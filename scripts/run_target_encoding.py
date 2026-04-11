"""Round 6: Target Encoding + Enhanced Features

Key insight from Round 5: The model over-relies on lag_1 which is stale during
test prediction. Instead of trying to fix lag features (recursive failed, removing
hurt), add TARGET ENCODING features that capture the same information without
depending on recent sales values.

Target encoding: compute historical sales statistics at various grouping levels
from TRAINING data only. These features are fully known at test time and don't
depend on lag features.

Grouping levels:
- (store_nbr, family, day_of_week)  — captures weekly patterns per product
- (store_nbr, family, month)         — captures seasonal patterns
- (store_nbr, day_of_week)           — captures store-level weekly patterns
- (family, day_of_week)              — captures product-level weekly patterns
- (store_nbr, family, is_payday)     — captures payday effects

Also try: ffill with adjusted prediction (scale factor to match expected mean).
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, merge_all_tables
from src.features.builder import build_features
from src.models.gbdt import train_lightgbm, predict_cv_models
from src.config import ModelConfig, CVConfig, SUBMISSIONS

start = time.time()

print("=" * 60)
print("Round 6: Target Encoding + Enhanced Features")
print("=" * 60)

# 1. Load data
print("\nSTEP 1: Load data")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# 2. Build features with ffill
print("\nSTEP 2: Build features (ffill)")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} base features")

# 3. Add target encoding features from TRAINING data only
print("\nSTEP 3: Target encoding from training data")

# Use original training data (before feature building) for clean target encoding
train_sales = train[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek
train_sales["month"] = train_sales["date"].dt.month
train_sales["day"] = train_sales["date"].dt.day
train_sales["is_weekend"] = (train_sales["day_of_week"] >= 5).astype(int)
train_sales["is_payday"] = ((train_sales["day"] == 15) | train_sales["date"].dt.is_month_end).astype(int)
train_sales["week_of_year"] = train_sales["date"].dt.isocalendar().week.astype(int)

# Compute target encodings at various grouping levels
target_encodings = {}

# Level 1: (store, family, day_of_week) — most granular temporal pattern
te1 = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median", "count"]
).reset_index()
te1.columns = ["store_nbr", "family", "day_of_week",
               "te_sf_dow_mean", "te_sf_dow_std", "te_sf_dow_median", "te_sf_dow_count"]
target_encodings["sf_dow"] = te1

# Level 2: (store, family, month) — seasonal pattern
te2 = train_sales.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te2.columns = ["store_nbr", "family", "month", "te_sf_month_mean", "te_sf_month_std"]
target_encodings["sf_month"] = te2

# Level 3: (store, family, is_payday) — payday effect
te3 = train_sales.groupby(["store_nbr", "family", "is_payday"])["sales"].agg(
    ["mean"]
).reset_index()
te3.columns = ["store_nbr", "family", "is_payday", "te_sf_payday_mean"]
target_encodings["sf_payday"] = te3

# Level 4: (store, day_of_week) — store-level weekly pattern
te4 = train_sales.groupby(["store_nbr", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te4.columns = ["store_nbr", "day_of_week", "te_s_dow_mean", "te_s_dow_std"]
target_encodings["s_dow"] = te4

# Level 5: (family, day_of_week) — product-level weekly pattern
te5 = train_sales.groupby(["family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te5.columns = ["family", "day_of_week", "te_f_dow_mean", "te_f_dow_std"]
target_encodings["f_dow"] = te5

# Level 6: (store, family, week_of_year) — annual pattern
te6 = train_sales.groupby(["store_nbr", "family", "week_of_year"])["sales"].agg(
    ["mean"]
).reset_index()
te6.columns = ["store_nbr", "family", "week_of_year", "te_sf_woy_mean"]
target_encodings["sf_woy"] = te6

# Level 7: Sales ratio features
# Ratio of store-family sales to overall family sales
family_mean = train_sales.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "te_family_mean"]
target_encodings["family"] = family_mean

# Add day_of_week and month to test/train features
for df in [train_fe, test_fe]:
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_payday"] = ((pd.to_datetime(df["date"]).dt.day == 15) |
                       pd.to_datetime(df["date"]).dt.is_month_end).astype(int)
    df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)

# Merge target encodings
for name, te_df in target_encodings.items():
    merge_cols = [c for c in te_df.columns if c in train_fe.columns or c in test_fe.columns]
    train_fe = train_fe.merge(te_df, on=merge_cols, how="left")
    test_fe = test_fe.merge(te_df, on=merge_cols, how="left")

# Add sales ratio feature: te_sf_dow_mean / te_family_mean
if "te_sf_dow_mean" in train_fe.columns and "te_family_mean" in train_fe.columns:
    train_fe["te_sf_ratio"] = train_fe["te_sf_dow_mean"] / (train_fe["te_family_mean"] + 1)
    test_fe["te_sf_ratio"] = test_fe["te_sf_dow_mean"] / (test_fe["te_family_mean"] + 1)

# Update feature columns
exclude_cols = {
    "id", "date", "sales", "is_train",
    "family", "city", "state", "type",
    "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
    "holiday_type_national", "holiday_type_regional", "holiday_type_local",
}
feat_cols_te = [c for c in train_fe.columns if c not in exclude_cols]
print(f"  Target encoding features added: {len(feat_cols_te) - len(feat_cols)}")
print(f"  Total features: {len(feat_cols_te)}")

# 4. Clean
print("\nSTEP 4: Clean")
lag_roll_cols = [c for c in feat_cols_te if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# 5. Train experiments
cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)
cv_cfg = CVConfig(n_folds=5, val_days=16)

# Experiment A: All features + target encoding
print("\n" + "=" * 60)
print("Experiment A: All features + target encoding")
print("=" * 60)
result_a = train_lightgbm(train_clean, feat_cols_te, cfg=cfg, cv_cfg=cv_cfg)
preds_a = predict_cv_models(result_a["cv_models"], test_fe, feat_cols_te)
print(f"  Preds: mean={preds_a.mean():.2f}, max={preds_a.max():.2f}")

# Experiment B: Safe features only + target encoding (no lag < 16)
print("\n" + "=" * 60)
print("Experiment B: Safe features + target encoding (no lag < 16)")
print("=" * 60)
safe_cols = []
for c in feat_cols_te:
    if "sales_lag_" in c:
        lag_n = int(c.split("_")[-1])
        if lag_n >= 16:
            safe_cols.append(c)
        else:
            continue
    elif "sales_roll_" in c or "sales_ewm_" in c:
        continue  # Remove all rolling/EWM (depend on recent sales)
    elif "onpromotion_lag_" in c:
        lag_n = int(c.split("_")[-1])
        if lag_n >= 16:
            safe_cols.append(c)
        else:
            continue
    elif "onpromotion_roll_" in c:
        continue
    elif "transactions_lag_" in c or "transactions_roll_" in c:
        continue
    else:
        safe_cols.append(c)

print(f"  Safe features: {len(safe_cols)}")
result_b = train_lightgbm(train_clean, safe_cols, cfg=cfg, cv_cfg=cv_cfg)
preds_b = predict_cv_models(result_b["cv_models"], test_fe, safe_cols)
print(f"  Preds: mean={preds_b.mean():.2f}, max={preds_b.max():.2f}")

# 6. Create submissions
print("\n" + "=" * 60)
print("STEP 6: Post-processing and save")
print("=" * 60)

# Zero-sales pairs
sf_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))

test_info = test_fe[["store_nbr", "family"]].copy()

# Save experiment A (all features + TE)
for name, preds, cv_val in [
    ("r6_all_te", preds_a, result_a["mean_cv"]),
    ("r6_safe_te", preds_b, result_b["mean_cv"]),
]:
    p = preds.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0
    sub = pd.DataFrame({"id": test_fe["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"  {name}: CV={cv_val:.5f}, mean={p.mean():.2f}, "
          f"max={p.max():.2f}, zeros={(p == 0).mean():.4f}")

# Also create blend submissions
for w_a, w_b in [(0.7, 0.3), (0.8, 0.2), (0.5, 0.5)]:
    blended = w_a * preds_a + w_b * preds_b
    p = blended.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0
    name = f"r6_blend_{int(w_a*100)}_{int(w_b*100)}"
    sub = pd.DataFrame({"id": test_fe["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  {name}: mean={p.mean():.2f}, max={p.max():.2f}")

# Scale experiments: scale predictions to match expected mean (~31)
for target_mean in [25, 30, 35]:
    scale = target_mean / preds_a.mean()
    scaled = preds_a * scale
    p = scaled.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0
    name = f"r6_scale_{target_mean}"
    sub = pd.DataFrame({"id": test_fe["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    sub.to_csv(path, index=False)
    print(f"  {name}: mean={p.mean():.2f}, scale={scale:.3f}")

print(f"\nTotal: {time.time() - start:.1f}s")
