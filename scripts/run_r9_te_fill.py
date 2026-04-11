"""Round 9: TE-Fill Strategy — Fix Stale Lag Features at the Source

KEY INSIGHT:
Instead of forward-filling test sales with the constant last training day value,
fill with Target Encoding estimates (per store-family-day_of_week mean).

This makes lag features VARY across test days (by day_of_week) instead of being
a constant. The model sees realistic, varying inputs instead of an out-of-distribution
constant pattern.

Why this should work:
- During training, lag features vary day-to-day (real sales values)
- With ffill, lag features are constant across all test days (one value repeated)
- With TE-fill, lag features vary by day_of_week (realistic seasonal pattern)
- The model's prediction should be at the correct magnitude without post-processing

Expected outcome:
- If this works, model predictions should have mean ≈ training mean (~467)
- No need for geometric mean blending post-processing
- LB score should approach CV score (0.36), near top of leaderboard (0.376)
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, merge_all_tables
from src.features.builder import build_features
from src.models.gbdt import train_lightgbm, predict_cv_models, time_series_split
from src.config import ModelConfig, CVConfig, SUBMISSIONS
from src.utils.metrics import rmsle

start = time.time()

print("=" * 60)
print("Round 9: TE-Fill Strategy")
print("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
print("\nSTEP 1: Load data")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# ============================================================
# STEP 2: Compute TE estimates for test sales
# ============================================================
print("\nSTEP 2: Compute TE estimates")

train_sales = train[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek

# Primary TE: per (store, family, day_of_week)
te_sf_dow = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].mean()
print(f"  TE (store, family, dow): {len(te_sf_dow)} groups, "
      f"mean={te_sf_dow.mean():.2f}")

# Map TE to test rows
test_dow = test["date"].dt.dayofweek
test_te = test.apply(
    lambda r: te_sf_dow.get((r["store_nbr"], r["family"], r["date"].dayofweek), np.nan),
    axis=1,
)
# Faster approach: merge
te_df = te_sf_dow.reset_index()
te_df.columns = ["store_nbr", "family", "day_of_week", "te_sales"]
test_for_te = test[["store_nbr", "family", "date"]].copy()
test_for_te["day_of_week"] = test_for_te["date"].dt.dayofweek
test_for_te = test_for_te.merge(te_df, on=["store_nbr", "family", "day_of_week"], how="left")
test_te_sales = test_for_te["te_sales"]

print(f"  Test TE sales: mean={test_te_sales.mean():.2f}, "
      f"median={test_te_sales.median():.2f}, NaN={test_te_sales.isna().sum()}")

# Also compute recent TE (last 3 months) for comparison
recent_3m = train_sales[train_sales["date"] >= "2017-05-01"]
te_sf_dow_recent = recent_3m.groupby(["store_nbr", "family", "day_of_week"])["sales"].mean()
te_recent_df = te_sf_dow_recent.reset_index()
te_recent_df.columns = ["store_nbr", "family", "day_of_week", "te_recent_sales"]
test_for_te2 = test[["store_nbr", "family", "date"]].copy()
test_for_te2["day_of_week"] = test_for_te2["date"].dt.dayofweek
test_for_te2 = test_for_te2.merge(te_recent_df, on=["store_nbr", "family", "day_of_week"], how="left")
test_te_recent_sales = test_for_te2["te_recent_sales"]
print(f"  Test TE recent sales: mean={test_te_recent_sales.mean():.2f}, "
      f"NaN={test_te_recent_sales.isna().sum()}")

# ============================================================
# STEP 3: Build features with different fill strategies
# ============================================================

# --- Strategy A: Standard ffill (baseline, same as r8) ---
print("\nSTEP 3A: Build features — ffill (baseline)")
train_fe_ffill, test_fe_ffill, feat_cols_ffill = build_features(train, test)
print(f"  {len(feat_cols_ffill)} features")

# --- Strategy B: TE-fill (all training TE) ---
print("\nSTEP 3B: Build features — TE-fill (all training)")
train_fe_te, test_fe_te, feat_cols_te = build_features(train, test, test_sales=test_te_sales)
print(f"  {len(feat_cols_te)} features")

# --- Strategy C: TE-fill (recent 3 months) ---
print("\nSTEP 3C: Build features — TE-fill (recent 3 months)")
train_fe_recent, test_fe_recent, feat_cols_recent = build_features(train, test, test_sales=test_te_recent_sales)
print(f"  {len(feat_cols_recent)} features")

# ============================================================
# STEP 4: Target Encoding (same as r8)
# ============================================================
print("\nSTEP 4: Target encoding")

train_sales_full = train[["store_nbr", "family", "date", "sales"]].copy()
train_sales_full["day_of_week"] = train_sales_full["date"].dt.dayofweek
train_sales_full["month"] = train_sales_full["date"].dt.month
train_sales_full["day"] = train_sales_full["date"].dt.day
train_sales_full["is_weekend"] = (train_sales_full["day_of_week"] >= 5).astype(int)
train_sales_full["is_payday"] = ((train_sales_full["day"] == 15) | (train_sales_full["date"].dt.is_month_end).astype(int)).astype(int)
train_sales_full["week_of_year"] = train_sales_full["date"].dt.isocalendar().week.astype(int)
train_sales_full["quarter"] = train_sales_full["date"].dt.quarter

target_encodings = {}

# (store, family, day_of_week) — PRIMARY expected level
te1 = train_sales_full.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median", "count"]
).reset_index()
te1.columns = ["store_nbr", "family", "day_of_week",
               "te_sf_dow_mean", "te_sf_dow_std", "te_sf_dow_median", "te_sf_dow_count"]
target_encodings["sf_dow"] = te1

# (store, family, month)
te2 = train_sales_full.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te2.columns = ["store_nbr", "family", "month", "te_sf_month_mean", "te_sf_month_std"]
target_encodings["sf_month"] = te2

# (store, family, is_payday)
te3 = train_sales_full.groupby(["store_nbr", "family", "is_payday"])["sales"].agg(
    ["mean"]
).reset_index()
te3.columns = ["store_nbr", "family", "is_payday", "te_sf_payday_mean"]
target_encodings["sf_payday"] = te3

# (store, family, week_of_year)
te6 = train_sales_full.groupby(["store_nbr", "family", "week_of_year"])["sales"].agg(
    ["mean"]
).reset_index()
te6.columns = ["store_nbr", "family", "week_of_year", "te_sf_woy_mean"]
target_encodings["sf_woy"] = te6

# (store, family, quarter)
te8 = train_sales_full.groupby(["store_nbr", "family", "quarter"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te8.columns = ["store_nbr", "family", "quarter", "te_sf_quarter_mean", "te_sf_quarter_std"]
target_encodings["sf_quarter"] = te8

# (store, family, is_weekend)
te9 = train_sales_full.groupby(["store_nbr", "family", "is_weekend"])["sales"].agg(
    ["mean"]
).reset_index()
te9.columns = ["store_nbr", "family", "is_weekend", "te_sf_weekend_mean"]
target_encodings["sf_weekend"] = te9

# (store, family) overall
te12 = train_sales_full.groupby(["store_nbr", "family"])["sales"].agg(
    ["mean", "std", "median", "count"]
).reset_index()
te12.columns = ["store_nbr", "family",
                "te_sf_mean", "te_sf_std", "te_sf_median", "te_sf_count"]
target_encodings["sf"] = te12

# Family overall mean
family_mean = train_sales_full.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "te_family_mean"]
target_encodings["family"] = family_mean

# Store overall
te13 = train_sales_full.groupby(["store_nbr"])["sales"].agg(["mean"]).reset_index()
te13.columns = ["store_nbr", "te_store_mean"]
target_encodings["store"] = te13

# Recent 3 months encoding
recent_3m_full = train_sales_full[train_sales_full["date"] >= "2017-05-01"]
te16 = recent_3m_full.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te16.columns = ["store_nbr", "family", "day_of_week", "te_sf_dow_recent_mean", "te_sf_dow_recent_std"]
target_encodings["sf_dow_recent"] = te16


def add_te_features(train_fe, test_fe):
    """Add target encoding features to both train and test."""
    for df in [train_fe, test_fe]:
        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["is_payday"] = ((pd.to_datetime(df["date"]).dt.day == 15) |
                           (pd.to_datetime(df["date"]).dt.is_month_end)).astype(int)
        df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
        df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
        df["is_weekend"] = (pd.to_datetime(df["date"]).dt.dayofweek >= 5).astype(int)

    for name, te_df in target_encodings.items():
        merge_cols = [c for c in te_df.columns if c in train_fe.columns or c in test_fe.columns]
        train_fe = train_fe.merge(te_df, on=merge_cols, how="left")
        test_fe = test_fe.merge(te_df, on=merge_cols, how="left")

    # Interaction features
    if "te_sf_dow_mean" in train_fe.columns and "te_family_mean" in train_fe.columns:
        train_fe["te_sf_ratio"] = train_fe["te_sf_dow_mean"] / (train_fe["te_family_mean"] + 1)
        test_fe["te_sf_ratio"] = test_fe["te_sf_dow_mean"] / (test_fe["te_family_mean"] + 1)

    if "te_sf_std" in train_fe.columns and "te_sf_mean" in train_fe.columns:
        train_fe["te_sf_cv"] = train_fe["te_sf_std"] / (train_fe["te_sf_mean"] + 1)
        test_fe["te_sf_cv"] = test_fe["te_sf_std"] / (test_fe["te_sf_mean"] + 1)

    if "te_sf_mean" in train_fe.columns and "te_store_mean" in train_fe.columns:
        train_fe["te_sf_store_ratio"] = train_fe["te_sf_mean"] / (train_fe["te_store_mean"] + 1)
        test_fe["te_sf_store_ratio"] = test_fe["te_sf_mean"] / (test_fe["te_store_mean"] + 1)

    exclude_cols = {
        "id", "date", "sales", "is_train",
        "family", "city", "state", "type",
        "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
        "holiday_type_national", "holiday_type_regional", "holiday_type_local",
    }
    feat_cols = [c for c in train_fe.columns if c not in exclude_cols]
    return train_fe, test_fe, feat_cols


# Add TE features to all three feature sets
train_fe_ffill, test_fe_ffill, feat_cols_ffill = add_te_features(train_fe_ffill, test_fe_ffill)
train_fe_te, test_fe_te, feat_cols_te = add_te_features(train_fe_te, test_fe_te)
train_fe_recent, test_fe_recent, feat_cols_recent = add_te_features(train_fe_recent, test_fe_recent)

print(f"  Features after TE: ffill={len(feat_cols_ffill)}, te={len(feat_cols_te)}, recent={len(feat_cols_recent)}")

# ============================================================
# STEP 5: Clean and prepare
# ============================================================
print("\nSTEP 5: Clean")

def clean_data(train_fe, feat_cols):
    lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
    train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
    return train_clean

train_clean_ffill = clean_data(train_fe_ffill, feat_cols_ffill)
train_clean_te = clean_data(train_fe_te, feat_cols_te)
train_clean_recent = clean_data(train_fe_recent, feat_cols_recent)

print(f"  Train clean: ffill={train_clean_ffill.shape}, te={train_clean_te.shape}, recent={train_clean_recent.shape}")

# Zero-sales pairs
sf_stats = train_clean_ffill.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))
print(f"  Zero-sales pairs (>99%): {len(zero_pairs)}")

test_info = test_fe_te[["store_nbr", "family"]].copy()

# ============================================================
# Helper
# ============================================================
def post_process(preds, name, cv_val):
    p = preds.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0
    sub = pd.DataFrame({"id": test_fe_te["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    print(f"  {name}: CV={cv_val:.5f}, mean={p.mean():.2f}, "
          f"max={p.max():.2f}, zeros={(p == 0).mean():.4f}")
    return p

# ============================================================
# Experiment A: ffill baseline (reproduce r8)
# ============================================================
print("\n" + "=" * 60)
print("Experiment A: ffill baseline")
print("=" * 60)
cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)
cv_cfg = CVConfig(n_folds=5, val_days=16)
result_ffill = train_lightgbm(train_clean_ffill, feat_cols_ffill, cfg=cfg, cv_cfg=cv_cfg)
preds_ffill = predict_cv_models(result_ffill["cv_models"], test_fe_ffill, feat_cols_ffill)
print(f"  Preds: mean={preds_ffill.mean():.2f}, max={preds_ffill.max():.2f}")

# ============================================================
# Experiment B: TE-fill (all training data)
# ============================================================
print("\n" + "=" * 60)
print("Experiment B: TE-fill (all training)")
print("=" * 60)
result_te = train_lightgbm(train_clean_te, feat_cols_te, cfg=cfg, cv_cfg=cv_cfg)
preds_te = predict_cv_models(result_te["cv_models"], test_fe_te, feat_cols_te)
print(f"  Preds: mean={preds_te.mean():.2f}, max={preds_te.max():.2f}")

# ============================================================
# Experiment C: TE-fill (recent 3 months)
# ============================================================
print("\n" + "=" * 60)
print("Experiment C: TE-fill (recent 3 months)")
print("=" * 60)
result_recent = train_lightgbm(train_clean_recent, feat_cols_recent, cfg=cfg, cv_cfg=cv_cfg)
preds_recent = predict_cv_models(result_recent["cv_models"], test_fe_recent, feat_cols_recent)
print(f"  Preds: mean={preds_recent.mean():.2f}, max={preds_recent.max():.2f}")

# ============================================================
# STEP 6: Post-process all predictions
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Post-process and submit")
print("=" * 60)

# Baseline: ffill
post_process(preds_ffill, "r9_ffill_baseline", result_ffill["mean_cv"])

# TE-fill variants
post_process(preds_te, "r9_te_fill", result_te["mean_cv"])
post_process(preds_recent, "r9_te_fill_recent", result_recent["mean_cv"])

# Geometric mean blend ON TOP of TE-fill (in case TE-fill still underpredicts)
te_level = test_fe_te["te_sf_dow_mean"].values.copy()
te_level = np.clip(te_level, 0.1, None)

for alpha in [0.01, 0.02, 0.05, 0.10, 0.20]:
    geo = np.expm1(alpha * np.log1p(np.clip(preds_te, 0, None)) + (1 - alpha) * np.log1p(te_level))
    name = f"r9_tefill_geo_{int(alpha*100):02d}_{int((1-alpha)*100)}"
    post_process(geo, name, result_te["mean_cv"])

# Also try geometric mean blend on ffill baseline for comparison
for alpha in [0.05, 0.10]:
    geo = np.expm1(alpha * np.log1p(np.clip(preds_ffill, 0, None)) + (1 - alpha) * np.log1p(te_level))
    name = f"r9_ffill_geo_{int(alpha*100):02d}_{int((1-alpha)*100)}"
    post_process(geo, name, result_ffill["mean_cv"])

# ============================================================
# STEP 7: Diagnostic — check underprediction ratio
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Diagnostic")
print("=" * 60)

train_mean = train["sales"].mean()
print(f"  Training sales mean: {train_mean:.2f}")
print(f"  ffill pred mean: {preds_ffill.mean():.2f} (ratio: {preds_ffill.mean()/train_mean:.3f})")
print(f"  TE-fill pred mean: {preds_te.mean():.2f} (ratio: {preds_te.mean()/train_mean:.3f})")
print(f"  Recent-fill pred mean: {preds_recent.mean():.2f} (ratio: {preds_recent.mean()/train_mean:.3f})")

if preds_te.mean() / train_mean > 0.5:
    print("\n  *** TE-fill significantly improved prediction magnitude! ***")
else:
    print("\n  *** TE-fill did not fully fix underprediction, geometric blend may still help ***")

print(f"\nTotal: {time.time() - start:.1f}s")
