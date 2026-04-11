"""Round 8: TE-Level Post-Processing to Fix Systematic Underprediction

KEY INSIGHT FROM ANALYSIS:
- The model systematically underpredicts by ~8x (pred mean=55, actual mean=467)
- Root cause: lag features (52% importance) are stale at test time (ffill)
- Target encoding (TE) features know the EXPECTED level correctly
- The model's lag-dependent predictions are like "ranking" - they get relative
  magnitudes right but absolute scale wrong

SOLUTION: Instead of trying to fix the model, POST-PROCESS the predictions
using TE-derived expected levels:

1. Model prediction = "ranking signal" (relative ordering is correct)
2. TE expected = "level signal" (absolute magnitude is correct)
3. Blend: final = alpha * model + (1-alpha) * TE_expected

Or better: use the model's prediction as a MULTIPLIER of the expected level.
The model gets the RATIO right even if the absolute value is wrong.

ratio_pred = model_pred / TE_expected_train  (what the model thinks vs expected)
final_pred = ratio_pred * TE_expected_test

Since TE_expected is the same in train and test, this gives:
final_pred ≈ model_pred (same as before)

Hmm, that won't work. Let me think again...

Actually the simplest approach that works:
- Use the raw te_sf_dow_mean as prediction (direct TE baseline)
- Blend model predictions with TE baseline
- The model adds signal BEYOND the TE baseline

Also try: use model to predict sales/te_sf_dow_mean ratio, then multiply back.
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
print("Round 8: TE-Based Post-Processing")
print("=" * 60)

# ============================================================
# STEP 1: Load data and build base features
# ============================================================
print("\nSTEP 1: Load data")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# ============================================================
# STEP 2: Build features with ffill
# ============================================================
print("\nSTEP 2: Build features (ffill)")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} base features")

# ============================================================
# STEP 3: Compute Target Encodings
# ============================================================
print("\nSTEP 3: Target encoding")

train_sales = train[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek
train_sales["month"] = train_sales["date"].dt.month
train_sales["day"] = train_sales["date"].dt.day
train_sales["is_weekend"] = (train_sales["day_of_week"] >= 5).astype(int)
train_sales["is_payday"] = ((train_sales["day"] == 15) | train_sales["date"].dt.is_month_end).astype(int)
train_sales["week_of_year"] = train_sales["date"].dt.isocalendar().week.astype(int)
train_sales["quarter"] = train_sales["date"].dt.quarter

target_encodings = {}

# (store, family, day_of_week) — PRIMARY expected level
te1 = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median", "count"]
).reset_index()
te1.columns = ["store_nbr", "family", "day_of_week",
               "te_sf_dow_mean", "te_sf_dow_std", "te_sf_dow_median", "te_sf_dow_count"]
target_encodings["sf_dow"] = te1

# (store, family, month)
te2 = train_sales.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te2.columns = ["store_nbr", "family", "month", "te_sf_month_mean", "te_sf_month_std"]
target_encodings["sf_month"] = te2

# (store, family, is_payday)
te3 = train_sales.groupby(["store_nbr", "family", "is_payday"])["sales"].agg(
    ["mean"]
).reset_index()
te3.columns = ["store_nbr", "family", "is_payday", "te_sf_payday_mean"]
target_encodings["sf_payday"] = te3

# (store, day_of_week)
te4 = train_sales.groupby(["store_nbr", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te4.columns = ["store_nbr", "day_of_week", "te_s_dow_mean", "te_s_dow_std"]
target_encodings["s_dow"] = te4

# (family, day_of_week)
te5 = train_sales.groupby(["family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te5.columns = ["family", "day_of_week", "te_f_dow_mean", "te_f_dow_std"]
target_encodings["f_dow"] = te5

# (store, family, week_of_year)
te6 = train_sales.groupby(["store_nbr", "family", "week_of_year"])["sales"].agg(
    ["mean"]
).reset_index()
te6.columns = ["store_nbr", "family", "week_of_year", "te_sf_woy_mean"]
target_encodings["sf_woy"] = te6

# Family overall mean
family_mean = train_sales.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "te_family_mean"]
target_encodings["family"] = family_mean

# (store, family, quarter)
te8 = train_sales.groupby(["store_nbr", "family", "quarter"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te8.columns = ["store_nbr", "family", "quarter", "te_sf_quarter_mean", "te_sf_quarter_std"]
target_encodings["sf_quarter"] = te8

# (store, family, is_weekend)
te9 = train_sales.groupby(["store_nbr", "family", "is_weekend"])["sales"].agg(
    ["mean"]
).reset_index()
te9.columns = ["store_nbr", "family", "is_weekend", "te_sf_weekend_mean"]
target_encodings["sf_weekend"] = te9

# (store, family) overall
te12 = train_sales.groupby(["store_nbr", "family"])["sales"].agg(
    ["mean", "std", "median", "count"]
).reset_index()
te12.columns = ["store_nbr", "family",
                "te_sf_mean", "te_sf_std", "te_sf_median", "te_sf_count"]
target_encodings["sf"] = te12

# Store overall
te13 = train_sales.groupby(["store_nbr"])["sales"].agg(["mean"]).reset_index()
te13.columns = ["store_nbr", "te_store_mean"]
target_encodings["store"] = te13

# Last 3 months encoding
recent_3m = train_sales[train_sales["date"] >= "2017-05-01"]
te16 = recent_3m.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te16.columns = ["store_nbr", "family", "day_of_week", "te_sf_dow_recent_mean", "te_sf_dow_recent_std"]
target_encodings["sf_dow_recent"] = te16

# Add date-derived columns
for df in [train_fe, test_fe]:
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_payday"] = ((pd.to_datetime(df["date"]).dt.day == 15) |
                       pd.to_datetime(df["date"]).dt.is_month_end).astype(int)
    df["week_of_year"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    df["is_weekend"] = (pd.to_datetime(df["date"]).dt.dayofweek >= 5).astype(int)

# Merge all target encodings
for name, te_df in target_encodings.items():
    merge_cols = [c for c in te_df.columns if c in train_fe.columns or c in test_fe.columns]
    train_fe = train_fe.merge(te_df, on=merge_cols, how="left")
    test_fe = test_fe.merge(te_df, on=merge_cols, how="left")

# Add interaction features
if "te_sf_dow_mean" in train_fe.columns and "te_family_mean" in train_fe.columns:
    train_fe["te_sf_ratio"] = train_fe["te_sf_dow_mean"] / (train_fe["te_family_mean"] + 1)
    test_fe["te_sf_ratio"] = test_fe["te_sf_dow_mean"] / (test_fe["te_family_mean"] + 1)

if "te_sf_std" in train_fe.columns and "te_sf_mean" in train_fe.columns:
    train_fe["te_sf_cv"] = train_fe["te_sf_std"] / (train_fe["te_sf_mean"] + 1)
    test_fe["te_sf_cv"] = test_fe["te_sf_std"] / (test_fe["te_sf_mean"] + 1)

if "te_sf_mean" in train_fe.columns and "te_store_mean" in train_fe.columns:
    train_fe["te_sf_store_ratio"] = train_fe["te_sf_mean"] / (train_fe["te_store_mean"] + 1)
    test_fe["te_sf_store_ratio"] = test_fe["te_sf_mean"] / (test_fe["te_store_mean"] + 1)

# Update feature columns
exclude_cols = {
    "id", "date", "sales", "is_train",
    "family", "city", "state", "type",
    "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
    "holiday_type_national", "holiday_type_regional", "holiday_type_local",
}
feat_cols_all = [c for c in train_fe.columns if c not in exclude_cols]
print(f"  Total features: {len(feat_cols_all)}")

# ============================================================
# STEP 4: Clean
# ============================================================
print("\nSTEP 4: Clean")
lag_roll_cols = [c for c in feat_cols_all if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# ============================================================
# STEP 5: Zero-sales pairs
# ============================================================
sf_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))
print(f"  Zero-sales pairs (>99%): {len(zero_pairs)}")

test_info = test_fe[["store_nbr", "family"]].copy()

# ============================================================
# Helper functions
# ============================================================

def post_process(preds, name, cv_val):
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
    return p


# ============================================================
# Experiment A: Standard approach (reproduce r6_all_te as baseline)
# ============================================================
print("\n" + "=" * 60)
print("Experiment A: Standard LightGBM (r6_all_te reproduction)")
print("=" * 60)
cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)
cv_cfg = CVConfig(n_folds=5, val_days=16)
result_a = train_lightgbm(train_clean, feat_cols_all, cfg=cfg, cv_cfg=cv_cfg)
preds_a = predict_cv_models(result_a["cv_models"], test_fe, feat_cols_all)
print(f"  Preds: mean={preds_a.mean():.2f}, max={preds_a.max():.2f}")

# ============================================================
# Experiment B: Tweedie objective
# ============================================================
print("\n" + "=" * 60)
print("Experiment B: Tweedie objective")
print("=" * 60)
cfg_b = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64,
                    min_child_samples=30, objective="tweedie")
result_b = train_lightgbm(train_clean, feat_cols_all, cfg=cfg_b, cv_cfg=cv_cfg)
preds_b = predict_cv_models(result_b["cv_models"], test_fe, feat_cols_all)
print(f"  Preds: mean={preds_b.mean():.2f}, max={preds_b.max():.2f}")

# ============================================================
# STEP 6: TE-Based Post-Processing
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: TE-Based Post-Processing")
print("=" * 60)

# Get TE expected levels for train (OOF) and test
train_te_level = train_clean["te_sf_dow_mean"].values.copy()
train_te_level = np.clip(train_te_level, 0.1, None)
test_te_level = test_fe["te_sf_dow_mean"].values.copy()
test_te_level = np.clip(test_te_level, 0.1, None)

# The key insight: the model's prediction has the right RANKING but wrong SCALE
# We can learn a scale correction from OOF predictions

# OOF correction: for each (store, family), compute ratio of actual/predicted
oof_preds = result_a["oof_preds"]
actual = train_clean["sales"].values

# Compute per-(store, family) scale factor from OOF
train_clean_copy = train_clean.copy()
train_clean_copy["oof_pred"] = oof_preds
train_clean_copy["actual"] = actual

# Scale factor: median(actual / oof_pred) per store-family
# This captures the systematic underprediction pattern
sf_scale = train_clean_copy.groupby(["store_nbr", "family"]).apply(
    lambda x: pd.Series({
        "scale_median": np.median(x["actual"] / (x["oof_pred"] + 0.1)),
        "scale_mean": np.mean(x["actual"]) / (np.mean(x["oof_pred"]) + 0.1),
    })
).reset_index()

print(f"  Scale factor distribution:")
print(f"    median scale: mean={sf_scale["scale_median"].mean():.2f}, "
      f"median={sf_scale["scale_median"].median():.2f}")
print(f"    mean scale: mean={sf_scale["scale_mean"].mean():.2f}, "
      f"median={sf_scale["scale_mean"].median():.2f}")

# Merge scale factors to test
test_scaled = test_fe[["store_nbr", "family"]].copy()
test_scaled = test_scaled.merge(sf_scale, on=["store_nbr", "family"], how="left")
test_scaled["scale_median"] = test_scaled["scale_median"].fillna(1.0)
test_scaled["scale_mean"] = test_scaled["scale_mean"].fillna(1.0)

# Apply per-store-family scaling
preds_a_scaled_median = preds_a * test_scaled["scale_median"].values
preds_a_scaled_mean = preds_a * test_scaled["scale_mean"].values

post_process(preds_a, "r8_standard", result_a["mean_cv"])
post_process(preds_b, "r8_tweedie", result_b["mean_cv"])
post_process(preds_a_scaled_median, "r8_scaled_median", result_a["mean_cv"])
post_process(preds_a_scaled_mean, "r8_scaled_mean", result_a["mean_cv"])

# ============================================================
# STEP 7: TE-level as direct prediction
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Direct TE prediction (no model)")
print("=" * 60)

# Direct TE prediction: just use te_sf_dow_mean
preds_te_direct = test_te_level.copy()
preds_te_direct = np.clip(preds_te_direct, 0, None)

# Compute CV for direct TE
train_te_cv = train_clean["te_sf_dow_mean"].values.copy()
train_te_cv = np.clip(train_te_cv, 0, None)
cv_te = rmsle(actual, train_te_cv)
print(f"  Direct TE CV: {cv_te:.5f}")
post_process(preds_te_direct, "r8_te_direct", cv_te)

# ============================================================
# STEP 8: Blending model predictions with TE level
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Blending experiments")
print("=" * 60)

best_cv = min(result_a["mean_cv"], result_b["mean_cv"])

# Blend standard model with TE direct
for w_model, w_te in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3), (0.2, 0.8), (0.1, 0.9)]:
    blended = w_model * preds_a + w_te * preds_te_direct
    name = f"r8_blend_model_te_{int(w_model*100)}_{int(w_te*100)}"
    post_process(blended, name, best_cv)

# Blend scaled model with TE direct
for w_model, w_te in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]:
    blended = w_model * preds_a_scaled_median + w_te * preds_te_direct
    name = f"r8_blend_scaled_te_{int(w_model*100)}_{int(w_te*100)}"
    post_process(blended, name, best_cv)

# Blend tweedie with TE direct
for w_model, w_te in [(0.5, 0.5), (0.3, 0.7), (0.7, 0.3)]:
    blended = w_model * preds_b + w_te * preds_te_direct
    name = f"r8_blend_tweedie_te_{int(w_model*100)}_{int(w_te*100)}"
    post_process(blended, name, best_cv)

# Geometric mean blends (better for RMSLE)
log_a = np.log1p(preds_a)
log_te = np.log1p(preds_te_direct)
for w_model, w_te in [(0.5, 0.5), (0.3, 0.7), (0.2, 0.8), (0.1, 0.9)]:
    geo = np.expm1(w_model * log_a + w_te * log_te)
    name = f"r8_geo_model_te_{int(w_model*100)}_{int(w_te*100)}"
    post_process(geo, name, best_cv)

# ============================================================
# STEP 9: Per-day-of-week scaling
# ============================================================
print("\n" + "=" * 60)
print("STEP 9: Per-day-of-week correction")
print("=" * 60)

# The model underpredicts more for early test days (lag features are most stale)
# Let's compute per-day correction factors from OOF
train_clean_copy["day_of_week"] = train_clean_copy["date"].dt.dayofweek
dow_scale = train_clean_copy.groupby("day_of_week").apply(
    lambda x: np.mean(x["actual"]) / (np.mean(x["oof_pred"]) + 0.1)
).reset_index()
dow_scale.columns = ["day_of_week", "dow_scale"]
print("  DoW scale factors from OOF:")
for _, row in dow_scale.iterrows():
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    print(f"    {days[int(row['day_of_week'])]}: {row['dow_scale']:.3f}")

# Apply per-DoW scaling
test_dow = test_fe.copy()
test_dow["day_of_week"] = pd.to_datetime(test_dow["date"]).dt.dayofweek
test_dow = test_dow.merge(dow_scale, on="day_of_week", how="left")
test_dow["dow_scale"] = test_dow["dow_scale"].fillna(1.0)

preds_a_dow_scaled = preds_a * test_dow["dow_scale"].values
preds_a_dow_scaled = np.clip(preds_a_dow_scaled, 0, None)
post_process(preds_a_dow_scaled, "r8_dow_scaled", result_a["mean_cv"])

# Combine per-SF and per-DoW scaling
combined_scale = test_scaled["scale_median"].values * test_dow["dow_scale"].values
preds_a_combined = preds_a * combined_scale
preds_a_combined = np.clip(preds_a_combined, 0, None)
post_process(preds_a_combined, "r8_combined_scaled", result_a["mean_cv"])

print(f"\nTotal: {time.time() - start:.1f}s")
