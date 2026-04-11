"""Round 10: Day-Specific Models (1st Place Approach)

Based on the 1st place Favorita competition solution by shixw125/sjv.

KEY INSIGHT: Instead of computing features for each test day separately (which
makes lag features stale), compute features ONCE from the last training date,
and train 16 separate models — one per prediction horizon.

For each day d (1 to 16):
- Training: features from date t → target = sales on date t+d
- Test: features from last training date (Aug 15) → predict d days ahead
- All lag features reference real training data — no stale features!

This completely avoids the stale lag problem because:
- lag_1 for ALL models = sales on Aug 14 (real, known)
- lag_7 for ALL models = sales on Aug 8 (real, known)
- rolling_mean_7 for ALL models = mean of Aug 8-14 (real, known)
- No forward-filling, no recursive prediction needed

Target date features (day_of_week, month, holiday, promotions) are added
separately for the specific day being predicted.
"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import sys
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, merge_all_tables
from src.features.builder import build_features
from src.config import ModelConfig, CVConfig, SUBMISSIONS
from src.utils.metrics import rmsle

start = time.time()

print("=" * 60)
print("Round 10: Day-Specific Models (1st Place Approach)")
print("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
print("\nSTEP 1: Load data")
data = load_raw_data()
train_raw, test_raw = merge_all_tables(data)
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

# Test period: 2017-08-16 to 2017-08-31 (16 days)
test_dates = sorted(test_raw["date"].unique())
print(f"  Test dates: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
last_train_date = train_raw["date"].max()
print(f"  Last train date: {last_train_date.date()}")

# ============================================================
# STEP 2: Build features on training data only (no test!)
# ============================================================
print("\nSTEP 2: Build features on training data")

# Create a dummy test with just the structure (no real predictions needed)
# We use the last 16 days of training as "pseudo-test" to get feature computation
# Actually, we need to compute features on the FULL training data
# The trick: compute lag features on training data only, then use last-row features

# Method: build features normally, but only on training data
# We need a small "test" set just so the feature builder works
# But we'll ignore those predictions

# Alternative approach: compute lag features directly on training data
# Then extract features for the last date per (store, family)

# Simpler: create a minimal test with the same columns, run build_features,
# but we only care about the training features

# Create a 1-row test to satisfy the builder
dummy_test = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe, _, feat_cols = build_features(train_raw, dummy_test)
print(f"  {len(feat_cols)} base features built on training data")

# ============================================================
# STEP 3: Create day-specific training datasets
# ============================================================
print("\nSTEP 3: Create day-specific datasets")

# For each day d (1-16), create a training set where:
# - Features are from date t
# - Target is sales on date t+d
# This means we can only use training dates where t+d is still in training data

train_fe["date"] = pd.to_datetime(train_fe["date"])
all_dates = sorted(train_fe["date"].unique())
print(f"  Training dates: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

# Get the features that are "reference date" features (computed from date t)
# These include: lag features, rolling features, EWM, time features of date t, etc.

# Identify which features are reference-date features vs target-date features
# Reference features: lag, rolling, ewm, store stats, oil at date t, etc.
# Target features: day_of_week of target, month of target, holiday on target, promotion on target

# We'll add target-date features separately
# For now, the base features from build_features are all reference-date features

# Also compute target encodings from training data
train_sales = train_raw[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek
train_sales["month"] = train_sales["date"].dt.month

# Primary TE: per (store, family, day_of_week)
te_sf_dow = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median"]
).reset_index()
te_sf_dow.columns = ["store_nbr", "family", "day_of_week",
                      "target_te_sf_dow_mean", "target_te_sf_dow_std", "target_te_sf_dow_median"]

# Per (store, family)
te_sf = train_sales.groupby(["store_nbr", "family"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf.columns = ["store_nbr", "family", "target_te_sf_mean", "target_te_sf_std"]

# Per (family, day_of_week)
te_f_dow = train_sales.groupby(["family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_f_dow.columns = ["family", "day_of_week", "target_te_f_dow_mean", "target_te_f_dow_std"]

# Per (store, family, month)
te_sf_month = train_sales.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_month.columns = ["store_nbr", "family", "month", "target_te_sf_month_mean", "target_te_sf_month_std"]

# Family mean
family_mean = train_sales.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "target_te_family_mean"]

# Recent 3 months TE
recent_3m = train_sales[train_sales["date"] >= "2017-05-01"]
te_sf_dow_recent = recent_3m.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_dow_recent.columns = ["store_nbr", "family", "day_of_week",
                             "target_te_sf_dow_recent_mean", "target_te_sf_dow_recent_std"]

print(f"  TE features computed")

# ============================================================
# STEP 4: Build day-specific training data
# ============================================================
print("\nSTEP 4: Build day-specific training data")

# For each day d, create training samples:
# Features from date t → target = sales on date t+d
# Only use dates t where t+d is still in training data and t has lag features

LAG_DROP_NAN_DAYS = 60  # approximate days needed for lag features to be valid
last_train_idx = len(all_dates) - 1

day_datasets = {}
for d in range(1, 17):
    # Valid reference dates: all dates where t+d is still in training
    # AND t is late enough that lag features are valid
    min_ref_idx = LAG_DROP_NAN_DAYS  # need enough history for lag features
    max_ref_idx = last_train_idx - d  # t+d must be in training data

    if max_ref_idx < min_ref_idx:
        print(f"  Day {d:2d}: Not enough training data, skipping")
        continue

    ref_dates = all_dates[min_ref_idx:max_ref_idx + 1]
    target_dates = [t + pd.Timedelta(days=d) for t in ref_dates]

    # Get reference features (avoid duplicates by selecting unique columns)
    ref_mask = train_fe["date"].isin(ref_dates)
    select_cols = ["store_nbr", "family", "date", "sales"] + [c for c in feat_cols if c not in ("store_nbr", "family", "date", "sales")]
    ref_data = train_fe[ref_mask][select_cols].copy()

    # Get target sales (the actual sales d days ahead)
    target_data = train_fe[train_fe["date"].isin(target_dates)][
        ["store_nbr", "family", "date", "sales"]
    ].copy()
    target_data.columns = ["store_nbr", "family", "target_date", "target_sales"]

    # Add d offset to ref_data dates to match target
    ref_data["target_date"] = ref_data["date"] + pd.Timedelta(days=d)

    # Merge reference features with target sales
    merged = ref_data.merge(
        target_data,
        on=["store_nbr", "family", "target_date"],
        how="inner"
    )

    # Add target-date features
    merged["target_day_of_week"] = merged["target_date"].dt.dayofweek
    merged["target_month"] = merged["target_date"].dt.month
    merged["target_day"] = merged["target_date"].dt.day
    merged["target_is_weekend"] = (merged["target_day_of_week"] >= 5).astype(int)
    merged["target_is_payday"] = ((merged["target_day"] == 15) |
                                   merged["target_date"].dt.is_month_end).astype(int)

    # Drop merge helper columns
    drop_cols = ["date", "sales", "target_date"]
    for c in drop_cols:
        if c in merged.columns:
            merged = merged.drop(columns=[c])

    # Merge target TE features (rename TE columns to match target_ prefix)
    te_sf_dow_r = te_sf_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_f_dow_r = te_f_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_sf_month_r = te_sf_month.rename(columns={"month": "target_month"})
    te_sf_dow_recent_r = te_sf_dow_recent.rename(columns={"day_of_week": "target_day_of_week"})

    merged = merged.merge(te_sf_dow_r, on=["store_nbr", "family", "target_day_of_week"], how="left")
    merged = merged.merge(te_sf, on=["store_nbr", "family"], how="left")
    merged = merged.merge(te_f_dow_r, on=["family", "target_day_of_week"], how="left")
    merged = merged.merge(te_sf_month_r, on=["store_nbr", "family", "target_month"], how="left")
    merged = merged.merge(family_mean, on=["family"], how="left")
    merged = merged.merge(te_sf_dow_recent_r, on=["store_nbr", "family", "target_day_of_week"], how="left")

    # TE interaction features
    if "target_te_sf_dow_mean" in merged.columns and "target_te_family_mean" in merged.columns:
        merged["target_te_sf_ratio"] = merged["target_te_sf_dow_mean"] / (merged["target_te_family_mean"] + 1)
    if "target_te_sf_dow_mean" in merged.columns and "target_te_sf_dow_recent_mean" in merged.columns:
        merged["target_te_trend"] = merged["target_te_sf_dow_recent_mean"] / (merged["target_te_sf_dow_mean"] + 1)

    # Replace inf with NaN, then fill
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Define feature columns for this day's model
    exclude = {"target_sales", "store_nbr", "family"}
    day_feat_cols = [c for c in merged.columns if c not in exclude and merged[c].dtype != 'object']

    # Fill NaN in features with 0 for training
    for c in day_feat_cols:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0)

    day_datasets[d] = {
        "data": merged,
        "feat_cols": day_feat_cols,
    }

    print(f"  Day {d:2d}: {len(merged):,} samples, {len(day_feat_cols)} features, "
          f"ref={ref_dates[0].date()}-{ref_dates[-1].date()}")

# ============================================================
# STEP 5: Train day-specific models
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Train day-specific LightGBM models")
print("=" * 60)

cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)

day_models = {}
day_oof_preds = {}
day_cv_scores = {}

for d in sorted(day_datasets.keys()):
    ds = day_datasets[d]
    df = ds["data"]
    feat = ds["feat_cols"]

    X = df[feat].values
    y = df["target_sales"].values

    # Simple train/val split: last 16 days as validation
    # Sort by target_date (we don't have it anymore, but data is roughly ordered)
    # Use a percentage-based split
    val_size = min(16 * 1782, int(len(df) * 0.05))  # ~16 days worth
    train_idx = np.arange(len(df) - val_size)
    val_idx = np.arange(len(df) - val_size, len(df))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Clip target to non-negative
    y_train = np.clip(y_train, 0, None)
    y_val = np.clip(y_val, 0, None)

    # Log1p transform for RMSLE optimization
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # OOF predictions for this day
    oof_pred = model.predict(X_val)
    oof_pred = np.expm1(oof_pred)
    oof_pred = np.clip(oof_pred, 0, None)

    cv_score = rmsle(y_val, oof_pred)
    day_cv_scores[d] = cv_score
    day_models[d] = model

    print(f"  Day {d:2d}: CV={cv_score:.5f}, best_iter={model.best_iteration_}, "
          f"train={len(train_idx):,}, val={len(val_idx):,}")

mean_cv = np.mean(list(day_cv_scores.values()))
print(f"\n  Mean CV RMSLE: {mean_cv:.5f}")

# Save models and feature columns for later use
import pickle
model_dir = Path(__file__).resolve().parent.parent / "outputs" / "r10_models"
model_dir.mkdir(parents=True, exist_ok=True)

for d, model in day_models.items():
    model.booster_.save_model(str(model_dir / f"day_{d:02d}.txt"))
# Save feature columns
with open(model_dir / "feat_cols.pkl", "wb") as f:
    pickle.dump({d: ds["feat_cols"] for d, ds in day_datasets.items()}, f)
print(f"  Models saved to {model_dir}")

# ============================================================
# STEP 6: Create test features from last training date
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Create test features and predict")
print("=" * 60)

# Get features from the last training date for all (store, family) pairs
last_date_features = train_fe[train_fe["date"] == last_train_date].copy()
print(f"  Last training date features: {last_date_features.shape}")

# Zero-sales pairs
sf_stats = train_raw.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))
print(f"  Zero-sales pairs (>99%): {len(zero_pairs)}")

# For each test day, create the feature vector and predict
all_preds = []

for d in range(1, 17):
    if d not in day_models:
        print(f"  Day {d:2d}: No model, skipping")
        continue

    target_date = last_train_date + pd.Timedelta(days=d)
    model = day_models[d]
    feat = day_datasets[d]["feat_cols"]

    # Start with reference features from last training date
    test_data = last_date_features[["store_nbr", "family"] + [c for c in feat if c in last_date_features.columns]].copy()

    # Add target-date features
    test_data["target_day_of_week"] = target_date.dayofweek
    test_data["target_month"] = target_date.month
    test_data["target_day"] = target_date.day
    test_data["target_is_weekend"] = int(target_date.dayofweek >= 5)
    test_data["target_is_payday"] = int(target_date.day == 15 or target_date.is_month_end)

    # Merge target TE features (rename TE columns to match target_ prefix)
    te_sf_dow_r = te_sf_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_f_dow_r = te_f_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_sf_month_r = te_sf_month.rename(columns={"month": "target_month"})
    te_sf_dow_recent_r = te_sf_dow_recent.rename(columns={"day_of_week": "target_day_of_week"})

    test_data = test_data.merge(te_sf_dow_r, on=["store_nbr", "family", "target_day_of_week"], how="left")
    test_data = test_data.merge(te_sf, on=["store_nbr", "family"], how="left")
    test_data = test_data.merge(te_f_dow_r, on=["family", "target_day_of_week"], how="left")
    test_data = test_data.merge(te_sf_month_r, on=["store_nbr", "family", "target_month"], how="left")
    test_data = test_data.merge(family_mean, on=["family"], how="left")
    test_data = test_data.merge(te_sf_dow_recent_r, on=["store_nbr", "family", "target_day_of_week"], how="left")

    # TE interaction features
    if "target_te_sf_dow_mean" in test_data.columns and "target_te_family_mean" in test_data.columns:
        test_data["target_te_sf_ratio"] = test_data["target_te_sf_dow_mean"] / (test_data["target_te_family_mean"] + 1)
    if "target_te_sf_dow_mean" in test_data.columns and "target_te_sf_dow_recent_mean" in test_data.columns:
        test_data["target_te_trend"] = test_data["target_te_sf_dow_recent_mean"] / (test_data["target_te_sf_dow_mean"] + 1)

    test_data = test_data.replace([np.inf, -np.inf], np.nan)

    # Ensure all feature columns exist
    for c in feat:
        if c not in test_data.columns:
            test_data[c] = 0
        elif test_data[c].isna().any():
            test_data[c] = test_data[c].fillna(0)

    # Predict
    X_test = test_data[feat].values
    preds = model.predict(X_test)
    preds = np.expm1(preds)
    preds = np.clip(preds, 0, None)

    test_data["pred"] = preds
    test_data["target_date"] = target_date
    test_data["day"] = d

    all_preds.append(test_data[["store_nbr", "family", "target_date", "day", "pred"]])
    print(f"  Day {d:2d} ({target_date.date()}): pred mean={preds.mean():.2f}, "
          f"max={preds.max():.2f}")

# Combine all day predictions
preds_df = pd.concat(all_preds, ignore_index=True)

# ============================================================
# STEP 7: Create submission
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Create submission")
print("=" * 60)

# Map predictions to test IDs
test_sub = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_sub["target_date"] = test_sub["date"]
sub = test_sub.merge(
    preds_df[["store_nbr", "family", "target_date", "pred"]],
    on=["store_nbr", "family", "target_date"],
    how="left"
)

# Fill missing predictions with 0
sub["sales"] = sub["pred"].fillna(0)

# Zero out known zero-sales pairs
for s, f in zero_pairs:
    mask = (sub["store_nbr"] == s) & (sub["family"] == f)
    sub.loc[mask, "sales"] = 0

# Clip negatives
sub["sales"] = sub["sales"].clip(0, None)
sub.loc[sub["sales"] < 0.1, "sales"] = 0

# Save submission
submission = sub[["id", "sales"]].sort_values("id")
path = SUBMISSIONS / "submission_r10_day_specific.csv"
path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(path, index=False)

print(f"  r10_day_specific: CV={mean_cv:.5f}, mean={submission['sales'].mean():.2f}, "
      f"max={submission['sales'].max():.2f}, zeros={(submission['sales']==0).mean():.4f}")
print(f"  Underprediction ratio: {submission['sales'].mean() / train_raw['sales'].mean():.3f}")

# ============================================================
# STEP 8: Also create geometric blend submission
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: Geometric blend with TE level")
print("=" * 60)

# Compute TE level for each test row
test_te = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_te["day_of_week"] = test_te["date"].dt.dayofweek
test_te = test_te.merge(
    te_sf_dow[["store_nbr", "family", "day_of_week", "target_te_sf_dow_mean"]],
    left_on=["store_nbr", "family", "day_of_week"],
    right_on=["store_nbr", "family", "day_of_week"],
    how="left"
)
te_level = test_te["target_te_sf_dow_mean"].values.copy()
te_level = np.clip(te_level, 0.1, None)

model_preds = submission.sort_values("id")["sales"].values.copy()

for alpha in [0.01, 0.02, 0.05, 0.10, 0.20]:
    geo = np.expm1(alpha * np.log1p(np.clip(model_preds, 0, None)) + (1-alpha) * np.log1p(te_level))
    geo = np.clip(geo, 0, None)
    # Apply zero pairs
    for s, f in zero_pairs:
        mask = (test_te["store_nbr"] == s) & (test_te["family"] == f)
        geo[mask.values] = 0
    geo[geo < 0.1] = 0
    sub_geo = pd.DataFrame({"id": submission["id"].values, "sales": geo})
    name = f"r10_dayspec_geo_{int(alpha*100):02d}_{int((1-alpha)*100)}"
    path_geo = SUBMISSIONS / f"submission_{name}.csv"
    sub_geo.to_csv(path_geo, index=False)
    print(f"  {name}: mean={geo.mean():.2f}, max={geo.max():.2f}, "
          f"zeros={(geo==0).mean():.4f}")

print(f"\nTotal: {time.time() - start:.1f}s")
