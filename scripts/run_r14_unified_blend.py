"""Round 14: Unified Model + Day-Specific Blend

Strategy from 1st place solution: final = ratio * day_specific + (1-ratio) * unified

- Day-specific model: same as R13 (16 separate models, one per day offset)
- Unified model: one single LightGBM model trained on ALL day offsets combined,
  with target_day_offset (1-16) as an additional feature
- The blend captures complementary strengths:
  day-specific handles horizon-specific patterns,
  unified handles cross-day patterns

Memory control: subsample to max ~12M rows for unified training data
by using every 4th day offset (1,5,9,13) and/or last 12 months of data.

Base: R13 (YoY features, 3-fold CV, noise removal, hierarchical fillna)
Post-processing: R10 simple (< 0.1 -> 0), zero pairs
"""
import warnings
warnings.filterwarnings("ignore")
import time
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
import sys
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, merge_all_tables
from src.features.builder import build_features
from src.config import ModelConfig, CVConfig, SUBMISSIONS
from src.utils.metrics import rmsle


def log(msg=""):
    print(msg, flush=True)


start = time.time()

log("=" * 60)
log("Round 14: Unified Model + Day-Specific Blend")
log("=" * 60)

# ============================================================
# STEP 1: Load data
# ============================================================
log("\nSTEP 1: Load data")
data = load_raw_data()
train_raw, test_raw = merge_all_tables(data)
log(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

test_dates = sorted(test_raw["date"].unique())
log(f"  Test dates: {test_dates[0].date()} to {test_dates[-1].date()} ({len(test_dates)} days)")
last_train_date = train_raw["date"].max()
log(f"  Last train date: {last_train_date.date()}")

# ============================================================
# STEP 2: Build features on training data (full 4-year, like R13)
# ============================================================
log("\nSTEP 2: Build features on training data (full 4-year)")

dummy_test = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe, _, feat_cols = build_features(train_raw, dummy_test)
log(f"  {len(feat_cols)} base features built")

# ============================================================
# STEP 3: Noise feature removal (same as R13)
# ============================================================
log("\nSTEP 3: Remove noise features")

NOISE_FEATURES = {
    "days_since_earthquake", "is_post_earthquake", "is_school_start",
    "day_of_year_sin", "day_of_year_cos",
}

# ============================================================
# STEP 4: Zero-sales pair detection
# ============================================================
log("\nSTEP 4: Detect zero-sales pairs")

sf_stats = train_raw.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))
log(f"  Zero-sales pairs (>99%): {len(zero_pairs)}")

# ============================================================
# STEP 5: Compute YoY features
# ============================================================
log("\nSTEP 5: Compute YoY 364-day features")

# Build a lookup table: (store_nbr, family, date) -> sales
sales_lookup = train_raw[["store_nbr", "family", "date", "sales"]].copy()
sales_lookup["date"] = pd.to_datetime(sales_lookup["date"])

# For 7-day average, precompute rolling 7-day mean per (store, family)
sales_sorted = sales_lookup.sort_values(["store_nbr", "family", "date"]).copy()
sales_sorted["sales_7d"] = (
    sales_sorted.groupby(["store_nbr", "family"])["sales"]
    .shift(1)
    .rolling(7, min_periods=1)
    .mean()
    .values
)

# Build date-offset versions for merging
# yoy_sales_364: sales from exactly 364 days ago
yoy_364 = sales_sorted[["store_nbr", "family", "date", "sales"]].copy()
yoy_364["ref_date"] = yoy_364["date"] + pd.Timedelta(days=364)
yoy_364 = yoy_364.rename(columns={"sales": "yoy_sales_364"})
yoy_364 = yoy_364[["store_nbr", "family", "ref_date", "yoy_sales_364"]]

# yoy_sales_364_7d_avg: 7-day rolling mean ending at t-364
yoy_364_7d = sales_sorted[["store_nbr", "family", "date", "sales_7d"]].copy()
yoy_364_7d["ref_date"] = yoy_364_7d["date"] + pd.Timedelta(days=364)
yoy_364_7d = yoy_364_7d.rename(columns={"sales_7d": "yoy_sales_364_7d_avg"})
yoy_364_7d = yoy_364_7d[["store_nbr", "family", "ref_date", "yoy_sales_364_7d_avg"]]

del sales_sorted
gc.collect()

log(f"  YoY lookup tables built: {len(yoy_364):,} rows (364d), {len(yoy_364_7d):,} rows (364d-7d)")

# ============================================================
# STEP 6: Target encoding computation (same as R13)
# ============================================================
log("\nSTEP 6: Compute target encodings")

train_sales = train_raw[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek
train_sales["month"] = train_sales["date"].dt.month

te_sf_dow = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median"]
).reset_index()
te_sf_dow.columns = [
    "store_nbr", "family", "day_of_week",
    "target_te_sf_dow_mean", "target_te_sf_dow_std", "target_te_sf_dow_median"
]

te_sf = train_sales.groupby(["store_nbr", "family"])["sales"].agg(["mean", "std"]).reset_index()
te_sf.columns = ["store_nbr", "family", "target_te_sf_mean", "target_te_sf_std"]

te_f_dow = train_sales.groupby(["family", "day_of_week"])["sales"].agg(["mean", "std"]).reset_index()
te_f_dow.columns = ["family", "day_of_week", "target_te_f_dow_mean", "target_te_f_dow_std"]

te_sf_month = train_sales.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_month.columns = [
    "store_nbr", "family", "month", "target_te_sf_month_mean", "target_te_sf_month_std"
]

family_mean = train_sales.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "target_te_family_mean"]

recent_3m = train_sales[train_sales["date"] >= "2017-05-01"]
te_sf_dow_recent = recent_3m.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_dow_recent.columns = [
    "store_nbr", "family", "day_of_week",
    "target_te_sf_dow_recent_mean", "target_te_sf_dow_recent_std"
]

del train_sales, recent_3m
gc.collect()

log(f"  TE features computed")


def merge_te_features(df):
    """Merge TE features and add interaction features."""
    te_sf_dow_r = te_sf_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_f_dow_r = te_f_dow.rename(columns={"day_of_week": "target_day_of_week"})
    te_sf_month_r = te_sf_month.rename(columns={"month": "target_month"})
    te_sf_dow_recent_r = te_sf_dow_recent.rename(columns={"day_of_week": "target_day_of_week"})

    df = df.merge(te_sf_dow_r, on=["store_nbr", "family", "target_day_of_week"], how="left")
    df = df.merge(te_sf, on=["store_nbr", "family"], how="left")
    df = df.merge(te_f_dow_r, on=["family", "target_day_of_week"], how="left")
    df = df.merge(te_sf_month_r, on=["store_nbr", "family", "target_month"], how="left")
    df = df.merge(family_mean, on=["family"], how="left")
    df = df.merge(te_sf_dow_recent_r, on=["store_nbr", "family", "target_day_of_week"], how="left")

    df["target_te_sf_ratio"] = df["target_te_sf_dow_mean"] / (df["target_te_family_mean"] + 1)
    df["target_te_trend"] = df["target_te_sf_dow_recent_mean"] / (df["target_te_sf_dow_mean"] + 1)

    return df


def hierarchical_fillna(df, te_cols):
    """Hierarchical fillna: sf_dow -> sf -> f_dow -> family mean."""
    sf_dow_mean_cols = [c for c in te_cols if "sf_dow" in c and "mean" in c]

    for col in sf_dow_mean_cols:
        missing_mask = df[col].isna()
        if missing_mask.any():
            indicator_col = col.replace("_mean", "_missing")
            df[indicator_col] = missing_mask.astype(int)
            df.loc[missing_mask, col] = df.loc[missing_mask, "target_te_sf_mean"]
            still_missing = df[col].isna()
            if still_missing.any():
                df.loc[still_missing, col] = df.loc[still_missing, "target_te_f_dow_mean"]
            still_missing = df[col].isna()
            if still_missing.any():
                df.loc[still_missing, col] = df.loc[still_missing, "target_te_family_mean"]

    for col in te_cols:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


def merge_yoy_features(df, date_col="date"):
    """Merge YoY features onto df using date_col as the reference date."""
    df = df.merge(
        yoy_364,
        left_on=["store_nbr", "family", date_col],
        right_on=["store_nbr", "family", "ref_date"],
        how="left",
    )
    if "ref_date" in df.columns:
        df = df.drop(columns=["ref_date"])

    df = df.merge(
        yoy_364_7d,
        left_on=["store_nbr", "family", date_col],
        right_on=["store_nbr", "family", "ref_date"],
        how="left",
    )
    if "ref_date" in df.columns:
        df = df.drop(columns=["ref_date"])

    # Fill NaN YoY features
    sf_mean_sales = train_raw.groupby(["store_nbr", "family"])["sales"].mean().reset_index()
    sf_mean_sales.columns = ["store_nbr", "family", "sf_mean_sales"]

    df = df.merge(sf_mean_sales, on=["store_nbr", "family"], how="left")

    for yoy_col in ["yoy_sales_364", "yoy_sales_364_7d_avg"]:
        if yoy_col in df.columns:
            missing = df[yoy_col].isna()
            if missing.any():
                df.loc[missing, yoy_col] = df.loc[missing, "sf_mean_sales"]

    df = df.drop(columns=["sf_mean_sales"])

    # Compute YoY ratio features
    if "sales_lag_1" in df.columns and "yoy_sales_364" in df.columns:
        df["yoy_ratio_1y"] = df["sales_lag_1"] / (df["yoy_sales_364"] + 1)

    if "sales_roll_mean_7" in df.columns and "yoy_sales_364_7d_avg" in df.columns:
        df["yoy_ratio_7d"] = df["sales_roll_mean_7"] / (df["yoy_sales_364_7d_avg"] + 1)

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill remaining NaN in YoY features with 0
    yoy_cols = [c for c in df.columns if c.startswith("yoy_")]
    for c in yoy_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(0)

    return df


# ============================================================
# STEP 7: Build day-specific training data with YoY features
# ============================================================
log("\nSTEP 7: Build day-specific training data with YoY features")

train_fe["date"] = pd.to_datetime(train_fe["date"])
all_dates = sorted(train_fe["date"].unique())
log(f"  Training dates: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

test_promo = test_raw[["store_nbr", "family", "date", "onpromotion"]].copy()
test_promo = test_promo.rename(columns={"date": "target_date", "onpromotion": "target_onpromotion"})

train_promo = train_raw[["store_nbr", "family", "date", "onpromotion"]].copy()
train_promo = train_promo.rename(columns={"date": "target_date", "onpromotion": "target_onpromotion"})

LAG_DROP_NAN_DAYS = 60
last_train_idx = len(all_dates) - 1
PAIRS_PER_DAY = 54 * 33  # 1782

day_datasets = {}
for d in range(1, 17):
    min_ref_idx = LAG_DROP_NAN_DAYS
    max_ref_idx = last_train_idx - d

    if max_ref_idx < min_ref_idx:
        log(f"  Day {d:2d}: Not enough training data, skipping")
        continue

    ref_dates = all_dates[min_ref_idx:max_ref_idx + 1]
    target_dates = [t + pd.Timedelta(days=d) for t in ref_dates]

    ref_mask = train_fe["date"].isin(ref_dates)
    select_cols = ["store_nbr", "family", "date", "sales"] + [
        c for c in feat_cols if c not in ("store_nbr", "family", "date", "sales")
    ]
    ref_data = train_fe.loc[ref_mask, select_cols].copy()

    target_data = train_fe.loc[train_fe["date"].isin(target_dates), [
        "store_nbr", "family", "date", "sales"
    ]].copy()
    target_data.columns = ["store_nbr", "family", "target_date", "target_sales"]

    ref_data["target_date"] = ref_data["date"] + pd.Timedelta(days=d)

    merged = ref_data.merge(target_data, on=["store_nbr", "family", "target_date"], how="inner")

    if "onpromotion" in merged.columns:
        merged = merged.drop(columns=["onpromotion"])

    # Target-date features
    merged["target_day_of_week"] = merged["target_date"].dt.dayofweek
    merged["target_month"] = merged["target_date"].dt.month
    merged["target_day"] = merged["target_date"].dt.day
    merged["target_is_weekend"] = (merged["target_day_of_week"] >= 5).astype(int)
    merged["target_is_payday"] = (
        (merged["target_day"] == 15) | merged["target_date"].dt.is_month_end
    ).astype(int)

    # Target-date onpromotion
    merged = merged.merge(
        train_promo[["store_nbr", "family", "target_date", "target_onpromotion"]],
        on=["store_nbr", "family", "target_date"],
        how="left"
    )
    merged["target_onpromotion"] = merged["target_onpromotion"].fillna(0)

    # YoY features -- merge using reference date
    merged = merge_yoy_features(merged, date_col="date")

    merged = merged.drop(columns=["date", "sales", "target_date"])

    # TE features
    merged = merge_te_features(merged)

    merged = merged.replace([np.inf, -np.inf], np.nan)

    exclude = {"target_sales", "store_nbr", "family"}
    day_feat_cols = [c for c in merged.columns if c not in exclude and merged[c].dtype != 'object']
    day_feat_cols = [c for c in day_feat_cols if c not in NOISE_FEATURES]

    te_cols = [c for c in day_feat_cols if c.startswith("target_te_")]
    merged = hierarchical_fillna(merged, te_cols)

    missing_cols = [c for c in merged.columns if c.endswith("_missing") and c not in day_feat_cols]
    day_feat_cols.extend(missing_cols)

    for c in day_feat_cols:
        if c in merged.columns and merged[c].isna().any():
            merged[c] = merged[c].fillna(0)

    yoy_feature_names = [c for c in day_feat_cols if c.startswith("yoy_")]
    log(f"  Day {d:2d}: {len(merged):,} samples, {len(day_feat_cols)} features "
        f"(YoY: {yoy_feature_names})")

    day_datasets[d] = {"data": merged, "feat_cols": day_feat_cols}

del train_fe
gc.collect()

# ============================================================
# STEP 8: Train day-specific LightGBM with 3-fold CV
# ============================================================
log("\n" + "=" * 60)
log("STEP 8: Train day-specific LightGBM with 3-fold CV")
log("=" * 60)

N_CV_FOLDS = 3
VAL_DAYS = 16
cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)

day_models = {}
day_cv_scores = {}

for d in sorted(day_datasets.keys()):
    log(f"\n--- Day {d} ---")
    ds = day_datasets[d]
    df = ds["data"]
    feat = ds["feat_cols"]

    X_all = df[feat].values.astype(np.float32)
    y_all = df["target_sales"].values.copy()
    y_all = np.clip(y_all, 0, None)
    y_all_log = np.log1p(y_all)

    n_samples = len(df)

    # CV splits
    cv_splits = []
    for fold_i in range(N_CV_FOLDS):
        val_end = n_samples - fold_i * VAL_DAYS * PAIRS_PER_DAY
        val_start = val_end - VAL_DAYS * PAIRS_PER_DAY
        if val_start < VAL_DAYS * PAIRS_PER_DAY * 2:
            break
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, val_end)
        cv_splits.append((train_idx, val_idx))

    if len(cv_splits) == 0:
        val_size = min(VAL_DAYS * PAIRS_PER_DAY, int(n_samples * 0.05))
        cv_splits = [(np.arange(n_samples - val_size), np.arange(n_samples - val_size, n_samples))]

    log(f"  {len(cv_splits)} CV folds, {n_samples:,} samples")

    fold_scores = []
    best_iters = []

    for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train_log = y_all_log[train_idx]
        y_val_log = y_all_log[val_idx]
        y_val_orig = y_all[val_idx]

        model = lgb.LGBMRegressor(
            objective="regression", metric="rmse",
            learning_rate=cfg.learning_rate,
            n_estimators=cfg.n_estimators,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        val_pred = np.expm1(model.predict(X_val))
        val_pred = np.clip(val_pred, 0, None)

        score = rmsle(y_val_orig, val_pred)
        fold_scores.append(score)
        best_iters.append(model.best_iteration_)

    mean_cv = np.mean(fold_scores)
    avg_best_iter = int(np.mean(best_iters))

    log(f"  CV={mean_cv:.5f} (folds: {', '.join(f'{s:.5f}' for s in fold_scores)}), "
        f"avg_iter={avg_best_iter}")

    # Train final model on all data with averaged best_iteration
    final_model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        learning_rate=cfg.learning_rate,
        n_estimators=avg_best_iter,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    final_model.fit(X_all, y_all_log)

    day_models[d] = final_model
    day_cv_scores[d] = mean_cv

    # Free memory
    day_datasets[d]["data"] = None
    del X_all, y_all, y_all_log
    gc.collect()

mean_cv_all = np.mean(list(day_cv_scores.values()))
log(f"\n  Day-specific Mean CV RMSLE: {mean_cv_all:.5f}")

# ============================================================
# STEP 9: Build unified training data (all day offsets combined)
# ============================================================
log("\n" + "=" * 60)
log("STEP 9: Build unified training data (all day offsets combined)")
log("=" * 60)

# Rebuild features for unified model
# We need train_fe again
dummy_test_u = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe_u, _, _ = build_features(train_raw, dummy_test_u)
train_fe_u["date"] = pd.to_datetime(train_fe_u["date"])

# Use the first day's feature columns as a reference (they should all be the same)
ref_feat_cols = day_datasets[1]["feat_cols"]

# Add target_day_offset to the feature set
unified_feat_cols = ref_feat_cols + ["target_day_offset"]

# Subsample for memory: use every 4th day offset AND last 12 months
UNIFIED_DAY_OFFSETS = list(range(1, 17))  # All 16 day offsets
UNIFIED_MAX_ROWS = 12_000_000
# Use last 12 months of reference dates to control memory
UNIFIED_MIN_DATE = pd.Timestamp("2016-08-16")

log(f"  Day offsets for unified model: {UNIFIED_DAY_OFFSETS}")
log(f"  Min reference date: {UNIFIED_MIN_DATE.date()}")
log(f"  Max rows target: {UNIFIED_MAX_ROWS:,}")

unified_parts = []
for d in UNIFIED_DAY_OFFSETS:
    min_ref_idx = LAG_DROP_NAN_DAYS
    max_ref_idx = last_train_idx - d

    if max_ref_idx < min_ref_idx:
        continue

    ref_dates = all_dates[min_ref_idx:max_ref_idx + 1]
    # Apply date filter for memory control
    ref_dates = [rd for rd in ref_dates if rd >= UNIFIED_MIN_DATE]

    if len(ref_dates) == 0:
        continue

    target_dates = [t + pd.Timedelta(days=d) for t in ref_dates]

    ref_mask = train_fe_u["date"].isin(ref_dates)
    select_cols = ["store_nbr", "family", "date", "sales"] + [
        c for c in feat_cols if c not in ("store_nbr", "family", "date", "sales")
    ]
    ref_data = train_fe_u.loc[ref_mask, select_cols].copy()

    target_data = train_fe_u.loc[train_fe_u["date"].isin(target_dates), [
        "store_nbr", "family", "date", "sales"
    ]].copy()
    target_data.columns = ["store_nbr", "family", "target_date", "target_sales"]

    ref_data["target_date"] = ref_data["date"] + pd.Timedelta(days=d)

    merged = ref_data.merge(target_data, on=["store_nbr", "family", "target_date"], how="inner")

    if "onpromotion" in merged.columns:
        merged = merged.drop(columns=["onpromotion"])

    # Target-date features
    merged["target_day_of_week"] = merged["target_date"].dt.dayofweek
    merged["target_month"] = merged["target_date"].dt.month
    merged["target_day"] = merged["target_date"].dt.day
    merged["target_is_weekend"] = (merged["target_day_of_week"] >= 5).astype(int)
    merged["target_is_payday"] = (
        (merged["target_day"] == 15) | merged["target_date"].dt.is_month_end
    ).astype(int)

    # Target-date onpromotion
    merged = merged.merge(
        train_promo[["store_nbr", "family", "target_date", "target_onpromotion"]],
        on=["store_nbr", "family", "target_date"],
        how="left"
    )
    merged["target_onpromotion"] = merged["target_onpromotion"].fillna(0)

    # Key feature for unified model: target_day_offset
    merged["target_day_offset"] = d

    # YoY features
    merged = merge_yoy_features(merged, date_col="date")

    merged = merged.drop(columns=["date", "sales", "target_date"])

    # TE features
    merged = merge_te_features(merged)

    merged = merged.replace([np.inf, -np.inf], np.nan)

    exclude = {"target_sales", "store_nbr", "family"}
    ufeat = [c for c in merged.columns if c not in exclude and merged[c].dtype != 'object']
    ufeat = [c for c in ufeat if c not in NOISE_FEATURES]

    te_cols_u = [c for c in ufeat if c.startswith("target_te_")]
    merged = hierarchical_fillna(merged, te_cols_u)

    missing_cols_u = [c for c in merged.columns if c.endswith("_missing") and c not in ufeat]
    ufeat.extend(missing_cols_u)

    for c in ufeat:
        if c in merged.columns and merged[c].isna().any():
            merged[c] = merged[c].fillna(0)

    # Ensure unified_feat_cols matches
    # (may have extra or missing vs ref_feat_cols; harmonize)
    unified_parts.append(merged)
    log(f"  Day offset {d:2d}: {len(merged):,} rows")

    del ref_data, target_data
    gc.collect()

del train_fe_u
gc.collect()

# Combine all day offsets
log("\n  Concatenating unified training data...")
unified_df = pd.concat(unified_parts, ignore_index=True)
del unified_parts
gc.collect()

log(f"  Unified training data: {unified_df.shape}")

# Subsample if still too large
if len(unified_df) > UNIFIED_MAX_ROWS:
    log(f"  Subsampling from {len(unified_df):,} to {UNIFIED_MAX_ROWS:,} rows...")
    sample_idx = np.random.RandomState(42).choice(
        len(unified_df), UNIFIED_MAX_ROWS, replace=False
    )
    sample_idx.sort()
    unified_df = unified_df.iloc[sample_idx].reset_index(drop=True)
    log(f"  After subsampling: {unified_df.shape}")

# Recompute unified_feat_cols from actual columns
exclude = {"target_sales", "store_nbr", "family"}
unified_feat_cols = [c for c in unified_df.columns if c not in exclude and unified_df[c].dtype != 'object']
unified_feat_cols = [c for c in unified_feat_cols if c not in NOISE_FEATURES]
# Ensure target_day_offset is included
if "target_day_offset" not in unified_feat_cols:
    unified_feat_cols.append("target_day_offset")

log(f"  Unified feature count: {len(unified_feat_cols)}")
log(f"  target_day_offset stats: {unified_df['target_day_offset'].describe().to_dict()}")

# ============================================================
# STEP 10: Train unified LightGBM with 3-fold CV
# ============================================================
log("\n" + "=" * 60)
log("STEP 10: Train unified LightGBM with 3-fold CV")
log("=" * 60)

X_unified = unified_df[unified_feat_cols].values.astype(np.float32)
y_unified = unified_df["target_sales"].values.copy()
y_unified = np.clip(y_unified, 0, None)
y_unified_log = np.log1p(y_unified)

n_unified = len(unified_df)
log(f"  Unified training samples: {n_unified:,}")

# For the unified model, use simple time-based CV
# Sort by store/family for consistent ordering, then use last portions as validation
# Since data comes from multiple day offsets, use the last VAL_DAYS*PAIRS_PER_DAY per day offset
# Simpler approach: just use tail for validation (data is roughly time-ordered)

cv_splits_u = []
val_size_per_fold = VAL_DAYS * PAIRS_PER_DAY * len(UNIFIED_DAY_OFFSETS)
for fold_i in range(N_CV_FOLDS):
    val_end = n_unified - fold_i * val_size_per_fold
    val_start = val_end - val_size_per_fold
    if val_start < val_size_per_fold * 2:
        break
    train_idx = np.arange(0, val_start)
    val_idx = np.arange(val_start, val_end)
    cv_splits_u.append((train_idx, val_idx))

if len(cv_splits_u) == 0:
    val_size = min(val_size_per_fold, int(n_unified * 0.05))
    cv_splits_u = [(np.arange(n_unified - val_size), np.arange(n_unified - val_size, n_unified))]

log(f"  {len(cv_splits_u)} CV folds")

fold_scores_u = []
best_iters_u = []

for fold_i, (train_idx, val_idx) in enumerate(cv_splits_u):
    X_train, X_val = X_unified[train_idx], X_unified[val_idx]
    y_train_log = y_unified_log[train_idx]
    y_val_log = y_unified_log[val_idx]
    y_val_orig = y_unified[val_idx]

    model = lgb.LGBMRegressor(
        objective="regression", metric="rmse",
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        num_leaves=cfg.num_leaves,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    val_pred = np.expm1(model.predict(X_val))
    val_pred = np.clip(val_pred, 0, None)

    score = rmsle(y_val_orig, val_pred)
    fold_scores_u.append(score)
    best_iters_u.append(model.best_iteration_)

    log(f"  Fold {fold_i+1}: CV={score:.5f}, best_iter={model.best_iteration_}")

mean_cv_unified = np.mean(fold_scores_u)
avg_best_iter_u = int(np.mean(best_iters_u))
log(f"  Unified CV={mean_cv_unified:.5f}, avg_iter={avg_best_iter_u}")

# Train final unified model on all data
log("  Training final unified model on all data...")
unified_model = lgb.LGBMRegressor(
    objective="regression", metric="rmse",
    learning_rate=cfg.learning_rate,
    n_estimators=avg_best_iter_u,
    num_leaves=cfg.num_leaves,
    min_child_samples=cfg.min_child_samples,
    subsample=cfg.subsample,
    colsample_bytree=cfg.colsample_bytree,
    random_state=42, verbose=-1, n_jobs=-1,
)
unified_model.fit(X_unified, y_unified_log)

# Log feature importance for target_day_offset and YoY features
importances_u = unified_model.feature_importances_
feat_imp_u = dict(zip(unified_feat_cols, importances_u))
log(f"  target_day_offset importance: {feat_imp_u.get('target_day_offset', 0)}")
for yoy_feat in sorted([c for c in unified_feat_cols if c.startswith("yoy_")]):
    log(f"    {yoy_feat}: importance={feat_imp_u.get(yoy_feat, 0)}")

# Free unified training data
del unified_df, X_unified, y_unified, y_unified_log
gc.collect()

# ============================================================
# STEP 11: Create test features and predict with both models
# ============================================================
log("\n" + "=" * 60)
log("STEP 11: Create test features and predict")
log("=" * 60)

# Rebuild last-date features for test predictions
dummy_test2 = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe2, _, _ = build_features(train_raw, dummy_test2)
last_date_features = train_fe2[train_fe2["date"] == last_train_date].copy()
del train_fe2
gc.collect()
log(f"  Last training date features: {last_date_features.shape}")

# Day-specific predictions
day_preds = {}
for d in range(1, 17):
    if d not in day_models:
        log(f"  Day {d:2d}: No model, skipping")
        continue

    target_date = last_train_date + pd.Timedelta(days=d)
    feat = day_datasets[d]["feat_cols"]
    model = day_models[d]

    test_data = last_date_features[
        ["store_nbr", "family", "date"] + [c for c in feat if c in last_date_features.columns]
    ].copy()

    test_data["target_day_of_week"] = target_date.dayofweek
    test_data["target_month"] = target_date.month
    test_data["target_day"] = target_date.day
    test_data["target_is_weekend"] = int(target_date.dayofweek >= 5)
    test_data["target_is_payday"] = int(target_date.day == 15 or target_date.is_month_end)

    # Target-date onpromotion from test data
    promo_for_day = test_promo[test_promo["target_date"] == target_date]
    test_data = test_data.merge(
        promo_for_day[["store_nbr", "family", "target_onpromotion"]],
        on=["store_nbr", "family"], how="left"
    )
    test_data["target_onpromotion"] = test_data["target_onpromotion"].fillna(0)

    # YoY features
    test_data = merge_yoy_features(test_data, date_col="date")

    # Drop date after YoY merge
    test_data = test_data.drop(columns=["date"])

    # TE features
    test_data = merge_te_features(test_data)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)

    te_cols = [c for c in feat if c.startswith("target_te_")]
    test_data = hierarchical_fillna(test_data, te_cols)

    for c in feat:
        if c not in test_data.columns:
            test_data[c] = 0
        elif test_data[c].isna().any():
            test_data[c] = test_data[c].fillna(0)

    X_test = test_data[feat].values.astype(np.float32)
    preds = np.expm1(model.predict(X_test))
    preds = np.clip(preds, 0, None)

    day_preds[d] = preds
    log(f"  Day {d:2d} day-specific ({target_date.date()}): pred mean={preds.mean():.2f}")

# Unified predictions
unified_preds = {}
for d in range(1, 17):
    target_date = last_train_date + pd.Timedelta(days=d)

    # Build test data using unified_feat_cols
    test_data = last_date_features[
        ["store_nbr", "family", "date"] + [c for c in unified_feat_cols
                                            if c in last_date_features.columns
                                            and c != "target_day_offset"]
    ].copy()

    test_data["target_day_of_week"] = target_date.dayofweek
    test_data["target_month"] = target_date.month
    test_data["target_day"] = target_date.day
    test_data["target_is_weekend"] = int(target_date.dayofweek >= 5)
    test_data["target_is_payday"] = int(target_date.day == 15 or target_date.is_month_end)
    test_data["target_day_offset"] = d

    # Target-date onpromotion from test data
    promo_for_day = test_promo[test_promo["target_date"] == target_date]
    test_data = test_data.merge(
        promo_for_day[["store_nbr", "family", "target_onpromotion"]],
        on=["store_nbr", "family"], how="left"
    )
    test_data["target_onpromotion"] = test_data["target_onpromotion"].fillna(0)

    # YoY features
    test_data = merge_yoy_features(test_data, date_col="date")

    # Drop date after YoY merge
    test_data = test_data.drop(columns=["date"])

    # TE features
    test_data = merge_te_features(test_data)
    test_data = test_data.replace([np.inf, -np.inf], np.nan)

    te_cols_u_test = [c for c in unified_feat_cols if c.startswith("target_te_")]
    test_data = hierarchical_fillna(test_data, te_cols_u_test)

    for c in unified_feat_cols:
        if c not in test_data.columns:
            test_data[c] = 0
        elif test_data[c].isna().any():
            test_data[c] = test_data[c].fillna(0)

    X_test = test_data[unified_feat_cols].values.astype(np.float32)
    preds = np.expm1(unified_model.predict(X_test))
    preds = np.clip(preds, 0, None)

    unified_preds[d] = preds
    log(f"  Day {d:2d} unified    ({target_date.date()}): pred mean={preds.mean():.2f}")

# ============================================================
# STEP 12: Blend and create submissions
# ============================================================
log("\n" + "=" * 60)
log("STEP 12: Blend day-specific + unified and create submissions")
log("=" * 60)

# Build the base test dataframe for submission
test_sub = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_sub["target_date"] = test_sub["date"]

# Build day-specific predictions array (ordered by test row)
day_specific_pred = np.zeros(len(test_sub))
unified_pred = np.zeros(len(test_sub))

for d in range(1, 17):
    target_date = last_train_date + pd.Timedelta(days=d)
    mask = test_sub["target_date"] == target_date

    if d in day_preds:
        day_specific_pred[mask.values] = day_preds[d]
    if d in unified_preds:
        unified_pred[mask.values] = unified_preds[d]

# Save pure day-specific submission (same as R13 baseline)
ds_sales = day_specific_pred.copy()
for s, f in zero_pairs:
    mask = (test_sub["store_nbr"] == s) & (test_sub["family"] == f)
    ds_sales[mask.values] = 0
ds_sales = np.clip(ds_sales, 0, None)
ds_sales[ds_sales < 0.1] = 0

sub_ds = pd.DataFrame({"id": test_sub["id"].values, "sales": ds_sales})
path_ds = SUBMISSIONS / "submission_r14_day_specific.csv"
sub_ds.to_csv(path_ds, index=False)
log(f"  day_specific_only: mean={ds_sales.mean():.2f}, zeros={(ds_sales==0).mean():.4f}")

# Save pure unified submission
u_sales = unified_pred.copy()
for s, f in zero_pairs:
    mask = (test_sub["store_nbr"] == s) & (test_sub["family"] == f)
    u_sales[mask.values] = 0
u_sales = np.clip(u_sales, 0, None)
u_sales[u_sales < 0.1] = 0

sub_u = pd.DataFrame({"id": test_sub["id"].values, "sales": u_sales})
path_u = SUBMISSIONS / "submission_r14_unified_only.csv"
sub_u.to_csv(path_u, index=False)
log(f"  unified_only:      mean={u_sales.mean():.2f}, zeros={(u_sales==0).mean():.4f}")

# Blend at multiple ratios
# ratio=1.0 means 100% day-specific, ratio=0.0 means 100% unified
BLEND_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]

for ratio in BLEND_RATIOS:
    blended = ratio * day_specific_pred + (1 - ratio) * unified_pred

    # Post-processing
    for s, f in zero_pairs:
        mask = (test_sub["store_nbr"] == s) & (test_sub["family"] == f)
        blended[mask.values] = 0
    blended = np.clip(blended, 0, None)
    blended[blended < 0.1] = 0

    name = f"r14_blend_{int(ratio*100)}_{int((1-ratio)*100)}"
    sub_blend = pd.DataFrame({"id": test_sub["id"].values, "sales": blended})
    path_blend = SUBMISSIONS / f"submission_{name}.csv"
    sub_blend.to_csv(path_blend, index=False)
    log(f"  {name}: mean={blended.mean():.2f}, zeros={(blended==0).mean():.4f}")

# ============================================================
# STEP 13: Save models
# ============================================================
log("\n" + "=" * 60)
log("STEP 13: Save models")
log("=" * 60)

model_dir = Path(__file__).resolve().parent.parent / "outputs" / "r14_models"
model_dir.mkdir(parents=True, exist_ok=True)

for d, model in day_models.items():
    model.booster_.save_model(str(model_dir / f"day_{d:02d}.txt"))

unified_model.booster_.save_model(str(model_dir / "unified.txt"))

with open(model_dir / "feat_cols.pkl", "wb") as f:
    pickle.dump({d: ds["feat_cols"] for d, ds in day_datasets.items()}, f)

with open(model_dir / "unified_feat_cols.pkl", "wb") as f:
    pickle.dump(unified_feat_cols, f)

with open(model_dir / "cv_scores.pkl", "wb") as f:
    pickle.dump({"day_specific": day_cv_scores, "unified": mean_cv_unified}, f)

log(f"  Models saved to {model_dir}")

# ============================================================
# Summary
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Experiment: R14 (Unified Model + Day-Specific Blend)")
log(f"  Base: R13 (YoY features, 3-fold CV, noise removal)")
log(f"  Day-specific Mean CV RMSLE: {mean_cv_all:.5f}")
log(f"  Unified Mean CV RMSLE:      {mean_cv_unified:.5f}")
log(f"  Blend ratios tested: {BLEND_RATIOS}")
log(f"  Post-processing: < 0.1 -> 0, zero pairs")
log(f"  R13 best LB: 0.39779")
log(f"  Target: improve on R13 LB=0.39779")
log(f"\n  Submissions saved to: {SUBMISSIONS}")
log(f"    submission_r14_day_specific.csv  (100% day-specific)")
log(f"    submission_r14_unified_only.csv  (100% unified)")
for ratio in BLEND_RATIOS:
    name = f"submission_r14_blend_{int(ratio*100)}_{int((1-ratio)*100)}.csv"
    log(f"    {name}  ({int(ratio*100)}% day-specific + {int((1-ratio)*100)}% unified)")
log(f"\n  Total: {time.time() - start:.1f}s")
