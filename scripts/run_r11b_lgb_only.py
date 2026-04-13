"""Round 11b: Day-Specific LightGBM Only — Full Data + R11 Improvements

Key changes from R10 (LB=0.39880):
1. Full 4-year training data (not truncated to 12 months)
2. Oil rolling bug fix (from R11)
3. Hierarchical fillna for TE missing values
4. Target-date onpromotion feature
5. Noise feature removal (earthquake, school, day_of_year_sin/cos)
6. Improved post-processing with store-family min-sales thresholds
7. 3-fold time-series CV for more robust evaluation
8. Uses averaged best_iteration from CV for final model
"""
import warnings
warnings.filterwarnings("ignore")
import time
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
log("Round 11b: Day-Specific LightGBM — Full Data + R11 Improvements")
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
# STEP 2: Build features on training data (full data, like R10)
# ============================================================
log("\nSTEP 2: Build features on training data (full 4-year)")

dummy_test = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe, _, feat_cols = build_features(train_raw, dummy_test)
log(f"  {len(feat_cols)} base features built")

# ============================================================
# STEP 3: Noise feature removal
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
# STEP 5: Target encoding computation
# ============================================================
log("\nSTEP 5: Compute target encodings")

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


# ============================================================
# STEP 6: Build day-specific training data (FULL data like R10)
# ============================================================
log("\nSTEP 6: Build day-specific training data (full history)")

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

    day_datasets[d] = {"data": merged, "feat_cols": day_feat_cols}
    log(f"  Day {d:2d}: {len(merged):,} samples, {len(day_feat_cols)} features")

del train_fe

# ============================================================
# STEP 7: Train day-specific LightGBM with 3-fold CV
# ============================================================
log("\n" + "=" * 60)
log("STEP 7: Train day-specific LightGBM with 3-fold CV")
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

mean_cv_all = np.mean(list(day_cv_scores.values()))
log(f"\n  Mean CV RMSLE: {mean_cv_all:.5f}")

# ============================================================
# STEP 8: Create test features and predict
# ============================================================
log("\n" + "=" * 60)
log("STEP 8: Create test features and predict")
log("=" * 60)

dummy_test2 = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe2, _, _ = build_features(train_raw, dummy_test2)
last_date_features = train_fe2[train_fe2["date"] == last_train_date].copy()
del train_fe2
log(f"  Last training date features: {last_date_features.shape}")

# Post-processing: store-family min non-zero sales
sf_min_sales = train_raw[train_raw["sales"] > 0].groupby(["store_nbr", "family"])["sales"].min().reset_index()
sf_min_sales.columns = ["store_nbr", "family", "min_nonzero_sales"]
sf_zero_ratio = sf_stats[["store_nbr", "family", "zr"]].copy()

all_preds = []

for d in range(1, 17):
    if d not in day_models:
        log(f"  Day {d:2d}: No model, skipping")
        continue

    target_date = last_train_date + pd.Timedelta(days=d)
    feat = day_datasets[d]["feat_cols"]
    model = day_models[d]

    test_data = last_date_features[
        ["store_nbr", "family"] + [c for c in feat if c in last_date_features.columns]
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

    test_data["pred"] = preds
    test_data["target_date"] = target_date
    all_preds.append(test_data[["store_nbr", "family", "target_date", "pred"]])
    log(f"  Day {d:2d} ({target_date.date()}): pred mean={preds.mean():.2f}")

preds_df = pd.concat(all_preds, ignore_index=True)

# ============================================================
# STEP 9: Create submission
# ============================================================
log("\n" + "=" * 60)
log("STEP 9: Create submission")
log("=" * 60)

test_sub = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_sub["target_date"] = test_sub["date"]
sub = test_sub.merge(
    preds_df[["store_nbr", "family", "target_date", "pred"]],
    on=["store_nbr", "family", "target_date"], how="left"
)

sub["sales"] = sub["pred"].fillna(0)

# Zero out known zero-sales pairs
for s, f in zero_pairs:
    mask = (sub["store_nbr"] == s) & (sub["family"] == f)
    sub.loc[mask, "sales"] = 0

# Improved post-processing: min-sales threshold
sub = sub.merge(sf_min_sales, on=["store_nbr", "family"], how="left")
sub = sub.merge(sf_zero_ratio, on=["store_nbr", "family"], how="left")
low_pred_mask = (
    (sub["sales"] > 0)
    & (sub["sales"] < sub["min_nonzero_sales"] * 0.5)
    & (sub["zr"] > 0.5)
)
sub.loc[low_pred_mask, "sales"] = 0
sub = sub.drop(columns=["min_nonzero_sales", "zr"])

sub["sales"] = sub["sales"].clip(0, None)

submission = sub[["id", "sales"]].sort_values("id")
path = SUBMISSIONS / "submission_r11b_lgb_full.csv"
path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(path, index=False)

log(f"  r11b_lgb_full: mean={submission['sales'].mean():.2f}, "
    f"max={submission['sales'].max():.2f}, zeros={(submission['sales']==0).mean():.4f}")
log(f"  Underprediction ratio: {submission['sales'].mean() / train_raw['sales'].mean():.3f}")

# ============================================================
# STEP 10: Geometric blend
# ============================================================
log("\n" + "=" * 60)
log("STEP 10: Geometric blend with TE level")
log("=" * 60)

test_te = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_te["day_of_week"] = test_te["date"].dt.dayofweek
test_te = test_te.merge(
    te_sf_dow[["store_nbr", "family", "day_of_week", "target_te_sf_dow_mean"]],
    on=["store_nbr", "family", "day_of_week"], how="left"
)
te_level = np.clip(test_te["target_te_sf_dow_mean"].values, 0.1, None)
model_preds = submission.sort_values("id")["sales"].values.copy()

for alpha in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    geo = np.expm1(
        alpha * np.log1p(np.clip(model_preds, 0, None))
        + (1 - alpha) * np.log1p(te_level)
    )
    geo = np.clip(geo, 0, None)
    for s, f in zero_pairs:
        mask = (test_te["store_nbr"] == s) & (test_te["family"] == f)
        geo[mask.values] = 0
    geo_df = pd.DataFrame({"id": submission["id"].values, "sales": geo})
    geo_df = geo_df.merge(test_raw[["id", "store_nbr", "family"]], on="id", how="left")
    geo_df = geo_df.merge(sf_min_sales, on=["store_nbr", "family"], how="left")
    geo_df = geo_df.merge(sf_zero_ratio, on=["store_nbr", "family"], how="left")
    low_mask = (
        (geo_df["sales"] > 0)
        & (geo_df["sales"] < geo_df["min_nonzero_sales"] * 0.5)
        & (geo_df["zr"] > 0.5)
    )
    geo_df.loc[low_mask, "sales"] = 0

    name = f"r11b_lgb_geo_{int(alpha*100):02d}_{int((1-alpha)*100)}"
    path_geo = SUBMISSIONS / f"submission_{name}.csv"
    geo_df[["id", "sales"]].to_csv(path_geo, index=False)
    log(f"  {name}: mean={geo_df['sales'].mean():.2f}, zeros={(geo_df['sales']==0).mean():.4f}")

# ============================================================
# STEP 11: Save models
# ============================================================
log("\n" + "=" * 60)
log("STEP 11: Save models")
log("=" * 60)

model_dir = Path(__file__).resolve().parent.parent / "outputs" / "r11b_models"
model_dir.mkdir(parents=True, exist_ok=True)

for d, model in day_models.items():
    model.booster_.save_model(str(model_dir / f"day_{d:02d}.txt"))

with open(model_dir / "feat_cols.pkl", "wb") as f:
    pickle.dump({d: ds["feat_cols"] for d, ds in day_datasets.items()}, f)

with open(model_dir / "cv_scores.pkl", "wb") as f:
    pickle.dump(day_cv_scores, f)

log(f"  Models saved to {model_dir}")

# ============================================================
# Summary
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)
log(f"  Mean CV RMSLE: {mean_cv_all:.5f}")
log(f"  R10 baseline CV: 0.407")
log(f"  R10 baseline LB: 0.39880")
log(f"  Target LB: < 0.39880")
log(f"\n  Total: {time.time() - start:.1f}s")
