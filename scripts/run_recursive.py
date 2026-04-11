"""Round 5: Recursive Prediction

Problem: ffill approach causes CV-LB gap (0.375 vs 1.86).
For 16-day test period, lag_1 is stale from day 2 onwards, rolling features degrade.

Solution: Multi-pass recursive prediction:
1. Predict with ffill features (Pass 0)
2. Replace ffill sales with predictions → rebuild features → re-predict (Pass 1)
3. Repeat until convergence

The key insight: ffill propagates the LAST training day's sales value to all 16 test days.
This means lag_1 for day 2 = last training sales (wrong), but it should = day 1's prediction.
Recursive prediction fixes this by using predictions as lag inputs for subsequent days.
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
from src.features.time_features import add_time_features
from src.features.lag_features import add_lag_features
from src.features.external_features import add_external_features
from src.models.gbdt import train_lightgbm, predict_cv_models
from src.config import ModelConfig, CVConfig, SUBMISSIONS, FeatureConfig


def build_features_with_test_sales(train_orig, test_orig, test_sales, cfg=None):
    """Build features for test set using predicted sales values.

    Same as build_features but uses provided test_sales instead of NaN→ffill.
    This allows recursive prediction: feed predictions back as sales for lag features.
    """
    if cfg is None:
        cfg = FeatureConfig()

    train = train_orig.copy()
    test = test_orig.copy()

    train["is_train"] = True
    test["is_train"] = False
    test["sales"] = test_sales

    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Ffill/bfill for any remaining NaN (shouldn't be any with predicted sales)
    df["sales"] = df.groupby(["store_nbr", "family"])["sales"].ffill()
    df["sales"] = df.groupby(["store_nbr", "family"])["sales"].bfill()
    df["sales"] = df["sales"].fillna(0)

    # Build features (same pipeline as builder.py)
    df = add_time_features(df)
    df = add_lag_features(df, cfg)
    df = add_external_features(df)

    # Define feature columns (same exclude set as builder.py)
    exclude_cols = {
        "id", "date", "sales", "is_train",
        "family", "city", "state", "type",
        "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
        "holiday_type_national", "holiday_type_regional", "holiday_type_local",
    }
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    # Extract test features only
    test_out = df[~df["is_train"]].drop(columns=["is_train"]).reset_index(drop=True)

    return test_out, feature_columns


start = time.time()

print("=" * 60)
print("Round 5: Recursive Prediction")
print("=" * 60)

# 1. Load data
print("\nSTEP 1: Load data")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# 2. Build features with ffill (baseline)
print("\nSTEP 2: Build features (ffill baseline)")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

# 3. Clean
print("\nSTEP 3: Clean")
lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# 4. Train LightGBM
print("\nSTEP 4: Train LightGBM")
cfg = ModelConfig(
    learning_rate=0.01, n_estimators=3000,
    num_leaves=64, min_child_samples=30,
)
cv_cfg = CVConfig(n_folds=5, val_days=16)
result = train_lightgbm(train_clean, feat_cols, cfg=cfg, cv_cfg=cv_cfg)

# 5. Predict: ffill (baseline)
print("\nSTEP 5: Predict with ffill (baseline)")
preds_ffill = predict_cv_models(result["cv_models"], test_fe, feat_cols)
print(f"  Ffill: mean={preds_ffill.mean():.2f}, max={preds_ffill.max():.2f}")

# 6. Predict: recursive (3 passes)
print("\nSTEP 6: Recursive prediction")
N_PASSES = 3
current_preds = preds_ffill.copy()

for pass_idx in range(N_PASSES):
    t0 = time.time()
    print(f"\n  Pass {pass_idx + 1}/{N_PASSES}")

    # Rebuild features with current predictions as test sales
    test_fe_new, feat_cols_new = build_features_with_test_sales(
        train, test, current_preds
    )

    # Verify feature columns match
    missing = set(feat_cols) - set(feat_cols_new)
    if missing:
        print(f"    WARNING: missing features: {missing}")

    # Predict using original feature column order
    new_preds = predict_cv_models(result["cv_models"], test_fe_new, feat_cols)
    new_preds = np.clip(new_preds, 0, None)

    # Convergence check
    diff = np.mean(np.abs(new_preds - current_preds))
    rel_diff = diff / (np.mean(current_preds) + 1e-8)
    print(f"    Mean={new_preds.mean():.2f}, Max={new_preds.max():.2f}")
    print(f"    Abs diff={diff:.4f}, Rel diff={rel_diff:.4f}")
    print(f"    Time: {time.time()-t0:.1f}s")

    current_preds = new_preds

preds_recursive = current_preds

# 7. Post-processing and save
print("\nSTEP 7: Post-processing and save")
sf_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))

test_info = test_fe[["store_nbr", "family"]].copy()

for name, preds in [("r5_ffill", preds_ffill), ("r5_recursive", preds_recursive)]:
    p = preds.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0

    sub = pd.DataFrame({"id": test_fe["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)

    cv_val = result["mean_cv"]
    print(f"  {name}: CV={cv_val:.5f}, mean={p.mean():.2f}, "
          f"max={p.max():.2f}, zeros={(p == 0).mean():.4f}")

print(f"\nTotal: {time.time() - start:.1f}s")
