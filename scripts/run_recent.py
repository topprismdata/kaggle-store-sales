"""Round 4: Feature ablation — safe lags vs all lags

Experiments:
A: Full data + all 69 features (baseline)
B: Full data + safe features only (remove lag_1, lag_7, lag_14 and related rolling)

Goal: Understand if short-lag features features hurt test generalization.
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
print("Round 4: Feature ablation")
print("=" * 60)

# 1. Load and build features
print("\nSTEP 1: Load + features")
data = load_raw_data()
train, test = merge_all_tables(data)
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

# 2. Clean
print("\nSTEP 2: Clean")
lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# 3. Identify short-lag features (lag < 16 = within prediction window)
short_lag_cols = []
for c in feat_cols:
    if "_lag_1" in c or "_lag_7" in c or "_lag_14" in c:
        short_lag_cols.append(c)

safe_cols = [c for c in feat_cols if c not in short_lag_cols]
print(f"  Short-lag features (remove): {len(short_lag_cols)}")
for c in short_lag_cols:
    print(f"    {c}")
print(f"  Safe features: {len(safe_cols)}")

# 4. Train
cfg = ModelConfig(learning_rate=0.01, n_estimators=3000, num_leaves=64, min_child_samples=30)
cv_cfg = CVConfig(n_folds=5, val_days=16)

print("\n" + "=" * 60)
print("Experiment A: All features (baseline)")
print("=" * 60)
result_a = train_lightgbm(train_clean, feat_cols, cfg=cfg, cv_cfg=cv_cfg)

print("\n" + "=" * 60)
print("Experiment B: Safe features only")
print("=" * 60)
result_b = train_lightgbm(train_clean, safe_cols, cfg=cfg, cv_cfg=cv_cfg)

# 5. Predict
print("\nSTEP 5: Predict")
preds_a = predict_cv_models(result_a["cv_models"], test_fe, feat_cols)
preds_b = predict_cv_models(result_b["cv_models"], test_fe, safe_cols)
print(f"  A (all):  mean={preds_a.mean():.2f}, max={preds_a.max():.2f}")
print(f"  B (safe): mean={preds_b.mean():.2f}, max={preds_b.max():.2f}")

# 6. Save submissions
print("\nSTEP 6: Save with post-processing")
sf_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zr"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, r in sf_stats[sf_stats["zr"] >= 0.99].iterrows():
    zero_pairs.add((r["store_nbr"], r["family"]))

test_info = test_fe[["store_nbr", "family"]].copy()

for name, preds in [("r4_all", preds_a), ("r4_safe", preds_b)]:
    p = preds.copy()
    for s, f in zero_pairs:
        mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
        p[mask.values] = 0
    p[p < 0.1] = 0

    sub = pd.DataFrame({"id": test_fe["id"].values, "sales": p})
    path = SUBMISSIONS / f"submission_{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    cv_a = result_a["mean_cv"] if "all" in name else result_b["mean_cv"]
    print(f"  {name}: CV={cv_a:.5f}, mean={p.mean():.2f}, zeros={(p==0).mean():.4f}")

print(f"\nTotal: {time.time()-start:.1f}s")
