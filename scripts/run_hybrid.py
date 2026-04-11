"""Round 3b: Hybrid — per-family for large, global for small

Per-family LB=2.10 (worse than global 1.86). Root cause: small families
(BOOKS, HARDWARE, etc.) have ~7万 rows but high CV variance, overfitting.
Fix: large families use per-family models, small families fall back to global.
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
print("Round 3b: Hybrid Per-Family + Global")
print("=" * 60)

# 1. Load data
print("\nSTEP 1: Load data")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# 2. Feature engineering
print("\nSTEP 2: Features")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

# 3. Clean
lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# 4. Train global model
print("\nSTEP 3: Global model")
global_result = train_lightgbm(train_clean, feat_cols)
print(f"  Global CV: {global_result['mean_cv']:.5f}")

# 5. Identify large vs small families
family_means = train_clean.groupby("family")["sales"].mean().sort_values(ascending=False)
LARGE_THRESHOLD = 50
large_families = family_means[family_means > LARGE_THRESHOLD].index.tolist()
small_families = family_means[family_means <= LARGE_THRESHOLD].index.tolist()
print(f"\nSTEP 4: Family split")
print(f"  Large families (>{LARGE_THRESHOLD}): {len(large_families)}")
print(f"  Small families (<={LARGE_THRESHOLD}): {len(small_families)}")

# 6. Train per-family for large families
print("\nSTEP 5: Per-family training (large families only)")
cv_cfg = CVConfig(n_folds=5, val_days=16)
all_test_preds = np.zeros(len(test_fe))

# Global predictions for ALL test rows first
global_preds = predict_cv_models(
    global_result["cv_models"], test_fe, feat_cols, log_transform=True
)

# Start with global predictions
all_test_preds = global_preds.copy()

# Override with per-family for large families
for i, family in enumerate(large_families):
    t0 = time.time()
    train_fam = train_clean[train_clean["family"] == family].copy()
    test_fam = test_fe[test_fe["family"] == family].copy()
    mean_sales = train_fam["sales"].mean()

    if mean_sales > 500:
        cfg = ModelConfig(
            num_leaves=64, min_child_samples=50,
            reg_alpha=0.3, reg_lambda=0.3,
        )
    else:
        cfg = ModelConfig(
            num_leaves=64, min_child_samples=30,
            reg_alpha=0.1, reg_lambda=0.1,
        )

    result = train_lightgbm(train_fam, feat_cols, cfg=cfg, cv_cfg=cv_cfg)
    test_pred = predict_cv_models(
        result["cv_models"], test_fam, feat_cols, log_transform=True
    )

    test_idx = test_fam.index.values
    all_test_preds[test_idx] = test_pred

    elapsed = time.time() - t0
    print(f"  [{i+1:2d}/{len(large_families)}] {family:30s}: "
          f"CV={result['mean_cv']:.5f}, mean={mean_sales:.0f}, {elapsed:.0f}s")

# 7. Post-processing
print("\nSTEP 6: Post-processing")
sf_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_stats["zero_rate"] = sf_stats["n_zeros"] / sf_stats["n_total"]
zero_pairs = set()
for _, row in sf_stats[sf_stats["zero_rate"] >= 0.99].iterrows():
    zero_pairs.add((row["store_nbr"], row["family"]))

test_info = test_fe[["store_nbr", "family"]].copy()
n_zero = 0
for s, f in zero_pairs:
    mask = (test_info["store_nbr"] == s) & (test_info["family"] == f)
    n_zero += mask.sum()
    all_test_preds[mask.values] = 0
print(f"  Zero-sales forced: {n_zero} rows")

n_small = (all_test_preds < 0.1).sum()
all_test_preds[all_test_preds < 0.1] = 0
print(f"  Small truncation: {n_small} rows")

# 8. Save
print("\nSTEP 7: Save")
submission = pd.DataFrame({
    "id": test_fe["id"].values,
    "sales": all_test_preds,
})
output_path = SUBMISSIONS / "submission_hybrid.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_path, index=False)
print(f"  Saved: {output_path}")
print(f"  Sales: range=[{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
print(f"  Mean: {submission['sales'].mean():.2f}")
print(f"  Zero rate: {(submission['sales'] == 0).mean():.4f}")
print(f"\nTotal time: {time.time()-start:.1f}s")
