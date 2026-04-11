"""Round 1: LightGBM baseline + post-processing

优化记录:
- v1: 原始提交 LB=2.67 (lag NaN cascade bug)
- v2: ffill修复 LB=1.90
- v3: 零销量后处理 + predict_cv_models (本轮)
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
from src.config import SUBMISSIONS

start = time.time()

print("=" * 60)
print("Round 1: LightGBM Baseline (optimized)")
print("=" * 60)

# 1. 数据加载
print("\nSTEP 1: 数据加载")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

# 2. 特征工程
print("\nSTEP 2: 特征工程")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

# 3. 清理
print("\nSTEP 3: 清理")
lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

# 4. 训练
print("\nSTEP 4: 训练 LightGBM")
result = train_lightgbm(train_clean, feat_cols)
print(f"  Mean CV RMSLE: {result['mean_cv']:.5f}")

# 5. 预测 — 使用CV models平均（比final model更稳定）
print("\nSTEP 5: 预测 (CV models average)")
preds = predict_cv_models(result["cv_models"], test_fe, feat_cols, log_transform=True)
print(f"  Pred range: [{preds.min():.2f}, {preds.max():.2f}], mean={preds.mean():.2f}")

# 6. 后处理
print("\nSTEP 6: 后处理")

# 6a. 识别零销量store-family组合
sf_zero_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    total_sales=("sales", "sum"),
    mean_sales=("sales", "mean"),
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_zero_stats["zero_rate"] = sf_zero_stats["n_zeros"] / sf_zero_stats["n_total"]

# 100%零销量的组合 → 强制为0
always_zero = sf_zero_stats[sf_zero_stats["zero_rate"] >= 1.0]
zero_pairs = set(zip(always_zero["store_nbr"], always_zero["family"]))
print(f"  100%零销量组合: {len(zero_pairs)} 个")

# 高零销量率(>95%)的组合 → 也设为0
near_zero = sf_zero_stats[(sf_zero_stats["zero_rate"] > 0.95) & (sf_zero_stats["zero_rate"] < 1.0)]
near_zero_pairs = set(zip(near_zero["store_nbr"], near_zero["family"]))
print(f"  >95%零销量组合: {len(near_zero_pairs)} 个")

# 应用后处理
test_with_info = test_fe.copy()
n_zero_forced = 0
n_near_zero_forced = 0

for _, row in always_zero.iterrows():
    mask = (test_with_info["store_nbr"] == row["store_nbr"]) & (test_with_info["family"] == row["family"])
    n_zero_forced += mask.sum()
    preds[mask.values] = 0

for _, row in near_zero.iterrows():
    mask = (test_with_info["store_nbr"] == row["store_nbr"]) & (test_with_info["family"] == row["family"])
    n_near_zero_forced += mask.sum()
    preds[mask.values] = 0

print(f"  强制归零: {n_zero_forced} 行 (100%零)")
print(f"  强制归零: {n_near_zero_forced} 行 (>95%零)")

# 6b. 微小预测截断
n_small = (preds < 0.1).sum()
preds[preds < 0.1] = 0
print(f"  微小截断(<0.1→0): {n_small} 行")

# 7. 保存
print("\nSTEP 7: 保存提交")
submission = pd.DataFrame({
    "id": test_fe["id"].values,
    "sales": preds,
})
output_path = SUBMISSIONS / "submission_baseline_v3.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_path, index=False)
print(f"  保存: {output_path}")
print(f"  Sales range: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
print(f"  Sales mean: {submission['sales'].mean():.2f}")
print(f"  零值占比: {(submission['sales'] == 0).mean():.4f}")

print(f"\n总用时: {time.time()-start:.1f}s")
