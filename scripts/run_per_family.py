"""Round 3: Per-family LightGBM models

NotebookLM TOP推荐: 按商品family独立训练模型
原因: 不同family的销售模式差异极大 (GROCERY I日均数百 vs BOOKS接近零)
      全局模型被大销量family主导,小销量family预测不准

策略: 33个family各训练一个LightGBM, 用各自最优的超参数
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
from src.models.gbdt import train_lightgbm, predict_cv_models, time_series_split
from src.utils.metrics import rmsle
from src.config import ModelConfig, CVConfig, SUBMISSIONS

start = time.time()

print("=" * 60)
print("Round 3: Per-Family LightGBM")
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

# 4. 识别零销量family-store组合
sf_zero_stats = train_clean.groupby(["store_nbr", "family"]).agg(
    total_sales=("sales", "sum"),
    mean_sales=("sales", "mean"),
    n_zeros=("sales", lambda x: (x == 0).sum()),
    n_total=("sales", "count"),
).reset_index()
sf_zero_stats["zero_rate"] = sf_zero_stats["n_zeros"] / sf_zero_stats["n_total"]
zero_pairs = set()
for _, row in sf_zero_stats[sf_zero_stats["zero_rate"] >= 0.99].iterrows():
    zero_pairs.add((row["store_nbr"], row["family"]))

# 5. Per-family训练
print("\nSTEP 4: Per-family LightGBM训练")
families = sorted(train_clean["family"].unique())
print(f"  {len(families)} families")

cv_cfg = CVConfig(n_folds=5, val_days=16)
all_oof_preds = np.zeros(len(train_clean))
all_test_preds = np.zeros(len(test_fe))

family_results = {}

for i, family in enumerate(families):
    t0 = time.time()

    # 筛选该family的数据
    train_fam = train_clean[train_clean["family"] == family].copy()
    test_fam = test_fe[test_fe["family"] == family].copy()

    if len(train_fam) < 100:
        print(f"  [{i+1:2d}/{len(families)}] {family:30s}: 跳过 (数据量={len(train_fam)})")
        continue

    # 对大销量family用更强的正则化
    mean_sales = train_fam["sales"].mean()
    if mean_sales > 500:
        cfg = ModelConfig(
            num_leaves=64, min_child_samples=50,
            reg_alpha=0.3, reg_lambda=0.3,
            learning_rate=0.005, n_estimators=3000,
        )
    elif mean_sales > 50:
        cfg = ModelConfig(
            num_leaves=64, min_child_samples=30,
            reg_alpha=0.1, reg_lambda=0.1,
            learning_rate=0.005, n_estimators=3000,
        )
    else:
        cfg = ModelConfig(
            num_leaves=32, min_child_samples=20,
            reg_alpha=0.5, reg_lambda=0.5,
            learning_rate=0.01, n_estimators=2000,
        )

    result = train_lightgbm(train_fam, feat_cols, cfg=cfg, cv_cfg=cv_cfg)

    # 收集OOF预测
    oof_idx = train_fam.index.values
    all_oof_preds[oof_idx] = result["oof_preds"]

    # 收集test预测
    test_idx = test_fam.index.values
    test_pred = predict_cv_models(result["cv_models"], test_fam, feat_cols, log_transform=True)
    all_test_preds[test_idx] = test_pred

    family_results[family] = {
        "mean_cv": result["mean_cv"],
        "mean_sales": mean_sales,
        "n_train": len(train_fam),
    }

    elapsed = time.time() - t0
    print(f"  [{i+1:2d}/{len(families)}] {family:30s}: CV={result['mean_cv']:.5f}, "
          f"mean_sales={mean_sales:8.2f}, n={len(train_fam):6d}, {elapsed:.0f}s")

# 6. 总体OOF评估
print("\n" + "=" * 60)
print("STEP 5: 总体OOF评估")

oof_valid_mask = all_oof_preds != 0
y_true = train_clean["sales"].values

# 使用所有有OOF预测的行
overall_rmsle = rmsle(y_true[oof_valid_mask], all_oof_preds[oof_valid_mask])
print(f"  Overall OOF RMSLE (valid rows): {overall_rmsle:.5f}")
print(f"  Valid rows: {oof_valid_mask.sum()} / {len(oof_valid_mask)}")

# 7. 后处理
print("\n" + "=" * 60)
print("STEP 6: 后处理")

# 零销量组合归零
test_info = test_fe[["store_nbr", "family"]].copy()
n_zero_forced = 0
for store, family in zero_pairs:
    mask = (test_info["store_nbr"] == store) & (test_info["family"] == family)
    n_zero_forced += mask.sum()
    all_test_preds[mask.values] = 0
print(f"  零销量组合归零: {n_zero_forced} 行")

# 微小预测截断
n_small = (all_test_preds < 0.1).sum()
all_test_preds[all_test_preds < 0.1] = 0
print(f"  微小截断(<0.1→0): {n_small} 行")

# 8. 保存提交
print("\n" + "=" * 60)
print("STEP 7: 保存提交")

submission = pd.DataFrame({
    "id": test_fe["id"].values,
    "sales": all_test_preds,
})
output_path = SUBMISSIONS / "submission_per_family.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_path, index=False)

print(f"  保存: {output_path}")
print(f"  Sales range: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
print(f"  Sales mean: {submission['sales'].mean():.2f}")
print(f"  零值占比: {(submission['sales'] == 0).mean():.4f}")

# 9. Per-family CV汇总
print("\n" + "=" * 60)
print("Per-Family CV汇总 (sorted by CV)")
sorted_results = sorted(family_results.items(), key=lambda x: x[1]["mean_cv"])
for name, res in sorted_results[:5]:
    print(f"  BEST  {name:30s}: CV={res['mean_cv']:.5f}, mean_sales={res['mean_sales']:.2f}")
for name, res in sorted_results[-5:]:
    print(f"  WORST {name:30s}: CV={res['mean_cv']:.5f}, mean_sales={res['mean_sales']:.2f}")

print(f"\n总用时: {time.time()-start:.1f}s")
