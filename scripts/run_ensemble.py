"""Round 2: 多模型集成训练脚本

流程:
1. 加载数据 + 特征工程 (复用Round 1 pipeline)
2. 训练 LightGBM + XGBoost + CatBoost
3. Hill Climbing / 网格搜索最优集成权重
4. 生成集成提交文件

Bug修复记录:
- OOF预测覆盖率问题: time_series_split每fold只覆盖16天的数据(约7.8%的行)
  其余行的oof_preds为0。集成搜索时必须只使用有OOF预测的行。
  修复: 用 valid_mask 筛选所有模型都有OOF预测的行
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
from src.models.gbdt import (
    train_lightgbm,
    train_xgboost,
    train_catboost,
    predict,
    predict_cv_models,
)
from src.ensemble.blender import hill_climbing, find_optimal_weights_grid
from src.config import SUBMISSIONS

start = time.time()

# ========== 1. 数据加载 + 特征工程 ==========
print("=" * 60)
print("STEP 1: 数据加载与特征工程")
print("=" * 60)

data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

y_true = train_clean["sales"].values

# ========== 2. 训练三种GBDT模型 ==========
print("\n" + "=" * 60)
print("STEP 2: 训练三种GBDT模型")
print("=" * 60)

model_results = {}

print("\n--- LightGBM ---")
t0 = time.time()
lgb_result = train_lightgbm(train_clean, feat_cols)
model_results["lightgbm"] = lgb_result
print(f"  LightGBM 用时: {time.time()-t0:.1f}s")

print("\n--- XGBoost ---")
t0 = time.time()
xgb_result = train_xgboost(train_clean, feat_cols)
model_results["xgboost"] = xgb_result
print(f"  XGBoost 用时: {time.time()-t0:.1f}s")

print("\n--- CatBoost ---")
t0 = time.time()
cb_result = train_catboost(train_clean, feat_cols)
model_results["catboost"] = cb_result
print(f"  CatBoost 用时: {time.time()-t0:.1f}s")

# ========== 3. 模型对比 ==========
print("\n" + "=" * 60)
print("STEP 3: 单模型CV对比")
print("=" * 60)
for name, res in model_results.items():
    print(f"  {name:12s}: Mean CV RMSLE = {res['mean_cv']:.5f}")

# ========== 4. 集成权重搜索 ==========
print("\n" + "=" * 60)
print("STEP 4: 集成权重搜索")
print("=" * 60)

oof_list = [res["oof_preds"] for res in model_results.values()]
model_names = list(model_results.keys())

# 关键修复: OOF预测只有验证fold覆盖的行有值，其余为0
# 只使用被CV覆盖的行进行集成搜索
oof_arr = np.array(oof_list)  # shape: (n_models, n_samples)
for i, name in enumerate(model_names):
    n_nonzero = (oof_arr[i] != 0).sum()
    print(f"  {name}: OOF非零行数={n_nonzero} ({n_nonzero/len(y_true)*100:.1f}%)")
# 所有模型用相同的CV split，所以用任意一个模型的非零行作为mask
valid_mask = oof_arr[0] != 0
print(f"  有效OOF行数: {valid_mask.sum()} / {len(valid_mask)} ({valid_mask.mean()*100:.1f}%)")

oof_valid = [oof[valid_mask] for oof in oof_list]
y_valid = y_true[valid_mask]

# 网格搜索
print("\n--- 网格搜索 ---")
grid_result = find_optimal_weights_grid(oof_valid, y_valid, n_steps=50)

# Hill Climbing
print("\n--- Hill Climbing ---")
hc_result = hill_climbing(oof_valid, y_valid, n_iterations=5000)

# 选择更优的权重
if grid_result["score"] <= hc_result["score"]:
    best_weights = grid_result["weights"]
    best_blend_score = grid_result["score"]
    print(f"\n  使用网格搜索权重 (RMSLE={best_blend_score:.5f})")
else:
    best_weights = hc_result["weights"]
    best_blend_score = hc_result["score"]
    print(f"\n  使用Hill Climbing权重 (RMSLE={best_blend_score:.5f})")

for name, w in zip(model_names, best_weights):
    print(f"    {name:12s}: {w:.4f}")

# ========== 5. 生成集成预测 ==========
print("\n" + "=" * 60)
print("STEP 5: 生成集成预测")
print("=" * 60)

test_preds_list = []
for name, res in model_results.items():
    # 使用CV models平均预测（比final model更稳定）
    pred = predict_cv_models(res["cv_models"], test_fe, feat_cols, log_transform=True)
    test_preds_list.append(pred)
    print(f"  {name}: pred range [{pred.min():.2f}, {pred.max():.2f}], mean={pred.mean():.2f}")

# 加权集成
ensemble_preds = np.zeros(len(test_fe))
for w, pred in zip(best_weights, test_preds_list):
    ensemble_preds += w * pred
ensemble_preds = np.clip(ensemble_preds, 0, None)

print(f"  集成预测: range [{ensemble_preds.min():.2f}, {ensemble_preds.max():.2f}], "
      f"mean={ensemble_preds.mean():.2f}")

# ========== 6. 后处理 + 保存提交 ==========
print("\n" + "=" * 60)
print("STEP 6: 后处理与保存")
print("=" * 60)

ensemble_preds[ensemble_preds < 0.1] = 0

submission = pd.DataFrame({
    "id": test_fe["id"].values,
    "sales": ensemble_preds,
})

output_path = SUBMISSIONS / "submission_ensemble.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(output_path, index=False)
print(f"提交文件已保存: {output_path}")
print(f"  Rows: {len(submission)}")
print(f"  Sales range: [{submission['sales'].min():.2f}, {submission['sales'].max():.2f}]")
print(f"  Sales mean: {submission['sales'].mean():.2f}")
print(f"  零值占比: {(submission['sales'] == 0).mean():.4f}")

print(f"\n总用时: {time.time()-start:.1f}s")
