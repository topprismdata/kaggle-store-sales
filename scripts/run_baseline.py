"""生成baseline提交文件的独立脚本"""
import warnings
warnings.filterwarnings("ignore")
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 手动导入
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import load_raw_data, merge_all_tables
from src.features.builder import build_features
from src.models.gbdt import train_lightgbm, generate_submission

start = time.time()

print("Loading data...")
data = load_raw_data()
train, test = merge_all_tables(data)
print(f"  Train: {train.shape}, Test: {test.shape}")

print("Building features...")
train_fe, test_fe, feat_cols = build_features(train, test)
print(f"  {len(feat_cols)} features")

print("Cleaning data...")
lag_roll_cols = [c for c in feat_cols if "lag" in c or "roll" in c or "ewm" in c]
train_clean = train_fe.dropna(subset=lag_roll_cols).reset_index(drop=True)
print(f"  Train clean: {train_clean.shape}")

print("Training LightGBM...")
result = train_lightgbm(train_clean, feat_cols)
print(f"  Mean CV RMSLE: {result['mean_cv']:.5f}")

print("Generating submission...")
submission = generate_submission(result["model"], train_clean, test_fe, feat_cols)
print(f"  Rows: {len(submission)}")
print(f"Total: {time.time()-start:.1f}s")
