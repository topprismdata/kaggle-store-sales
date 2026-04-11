"""Round 10: Day-Specific Model Prediction (load saved models)

Quick prediction script that loads saved day-specific models and generates submissions.
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
from src.config import SUBMISSIONS

start = time.time()

print("=" * 60)
print("Round 10: Day-Specific Prediction (from saved models)")
print("=" * 60)

# Load models
model_dir = Path(__file__).resolve().parent.parent / "outputs" / "r10_models"
with open(model_dir / "feat_cols.pkl", "rb") as f:
    day_feat_cols = pickle.load(f)

day_models = {}
for d in range(1, 17):
    model_path = model_dir / f"day_{d:02d}.txt"
    if model_path.exists():
        day_models[d] = lgb.Booster(model_file=str(model_path))

print(f"Loaded {len(day_models)} day-specific models")

# ============================================================
# STEP 1: Load data and build features
# ============================================================
print("\nSTEP 1: Load data and build features")
data = load_raw_data()
train_raw, test_raw = merge_all_tables(data)
last_train_date = train_raw["date"].max()
print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")
print(f"  Last train date: {last_train_date.date()}")

dummy_test = train_raw[train_raw["date"] == last_train_date].head(1).copy()
train_fe, _, _ = build_features(train_raw, dummy_test)
train_fe["date"] = pd.to_datetime(train_fe["date"])

# ============================================================
# STEP 2: Compute target TE features
# ============================================================
print("\nSTEP 2: Compute TE features")
train_sales = train_raw[["store_nbr", "family", "date", "sales"]].copy()
train_sales["day_of_week"] = train_sales["date"].dt.dayofweek
train_sales["month"] = train_sales["date"].dt.month

te_sf_dow = train_sales.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std", "median"]
).reset_index()
te_sf_dow.columns = ["store_nbr", "family", "day_of_week",
                      "target_te_sf_dow_mean", "target_te_sf_dow_std", "target_te_sf_dow_median"]

te_sf = train_sales.groupby(["store_nbr", "family"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf.columns = ["store_nbr", "family", "target_te_sf_mean", "target_te_sf_std"]

te_f_dow = train_sales.groupby(["family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_f_dow.columns = ["family", "day_of_week", "target_te_f_dow_mean", "target_te_f_dow_std"]

te_sf_month = train_sales.groupby(["store_nbr", "family", "month"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_month.columns = ["store_nbr", "family", "month", "target_te_sf_month_mean", "target_te_sf_month_std"]

family_mean = train_sales.groupby("family")["sales"].mean().reset_index()
family_mean.columns = ["family", "target_te_family_mean"]

recent_3m = train_sales[train_sales["date"] >= "2017-05-01"]
te_sf_dow_recent = recent_3m.groupby(["store_nbr", "family", "day_of_week"])["sales"].agg(
    ["mean", "std"]
).reset_index()
te_sf_dow_recent.columns = ["store_nbr", "family", "day_of_week",
                             "target_te_sf_dow_recent_mean", "target_te_sf_dow_recent_std"]

# Rename day_of_week to target_day_of_week for merging
te_sf_dow_r = te_sf_dow.rename(columns={"day_of_week": "target_day_of_week"})
te_f_dow_r = te_f_dow.rename(columns={"day_of_week": "target_day_of_week"})
te_sf_month_r = te_sf_month.rename(columns={"month": "target_month"})
te_sf_dow_recent_r = te_sf_dow_recent.rename(columns={"day_of_week": "target_day_of_week"})

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

# ============================================================
# STEP 3: Get last training date features
# ============================================================
print("\nSTEP 3: Create test features")
last_date_features = train_fe[train_fe["date"] == last_train_date].copy()
print(f"  Last training date features: {last_date_features.shape}")

# ============================================================
# STEP 4: Predict each day
# ============================================================
print("\nSTEP 4: Predict")
all_preds = []

for d in range(1, 17):
    if d not in day_models:
        print(f"  Day {d:2d}: No model, skipping")
        continue

    target_date = last_train_date + pd.Timedelta(days=d)
    model = day_models[d]
    feat = day_feat_cols[d]

    # Start with reference features from last training date
    available_feat = [c for c in feat if c in last_date_features.columns]
    test_data = last_date_features[["store_nbr", "family"] + available_feat].copy()

    # Add target-date features
    test_data["target_day_of_week"] = target_date.dayofweek
    test_data["target_month"] = target_date.month
    test_data["target_day"] = target_date.day
    test_data["target_is_weekend"] = int(target_date.dayofweek >= 5)
    test_data["target_is_payday"] = int(target_date.day == 15 or target_date.is_month_end)

    # Merge target TE features
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

preds_df = pd.concat(all_preds, ignore_index=True)
overall_mean = preds_df["pred"].mean()
train_mean = train_raw["sales"].mean()
print(f"\n  Overall prediction mean: {overall_mean:.2f}")
print(f"  Training sales mean: {train_mean:.2f}")
print(f"  Underprediction ratio: {overall_mean / train_mean:.3f}")

# ============================================================
# STEP 5: Create submissions
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Create submissions")
print("=" * 60)

# Map predictions to test IDs
test_sub = test_raw[["id", "store_nbr", "family", "date"]].copy()
test_sub["target_date"] = test_sub["date"]
sub = test_sub.merge(
    preds_df[["store_nbr", "family", "target_date", "pred"]],
    on=["store_nbr", "family", "target_date"],
    how="left"
)

sub["sales"] = sub["pred"].fillna(0)
for s, f in zero_pairs:
    mask = (sub["store_nbr"] == s) & (sub["family"] == f)
    sub.loc[mask, "sales"] = 0
sub["sales"] = sub["sales"].clip(0, None)
sub.loc[sub["sales"] < 0.1, "sales"] = 0

submission = sub[["id", "sales"]].sort_values("id")
path = SUBMISSIONS / "submission_r10_day_specific.csv"
path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(path, index=False)
print(f"  r10_day_specific: mean={submission['sales'].mean():.2f}, "
      f"max={submission['sales'].max():.2f}, zeros={(submission['sales']==0).mean():.4f}")

# Geometric blend with TE level
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
