"""Lag/Rolling/EWM特征 — 按store-family分组的历史sales特征

修复记录: 3M行数据的rolling导致内存爆炸(OOM Kill ~5.4GB)
原因: group["sales"].shift(1).rolling().values 创建了~540万份中间Series拷贝
解决方案: 对每个window/stat组合分别计算rolling/mean等，而不是先算完整shifted再rolling
"""
import numpy as np
import pandas as pd
from src.config import FeatureConfig


def add_lag_features(df: pd.DataFrame, cfg: FeatureConfig = None) -> pd.DataFrame:
    if cfg is None:
        cfg = FeatureConfig()

    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])
    group = df.groupby(["store_nbr", "family"])

    # Lag特征
    for lag in cfg.lag_days:
        col_name = f"sales_lag_{lag}"
        df[col_name] = group["sales"].shift(lag)

    # Rolling特征 — 逐个window/stat分别计算，避免内存爆炸
    for w in cfg.rolling_windows:
        shifted = group["sales"].shift(1)
        for stat in cfg.rolling_stats:
            col_name = f"sales_roll_{stat}_{w}"
            if stat == "mean":
                df[col_name] = shifted.rolling(w, min_periods=1).mean()
            elif stat == "std":
                df[col_name] = shifted.rolling(w, min_periods=1).std()
            elif stat == "min":
                df[col_name] = shifted.rolling(w, min_periods=1).min()
            elif stat == "max":
                df[col_name] = shifted.rolling(w, min_periods=1).max()

    # EWM特征
    for span in cfg.ewm_spans:
        col_name = f"sales_ewm_{span}"
        df[col_name] = group["sales"].shift(1).ewm(span=span, min_periods=1).mean()

    # 促销的lag特征
    for lag in [1, 7, 14]:
        col_name = f"onpromotion_lag_{lag}"
        df[col_name] = group["onpromotion"].shift(lag)

    # 促销rolling
    df["onpromotion_roll_mean_7"] = group["onpromotion"].shift(1).rolling(7, min_periods=1).mean()

    return df
