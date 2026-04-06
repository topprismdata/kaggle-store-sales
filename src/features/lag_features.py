"""Lag/Rolling/EWM特征 — 按store-family分组的历史sales特征"""
import numpy as np
import pandas as pd
from src.config import FeatureConfig


def add_lag_features(
    df: pd.DataFrame,
    cfg: FeatureConfig = None,
) -> pd.DataFrame:
    """添加lag/rolling/ewm特征。
    关键: 必须按(store_nbr, family)分组，且shift避免数据泄露。
    """
    if cfg is None:
        cfg = FeatureConfig()

    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    group = df.groupby(["store_nbr", "family"])

    for lag in cfg.lag_days:
        col_name = f"sales_lag_{lag}"
        df[col_name] = group["sales"].shift(lag)

    # Rolling — 先shift(1)再rolling，避免当前值泄露
    for w in cfg.rolling_windows:
        shifted = group["sales"].shift(1)
        for stat in cfg.rolling_stats:
            col_name = f"sales_roll_{stat}_{w}"
            if stat == "mean":
                df[col_name] = shifted.rolling(w, min_periods=1).mean().values
            elif stat == "std":
                df[col_name] = shifted.rolling(w, min_periods=1).std().values
            elif stat == "min":
                df[col_name] = shifted.rolling(w, min_periods=1).min().values
            elif stat == "max":
                df[col_name] = shifted.rolling(w, min_periods=1).max().values

    # EWM
    for span in cfg.ewm_spans:
        col_name = f"sales_ewm_{span}"
        df[col_name] = group["sales"].shift(1).ewm(span=span, min_periods=1).mean().values

    # 促销的lag特征
    for lag in [1, 7, 14]:
        col_name = f"onpromotion_lag_{lag}"
        df[col_name] = group["onpromotion"].shift(lag)

    # 促销rolling
    df["onpromotion_roll_mean_7"] = group["onpromotion"].shift(1).rolling(7, min_periods=1).mean().values

    return df
