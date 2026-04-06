"""特征构建流水线 — 串联所有特征工程步骤"""
import pandas as pd
import numpy as np
from src.config import DATA_PROCESSED, FeatureConfig
from src.features.time_features import add_time_features
from src.features.lag_features import add_lag_features
from src.features.external_features import add_external_features


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: FeatureConfig = None,
) -> tuple:
    """构建完整的特征矩阵。
    返回: (train_featured, test_featured, feature_columns)
    """
    if cfg is None:
        cfg = FeatureConfig()

    train = train.copy()
    test = test.copy()

    # 拼接统一处理
    train["is_train"] = True
    test["is_train"] = False
    test["sales"] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Step 1: 时间特征
    df = add_time_features(df)

    # Step 2: Lag/Rolling特征
    df = add_lag_features(df, cfg)

    # Step 3: 外部数据特征
    df = add_external_features(df)

    # 定义要排除的列
    exclude_cols = {
        "id", "date", "sales", "is_train",
        "family", "city", "state", "type",
        "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
        "holiday_type_national", "holiday_type_regional", "holiday_type_local",
    }
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    # 拆分回train/test
    train_out = df[df["is_train"]].drop(columns=["is_train"]).reset_index(drop=True)
    test_out = df[~df["is_train"]].drop(columns=["is_train"]).reset_index(drop=True)

    if "sales" in test_out.columns:
        test_out = test_out.drop(columns=["sales"])

    return train_out, test_out, feature_columns


def get_feature_columns(df: pd.DataFrame) -> list:
    """从dataframe中提取特征列名"""
    exclude = {
        "id", "date", "sales", "is_train",
        "family", "city", "state", "type",
        "holiday_desc_national", "holiday_desc_regional", "holiday_desc_local",
        "holiday_type_national", "holiday_type_regional", "holiday_type_local",
    }
    return [c for c in df.columns if c not in exclude]
