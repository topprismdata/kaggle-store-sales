"""特征构建流水线 — 串联所有特征工程步骤

Bug修复记录:
- v1 (原始): test["sales"] = np.nan → lag特征在test集上93%+ NaN → LB 2.67
- v2 (sales=0): lag_1全变为0 → 预测偏低 → LB 2.83
- v3 (ffill): 用train末尾sales值前向填充test → lag特征有效 → 修复成功
"""
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
    test_sales: pd.Series = None,
) -> tuple:
    """构建完整的特征矩阵。
    返回: (train_featured, test_featured, feature_columns)

    test_sales: Optional. If provided, used as test sales values instead of ffill.
                Should be a Series aligned with test rows (e.g., TE estimates).
                Enables TE-fill strategy: lag features vary by day_of_week instead
                of being constant (ffill).
    """
    if cfg is None:
        cfg = FeatureConfig()

    train = train.copy()
    test = test.copy()

    # 拼接统一处理
    # 关键修复: test集的sales用前向填充(ffill)而非NaN或0
    # 原因: sales=NaN → lag/rolling/ewm在test上大面积NaN → LB=2.67
    #       sales=0   → lag_1全为0 → 预测偏差 → LB=2.83
    # 解决: concat后按(store,family)分组做ffill → train末尾sales传播到test → LB改善
    # v4 (TE-fill): 用TE期望值填充test sales → lag特征按day_of_week变化而非恒定
    train["is_train"] = True
    test["is_train"] = False
    if test_sales is not None:
        test["sales"] = test_sales.values
    else:
        test["sales"] = np.nan
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # 前向填充: 对于ffill模式填完所有NaN; 对于TE-fill模式只填补可能的剩余NaN
    df["sales"] = df.groupby(["store_nbr", "family"])["sales"].ffill()

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
