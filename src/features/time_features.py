"""时间/日历特征 — 年月日星期、发薪日、地震标记"""
import numpy as np
import pandas as pd
from src.config import EARTHQUAKE_DATE


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加所有时间相关特征（不依赖历史sales数据）"""
    df = df.copy()
    dt = df["date"]

    # 基础日历
    df["year"] = dt.dt.year
    df["month"] = dt.dt.month
    df["day"] = dt.dt.day
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_month"] = (dt.dt.day - 1) // 7 + 1
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)

    # 发薪日 — 厄瓜多尔公共部门15日和月末发薪
    df["is_payday_15"] = (dt.dt.day == 15).astype(int)
    df["is_payday_end"] = dt.dt.is_month_end.astype(int)
    df["is_payday"] = (df["is_payday_15"] | df["is_payday_end"]).astype(int)
    df["days_to_payday"] = df.apply(_days_to_next_payday, axis=1)

    # 地震特征 — 2016-04-16 厄瓜多尔7.8级地震
    eq_date = pd.Timestamp(EARTHQUAKE_DATE)
    df["days_since_earthquake"] = (dt - eq_date).dt.days
    df["is_post_earthquake"] = (
        (dt >= eq_date) & (dt <= eq_date + pd.Timedelta(days=30))
    ).astype(int)
    df.loc[df["days_since_earthquake"] < 0, "days_since_earthquake"] = 0

    # 学年开始 (厄瓜多尔9月开学)
    df["is_school_start"] = ((dt.dt.month == 9) & (dt.dt.day <= 15)).astype(int)

    # 周期性编码 — sin/cos
    df["day_of_week_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365)

    return df


def _days_to_next_payday(row) -> int:
    """距下一个发薪日(15号或月末)的天数"""
    day = row["date"].day
    last_day = row["date"].days_in_month
    paydays = [15, last_day]
    future_paydays = [p for p in paydays if p >= day]
    if future_paydays:
        return future_paydays[0] - day
    else:
        return 15 + (last_day - day)
