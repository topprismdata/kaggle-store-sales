"""外部数据特征 — 油价rolling/变化率, 交易量特征, store-family统计, 分类编码"""
import numpy as np
import pandas as pd


def add_external_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加油价、交易量的衍生特征"""
    df = df.copy()
    df = df.sort_values(["store_nbr", "family", "date"])

    # === 油价特征 ===
    # Fix: compute oil rolling/pct_change/diff on unique dates first to avoid
    # cross store-family contamination, then merge back.
    if "dcoilwtico" in df.columns:
        oil_by_date = df[["date", "dcoilwtico"]].drop_duplicates("date").sort_values("date")
        oil_by_date = oil_by_date.set_index("date")
        oil_by_date["oil_roll_mean_7"] = oil_by_date["dcoilwtico"].rolling(7, min_periods=1).mean()
        oil_by_date["oil_roll_mean_14"] = oil_by_date["dcoilwtico"].rolling(14, min_periods=1).mean()
        oil_by_date["oil_roll_mean_28"] = oil_by_date["dcoilwtico"].rolling(28, min_periods=1).mean()
        oil_by_date["oil_pct_change_7"] = oil_by_date["dcoilwtico"].pct_change(7)
        oil_by_date["oil_pct_change_28"] = oil_by_date["dcoilwtico"].pct_change(28)
        oil_by_date["oil_diff_7"] = oil_by_date["dcoilwtico"].diff(7)
        oil_features = oil_by_date[["oil_roll_mean_7", "oil_roll_mean_14", "oil_roll_mean_28",
                                     "oil_pct_change_7", "oil_pct_change_28", "oil_diff_7"]].reset_index()
        df = df.drop(columns=["oil_roll_mean_7", "oil_roll_mean_14", "oil_roll_mean_28",
                               "oil_pct_change_7", "oil_pct_change_28", "oil_diff_7"],
                      errors="ignore")
        df = df.merge(oil_features, on="date", how="left")

    # === 交易量特征 ===
    if "transactions" in df.columns:
        group = df.groupby(["store_nbr", "family"])
        df["transactions_lag_1"] = group["transactions"].shift(1)
        df["transactions_lag_7"] = group["transactions"].shift(7)
        df["transactions_roll_mean_7"] = (
            group["transactions"].shift(1).rolling(7, min_periods=1).mean()
        )

    # === Store-Family 统计特征 ===
    if "sales" in df.columns:
        stats = df.groupby(["store_nbr", "family"])["sales"].agg(["mean", "median", "std"]).reset_index()
        stats.columns = ["store_nbr", "family", "store_family_mean", "store_family_median", "store_family_std"]
        df = df.merge(stats, on=["store_nbr", "family"], how="left")

    # === 分类变量编码 ===
    for col in ["family", "city", "state", "type"]:
        if col in df.columns:
            df[f"{col}_code"] = df[col].astype("category").cat.codes

    if "cluster" in df.columns:
        df["cluster"] = df["cluster"].astype(int)

    return df
