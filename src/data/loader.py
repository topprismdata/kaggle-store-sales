"""数据加载器 — 合并train/test与所有辅助表"""
import numpy as np
import pandas as pd
from src.config import DATA_RAW


def load_raw_data() -> dict:
    """加载所有原始CSV，返回dict[dataframe]"""
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "stores": "stores.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
        "sample_submission": "sample_submission.csv",
    }
    data = {}
    for key, filename in files.items():
        path = DATA_RAW / filename
        data[key] = pd.read_csv(path)
        if "date" in data[key].columns:
            data[key]["date"] = pd.to_datetime(data[key]["date"])
    return data


def merge_all_tables(data: dict) -> tuple:
    """合并train/test与stores/oil/holidays/transactions
    返回: (train_merged, test_merged)
    """
    train = data["train"].copy()
    test = data["test"].copy()

    train["is_train"] = True
    test["is_train"] = False
    test["sales"] = np.nan
    df = pd.concat([train, test], ignore_index=True)

    # 合并 stores
    df = df.merge(data["stores"], on="store_nbr", how="left")

    # 合并 oil
    df = df.merge(data["oil"], on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    # 合并 transactions
    df = df.merge(data["transactions"], on=["date", "store_nbr"], how="left")
    df["transactions"] = df["transactions"].fillna(0)

    # 合并 holidays
    df = _merge_holidays(df, data["holidays"])

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    train_merged = df[df["is_train"]].drop(columns=["is_train"]).reset_index(drop=True)
    test_merged = df[~df["is_train"]].drop(columns=["is_train", "sales"]).reset_index(drop=True)

    return train_merged, test_merged


def _merge_holidays(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """假日合并 — National/Regional/Local三种级别"""
    holidays = holidays.copy()
    actual_holidays = holidays[
        (holidays["transferred"] == False) | (holidays["type"] == "Transfer")
    ].copy()

    # National
    national = actual_holidays[actual_holidays["locale"] == "National"]
    national_flags = national[["date", "type", "description"]].drop_duplicates("date")
    national_flags = national_flags.rename(columns={
        "type": "holiday_type_national",
        "description": "holiday_desc_national"
    })
    df = df.merge(national_flags, on="date", how="left")
    df["is_national_holiday"] = df["holiday_type_national"].notna().astype(int)

    # Regional
    regional = actual_holidays[actual_holidays["locale"] == "Regional"]
    regional_flags = regional[["date", "locale_name", "type", "description"]].drop_duplicates()
    regional_flags = regional_flags.rename(columns={
        "locale_name": "state",
        "type": "holiday_type_regional",
        "description": "holiday_desc_regional"
    })
    df = df.merge(regional_flags, on=["date", "state"], how="left")
    df["is_regional_holiday"] = df["holiday_type_regional"].notna().astype(int)

    # Local
    local = actual_holidays[actual_holidays["locale"] == "Local"]
    local_flags = local[["date", "locale_name", "type", "description"]].drop_duplicates()
    local_flags = local_flags.rename(columns={
        "locale_name": "city",
        "type": "holiday_type_local",
        "description": "holiday_desc_local"
    })
    df = df.merge(local_flags, on=["date", "city"], how="left")
    df["is_local_holiday"] = df["holiday_type_local"].notna().astype(int)

    # 合并
    df["is_holiday"] = (
        df["is_national_holiday"]
        | df["is_regional_holiday"]
        | df["is_local_holiday"]
    ).astype(int)

    return df
