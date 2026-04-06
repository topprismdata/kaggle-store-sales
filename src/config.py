"""全局配置 — 所有路径和超参数集中管理"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
SUBMISSIONS = OUTPUTS / "submissions"
MODELS = OUTPUTS / "models"
FIGURES = OUTPUTS / "figures"

# 竞赛关键日期
TRAIN_START = "2013-01-01"
TRAIN_END = "2017-08-15"
TEST_START = "2017-08-16"
TEST_END = "2017-08-31"
EARTHQUAKE_DATE = "2016-04-16"

# 数据维度
N_STORES = 54
N_FAMILIES = 33

@dataclass
class FeatureConfig:
    """特征工程配置"""
    lag_days: List[int] = field(default_factory=lambda: [1, 7, 14, 28, 365])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 14, 28, 60])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])
    ewm_spans: List[int] = field(default_factory=lambda: [7, 14, 28])
    use_oil: bool = True
    use_holidays: bool = True
    use_transactions: bool = True
    use_earthquake: bool = True
    use_payday: bool = True

@dataclass
class ModelConfig:
    """模型超参数 — LightGBM baseline默认值"""
    objective: str = "regression"
    metric: str = "rmse"
    boosting_type: str = "gbdt"
    n_estimators: int = 3000
    learning_rate: float = 0.005
    max_depth: int = -1
    num_leaves: int = 64
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    random_state: int = 42
    early_stopping_rounds: int = 100

@dataclass
class CVConfig:
    """交叉验证配置"""
    n_folds: int = 5
    val_days: int = 16
