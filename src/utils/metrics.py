"""评估指标"""
import numpy as np


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error
    注意: 竞赛评估指标，需要clip防止log(0)
    """
    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return np.sqrt(np.mean(log_diff ** 2))
