from __future__ import annotations

import numpy as np


def lagged_rolling_mean(values, window: int) -> np.ndarray:
    """At t, return mean(values[t-window:t]); current target y[t] is excluded."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if window < 1:
        raise ValueError("window must be >= 1")
    result = np.full(arr.shape, np.nan, dtype=np.float64)
    if arr.size <= window:
        return result
    cumulative = np.concatenate(([0.0], np.cumsum(arr)))
    for t in range(window, arr.size):
        result[t] = (cumulative[t] - cumulative[t - window]) / window
    return result
