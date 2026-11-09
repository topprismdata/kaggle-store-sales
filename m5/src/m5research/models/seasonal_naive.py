from __future__ import annotations

"""Simple M5 Accuracy weekly baselines, using only pre-origin sales."""

import numpy as np

HORIZON = 28


def seasonal_naive_7(sales: np.ndarray, origin: int) -> np.ndarray:
    """Repeat the last fully observed week for 28 future days."""
    if origin < 7:
        raise ValueError("origin must be >= 7 for a weekly seasonal naive")
    last_week = sales[:, origin - 7:origin]
    return np.maximum(np.tile(last_week, (1, 4)), 0.0)


def seasonal_mean_28(sales: np.ndarray, origin: int) -> np.ndarray:
    """Average equal weekdays across the last four fully observed weeks."""
    if origin < 28:
        raise ValueError("origin must be >= 28 for a 4-week seasonal mean")
    window = sales[:, origin - 28:origin]
    mean_week = window.reshape(window.shape[0], 4, 7).mean(axis=1)
    return np.maximum(np.tile(mean_week, (1, 4)), 0.0)


BASELINES = {
    "seasonal_naive_7": seasonal_naive_7,
    "seasonal_mean_28": seasonal_mean_28,
}
