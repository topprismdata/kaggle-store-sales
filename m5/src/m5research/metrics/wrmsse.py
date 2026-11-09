from __future__ import annotations

import numpy as np


class UndefinedRMSSEScaleError(ValueError):
    """Raised when the official active-history RMSSE scale is zero or undefined."""


def rmsse_scale(train_values: np.ndarray | list[float]) -> float:
    """Compute the M5 RMSSE scale after the first non-zero observation.

    Leading zeros are excluded, matching the M5 guide's active-sales rule.
    No epsilon or silent fallback is applied when the resulting scale is zero.
    """
    train = np.asarray(train_values, dtype=np.float64)
    if train.ndim != 1:
        raise ValueError("train_values must be one-dimensional")
    if train.size < 2:
        raise UndefinedRMSSEScaleError("at least two observations are required")

    nonzero = np.flatnonzero(train != 0)
    if nonzero.size == 0:
        raise UndefinedRMSSEScaleError("all-zero history: no first non-zero observation")

    active = train[nonzero[0] :]
    if active.size < 2:
        raise UndefinedRMSSEScaleError("fewer than two active-history observations")

    scale = float(np.mean(np.diff(active) ** 2))
    if not np.isfinite(scale) or scale <= 0.0:
        raise UndefinedRMSSEScaleError(
            f"active-history scale is zero or non-finite ({scale}); "
            "do not inject epsilon or silently drop this series"
        )
    return scale


def rmsse(
    train_values: np.ndarray | list[float],
    actual_values: np.ndarray | list[float],
    forecast_values: np.ndarray | list[float],
) -> float:
    """Compute RMSSE for one series and one fixed forecast horizon."""
    actual = np.asarray(actual_values, dtype=np.float64)
    forecast = np.asarray(forecast_values, dtype=np.float64)
    if actual.ndim != 1 or forecast.ndim != 1:
        raise ValueError("actual_values and forecast_values must be one-dimensional")
    if actual.shape != forecast.shape:
        raise ValueError("actual_values and forecast_values must have identical shapes")
    if actual.size == 0:
        raise ValueError("forecast horizon must not be empty")
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(forecast)):
        raise ValueError("actual and forecast values must be finite")

    scale = rmsse_scale(train_values)
    numerator = float(np.mean((actual - forecast) ** 2))
    return float(np.sqrt(numerator / scale))


def weighted_level_wrmsse(
    rmsse_values: np.ndarray | list[float],
    weights: np.ndarray | list[float],
    *,
    atol: float = 1e-8,
) -> float:
    """Aggregate one hierarchy level; its weights must sum to one."""
    scores = np.asarray(rmsse_values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 1 or w.ndim != 1 or scores.shape != w.shape:
        raise ValueError("rmsse_values and weights must be equal-length 1D arrays")
    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(w)):
        raise ValueError("scores and weights must be finite")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if not np.isclose(float(w.sum()), 1.0, atol=atol):
        raise ValueError(f"weights for one level must sum to 1; got {w.sum()}")
    return float(np.dot(scores, w))


def official_wrmsse(level_scores: dict[str, float]) -> float:
    """Mean the 12 level-specific weighted scores after each level is verified."""
    if len(level_scores) != 12:
        raise ValueError(f"expected 12 level scores, got {len(level_scores)}")
    values = np.asarray(list(level_scores.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("all level scores must be finite")
    return float(values.mean())
