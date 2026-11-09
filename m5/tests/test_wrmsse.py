import numpy as np
import pytest

from m5research.metrics.wrmsse import (
    UndefinedRMSSEScaleError,
    official_wrmsse,
    rmsse,
    rmsse_scale,
    weighted_level_wrmsse,
)


def test_leading_zeros_are_trimmed_before_scale():
    values = [0, 0, 2, 4, 2]
    assert rmsse_scale(values) == pytest.approx(np.mean(np.diff([2, 4, 2]) ** 2))


def test_all_zero_history_is_explicitly_undefined():
    with pytest.raises(UndefinedRMSSEScaleError, match="all-zero history"):
        rmsse_scale([0, 0, 0, 0])


def test_constant_active_history_is_explicitly_undefined():
    with pytest.raises(UndefinedRMSSEScaleError, match="scale is zero"):
        rmsse_scale([0, 0, 5, 5, 5])


def test_single_active_observation_is_undefined():
    with pytest.raises(UndefinedRMSSEScaleError, match="fewer than two"):
        rmsse_scale([0, 0, 5])


def test_rmsse_uses_mean_squared_error_over_active_scale():
    train = [0, 0, 2, 4, 2]
    actual = [3, 4]
    forecast = [2, 2]
    expected = np.sqrt(np.mean((np.array(actual) - np.array(forecast)) ** 2) / 5.0)
    assert rmsse(train, actual, forecast) == pytest.approx(expected)


def test_weighted_level_requires_normalized_weights():
    assert weighted_level_wrmsse([1.0, 2.0], [0.25, 0.75]) == pytest.approx(1.75)
    with pytest.raises(ValueError, match="sum to 1"):
        weighted_level_wrmsse([1.0, 2.0], [0.2, 0.2])


def test_official_wrmsse_requires_twelve_levels():
    with pytest.raises(ValueError, match="expected 12"):
        official_wrmsse({"Level1": 1.0})
