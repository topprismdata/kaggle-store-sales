import numpy as np
import pytest

from m5research.features import lagged_rolling_mean


def test_rolling_mean_excludes_current_target():
    values = np.array([1.0, 2.0, 3.0, 1000.0])
    result = lagged_rolling_mean(values, window=3)
    assert np.isnan(result[:3]).all()
    assert result[3] == pytest.approx((1.0 + 2.0 + 3.0) / 3.0)


def test_rolling_mean_is_not_affected_by_current_value():
    first = lagged_rolling_mean([1, 2, 3, 4], window=2)
    second = lagged_rolling_mean([1, 2, 3, 999], window=2)
    assert first[3] == second[3]


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        lagged_rolling_mean([1, 2, 3], window=0)
