import numpy as np
import pytest

from m5research.models.seasonal_naive import seasonal_naive_7, seasonal_mean_28


def test_naive_repeats_last_week_without_future_data():
    past = np.arange(35, dtype=float).reshape(1, 35)
    result = seasonal_naive_7(past, 35)
    np.testing.assert_array_equal(result[0, :7], past[0, -7:])
    np.testing.assert_array_equal(result[0, 7:14], past[0, -7:])


def test_mean28_calculates_same_weekday_average():
    sales = np.zeros((1, 28), dtype=float)
    sales[0, ::7] = [4, 8, 12, 16]
    result = seasonal_mean_28(sales, 28)
    assert result.shape == (1, 28)
    assert result[0, 0] == pytest.approx(10.0)
    assert result[0, 1] == pytest.approx(0.0)


def test_rejects_short_histories():
    with pytest.raises(ValueError):
        seasonal_naive_7(np.zeros((1, 6)), 6)
    with pytest.raises(ValueError):
        seasonal_mean_28(np.zeros((1, 27)), 27)
