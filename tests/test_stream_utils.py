"""Tests for the streaming helpers added to numcompute_stream/utils.py."""

import numpy as np
import pytest

from numcompute_stream.utils import check_2d, check_labels, check_numeric, memory_bytes, update_mean_var


def test_check_numeric_accepts_numbers_and_rejects_bad_input():
    X = check_numeric([[1, 2], [3, np.nan]])
    assert X.dtype == float and X.shape == (2, 2)
    with pytest.raises(TypeError):
        check_numeric([["a", "b"]])
    with pytest.raises(ValueError, match="infinite"):
        check_numeric([[1.0, np.inf]])
    with pytest.raises(ValueError, match="NaN"):
        check_numeric([[np.nan]], allow_nan=False)
    with pytest.raises(ValueError, match="features"):
        check_numeric([[1.0, 2.0]], n_features=3)


def test_check_2d_and_check_labels_shapes():
    with pytest.raises(ValueError, match="2D"):
        check_2d([1.0, 2.0])
    with pytest.raises(ValueError, match="at least one row"):
        check_2d(np.empty((0, 3)))
    assert check_labels([0, 1], n_samples=2).shape == (2,)
    with pytest.raises(ValueError):
        check_labels([0, 1, 1], n_samples=2)


def test_update_mean_var_chunks_match_numpy():
    data = np.random.default_rng(0).normal(5.0, 2.0, size=(500, 3))
    count, mean, m2 = np.zeros(3), np.zeros(3), np.zeros(3)
    for start in range(0, 500, 70):  # uneven last chunk on purpose
        count, mean, m2 = update_mean_var(count, mean, m2, data[start : start + 70])
    np.testing.assert_allclose(mean, data.mean(axis=0))
    np.testing.assert_allclose(m2 / count, data.var(axis=0))


def test_update_mean_var_skips_nan_and_is_stable():
    chunk = np.array([[1.0, np.nan], [3.0, np.nan]])
    count, mean, m2 = update_mean_var(np.zeros(2), np.zeros(2), np.zeros(2), chunk)
    np.testing.assert_array_equal(count, [2, 0])  # all-NaN column stays empty
    assert mean[0] == 2.0 and mean[1] == 0.0
    # Large offset: the sum-of-squares formula would lose all precision here.
    big = 1e9 + np.random.default_rng(1).normal(size=(1000, 1))
    count, mean, m2 = np.zeros(1), np.zeros(1), np.zeros(1)
    for start in range(0, 1000, 100):
        count, mean, m2 = update_mean_var(count, mean, m2, big[start : start + 100])
    np.testing.assert_allclose(m2 / count, big.var(axis=0), rtol=1e-6)


def test_memory_bytes_counts_arrays_and_lists_of_arrays():
    class Holder:
        pass

    holder = Holder()
    holder.weights = np.zeros(10)                  # 80 bytes
    holder.buffer = [np.zeros(5), np.zeros(5)]     # 80 bytes
    holder.name = "not an array"                   # ignored
    assert memory_bytes(holder) == 160
