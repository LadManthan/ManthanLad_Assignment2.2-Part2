import numpy as np
import pytest

from numcompute import preprocessing as a1
from numcompute_stream.preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler


def test_scaler_partial_fit_matches_fit():
    x = np.random.default_rng(0).normal(3.0, 2.0, size=500)
    streamed = StandardScaler()
    for start in range(0, 500, 37):  # uneven chunk sizes on purpose
        streamed.partial_fit(x[start : start + 37])
    batch = a1.StandardScaler().fit(x)  # Assignment 1 version, batch fit
    assert streamed.mean_ == pytest.approx(batch.mean_)
    assert streamed.scale_ == pytest.approx(batch.scale_)
    np.testing.assert_allclose(streamed.transform(x), batch.transform(x))


def test_scaler_is_stable_with_large_offset():
    # The sum-of-squares formula loses all precision here; Welford does not.
    x = 1e9 + np.random.default_rng(1).normal(size=1000)
    scaler = StandardScaler()
    for start in range(0, 1000, 100):
        scaler.partial_fit(x[start : start + 100])
    assert scaler.scale_ == pytest.approx(x.std(), rel=1e-6)


def test_scaler_zero_variance_and_nan():
    scaler = StandardScaler().partial_fit([5.0, 5.0, 5.0])
    np.testing.assert_allclose(scaler.transform([5.0, 5.0]), 0.0)  # scale 1, no division by 0
    skips_nan = StandardScaler().partial_fit([1.0, np.nan, 3.0])
    assert skips_nan.mean_ == 2.0


def test_minmax_scaler_uses_running_range():
    scaler = MinMaxScaler()
    scaler.partial_fit([0.0, 5.0])
    scaler.partial_fit([10.0, np.nan])
    np.testing.assert_allclose(scaler.transform([0.0, 5.0, 10.0]), [0, 0.5, 1])


def test_imputer_running_mean_and_all_nan_start():
    imputer = Imputer(fill_value=-1.0)
    imputer.partial_fit([np.nan, np.nan])
    assert imputer.statistics_ == -1.0  # no real value seen yet
    imputer.partial_fit([1.0, np.nan])
    imputer.partial_fit([3.0])
    np.testing.assert_array_equal(imputer.transform([np.nan, 7.0]), [2.0, 7.0])
    with pytest.raises(ValueError, match="median"):
        Imputer(strategy="median").partial_fit([1.0])


def test_one_hot_encoder_adds_columns_over_time():
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.partial_fit(np.array(["red", "blue"]))
    first = encoder.transform(np.array(["red", "blue", "green"]))
    assert first.shape == (3, 2)
    np.testing.assert_array_equal(first[2], [0, 0])  # unseen -> all zeros

    encoder.partial_fit(np.array(["green"]))
    second = encoder.transform(np.array(["red", "blue", "green"]))
    assert second.shape == (3, 3)
    np.testing.assert_array_equal(second[:, :2], first)  # old columns did not move

    # fit (Assignment 1) and partial_fit can be mixed.
    strict = OneHotEncoder().fit(np.array([1, 2])).partial_fit(np.array([5]))
    np.testing.assert_array_equal(strict.transform(np.array([5, 1])), [[0, 0, 1], [1, 0, 0]])
    with pytest.raises(ValueError, match="unknown"):
        strict.transform(np.array([3]))


def test_bad_chunks_raise():
    with pytest.raises(ValueError, match="1D"):
        StandardScaler().partial_fit([[1.0, 2.0]])
    with pytest.raises(ValueError, match="infinite"):
        Imputer().partial_fit([np.inf])
    with pytest.raises(TypeError, match="numeric"):
        MinMaxScaler().partial_fit(np.array(["a"]))
    with pytest.raises(ValueError, match="at least one"):
        StandardScaler().partial_fit([])
