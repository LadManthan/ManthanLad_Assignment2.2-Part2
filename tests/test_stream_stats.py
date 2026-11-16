import numpy as np
import pytest

from numcompute_stream.stats import StreamingHistogram, StreamingStats


def test_running_statistics_match_numpy_with_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    X[rng.random(X.shape) < 0.1] = np.nan
    stats = StreamingStats()
    for start in range(0, 300, 40):
        stats.update_stats(X[start : start + 40])
    np.testing.assert_allclose(stats.mean(), np.nanmean(X, axis=0))
    np.testing.assert_allclose(stats.variance(ddof=1), np.nanvar(X, axis=0, ddof=1))
    np.testing.assert_allclose(stats.minimum(), np.nanmin(X, axis=0))
    np.testing.assert_allclose(stats.maximum(), np.nanmax(X, axis=0))
    np.testing.assert_allclose(stats.quantile(0.5), np.nanmedian(X, axis=0))


def test_quantiles_and_histogram_use_window():
    stats = StreamingStats(window_size=10)
    stats.update_stats(np.zeros((50, 1)))
    stats.update_stats(np.arange(10.0).reshape(-1, 1))
    # Only the last 10 rows (0..9) are in the window; the mean covers all 60.
    assert stats.quantile(0.5)[0] == pytest.approx(4.5)
    assert stats.mean()[0] == pytest.approx(45 / 60)
    counts, _ = stats.histogram(0, bins=2)
    assert counts.sum() == 10


def test_edge_cases():
    stats = StreamingStats()
    with pytest.raises(ValueError, match="No data"):
        stats.mean()
    stats.update_stats([[1.0, np.nan]])
    assert np.isnan(stats.variance(ddof=1)).all()  # one value: sample variance undefined
    assert np.isnan(stats.mean()[1])  # only NaN seen in column 1
    with pytest.raises(ValueError, match="features"):
        stats.update_stats([[1.0, 2.0, 3.0]])


def test_streaming_histogram_matches_numpy():
    values = np.random.default_rng(2).normal(size=1000)
    hist = StreamingHistogram(bins=8, range=(-2, 2))
    for start in range(0, 1000, 100):
        hist.update(values[start : start + 100])
    expected, edges = np.histogram(values, bins=8, range=(-2, 2))
    counts, out_edges = hist.result()
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_allclose(out_edges, edges)
    assert hist.n_outside_ == 1000 - expected.sum()


def test_streaming_histogram_sliding_window():
    hist = StreamingHistogram(bins=2, range=(0, 2), window_chunks=1)
    hist.update([0.5, 0.5, np.nan])
    hist.update([1.5])
    counts, _ = hist.result()
    np.testing.assert_array_equal(counts, [0, 1])
    density, edges = hist.result(density=True)
    assert np.sum(density * np.diff(edges)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="range"):
        StreamingHistogram(bins=5)
