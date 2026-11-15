import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from numcompute_stream.utils import check_numeric, update_mean_var


def _as_numeric_array(
    data: npt.ArrayLike,
) -> NDArray[np.float64]:
    """
    Convert input to a numeric NumPy array.
    
    Parameters
    ----------
    data : array-like
        Input data containing real numeric values.

    Returns
    -------
    numpy.ndarray
        Floating-point array containing the input data.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = np.asarray(data)

    if arr.dtype.kind not in "iuf":
        raise TypeError("Data must contain real numeric values")

    return arr.astype(float, copy=False)


def _validate_data(
    data: npt.ArrayLike,
    allow_empty: bool = False,
) -> NDArray[np.float64]:
    """Validate numeric input and optionally allow empty arrays.
    Parameters
    ----------
    data : array-like
        Input data to validate.
    allow_empty : bool, default=False
        Whether to allow an empty array.

    Returns
    -------
    numpy.ndarray
        Validated floating-point array.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the array is empty and allow_empty is False.

    Complexity: O(n) time and O(n) space for n input elements.
    """
    arr = _as_numeric_array(data)

    if arr.size == 0 and not allow_empty:
        raise ValueError("Data must not be empty")

    return arr


def mean(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the arithmetic mean.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the mean.
        None calculates the mean over all elements.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Mean value or array of mean values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmean(arr, axis=axis)

    return np.mean(arr, axis=axis)


def median(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """Return the median.
    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate the median.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Median value or array of median values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmedian(arr, axis=axis)

    return np.median(arr, axis=axis)


def std(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    ddof: int = 0,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the standard deviation of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate standard deviation.
    ddof : int, default=0
        Delta degrees of freedom. Zero computes population standard
        deviation; one computes sample standard deviation.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Standard deviation value or array of values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty, ddof is invalid, or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")

    if skipna:
        return np.nanstd(arr, axis=axis, ddof=ddof)

    return np.std(arr, axis=axis, ddof=ddof)


def minimum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the minimum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the minimum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Minimum value or array of minimum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmin(arr, axis=axis)

    return np.min(arr, axis=axis)


def maximum(
    data: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Find the maximum value in the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to find the maximum.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Maximum value or array of maximum values.

    Raises
    ------
    TypeError
        If the input contains non-numeric or complex values.
    ValueError
        If the input is empty or the axis is invalid.

    Complexity: O(n) time for n input elements.    
    """
    arr = _validate_data(data)

    if skipna:
        return np.nanmax(arr, axis=axis)

    return np.max(arr, axis=axis)


def quantile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested quantile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Quantile value(s) between 0 and 1, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate quantiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Quantile value or array of quantile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 1], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    arr = _validate_data(data)
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 1)):
        raise ValueError("q must be between 0 and 1")

    if skipna:
        return np.nanquantile(arr, q_arr, axis=axis)

    return np.quantile(arr, q_arr, axis=axis)


def percentile(
    data: npt.ArrayLike,
    q: npt.ArrayLike,
    axis: int | tuple[int, ...] | None = None,
    skipna: bool = False,
) -> float | NDArray[np.float64]:
    """
    Calculate the requested percentile(s) of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    q : float or array-like
        Percentile value(s) between 0 and 100, inclusive.
    axis : int, tuple of int, or None, default=None
        Axis or axes along which to calculate percentiles.
    skipna : bool, default=False
        If True, ignore NaN values.

    Returns
    -------
    float or numpy.ndarray
        Percentile value or array of percentile values.

    Raises
    ------
    TypeError
        If the input data is non-numeric or complex.
    ValueError
        If the data is empty, q is outside [0, 100], q contains
        non-finite values, or the axis is invalid.

    Complexity: Typically O(n log n) time for n input elements.
    """
    q_arr = np.asarray(q, dtype=float)

    if not np.all(np.isfinite(q_arr)):
        raise ValueError("q must contain finite values")

    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be between 0 and 100")

    return quantile(data, q_arr / 100.0, axis=axis, skipna=skipna)


def histogram(
    data: npt.ArrayLike,
    bins: int | str | npt.ArrayLike = 10,
    range: tuple[float, float] | None = None,
    density: bool = False,
    skipna: bool = False,
) -> tuple[NDArray[np.int64] | NDArray[np.float64], NDArray[np.float64]]:
    """Calculate a histogram of the input data.

    Parameters
    ----------
    data : array-like
        Numeric input data.
    bins : int, str, or array-like, default=10
        Number of bins, a supported binning strategy, or explicit
        bin edges.
    range : tuple of float, optional
        Lower and upper range of the bins.
    density : bool, default=False
        If True, return normalized density values instead of counts.
    skipna : bool, default=False
        If True, ignore NaN values. Infinite values are always rejected.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing histogram counts or densities and bin edges.

    Raises
    ------
    TypeError
        If the data contains non-numeric or complex values.
    ValueError
        If the data has no valid elements, contains NaN without
        skipna=True, contains infinity, or the bin configuration is invalid.

    Complexity: O(n + b) time, where n is the number of observations
    and b is the number of bins.
    """
    arr = _validate_data(data, allow_empty=True).ravel()

    if np.any(np.isinf(arr)):
        raise ValueError("Data must not contain infinite values")

    if skipna:
        arr = arr[~np.isnan(arr)]
    elif np.any(np.isnan(arr)):
        raise ValueError("Data contains NaN; use skipna=True to ignore it")

    if arr.size == 0:
        raise ValueError("Data must contain at least one valid value")

    if isinstance(bins, (int, np.integer)):
        if bins <= 0:
            raise ValueError("Bins must be a positive integer")
    elif not isinstance(bins, str):
        edges = np.asarray(bins, dtype=float)

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                "Bin edges must be a 1D array with at least 2 values"
            )

        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(
                "Bin edges must be finite and strictly increasing"
            )

    counts, edges = np.histogram(
        arr,
        bins=bins,
        range=range,
        density=density,
    )

    return counts, edges


class StreamingStats:
    """
    Running mean, variance, min, max, quantiles and histograms of chunked data.

    Mean, variance, min and max are exact for everything seen and only keep a
    few numbers per feature. 
    Quantiles and histograms need the actual values, so they are computed on a sliding window of the newest ``window_size`` rows. With ``window_size=None`` every row is kept (exact, but memory grows).

    NaN values are skipped everywhere.

    Parameters
    ----------
    window_size : int, optional
        Number of newest rows kept for quantiles and histograms.
    """

    def __init__(self, window_size=None):
        if window_size is not None and (not isinstance(window_size, int) or window_size < 1):
            raise ValueError("window_size must be a positive integer or None.")
        self.window_size = window_size
        self.reset()

    def reset(self):
        """Forget everything seen so far."""
        self.count_ = None
        self.mean_ = None
        self.m2_ = None
        self.min_ = None
        self.max_ = None
        self.window_ = None
        return self

    def update_stats(self, X_chunk):
        """
        Add a chunk of shape (n_chunk, n_features) to the statistics.

        Raises
        ------
        ValueError
            If the number of features changes between chunks.
        """
        n_features = None if self.mean_ is None else len(self.mean_)
        chunk = check_numeric(X_chunk, "X_chunk", n_features=n_features)
        if self.mean_ is None:
            width = chunk.shape[1]
            self.count_ = np.zeros(width)
            self.mean_ = np.zeros(width)
            self.m2_ = np.zeros(width)
            self.min_ = np.full(width, np.inf)
            self.max_ = np.full(width, -np.inf)
            self.window_ = np.empty((0, width))

        self.count_, self.mean_, self.m2_ = update_mean_var(
            self.count_, self.mean_, self.m2_, chunk
        )
        is_nan = np.isnan(chunk)
        self.min_ = np.minimum(self.min_, np.where(is_nan, np.inf, chunk).min(axis=0))
        self.max_ = np.maximum(self.max_, np.where(is_nan, -np.inf, chunk).max(axis=0))

        self.window_ = np.vstack([self.window_, chunk])
        if self.window_size is not None:
            self.window_ = self.window_[-self.window_size :]
        return self

    def _check_has_data(self):
        if self.mean_ is None:
            raise ValueError("No data seen yet. Call update_stats first.")

    def mean(self):
        """Mean of every feature, shape (n_features,). NaN for a feature with no values."""
        self._check_has_data()
        return np.where(self.count_ > 0, self.mean_, np.nan)

    def variance(self, ddof=0):
        """
        Variance of every feature.

        ddof=0 gives the population variance and ddof=1 the sample variance.
        A feature with ddof or fewer values gives NaN.
        """
        self._check_has_data()
        if ddof < 0:
            raise ValueError("ddof must not be negative.")
        divisor = self.count_ - ddof
        return np.where(divisor > 0, self.m2_ / np.maximum(divisor, 1), np.nan)

    def std(self, ddof=0):
        """Standard deviation of every feature."""
        return np.sqrt(self.variance(ddof))

    def minimum(self):
        """Smallest value seen in every feature."""
        self._check_has_data()
        return np.where(self.count_ > 0, self.min_, np.nan)

    def maximum(self):
        """Largest value seen in every feature."""
        self._check_has_data()
        return np.where(self.count_ > 0, self.max_, np.nan)

    def quantile(self, q):
        """
        Quantile(s) of every feature over the window.

        Parameters
        ----------
        q : float or array-like
            Values between 0 and 1.

        Returns
        -------
        np.ndarray
            Shape (n_features,) for one q, or (len(q), n_features).
        """
        self._check_has_data()
        return np.asarray(quantile(self.window_, q, axis=0, skipna=True))

    def histogram(self, feature=0, bins=10, range=None, density=False):
        """Histogram (counts, edges) of one feature over the window."""
        self._check_has_data()
        if not 0 <= feature < self.window_.shape[1]:
            raise ValueError(
                f"feature must be between 0 and {self.window_.shape[1] - 1}, got {feature}."
            )
        return histogram(
            self.window_[:, feature], bins=bins, range=range, density=density, skipna=True
        )


class StreamingHistogram:
    """
    Histogram with fixed bins whose counts build up as chunks arrive.

    Only the bin counts are stored, so memory does not grow with the stream.
    With ``window_chunks`` set, only the newest chunks are counted (a sliding window). Values outside the bins are counted in ``n_outside_``. NaN values are skipped.

    Parameters
    ----------
    bins : int or array-like, default=10
        Number of equal-width bins (``range`` is then required), or the bin edges.
    range : tuple of float, optional
        (low, high) of the bins when ``bins`` is an int.
    window_chunks : int, optional
        Number of newest chunks to keep. None keeps every chunk.
    """

    def __init__(self, bins=10, range=None, window_chunks=None):
        if isinstance(bins, int):
            if range is None:
                raise ValueError("range is required when bins is an integer.")
            if bins < 1 or range[0] >= range[1]:
                raise ValueError("bins must be positive and range must be (low, high) with low < high.")
            self.edges_ = np.linspace(range[0], range[1], bins + 1)
        else:
            edges = np.asarray(bins, dtype=float)
            if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("Bin edges must be a 1D, strictly increasing array.")
            self.edges_ = edges
        if window_chunks is not None and (not isinstance(window_chunks, int) or window_chunks < 1):
            raise ValueError("window_chunks must be a positive integer or None.")
        self.window_chunks = window_chunks
        self.reset()

    def reset(self):
        """Forget every chunk."""
        self.chunk_counts = []
        self.chunk_outside = []
        self.n_outside_ = 0
        return self

    def update(self, values):
        """Add the values of one chunk (any shape; it is flattened)."""
        values = np.asarray(values)
        if values.dtype.kind not in "biuf":
            raise TypeError("values must contain real numbers only.")
        values = values.astype(float).ravel()
        if np.isinf(values).any():
            raise ValueError("values must not contain infinite values.")
        values = values[~np.isnan(values)]

        counts, _ = np.histogram(values, bins=self.edges_)
        self.chunk_counts.append(counts)
        self.chunk_outside.append(len(values) - counts.sum())
        if self.window_chunks is not None and len(self.chunk_counts) > self.window_chunks:
            self.chunk_counts.pop(0)
            self.chunk_outside.pop(0)
        self.n_outside_ = int(sum(self.chunk_outside))
        return self

    def result(self, density=False):
        """
        Return (counts, edges). With density=True the counts are divided so the
        histogram integrates to 1.
        """
        if self.chunk_counts:
            counts = np.sum(self.chunk_counts, axis=0)
        else:
            counts = np.zeros(len(self.edges_) - 1, dtype=int)
        if not density:
            return counts, self.edges_
        total = counts.sum()
        if total == 0:
            return np.zeros(len(counts)), self.edges_
        return counts / (total * np.diff(self.edges_)), self.edges_
