from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def euclidean_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute Euclidean distance between equal-shaped numeric arrays.

    Returns a scalar; shape mismatch raises ValueError. Complexity is O(n) time and O(n) temporary space for n elements.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.linalg.norm(left - right))


def manhattan_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute Manhattan distance between equal-shaped numeric arrays.

    Returns a scalar; shape mismatch raises ValueError. Complexity is O(n) time and O(n) temporary space.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    return float(np.sum(np.abs(left - right)))


def cosine_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Compute cosine similarity of equal-shaped numeric arrays.

    Returns zero if either vector has zero norm. 
    Shape mismatch raises ValueError. 
    Complexity is O(n) time and O(n) temporary space.
    """
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.shape != right.shape:
        raise ValueError("a and b must have the same shape.")
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return (
        0.0
        if denominator == 0
        else float(np.dot(left.ravel(), right.ravel()) / denominator)
    )


def relu(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Apply ReLU elementwise and return an array with x's shape."""
    return np.maximum(np.asarray(x, dtype=float), 0.0)


def sigmoid(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Apply stable logistic sigmoid elementwise, preserving the input shape."""
    values = np.asarray(x, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0
    
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def softmax(x: npt.ArrayLike, axis: int = -1) -> npt.NDArray[np.float64]:
    """Apply stable softmax along axis; output has the same shape as x.

    Empty inputs raise ValueError. Positive-infinity logits share probability uniformly; an all-negative-infinity slice also maps to a uniform vector.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("x must not be empty.")
    maximum = np.max(values, axis=axis, keepdims=True)
    positive_infinity = np.isposinf(values)
    positive_count = positive_infinity.sum(axis=axis, keepdims=True)
    negative_infinity_only = np.isneginf(maximum)
    
    shifted = np.where(
        positive_infinity,
        0.0,
        values - np.where(np.isfinite(maximum), maximum, 0.0),
    )
    exponentials = np.exp(shifted)
    denominator = np.sum(exponentials, axis=axis, keepdims=True)
    probabilities = np.divide(
        exponentials,
        denominator,
        out=np.zeros_like(values),
        where=denominator > 0,
    )
    inf_probabilities = np.divide(
        positive_infinity,
        positive_count,
        out=np.zeros_like(values),
        where=positive_count > 0,
    )
    uniform = np.full_like(values, 1.0 / values.shape[axis])
    
    probabilities = np.where(positive_count > 0, inf_probabilities, probabilities)
    probabilities = np.where(negative_infinity_only, uniform, probabilities)
    return np.where(np.isnan(maximum), np.nan, probabilities)


def logsumexp(
    x: npt.ArrayLike, axis: int | None = None, keepdims: bool = False
) -> npt.NDArray[np.float64] | np.float64:
    """Compute log(sum(exp(x))) stably along the selected axis.

    Positive- and negative-infinite input maxima return their corresponding infinite result rather than generating ``inf - inf`` NaNs.
    """
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("x must not be empty.")
    maximum = np.max(values, axis=axis, keepdims=True)
    
    finite_shift = np.where(np.isfinite(maximum), maximum, 0.0)
    shifted = np.where(np.isfinite(maximum), values - finite_shift, -np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        finite_result = finite_shift + np.log(
            np.sum(np.exp(shifted), axis=axis, keepdims=True)
        )
    result = np.where(np.isposinf(maximum), np.inf, finite_result)
    result = np.where(np.isneginf(maximum), -np.inf, result)
    result = np.where(np.isnan(maximum), np.nan, result)
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    elif not keepdims and axis is None:
        result = np.squeeze(result)
    return result


def topk_indices(
    values: npt.ArrayLike, k: int, largest: bool = True
) -> npt.NDArray[np.intp]:
    """Return ordered indices of k largest/smallest values in a 1D array.

    Requires ``1 <= k <= len(values)``. Expected selection work is O(n), plus O(k log k) sorting, with O(n) workspace.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, (bool, np.bool_))
        or k < 1
        or k > array.size
    ):
        raise ValueError("k must be between 1 and the number of values.")
    
    split = array.size - k if largest else k - 1
    candidates = (
        np.argpartition(array, split)[-k:]
        if largest
        else np.argpartition(array, split)[:k]
    )
    order = np.argsort(array[candidates], kind="stable")
    if largest:
        order = order[::-1]
    return candidates[order]


def batches(data: npt.ArrayLike, batch_size: int) -> Iterator[npt.NDArray[np.generic]]:
    """Yield slices along data's first axis, including a final partial batch.

    Requires data with at least one dimension and positive integer batch_size.
    Each yielded array has at most ``batch_size`` rows and preserves trailing dimensions. Slicing itself is O(1) view construction per batch.
    """
    values = np.asarray(data)
    if values.ndim == 0:
        raise ValueError("data must have at least one dimension.")
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def check_2d(X, name="X"):
    """
    Return X as a 2D NumPy array with at least one row.

    Parameters
    ----------
    X : array-like
        Input data, shape (n_samples, n_features).
    name : str, default="X"
        Name used in error messages.

    Returns
    -------
    np.ndarray
        X as a 2D array.

    Raises
    ------
    ValueError
        If X is not 2D or has no rows.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {X.shape}.")
    if X.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    return X


def check_numeric(X, name="X", n_features=None, allow_nan=True):
    """
    Return X as a 2D float array after checking its type, width and values.

    Parameters
    ----------
    X : array-like
        Input data, shape (n_samples, n_features).
    name : str, default="X"
        Name used in error messages.
    n_features : int, optional
        Expected number of columns. Not checked when None.
    allow_nan : bool, default=True
        If False, NaN values raise an error. Infinity is always rejected.

    Returns
    -------
    np.ndarray
        Float array with shape (n_samples, n_features).

    Raises
    ------
    TypeError
        If X does not hold real numbers.
    ValueError
        If X has the wrong shape or holds infinity (or NaN when not allowed).
    """
    X = check_2d(X, name)
    if X.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numbers only.")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"{name} has {X.shape[1]} features but {n_features} were expected."
        )
    X = X.astype(float)
    if np.isinf(X).any():
        raise ValueError(f"{name} must not contain infinite values.")
    if not allow_nan and np.isnan(X).any():
        raise ValueError(f"{name} must not contain NaN values. Use an Imputer first.")
    return X


def check_labels(y, n_samples, name="y"):
    """
    Return y as a 1D array with one label per sample.

    Raises
    ------
    ValueError
        If y is not 1D or its length is not n_samples.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {y.shape}.")
    if len(y) != n_samples:
        raise ValueError(f"{name} has {len(y)} labels but X has {n_samples} rows.")
    return y


def update_mean_var(count, mean, m2, chunk):
    """
    Merge a chunk into the running count, mean and M2 of every column.

    M2 is the sum of squared differences from the mean, so the variance is M2 / count. 
    The chunk's own mean and M2 are computed first and then merged with the running values (Chan et al.'s parallel form of Welford's method).
    This avoids the cancellation problem of the sum-of-squares formula.
    NaN values in the chunk are skipped.

    Parameters
    ----------
    count, mean, m2 : np.ndarray
        Running values, each of shape (n_features,).
    chunk : np.ndarray
        New rows, shape (n_chunk, n_features).

    Returns
    -------
    tuple of np.ndarray
        Updated (count, mean, m2).
    """
    valid = ~np.isnan(chunk)
    chunk_count = valid.sum(axis=0)
    
    chunk_mean = np.where(valid, chunk, 0.0).sum(axis=0) / np.maximum(chunk_count, 1)
    chunk_m2 = (np.where(valid, chunk - chunk_mean, 0.0) ** 2).sum(axis=0)

    new_count = count + chunk_count
    delta = chunk_mean - mean
    weight = chunk_count / np.maximum(new_count, 1)
    new_mean = mean + delta * weight
    new_m2 = m2 + chunk_m2 + delta**2 * count * weight
    return new_count, new_mean, new_m2


def memory_bytes(obj):
    """
    Add up the bytes of the NumPy arrays stored on an object.

    Arrays kept directly as attributes, or inside list attributes (like the tree's buffer), are counted. 
    Other Python objects are ignored, so this is an estimate of the data held, not of total process memory.
    """
    total = 0
    for value in vars(obj).values():
        if isinstance(value, np.ndarray):
            total += value.nbytes
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, np.ndarray):
                    total += item.nbytes
    return total
