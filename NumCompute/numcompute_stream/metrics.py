import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def _validate_labels(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> tuple[NDArray, NDArray]:
    """
    Validate true and predicted labels.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.

    Returns
    -------
    tuple of numpy.ndarray
        Validated true and predicted label arrays.

    Raises
    ------
    ValueError
        If either input is not one-dimensional, is empty, has a
        mismatched shape, or contains NaN or infinity in a
        floating-point or complex array.

    Complexity: O(n) time and O(n) space for n labels.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")

    if true.size == 0 or pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    if pred.dtype.kind in "fc" and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred must not contain NaN or infinity")

    return true, pred


def _safe_divide(
    numerator: int | float | np.integer | np.floating,
    denominator: int | float | np.integer | np.floating,
    zero_division: int = 0,
) -> float:
    """
    Divide two values while handling a zero denominator.

    Parameters
    ----------
    numerator : int or float
        Numerator of the division.
    denominator : int or float
        Denominator of the division.
    zero_division : int, default=0
        Value to return when the denominator is zero. Must be 0 or 1.

    Returns
    -------
    float
        Division result, or zero_division if the denominator is zero.

    Raises
    ------
    ValueError
        If zero_division is not 0 or 1.

    Complexity: O(1) time and O(1) space.
    """
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")

    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def accuracy(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Calculate the fraction of correct predictions.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels, with the same shape as y_true.

    Returns
    -------
    float
        Number of correct predictions divided by total predictions.

    Raises
    ------
    ValueError
        If the arrays are empty, not one-dimensional, or have
        different shapes.

    Complexity: O(n) time for n predictions.
    """
    true, pred = _validate_labels(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    labels: npt.ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Calculate the confusion matrix for classification predictions.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    labels : array-like, optional
        Ordered list of class labels to include. If omitted, observed labels from both arrays are sorted automatically.

    Returns
    -------
    numpy.ndarray
        Integer matrix where rows represent actual classes and columns represent predicted classes.

    Raises
    ------
    ValueError
        If the input arrays are invalid, labels are empty or duplicated, or an observed label is missing from the provided labels.

    Complexity: O(n log k + k^2) time, where n is the number of samples and k is the number of classes.
    """
    true, pred = _validate_labels(y_true, y_pred)

    if labels is None:
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if labels.size == 0:
            raise ValueError("labels must not be empty")

        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")

    # Sort the labels once, then find every sample's label position with a
    # binary search instead of a Python loop over the samples.
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    true_pos = np.clip(np.searchsorted(sorted_labels, true), 0, labels.size - 1)
    pred_pos = np.clip(np.searchsorted(sorted_labels, pred), 0, labels.size - 1)

    true_found = sorted_labels[true_pos] == true
    if not np.all(true_found):
        raise ValueError(f"actual label {true[~true_found][0]!r} is not in labels")
    
    pred_found = sorted_labels[pred_pos] == pred
    if not np.all(pred_found):
        raise ValueError(f"predicted label {pred[~pred_found][0]!r} is not in labels")

    # Row i, column j is stored as the single number i * k + j, so one
    # bincount fills the whole k x k matrix.
    k = labels.size
    flat = np.bincount(order[true_pos] * k + order[pred_pos], minlength=k * k)
    matrix = flat.reshape(k, k).astype(np.int64)

    return matrix


def precision(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """Calculate binary precision for the selected positive class.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value returned when there are no predicted positive samples.
        Must be 0 or 1.

    Returns
    -------
    float
        Precision value: True positives divided by the sum of true and false positives.

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fp = np.count_nonzero(~actual_positive & predicted_positive)

    return _safe_divide(tp, tp + fp, zero_division)


def recall(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """Calculate binary recall for the selected positive class.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value returned when there are no actual positive samples. Must be 0 or 1.

    Returns
    -------
    float
        Recall value: True positives divided by the sum of true positives and
        false negatives.

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fn = np.count_nonzero(actual_positive & ~predicted_positive)

    return _safe_divide(tp, tp + fn, zero_division)


def f1(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    positive_label: object = 1,
    zero_division: int = 0,
) -> float:
    """
    Calculate the binary F1 score.

    Parameters
    ----------
    y_true : array-like
        Actual class labels.
    y_pred : array-like
        Predicted class labels.
    positive_label : object, default=1
        Label treated as the positive class.
    zero_division : int, default=0
        Value returned when a division denominator is zero. Must be 0 or 1.

    Returns
    -------
    float
        Harmonic mean of precision and recall.

    Raises
    ------
    ValueError
        If the labels are invalid or zero_division is not 0 or 1.

    Complexity: O(n) time for n samples.
    """
    p = precision(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    r = recall(
        y_true,
        y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )

    return _safe_divide(2 * p * r, p + r, zero_division)


def mse(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> float:
    """
    Calculate mean squared error for regression predictions.

    Parameters
    ----------
    y_true : array-like
        Actual numeric target values.
    y_pred : array-like
        Predicted numeric target values, matching y_true in shape.

    Returns
    -------
    float
        Mean of the squared differences between actual and predicted values.

    Raises
    ------
    TypeError
        If either input contains non-real numeric values.
    ValueError
        If the arrays have different shapes, are empty, or contain NaN or infinity.

    Complexity: O(n) time for n target values.
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.dtype.kind not in "iuf" or pred.dtype.kind not in "iuf":
        raise TypeError("y_true and y_pred must contain real numeric values")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must not contain NaN or infinity")

    true = true.astype(float, copy=False)
    pred = pred.astype(float, copy=False)

    return float(np.mean((true - pred) ** 2))


def roc_curve(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate ROC curve points for binary classification.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Numeric scores for the positive class. Higher values indicate
        a stronger prediction of the positive class.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    tuple of numpy.ndarray
        False positive rates, true positive rates, and descending classification thresholds. The first threshold is infinity.

    Raises
    ------
    TypeError
        If y_score contains non-real numeric values.
    ValueError
        If inputs are empty, not one-dimensional, have different shapes, contain non-finite values, or y_true does not contain exactly two classes.

    Complexity: O(n log n) time due to sorting n prediction scores.
    """
    true = np.asarray(y_true)
    scores = np.asarray(y_score)

    if true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")

    if true.size == 0 or scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")

    if true.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if scores.dtype.kind not in "iuf":
        raise TypeError("y_score must contain real numeric values")

    scores = scores.astype(float, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must not contain NaN or infinity")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    classes = np.unique(true)

    if classes.size != 2:
        raise ValueError("ROC curve requires exactly two classes")

    if not np.any(true == positive_label):
        raise ValueError("positive_label must be present in y_true")

    is_positive = true == positive_label
    n_positive = np.count_nonzero(is_positive)
    n_negative = true.size - n_positive

    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_positive = is_positive[order]

    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.cumsum(~sorted_positive)

    distinct_indices = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_indices = np.r_[
        distinct_indices,
        sorted_scores.size - 1,
    ]

    tp = cumulative_tp[threshold_indices]
    fp = cumulative_fp[threshold_indices]

    tpr = np.r_[0.0, tp / n_positive]
    fpr = np.r_[0.0, fp / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    return fpr, tpr, thresholds


def auc(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
) -> float:
    """
    Calculate the area under a curve using the trapezoidal rule.

    Parameters
    ----------
    x : array-like
        Monotonic x-coordinates of the curve.
    y : array-like
        Corresponding y-coordinates.

    Returns
    -------
    float
        Non-negative area under the curve.

    Raises
    ------
    TypeError
        If x or y contains non-real numeric values.
    ValueError
        If the inputs are not one-dimensional, have different lengths, contain fewer than two points, contain non-finite values, or x is not monotonic.

    Complexity: O(n) time for n curve points.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")

    if x_arr.size < 2:
        raise ValueError("at least two points are required")

    if x_arr.dtype.kind not in "iuf" or y_arr.dtype.kind not in "iuf":
        raise TypeError("x and y must contain real numeric values")

    x_arr = x_arr.astype(float, copy=False)
    y_arr = y_arr.astype(float, copy=False)

    if (
        not np.all(np.isfinite(x_arr))
        or not np.all(np.isfinite(y_arr))
    ):
        raise ValueError("x and y must not contain NaN or infinity")

    differences = np.diff(x_arr)

    if not (
        np.all(differences >= 0)
        or np.all(differences <= 0)
    ):
        raise ValueError("x values must be monotonic")

    return float(abs(np.trapezoid(y_arr, x_arr)))


def roc_auc_score(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    positive_label: object = 1,
) -> float:
    """
    Calculate the area under the receiver operating characteristic curve.

    Parameters
    ----------
    y_true : array-like
        Actual binary class labels.
    y_score : array-like
        Numeric scores for the positive class.
    positive_label : object, default=1
        Label treated as the positive class.

    Returns
    -------
    float
        ROC AUC score, generally between 0 and 1.

    Raises
    ------
    TypeError
        If y_score contains non-real numeric values.
    ValueError
        If the ROC curve inputs are invalid or y_true does not contain exactly two classes.

    Complexity: O(n log n) time for n prediction scores.
    """
    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
        positive_label=positive_label,
    )

    return auc(fpr, tpr)

def check_window(window_size):
    if window_size is not None and (not isinstance(window_size, int) or window_size < 1):
        raise ValueError("window_size must be a positive integer or None.")


class StreamingAccuracy:
    """
    Fraction of correct predictions over everything seen (or over a window).

    Without a window only two numbers are stored: correct and total. With a window the newest ``window_size`` right/wrong flags are kept.
    """

    def __init__(self, window_size=None):
        check_window(window_size)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.n_correct = 0
        self.n_seen = 0
        self.recent = np.empty(0, dtype=bool)
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = _validate_labels(y_true_chunk, y_pred_chunk)
        correct = y_true == y_pred
        self.n_correct += int(correct.sum())
        self.n_seen += len(correct)
        if self.window_size is not None:
            self.recent = np.concatenate([self.recent, correct])[-self.window_size :]
        return self

    def result(self):
        """Accuracy in [0, 1], or NaN before the first update."""
        if self.n_seen == 0:
            return float("nan")
        if self.window_size is not None:
            return float(self.recent.mean())
        return self.n_correct / self.n_seen


class StreamingConfusionMatrix:
    """
    Confusion matrix that adds up over chunks.

    Rows are true labels and columns are predicted labels, both in the sorted order of ``labels_``. A label that first shows up in a later chunk adds a new row and column; the old counts are moved into the bigger matrix.

    Parameters
    ----------
    window_size : int, optional
        If given, only the newest ``window_size`` samples are counted.
    """

    def __init__(self, window_size=None):
        check_window(window_size)
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.labels_ = None
        self.counts = None
        self.recent_true = None
        self.recent_pred = None
        self.n_seen = 0
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        y_true, y_pred = _validate_labels(y_true_chunk, y_pred_chunk)
        self.n_seen += len(y_true)

        if self.window_size is not None:
            # Old samples drop out of a window, so recount the window every time.
            if self.recent_true is None:
                self.recent_true, self.recent_pred = y_true, y_pred
            else:
                self.recent_true = np.concatenate([self.recent_true, y_true])
                self.recent_pred = np.concatenate([self.recent_pred, y_pred])
            self.recent_true = self.recent_true[-self.window_size :]
            self.recent_pred = self.recent_pred[-self.window_size :]
            self.labels_ = np.union1d(self.recent_true, self.recent_pred)
            self.counts = confusion_matrix(self.recent_true, self.recent_pred, labels=self.labels_)
            return self

        chunk_labels = np.union1d(y_true, y_pred)
        if self.labels_ is None:
            self.labels_ = chunk_labels
            self.counts = np.zeros((len(chunk_labels), len(chunk_labels)), dtype=int)
        else:
            all_labels = np.union1d(self.labels_, chunk_labels)
            if len(all_labels) > len(self.labels_):
                grown = np.zeros((len(all_labels), len(all_labels)), dtype=int)
                old = np.searchsorted(all_labels, self.labels_)
                grown[old[:, None], old] = self.counts
                self.labels_ = all_labels
                self.counts = grown
        self.counts = self.counts + confusion_matrix(y_true, y_pred, labels=self.labels_)
        return self

    def result(self):
        """The matrix, shape (n_labels, n_labels). Empty before the first update."""
        if self.counts is None:
            return np.zeros((0, 0), dtype=int)
        return self.counts.copy()


class _ConfusionMetric:
    """
    Shared code for precision, recall and F1, which all come from the
    confusion matrix through the per-class TP, FP and FN counts.

    Parameters
    ----------
    average : {"macro", "micro", "weighted", "binary"}, default="macro"
        "macro" is the plain mean over classes
        "weighted" weights each class by its number of true samples
        "micro" pools the counts of all classes,
        "binary" reports only ``positive_label``.
    positive_label : default=1
        Class used when average="binary".
    window_size : int, optional
        Count only the newest ``window_size`` samples.
    zero_division : int, default=0
        Value (0 or 1) used when a class has a 0 / 0 score
    """

    def __init__(self, average="macro", positive_label=1, window_size=None, zero_division=0):
        if average not in ("macro", "micro", "weighted", "binary"):
            raise ValueError(f"average must be macro, micro, weighted or binary, got {average!r}.")
        if zero_division not in (0, 1):
            raise ValueError("zero_division must be 0 or 1")
        self.average = average
        self.positive_label = positive_label
        self.zero_division = zero_division
        self.matrix = StreamingConfusionMatrix(window_size)

    def reset(self):
        self.matrix.reset()
        return self

    def update(self, y_true_chunk, y_pred_chunk):
        self.matrix.update(y_true_chunk, y_pred_chunk)
        return self

    def _divide(self, top, bottom):
        safe_bottom = np.where(bottom > 0, bottom, 1.0)
        return np.where(bottom > 0, top / safe_bottom, self.zero_division)

    def _per_class(self, tp, fp, fn):
        """Score of every class at once (vectorised)."""
        top, bottom = self._fraction(tp, fp, fn)
        return self._divide(top, bottom)

    def _one_score(self, tp, fp, fn):
        """Score for a single class (or pooled counts)."""
        top, bottom = self._fraction(tp, fp, fn)
        return _safe_divide(top, bottom, self.zero_division)

    def result(self):
        """Current value of the metric, or NaN before the first update."""
        counts = self.matrix.result().astype(float)
        if counts.size == 0:
            return float("nan")
        tp = np.diag(counts)
        fp = counts.sum(axis=0) - tp
        fn = counts.sum(axis=1) - tp

        if self.average == "binary":
            where = np.flatnonzero(self.matrix.labels_ == self.positive_label)
            if len(where) == 0:
                raise ValueError(f"positive_label {self.positive_label!r} has not been seen yet.")
            i = where[0]
            return self._one_score(tp[i], fp[i], fn[i])
        if self.average == "micro":
            return self._one_score(tp.sum(), fp.sum(), fn.sum())

        scores = self._per_class(tp, fp, fn)
        if self.average == "macro":
            return float(scores.mean())
        support = counts.sum(axis=1)
        return float(np.sum(scores * support) / support.sum())


class StreamingPrecision(_ConfusionMetric):
    """Precision = TP / (TP + FP), updated chunk by chunk."""

    def _fraction(self, tp, fp, fn):
        return tp, tp + fp


class StreamingRecall(_ConfusionMetric):
    """Recall = TP / (TP + FN), updated chunk by chunk."""

    def _fraction(self, tp, fp, fn):
        return tp, tp + fn


class StreamingF1(_ConfusionMetric):
    """F1 = 2TP / (2TP + FP + FN), updated chunk by chunk."""

    def _fraction(self, tp, fp, fn):
        return 2 * tp, 2 * tp + fp + fn


class StreamingAUC:
    """
    ROC AUC for two classes, updated chunk by chunk.
    AUC depends on the ranking of all scores, so the labels and scores are stored. Use ``window_size`` to keep memory bounded (a rolling AUC).

    Parameters
    ----------
    positive_label : default=1
    window_size : int, optional
    """

    def __init__(self, positive_label=1, window_size=None):
        check_window(window_size)
        self.positive_label = positive_label
        self.window_size = window_size
        self.reset()

    def reset(self):
        self.y_true = None
        self.y_score = None
        return self

    def update(self, y_true_chunk, y_score_chunk):
        """Add true labels and the scores for the positive class."""
        y_true, y_score = _validate_labels(y_true_chunk, y_score_chunk)
        y_score = y_score.astype(float)
        if self.y_true is None:
            self.y_true, self.y_score = y_true, y_score
        else:
            self.y_true = np.concatenate([self.y_true, y_true])
            self.y_score = np.concatenate([self.y_score, y_score])
        if self.window_size is not None:
            self.y_true = self.y_true[-self.window_size :]
            self.y_score = self.y_score[-self.window_size :]
        return self

    def result(self):
        """AUC in [0, 1], or NaN until both classes have been seen."""
        if self.y_true is None or len(np.unique(self.y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(self.y_true, self.y_score, positive_label=self.positive_label))
