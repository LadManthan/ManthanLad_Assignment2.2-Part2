import numpy as np
import pytest

from numcompute import metrics as a1_metrics
from numcompute_stream.metrics import confusion_matrix, roc_auc_score
from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingPrecision,
    StreamingRecall,
)

Y_TRUE = np.array([0, 1, 1, 0, 2, 2, 1, 0])
Y_PRED = np.array([0, 1, 0, 0, 2, 1, 1, 2])


def feed(metric, size=3):
    for start in range(0, len(Y_TRUE), size):
        metric.update(Y_TRUE[start : start + size], Y_PRED[start : start + size])
    return metric


def test_accuracy_over_chunks_and_reset():
    accuracy = feed(StreamingAccuracy())
    assert accuracy.result() == pytest.approx(np.mean(Y_TRUE == Y_PRED))
    accuracy.reset()
    assert np.isnan(accuracy.result())


def test_rolling_accuracy_only_counts_window():
    accuracy = StreamingAccuracy(window_size=2)
    accuracy.update([0, 0, 0], [1, 1, 1])  # all wrong
    accuracy.update([1, 1], [1, 1])  # all right
    assert accuracy.result() == 1.0


def test_confusion_matrix_grows_with_new_labels():
    matrix = StreamingConfusionMatrix()
    matrix.update([0, 1], [0, 1])
    matrix.update([2, 2], [2, 0])  # label 2 appears for the first time
    np.testing.assert_array_equal(matrix.labels_, [0, 1, 2])
    np.testing.assert_array_equal(matrix.result(), [[1, 0, 0], [0, 1, 0], [1, 0, 1]])


def test_precision_recall_f1_against_hand_counts():
    # Class 1 in Y_TRUE/Y_PRED: TP=2, FP=1, FN=1
    assert feed(StreamingPrecision(average="binary")).result() == pytest.approx(2 / 3)
    assert feed(StreamingRecall(average="binary")).result() == pytest.approx(2 / 3)
    # Per class F1 = [2/3, 2/3, 1/2] -> macro 11/18; micro equals accuracy.
    assert feed(StreamingF1(average="macro")).result() == pytest.approx(11 / 18)
    assert feed(StreamingF1(average="micro")).result() == pytest.approx(5 / 8)
    assert feed(StreamingRecall(average="weighted")).result() == pytest.approx(5 / 8)


def test_zero_division_and_missing_positive_label():
    precision = StreamingPrecision(average="macro", zero_division=0.0)
    precision.update([0, 1], [0, 0])  # class 1 never predicted -> 0 / 0
    assert precision.result() == pytest.approx(0.25)
    with pytest.raises(ValueError, match="positive_label"):
        StreamingPrecision(average="binary", positive_label=5).update([0], [0]).result()


def test_auc_accumulates_and_needs_both_classes():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    score = y + rng.normal(size=200)
    auc = StreamingAUC()
    auc.update(np.zeros(5), np.zeros(5))
    assert np.isnan(auc.result())  # one class only
    auc.reset()
    for start in range(0, 200, 50):
        auc.update(y[start : start + 50], score[start : start + 50])
    assert auc.result() == pytest.approx(roc_auc_score(y, score))


def test_bad_chunks_raise():
    with pytest.raises(ValueError, match="same shape"):
        StreamingAccuracy().update([0, 1], [0])
    with pytest.raises(ValueError, match="1D"):
        StreamingF1().update([[0]], [[0]])
    with pytest.raises(ValueError, match="window_size"):
        StreamingAccuracy(window_size=0)


def test_vectorised_confusion_matrix_matches_assignment1_loop():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 4, size=500)
    y_pred = rng.integers(0, 4, size=500)
    labels = [3, 0, 2, 1]  # unsorted on purpose
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred, labels=labels),
        a1_metrics.confusion_matrix(y_true, y_pred, labels=labels),
    )
