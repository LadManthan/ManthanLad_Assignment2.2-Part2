import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, impurity


def make_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


def test_impurity_values():
    assert impurity([5, 0]) == 0.0
    assert impurity([5, 5]) == pytest.approx(0.5)
    assert impurity([5, 5], "entropy") == pytest.approx(1.0)
    # An empty node has no impurity rather than a division by zero.
    assert impurity([0, 0]) == 0.0


def test_fit_learns_a_single_threshold():
    X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
    y = np.array([0, 0, 0, 1, 1, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.get_depth() == 1
    assert 3.0 < tree.threshold_[0] < 10.0
    assert tree.score(X, y) == 1.0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_max_depth_is_respected(criterion):
    X, y = make_data()
    tree = DecisionTreeClassifier(criterion=criterion, max_depth=3).fit(X, y)
    assert tree.get_depth() <= 3
    assert tree.get_n_leaves() <= 8
    assert tree.score(X, y) > 0.85


def test_partial_fit_waits_for_full_buffer():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=250)
    tree.partial_fit(X[:100], y[:100])
    tree.partial_fit(X[100:200], y[100:200])
    assert tree.n_retrains_ == 0
    with pytest.raises(RuntimeError, match="not fitted"):
        tree.predict(X)

    tree.partial_fit(X[200:300], y[200:300])  # 300 >= 250 -> retrain
    assert tree.n_retrains_ == 1
    assert tree.n_buffered == 0
    assert tree.predict(X).shape == (400,)


def test_predict_uses_last_tree_between_retrains():
    X, y = make_data()
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=200)
    tree.partial_fit(X[:200], y[:200])
    before = tree.predict(X)
    # Opposite labels go into the buffer but do not change the tree yet.
    tree.partial_fit(X[200:300], 1 - y[200:300])
    np.testing.assert_array_equal(tree.predict(X), before)


def test_buffered_retrain_matches_fit_on_same_data():
    X, y = make_data()
    streamed = DecisionTreeClassifier(max_depth=5, buffer_size=400)
    for start in range(0, 400, 50):
        streamed.partial_fit(X[start : start + 50], y[start : start + 50])
    batch = DecisionTreeClassifier(max_depth=5).fit(X, y)
    np.testing.assert_array_equal(streamed.predict(X), batch.predict(X))


def test_constant_features_and_ties_give_majority_leaf():
    # Zero variance: no threshold can separate equal values.
    X = np.ones((6, 2))
    y = np.array([0, 1, 1, 1, 0, 1])
    tree = DecisionTreeClassifier().fit(X, y)
    assert tree.get_n_leaves() == 1
    np.testing.assert_array_equal(tree.predict(X), np.ones(6))
    # A 50/50 leaf picks the smaller class label.
    tie = DecisionTreeClassifier().fit(np.zeros((4, 1)), np.array([1, 0, 1, 0]))
    assert tie.predict([[0.0]])[0] == 0


def test_string_labels_and_probabilities():
    X, y = make_data()
    labels = np.where(y == 1, "yes", "no")
    tree = DecisionTreeClassifier(max_depth=3, max_features="sqrt", random_state=0).fit(X, labels)
    assert set(tree.predict(X)) <= {"yes", "no"}
    np.testing.assert_allclose(tree.predict_probability(X).sum(axis=1), 1.0)


def test_chunks_with_more_columns_are_padded():
    # e.g. a OneHotEncoder that found a new category in the second chunk
    tree = DecisionTreeClassifier(buffer_size=4)
    tree.partial_fit(np.array([[0.0, 1.0], [1.0, 0.0]]), [0, 1])
    tree.partial_fit(np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), [0, 1])
    assert tree.n_features_ == 3
    assert tree.predict(np.array([[1.0, 0.0]])).shape == (1,)


def test_invalid_input_raises():
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="NaN"):
        tree.fit([[1.0], [np.nan]], [0, 1])
    with pytest.raises(ValueError, match="2D"):
        tree.fit([1.0, 2.0], [0, 1])
    with pytest.raises(ValueError, match="labels"):
        tree.fit([[1.0], [2.0]], [0, 1, 1])
    with pytest.raises(TypeError):
        tree.fit([["a"], ["b"]], [0, 1])


@pytest.mark.parametrize(
    "kwargs",
    [{"criterion": "mse"}, {"max_depth": 0}, {"min_samples_split": 1},
     {"buffer_size": 0}, {"max_features": 0}, {"max_features": 1.5}],
)
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**kwargs)
