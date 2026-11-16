import numpy as np
import pytest

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import Imputer, OneHotEncoder, StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier


def make_stream(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)) * [1, 100, 1, 1] + [0, 5000, 0, 0]
    y = (X[:, 0] > 0).astype(int)
    X[rng.random(X.shape) < 0.05] = np.nan  # missing values
    return X, y


def make_pipeline(buffer_size=300):
    return Pipeline(
        [
            ("impute", Imputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
        ]
    )


def test_pipeline_partial_fit_end_to_end():
    X, y = make_stream()
    pipe = make_pipeline()
    for start in range(0, 1000, 100):
        pipe.partial_fit(X[start : start + 100], y[start : start + 100])
    assert pipe.named_steps["model"].n_retrains_ == 3
    assert pipe.score(X, y) > 0.85
    assert pipe.transform(X).shape == (1000, 4)
    # Each column got its own scaler copy, as in the Assignment 1 Pipeline.
    scalers = pipe._fitted_transformers["scale"]
    assert len(scalers) == 4
    assert scalers[1].mean_ == pytest.approx(5000, rel=0.01)


def test_pipeline_one_hot_encoder_grows_inside_stream():
    pipe = Pipeline([("encode", OneHotEncoder(handle_unknown="ignore")),
                     ("model", DecisionTreeClassifier(buffer_size=4))])
    pipe.partial_fit(np.array([["a"], ["b"]]), [0, 1])
    pipe.partial_fit(np.array([["c"], ["a"]]), [1, 0])  # new category -> new column
    assert pipe.transform(np.array([["a"], ["c"]])).shape == (2, 3)
    np.testing.assert_array_equal(pipe.predict(np.array([["a"], ["b"]])), [0, 1])


def test_pipeline_errors():
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", StandardScaler())])
    pipe = make_pipeline()
    with pytest.raises(RuntimeError, match="fitted"):
        pipe.predict(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="y is required"):
        pipe.partial_fit(np.zeros((2, 4)))


def test_trainer_logs_every_chunk():
    X, y = make_stream()
    trainer = StreamTrainer(make_pipeline(buffer_size=300), window_size=200)
    for start in range(0, 1000, 100):
        trainer.process_chunk(X[start : start + 100], y[start : start + 100])

    accuracy = trainer.history("accuracy")
    assert len(accuracy) == 10
    # The first 3 chunks are scored before any tree exists.
    assert np.isnan(accuracy[:3]).all() and not np.isnan(accuracy[3:]).any()
    np.testing.assert_array_equal(trainer.history("n_retrains")[[2, 5, 9]], [1, 2, 3])
    assert trainer.history("n_samples_seen")[-1] == 1000
    assert (trainer.history("memory_bytes") > 0).all()
    assert trainer.history("cumulative_accuracy")[-1] == pytest.approx(np.nanmean(accuracy[3:]))
    with pytest.raises(KeyError):
        trainer.history("loss")


def test_trainer_with_bare_tree_and_bad_model():
    X, y = make_stream()
    X = np.nan_to_num(X)
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=100))
    trainer.run((X[s : s + 100], y[s : s + 100]) for s in range(0, 500, 100))
    assert len(trainer.fit_log) == 5
    assert trainer.last_y_pred.shape == (100,)
    with pytest.raises(TypeError, match="partial_fit"):
        StreamTrainer(object())
