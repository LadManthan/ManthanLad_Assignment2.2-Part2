import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  #drawing graph to memory only no window

from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
    plot_tree,
)


def test_plots_are_saved_to_file(tmp_path):
    accuracy = np.array([np.nan, 0.6, 0.7, 0.8])  # NaN chunks before the first retrain
    plot_metric_over_time(accuracy, "Accuracy", "Accuracy", save_path=tmp_path / "a.png", show=False)
    compare_models(accuracy, accuracy[::-1], ("small", "large"), save_path=tmp_path / "b.png", show=False)
    fig = plot_predictions_vs_ground_truth(
        ["cat", "dog", "dog"], ["cat", "cat", "dog"], save_path=tmp_path / "c.png", show=False
    )
    assert "1 wrong out of 3" in fig.axes[0].get_title()
    for name in ("a.png", "b.png", "c.png"):
        assert (tmp_path / name).stat().st_size > 0


def test_plot_input_errors():
    with pytest.raises(ValueError, match="non-empty"):
        plot_metric_over_time([], show=False)
    with pytest.raises(ValueError, match="exactly 2"):
        compare_models([1.0], [1.0], labels=("a",), show=False)
    with pytest.raises(ValueError, match="same length"):
        plot_predictions_vs_ground_truth([0, 1], [0], show=False)


def test_plot_tree_draws_every_node(tmp_path):
    X = np.array([[1.0, 0.0], [2.0, 1.0], [8.0, 0.0], [9.0, 1.0]])
    tree = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    fig = plot_tree(tree, ["age", "fare"], ["died", "survived"], save_path=tmp_path / "t.png", show=False)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("age <= 5" in t for t in texts)
    assert sum("class = " in t for t in texts) == tree.get_n_leaves()
    with pytest.raises(ValueError, match="not been trained"):
        plot_tree(DecisionTreeClassifier(), show=False)

def test_predictions_plot_joins_each_pair(tmp_path):
    fig = plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 1], save_path=tmp_path / "p.png", show=False)
    ax = fig.axes[0]
    assert "2 wrong out of 4" in ax.get_title()
    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["Prediction", "Ground Truth"]