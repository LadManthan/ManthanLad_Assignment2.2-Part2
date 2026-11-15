import matplotlib.pyplot as plt
import numpy as np


def _finish(fig, save_path, show):
    """Save and/or show a figure. A figure that is not shown is closed."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def _as_series(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty 1D array, got shape {values.shape}.")
    return values


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric",
                          save_path=None, show=True):
    """
    Line plot of one value per chunk (e.g. accuracy). NaN values leave a gap.

    Parameters
    ----------
    metric_values : array-like, shape (n_chunks,)
    title, ylabel : str
    save_path : str, optional
    show : bool, default=True
    """
    values = _as_series(metric_values, "metric_values")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(values)), values, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), title="Model comparison",
                   ylabel="Metric", save_path=None, show=True):
    """
    Plot the same metric for two models on one chart, e.g. trees with
    different ``buffer_size`` values.

    Parameters
    ----------
    metric1, metric2 : array-like, shape (n_chunks,)
    labels : two strings for the legend
    """
    first = _as_series(metric1, "metric1")
    second = _as_series(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError(f"labels must have exactly 2 names, got {len(labels)}.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(first)), first, marker="o", markersize=3, label=labels[0])
    ax.plot(np.arange(len(second)), second, marker="s", markersize=3, label=labels[1])
    ax.set_title(title)
    ax.set_xlabel("Chunk")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth",
                                     save_path=None, show=True):
    """
    Plot the prediction and the true value of every sample in a chunk.

    Predictions are green circles and ground truth values are red triangles, with a line joining the two for each sample, so the length of the line shows how far off the prediction is. For class labels a correct prediction sits on its ground truth (no line), so the lines mark the mistakes.

    Numbers are plotted as they are. Text labels (e.g. "yes"/"no") are given one row each on the y axis.

    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
    title : str
    save_path : str, optional
    show : bool, default=True
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or len(y_true) == 0:
        raise ValueError(
            "y_true and y_pred must be non-empty 1D arrays of the same length, "
            f"got shapes {y_true.shape} and {y_pred.shape}."
        )

    if y_true.dtype.kind in "biuf" and y_pred.dtype.kind in "biuf":
        true_values = y_true.astype(float)
        pred_values = y_pred.astype(float)
        classes = None
    else:
        classes = np.union1d(y_true.astype(str), y_pred.astype(str))
        true_values = np.searchsorted(classes, y_true.astype(str)).astype(float)
        pred_values = np.searchsorted(classes, y_pred.astype(str)).astype(float)
    n_wrong = int(np.sum(true_values != pred_values))
    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.vlines(x, pred_values, true_values, color="black", linewidth=0.8, zorder=1)
    ax.plot(x, pred_values, "o", color="green", markersize=7, label="Prediction", zorder=2)
    ax.plot(x, true_values, "^", color="red", markersize=6, label="Ground Truth", zorder=3)
    if classes is not None:
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes.tolist())
    ax.set_xlim(-1, len(y_true))
    ax.set_title(f"{title} ({n_wrong} wrong out of {len(y_true)})")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value" if classes is None else "Class")
    ax.legend()
    return _finish(fig, save_path, show)


def plot_tree(tree, feature_names=None, class_names=None, title="Decision tree",
              save_path=None, show=True):
    """
    Draw a fitted DecisionTreeClassifier as boxes joined by lines.

    Each split node shows its question ("feature <= threshold"), the number of samples and the class counts. 
    Each leaf shows the predicted class.
    The left child answers "yes" to the question and the right child "no".

    Parameters
    ----------
    tree : DecisionTreeClassifier
        A tree that has been trained at least once.
    feature_names : list of str, optional
        Names of the input columns. Defaults to x0, x1, ...
    class_names : list of str, optional
        Names for ``tree.classes_``. Defaults to the class labels themselves.
    title : str
    save_path : str, optional
    show : bool, default=True

    Returns
    -------
    matplotlib.figure.Figure
    """
    if not getattr(tree, "fitted", False):
        raise ValueError("The tree has not been trained yet, so there is nothing to plot.")
    n_nodes = len(tree.feature_)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(tree.n_features_)]
    if class_names is None:
        class_names = [str(c) for c in tree.classes_]
    if len(feature_names) < tree.n_features_ or len(class_names) != len(tree.classes_):
        raise ValueError("feature_names and class_names must match the tree's features and classes.")

    # Depth of every node. Children are stored after their parent.
    depth = np.zeros(n_nodes, dtype=int)
    for node in range(n_nodes):
        if tree.feature_[node] >= 0:
            depth[tree.left_[node]] = depth[node] + 1
            depth[tree.right_[node]] = depth[node] + 1

    # Leaves are spread evenly from left to right (depth-first order), and each split node is halfway between its two children.
    x = np.zeros(n_nodes)
    next_leaf = 0
    stack = [0]
    while stack:
        node = stack.pop()
        if tree.feature_[node] < 0:
            x[node] = next_leaf
            next_leaf += 1
        else:
            stack.append(tree.right_[node])
            stack.append(tree.left_[node])
    for node in range(n_nodes - 1, -1, -1):
        if tree.feature_[node] >= 0:
            x[node] = (x[tree.left_[node]] + x[tree.right_[node]]) / 2

    width = max(6.0, 1.6 * next_leaf)
    height = max(3.0, 1.5 * (depth.max() + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    font_size = 8 if next_leaf <= 12 else 6

    for node in range(n_nodes):
        counts = tree.value_[node].astype(int)
        if tree.feature_[node] >= 0:
            for child, answer in ((tree.left_[node], "yes"), (tree.right_[node], "no")):
                ax.plot([x[node], x[child]], [-depth[node], -depth[child]], color="grey", lw=1, zorder=1)
                ax.text((x[node] + x[child]) / 2, -(depth[node] + depth[child]) / 2, answer,
                        fontsize=font_size - 1, color="grey", ha="center", va="center",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5})
            name = feature_names[tree.feature_[node]]
            text = f"{name} <= {tree.threshold_[node]:.3g}\nsamples = {counts.sum()}\ncounts = {counts.tolist()}"
            colour = "#dde8f5"
        else:
            predicted = class_names[int(np.argmax(counts))]
            text = f"class = {predicted}\nsamples = {counts.sum()}\ncounts = {counts.tolist()}"
            colour = "#e3f1df"
        ax.text(x[node], -depth[node], text, fontsize=font_size, ha="center", va="center", zorder=2,
                bbox={"boxstyle": "round", "facecolor": colour, "edgecolor": "grey"})

    ax.set_xlim(-0.8, next_leaf - 0.2)
    ax.set_ylim(-depth.max() - 0.6, 0.6)
    ax.set_title(f"{title} (depth {depth.max()}, {next_leaf} leaves)")
    ax.axis("off")
    return _finish(fig, save_path, show)
