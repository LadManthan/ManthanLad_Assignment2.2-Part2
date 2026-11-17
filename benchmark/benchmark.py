"""Benchmarks for the streaming decision tree.

Part 1 - loop vs vectorised:
    Five calculations used by the package, each written once with plain
    Python loops and once with NumPy. Both versions are checked to give the
    same answer before they are timed.

Part 2 - buffer_size:
    The same data stream is run through a StandardScaler + DecisionTree pipeline with buffer_size values, recording accuracy, number of retrains and training time.

Run from the project folder:
    python benchmark/benchmark.py
    python benchmark/benchmark.py --part loop --repeats 3
    python benchmark/benchmark.py --part buffer --buffer-sizes 100 500 2000
"""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

# Make the package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "NumCompute"))

from numcompute_stream.benchmarking import compare, format_table
from numcompute_stream.metrics import confusion_matrix
from numcompute_stream.utils import batches
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.utils import update_mean_var
from numcompute_stream.visualise import compare_models


# ---------------------------------------------------------------------------
# Part 1: loop vs vectorisation of same calculations.
# ---------------------------------------------------------------------------

# 1. Pairwise Euclidean distance: D[i, j] = sqrt(sum_k (a[i, k] - b[j, k])^2)
def pairwise_distance_loop(a, b):
    """Distances between every row of a (n, d) and every row of b (m, d), shape (n, m)."""
    distances = []
    for point_a in a.tolist():
        row = []
        for point_b in b.tolist():
            total = 0.0
            for value_a, value_b in zip(point_a, point_b):
                total += (value_a - value_b) ** 2
            row.append(math.sqrt(total))
        distances.append(row)
    return np.array(distances)


def pairwise_distance_vectorised(a, b):
    """Same as the loop version. Broadcasting builds all (n, m, d) differences at once."""
    difference = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(difference**2, axis=2))


# 2. Gini impurity of each row of class counts: G = 1 - sum_k (c_k / n)^2
def gini_loop(counts):
    """Gini impurity of every row of a (n_rows, n_classes) count matrix."""
    result = []
    for row in counts.tolist():
        total = sum(row)
        squares = 0.0
        for count in row:
            squares += (count / total) ** 2
        result.append(1.0 - squares)
    return np.array(result)


def gini_vectorised(counts):
    p = counts / counts.sum(axis=1, keepdims=True)
    return 1.0 - np.sum(p**2, axis=1)


# 3. Best split of one feature: the threshold with the lowest weighted Gini,
#    (n_left * G_left + n_right * G_right) / n, over all positions in sorted order.

def best_split_loop(x, y, n_classes):
    """
    Best threshold of feature x for labels y (ints 0 .. n_classes - 1).

    Walks through the samples in sorted order, moving one sample at a time from the right to left side and updating the class counts.

    Returns
    -------
    tuple
        (threshold, weighted Gini at that threshold)
    """
    pairs = sorted(zip(x.tolist(), y.tolist()))
    n = len(pairs)
    left = [0] * n_classes
    right = [0] * n_classes
    for _, label in pairs:
        right[label] += 1

    best_score, best_threshold = math.inf, None
    for i in range(n - 1):
        label = pairs[i][1]
        left[label] += 1
        right[label] -= 1
        
        if pairs[i][0] == pairs[i + 1][0]:
            continue  
        
        n_left, n_right = i + 1, n - i - 1
        gini_left = 1.0 - sum((c / n_left) ** 2 for c in left)
        gini_right = 1.0 - sum((c / n_right) ** 2 for c in right)
        score = (n_left * gini_left + n_right * gini_right) / n
        
        if score < best_score:
            best_score = score
            best_threshold = (pairs[i][0] + pairs[i + 1][0]) / 2
    
    return best_threshold, best_score


def best_split_vectorised(x, y, n_classes):
    order = np.argsort(x, kind="stable")
    x_sorted = x[order]
    one_hot = np.zeros((len(x), n_classes))
    one_hot[np.arange(len(x)), y[order]] = 1.0
    left = np.cumsum(one_hot, axis=0)[:-1]
    right = one_hot.sum(axis=0) - left
    n_left = np.arange(1, len(x))[:, None]
    n_right = len(x) - n_left
    gini_left = 1.0 - np.sum((left / n_left) ** 2, axis=1)
    gini_right = 1.0 - np.sum((right / n_right) ** 2, axis=1)
    score = (n_left[:, 0] * gini_left + n_right[:, 0] * gini_right) / len(x)
    score[x_sorted[1:] == x_sorted[:-1]] = np.inf
    i = np.argmin(score)
    
    return (x_sorted[i] + x_sorted[i + 1]) / 2, score[i]


# 4. Streaming mean and variance (Welford):
#    mean_n = mean_{n-1} + (x - mean_{n-1}) / n
#    M2_n   = M2_{n-1} + (x - mean_{n-1}) * (x - mean_n),   variance = M2 / n

def running_variance_loop(data):
    """Mean and variance of every column, updated one value at a time."""
    n_features = data.shape[1]
    count = 0
    mean = [0.0] * n_features
    m2 = [0.0] * n_features
    
    for row in data.tolist():
        count += 1
        for j in range(n_features):
            delta = row[j] - mean[j]
            mean[j] += delta / count
            m2[j] += delta * (row[j] - mean[j])
    
    return np.array(mean), np.array(m2) / count


def running_variance_vectorised(data, chunk_size=500):
    """Same result, but each chunk of rows is merged in one NumPy step (as the scaler does)."""
    n_features = data.shape[1]
    count = np.zeros(n_features)
    mean = np.zeros(n_features)
    m2 = np.zeros(n_features)
    
    for chunk in batches(data, chunk_size):
        count, mean, m2 = update_mean_var(count, mean, m2, chunk)
    
    return mean, m2 / count


# 5. Confusion matrix: M[i, j] = number of samples with true class i, predicted j
def confusion_matrix_loop(y_true, y_pred, n_classes):
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        matrix[t][p] += 1
    return np.array(matrix)


def confusion_matrix_vectorised(y_true, y_pred, n_classes):
    return confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))


def build_cases(seed=0):
    """
    Inputs for every calculation.

    Returns
    -------
    dict
        name -> (loop function, vectorised function, arguments)
    """
    rng = np.random.default_rng(seed)
    points_a = rng.normal(size=(200, 10))
    points_b = rng.normal(size=(150, 10))
    counts = rng.integers(0, 50, size=(20_000, 5)).astype(float)
    counts[:, 0] += 1  
    
    x = rng.normal(size=5_000)
    y = (x + rng.normal(scale=0.8, size=5_000) > 0).astype(int)
    data = rng.normal(loc=1000.0, size=(20_000, 8))
    y_true = rng.integers(0, 5, size=100_000)
    y_pred = rng.integers(0, 5, size=100_000)
    
    return {
        "pairwise distance (200 x 150, d=10)": (
            pairwise_distance_loop, pairwise_distance_vectorised, (points_a, points_b)
        ),
        "Gini impurity (20000 rows x 5 classes)": (
            gini_loop, gini_vectorised, (counts,)
        ),
        "best split threshold (5000 samples)": (
            best_split_loop, best_split_vectorised, (x, y, 2)
        ),
        "running mean/variance (20000 x 8)": (
            running_variance_loop, running_variance_vectorised, (data,)
        ),
        "confusion matrix (100000 labels, 5 classes)": (
            confusion_matrix_loop, confusion_matrix_vectorised, (y_true, y_pred, 5)
        ),
    }


def run_loop_benchmark(seed=0, repeats=5, warmup=1):
    """
    Check every loop/vectorised pair gives the same answer, then time both.

    Returns
    -------
    dict
        name -> result of ``numcompute_stream.benchmarking.compare`` with the keys
        "Python loop" and "NumPy vectorised".
    """
    results = {}
    
    for name, (loop_function, vectorised_function, arguments) in build_cases(seed).items():
        np.testing.assert_allclose(
            np.concatenate([np.ravel(v) for v in np.atleast_1d(loop_function(*arguments))]),
            np.concatenate([np.ravel(v) for v in np.atleast_1d(vectorised_function(*arguments))]),
            rtol=1e-9,
            err_msg=f"{name}: loop and vectorised results differ",
        )
        results[name] = compare(
            {"Python loop": loop_function, "NumPy vectorised": vectorised_function},
            *arguments,
            repeats=repeats,
            warmup=warmup,
        )
    return results


# ---------------------------------------------------------------------------
# Part 2: effect of buffer_size on accuracy and retraining time.
# ---------------------------------------------------------------------------

def make_stream(n_samples=20_000, n_features=8, seed=0):
    """
    Two-class data whose rule changes halfway through (concept drift).

    The label is the sign of a weighted sum of the features plus an interaction between the first two. Halfway through the weights flip sign, so a model has to retrain to keep up.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    weights = rng.normal(size=n_features)
    interaction = 0.8 * X[:, 0] * X[:, 1]
    score = X @ weights + interaction
    half = n_samples // 2
    score[half:] = X[half:] @ -weights + interaction[half:]
    score += rng.normal(scale=0.5, size=n_samples)
    
    return X, (score > 0).astype(int)


def run_stream(X, y, buffer_size, chunk_size=100, max_depth=6):
    """Run one test-then-train pass over the stream and return the StreamTrainer."""
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("tree", DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size)),
        ]
    )
    trainer = StreamTrainer(pipeline, window_size=1000)
    trainer.run(zip(batches(X, chunk_size), batches(y, chunk_size)))
    return trainer


def run_buffer_benchmark(X, y, buffer_sizes=(100, 250, 500, 1000, 2000), chunk_size=100):
    """
    Stream the data once per buffer size.

    Returns
    -------
    rows : list of dict
        One row per buffer size: accuracy over the whole stream, rolling accuracy over the last 1000 samples, number of retrains, total training time and average time per retrain.
    trainers : dict
        buffer size -> StreamTrainer, so the per-chunk logs can be plotted.
    """
    rows = []
    trainers = {}
    for buffer_size in buffer_sizes:
        trainer = run_stream(X, y, buffer_size, chunk_size)
        n_retrains = int(trainer.history("n_retrains")[-1])
        train_seconds = trainer.history("fit_seconds").sum()
        rows.append(
            {
                "buffer_size": buffer_size,
                "accuracy": trainer.accuracy.result(),
                "rolling_accuracy": trainer.rolling_accuracy.result(),
                "n_retrains": n_retrains,
                "train_seconds": train_seconds,
                "seconds_per_retrain": train_seconds / max(n_retrains, 1),
            }
        )
        trainers[buffer_size] = trainer
    
    return rows, trainers


def format_buffer_table(rows):
    """Text table of the buffer size results."""
    if len(rows) == 0:
        raise ValueError("rows must contain at least one result.")
    header = "buffer | accuracy | last 1000 acc | retrains | train (s) | s / retrain"
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['buffer_size']:>6} | {row['accuracy']:>8.4f} | {row['rolling_accuracy']:>13.4f} | "
            f"{row['n_retrains']:>8} | {row['train_seconds']:>9.4f} | {row['seconds_per_retrain']:>11.5f}"
        )
    return "\n".join(lines)


def _positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv=None):
    """Run the benchmarks from the command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=["all", "loop", "buffer"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=_positive_int, default=20_000)
    parser.add_argument("--chunk-size", type=_positive_int, default=100)
    parser.add_argument("--buffer-sizes", type=_positive_int, nargs="+", default=[100, 250, 500, 1000, 2000])
    parser.add_argument("--plot", help="save a rolling-accuracy plot of the smallest vs largest buffer here")
    args = parser.parse_args(argv)

    if args.part in ("all", "loop"):
        print("=== Part 1: Python loop vs NumPy vectorised ===\n")
        for name, result in run_loop_benchmark(args.seed, args.repeats, args.warmup).items():
            speedup = result["Python loop"]["median_seconds"] / result["NumPy vectorised"]["median_seconds"]
            print(name)
            print(format_table(result))
            print(f"Median speedup: {speedup:.1f}x\n")

    if args.part in ("all", "buffer"):
        print(f"=== Part 2: buffer_size ({args.samples} samples, chunks of {args.chunk_size}) ===\n")
        X, y = make_stream(args.samples, seed=args.seed)
        rows, trainers = run_buffer_benchmark(X, y, args.buffer_sizes, args.chunk_size)
        print(format_buffer_table(rows))
        if args.plot:
            small, large = min(args.buffer_sizes), max(args.buffer_sizes)
            compare_models(
                trainers[small].history("rolling_accuracy"),
                trainers[large].history("rolling_accuracy"),
                labels=(f"buffer_size={small}", f"buffer_size={large}"),
                title="Rolling accuracy (last 1000 samples)",
                ylabel="Accuracy",
                save_path=args.plot,
                show=False,
            )
            print(f"\nPlot saved to {args.plot}")


if __name__ == "__main__":
    main()
