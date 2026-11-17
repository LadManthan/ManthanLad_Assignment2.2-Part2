# NumCompute

NumCompute is an educational numerical computing toolkit built primarily with **Python and NumPy**. Matplotlib is used for plotting and histogram display.

The project includes batch-oriented numerical utilities and a separate set of streaming components for processing data in chunks.

It does not depend on scikit-learn, pandas, or PyTorch.

---

## Features

- Load CSV data either all at once or in bounded-size chunks.
- Preprocess numeric and categorical features with imputers, scalers, and encoders.
- Sort, search, select, rank, and calculate descriptive statistics.
- Calculate classification and regression metrics, including streaming metrics.
- Estimate gradients and Jacobians using finite differences.
- Compose transformers and estimators into pipelines, including incremental pipelines.
- Train and evaluate a buffered decision tree over data streams.
- Visualize histograms, stream metrics, predictions, and decision trees.
- Benchmark Python and NumPy implementations.

---

## Project Structure

The tree below shows the project layout and modules.

```text
.
├── NumCompute/
│   ├── numcompute/
│   │   ├── __init__.py
│   │   ├── io.py
│   │   ├── preprocessing.py
│   │   ├── sort_search.py
│   │   ├── rank.py
│   │   ├── stats.py
│   │   ├── metrics.py
│   │   ├── optim.py
│   │   ├── pipeline.py
│   │   ├── utils.py
│   │   ├── histogram_display.py
│   │   └── benchmarking.py
│   └── numcompute_stream/
│       ├── io.py
│       ├── preprocessing.py
│       ├── sort_search.py
│       ├── rank.py
│       ├── stats.py
│       ├── metrics.py
│       ├── optim.py
│       ├── pipeline.py
│       ├── tree.py
│       ├── stream.py
│       ├── utils.py
│       ├── visualise.py
│       └── benchmarking.py
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   ├── test_pipeline.py
│   ├── test_utils.py
│   ├── test_benchmarking.py
│   ├── test_stream_benchmark.py
│   ├── test_stream_metrics.py
│   ├── test_stream_pipeline.py
│   ├── test_stream_preprocessing.py
│   ├── test_stream_stats.py
│   ├── test_stream_tree.py
│   ├── test_stream_utils.py
│   └── test_stream_visualise.py
├── benchmark/
│   └── benchmark.py
├── demo/
│   ├── quickstart.ipynb
│   └── stream_demo.ipynb
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── CODING_STYLE.md
├── .github/
│   └── workflows/
│       └── pytest.yml
└── README.md
```

---

## Requirements

* Python 3.10 or newer
* NumPy and Matplotlib (installed from `requirements.txt`)
* Pytest for development and tests (installed from `requirements-dev.txt`)

The core numerical algorithms are implemented in the project; the toolkit does not use external ML/DL libraries such as scikit-learn, pandas, or PyTorch.

---

## Installation

Clone the repository and move into its root directory:

```bash
git clone https://github.com/LadManthan/ManthanLad_Assignment2.2-Part2.git
cd ManthanLad_Assignment2.2-Part2
```

Create and activate a virtual environment.

**Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the package and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

The project provides the `numcompute` and `numcompute_stream` packages. Install Jupyter separately if you want to run the notebooks:

```bash
python -m pip install jupyter
```

---

## Modules

### `io.py` — Data loading

Supports:

* `load_csv(path, ...) -> (header, data)` for complete-table loading
* `iter_csv(path, chunksize=1000, ...)` for bounded-memory iteration
* Custom delimiters, requested or inferred dtypes, and fill/drop missing modes
* Numeric arrays for numeric tables and object arrays for mixed columns

`load_csv` returns the column names and a 2D NumPy array using
`numpy.loadtxt`. Use `iter_csv` to process data in bounded-memory chunks.

---
### `preprocessing.py` — Data preprocessing

Preprocessing components:

* `StandardScaler`
* `MinMaxScaler`
* `OneHotEncoder`
* `Imputer` for replacing NaN or configured missing-value markers

Preprocessors provide `fit(X)`, `transform(X)`, and `fit_transform(X)` methods.

```python
fit(X) -> self
transform(X) -> X_out
fit_transform(X) -> X_out
```

---

### `sort_search.py` - Sorting and searching algorithms

Includes:

* Stable sorting
* Multi-key sorting
* `topk(values, k, largest=True, return_indices=True)`
* Top-k selection using `np.argpartition`
* Quickselect
* Binary search

Binary search returns the insertion index and whether the value exists.

---

### `rank.py` - Ranking and percentile

`Ranker.rank(data, method="average")` supports the `average`, `dense`, and `ordinal` ranking methods and handles tied values.

```python
from numcompute.rank import Ranker

ranker = Ranker()
ranker.rank(data, method="average")
```

Percentile interpolation supports `linear`, `lower`, `higher`, and `midpoint` methods.

```python
ranker.percentile(data, q, interpolation="linear")
```

---

### `stats.py` — Descriptive statistics

Basic statistical functions:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Histogram
* Quantiles
* NaN-aware batch statistics and exact quantiles

---

### `metrics.py` — Model evaluation

Classification metrics:

```python
accuracy(y_true, y_pred)
precision(y_true, y_pred)
recall(y_true, y_pred)
f1(y_true, y_pred)
confusion_matrix(y_true, y_pred)
```

Regression metric:

```python
mse(y_true, y_pred)
```

Optional bonus functionality:

```python
roc_curve(y_true, y_score)
auc(x, y)
```

Precision, recall, and F1 accepts a `positive_label` argument. In the confusion matrix, rows represent true labels and columns represent predicted labels.

---

### `optim.py` — Numerical differentiation

The `grad` function estimates gradients using finite differences:

```python
grad(f, x, h=1e-5, method="central")
```

Supported methods are `central` and  `forward`. Jacobian estimation is provided through:

```python
jacobian(F, x, ...)
```

---

### `pipeline.py` — Pipeline composition

Provides a minimal interface for composing transformers and estimators.

Preprocessors use:

```python
fit(X)
transform(X)
```

Models use:

```python
fit(X, y)
predict(X)
```

The batch package does not include a built-in estimator; the streaming package includes a buffered `DecisionTreeClassifier`.

The module also provides pipeline chaining, `Compose` for sequential
transformers, and `FeatureUnion` for concatenating independent feature outputs.

Example:

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler())])
X_transformed = pipe.fit_transform(X)
```

---

### `utils.py` — Shared utilities

Contains reusable helper functionality shared across the project.

Utilities should only be added here when they are genuinely reusable across multiple modules.

---

### `histogram_display.py` — Histogram display

`display_histogram(heights, edges, density=False)` displays precomputed histogram values in Jupyter. It returns `False` when called outside a notebook.

---

### `benchmarking.py` — Performance measurement

Contains reusable micro-benchmarking functionality for measuring implementation performance.

`benchmark(function, ..., repeats=5, warmup=1)` returns minimum, median, and
mean elapsed times in seconds plus Python, NumPy, OS/platform, processor, and
CPU-count metadata. `compare()` benchmarks named implementations with shared
arguments, and `format_table()` renders timing and environment information.

---

## Streaming package

The `numcompute_stream` package provides streaming counterparts for CSV loading, preprocessing, sorting/searching, ranking, statistics, metrics, optimization, pipelines, utilities, and benchmarking. Streaming preprocessors and metrics maintain state across chunks; `Pipeline.partial_fit` passes each chunk through incrementally updated steps.

The package also provides:

* `DecisionTreeClassifier` — a depth-limited classifier that buffers chunks and periodically retrains.
* `StreamTrainer` — processes chunks using test-then-train evaluation and records timing, memory, retrain counts, and accuracy history.
* `visualise.py` — plots streaming metrics, compares models, displays predictions against ground truth, and draws decision trees.

Example:

```python
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(buffer_size=100)
trainer = StreamTrainer(model, window_size=5)
trainer.run(chunks)  # iterable of (X_chunk, y_chunk) pairs

rolling_accuracy = trainer.history("rolling_accuracy")
```

---

## Running Tests

Run the full test suite from the repository root:

```bash
python -m pytest
```

Run an individual test file when focusing on a particular module. For example:

```bash
python -m pytest tests/test_optim.py
```

The tests cover the batch modules and streaming functionality, including boundary conditions, invalid inputs, empty arrays, duplicate and tied values, NaN handling, input shapes, top-k limits, and chunked processing.

---

# Demo

The demonstrations are in:

```text
demo/quickstart.ipynb
demo/stream_demo.ipynb
```

Open either notebook in Jupyter and run the cells in order. `quickstart.ipynb` demonstrates the batch API, including:

1. CSV loading
2. Preprocessing
3. Sorting and searching
4. Ranking and percentiles
5. Statistics
6. Gradients and Jacobians
7. Pipeline usage
8. Benchmark results

`stream_demo.ipynb` demonstrates chunked loading, incremental preprocessing and model training, stream evaluation, and visualisation.

# Benchmark

The benchmark script and notebook compare implementations, particularly:

```text
Python loops
        vs.
NumPy vectorised implementations
```


Use `python benchmark/benchmark.py --help` to see the benchmark script options. Results depend on input size, software versions, hardware, and system load.

The loop-versus-NumPy benchmark checks that each implementation produces the same result, then measures each five times after one warm-up run. The following medians were measured on Windows 10 with Python 3.11.0 and NumPy 2.4.4:

| Calculation | Input size | Python loop (s) | NumPy vectorised (s) | Median speedup |
|---|---:|---:|---:|---:|
| Pairwise distances | 200 x 150 points, 10 features | 0.067501 | 0.003093 | 21.8x |
| Gini impurity | 20,000 rows x 5 classes | 0.038086 | 0.003515 | 10.8x |
| Best split threshold | 5,000 samples | 0.020707 | 0.001284 | 16.1x |
| Running mean and variance | 20,000 rows x 8 features | 0.090385 | 0.006230 | 14.5x |
| Confusion matrix | 100,000 labels, 5 classes | 0.015431 | 0.011158 | 1.4x |

These are hardware-specific measurements; rerun the benchmark on your system to obtain current results.

---