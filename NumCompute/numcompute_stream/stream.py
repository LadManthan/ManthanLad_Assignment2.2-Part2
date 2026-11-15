from time import perf_counter

import numpy as np

from numcompute_stream.metrics import StreamingAccuracy
from numcompute_stream.utils import check_labels, memory_bytes


class StreamTrainer:
    """
    Runs a model (or Pipeline) over a stream of chunks and keeps a log.

    ``fit_chunk`` trains on a chunk and ``score_chunk`` evaluates on one.
    ``process_chunk`` does both on the same chunk, scoring first and training second ("test-then-train"), so every chunk is scored by a model that has not seen it yet.

    Every call adds a record (a dict) to ``fit_log`` or ``score_log``:

    - fit_log:   chunk, n_samples, n_samples_seen, fit_seconds, memory_bytes, n_retrains
    - score_log: chunk, n_samples, accuracy, cumulative_accuracy, rolling_accuracy, score_seconds

    Parameters
    ----------
    model : object
        Anything with ``partial_fit(X, y)`` and ``predict(X)``.
    window_size : int, optional
        Number of newest samples for the rolling accuracy. None makes the
        rolling accuracy the same as the cumulative one.
    verbose : bool, default=False
        Print one line per processed chunk.

    Notes
    -----
    A buffered tree cannot predict until its first buffer is full. While that is the case ``predict`` raises RuntimeError, and the chunk's accuracy is logged as NaN instead.
    """

    def __init__(self, model, window_size=None, verbose=False):
        if not hasattr(model, "partial_fit") or not hasattr(model, "predict"):
            raise TypeError("model must have partial_fit and predict methods.")
        self.model = model
        self.window_size = window_size
        self.verbose = verbose
        self.accuracy = StreamingAccuracy()
        self.rolling_accuracy = StreamingAccuracy(window_size=window_size)
        self.fit_log = []
        self.score_log = []

    def _memory(self):
        """Bytes held in NumPy arrays by the model, or by every pipeline step."""
        
        if hasattr(self.model, "steps"):
            return sum(memory_bytes(step) for _, step in self.model.steps)
        return memory_bytes(self.model)

    def _n_retrains(self):
        model = self.model.steps[-1][1] if hasattr(self.model, "steps") else self.model
        return getattr(model, "n_retrains_", np.nan)

    def fit_chunk(self, X, y):
        """Train on one chunk and log the time, memory and retrain count."""
        
        X = np.asarray(X)
        y = check_labels(y, len(X))
        start = perf_counter()
        self.model.partial_fit(X, y)
        seconds = perf_counter() - start

        seen_before = self.fit_log[-1]["n_samples_seen"] if self.fit_log else 0
        self.fit_log.append(
            {
                "chunk": len(self.fit_log),
                "n_samples": len(y),
                "n_samples_seen": seen_before + len(y),
                "fit_seconds": seconds,
                "memory_bytes": self._memory(),
                "n_retrains": self._n_retrains(),
            }
        )
        return self

    def score_chunk(self, X, y):
        """
        Predict one chunk, update the accuracies and log them.

        Returns
        -------
        float
            Accuracy on this chunk, or NaN if the model cannot predict yet.
        """
        X = np.asarray(X)
        y = check_labels(y, len(X))
        start = perf_counter()
        try:
            y_pred = np.asarray(self.model.predict(X))
        except RuntimeError:
            y_pred = None  # not trained yet
        seconds = perf_counter() - start

        if y_pred is None:
            chunk_accuracy = float("nan")
        else:
            chunk_accuracy = float(np.mean(y_pred == y))
            self.accuracy.update(y, y_pred)
            self.rolling_accuracy.update(y, y_pred)
            self.last_y_true = y
            self.last_y_pred = y_pred

        self.score_log.append(
            {
                "chunk": len(self.score_log),
                "n_samples": len(y),
                "accuracy": chunk_accuracy,
                "cumulative_accuracy": self.accuracy.result(),
                "rolling_accuracy": self.rolling_accuracy.result(),
                "score_seconds": seconds,
            }
        )
        return chunk_accuracy

    def process_chunk(self, X, y):
        """Score a chunk, then train on it. Returns the two log records merged."""
        
        self.score_chunk(X, y)
        self.fit_chunk(X, y)
        record = dict(self.score_log[-1])
        record.update(self.fit_log[-1])
        if self.verbose:
            print(
                f"chunk {record['chunk']:4d} | acc {record['accuracy']:.3f} | "
                f"cumulative {record['cumulative_accuracy']:.3f} | "
                f"retrains {record['n_retrains']} | memory {record['memory_bytes'] / 1024:.1f} KB"
            )
        return record

    def run(self, chunks):
        """Call ``process_chunk`` on every (X, y) pair of an iterable."""
        
        for X_chunk, y_chunk in chunks:
            self.process_chunk(X_chunk, y_chunk)
        return self

    def history(self, name):
        """
        One logged value per chunk as an array, e.g. ``history("rolling_accuracy")``.

        Raises
        ------
        KeyError
            If neither log has a value with that name.
        """
        
        for log in (self.score_log, self.fit_log):
            if log and name in log[0]:
                return np.array([record[name] for record in log], dtype=float)
        raise KeyError(f"No logged value named {name!r}.")
