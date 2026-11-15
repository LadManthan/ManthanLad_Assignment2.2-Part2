import numpy as np

from numcompute_stream.utils import check_labels, check_numeric

# Gains smaller than this are treated as rounding noise, not a real split.
MIN_GAIN = 1e-12


def impurity(counts, criterion="gini"):
    """
    Gini impurity or entropy for every row of class counts.

    Gini    = 1 - sum(p_k^2)          (0 when pure, 0.5 at 50/50 for 2 classes)
    Entropy = -sum(p_k * log2(p_k))   (0 when pure, 1 bit at 50/50)

    Parameters
    ----------
    counts : array-like
        Class counts, shape (n_classes,) or (n_rows, n_classes).
    criterion : {"gini", "entropy"}, default="gini"

    Returns
    -------
    float or np.ndarray
        Impurity of each row. A row with no samples has impurity 0.
    """
    counts = np.asarray(counts, dtype=float)
    totals = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(totals, 1.0)
    if criterion == "gini":
        result = 1.0 - np.sum(p**2, axis=-1)
    else:
        # p * log2(p) goes to 0 as p goes to 0, so log2(1) = 0 is used for p = 0.
        log_p = np.log2(np.where(p > 0, p, 1.0))
        result = -np.sum(p * log_p, axis=-1)
    return np.where(totals[..., 0] > 0, result, 0.0)


class DecisionTreeClassifier:
    """
    Depth-limited classification tree with buffered retraining.

    ``fit(X, y)`` trains a new tree from scratch. 
    ``partial_fit(X, y)`` adds the chunk to a buffer. Once the buffer holds at least ``buffer_size`` samples, the tree is retrained on the buffer and the buffer is cleared. 
    Between retrains, ``predict`` uses the most recently trained tree.

    The tree is stored in flat arrays (one entry per node) instead of node objects, so prediction can move every sample down one level at a time with NumPy indexing.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
    max_depth : int, optional
        Maximum depth. None grows until leaves are pure or too small.
    min_samples_split : int, default=2
        A node with fewer samples than this becomes a leaf.
    max_features : int, float, "sqrt" or None, default=None
        Number of features tried at each split: an int count, a float fraction
        of all features, "sqrt" for sqrt(n_features), or None for all.
    buffer_size : int, default=500
        Samples collected by ``partial_fit`` before each retrain.
    random_state : int, optional
        Seed for picking feature subsets when max_features is set.

    Attributes
    ----------
    classes_ : np.ndarray
        Labels seen in the last training set, sorted.
    n_retrains_ : int
        Number of times a tree has been trained.
    feature_, threshold_, left_, right_, value_ : np.ndarray
        The tree. A node with feature_ == -1 is a leaf. Samples with
        X[:, feature] <= threshold go to left_, the rest to right_.
        value_ holds the class counts of the training samples in each node.
    """

    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        max_features=None,
        buffer_size=500,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}.")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 1):
            raise ValueError("max_depth must be a positive integer or None.")
        if not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer of at least 2.")
        if not isinstance(buffer_size, int) or buffer_size < 1:
            raise ValueError("buffer_size must be a positive integer.")
        if not (
            max_features is None
            or max_features == "sqrt"
            or (isinstance(max_features, int) and max_features >= 1)
            or (isinstance(max_features, float) and 0.0 < max_features <= 1.0)
        ):
            raise ValueError(
                "max_features must be None, 'sqrt', an int >= 1 or a float in (0, 1]."
            )

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.buffer_size = buffer_size
        self.random_state = random_state
        self.n_retrains_ = 0
        self.buffer_X = []
        self.buffer_y = []
        self.n_buffered = 0
        self.fitted = False

    #training

    def fit(self, X, y):
        """
        Train a new tree on X and y. Anything in the buffer is thrown away.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Finite numbers only (put an Imputer first if there are NaN values).
        y : array-like, shape (n_samples,)

        Returns
        -------
        self
        """
        X = check_numeric(X, allow_nan=False)
        y = check_labels(y, X.shape[0])
        self.clear_buffer()
        self._train(X, y)
        return self

    def partial_fit(self, X, y):
        """
        Add a chunk to the buffer and retrain once the buffer is full.

        Parameters
        ----------
        X : array-like, shape (n_chunk, n_features)
        y : array-like, shape (n_chunk,)

        Returns
        -------
        self
        """
        X = check_numeric(X, allow_nan=False)
        y = check_labels(y, X.shape[0])
        self.buffer_X.append(X)
        self.buffer_y.append(y)
        self.n_buffered += X.shape[0]
        if self.n_buffered < self.buffer_size:
            return self


        width = max(chunk.shape[1] for chunk in self.buffer_X)
        rows = []
        for chunk in self.buffer_X:
            rows.append(np.pad(chunk, ((0, 0), (0, width - chunk.shape[1]))))
        self._train(np.vstack(rows), np.concatenate(self.buffer_y))
        self.clear_buffer()
        return self

    def clear_buffer(self):
        """Drop every buffered sample."""
        self.buffer_X = []
        self.buffer_y = []
        self.n_buffered = 0

    def _n_features_to_try(self, n_features):
        """How many features each split looks at."""
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return max(1, int(self.max_features * n_features))

    def _best_split(self, X, y, n_classes, features):
        """
        Find the (feature, threshold) with the largest information gain.

        For one feature, sort the samples by value. Row i of the cumulative sum of the one-hot labels is then the class counts of the left side when the split is after sorted sample i, so every threshold of that feature is scored in one vectorised step instead of a loop over thresholds.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray of int, shape (n_samples,)
            Class index of each sample (0 .. n_classes - 1).
        n_classes : int
        features : np.ndarray
            Columns to try.

        Returns
        -------
        tuple or None
            (feature, threshold), or None if no split has a positive gain.
        """
        n_samples = len(y)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y] = 1.0
        total = one_hot.sum(axis=0)
        before = impurity(total, self.criterion)
        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left

        best = None
        best_gain = MIN_GAIN
        for feature in features:
            order = np.argsort(X[:, feature], kind="stable")
            values = X[order, feature]
            left = np.cumsum(one_hot[order], axis=0)[:-1]
            right = total - left
            after = (
                n_left * impurity(left, self.criterion)
                + n_right * impurity(right, self.criterion)
            ) / n_samples
            gain = before - after
            
            gain[values[1:] == values[:-1]] = -1.0
            i = np.argmax(gain)
            if gain[i] > best_gain:
                best_gain = gain[i]
                threshold = (values[i] + values[i + 1]) / 2.0
                
                if threshold >= values[i + 1]:
                    threshold = values[i]
                best = (int(feature), float(threshold))
        return best

    def _train(self, X, y):
        """Grow a new tree on X and y, replacing the current one."""
        rng = np.random.default_rng(self.random_state)
        self.classes_, y_index = np.unique(y, return_inverse=True)
        y_index = y_index.ravel()
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        n_try = self._n_features_to_try(n_features)

        
        feature, threshold, left, right, value = [], [], [], [], []

        def add_node(rows):
            feature.append(-1)
            threshold.append(0.0)
            left.append(-1)
            right.append(-1)
            value.append(np.bincount(y_index[rows], minlength=n_classes))
            return len(feature) - 1

        all_rows = np.arange(X.shape[0])
        stack = [(add_node(all_rows), all_rows, 0)]
        while stack:
            node, rows, depth = stack.pop()
            is_pure = np.count_nonzero(value[node]) == 1
            too_deep = self.max_depth is not None and depth >= self.max_depth
            if is_pure or too_deep or len(rows) < self.min_samples_split:
                continue

            if n_try < n_features:
                features = rng.choice(n_features, size=n_try, replace=False)
            else:
                features = np.arange(n_features)
            split = self._best_split(X[rows], y_index[rows], n_classes, features)
            if split is None:
                continue  # no question lowers the impurity, so this is a leaf

            feature[node], threshold[node] = split
            goes_left = X[rows, split[0]] <= split[1]
            left[node] = add_node(rows[goes_left])
            right[node] = add_node(rows[~goes_left])
            stack.append((left[node], rows[goes_left], depth + 1))
            stack.append((right[node], rows[~goes_left], depth + 1))

        self.feature_ = np.array(feature)
        self.threshold_ = np.array(threshold)
        self.left_ = np.array(left)
        self.right_ = np.array(right)
        self.value_ = np.array(value, dtype=float)
        self.n_features_ = n_features
        self.n_retrains_ += 1
        self.fitted = True

    # prediction

    def _check_fitted(self):
        if not self.fitted:
            raise RuntimeError(
                "DecisionTreeClassifier is not fitted yet. Call fit, or call "
                f"partial_fit until {self.buffer_size} samples are buffered."
            )

    def _leaves(self, X):
        """Return the index of the leaf each row of X ends up in."""
        n = X.shape[0]
        node = np.zeros(n, dtype=int)
        
        while True:
            node_feature = self.feature_[node]
            at_leaf = node_feature < 0
            if at_leaf.all():
                return node
            
            values = X[np.arange(n), np.where(at_leaf, 0, node_feature)]
            child = np.where(
                values <= self.threshold_[node], self.left_[node], self.right_[node]
            )
            node = np.where(at_leaf, node, child)

    def predict_probability(self, X):
        """
        Class probabilities from the class mix of each sample's leaf.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Extra columns are ignored and missing columns are taken as 0, to
            match how the buffer pads chunks.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Columns are in the order of ``classes_``.
        """
        self._check_fitted()
        X = check_numeric(X, allow_nan=False)
        if X.shape[1] < self.n_features_:
            X = np.pad(X, ((0, 0), (0, self.n_features_ - X.shape[1])))
        X = X[:, : self.n_features_]
        counts = self.value_[self._leaves(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Predict labels with the most recently trained tree.

        Returns
        -------
        np.ndarray, shape (n_samples,)

        Raises
        ------
        RuntimeError
            If no tree has been trained yet.
        """
        probabilities = self.predict_probability(X)
        # On a tie argmax picks the first (smallest) class.
        return self.classes_[np.argmax(probabilities, axis=1)]

    def score(self, X, y):
        """Accuracy of ``predict(X)`` against y."""
        X = check_numeric(X, allow_nan=False)
        y = check_labels(y, X.shape[0])
        return float(np.mean(self.predict(X) == y))

    def get_depth(self):
        """Depth of the current tree. A tree that is a single leaf has depth 0."""
        self._check_fitted()
        depth = np.zeros(len(self.feature_), dtype=int)
        
        for node in range(len(self.feature_)):
            if self.feature_[node] >= 0:
                depth[self.left_[node]] = depth[node] + 1
                depth[self.right_[node]] = depth[node] + 1
        return int(depth.max())

    def get_n_leaves(self):
        """Number of leaves in the current tree."""
        self._check_fitted()
        return int(np.sum(self.feature_ < 0))
