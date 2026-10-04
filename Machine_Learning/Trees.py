"""
Decision trees implemented from scratch, with scikit-learn as the reference.

Covers the classification tree (Gini / entropy criteria) and the regression
tree (variance criterion). Cost-complexity pruning and the bagged ensemble
(random forest) build on this module.

Conventions:
X : np.ndarray, shape (n_samples, n_features), dtype float64
y : np.ndarray, shape (n_samples,) -- int labels for classification,
    float targets for regression

Metric signatures are (y_true, y_pred), in that order, like sklearn.metrics.
Impurity functions take a probability vector p (it sums to 1); the split
machinery passes class *counts* and normalises at the boundary.

pandas is used only at the I/O boundary (read_csv), then converted once via
.to_numpy(). The split search is a numerical inner loop over homogeneous
floats: a DataFrame pays for an index and per-element dtype dispatch on every
access, whereas an ndarray is one contiguous block whose slices are views.

Correctness bar: each hand-written component must match scikit-learn's
equivalent to within numerical noise given identical hyperparameters.
Run `python3 Machine_Learning/test_trees.py` -- it is the contract.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_xy(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV whose last column is the target.

    Returns X (n_samples, n_features) float64 and y (n_samples,), keeping y's
    own dtype: int for classification labels, float for regression targets.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"{path}: expected at least one feature column plus a target")
    X = frame.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = frame.iloc[:, -1].to_numpy()
    if y.dtype == object:
        raise ValueError(
            f"{path}: target column is non-numeric; encode labels to ints before loading"
        )
    return X, y


def load_split(
    path: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """X_train, X_test, y_train, y_test, sklearn's order, seeded explicitly."""
    X, y = load_xy(path)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _check_same_shape(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if np.shape(y_true) != np.shape(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same shape, got "
            f"{np.shape(y_true)} and {np.shape(y_pred)}"
        )


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check_same_shape(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check_same_shape(y_true, y_pred)
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _check_same_shape(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_true_mean = np.full(y_true.shape, np.mean(y_true))
    if np.array_equal(y_true, y_pred):
        return 1.0
    if mse(y_true, y_true_mean) == 0:
        return 0.0
    return 1 - mse(y_true, y_pred) / mse(y_true, y_true_mean)


"""
Best split metrics.
Many options. To ensure each split never increase impurity, we need the evaluation functions to
be concave wrt to p, the probability of class 1.
Indeed, p = p_left * n_left/n + p_right * n_right/n. And the function being concave means that
n_left/n * Impurity(p_left) + n_right/n * Impurity(p_right) <= Impurity(p)
So the weighted haverage impurity decrease with splits.
Gini and entropy are stricly concave, ensure a strictly positive decrease in impurity at each split
if p_left != p_right .
"""


def gini(p: np.ndarray) -> float:
    """
    Compute gini index from p 
    """
    return 1 - np.sum(p**2)

def entropy(p: np.ndarray)-> float:
    return np.sum(np.nan_to_num(-p * np.log2(p)))

def misclassification_rate(p: np.ndarray)-> float:
    return 1 - np.max(p)
    #the node predicts the mode, the error is everything else


# Criterion registry: the single place a criterion name is bound to a function.
# Phase 2 registers "variance" here and nothing else changes.
CRITERIA = {
    "gini": gini,
    "entropy": entropy,
    "misclassification_rate": misclassification_rate,
}


def weighted_impurity_decrease(
    counts_left: np.ndarray, counts_right: np.ndarray, criterion: str
) -> float:
    """Impurity decrease of a split, from the two children's class counts.

    counts_left[k] / counts_right[k] = number of samples of class k on that side.
    An empty child contributes nothing: its weight is 0, so the result is 0.0
    for a split that puts every sample on one side -- never nan.
    """
    if criterion not in CRITERIA:
        raise ValueError(
            f"unknown criterion {criterion!r}; expected one of {sorted(CRITERIA)}"
        )
    impurity = CRITERIA[criterion]

    counts_left = np.asarray(counts_left, dtype=np.float64)
    counts_right = np.asarray(counts_right, dtype=np.float64)
    if counts_left.shape != counts_right.shape:
        raise ValueError(
            f"children must cover the same classes, got shapes "
            f"{counts_left.shape} and {counts_right.shape}"
        )
    if np.any(counts_left < 0) or np.any(counts_right < 0):
        raise ValueError("class counts must be non-negative")

    n_left = counts_left.sum()
    n_right = counts_right.sum()
    n = n_left + n_right
    if n == 0:
        raise ValueError("cannot score a split of an empty node")

    parent = (counts_left + counts_right) / n
    total = impurity(parent)
    if n_left > 0:
        total -= (n_left / n) * impurity(counts_left / n_left)
    if n_right > 0:
        total -= (n_right / n) * impurity(counts_right / n_right)
    return float(total)


"""
Naive tree: 
get data
get best split based on impurity metric improvement
build tree
"""