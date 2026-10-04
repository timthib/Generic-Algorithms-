"""
Contract tests for Trees.py -- run this BEFORE writing each phase, not after.

    python3 Machine_Learning/test_trees.py            # everything
    python3 Machine_Learning/test_trees.py M1         # one milestone
    python3 Machine_Learning/test_trees.py M1b        # one sub-step

Labels match ROADMAP.md milestones: M0, M1a..M1e, M2a...  A prefix selects
everything under it.

Each test states a contract. A phase whose functions do not exist yet reports
PENDING with the exact signature to implement -- that is the structure to code
against. FAIL means a contract is broken. The suite is the pass/fail gate of
ROADMAP.md; no phase is done while its tests are red.

No pytest: plain asserts, so it runs anywhere python3 and sklearn do.
"""
import os
import sys
import traceback

import numpy as np
import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier

import Trees as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CLASS = os.path.join(_ROOT, "data", "classification_data.csv")
DATA_REG = os.path.join(_ROOT, "data", "regression_data.csv")

_TESTS = []
_PENDING = []


def test(phase, contract):
    """Register a test under a phase, labelled by the contract it enforces."""
    def wrap(fn):
        _TESTS.append((phase, contract, fn))
        return fn
    return wrap


def requires(*names):
    """Skip-with-PENDING until Trees.py exposes these names."""
    missing = [n for n in names if not hasattr(T, n)]
    if missing:
        raise Pending(", ".join(missing))


class Pending(Exception):
    pass


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, f"expected {b}, got {a}"


# ------------------------------------------------------------------- M0 --
# Contract: data in, metrics out, both matching sklearn exactly.

@test("M0", "load_xy: X is float64 (n, d), y keeps its own dtype, last column is y")
def t_load_shapes():
    Xc, yc = T.load_xy(DATA_CLASS)
    Xr, yr = T.load_xy(DATA_REG)
    assert Xc.shape == (500, 2) and yc.shape == (500,), (Xc.shape, yc.shape)
    assert Xr.shape == (500, 2) and yr.shape == (500,), (Xr.shape, yr.shape)
    assert Xc.dtype == np.float64 and Xr.dtype == np.float64
    assert yc.dtype.kind in "iu", f"classification labels must be integer, got {yc.dtype}"
    assert yr.dtype.kind == "f", f"regression targets must be float, got {yr.dtype}"
    assert set(np.unique(yc)) == {0, 1}


@test("M0", "load_split: seeded, reproducible, 80/20, rows stay aligned with labels")
def t_load_split():
    a = T.load_split(DATA_CLASS, test_size=0.2, random_state=42)
    b = T.load_split(DATA_CLASS, test_size=0.2, random_state=42)
    c = T.load_split(DATA_CLASS, test_size=0.2, random_state=7)
    X_train, X_test, y_train, y_test = a
    assert len(X_train) == 400 and len(X_test) == 100
    assert len(y_train) == 400 and len(y_test) == 100
    for lhs, rhs in zip(a, b):
        assert np.array_equal(lhs, rhs), "same seed must give the same split"
    assert not np.array_equal(a[0], c[0]), "different seeds must give different splits"
    # rows must not be shuffled independently of their labels
    X_all, y_all = T.load_xy(DATA_CLASS)
    lookup = {row.tobytes(): label for row, label in zip(X_all, y_all)}
    for row, label in zip(X_test, y_test):
        assert lookup[row.tobytes()] == label, "a test row lost its label"


@test("M0", "metrics take (y_true, y_pred) and match sklearn")
def t_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, 3, 200))
    close(T.accuracy(y_true, y_pred), sklearn.metrics.accuracy_score(y_true, y_pred))

    y_true = rng.normal(size=200) * 10
    y_pred = y_true + rng.normal(size=200)
    close(T.mse(y_true, y_pred), sklearn.metrics.mean_squared_error(y_true, y_pred))
    close(T.r2(y_true, y_pred), sklearn.metrics.r2_score(y_true, y_pred))


@test("M0", "r2 is asymmetric: swapping the arguments must change the answer")
def t_r2_argument_order():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=200) * 10
    y_pred = 0.5 * y_true + rng.normal(size=200)
    assert abs(T.r2(y_true, y_pred) - T.r2(y_pred, y_true)) > 1e-6, (
        "r2(y_true, y_pred) == r2(y_pred, y_true): the arguments are not being used "
        "in sklearn's order"
    )


@test("M0", "r2 on a constant target matches sklearn (1.0 if exact, else 0.0)")
def t_r2_constant_target():
    y = np.ones(10)
    close(T.r2(y, y), sklearn.metrics.r2_score(y, y))
    close(T.r2(y, np.zeros(10)), sklearn.metrics.r2_score(y, np.zeros(10)))


# --------------------------------------------------------------- M1a --
# Contract: impurity of a probability vector; decrease from two count vectors.

@test("M1a", "impurity functions take a probability vector, on 2 and on k classes")
def t_impurity_values():
    close(T.gini(np.array([0.5, 0.5])), 0.5)
    close(T.gini(np.array([1.0, 0.0])), 0.0)
    close(T.gini(np.full(4, 0.25)), 0.75)          # max for k classes is 1 - 1/k
    close(T.misclassification_rate(np.array([0.5, 0.5])), 0.5)
    close(T.misclassification_rate(np.array([0.9, 0.1])), 0.1)


@test("M1a", "entropy handles a pure node: 0 * log2(0) is 0, not nan, and not a crash")
def t_entropy_zero_component():
    close(T.entropy(np.array([0.5, 0.5])), 1.0)
    close(T.entropy(np.full(4, 0.25)), 2.0)
    close(T.entropy(np.array([1.0, 0.0])), 0.0)
    close(T.entropy(np.array([0.0, 0.25, 0.75])), 0.8112781244591328)


@test("M1a", "weighted_impurity_decrease matches values computed by hand")
def t_decrease_known_values():
    # parent [400, 400]: gini 0.5
    close(T.weighted_impurity_decrease([300, 100], [100, 300], "gini"), 0.125)
    close(T.weighted_impurity_decrease([200, 400], [200, 0], "gini"), 0.5 - 1 / 3)
    close(T.weighted_impurity_decrease([400, 0], [0, 400], "gini"), 0.5)
    close(T.weighted_impurity_decrease([400, 0], [0, 400], "entropy"), 1.0)


@test("M1a", "gini separates split A from split B where misclassification ties them")
def t_gini_beats_misclassification():
    a = ([300, 100], [100, 300])
    b = ([200, 400], [200, 0])
    close(
        T.weighted_impurity_decrease(*a, "misclassification_rate"),
        T.weighted_impurity_decrease(*b, "misclassification_rate"),
    )
    assert T.weighted_impurity_decrease(*b, "gini") > T.weighted_impurity_decrease(*a, "gini")


@test("M1a", "an unknown criterion raises, it does not return None")
def t_unknown_criterion():
    for bad in ("giny", "Gini", "", "mse"):
        try:
            T.weighted_impurity_decrease([1, 1], [1, 1], bad)
        except ValueError:
            continue
        raise AssertionError(f"criterion {bad!r} was accepted silently")


@test("M1a", "a one-sided split scores 0.0, never nan")
def t_empty_child():
    for criterion in T.CRITERIA:
        got = T.weighted_impurity_decrease([400, 200], [0, 0], criterion)
        assert not np.isnan(got), f"{criterion}: empty child gave nan"
        close(got, 0.0)


@test("M1a", "impurity decrease is never negative (concavity), on random splits")
def t_decrease_non_negative():
    rng = np.random.default_rng(3)
    for criterion in T.CRITERIA:
        for _ in range(200):
            k = rng.integers(2, 5)
            left = rng.integers(0, 50, k)
            right = rng.integers(0, 50, k)
            if left.sum() + right.sum() == 0:
                continue
            got = T.weighted_impurity_decrease(left, right, criterion)
            assert got >= -1e-12, f"{criterion}: negative decrease {got} on {left}/{right}"


# --------------------------------------------------------------- M1b --
# Contract to implement:
#   class_counts(y, n_classes) -> np.ndarray[int], shape (n_classes,)
#   best_split(X, y, criterion, min_samples_leaf=1)
#       -> (feature_idx, threshold, decrease), or (None, None, 0.0) if no
#          valid split exists. threshold is the midpoint of two consecutive
#          distinct values; the split sends X[:, f] <= threshold left.

@test("M1b", "class_counts(y, n_classes) -> counts per class, zeros included")
def t_class_counts():
    requires("class_counts")
    assert np.array_equal(T.class_counts(np.array([0, 0, 1, 2, 2, 2]), 4), [2, 1, 3, 0])
    assert np.array_equal(T.class_counts(np.array([], dtype=int), 3), [0, 0, 0])


@test("M1b", "best_split finds the separating threshold and puts it at the midpoint")
def t_best_split_obvious():
    requires("best_split")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    f, thr, dec = T.best_split(X, y, "gini")
    assert f == 0
    close(thr, 1.5)
    close(dec, 0.5)


@test("M1b", "best_split picks the informative feature and ignores noise")
def t_best_split_feature_choice():
    requires("best_split")
    rng = np.random.default_rng(5)
    y = np.array([0] * 50 + [1] * 50)
    X = np.column_stack([rng.normal(size=100), y + rng.normal(scale=0.01, size=100)])
    f, thr, dec = T.best_split(X, y, "gini")
    assert f == 1, f"expected the clean feature 1, got {f}"


@test("M1b", "no valid split: constant feature, single sample, already pure")
def t_best_split_no_split():
    requires("best_split")
    for X, y in [
        (np.array([[1.0], [1.0], [1.0]]), np.array([0, 1, 0])),
        (np.array([[1.0]]), np.array([0])),
        (np.array([[1.0], [2.0], [3.0]]), np.array([1, 1, 1])),
    ]:
        f, thr, dec = T.best_split(X, y, "gini")
        assert f is None and thr is None and dec == 0.0, (
            f"expected (None, None, 0.0) for X={X.ravel()}, y={y}, got {(f, thr, dec)}"
        )


@test("M1b", "duplicate feature values: the reported decrease is the real one")
def t_best_split_tied_values():
    requires("best_split")
    # every value appears twice, so only 1.5 and 2.5 are candidate thresholds.
    # By hand: parent gini = 1 - (2/6)^2 - (4/6)^2 = 4/9.
    #   thr 1.5 -> (2,0) | (0,4), both pure      -> decrease 4/9
    #   thr 2.5 -> (2,2) | (0,2)                 -> decrease 1/9
    X = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    y = np.array([0, 0, 1, 1, 1, 1])
    f, thr, dec = T.best_split(X, y, "gini")
    assert f == 0
    close(thr, 1.5)
    close(dec, 4 / 9)


@test("M1b", "3 classes need no special case: labels enter only as counts")
def t_best_split_multiclass():
    requires("best_split")
    X = np.arange(6.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1, 2, 2])
    f, thr, dec = T.best_split(X, y, "gini")
    assert f == 0
    # 1.5 and 3.5 both score 1/3 exactly -- a real tie, either is acceptable
    assert thr in (1.5, 3.5), f"expected a best threshold of 1.5 or 3.5, got {thr}"
    close(dec, 1 / 3)


@test("M1b", "best_split does not modify the arrays it is given")
def t_best_split_pure_function():
    requires("best_split")
    rng = np.random.default_rng(11)
    X = rng.normal(size=(40, 3))
    y = rng.integers(0, 2, size=40)
    X_ref, y_ref = X.copy(), y.copy()
    T.best_split(X, y, "gini")
    assert np.array_equal(X, X_ref), "X was reordered in place"
    assert np.array_equal(y, y_ref), "y was reordered in place"


@test("M1b", "min_samples_leaf is enforced during the scan, not after")
def t_best_split_min_samples_leaf():
    requires("best_split")
    X = np.arange(10.0).reshape(-1, 1)
    y = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 0])   # perfect split needs a leaf of 1
    f, thr, dec = T.best_split(X, y, "gini", min_samples_leaf=3)
    assert thr is None or 1.5 <= thr <= 7.5, f"threshold {thr} leaves a leaf under 3 samples"


@test("M2a", "the scan is O(n log n): 10x the rows must not cost ~100x the time")
def t_best_split_complexity():
    requires("best_split")
    import time
    rng = np.random.default_rng(7)

    def timed(n):
        X = rng.normal(size=(n, 1))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        start = time.perf_counter()
        T.best_split(X, y, "gini")
        return time.perf_counter() - start

    small = min(timed(2000) for _ in range(3))
    large = min(timed(20000) for _ in range(3))
    ratio = large / max(small, 1e-9)
    assert ratio < 30, f"10x rows cost {ratio:.0f}x time -- that scan is quadratic"


# --------------------------------------------------------------- M1c --
# Contract to implement:
#   class DecisionTree(criterion="gini", max_depth=None, min_samples_split=2,
#                      min_samples_leaf=1, min_impurity_decrease=0.0)
#       .fit(X, y) -> self      .predict(X) -> np.ndarray      .depth() -> int
#   A node is a split (feature_idx, threshold, left, right) or a leaf (value),
#   never both. No exception is raised for a pure node or a depth limit.

@test("M1c", "fit/predict round-trip: a deep tree reproduces its training labels")
def t_tree_fits_training_data():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(criterion="gini").fit(X, y)
    pred = tree.predict(X)
    assert pred.shape == y.shape
    close(T.accuracy(y, pred), 1.0)


@test("M1c", "max_depth is respected and depth 0 is a single leaf predicting the mode")
def t_tree_max_depth():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    stump = T.DecisionTree(max_depth=0).fit(X, y)
    assert stump.depth() == 0
    mode = np.bincount(y).argmax()
    assert set(np.unique(stump.predict(X))) == {mode}
    for d in (1, 3, 5):
        assert T.DecisionTree(max_depth=d).fit(X, y).depth() <= d


@test("M1c", "a pure node or an unsplittable node becomes a leaf, it does not raise")
def t_tree_pure_node():
    requires("DecisionTree")
    X = np.array([[1.0], [1.0], [1.0]])
    tree = T.DecisionTree().fit(X, np.array([1, 1, 1]))
    assert np.array_equal(tree.predict(X), [1, 1, 1])
    tree = T.DecisionTree().fit(X, np.array([0, 1, 1]))   # constant feature, impure
    assert np.array_equal(tree.predict(X), [1, 1, 1])


@test("M1c", "min_samples_leaf holds for every leaf of the fitted tree")
def t_tree_min_samples_leaf():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(min_samples_leaf=20).fit(X, y)
    if not hasattr(tree, "apply"):
        raise Pending("DecisionTree.apply(X) -> leaf id per row")
    _, sizes = np.unique(tree.apply(X), return_counts=True)
    assert sizes.min() >= 20, f"smallest leaf holds {sizes.min()} samples, min_samples_leaf=20"


# --------------------------------------------------------------- M1d --

@test("M1d", "GATE: identical predictions to sklearn DecisionTreeClassifier")
def t_gate_vs_sklearn():
    requires("DecisionTree")
    X_train, X_test, y_train, y_test = T.load_split(DATA_CLASS)
    for max_depth in (1, 3, 5, None):
        mine = T.DecisionTree(criterion="gini", max_depth=max_depth).fit(X_train, y_train)
        ref = DecisionTreeClassifier(
            criterion="gini", max_depth=max_depth, random_state=0
        ).fit(X_train, y_train)
        mine_pred, ref_pred = mine.predict(X_test), ref.predict(X_test)
        disagree = int(np.sum(mine_pred != ref_pred))
        assert disagree == 0, (
            f"max_depth={max_depth}: {disagree}/{len(y_test)} test predictions differ "
            f"from sklearn -- explain every one before moving on"
        )


# ------------------------------------------------- M1e: regression trees --

@test("M1e", "variance is a registered criterion and scores a regression split")
def t_variance_criterion():
    if "variance" not in getattr(T, "CRITERIA", {}):
        raise Pending("CRITERIA['variance']")
    requires("weighted_variance_decrease")
    left = np.array([1.0, 1.0, 1.0])
    right = np.array([5.0, 5.0, 5.0])
    close(T.weighted_variance_decrease(left, right), 4.0)
    close(T.weighted_variance_decrease(left, left), 0.0)


# ----------------------------------------------- M1b: split contracts --
# Added for the split-search / builder session. Same contract as above:
#   best_split(X, y, criterion, min_samples_leaf=1)
#       -> (feature_idx, threshold, decrease) | (None, None, 0.0)

@test("M1b", "the returned decrease is the decrease of the returned split")
def t_best_split_decrease_is_consistent():
    requires("best_split")
    rng = np.random.default_rng(23)
    for criterion in T.CRITERIA:
        for _ in range(20):
            X = rng.normal(size=(60, 3))
            y = rng.integers(0, 3, size=60)
            f, thr, dec = T.best_split(X, y, criterion)
            if f is None:
                continue
            mask = X[:, f] <= thr
            assert mask.any() and (~mask).any(), "the reported split has an empty child"
            left = np.bincount(y[mask], minlength=3)
            right = np.bincount(y[~mask], minlength=3)
            close(dec, T.weighted_impurity_decrease(left, right, criterion))


@test("M1b", "the threshold sits between two distinct values, never on a sample")
def t_best_split_threshold_is_a_midpoint():
    requires("best_split")
    rng = np.random.default_rng(29)
    X = rng.integers(0, 6, size=(80, 2)).astype(np.float64)   # heavy ties on purpose
    y = (X[:, 0] + rng.normal(scale=0.5, size=80) > 2).astype(int)
    f, thr, dec = T.best_split(X, y, "gini")
    assert f is not None
    values = np.unique(X[:, f])
    assert thr not in set(values), f"threshold {thr} is a sample value, not a midpoint"
    below = values[values < thr]
    above = values[values > thr]
    assert len(below) and len(above), f"threshold {thr} is outside the value range"
    close(thr, (below[-1] + above[0]) / 2)


@test("M1b", "min_samples_leaf larger than any admissible child gives the null triple")
def t_best_split_min_samples_leaf_blocks_all():
    requires("best_split")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    assert T.best_split(X, y, "gini", min_samples_leaf=3) == (None, None, 0.0)


# ------------------------------------- M1c: stopping rules and wiring --

@test("M1c", "fit returns self and leaves the hyperparameters exactly as given")
def t_tree_fit_contract():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(criterion="gini", max_depth=3, min_samples_split=10,
                          min_samples_leaf=4, min_impurity_decrease=0.001)
    assert tree.fit(X, y) is tree, "fit must return self (sklearn's contract)"
    assert tree.criterion == "gini" and tree.max_depth == 3
    assert tree.min_samples_split == 10 and tree.min_samples_leaf == 4
    close(tree.min_impurity_decrease, 0.001)


@test("M1c", "predict before fit raises; it does not return zeros")
def t_tree_predict_unfitted():
    requires("DecisionTree")
    X, _ = T.load_xy(DATA_CLASS)
    try:
        T.DecisionTree().predict(X[:5])
    except Exception:
        return
    raise AssertionError("predict on an unfitted tree returned something")


@test("M1c", "min_samples_split stops the root: 4 separable samples, threshold 5")
def t_tree_min_samples_split():
    requires("DecisionTree")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])
    assert T.DecisionTree(min_samples_split=5).fit(X, y).depth() == 0
    assert T.DecisionTree(min_samples_split=4).fit(X, y).depth() == 1


@test("M1c", "min_impurity_decrease at the root: 0.5 available, 0.6 stops, 0.4 splits")
def t_tree_min_impurity_decrease():
    requires("DecisionTree")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])                       # best decrease is exactly 0.5
    assert T.DecisionTree(min_impurity_decrease=0.6).fit(X, y).depth() == 0
    assert T.DecisionTree(min_impurity_decrease=0.4).fit(X, y).depth() == 1


@test("M1c", "a point landing exactly on a threshold goes left")
def t_tree_boundary_goes_left():
    requires("DecisionTree")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 0, 1, 1])                       # threshold must be 1.5
    tree = T.DecisionTree(max_depth=1).fit(X, y)
    assert tree.predict(np.array([[1.5]]))[0] == 0, (
        "X[:, f] <= threshold goes left, so 1.5 must get the left leaf's label"
    )
    assert tree.predict(np.array([[1.5 + 1e-12]]))[0] == 1


@test("M1c", "depth() is the longest root-to-leaf path, not a running counter")
def t_tree_depth_is_longest_path():
    requires("DecisionTree")
    X = np.arange(4.0).reshape(-1, 1)
    y = np.array([0, 1, 0, 1])           # sklearn: one shallow branch, one 3 deep
    assert T.DecisionTree().fit(X, y).depth() == 3


@test("M1c", "fit is repeatable: refitting the same data gives the same tree")
def t_tree_refit_idempotent():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(max_depth=4)
    first_pred = tree.fit(X, y).predict(X)
    first_depth = tree.depth()
    small = np.arange(4.0).reshape(-1, 1)
    tree.fit(small, np.array([0, 0, 1, 1]))          # fit on something else
    second_pred = tree.fit(X, y).predict(X)          # then back again
    assert np.array_equal(first_pred, second_pred), "refitting changed the predictions"
    assert tree.depth() == first_depth, (
        f"depth() went {first_depth} -> {tree.depth()} across refits: it is accumulating "
        f"state instead of being derived from the tree"
    )


@test("M1c", "apply is stable and every row in a leaf gets that leaf's prediction")
def t_tree_apply_matches_predict():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(max_depth=3).fit(X, y)
    if not hasattr(tree, "apply"):
        raise Pending("DecisionTree.apply(X) -> leaf id per row")
    ids, pred = tree.apply(X), tree.predict(X)
    assert np.array_equal(ids, tree.apply(X)), "apply is not deterministic"
    assert ids.shape == (len(X),)
    for leaf in np.unique(ids):
        assert len(np.unique(pred[ids == leaf])) == 1, (
            f"leaf {leaf} returns more than one prediction"
        )
    assert len(np.unique(ids)) <= 2 ** 3, "more leaves than max_depth=3 allows"


@test("M1c", "duplicate rows with conflicting labels: a leaf holding the mode, no crash")
def t_tree_duplicate_rows():
    requires("DecisionTree")
    X = np.array([[1.0, 2.0]] * 5)
    y = np.array([0, 1, 1, 1, 0])
    tree = T.DecisionTree().fit(X, y)
    assert tree.depth() == 0
    assert np.array_equal(tree.predict(X), np.ones(5, dtype=y.dtype))


@test("M1c", "stopping rules compose: depth, node size and leaf size hold together")
def t_tree_stopping_rules_compose():
    requires("DecisionTree")
    X, y = T.load_xy(DATA_CLASS)
    tree = T.DecisionTree(max_depth=3, min_samples_split=40, min_samples_leaf=15).fit(X, y)
    if not hasattr(tree, "apply"):
        raise Pending("DecisionTree.apply(X) -> leaf id per row")
    assert tree.depth() <= 3
    _, sizes = np.unique(tree.apply(X), return_counts=True)
    assert sizes.min() >= 15, f"smallest leaf holds {sizes.min()}, min_samples_leaf=15"
    assert T.accuracy(y, tree.predict(X)) > 0.8, "a depth-3 tree should still be useful here"


def main() -> int:
    wanted = sys.argv[1:]
    passed = failed = pending = 0
    current_phase = None
    for phase, contract, fn in sorted(_TESTS, key=lambda t: t[0]):
        if wanted and not any(phase.startswith(w) for w in wanted):
            continue
        if phase != current_phase:
            print(f"\n--- {phase} " + "-" * (66 - len(phase)))
            current_phase = phase
        try:
            fn()
        except Pending as exc:
            print(f"  PENDING  {contract}\n           not implemented yet: {exc}")
            pending += 1
        except Exception:
            print(f"  FAIL     {contract}")
            for line in traceback.format_exc().strip().splitlines()[-3:]:
                print(f"           {line.strip()}")
            failed += 1
        else:
            print(f"  ok       {contract}")
            passed += 1
    print(f"\n{passed} passed, {failed} failed, {pending} pending")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
