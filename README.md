# Generic Algorithms

Classic algorithms in machine learning and statistics, implemented twice: once
by hand from the mathematics, once with the standard library. The two versions
are then compared on correctness and on speed.

The point is not to reinvent scikit-learn. It is to be able to say exactly what
a model does, why each design choice is the one it is, and what it costs — and
to prove that claim by matching the reference implementation numerically.

## Standard of correctness

Every hand-written component must reproduce its scikit-learn equivalent to
within numerical noise, given identical hyperparameters and a fixed seed. That
is the pass criterion — not "the output looks plausible".

For a decision tree this means agreeing on the *predictions themselves*, not
just on aggregate accuracy: two trees can score identically while splitting on
different features. Where predictions do diverge, the divergence has to be
explained (tie-breaking in the split search, threshold placement, floating-point
ordering) rather than absorbed into a tolerance.

Performance is measured the same way: fit time and predict time against the
reference, on the same data, rather than asserted.

## Design conventions

- **pandas at the boundary only.** `read_csv` on the way in, then `.to_numpy()`
  once. The split search is a numerical inner loop over homogeneous float64; a
  DataFrame pays for an index and per-element dtype dispatch on every access,
  while an ndarray is one contiguous block whose slices are views, not copies.
- `X` is `(n_samples, n_features)` float64; `y` is `(n_samples,)` — integer
  labels for classification, float targets for regression.
- Randomness is seeded through function arguments, never a global
  `np.random.seed()`, so that any result can be reproduced in isolation.
- One module per concern, tests co-located with the source.

## Layout

```
Machine_Learning/     decision trees, ensembles
Statistics/           statistical methods
data/                 datasets used throughout
```

## Status

| Component | State |
| --- | --- |
| Metrics — `accuracy`, `mse`, `r2` | implemented by hand |
| Impurity criteria — Gini, entropy, variance | in progress |
| Decision tree — split search, recursive builder | in progress |
| Cost-complexity pruning | planned |
| Bagging + out-of-bag error | planned |
| Random forest — feature subsampling, importances | planned |
| Optimisation — pre-sorting, histogram binning, `joblib` | planned |

Actively being built. Benchmark tables go in as each component clears the
matched-against-sklearn gate above, so anything listed as implemented has
already passed it.

## Data

Two synthetic datasets, 500 rows and 2 features each:

- `classification_data.csv` — binary target, near-balanced (249 / 251).
- `regression_data.csv` — continuous target, roughly ±65.

Small on purpose. At this size a single train/test split is noisy enough that
results have to be checked across seeds, which is the habit worth keeping.

## Running

Requires Python 3.11+ with `numpy`, `pandas` and `scikit-learn`
(developed against numpy 2.3, pandas 2.3, scikit-learn 1.8).

Run from the repository root, since data paths are relative to it:

```bash
python -m Machine_Learning.Trees
```
