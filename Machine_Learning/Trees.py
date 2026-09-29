"""
Decision trees implemented from scratch, with scikit-learn as the reference.

Covers the classification tree (Gini / entropy criteria) and the regression
tree (variance criterion). Cost-complexity pruning and the bagged ensemble
(random forest) build on this module.

Conventions:
X : np.ndarray, shape (n_samples, n_features), dtype float64
y : np.ndarray, shape (n_samples,) -- int labels for classification,
    float targets for regression

pandas is used only at the I/O boundary (read_csv), then converted once via
.to_numpy(). The split search is a numerical inner loop over homogeneous
floats: a DataFrame pays for an index and per-element dtype dispatch on every
access, whereas an ndarray is one contiguous block whose slices are views.

Correctness bar: each hand-written component must match scikit-learn's
equivalent to within numerical noise given identical hyperparameters.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data():
    data_class = pd.read_csv('data/classification_data.csv').to_numpy()
    data_reg = pd.read_csv('data/regression_data.csv').to_numpy()
    train_class, test_class = train_test_split(data_class, test_size=0.2, random_state=42)
    train_reg, test_reg = train_test_split(data_reg, test_size=0.2, random_state=42)
    return train_class, test_class,train_reg, test_reg


def accuracy(pred,y):
    if pred.shape != y.shape :
        raise ValueError("prediction and ground truth dataset must have the same dimensions")
    return np.mean(y == pred) 

def mse(pred,y):
    if pred.shape != y.shape :
        raise ValueError("prediction and ground truth dataset must have the same dimensions")
    return np.mean((pred-y)**2)

def r2(pred,y):
    if pred.shape != y.shape :
        raise ValueError("prediction and ground truth dataset must have the same dimensions")
    avg = np.mean(y)*np.ones(y.size)
    if mse(y,avg) == 0 : 
        return 0
    return 1 - mse(pred,y)/mse(y,avg)


"""
Best split metrics. 
Many options. To ensure each split never increase impurity, we need the evaluation functions to
be concave wrt to p, the probability of class 1. 
Indeed, p = p_left * n_left/n + p_right * n_right/n. And the function being concave means that 
n_left/n * Impurity(p_left) + n_right/n * Impurity(p_right) <= Impurity(p)
So the weighted haverage impurity decrease with splits. 
Gini and entropy are stricly concave, ensure a strictly positive decrease in impurity at each split.
"""

def gini(p):
    return 1 - p**2 - (1-p)**2

def entropy(p):
    if p == 0 or p == 1: 
        return 0 
    return -p* np.log2(p) - (1-p)*np.log2(1-p)

def misclassification_rate(p):
    return min(p,1-p)
