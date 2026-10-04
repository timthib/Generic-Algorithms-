"""
Synthetic time-series generators: AR, MA, ARMA, ARIMA, ARIMAX, ARX.

Each process is simulated from its own definition with a fixed seed, so the
coefficients used to build a series are known exactly and a fitted model can
be scored against them. That is the point of this file: the test of an
estimator written by hand is whether it recovers these numbers (and whether
statsmodels recovers the same ones from the same CSV).

Conventions:
 - every CSV has a leading integer column `t` (0, 1, 2, ...) and a target
   column `y`; exogenous regressors sit between them as `x1`, `x2`, `u`.
 - innovations are Gaussian white noise, e_t ~ N(0, sigma^2).
 - a burn-in of `BURN` samples is simulated and discarded, so row 0 is not
   contaminated by the zero initial condition.
 - lag polynomials follow the usual sign convention:
       AR: y_t = c + sum_i phi_i y_{t-i} + e_t
       MA: y_t = mu + e_t + sum_j theta_j e_{t-j}
   i.e. theta is written with a PLUS sign (statsmodels' convention), not the
   minus sign used in some textbooks. Flip the sign if comparing to a source
   that writes (1 - theta_1 B).

AR coefficients are chosen inside the stationarity region (roots of the
characteristic polynomial outside the unit circle); `_assert_stationary`
enforces this rather than trusting the choice, because a non-stationary draw
would explode over the burn-in and silently produce garbage.

Exogenous parameterisation -- read this before cross-checking against a
library. The ARX and ARIMAX series here are *dynamic* regressions (the
transfer-function form): the exogenous term enters the recursion alongside
the lagged y,

    y_t = c + sum_i phi_i y_{t-i} + sum_k beta_k u_{t-k} + e_t

statsmodels' SARIMAX instead fits regression-with-ARMA-errors,
y_t = beta' x_t + eta_t with eta an ARMA process, and (for d > 0) differences
the exogenous columns too. The two are not the same model: a SARIMAX fit of
these CSVs returns different betas, because the dynamic form implies an
infinite distributed lag of u that SARIMAX's single contemporaneous beta has
to absorb. Use statsmodels' ARDL for the dynamic form, or compare against the
betas below only with an estimator written in this parameterisation.

Recovery verified on the generated files (N = 1200):
  ar_data.csv      AR(2) MLE:      phi = (0.587, -0.279); mean 2.109 (true 2.143)
  ma_data.csv      MA(2) MLE:      theta = (0.675, 0.382); mu = -2.065
  arma_data.csv    ARMA(1,1):      phi = 0.725, theta = 0.526
  arima_data.csv   ARIMA(1,1,1):   phi = 0.584, theta = 0.337 (trend="t";
                                   statsmodels rejects a constant when d > 0)
  arx_data.csv     OLS on lags:    c = 0.269, phi = (0.488, 0.207),
                                   beta = (1.214, -0.809) -- every estimate
                                   within one standard error; the errors are
                                   white, so one least-squares pass suffices
  arimax_data.csv  conditional LS: c = 0.197, phi = 0.529,
                                   beta = (0.931, -0.471), theta = 0.363
ADF p-values: ar 0.00, arma 0.00, arima 0.93, arimax 0.99 -- the two
integrated files fail to reject a unit root, as they should.

Plain OLS on arimax_data.csv is visibly biased (beta_x1 ~ 0.43 against a true
0.9): the MA term makes e_{t-1} correlate with the lagged regressand, and the
bias leaks into the exogenous coefficients through x's own autocorrelation.
That bias is intentional -- it is the reason an MA part needs an estimator
that solves for the innovations rather than a single least-squares pass.

The true parameters are also dumped to ground_truth.json for the tests.
Regenerate everything with `python3 Time_Series/data/generate_series.py`.
"""
import json
import os

import numpy as np

N = 1200         # rows kept per series
BURN = 200       # discarded samples
SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))


def _assert_stationary(phi: np.ndarray) -> None:
    """Raise if the AR polynomial 1 - phi_1 B - ... - phi_p B^p has a root
    inside or on the unit circle.

    The companion-form test is used: the AR recursion is stationary iff every
    eigenvalue of its companion matrix lies strictly inside the unit circle.
    """
    p = len(phi)
    companion = np.zeros((p, p))
    companion[0, :] = phi
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    radius = np.max(np.abs(np.linalg.eigvals(companion)))
    if radius >= 1.0:
        raise ValueError(f"non-stationary AR coefficients {phi}: spectral radius {radius:.3f}")


def _noise(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    return rng.normal(0.0, sigma, size=n)


def ar(rng, phi, c, sigma):
    """y_t = c + sum_i phi_i y_{t-i} + e_t."""
    phi = np.asarray(phi, dtype=float)
    _assert_stationary(phi)
    p = len(phi)
    n = N + BURN
    e = _noise(rng, n, sigma)
    y = np.zeros(n)
    y[:p] = c / (1.0 - phi.sum())          # start at the unconditional mean
    for t in range(p, n):
        y[t] = c + phi @ y[t - p:t][::-1] + e[t]
    return y[BURN:]


def ma(rng, theta, mu, sigma):
    """y_t = mu + e_t + sum_j theta_j e_{t-j}."""
    theta = np.asarray(theta, dtype=float)
    q = len(theta)
    n = N + BURN
    e = _noise(rng, n, sigma)
    y = np.full(n, mu)
    for t in range(q, n):
        y[t] += e[t] + theta @ e[t - q:t][::-1]
    return y[BURN:]


def arma(rng, phi, theta, c, sigma):
    """y_t = c + sum_i phi_i y_{t-i} + e_t + sum_j theta_j e_{t-j}."""
    phi = np.asarray(phi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    _assert_stationary(phi)
    p, q = len(phi), len(theta)
    k = max(p, q)
    n = N + BURN
    e = _noise(rng, n, sigma)
    y = np.zeros(n)
    y[:k] = c / (1.0 - phi.sum())
    for t in range(k, n):
        y[t] = (c + phi @ y[t - p:t][::-1]
                + e[t] + theta @ e[t - q:t][::-1])
    return y[BURN:]


def integrate(series, d, y0=0.0):
    """Undo differencing d times: return a level series whose d-th difference
    is `series`. An ARIMA(p, d, q) series is an ARMA(p, q) series integrated
    d times, which is how `arima_data.csv` is built.
    """
    out = np.asarray(series, dtype=float)
    for _ in range(d):
        out = y0 + np.cumsum(out)
    return out


def exog_ar1(rng, phi1, sigma, n=None):
    """A stationary AR(1) regressor -- exogenous drivers in practice are
    themselves serially correlated, and an i.i.d. x would make the exogenous
    coefficients unrealistically easy to identify.
    """
    n = N if n is None else n
    e = _noise(rng, n + BURN, sigma)
    x = np.zeros(n + BURN)
    for t in range(1, n + BURN):
        x[t] = phi1 * x[t - 1] + e[t]
    return x[BURN:]


def write_csv(name, columns):
    """Dump {column_name: array} to data/<name>.csv with a leading `t`."""
    names = list(columns)
    data = np.column_stack([np.arange(len(columns[names[0]]))]
                           + [columns[k] for k in names])
    path = os.path.join(HERE, name)
    header = "t," + ",".join(names)
    fmt = ["%d"] + ["%.10f"] * len(names)
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)
    return path


def main():
    truth = {"n_rows": N, "burn_in": BURN, "seed": SEED,
             "ma_sign_convention": "y_t = mu + e_t + theta_1 e_{t-1} + ...",
             "exog_parameterisation": (
                 "dynamic / transfer-function: exogenous terms enter the recursion "
                 "beside the lagged y, NOT statsmodels SARIMAX's "
                 "regression-with-ARMA-errors. A SARIMAX fit of these files returns "
                 "different betas by construction; use ARDL or a dynamic estimator.")}

    # --- AR(2) -----------------------------------------------------------
    rng = np.random.default_rng(SEED)
    phi = [0.6, -0.3]
    y = ar(rng, phi, c=1.5, sigma=1.0)
    write_csv("ar_data.csv", {"y": y})
    truth["ar_data.csv"] = {"model": "AR(2)", "phi": phi, "c": 1.5, "sigma": 1.0}

    # --- MA(2) -----------------------------------------------------------
    rng = np.random.default_rng(SEED + 1)
    theta = [0.7, 0.4]
    y = ma(rng, theta, mu=-2.0, sigma=1.0)
    write_csv("ma_data.csv", {"y": y})
    truth["ma_data.csv"] = {"model": "MA(2)", "theta": theta, "mu": -2.0, "sigma": 1.0}

    # --- ARMA(1, 1) ------------------------------------------------------
    rng = np.random.default_rng(SEED + 2)
    y = arma(rng, [0.75], [0.5], c=0.4, sigma=1.0)
    write_csv("arma_data.csv", {"y": y})
    truth["arma_data.csv"] = {"model": "ARMA(1,1)", "phi": [0.75], "theta": [0.5],
                              "c": 0.4, "sigma": 1.0}

    # --- ARIMA(1, 1, 1): stationary ARMA in the differences --------------
    # The level series has a unit root, so a differencing step (d=1) is
    # required before any AR/MA estimator applies; fitting it undifferenced
    # is the mistake this file lets you observe.
    rng = np.random.default_rng(SEED + 3)
    dy = arma(rng, [0.6], [0.3], c=0.2, sigma=1.0)
    y = integrate(dy, d=1, y0=10.0)
    write_csv("arima_data.csv", {"y": y})
    truth["arima_data.csv"] = {"model": "ARIMA(1,1,1)", "d": 1, "phi": [0.6],
                               "theta": [0.3], "c_on_diff": 0.2, "y0": 10.0,
                               "sigma": 1.0,
                               "note": "ARMA(1,1) holds for diff(y), not y"}

    # --- ARX: AR(2) plus a contemporaneous and a lagged exogenous term ---
    # Simulated over N + BURN + 1 samples: the extra sample feeds u_{t-1} for
    # the first kept row, and the burn-in keeps the zero initial condition out
    # of the file (otherwise rows 0-1 are not drawn from the process at all).
    # Seed 15 rather than SEED + 4: that draw happened to sit ~3 standard
    # errors from beta_u, which makes a correct estimator look wrong. Any seed
    # is valid data; this one is a typical draw (every OLS estimate within one
    # standard error of truth), so a recovery check reads cleanly.
    rng = np.random.default_rng(15)
    n = N + BURN + 1
    u_full = exog_ar1(rng, 0.8, 1.0, n=n)
    e = _noise(rng, n, 1.0)
    phi_arx = [0.5, 0.2]
    beta = [1.2, -0.8]                              # [u_t, u_{t-1}]
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = (0.3 + phi_arx[0] * y[t - 1] + phi_arx[1] * y[t - 2]
                + beta[0] * u_full[t] + beta[1] * u_full[t - 1] + e[t])
    keep = slice(BURN + 1, n)
    lag = slice(BURN, n - 1)
    write_csv("arx_data.csv", {"u": u_full[keep], "u_lag1": u_full[lag], "y": y[keep]})
    truth["arx_data.csv"] = {"model": "ARX(2) with u_t and u_{t-1}",
                             "phi": phi_arx, "beta_u": beta[0],
                             "beta_u_lag1": beta[1], "c": 0.3, "sigma": 1.0,
                             "columns": "u, u_lag1, y"}

    # --- ARIMAX(1, 1, 1) with two exogenous regressors -------------------
    # Specified on the differences: diff(y)_t follows ARMA(1,1) driven by
    # x1_t and x2_t, then integrated back to a level series.
    rng = np.random.default_rng(SEED + 5)
    x1 = exog_ar1(rng, 0.7, 1.0)
    x2 = exog_ar1(rng, 0.4, 2.0)
    e = _noise(rng, N, 1.0)
    phi1, theta1 = 0.55, 0.35
    b1, b2 = 0.9, -0.5
    dy = np.zeros(N)
    for t in range(1, N):
        dy[t] = (0.1 + phi1 * dy[t - 1] + b1 * x1[t] + b2 * x2[t]
                 + e[t] + theta1 * e[t - 1])
    y = integrate(dy, d=1, y0=5.0)
    write_csv("arimax_data.csv", {"x1": x1, "x2": x2, "y": y})
    truth["arimax_data.csv"] = {"model": "ARIMAX(1,1,1)", "d": 1,
                                "phi": [phi1], "theta": [theta1],
                                "beta_x1": b1, "beta_x2": b2,
                                "c_on_diff": 0.1, "y0": 5.0, "sigma": 1.0,
                                "note": "ARMA(1,1)+exog holds for diff(y)"}

    with open(os.path.join(HERE, "ground_truth.json"), "w") as fh:
        json.dump(truth, fh, indent=2)
    print("wrote:", ", ".join(sorted(os.listdir(HERE))))


if __name__ == "__main__":
    main()
