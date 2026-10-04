"""
ARX: AutoRegressive with eXogenous input.

    y_t = c + sum_{i=1..p} phi_i y_{t-i} + sum_{k=0..s} beta_k u_{t-k} + e_t

Linear in the parameters, so stacking the regressors into a design matrix X
turns the fit into min_b ||y - X b||^2, solved in one pass by the normal
equations X'X b = X'y. Nothing iterates. (Iteration only becomes unavoidable
once an MA term makes the residuals depend on unobserved past innovations:
see data/arimax_data.csv.)

OLS is consistent here because e_t is white and every regressor
(y_{t-1}, y_{t-2}, u_t, u_{t-1}) is known at time t, hence uncorrelated with
e_t. In arimax_data.csv an MA term correlates e_{t-1} with the regressors and
plain OLS returns beta_x1 ~ 0.43 against a true 0.9.

Data: data/arx_data.csv, true parameters in data/ground_truth.json
    c = 0.3, phi = (0.5, 0.2), beta = (1.2 on u_t, -0.8 on u_{t-1}), sigma = 1
    1200 rows, columns: t, u, u_lag1, y
    u is itself an AR(1) with phi = 0.8, so beta is not trivially identified.
    u_lag1 is u shifted by one row: build your own lags from u and use it only
    to check the shift direction.

Library reference is statsmodels ARDL, not SARIMAX. SARIMAX fits
regression-with-ARMA-errors, a different model, and returns different betas on
this file by construction.

# ======================================================================
# PIPELINE
# ======================================================================
#
# 0. Stationarity, before anything else.
#    ADF + KPSS on y and on every column of u. Opposite nulls, which is the
#    point of running both: ADF nulls a unit root (small p => stationary),
#    KPSS nulls stationarity (small p => non-stationary). Disagreement means
#    near unit root and too little power to tell.
#
#    This step is not optional and it does not come last. Everything below:
#    the PACF band, the t stats, AIC/BIC, Ljung-Box, has a null distribution
#    derived under stationarity. Worse, the correlogram cannot detect its own
#    invalidity: the sample PACF of a pure random walk is [0.997, 0.02, ...],
#    a textbook AR(1) cutoff. And two independent I(1) series regressed on
#    each other give |t| ~ 10 and R^2 ~ 0.7 (Granger-Newbold). Non stationary
#    => difference, or fit an ECM if y and u turn out cointegrated.
#
# 1. Build the design.
#    y_t against y_{t-1}..y_{t-p} plus, for each exogenous column l,
#    u_{l,t-k} .. u_{l,t-k-s}. Per feature lags k_l, s_l are a real modelling
#    choice (a transfer function with its own delay per input);
#    statsmodels ARDL takes 'order' as a dict for exactly this.
#    The first max(p, s) rows have no complete history and get dropped.
#
# 2. AR order p, from the PACF.
#    phi_kk is the coefficient on the k-th lag of an AR(k) fitted by OLS,
#    One regression per lag, keep the last coefficient.
#
#    Plot against LAG, in lag order.
#
# 3. Exogenous lags, from the prewhitened CCF.
#    Fit an AR to u, apply that same filter to u AND y (polynomials in L
#    commute, so v(L) survives untouched), and the filtered input is white:
#        gamma_beta_alpha(k) = v_k sigma_alpha^2
#    i.e. the CCF becomes the impulse response, up to scale. It also fixes the
#    band, since Bartlett's variance sum collapses to 1/n only for a white
#    input.
#
#    Read the delay off the first spike, not the order off the last: for ARX
#    v(L) = B(L)/Phi(L), so the AR denominator leaves a geometric tail past
#    the true s. Delay from the CCF, order from BIC.
#
# 4. Residual check.
#    ACF of the residuals inside +-1.96/sqrt(N), plus Ljung-Box with
#    df = nlags - n_params. If they are white noise, the specification holds and the
#    standard errors from step 6 mean something; until then they do not.
#
# 5. Ties between orders -> BIC.
#    log(n) > 2 for n > 7, so BIC penalises a parameter more heavily than AIC
#    and never picks the larger of two models. BIC is consistent for order
#    selection when the true order is in the grid; AIC is efficient but
#    overfits by O(1) lags asymptotically. Consistency is what identification
#    wants.
#
#    Compare every candidate on the same. n_eff = n - max(p, s), so a
#    bigger p silently drops more rows: (p=1,s=1) is scored on 1199 rows of
#    arx_data.csv, (p=4,s=1) on 1196. Both halves of the criterion are sums
#    over observations, so that compares a total of 1199 terms against a total
#    of 1196: the gap carries three rows' worth of fit, which says nothing
#    about which model is the best.
#
#    The artifact is (n_i - n_j) * log(sigma2), so its sign follows the scale
#    of the data: sigma2 > 1 penalises whichever model kept more rows and
#    hands the win to the largest p, sigma2 < 1 does the reverse. 
#    Hence the `warmup` argument of make_lag_matrix: select_order pins every
#    fit to warmup=max(p_max, s_max).
#
# 6. Fit and forecast.
#    OLS once on the chosen orders. Forecasts start at t = max(p, max exoglag).
#    One step is the regression equation evaluated at the end of the
#    sample (the conditional mean, so e contributes 0); multi-step feeds each
#    forecast back in as the next step's lagged y and needs the exogenous path
#    supplied. An ARX model cannot forecast further than u can be stated.
#    A pure delay helps: for h <= k every u needed is already observed.
"""
import numpy as np
import pandas as pd
from scipy import stats


def load_series(path: str, exog_cols: list[str], target_col: str = "y"
                ) -> tuple[np.ndarray, np.ndarray]:
    """Read the CSV and return (y, U) in time order.

    y : (n,)    target
    U : (n, m)  exogenous columns, in the order given by exog_cols
    """
    data = pd.read_csv(path, dtype=np.float64).sort_values("t", ascending=True)
    y = data[target_col].to_numpy(dtype=np.float64)
    # reshape keeps U two-dimensional when a single exogenous column is asked for
    U = data[exog_cols].to_numpy(dtype=np.float64).reshape(y.size, len(exog_cols))
    return y, U


def make_lag_matrix(y: np.ndarray, U: np.ndarray, p: int, s: int,
                    include_const: bool = True, warmup: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the ARX regression problem.

    p : number of lags of y, y_{t-1} .. y_{t-p}
    s : highest lag of u, u_t .. u_{t-s}. s=0 is contemporaneous only. Note the
        asymmetry with p: u_t is a regressor, y_t obviously is not.
    warmup : rows to drop from the front. Defaults to max(p, s), the smallest
        value for which every regressor exists. Pass something larger to pin
        the estimation sample across a grid of (p, s) -- AIC and BIC are only
        comparable at identical n, and select_order relies on this.

    Returns X (n_eff, k), the aligned y, and the column names. Column order is
    [const?, y_{t-1}..y_{t-p}, u_t..u_{t-s}], lag-major: all m exogenous
    columns for one lag before moving to the next. forecast_one_step rebuilds
    a row in this order, so the two must stay in step.
    """
    n = y.size
    m = U.shape[1]
    start = max(p, s) if warmup is None else warmup
    if start < max(p, s):
        raise ValueError(f"warmup={start} too small for p={p}, s={s}")
    n_eff = n - start
    if n_eff <= 0:
        raise ValueError(f"no usable rows: n={n}, warmup={start}")

    k = int(include_const) + p + (s + 1) * m
    X = np.empty((n_eff, k), dtype=np.float64)
    names: list[str] = []

    col = 0
    if include_const:
        X[:, 0] = 1.0
        names.append("const")
        col = 1
    for j in range(1, p + 1):
        X[:, col] = y[start - j:n - j]
        names.append(f"y_t-{j}")
        col += 1
    for j in range(s + 1):
        for l in range(m):
            X[:, col] = U[start - j:n - j, l]
            names.append(f"u{l}_t" if j == 0 else f"u{l}_t-{j}")
            col += 1

    return X, y[start:], names


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least-squares coefficients, shape (k,).

    lstsq factorises X directly (SVD) instead of forming X'X, which would
    square the condition number and cost twice the digits. rcond=None selects
    the current singular-value cutoff and silences the legacy warning.
    """
    return np.linalg.lstsq(X, y, rcond=None)[0]


def ols_stats(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> dict:
    """Standard errors, t-stats, p-values and R^2 for a fitted model.

    Only valid once the residuals pass ljung_box; until then the numbers are
    provisional.
    """
    n, k = X.shape
    e = y - X @ coef
    sse = float(e @ e)
    # n-k, not n: k degrees of freedom were already spent making e small, so
    # dividing by n underestimates the innovation variance.
    sigma2 = sse / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = coef / se
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "resid": e,
        "sigma2": sigma2,
        "cov": cov,
        "se": se,
        "t": t,
        "p_value": 2.0 * stats.t.sf(np.abs(t), df=n - k),
        "r2": 1.0 - sse / sst,
        "n": n,
        "k": k,
    }


def t_test_against(coef: np.ndarray, se: np.ndarray, j: int, value: float,
                   df: int) -> tuple[float, float]:
    """Test H0: b_j == value. Returns (t, two-sided p).

    Same standard error, different null: value=0 asks whether the regressor is
    needed at all, value=-0.8 asks whether the estimate agrees with a known
    truth. Easy to conflate.
    """
    t = (coef[j] - value) / se[j]
    return float(t), float(2.0 * stats.t.sf(abs(t), df=df))


def residuals(X: np.ndarray, y: np.ndarray, coef: np.ndarray) -> np.ndarray:
    """e_hat = y - X @ coef, shape (n_eff,)."""
    return y - X @ coef


def acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelations at lags 1..nlags, shape (nlags,).

    Divisor is the single n rather than n-j per lag. That is what keeps the
    autocovariance sequence positive semi-definite, and it matches
    statsmodels.acf().
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    z = x - x.mean()
    gamma0 = float(z @ z) / n
    out = np.empty(nlags, dtype=np.float64)
    for j in range(1, nlags + 1):
        out[j - 1] = float(z[j:] @ z[:n - j]) / n
    return out / gamma0


def pacf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample partial autocorrelations at lags 1..nlags, shape (nlags,).

    One AR(k) regression per lag, keeping only the last coefficient. See step 2
    of the pipeline for why this is left unpenalised.
    """
    x = np.asarray(x, dtype=np.float64)
    no_exog = np.zeros((x.size, 0), dtype=np.float64)  # empty U: pure AR fit
    out = np.empty(nlags, dtype=np.float64)
    for lag in range(1, nlags + 1):
        X, y_eff, _ = make_lag_matrix(x, no_exog, p=lag, s=0, include_const=True)
        out[lag - 1] = fit_ols(X, y_eff)[-1]
    return out


def significance_band(n: int, alpha: float = 0.05) -> float:
    """Half-width of the white-noise band for a correlogram: z / sqrt(n).

    n is the effective sample size. On the residuals of a fitted model this
    band is slightly too wide at the first few lags, since estimation has
    already pushed those autocorrelations toward zero; ljung_box corrects for
    that properly through its degrees of freedom.
    """
    return float(stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n))


def ljung_box(e: np.ndarray, nlags: int, n_params: int = 0
              ) -> tuple[float, float]:
    """Portmanteau test for residual autocorrelation: (statistic, p_value).

        Q = n (n + 2) sum_{j=1..nlags} r_j^2 / (n - j)

    H0 is joint whiteness up to nlags. Q is a weighted sum of squares, so only
    the upper tail rejects: a small p-value is bad news for the model.

    n_params is the number of fitted coefficients. Each one removes a free
    direction from the residual autocorrelations, so leaving it at 0 on fitted
    residuals makes the test too conservative.
    """
    e = np.asarray(e, dtype=np.float64)
    n = e.size
    r = acf(e, nlags)
    j = np.arange(1, nlags + 1)
    q = n * (n + 2) * float(np.sum(r ** 2 / (n - j)))
    df = nlags - n_params
    if df <= 0:
        raise ValueError(f"nlags={nlags} must exceed n_params={n_params}")
    return q, float(stats.chi2.sf(q, df))


def aic(n: int, k: int, sigma2: float) -> float:
    """Gaussian AIC = n log(sigma2) + 2k, additive constants dropped."""
    return n * np.log(sigma2) + 2.0 * k


def bic(n: int, k: int, sigma2: float) -> float:
    """Gaussian BIC = n log(sigma2) + k log(n), additive constants dropped."""
    return n * np.log(sigma2) + k * np.log(n)


def select_order(y: np.ndarray, U: np.ndarray, p_max: int, s_max: int,
                 criterion: str = "aic") -> tuple[int, int, dict]:
    """Grid-search (p, s) over 1..p_max x 0..s_max.

    Returns (p_best, s_best, table) with table mapping (p, s) -> criterion.
    Every candidate is fitted on the same rows, pinned by warmup=max(p_max,
    s_max); sigma2 uses the ML divisor n because that is the one the criteria
    are derived with.
    """
    score = {"aic": aic, "bic": bic}[criterion]
    warmup = max(p_max, s_max)
    table: dict[tuple[int, int], float] = {}
    for p in range(1, p_max + 1):
        for s in range(0, s_max + 1):
            X, y_eff, _ = make_lag_matrix(y, U, p, s, warmup=warmup)
            coef = fit_ols(X, y_eff)
            n, k = X.shape
            e = y_eff - X @ coef
            table[(p, s)] = score(n, k, float(e @ e) / n)
    p_best, s_best = min(table, key=table.get)
    return p_best, s_best, table


def _design_row(coef_len: int, y_lags: np.ndarray, u_at_date: np.ndarray,
                u_lags: np.ndarray, p: int, s: int, m: int,
                include_const: bool) -> np.ndarray:
    """One design row in make_lag_matrix's column order.

    y_lags    : (p,)    y_{t-1} .. y_{t-p}, most recent first
    u_at_date : (m,)    u_t
    u_lags    : (s, m)  u_{t-1} .. u_{t-s}, most recent first
    """
    row = np.empty(coef_len, dtype=np.float64)
    col = 0
    if include_const:
        row[0] = 1.0
        col = 1
    row[col:col + p] = y_lags
    col += p
    row[col:col + m] = u_at_date
    col += m
    for j in range(s):
        row[col:col + m] = u_lags[j]
        col += m
    return row


def forecast_one_step(coef: np.ndarray, y_hist: np.ndarray,
                      u_hist: np.ndarray, u_next: np.ndarray,
                      p: int, s: int, include_const: bool = True) -> float:
    """Predict y_{T+1} from history up to T plus the known u_{T+1}.

    y_hist : (>=p,)    recent y, oldest first
    u_hist : (>=s, m)  recent u, oldest first
    u_next : (m,)      exogenous value AT the forecast date, not before it
    """
    y_hist = np.asarray(y_hist, dtype=np.float64)
    u_hist = np.atleast_2d(np.asarray(u_hist, dtype=np.float64))
    u_next = np.atleast_1d(np.asarray(u_next, dtype=np.float64))
    m = u_next.size
    if y_hist.size < p:
        raise ValueError(f"need >= {p} past y, got {y_hist.size}")
    if s > 0 and u_hist.shape[0] < s:
        raise ValueError(f"need >= {s} past u, got {u_hist.shape[0]}")

    y_lags = y_hist[::-1][:p]                      # reverse: y_{t-1} first
    u_lags = u_hist[::-1][:s] if s > 0 else np.empty((0, m))
    row = _design_row(coef.size, y_lags, u_next, u_lags, p, s, m, include_const)
    return float(row @ coef)


def forecast_multi_step(coef: np.ndarray, y_hist: np.ndarray,
                        u_hist: np.ndarray, u_future: np.ndarray,
                        p: int, s: int, include_const: bool = True
                        ) -> np.ndarray:
    """Recursive h-step forecast, shape (h,). u_future is (h, m) and required.

    The forecast error grows with h even though u is treated as known exactly:
    the unobserved e_{T+1}..e_{T+h-1} enter through the lagged y channel and
    are propagated by powers of the AR polynomial, giving
    var = sigma2 * sum_{j<h} psi_j^2. Nothing accumulates through u.
    """
    y_hist = list(np.asarray(y_hist, dtype=np.float64))
    u_hist = np.atleast_2d(np.asarray(u_hist, dtype=np.float64))
    u_future = np.atleast_2d(np.asarray(u_future, dtype=np.float64))
    m = u_future.shape[1]
    n_hist = u_hist.shape[0]
    # past and future u in one array so a lag is just an index offset
    u_all = np.vstack([u_hist.reshape(n_hist, m), u_future])

    out = np.empty(u_future.shape[0], dtype=np.float64)
    for i in range(u_future.shape[0]):
        at = n_hist + i
        if at - s < 0:
            raise ValueError(f"need >= {s} past u, got {n_hist}")
        y_lags = np.array(y_hist[::-1][:p])
        u_lags = u_all[at - s:at][::-1] if s > 0 else np.empty((0, m))
        row = _design_row(coef.size, y_lags, u_all[at], u_lags, p, s, m,
                          include_const)
        out[i] = row @ coef
        y_hist.append(out[i])      # the forecast becomes the next lagged y
    return out


# Identification: steps 0 and 3 of the pipeline.

def stationarity_report(x: np.ndarray, regression: str = "c") -> dict:
    """ADF and KPSS on one series. Run on y and on every exogenous column.

    regression="c" allows a non-zero mean but no trend; use "ct" if the series
    plainly trends.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    adf_stat, adf_p, *_ = adfuller(x, regression=regression, autolag="AIC")
    # kpss p-values are clipped to [0.01, 0.10] -- they come from a lookup
    # table, so 0.1 means ">= 0.1", not exactly 0.1
    kpss_stat, kpss_p, *_ = kpss(x, regression=regression, nlags="auto")
    return {
        "adf_stat": float(adf_stat), "adf_p": float(adf_p),
        "kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p),
        "adf_says_stationary": adf_p < 0.05,
        "kpss_says_stationary": kpss_p > 0.05,
    }


def ccf_prewhitened(y: np.ndarray, u: np.ndarray, max_lag: int,
                    ar_order: int | None = None, ar_max: int = 12
                    ) -> tuple[np.ndarray, int]:
    """Box-Jenkins prewhitened cross-correlation at lags 0..max_lag.

    Returns (ccf, ar_order). The filter is an AR fitted to u alone and applied
    to both series; ar_order is chosen by BIC over 1..ar_max if not given.

    The result estimates the impulse response v(L) up to a scale factor, so
    read the delay off the first significant spike. Do not read the order off
    the last one: for ARX v(L) = B(L)/Phi(L) and the AR denominator leaves a
    geometric tail past the true s. On arx_data.csv (true s=1) lags 1 and 2
    both clear the band for that reason.
    """
    from statsmodels.tsa.ar_model import AutoReg
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    if ar_order is None:
        best = (np.inf, 1)
        for q in range(1, ar_max + 1):
            res = AutoReg(u, lags=q).fit()
            if res.bic < best[0]:
                best = (res.bic, q)
        ar_order = best[1]

    phi = AutoReg(u, lags=ar_order).fit().params[1:]   # params[0] is the intercept

    def whiten(v: np.ndarray) -> np.ndarray:
        """Apply 1 - phi_1 L - ... - phi_q L^q, dropping the first q rows."""
        out = v[ar_order:].copy()
        for j in range(1, ar_order + 1):
            out -= phi[j - 1] * v[ar_order - j:v.size - j]
        return out

    a = whiten(u)          # filtered input, approximately white
    b = whiten(y)          # filtered output, same filter so v(L) is unchanged
    a = a - a.mean()
    b = b - b.mean()
    n = a.size
    denom = np.sqrt(float(a @ a) * float(b @ b)) / n

    out = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        out[k] = (float(b[k:] @ a[:n - k]) / n) / denom
    return out, ar_order


def plot_correlogram(values: np.ndarray, n: int, title: str = "",
                     start_lag: int = 1, ax=None):
    """Stem plot of an ACF/PACF/CCF against lag, with the +-1.96/sqrt(n) band.

    start_lag=1 for acf/pacf, which skip lag 0; 0 for ccf_prewhitened, which
    includes it.
    """
    import matplotlib.pyplot as plt
    values = np.asarray(values, dtype=np.float64)
    lags = np.arange(start_lag, start_lag + values.size)
    band = significance_band(n)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.2))
    ax.stem(lags, values, basefmt=" ")   # basefmt=" " drops the baseline stub
    ax.axhline(0.0, lw=0.8, color="black")
    ax.axhline(band, ls="--", lw=0.9, color="crimson")
    ax.axhline(-band, ls="--", lw=0.9, color="crimson")
    ax.set_xlabel("lag")
    ax.set_title(title or f"correlogram (band = +-{band:.3f})")
    return ax


def fit_statsmodels_ardl(y: np.ndarray, U: np.ndarray, p: int, s: int):
    """Reference fit. Should match fit_ols to ~1e-15.

    ARDL is the dynamic parameterisation at the top of this file, SARIMAX is
    not; check exog_parameterisation in ground_truth.json before concluding a
    hand fit is wrong. `order` also accepts a dict {column: lag} for
    per-feature exogenous lags.
    """
    from statsmodels.tsa.ardl import ARDL
    return ARDL(endog=y, lags=p, exog=U, order=s, trend="c").fit()
