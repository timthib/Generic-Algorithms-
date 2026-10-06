"""
QRT HackerRank mock · h5 · Daily rank IC of a signal (pandas)   [Hard]

`df` has columns date, ticker, signal, fwd_ret (long format, unsorted).
For each date, compute the Spearman rank correlation between signal and fwd_ret over
rows where BOTH are not NaN (ties get average ranks). Drop dates with fewer than 3
valid rows. If a column is constant within a date, ic is NaN (keep the row).
Return a DataFrame with columns ["date", "ic", "n"], sorted by date, index 0..m-1.

HOW TO USE: write your solution in `daily_rank_ic` below and run the file.
Visible tests are what the exam shows you; hidden tests are the kind used to score you.
"""
import numpy as np
import pandas as pd

def daily_rank_ic(df):
    def valid_count(row) : 
        if len(row[(row['signal'] != "NaN") & (row['fwd_ret'] != "NaN")]) < 3 : 
            return False 
        return True

    for row in df.iterrows():
        if valid_count(row) == False: 
            df.drop_row(row)
    
    


# ======================================================================
# Everything below is the test harness. You don't need to edit it.
# ======================================================================
# ---------------------------------------------------------------- test runner
import math, time, traceback, signal

class _Timeout(Exception):
    pass

def _alarm(seconds):
    # Hard stop for runaway solutions (Unix/macOS; on Windows tests just run to the end).
    if hasattr(signal, "setitimer"):
        try:
            signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_Timeout()))
            signal.setitimer(signal.ITIMER_REAL, seconds)
            return True
        except ValueError:          # not in the main thread
            return False
    return False

def _clear():
    if hasattr(signal, "setitimer"):
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except ValueError:
            pass

class CALL:
    """A test that calls a custom function with your solution as its argument."""
    def __init__(self, fn): self.fn = fn

def _same(got, exp, tol=1e-6):
    if isinstance(exp, float):
        if exp != exp:
            return isinstance(got, float) and got != got
        return isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - exp) <= tol * max(1, abs(exp))
    if isinstance(exp, (tuple, list)) and isinstance(got, (tuple, list)):
        return len(got) == len(exp) and all(_same(g, e, tol) for g, e in zip(got, exp))
    return got == exp

def _run(fn, case, check, time_limit):
    name, args, expected = case[:3]
    t0 = time.perf_counter()
    _alarm(max(10.0, 5 * time_limit))
    try:
        try:
            got = args.fn(fn) if isinstance(args, CALL) else fn(*args)
        finally:
            _clear()
        dt = time.perf_counter() - t0
        if isinstance(args, CALL):
            ok = bool(got) if expected is True else bool(check(got, expected))
        elif callable(expected):
            ok = bool(expected(got))
        else:
            ok = bool(check(got, expected))
        status = "PASS" if ok and dt <= time_limit else ("SLOW" if ok else "FAIL")
        return status, dt, got, None
    except _Timeout:
        return "SLOW", time.perf_counter() - t0, None, "stopped after %.0fs: far too slow, this would time out" % max(10.0, 5 * time_limit)
    except Exception as ex:
        return "ERROR", time.perf_counter() - t0, None, f"{type(ex).__name__}: {ex}"

def run_suite(fn, visible, hidden, check=None, time_limit=2.0, show_hidden_reasons=True):
    check = check or _same
    total = {"visible": [0, len(visible)], "hidden": [0, len(hidden)]}
    for section, cases in (("VISIBLE TESTS (shown in the exam)", visible), ("HIDDEN TESTS (used for scoring, not shown in the exam)", hidden)):
        print("\n" + section + "\n" + "-" * len(section))
        key = "visible" if section.startswith("VISIBLE") else "hidden"
        for case in cases:
            status, dt, got, err = _run(fn, case, check, time_limit)
            total[key][0] += status == "PASS"
            print(f"[{status:5}] {case[0]}  ({dt*1000:.0f} ms)")
            if status in ("FAIL", "ERROR", "SLOW"):
                if err:
                    print(f"         {err}")
                elif status == "FAIL":
                    exp = case[2]
                    shown = "see the check" if callable(exp) or isinstance(case[1], CALL) or exp is None else repr(exp)
                    print(f"         got:      {repr(got)[:200]}\n         expected: {shown[:200]}")
                else:
                    print(f"         over {time_limit:.0f}s: likely a timeout on HackerRank")
                if len(case) > 3 and show_hidden_reasons:
                    print(f"         why this test exists: {case[3]}")
    v, h = total["visible"], total["hidden"]
    allp = v[0] + h[0]; alln = v[1] + h[1]
    print(f"\nVisible {v[0]}/{v[1]} · Hidden {h[0]}/{h[1]} · Score {allp}/{alln} ({100*allp/max(alln,1):.0f}%)")

def _df(rows): return pd.DataFrame(rows, columns=["date", "ticker", "signal", "fwd_ret"])
def CHECK(got, exp):
    if not isinstance(got, pd.DataFrame) or list(got.columns) != ["date", "ic", "n"]:
        return False
    g = got.reset_index(drop=True)
    return (list(g["date"]) == exp["date"] and list(map(int, g["n"])) == exp["n"]
            and np.allclose(g["ic"].astype(float), exp["ic"], atol=1e-9, equal_nan=True))
_base = [("d1", t, s, r) for t, s, r in [("A", 1, .01), ("B", 2, .02), ("C", 3, .03), ("D", 4, .05)]]
VISIBLE = [
    ("perfect ranking", (_df(_base),), {"date": ["d1"], "ic": [1.0], "n": [4]}),
    ("reversed ranking", (_df([("d2", t, -s, r) for _, t, s, r in _base]),), {"date": ["d2"], "ic": [-1.0], "n": [4]}),
]
HIDDEN = [
    ("NaNs excluded from n", (_df(_base + [("d1", "E", np.nan, .1), ("d1", "F", 5, np.nan)]),), {"date": ["d1"], "ic": [1.0], "n": [4]}, "Drop rows where EITHER value is NaN before ranking."),
    ("date with 2 valid rows dropped", (_df(_base + [("d0", "A", 1, .1), ("d0", "B", 2, .2)]),), {"date": ["d1"], "ic": [1.0], "n": [4]}, "Minimum of 3 observations per date."),
    ("ties use average ranks", (_df([("d1", "A", 1, 1), ("d1", "B", 1, 2), ("d1", "C", 2, 3), ("d1", "D", 3, 4)]),), {"date": ["d1"], "ic": [0.9486832980505138], "n": [4]}, "Spearman = Pearson on average ranks; argsort-based ranks break ties wrongly."),
    ("unsorted dates", (_df([(d, t, s, r) for d in ["d3", "d1"] for _, t, s, r in _base]),), {"date": ["d1", "d3"], "ic": [1.0, 1.0], "n": [4, 4]}, "Output sorted by date."),
    ("constant signal gives NaN", (_df([("d1", t, 7, r) for _, t, s, r in _base]),), {"date": ["d1"], "ic": [np.nan], "n": [4]}, "Keep the date with ic = NaN; don't crash on zero variance."),
]


if __name__ == "__main__":
    run_suite(daily_rank_ic, VISIBLE, HIDDEN, check=globals().get("CHECK"), time_limit=2.0)
