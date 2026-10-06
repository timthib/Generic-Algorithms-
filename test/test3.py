"""
QRT HackerRank mock · h2 · Sliding-window median   [Hard]

Return the list of medians of every contiguous window of size k in `a`, as floats.
For even k the median is the mean of the two middle values.
If k <= 0 or k > len(a), return [].

Example: a=[1,3,-1,-3,5,3,6,7], k=3  ->  [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]
Constraints: len(a) <= 10^5, k <= 10^3 in the large tests

HOW TO USE: write your solution in `window_medians` below and run the file.
Visible tests are what the exam shows you; hidden tests are the kind used to score you.
"""

import bisect

def window_medians(a, k):
    medians = []
    window = a[:k]
    window_sorted = sorted(window)
    # numbers = {}
    # for i in range(len(window_sorted)):
    #     for nb_available in range(window_sorted[i],window_sorted[i+1]):
    #         numbers[nb_available] = i

    d = {position:number for i, group in enumerate(window_sorted) for number, position in group}

    if k%2 == 0: 
        medians.append(1/2 * (window_sorted[k//2-1]+window_sorted[k//2]))
    else :
        medians.append(window_sorted[k//2])
    for i in range(k+1,len(a)):
        window_sorted.remove(a[i-k-1])
        bisect.insort(window_sorted,a[i])
        if k%2 == 0: 
            medians.append(1/2 * (window_sorted[k//2-1]+window_sorted[k//2]))
        else :
            medians.append(window_sorted[k//2])

    return medians


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

_big = [(i * 7919) % 10007 for i in range(100_000)]
VISIBLE = [
    ("sample 1", ([1, 3, -1, -3, 5, 3, 6, 7], 3), [1.0, -1.0, -1.0, 3.0, 5.0, 6.0]),
    ("sample 2 (even k)", ([1, 2, 3, 4], 2), [1.5, 2.5, 3.5]),
]
HIDDEN = [
    ("k = 1", ([4, -2, 7], 1), [4.0, -2.0, 7.0], "Every element is its own median."),
    ("k = len(a)", ([5, 1, 3], 3), [3.0], "Exactly one window."),
    ("k > len(a)", ([1, 2], 3), [], "No full window: return []."),
    ("duplicates", ([2, 2, 2, 2], 2), [2.0, 2.0, 2.0], "Removing the outgoing value must remove ONE copy only."),
    ("large values", ([10**9, 10**9 + 1], 2), [1000000000.5], "Average as a float."),
    ("n = 10^5, k = 1000", (_big, 1000), None, "Re-sorting every window is O(n k log k) and times out; keep a sorted window (bisect) or two heaps."),
]
def CHECK(got, exp):
    if exp is None:
        return isinstance(got, list) and len(got) == len(_big) - 999 and abs(sum(got) - 495_306_597.0) < 1e-3
    return _same(got, exp)


if __name__ == "__main__":
    run_suite(window_medians, VISIBLE, HIDDEN, check=globals().get("CHECK"), time_limit=2.0)
