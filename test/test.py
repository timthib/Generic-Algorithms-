"""
QRT HackerRank mock · a2 · Subarrays summing to K   [Standard]

Given integers `a` (may be negative or zero) and `k`, return the number of
contiguous subarrays whose sum equals `k`.

Example: a=[1,2,3], k=3  ->  2   ([1,2] and [3])
Constraints: 0 <= len(a) <= 10^5, |a[i]| <= 10^4

HOW TO USE: write your solution in `count_subarrays` below and run the file.
Visible tests are what the exam shows you; hidden tests are the kind used to score you.
"""


# def count_subarrays(a, k):
#     s={}
#     nb_sol = 0
#     for i in range(len(a)):
#         for key,val in s.items():
#             s[key] += a[i]
#             if s[key] == k:
#                 nb_sol += 1
#         s[i]=a[i]
#         if a[i] == k : 
#             nb_sol +=1
#     return nb_sol
    
def count_subarrays(a,k):
    d = {}
    current_sum = 0
    nb_sol = 0
    for e in range(0,len(a)):
        current_sum += a[e]
        needed_val = current_sum - k
        if needed_val == 0: 
            nb_sol += 1
        if needed_val in d.keys(): 
            nb_sol += d[needed_val]
        d[current_sum] = d.get(current_sum,0)+1
    return nb_sol


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

VISIBLE = [
    ("sample 1", ([1, 2, 3], 3), 2),
    ("sample 2", ([1, -1, 0], 0), 3),
]
HIDDEN = [
    ("empty list", ([], 0), 0, "No subarrays at all."),
    ("all zeros", ([0, 0, 0], 0), 6, "Prefix sums repeat: you need COUNTS per prefix sum, not a set."),
    ("alternating signs", ([1, -1, 1, -1], 0), 4, "Negative numbers break the sliding-window idea."),
    ("mixed values", ([3, 4, 7, 2, -3, 1, 4, 2], 7), 4, "Subarrays starting at index 0 count only if prefix 0 is seeded."),
    ("no match", ([1, 2, 3], 100), 0, "Return 0, not None."),
    ("n = 10^5", ([1] * 100_000, 1), 100_000, "O(n^2) over all pairs times out; needs O(n) prefix sums + hash map."),
]


run_suite(count_subarrays, VISIBLE, HIDDEN, check=globals().get("CHECK"), time_limit=2.0)
