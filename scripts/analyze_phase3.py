"""
The phase 3 decision rule, fixed in the experiment plan before any phase 3 result. Standard library only.
Run from the repository root:

$ python3 scripts/analyze_phase3.py n RUNS_DIR               # seeds per arm, from the first 3 S seeds
$ python3 scripts/analyze_phase3.py compare RUNS_DIR P_EQ     # all planned comparisons, e.g. P_EQ=peq105x70

RUNS_DIR holds <arm>_seed<k>/eval.json; the score is eval.json's fp32 "nll" (lower is better).

- n: S standard deviation at most 0.0011 gives 3 seeds, at most 0.0014 gives 4, otherwise 5. Not extended later.
- Win: Welch 95% CI of the difference excludes 0 and |difference| is at least 0.005 nats.
- Equivalent: the Welch 90% CI lies inside +-0.005.
- Anything else is inconclusive and never passes. Two claims are judged separately (Gary, 2026-11-07):
  sharing does not hurt (D vs P-free, D vs best P-eq, S vs best separated, each better or equivalent) and
  coupling helps (S vs D, better).
"""
import glob
import json
import math
import os
import statistics
import sys

MARGIN = 0.005


def betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta function (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / d if abs(d) > 1e-300 else 1e300
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)), -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / d if abs(d) > 1e-300 else 1e300
            c = 1 + aa / c if abs(c) > 1e-300 else 1e300
            h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    return h


def t_cdf(t, df):
    x = df / (df + t * t)
    lb = math.lgamma(df / 2 + 0.5) - math.lgamma(df / 2) - math.lgamma(0.5)
    ib = math.exp(lb + (df / 2) * math.log(x) + 0.5 * math.log(1 - x))
    tail = ib * betacf(df / 2, 0.5, x) / (df / 2) if x < (df / 2 + 1) / (df / 2 + 0.5 + 2) else 1 - ib * betacf(0.5, df / 2, 1 - x) / 0.5
    return 1 - tail / 2 if t >= 0 else tail / 2


def t_quantile(p, df):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def welch(a, b, level):
    """CI of mean(a) - mean(b)."""
    va, vb = statistics.variance(a) / len(a), statistics.variance(b) / len(b)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    half = t_quantile(1 - (1 - level) / 2, df) * se
    diff = statistics.mean(a) - statistics.mean(b)
    return diff, diff - half, diff + half, df


def scores(runs, arm):
    out = [json.load(open(p))["nll"] for p in sorted(glob.glob(os.path.join(runs, f"{arm}_seed*", "eval.json")))]
    if len(out) < 2:
        raise SystemExit(f"{arm}: need at least 2 finished seeds, have {len(out)}")
    return out


def verdict(a, b):
    diff, lo95, hi95, df = welch(a, b, 0.95)
    _, lo90, hi90, _ = welch(a, b, 0.90)
    if (lo95 > 0 or hi95 < 0) and abs(diff) >= MARGIN:
        v = "first worse" if diff > 0 else "first better"
    elif -MARGIN < lo90 and hi90 < MARGIN:
        v = "equivalent"
    else:
        v = "inconclusive"
    return v, f"diff {diff:+.4f}, 95% CI [{lo95:+.4f}, {hi95:+.4f}], 90% CI [{lo90:+.4f}, {hi90:+.4f}], df {df:.1f}"


def seeds_per_arm(runs):
    s = scores(runs, "s")[:3]
    if len(s) < 3:
        raise SystemExit("need the first 3 S seeds")
    sd = statistics.stdev(s)
    n = 3 if sd <= 0.0011 else 4 if sd <= 0.0014 else 5
    print(f"S seeds {', '.join(f'{x:.4f}' for x in s)}: sd {sd:.5f}, n = {n}")


def compare(runs, peq):
    arms = {a: scores(runs, a) for a in ["s", "d", "pfree", peq, "a"]}
    for a, x in arms.items():
        print(f"{a:10s} n={len(x)} mean {statistics.mean(x):.4f} sd {statistics.stdev(x):.5f}")
    best_sep = min([peq, "a"], key=lambda a: statistics.mean(arms[a]))
    # (first, second, what the unification claim needs from the first against the second)
    plan = [("s", "d", "S vs D: does the attenuation help"),
            ("d", "pfree", "D vs P-free: what releasing the shared address gains"),
            ("d", peq, "D vs best P-eq: shared against separate address at equal params"),
            ("s", best_sep, f"S vs best separated ({best_sep}): the package within this family")]
    verdicts = {}
    for first, second, what in plan:
        v, detail = verdict(arms[first], arms[second])
        verdicts[(first, second)] = v
        print(f"{what}\n  {first} - {second}: {v} ({detail})")
    sharing = all(verdicts[k] in ("first better", "equivalent") for k in [("d", "pfree"), ("d", peq), ("s", best_sep)])
    coupling = verdicts[("s", "d")] == "first better"
    print(f"sharing does not hurt: {'yes' if sharing else 'no'} (three comparisons, each better or equivalent)")
    print(f"coupling helps: {'yes' if coupling else 'no'} (S better than D)")


if __name__ == "__main__":
    if sys.argv[1] == "n":
        seeds_per_arm(sys.argv[2])
    else:
        compare(sys.argv[2], sys.argv[3])
