"""
Applies the M1 rules of scripts/reports/memory-plan.md section 6 to finished runs (each a directory with
eval.json from mqar.py). Standard library only.

$ python scripts/memory/analyze.py plan STEPS            # prints the screening plan, 36 lines
$ python scripts/memory/analyze.py select RUNS_DIR       # picks each (model, width) LR; prints the seed plan
$ python scripts/memory/analyze.py report RUNS_DIR       # seed means per N and the S vs P verdict per width
"""
import json
import os
import sys

MODELS, DIMS, LRS = ["S", "P", "T"], [32, 64, 128], ["3e-4", "1e-3", "3e-3", "1e-2"]
EDGE = {"3e-4": "1e-4", "1e-2": "3e-2"}
SEEDS = [2001, 2002, 2003]


def load(root):
    runs = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "eval.json")
        if os.path.exists(p):
            runs.append({"name": name, **json.load(open(p))})
    return runs


def mean_acc(split):
    return sum(split.values()) / len(split)


def flags(model, dim, lr, steps, seed=None):
    return f"--model={model} --dim={dim} --lr={lr} --steps={steps}" + (f" --seed={seed}" if seed else "")


def select(root):
    runs = [r for r in load(root) if r["seed"] == 1337]
    steps = {r["steps"] for r in runs}
    assert len(steps) == 1, f"mixed step budgets {steps}"
    steps = steps.pop()
    out, extra = [], []
    for model in MODELS:
        for dim in DIMS:
            cand = [r for r in runs if r["model"] == model and r["dim"] == dim]
            if not cand:
                continue
            cand.sort(key=lambda r: r["lr"])  # ties go to the smaller LR
            best = max(cand, key=lambda r: (0 if r["diverged"] else mean_acc(r["val"])))
            best = next(r for r in cand if not r["diverged"] and mean_acc(r["val"]) >= mean_acc(best["val"]) - 0.005)
            lr = next(k for k in LRS + list(EDGE.values()) if float(k) == best["lr"])
            tried = {r["lr"] for r in cand}
            line = " ".join(f"{r['lr']:g}:{mean_acc(r['val']):.4f}" for r in cand)
            if lr in EDGE and float(EDGE[lr]) not in tried:
                extra.append(f"m1_{model.lower()}_d{dim}_lr{EDGE[lr]} {flags(model, dim, EDGE[lr], steps)}")
                print(f"# {model} d={dim}: best {lr} is at the grid edge, try {EDGE[lr]} first   [{line}]")
                continue
            print(f"# {model} d={dim}: lr {lr}   [{line}]")
            out += [f"m1s_{model.lower()}_d{dim}_s{s} {flags(model, dim, lr, steps, s)}" for s in SEEDS]
    if extra:
        print("\n".join(["# edge extensions, run these and select again:"] + extra))
    else:
        print("\n".join(out))


def report(root):
    runs = [r for r in load(root) if r["seed"] in SEEDS]
    table = {}
    for r in runs:
        table.setdefault((r["model"], r["dim"]), []).append(r)
    for dim in DIMS:
        print(f"d={dim}")
        means = {}
        for model in MODELS:
            rs = table.get((model, dim), [])
            if len(rs) != len(SEEDS):
                print(f"  {model}: {len(rs)} of {len(SEEDS)} seeds")
                continue
            counts = rs[0]["counts"]
            means[model] = {N: sum(r["test"][str(N)] for r in rs) / len(rs) for N in counts}
            print(f"  {model}: " + " ".join(f"N{N} {means[model][N]:.3f}" for N in counts))
        if "S" in means and "P" in means:
            gaps = {N: means["S"][N] - means["P"][N] for N in means["S"]}
            worse = [N for N, g in gaps.items() if g <= -0.05]
            better = [N for N, g in gaps.items() if g >= 0.05]
            verdict = ("sharing hurts memory" if worse else "sharing is better" if better else "no difference")
            print("  S-P: " + " ".join(f"N{N} {g:+.3f}" for N, g in gaps.items()) + f"  -> {verdict}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "plan":
        steps = int(sys.argv[2])
        for model in MODELS:
            for dim in DIMS:
                for lr in LRS:
                    print(f"m1_{model.lower()}_d{dim}_lr{lr} {flags(model, dim, lr, steps)}")
    elif mode == "select":
        select(sys.argv[2])
    elif mode == "report":
        report(sys.argv[2])
