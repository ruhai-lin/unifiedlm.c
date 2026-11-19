"""
Applies scripts/reports/phase2-protocol.md to finished screening runs. Run from the repository root:

$ python3 scripts/select_phase2.py 2a RUNS_DIR    # prints the phase 2b plan file
$ python3 scripts/select_phase2.py 2b RUNS_DIR    # prints the frozen choices for phase 3
$ python3 scripts/select_phase2.py 2c RUNS_ROOT   # 2c: next doublings to run, or the frozen choices

For 2c, RUNS_ROOT holds phase2b/, explore/ and phase2c/; every 20K run of a config is gathered from them.

An optional third argument, a comma-separated list such as s,d,a, restricts it to those configs; the
P-eq split is then chosen only among the P-eq configs listed. RUNS_DIR holds one directory per run, named as in the plan file, each with train.log. Standard library only.
"""
import math
import os
import re
import sys

CONFIGS = ["s", "d", "pfree", "peq69x140", "peq93x93", "peq105x70", "a"]
PEQ = ["peq69x140", "peq93x93", "peq105x70"]
LRS = ["5e-4", "1e-3", "2e-3"]
VAL = re.compile(r"^step (\d+): train loss (\S+), val loss (\S+)$")
STEP = re.compile(r"^\d+ \| loss (\S+) \|")


def read(path):
    """Val losses by step, and whether every logged loss is finite."""
    vals, finite = [], True
    with open(path) as f:
        for line in f:
            line = line.strip()
            if m := VAL.match(line):
                train, val = float(m[2]), float(m[3])
                vals.append((int(m[1]), val))
                finite &= math.isfinite(train) and math.isfinite(val)
            elif m := STEP.match(line):
                finite &= math.isfinite(float(m[1]))
    return vals, finite


def run(runs, config, lr, tag):
    path = os.path.join(runs, f"{config}_lr{lr}", "train.log")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    vals, finite = read(path)
    if not vals:
        raise SystemExit(f"no val loss in {path}")
    final = vals[-1][1]
    stable = finite and final <= min(v for _, v in vals) + 0.02
    print(f"# {tag} {config} lr={lr}: final val {final:.4f} at step {vals[-1][0]}, "
          f"min {min(v for _, v in vals):.4f}, {'stable' if stable else 'UNSTABLE'}", file=sys.stderr)
    return final, stable


def plan_2b(runs, configs):
    plan2a = {l.split(" ", 1)[0]: l.split(" ", 1)[1].strip() for l in open("scripts/nrp/phase2a.txt") if l.strip()}
    for config in configs:
        res = {lr: run(runs, config, lr, "2a") for lr in LRS}
        stable = [lr for lr in LRS if res[lr][1]]
        others = sorted((lr for lr in stable if lr != "1e-3"), key=lambda lr: res[lr][0])
        if "1e-3" in stable:
            pair = ["1e-3"] + others[:1]
        else:
            pair = others[:2]
        if not pair:
            raise SystemExit(f"{config}: no stable learning rate, stop and report")
        for lr in pair:
            flags = plan2a[f"{config}_lr{lr}"]
            flags = flags.replace("--warmup_iters=50", "--warmup_iters=200").replace("--max_iters=5000", "--max_iters=20000")
            flags = flags.replace("--eval_interval=1000", "--eval_interval=2000")
            print(f"{config}_lr{lr} {flags}")


def freeze(runs, configs):
    chosen = {}
    for config in configs:
        res = {lr: run(runs, config, lr, "2b")[0] for lr in LRS
               if os.path.exists(os.path.join(runs, f"{config}_lr{lr}", "train.log"))}
        best = min(res, key=res.get)
        lr = "1e-3" if "1e-3" in res and res["1e-3"] <= res[best] + 0.01 else best
        chosen[config] = (lr, res[lr])
    peq = [min((c for c in PEQ if c in chosen), key=lambda c: chosen[c][1])] if any(c in chosen for c in PEQ) else []
    for config in [c for c in ["s", "d", "pfree"] if c in chosen] + peq + [c for c in ["a"] if c in chosen]:
        print(f"{config} lr={chosen[config][0]} 20K val {chosen[config][1]:.4f}")


def flags_20k(config, lr):
    plan2a = {l.split(" ", 1)[0]: l.split(" ", 1)[1].strip() for l in open("scripts/nrp/phase2a.txt") if l.strip()}
    flags = plan2a[f"{config}_lr2e-3"].replace("--learning_rate=2e-3", f"--learning_rate={lr}")
    flags = flags.replace("--warmup_iters=50", "--warmup_iters=200").replace("--max_iters=5000", "--max_iters=20000")
    return flags.replace("--eval_interval=1000", "--eval_interval=2000")


def phase_2c(root, configs):
    chosen, todo = {}, []
    for config in configs:
        res = {}
        for lr in ["1e-3", "2e-3", "4e-3", "8e-3", "1.6e-2", "3.2e-2", "6.4e-2"]:
            for path in (f"phase2c/{config}_lr{lr}", f"explore/{config}_20k_lr{lr}", f"phase2b/{config}_lr{lr}"):
                log = os.path.join(root, path, "train.log")
                if os.path.exists(log):
                    vals, finite = read(log)
                    final = vals[-1][1]
                    res[lr] = (final, finite and final <= min(v for _, v in vals) + 0.02)
                    print(f"# 2c {config} lr={lr}: final val {final:.4f} ({path}), {'stable' if res[lr][1] else 'UNSTABLE'}",
                          file=sys.stderr)
                    break
        tested = sorted(res, key=float)
        stable = [lr for lr in tested if res[lr][1]]
        if not stable:
            raise SystemExit(f"{config}: no stable learning rate, stop and report")
        best = min(stable, key=lambda lr: res[lr][0])
        if best == tested[-1]:
            nxt = f"{2 * float(best):.1e}".replace("e-0", "e-").replace(".0e", "e")
            todo.append(f"{config}_lr{nxt} {flags_20k(config, nxt)}")
        else:
            chosen[config] = (best, res[best][0])
    for line in todo:
        print(line)
    if not todo:
        peq = [min((c for c in PEQ if c in chosen), key=lambda c: chosen[c][1])]
        for config in ["s", "d", "pfree"] + peq + ["a"]:
            print(f"# frozen {config} lr={chosen[config][0]} 20K val {chosen[config][1]:.4f}")


if __name__ == "__main__":
    configs = sys.argv[3].split(",") if len(sys.argv) > 3 else CONFIGS
    {"2a": plan_2b, "2b": freeze, "2c": phase_2c}[sys.argv[1]](sys.argv[2], configs)
