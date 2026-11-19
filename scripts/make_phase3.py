"""
Writes the phase 3 plan from the frozen choices: 100K iters with the phase 0 recipe, warmup 1000, eval every 2000.
Seeds start at 2001 and never use 1337, the screening seed. Standard library only. Run from the repository root:

$ python3 scripts/make_phase3.py SEEDS ARM:LR ...    # e.g. 3 s:2e-3 d:2e-3 pfree:2e-3 peq105x70:2e-3 a:2e-3

Use SEEDS=3 with only s:LR first: those three S seeds fix n (analyze_phase3.py n) and are also the S arm's first
three confirmatory runs.
"""
import sys

base = open("scripts/nrp/phase0b.txt").read().split(" ", 1)[1].strip()
variant = {l.split(" ", 1)[0].rsplit("_lr", 1)[0]: l.split(" --vocab_source", 1)[0].split(" ", 1)[1]
           for l in open("scripts/nrp/phase2a.txt") if l.strip()}
base = base.replace("--variant=S ", "").replace("--seed=1337", "--seed={seed}")
n = int(sys.argv[1])
for spec in sys.argv[2:]:
    arm, lr = spec.split(":")
    for k in range(n):
        seed = 2001 + k
        flags = base.format(seed=seed).replace("--learning_rate=1e-3", f"--learning_rate={lr}")
        print(f"{arm}_seed{seed} {variant[arm]} {flags}")
