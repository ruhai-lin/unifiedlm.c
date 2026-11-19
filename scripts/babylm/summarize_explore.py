"""
Collects the exploratory BabyLM follow-ups (babylm-plan.md section 13) across seeds into one JSON:
  share growth   median over blocks of context share (1024-2048) / (256-512): normal stream with the block's own
                 or the other model's time constant, and the clamped stream (no growth passed down) likewise
  context        same tokens, long window minus 512 window, by position in the long window
  long           per-position NLL on 4096 and 8192 windows (seed 0 only)

$ python scripts/babylm/summarize_explore.py scripts/reports
"""
import json
import os
import statistics
import sys

R = sys.argv[1]
E0, ES = os.path.join(R, "babylm", "explore"), os.path.join(R, "babylm_seeds")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def growth(block_stats, key):
    return round(statistics.median(a / b for a, b in zip(block_stats[key]["1024-2048"], block_stats[key]["256-512"])), 3)


out = {"share_growth": {}, "context": {}, "long": None}
for k in (0, 1, 2):
    tau = load(os.path.join(E0, "explore_seed0.json")) if k == 0 else load(os.path.join(ES, f"explore_seed{k}.json"))
    intr = load(os.path.join(E0, "intrinsic_seed0.json")) if k == 0 else tau
    if not tau or "tau" not in tau or not intr or "intrinsic" not in intr:
        continue
    out["share_growth"][k] = {v: {
        "normal, own tau": growth(tau["tau"][v], "share_own_tau"),
        "normal, other tau": growth(tau["tau"][v], "share_other_tau"),
        "no growth passed down, own tau": growth(intr["intrinsic"][v], "share_own_tau"),
        "no growth passed down, other tau": growth(intr["intrinsic"][v], "share_other_tau"),
        "tau range": [min(tau["tau"][v]["tau_own"]), max(tau["tau"][v]["tau_own"])]} for v in ("S", "P")}

ctx = {0: load(os.path.join(E0, "context_seed0.json")) or {}}
for k in (1, 2):
    ctx[k] = {}
    for f in ("context_p_seeds.json", f"context_s_seed{k}.json", "context_t_seeds.json"):
        for name, r in (load(os.path.join(ES, f)) or {}).items():
            if name.endswith(f"seed{k}"):
                ctx[k][name.rsplit("_seed", 1)[0]] = r
for k, runs in ctx.items():
    for name, r in runs.items():
        out["context"].setdefault(name, {})[k] = {L: {b: round(x["long_minus_short"], 4) for b, x in r[L].items()}
                                                 for L in r if L != "variant"}

e0 = load(os.path.join(E0, "explore_seed0.json"))
if e0 and "long" in e0:
    out["long"] = {L: {v: {m: {b: round(x, 4) for b, x in e0["long"][L][v][m]["nll"].items()} for m in e0["long"][L][v]}
                       for v in ("S", "P")} for L in e0["long"]}
t0 = load(os.path.join(E0, "t_long_seed0.json"))
if t0:
    out["t_2048_seed0"] = {b: round(x, 4) for b, x in t0["by_position"].items()}
print(json.dumps(out, indent=1))
