"""Print one JSON object with everything the result pages need from /workspace/runs: per run its eval.json,
diag_long.json, timing.json, variant.json and the val-loss curve from train.log, plus any bench files.
Runs inside a pod (stdlib only):

$ python3 scripts/fineweb/dump_runs.py > runs.json
"""
import glob
import json
import os
import re

ROOT = "/workspace/runs"
out = {"runs": {}, "bench": {}}
for d in sorted(glob.glob(f"{ROOT}/fw*/*/")):
    name = os.path.basename(d.rstrip("/"))
    r = {"plan": os.path.basename(os.path.dirname(d.rstrip("/")))}
    for f in ("eval", "diag_long", "timing", "variant", "gates"):
        p = os.path.join(d, f + ".json")
        if os.path.exists(p):
            r[f] = json.load(open(p))
    log = os.path.join(d, "train.log")
    if os.path.exists(log):
        r["curve"] = [[k, v] for k, v in sorted({int(m[1]): float(m[2]) for m in re.finditer(  # a resumed run repeats its first eval
                r"step (\d+): train loss [\d.]+, val loss ([\d.]+)", open(log).read())}.items())]
    if len(r) > 1:   # skip directories that hold no run (fwfkdiag/tokgap*: analysis output)
        out["runs"][name] = r
for p in sorted(glob.glob(f"{ROOT}/bench/*.jsonl")):
    out["bench"][os.path.basename(p)] = [json.loads(l) for l in open(p) if l.startswith("{")]
print(json.dumps(out))
