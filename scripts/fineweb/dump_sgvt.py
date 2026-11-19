"""Prints every /workspace/runs/sgvt/<run> (long_eval.json and samples.jsonl, scripts/fineweb/sgvt.sh) as one JSON
object, for scripts/explainer/gen_sgvt.py. Run on a pod."""
import glob
import json
import os

out = {}
for d in sorted(glob.glob("/workspace/runs/sgvt/*/long_eval.json")):
    d = os.path.dirname(d)
    out[os.path.basename(d)] = {"long": json.load(open(os.path.join(d, "long_eval.json"))),
                                "samples": [json.loads(l) for l in open(os.path.join(d, "samples.jsonl")) if l.strip()]}
print(json.dumps(out, ensure_ascii=False))
