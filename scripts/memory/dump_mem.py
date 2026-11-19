"""Prints the S, SG and SH memory-task results on the PVC as JSON, for scripts/explainer/gen_sg.py. Run on a pod."""
import json, os
P = {"m2_d128": ("mem_m23/m2_s_d128", "memsg/m2_sg_d128", "memsh/m2_sh_d128"), "m3_d128": ("mem_m23/m3_s_d128", "memsg/m3_sg_d128"),
     "m2kv_d32_seed1": ("mem_seeds/m2kv_s_d32_seed1", "memsg/m2kv_sg_d32_seed1", "memsh/m2kv_sh_d32_seed1")}
out = {}
for k, pair in P.items():
    out[k] = {}
    for v, p in zip(("S", "SG", "SH"), pair):
        f = "/workspace/runs/" + p + "/eval.json"
        if os.path.exists(f):
            e = json.load(open(f))
            out[k][v] = {x: e.get(x) for x in ("task", "dim", "slots", "params", "seq", "counts", "gaps", "filler", "test", "diverged")}
print(json.dumps(out))
