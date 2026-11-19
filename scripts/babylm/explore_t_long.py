"""
Exploratory: the Transformer's per-position NLL on the 2048-token validation windows, which eval_babylm.py skips by
rule (its RoPE positions beyond 512 were never trained), so it can sit beside S and P-free beyond the training length.

$ python scripts/babylm/explore_t_long.py data/babylm10m/runs/babylm_t out.json
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
import eval_babylm as ev  # noqa: E402

torch.set_num_threads(os.cpu_count())
net, meta, _ = ev.load(sys.argv[1], "cpu")
assert meta["variant"] == "T"
r = ev.nll_by_position(net, 2048, 4, "cpu", 10 ** 9)
json.dump(r, open(sys.argv[2], "w"), indent=1)
print(json.dumps(r))
