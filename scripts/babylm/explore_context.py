"""
Exploratory: does context beyond the training length help or hurt the same tokens?

Scores every validation token twice, once in the 512-token windows and once in L-token windows (both cut in order
from the start of val.bin), and compares the two only on tokens at positions 256-511 of their 512-token window,
so the short context always has at least 256 tokens. Grouped by position in the long window. In the long
window's 256-512 range both contexts are the same tokens, so the difference there must be about 0 (a check).
Negative differences mean the longer context lowers the loss.

$ python scripts/babylm/explore_context.py --out=x.json data/babylm10m/runs/babylm_s data/babylm10m/runs/babylm_p ...
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
import eval_babylm as ev  # noqa: E402

EDGES = [0, 256, 512, 1024, 2048, 4096, 8192]


@torch.no_grad()
def per_token(net, L, batch):
    m = np.memmap(os.path.join(ev.DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    n = (len(m) - 1) // L
    out = torch.empty(n * L, dtype=torch.float64)
    for i in range(0, n, batch):
        xy = torch.from_numpy(np.stack([m[j * L: (j + 1) * L + 1].astype(np.int64) for j in range(i, min(n, i + batch))]))
        x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
        nll = F.cross_entropy(net(x, y).transpose(1, 2), y, reduction="none")
        out[i * L: i * L + nll.numel()] = nll.flatten().double()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2048,8192")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    res = {}
    for run in a.runs:
        net, meta, _ = ev.load(run, "cpu")
        short = per_token(net, 512, 16)
        res[os.path.basename(run)] = r = {"variant": meta["variant"]}
        for L in map(int, a.lengths.split(",")):
            long = per_token(net, L, max(1, 4096 // L))
            N = min(len(long), len(short))
            pos = torch.arange(N)
            keep = (pos % 512) >= 256
            local = pos % L
            r[str(L)] = {}
            for lo, hi in zip(EDGES, EDGES[1:]):
                mask = keep & (local >= lo) & (local < hi)
                if mask.any():
                    d = long[:N][mask] - short[:N][mask]
                    r[str(L)][f"{lo}-{hi}"] = {"tokens": int(mask.sum()), "short": short[:N][mask].mean().item(),
                                               "long": long[:N][mask].mean().item(), "long_minus_short": d.mean().item(),
                                               "se": (d.std() / mask.sum().sqrt()).item()}
            print(os.path.basename(run), L, json.dumps({b: round(x["long_minus_short"], 4) for b, x in r[str(L)].items()}), flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
