"""NLL by position out to 16 L on the FineWeb validation shard, for the SG versus T comparison. Run from the repo root:

$ python scripts/fineweb/long_eval.py --out_dir=<run> --dest=<dir>            # SG / S: one pass
$ python scripts/fineweb/long_eval.py --out_dir=<run> --dest=<dir> --window   # T: full context, then a window of L

Windows of 16 L consecutive tokens (documents concatenated with BOS, as in eval_fineweb.py), the same windows for
every model, so the per-bucket differences are paired. With --window, T is evaluated a second time with each query
seeing only the last L keys (diag_long.py's band mask). Writes <dest>/long_eval.json.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev   # noqa: E402
import variants            # noqa: E402


def run(net, val, T, n, device):
    total = torch.zeros(T, dtype=torch.float64)
    with torch.no_grad():
        for j in range(n):
            xy = torch.from_numpy(val[j * T: (j + 1) * T + 1].astype(np.int64))[None].to(device)
            logits = net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
            total += F.cross_entropy(logits.transpose(1, 2), xy[:, 1:], reduction="none")[0].double().cpu()
    per = total / n
    edges = [0, 64, 128, 256, 512] + [512 * 2 ** i for i in range(1, 20) if 512 * 2 ** i <= T]
    edges += [T] if edges[-1] < T else []   # a last partial bucket when T is not a power of two (e.g. 20 L)
    b = {f"{a}-{c}": per[a:c].mean().item() for a, c in zip(edges, edges[1:])}
    return {"nll_by_position": b, "ppl_by_position": {k: math.exp(v) for k, v in b.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--mult", type=int, default=16, help="window length in units of the training length L")
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--window", action="store_true", help="T only: also evaluate with a sliding window of L")
    ap.add_argument("--logn", action="store_true", help="T only: also evaluate the full context with log-n scaling")
    a = ap.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    variants.HEAD_DIM = meta.get("head_dim", 48)
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, device)
    L = ckpt["model_args"]["max_seq_len"]
    T = a.mult * L
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    out = {"run": os.path.basename(a.out_dir.rstrip("/")), "variant": meta["variant"], "seed": ckpt["config"]["seed"],
           "L": L, "T": T, "windows": a.windows, "full": run(net, val, T, a.windows, device)}
    print(json.dumps(out), flush=True)
    if a.logn:   # full context again, with log-n scaling past L
        variants.LOGN = L
        out["logn"] = run(net, val, T, a.windows, device)
        variants.LOGN = None
        print(json.dumps(out["logn"]), flush=True)
    if a.window:
        q = torch.arange(T, device=device)
        band = (q[None, :] <= q[:, None]) & (q[None, :] > q[:, None] - L)   # key within the last L positions

        def attn(self, x):
            B, T_, d = x.shape
            h = self.mix(self.norm(x))
            qq, k, v = (w(h).view(B, T_, self.h, variants.HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
            y = F.scaled_dot_product_attention(variants._rope(self._scale(qq, window=L)), variants._rope(k), v, attn_mask=band[:T_, :T_])
            return x + self.mu(y.transpose(1, 2).reshape(B, T_, d))
        variants.AttentionHalf.forward = attn
        out["window"] = run(net, val, T, a.windows, device)
        print(json.dumps(out["window"]), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "long_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
