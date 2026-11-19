"""Two measurements past the training length L on the FineWeb validation shard, at 4L. Run from the repo root:

$ python scripts/fineweb/diag_long.py --out_dir=<run>

- T: NLL by position with a sliding window of L (each query sees at most the last L keys), next to the full-context
  numbers in eval.json. The window keeps every relative distance inside what training saw and caps the KV cache at L.
- S / SG / SR: the prior share of the read by position, sum_a w_a q_{t,a} / sum_a r_{t,a} = the r-weighted mean of
  w_a / c_{t,a}, per layer. It is 1 before any evidence and falls as evidence c accumulates; if it keeps falling
  past L into values never seen in training, the slots' static contents (the FFN part, mu) are being washed out.
Writes diag_long.json into the run directory.
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev   # noqa: E402
import model               # noqa: E402
import variants            # noqa: E402

EDGES = [0, 64, 128, 256, 512, 1024, 2048, 4096]


def buckets(per):
    return {f"{a}-{b}": per[a:b].mean().item() for a, b in zip(EDGES, EDGES[1:]) if b <= len(per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    variants.HEAD_DIM = meta.get("head_dim", 48)
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, device)
    L = ckpt["model_args"]["max_seq_len"]
    T = 4 * L
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    out = {"variant": meta["variant"], "L": L, "T": T, "windows": a.windows}
    rec = {}

    if meta["variant"] in ("T", "TC"):
        band = torch.ones(T, T, dtype=torch.bool, device=device).tril()
        band &= ~torch.ones(T, T, dtype=torch.bool, device=device).tril(-L)   # key within the last L positions

        def attn(self, x):
            B, T_, d = x.shape
            h = self.mix(self.norm(x))
            q, k, v = (w(h).view(B, T_, self.h, variants.HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
            y = F.scaled_dot_product_attention(variants._rope(q), variants._rope(k), v, attn_mask=band[:T_, :T_])
            return x + self.mu(y.transpose(1, 2).reshape(B, T_, d))
        variants.AttentionHalf.forward = attn
    else:
        def wrap(fn, w_of):
            def f(r, v, g, w):
                q, fast = fn(r, v, g, w)
                num = (q.float() * w.float()).sum(-1).sum(0)
                den = r.float().sum(-1).sum(0)
                i = rec.setdefault("_i", 0)
                rec.setdefault(i, [torch.zeros_like(num), torch.zeros_like(den)])
                rec[i][0] += num
                rec[i][1] += den
                rec["_i"] = i + 1
                return q, fast
            return f
        if meta["variant"] in ("SG", "SG64", "SGh", "SGs", "SR"):
            import gated
            gated.gated_memory = wrap(gated.gated_memory, None)
        else:
            model.unified_memory = wrap(model.unified_memory, None)

    total = torch.zeros(T, dtype=torch.float64)
    n = 0
    with torch.no_grad():
        for i in range(0, a.windows, a.batch):
            rows = [val[j * T: (j + 1) * T + 1].astype(np.int64) for j in range(i, min(a.windows, i + a.batch))]
            xy = torch.from_numpy(np.stack(rows)).to(device)
            x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
            rec["_i"] = 0
            logits = net(x, y)
            total += F.cross_entropy(logits.transpose(1, 2), y, reduction="none").sum(0).double().cpu()
            n += len(rows)
    per = total / n
    out["nll_by_position"] = buckets(per)
    if meta["variant"] not in ("T", "TC"):
        layers = sorted(k for k in rec if k != "_i")
        out["prior_share_by_layer"] = [buckets((rec[k][0] / rec[k][1]).double().cpu()) for k in layers]
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "diag_long.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
