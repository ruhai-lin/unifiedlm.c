"""What the forget gate learned, for an S, SR or SG FineWeb run, on validation windows of length L. Per layer:
- the per-slot base horizon 1 / rate_a (S: one number per layer), median and 10th / 90th percentile;
- for SG / SR, the forgetting actually applied, -log lam_{t,a}, averaged over slots, then over tokens: as an
  effective horizon 1 / mean, its spread over tokens (10th / 90th percentile of the per-token horizon), and the
  mean at document starts (the token after a BOS) against all other tokens;
- beta (SG).
Writes gates.json into the run directory. Run from the repo root:

$ python scripts/fineweb/gates.py --out_dir=<run>
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev   # noqa: E402
import variants            # noqa: E402


def q(x, p):
    return torch.quantile(x.float().flatten()[:1_000_000], p).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=64)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, "cuda")
    L = ckpt["model_args"]["max_seq_len"]
    out = {"variant": meta["variant"], "layers": []}
    for layer in net.layers:
        rate = layer.log_rate.detach().exp().flatten()
        hz = 1 / rate
        out["layers"].append({"base_horizon_median": q(hz, .5), "base_horizon_p10": q(hz, .1), "base_horizon_p90": q(hz, .9)})
        if hasattr(layer, "log_pm1"):                       # SDp: the learned address power
            out["layers"][-1]["power"] = 1.0 + layer.log_pm1.detach().clamp(-5.0, 2.0).exp().item()
    if meta["variant"] in ("SG", "SG64", "SGh", "SGs", "SR", "SD", "SDp", "SDr", "SDw", "SDl", "SDl100", "SDl1000", "SDv", "SDu", "SDul", "SDula", "SDulag", "SDulas", "SDulagn", "SDulagq", "SDulagnq", "UD2-gate-q128", "UD2-gate-norm-q32-rope", "UD2-gate-norm-q128-rope", "UD2-gate-norm-h4q8", "UD2-gate-mass-q32", "SDulc", "SDb", "SD2", "SD2c"):
        val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
        x = torch.from_numpy(np.stack([val[j * L:(j + 1) * L].astype(np.int64) for j in range(a.windows)])).cuda()
        start = torch.zeros_like(x, dtype=torch.bool)
        start[:, 1:] = x[:, :-1] == 1                       # the token right after a BOS opens a new document
        rec = []
        for layer in net.layers:
            f = layer.log_decay
            layer.log_decay = (lambda f: lambda *a: (rec.append(f(*a).detach()), rec[-1])[1])(f)
        with torch.no_grad():
            for i in range(0, a.windows, 8):
                net(x[i:i + 8], x[i:i + 8])
        n = len(net.layers)
        for li, layer in enumerate(net.layers):
            forget = -torch.cat(rec[li::n], 0).float()       # (windows, L, m)
            per_tok = forget.mean(-1)                        # (windows, L)
            s = out["layers"][li]
            s["horizon_effective"] = 1 / per_tok.mean().item()
            s["horizon_token_p10"] = 1 / q(per_tok, .9)
            s["horizon_token_p90"] = 1 / q(per_tok, .1)
            s["forget_doc_start"] = per_tok[start].mean().item() if start.any() else None
            s["forget_other"] = per_tok[~start].mean().item()
            if meta["variant"] in ("SG", "SG64", "SGh", "SGs"):
                s["beta"] = layer.log_beta.exp().item()
        out["doc_starts"] = int(start.sum())
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "gates.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
