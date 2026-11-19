"""How SD-family slots lose their content: decay against delta-rule overwrite, per slot, grouped by base horizon.
The delta update E_t = (I - beta_t k_t k_t^T) Lam_t E_{t-1} + ... removes, from slot a, a share of about
beta_t k_{t,a}^2 (k unit; the diagonal of beta k k^T, ignoring the mixing between slots) on top of the decay
1 - lam_{t,a}. Averaged over tokens per slot:
    decay     d_a = mean_t -log lam_{t,a}
    overwrite o_a = mean_t -log(1 - beta_t k_{t,a}^2)
Horizons 1 / d, 1 / o and 1 / (d + o) per group of slots (by base horizon 1 / rate_a). Prints JSON and writes
overwrite.json into the run directory. Run from the repo root:

$ python scripts/fineweb/overwrite.py --out_dir=<run> [--device=cpu]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev   # noqa: E402
import delta               # noqa: E402

BINS = [(0, 1024), (1024, 4096), (4096, 8192), (8192, float("inf"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, a.device)
    L = ckpt["model_args"]["max_seq_len"]
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    x = torch.from_numpy(np.stack([val[j * L:(j + 1) * L].astype(np.int64) for j in range(a.windows)])).to(a.device)
    n = len(net.layers)
    d = [0.0] * n
    o = [0.0] * n
    calls = [0]
    f = delta.delta_memory

    def rec(k, u, beta, logl, q, *rest, **kw):     # k is already unit; called once per layer, in layer order
        li = calls[0] % n
        calls[0] += 1
        d[li] = d[li] + (-logl).sum((0, 1)).double()
        o[li] = o[li] + (-torch.log1p(-(beta[..., None] * k.square()).clamp(max=1 - 1e-6))).sum((0, 1)).double()
        return f(k, u, beta, logl, q, *rest, **kw)
    delta.delta_memory = rec
    with torch.no_grad():
        for i in range(0, a.windows, 4):
            net(x[i:i + 4], x[i:i + 4])
    T = a.windows * L
    out = {"variant": meta["variant"], "windows": a.windows, "layers": [], "all": []}
    hz_all, d_all, o_all = [], [], []
    for li, layer in enumerate(net.layers):
        hz = (1 / layer.log_rate.detach().exp().flatten()).double().cpu()
        dl, ol = (d[li] / T).cpu(), (o[li] / T).cpu()
        hz_all.append(hz); d_all.append(dl); o_all.append(ol)
        out["layers"].append({"decay_horizon": 1 / dl.mean().item(), "overwrite_horizon": 1 / ol.mean().item(),
                              "total_horizon": 1 / (dl + ol).mean().item()})
    hz, dl, ol = torch.cat(hz_all), torch.cat(d_all), torch.cat(o_all)
    for lo, hi in BINS:
        s = (hz >= lo) & (hz < hi)
        if s.any():
            out["all"].append({"base_horizon": [lo, None if hi == float("inf") else hi], "slots": int(s.sum()),
                               "decay_horizon_median": (1 / dl[s]).median().item(),
                               "overwrite_horizon_median": (1 / ol[s]).median().item(),
                               "total_horizon_median": (1 / (dl[s] + ol[s])).median().item(),
                               "overwrite_share": (ol[s] / (dl[s] + ol[s])).median().item()})
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "overwrite.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
