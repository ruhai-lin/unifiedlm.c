"""How long do the slots of an SD-family model keep what is written to them? Per layer and slot a, the share of a write
at step s still there D tokens later by decay alone, prod_{s<t<=s+D} lam_{t,a}, averaged over the writes with their
mass beta_s k_{s,a}^2 as weight, for D = 16, 64, 256. The gate makes forgetting bursty and the average forgetting rate
of gates.py is ruled by the fast slots, so this is the decay the writes actually meet. Per D: the slots that keep
more than 1/e, the share of all write mass that goes to them, and the share of all write mass that survives D tokens.
Decay only: overwriting by later writes is left out (see overwrite.py). For models with one memory per block (not SD2,
SD2c). Prints JSON and writes slotlen.json into --dest. Run from the repo root:

$ python scripts/fineweb/slotlen.py --out_dir=<run> --dest=<dir>
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
import delta               # noqa: E402

DS = (16, 64, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, a.device)
    L = ckpt["model_args"]["max_seq_len"]
    n = len(net.layers)
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    f = delta.delta_memory
    calls = [0]
    mass = [0.0] * n
    kept = [{D: [0.0, 0.0] for D in DS} for _ in range(n)]     # per layer and D: sum of w * kept, sum of w (per slot)

    def hook(k, u, beta, logl, q, *rest, **kw):     # called once per layer, in layer order; k is already unit
        li = calls[0] % n
        calls[0] += 1
        w = beta.float()[..., None] * k.float() ** 2
        C = logl.float().cumsum(1)
        mass[li] = mass[li] + w.sum((0, 1)).double().cpu()
        for D in DS:
            s = kept[li][D]
            s[0] = s[0] + (w[:, :-D] * (C[:, D:] - C[:, :-D]).exp()).sum((0, 1)).double().cpu()
            s[1] = s[1] + w[:, :-D].sum((0, 1)).double().cpu()
        return f(k, u, beta, logl, q, *rest, **kw)
    delta.delta_memory = hook
    with torch.no_grad():
        for i in range(0, a.windows, 4):
            xy = torch.from_numpy(np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64)
                                            for j in range(i, min(a.windows, i + 4))])).to(a.device)
            calls[0] = 0
            net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
    out = {"variant": meta["variant"], "windows": a.windows, "layers": []}
    for li in range(n):
        e = {"slots": int(mass[li].numel())}
        for D in DS:
            num, den = kept[li][D]
            frac = torch.where(den > 0, num / den.clamp_min(1e-30), torch.zeros_like(num))
            keep = frac > 1 / math.e
            e[str(D)] = {"slots_keep": int(keep.sum()), "write_share_keep": float(den[keep].sum() / den.sum()),
                         "mass_surviving": float(num.sum() / den.sum())}
        out["layers"].append(e)
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "slotlen.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
