"""How strongly does an SD-family block read its memory, by token class and position? Per layer and token: the memory
read |E^T q| (before any read gate or norm; all heads together for -h4q8, without the mass channel for -mass), the
read gate (SDulag, SDulagn: o_t = 2 sigmoid(w . h_t + b); SDulas: its one scale; else 1), what the memory actually adds to the block output (out: the read after its gate and norm), and the
static half |mu z| for scale. Grouped as in tokgap.py by the class of the token the position predicts (bigram /
seen / new) and by position (0-64, 64-256, 256-1024). Prints JSON and writes readstats.json into --dest. Run from the
repo root:

$ python scripts/fineweb/readstats.py --out_dir=<run> --dest=<dir> [--device=cpu]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import tokgap  # noqa: E402  sets the other paths
import delta   # noqa: E402

POS = [(0, 64), (64, 256), (256, 1024)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = tokgap.ev.load(a.out_dir, a.device)
    L = ckpt["model_args"]["max_seq_len"]
    n = len(net.layers)
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    x = np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64) for j in range(a.windows)])
    c = tokgap.classes(x)
    pos = np.zeros(c.shape, dtype=np.int64) + np.arange(L)
    f = delta.delta_memory
    rec = {k: [[] for _ in range(n)] for k in ("read", "gate", "out", "ffn")}
    calls = [0]

    def hook(*args, **kw):          # called once per layer, in layer order
        o = f(*args, **kw)
        layer, r = net.layers[calls[0] % n], o
        if isinstance(layer, delta.MultiHeadReadNormGateAnchorDeltaBlock):   # -h4q8: heads folded into the batch
            r = o.unflatten(0, (-1, layer.heads)).transpose(1, 2).flatten(-2)
        elif isinstance(layer, delta.MassQueryGateAnchorDeltaBlock):         # -mass: the last channel is the mass
            r = o[..., :-1]
        rec["read"][calls[0] % n].append(r.norm(dim=-1).float().cpu())
        calls[0] += 1
        return o
    delta.delta_memory = hook

    def gate(m, i, li):
        h = m.norm(i[0])
        g = 2 * torch.sigmoid(m.read(h)).squeeze(-1) if hasattr(m, "read") else (2 * torch.sigmoid(m.read_b)).expand(h.shape[:2])
        rec["gate"][li].append(g.float().cpu())

    def wrap(read_out, li):
        def g(o, h):
            y = read_out(o, h)
            rec["out"][li].append(y.norm(dim=-1).float().cpu())
            return y
        return g
    for li, layer in enumerate(net.layers):
        layer.mu.register_forward_hook(lambda m, i, o, li=li: rec["ffn"][li].append(o.norm(dim=-1).float().cpu()))
        if hasattr(layer, "read_out"):
            layer.register_forward_pre_hook(lambda m, i, li=li: gate(m, i, li))
            layer.read_out = wrap(layer.read_out, li)
    with torch.no_grad():
        for i in range(0, a.windows, 4):
            xy = torch.from_numpy(x[i:i + 4]).to(a.device)
            calls[0] = 0
            net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
    groups = {"all": np.ones_like(c, dtype=bool)}
    for ci, cn in enumerate(("bigram", "seen", "new")):
        groups[cn] = c == ci
    for lo, hi in POS:
        groups[f"pos{lo}-{hi}"] = (pos >= lo) & (pos < hi)
        for ci, cn in enumerate(("bigram", "seen", "new")):
            groups[f"{cn}@{lo}-{hi}"] = (c == ci) & (pos >= lo) & (pos < hi)
    out = {"variant": meta["variant"], "windows": a.windows, "layers": []}
    for li in range(n):
        r = torch.cat(rec["read"][li]).numpy()
        z = torch.cat(rec["ffn"][li]).numpy()
        g = torch.cat(rec["gate"][li]).numpy() if rec["gate"][li] else np.ones_like(r)
        o = torch.cat(rec["out"][li]).numpy() if rec["out"][li] else r
        out["layers"].append({k: {"read": float(r[m].mean()), "gate": float(g[m].mean()), "out": float(o[m].mean()),
                                  "ffn": float(z[m].mean())} for k, m in groups.items()})
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "readstats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
