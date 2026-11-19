"""How much does an SD-family model lean on its memory? NLL by position on the FineWeb validation shard with the memory
read E^T q of every layer zeroed at inference (the block then outputs its FFN half alone), and with one layer's
memory zeroed at a time. The model was trained with its memory, so this measures reliance, not what a model trained
without memory could do. Windows of L and 4 L (the same windows for every model, so differences are paired). Prints
JSON and writes memablate.json into --dest (default: the run directory). Run from the repo root:

$ python scripts/fineweb/memablate.py --out_dir=<run> [--dest=<dir>]
"""
import argparse
import glob
import json
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
import delta               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=128, help="windows of L; a quarter as many of 4 L")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, a.device)
    L = ckpt["model_args"]["max_seq_len"]
    n = len(net.layers)
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    f = delta.delta_memory
    off = set()
    calls = [0]

    def hook(k, u, *rest, **kw):     # called once per layer, in layer order
        li = calls[0] % n
        calls[0] += 1
        return torch.zeros_like(u) if li in off else f(k, u, *rest, **kw)
    delta.delta_memory = hook

    def nll(T, windows, batch):
        total = torch.zeros(T, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, windows, batch):
                xy = torch.from_numpy(np.stack([val[j * T:(j + 1) * T + 1].astype(np.int64)
                                                for j in range(i, min(windows, i + batch))])).to(a.device)
                calls[0] = 0
                logits = net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
                total += F.cross_entropy(logits.transpose(1, 2), xy[:, 1:], reduction="none").sum(0).double().cpu()
        per = total / windows
        edges = [e for e in (0, 64, 128, 256, 512, 1024, 2048) if e < T] + [T]
        return {"nll": per.mean().item(), "by_position": {f"{p}-{q}": per[p:q].mean().item() for p, q in zip(edges, edges[1:])}}

    out = {"variant": meta["variant"], "windows": a.windows, "layers": n}
    for T, w, b in ((L, a.windows, 8), (4 * L, a.windows // 4, 2)):
        off.clear()
        full = nll(T, w, b)
        off.update(range(n))
        none = nll(T, w, b)
        out[str(T)] = {"full": full, "no_memory": none}
    off.clear()
    full = out[str(L)]["full"]["nll"]
    out["per_layer"] = []
    for li in range(n):
        off.clear()
        off.add(li)
        r = nll(L, a.windows, 8)
        out["per_layer"].append({"layer": li, "nll": r["nll"], "delta": r["nll"] - full})
    print(json.dumps(out), flush=True)
    dest = a.dest or a.out_dir
    os.makedirs(dest, exist_ok=True)
    json.dump(out, open(os.path.join(dest, "memablate.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
