"""Where does a model lose to a reference (e.g. SDu 42M against T 42M)? Per-token NLL of every model on the same
windows of L from the FineWeb validation shard, each target x_{t+1} classed by what the window before it holds:
    bigram  the pair (x_t, x_{t+1}) occurred earlier in the window (a copy / induction target)
    seen    x_{t+1} occurred earlier in the window, but not after x_t
    new     x_{t+1} has not occurred in the window
and by its frequency rank in the first training shard (top 100, 100-1K, 1K-10K, rest), and the class by position.
Per group: token count, mean NLL of every model, and each model's mean difference from --ref with the share of the
total difference the group carries. Prints JSON and writes tokgap.json into --dest. Run from the repo root:

$ python scripts/fineweb/tokgap.py --ref=<run> --runs <run> [<run> ...] --dest=<dir>
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
import variants            # noqa: E402
import delta               # noqa: E402,F401  gated.py subclasses model.UnifiedBlock on import: bind it before T replaces it

POS = [(0, 64), (64, 256), (256, 1024)]


def per_token(run, x, device):
    meta = json.load(open(os.path.join(run, "variant.json")))
    variants.HEAD_DIM = meta.get("head_dim", 48)
    net, meta, ckpt = ev.load(run, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            xy = torch.from_numpy(x[i:i + 8]).to(device)
            logits = net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous())
            out.append(F.cross_entropy(logits.transpose(1, 2), xy[:, 1:], reduction="none").double().cpu().numpy())
    del net
    torch.cuda.empty_cache()
    return meta["variant"], np.concatenate(out)


def classes(x):
    """(windows, L) class of each target: 0 bigram, 1 seen, 2 new."""
    c = np.empty((x.shape[0], x.shape[1] - 1), dtype=np.int8)
    for w, row in enumerate(x.tolist()):
        seen, pairs = {row[0]}, set()
        for t in range(len(row) - 1):
            p, y = row[t], row[t + 1]
            c[w, t] = 0 if (p, y) in pairs else 1 if y in seen else 2
            seen.add(y)
            pairs.add((p, y))
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.ref, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    shards = sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))
    L = torch.load(os.path.join(a.ref, "ckpt.pt"), map_location="cpu")["model_args"]["max_seq_len"]
    val = np.memmap(shards[-1], dtype=np.uint16, mode="r")
    x = np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64) for j in range(a.windows)])
    rank = np.empty(65536, dtype=np.int64)
    rank[np.argsort(-np.bincount(np.memmap(shards[0], dtype=np.uint16, mode="r"), minlength=65536), kind="stable")] = np.arange(65536)
    r = rank[x[:, 1:]]
    groups = {"all": np.ones_like(r, dtype=bool)}
    for ci, cn in enumerate(("bigram", "seen", "new")):
        groups[cn] = classes(x) == ci
    for lo, hi, name in ((0, 100, "top100"), (100, 1000, "100-1K"), (1000, 10000, "1K-10K"), (10000, 1 << 20, "rest")):
        groups[f"freq:{name}"] = (r >= lo) & (r < hi)
    pos = np.zeros_like(r, dtype=np.int64) + np.arange(r.shape[1])
    for cn in ("bigram", "seen", "new"):
        for lo, hi in POS:
            groups[f"{cn}@{lo}-{hi}"] = groups[cn] & (pos >= lo) & (pos < hi)
    nll = {}
    for run in [a.ref] + a.runs:
        v, z = per_token(run, x, a.device)
        nll[os.path.basename(run.rstrip("/"))] = (v, z)
    ref = os.path.basename(a.ref.rstrip("/"))
    out = {"ref": ref, "windows": a.windows, "L": L, "models": {k: v for k, (v, _) in nll.items()}, "groups": {}}
    for g, m in groups.items():
        e = {"tokens": int(m.sum()), "share_of_tokens": float(m.mean()), "nll": {}, "minus_ref": {}, "share_of_gap": {}}
        for k, (_, z) in nll.items():
            e["nll"][k] = float(z[m].mean())
            if k != ref:
                d = z - nll[ref][1]
                e["minus_ref"][k] = float(d[m].mean())
                e["share_of_gap"][k] = float(d[m].sum() / d.sum())
        out["groups"][g] = e
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "tokgap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
