"""How much does a model get from far context? Per-token NLL on the tokgap.py windows of L, with the full window and
with the context cut to the most recent w + 1 to 2 w tokens: the window is split into blocks of w, and the targets
of block b are scored on a sequence of their own made of blocks b - 1 and b, so no state crosses from earlier text.
Targets before position 2 w see the same context either way and are left out. Classes (bigram / seen / new, and new
tokens in the 100 most frequent) are taken on the full window, as in tokgap.py, so a new token is new under both.
Per w and group: every model's NLL full and cut, and cut - full, what the text beyond the last w + 1 to 2 w tokens
gives it. Prints JSON and writes reach.json into --dest. Run from the repo root:

$ python scripts/fineweb/reach.py --runs <run> [<run> ...] --dest=<dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import tokgap  # noqa: E402  sets the other paths, and binds delta before T replaces model.UnifiedBlock

CUTS = (64, 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.runs[0], "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    shards = sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))
    L = torch.load(os.path.join(a.runs[0], "ckpt.pt"), map_location="cpu")["model_args"]["max_seq_len"]
    val = np.memmap(shards[-1], dtype=np.uint16, mode="r")
    x = np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64) for j in range(a.windows)])
    rank = np.empty(65536, dtype=np.int64)
    rank[np.argsort(-np.bincount(np.memmap(shards[0], dtype=np.uint16, mode="r"), minlength=65536), kind="stable")] = np.arange(65536)
    c = tokgap.classes(x)
    groups = {"all": np.ones_like(c, dtype=bool), "bigram": c == 0, "seen": c == 1, "new": c == 2,
              "new_top100": (c == 2) & (rank[x[:, 1:]] < 100)}
    pos = np.zeros(c.shape, dtype=np.int64) + np.arange(L)
    full, cut, models = {}, {w: {} for w in CUTS}, {}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        models[name], full[name] = tokgap.per_token(run, x, a.device)
        for w in CUTS:
            nb = L // w - 1                                     # blocks 1 .. L / w - 1, each with the block before it
            seq = np.stack([x[:, (b - 1) * w:(b + 1) * w + 1] for b in range(1, nb + 1)], 1).reshape(-1, 2 * w + 1)
            _, z = tokgap.per_token(run, seq, a.device)
            cut[w][name] = np.concatenate([full[name][:, :w], z[:, w:].reshape(len(x), nb * w)], 1)
    out = {"windows": a.windows, "L": L, "models": models, "cuts": {}}
    for w in CUTS:
        e = out["cuts"][str(w)] = {"context": [w + 1, 2 * w], "from_position": 2 * w, "groups": {}}
        for g, m in groups.items():
            m = m & (pos >= 2 * w)
            e["groups"][g] = {"tokens": int(m.sum()),
                              "full": {k: float(z[m].mean()) for k, z in full.items()},
                              "cut": {k: float(z[m].mean()) for k, z in cut[w].items()},
                              "cut_minus_full": {k: float((cut[w][k] - full[k])[m].mean()) for k in full}}
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "reach.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
