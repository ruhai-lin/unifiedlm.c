"""Can a model look up one item among many in its context, as attention compares a query with each past token?
In-context key-value recall in the MQAR format of Zoology (Arora et al. 2023), with no training on it:
    k_1 v_1 k_2 v_2 ... k_N v_N  then the same N pairs again in a random order,
keys distinct, keys and values drawn from disjoint halves of vocabulary ranks 100 to 10,000 of the first training
shard (as copyprobe.py's random probe). In the second part each value v follows its key k for the second time; with
the order shuffled, the only way to predict v is to find k in the first part and read what came after it.
N = 8, 16, 32, 64, 128, 256 (4N tokens, at most 1024, the training length).
Per model and N: mean NLL and top-1 accuracy on the recalled values; NLL on the values of the first part (nothing
points at them: the no-recall reference, above ln 4,950 = 8.5 since a text model is no uniform guesser); recall accuracy by distance from the first
occurrence of the pair. Prints JSON and writes kvrecall.json into --dest. Run from the repo root:

$ python scripts/fineweb/kvrecall.py --runs <run> [<run> ...] --dest=<dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import tokgap  # noqa: E402  sets the other paths, and binds delta before T replaces model.UnifiedBlock

NS = (8, 16, 32, 64, 128, 256)
DIST = [(0, 64), (64, 256), (256, 512), (512, 1024)]


def sequences(rank, N, n_seq, rng):
    """(n_seq, 4N) tokens; the recalled value of query j sits at 2N + 2j + 1, its first occurrence at 2 pi_j + 1."""
    pool = rng.permutation(np.flatnonzero((rank >= 100) & (rank < 10000)))
    kpool, vpool = pool[: len(pool) // 2], pool[len(pool) // 2:]
    x, firstpos = np.empty((n_seq, 4 * N), dtype=np.int64), np.empty((n_seq, N), dtype=np.int64)
    for i in range(n_seq):
        k, v, p = rng.choice(kpool, N, replace=False), rng.choice(vpool, N), rng.permutation(N)
        x[i, 0:2 * N:2], x[i, 1:2 * N:2] = k, v
        x[i, 2 * N::2], x[i, 2 * N + 1::2] = k[p], v[p]
        firstpos[i] = 2 * p + 1
    return x, firstpos


def score(run, x, device):
    """NLL and argmax of every target x_{t+1} (index t)."""
    meta = json.load(open(os.path.join(run, "variant.json")))
    tokgap.variants.HEAD_DIM = meta.get("head_dim", 48)
    net, meta, _ = tokgap.ev.load(run, device)
    nll, hit = [], []
    with torch.no_grad():
        for i in range(0, len(x), 8):
            xy = torch.from_numpy(x[i:i + 8]).to(device)
            logits = net(xy[:, :-1].contiguous(), xy[:, 1:].contiguous()).float()
            nll.append(F.cross_entropy(logits.transpose(1, 2), xy[:, 1:], reduction="none").double().cpu().numpy())
            hit.append((logits.argmax(-1) == xy[:, 1:]).cpu().numpy())
    del net
    return meta["variant"], np.concatenate(nll), np.concatenate(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--seqs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.runs[0], "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    shards = sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))
    rank = np.empty(65536, dtype=np.int64)
    rank[np.argsort(-np.bincount(np.memmap(shards[0], dtype=np.uint16, mode="r"), minlength=65536), kind="stable")] = np.arange(65536)
    rank = rank[:32000]                                          # the llama2 vocabulary
    rng = np.random.default_rng(0)
    P = {N: sequences(rank, N, a.seqs, rng) for N in NS}
    out = {"seqs": a.seqs, "models": {}, "N": {str(N): {} for N in NS}}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        for N, (x, firstpos) in P.items():
            out["models"][name], nll, hit = score(run, x, a.device)
            t_rec = 2 * N + 2 * np.arange(N)                     # target index t predicts x[t + 1], a recalled value
            dist = (t_rec[None, :] + 1) - firstpos               # tokens since the pair's first occurrence
            rec_nll, rec_hit = nll[:, t_rec], hit[:, t_rec]
            by_d = {}
            for lo, hi in DIST:
                m = (dist >= lo) & (dist < hi)
                if m.any():
                    by_d[f"{lo}-{hi}"] = {"acc": float(rec_hit[m].mean()), "nll": float(rec_nll[m].mean()), "n": int(m.sum())}
            out["N"][str(N)][name] = {"recall_nll": float(rec_nll.mean()), "recall_acc": float(rec_hit.mean()),
                                      "first_nll": float(nll[:, 2 * np.arange(N)].mean()), "by_distance": by_d}
            print(f"{name:32s} N {N:3d}  recall acc {rec_hit.mean():.3f} nll {rec_nll.mean():6.3f}  first nll "
                  f"{nll[:, 2 * np.arange(N)].mean():6.3f}  by distance "
                  + " ".join(f"{k}:{v['acc']:.2f}" for k, v in by_d.items()), flush=True)
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "kvrecall.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
