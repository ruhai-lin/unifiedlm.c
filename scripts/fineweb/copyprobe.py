"""How well does a model copy from its context? Three probes, scored as mean NLL per target (nats):
- random: n random tokens (uniform over vocabulary ranks 100 to 10,000 of the first training shard, so neither
  trivial nor rare), then the same n tokens again. The second copy's first token has nothing pointing at it and is
  left out. A perfect copier scores about 0 on the copy; the first occurrence (about ln 9,900 = 9.2) is the
  no-copy reference. n = 32, 128, 512.
- text: 512 tokens of validation text, then the same 512 again: the copy against the first occurrence, which the
  model can only predict from what it knows of language.
- gap: 128 tokens of text A, G tokens of other validation text, then A again; the repeat of A at distance G + 128,
  for G = 0, 256, 768.
Per model and probe: mean NLL on the first occurrence and on the copy, and on the copy by quarter (does copying lock
in as more of the copy has matched?). Prints JSON and writes copyprobe.json into --dest. Run from the repo root:

$ python scripts/fineweb/copyprobe.py --runs <run> [<run> ...] --dest=<dir>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import tokgap  # noqa: E402  sets the other paths, and binds delta before T replaces model.UnifiedBlock


def probes(val, rank, n_seq, rng):
    """name -> (sequences (n_seq, len), first-occurrence target slice, copy target slice); target t predicts x[t + 1]."""
    pool = np.flatnonzero((rank >= 100) & (rank < 10000))
    win = lambda j, w: val[j * w:(j + 1) * w].astype(np.int64)  # noqa: E731
    out = {}
    for n in (32, 128, 512):
        a = rng.choice(pool, size=(n_seq, n))
        out[f"random_{n}"] = (np.concatenate([a, a], 1), slice(0, n - 1), slice(n, 2 * n - 1))
    c = np.stack([win(j, 512) for j in range(n_seq)])
    out["text_512"] = (np.concatenate([c, c], 1), slice(0, 511), slice(512, 1023))
    for g in (0, 256, 768):
        a = np.stack([win(j, 128) for j in range(n_seq)])
        f = np.stack([win(1000 + j, 768)[:g] for j in range(n_seq)])
        out[f"gap_{g}"] = (np.concatenate([a, f, a], 1), slice(0, 127), slice(128 + g, 255 + g))
    return out


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
    val = np.memmap(shards[-1], dtype=np.uint16, mode="r")
    rank = np.empty(65536, dtype=np.int64)
    rank[np.argsort(-np.bincount(np.memmap(shards[0], dtype=np.uint16, mode="r"), minlength=65536), kind="stable")] = np.arange(65536)
    rank = rank[:32000]                                          # the llama2 vocabulary
    P = probes(val, rank, a.seqs, np.random.default_rng(0))
    out = {"seqs": a.seqs, "models": {}, "probes": {k: {} for k in P}}
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        for k, (x, first, copy) in P.items():
            out["models"][name], z = tokgap.per_token(run, x, a.device)
            zc = z[:, copy]
            q = np.array_split(np.arange(zc.shape[1]), 4)
            out["probes"][k][name] = {"first": float(z[:, first].mean()), "copy": float(zc.mean()),
                                      "copy_by_quarter": [float(zc[:, i].mean()) for i in q]}
            print(f"{name:32s} {k:11s} first {z[:, first].mean():6.3f} copy {zc.mean():6.3f}  quarters "
                  + " ".join(f"{zc[:, i].mean():.3f}" for i in q), flush=True)
    print(json.dumps(out), flush=True)
    os.makedirs(a.dest, exist_ok=True)
    json.dump(out, open(os.path.join(a.dest, "copyprobe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
