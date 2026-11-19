"""Text samples from one FineWeb-Edu checkpoint, to show what its outputs look like. One model per process, because
variants.install and HEAD_DIM are module-global. Temperature 0.8, top-k 40, seed 0, one draw per case. The long
cases take a document from the validation shard and continue it after 900 tokens (inside the 1024 training length)
and after 2000 tokens (past it). Not evidence about which model is better; the losses are. Run from the repo root:

$ python scripts/fineweb/samples_fineweb.py --out_dir=<run>     # prints one JSON line
$ python scripts/fineweb/samples_fineweb.py --out_dir=<run> --window   # T: each step sees only the last L tokens
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev     # noqa: E402
import variants              # noqa: E402
from tokenizer import Tokenizer  # noqa: E402

PROMPTS = ["Photosynthesis is the process by which", "The French Revolution began in",
           "To find the area of a circle, you", "Children learn to read by"]
N_GEN, CUT = 60, 2000


@torch.no_grad()
def gen(net, ctx, n, device, temp=0.8, top_k=40, window=0):
    g = torch.Generator(device=device).manual_seed(0)
    idx = torch.tensor([ctx], dtype=torch.long, device=device)
    for _ in range(n):
        logits = net(idx[:, -window:] if window else idx)[:, -1, :].float() / temp      # whole context unless --window
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
        idx = torch.cat([idx, nxt], 1)
    return idx[0, len(ctx):].tolist()


def long_doc(val):
    """First document in the validation shard with at least CUT + N_GEN tokens after its BOS."""
    bos = np.flatnonzero(val[:5_000_000] == 1)
    for a, b in zip(bos, bos[1:]):
        if b - a > CUT + N_GEN + 1:
            return int(a)
    raise SystemExit("no long document")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--window", action="store_true", help="crop the context to the training length L")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    variants.HEAD_DIM = meta.get("head_dim", 48)
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, device)
    W = ckpt["model_args"]["max_seq_len"] if a.window else 0
    enc = Tokenizer()
    dec = lambda ids: enc.decode([int(i) for i in ids])
    out = {"run": os.path.basename(a.out_dir.rstrip("/")), "variant": meta["variant"], "window": W, "prompts": []}
    for p in PROMPTS:
        out["prompts"].append({"prompt": p, "text": dec(gen(net, enc.encode(p, bos=True, eos=False), N_GEN, device, window=W))})
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    s = long_doc(val)
    doc = val[s:s + CUT + N_GEN].astype(np.int64).tolist()
    out["long"] = {"doc_start": s, "context_tail": dec(doc[CUT - 120:CUT]), "truth": dec(doc[CUT:CUT + N_GEN]),
                   "after_900": dec(gen(net, doc[CUT - 900:CUT], N_GEN, device, window=W)),
                   "after_2000": dec(gen(net, doc[:CUT], N_GEN, device, window=W))}
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
