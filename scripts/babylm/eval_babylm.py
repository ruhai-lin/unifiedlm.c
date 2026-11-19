"""
Score one BabyLM run in fp32 (scripts/reports/babylm-plan.md): validation NLL by position at the training length
and at 4x it, and BLiMP accuracy. Run from the repository root:

$ python scripts/babylm/eval_babylm.py --out_dir=runs/x

Reads out_dir/ckpt.pt (the final state) and out_dir/variant.json, writes out_dir/eval.json. The validation windows
are the held-out text cut in order into non-overlapping windows, so every run is scored on the same tokens.
The Transformer is not run beyond its training length (its positions there were never trained).
Without CUDA the dense reference stands in for the Triton kernel (slow; for smoke tests).
"""
import argparse
import json
import os
import sys
import types
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
if not torch.cuda.is_available():
    from dense_memory import dense_memory
    sys.modules["kernel"] = types.SimpleNamespace(CHUNK=64, unified_memory=dense_memory)
import variants                         # noqa: E402
from model import ModelArgs, UnifiedLM  # noqa: E402

DATA_DIR = "data/babylm10m"


def load(out_dir, device):
    meta = json.load(open(os.path.join(out_dir, "variant.json")))
    variants.install(meta["variant"], meta["m_s"], meta["m_f"])
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    net = UnifiedLM(ModelArgs(**ckpt["model_args"]))
    net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    return net.to(device).eval(), meta, ckpt


@torch.no_grad()
def nll_by_position(net, L, batch, device, max_windows):
    m = np.memmap(os.path.join(DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    n = min((len(m) - 1) // L, max_windows)
    total = torch.zeros(L, dtype=torch.float64)
    for i in range(0, n, batch):
        rows = [m[j * L: (j + 1) * L + 1].astype(np.int64) for j in range(i, min(n, i + batch))]
        xy = torch.from_numpy(np.stack(rows)).to(device)
        x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
        logits = net(x, y)
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        total += nll.sum(0).double().cpu()
    per = total / n
    edges = [e for e in (0, 64, 128, 256, 512, 1024, 2048) if e < L] + [L]
    return {"windows": n, "tokens": n * L, "nll": per.mean().item(),
            "by_position": {f"{a}-{b}": per[a:b].mean().item() for a, b in zip(edges, edges[1:])}}


@torch.no_grad()
def blimp(net, device, batch, limit):
    pairs = json.load(open(os.path.join(DATA_DIR, "blimp.json")))[:limit]
    sents = [p[k] for p in pairs for k in ("good", "bad")]
    scores = torch.empty(len(sents), dtype=torch.float64)
    order = sorted(range(len(sents)), key=lambda i: len(sents[i]))  # similar lengths share a batch
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        n = max(len(sents[j]) for j in idx)
        T = -(-n // 64) * 64  # the kernel works in chunks of 64; padding after a sentence cannot affect it
        x = torch.full((len(idx), T), 2, dtype=torch.long)
        y = torch.full((len(idx), T), -1, dtype=torch.long)
        for r, j in enumerate(idx):
            s = torch.tensor(sents[j])
            x[r, :len(s)] = s
            y[r, :len(s) - 1] = s[1:]
        x, y = x.to(device), y.to(device)
        logits = net(x, y)
        lp = -F.cross_entropy(logits.transpose(1, 2), y, reduction="none", ignore_index=-1)
        scores[idx] = lp.sum(1).double().cpu()
    right = (scores[0::2] > scores[1::2]).tolist()
    groups = defaultdict(list)
    for p, ok in zip(pairs, right):
        for key in ("uid", "term", "field"):
            groups[(key, p[key])].append(ok)
    acc = lambda key: {name: sum(v) / len(v) for (k, name), v in sorted(groups.items()) if k == key}
    return {"pairs": len(pairs), "accuracy": sum(right) / len(right),
            "by_field": acc("field"), "by_term": acc("term"), "by_paradigm": acc("uid")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_windows", type=int, default=10**9, help="smoke tests only")
    ap.add_argument("--blimp_limit", type=int, default=10**9, help="smoke tests only")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    net, meta, ckpt = load(a.out_dir, device)
    L = ckpt["model_args"]["max_seq_len"]
    res = {"variant": meta["variant"], "m_s": meta["m_s"], "m_f": meta["m_f"], "seed": ckpt["config"]["seed"],
           "iter_num": ckpt["iter_num"], "params": sum(p.numel() for p in net.parameters()),
           "val": {str(L): nll_by_position(net, L, a.batch, device, a.max_windows)}}
    print(json.dumps(res), flush=True)
    if meta["variant"] != "T":
        res["val"][str(4 * L)] = nll_by_position(net, 4 * L, max(1, a.batch // 4), device, a.max_windows)
        print(json.dumps(res["val"][str(4 * L)]), flush=True)
    res["blimp"] = blimp(net, device, 4 * a.batch, a.blimp_limit)
    print("blimp", res["blimp"]["accuracy"], json.dumps(res["blimp"]["by_field"]), flush=True)
    json.dump(res, open(os.path.join(a.out_dir, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
