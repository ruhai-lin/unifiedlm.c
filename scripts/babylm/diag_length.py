"""
Why does P-free collapse beyond its training length on BabyLM while S holds (babylm-plan.md section 9)?

Reruns the saved S and P-free models on the 2048-token validation windows with the dense form of the context
memory, layer by layer, so the memory can be changed at test time:
  normal   the model as trained
  window   each slot's memory only holds the last 512 writes (the training length)
  clamp    all writes are kept, but each slot's evidence E is capped at the largest value it reached before
           position 512 in that window, with its stored values scaled by the same factor, so the read keeps the
           same average content and only its weight against the prior mass w is held to the trained range
Either change can be applied to every block or to one block. Both leave positions before 512 untouched.
Also records, per block and position range, the context share of the read mass, sum r E/(w+E) / sum r, and the
RMS of the block's two read terms. CPU only.

$ python scripts/babylm/diag_length.py --s_dir=data/babylm10m/runs/babylm_s --p_dir=data/babylm10m/runs/babylm_p \
      --s_eval=scripts/reports/babylm/babylm_s/eval.json --p_eval=scripts/reports/babylm/babylm_p/eval.json --out=diag.json
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from dense_memory import dense_memory  # noqa: E402
sys.modules["kernel"] = types.SimpleNamespace(CHUNK=64, unified_memory=dense_memory)
import variants                         # noqa: E402
from model import ModelArgs, UnifiedLM  # noqa: E402

L, TRAINED = 2048, 512
EDGES = [0, 64, 128, 256, 512, 1024, 2048]
BUCKETS = [f"{a}-{b}" for a, b in zip(EDGES, EDGES[1:])]


def load(run_dir, variant):
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location="cpu")
    variants.install(variant, 712 if variant == "P" else 0, 712 if variant == "P" else 0)
    net = UnifiedLM(ModelArgs(**ckpt["model_args"]))
    net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    return net.eval()


def memory(r, v, g, w, mode):
    """Dense context memory. Returns q = r / c (S's static read is mu(q w)), the fast read and the context share."""
    T = r.size(1)
    t = torch.arange(T)
    lag = (t[:, None] - t[None, :]).clamp(min=0).to(r.dtype)
    D = (t[:, None] >= t[None, :]).to(r.dtype) * torch.exp(g.to(r.dtype) * lag)
    if mode == "window":
        D = D * (lag < TRAINED).to(r.dtype)
    k = F.pad(r, (0, 0, 1, 0))[:, :-1]
    E = D @ k
    f = torch.ones_like(E)
    if mode == "clamp":
        f = (E[:, :TRAINED].amax(1, keepdim=True) / E.clamp_min(1e-30)).clamp(max=1.0)
    c = w + f * E
    q = r / c
    fast = (((q * f) @ k.transpose(1, 2)) * D) @ v
    share = (r * f * E / c).sum(-1) / r.sum(-1).clamp_min(1e-12)
    return q, fast, share


def rms(x):
    return x.pow(2).mean(-1).sqrt()


@torch.no_grad()
def forward(net, variant, x, mode, layers):
    """Returns logits and per block (share, rms of the static term, rms of the fast term, rms of their sum), each (B,T)."""
    h = net.tok_embeddings(x)
    stats = []
    for i, blk in enumerate(net.layers):
        m = mode if i in layers else "normal"
        a = blk.norm(h)
        u = blk.conv(a.transpose(1, 2))[..., : a.size(1)].transpose(1, 2)
        g = -blk.log_rate.exp()
        if variant == "S":
            r, w = blk.code(u), blk._w()
            q, fast, share = memory(r, blk.val(a), g, w, m)
            static = blk.mu(q * w)
        else:
            static = blk.mu(blk._code(blk.key_s, blk.log_eps_s, u))
            r, w = blk._code(blk.key_f, blk.log_eps_f, u), blk.log_w.clamp(-7.0, 12.0).exp()
            _, fast, share = memory(r, blk.val(a), g, w, m)
        h = h + static + fast
        stats.append(torch.stack([share, rms(static), rms(fast), rms(static + fast)]))
    return net.output(net.norm(h)), torch.stack(stats)  # stats: (blocks, 4, B, T)


def windows(n):
    m = np.memmap("data/babylm10m/val.bin", dtype=np.uint16, mode="r")
    n = min(n, (len(m) - 1) // L)
    return torch.from_numpy(np.stack([m[j * L: (j + 1) * L + 1].astype(np.int64) for j in range(n)]))


def by_bucket(per_position):
    return {b: per_position[..., a:e].mean(-1).tolist() for b, a, e in zip(BUCKETS, EDGES, EDGES[1:])}


def evaluate(net, variant, xy, mode, layers, batch):
    nll = torch.zeros(L, dtype=torch.float64)
    stats = torch.zeros(len(net.layers), 4, L, dtype=torch.float64)
    for i in range(0, xy.size(0), batch):
        x, y = xy[i:i + batch, :-1], xy[i:i + batch, 1:]
        logits, st = forward(net, variant, x, mode, layers)
        nll += F.cross_entropy(logits.transpose(1, 2), y, reduction="none").sum(0).double()
        stats += st.sum(2).double()
    nll, stats = nll / xy.size(0), stats / xy.size(0)
    names = ["context_share", "rms_static", "rms_fast", "rms_block"]
    return {"nll": by_bucket(nll), **{n: by_bucket(stats[:, j]) for j, n in enumerate(names)}}


def check(net, variant, xy):
    """The layer-by-layer form must reproduce the model's own forward."""
    x = xy[:2, :-1].contiguous()
    with torch.no_grad():
        ref = net(x, xy[:2, 1:].contiguous())
        mine, _ = forward(net, variant, x, "normal", set())
    rel = ((ref - mine).abs().max() / ref.abs().max()).item()
    assert rel < 1e-5, f"{variant}: layer-by-layer form differs from the model by {rel:.1e} of the largest logit"
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_dir", required=True)
    ap.add_argument("--p_dir", required=True)
    ap.add_argument("--s_eval", required=True)
    ap.add_argument("--p_eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=145, help="2048-token windows (145 is all of them)")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--per_layer", type=int, default=1, help="also apply window to one P-free block at a time")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    xy = windows(a.n)
    res = {"windows": xy.size(0), "time_constant": {}, "check_rel_err": {}, "recorded": {}, "S": {}, "P": {}}
    runs = [("S", m, "all") for m in ("normal", "window", "clamp")] + [("P", m, "all") for m in ("normal", "window", "clamp")]
    if a.per_layer:
        runs += [("P", "window", str(i)) for i in range(12)]
    nets = {"S": load(a.s_dir, "S"), "P": load(a.p_dir, "P")}
    for v, path in (("S", a.s_eval), ("P", a.p_eval)):
        res["time_constant"][v] = [round(1 / b.log_rate.exp().item(), 1) for b in nets[v].layers]
        res["check_rel_err"][v] = check(nets[v], v, xy)
        res["recorded"][v] = json.load(open(path))["val"][str(L)]["by_position"]
        print(v, "time constants", res["time_constant"][v], "check", f"{res['check_rel_err'][v]:.1e}", flush=True)
    for v, mode, which in runs:
        t0 = time.time()
        layers = set(range(12)) if which == "all" else {int(which)}
        r = evaluate(nets[v], v, xy, mode, layers, a.batch)
        res[v].setdefault(mode, {})[which] = r
        print(v, mode, which, json.dumps({b: round(x, 4) for b, x in r["nll"].items()}), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
