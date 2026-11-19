"""
Exploratory follow-ups to babylm-plan.md section 10 (CPU, dense memory, reuses diag_length.py):

tau    Why does S's context share grow less than P-free's beyond 512? On the 2048-token windows, with each
       block's real inputs, recompute the block's context share with its own time constant and with the other
       model's time constant at the same depth (the swap changes only that number; the stream that flows on is
       the normal one). Also the address-weighted evidence sum r E / sum r, whose growth the time constant sets.
intrinsic  The same, but the stream that flows on is the clamped one, so each block sees inputs near the trained
       range: its share growth without the part that upstream blocks pass down.
long   Does S keep holding further out? Per-position NLL on 4096- and 8192-token windows, normal and clamp.

$ python scripts/babylm/explore_length.py --s_dir=data/babylm10m/runs/babylm_s --p_dir=data/babylm10m/runs/babylm_p --out=x.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
import diag_length as dl  # noqa: E402


def share_and_evidence(r, w, g):
    T = r.size(1)
    t = torch.arange(T)
    lag = (t[:, None] - t[None, :]).clamp(min=0).to(r.dtype)
    D = (t[:, None] >= t[None, :]).to(r.dtype) * torch.exp(g.to(r.dtype) * lag)
    E = D @ F.pad(r, (0, 0, 1, 0))[:, :-1]
    mass = r.sum(-1).clamp_min(1e-12)
    return (r * E / (w + E)).sum(-1) / mass, (r * E).sum(-1) / mass


@torch.no_grad()
def tau_swap(nets, xy, batch, stream="normal"):
    """Per model and block: share and evidence with its own and the other model's time constant, by bucket.
    stream="clamp" runs every block clamped, so each block's inputs stay near the trained range, and still
    records the block's unclamped share: its own growth, without what upstream blocks pass down."""
    g = {v: [-b.log_rate.exp() for b in nets[v].layers] for v in nets}
    out = {}
    for v, other in (("S", "P"), ("P", "S")):
        net, acc = nets[v], torch.zeros(len(nets[v].layers), 4, dl.L, dtype=torch.float64)
        for i in range(0, xy.size(0), batch):
            h = net.tok_embeddings(xy[i:i + batch, :-1])
            for j, blk in enumerate(net.layers):
                a = blk.norm(h)
                u = blk.conv(a.transpose(1, 2))[..., : a.size(1)].transpose(1, 2)
                if v == "S":
                    r, w = blk.code(u), blk._w()
                    q, fast, _ = dl.memory(r, blk.val(a), g[v][j], w, stream)
                    static = blk.mu(q * w)
                else:
                    static = blk.mu(blk._code(blk.key_s, blk.log_eps_s, u))
                    r, w = blk._code(blk.key_f, blk.log_eps_f, u), blk.log_w.clamp(-7.0, 12.0).exp()
                    _, fast, _ = dl.memory(r, blk.val(a), g[v][j], w, stream)
                s0, e0 = share_and_evidence(r, w, g[v][j])
                s1, e1 = share_and_evidence(r, w, g[other][j])
                acc[j] += torch.stack([s0, e0, s1, e1]).sum(1).double()
                h = h + static + fast
        acc /= xy.size(0)
        names = ["share_own_tau", "evidence_own_tau", "share_other_tau", "evidence_other_tau"]
        out[v] = {n: dl.by_bucket(acc[:, k]) for k, n in enumerate(names)}
        out[v]["tau_own"] = [round(-1 / x.item(), 1) for x in g[v]]
        out[v]["tau_other"] = [round(-1 / x.item(), 1) for x in g[other]]
    return out


def set_length(L):
    dl.L = L
    dl.EDGES = [e for e in [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192] if e <= L]
    dl.BUCKETS = [f"{a}-{b}" for a, b in zip(dl.EDGES, dl.EDGES[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_dir", required=True)
    ap.add_argument("--p_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="tau,long")
    ap.add_argument("--lengths", default="4096,8192")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    nets = {"S": dl.load(a.s_dir, "S"), "P": dl.load(a.p_dir, "P")}
    res = {}
    for part, stream in (("tau", "normal"), ("intrinsic", "clamp")):
        if part not in a.parts.split(","):
            continue
        set_length(2048)
        t0 = time.time()
        res[part] = tau_swap(nets, dl.windows(145), 5, stream)
        for v in ("S", "P"):
            r = res[part][v]
            for k in ("share_own_tau", "share_other_tau", "evidence_own_tau", "evidence_other_tau"):
                ratio = [x / y for x, y in zip(r[k]["1024-2048"], r[k]["256-512"])]
                print(part, v, k, "growth", [round(x, 2) for x in ratio], flush=True)
        print(f"{part} {time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    if "long" in a.parts.split(","):
        for L in map(int, a.lengths.split(",")):
            set_length(L)
            xy = dl.windows(10 ** 6)
            for v in ("S", "P"):
                for mode in ("normal", "clamp"):
                    t0 = time.time()
                    r = dl.evaluate(nets[v], v, xy, mode, set(range(12)), max(1, 4096 // L))
                    res.setdefault("long", {}).setdefault(str(L), {"windows": xy.size(0)}).setdefault(v, {})[mode] = r
                    print(L, v, mode, json.dumps({b: round(x, 4) for b, x in r["nll"].items()}), f"{time.time() - t0:.0f}s", flush=True)
                    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
