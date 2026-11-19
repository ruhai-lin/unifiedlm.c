"""
Why does S collapse on the interfering distance task at gap 1024 while P-free holds? (memory-plan.md section 12)

Reruns the saved S and P models token by token, with the fast memory split into the writes that involve a filler
token and the rest, so filler writes can be removed from the static attenuation, from the fast read, or from both.
CPU only, standard library plus torch.

$ python scripts/memory/diag_collapse.py --s_dir=<run with model.pt> --p_dir=<run with model.pt> --out=diag.json
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mqar  # noqa: E402  (installs the CPU memory reference as `kernel`)

V, D_MODEL, N, SEQ = 8192, 32, 16, 320
THIRD = V // 3


def load(kind, run_dir):
    net, m = mqar.build(kind, D_MODEL, V, SEQ)
    net.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu"))
    return net.eval()


def batches(G, n=1024, eb=32):
    """The test set of mqar.evaluate: the same generator seed and the same batch draws."""
    g = torch.Generator().manual_seed(2 * 100003 + G)
    xs, ys = [], []
    for _ in range(n // eb):
        x, y, pos, _ = mqar.m2_batch(eb, N, G, mqar.round_up(4 * N + G), V, g, "cpu", "kv")
        xs.append(x); ys.append(y)
    return torch.cat(xs), torch.cat(ys), pos


def memory(r, v, lam, w, filler, mode):
    """
    Token-by-token unified memory with the writes split. r: (B,T,m) read address, v: (B,T,d), filler: (T,) bool,
    true where the write (key r_{t-1}, value v_t) involves a filler token. Returns the static read weight r*w/c_static
    (B,T,m), the fast read (B,T,d) and the static share sum(r w/c)/sum(r) per position (B,T).
      normal: c = w + E_all, fast = N_all r / c
      A:      filler never written: static and fast both use the non-filler writes
      B:      filler written to the fast read, but not counted in the evidence that attenuates the static read
      C:      filler counted in the evidence that attenuates the static read, but not written to the fast read
    """
    B, T, m = r.shape
    d = v.size(-1)
    E_all = torch.zeros(B, m); E_keep = torch.zeros(B, m)
    N_all = torch.zeros(B, d, m); N_keep = torch.zeros(B, d, m)
    r_prev = torch.zeros(B, m)
    qw, fast, share = torch.empty(B, T, m), torch.empty(B, T, d), torch.empty(B, T)
    for t in range(T):
        k, vt = r_prev, v[:, t]
        E_all = lam * E_all + k
        N_all = lam * N_all + vt[:, :, None] * k[:, None, :]
        if not filler[t]:
            E_keep = lam * E_keep + k
            N_keep = lam * N_keep + vt[:, :, None] * k[:, None, :]
        else:
            E_keep = lam * E_keep
            N_keep = lam * N_keep
        rt = r[:, t]
        c_static = w + (E_keep if mode in ("A", "B") else E_all)
        c_fast, N_fast = (w + E_keep, N_keep) if mode in ("A", "C") else (w + E_all, N_all)
        qw[:, t] = rt * w / c_static
        fast[:, t] = torch.einsum("bdm,bm->bd", N_fast, rt / c_fast)
        share[:, t] = qw[:, t].sum(-1) / rt.sum(-1).clamp_min(1e-12)
        r_prev = rt
    return qw, fast, share


def conv(block, h):
    return block.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)


@torch.no_grad()
def run(net, kind, x, filler, mode):
    """Logits-free forward to the final norm; returns hidden states and, for S, the per-layer static share."""
    h = net.tok_embeddings(x)
    shares = []
    for blk in net.layers:
        a = blk.norm(h)
        u = conv(blk, a)
        lam = torch.exp(-blk.log_rate.exp())
        if kind == "S":
            r = blk.code(u)
            qw, fast, share = memory(r, blk.val(a), lam, blk._w(), filler, mode)
            h = h + blk.mu(qw) + fast
            shares.append(share)
        else:
            static = blk.mu(blk._code(blk.key_s, blk.log_eps_s, u))
            r = blk._code(blk.key_f, blk.log_eps_f, u)
            w = blk.log_w.clamp(-7.0, 12.0).exp()
            _, fast, _ = memory(r, blk.val(a), lam, w, filler, "A" if mode in ("A", "C") else "normal")
            h = h + static + fast
    return net.norm(h), shares


def filler_writes(T, G):
    t = torch.arange(T)
    return (t >= 2 * N) & (t <= 2 * N + G) if G > 0 else torch.zeros(T, dtype=torch.bool)


@torch.no_grad()
def evaluate(net, kind, G, mode, n=1024, chunk=256):
    x, y, pos = batches(G, n)
    filler = filler_writes(x.size(1), G)
    hit, cats, share_q = 0, torch.zeros(3), []
    for i in range(0, x.size(0), chunk):
        hs, shares = run(net, kind, x[i:i + chunk], filler, mode)
        pred = net.output(hs[:, pos]).argmax(-1)
        t = y[i:i + chunk][:, pos]
        hit += (pred == t).sum().item()
        wrong = pred[pred != t]
        cats += torch.tensor([((wrong >= THIRD) & (wrong < 2 * THIRD)).sum().item(),  # some other value
                              ((wrong >= 1) & (wrong < THIRD)).sum().item(),            # a key
                              ((wrong < 1) | (wrong >= 2 * THIRD)).sum().item()])       # anything else
        if shares:
            share_q.append(torch.stack([s[:, pos].mean() for s in shares]))
    total = x.size(0) * len(pos)
    out = {"acc": hit / total, "wrong_as": dict(zip(["value", "key", "other"], (cats / max(1, cats.sum().item())).tolist()))}
    if share_q:
        out["static_share_at_queries"] = torch.stack(share_q).mean(0).tolist()  # per layer
    return out


def check(net, kind):
    """The token-by-token form must reproduce the model's own forward (CPU reference memory)."""
    x, y, pos = batches(64, n=32, eb=32)
    x = x[:4]
    with torch.no_grad():
        ref = mqar.Body(net)(x)
        mine, _ = run(net, kind, x, torch.zeros(x.size(1), dtype=torch.bool), "normal")
    err = (ref - mine).abs().max().item()
    rel = err / ref.abs().max().item()  # relative: fp32 accumulation error grows with the size of the hidden state
    assert rel < 1e-5, f"{kind}: token-by-token form differs from the model by {err} ({rel:.1e} of the largest value)"
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s_dir", required=True)
    ap.add_argument("--p_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gaps", default="0,256,384,512,640,768,896,1024")
    ap.add_argument("--n", type=int, default=1024, help="test sequences per gap (1024 is the recorded test set)")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    gaps = [int(g) for g in a.gaps.split(",")]
    res = {"time_constant": {}, "check_max_abs_err": {}, "recorded_test": {}, "gaps": gaps, "S": {}, "P": {}}
    nets = {"S": load("S", a.s_dir), "P": load("P", a.p_dir)}
    for kind, run_dir in (("S", a.s_dir), ("P", a.p_dir)):
        net = nets[kind]
        res["time_constant"][kind] = [round(1 / blk.log_rate.exp().item(), 1) for blk in net.layers]
        res["check_max_abs_err"][kind] = check(net, kind)
        res["recorded_test"][kind] = json.load(open(os.path.join(run_dir, "eval.json")))["test"]
        print(kind, "time constants", res["time_constant"][kind], "check err", res["check_max_abs_err"][kind], flush=True)
    for kind in ("S", "P"):
        modes = ("normal", "A", "B", "C") if kind == "S" else ("normal", "A")
        for G in gaps:
            for mode in (modes if G in (256, 1024) else ("normal",)):
                t0 = time.time()
                r = evaluate(nets[kind], kind, G, mode, a.n)
                res[kind].setdefault(mode, {})[str(G)] = r
                print(kind, mode, f"G{G}", json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
                json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
