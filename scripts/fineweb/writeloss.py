"""Does SD write less where it is not surprised? Per token and layer, the delta-rule write size
    |delta_t| = beta_t |u_t - E_{t-1}^T diag(lam_t) k_t|      (k unit, u = v - mu k: the part the weights miss)
against the loss on that token, -log p(x_t | x_<t). The write at step t carries token t's value under the address of
t - 1, so it pairs with the logits at t - 1. Reports Spearman correlations (write, |u| and beta against the loss) and the
write size by loss decile. Then the same model on its own text: after a 512-token real prefix, 256 tokens sampled at
temperature 1 and 256 greedy, against the real continuation at the same positions. Prints JSON and writes
writeloss.json into the run directory. Run from the repo root:

$ python scripts/fineweb/writeloss.py --out_dir=<run> [--device=cpu]
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

P, G = 512, 256


def spearman(a, b):
    ra, rb = (x.argsort().argsort().double() for x in (a, b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm())).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # noqa: E402
    net, meta, ckpt = ev.load(a.out_dir, a.device)
    L = ckpt["model_args"]["max_seq_len"]
    val = np.memmap(sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1], dtype=np.uint16, mode="r")
    xy = torch.from_numpy(np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64) for j in range(a.windows)])).to(a.device)
    n = len(net.layers)
    f = delta.delta_memory
    rec = []     # per call (one per layer, in layer order): (write, |u|, beta), each (B, T)
    err = [0.0]

    def hook(k, u, beta, logl, q, *rest, **kw):     # k is already unit
        o = f(k, u, beta, logl, q, *rest, **kw)
        B, T, m = k.shape
        E = k.new_zeros(B, m, u.size(-1))
        lam = logl.exp()
        w, ob = [], []
        for t in range(T):
            d = beta[:, t, None] * (u[:, t] - torch.einsum("bm,bmd->bd", lam[:, t] * k[:, t], E))
            E = lam[:, t, :, None] * E + k[:, t, :, None] * d[:, None, :]
            w.append(d.norm(dim=-1))
            ob.append(torch.einsum("bm,bmd->bd", q[:, t], E))
        err[0] = max(err[0], ((torch.stack(ob, 1) - o).norm() / o.norm()).item())
        rec.append((torch.stack(w, 1), u.norm(dim=-1), beta))
        return o

    def measure(x, y):
        """Per-layer (write, |u|, beta) (B, T) and the loss (B, T) on token t (the logits at t - 1; position 0 is nan)."""
        rec.clear()
        delta.delta_memory = hook
        x, y = x.contiguous(), y.contiguous()
        logits = net(x, y)
        delta.delta_memory = f
        nll = F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
        return list(rec), torch.cat([torch.full_like(nll[:, :1], float("nan")), nll[:, :-1]], 1)

    out = {"variant": meta["variant"], "windows": a.windows, "layers": []}
    with torch.no_grad():
        per = [[[], [], []] for _ in range(n)]
        nl = []
        for i in range(0, a.windows, 4):
            r, nll = measure(xy[i:i + 4, :-1], xy[i:i + 4, 1:])
            nl.append(nll[:, 1:].flatten().cpu())
            for li in range(n):
                for j in range(3):
                    per[li][j].append(r[li][j][:, 1:].flatten().cpu())
        nl = torch.cat(nl)
        edges = torch.quantile(nl, torch.linspace(0, 1, 11)).tolist()
        dec = torch.bucketize(nl, torch.tensor(edges[1:-1]))
        for li in range(n):
            w, un, be = (torch.cat(x) for x in per[li])
            out["layers"].append({"spearman_write_nll": spearman(w, nl), "spearman_target_nll": spearman(un, nl),
                                  "spearman_beta_nll": spearman(be, nl),
                                  "write_by_nll_decile": [(w[dec == b].mean() / w.mean()).item() for b in range(10)]})
        out["nll_decile_edges"] = edges
        out["mean"] = {k: float(np.mean([l[k] for l in out["layers"]])) for k in ("spearman_write_nll", "spearman_target_nll", "spearman_beta_nll")}
        out["write_by_nll_decile_mean"] = np.mean([l["write_by_nll_decile"] for l in out["layers"]], 0).tolist()

        x0 = xy[:8, :P + G]
        gen = {"real": x0}
        for mode in ("sample", "greedy"):
            x = x0[:, :P]
            for _ in range(G):
                lg = net(x)[:, -1]
                nxt = torch.multinomial(F.softmax(lg, -1), 1) if mode == "sample" else lg.argmax(-1, keepdim=True)
                x = torch.cat([x, nxt], 1)
            gen[mode] = x
        res = {}
        for mode, x in gen.items():
            r, nll = measure(x[:, :-1], x[:, 1:])
            res[mode] = {"nll": nll[:, P:].mean().item(), "write": [r[li][0][:, P:].mean().item() for li in range(n)]}
        out["generation"] = {"prefix": P, "generated": G, "rows": 8}
        for mode in ("real", "sample", "greedy"):
            out["generation"][mode] = {"nll": res[mode]["nll"],
                                       "write_vs_real": [w / wr for w, wr in zip(res[mode]["write"], res["real"]["write"])]}
            out["generation"][mode]["write_vs_real_mean"] = float(np.mean(out["generation"][mode]["write_vs_real"]))
    out["loop_vs_kernel_rel_err"] = err[0]
    print(json.dumps(out), flush=True)
    json.dump(out, open(os.path.join(a.out_dir, "writeloss.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
