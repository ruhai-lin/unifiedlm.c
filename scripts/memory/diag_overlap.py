"""Exploratory, after the pre-registered tests: why do filler writes hurt S more than P?
Per layer at gap 1024: how many slots one token's address spreads over (participation ratio), and what share of the
fast read at the questions comes from filler writes."""
import json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diag_collapse as dc
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "mem_diag") + "/"
torch.set_num_threads(10)
G, n = 1024, 256
x, y, pos = dc.batches(G, n)
T = x.size(1); N = dc.N
filler = dc.filler_writes(T, G)
pair_keys = torch.arange(0, 2 * N, 2); fill_pos = torch.arange(2 * N, 2 * N + G)
out = {}
for kind in ("S", "P"):
    net = dc.load(kind, R + f"m2kv_{kind.lower()}_d32_save")
    h = net.tok_embeddings(x); rows = []
    with torch.no_grad():
        for blk in net.layers:
            a = blk.norm(h); u = dc.conv(blk, a); lam = torch.exp(-blk.log_rate.exp())
            if kind == "S":
                r = blk.code(u); w = blk._w()
            else:
                r = blk._code(blk.key_f, blk.log_eps_f, u); w = blk.log_w.clamp(-7.0, 12.0).exp()
            pr = lambda z: ((z.sum(-1) ** 2) / (z ** 2).sum(-1).clamp_min(1e-12)).mean().item()
            # filler share of the evidence the questions read: E_filler / E over slots weighted by r/c
            B, _, m = r.shape
            qset = set(pos.tolist())
            E_all = torch.zeros(B, m); E_keep = torch.zeros(B, m); rp = torch.zeros(B, m); share = []
            for t in range(T):
                E_all = lam * E_all + rp
                E_keep = lam * E_keep + (0 if filler[t] else rp)
                if t in qset:
                    q = r[:, t] / (w + E_all)
                    share.append(((q * (E_all - E_keep)).sum(-1) / (q * E_all).sum(-1).clamp_min(1e-12)).mean().item())
                rp = r[:, t]
            rows.append({"slots_per_key": pr(r[:, pair_keys]), "slots_per_filler": pr(r[:, fill_pos]), "slots_per_question": pr(r[:, pos]),
                         "filler_share_of_question_read": sum(share) / len(share)})
            # advance the hidden state token by token (the dense reference would need T^2 memory)
            qw, fast, _ = dc.memory(r, blk.val(a), lam, w, filler, "normal")
            static = blk.mu(qw) if kind == "S" else blk.mu(blk._code(blk.key_s, blk.log_eps_s, u))
            h = h + static + fast
    out[kind] = rows
    for i, row in enumerate(rows):
        print(kind, "layer", i + 1, {k: round(v, 3) for k, v in row.items()})
json.dump(out, open(R + "explore_overlap.json", "w"), indent=1)
