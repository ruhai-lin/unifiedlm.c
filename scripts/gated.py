"""
SG: the unified block with a data-dependent, per-slot forget gate. Not part of the standard model.

S forgets every slot at one learned scalar rate per layer, lam = exp(-rate). SG replaces it with

    lam_{t,a} = exp(expm1(-x_{t,a})),  x_{t,a} = rate_a * g_{t,a} + beta * k_{t,a}
    g_t       = softplus(P Q rms(r_t) + b) / ln 2

- rate_a: one learned base rate per slot, initialised to S's rate;
- g_t: a gate read off the relu^2 address r_t through a rank-16 bottleneck (Q: m -> 16, P: 16 -> m), so the same
  address that picks the slots also decides how much each slot forgets; ln 2 makes g = 1 when P Q r + b = 0;
- beta * k_t: the slots being written forget a little first (delta-rule style), k_t = r_{t-1}; beta starts at e^-7;
- expm1(-x) maps x in [0, inf) to log lam in (-1, 0]: ~ -x for small x, at most 63% forgotten per token, which
  keeps the chunked form below inside fp32 range.

At initialisation (bias 0, P and Q at std 0.02) g ~ 1, so SG starts within a small perturbation of S.
The state update is S's with lam per slot and per token:

    U_t = lam_t * U_{t-1} + (1 - lam_t) * mu w + v_t k_t^T,   c_t = lam_t * c_{t-1} + (1 - lam_t) * w + k_t
    y_t = U_t (r_t / c_t)

computed in parallel by gated_memory (chunked, PyTorch autograd, fp32 with TF32 matmuls; no Triton yet).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import UnifiedBlock, _hi

CHUNK = 64
RANK = 16


def gated_memory(r, v, logl, w, chunk=CHUNK):
    """r: (B,T,m) read addresses, v: (B,T,d) values, logl: (B,T,m) log decay applied at step t (<= 0, >= -1),
    w: (m,) prior mass. Returns q = r / c and fast_t = sum_a q_{t,a} sum_{s<=t} D_{t,s,a} k_{s,a} v_s,
    D_{t,s,a} = prod_{s<u<=t} lam_{u,a}, k_s = r_{s-1}."""
    B, T, m = r.shape
    d = v.size(-1)
    k = F.pad(r[:, :-1], (0, 0, 1, 0))
    pad = (-T) % chunk
    if pad:
        r, k, v, logl = (F.pad(x, (0, 0, 0, pad)) for x in (r, k, v, logl))
    N = (T + pad) // chunk
    rc, kc, vc, lc = (x.reshape(B, N, chunk, -1) for x in (r, k, v, logl))
    Bc = lc.cumsum(2)                       # log decay from the state entering the chunk to step i
    Bl = Bc[:, :, -1]                       # whole chunk
    kh = kc * torch.exp(-Bc)                # k_s e^{-B_s}
    ks = kc * torch.exp(Bl[:, :, None] - Bc)  # k_s e^{B_last - B_s}, <= k_s
    # evidence mass: c~ = c - w carried across chunks
    dc = ks.sum(2)                          # (B,N,m)
    dH = ks.transpose(-1, -2) @ vc          # (B,N,m,d)
    el = torch.exp(Bl)
    cprev, hprev = [], []
    c = r.new_zeros(B, m)
    h = r.new_zeros(B, m, d)
    for n in range(N):
        cprev.append(c)
        hprev.append(h)
        c = el[:, n] * c + dc[:, n]
        h = el[:, n, :, None] * h + dH[:, n]
    cprev, hprev = torch.stack(cprev, 1), torch.stack(hprev, 1)
    eB = torch.exp(Bc)
    ct = eB * (kh.cumsum(2) + cprev[:, :, None])
    q = rc / (w + ct)
    qh = q * eB
    A = (qh @ kh.transpose(-1, -2)).tril()  # (B,N,C,C), s <= t
    fast = A @ vc + qh @ hprev
    q, fast = q.reshape(B, N * chunk, m)[:, :T], fast.reshape(B, N * chunk, d)[:, :T]
    return q, fast


class GatedBlock(UnifiedBlock):

    def __init__(self, args, rank=RANK, gate_from="r", silu=False):
        """gate_from="h" reads the gate off the block input h = norm(x) instead of the address r (SGh);
        silu=True puts a SiLU between Q and P, a two-layer MLP gate (SGs)."""
        super().__init__(args)
        m = args.hidden_dim
        self.gate_from, self.silu = gate_from, silu
        self.log_rate = nn.Parameter(torch.full((m,), math.log(1.0 / (4 * args.max_seq_len))))
        self.gate_q = nn.Linear(m if gate_from == "r" else args.dim, rank, bias=False)
        self.gate_p = nn.Linear(rank, m, bias=True)
        self.log_beta = nn.Parameter(torch.tensor(-7.0))

    def log_decay(self, r, h=None):
        k = F.pad(r[:, :-1], (0, 0, 1, 0))
        z = r * torch.rsqrt(r.square().mean(-1, keepdim=True) + 1e-12) if self.gate_from == "r" else _hi(h)
        z = self.gate_q(z)
        if self.silu:
            z = F.silu(z)
        g = F.softplus(_hi(self.gate_p(z))) / math.log(2.0)
        x = self.log_rate.exp() * g + self.log_beta.exp() * k
        return torch.expm1(-x)

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        r = self.code(u)
        v = self.val(h)
        w = self._w()
        with torch.autocast(device_type=x.device.type, enabled=False):
            r, v = _hi(r), _hi(v)
            q, fast = gated_memory(r, v, self.log_decay(r, h), w.to(r.dtype))
        return x + self.dropout(self.mu((q * w).type_as(x)) + fast.type_as(x))

    def step(self, x, s):
        raise NotImplementedError("token-by-token SG lives in scripts/check_gated.py until SG wins")


class RateBlock(GatedBlock):
    """SR, the ablation of SG: per-slot rates only, no data-dependent gate and no write erase (lam_a = exp(expm1(-rate_a)))."""

    def __init__(self, args):
        super().__init__(args)
        del self.gate_q, self.gate_p, self.log_beta

    def log_decay(self, r, h=None):
        return torch.expm1(-self.log_rate.exp()).expand_as(r)


class ShortBlock(UnifiedBlock):
    """SH, a control for SG: S unchanged except that its one forgetting rate starts at 1/64 instead of
    1/(4 L), the memory length SG's gate settles near; still learned, still one scalar per layer."""

    def __init__(self, args):
        super().__init__(args)
        with torch.no_grad():
            self.log_rate.fill_(math.log(1.0 / 64))
