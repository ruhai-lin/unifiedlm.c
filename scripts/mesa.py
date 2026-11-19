"""
Mesa layer (von Oswald et al., "MesaNet: Sequence Modeling by Locally Optimal Test-Time Training", arXiv 2506.05233)
in place of attention: the MESA baseline, run like GDN (Mesa halves alternating with FFN halves). Not part of the
standard model.

Per head, with decay gamma_t, write strength beta_t and unit q_t, k_t, each token solves a regularised least-squares
problem over everything written so far, instead of taking one delta-rule step towards it as GDN and SD do:

    H_t = gamma_t H_{t-1} + beta_t k_t k_t^T,   G_t = gamma_t G_{t-1} + beta_t v_t k_t^T
    o_t = G_t (H_t + diag(lambda))^-1 q_t

This is the "optimal linear store" of alexwang.ai/posts/associative-memory: GDN and SD approximate it online.

The solve runs batched over all tokens by conjugate gradient with a Jacobi preconditioner, a fixed number of
steps; every product H_t p_t is a gated linear-attention pass (keys k, values beta k, query p_t), so H_t is never
stored. The gradient is implicit: the backward pass solves (H_t + Lambda) u_t = g_t with the same operator, then
grad q = u and the operator's parameters get -d/dtheta sum_t u_t^T (H_t + Lambda) x_t, one differentiable pass.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import RMSNorm, _hi

CHUNK = 64
CG_STEPS = 24      # relative error about 1e-3 with strongly correlated unit keys (hd 48, T 1024); 12 steps leave 4%


def _decay(lg, chunk):
    """lg: (B,N,C) log decay per step. Returns b (cumulative within the chunk, inclusive) and the (B,N,C,C) causal
    matrix exp(b_i - b_j) for j <= i (0 above the diagonal)."""
    b = lg.cumsum(-1)
    diff = b[..., :, None] - b[..., None, :]
    mask = torch.ones(chunk, chunk, dtype=torch.bool, device=lg.device).tril()
    return b, torch.exp(diff.masked_fill(~mask, -float("inf")))


def gla(q, k, v, lg, chunk=CHUNK):
    """Gated linear attention. q, k: (B,T,dk), v: (B,T,dv), lg: (B,T) log decay (<= 0) applied at step t.
    o_t = sum_{s<=t} exp(lg_{s+1} + ... + lg_t) (q_t . k_s) v_s, i.e. S_t = gamma_t S_{t-1} + k_t v_t^T, o_t = S_t^T q_t."""
    B, T, dk = k.shape
    dv = v.size(-1)
    pad = (-T) % chunk
    if pad:
        q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        lg = F.pad(lg, (0, pad))
    N = (T + pad) // chunk
    qc, kc, vc = q.reshape(B, N, chunk, dk), k.reshape(B, N, chunk, dk), v.reshape(B, N, chunk, dv)
    b, D = _decay(lg.reshape(B, N, chunk), chunk)
    intra = ((qc @ kc.transpose(-1, -2)) * D) @ vc
    qd = qc * torch.exp(b)[..., None]
    kv = (kc * torch.exp(b[..., -1:] - b)[..., None]).transpose(-1, -2) @ vc      # (B,N,dk,dv) chunk's own writes
    el = torch.exp(b[..., -1])
    S = k.new_zeros(B, dk, dv)
    out = []
    for n in range(N):
        out.append(intra[:, n] + qd[:, n] @ S)
        S = el[:, n, None, None] * S + kv[:, n]
    return torch.stack(out, 1).reshape(B, N * chunk, dv)[:, :T]


def decayed_sum(x, lg, chunk=CHUNK):
    """c_t = sum_{s<=t} exp(lg_{s+1} + ... + lg_t) x_s for x: (B,T,d), the diagonal of H_t when x = beta k^2."""
    B, T, d = x.shape
    pad = (-T) % chunk
    if pad:
        x, lg = F.pad(x, (0, 0, 0, pad)), F.pad(lg, (0, pad))
    N = (T + pad) // chunk
    xc = x.reshape(B, N, chunk, d)
    b, D = _decay(lg.reshape(B, N, chunk), chunk)
    intra = D @ xc
    tail = (xc * torch.exp(b[..., -1:] - b)[..., None]).sum(2)
    el, eb = torch.exp(b[..., -1]), torch.exp(b)
    c = x.new_zeros(B, d)
    out = []
    for n in range(N):
        out.append(intra[:, n] + eb[:, n, :, None] * c[:, None])
        c = el[:, n, None] * c + tail[:, n]
    return torch.stack(out, 1).reshape(B, N * chunk, d)[:, :T]


def _apply(p, k, kb, lg, lam):
    """(H_t + diag(lam)) p_t for every t; kb = beta k."""
    return gla(p, k, kb, lg) + lam * p


def _cg(rhs, k, kb, lg, lam, prec, steps):
    """Preconditioned conjugate gradient for (H_t + diag(lam)) x_t = rhs_t, all t at once, from x = 0."""
    x = torch.zeros_like(rhs)
    r = rhs
    z = r / prec
    p = z
    rz = (r * z).sum(-1, keepdim=True)
    for _ in range(steps):
        Ap = _apply(p, k, kb, lg, lam)
        alpha = rz / (p * Ap).sum(-1, keepdim=True).clamp_min(1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        z = r / prec
        rz_new = (r * z).sum(-1, keepdim=True)
        p = z + (rz_new / rz.clamp_min(1e-30)) * p
        rz = rz_new
    return x


class MesaSolve(torch.autograd.Function):
    """x_t = (H_t + diag(lam))^-1 q_t with an implicit gradient. q, k: (B,T,d), beta, lg: (B,T), lam: (d,)."""

    @staticmethod
    def forward(ctx, q, k, beta, lg, lam, steps):
        kb = beta[..., None] * k
        prec = decayed_sum(kb * k, lg) + lam
        x = _cg(q, k, kb, lg, lam, prec, steps)
        ctx.save_for_backward(x, k, beta, lg, lam)
        ctx.steps = steps
        return x

    @staticmethod
    def backward(ctx, g):
        x, k, beta, lg, lam = ctx.saved_tensors
        kb = beta[..., None] * k
        prec = decayed_sum(kb * k, lg) + lam
        u = _cg(g.contiguous(), k, kb, lg, lam, prec, ctx.steps)
        with torch.enable_grad():
            k_, b_, l_, m_ = (t.detach().requires_grad_() for t in (k, beta, lg, lam))
            s = (u * _apply(x, k_, b_[..., None] * k_, l_, m_)).sum()
            gk, gb, gl, gm = torch.autograd.grad(-s, (k_, b_, l_, m_))
        return u, gk, gb, gl, gm, None


class MesaHalf(nn.Module):
    """The Mesa layer in place of attention, wrapped like GatedDeltaHalf: q, k, v through a causal depthwise conv
    (size 4) and SiLU, unit q and k, gamma_t = exp(-exp(A) softplus(W_a h + dt)) and beta_t = sigmoid(W_b h) per head
    (GDN's parametrisation, no clamp: the chunked form uses differences of log decays), lambda > 0 per head and
    dimension (started at 1), then RMSNorm per head -> * SiLU(W_g h) -> output projection."""

    def __init__(self, args, head_dim):
        super().__init__()
        d = args.dim
        self.hd, self.h = head_dim, d // head_dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.wq, self.wk, self.wv, self.wg, self.mu = (nn.Linear(d, d, bias=False) for _ in range(5))
        self.conv = nn.Conv1d(3 * d, 3 * d, 4, groups=3 * d, padding=3, bias=False)
        self.wa, self.wb = nn.Linear(d, self.h, bias=False), nn.Linear(d, self.h, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.h).uniform_(1, 16).log())
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_shift = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.lam_raw = nn.Parameter(torch.full((self.h, head_dim), math.log(math.e - 1)))   # softplus^-1(1)
        self.onorm = RMSNorm(head_dim, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    @torch.compiler.disable
    def _mix(self, q, k, v, beta, lg):
        B, T, d = q.shape
        H, hd = self.h, self.hd
        heads = lambda z: z.reshape(B, T, H, hd).transpose(1, 2).reshape(B * H, T, hd)
        q, k, v = (heads(_hi(z)) for z in (q, k, v))
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = beta.transpose(1, 2).reshape(B * H, T)
        lg = lg.transpose(1, 2).reshape(B * H, T)
        lam = (F.softplus(self.lam_raw) + 1e-3)[None].expand(B, H, hd).reshape(B * H, 1, hd)
        x = MesaSolve.apply(q, k, beta, lg, lam, CG_STEPS)
        return gla(x, k, beta[..., None] * v, lg).reshape(B, H, T, hd).transpose(1, 2)

    def forward(self, x):
        B, T, d = x.shape
        a = self.norm(x)
        qkv = torch.cat([self.wq(a), self.wk(a), self.wv(a)], -1).transpose(1, 2)
        q, k, v = F.silu(self.conv(qkv)[..., :T]).transpose(1, 2).chunk(3, -1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            beta = torch.sigmoid(_hi(self.wb(a)))
            lg = -self.A_log.exp() * F.softplus(_hi(self.wa(a)) + self.dt_shift)
            o = self._mix(q, k, v, beta, lg)
        o = self.onorm(o).reshape(B, T, d).type_as(x)
        return x + self.dropout(self.mu(o * F.silu(self.wg(a))))
