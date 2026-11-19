"""
Ablation variants for the unification experiment. Not part of the standard model.

    S  model.UnifiedBlock          y = mu(r w/c) + fast(r)    shared address, evidence attenuates the static read
    D  DecoupledBlock              y = mu(r)     + fast(r)    shared address, no attenuation
    P  SeparateBlock(m_s, m_f)     y = mu(r_s)   + fast(r_f)  one address per path, two slot sets
    A  SeparateBlock alternating   y = fast(r_f), then y = mu(r_s)
    T  llama2.c Transformer        attention with RoPE, then a SwiGLU feed-forward of hidden_dim units, per block pair

fast(r) is unified_memory(r, v, g, w): read address r, write address r shifted by one, normalized by
c = w + decayed evidence. Variants only train and evaluate; forward_recurrent, export and run.c stay
with the standard model until a variant wins.
"""
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import model
from kernel import unified_memory
from model import RMSNorm, UnifiedBlock, _hi


class DecoupledBlock(UnifiedBlock):

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        r = self.code(u)
        _, fast = unified_memory(r, self.val(h), -self.log_rate.exp(), self._w())
        return x + self.dropout(self.mu(r.type_as(x)) + fast.type_as(x))

    def step(self, x, s):
        raise NotImplementedError("the inherited token-by-token form computes S, not D")


class SeparateBlock(nn.Module):
    """m_s static slots and m_f context slots with independent addresses; either count may be 0."""

    def __init__(self, args, m_s, m_f):
        super().__init__()
        d = args.dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.conv = nn.Conv1d(d, d, args.conv_size, groups=d, padding=args.conv_size - 1, bias=False)
        self.dropout = nn.Dropout(args.dropout)
        if m_s:
            self.key_s = nn.Linear(d, m_s, bias=True)
            self.log_eps_s = nn.Parameter(torch.zeros(()))
            self.mu = nn.Linear(m_s, d, bias=False)
        if m_f:
            self.key_f = nn.Linear(d, m_f, bias=True)
            self.log_eps_f = nn.Parameter(torch.zeros(()))
            self.val = nn.Linear(d, d, bias=False)
            self.log_rate = nn.Parameter(torch.tensor(math.log(1.0 / (4 * args.max_seq_len))))
            self.log_w = nn.Parameter(torch.full((m_f,), math.log(args.prior)))
        self.m_s, self.m_f = m_s, m_f

    @staticmethod
    def _code(key, log_eps, u):
        z = F.relu(_hi(key(u))).square()
        return z / (log_eps.clamp(-10, 10).exp() + z.sum(-1, keepdim=True))

    def forward(self, x):
        h = self.norm(x)
        u = self.conv(h.transpose(1, 2))[..., : h.size(1)].transpose(1, 2)
        y = 0
        if self.m_s:
            y = y + self.mu(self._code(self.key_s, self.log_eps_s, u).type_as(x))
        if self.m_f:
            r = self._code(self.key_f, self.log_eps_f, u)
            w = self.log_w.clamp(-7.0, 12.0).exp()
            _, fast = unified_memory(r, self.val(h), -self.log_rate.exp(), w)
            y = y + fast.type_as(x)
        return x + self.dropout(y)


HEAD_DIM = 48  # llama2.c 15M: 6 heads of 48 at dim 288


def _rope(x):
    T, hd = x.size(-2), x.size(-1)
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = torch.outer(torch.arange(T, device=x.device).float(), inv)
    c, s = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()
    return torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), -1).flatten(-2).type_as(x)


class CausalConv(nn.Module):
    """SD's causal depthwise conv over time (size 4, no activation). UnifiedLM's init starts it as the identity."""

    def __init__(self, d, size=4):
        super().__init__()
        self.conv = nn.Conv1d(d, d, size, groups=d, padding=size - 1, bias=False)

    def forward(self, x):
        return self.conv(x.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)


LOGN = None   # evaluation only: Qwen's log-n scaling for T past its training length L, q_t *= max(1, log n_t / log L)


class AttentionHalf(nn.Module):
    """The attention half of a llama2.c layer. The output projection is named mu so that UnifiedLM's init scales
    it to 0.02/sqrt(n_layers), which with two blocks per layer is GPT-2's scaling of residual projections.
    conv=True (TC) passes the normed input through SD's causal conv before q, k, v.
    ssmax=True (TS) is Scalable-Softmax (Nakanishi 2025, arXiv 2501.19399): the logits of a query that sees n keys
    are multiplied by s log n, s learnable per head, started at 1 / log L so that at n = L it is the usual softmax."""

    def __init__(self, args, conv=False, ssmax=False):
        super().__init__()
        d = args.dim
        self.h = d // HEAD_DIM
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.mix = CausalConv(d) if conv else nn.Identity()
        self.wq, self.wk, self.wv, self.mu = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.ssmax = ssmax
        if ssmax:
            self.s = nn.Parameter(torch.full((self.h,), 1.0 / math.log(args.max_seq_len)))
        self.dropout = nn.Dropout(args.dropout)

    def _scale(self, q, window=None):
        """SSMax and LOGN on q (B, h, T, hd); n_t = t + 1 keys, at most `window` under a sliding window."""
        if not (self.ssmax or LOGN):
            return q
        n = torch.arange(1, q.size(2) + 1, device=q.device, dtype=torch.float32)
        if window:
            n = n.clamp(max=window)
        if self.ssmax:
            q = q * (self.s[:, None] * n.log())[:, :, None].to(q.dtype)
        if LOGN:
            q = q * (n.log() / math.log(LOGN)).clamp(min=1.0)[:, None].to(q.dtype)
        return q

    def forward(self, x):
        B, T, d = x.shape
        a = self.mix(self.norm(x))
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        y = F.scaled_dot_product_attention(_rope(self._scale(q)), _rope(k), v, is_causal=True)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d)))


class WindowAttentionHalf(AttentionHalf):
    """AttentionHalf where each query sees only the last `window` keys (itself included): SDA's local attention.
    With RoPE and a band this short, it behaves the same at any sequence length."""

    def __init__(self, args, window=128):
        super().__init__(args)
        self.window = window

    def forward(self, x):
        B, T, d = x.shape
        a = self.norm(x)
        q, k, v = (w(a).view(B, T, self.h, HEAD_DIM).transpose(1, 2) for w in (self.wq, self.wk, self.wv))
        i = torch.arange(T, device=x.device)
        band = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v, attn_mask=band)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d)))


class GatedAttentionHalf(nn.Module):
    """The attention half of the Qwen3-Next / Qwen3.5 / Qwen3.6 hybrids: gated attention (Qiu et al. 2025,
    arXiv 2505.06708), a sigmoid gate per head channel on the attention output, y * sigmoid(W_g h), before the
    output projection; grouped-query attention with kv_heads key/value heads; RMSNorm on each q and k head
    (Qwen3's QK-norm); full RoPE as in AttentionHalf (the Qwen hybrids rotate part of each head)."""

    def __init__(self, args, kv_heads=2):
        super().__init__()
        d = args.dim
        self.h, self.kv = d // HEAD_DIM, kv_heads
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.wq, self.wg, self.mu = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.wk, self.wv = (nn.Linear(d, kv_heads * HEAD_DIM, bias=False) for _ in range(2))
        self.qnorm, self.knorm = RMSNorm(HEAD_DIM, eps=args.norm_eps), RMSNorm(HEAD_DIM, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        B, T, d = x.shape
        a = self.norm(x)
        q = self.qnorm(self.wq(a).view(B, T, self.h, HEAD_DIM)).transpose(1, 2)
        k, v = (w(a).view(B, T, self.kv, HEAD_DIM) for w in (self.wk, self.wv))
        k, v = (z.transpose(1, 2).repeat_interleave(self.h // self.kv, 1) for z in (self.knorm(k), v))
        y = F.scaled_dot_product_attention(_rope(q), _rope(k), v, is_causal=True)
        return x + self.dropout(self.mu(y.transpose(1, 2).reshape(B, T, d) * torch.sigmoid(self.wg(a))))


class GatedDeltaHalf(nn.Module):
    """Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025, arXiv 2412.06464) in place of attention: the GDN
    baseline, a recurrent (SSM-family) Transformer. Per head, with unit q_t, k_t (q also scaled by 1/sqrt(dk)):
        S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t
        alpha_t = exp(-exp(A) softplus(W_a h + dt)),  beta_t = sigmoid(W_b h)
    then o -> RMSNorm per head -> * SiLU(W_g h) -> output projection, as in the paper and fla's GatedDeltaNet.
    q, k, v pass a causal depthwise conv (size 4) and SiLU. delta.delta_memory computes the recurrence (chunk 32);
    log alpha is clamped at -2 per step so its within-chunk exp(-cumsum) stays finite in fp32. soft=True (GDNs)
    bounds it smoothly instead, 2 expm1(log alpha / 2) in (-2, 0], so a head near the bound still gets a gradient."""

    def __init__(self, args, soft=False):
        super().__init__()
        self.soft = soft
        d = args.dim
        self.h = d // HEAD_DIM
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.wq, self.wk, self.wv, self.wg, self.mu = (nn.Linear(d, d, bias=False) for _ in range(5))
        self.conv = nn.Conv1d(3 * d, 3 * d, 4, groups=3 * d, padding=3, bias=False)
        self.wa, self.wb = nn.Linear(d, self.h, bias=False), nn.Linear(d, self.h, bias=False)
        self.A_log = nn.Parameter(torch.empty(self.h).uniform_(1, 16).log())          # fla's init
        dt = torch.exp(torch.rand(self.h) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_shift = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))                # softplus^-1(dt)
        self.onorm = RMSNorm(HEAD_DIM, eps=args.norm_eps)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        import delta
        B, T, d = x.shape
        H, hd = self.h, HEAD_DIM
        a = self.norm(x)
        qkv = torch.cat([self.wq(a), self.wk(a), self.wv(a)], -1).transpose(1, 2)
        q, k, v = F.silu(self.conv(qkv)[..., :T]).transpose(1, 2).chunk(3, -1)
        heads = lambda z: z.reshape(B, T, H, hd).transpose(1, 2).reshape(B * H, T, hd)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (heads(_hi(z)) for z in (q, k, v))
            q = F.normalize(q, dim=-1) / math.sqrt(hd)
            k = F.normalize(k, dim=-1)
            beta = torch.sigmoid(_hi(self.wb(a))).transpose(1, 2).reshape(B * H, T)
            la = -self.A_log.exp() * F.softplus(_hi(self.wa(a)) + self.dt_shift)
            la = (2 * torch.expm1(la / 2) if self.soft else la.clamp(min=-2.0)).transpose(1, 2).reshape(B * H, T, 1).expand(B * H, T, hd)
            o = delta.delta_memory(k, v, beta, la, q, chunk=32)
        o = self.onorm(o.reshape(B, H, T, hd).transpose(1, 2)).reshape(B, T, d).type_as(x)
        return x + self.dropout(self.mu(o * F.silu(self.wg(a))))


def erase_write_memory(k, e, z, logl, q, chunk=32):
    """Gated Delta Rule-2: k: (B,T,dk) unit write keys, e = b * k erase keys, z = w * v gated values (B,T,dv),
    logl: (B,T,dk) channel-wise log decay, q: (B,T,dk). With S_0 = 0 and D_t = diag(exp(logl_t)),
        S_t = (I - k_t e_t^T) D_t S_{t-1} + k_t z_t^T = D_t S_{t-1} + k_t delta_t^T,  delta_t = z_t - (D_t S_{t-1})^T e_t,
    o_t = S_t^T q_t. delta.delta_memory with the erase key and the write strength split; e = beta k, z = beta u gives it."""
    B, T, m = k.shape
    d = z.size(-1)
    pad = (-T) % chunk
    if pad:
        k, e, z, logl, q = (F.pad(x, (0, 0, 0, pad)) for x in (k, e, z, logl, q))
    N = (T + pad) // chunk
    kc, ec, zc, lc, qc = (x.reshape(B, N, chunk, -1) for x in (k, e, z, logl, q))
    b = lc.cumsum(2)
    bl = b[:, :, -1]
    ed = ec * torch.exp(b)                    # D_i e_i
    ki = kc * torch.exp(-b)                   # D_j^-1 k_j
    qd = qc * torch.exp(b)
    ks = kc * torch.exp(bl[:, :, None] - b)
    eye = torch.eye(chunk, device=k.device, dtype=k.dtype)
    A = (ed @ ki.transpose(-1, -2)).tril(-1)  # A_ij = D_i e_i . D_j^-1 k_j, j < i
    M = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False, unitriangular=True)
    PU, PK = M @ zc, M @ ed
    Q = (qd @ ki.transpose(-1, -2)).tril()
    el = torch.exp(bl)
    S = k.new_zeros(B, m, d)
    out = []
    for n in range(N):
        delta = PU[:, n] - PK[:, n] @ S
        out.append(qd[:, n] @ S + Q[:, n] @ delta)
        S = el[:, n, :, None] * S + ks[:, n].transpose(-1, -2) @ delta
    return torch.stack(out, 1).reshape(B, N * chunk, d)[:, :T]


class GatedDeltaNet2Half(GatedDeltaHalf):
    """Gated DeltaNet-2 (Hatamizadeh, Choi, Kautz, NeurIPS 2026, arXiv 2605.22791) in place of attention: GDN with
    the scalar beta split into a channel-wise erase gate b_t = sigmoid(W_b h) over key channels and a write gate
    w_t = sigmoid(W_w h) over value channels, and KDA's channel-wise decay
        g_t = -exp(a) * softplus(W_f h + delta)  (a per head, delta per key channel),
        S_t = (I - k_t (b_t * k_t)^T) diag(exp(g_t)) S_{t-1} + k_t (w_t * v_t)^T.
    The three gate projections are low-rank (d -> head_dim -> d, as fla's KDA), so the layer costs about 80K more
    parameters than GDN's instead of 250K. Everything else as GatedDeltaHalf (conv + SiLU, unit q and k, RMSNorm per
    head, SiLU output gate); g is clamped
    at -2 per step like GDN's log alpha, for the same within-chunk exp(-cumsum) reason."""

    def __init__(self, args):
        super().__init__(args)
        d = args.dim
        del self.wa, self.wb, self.dt_shift
        low_rank = lambda: nn.Sequential(nn.Linear(d, HEAD_DIM, bias=False), nn.Linear(HEAD_DIM, d, bias=False))
        self.wf, self.wb, self.ww = low_rank(), low_rank(), low_rank()   # rank head_dim, as fla's KDA gates
        dt = torch.exp(torch.rand(d) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_shift = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))                # per key channel

    def forward(self, x):
        B, T, d = x.shape
        H, hd = self.h, HEAD_DIM
        a = self.norm(x)
        qkv = torch.cat([self.wq(a), self.wk(a), self.wv(a)], -1).transpose(1, 2)
        q, k, v = F.silu(self.conv(qkv)[..., :T]).transpose(1, 2).chunk(3, -1)
        heads = lambda z: z.reshape(B, T, H, hd).transpose(1, 2).reshape(B * H, T, hd)
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = (heads(_hi(z)) for z in (q, k, v))
            q = F.normalize(q, dim=-1) / math.sqrt(hd)
            k = F.normalize(k, dim=-1)
            bg, wg = (heads(torch.sigmoid(_hi(w(a)))) for w in (self.wb, self.ww))
            g = -self.A_log.exp().repeat_interleave(hd) * F.softplus(_hi(self.wf(a)) + self.dt_shift)
            o = erase_write_memory(k, bg * k, wg * v, heads(g.clamp(min=-2.0)), q)
        o = self.onorm(o.reshape(B, H, T, hd).transpose(1, 2)).reshape(B, T, d).type_as(x)
        return x + self.dropout(self.mu(o * F.silu(self.wg(a))))


class FeedForwardHalf(nn.Module):
    """The SwiGLU feed-forward half of a llama2.c layer; mu is its output projection (w2 in llama2.c)."""

    def __init__(self, args):
        super().__init__()
        d, hidden = args.dim, args.hidden_dim
        self.norm = RMSNorm(d, eps=args.norm_eps)
        self.w1, self.w3 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.mu = nn.Linear(hidden, d, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        f = self.norm(x)
        return x + self.dropout(self.mu(F.silu(self.w1(f)) * self.w3(f)))


def install(variant, m_s=0, m_f=0):
    """UnifiedLM builds its layers through the global model.UnifiedBlock, so swapping it is enough."""
    if variant == "S":
        model.UnifiedBlock = UnifiedBlock
    elif variant == "D":
        model.UnifiedBlock = DecoupledBlock
    elif variant == "P":
        model.UnifiedBlock = lambda args: SeparateBlock(args, m_s, m_f)
    elif variant == "A":
        i = itertools.count()
        model.UnifiedBlock = lambda args: SeparateBlock(args, 0, m_f) if next(i) % 2 == 0 else SeparateBlock(args, m_s, 0)
    elif variant == "SG":  # S with a per-slot, per-token forget gate read off the address, scripts/gated.py
        import gated
        model.UnifiedBlock = gated.GatedBlock
    elif variant == "SG64":  # SG with a rank-64 gate bottleneck
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, rank=64)
    elif variant == "SGh":  # SG with the gate read off the block input h instead of the address r
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, gate_from="h")
    elif variant == "SGs":  # SG with a SiLU inside the rank-16 gate (a two-layer MLP gate)
        import gated
        model.UnifiedBlock = lambda args: gated.GatedBlock(args, silu=True)
    elif variant == "SR":  # SG's ablation: per-slot rates, no gate
        import gated
        model.UnifiedBlock = gated.RateBlock
    elif variant == "SH":  # control: S with its forgetting rate started at 1/64
        import gated
        model.UnifiedBlock = gated.ShortBlock
    elif variant == "SD":  # S with a delta-rule memory anchored at the FFN, scripts/delta.py
        import delta
        model.UnifiedBlock = delta.DeltaBlock
    elif variant == "SDp":  # SD with a learnable address power per layer (sharpness), from 2
        import delta
        model.UnifiedBlock = delta.PowerDeltaBlock
    elif variant == "SDr":  # SD with plain ReLU addresses (p = 1) in the first 4 blocks, relu^2 above
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: delta.FixedPowerDeltaBlock(args, 1) if next(i) < 4 else delta.DeltaBlock(args)
    elif variant == "SDw":  # SD with per-slot base rates spread from 1/(L/4) to 1/(16 L): fast and slow slots in one memory
        import delta
        model.UnifiedBlock = delta.WideRateDeltaBlock
    elif variant == "SDl":  # SD with a 10x learning rate on the per-slot base rates (log_rate), by reparametrization
        import delta
        model.UnifiedBlock = delta.FastRateDeltaBlock
    elif variant == "SDu":  # SD with the static half on the unnormalized relu^2 code (a plain ReLU^2 FFN)
        import delta
        model.UnifiedBlock = delta.RawFFNDeltaBlock
    elif variant == "SDul":  # SDu with SDl1000's 1000x learning rate on log_rate
        import delta
        model.UnifiedBlock = delta.RawFFNFastRate1000DeltaBlock
    elif variant == "SDn":  # control: SDu with the memory removed (FFN on the conv output only)
        import delta
        model.UnifiedBlock = delta.NoMemoryDeltaBlock
    elif variant == "SDula":  # SDul with the write target anchored at the FFN output it gives, v - mu z_{t-1}
        import delta
        model.UnifiedBlock = delta.AnchorRawFFNDeltaBlock
    elif variant == "SDulag":  # SDula with a per-token gate on the memory read, 2 sigmoid(w . h + b)
        import delta
        model.UnifiedBlock = delta.ReadGateAnchorDeltaBlock
    elif variant == "SDulas":  # control for SDulag: one learned read scale per block instead of a per-token gate
        import delta
        model.UnifiedBlock = delta.ReadScaleAnchorDeltaBlock
    elif variant == "SDulagn":  # SDulag with the memory read RMS-normalised before the gate (GatedDeltaNet's output norm)
        import delta
        model.UnifiedBlock = delta.ReadNormGateAnchorDeltaBlock
    elif variant == "SDulagq":  # SDulag with a low-rank correction on the read address only (asymmetric match)
        import delta
        model.UnifiedBlock = delta.ReadQueryGateAnchorDeltaBlock
    elif variant == "SDulagnq":  # SDulagq with SDulagn's normalised read: both read changes together
        import delta
        model.UnifiedBlock = delta.ReadNormQueryGateAnchorDeltaBlock
    elif variant == "UD2-gate-q128":  # SDulagq (UD2-gate-q32) with the read-address correction at rank 128
        import delta
        model.UnifiedBlock = delta.ReadQuery128GateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-q32-rope":  # SDulagnq (UD2-gate-norm-q32) with rotary relative position in half the memory
        import delta
        model.UnifiedBlock = delta.RopeReadNormQueryGateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-q32-ropelo":  # UD2-gate-norm-q32-rope with rope in the lower half of the blocks only
        import delta
        i = itertools.count()   # every model builds n_layers blocks, so the index modulo n_layers is the block's depth
        model.UnifiedBlock = lambda args: (delta.RopeReadNormQueryGateAnchorDeltaBlock if next(i) % args.n_layers < args.n_layers // 2
                                           else delta.ReadNormQueryGateAnchorDeltaBlock)(args)
    elif variant == "UD2-gate-norm-q32-rope-ev":  # UD2-gate-norm-q32-rope with the token embedding in the write value
        import delta
        model.UnifiedBlock = delta.EmbedValueRopeReadNormQueryGateAnchorDeltaBlock
        model.UnifiedLM.forward = delta.embed_value_forward
    elif variant == "UD2-gate-norm-q32-ev":  # UD2-gate-norm-q32 (no rope) with -rope-ev's token embedding in the write value
        import delta
        model.UnifiedBlock = delta.EmbedValueReadNormQueryGateAnchorDeltaBlock
        model.UnifiedLM.forward = delta.embed_value_forward
    elif variant == "UD2-gate-norm-q32-ropelo-ev":  # -ropelo with the -ev write value in every block
        import delta
        i = itertools.count()   # as -ropelo: the index modulo n_layers is the block's depth
        model.UnifiedBlock = lambda args: (delta.EmbedValueRopeReadNormQueryGateAnchorDeltaBlock if next(i) % args.n_layers < args.n_layers // 2
                                           else delta.EmbedValueReadNormQueryGateAnchorDeltaBlock)(args)
        model.UnifiedLM.forward = delta.embed_value_forward
    elif variant == "UD2-gate-norm-q32-decay":  # UD2-gate-norm-q32 with fixed per-slot half-lives 2..8192 and a bounded gate
        import delta
        model.UnifiedBlock = delta.SpectrumReadNormQueryGateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-q32-decay-ev":  # UD2-gate-norm-q32-decay with the -ev write value
        import delta
        model.UnifiedBlock = delta.EmbedValueSpectrumReadNormQueryGateAnchorDeltaBlock
        model.UnifiedLM.forward = delta.embed_value_forward
    elif variant == "UD2-gate-norm-q32-decay-half":  # half the slots as -decay (fixed spectrum), half as the base (learned)
        import delta
        model.UnifiedBlock = delta.HalfSpectrumReadNormQueryGateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-q32-conv8":  # UD2-gate-norm-q32 trained with --conv_size=8 (the flag sets the width)
        import delta
        model.UnifiedBlock = delta.ReadNormQueryGateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-q128-rope":  # UD2-gate-norm-q32-rope with the read-address correction at rank 128
        import delta
        model.UnifiedBlock = delta.RopeReadNormQuery128GateAnchorDeltaBlock
    elif variant == "UD2-gate-norm-h4q8":  # UD2-gate-norm-q32 with the read split into 4 heads of rank 8 (same parameters)
        import delta
        model.UnifiedBlock = delta.MultiHeadReadNormGateAnchorDeltaBlock
    elif variant == "UD2-gate-mass-q32":  # UD2-gate-q32 with the read divided by the stored mass its address hits
        import delta
        model.UnifiedBlock = delta.MassQueryGateAnchorDeltaBlock
    elif variant == "SDulc":  # SDul with the memory carried across consecutive training windows (gradient stopped)
        import sys
        import delta
        import fineweb
        flags = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
        if "batch_size" in flags and "gradient_accumulation_steps" in flags:   # training; evaluation needs neither
            delta.CARRY["batch"] = int(flags["batch_size"])
            delta.CARRY["accum"] = int(flags["gradient_accumulation_steps"])
            fineweb.PretokDataset.__iter__ = delta.carry_stream(fineweb.PretokDataset.__iter__)
        i = itertools.count()
        model.UnifiedBlock = lambda args: delta.CarryRawFFNDeltaBlock(args, first=next(i) == 0)
    elif variant == "SDb":  # SD with the memory emptied at every BOS
        import delta
        model.UnifiedBlock = delta.DocResetDeltaBlock
        model.UnifiedLM.forward = delta.doc_reset_forward
    elif variant == "SDv":  # SD with base horizons spread from 4 to 64 L and a gate bounded in (0, 2): diverse lambda
        import delta
        model.UnifiedBlock = delta.DiverseRateDeltaBlock
    elif variant in ("SDl100", "SDl1000"):  # SDl with 100x / 1000x
        import delta
        model.UnifiedBlock = delta.FastRate100DeltaBlock if variant == "SDl100" else delta.FastRate1000DeltaBlock
    elif variant == "SD2":  # SD plus a slow memory and a read mix
        import delta
        model.UnifiedBlock = delta.TwoDeltaBlock
    elif variant == "SD2c":  # SD2's control: two memories like SD's, same state as SD2, no fast/slow split
        import delta
        model.UnifiedBlock = delta.TwoFastDeltaBlock
    elif variant == "SDA":  # hybrid: after every 4 SD blocks a local attention half (window 128); n_layers counts both
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: WindowAttentionHalf(args) if next(i) % 5 == 4 else delta.DeltaBlock(args)
    elif variant == "GDN":  # baseline: T with Gated DeltaNet in place of attention
        import delta  # before UnifiedBlock is replaced: delta imports gated, which subclasses it
        i = itertools.count()
        model.UnifiedBlock = lambda args: GatedDeltaHalf(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "GDNs":  # GDN with the soft bound on log alpha; run with large heads (--head_dim=144)
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: GatedDeltaHalf(args, soft=True) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "MESA":  # baseline: T with the Mesa layer (exact least squares per token) in place of attention
        import mesa
        i = itertools.count()
        model.UnifiedBlock = lambda args: mesa.MesaHalf(args, HEAD_DIM) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "GDN2":  # baseline: T with Gated DeltaNet-2 in place of attention
        import delta
        i = itertools.count()
        model.UnifiedBlock = lambda args: GatedDeltaNet2Half(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "QHybrid":  # baseline: the Qwen3-Next / 3.5 / 3.6 layout, 3 GDN halves then 1 gated attention half, each before a SwiGLU half
        import delta
        i = itertools.count()

        def block(args):
            n = next(i)
            if n % 2:
                return FeedForwardHalf(args)
            return GatedAttentionHalf(args) if (n // 2) % 4 == 3 else GatedDeltaHalf(args)
        model.UnifiedBlock = block
    elif variant == "T":
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "TS":  # T with Scalable-Softmax in every attention half
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args, ssmax=True) if next(i) % 2 == 0 else FeedForwardHalf(args)
    elif variant == "TC":  # control: T with SD's causal conv before attention
        i = itertools.count()
        model.UnifiedBlock = lambda args: AttentionHalf(args, conv=True) if next(i) % 2 == 0 else FeedForwardHalf(args)
    else:
        raise ValueError(variant)
