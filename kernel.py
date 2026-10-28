"""
Triton kernels for UnifiedBlock's fast memory (training path).

unified_memory(r, v, g, w) -> (q, fast), for every batch row and position t:

    k_t    = r_{t-1}                                     (k_0 = 0, write address)
    c_t    = w + sum_{s<=t} lam^(t-s) k_s                (slot evidence mass, lam = exp(g))
    q_t    = r_t / c_t
    fast_t = sum_{s<=t} lam^(t-s) (q_t . k_s) v_s

The fast read is a causal linear attention with key dimension = slots, value dimension = dim and one
scalar decay. The four products it needs (forward, dq, dk, dv) are the same chunked kernel `_la` with
roles swapped and time optionally reversed. Matmul operands are bf16 with fp32 accumulation; the slot
statistics c and q, and all gradients of them, stay fp32.
"""
import torch
import triton
import triton.language as tl

CHUNK = 64


@triton.autotune(configs=[triton.Config({"BK": bk, "BV": bv}, num_warps=nw, num_stages=ns)
                          for bk in (32, 64, 128) for bv in (32, 64, 128) for nw in (4, 8) for ns in (1, 2, 3)
                          if bk * bv <= 8192], key=["K", "V"])
@triton.jit
def _la_state_kernel(b_ptr, c_ptr, h_ptr, g_ptr, T, K, V, NT,
                     REVERSE: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """h_n = state carried into chunk n: sum over earlier chunks of lam^(distance) b_s c_s^T."""
    i_k, i_v, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g = tl.load(g_ptr).to(tl.float32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)
    m_k, m_v = o_k < K, o_v < V
    h = tl.zeros([BK, BV], dtype=tl.float32)
    for n in range(NT):
        tl.store(h_ptr + ((i_b * NT + n) * K + o_k[:, None]) * V + o_v[None, :], h.to(h_ptr.dtype.element_ty),
                 mask=m_k[:, None] & m_v[None, :])
        t = n * C + o_c
        m_t = t < T
        pos = tl.where(REVERSE, T - 1 - t, t)
        L = tl.minimum(C, T - n * C)
        bt = tl.load(b_ptr + (i_b * T + pos[:, None]) * K + o_k[None, :], mask=m_t[:, None] & m_k[None, :], other=0.)
        ct = tl.load(c_ptr + (i_b * T + pos[:, None]) * V + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.)
        tail = tl.where(m_t, tl.exp(g * (L - 1 - o_c).to(tl.float32)), 0.)
        bt = (bt.to(tl.float32) * tail[:, None]).to(ct.dtype)
        h = h * tl.exp(g * L.to(tl.float32)) + tl.dot(tl.trans(bt), ct)


@triton.autotune(configs=[triton.Config({"BK": bk}, num_warps=nw, num_stages=ns)
                          for bk in (32, 64, 128) for nw in (4, 8) for ns in (1, 2, 3)], key=["K"])
@triton.jit
def _la_attn_kernel(a_ptr, b_ptr, A_ptr, g_ptr, T, K, NT,
                    REVERSE: tl.constexpr, C: tl.constexpr, BK: tl.constexpr):
    """A_n[i,j] = lam^(i-j) (a_i . b_j) for j <= i inside chunk n."""
    i_n, i_b = tl.program_id(0), tl.program_id(1)
    g = tl.load(g_ptr).to(tl.float32)
    o_c = tl.arange(0, C)
    t = i_n * C + o_c
    m_t = t < T
    pos = tl.where(REVERSE, T - 1 - t, t)
    A = tl.zeros([C, C], dtype=tl.float32)
    for k0 in range(0, K, BK):
        o_k = k0 + tl.arange(0, BK)
        m = m_t[:, None] & (o_k < K)[None, :]
        at = tl.load(a_ptr + (i_b * T + pos[:, None]) * K + o_k[None, :], mask=m, other=0.)
        bt = tl.load(b_ptr + (i_b * T + pos[:, None]) * K + o_k[None, :], mask=m, other=0.)
        A += tl.dot(at, tl.trans(bt))
    lag = (o_c[:, None] - o_c[None, :]).to(tl.float32)
    A = tl.where(o_c[:, None] >= o_c[None, :], A * tl.exp(g * lag), 0.)
    tl.store(A_ptr + ((i_b * NT + i_n) * C + o_c[:, None]) * C + o_c[None, :], A)


@triton.autotune(configs=[triton.Config({"BK": bk, "BV": bv}, num_warps=nw, num_stages=ns)
                          for bk in (32, 64, 128) for bv in (32, 64, 128) for nw in (4, 8) for ns in (1, 2, 3)
                          if bk * bv <= 8192], key=["K", "V"])
@triton.jit
def _la_out_kernel(a_ptr, c_ptr, h_ptr, A_ptr, o_ptr, g_ptr, T, K, V, NT,
                   REVERSE: tl.constexpr, C: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    """o_t = lam^(i+1) a_t h_n + sum_{j<=i} A_n[i,j] c_j inside chunk n."""
    i_v, i_n, i_b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g = tl.load(g_ptr).to(tl.float32)
    o_c = tl.arange(0, C)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    t = i_n * C + o_c
    m_t = t < T
    pos = tl.where(REVERSE, T - 1 - t, t)
    acc = tl.zeros([C, BV], dtype=tl.float32)
    for k0 in range(0, K, BK):
        o_k = k0 + tl.arange(0, BK)
        m_k = o_k < K
        at = tl.load(a_ptr + (i_b * T + pos[:, None]) * K + o_k[None, :], mask=m_t[:, None] & m_k[None, :], other=0.)
        h = tl.load(h_ptr + ((i_b * NT + i_n) * K + o_k[:, None]) * V + o_v[None, :], mask=m_k[:, None] & m_v[None, :], other=0.)
        acc += tl.dot(at, h)
    acc = acc * tl.exp(g * (o_c + 1).to(tl.float32))[:, None]
    A = tl.load(A_ptr + ((i_b * NT + i_n) * C + o_c[:, None]) * C + o_c[None, :])
    ct = tl.load(c_ptr + (i_b * T + pos[:, None]) * V + o_v[None, :], mask=m_t[:, None] & m_v[None, :], other=0.)
    acc += tl.dot(A.to(ct.dtype), ct)
    tl.store(o_ptr + (i_b * T + pos[:, None]) * V + o_v[None, :], acc.to(o_ptr.dtype.element_ty), mask=m_t[:, None] & m_v[None, :])


@triton.jit
def _decay_combine(a1, x1, a2, x2):
    return a1 * a2, x1 * a2 + x2


@triton.jit
def _fwd_stats_kernel(r_ptr, w_ptr, g_ptr, q_ptr, c_ptr, q16_ptr, k16_ptr, T, M, NT,
                      C: tl.constexpr, BM: tl.constexpr):
    """k = shift(r), c = w + decayed cumsum of k, q = r / c; also bf16 copies of q and k."""
    i_m, i_b = tl.program_id(0), tl.program_id(1)
    lam = tl.exp(tl.load(g_ptr).to(tl.float32))
    o_c = tl.arange(0, C)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < M
    w = tl.load(w_ptr + o_m, mask=m_m, other=1.)
    decay = tl.full([C, BM], 1.0, tl.float32) * lam
    carry = tl.zeros([BM], dtype=tl.float32)
    for n in range(NT):
        t = n * C + o_c
        m = (t < T)[:, None] & m_m[None, :]
        off = (i_b * T + t[:, None]) * M + o_m[None, :]
        r = tl.load(r_ptr + off, mask=m, other=0.)
        k = tl.load(r_ptr + off - M, mask=m & (t > 0)[:, None], other=0.)
        x = tl.where(o_c[:, None] == 0, k + lam * carry[None, :], k)
        _, n_t = tl.associative_scan((decay, x), 0, _decay_combine)
        c = w[None, :] + n_t
        q = r / c
        tl.store(q_ptr + off, q, mask=m)
        tl.store(c_ptr + off, c, mask=m)
        tl.store(q16_ptr + off, q.to(tl.bfloat16), mask=m)
        tl.store(k16_ptr + off, k.to(tl.bfloat16), mask=m)
        L = tl.minimum(C, T - n * C)
        carry = tl.sum(tl.where((o_c == L - 1)[:, None], n_t, 0.), axis=0)


@triton.jit
def _bwd_stats_kernel(dqf_ptr, dqo_ptr, q_ptr, c_ptr, r_ptr, w_ptr, dk_ptr, g_ptr, dr_ptr, dkt_ptr, dg_ptr, dw_ptr,
                      T, M, NT, HAS_DQO: tl.constexpr, C: tl.constexpr, BM: tl.constexpr):
    """Backward through q = r / c and c = w + decayed cumsum of k, walking time in reverse:
    dr = dq / c, dc = -dq q / c, dkc = reverse decayed cumsum of dc, dk_total = dk + dkc,
    and the scalar d/dg and per-slot d/dw contributions (atomically accumulated)."""
    i_m, i_b = tl.program_id(0), tl.program_id(1)
    lam = tl.exp(tl.load(g_ptr).to(tl.float32))
    o_c = tl.arange(0, C)
    o_m = i_m * BM + tl.arange(0, BM)
    m_m = o_m < M
    w = tl.load(w_ptr + o_m, mask=m_m, other=0.)
    decay = tl.full([C, BM], 1.0, tl.float32) * lam
    carry = tl.zeros([BM], dtype=tl.float32)
    dg = tl.zeros([BM], dtype=tl.float32)
    dw = tl.zeros([BM], dtype=tl.float32)
    for n in range(NT):
        t = n * C + o_c
        pos = T - 1 - t
        m = (t < T)[:, None] & m_m[None, :]
        off = (i_b * T + pos[:, None]) * M + o_m[None, :]
        dqf = tl.load(dqf_ptr + off, mask=m, other=0.)
        q = tl.load(q_ptr + off, mask=m, other=0.)
        c = tl.load(c_ptr + off, mask=m, other=1.)
        k = tl.load(r_ptr + off - M, mask=m & (pos > 0)[:, None], other=0.)
        dk = tl.load(dk_ptr + off, mask=m, other=0.)
        dq = dqf
        if HAS_DQO:
            dq += tl.load(dqo_ptr + off, mask=m, other=0.)
        dc = -dq * q / c
        x = tl.where(o_c[:, None] == 0, dc + lam * carry[None, :], dc)
        _, dkc = tl.associative_scan((decay, x), 0, _decay_combine)
        tl.store(dr_ptr + off, dq / c, mask=m)
        tl.store(dkt_ptr + off, dk + dkc, mask=m)
        # d/dg of sum_{s<=t} lam^(t-s) (...) = sum_t t (.)_t - sum_s s (.)_s
        tpos = pos.to(tl.float32)[:, None]
        dg += tl.sum(tl.where(m, tpos * (q * dqf - k * dk + dc * (c - w[None, :]) - k * dkc), 0.), axis=0)
        dw += tl.sum(tl.where(m, dc, 0.), axis=0)
        L = tl.minimum(C, T - n * C)
        carry = tl.sum(tl.where((o_c == L - 1)[:, None], dkc, 0.), axis=0)
    tl.atomic_add(dg_ptr, tl.sum(dg, axis=0))
    tl.atomic_add(dw_ptr + o_m, dw, mask=m_m)


def _la(a, b, c, g, reverse=False, out_dtype=None):
    """o_t = sum_{s<=t} lam^(t-s) (a_t . b_s) c_s (s>=t when reverse). a, b: (B,T,K), c: (B,T,V), bf16."""
    B, T, K = a.shape
    V = c.size(-1)
    NT = triton.cdiv(T, CHUNK)
    h = torch.empty(B, NT, K, V, device=a.device, dtype=c.dtype)
    A = torch.empty(B, NT, CHUNK, CHUNK, device=a.device, dtype=torch.float32)
    o = torch.empty(B, T, V, device=a.device, dtype=out_dtype or c.dtype)
    _la_state_kernel[lambda m: (triton.cdiv(K, m["BK"]), triton.cdiv(V, m["BV"]), B)](
        b, c, h, g, T, K, V, NT, REVERSE=reverse, C=CHUNK)
    _la_attn_kernel[(NT, B)](a, b, A, g, T, K, NT, REVERSE=reverse, C=CHUNK)
    _la_out_kernel[lambda m: (triton.cdiv(V, m["BV"]), NT, B)](
        a, c, h, A, o, g, T, K, V, NT, REVERSE=reverse, C=CHUNK)
    return o


class UnifiedMemory(torch.autograd.Function):

    @staticmethod
    def forward(ctx, r, v, g, w):
        B, T, M = r.shape
        r = r.float().contiguous()
        v16 = v.to(torch.bfloat16).contiguous()
        g1 = g.float().reshape(1)
        q, c = torch.empty_like(r), torch.empty_like(r)
        q16, k16 = torch.empty_like(r, dtype=torch.bfloat16), torch.empty_like(r, dtype=torch.bfloat16)
        _fwd_stats_kernel[(triton.cdiv(M, 64), B)](r, w.float().contiguous(), g1, q, c, q16, k16, T, M,
                                                    triton.cdiv(T, CHUNK), C=CHUNK, BM=64)
        fast = _la(q16, k16, v16, g1, out_dtype=v.dtype)
        ctx.save_for_backward(r, v16, g1, w, q, c, q16, k16)
        return q, fast

    @staticmethod
    def backward(ctx, dq_out, dfast):
        r, v16, g1, w, q, c, q16, k16 = ctx.saved_tensors
        B, T, M = r.shape
        do16 = dfast.to(torch.bfloat16).contiguous()
        # fast read: fast = LA(q, k, v)
        dq = _la(do16, v16, k16, g1, out_dtype=torch.float32)
        dk = _la(v16, do16, q16, g1, reverse=True, out_dtype=torch.float32)
        dv = _la(k16, q16, do16, g1, reverse=True, out_dtype=dfast.dtype)
        # q = r / c, c = w + decayed cumsum of k (k = r shifted by one position)
        dr, dkt = torch.empty_like(r), torch.empty_like(r)
        dg = torch.zeros(1, device=r.device, dtype=torch.float32)
        dw = torch.zeros(M, device=r.device, dtype=torch.float32)
        dqo = dq_out.float().contiguous() if dq_out is not None else dq
        _bwd_stats_kernel[(triton.cdiv(M, 32), B)](dq, dqo, q, c, r, w.float().contiguous(), dk, g1, dr, dkt, dg, dw,
                                                    T, M, triton.cdiv(T, CHUNK), HAS_DQO=dq_out is not None,
                                                    C=CHUNK, BM=32)
        dr[:, :-1] += dkt[:, 1:]
        return dr, dv, dg.reshape(()), dw


def unified_memory(r, v, g, w):
    """r: (B,T,M) read addresses, v: (B,T,D) values, g: scalar log decay, w: (M,) prior mass."""
    return UnifiedMemory.apply(r, v, g, w)
