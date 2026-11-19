"""The dense quadratic definition of kernel.unified_memory, used as the reference for the Triton kernel."""
import torch
import torch.nn.functional as F


def dense_memory(r, v, g, w):
    """q = r / c, c = w + sum_{s<=t} lam^(t-s) k_s, fast_t = sum_{s<=t} lam^(t-s) (q_t . k_s) v_s, k_t = r_{t-1}."""
    t = torch.arange(r.size(1), device=r.device)
    lag = (t[:, None] - t[None, :]).clamp(min=0).to(r.dtype)
    D = (t[:, None] >= t[None, :]).to(r.dtype) * torch.exp(g.to(r.dtype) * lag)
    k = F.pad(r, (0, 0, 1, 0))[:, :-1]
    q = r / (w.to(r.dtype) + D @ k)
    return q, ((q @ k.transpose(1, 2)) * D) @ v.to(r.dtype)
