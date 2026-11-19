"""
Where does the kernel's log_rate gradient error come from? Needs CUDA. Run from the repository root:

$ python scripts/diag_log_rate.py

For check_kernel's models, the inputs and upstream gradients of every unified_memory call are captured from
the float64 dense model. Per call, d/dg is then computed exactly (float64 autograd), by the real kernel and by
the candidate fix (fp32 inputs), and by two float64 simulations of the kernel's formula
    dg = sum_t t (q.dqf - k.dk + dc (c - w) - k.dkc),
in which dqf and dk are built from bf16-rounded operands as in the kernel. "as kernel" pairs them with fp32
q and k, as kernel.py does; "consistent" pairs them with the same bf16 q and k. If "as kernel" reproduces the
kernel's error and "consistent" removes it, the error is the rounding mismatch weighted by absolute position t.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))

import check_kernel as ck          # noqa: E402
import kernel                      # noqa: E402
import kernel_dgfix                # noqa: E402
from dense_memory import dense_memory  # noqa: E402

CONFIGS = [("S", 0, 0, 64, 140, 512), ("D", 0, 0, 64, 140, 512), ("P", 93, 93, 64, 140, 512),
           ("P", 105, 70, 64, 140, 512), ("S", 0, 0, 64, 140, 200), ("S", 0, 0, 288, 712, 256)]
calls = []


def recorder(r, v, g, w):
    q, fast = dense_memory(r, v, g, w)
    q.retain_grad()
    fast.retain_grad()
    calls.append({"r": r.detach(), "v": v.detach(), "g": g.detach(), "w": w.detach(), "q": q, "fast": fast})
    return q, fast


def exact(c):
    g = c["g"].clone().requires_grad_()
    q, fast = dense_memory(c["r"], c["v"], g, c["w"])
    out = (fast * c["do"]).sum() + ((q * c["dqo"]).sum() if c["dqo"] is not None else 0)
    return torch.autograd.grad(out, g)[0].item()


def by_kernel(c, mod):
    g = c["g"].float().clone().requires_grad_()
    q, fast = mod.UnifiedMemory.apply(c["r"].float(), c["v"].float(), g, c["w"].float())
    outs, grads = [fast], [c["do"].float()]
    if c["dqo"] is not None:
        outs.append(q)
        grads.append(c["dqo"].float())
    torch.autograd.backward(outs, grads)
    return g.grad.item()


def simulated(c, consistent, rounding=True):
    r, v, g, w, do, dqo = c["r"], c["v"], c["g"], c["w"], c["do"], c["dqo"]
    T = r.size(1)
    t = torch.arange(T, device=r.device, dtype=r.dtype)
    lag = t[:, None] - t[None, :]
    D = (lag >= 0).to(r.dtype) * torch.exp(g * lag.clamp(min=0))
    k = F.pad(r, (0, 0, 1, 0))[:, :-1]
    c_ = w + D @ k
    q = r / c_
    bf = (lambda x: x.to(torch.bfloat16).to(r.dtype)) if rounding else (lambda x: x)
    q16, k16, v16, do16 = bf(q), bf(k), bf(v), bf(do)
    A = (do16 @ v16.transpose(1, 2)) * D             # [t, s] = lam^(t-s) (do_t . v_s)
    dqf = A @ k16                                    # kernel's dq = LA(do, v, k)
    dk = A.transpose(1, 2) @ q16                     # kernel's dk = LA(v, do, q, reverse)
    dq = dqf + (dqo if dqo is not None else 0)
    dc = -dq * q / c_
    dkc = D.transpose(0, 1) @ dc                     # reverse decayed cumsum of dc
    qq, kk = (q16, k16) if consistent else (q, k)
    return (t[None, :, None] * (qq * dqf - kk * dk + dc * (c_ - w) - k * dkc)).sum().item()


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    for config in CONFIGS:
        for rate in ("test", "train"):
            net = ck.build(*config)
            if rate == "train":
                with torch.no_grad():
                    for n, p in net.named_parameters():
                        if n.endswith("log_rate"):
                            p.fill_(math.log(1 / (4 * config[-1])))
            net = net.cuda().double().train()
            x = torch.randint(0, 512, (2, config[-1]), device="cuda")
            calls.clear()
            ck.set_memory(recorder)
            net(x, x)
            net.last_loss.backward()
            ck.set_memory(kernel.unified_memory)
            print(f"\n{config[0]} m_s={config[1]} m_f={config[2]} d={config[3]} T={config[5]} rate={rate}")
            for i, c in enumerate(calls):
                c["do"] = c["fast"].grad
                c["dqo"] = c["q"].grad
                ref = exact(c)
                print(f"  call {i}: dg {ref:+.3e}  formula in float64 {rel(simulated(c, False, rounding=False), ref):.1e}"
                      f"  kernel {rel(by_kernel(c, kernel), ref):.2e}  fix {rel(by_kernel(c, kernel_dgfix), ref):.2e}"
                      f"  simulated as kernel {rel(simulated(c, False), ref):.2e}  consistent {rel(simulated(c, True), ref):.2e}",
                      flush=True)
