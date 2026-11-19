"""
Checks for SG (scripts/gated.py). Runs on CPU in float64, plus the whole model on CUDA when available.

1. gated_memory (chunked) against the token-by-token recurrence of the docstring, U and c carried explicitly:
   q, fast and the gradients of r, v, logl, w, for random per-slot, per-token decays, T not a multiple of the chunk.
2. With a decay that is constant over slots and tokens, gated_memory equals S's dense definition (dense_memory).
3. A whole SG model: the parallel forward (chunked) against step-by-step recurrence through every layer, loss and
   every parameter gradient (float64 CPU).
4. CUDA only: the SG model in fp32 and in bf16 autocast, eager and compiled, against float64; loss and gradient error.

$ python scripts/check_gated.py
"""
import copy
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import model                            # noqa: E402
import variants                         # noqa: E402
from dense_memory import dense_memory   # noqa: E402
from gated import GatedBlock, gated_memory  # noqa: E402
from model import ModelArgs, UnifiedLM   # noqa: E402

FAIL = []


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def report(name, err, tol):
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


def recurrent_memory(r, v, logl, w):
    B, T, m = r.shape
    d = v.size(-1)
    U = torch.zeros(B, d, m, dtype=r.dtype)
    c = torch.zeros(B, m, dtype=r.dtype)
    k = torch.zeros(B, m, dtype=r.dtype)
    qs, fs = [], []
    for t in range(T):
        lam = logl[:, t].exp()
        U = lam[:, None, :] * U + v[:, t, :, None] * k[:, None, :]
        c = lam * c + k
        q = r[:, t] / (w + c)
        qs.append(q)
        fs.append((U @ q[:, :, None]).squeeze(-1))
        k = r[:, t]
    return torch.stack(qs, 1), torch.stack(fs, 1)


def memory_checks():
    torch.manual_seed(0)
    B, T, m, d = 2, 150, 24, 8
    r = F.relu(torch.randn(B, T, m, dtype=torch.float64)).square()
    r = r / (1 + r.sum(-1, keepdim=True))
    v = torch.randn(B, T, d, dtype=torch.float64)
    logl = torch.expm1(-torch.rand(B, T, m, dtype=torch.float64) * torch.logspace(-3, 1, m, dtype=torch.float64))
    w = torch.rand(m, dtype=torch.float64) * 5 + 0.5
    ins = [x.clone().requires_grad_() for x in (r, v, logl, w)]
    for chunk in (16, 64):
        a = [x.detach().clone().requires_grad_() for x in ins]
        b = [x.detach().clone().requires_grad_() for x in ins]
        qa, fa = gated_memory(*a, chunk=chunk)
        qb, fb = recurrent_memory(*b)
        report(f"chunk {chunk}: q", rel(qa, qb), 1e-10)
        report(f"chunk {chunk}: fast", rel(fa, fb), 1e-10)
        gq, gf = torch.randn_like(qa), torch.randn_like(fa)
        ((qa * gq).sum() + (fa * gf).sum()).backward()
        ((qb * gq).sum() + (fb * gf).sum()).backward()
        for name, x, y in zip("r v logl w".split(), a, b):
            report(f"chunk {chunk}: d{name}", rel(x.grad, y.grad), 1e-10)
    g = torch.tensor(-0.05, dtype=torch.float64)
    q1, f1 = gated_memory(r, v, torch.full_like(r, g.item()), w)
    q2, f2 = dense_memory(r, v, g, w)
    report("constant decay = S dense: q", rel(q1, q2), 1e-10)
    report("constant decay = S dense: fast", rel(f1, f2), 1e-10)


def build(d=32, m=48, T=150, layers=3, seed=0):
    model.UnifiedBlock = GatedBlock
    torch.manual_seed(seed)
    net = UnifiedLM(ModelArgs(dim=d, n_layers=layers, hidden_dim=m, vocab_size=97, max_seq_len=T))
    with torch.no_grad():  # move off the near-S initialisation so every term is exercised
        for name, p in net.named_parameters():
            if name.endswith(("conv.weight", "val.weight", "gate_p.weight", "gate_q.weight")):
                p.normal_(std=0.3)
            elif name.endswith("log_rate"):
                p.uniform_(-5, 0)
            elif name.endswith("log_beta"):
                p.fill_(0.0)
    return net


def step_forward(net, tokens):
    """Token by token through every layer: the conv, the gate and the state update all causal by construction."""
    B, T = tokens.shape
    x = net.tok_embeddings(tokens)
    for layer in net.layers:
        d, m = layer.mu.out_features, layer.mu.in_features
        h = layer.norm(x)
        u = layer.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)
        r = layer.code(u)
        v = layer.val(h)
        w = layer._w()
        ld = layer.log_decay(r)  # log_decay uses r_t and k_t = r_{t-1} only
        U = (layer.mu.weight * w).expand(B, d, m).clone()
        c = w.expand(B, m).clone()
        kprev = torch.zeros(B, m, dtype=x.dtype)
        ys = []
        for t in range(T):
            lam = ld[:, t].exp()
            U = lam[:, None, :] * U + (1 - lam)[:, None, :] * layer.mu.weight * w + v[:, t, :, None] * kprev[:, None, :]
            c = lam * c + (1 - lam) * w + kprev
            ys.append((U @ (r[:, t] / c)[:, :, None]).squeeze(-1))
            kprev = r[:, t]
        x = x + torch.stack(ys, 1)
    return net.output(net.norm(x))


def model_checks():
    net = build().double()
    tokens = torch.randint(0, 97, (2, 150))
    targets = torch.randint(0, 97, (2, 150))
    ref = copy.deepcopy(net)
    loss_a = net(tokens, targets)
    loss_a = net.last_loss
    loss_a.backward()
    logits = step_forward(ref, tokens)
    loss_b = F.cross_entropy(logits.view(-1, 97), targets.view(-1))
    loss_b.backward()
    report("model: loss parallel vs recurrent", abs(loss_a.item() - loss_b.item()) / loss_b.item(), 1e-10)
    for (n, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
        report(f"model: grad {n}", rel(p.grad, q.grad), 1e-8)


def cuda_checks():
    if not torch.cuda.is_available():
        print("no CUDA: skipping the precision checks")
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    net64 = build(d=288, m=712, T=1024, layers=2).double().cuda()
    tokens = torch.randint(0, 97, (4, 1024), device="cuda")
    targets = torch.randint(0, 97, (4, 1024), device="cuda")
    net64(tokens, targets)
    net64.last_loss.backward()
    ref = {n: p.grad.clone() for n, p in net64.named_parameters()}
    ref_loss = net64.last_loss.item()
    ref_all = torch.cat([g.flatten() for g in ref.values()])
    for dtype, comp in [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]:
        net = build(d=288, m=712, T=1024, layers=2).cuda()
        net.load_state_dict({k: v.float() for k, v in net64.state_dict().items()})
        f = torch.compile(net) if comp else net
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            f(tokens, targets)
            loss = net.last_loss if not comp else f._orig_mod.last_loss
        loss.backward()
        g = torch.cat([p.grad.double().flatten() for p in net.parameters()])
        tag = f"{str(dtype).split('.')[-1]} {'compiled' if comp else 'eager'}"
        report(f"cuda {tag}: loss", abs(loss.item() - ref_loss) / ref_loss, 1e-2)
        report(f"cuda {tag}: whole gradient", rel(g, ref_all), 5e-2 if dtype == torch.bfloat16 else 1e-2)


if __name__ == "__main__":
    memory_checks()
    model_checks()
    cuda_checks()
    print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)
