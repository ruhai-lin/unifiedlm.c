"""
Checks for SD and SD2 (scripts/delta.py). Runs on CPU in float64, plus whole models on CUDA when available.

1. delta_memory (chunked) against the token-by-token delta rule of its docstring, E carried explicitly: the output
   and the gradients of k, u, beta, logl, q, for random per-slot, per-token decays, T not a multiple of the chunk.
2. With beta = 1 and a single repeated key, the read at that key returns the last target (the delta rule overwrites).
3. Whole SD and SD2 models: the parallel forward against a step-by-step recurrence through every layer, loss and
   every parameter gradient (float64 CPU).
4. CUDA only: SD and SD2 in fp32 and in bf16 autocast, eager and compiled, against float64; loss and gradient error.

$ python scripts/check_delta.py
"""
import copy
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import model                                         # noqa: E402
from delta import DeltaBlock, TwoDeltaBlock, delta_memory, _unit  # noqa: E402
from model import ModelArgs, UnifiedLM                # noqa: E402

FAIL = []


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def report(name, err, tol):
    ok = err <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


def recurrent_delta(k, u, beta, logl, q):
    B, T, m = k.shape
    E = torch.zeros(B, m, u.size(-1), dtype=k.dtype)
    out = []
    for t in range(T):
        E = logl[:, t].exp()[:, :, None] * E
        pred = (E * k[:, t, :, None]).sum(1)                       # E^T k
        E = E + beta[:, t, None, None] * k[:, t, :, None] * (u[:, t] - pred)[:, None, :]
        out.append((E * q[:, t, :, None]).sum(1))                 # E^T q
    return torch.stack(out, 1)


def memory_checks():
    torch.manual_seed(0)
    B, T, m, d = 2, 150, 24, 8
    k = _unit(F.relu(torch.randn(B, T, m, dtype=torch.float64)).square())
    q = _unit(F.relu(torch.randn(B, T, m, dtype=torch.float64)).square())
    u = torch.randn(B, T, d, dtype=torch.float64)
    beta = torch.rand(B, T, dtype=torch.float64)
    logl = torch.expm1(-torch.rand(B, T, m, dtype=torch.float64) * torch.logspace(-3, 0.5, m, dtype=torch.float64))
    ins = [k, u, beta, logl, q]
    for chunk in (16, 64):
        a = [x.detach().clone().requires_grad_() for x in ins]
        b = [x.detach().clone().requires_grad_() for x in ins]
        oa = delta_memory(*a, chunk=chunk)
        ob = recurrent_delta(*b)
        report(f"chunk {chunk}: out", rel(oa, ob), 1e-10)
        g = torch.randn_like(oa)
        (oa * g).sum().backward()
        (ob * g).sum().backward()
        for name, x, y in zip("k u beta logl q".split(), a, b):
            report(f"chunk {chunk}: d{name}", rel(x.grad, y.grad), 1e-10)
    key = _unit(torch.rand(1, 1, m, dtype=torch.float64)).expand(1, 100, m)
    tgt = torch.randn(1, 100, d, dtype=torch.float64)
    o = delta_memory(key, tgt, torch.ones(1, 100, dtype=torch.float64), torch.zeros(1, 100, m, dtype=torch.float64), key)
    report("beta 1, one key: read = last target", rel(o[0, -1], tgt[0, -1]), 1e-10)


def build(block, d=32, m=48, T=150, layers=3, seed=0):
    model.UnifiedBlock = block
    torch.manual_seed(seed)
    net = UnifiedLM(ModelArgs(dim=d, n_layers=layers, hidden_dim=m, vocab_size=97, max_seq_len=T))
    with torch.no_grad():  # move off the initialisation so every term is exercised
        for name, p in net.named_parameters():
            if name.endswith(("conv.weight", "val.weight", "gate_p.weight", "gate_q.weight", "beta.weight", "beta_slow.weight", "rho.weight")):
                p.normal_(std=0.3)
            elif name.endswith(("log_rate", "log_rate_slow")):
                p.uniform_(-5, 0)
    return net


def step_forward(net, tokens):
    """Token by token through every layer, E carried explicitly (fast, and slow for SD2)."""
    B, T = tokens.shape
    x = net.tok_embeddings(tokens)
    for layer in net.layers:
        d, m = layer.mu.out_features, layer.mu.in_features
        h = layer.norm(x)
        u = layer.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)
        r = layer.code(u)
        v = layer.val(h)
        ld = layer.log_decay(r)
        two = isinstance(layer, TwoDeltaBlock)
        mems = [(ld, torch.sigmoid(layer.beta(h)).squeeze(-1))]
        if two:
            mems.append((torch.expm1(-layer.log_rate_slow.exp()).expand_as(r), torch.sigmoid(layer.beta_slow(h)).squeeze(-1)))
        Es = [torch.zeros(B, m, d, dtype=x.dtype) for _ in mems]
        kprev = torch.zeros(B, m, dtype=x.dtype)
        ys = []
        for t in range(T):
            kh = _unit(kprev)
            target = v[:, t] - kprev @ layer.mu.weight.T
            reads = []
            for i, (logl, beta) in enumerate(mems):
                E = logl[:, t].exp()[:, :, None] * Es[i]
                pred = (E * kh[:, :, None]).sum(1)
                Es[i] = E + beta[:, t, None, None] * kh[:, :, None] * (target - pred)[:, None, :]
                reads.append((Es[i] * _unit(r[:, t])[:, :, None]).sum(1))
            mem = reads[0]
            if two:
                rho = torch.sigmoid(layer.rho(h[:, t]))
                mem = (1 - rho) * reads[0] + rho * reads[1]
            ys.append(r[:, t] @ layer.mu.weight.T + mem)
            kprev = r[:, t]
        x = x + torch.stack(ys, 1)
    return net.output(net.norm(x))


def model_checks(block, tag):
    net = build(block).double()
    tokens = torch.randint(0, 97, (2, 150))
    targets = torch.randint(0, 97, (2, 150))
    ref = copy.deepcopy(net)
    net(tokens, targets)
    loss_a = net.last_loss
    loss_a.backward()
    logits = step_forward(ref, tokens)
    loss_b = F.cross_entropy(logits.view(-1, 97), targets.view(-1))
    loss_b.backward()
    report(f"{tag} model: loss parallel vs recurrent", abs(loss_a.item() - loss_b.item()) / loss_b.item(), 1e-10)
    for (n, p), (_, q) in zip(net.named_parameters(), ref.named_parameters()):
        report(f"{tag} model: grad {n}", rel(p.grad, q.grad), 1e-8)


def cuda_checks(block, tag):
    if not torch.cuda.is_available():
        print("no CUDA: skipping the precision checks")
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    net64 = build(block, d=288, m=712, T=1024, layers=2).double().cuda()
    tokens = torch.randint(0, 97, (4, 1024), device="cuda")
    targets = torch.randint(0, 97, (4, 1024), device="cuda")
    net64(tokens, targets)
    net64.last_loss.backward()
    ref_loss = net64.last_loss.item()
    ref_all = torch.cat([p.grad.flatten() for p in net64.parameters()])
    for dtype, comp in [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]:
        net = build(block, d=288, m=712, T=1024, layers=2).cuda()
        net.load_state_dict({k: v.float() for k, v in net64.state_dict().items()})
        f = torch.compile(net) if comp else net
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            f(tokens, targets)
            loss = net.last_loss if not comp else f._orig_mod.last_loss
        loss.backward()
        g = torch.cat([p.grad.double().flatten() for p in net.parameters()])
        name = f"{tag} cuda {str(dtype).split('.')[-1]} {'compiled' if comp else 'eager'}"
        report(f"{name}: loss", abs(loss.item() - ref_loss) / ref_loss, 1e-2)
        report(f"{name}: whole gradient", rel(g, ref_all), 5e-2 if dtype == torch.bfloat16 else 1e-2)


if __name__ == "__main__":
    memory_checks()
    for block, tag in ((DeltaBlock, "SD"), (TwoDeltaBlock, "SD2")):
        model_checks(block, tag)
        cuda_checks(block, tag)
    print("ALL OK" if not FAIL else f"{len(FAIL)} FAILED: {FAIL}")
    sys.exit(1 if FAIL else 0)
