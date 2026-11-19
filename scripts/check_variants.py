"""
CPU math checks for scripts/variants.py, no Triton or CUDA. Run from the repository root:

$ uv run --no-project --python 3.12 --with torch==2.10.0 --with sentencepiece --with numpy python scripts/check_variants.py

kernel.unified_memory is replaced by its dense quadratic definition, so this checks the math and the wiring,
not the Triton kernel. Kernel agreement still needs CUDA (test_all.py::test_kernel, plus the same check for each variant).
"""
import math
import os
import sys
import types

import torch

from dense_memory import dense_memory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

sys.modules["kernel"] = types.SimpleNamespace(CHUNK=64, unified_memory=dense_memory)

import test_all                                             # noqa: E402
import variants                                             # noqa: E402
from model import ModelArgs, UnifiedLM                      # noqa: E402
from variants import DecoupledBlock, SeparateBlock          # noqa: E402

TOL = 1e-10
SCALES = {"260K": (64, 140, 10, 512, 512), "15M": (288, 712, 12, 32000, 256)}
#          variant, m_s, m_f for each scale
CONFIGS = {
    "260K": [("S", 0, 0), ("D", 0, 0), ("P", 140, 140), ("P", 69, 140), ("P", 93, 93), ("P", 105, 70), ("A", 207, 207)],
    "15M": [("S", 0, 0), ("D", 0, 0), ("P", 712, 712), ("P", 355, 712), ("P", 474, 474), ("P", 534, 356), ("A", 1045, 1045)],
}


def build(variant, m_s=0, m_f=0, seed=0, nontrivial=True, **kw):
    variants.install(variant, m_s, m_f)
    torch.manual_seed(seed)
    args = {**dict(dim=32, n_layers=4, hidden_dim=48, vocab_size=97, max_seq_len=40), **kw}
    net = UnifiedLM(ModelArgs(**args)).eval()
    if nontrivial:
        with torch.no_grad():
            for name, p in net.named_parameters():
                if name.endswith("conv.weight"):
                    p.normal_(std=0.5)
                elif name.endswith(("key.bias", "key_s.bias", "key_f.bias")):
                    p.normal_(std=0.5)
                elif name.endswith("val.weight"):
                    p.normal_(std=0.2)
                elif name.endswith("log_rate"):
                    p.fill_(math.log(0.05))
                elif name.endswith("log_w"):
                    p.normal_(std=1.0)
    return net.double()


def logits(net, x):
    with torch.no_grad():
        return net(x, x)


def check_upstream_reference():
    for fn in (test_all.test_chunked_equals_dense, test_all.test_parallel_equals_recurrent,
               test_all.test_causality, test_all.test_gradients_finite, test_all.test_runc):
        fn()
    net = test_all.tiny_model().double()
    x = torch.randint(0, 97, (3, 40))
    with torch.no_grad():
        ref, _ = test_all.reference_forward(net, x, x, test_all.dense_mix)
        ours = net(x, x)
        rec = net.forward_recurrent(x)
    print(f"  S forward (dense stub) vs reference: {(ours - ref).abs().max():.2e}, vs recurrent: {(ours - rec).abs().max():.2e}")
    assert (ours - ref).abs().max() < TOL and (ours - rec).abs().max() < TOL


def check_param_counts():
    expected = {"260K": [259012, 259012, 350022, 258432, 258372, 258672, 258352],
                "15M": [15167352, 15167352, 17636580, 15164712, 15160428, 15165228, 15167340]}
    for scale, (d, m, L, V, T) in SCALES.items():
        base = None
        for (variant, ms, mf), want in zip(CONFIGS[scale], expected[scale]):
            variants.install(variant, ms, mf)
            net = UnifiedLM(ModelArgs(dim=d, n_layers=L, hidden_dim=m, vocab_size=V, max_seq_len=T))
            n = sum(p.numel() for p in net.parameters())
            base = base or n
            print(f"  {scale} {variant} m_s={ms:<5} m_f={mf:<5} params={n:>11,} ({100 * (n / base - 1):+.2f}%)")
            assert n == want, (scale, variant, ms, mf, n)
            assert (ms, mf) == (m, m) or abs(n / base - 1) < 0.005, "equal-budget variants stay within 0.5% of S"


def check_init():
    variants.install("A", 30, 30)
    torch.manual_seed(0)
    net = UnifiedLM(ModelArgs(dim=32, n_layers=4, hidden_dim=48, vocab_size=97, max_seq_len=40))
    a, b = net.layers[0], net.layers[1]
    assert a.m_s == 0 and a.m_f == 30 and b.m_s == 30 and b.m_f == 0, "A must alternate fast, static"
    assert (a.val.weight == 0).all(), "val starts at zero"
    assert (a.conv.weight[:, 0, -1] == 1).all() and a.conv.weight[:, 0, :-1].abs().sum() == 0, "conv starts as identity"
    assert abs(b.mu.weight.std().item() - 0.02 / math.sqrt(4)) < 0.003, "mu keeps the scaled init"
    print("  name-based init reaches SeparateBlock; A order is fast, static")


def check_s_to_d_limit():
    x = torch.randint(0, 97, (3, 40))
    s, d = build("S"), build("D")
    d.load_state_dict(s.state_dict())
    print(f"  normal w: |S - D| = {(logits(s, x) - logits(d, x)).abs().max():.2e}  (must be clearly nonzero)")
    assert (logits(s, x) - logits(d, x)).abs().max() > 1e-3
    prev = ratio = None
    for lw in (4.0, 8.0, 12.0):
        with torch.no_grad():
            for net in (s, d):
                for layer in net.layers:
                    layer.log_w.fill_(lw)
        e = (logits(s, x) - logits(d, x)).abs().max().item()
        ratio = prev / e if prev else None
        print(f"  log_w={lw:>4}: |S - D| = {e:.2e}" + (f"  (ratio {ratio:.1f}, e^4 = {math.exp(4):.1f} if error ~ 1/w)" if prev else ""))
        prev = e
    assert prev < 1e-3 and abs(ratio / math.exp(4) - 1) < 0.1, "error must scale as 1/w"


def check_d_refuses_recurrent():
    try:
        build("D").forward_recurrent(torch.randint(0, 97, (1, 4)))
    except NotImplementedError:
        print("  D refuses the inherited token-by-token form")
    else:
        raise AssertionError("D must not silently run S's recurrence")


def check_tied_p_equals_d():
    x = torch.randint(0, 97, (3, 40))
    d = build("D", seed=1)
    p = build("P", 48, 48, seed=2)
    print(f"  untied: |P - D| = {(logits(p, x) - logits(d, x)).abs().max():.2e}  (must be clearly nonzero)")
    assert (logits(p, x) - logits(d, x)).abs().max() > 1e-3
    with torch.no_grad():
        p.tok_embeddings.weight.copy_(d.tok_embeddings.weight)
        p.norm.weight.copy_(d.norm.weight)
        for a, b in zip(p.layers, d.layers):
            for suffix in ("key_s", "key_f"):
                getattr(a, suffix).weight.copy_(b.key.weight)
                getattr(a, suffix).bias.copy_(b.key.bias)
            a.log_eps_s.copy_(b.log_eps)
            a.log_eps_f.copy_(b.log_eps)
            a.norm.weight.copy_(b.norm.weight)
            a.conv.weight.copy_(b.conv.weight)
            a.mu.weight.copy_(b.mu.weight)
            a.val.weight.copy_(b.val.weight)
            a.log_rate.copy_(b.log_rate)
            a.log_w.copy_(b.log_w)
    e = (logits(p, x) - logits(d, x)).abs().max().item()
    print(f"  tied:   |P - D| = {e:.2e}")
    assert e < TOL


def check_causality_and_grads():
    x = torch.randint(0, 97, (3, 40))
    x2 = x.clone()
    x2[:, 25:] = torch.randint(0, 97, (3, 15))
    for variant, ms, mf in [("D", 0, 0), ("P", 48, 48), ("P", 30, 20), ("A", 30, 30)]:
        net = build(variant, ms, mf)
        leak = (logits(net, x)[:, :25] - logits(net, x2)[:, :25]).abs().max().item()
        assert leak < TOL, (variant, leak)
        net.train()
        net(x, x)
        net.last_loss.backward()
        bad = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
        assert not bad, (variant, bad)
        print(f"  {variant} m_s={ms} m_f={mf}: leak {leak:.1e}, every parameter has a finite nonzero gradient")


def check_gain_at_init():
    """r-weighted static gain w/(w+n) at init on random tokens, against the uniform-address estimate."""
    for scale, (d, m, L, V, T) in SCALES.items():
        variants.install("S")
        torch.manual_seed(0)
        net = UnifiedLM(ModelArgs(dim=d, n_layers=L, hidden_dim=m, vocab_size=V, max_seq_len=T)).eval()
        x = net.tok_embeddings(torch.randint(0, V, (2, T))).detach()
        rows = []
        with torch.no_grad():
            for layer in net.layers:
                h = layer.norm(x)
                u = layer.conv(h.transpose(1, 2))[..., :T].transpose(1, 2)
                r = layer.code(u)
                lam, w = torch.exp(-layer.log_rate.exp()), layer._w()
                n, gains = torch.zeros(2, m), []
                for t in range(T):
                    if t:
                        n = lam * n + r[:, t - 1]
                    gains.append((r[:, t] * w / (w + n)).sum(-1) / r[:, t].sum(-1))
                rows.append((r.sum(-1).mean().item(), gains[-1].mean().item(), torch.stack(gains).mean().item()))
                x = layer(x)
        lam = math.exp(-1 / (4 * T))
        uniform = 4 / (4 + sum(lam ** k for k in range(T - 1)) / m)
        s, last, avg = (sum(c) / L for c in zip(*rows))
        print(f"  {scale}: T/m={T / m:.2f} sum(r)={s:.2f} gain last={last:.3f} mean over t={avg:.3f}; uniform estimate last={uniform:.3f}")


def check_unused_q_grad():
    seen = {}

    class F2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a):
            return a * 2, a * 3

        @staticmethod
        def backward(ctx, d1, d2):
            seen["d1"] = d1
            return d1 * 2 + d2 * 3 if d1 is not None else d2 * 3

    a = torch.ones(2, requires_grad=True)
    _, y = F2.apply(a)
    y.sum().backward()
    d1 = seen["d1"]
    assert (d1 is None or bool((d1 == 0).all())) and a.grad.tolist() == [3.0, 3.0]
    print(f"  backward gets an unused first output as: {'None' if d1 is None else 'an all-zero tensor'}; gradient {a.grad.tolist()}")


def check_mac_counts():
    """Cost proxies, not FLOPs: matrix-sized operations per block with dense r, as a convention over run.c:228-255.
    Combined S = K, read U, write, decay U, restore prior (restore precomputed; as written it is one more multiply).
    A read-only mu needs K and a read. A context store needs K, read, write, decay.
    lazy: S_t = a_t H_t, a_t = lam a_{t-1}, H_t = H_{t-1} + v_t k_t^T / a_t removes the dense decay of a separate
    context store (periodic rescaling aside). The combined U cannot do this, since it also restores the prior."""
    tail = lambda d: d * d + 4 * d                                     # val + conv
    for scale, (d, m, L, V, T) in SCALES.items():
        s = 5 * d * m + tail(d)
        s_lazy = 4 * d * m + tail(d)                                   # split S or D: K, read mu, read H, write
        print(f"  {scale}: combined S {s:,}; split S or D {s:,}; split S or D with lazy decay {s_lazy:,}")
        for variant, ms, mf in CONFIGS[scale]:
            if variant in "SD":
                eager, lazy, state = s, s_lazy, L * d * m
            elif variant == "P":
                eager = 2 * d * ms + 4 * d * mf + tail(d)
                lazy = 2 * d * ms + 3 * d * mf + tail(d)
                state = L * d * mf
            else:
                eager = (4 * d * mf + tail(d) + 2 * d * ms + 4 * d) / 2
                lazy = (3 * d * mf + tail(d) + 2 * d * ms + 4 * d) / 2
                state = L // 2 * d * mf
            print(f"    {variant} m_s={ms:<5} m_f={mf:<5} eager {eager / s:.2f}x  lazy {lazy / s_lazy:.2f}x  matrix state {state / (L * d * m):.2f}x")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("check_"):
            print(name)
            fn()
    print("all checks passed")
