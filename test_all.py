"""
Run simply with
$ python test_all.py
or, if pytest is installed,
$ pytest test_all.py

The reference tests run on CPU in float64. The kernel test needs CUDA and the run.c test
needs a compiled ./run (make run); each is skipped otherwise.
"""
import math
import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F

from export import model_export
from model import UnifiedLM, ModelArgs
from tokenizer import Tokenizer

TOL = 1e-8


# -----------------------------------------------------------------------------
# pure PyTorch reference forms of UnifiedBlock's mixing

def chunked_mix(block, x, r, k, v, chunk):
    """Chunkwise form: quadratic inside chunks, fast state carried between them."""
    B, T, m = r.shape
    lam = torch.exp(-block.log_rate.exp()).to(r.dtype)
    w = block._w()
    Up = r.new_zeros(B, m, v.size(-1))                 # sum_s lam^(t-s) k_s v_s^T
    cp = r.new_zeros(B, 1, m)                          # sum_s lam^(t-s) k_s
    idx = torch.arange(chunk, device=r.device, dtype=r.dtype)
    qs, ys = [], []
    for t0 in range(0, T, chunk):
        L = min(chunk, T - t0)
        i = idx[:L]
        Dc = (i[:, None] >= i[None, :]).to(r.dtype) * lam ** (i[:, None] - i[None, :]).clamp(min=0)
        din = (lam ** (i + 1))[None, :, None]          # carried state -> position t0+i
        kc, rc, vc = k[:, t0:t0 + L], r[:, t0:t0 + L], v[:, t0:t0 + L].to(r.dtype)
        q = rc / (w + din * cp + Dc @ kc)
        inter = (q * din) @ Up
        tail = (lam ** (L - 1 - i))[None, :, None]     # position t0+i -> chunk end
        Up = lam ** L * Up + (kc * tail).transpose(1, 2) @ vc
        cp = lam ** L * cp + (kc * tail).sum(1, keepdim=True)
        ys.append(inter + ((q @ kc.transpose(1, 2)) * Dc) @ vc)
        qs.append(q)
    q, fast = torch.cat(qs, 1), torch.cat(ys, 1)
    return block.mu((q * w).type_as(x)) + fast.type_as(x)


def dense_mix(block, x, r, k, v):
    """Quadratic (T x T) form."""
    T = r.size(1)
    t = torch.arange(T, device=r.device)
    D = (t[:, None] >= t[None, :]).to(r.dtype) * torch.exp(-block.log_rate.exp() * (t[:, None] - t[None, :]).clamp(min=0))
    w = block._w()
    q = r / (w + D @ k)
    return block.mu((q * w).type_as(x)) + (((q @ k.transpose(1, 2)) * D) @ v.to(r.dtype)).type_as(x)


def reference_forward(model, tokens, targets, mix):
    """UnifiedLM.forward with the block's mixing replaced by a reference form."""
    h = model.tok_embeddings(tokens)
    for layer in model.layers:
        hn = layer.norm(h)
        u = layer.conv(hn.transpose(1, 2))[..., : hn.size(1)].transpose(1, 2)
        r = layer.code(u)
        k = F.pad(r, (0, 0, 1, 0))[:, :-1]
        h = h + mix(layer, h, r, k, layer.val(hn))
    logits = model.output(model.norm(h))
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1))
    return logits, loss


def tiny_model(seed=0, **kw):
    """A small model with every path made non-trivial."""
    torch.manual_seed(seed)
    args = ModelArgs(**{**dict(dim=32, n_layers=3, hidden_dim=48, vocab_size=97, max_seq_len=40), **kw})
    model = UnifiedLM(args).eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.conv.weight.normal_(std=0.5)
            layer.key.bias.normal_(std=0.5)
            layer.val.weight.normal_(std=0.2)
            layer.log_rate.fill_(math.log(0.05))
            layer.log_w.normal_(std=1.0)
    return model


def chunked(chunk):
    return lambda block, x, r, k, v: chunked_mix(block, x, r, k, v, chunk)


# -----------------------------------------------------------------------------
# tests

def test_chunked_equals_dense():
    model = tiny_model().double()
    x = torch.randint(0, 97, (3, 40))
    with torch.no_grad():
        a, _ = reference_forward(model, x, x, chunked(7))
        b, _ = reference_forward(model, x, x, dense_mix)
    err = (a - b).abs().max().item()
    assert err < TOL, err


def test_parallel_equals_recurrent():
    model = tiny_model().double()
    x = torch.randint(0, 97, (3, 40))
    with torch.no_grad():
        a, _ = reference_forward(model, x, x, chunked(16))
        err = (a - model.forward_recurrent(x)).abs().max().item()
    assert err < TOL, err


def test_causality():
    model = tiny_model().double()
    x = torch.randint(0, 97, (3, 40))
    x2 = x.clone()
    x2[:, 25:] = torch.randint(0, 97, (3, 15))
    with torch.no_grad():
        a, _ = reference_forward(model, x, x, chunked(7))
        b, _ = reference_forward(model, x2, x2, chunked(7))
    leak = (a[:, :25] - b[:, :25]).abs().max().item()
    assert leak < TOL, leak


def test_gradients_finite():
    model = tiny_model().double().train()
    x = torch.randint(0, 97, (3, 40))
    _, loss = reference_forward(model, x, x, chunked(7))
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def kernel_vs_reference(dim=64, n_layers=2, hidden_dim=140, T=200, B=2, dtype=torch.float32):
    """Loss and every parameter gradient of UnifiedLM.forward against the float64 reference."""
    model = tiny_model(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim, vocab_size=512, max_seq_len=T).cuda().train()
    ref = tiny_model(dim=dim, n_layers=n_layers, hidden_dim=hidden_dim, vocab_size=512, max_seq_len=T).cuda().double().train()
    x = torch.randint(0, 512, (B, T), device="cuda")
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        model(x, x)
    model.last_loss.backward()
    _, loss = reference_forward(ref, x, x, chunked(64))
    loss.backward()
    out = {"loss": abs(model.last_loss.item() - loss.item()) / abs(loss.item())}
    for (name, p), (_, q) in zip(model.named_parameters(), ref.named_parameters()):
        out[name.split(".", 2)[-1]] = max(out.get(name.split(".", 2)[-1], 0.0),
                                          ((p.grad.double() - q.grad).norm() / q.grad.norm().clamp(min=1e-30)).item())
    return out


def test_kernel():
    """The training kernel against the float64 reference: loss and every parameter gradient.
    The kernel computes in float32 with TF32 matmuls inside, hence the 1e-2 relative tolerance."""
    if not torch.cuda.is_available():
        print("skipping test_kernel: needs CUDA")
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    rel = kernel_vs_reference()
    worst = max(rel.values())
    assert worst < 1e-2, rel


def test_runc():
    """Greedy text from ./run must match the Python token-by-token reference."""
    if not os.path.exists("run"):
        print("skipping test_runc: build ./run with `make run` first")
        return
    torch.manual_seed(0)
    model = UnifiedLM(ModelArgs(dim=32, n_layers=2, hidden_dim=48, vocab_size=512, max_seq_len=64)).eval()
    with torch.no_grad():
        for layer in model.layers:
            layer.val.weight.normal_(std=0.2)
            layer.conv.weight.normal_(std=0.5)
    enc = Tokenizer("tok512.model")
    prompt = "Once upon a time"
    tokens = enc.encode(prompt, bos=True, eos=False)
    steps = 40
    while len(tokens) < steps + 1:  # ./run emits `steps` tokens after BOS
        logits = model.forward_recurrent(torch.tensor([tokens]))[0, -1]
        tokens.append(int(logits.argmax()))
    expected = enc.decode(tokens[1:])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.bin")
        model_export(model, path)
        out = subprocess.run(["./run", path, "-z", "tok512.bin", "-t", "0", "-n", str(steps), "-i", prompt],
                             capture_output=True, text=True, check=True).stdout
    assert out.strip() == expected.strip(), (out, expected)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
