"""
The real Triton kernel against the dense float64 definition, for every variant and slot width the experiment
uses: loss and every parameter gradient, in fp32 and bf16, eager and compiled, plus one optimizer update.
Needs CUDA. Run from the repository root:

$ python scripts/check_kernel.py

The gate was amended after preflight 1, with Gary's approval (scripts/reports/preflight-1.md). The control is the
dense definition at the same precision and in the same compile mode, with no Triton, measured against float64 too.

1. Loss and whole-gradient relative error at most TOL; in bf16, whole-gradient error at most 1.10 x the control's.
2. In each parameter holding at least 0.1% of the total gradient norm, error at most TOL above the control's.
   log_rate is left out of this and printed on its own, with its known kernel-specific error.
3. A variant's whole-gradient error over its control's, at most 1.10 x the same ratio for S at the same d, T and
   mode (adopted by Gary after preflight 2, where the raw ratio failed two bf16 eager cases whose control was
   just as far off). Per-parameter excess is printed for comparison but not gated by ratio.
"""
import copy
import math
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

import kernel                      # noqa: E402
import model                       # noqa: E402
import variants                    # noqa: E402
from dense_memory import dense_memory  # noqa: E402
from model import ModelArgs, UnifiedLM  # noqa: E402

TOL = 1e-2
MARGIN = 1.10
MIN_SHARE = 1e-3
#         variant, m_s, m_f, d, m, T
CONFIGS = [
    ("S", 0, 0, 64, 140, 512), ("D", 0, 0, 64, 140, 512), ("P", 140, 140, 64, 140, 512), ("P", 69, 140, 64, 140, 512),
    ("P", 93, 93, 64, 140, 512), ("P", 105, 70, 64, 140, 512), ("A", 207, 207, 64, 140, 512),
    ("S", 0, 0, 288, 712, 256), ("D", 0, 0, 288, 712, 256), ("P", 712, 712, 288, 712, 256), ("P", 355, 712, 288, 712, 256),
    ("P", 474, 474, 288, 712, 256), ("P", 534, 356, 288, 712, 256), ("A", 1045, 1045, 288, 712, 256),
    # partial last chunk
    ("S", 0, 0, 64, 140, 200), ("D", 0, 0, 64, 140, 200), ("P", 105, 70, 64, 140, 200), ("A", 207, 207, 64, 140, 200),
]
MODES = [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)]


def set_memory(fn):
    model.unified_memory = fn
    variants.unified_memory = fn


def build(variant, m_s, m_f, d, m, T):
    variants.install(variant, m_s, m_f)
    torch.manual_seed(0)
    net = UnifiedLM(ModelArgs(dim=d, n_layers=4, hidden_dim=m, vocab_size=512, max_seq_len=T))
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
    return net


def grads(net, x, dtype, compiled, memory):
    set_memory(memory)
    torch._dynamo.reset()  # otherwise the recompile limit silently turns later "compiled" cases into eager ones
    run = torch.compile(net) if compiled else net
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        run(x, x)
        loss = run.last_loss
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    return run, loss.item(), {name: p.grad.double().clone() for name, p in net.named_parameters()}


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


def check(config, dtype, compiled):
    net = build(*config)
    ref = copy.deepcopy(net).cuda().double().train()
    ctrl = copy.deepcopy(net).cuda().train()
    net = net.cuda().train()
    x = torch.randint(0, 512, (2, config[-1]), device="cuda")

    _, loss0, g0 = grads(ref, x, torch.float32, False, dense_memory)
    _, _, gc = grads(ctrl, x, dtype, compiled, dense_memory)
    run, loss, gk = grads(net, x, dtype, compiled, kernel.unified_memory)

    all0 = torch.cat([g.flatten() for g in g0.values()])
    allk = torch.cat([gk[n].flatten() for n in g0])
    allc = torch.cat([gc[n].flatten() for n in g0])
    r = {
        "loss": abs(loss - loss0) / abs(loss0),
        "whole": rel(allk, all0),
        "whole_ctrl": rel(allc, all0),
        "cosine": torch.cosine_similarity(allk, all0, dim=0).item(),
        "excess": (float("-inf"), "none"),
        "raw": (0.0, "none", 0.0),
        "log_rate": (0.0, 0.0, 0.0),
    }
    total = all0.norm().item()
    for name in g0:
        key = name.split(".", 2)[-1]
        err, err_ctrl, share = rel(gk[name], g0[name]), rel(gc[name], g0[name]), g0[name].norm().item() / total
        r["raw"] = max(r["raw"], (err, key, share))
        if key == "log_rate":
            r["log_rate"] = max(r["log_rate"], (err, err_ctrl, share))
        elif share >= MIN_SHARE:
            r["excess"] = max(r["excess"], (err - err_ctrl, key))

    # one update through the real optimizer must stay finite and move the weights
    before = {n: p.detach().clone() for n, p in net.named_parameters()}
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, fused=True)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    assert all(torch.isfinite(p).all() for p in net.parameters())
    assert any((p != before[n]).any() for n, p in net.named_parameters())
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        run(x, x)
    assert math.isfinite(run.last_loss.item())
    return r


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    failed = 0
    yardstick = {}
    for config in CONFIGS:
        for dtype, compiled in MODES:
            r = check(config, dtype, compiled)
            shape = (config[3], config[5], dtype, compiled)
            if config[0] == "S":
                yardstick[shape] = r["whole"] / r["whole_ctrl"]
            why = []
            if r["loss"] > TOL:
                why.append("loss")
            if r["whole"] > TOL:
                why.append("whole")
            if dtype != torch.float32 and r["whole"] > MARGIN * r["whole_ctrl"]:
                why.append("control")
            if r["excess"][0] > TOL:
                why.append("group")
            if r["whole"] / r["whole_ctrl"] > MARGIN * yardstick[shape]:
                why.append("vs_S")
            failed += bool(why)
            print(f"{'ok  ' if not why else 'FAIL ' + ','.join(why)} {config[0]} m_s={config[1]:<5} m_f={config[2]:<5} d={config[3]:<4} T={config[5]:<4} "
                  f"{str(dtype).split('.')[-1]:<8} compiled={compiled!s:<5} "
                  f"whole {r['whole']:.2e} ctrl {r['whole_ctrl']:.2e} ratio/S {r['whole'] / r['whole_ctrl'] / yardstick[shape]:.3f} cos {r['cosine']:.6f} loss {r['loss']:.2e} "
                  f"excess {r['excess'][0]:.2e} ({r['excess'][1]}) "
                  f"log_rate {r['log_rate'][0]:.2e} ctrl {r['log_rate'][1]:.2e} share {r['log_rate'][2]:.1e} "
                  f"raw {r['raw'][0]:.2e} ({r['raw'][1]}, share {r['raw'][2]:.1e})", flush=True)
    print(f"{failed} failures")
    sys.exit(1 if failed else 0)
