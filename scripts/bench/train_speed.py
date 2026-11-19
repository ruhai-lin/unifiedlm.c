"""Training throughput and peak memory of S and T (BENCH_VARIANTS="S T SG P" adds the gated S and P-free) at the 15M / 42M / 110M widths, on random tokens.
One full step = forward, backward, grad clip, fused AdamW, bf16 autocast (BENCH_DTYPE=float16 for fp16), torch.compile, as in train.py.
Each (variant, size, micro-batch) runs in its own process so an OOM cannot affect the next one; the
micro-batch doubles until it no longer fits. Run from the repository root:

$ python -u scripts/bench/train_speed.py [15M 42M 110M] > speed.jsonl
"""
import json
import os
import subprocess
import sys
import time

# dim, UnifiedBlocks, S slots (docs/models.md), T SwiGLU hidden (llama2.c), T head dim (llama2.c)
SIZES = {"15M": (288, 12, 712, 768, 48), "42M": (512, 16, 1280, 1376, 64), "110M": (768, 24, 1912, 2048, 64)}
SEQ, VOCAB, STEPS, WARM = 1024, 32000, 20, 5
DTYPE = os.environ.get("BENCH_DTYPE", "bfloat16")  # float16 on cards without bf16 (T4)


def one(variant, size, bs):
    """BENCH_BS="15M:16,42M:8" runs only that micro-batch per size instead of doubling until OOM."""
    import torch
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
    import model
    import variants
    dim, n_layers, slots, ffn, head_dim = SIZES[size]
    variants.HEAD_DIM = head_dim
    variants.install(variant, slots, slots)  # m_s, m_f: P-free gives each path all the slots
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    net = model.UnifiedLM(model.ModelArgs(dim=dim, n_layers=n_layers, hidden_dim=ffn if variant == "T" else slots,
                                          vocab_size=VOCAB, max_seq_len=SEQ)).cuda()
    params = sum(p.numel() for p in net.parameters())
    opt = net.configure_optimizers(0.1, 5e-4, (0.9, 0.95), "cuda")
    cnet = torch.compile(net)
    x = torch.randint(0, VOCAB, (bs, SEQ + 1), device="cuda")
    X, Y = x[:, :-1].contiguous(), x[:, 1:].contiguous()
    t_start = time.time()
    for i in range(WARM + STEPS):
        if i == WARM:
            torch.cuda.synchronize()
            t0 = time.time()
            warmup = t0 - t_start
        with torch.amp.autocast("cuda", dtype=getattr(torch, DTYPE)):
            cnet(X, Y)
        net.last_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / STEPS
    return {"variant": variant, "size": size, "params": params, "bs": bs, "seq": SEQ, "ms_per_step": round(dt * 1e3, 1),
            "tok_per_s": round(bs * SEQ / dt), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "warmup_s": round(warmup, 1), "gpu": torch.cuda.get_device_name(), "dtype": DTYPE, "loss": round(net.last_loss.item(), 3)}


def main(sizes):
    import torch
    print(json.dumps({"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cpus": len(os.sched_getaffinity(0))}),
          flush=True)
    plan = [(v, s) for s in sizes for v in os.environ.get("BENCH_VARIANTS", "S T").split()]
    fixed = dict(kv.split(":") for kv in os.environ.get("BENCH_BS", "").split(",") if kv)  # e.g. 15M:16,42M:8
    for variant, size in plan:
        bs = int(fixed.get(size, 8))
        top = bs if size in fixed else 256
        while bs <= top:
            r = subprocess.run([sys.executable, "-u", __file__, variant, size, str(bs)], capture_output=True, text=True)
            line = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
            if line is None:
                oom = "OutOfMemory" in r.stderr or "out of memory" in r.stderr
                print(json.dumps({"variant": variant, "size": size, "bs": bs, "failed": "oom" if oom else r.stderr[-600:]}),
                      flush=True)
                break
            print(line, flush=True)
            bs *= 2


if __name__ == "__main__":
    if len(sys.argv) == 4:
        print(json.dumps(one(sys.argv[1], sys.argv[2], int(sys.argv[3]))))
    else:
        main(sys.argv[1:] or list(SIZES))
