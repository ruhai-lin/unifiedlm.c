"""Why does check_kernel fail? Compares, against dense float64:
   kernel = real Triton kernel, ctrl = dense definition at the same precision with no Triton at all.
   If ctrl shows the same error, the kernel is not the cause. rate=test is check_kernel's rate 0.05;
   rate=train is the value training starts from, 1/(4T)."""
import copy, math, os, sys
import torch
sys.path.insert(0, "scripts"); sys.path.insert(0, os.getcwd())
import check_kernel as ck, kernel
from dense_memory import dense_memory
torch.backends.cuda.matmul.allow_tf32 = False
CONFIGS = [("S",0,0,64,140,512),("D",0,0,64,140,512),("P",105,70,64,140,512),("A",207,207,64,140,512),
           ("S",0,0,64,140,200),("P",105,70,64,140,200),("S",0,0,288,712,256)]
def grads(net, x, dtype, mem):
    ck.set_memory(mem); net.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        net(x, x)
    net.last_loss.backward()
    return {n: p.grad.double().clone() for n, p in net.named_parameters()}
def group(n): return n.split(".", 2)[-1]
for config in CONFIGS:
  for rate in ("test", "train"):
    base = ck.build(*config)
    if rate == "train":
        with torch.no_grad():
            for n, p in base.named_parameters():
                if n.endswith("log_rate"): p.fill_(math.log(1 / (4 * config[-1])))
    ref = copy.deepcopy(base).cuda().double().train()
    net = base.cuda().train()
    torch.manual_seed(7); x = torch.randint(0, 512, (2, config[-1]), device="cuda")
    g0 = grads(ref, x, torch.float32, dense_memory)
    total = torch.cat([v.flatten() for v in g0.values()]).norm()
    for dtype in (torch.float32, torch.bfloat16):
        gk = grads(net, x, dtype, kernel.unified_memory); gc = grads(net, x, dtype, dense_memory)
        allk = torch.cat([gk[n].flatten() for n in g0]); allc = torch.cat([gc[n].flatten() for n in g0]); all0 = torch.cat([g0[n].flatten() for n in g0])
        print(f"\n{config[0]} ms={config[1]} mf={config[2]} d={config[3]} T={config[5]} rate={rate} {str(dtype).split('.')[-1]}: whole-gradient cosine kernel {torch.cosine_similarity(allk,all0,dim=0):.6f} ctrl {torch.cosine_similarity(allc,all0,dim=0):.6f}, rel err kernel {(allk-all0).norm()/total:.2e} ctrl {(allc-all0).norm()/total:.2e}, |g|={total:.3e}", flush=True)
        rows = {}
        for n in g0:
            k = group(n); nr = g0[n].norm().item()
            ek = (gk[n]-g0[n]).norm().item()/max(nr,1e-30); ec = (gc[n]-g0[n]).norm().item()/max(nr,1e-30)
            r = rows.setdefault(k, [0,0,0]); 
            if ek > r[0]: rows[k] = [ek, ec, nr]
        for k, (ek, ec, nr) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:5]:
            print(f"   {k:<14} kernel {ek:.2e}  ctrl {ec:.2e}  |g|={nr:.2e}  |g|/total={nr/total:.1e}")
        if config[-1] in (512,200) and config[0] in "SD":
            print("   log_rate per layer  ref:", " ".join(f"{g0[n].item():+.2e}" for n in g0 if n.endswith("log_rate")), "\n                    kernel:", " ".join(f"{gk[n].item():+.2e}" for n in g0 if n.endswith("log_rate")))
