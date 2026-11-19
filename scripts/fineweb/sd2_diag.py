"""SD2 diagnostic on CPU: rho usage, slow rates, and NLL with the slow (rho=0) or fast (rho=1) memory switched off.
Run from /scratch/src:  python sd2_diag.py /workspace/runs/fwsd/fwsd2_15m 32"""
import glob, json, math, os, sys, types
sys.modules["kernel"] = types.SimpleNamespace(unified_memory=None, CHUNK=64)
for p in ("", "scripts", "scripts/babylm", "scripts/fineweb"):
    sys.path.insert(0, os.path.join(os.getcwd(), p))
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(6)
import eval_babylm as ev, variants  # noqa

out_dir, windows = sys.argv[1], int(sys.argv[2])
net, meta, ckpt = ev.load(out_dir, "cpu")
import delta
from delta import _hi, _lin, _unit, delta_memory
MODE, REC = ["mix"], {}

def fwd(self, x):
    h, r, v = self._inputs(x)
    r, v = _hi(r), _hi(v)
    k = F.pad(r[:, :-1], (0, 0, 1, 0))
    target = v - F.linear(k, _hi(self.mu.weight))
    kh, qh = _unit(k), _unit(r)
    fast = delta_memory(kh, target, torch.sigmoid(_lin(self.beta, h)).squeeze(-1), self.log_decay(r), qh)
    slow = delta_memory(kh, target, torch.sigmoid(_lin(self.beta_slow, h)).squeeze(-1),
                        torch.expm1(-self.log_rate_slow.exp()).expand_as(r), qh)
    rho = torch.sigmoid(_lin(self.rho, h))
    if MODE[0] == "mix":
        R = REC.setdefault(self._i, {"rho": [], "fast": [], "slow": [], "bslow": [], "bfast": []})
        R["rho"].append(rho.flatten())
        R["fast"].append(((1 - rho) * fast).norm(dim=-1).flatten())
        R["slow"].append((rho * slow).norm(dim=-1).flatten())
        R["bslow"].append(torch.sigmoid(_lin(self.beta_slow, h)).flatten())
        R["bfast"].append(torch.sigmoid(_lin(self.beta, h)).flatten())
    rho = {"mix": rho, "fast": torch.zeros_like(rho), "slow": torch.ones_like(rho)}[MODE[0]]
    return x + self.mu(r) + (1 - rho) * fast + rho * slow

delta.TwoDeltaBlock.forward = fwd
blocks = [b for b in net.modules() if isinstance(b, delta.TwoDeltaBlock)]
for i, b in enumerate(blocks):
    b._i = i
L = ckpt["model_args"]["max_seq_len"]
val = np.memmap("/workspace/data/FineWebEdu_2of14/001_00000_008.bin", dtype=np.uint16, mode="r")  # the 2of14 validation shard
EDGES = [0, 64, 128, 256, 512, 1024]

def run():
    tot, n = torch.zeros(L, dtype=torch.float64), 0
    with torch.no_grad():
        for i in range(0, windows, 4):
            xy = torch.from_numpy(np.stack([val[j * L:(j + 1) * L + 1].astype(np.int64) for j in range(i, min(windows, i + 4))]))
            x, y = xy[:, :-1].contiguous(), xy[:, 1:].contiguous()
            tot += F.cross_entropy(net(x, y).transpose(1, 2), y, reduction="none").sum(0).double()
            n += len(xy)
    per = tot / n
    return {"nll": per.mean().item(), **{f"{a}-{b}": per[a:b].mean().item() for a, b in zip(EDGES, EDGES[1:])}}

res = {"windows": windows, "L": L}
for mode in ("mix", "fast", "slow"):
    MODE[0] = mode
    res[mode] = run()
    print(mode, json.dumps(res[mode]), flush=True)
q = lambda t, p: torch.quantile(t[torch.randperm(len(t))[:200000]], p).item()
res["layers"] = []
for i, b in enumerate(blocks):
    R = {k: torch.cat(v) for k, v in REC[i].items()}
    hs = 1.0 / b.log_rate_slow.exp().detach()
    res["layers"].append({
        "rho_mean": R["rho"].mean().item(), "rho_p10": q(R["rho"], .1), "rho_p90": q(R["rho"], .9),
        "fast_norm": R["fast"].mean().item(), "slow_norm": R["slow"].mean().item(),
        "beta_fast": R["bfast"].mean().item(), "beta_slow": R["bslow"].mean().item(),
        "slow_horizon_median": hs.median().item(), "slow_horizon_p10": torch.quantile(hs, .1).item(),
        "slow_horizon_p90": torch.quantile(hs, .9).item(), "slow_horizon_init": 16.0 * L})
    print(i, json.dumps({k: round(v, 4) for k, v in res["layers"][-1].items()}), flush=True)
json.dump(res, open("/scratch/sd2_diag.json", "w"), indent=1)
print("DONE", flush=True)
