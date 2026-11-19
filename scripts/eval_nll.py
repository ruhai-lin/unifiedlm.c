"""
Validation NLL of one run in fp32 on a fixed set of batches, overall and by position. Needs CUDA.
Run from the repository root:

$ python scripts/eval_nll.py --out_dir=runs/x

Reads out_dir/ckpt.pt and out_dir/variant.json, writes out_dir/eval.json. The batches are fixed by the
dataset's seed, max_seq_len and batch_size, so runs with the same settings are scored on the same tokens.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())

import variants                    # noqa: E402
from model import ModelArgs, UnifiedLM  # noqa: E402
from tinystories import Task       # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", required=True)
parser.add_argument("--checkpoint", default="ckpt.pt")
parser.add_argument("--eval_iters", type=int, default=100)
args = parser.parse_args()

with open(os.path.join(args.out_dir, "variant.json")) as f:
    meta = json.load(f)
variants.install(meta["variant"], meta["m_s"], meta["m_f"])

ckpt = torch.load(os.path.join(args.out_dir, args.checkpoint), map_location="cuda")
net = UnifiedLM(ModelArgs(**ckpt["model_args"]))
net.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
net.cuda().eval()

cfg = ckpt["config"]
T, B = cfg["max_seq_len"], cfg["batch_size"]
batches = Task.iter_batches(batch_size=B, device="cuda", num_workers=0, split="val", max_seq_len=T,
                            vocab_size=cfg["vocab_size"], vocab_source=cfg["vocab_source"])
torch.backends.cuda.matmul.allow_tf32 = False
total = torch.zeros(T, dtype=torch.float64)
with torch.no_grad():
    for _ in range(args.eval_iters):
        x, y = next(batches)
        logits = net(x, y)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="none").view(x.shape)
        total += nll.sum(0).double().cpu()
per_position = total / (B * args.eval_iters)

edges = [e for e in (0, 64, 128, 256, 512, 1024) if e < T] + [T]
result = {
    "variant": meta["variant"], "m_s": meta["m_s"], "m_f": meta["m_f"], "seed": cfg["seed"],
    "iter_num": ckpt["iter_num"], "tokens": B * T * args.eval_iters,
    "params": sum(p.numel() for p in net.parameters()),
    "nll": per_position.mean().item(),
    "nll_by_position": {f"{a}-{b}": per_position[a:b].mean().item() for a, b in zip(edges, edges[1:])},
}
with open(os.path.join(args.out_dir, "eval.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result))
