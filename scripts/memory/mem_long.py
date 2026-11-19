"""Evaluate a trained m2 run (mqar.py --save=1) at longer gaps, with mqar.py's test sets (the same generator seeds).
Writes long.json next to model.pt. Run from the repo root:

$ python scripts/memory/mem_long.py --run_dir=<run> --gaps=4096,8192,16384,20000 --eval_size=256
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mqar  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--gaps", default="4096,8192,16384,20000")
    ap.add_argument("--eval_size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    e = json.load(open(os.path.join(a.run_dir, "eval.json")))
    assert e["task"] == "m2", e["task"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, _ = mqar.build(e["model"], e["dim"], e["vocab"], e["seq"])
    net.load_state_dict(torch.load(os.path.join(a.run_dir, "model.pt"), map_location=dev))
    net.to(dev).eval()
    body = mqar.Body(net)
    N = e["counts"][0]
    out = {"model": e["model"], "eval_size": a.eval_size, "trained_gmax": e["gmax"], "test": {}}
    with torch.no_grad():
        for G in (int(x) for x in a.gaps.split(",")):
            g = torch.Generator(device=dev).manual_seed(2 * 100003 + G)   # mqar.py's test split
            # mqar.py draws its test batches 32 sequences at a time; draw the same way, then evaluate in smaller pieces
            hit = n = 0
            for _ in range(max(1, a.eval_size // 32)):
                x, y, pos, _ = mqar.m2_batch(32, N, G, mqar.round_up(4 * N + G), e["vocab"], g, dev, e["filler"])
                for i in range(0, 32, a.batch):
                    h = body(x[i:i + a.batch])
                    pred = net.output(h[:, pos]).argmax(-1)
                    hit += (pred == y[i:i + a.batch][:, pos]).sum().item()
                    n += pred.numel()
            out["test"][f"G{G}"] = hit / n
            print(e["model"], f"G{G}", f"{hit / n:.4f}", flush=True)
    json.dump(out, open(os.path.join(a.run_dir, "long.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
