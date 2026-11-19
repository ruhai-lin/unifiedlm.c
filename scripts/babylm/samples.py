"""Exploratory text samples from the seed-0 BabyLM checkpoints (S, P-free, T), to show what their outputs look like.
Sampling at temperature 0.8, top-k 40, one draw per case; the context is never cropped, so the long case runs past
the 512-token training length. Not evidence about which model is better; the losses are.

$ python scripts/babylm/samples.py > scripts/reports/babylm/explore/samples_seed0.txt
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
import eval_babylm as ev  # noqa: E402
from babylm import load_tokenizer  # noqa: E402

TOK = load_tokenizer(json.load(open(os.path.join(ev.DATA_DIR, "meta.json")))["tokenizer"])
RUNS = {"S": "babylm_s", "P-free": "babylm_p", "T": "babylm_t"}
dec = lambda ids: TOK.decode([int(i) for i in ids], skip_special_tokens=False)


@torch.no_grad()
def gen(net, ctx, n, seed, temp=0.8, top_k=40):
    g = torch.Generator().manual_seed(seed)
    idx = torch.tensor([ctx], dtype=torch.long)
    for _ in range(n):
        logits = net(idx)[:, -1, :] / temp           # whole context, never cropped
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g)
        idx = torch.cat([idx, nxt], 1)
        if nxt.item() == 2:
            break
    return idx[0, len(ctx):].tolist()


def main():
    torch.set_num_threads(os.cpu_count())
    nets = {k: ev.load(os.path.join("data/babylm10m/runs", v), "cpu")[0] for k, v in RUNS.items()}

    s = "The little dog ran to the park."
    print("== tokens ==")
    print(" | ".join(TOK.decode([i]) for i in TOK.encode(s, add_special_tokens=False).ids))

    prompts = ["Mommy, can I have", "Once upon a time there was a king who", "Paris is the capital of"]
    print("\n== short prompts (context < 512) ==")
    for p in prompts:
        ctx = [1] + TOK.encode(p, add_special_tokens=False).ids
        print(f"\nPROMPT: {p}")
        for k, net in nets.items():
            print(f"  [{k}] {dec(gen(net, ctx, 40, seed=0)).strip()!r}")

    val = np.memmap(os.path.join(ev.DATA_DIR, "val.bin"), dtype=np.uint16, mode="r")
    start = 140000                                   # inside the Gutenberg part of val.bin
    long_ctx = val[start:start + 1800].astype(np.int64).tolist()
    short_ctx = long_ctx[-400:]
    truth = val[start + 1800:start + 1860].tolist()
    print("\n== long context ==")
    print("LAST 400 TOKENS OF CONTEXT:", repr(dec(short_ctx)[-600:]))
    print("TRUE CONTINUATION:", repr(dec(truth)))
    for k, net in nets.items():
        print(f"\n  [{k}] after 400 tokens : {dec(gen(net, short_ctx, 50, seed=0))!r}")
        print(f"  [{k}] after 1800 tokens: {dec(gen(net, long_ctx, 50, seed=0))!r}")


if __name__ == "__main__":
    main()
