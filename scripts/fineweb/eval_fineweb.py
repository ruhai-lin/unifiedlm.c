"""Validation NLL by position for a FineWeb-Edu run, at the trained length L and at 4L, on the subset's
validation shard (the last .bin, never trained on). Same code as the BabyLM eval; no BLiMP, whose tokens are in
the BabyLM vocabulary. Run from the repository root:

$ python scripts/fineweb/eval_fineweb.py --out_dir=<run>
"""
import argparse
import glob
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "babylm"))
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "fineweb"))
import eval_babylm as ev   # noqa: E402
import variants            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--windows", type=int, default=2048, help="windows of length L; 4L uses a quarter as many")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = json.load(open(os.path.join(a.out_dir, "variant.json")))
    variants.HEAD_DIM = meta.get("head_dim", 48)
    os.environ["FW_SUBSET"] = "4of14" if meta["dataset"] == "fineweb4" else "2of14"
    import fineweb_subset  # both subsets share the validation shard
    net, meta, ckpt = ev.load(a.out_dir, device)
    val = sorted(glob.glob(os.path.join(fineweb_subset.fineweb.DATA_DIR, "*.bin")))[-1]
    ev.DATA_DIR = tempfile.mkdtemp()
    os.symlink(os.path.abspath(val), os.path.join(ev.DATA_DIR, "val.bin"))
    L = ckpt["model_args"]["max_seq_len"]
    a.batch = max(1, a.batch * 1024 // L)  # --batch is for L = 1024; keep tokens per batch (and the logits) fixed
    res = {"variant": meta["variant"], "seed": ckpt["config"]["seed"], "iter_num": ckpt["iter_num"],
           "params": sum(p.numel() for p in net.parameters()), "val_shard": os.path.basename(val),
           "val": {str(L): ev.nll_by_position(net, L, a.batch, device, a.windows)}}
    print(json.dumps(res), flush=True)
    try:  # at L = 16K the 4L logits alone are 8 GB in fp32; a 24 GB card may not fit them, keep the L result anyway
        res["val"][str(4 * L)] = ev.nll_by_position(net, 4 * L, max(1, a.batch // 4), device, a.windows // 4)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        res["val"][str(4 * L)] = {"skipped": "out of memory"}
    print(json.dumps(res["val"][str(4 * L)]), flush=True)
    json.dump(res, open(os.path.join(a.out_dir, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
