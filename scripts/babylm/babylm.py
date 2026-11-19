"""
BabyLM 2026 Strict-Small (10M words) as a train.py dataset, tokenized with the official 2026 Strict-Small
tokenizer (byte-level BPE, 16384 tokens). See scripts/reports/babylm-plan.md.

Prepare once, on any machine with the `tokenizers` and `pyarrow` packages (training needs only numpy):
$ python scripts/babylm/babylm.py prepare --raw=data/babylm10m/raw --tokenizer=<dir with tokenizer.json>
$ python scripts/babylm/babylm.py blimp --blimp=data/babylm10m/blimp --tokenizer=<same dir>

Each of the six source files is one continuous text with no document breaks. The last 2% of each file's lines
are held out for validation, so training sees about 98% of the 10M words. Each part is encoded as
<s> text </s> and the parts are concatenated in a fixed order. Batches are the non-overlapping windows of
max_seq_len + 1 tokens, every window once per pass in a shuffled order, as in tinystories.py.
"""
import argparse
import glob
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist

DATA_DIR = "data/babylm10m"
SOURCES = ["bnc_spoken", "childes", "gutenberg", "open_subtitles", "simple_wiki", "switchboard"]
VAL_FRACTION = 0.02
BOS, EOS = 1, 2
DATA_SEED = 42  # train order; scripts/train_variant.py shifts it with --seed, so the default seed keeps this order


def load_tokenizer(tok_dir):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))


def encode(tok, text):
    return [BOS] + tok.encode(text, add_special_tokens=False).ids + [EOS]


def prepare(raw, tok_dir):
    tok = load_tokenizer(tok_dir)
    out = {"train": [], "val": []}
    meta = {"tokenizer": tok_dir, "val_fraction": VAL_FRACTION, "sources": {}}
    for name in SOURCES:
        lines = open(os.path.join(raw, f"{name}.train.txt"), encoding="utf-8").read().split("\n")
        while lines and not lines[-1].strip():
            lines.pop()
        cut = len(lines) - max(1, round(VAL_FRACTION * len(lines)))
        info = {}
        for split, part in (("train", lines[:cut]), ("val", lines[cut:])):
            text = "\n".join(part)
            ids = encode(tok, text)
            out[split].append(np.array(ids, dtype=np.uint16))
            info[split] = {"lines": len(part), "words": len(text.split()), "tokens": len(ids)}
        meta["sources"][name] = info
        print(name, json.dumps(info))
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, parts in out.items():
        arr = np.concatenate(parts)
        assert arr.max() < tok.get_vocab_size()
        arr.tofile(os.path.join(DATA_DIR, f"{split}.bin"))
        meta[split] = {k: sum(s[split][k] for s in meta["sources"].values()) for k in ("lines", "words", "tokens")}
    json.dump(meta, open(os.path.join(DATA_DIR, "meta.json"), "w"), indent=1)
    print(json.dumps({k: meta[k] for k in ("train", "val")}))


def blimp(blimp_dir, tok_dir):
    """BLiMP minimal pairs as token ids, each sentence encoded as <s> sentence (no </s>)."""
    import pyarrow.parquet as pq
    tok = load_tokenizer(tok_dir)
    pairs = []
    for path in sorted(glob.glob(os.path.join(blimp_dir, "*.parquet"))):
        for r in pq.read_table(path).to_pylist():
            pairs.append({"uid": r["UID"], "term": r["linguistics_term"], "field": r["field"],
                          "good": [BOS] + tok.encode(r["sentence_good"], add_special_tokens=False).ids,
                          "bad": [BOS] + tok.encode(r["sentence_bad"], add_special_tokens=False).ids})
    json.dump(pairs, open(os.path.join(DATA_DIR, "blimp.json"), "w"))
    print(len(pairs), "pairs in", len({p["uid"] for p in pairs}), "paradigms")


class PretokDataset(torch.utils.data.IterableDataset):

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        assert vocab_source == "custom" and vocab_size == 16384, "BabyLM uses its own 16384-token vocabulary"
        self.split, self.max_seq_len = split, max_seq_len

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        rank = dist.get_rank() if dist.is_initialized() else 0
        rng = random.Random(DATA_SEED + worker_id + 1337 * rank)
        m = np.memmap(os.path.join(DATA_DIR, f"{self.split}.bin"), dtype=np.uint16, mode="r")
        n = (len(m) - 1) // self.max_seq_len
        while True:
            ixs = list(range(n))
            rng.shuffle(ixs)
            for ix in ixs:
                chunk = torch.from_numpy(m[ix * self.max_seq_len: (ix + 1) * self.max_seq_len + 1].astype(np.int64))
                yield chunk[:-1], chunk[1:]


class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        dl = torch.utils.data.DataLoader(PretokDataset(**dataset_kwargs), batch_size=batch_size,
                                         pin_memory=True, num_workers=num_workers)
        for x, y in dl:
            yield x.to(device, non_blocking=True), y.to(device, non_blocking=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "blimp"])
    ap.add_argument("--raw", default=os.path.join(DATA_DIR, "raw"))
    ap.add_argument("--blimp", default=os.path.join(DATA_DIR, "blimp"))
    ap.add_argument("--tokenizer", required=True)
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare(a.raw, a.tokenizer)
    else:
        blimp(a.blimp, a.tokenizer)
