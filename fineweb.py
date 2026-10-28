"""
Download, preprocess and serve the FineWeb-Edu sample-10BT dataset as a DataLoader.

Mirrors climbmix.py. The source is the 10B-token sample of HuggingFaceFW/fineweb-edu, 14 large
parquet files kept in the HuggingFace cache (~27 GB). Pretokenization streams them row group by
row group with the Llama 2 tokenizer into uint16 .bin shards of ~100M tokens in data/FineWebEdu_10BT/.
The last shard is the validation split.
"""

import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download

from tokenizer import Tokenizer

DATA_CACHE_DIR = "data"
DATA_DIR = os.path.join(DATA_CACHE_DIR, "FineWebEdu_10BT")
REPO_ID = "HuggingFaceFW/fineweb-edu"
PATTERN = "sample/10BT/*.parquet"
TOKENS_PER_SHARD = 100_000_000


def parquet_files():
    """Paths of the sample-10BT parquet files in the HuggingFace cache, downloading them if needed."""
    root = snapshot_download(repo_id=REPO_ID, repo_type="dataset", allow_patterns=[PATTERN])
    return sorted(glob.glob(os.path.join(root, PATTERN)))


def download():
    """Downloads the 14 sample-10BT parquet files (~27 GB) into the HuggingFace cache."""
    files = parquet_files()
    print(f"Number of parquet files: {len(files)}")
    pf = pq.ParquetFile(files[0])
    example = pf.read_row_group(0, columns=["text"]).column("text").to_pylist()[0]
    print(f"Example document:\n{example[:500]}")


def process_file(args):
    file_id, path = args
    enc = Tokenizer()
    pf = pq.ParquetFile(path)
    stem = os.path.basename(path).replace(".parquet", "")
    buf, n_buf, part, total = [], 0, 0, 0

    def flush():
        nonlocal buf, n_buf, part
        tokens = np.concatenate(buf).astype(np.uint16)
        filename = os.path.join(DATA_DIR, f"{stem}_{part:03d}.bin")
        with open(filename, "wb") as f:
            f.write(tokens.tobytes())
        print(f"Saved {filename}, {tokens.size:,} tokens, average seqlen: {tokens.size / (tokens == 1).sum():.2f}")
        buf, n_buf, part = [], 0, part + 1

    for rg_idx in range(pf.num_row_groups):
        texts = [t.strip() for t in pf.read_row_group(rg_idx, columns=["text"]).column("text").to_pylist()]
        for tokens in enc.sp_model.encode(texts):  # batched; every document gets BOS, no EOS
            buf.append(np.array([enc.bos_id] + tokens, dtype=np.uint16))
            n_buf += len(tokens) + 1
            total += len(tokens) + 1
        if n_buf >= TOKENS_PER_SHARD:
            flush()
    if n_buf:
        flush()
    return total


def pretokenize(num_workers):
    os.makedirs(DATA_DIR, exist_ok=True)
    files = parquet_files()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        total = sum(executor.map(process_file, enumerate(files)))
    shards = sorted(glob.glob(os.path.join(DATA_DIR, "*.bin")))
    print(f"Done. {len(shards)} shards, {total:,} tokens; validation shard: {shards[-1]}")


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source):
        super().__init__()
        assert vocab_source == "llama2", "FineWeb-Edu is pretokenized with the Llama 2 tokenizer only"
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        shard_filenames = sorted(glob.glob(os.path.join(DATA_DIR, "*.bin")))
        # train/test split. the last shard is val, rest are train
        shard_filenames = shard_filenames[:-1] if self.split == "train" else shard_filenames[-1:]
        assert len(shard_filenames) > 0, f"No bin files found in {DATA_DIR}"
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y

# -----------------------------------------------------------------------------
# public interface functions

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order:
    python fineweb.py download
    python fineweb.py pretokenize
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize"])
    parser.add_argument("-w", "--num_workers", type=int, default=14, help="parallel tokenizer processes (one per parquet file)")
    args = parser.parse_args()

    if args.stage == "download":
        download()
    elif args.stage == "pretokenize":
        pretokenize(num_workers=args.num_workers)
