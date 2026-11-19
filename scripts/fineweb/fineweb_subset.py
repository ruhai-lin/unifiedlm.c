"""FineWeb-Edu sample-10BT subsets, pretokenized by the unmodified fineweb.py (Llama 2 tokenizer, ~100M-token
uint16 shards, last shard in sorted order = validation). Training imports this module in place of fineweb so both
sides agree on DATA_DIR. FW_SUBSET picks the subset:
- 2of14 (default): parquet files 000 and 001; validation is 001_00000_008.bin.
- 4of14: files 000-003, built by scripts/fineweb/prep_4of14.sh; the same validation shard, renamed
  val_001_00000_008.bin so it still sorts last and is never trained on.
DATA_SEED shifts the shard and window order (fineweb.py seeds it with 42); 0 keeps fineweb.py's order.

$ python scripts/fineweb/fineweb_subset.py pretokenize      # builds 2of14
"""
import os
import random as _random
import sys
import types

sys.path.insert(0, os.getcwd())
import fineweb  # noqa: E402

SUBSET = os.environ.get("FW_SUBSET", "2of14")
assert SUBSET in ("2of14", "4of14"), SUBSET
N_FILES = 2
fineweb.PATTERN = "sample/10BT/00[0-1]_*.parquet"
fineweb.DATA_DIR = os.path.join(fineweb.DATA_CACHE_DIR, f"FineWebEdu_{SUBSET}")
DATA_SEED = 0
fineweb.random = types.SimpleNamespace(Random=lambda seed: _random.Random(seed + DATA_SEED))
Task = fineweb.Task

if __name__ == "__main__" and sys.argv[1:] == ["pretokenize"]:
    assert SUBSET == "2of14"
    files = fineweb.parquet_files()
    print("parquet files:", [os.path.basename(f) for f in files])
    assert len(files) == N_FILES, files
    fineweb.pretokenize(num_workers=N_FILES)
