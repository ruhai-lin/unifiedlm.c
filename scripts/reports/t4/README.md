# T4 smoke test of the 15M model in fp16 (Colab, 2026-11-10)

Run by `scripts/colab/t4_smoke.sh` and `scripts/colab/check_kernel_fp16.py` on one Tesla T4 (sm 7.5, 15 GB; no bf16),
torch 2.10.0+cu128, triton 3.6.0. PoC smoke test, one run each.

- Kernel check, S configurations only, same tolerances as `scripts/check_kernel.py` (`kernel.log`):
  - kernel as it is (bf16 inside): fp32 passes; all 6 fp16 cases fail rule 1's control test (whole-gradient error
    1.3-1.8x the dense fp16 control's, limit 1.10x). Gradients still have cosine >= 0.99999.
  - copy with fp16 inside (`sed s/bfloat16/float16/` on kernel.py): 0 of 9 failures.
- Throughput, fp16 kernel copy, T=1024 (`speed.jsonl`): 15M S 22-23K tok/s (batch 16 max), 15M T 95-105K tok/s
  (batch 32 max). On an A6000 in bf16 the same S runs at about 200K tok/s.
- Convergence (`train.log`): the BabyLM S run with the same flags and seed, fp16 + GradScaler instead of bf16, with the
  fp16 kernel copy. Val loss 3.8684 at step 500 and 3.5252 at step 1000, against bf16's 3.8739 and 3.5270.
