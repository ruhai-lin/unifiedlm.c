# FineWeb-Edu results (2026-11-10)

Validation shard 001_00000_008.bin, never trained on. Training length L=1024; the 4L eval uses 512 windows of 4096.
Numbers from each run's eval.json on /workspace/runs.

## Val NLL at L=1024

| Run | Params | seed 1337 | 1338 | 1339 |
|---|---|---|---|---|
| 15M S | 15.17M | 4.116 | 4.113 | 4.107 |
| 15M T | 15.19M | 3.928 | 3.957 | 3.952 |
| 15M P-free | 17.64M | 4.040 | 4.034 | 4.039 |

15M means over 3 seeds: S 4.112, T 3.946, P-free 4.038. S minus T 0.166 (seed spread about 0.03); P-free minus T 0.092.
| 42M S | 41.63M | 3.423 | | |
| 42M T | 41.69M | 3.214 | | |
| 42M P-free | 52.14M | 3.368 | | |

S minus T: 0.19 at 15M (seed 1337), 0.21 at 42M. The gap does not shrink with size.

## Beyond the training length (4096 eval, NLL by position)

| Run | 1024-2048 | 2048-4096 |
|---|---|---|
| 15M S 1337 / 1338 / 1339 | 4.30 / 4.50 / 4.75 | 5.85 / 7.24 / 7.92 |
| 15M T 1337 / 1338 / 1339 | 4.79 / 4.94 / 4.90 | 5.59 / 5.68 / 5.60 |
| 15M P-free 1337 / 1338 / 1339 | 4.79 / 5.84 / 4.72 | 7.47 / 8.51 / 7.69 |
| 42M S | 8.52 | 11.44 |
| 42M T | 4.69 | 5.79 |
| 42M P-free | 7.13 | 9.87 |

At 15M, S is better than T on 1024-2048 on all three seeds (means 4.52 vs 4.88) but varies by 0.45 between seeds;
on 2048-4096 it is worse than T on every seed (means 7.00 vs 5.62). At 42M S collapses past L, so the 1024-2048
advantage is not there at the next size.

## In progress

- 110M S (A6000): step ~980 of 16700 at 28K tok/s, about 20 h left. Initial loss 15.09 (uniform is 10.37).
- 110M T (RTX 5000 Ada): step ~4700, val 3.21 at step 4000, about 3.7 h left.
