# Phase 2 protocol (written 2026-11-07, before any phase 0 or phase 2 result)

Fixed before seeing results, so selection cannot follow the data. Screening uses seed 1337 only; phase 3 uses
new seeds 2001 to 2005 and never 1337. Every number below is the validation loss train.py prints at the
final eval ("step N: ... val loss X"), which is also what selects the saved checkpoint.

## Phase 0 check

- s_short (5K iters, the README short budget of 327.68M tokens): val loss within 0.01 of 1.511, else stop
  before phase 2 and report.
- s_full (100K iters): within 0.005 of 1.2905. It runs beside phase 2 if cards allow; a miss stops the work
  before phase 3, not phase 2 results already in hand.

## Phase 2a: 7 configs x {5e-4, 1e-3, 2e-3}, 5K iters, warmup 50 (plan file phase2a.txt)

An LR is unstable for a config if any logged loss is not finite, or its final val loss is more than 0.02 above
the lowest val loss of the same run. 2a ranks nothing else.

Phase 2b pair per config: 1e-3 plus the stable other LR with the lower 2a final val loss. If 1e-3 is unstable,
the two stable LRs. If only one LR is stable, 2b runs that one.

## Phase 2b: 7 configs x their pair, 20K iters, warmup 200, eval every 2000

- LR per config: the one with the lower 20K final val loss, but 1e-3 if it is within 0.01 of the other.
- P-eq split: the lowest 20K final val loss among the three P-eq configs at their chosen LRs.
- Frozen for phase 3: S, D, P-free, the chosen P-eq, A, each at its chosen LR.

Phase 3 does not start until Gary has seen the frozen choices and the measured cost.

## Exploratory addition (2026-11-07, before its results; plan file explore.txt, runs under runs/explore)

2e-3, the top of the grid, won in every config. To inform Gary's decision, not to change the rules above:
s_full at 2e-3 (100K iters, warmup 1000) and 4e-3 at the 2a and 2b budgets for the five configs that ran.
The interim freeze stays as the rules above give it; whether to extend the grid is Gary's call.

Second exploratory addition (2026-11-07, before its results; explore2.txt, runs under runs/explore2): 4e-3 also
won everywhere, so 8e-3 at the 2b budget for the same five configs, and s_full at 4e-3. Same status as above.

## Phase 2c, adopted by Gary on 2026-11-07, written before any 2c result

The LR grid moves up because its top won everywhere. This replaces the 2b LR rule ("1e-3 if within 0.01");
the 2b runs stay as data points.

- P(140,140) and P(93,93) first run their 2a lines, as the other configs did.
- Every config is scored at the 2b budget (20K iters, warmup 200, seed 1337) at 1e-3, 2e-3, 4e-3 and 8e-3.
  Runs that already exist with exactly these flags are reused: phase2b/<c>_lr1e-3 and _lr2e-3, and
  explore/<c>_20k_lr4e-3.
- Unstable means a non-finite logged loss or a final val loss more than 0.02 above the run's minimum.
- While a config's lowest final val loss among its stable LRs is at its highest tested LR, its next doubling
  is run. A config stops once its best LR is not the highest tested.
- LR per config: the stable LR with the lowest 20K final val loss. P-eq split: the lowest among the three
  P-eq configs at their chosen LRs.
- Check before phase 3: S at its chosen LR for 100K iters (s_full recipe). If it is worse than S at 2e-3 over
  100K (val 1.2716) by more than 0.005, stop and report.
- Frozen choices and cost go to Gary before phase 3.
- The exploratory Job ulm-explore2-3f0c7a25 (8e-3 for five configs, s_full at 4e-3) was deleted unrun.

## Decision rule per comparison, adopted by Gary on 2026-11-07, before any phase 3 result

Two claims, judged separately; inconclusive never passes.
- Sharing does not hurt: D vs P-free, D vs best P-eq and S vs best separated must each be better or equivalent.
- Coupling helps: S vs D must be better.
