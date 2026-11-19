# Preflight 2 (2026-11-07, A40 bak-hpc1, source 208d9e1, the real Job environment)

First run inside the Job image: apt gcc, the venv tarball, the tok512 checksum and the repo tests all worked.
Raw output: preflight-2-check_kernel.txt.

## Result: FAILED, 2 of 54. No training started.

Both failures are rule 3 (whole-gradient error at most 1.10 x S at the same d, T and mode), in bf16 eager:

- P(140,140), d=64, T=512: kernel 5.76e-3, control 5.65e-3, S 5.12e-3, ratio 1.13.
- P(93,93), d=64, T=512: kernel 5.77e-3, control 5.73e-3, S 5.12e-3, ratio 1.13.

In both, the dense control with no Triton is just as far from float64, so the extra error belongs to how
these architectures round in bf16, not to the kernel. In bf16 compiled, the training mode, both pass
(5.52e-3 and 5.51e-3 against S 5.10e-3, ratio 1.08). Rules 1 and 2 pass in all 54 cases.

scripts/preflight.py did not run, because preflight.sh stops at the first failure. It passed on the dev pod
in preflight 1 with the same code.

## Other findings

- The compiler reset works: compiled results now differ from eager in every case, including from
  `D d=288` on, where preflight 1 printed identical numbers.
- The 2.61e+01 for log_eps_s of P(105,70) at T=200 reproduced with the same tokens. That parameter holds
  9.4e-9 of the total gradient norm, so it is the ill-conditioned case, not a kernel fault.
- log_rate, not gated: the largest kernel-specific errors where its share is at least 0.1% are fp32 S
  1.05e-1 (share 1.3e-3), D 8.6e-2, P(93,93) 8.7e-2 in fp32 and 9.1e-2 in bf16 (share 4.4e-3), against
  controls below 4e-2 and mostly below 1e-3. With these tokens it is not one-sided against P: S is worse
  than P(105,70) and P(140,140).

## log_rate diagnosis

Hypothesis: d/dg in _bwd_stats_kernel is sum_t t (q dqf - k dk + ...). dqf and dk are built from bf16 q and k,
but are paired with fp32 q and k in that sum, so the rounding difference survives weighted by absolute
position t instead of cancelling to (t - s). scripts/diag_log_rate.py tests this at operator level, with
scripts/kernel_dgfix.py as a diagnosis-only copy that pairs them consistently. Raw output:
preflight-2-diag_log_rate.txt, 48 calls (6 models x 2 decay rates x 4 layers).

- The formula itself is right: in float64 it matches autograd to 1e-14.
- The hypothesis explains only part of it. Simulating bf16 inputs reproduces the kernel's error in some calls
  (P(105,70) rate=test layer 2: simulated 1.38e-1, kernel 1.24e-1, fix 2.0e-2) but not in most, and the fix
  does not lower the error in general: median absolute dg error 3.8e-4 for both, max 1.5e-3 and 1.2e-3.
  The kernel also rounds the in-chunk weights and the chunk state to bf16 (_la), which the simulation omits.
- What the data do show: the absolute dg error is roughly fixed, about 1e-4 to 1.5e-3 per layer, whatever dg
  is. Relative error is large only where a layer's dg is small: dg 1.3e-3 gives 12%, dg 0.35 gives 7e-5.
  At the decay rate training starts from, every layer with |dg| above 0.08 is below 5.1e-3.
- So log_rate gets a small, roughly constant additive error on its gradient. It is not specific to any variant.
  kernel_dgfix.py is not a fix and is not proposed.

## Rule 3 under other readings (for Gary; the gate was not changed)

- Training mode only (bf16 compiled): all 54 pass; worst variant/S ratio is 1.08.
- Kernel error over control, relative to the same quantity for S: all 54 pass; worst 1.069, P(474,474) fp32.
- As written: 2 fail, both bf16 eager, both with the control as far off as the kernel.

## What ran overnight (decided by Claude while Gary slept, after he asked for the work to continue)

The gate failed for two configurations only. Every other configuration passed all three rules in all three
modes, so training was limited to those: phase 0 (S) and phase 2a for S, D, P(69,140), P(105,70) and A.
The six 2a lines of P(140,140) and P(93,93) are held by hand-made claims in
/workspace/runs/phase2a/.claims, which say why; removing them releases the lines. Nothing about the gate or
the protocol was changed. Job ulm-night-bb7cacac, source bb7caca: s_short, then 2a, then s_full.

## Gary's decision (2026-11-07, after reading the readings above)

Rule 3 becomes the second reading: a variant's kernel-over-control ratio at most 1.10 x S's. check_kernel.py
now gates on it. Applied to preflight-2-check_kernel.txt, all 54 cases pass, worst 1.069 (P(474,474) fp32);
the GPU check was not rerun, since the inputs are fixed and the printed figures suffice. This is the second
change made after seeing results; it is final. P(140,140) and P(93,93) are released.
