# Preflight 1 (2026-11-06, A40, source f8e76ee)

Run by exec on gpu-a40-dev and gpu-a40-dev2, with Gary's permission, using the same uv.lock as the Job environment.
This is not the Job image: Python 3.12.16 from uv on Ubuntu 22.04, gcc installed by hand.

## Result

- Repo tests and end-to-end run path: PASSED. Seeds, S/D shared init, parameter counts, variant.json and resume refusal all behave.
- check_kernel.py as written: FAILED, 37 of 54 configurations exceed TOL = 1e-2. Training has not started.

## Diagnosis (diag_kernel.py, raw output in preflight-1-diag.txt)

Control: the dense definition at the same precision, with no Triton.

- Whole gradient, 28 cases: cosine to float64 at least 0.999983.
- bf16 (the training mode): relative error kernel at most 5.99e-3, control at most 5.90e-3. The two match case by case, so bf16 autocast dominates, not the kernel.
- fp32: kernel at most 2.74e-3, control at most 4.83e-7. This is the kernel's internal bf16 and agrees with docs/kernel.md (about 3e-3).
- Variants versus S at d=64, T=512, rate=train, bf16: S 5.71e-3, D 5.81e-3, P(105,70) 5.88e-3, A 5.70e-3. No variant-specific problem.
- Most FAIL lines come from scalar parameters with very small true gradients (log_rate, log_eps_s, at times 1e-7 to 1e-6 of the total norm), where relative error is ill-conditioned. With bf16 the control shows the same size of error, for example log_rate 0.83 against the kernel's 1.44.

## Not explained away

- log_rate has kernel-specific error on top of bf16: up to about 2% in fp32 and 6.5% in bf16 where its gradient is not tiny (control below 0.5%). S has it too, and docs/kernel.md reports up to 1.7e-2 at operator level. Cause not established; d/dg subtracts two position-weighted sums, so cancellation is a hypothesis only.
- The largest single figure, log_eps_s 2.61e+01 for P(105,70) at T=200, was not reproduced by the diagnostic, whose worst value for it there is 2.50e-2. The two runs drew different tokens. Its gradient does not pass through the kernel directly.
- test_all.py::test_kernel passes because it only checks fp32, 2 layers, T=200.

## Amended gate, approved by Gary on 2026-11-07 after seeing these results

1. Whole-gradient relative error at most 1e-2 in every mode, and in bf16 within 10% of the same-precision dense control.
2. Per-group relative error at most 1e-2 above the control's, only for groups holding at least 0.1% of the total norm, with log_rate reported separately.
3. Variants no worse than S at the same shape, by the same measures.

The original failures stay on record in preflight-1-check_kernel.txt. This gate was written after seeing them.

Choices made when implementing it, fixed before the rerun:

- Rule 3 is gated on whole-gradient error at 1.10 x S. Per-parameter excess over the control sits near zero and can be negative, so a ratio on it means nothing; it is printed beside S's, and rule 2 is its only hard limit.
- log_rate is printed for every case and gated by nothing. Its kernel-specific error remains unexplained.
- The control runs in the same compile mode as the kernel.

Gap found in the first run: from `D d=288 T=256` on, torch.compile hit its recompile limit and the "compiled" results equal the eager ones to every printed digit, so those cases probably never ran compiled. The check now resets the compiler for every case.
