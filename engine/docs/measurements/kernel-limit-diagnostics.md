# Kernel limit diagnostics

Cost: standalone consumed-output probes and checked kernel replays. No served path changes.
Gain: compare a traced kernel mean with measured instruction and memory references.

The source base is `decc380f497b7bcf7ca54aafb7def8b78f8c0353`.
`kernel-bench --diag-limits PRIMARY_PTX TC_PTX MODE` selects the probes.
The `limits`, `scalar`, `rates2`, `replay`, `all`, and `rest` modes keep their result checks.
Operands depend on runtime input. Control cases change inputs and must change outputs.
Every replay prints its dimensions before launch and checks the consumed output.
The primary kernel image does not change.

The following references come from runs `diag-limits-all-1` and
`diag-limits-rates2-1`. Read means come from `lead2-split-main-1`, rank 0.
All times are milliseconds per launch. Read means and fixture times are
MEASURED. Instruction times and ratios are ESTIMATES from logical work counts.
A reference is the larger of the instruction time and the memory fixture time.
It is a conditional comparison, not a proven physical minimum or a speed promise.

| Kernel | Read mean | Instruction reference | Memory fixture | Read/reference |
|---|---:|---:|---:|---:|
| mx_tc_tile | 1.479438 | 0.129144 | 0.227440 | 6.505 |
| fp4_cohort | 4.916180 | 0.297865 | 3.386953 | 1.452 |
| fp4_cohort_t64 | 6.084500 | 0.278388 | 3.649641 | 1.667 |
| mhc_boundary_block | 1.346556 | 0.056188 | 0.885310 | 1.521 |
| fp8_rows_tile3 | 0.424546 | 0.113835 | 0.092784 | 3.729 |

The expert fixtures use synthetic distributions, not the original routes.
They retain the recorded grids and print every cohort count. The original
route lists were not saved. Exact original-route replay is NOT MEASURED.
The mHC row assumes no normalized-plane store. With that store the memory
fixture is MEASURED 1.231950 ms and the ratio is ESTIMATE 1.093. The instruction
cell counts the core dot; the boundary also does normalization, gates and Sinkhorn.
The block FP8 kernel mixes instructions. Its pure instruction reference does
not measure the issue rate of the full mixed loop.

MEASURED empty-launch cadence is 0.002055848 ms; the checked tiny kernel is
0.002078344 ms in `diag-limits-rates2-1`. Multiplying the empty cadence by
28,990 launches gives ESTIMATE 59.598 ms. This is not an additive saved time.

Validation: top-five product and boundary replays passed consumed-output checks
in `diag-limits-all-1`. Dynamic instruction controls and matched memory fixtures
passed in `diag-limits-rates2-1`. The Linux benchmark target passed an offline,
locked check with warnings denied. `diag-limits-replay-1` is invalid: it used a
wrong tensor-image path and its bench exited 1. No result from it is accepted.

The `rest` replay source compiles but has NOT RUN. The lead stopped limit work
after `diag-limits-rates2-1`. Original descriptor captures, matched latent gather
and range joins, and the mixed FP8/BF16/conversion issue loop remain NOT MEASURED.
