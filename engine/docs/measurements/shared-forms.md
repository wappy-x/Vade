# PAIR2 shared-projection dispatch — final proofs in progress

Per the lead's new 15:36 order, final proofs are active on pair four: k32 base/candidate, then the predeclared one-row decision with 14 start pairs, then saved served counters. No main merge by this seat; the second-turn seat reviews the pinned measured candidate. The broad selector remains rejected; the narrow selector and draft PR are retained.

Measured candidate: `455a2a295614942afe8b52d0c60a7927f350f7fa`, branch `day-pair2-1`. Measured base: `585071ce3ce987d00680d9fb8da946a4223d4161` (the requested day-pair-1 head, containing main `42c6b6096e1f551b51968f4fc1a74013b82cf651`). Later commits change only receipts. The lead explicitly retained this base after main advanced; these are not measurements of the subsequently merged main.

MEASURED on pair four, `pair2-narrow-b1/c1/c2/b2`: two-chat mean gain 2.269110 ms (3.345 percent), against base spread 0.003484 ms. Four-chat mean loss 0.007358 ms is smaller than base spread 0.092953 ms; four-row verification mean gain 0.077002 ms is smaller than base spread 0.498047 ms. The repeated broad four-row regression is absent in this bracket. Three-row verification is flat within the baseline spread. The brief's absolute two-chat and three-row targets remain unmet.

MEASURED one-row caution, same tags and pair: prose/code/structured mean penalties are 0.058468/0.038461/0.093238 ms; the corresponding base spreads are 0.120819/0.090889/0.106162 ms. These mean penalties are below the observed spreads, but both structured candidate means exceed both baselines. One-row non-regression needs confirmation before a train; no cause is assigned from this bracket.

MEASURED answers, same tags and pair: every run covers all four requested loads (`one chats read draft`), 51 of 51 token-ID lists and 7264 tokens, zero differences and no missing baseline. Every driver exited zero, released its claim, and removed its device folders. The final `PAIR four FREE` handoff is logged. No PAIR2 Mac build/test overlapped either bracket. No extra four-load answers run is currently missing for this candidate.

MEASURED images, same tags and pair: all four full PTX hashes are `74075403ccee3ec9600a3e93a8d9e3f5cb3d61bda36b8151a5710cfb4c516d3b`. Both baseline binaries match; both candidate binaries match. No kernel entry, arithmetic or output chain order was edited. The candidate's rank-zero shared_plain_fold counter maximum is 10668 in plain decode and zero in strict drafting in both candidate runs; these are recorded launches, not graph-replay counts.

## Narrow implementation and remaining work

The narrow selector changes only two-row calls: FP32 router uses the existing ordered one-row fold per descriptor; BF16 latent uses the existing packed row form; wide MX QKV and gate/up use existing small2 calls with existing rounding; single-matrix FP8 down/output/QB use the existing few form. Shared FP8 gate/up stays multi. Three/four-row MX, FP8 and BF16 choices and the FP32 batch are restored to the base.

WHY four rows: the rejected broad selector changed FP32 router mixed batches to scalar folds, BF16 sparse-index rows to folds, wide-output MX small4 to tile, FP8 QB multi to few, and FP8 QA small to few. Its served logs have no per-family timing. The form screen alone cannot identify which change caused the whole-step regression; no causal assignment is made. A served four-row cost trace of broad head `0110422707682c08511f1f2bb8f00d47a2687685` against the base is needed to locate it.


Pending: the required base/candidate k32 proof, one-row non-regression confirmation, a second implementation/review turn, and integration with then-current main before a train. The 15:36 order supersedes the earlier stop window; the requested READY/HELP line is due by 16:50. Vocabulary-head form screening remains the separate follow-up described below. The worktree is retained; current run ownership and releases are recorded in PAIR2.log.

## Narrow host/source validation

MEASURED Mac, no pair: `PAIR2-narrow-gpu` 452 passed, zero failed, 11 existing ignores; `PAIR2-narrow-tp` 527 passed, zero failed, one existing ignore. The full suites passed before a declaration-only relocation of the unchanged two_wins function ahead of the test module. `PAIR2-narrow-selector` then passed all four selector tests. Benchmark code is unchanged since `PAIR2-bench-green`, 71 passed, zero failed.

MEASURED Mac, no pair, `PAIR2-narrow-red`: the restored-three/four-row test fails on the rejected broad selector (zero passed, one expected assertion failure: Tile versus the base Small entry). It checks base entries and launch geometry plus row-alone bit equality after a neighbor changes. BF16 source parts and FP32 two-descriptor admission have their own host cases. The original dispatch red control is retained below.

MEASURED Mac, no pair: `five/day-pair2-1-narrow-final/SUMMARY` has seven checks exit zero and a clean tree (normal/TC kernels, host/Linux clippy, unsafe sites, one path, CLI ratchet). It began at `2e095986ebfc0a7ecd6c5963880b2b8fb918484d` and ended at `017c8d23487577d402e5e1c4d5c026d1cf96caa8`, with only a receipt document changing. The earlier gate attempt's sole issue was items_after_test_module; it was fixed by relocating the unchanged declaration, not by an allow attribute.

## Narrow served bracket

MEASURED on pair four. Columns in order: `pair2-narrow-b1`, `pair2-narrow-c1`, `pair2-narrow-c2`, `pair2-narrow-b2`.
Engine step/pass means omit the first eight; read8k is total prefill time. All are milliseconds, extracted by PAIR-metrics.py. Client rates are cross-checks; launch Mac loads are in PAIR2.log.

| Load | Base | Candidate | Candidate | Base | Base spread | Mean gain |
|---|---:|---:|---:|---:|---:|---:|
| one_prose | 40.984247 | 41.095516 | 41.110733 | 41.105066 | 0.120819 | -0.058468 |
| one_code | 41.026995 | 41.098123 | 41.123679 | 41.117884 | 0.090889 | -0.038461 |
| one_structured | 41.022320 | 41.146413 | 41.190866 | 41.128483 | 0.106162 | -0.093238 |
| c2 | 67.832487 | 65.567579 | 65.562659 | 67.835971 | 0.003484 | 2.269110 |
| c3 | 81.583439 | 81.571377 | 81.556657 | 81.503490 | 0.079949 | -0.020552 |
| c4 | 91.215894 | 91.142400 | 91.211151 | 91.122942 | 0.092953 | -0.007358 |
| c8 | 143.066271 | 142.863622 | 142.945009 | 143.029351 | 0.036920 | 0.143495 |
| s2 | 66.781731 | 64.683877 | 64.776011 | 66.757370 | 0.024361 | 2.039606 |
| s8 | 94.674827 | 94.542189 | 94.462039 | 94.534732 | 0.140095 | 0.102666 |
| verify3 | 87.201174 | 87.019771 | 87.045191 | 86.783800 | 0.417374 | -0.039994 |
| verify4 | 96.036731 | 95.501157 | 95.920254 | 95.538684 | 0.498047 | 0.077002 |
| read8k prefill total | 31981.905927 | 31831.165961 | 31921.142658 | 31860.229232 | 121.676695 | 44.913270 |


MEASURED answers and served dispatch receipts:

`pair2-narrow-b1`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum absent on baseline.
- DR: shared_plain_fold counter maximum absent on baseline.

`pair2-narrow-c1`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum 10668.
- DR: shared_plain_fold counter maximum 0.

`pair2-narrow-c2`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum 10668.
- DR: shared_plain_fold counter maximum 0.

`pair2-narrow-b2`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum absent on baseline.
- DR: shared_plain_fold counter maximum absent on baseline.

## Rejected broad selector and original form screen

The isolated form timings did not predict the integrated result at three/four rows. No cause is claimed from these timings alone. The first candidate also had a broader one/eight-chat slowdown; the repeated four-chat and four-row-verification regressions are enough to reject the cut without relying on that disturbance.

MEASURED pair four, `pair2-dispatch-b1/c1/c2/b2`: two-chat mean gain 2.032150 ms exceeds base spread 0.183198 ms. Four-chat mean loss 0.964095 ms exceeds base spread 0.109296 ms, and both candidate means exceed both baselines. Four-row-verification mean loss 1.071105 ms exceeds base spread 0.419649 ms. Three-row verification has no demonstrated gain. The target two-chat/three-row-verification times were not reached.

Both baselines use `585071ce3ce987d00680d9fb8da946a4223d4161`; both candidates use `0110422707682c08511f1f2bb8f00d47a2687685`. Commits through 363d2d9803f8cc627e4f665cf85be2c9c9bdcfd7 only retain receipts; dae0b33dac9d68079f234e5b4f1b29d204d6736f implements the later narrow variant. The lead explicitly kept this base valid after main advanced.

MEASURED full kernel-image SHA256 is identical in all four runs and the form screen: `74075403ccee3ec9600a3e93a8d9e3f5cb3d61bda36b8151a5710cfb4c516d3b`. Baseline binaries match each other; candidate binaries match each other. No kernel entry or arithmetic was edited.

MEASURED answers, same tags and pair: each run checks all 51 lists and 7264 tokens, with zero differences and zero lists lacking a baseline. Every driver exited zero, released its claim and removed its device folders. The PAIR four FREE handoff is logged. The Mac ran no PAIR2 build or test during the bracket.

## Local validation

MEASURED on the Mac, no pair: `PAIR2-gpu-green` 451 passed, zero failed, 11 existing ignores; `PAIR2-tp-green` 527 passed, zero failed, one existing ignore; `PAIR2-bench-green` 71 passed, zero failed. The `PAIR2-fold-red` control failed its dispatch assertion before implementation (zero passed, one expected failure); the green case also checks independent two/three-row words and a changed neighbor.

MEASURED `five/day-pair2-1/SUMMARY`, Mac, no pair: seven checks exited zero, final tree clean (normal/TC kernels, host/Linux clippy, unsafe sites, one path, CLI ratchet). Checks began at cc772853e3e7ab10c3d2f49db4817abcaf502af9 and ended at 94273ea97d83ffe59404f24634822f76579c9053, with only a receipt document changing. The benchmark Linux compilation also passed before the device form screen.

## Served bracket

MEASURED on pair four. Columns in order: `pair2-dispatch-b1`, `pair2-dispatch-c1`, `pair2-dispatch-c2`, `pair2-dispatch-b2`.
Engine step/pass means omit the first eight; read8k is total prefill time. All are milliseconds, extracted by PAIR-metrics.py. Client rates below are cross-checks only; launch Mac loads are in PAIR2.log.

| Load | Base | Candidate | Candidate | Base | Base spread | Mean gain |
|---|---:|---:|---:|---:|---:|---:|
| one_prose | 40.919298 | 41.274613 | 41.020562 | 41.071965 | 0.152667 | -0.151956 |
| one_code | 40.987460 | 41.820080 | 41.077644 | 41.103584 | 0.116124 | -0.403340 |
| one_structured | 41.020146 | 41.246774 | 41.076619 | 41.104888 | 0.084741 | -0.099180 |
| c2 | 67.617199 | 65.690523 | 65.662773 | 67.800397 | 0.183198 | 2.032150 |
| c3 | 81.462832 | 81.728837 | 81.696961 | 81.531315 | 0.068484 | -0.215826 |
| c4 | 91.101719 | 92.172087 | 92.068835 | 91.211015 | 0.109296 | -0.964095 |
| c8 | 142.787560 | 145.202630 | 142.908186 | 142.767601 | 0.019960 | -1.277828 |
| s2 | 66.519736 | 64.704317 | 64.608299 | 66.779336 | 0.259600 | 1.993228 |
| s8 | 94.234943 | 94.513156 | 94.741802 | 94.412449 | 0.177507 | -0.303783 |
| verify3 | 86.833545 | 86.703280 | 86.850192 | 86.414864 | 0.418682 | -0.152531 |
| verify4 | 95.599138 | 96.574735 | 96.346101 | 95.179488 | 0.419649 | -1.071105 |
| read8k prefill total | 31819.769783 | 31772.342171 | 31809.565060 | 31819.830356 | 0.060573 | 28.846454 |

MEASURED client tokens/second, same run tags and pair:

| Load | Base | Candidate | Candidate | Base |
|---|---:|---:|---:|---:|
| DR-d-code-s1-P | 24.241097 | 24.253569 | 24.218654 | 24.241405 |
| DR-d-code-s2-S2 | 28.141084 | 28.135457 | 28.121831 | 28.143291 |
| DR-d-code-s3-S3 | 29.246727 | 28.968409 | 28.992681 | 29.300573 |
| DR-d-prose-s1-P | 24.286027 | 24.275604 | 24.218345 | 24.271795 |
| DR-d-prose-s2-S2 | 23.641178 | 23.491049 | 23.446408 | 23.529055 |
| DR-d-prose-s3-S3 | 23.455161 | 23.223566 | 23.249396 | 23.033407 |
| DR-d-structured-s1-P | 24.240785 | 24.229619 | 24.211238 | 24.228439 |
| DR-d-structured-s2-S2 | 30.587559 | 30.614256 | 30.552895 | 30.667279 |
| DR-d-structured-s3-S3 | 35.411325 | 35.692122 | 35.172437 | 36.154663 |
| DR-warm-structured-s0-P | 24.079117 | 24.060606 | 24.073201 | 24.075313 |
| DR-warmS-structured-s10-S3 | 30.949016 | 30.597278 | 30.637796 | 30.935012 |
| PL-c2 | 29.551606 | 30.416885 | 30.424697 | 29.472873 |
| PL-c3 | 36.835236 | 36.737656 | 36.751197 | 36.813324 |
| PL-c4 | 43.865694 | 43.364961 | 43.409934 | 43.816136 |
| PL-c8 | 55.930610 | 55.047283 | 55.881762 | 55.936999 |
| PL-one-code-s1-P | 24.353233 | 23.849968 | 24.302994 | 24.281947 |
| PL-one-prose-s1-P | 24.392823 | 24.186764 | 24.338421 | 24.308968 |
| PL-one-structured-s1-P | 24.333036 | 24.208167 | 24.302249 | 24.288245 |
| PL-s2 | 29.938884 | 30.778718 | 30.821274 | 29.830447 |
| PL-s8 | 83.892817 | 83.667578 | 83.466643 | 83.725082 |
| PL-warm | 56.184790 | 54.543398 | 56.058201 | 56.240189 |
| PL-warm-prose-s0-P | 24.067404 | 23.329812 | 24.045487 | 24.023611 |

MEASURED answers and served dispatch receipts:

`pair2-dispatch-b1`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum absent on baseline.
- DR: shared_plain_fold counter maximum absent on baseline.

`pair2-dispatch-c1`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum 21336.
- DR: shared_plain_fold counter maximum 16128.

`pair2-dispatch-c2`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum 21336.
- DR: shared_plain_fold counter maximum 15456.

`pair2-dispatch-b2`, pair four: ANSWERS VERDICT: SAME AS MAIN
- PL: shared_plain_fold counter maximum absent on baseline.
- DR: shared_plain_fold counter maximum absent on baseline.

## Original form screen

MEASURED times in microseconds from `pair2-forms-02`, pair four, unchanged base `585071ce3ce987d00680d9fb8da946a4223d4161`, bench head `6f30245c9f6ed76ee42a13e066f45f1b37ceb964`. Device timestamps inside captured graphs; rotating weight copies; each timed operand set is byte-checked. Image SHA256 `74075403ccee3ec9600a3e93a8d9e3f5cb3d61bda36b8151a5710cfb4c516d3b`.

Calls are ESTIMATE by checkpoint inventory and current grouping, cross-checked against MEASURED family counts in `pair-main-trace-01`, pair one. No cross-pair speed comparison. Savings are ESTIMATE, not served measurements.

Storage changes: `bf16_latent` currently reads widened FP32 at two/three rows; its alternatives read the same BF16 patterns in packed storage. FP32 router has no packed copy and uses FP32 in both arms. Standalone index rows currently read widened FP32; the one-row reader reads the matching packed BF16. Narrow current folds and row forms use packed BF16.

| Family | m | Current us | One-row form repeated us | Other forms us | Bytes | Calls/step | Best admitted | Saving ms ESTIMATE |
|---|---:|---:|---:|---|---|---:|---|---:|
| fp32_router | 2 | mixed 62.419 | scalar_fold 24.141 | loop 123.053 | SAME | 42 | scalar_fold | 1.607676 |
| fp32_router | 3 | mixed 62.600 | scalar_fold 34.662 | loop 184.312 | SAME | 42 | scalar_fold | 1.173396 |
| fp32_router | 4 | mixed 62.840 | scalar_fold 45.032 | loop 245.734 | SAME | 42 | scalar_fold | 0.747936 |
| bf16_narrow | 2 | fold 44.998 | packed_scalar 190.336 | mixed 102.550, loop 151.076, rows 78.601 | SAME | 34 | fold | 0.000000 |
| bf16_narrow | 3 | fold 64.122 | packed_scalar 285.386 | mixed 100.676, loop 224.571, rows 78.847 | SAME | 34 | fold | 0.000000 |
| bf16_narrow | 4 | rows 80.239 | packed_scalar 380.565 | mixed 200.286, loop 299.851, fold 83.922 | SAME | 34 | rows | 0.000000 |
| bf16_index_weights | 2 | fold 33.885 | packed_scalar 188.489 | mixed 81.172, loop 145.938, rows 73.678 | SAME | 11 | fold | 0.000000 |
| bf16_index_weights | 3 | fold 49.769 | packed_scalar 282.675 | mixed 80.152, loop 218.708, rows 73.560 | SAME | 11 | fold | 0.000000 |
| bf16_index_weights | 4 | rows 73.590 | packed_scalar 376.940 | mixed 142.974, loop 290.078, fold 60.700 | SAME | 11 | fold | 0.141790 |
| bf16_latent | 2 | mixed 28.212 | packed_scalar 18.451 | loop 19.766, rows 15.038 | SAME | 34 | rows | 0.447916 |
| bf16_latent | 3 | mixed 31.309 | packed_scalar 23.290 | loop 24.792, rows 18.003 | SAME | 34 | rows | 0.452404 |
| bf16_latent | 4 | rows 16.952 | packed_scalar 28.181 | mixed 28.800, loop 29.818 | SAME | 34 | rows | 0.000000 |
| mx_qkv | 2 | rows 256.251 | loop 487.109 | small 238.016, pair 295.397, tile 285.477 | SAME | 34 | small | 0.619990 |
| mx_qkv | 3 | rows 254.149 | loop 730.112 | small 325.307, pair 323.477, tile 285.973 | SAME | 34 | rows | 0.000000 |
| mx_qkv | 4 | rows 262.901 | loop 976.720 | small 349.253, pair 364.277, tile 287.328 | SAME | 34 | rows | 0.000000 |
| mx_gate_up | 2 | rows 245.125 | loop 490.507 | small 229.675, pair 261.637, tile 277.435 | SAME | 3 | small | 0.046350 |
| mx_gate_up | 3 | rows 242.795 | loop 735.867 | small 297.915, pair 286.155, tile 278.800 | SAME | 3 | rows | 0.000000 |
| mx_gate_up | 4 | rows 249.755 | loop 1003.744 | small 315.963, pair 338.683, tile 278.427 | SAME | 3 | rows | 0.000000 |
| mx_output | 2 | small 88.144 | loop 148.524 | rows 99.602, pair 112.726, tile 92.496 | SAME | 34 | small | 0.000000 |
| mx_output | 3 | small 122.440 | loop 200.556 | rows 102.982, pair 128.468, tile 93.148 | SAME | 34 | tile | 0.995928 |
| mx_output | 4 | small 137.710 | loop 250.706 | rows 112.154, pair 152.760, tile 93.430 | SAME | 34 | tile | 1.505520 |
| mx_down | 2 | small 129.987 | loop 240.209 | rows 152.652, pair 166.900, tile 134.767 | SAME | 3 | small | 0.000000 |
| mx_down | 3 | small 181.935 | loop 343.244 | rows 159.372, pair 191.020, tile 134.595 | SAME | 3 | tile | 0.142020 |
| mx_down | 4 | small 202.656 | loop 436.890 | rows 169.449, pair 223.404, tile 134.967 | SAME | 3 | tile | 0.203067 |
| fp8_gate_up | 2 | fp8_rows_multi 49.645 | expert_operation 64.275 | fp8_rows_small 51.508, fp8_rows_few 45.824, expert_group 128.366, fp8_rows_tile 185.999 | SAME | 42 | fp8_rows_multi | 0.000000 |
| fp8_gate_up | 3 | fp8_rows_multi 49.739 | expert_operation 87.297 | fp8_rows_small 72.863, fp8_rows_few 48.280, expert_group 135.615, fp8_rows_tile 184.568 | SAME | 42 | fp8_rows_multi | 0.000000 |
| fp8_gate_up | 4 | fp8_rows_multi 49.850 | expert_operation 110.202 | fp8_rows_small 95.223, fp8_rows_few 49.544, expert_group 133.553, fp8_rows_tile 184.516 | SAME | 42 | fp8_rows_multi | 0.000000 |
| fp8_down | 2 | fp8_rows_small 29.200 | expert_operation 35.069 | fp8_rows_multi 29.848, fp8_rows_few 26.044, expert_group 64.177, fp8_rows_tile 95.354 | SAME | 42 | fp8_rows_few | 0.132552 |
| fp8_down | 3 | fp8_rows_few 27.837 | expert_operation 47.538 | fp8_rows_small 39.770, fp8_rows_multi 29.752, expert_group 68.343, fp8_rows_tile 94.031 | SAME | 42 | fp8_rows_few | 0.000000 |
| fp8_down | 4 | fp8_rows_few 28.387 | expert_operation 60.154 | fp8_rows_small 50.743, fp8_rows_multi 30.096, expert_group 66.062, fp8_rows_tile 94.001 | SAME | 42 | fp8_rows_few | 0.000000 |
| fp8_output | 2 | fp8_rows_small 182.632 | expert_operation 293.992 | fp8_rows_multi 169.036, fp8_rows_few 166.012, expert_group 548.744, fp8_rows_tile 737.660 | SAME | 11 | fp8_rows_few | 0.182820 |
| fp8_output | 3 | fp8_rows_multi 170.060 | expert_operation 432.896 | fp8_rows_small 268.708, fp8_rows_few 171.224, expert_group 524.200, fp8_rows_tile 726.416 | SAME | 11 | fp8_rows_multi | 0.000000 |
| fp8_output | 4 | fp8_rows_multi 169.544 | expert_operation 577.268 | fp8_rows_small 355.784, fp8_rows_few 176.312, expert_group 527.192, fp8_rows_tile 726.136 | SAME | 11 | fp8_rows_multi | 0.000000 |
| fp8_qb | 2 | fp8_rows_small 75.420 | expert_operation 98.214 | fp8_rows_multi 76.687, fp8_rows_few 70.420, expert_group 114.929, fp8_rows_tile 198.340 | SAME | 11 | fp8_rows_few | 0.055000 |
| fp8_qb | 3 | fp8_rows_multi 77.360 | expert_operation 134.932 | fp8_rows_small 106.087, fp8_rows_few 71.153, expert_group 117.584, fp8_rows_tile 198.249 | SAME | 11 | fp8_rows_few | 0.068277 |
| fp8_qb | 4 | fp8_rows_multi 78.269 | expert_operation 170.813 | fp8_rows_small 139.699, fp8_rows_few 71.959, expert_group 142.262, fp8_rows_tile 199.683 | SAME | 11 | fp8_rows_few | 0.069410 |
| fp8_qa | 2 | fp8_rows_small 22.159 | expert_operation 37.409 | fp8_rows_multi 34.439, fp8_rows_few 30.951, expert_group 124.561, fp8_rows_tile 189.906 | SAME | 11 | fp8_rows_small | 0.000000 |
| fp8_qa | 3 | fp8_rows_small 25.830 | expert_operation 54.076 | fp8_rows_multi 34.598, fp8_rows_few 31.091, expert_group 133.869, fp8_rows_tile 187.138 | SAME | 11 | fp8_rows_small | 0.000000 |
| fp8_qa | 4 | fp8_rows_small 48.123 | expert_operation 70.727 | fp8_rows_multi 34.853, fp8_rows_few 31.391, expert_group 142.046, fp8_rows_tile 190.060 | SAME | 11 | fp8_rows_few | 0.184052 |
| fp8_kva | 2 | fp8_rows_small 17.707 | expert_operation 34.023 | fp8_rows_multi 31.525, fp8_rows_few 29.437, expert_group 126.279, fp8_rows_tile 184.978 | SAME | 11 | fp8_rows_small | 0.000000 |
| fp8_kva | 3 | fp8_rows_small 17.730 | expert_operation 49.644 | fp8_rows_multi 31.658, fp8_rows_few 29.553, expert_group 120.744, fp8_rows_tile 188.353 | SAME | 11 | fp8_rows_small | 0.000000 |
| fp8_kva | 4 | fp8_rows_small 20.946 | expert_operation 66.086 | fp8_rows_multi 31.944, fp8_rows_few 29.778, expert_group 129.699, fp8_rows_tile 188.546 | SAME | 11 | fp8_rows_small | 0.000000 |
| plain_index_rows | 2 | plain_rows_small 64.007 | project_bf16_ordered 46.886 | plain_rows_tile 89.348, prompt_rows 105.063 | SAME | 11 | plain_rows_small | 0.000000 |
| plain_index_rows | 3 | plain_rows_small 64.883 | project_bf16_ordered 62.452 | plain_rows_tile 89.180, prompt_rows 104.420 | SAME | 11 | plain_rows_small | 0.000000 |
| plain_index_rows | 4 | plain_rows_small 65.834 | project_bf16_ordered 72.140 | plain_rows_tile 89.764, prompt_rows 103.849 | SAME | 11 | plain_rows_small | 0.000000 |

FP8 shared gate/up keeps `multi`: `few` rejects multi-part tables and small/few timings duplicate the input before timing. They are not candidates for this dispatch-only cut. The standalone one-row FP8 and plain-index timings exclude a layout conversion and are also excluded. All other timing arms include their required output rounding. Quantization common to MX arms and gathers common to every arm are outside the timer.

The narrow plain groups are 128+32+128 x 4096 (recurrent) and 64+64+16 x 4096 (index). The short-input group is two 4096 x 128 matrices. Their final outputs are unrounded, as in the service.

Initial run `pair2-forms-01`, pair four, is diagnostic only and superseded: its plain fallback storage and narrow groups did not match the service. Both drivers exited zero and released their pairs.

ESTIMATE saving by admissible existing dispatch: 3.092304 ms at 2 rows, 2.832025 ms at 3 rows, 2.851775 ms at 4 rows. This clears the required two-row threshold. No served speed or token verdict is claimed.

Next, outside this first cut per the lead: vocabulary-head MXFP8. MEASURED `pair-main-trace-01`, pair one, rank zero: one call per step, 1.45647 ms at one row and 1.47462 ms at two rows (kernel time, after the trace warmup). This is a separate study band; it has no form screen in this branch. Absorbed KV and mHC remain attention math outside PAIR2.

Client-side launch cross-check for the bracket (pair four): no local host build/test overlapped any of the four measured starts (`pair2-dispatch-b1`, `pair2-dispatch-c1`, `pair2-dispatch-c2`, `pair2-dispatch-b2`).


## One-row decision protocol, declared before the run

ONE-ROW PREDECLARATION for pair2-one-ab, pair four, before any decision run: fixed base 585071ce3ce987d00680d9fb8da946a4223d4161 and narrow 455a2a295614942afe8b52d0c60a7927f350f7fa; 14 fresh candidate starts bracketed by 15 base starts after the excluded warm start. Each measured start uses the unchanged kit warm request and prose/code/structured requests of 352 tokens. The primary observation is the rank-zero engine SERVED STEP seconds converted to milliseconds, pooling the three measured requests after dropping each request's first eight steps. Arm-blind disturbed-run rule: discard a whole measured start iff strictly more than 5 percent of its timed steps exceed that start's timed-step median plus 1.5 ms. No other discretionary exclusions. A clean candidate uses the nearest clean base start on each side, interpolated at its decode midpoint; without both sides it contributes no paired difference. Primary interval: the kit paired() mean narrow-minus-interpolated-base and Student-t 95 percent interval, with its covariance correction for shared base starts, on retained runs. Report unfiltered interval, retained counts, every exclusion, per-kind results, and all ID comparisons as cross-checks. PASS iff the filtered interval includes zero or its upper endpoint is at most +0.05 ms; if its lower endpoint exceeds +0.05 ms, write HELP with the numbers, not READY; a positive interval straddling +0.05 ms is inconclusive and also goes to HELP. Too few valid pairs or incomplete/unequal IDs cannot pass. Do not stop early based on a favorable interim interval.

## k32 receipt: pair2-k32-b

MEASURED `pair2-k32-b`, pair four: burst 9/9 requests passed; all 18 requests complete; turn-2 IDs equal cold. Turn-1/turn-2/cold-turn-2 first-token times 33.877519/2.704466/35.713939 s; total times 36.823917/5.631471/38.632101 s. Cached 8192 tokens; burst widest decode batch 5 rows, longest admission queue 115719.040000 ms. Driver exit zero and pair released.
