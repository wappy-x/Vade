State now: two-chat batches use the existing grouped route; the completed same-pair bracket proves a modest win. Merged branch is ready for the second turn and lead gate.
Proven: all four runs preserve 51/51 ID lists and 7264 tokens each; merged host suite 526 passed, 0 failed, 1 ignored; all seven source checks passed, 0 dirty.
Next step: second-model review and the lead's merged-head four-load/golden/k32 gate. The 55 ms decode and 85 ms verify targets need further cuts.

# PAIR first shot

Branch `day-pair-1`, draft [PR 228](https://github.com/wappy-x/vade-next/pull/228). Measured candidate `566adaeb8f4a724db7c34dfdacf092a0cd4c3924`; measured main `64cd96e586485aa2bcbfdb9cc2c3560aafd764b9`. Current main `42c6b6096e1f551b51968f4fc1a74013b82cf651` merged cleanly at `be2552fe616e2019ba537627173a02fac5255371`. Merged source validation head: `d9dd1aeaa341ee84c071cfb46652361e5563f329`; later commits contain receipts only.

## Change and proof

`ROUTED_CHATS` changes from 2 to 1. Two chats use the existing grouped row route, including its sparse-sharing device-route selection. The two copied chats can share expert work. No kernel entry, arithmetic chain, allocation type, memory charge, prompt path or drafting policy changes. One-row dispatch is unchanged. The emitted PTX is identical between main and both candidates.

The new host test covers widths 2 through 8, decreasing and increasing membership, row-slot rotation, differing positions, and exact logits, IDs and every state plane against each chat alone. Copies must have twice as many expert picks as distinct experts. It fails on unchanged main (0 passed/1 failed, grouped picks 0 instead of 4 at width 2), and passes with the route change (1 passed/0 failed). The served counters below prove the device runs took the intended path.

## Measure first

`pair-main-trace-01` ran on unchanged main, pair one, before any code change. Added windows cover three chats, two copies and three-row drafting. All 50/50 lists and 7104 tokens equal the fixed baseline. `reports/PAIR-trace.md` contains the full group/kernel, launch, message and idle table, plus 1/2/3/4/8-row/copy windows and both ranks of the verify interiors.

The measured one-to-two-row delta is 28.023 ms (41.959 to 69.983). Shared projections add 11.478 ms across several replaced forms; routed NVFP4 adds 8.782 ms in one family; attention adds 2.804 ms. The first cut tests the largest individually attributable repeated-kernel increase. Two-copy NVFP4 is 18.098 ms versus 18.241 for different chats and 9.460 for one chat. Actual DRAM/L2 bytes are unavailable in this capture; logical-byte figures are explicitly ESTIMATES. HELP for an authorized counter kit is recorded. No baseline or golden changed.

## Verdict

Two different chats: M1/C1/C2/M2 = 69.292169 / 67.876506 / 67.950445 / 69.238030 ms. Mean latency improves **1.95%**. The slowest candidate beats the faster main by **1.287585 ms**, larger than the two-main spread of **0.054139 ms**.

Two copies: M1/C1/C2/M2 = 68.476524 / 66.855120 / 67.082186 / 68.398120 ms. Mean latency improves **2.15%**. The slowest candidate beats the faster main by **1.315934 ms**, larger than the two-main spread of **0.078404 ms**.

Every one-chat candidate mean is below both main means. The untargeted load means show no regression larger than their two-main spread; no speed gain is claimed for those controls. Four-row verify remains about 102 ms, so the end target remains unmet.

## Untraced bracket, pair one

Both candidates use archived head `566adaeb8f4a724db7c34dfdacf092a0cd4c3924`; both references use `64cd96e586485aa2bcbfdb9cc2c3560aafd764b9`. Engine means omit the first eight steps.

| Engine ms | M1 | C1 | C2 | M2 |
|---|---:|---:|---:|---:|
| one_prose | 41.140828 | 41.042562 | 41.117037 | 41.142497 |
| one_code | 41.189972 | 41.102823 | 41.158261 | 41.185235 |
| one_structured | 41.199394 | 41.130253 | 41.159909 | 41.206901 |
| c2 | 69.292169 | 67.876506 | 67.950445 | 69.238030 |
| s2 | 68.476524 | 66.855120 | 67.082186 | 68.398120 |
| c3 | 81.757433 | 81.656170 | 81.850482 | 81.823220 |
| c4 | 91.422607 | 91.281311 | 91.303794 | 91.405585 |
| c8 | 143.296194 | 143.127636 | 143.225462 | 143.200074 |
| s8 | 95.028485 | 94.671636 | 94.930529 | 95.057375 |
| verify3 | 93.277575 | 93.510419 | 93.320787 | 93.565791 |
| verify4 | 102.019120 | 101.776415 | 101.752632 | 101.192867 |
| read8k_engine_ms | 32323.326452 | 32431.639325 | 32146.130707 | 32095.654599 |

The read row sums the eight engine prefill-group timers for the 8192-token prompt; it is not client time to first token.

| Client tokens/s (cross-check) | M1 | C1 | C2 | M2 |
|---|---:|---:|---:|---:|
| PL-one-prose-s1-P | 24.265066 | 24.318965 | 24.280905 | 24.266157 |
| PL-one-code-s1-P | 24.224718 | 24.289886 | 24.252182 | 24.230705 |
| PL-one-structured-s1-P | 24.226841 | 24.264057 | 24.247076 | 24.222931 |
| PL-c2 | 28.739757 | 29.435574 | 29.408750 | 28.770549 |
| PL-s2 | 29.087209 | 29.792938 | 29.693224 | 29.115912 |
| PL-c3 | 36.699515 | 36.752065 | 36.668774 | 36.674603 |
| PL-c4 | 43.700694 | 43.764560 | 43.758175 | 43.710155 |
| PL-c8 | 55.661286 | 55.765708 | 55.746327 | 55.761320 |
| PL-s8 | 83.104530 | 83.345506 | 83.309337 | 83.181655 |

| Run | Mac load at launch (1/5/15 min) | At completion |
|---|---|---|
| M1 | 3.83 / 13.91 / 17.66 | 15.20 / 10.25 / 13.44 |
| C1 | 16.38 / 10.58 / 13.53 | 10.67 / 10.18 / 12.26 |
| C2 | 10.30 / 10.11 / 12.22 | 13.07 / 14.92 / 14.47 |
| M2 | 17.66 / 15.91 / 14.81 | 9.79 / 12.36 / 13.55 |

No local host suite, build or trace analysis overlapped the untraced loads. M2 waited for the reserved pair-one slot; the run starts were 11:42:29 / 11:50:57 / 11:59:22 / 12:32:41. Each candidate is within 45 minutes of both reference starts. M1 and M2 are 50:12 apart; that wait is disclosed rather than presented as back-to-back runs.

- M1 `pair-route-m1-01`: **ANSWERS VERDICT: SAME AS MAIN**, 51/51 lists, 7264 tokens; all four loads. Driver exit 0, pair released, run folders removed.
- C1 `pair-route-c1-01`: **ANSWERS VERDICT: SAME AS MAIN**, 51/51 lists, 7264 tokens; all four loads. Driver exit 0, pair released, run folders removed.
- C2 `pair-route-c2-01`: **ANSWERS VERDICT: SAME AS MAIN**, 51/51 lists, 7264 tokens; all four loads. Driver exit 0, pair released, run folders removed.
- M2 `pair-route-m2-02`: **ANSWERS VERDICT: SAME AS MAIN**, 51/51 lists, 7264 tokens; all four loads. Driver exit 0, pair released, run folders removed.

## Served path and image proof

Counter deltas span the last prompt of the two-chat window through the next prompt. Comparing identical requests removes common next-prompt work. Each candidate adds 127 × 42 = 5334 route launches for two different chats. Two copies likewise add 5334 routes and 16002 tensor-core expert launches.

| Run | c2 prompt_route delta | c2 fp4_cohort delta | s2 prompt_route delta | s2 fp4_cohort delta |
|---|---:|---:|---:|---:|
| M1 | 42 | 126 | 1344 | 1008 |
| C1 | 5376 | 207 | 6678 | 17010 |
| C2 | 5376 | 207 | 6678 | 17010 |
| M2 | 42 | 126 | 1344 | 1008 |

All four PTX images have SHA256 `74075403ccee3ec9600a3e93a8d9e3f5cb3d61bda36b8151a5710cfb4c516d3b`. Full build receipts are in the corresponding run directories.

## Merged validation and handoff

- Merged TP suite: 526 passed, 0 failed, 1 ignored, exit 0 (`logs/PAIR-tests-merged.out`). Initial candidate suite: 499 passed, 0 failed, 1 ignored.
- All seven source checks pass, 0 dirty: kernels, tensor-core kernels, host and Linux clippy, unsafe-site check, one-path check, CLI ratchet (`five/day-pair-1-merged/SUMMARY`). CLI ratchet: 1 passed.
- Privacy gate: 0 hits before each push. No local test, build or analysis overlapped the untraced loads.
- No kernel entry changed; the focused host negative and full served exactness are saved. No new buffer/charge requires a lane k32 run. Per the lead's 11:46/12:08 instructions, the merged head's four-load, golden and k32 gate is run by the lead; a merge-only second comparison is not required.

The failed pre-claim attempt `pair-route-m2-01` changed no device state. The completed M2 is `pair-route-m2-02`, on pair one. The branch includes current main `42c6b6096e1f551b51968f4fc1a74013b82cf651`; the second turn starts from this merged branch. No cross-pair result is used.

The 55 ms two-chat and 85 ms four-row verify targets remain unmet. Further shared-projection work needs a separate measured cut and an old/new byte bench with an intentional negative for any changed kernel. The three-chat arm here is an unchanged grouped control; a separate three-chat once-per-chat arm has not run. CHATS3's incoming 12:11 trace note also identifies host route-read waits in copied chats as a follow-up. Actual memory-traffic counters, second-model review and the lead's merged-head device gate remain outstanding.
