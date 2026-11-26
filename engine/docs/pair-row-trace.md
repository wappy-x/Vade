# PAIR matched main trace

State: matched main trace and all secondary-window analyses complete.
Proven: 50/50 token lists, 7104 IDs equal the fixed main baseline. No engine code changed for this measurement.
Next: test the largest attributable individual kernel increase, repeated routed expert work, using the existing grouped path.

MEASURED run `pair-main-trace-01`, main `64cd96e586485aa2bcbfdb9cc2c3560aafd764b9`, pair one. All values below are traced. Do not compare them with untraced speed claims. The driver exited 0, released the pair and removed its run folders.

## Exclusive path across both ranks

Each instant belongs to one band. The measured delta is 28.023 ms; it agrees with the earlier 28.095 ms untraced delta within 0.072 ms, but these are separate measurements. The table accounts for the entire fresh delta. Categories: a = repeated weight work; b = a different row form or extra row arithmetic; c = launch/host wait; d = message work. Categories explain source dispatch, not hardware-counter causality.

| Group | 1 row ms | 2 rows ms | Delta ms | Category |
|---|---:|---:|---:|---|
| routed expert weights (NVFP4) | 9.459701 | 18.241399 | 8.781697 | a |
| shared weights (MXFP8, block FP8, BF16) | 18.658726 | 30.136739 | 11.478012 | a/b |
| output head | 1.458550 | 1.473492 | 0.014942 | b |
| attention math | 4.573114 | 7.377604 | 2.804490 | b |
| latent attention expansion | 0.000000 | 0.000000 | 0.000000 | b |
| mHC | 2.071891 | 2.820881 | 0.748990 | b |
| messages between Sparks | 3.840068 | 5.932135 | 2.092067 | d |
| expert routing and quantize | 0.436117 | 0.901325 | 0.465208 | b |
| small math and copies | 0.778587 | 1.748573 | 0.969987 | b/c |
| idle (GPU waiting) | 0.682708 | 1.350515 | 0.667808 | c |
| Total | 41.959462 | 69.982662 | 28.023200 | |

Windows contain 246 one-row steps and 118 two-row steps, after discarding the first eight. Cross-rank fitting uses matching put/take counts and the supplied xrank tool.

## Kernel family detail

Times are exclusive cross-rank path times. Launch counts are rank-zero executions per step, including graph nodes, from the family census. A new family at two rows can replace a one-row family; its full time is not an incremental penalty.

| Family | 1 row ms | 2 rows ms | Delta ms | 1 row calls | 2 row calls | Category |
|---|---:|---:|---:|---:|---:|---|
| flash_resident_projection_bf16_rows | 0.000000 | 9.497116 | 9.497116 | 0.0 | 37.0 | b |
| NVFP4 | 9.459701 | 18.241399 | 8.781697 | 84.0 | 168.0 | a |
| mixed | 0.000000 | 4.088990 | 4.088990 | 0.0 | 76.0 | b |
| fp8_rows_small | 0.000000 | 4.060077 | 4.060077 | 0.0 | 44.0 | b |
| mx_rows_small2 | 0.000000 | 3.511205 | 3.511205 | 0.0 | 37.0 | b |
| fp8_rows_multi | 0.000000 | 2.630306 | 2.630306 | 0.0 | 42.0 | b |
| plain_batch | 0.101070 | 2.304031 | 2.202961 | 11.0 | 45.0 | b |
| causal_fast | 1.184911 | 2.991212 | 1.806301 | 197.8 | 366.6 | b |
| flash_next_mxfp8_pair | 0.000000 | 1.473492 | 1.473492 | 0.0 | 1.0 | b |
| prompt_rows | 0.000000 | 0.911033 | 0.911033 | 0.0 | 174.0 | b |
| causal | 1.224322 | 2.133099 | 0.908776 | 132.8 | 260.6 | b |
| plain_rows_small | 0.000000 | 0.796137 | 0.796137 | 0.0 | 11.0 | b |
| mhc_boundary_fused | 2.071891 | 2.820881 | 0.748990 | 90.0 | 180.0 | b |
| absorb_dense | 0.874343 | 1.389734 | 0.515391 | 22.0 | 44.0 | b |
| device_route | 0.436117 | 0.901325 | 0.465208 | 84.0 | 168.0 | b |
| put_join | 1.172581 | 1.636146 | 0.463565 | 245.0 | 291.0 | d |
| copy | 0.081374 | 0.500389 | 0.419015 | 18.8 | 141.4 | b |
| absorb_attention | 0.475806 | 0.863559 | 0.387753 | 22.0 | 44.0 | b |
| take_scatter | 1.532292 | 1.871192 | 0.338901 | 245.0 | 291.0 | d |
| idle: host between driver calls | 0.289922 | 0.600757 | 0.310835 | 0.0 | 0.0 | c |
| pointwise | 0.027896 | 0.337288 | 0.309392 | 11.0 | 138.8 | b |
| prompt_prepare | 0.000000 | 0.264039 | 0.264039 | 0.0 | 34.0 | b |
| put | 0.000000 | 0.190936 | 0.190936 | 0.0 | 48.0 | d |
| prompt_gated_norm | 0.000000 | 0.170297 | 0.170297 | 0.0 | 34.0 | b |
| idle: host in cuGraphInstantiate_v2 | 0.010955 | 0.162367 | 0.151413 | 0.0 | 0.0 | c |
| idle: host in cuStreamSynchronize | 0.313742 | 0.452553 | 0.138811 | 0.0 | 0.0 | c |
| take | 0.000000 | 0.119665 | 0.119665 | 0.0 | 49.0 | d |
| prompt_permute | 0.000000 | 0.090171 | 0.090171 | 0.0 | 38.0 | b |
| prompt_select_rows | 0.030036 | 0.079437 | 0.049401 | 11.0 | 22.0 | b |
| copy H2D | 0.080668 | 0.117743 | 0.037075 | 20.0 | 30.0 | b |
| model_rms_partials | 0.000000 | 0.032846 | 0.032846 | 0.0 | 23.0 | b |
| model_rms_apply | 0.000000 | 0.032581 | 0.032581 | 0.0 | 23.0 | b |
| idle: host in cuLaunchKernel | 0.005201 | 0.034150 | 0.028949 | 0.0 | 0.0 | c |
| idle: host in cuGraphLaunch | 0.040061 | 0.063886 | 0.023825 | 0.0 | 0.0 | c |
| put_wide | 0.000000 | 0.015267 | 0.015267 | 0.0 | 1.0 | d |
| idle: host in cuGraphDestroy | 0.000440 | 0.007621 | 0.007181 | 0.0 | 0.0 | c |
| idle: host in cuCtxSetCurrent | 0.006732 | 0.013291 | 0.006558 | 0.0 | 0.0 | c |
| idle: host in cuEventCreate | 0.000241 | 0.001476 | 0.001235 | 0.0 | 0.0 | c |
| idle: host in cuEventRecord | 0.000291 | 0.001312 | 0.001021 | 0.0 | 0.0 | c |
| idle: host in cuStreamWaitEvent | 0.000227 | 0.001190 | 0.000963 | 0.0 | 0.0 | c |
| idle: host in cuEventDestroy_v2 | 0.000086 | 0.000914 | 0.000827 | 0.0 | 0.0 | c |
| idle: host in cuStreamEndCapture | 0.000055 | 0.000711 | 0.000655 | 0.0 | 0.0 | c |
| combine | 0.003945 | 0.004561 | 0.000617 | 1.0 | 1.0 | b |
| idle: host in cuStreamBeginCapture_v2 | 0.000040 | 0.000437 | 0.000397 | 0.0 | 0.0 | c |
| copy D2H | 0.004710 | 0.005014 | 0.000304 | 2.0 | 2.0 | b |
| idle: host in cuGraphGetNodes | 0.000009 | 0.000123 | 0.000114 | 0.0 | 0.0 | c |
| logits_argmax_finish | 0.001834 | 0.001883 | 0.000049 | 1.0 | 1.0 | b |
| idle: host in cuGraphExecUpdate_v2 | 0.000074 | 0.000117 | 0.000043 | 0.0 | 0.0 | c |
| logits_argmax | 0.002554 | 0.002549 | -0.000005 | 1.0 | 1.0 | b |
| idle: host in cuMemcpyDtoHAsync_v2 | 0.000082 | 0.000070 | -0.000012 | 0.0 | 0.0 | c |
| idle: host in cuMemAlloc_v2 | 0.000394 | 0.000044 | -0.000350 | 0.0 | 0.0 | c |
| batch_embedding | 0.004408 | 0.003916 | -0.000493 | 1.0 | 1.0 | b |
| idle: host in cuMemsetD8Async | 0.001406 | 0.000000 | -0.001406 | 0.0 | 0.0 | c |
| copy_words | 0.020766 | 0.018255 | -0.002512 | 11.0 | 11.1 | b |
| idle: host in cuMemcpyHtoDAsync_v2 | 0.012750 | 0.009496 | -0.003253 | 0.0 | 0.0 | c |
| side branch alone: shadow_side | 0.040907 | 0.029334 | -0.011573 | 0.0 | 0.0 | b |
| fill | 0.018240 | 0.000000 | -0.018240 | 11.0 | 0.0 | b |
| rms_partials | 0.089983 | 0.000000 | -0.089983 | 23.0 | 0.0 | b |
| norm_tail | 0.151979 | 0.000000 | -0.151979 | 23.0 | 0.0 | b |
| round | 0.219287 | 0.058270 | -0.161016 | 130.0 | 34.0 | b |
| plain_bf16 | 0.758156 | 0.000000 | -0.758156 | 79.0 | 0.0 | b |
| causal_chain | 0.813732 | 0.000000 | -0.813732 | 34.0 | 0.0 | b |
| plain | 1.049281 | 0.000000 | -1.049281 | 42.0 | 0.0 | b |
| output head | 1.458550 | 0.000000 | -1.458550 | 1.0 | 0.0 | b |
| MXFP8 | 2.636115 | 0.000000 | -2.636115 | 37.0 | 0.0 | b |
| FP8 block | 5.509307 | 2.337843 | -3.171464 | 128.0 | 84.0 | b |
| mixed_bf16 | 8.604798 | 0.000000 | -8.604798 | 37.0 | 0.0 | b |
| Fitted transit | 1.135196 | 2.098930 | 0.963734 | | | d |

## Bytes, messages, launches and idle

Actual DRAM bytes are NOT MEASURED by this nsys capture. The trace contains kernel/copy timing, not DRAM/L2 counters. Kernel bytes read must not be invented from duration. The following are ESTIMATE logical operands based on the saved checkpoint inventory in `reports/LIMIT-data/weight-recount.json` and current dispatch:

| Work | 1 row bytes | 2 row bytes | Interpretation |
|---|---:|---:|---|
| Selected NVFP4 matrix codes/scales | 2378170368 | 4756340736 | Nominal full operands; two separate per-chat passes. Zero columns, cache and prefetch change actual traffic. |
| Shared projection matrix inventory | 4001761280 | 4001761280 unique | Same matrices; some row forms reread them. Actual traffic unavailable. |
| Output head matrix inventory | 327106560 | 327106560 unique | Same matrix. Actual traffic unavailable. |
| Recurrent state minimum read/write | 142606336 | 285212672 | Independent state per chat; not weight traffic. |

**1chat MEASURED:** == 1chat/rank0: 246 steps; step wall 41.959 ms (median 41.758); union device busy 41.290; GPU idle 0.670
steps with equal message counts 246 of 246; wire messages a step each way 245.0
HOST DRIVER CALLS of the launch thread, a step: cuStreamSynchronize 4.0 (33.853 ms), cuGraphLaunch 19.0 (1.699 ms), cuLaunchKernel 76.2 (0.142 ms), cuCtxSetCurrent 1865.0 (0.129 ms), cuMemcpyHtoDAsync_v2 20.0 (0.102 ms), cuMemsetD8Async 11.0 (0.070 ms), cuGraphExecUpdate_v2 0.5 (0.032 ms), cuGraphDestroy 0.5 (0.016 ms), cuMemcpyDtoHAsync_v2 2.0 (0.014 ms), cuGraphInstantiate_v2 0.0 (0.011 ms), cuMemAlloc_v2 0.7 (0.006 ms), cuEventCreate 12.5 (0.006 ms), cuEventRecord 20.8 (0.005 ms), cuStreamWaitEvent 20.8 (0.005 ms), cuEventDestroy_v2 12.5 (0.002 ms), cuStreamEndCapture 0.5 (0.002 ms)

**2chat MEASURED:** == 2chat/rank0: 118 steps; step wall 69.982 ms (median 69.481); union device busy 68.662; GPU idle 1.320
steps with equal message counts 118 of 118; wire messages a step each way 340.0
HOST DRIVER CALLS of the launch thread, a step: cuStreamSynchronize 3.2 (55.590 ms), cuGraphLaunch 29.0 (2.543 ms), cuLaunchKernel 156.3 (0.313 ms), cuCtxSetCurrent 3173.8 (0.195 ms), cuMemcpyHtoDAsync_v2 30.0 (0.174 ms), cuGraphInstantiate_v2 0.2 (0.156 ms), cuGraphExecUpdate_v2 0.3 (0.030 ms), cuGraphDestroy 0.5 (0.026 ms), cuMemcpyDtoHAsync_v2 2.0 (0.014 ms), cuMemAlloc_v2 1.5 (0.012 ms), cuEventCreate 12.2 (0.005 ms), cuEventRecord 18.3 (0.004 ms), cuStreamWaitEvent 18.3 (0.004 ms), cuEventDestroy_v2 12.2 (0.003 ms), cuStreamEndCapture 0.5 (0.002 ms), cuStreamBeginCapture_v2 0.5 (0.002 ms)

## Other matched windows

MEASURED server means, first eight steps omitted: two different 69.836 ms; two copies 69.105; three different 82.731; four different 92.452; eight copies 96.838. The matching client rates are 28.498594, 28.794643, 36.250324, 43.134945 and 81.163458 tok/s. Mac load at launch is recorded in `logs/PAIR.log`; client numbers are cross-checks only.

All 1/2/3/4/8 and copy windows, plus S3 and S2 drafting windows, are saved in `runs/pair-main-trace-01/rank0` and `rank1`. `SUMMARY.txt` has the complete server timer record.

## Cut selection

The largest individual kernel increase is NVFP4, 9.459701 to 18.241399 ms (+8.781697). `ROUTED_CHATS=2` runs the routed span once per chat. Two copies also repeat it. The first small test changes two chats to the already existing grouped route; three chats already use that route and serve as a control. No arithmetic or kernel entry changes. Shared projections add more in aggregate (+11.478 ms), but this is spread across several replaced forms and repeated shared-down work. This selection does not claim all NV time is avoidable.

Other supported follow-up: unrounded narrow FP32 projections use `mixed` at two rows (4.088990 ms) instead of the one-row `plain` fold (1.049281 ms). A shape-based dispatch to the existing fold is a candidate, not yet measured. The BF16 part extension alone should not be assumed faster: prior row_deep trials lost.

Limits: no hardware byte counters; no implementation speedup; no all-four-load candidate proof; no golden reset. First-cut speed requires untraced main/candidate/candidate/main on one pair.

## Matched secondary windows

Exclusive cross-rank path, first eight steps omitted; milliseconds a step.

| Group | 2 copies | 3 chats | 4 chats | 8 chats | 8 copies |
|---|---:|---:|---:|---:|---:|
| routed expert weights (NVFP4) | 18.098 | 28.710 | 35.784 | 67.754 | 15.320 |
| shared weights (MXFP8, block FP8, BF16) | 29.859 | 30.305 | 30.188 | 35.892 | 35.754 |
| output head | 1.472 | 1.542 | 1.850 | 2.982 | 2.980 |
| attention math | 7.297 | 7.288 | 9.633 | 11.751 | 11.169 |
| latent attention expansion | 0.000 | 0.000 | 0.000 | 5.393 | 5.362 |
| mHC | 2.780 | 4.027 | 2.288 | 3.283 | 3.218 |
| messages between Sparks | 6.069 | 4.295 | 5.091 | 7.259 | 9.827 |
| expert routing and quantize | 0.766 | 1.543 | 1.780 | 2.633 | 1.128 |
| small math and copies | 1.720 | 2.613 | 3.355 | 4.586 | 5.330 |
| idle (GPU waiting) | 1.197 | 2.541 | 2.637 | 3.345 | 6.968 |
| Total | 69.258 | 82.864 | 92.605 | 144.878 | 97.056 |

## Verify spans

MEASURED full engine `SERVED VERIFY` means after eight passes: structured three rows 94.389 ms; structured four rows 103.055 ms. The table below is only the device interior from the multi-row embedding start to the vocabulary-head end, after four passes. These windows exclude the proposal work and some host setup/tail work; do not substitute them for the full server timer.

| Window/rank | Passes | Interior ms | Device busy ms | Device idle ms |
|---|---:|---:|---:|---:|
| dstruct-rank0-interior4 | 59 | 101.783 | 91.427 | 10.356 |
| dstruct-rank1-interior4 | 59 | 101.790 | 91.559 | 10.232 |
| dstruct2-rank0-interior3 | 79 | 93.384 | 83.064 | 10.319 |
| dstruct2-rank1-interior3 | 79 | 93.386 | 83.247 | 10.139 |
| dprose-rank0-interior4 | 18 | 99.319 | 88.903 | 10.416 |
| dprose-rank1-interior4 | 18 | 99.303 | 89.043 | 10.260 |

Each embedding has four residual streams: grid Y of 12 means three token rows; grid Y of 16 means four. The corrected extraction is `kit/PAIR-verify.py`; use `pair-analysis/verify-interior.json`, not an empty `verify.json` from the initial extraction.
