# Standalone gateway: historical measurements

Measured 2026-10-20 on one Spark, engine main `26b1692`, FP4-Local,
thinking off. The model remains `qwen3.8-flash-next-fp4-local` and Codex uses
Responses with authentication disabled, following the workshop provider settings.
No engine change or third-party Rust dependency is part of this gateway change.

All 27 recorded greedy request pairs have identical answer bytes. Default sampled
Codex assistant text matches in 8 of 21 pairs; that row preserves sampling.
Some latency and write-rate medians are worse through the gateway. These samples
do **not** establish uniform performance non-regression. Opt-in greedy drafting
has a separate measured throughput benefit and can change answers.

## Measurement and scope

Actual Codex uses one shared isolated home in both arms, keeping injected skill
paths and instructions identical. An offline installed-client rehearsal verified
request-content identity for all seven prompts, excluding message ids only.
Earlier separate-home client timings were excluded because their injected paths
differed. Three repetitions alternate direct/gateway order. Each real arm runs
three prose tasks, three code tasks and one shell-tool user turn. Sampled tool
turns can execute different numbers of model/tool calls; all finish with the
requested file contents. Tool-call latency reflects the frontend's first visible
notification, and write rates use response spans with observable generated events.

Recorded replay uses exact captured Codex request bodies, with temperature zero
in both arms and no seed. All three requests from the captured tool conversation
are retained. That captured session predates the shared-home timing correction;
its frozen request bytes are identical in both replay arms. Complete Responses
output item arrays are decoded, then serialized as UTF-8 for comparison. The only
normalized fields are `response.output[*].id` and
`response.output[function_call|custom_tool_call].call_id`. Text, reasoning,
arguments, tool names, annotations, item order and status are compared unchanged.
Root response ids, created timestamps, usage and vade timings are outside this
answer projection. This is not a claim that timing-bearing SSE envelopes from
separate engine requests are byte-identical. Opaque relay tests separately check
unchanged full SSE/binary payload bytes, headers and error status.

Client comparisons and greedy opt-in controls use context 131072 throughout.
Recorded replay has a context column: each case has three pairs at one context.
The case crossing the window boundary was repeated on 131072; its old samples
are retained as excluded receipts. Greedy opt-in has fresh sampled direct controls
on the same 131072 server. Main advertises prefix caching disabled: every main
cache observation is honestly zero.

Cells below are medians [minimum, maximum] of three runs. D/G means direct/gateway;
positive latency delta is slower through the gateway, positive write delta faster.

## Actual Codex, default sampling

| Case | Context | First token D ms | First token G ms | Delta ms | Write D tok/s | Write G tok/s | Delta tok/s | Answer pairs equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1 | 131072 | 7344.478 [7334.047, 7370.332] | 7341.691 [7336.997, 7380.519] | -2.787 | 26.170 [26.088, 26.237] | 26.175 [26.048, 26.238] | +0.005 | 3/3 |
| code-2 | 131072 | 7365.640 [7344.781, 7365.842] | 7346.323 [7345.215, 7363.503] | -19.317 | 25.562 [25.349, 25.638] | 25.367 [25.356, 25.558] | -0.195 | 2/3 |
| code-3 | 131072 | 7359.543 [7331.779, 7364.672] | 7331.097 [7311.537, 7368.165] | -28.446 | 25.891 [25.888, 26.952] | 25.738 [25.654, 25.800] | -0.154 | 3/3 |
| prose-1 | 131072 | 7328.605 [7175.635, 7375.399] | 7344.717 [7266.360, 7390.062] | +16.111 | 25.238 [24.854, 25.292] | 25.338 [25.186, 25.445] | +0.101 | 0/3 |
| prose-2 | 131072 | 7343.569 [7275.813, 7379.108] | 7330.369 [7308.559, 7374.324] | -13.201 | 24.590 [24.516, 24.797] | 24.775 [24.686, 24.825] | +0.185 | 0/3 |
| prose-3 | 131072 | 7348.836 [7309.506, 7369.359] | 7320.024 [7312.903, 7375.353] | -28.812 | 25.004 [24.773, 25.079] | 25.053 [24.682, 25.125] | +0.049 | 0/3 |
| tool-1 | 131072 | 7288.871 [7265.749, 9316.773] | 7243.540 [7243.400, 7273.165] | -45.331 | 25.402 [24.821, 34.668] | 25.105 [24.844, 25.259] | -0.296 | 0/3 |

## Recorded requests, greedy in both arms

| Case | Context | First token D ms | First token G ms | Delta ms | Write D tok/s | Write G tok/s | Delta tok/s | Answer pairs equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1 | 32768 | 6979.220 [6972.046, 6985.553] | 6975.015 [6967.494, 6975.216] | -4.205 | 47.169 [47.018, 47.642] | 47.140 [46.631, 47.300] | -0.029 | 3/3 |
| code-2 | 131072 | 7247.312 [7242.808, 7254.540] | 7246.800 [7245.033, 7253.058] | -0.512 | 45.798 [45.701, 45.984] | 45.961 [45.921, 46.060] | +0.163 | 3/3 |
| code-3 | 131072 | 7237.175 [7221.886, 7245.758] | 7256.745 [7251.011, 7260.031] | +19.570 | 45.888 [45.743, 46.107] | 46.021 [45.708, 46.574] | +0.134 | 3/3 |
| prose-1 | 32768 | 6971.007 [6947.251, 6971.628] | 6973.426 [6961.749, 6978.581] | +2.419 | 34.213 [34.206, 34.231] | 34.206 [34.183, 34.261] | -0.008 | 3/3 |
| prose-2 | 32768 | 6970.923 [6966.462, 6972.911] | 6978.965 [6955.257, 6982.457] | +8.043 | 29.536 [29.374, 29.545] | 29.430 [29.409, 29.554] | -0.106 | 3/3 |
| prose-3 | 32768 | 6960.291 [6954.267, 6974.395] | 6976.952 [6965.022, 6999.716] | +16.661 | 31.385 [31.121, 31.465] | 31.443 [31.372, 31.474] | +0.058 | 3/3 |
| tool-1 | 131072 | 7190.709 [7185.366, 7200.652] | 7169.507 [7162.595, 7178.922] | -21.203 | 51.705 [51.504, 51.876] | 51.968 [51.260, 52.128] | +0.263 | 3/3 |
| tool-continuation | 131072 | 7259.895 [7252.731, 7263.196] | 7273.126 [7257.343, 7277.085] | +13.231 | 55.713 [55.688, 55.757] | 55.703 [55.497, 55.864] | -0.010 | 3/3 |
| tool-continuation-2 | 131072 | 7488.273 [7475.240, 7503.309] | 7496.305 [7494.885, 7504.272] | +8.031 | 36.261 [36.225, 36.692] | 36.490 [36.150, 38.114] | +0.229 | 3/3 |

## Opt-in greedy gateway versus sampled direct

| Case | Context | First token D ms | First token G ms | Delta ms | Write D tok/s | Write G tok/s | Delta tok/s | Answer pairs equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1 | 131072 | 7262.059 [7244.077, 7265.339] | 7341.887 [7321.187, 9576.656] | +79.827 | 26.165 [25.999, 26.908] | 48.130 [47.901, 33710.626] | +21.965 | 3/3 |
| code-2 | 131072 | 7272.310 [7252.028, 7276.089] | 7317.997 [7296.282, 7336.392] | +45.687 | 25.498 [25.415, 25.614] | 44.467 [44.176, 44.899] | +18.969 | 1/3 |
| code-3 | 131072 | 7271.312 [7256.651, 9535.956] | 7313.102 [7219.638, 7335.517] | +41.790 | 25.979 [25.971, 27207.811] | 45.755 [45.332, 46.232] | +19.776 | 3/3 |
| prose-1 | 131072 | 7269.124 [7266.703, 7313.236] | 7318.991 [7306.199, 7347.909] | +49.867 | 25.130 [25.053, 25.156] | 31.366 [31.273, 31.612] | +6.236 | 0/3 |
| prose-2 | 131072 | 7311.411 [7253.787, 7338.853] | 7324.813 [7324.247, 7327.424] | +13.403 | 24.763 [24.631, 24.821] | 33.113 [32.929, 33.219] | +8.350 | 0/3 |
| prose-3 | 131072 | 7262.789 [7246.757, 7287.032] | 7325.257 [7313.614, 7332.859] | +62.468 | 25.244 [24.877, 25.334] | 31.822 [31.723, 32.357] | +6.578 | 0/3 |
| tool-1 | 131072 | 9269.476 [7154.752, 9290.883] | 7226.476 [7213.016, 7251.607] | -2043.000 | 35.922 [25.595, 35.981] | 45.148 [44.752, 45.215] | +9.226 | 0/3 |

## Relay cost, cache and qualification

Across 63 complete observed gateway replies, weighted downstream framing/write
cost is **0.030544 ms per output token**. Median first-token observation minus
engine admission-to-first-logits is **79.770 ms**. The latter includes request,
return transport and boundary differences; it is not an isolated gateway cost.

On the separately authorized cache-fix build `20a41232`, context 32768, gateway
stable-prefix and affinity enabled, six greedy replay pairs remain identical.
The first incremental second turn restores **98.125%**, first token **444.844 ms**.
Repeated identical warm requests restore **100%** in both arms: three-run warm
medians are **97.223 / 88.725 ms** first token and **54.595 / 54.950 tok/s** D/G.
First-turn reuse is zero. Later exact replay can restore the entire input; these
figures do not attribute the engine cache implementation or its warm state to the
gateway. Cached-token counts equal `vade.prefill_cache_restored_tokens` throughout.

Earlier flag qualification measured two output/reservation clamps, one original
oversized input relayed with the engine's exact 400 body, seven input fields
unchanged, two stable keys out of two turns, and 100 affinity keys preserving
placement under list reordering (moves on addition only to the added server).
That same-window 128-token code control measured greedy off/on **25.842 / 53.573
tok/s**. No input is truncated, reordered or rewritten.

Historical qualification: **514 harness tests pass**, including 13 standalone gateway
tests; Clippy retains 150 baseline diagnostics with zero new gateway diagnostics.
Doctor passes through the candidate on the measured build, with the expected
cache-disabled warning. Installed OpenCode and Pi pass direct/gateway smokes
(4/4, **10-20, no tools**, one-line prompt, 128-token cap); Hermes is blocked by
missing installed dependencies, with no installation or repair attempted.
Claude Code remains parked.

The candidate fixes HTTP/1.0 close-delimited framing, arbitrary-path forwarding
with an API base suffix, busy-refusal logging and validated final-event recording
when Codex closes before hop framing. Explicit logit bias also prevents optional
greedy defaults. No native-harness feature was added.

Raw request recordings remain private. This document preserves the published
summary and its limitations. The [historical gateway card](harness-card.md)
records the longer-prefix measurements; its private launcher is not shipped.
