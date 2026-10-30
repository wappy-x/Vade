# Gateway measurements from 2026-10-20

This page keeps an older record. Installed Codex talked to the engine on one
Spark twice for every task: once directly, and once through `vade-gateway`. The
point was to see what the gateway changes and what it costs. It describes the
build named below, not the current release.

| Setting | Value |
| --- | --- |
| Date | 2026-10-20 |
| Machine | One Spark |
| Engine | main `26b1692`, FP4-Local, thinking off |
| Model ID | `qwen3.8-flash-next-fp4-local` |
| Client | Codex over the Responses API, authentication off, with the provider settings from our private test launcher (not shipped) |

The gateway work changed no engine code and added no third-party Rust crate.

## What the runs show

- **Greedy requests give the same answer both ways.** All 27 recorded greedy
  request pairs have identical answer bytes, direct and through the gateway.
- **Sampled requests differ, as sampling does.** With Codex's default sampling,
  the answer text matched in 8 of 21 pairs. Sampling stays on in that table.
- **The gateway is not always free.** Some first-token and writing medians are
  a little worse through the gateway, so these samples do not show that it
  never slows things down.
- **The greedy default was a separate, opt-in choice on this build.** With
  `--greedy-when-unset`, the gateway made a request greedy when the client
  named no sampling fields, and greedy requests can use drafting (the model
  guesses a few tokens ahead and the engine checks them). The third table shows
  the effect on speed. Its direct arm was sampled, so answers that differ from
  it are expected; that table does not compare drafting ON and OFF. Current
  gateways insert the greedy default whenever no sampling field is present, and
  keep the flag only for compatibility (see
  [the gateway guide](invisible-gateway.md)).

## How we measured

- **One home for both arms.** The real Codex runs used one shared, isolated
  home folder for the direct and the gateway arm, so the injected skill paths
  and instructions were identical. An offline rehearsal with the installed
  client confirmed that both arms sent the same request content for all seven
  prompts, apart from message ids. Earlier timings from separate homes were
  dropped, because their injected paths differed.
- **Seven tasks, three rounds.** Each arm ran three prose tasks, three code
  tasks and one turn that uses a shell tool. The three rounds alternate which
  arm goes first. Sampled tool turns can make different numbers of model and
  tool calls; all of them finished with the requested file contents.
- **What the times mean.** First token is when the Codex frontend shows its
  first notification. Writing rates use only response spans with observable
  generated events.
- **Recorded replay.** The second table resends the exact captured Codex
  request bodies, with temperature zero in both arms and no seed. All three
  requests from the captured tool conversation are kept. That capture is older
  than the shared-home fix, but its frozen request bytes are the same in both
  replay arms.

### What counts as an equal answer

For the answer comparison, complete Responses output item arrays are decoded,
then serialized as UTF-8. Only two fields are normalized:
`response.output[*].id` and
`response.output[function_call|custom_tool_call].call_id`. Text, reasoning,
arguments, tool names, annotations, item order and status are compared
unchanged. Root response ids, created timestamps, usage and vade timings are
not compared.

This does not claim that the SSE envelopes of two separate engine requests are
byte-identical, since they carry timings. Separate relay tests check that full
SSE and binary payload bytes, headers and error status pass through unchanged.

### Context sizes

The client comparisons and the greedy opt-in runs use context 131072
throughout. The recorded-replay table has a context column, and each case has
its three pairs at one context. The case that crossed the window boundary was
repeated at 131072; its old samples are kept as records but left out of the
results. The greedy opt-in table uses fresh sampled direct runs, on the same
131072 server, as its controls. The `26b1692` build advertises prefix caching
as off, so every cache reading on it is zero, as it should be.

### Reading the tables

Each cell is the median of three runs, then [minimum, maximum]. A positive
latency change means slower through the gateway; a positive writing change
means faster.

Two writing maxima in the third table are shown as —: code-1 through the
gateway and code-3 direct. The published summary gave values above 27,000
tok/s there, which cannot be real write rates. No committed measurement file
holds these runs (the raw recordings are private), so we removed the two
numbers rather than repeat them.

## Real Codex, default sampling

| Case | Context | First token, direct (ms) | First token, gateway (ms) | Change (ms) | Writing, direct (tok/s) | Writing, gateway (tok/s) | Change (tok/s) | Equal answer pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1 | 131072 | 7344.478 [7334.047, 7370.332] | 7341.691 [7336.997, 7380.519] | -2.787 | 26.170 [26.088, 26.237] | 26.175 [26.048, 26.238] | +0.005 | 3/3 |
| code-2 | 131072 | 7365.640 [7344.781, 7365.842] | 7346.323 [7345.215, 7363.503] | -19.317 | 25.562 [25.349, 25.638] | 25.367 [25.356, 25.558] | -0.195 | 2/3 |
| code-3 | 131072 | 7359.543 [7331.779, 7364.672] | 7331.097 [7311.537, 7368.165] | -28.446 | 25.891 [25.888, 26.952] | 25.738 [25.654, 25.800] | -0.154 | 3/3 |
| prose-1 | 131072 | 7328.605 [7175.635, 7375.399] | 7344.717 [7266.360, 7390.062] | +16.111 | 25.238 [24.854, 25.292] | 25.338 [25.186, 25.445] | +0.101 | 0/3 |
| prose-2 | 131072 | 7343.569 [7275.813, 7379.108] | 7330.369 [7308.559, 7374.324] | -13.201 | 24.590 [24.516, 24.797] | 24.775 [24.686, 24.825] | +0.185 | 0/3 |
| prose-3 | 131072 | 7348.836 [7309.506, 7369.359] | 7320.024 [7312.903, 7375.353] | -28.812 | 25.004 [24.773, 25.079] | 25.053 [24.682, 25.125] | +0.049 | 0/3 |
| tool-1 | 131072 | 7288.871 [7265.749, 9316.773] | 7243.540 [7243.400, 7273.165] | -45.331 | 25.402 [24.821, 34.668] | 25.105 [24.844, 25.259] | -0.296 | 0/3 |

## Recorded requests, greedy in both arms

| Case | Context | First token, direct (ms) | First token, gateway (ms) | Change (ms) | Writing, direct (tok/s) | Writing, gateway (tok/s) | Change (tok/s) | Equal answer pairs |
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

## Gateway with the greedy default versus sampled direct

| Case | Context | First token, direct (ms) | First token, gateway (ms) | Change (ms) | Writing, direct (tok/s) | Writing, gateway (tok/s) | Change (tok/s) | Equal answer pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code-1 | 131072 | 7262.059 [7244.077, 7265.339] | 7341.887 [7321.187, 9576.656] | +79.827 | 26.165 [25.999, 26.908] | 48.130 [47.901, —] | +21.965 | 3/3 |
| code-2 | 131072 | 7272.310 [7252.028, 7276.089] | 7317.997 [7296.282, 7336.392] | +45.687 | 25.498 [25.415, 25.614] | 44.467 [44.176, 44.899] | +18.969 | 1/3 |
| code-3 | 131072 | 7271.312 [7256.651, 9535.956] | 7313.102 [7219.638, 7335.517] | +41.790 | 25.979 [25.971, —] | 45.755 [45.332, 46.232] | +19.776 | 3/3 |
| prose-1 | 131072 | 7269.124 [7266.703, 7313.236] | 7318.991 [7306.199, 7347.909] | +49.867 | 25.130 [25.053, 25.156] | 31.366 [31.273, 31.612] | +6.236 | 0/3 |
| prose-2 | 131072 | 7311.411 [7253.787, 7338.853] | 7324.813 [7324.247, 7327.424] | +13.403 | 24.763 [24.631, 24.821] | 33.113 [32.929, 33.219] | +8.350 | 0/3 |
| prose-3 | 131072 | 7262.789 [7246.757, 7287.032] | 7325.257 [7313.614, 7332.859] | +62.468 | 25.244 [24.877, 25.334] | 31.822 [31.723, 32.357] | +6.578 | 0/3 |
| tool-1 | 131072 | 9269.476 [7154.752, 9290.883] | 7226.476 [7213.016, 7251.607] | -2043.000 | 35.922 [25.595, 35.981] | 45.148 [44.752, 45.215] | +9.226 | 0/3 |

## What the relay itself costs

Across 63 complete gateway replies, the gateway's own framing and write work
came to **0.030544 ms per output token** (weighted by tokens). The median gap
between the first token the client saw and the engine's own
admission-to-first-logits time was **79.770 ms**. That gap also includes the
request, the trip back, and the two clocks starting and stopping at different
points, so it is not the gateway's cost on its own.

## Cache results on a later build

Build `20a41232` added a cache fix. On it, at context 32768 with the gateway's
`--stable-prefix` and `--affinity` flags on, six greedy replay pairs stayed
identical.

- The first follow-up turn restored **98.125%** of its prompt from the cache,
  with a first token at **444.844 ms**.
- Repeating an identical warm request restored **100%** in both arms. Over
  three runs, the warm medians were **97.223 / 88.725 ms** to the first token
  and **54.595 / 54.950 tok/s**, direct / gateway.

A conversation's first turn restores nothing. A later exact replay can restore
the whole input. The cache and its warm state belong to the engine, so these
figures do not credit them to the gateway. Cached-token counts equal
`vade.prefill_cache_restored_tokens` throughout.

## Claude Code prefix reuse, 2026-10-30

The engine reuses a cached prefix only for a request that sends `cache: true`
or a `prompt_cache_key`. Claude Code sends neither. Before this change,
`--stable-prefix` added those fields only to Chat Completions and Responses
requests, so each Claude Code turn prefilled its whole prompt again.

Setup:

- Engine: main `a5357b3`, prebuilt, on one DGX Spark (NVIDIA GB10), with the
  `nvidia/Qwen3.8-Flash-Next-FP4-Local` checkpoint (revision
  `local-sha256-d99ccbed3a9db8a2f099eca98e17c0bc3a2ebe84df95852f3b07c87c20a32716`).
  The context was 32768 tokens, the output limit 8192, and the prefix cache
  was at its default (on).
- Gateway: `vade-gateway --upstream ENGINE --stable-prefix --stats FILE` on a
  loopback port. "Before" is the build without this change; "after" has the
  code of commit `8922b8a`.
- Client: Claude Code 2.1.314, set up with `vade-harness setup claude
  --target gateway --dedicated-home`. Two runs of `launch.sh -p PROMPT
  --output-format json`: "Reply exactly OK." and then "Read marker.txt using a
  tool and report its exact contents." The second run makes two model calls.

The `--stats` rows for the three Messages requests:

| Request | Before: prompt / cached / first token | After: prompt / cached / first token |
| --- | --- | --- |
| Run 1 | 16044 / 0 / 7716 ms | 16044 / 0 / 7927 ms |
| Run 2, first call | 16051 / 0 / 7785 ms | 16051 / 0 / 8014 ms |
| Run 2, after the tool result | 16178 / 0 / 7832 ms | 16180 / 16000 / 472 ms |

Each run is a new Claude Code session, so its first call reuses nothing. The
call after the tool result shares its prefix with the call before it, and only
the "after" build lets the engine reuse it. This is one run of each build, not
a repeated benchmark.

## Other gateway checks

Earlier checks of the gateway's optional flags found:

- Limit clamping (`--clamp-limits`): two output or reservation limits were cut
  down to what the engine allows.
- An oversized input was passed to the engine unchanged, and the engine's exact
  400 error came back to the client.
- All seven input fields we checked came through unchanged.
- The stable-key option kept the same cache key on both turns tested.
- 100 affinity keys kept their placement when the server list was reordered;
  when a server was added, keys moved only to that new server.
- A 128-token code control run in the same session measured
  **25.842 tok/s** with the greedy default off and **53.573 tok/s** with it on.

The gateway never truncates, reorders or rewrites an input.

## Test status at the time

- **514 harness tests passed**, including 13 standalone gateway tests.
- Clippy still reported its 150 baseline diagnostics, and none were new from
  the gateway.
- The doctor check passed through the gateway on the measured build, with the
  expected warning that the cache is off.
- The installed OpenCode and Pi clients passed quick direct and gateway smoke
  tests on 10-20 (4/4, no tools, a one-line prompt and a 128-token cap).
- Hermes was blocked by missing installed dependencies; we did not install or
  repair anything.
- Claude Code testing remains on hold.

This gateway build also fixed:

- HTTP/1.0 close-delimited framing;
- arbitrary-path forwarding with an API base suffix;
- busy-refusal logging;
- recording the validated final event when Codex closes the connection before
  the HTTP response framing is finished.

A request that sets its own `logit_bias` (or any other sampling field) gets no
greedy defaults. No native-harness feature was added.

Raw request recordings stay private. This page keeps the published summary and
its limits. The [historical gateway card](harness-card.md), a one-page summary
of results, records the longer-prompt measurements; its private launcher is not
shipped.
