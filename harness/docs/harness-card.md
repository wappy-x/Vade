# Codex gateway card

This is a historical measurement record. The private launcher used to collect
it is not distributed with this repository. The method and results are retained
below; this is not a runnable installation recipe.

Each size prints three JSON rows, with direct and gateway measurements nested
under those names:

- `a_real_codex_sampled`: the actual installed Codex app-server in both arms,
  preserving the engine's default sampling and using an undecorated gateway.
- `b_recorded_request_greedy_replay`: one captured Codex warm-turn request per
  repetition, replayed directly and through the gateway with temperature zero
  in both arms and no seed. Different answer bytes fail the run.
- `c_gateway_greedy_opt_in_vs_direct_sampled`: Codex through the gateway with
  stable-prefix, affinity and greedy-when-unset flags. Its direct baseline is
  the three sampled runs already measured in row (a). Answers may differ.

Codex uses Responses, model `qwen3.8-flash-next-fp4-local`, authentication disabled,
thinking off and the provider settings from the workshop launcher. Both arms share
one isolated Codex home, keeping injected skill paths identical. The tool captures the installed client's request shape
using an offline loopback fixture. Stable background vocabulary comes from the
retained conversation fixture. Exact admission calibrates the cold request
within 32 tokens of each target. The actual Codex tool schema and instructions
remain present; background text never requests tool execution. A follow-up turn
retains the full conversation and session key.

`first_token_ms`, `cache_reuse_pct` and `write_tok_s` are medians of three warm
(second) turns. Rows also report cold first-token latency, reuse per turn, sample
ranges, actual token counts and answer equality. First token is the first model
delta observed by the Codex frontend; total process startup is excluded. Write
speed uses output tokens minus one divided by the first-to-last generated-event
interval. Replay observes the corresponding Responses SSE events directly.
Missing observations remain null. A reported cache hit requires actual restored
input tokens; a stable scheduling key alone is insufficient.

Equality in row (b) compares complete `response.output` item arrays as serialized
UTF-8, including text, reasoning, tool names/arguments, annotations, order and
status. Only `response.output[*].id` and function/custom-tool `call_id` are
normalized. Root response ids, creation times, usage and timing metadata are
outside this answer projection; reply bodies and measured metadata are retained
separately. Row (a) compares completed assistant text and phase bytes honestly.

Receipts contain client events, exact recorded requests/replies, per-request
stats, admission counts and completed samples. Keep their output directory
outside the tracked tree: real recorded sessions can contain private content.
## Historical measurements

Measured 2026-10-20 on the authorized one-Spark main-`26b1692` windows,
context 131072, thinking off. Three runs per arm; entries are warm-turn medians.
The size is the exact admitted cold-prefix target; follow-up prompts also contain
the prior answer and a short question. Actual warm prompt counts and min/max
ranges remain in the JSON receipts. Main reports prefix caching disabled: all
reuse observations below are measured zero. The separate cache-fix proof is in
[the gateway qualification](gateway-measurements.md).

A preserves sampled Codex defaults, B replays frozen requests with temperature
zero in both arms (no seed), and C explicitly opts the gateway into greedy
sampling. C uses the sampled direct control measured in A. Different sampled
answers are reported honestly. Some timing medians are worse through the gateway;
these measurements do not establish uniform performance non-regression.

| Mode | Prefix tokens | First token direct ms | Gateway ms | Reuse direct/gateway % | Write direct tok/s | Gateway tok/s | All answer pairs equal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 10000 | 7327.895 | 7300.655 | 0.0/0.0 | 25.011 | 24.960 | no |
| B | 10000 | 7262.446 | 7253.517 | 0.0/0.0 | 28.722 | 28.221 | yes |
| C | 10000 | 7327.895 | 7285.837 | 0.0/0.0 | 25.011 | 30.561 | no |
| A | 50000 | 35107.389 | 35121.954 | 0.0/0.0 | 24.580 | 24.631 | no |
| B | 50000 | 34946.709 | 34666.026 | 0.0/0.0 | 28.754 | 29.322 | yes |
| C | 50000 | 35107.389 | 35057.728 | 0.0/0.0 | 24.580 | 32.363 | no |
| A | 100000 | 71569.526 | 71510.027 | 0.0/0.0 | 24.344 | 24.431 | no |
| B | 100000 | 71372.809 | 71334.065 | 0.0/0.0 | 29.221 | 29.509 | yes |
| C | 100000 | 71569.526 | 71973.890 | 0.0/0.0 | 24.344 | 29.212 | no |

Raw samples were retained outside the repository. The table above is the
published historical summary; it does not qualify the current build. The
private collection scripts are not shipped.
