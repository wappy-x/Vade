# Native agent workload measurement

`vade-harness workload` drives independent, concurrent native conversations through
`agent::run`, `Conversation`, `NativeModel`, the append-only request builder,
capability negotiation, exact admission when advertised, and the real HTTP/SSE
transport. This is a harness product mode, not a second agent loop or an Engine
benchmark implementation. Verification was on macOS ARM64 with first-party
loopback mocks only. No Engine, Spark, GPU, external agent, or live model was used.

## Run

From `harness/`, build once, then supply the coordinator's selected endpoint and
model. The following command is for the coordinator's live run; it was **not executed**
against a live upstream during development. The shell checks require explicit
configuration rather than guessing a server/model. Use a new JSON filename for
each run; existing files are never overwritten.

```sh
cargo build --release --offline
: "${ENGINE_ENDPOINT:?Set the HTTP chat-completions endpoint}"
: "${ENGINE_MODEL:?Set the model advertised by that endpoint}"
./target/release/vade-harness workload \
  --endpoint "$ENGINE_ENDPOINT" --model "$ENGINE_MODEL" \
  --sessions 6 --main-sessions 1 --turns 8 \
  --main-system-bytes 16384 --sub-system-bytes 4096 \
  --main-user-bytes 2048 --sub-user-bytes 512 \
  --main-tool-bytes 8192 --sub-tool-bytes 2048 \
  --output-tokens 128 --seed 41 --max-seconds 900 \
  --json workload-live.json
```

For two mains and six subordinates, change to `--sessions 8 --main-sessions 2`.
`workload --help` lists all flags and bounds. N includes mains. K includes the
first turn, which is retained but excluded from performance/cache summaries.
The default composition is one main plus five subordinates, with eight turns
per session. Role names describe workload sizes, not an actual agent delegation
hierarchy. Sessions start together and then proceed independently without a
per-turn barrier. Cold first turns can overlap another session's measured turns.

Sizes configure **ASCII padding bytes**, not tokens. Each synthetic message also
has a short descriptive envelope. Main prompts grow by larger user/tool payloads
than subordinate prompts. Fixed seeded content is repeatable; a session/turn/
channel-specific linear congruential generator chooses words from an original
small vocabulary. It is not cryptography. Model output is real and retained, so
that output and engine scheduling can vary even with `temperature: 0`. Random
session affinity remains a non-secret scheduling hint and is not a content seed.

The fixed system prompt identifies a unique synthetic session. Before every
request, the history gains a closed, explicitly synthetic assistant tool-call /
tool-result pair plus a user message. The result describes an invented observation;
no tool runs. Previous real assistant messages retain their complete typed content,
reasoning and provider fields. Historical tool records use ordinary Chat Completions
fields; the upstream must accept that history. No executable tool is advertised.
An unexpected new model tool call stops that session under a zero-tool budget.
No workspace, shell, filesystem content, subordinate process or external agent
is accessed. Reconstructing a validated conversation to append the synthetic pair
does not reset the persistent model or its byte-prefix checks.

Replies containing only reasoning, including `finish_reason: "length"`, enter
history with `content: ""` when content is absent/null and there are no tool
calls. The original reasoning and provider fields remain intact. This is done
once during response ingestion for both SSE and JSON fallback, before immutable
serialization, so later requests keep the complete existing byte prefix.

The endpoint must be plain HTTP. Base URLs and `/v1` normalize to chat completions;
custom exact routes use standard chat without contract discovery. HTTPS,
credentials in URLs, queries and fragments fail explicitly. There is no auth
header flag or credential discovery. Use a suitable explicitly configured local
HTTP gateway if the selected deployment needs another transport/auth boundary.
This mode does not configure that gateway.

## Cache denominator: exact counts versus exact LCP

The requested criterion, credited to
[andreasgru/ds41-flash-hybrid, METHODS.md](https://github.com/andreasgru/ds41-flash-hybrid/blob/main/METHODS.md),
is evaluated separately by role from turn 2 onward:

```text
U = token length of LCP(current prompt, previous prompt + previous generated output)
C = usage.prompt_tokens_details.cached_tokens
R = sum(min(C, U)) / sum(U), target >= 0.98 for each role
```

**The current `vade.engine.v1` admission contract returns exact rendered prompt
lengths, but not rendered token IDs or an exact LCP. Exact lengths alone cannot
establish that two token sequences match.** Before each warm turn, the driver
uses a separate admission-only request to count the retained previous history,
including the full validated assistant content, reasoning and provider fields.
The request has the frozen generation options and affinity, no KV retention
request, and no new synthetic observation/user messages. It runs no inference.
The next actual generation independently negotiates and admits its complete body
byte for byte, preserving the existing append-only client state.

```text
U_est = min(current exact admitted prompt count,
            exact admitted rendering of previous history + full assistant reply)
```

This avoids adding generated output-token counts to an input-token count. Those
counts can differ when the renderer handles reasoning, assistant terminators,
retokenization, or hidden framing. No reported output count is needed for this
path. JSON labels it `u_basis: "exact_rendered_history_counts"` and retains the
raw count as `rendered_previous_tokens`. The renderer can append a generation
header to the counted transcript, or change an earlier token boundary on the next
request. Therefore **U and R remain estimated LCP metrics**, even when both
rendered lengths are exact. The human table prefixes R with `~`;
`exact_target_verified` stays false. An engine-provided LCP/IDs or independently
verified render-continuity contract is required to prove the exact 0.98 criterion.
No invented field or count-only generation is sent.

Each sample retains `prefix_count_requests`, `prefix_count_elapsed_ms` and
`prefix_count_error`. The top-level `prefix_count_requests` sums these extra
measurement requests separately from `all_turn_requests`, which still describes
ordinary generation capability/admission/HTTP activity. Count refusal, HTTP
failure or malformed telemetry leaves U unknown for that turn; it does not
silently fall back to output/byte estimates or retry the count. Generations can
complete while cache target coverage remains insufficient. The extra admission
adds measurement overhead; its time is outside native TTFT, while intervening
counts are included in the overlapping warm wall window used for throughput.

Without exact admission, server-reported prompt usage is preferred and
`U_est = min(current prompt count, previous prompt count + previous output count)`.
If usage is absent, input length uses `ceil(request wire bytes / 4) + 256`; absent
output usage uses `ceil(serialized assistant bytes / 4)`. This rough metric
estimate is separate from the existing context-budget estimate retained in each
sample's `input` field. Neither is a tokenizer or a context-fit guarantee.
`u_basis` identifies `reported_counts` or `wire_bytes_estimate` fallback. A cache
identity/domain change leaves U unknown for the boundary turn. An exact admitted
count conflicting with reported prompt usage is a measurement error and stops
the session; output usage beyond its requested reservation does likewise.

The text report includes every attempted turn, including the cold first turn,
with P (admitted/reported prompt count), raw C, U, `100 * min(C,U) / U` percent,
U basis and outcome/error. Missing or zero denominators print an unknown ratio.
This makes a zero-cache turn, changed identity, or failed count visible alongside
the role summaries; turn 1 remains excluded from their measurements.

Missing C differs from zero. Invalid/conflicting telemetry remains unknown.
C greater than U is capped at U. Zero U contributes zero to both sums; an all-zero
denominator cannot pass. `cache_ratio` is null unless every planned measured
turn has a usable C/U pair and the denominator is positive. `paired_cache_ratio`
is a clearly separate diagnostic over the covered subset. Coverage, raw C, U,
basis, numerator and denominator are retained. Incomplete/failed sessions cannot
silently disappear from the expected denominator coverage. No aggregate cache
target is calculated across roles. The 0.98 comparison uses integer arithmetic.

## Timing, output and failures

TTFT is measured with a monotonic clock at the native model request boundary,
before negotiation/admission, through the first generated SSE delta. Nonempty
content, reasoning and function name/argument deltas qualify; role-only,
finish-only and usage-only events do not. It includes local preparation,
capability discovery, admission, queueing, network time, and parsing of that first
event. The metric observes a generated delta, not a tokenizer count or physical
terminal render. JSON fallback has no TTFT. Failed/truncated responses have no
accepted TTFT or provisional cache usage. Percentiles are nearest rank over
known successful measured samples; their coverage is explicit.

`output_tok_s` is the sum of reported completion tokens divided by the measured
wall window (earliest turn-2-or-later start to latest measured end). Per-session
rates include time between its measured requests. Role/overall rates use the
union's wall span, so overlapping concurrency is counted once, never by adding
session rates. This is end-to-end output throughput, including admission and
prefill, **not decode-only speed**. It is null if any planned output count is
missing; `measured_output_tok_s` reports only known output over that same span.
The reservation flag is a maximum; it does not force a model to emit that many
tokens. `stop` and validated `length` finishes both yield workload samples and
preserve history; length is expected in a token-limited workload. Other finishes,
unexpected tools and errors stop only the affected session, with no retry.

The human table shows main/subordinate role totals, overall throughput, and each
session. Admission attempts/refusals, all HTTP 429/503 responses (including
capability/admission failures), and errors are counted from turn 2 onward. An
all-turn footer also exposes cold-turn failures. JSON includes each attempt's
stage/status, counts, timing, input admission/estimate, output/cache usage,
finish/error and history size. Counts describe requests actually attempted, not
requests scheduled but never sent; context-fit refusal differs from HTTP overload.

Exit 0 means every requested session/turn completed with an accepted finish; it
does not assert a cache target or complete metric telemetry. Exit 1 denotes a
stopped/failed/timed-out workload or report I/O failure. Exit 2 denotes invalid
flags or an unusable/existing output destination. A partial run still writes its
metrics; setup/worker failure or external process termination can leave an empty
or incomplete file. Cache target statuses are `estimate_meets_target`,
`estimate_below_target`, or `insufficient_data`.

The report destination is checked before network I/O, created with `create_new`
and private Unix mode 0600, and never clobbered. It contains counts and generic
session IDs, not endpoint/model names, credentials, affinity keys, fingerprints,
prompts, model responses, tool contents, private paths or raw upstream errors.

## Bounds and verification

At most 32 sessions, 1–2 mains, 1,000 turns/session and 1,024 session-turns total.
Each padding field is capped at 262,144 bytes. Output reservation is 1–8,192 tokens.
Native bounds are 4 MiB of history, 4 MiB per message, 64 MiB of response-stream
payloads and 262,144 events per response. Histories are never pruned or compacted
to improve the cache score.
The workload deadline is 600 seconds by default (maximum 86,400); it signals
cancellation and joins all owned workers. Native generation response reads use
the full 3,600-second HTTP lifetime, including silent queue/prefill waits before
headers or after role-only events. The five-second progress notice counts
sessions **queued / awaiting upstream** until their first generated delta (or
complete JSON response); this is a local waiting observation, not server queue
telemetry. No request is replayed. Other transport bounds remain: 5-second
connect, 600-second idle for discovery/admission/request writes, and 250 ms
cancellation polling. Std DNS and an in-progress connect cannot be interrupted. SIGINT/process
termination does not currently flush a partial report; prefer `--max-seconds`
when a timed partial report is required.

These limits include the output extension from `824ab96f`, which changed native
message, stream and event bounds and HTTP timeouts;
see the [dated branch comparison](engine-contract.md#output-extension-provenance-824ab96f).
The workload keeps its 1–8,192-token reservation and disables automatic
continuation to preserve the single-response timing boundary.

```sh
cargo test --offline --all-targets
cargo test --offline --test contract_mock workload_ -- --nocapture
RUSTFLAGS='-D warnings' cargo check --offline --all-targets
cargo fmt --all -- --check
cargo build --release --offline
```

The existing macOS mock example can also launch the release workload CLI and
leave a report for inspection, with no separately configured server:

```sh
cargo run --release --offline --example perf_mock -- \
  target/release/vade-harness 1 workload target/workload-mock.json
```

This selects six sessions, three turns, a 17-token reservation, and invented
fixture counts. Both roles deliberately receive full reuse of their
admission-counted, estimated prefix. The process binds its own ephemeral loopback port and cannot select an
Engine URL. Choose a new output filename on each invocation.

The existing first-party contract mock fixtures now check six simultaneous
native sessions, admitted/generation byte equality, persistent independent keys,
unchanged historical bytes, real assistant retention, synthetic tool pairing,
per-role weighted sums and clamping, separate full-history count requests with
reasoning/opaque fields, counts differing from generated-output arithmetic,
failed/refused/malformed count telemetry, per-turn C/U text, overlapping aggregate
throughput, missing usage, standard fallback, JSON-only timing, 429/503 at both admission and
generation, context refusal without generation, no replay, reasoning-first TTFT,
zero cache/denominator, length finish, malformed/truncated streams, unexpected
tools, identity changes, inconsistent usage, cancellation/joining, eight-turn
reasoning-only SSE/JSON histories, 31-second silent generation waits, and actual CLI
JSON permissions/overwrite refusal. No mock timing is an inference-performance
claim. The deliberate mixed-cache fixture yields main R=1.0000 and subordinate
R=760/1530≈0.4967, demonstrating that one role cannot hide another's poor reuse.

The linked method was read for the criterion only. All driver, fixture, prompt
and filler code is original Rust using the standard library. No competitor code,
tokenizer, cryptography implementation or dependency was added.
