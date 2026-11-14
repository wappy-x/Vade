# Native agent workload measurement

`vade-harness workload` load-tests a chat-completions endpoint. It runs several
agent conversations at once and measures two things:

- how well the server keeps up
- how much of each prompt the server serves from cache

It is neither a second agent loop nor an engine benchmark. It drives the
harness's real agent code:

- the same conversation and append-only request builder (`agent::run`,
  `Conversation`, `NativeModel`)
- the capability check
- exact admission, when the engine offers it
- the real HTTP/SSE transport

Exact admission is the [engine contract](engine-contract.md#exact-admission)
step where the harness sends the complete request to the engine first and gets
back its exact prompt length, without running the model.

**Where it has run so far:** only on macOS ARM64 against our own
loopback mocks, with no engine, Spark, GPU, external agent or real model.

## Run it against your endpoint

Build once from `harness/`. Then give it the endpoint and model you want to
test.

The command below has **not yet been run** against a real server. It refuses to
start until you set both variables, so it never guesses a server or model. Use
a new JSON filename for each run, because it never overwrites an existing file.

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

`workload --help` lists every flag and limit. Two counts are easy to misread:

- `--sessions` counts every conversation, mains included. For two mains and
  six subordinates, use `--sessions 8 --main-sessions 2`.
- `--turns` counts every turn, including the first. The first turn is kept in
  history but left out of the speed and cache figures.

By default you get one main and five subordinates, eight turns each.

### What the endpoint must accept

The endpoint must be plain HTTP. Base URLs and `/v1` are normalized to chat
completions. A custom exact route uses standard chat, without contract
discovery. HTTPS, credentials in URLs, queries and fragments fail explicitly.

There is no auth header flag and no credential discovery. If your deployment
needs another transport or auth boundary, use a suitable, explicitly configured
local HTTP gateway. This mode does not configure that gateway.

## What each session sends

"Main" and "subordinate" only mean bigger and smaller prompts. The sessions do
not delegate to each other. They start together and then run at their own
pace, so a slow first turn in one can overlap measured turns in another.

Each session has its own fixed system prompt. Before every request, the driver
adds three messages to the history:

- a made-up tool call
- its made-up result
- a new user message

No tool runs. No tools are offered either, so if the model calls one anyway,
that session stops. The test never touches a workspace, shell, files,
subprocess or external agent.

The made-up tool records use ordinary Chat Completions fields, so the server
must accept that history. Adding the made-up pair does not reset the persistent
model (`NativeModel`) or its byte-prefix checks.

### Prompt sizes and filler

Sizes are **ASCII padding bytes**, not tokens, and each message also gets a
short label. Main prompts grow faster than subordinate ones, because their user
and tool payloads are larger.

The filler is repeatable. A simple seeded generator (a linear congruential
generator, not cryptographic) picks words from a small vocabulary of our own,
separately for each session, turn and channel. The random session-affinity key
is only a scheduling hint, never a content seed.

### The model's replies

The model's replies are real and kept, so output and scheduling can vary even
at `temperature: 0`. Earlier replies are kept whole, reasoning and provider
fields included.

Some replies contain only reasoning, including ones that end with
`finish_reason: "length"`. If such a reply has no tool calls and its content is
absent or null, it enters history with `content: ""`. The original reasoning
and provider fields stay intact.

This happens once, when the response is ingested, for both SSE and the JSON
fallback, before immutable serialization. That way later requests keep the
complete existing byte prefix.

## Reading the results

The human table shows the main and subordinate role totals, overall
throughput, and each session. From turn 2 onward it counts:

- admission attempts and refusals
- all HTTP 429/503 responses, capability and admission failures included
- errors

A footer covering all turns also shows cold-turn failures.

JSON includes each attempt's stage and status, counts, timing, input admission
or estimate, output and cache usage, finish or error, and history size. Counts
describe requests that were actually attempted, not requests that were
scheduled but never sent. A context-fit refusal is different from an HTTP
overload.

The cache target status is `estimate_meets_target`, `estimate_below_target` or
`insufficient_data`. The next section explains how it is worked out.

### Exit codes

| Exit | Meaning |
| --- | --- |
| 0 | Every requested session and turn completed with an accepted finish. It does not claim a cache target or complete metric telemetry. |
| 1 | A stopped, failed or timed-out workload, or a report I/O failure. |
| 2 | Invalid flags, or an output destination that is unusable or already exists. |

A partial run still writes its metrics. A setup or worker failure, or the
process being killed from outside, can leave an empty or incomplete file.

### What the report file holds

The report destination is checked before any network I/O. It is created with
`create_new` and private Unix mode 0600, and never overwritten.

It holds counts and generic session IDs. It does not hold endpoint or model
names, credentials, affinity keys, fingerprints, prompts, model responses, tool
contents, private paths or raw upstream errors.

## How the cache score is counted

The target is that each role gets at least 98% of its reusable prefix from
cache. It comes from [andreasgru/ds41-flash-hybrid,
METHODS.md](https://github.com/andreasgru/ds41-flash-hybrid/blob/main/METHODS.md),
and we check it separately for each role from turn 2 onward:

```text
U = token length of LCP(current prompt, previous prompt + previous generated output)
C = usage.prompt_tokens_details.cached_tokens
R = sum(min(C, U)) / sum(U), target >= 0.98 for each role
```

LCP is the longest common prefix of two token sequences. So U is the reusable
prefix of a turn, C is what the server reports as cached, and R is the cached
share of the reusable prefix across a role's turns.

### Why the score is an estimate

Proving the target exactly needs token IDs, but the current engine contract
returns only exact prompt lengths. So the workload estimates U, the table marks
the score with `~`, and `exact_target_verified` stays false.

Two things keep it an estimate:

- The current `vade.engine.v1` admission contract returns exact rendered prompt
  lengths, but not rendered token IDs or an exact LCP. Equal lengths alone
  cannot show that two token sequences match.
- The chat template can add a header or shift a token boundary.

So **U and R stay estimates** even when both rendered lengths are exact.
Proving the exact 0.98 criterion needs an LCP or token IDs from the engine, or
an independently verified render-continuity contract. No made-up field or
count-only generation is sent.

### With exact admission

Before each warm turn, the driver asks the engine, without running the model,
how many tokens the previous history plus the full reply comes to. It counts
the smaller of that and the current prompt's length as the reusable prefix:

```text
U_est = min(current exact admitted prompt count,
            exact admitted rendering of previous history + full assistant reply)
```

That count is a separate admission-only request. It covers the kept previous
history, including the full assistant content, reasoning and provider fields.
It carries the frozen generation options and affinity, no request to keep KV,
and no new synthetic observation or user messages. It runs no inference. The
next real generation separately negotiates and admits its complete body byte
for byte, keeping the existing append-only client state.

This avoids adding generated output-token counts to an input-token count. Those
counts can differ when the renderer handles reasoning, assistant terminators,
retokenization or hidden framing. This path needs no reported output count.
JSON labels it `u_basis: "exact_rendered_history_counts"` and keeps the raw
count as `rendered_previous_tokens`.

Each sample keeps `prefix_count_requests`, `prefix_count_elapsed_ms` and
`prefix_count_error`. The top-level `prefix_count_requests` sums these extra
measurement requests. It is kept apart from `all_turn_requests`, which still
describes ordinary generation capability, admission and HTTP activity.

A count refusal, HTTP failure or malformed telemetry leaves U unknown for that
turn. It does not quietly fall back to output or byte estimates, or retry the
count. Generations can complete while cache target coverage is still
insufficient.

The extra admission adds measurement overhead. Its time does not count toward
native TTFT. But count requests that run in between do fall inside the warm
wall-clock window used to compute throughput.

### Without exact admission

Server-reported prompt usage is preferred, and
`U_est = min(current prompt count, previous prompt count + previous output count)`.
If usage is absent:

- input length uses `ceil(request wire bytes / 4) + 256`
- output length uses `ceil(serialized assistant bytes / 4)`

This rough metric estimate is separate from the existing context-budget
estimate kept in each sample's `input` field. Neither is a tokenizer or a
guarantee that the context fits. `u_basis` names the `reported_counts` or
`wire_bytes_estimate` fallback.

### When U is unknown or a session stops

- A change of cache identity or domain leaves U unknown for the turn at that
  boundary.
- An exact admitted count that conflicts with reported prompt usage is a
  measurement error and stops the session.
- Output usage beyond its requested reservation also stops the session.

### The per-turn lines

The text report includes every attempted turn, the cold first turn included,
with:

- P, the admitted or reported prompt count
- raw C, and U
- `100 * min(C,U) / U` percent
- the U basis
- the outcome or error

Missing or zero denominators print an unknown ratio. This makes a turn with
zero cache, a changed identity, or a failed count visible next to the role
summaries. Turn 1 is still left out of their measurements.

### How the sums treat edge cases

- Missing C is unknown, not zero, and invalid or conflicting telemetry also
  stays unknown.
- C greater than U is capped at U.
- Zero U adds zero to both sums, and an all-zero denominator cannot pass.
- `cache_ratio` is null unless every planned measured turn has a usable C/U
  pair and the denominator is positive.
- `paired_cache_ratio` is a separate diagnostic over just the covered turns.
- Coverage, raw C, U, basis, numerator and denominator are all kept.
- Incomplete or failed sessions cannot quietly drop out of the expected
  coverage.
- No combined cache target is calculated across roles.
- The 0.98 comparison uses integer arithmetic.

## How time and throughput are measured

### Time to first token

TTFT (time to first token) is measured with a monotonic clock. It runs from the
native model request boundary, before negotiation and admission, to the first
generated SSE delta. Nonempty content, reasoning and function name/argument
deltas count. Role-only, finish-only and usage-only events do not.

So TTFT includes local preparation, capability discovery, admission, queueing,
network time and parsing of that first event. It observes a generated delta,
not a tokenizer count or physical terminal rendering.

- JSON fallback has no TTFT.
- Failed or truncated responses have no accepted TTFT or provisional cache
  usage.
- Percentiles are nearest-rank over known successful measured samples, and
  their coverage is stated.

### Output throughput

`output_tok_s` is the sum of reported completion tokens divided by the measured
wall window. That window runs from the earliest start at turn 2 or later to the
latest measured end.

- Per-session rates include time between that session's measured requests.
- Role and overall rates use the wall span of the union, so overlapping
  concurrency is counted once, never by adding up session rates.

This is end-to-end output throughput, including admission and
[prefill](../../docs/GLOSSARY.md#pr), **not decode-only speed**. It is null if
any planned output count is missing. `measured_output_tok_s` reports only the
known output over that same span.

### Finishes and failures

The reservation flag is a maximum; it does not force a model to produce that
many tokens. `stop` and validated `length` finishes both give workload samples
and keep history, and length is expected in a token-limited workload. Other
finishes, unexpected tools and errors stop only the affected session, with no
retry.

## Limits and timeouts

| Limit | Value |
| --- | --- |
| Sessions | At most 32 |
| Mains | 1–2 |
| Turns per session | At most 1,000 |
| Session-turns in total | At most 1,024 |
| Each padding field | At most 262,144 bytes |
| Output reservation | 1–8,192 tokens |
| Native history | 4 MiB |
| Native message | 4 MiB per message |
| Native response-stream payload | 64 MiB |
| Native events | 262,144 per response |

Histories are never pruned or compacted to improve the cache score.

### Deadline and waits

The workload deadline is 600 seconds by default (maximum 86,400). When the
deadline is reached, it signals cancellation and joins all owned workers.

Native generation response reads use the full 3,600-second HTTP lifetime. That
includes silent queue/prefill waits before headers or after role-only events.
No request is replayed.

The five-second progress notice counts sessions **queued / awaiting upstream**
until their first generated delta (or complete JSON response). This is a local
observation of waiting, not server queue telemetry.

The other transport bounds still apply:

- a 5-second connect
- a 600-second idle timeout for discovery, admission and request writes
- 250 ms cancellation polling

Standard-library DNS and a connect already in progress cannot be interrupted.
SIGINT or process termination does not currently flush a partial report, so use
`--max-seconds` when you need a timed partial report.

### Where these limits come from

These limits include the output extension that came in with commit `824ab96f`,
which changed the native message, stream and event bounds and the HTTP
timeouts; see the [dated branch
comparison](engine-contract.md#output-extension-provenance-824ab96f). The
workload keeps its 1–8,192-token reservation and turns off automatic
continuation, to keep the single-response timing boundary.

## Check it yourself

To run the tests and checks:

```sh
cargo test --offline --all-targets
cargo test --offline --test workload_mock workload_ -- --nocapture
RUSTFLAGS='-D warnings' cargo check --offline --all-targets
cargo build --release --offline
```

### A mock run with no server

The existing macOS mock example can also launch the release workload CLI and
leave a report for you to inspect, with no separately configured server:

```sh
cargo run --release --offline --example perf_mock -- \
  target/release/vade-harness 1 workload target/workload-mock.json
```

This picks six sessions, three turns, a 17-token reservation and made-up
fixture counts. Both roles are deliberately given full reuse of their
admission-counted, estimated prefix. The process binds its own ephemeral
loopback port and cannot pick an engine URL. Choose a new output filename every
time you run it.

### What the mock fixtures check

The first-party contract mock fixtures check:

- six simultaneous native sessions, byte equality between admission and
  generation, persistent independent keys and unchanged historical bytes
- retention of real assistant messages, synthetic tool pairing, and per-role
  weighted sums and clamping
- separate full-history count requests with reasoning and opaque fields, counts
  that differ from generated-output arithmetic, and failed, refused and
  malformed count telemetry
- per-turn C/U text, overlapping combined throughput, missing usage, the
  standard fallback and JSON-only timing
- 429/503 at both admission and generation, context refusal without generation,
  and no replay
- reasoning-first TTFT, zero cache and zero denominator, length finishes,
  malformed and truncated streams, unexpected tools, identity changes and
  inconsistent usage
- cancellation and joining, eight-turn reasoning-only SSE/JSON histories, and
  31-second silent generation waits
- real CLI JSON permissions and overwrite refusal

No mock timing is a claim about inference performance. The deliberately
mixed-cache fixture gives main R=1.0000 and subordinate R=760/1530≈0.4967. That
shows one role cannot hide another role's poor reuse.

## Method source and code

The linked method was read for the criterion only. All driver, fixture, prompt
and filler code is original Rust using the standard library. No outside code,
tokenizer, cryptography implementation or dependency was added.
