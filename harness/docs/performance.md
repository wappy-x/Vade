# Model path performance and prefix contract

This round uses first-party Rust, no dependencies, loopback mock HTTP servers and
an explicit first-party app-server mock child. No Engine, Spark, GPU, installed
Codex executable or live model was used. The measurements concern harness cost;
there is no inference throughput or engine cache-hit claim.

The product's [native workload mode](workload.md) separately drives concurrent
growing agent histories and reports per-role cache/timing/throughput metrics.
Its exact-admission-based LCP approximation is explicitly labeled; current
admission lengths alone do not establish token-prefix equality.

## What stays byte-identical

A native `Message` is validated and serialized once. Its JSON bytes never change.
The conversation serializes options (including tools, model and sampling fields)
once. `NativeModel` retains an `AppendOnlyRequest`, checks every old record by
exact byte comparison, and appends only new message records. It refuses a changed
system prompt, tool definition, option or historical record before generation dispatch. Optional capability
discovery can precede that local check.
No turn counter, timestamp, workspace path or display text enters that prefix.

For conventional chat JSON, the precise invariant is:

```rust
assert!(next_body.starts_with(&previous_body[..previous_body.len() - 2]));
// The two excluded bytes are the messages-array/object terminator: ]}
```

When native KV checkpoints are advertised, `kv_checkpoint` and `kv_handle` are
added after the messages array, outside this canonical builder. Those changing
request-local hints are excluded from the byte-prefix invariant. They do not
modify options, schema order or historical messages; exact admission sees the
complete decorated generation body. See [checkpoint semantics](engine-checkpoints.md).

The entire previous HTTP/JSON request cannot be a literal prefix of a longer
request: `Content-Length` changes and closing JSON delimiters must move. Engine
chat templates, assistant generation markers and tokenization are outside this
repository. Byte equality here does not prove token equality or KV reuse there.

`NativeModel::reset_prompt` starts an explicit cache boundary, for a new
conversation or caller-directed compaction. The UI's `/new` replaces the model
session. Integration of future context budgeting must keep that reset explicit.
The native runtime, tool executor and session policy remain separately owned;
`agent.rs` only exposes the cached options and reads cached message serialization.

Direct chat places fixed options before messages and preserves exact successful
exchanges. If its bounded Session evicts an exchange or excludes an oversized
response, the UI requires `/new` before another model request, so trimmed history
is never silently sent. Explicit `/new` and `/undo`, failed/canceled direct turns,
and provider-owned delegation state are boundaries to the successful-turn prefix
contract. Direct failures still discard their provisional exchange.

The gateway preserves caller request bytes, except that requests with no sampling
fields receive the [shared greedy defaults](passthrough-matrix.md). Any named
sampling field, including `null`, preserves the caller's body. Stable-prefix
cache decoration remains an explicit opt-in; it is not a prefix repair. The three-request Responses mock includes
function calls/results and proves unchanged source prefixes reach the upstream.
Native tests capture six actual HTTP requests (three user turns and three tool
continuations), including reasoning and provider extensions, on one socket.
Actual CLI integration captures three user turns on one socket in both direct
and native modes. The newer [Engine contract client](engine-contract.md) adds
negotiated fixed decoration and exact admission, with complete decorated-prefix
and tool-continuation fixtures. Those mocks do not execute the Engine.

Delegation sends only the exact current prompt, on the same client and thread.
A three-turn first-party app-server mock checks whitespace, Unicode, unchanged
input shape, absence of injected history/instructions, and mock-owned tool
history. This proves the adapter boundary, not the installed Codex model prompt.
Actual Codex token-prefix behavior remains unverified.

## Transport and rendering

`ChatClient` and `NativeModel` retain idle HTTP/1.1 connections across successful
turns. The gateway shares a bounded upstream pool across downstream requests.
Optional [cluster routing](gateway-routing.md) adds capability-verified endpoint
affinity, local in-flight load balancing and passive reported-usage metrics.
Three-endpoint mock counts in that guide demonstrate routing, not Engine speed.
The pool retains at most 16 idle sockets and expires them after 15 seconds. A
nonblocking probe discards already-closed or unexpectedly readable idle sockets.
A peer can still close after the probe: that POST fails and is never replayed.

Only fully consumed, self-delimited responses can return a socket to the pool.
HTTP/1.0, `Connection: close`, close-delimited bodies, errors, cancellations,
truncation and buffered extra response bytes retire the socket. Chunk terminators
and trailers are consumed before reuse. `[DONE]` never substitutes for complete
HTTP framing. The downstream gateway still closes after one request; there is
no HTTP pipelining, WebSocket, TLS or compression support. Unsupported TLS fails
explicitly. Gateway binding remains loopback-only.

`BodyReader::fill_buf` lends the existing socket buffer to SSE parsing and gateway
relay. Complete single-data-line SSE events borrow that buffer. JSON views validate
all fields and duplicate keys without constructing a value tree; unescaped text
borrows input through the display callback. Escaped strings need decoding,
fragmented/multiline SSE events use reusable bounded scratch, and retained
conversation state and cross-thread display messages still require ownership.
This is a borrowed common path, not an allocation-free or end-to-end zero-copy
claim. Tools and unfamiliar extension deltas retain the full owned parser.

Input and model queues wake the UI thread. The old 20 ms display polling delay
and mandatory 5 ms producer backpressure sleeps are removed from normal delivery.
The merged bridge keeps separate control, receipt, completion, and render lanes.
Adjacent deltas coalesce without a batching delay in a 64 KiB render buffer;
each pass takes at most 16 KiB. A full render buffer stops provisional display
at its first gap; authoritative completion repairs the display. Workers never
wait for rendering. Receipt and completion channels use `wake::channel`; render,
notice, and state changes unpark the same receiver. Input uses `try_recv` in
bounded batches, and retained input/render work schedules another immediate
pass. `park_timeout` only bounds spinner, resize, and signal checks; it does not
gate delivery. Follow-up work and steering retain their separate bounds.

DNS, connect and reads stay on workers. Raw-terminal integration checks send Esc
while an upstream sends no response bytes, observe the cancellation notice within
1 s, and verify socket closure. Blocking terminal output still cannot be preempted;
OS DNS and an in-progress connect retain their existing cancellation limitations.

## Reproduce locally

```sh
cargo test --offline --all-targets
RUSTFLAGS='-D warnings' cargo check --offline --all-targets
cargo fmt --all -- --check
cargo build --release --offline
cargo run --release --offline --example perf_mock -- target/release/vade-harness 3
cargo run --release --offline --example prompt_memory
```

`perf_mock` is macOS-only. It binds fresh ephemeral loopback ports and uses a
mock in the benchmark parent; CPU/RSS accounting covers only the harness child.
Each run renders 100,000 synthetic `x` characters. Direct mode sends one character
per SSE event; native mode sends eight (12,500 events). The native host prose
budget is explicitly disabled with `--reply-tokens 0` so this intentionally long
fixture completes in one model call. These are characters, not tokenizer tokens,
and no model is loaded. The fixture exercises production HTTP, parsing, queues,
sanitization, native journaling and output paths.

Each stream begins and ends with unique text markers. `first_output_ms` starts
immediately before the mock writes the first marker and ends when the parent
observes it. Raw PTY receipts wait for the synchronized terminal frame to close;
read boundaries alone are not frame completion. Direct pipe output and both raw
terminal modes must display the first marker before the mock sends the rest.
This handshake proves progressive delivery. Native redirected output intentionally
publishes only the authoritative completed answer, so its `progressive=false`
row measures completed-answer latency, **not** per-token display overhead.
Plain rows must contain the entire exact answer; raw rows verify complete start
and end frames. Physical display paint and engine computation are excluded.

`wall_s` ends when the final marker is observed. Live CPU samples bracket that
interval via `proc_pid_rusage`, converting Mach ticks with `mach_timebase_info`;
100% is one fully used CPU core. The fixture's progress handshake and parent
scheduling are included in wall time. Resident bytes are sampled at both ends;
`wait4` records whole-process CPU and peak RSS through exit. Live CPU totals must
not exceed whole-process totals beyond a 2 ms accounting tolerance.
`NO_COLOR=1` preserves the raw TUI. Startup measures spawn to a complete ready
raw demo frame, with no request sent. Shutdown drains the PTY so a final redraw
cannot block an otherwise finished child.

The benchmark has a 240-second overall deadline, 10-second startup and
30-second per-stream bounds, limits repetitions to 1..31, and cleans up only
its own child and temporary workspace. A missing marker reports the escaped
output tail. `TMPDIR` controls the workspace location. Four example tests cover
split-frame receipts, terminal-buffer-sized shutdown output and owned-child
timeout/reaping. `pty` selects just the two raw rows; `workload NEW_JSON_FILE`
retains the separate six-session/three-turn synthetic contract fixture:

```sh
cargo test --offline --example perf_mock
cargo run --release --offline --example perf_mock -- target/release/vade-harness 7 pty
cargo run --release --offline --example perf_mock -- target/release/vade-harness 1 workload target/workload-mock.json
```

The current marker/frame protocol differs from the historical benchmark below.
The older first-`x` samples do not establish current terminal-frame latency and
must not be compared as a before/after speedup for this benchmark repair.

`prompt_memory` measures live allocator-requested bytes for a native conversation
and its append-only request builder: 100 user turns, 100 mock tool results of
1,024 bytes each, and user messages of 256 bytes each. It excludes allocator
metadata, fragmentation, OS/socket buffers and thread stacks. It also times 1,000
preparations of the retained request versus reserializing cached records or full
owned values. Its counting allocator adds overhead to those microbenchmarks.
Whole-process RSS in `perf_mock` is a separate measurement.

Raw samples, units, sample counts and host/toolchain characteristics accompany
this document in `performance-samples.txt`. Small sample sizes, scheduler/load
variation and the first launch after rebuilding limit comparisons. No startup
speed improvement is claimed.

## Current host baseline (2026-10-23)

Apple M4 Pro, macOS 27.0.1 ARM64, Rust 1.99.0, release `-D warnings`, seven
repetitions per row with the current marker/frame protocol. The raw samples and
source identity are in [performance-current-samples.txt](performance-current-samples.txt).
These measurements identified a first-frame redraw wait in the current TUI.
They do not show an engine or tokenizer speed change.

| Mode/output | Output policy | First output median (ms) | Stream wall median (s) |
| --- | --- | ---: | ---: |
| direct / pipe | progressive | 0.134 | 0.168485 |
| agent / pipe | completed answer | 31.092 | 0.031270 |
| direct / PTY | progressive, complete frame | 35.428 | 0.192408 |
| agent / PTY | progressive, complete frame | 33.472 | 0.066939 |

## First visible frame (2026-10-23)

The first visible answer now triggers an immediate raw-terminal redraw. Later
stream frames keep the 33 ms budget. Redirected native output still waits for the
completed answer. An alternating `A X X A X A A X` comparison, four repetitions
per batch, measured 16 samples per row per arm. Every plain answer matched the
full 100,000-character fixture; every raw stream passed its start/end-frame
receipts and progressive-output handshake.

| Raw terminal | Before median (ms) | After median (ms) | After min–max (ms) |
| --- | ---: | ---: | ---: |
| direct | 34.870 | 0.158 | 0.049–0.782 |
| agent | 34.924 | 0.133 | 0.060–5.525 |

This removes over 99.5% of the measured median first-frame delay. These are
loopback-to-complete-frame observations, not GPU or physical-paint timings.
Scheduler/load variation remains visible in the ranges. Whole-stream CPU and
pipe measurements were noisy; no throughput or CPU reduction is claimed.
All samples, binary hashes and source identities are in
[performance-first-frame-samples.txt](performance-first-frame-samples.txt).

A separate opt-in regression checks seven trials per mode against a 20 ms median
bound, leaving room for scheduling while detecting the old 33 ms wait. It is
excluded from ordinary CI timing assertions and can be run serially:

```sh
cargo test --release --offline --test snappy first_answer_frame_is_immediate -- --ignored --test-threads=1 --nocapture
```

## Historical measured results (2026-10-08)

Apple M4 Pro, macOS 27.0.1 ARM64, rustc 1.98.1, release builds. All rows have
n=3; times are medians, with final latency min–max in parentheses. CPU is total
user + system time during the stream. Peak RSS is the maximum of the three
whole-process peaks, in bytes. Every row renders 100,000 synthetic characters.
The native rows have one eighth as many SSE/JSON events as direct rows.

| Mode/output | Baseline first output (ms) | Final first output (ms) | Baseline → final stream wall (s) | Final streaming CPU (s) | Final max peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct / pipe | 22.561 | 0.103 (0.097–0.332) | 80.784015 → 0.090922 | 0.157566 | 3,276,800 |
| agent / pipe | 23.241 | 0.059 (0.056–0.106) | 10.233990 → 0.012582 | 0.021710 | 3,194,880 |
| direct / PTY | 28.207 | 0.093 (0.078–0.098) | 83.233605 → 0.075459 | 0.149030 | 3,014,656 |
| agent / PTY | 23.719 | 0.092 (0.091–0.096) | 10.361180 → 0.010939 | 0.021291 | 3,276,800 |

The final raw demo ready prompt took 2.791, 11.969 and 12.772 ms (n=3; median
11.969 ms). An earlier first launch after rebuilding took 76.705 ms (n=1). Caches
and scheduler load were uncontrolled; no startup speedup claim is made.

The 100-turn native session retained 685,662 heap bytes (n=3, identical), including
215,808 bytes of request-buffer capacity, for 152,160 serialized history bytes.
Its empty conversation/mock builder used 2,106 heap bytes. Median request
preparation was 4.643 µs, cached-record serialization 5.757 µs, and reserialization
of owned message values 318.712 µs (n=3 batches of 1,000 iterations, counting
allocator enabled). These are harness serialization costs, not engine cache metrics.

## Historical contract telemetry check (2026-10-08)

After adding capability/admission and usage handling, the same release mock
benchmark was run again (n=3 per row, same host/toolchain and 100,000-character
workloads). Discovery receives a mock 404; these samples exercise ordinary-chat
fallback. Stream timing begins at the first SSE write, so it excludes discovery
and admission latency. Engine count/compute performance is not measured.

| Mode/output | First output median (ms; min–max) | Stream wall median (s) | Streaming CPU median (s) | Max peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| direct / pipe | 0.079 (0.034–0.081) | 0.096153 | 0.169356 | 3,293,184 |
| agent / pipe | 0.040 (0.031–0.042) | 0.012840 | 0.022873 | 3,293,184 |
| direct / PTY | 0.047 (0.041–0.053) | 0.085006 | 0.169691 | 3,293,184 |
| agent / PTY | 0.041 (0.034–0.042) | 0.012031 | 0.023899 | 3,375,104 |

Demo ready-prompt samples were 90.467, 10.713 and 7.670 ms, including the first
launch after rebuilding. No startup improvement is claimed. Small sample sizes
and uncontrolled load prevent attributing every difference from the initial samples to code.
These checks retain prompt responsiveness; they do not claim a new speedup.
Usage collection reuses direct events' validated JSON views and avoids rescanning
ordinary native deltas with no possible usage key (escaped keys take the full
path). All raw samples are in
[performance-contract-samples.txt](performance-contract-samples.txt).

## References and implementation ownership

Protocol ideas follow RFC 9112 (HTTP/1.1), RFC 8259 (JSON), and the WHATWG SSE
format already cited by the core. The macOS examples use the Darwin SDK's
`sys/resource.h`, `libproc.h` and Mach timebase ABI declarations for accounting,
and `openpty` for a local terminal. These are OS interfaces, not Rust dependencies.
All implementation and fixture code is first-party. No competitor code was copied.
