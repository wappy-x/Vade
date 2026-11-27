# Harness overhead and the stable prefix

This page answers two questions for anyone who wants the harness to be fast:

- **What does the harness itself cost?** That is the time and memory it adds
  on top of the model.
- **Does it let the engine reuse its prompt cache?** An engine can skip work it
  has already done only when a new request starts with exactly the same bytes
  as the last one. The harness is built so that it never rewrites what it has
  already sent.

Every measurement here ran against small mock servers on
[loopback](../../docs/GLOSSARY.md#ho) (the machine talking to itself), plus a
mock app server, all written in our own dependency-free Rust. No engine, Spark,
GPU, installed Codex or real model was involved. So nothing on this page says
how fast inference is, or how often the engine cache hits.

For cache behaviour across many agent conversations, see the harness's separate
[native workload mode](workload.md). It drives several growing agent histories
at once and reports cache, timing and throughput figures for each role. One of
those figures approximates the LCP (longest common prefix) from exact admission
counts, which are the engine's own prompt lengths (see
[exact admission](engine-contract.md#exact-admission)). The report labels it
as an approximation, because equal admission lengths alone do not prove that
two token prefixes match.

## How the harness keeps the prompt prefix stable

Each way of talking to a model (native agent, direct chat, gateway and
delegation) keeps earlier bytes unchanged in its own way. This section goes
through them in turn, then lists what the tests prove.

### Native agent mode

Each native `Message` is checked and turned into JSON once, and those bytes
never change. The options (tools, model, sampling) are serialized once too.

`NativeModel` keeps an append-only request (`AppendOnlyRequest`). Before
sending, it compares every old record byte for byte, then adds only the new
messages. It refuses to send if any of these has changed:

- the system prompt
- a tool definition
- an option
- an earlier message

Optional capability discovery may run before this check. Turn counters,
timestamps, workspace paths and display text never go into the prefix.

The code keeps these jobs apart. The native runtime, tool executor and session
policy each stay in their own module; `agent.rs` only exposes the cached
options and reads the cached message serialization.

### The exact rule, and what it does not cover

For conventional chat JSON, each new body must start with the previous body
minus its last two bytes. In code, the precise invariant is:

```rust
assert!(next_body.starts_with(&previous_body[..previous_body.len() - 2]));
// The two excluded bytes are the messages-array/object terminator: ]}
```

When the engine advertises native KV checkpoints, `kv_checkpoint` and
`kv_handle` are added after the messages array, outside this canonical builder.
These hints belong to a single request and change from one request to the
next, so the byte-prefix rule leaves them out. They do not change options,
schema order or earlier messages, and exact admission still sees the complete
decorated generation body. See [checkpoint semantics](engine-checkpoints.md).

Two things sit outside this guarantee:

- **The HTTP request as a whole.** The whole previous HTTP/JSON request can
  never be a literal prefix of a longer one, because `Content-Length` changes
  and the closing JSON delimiters have to move.
- **What the engine does with the bytes.** Engine chat templates, assistant
  generation markers and tokenization live outside this repository. Byte
  equality here does not prove token equality or KV reuse there.

### Direct chat

Direct chat also puts the fixed options first and keeps every successful
exchange exactly as sent. Its session has a size limit. If the session has to
drop an exchange, or leaves out a reply that was too big, the UI makes you type
`/new` before the next request. That way it never quietly sends a trimmed
history.

### When a conversation starts fresh

`NativeModel::reset_prompt` starts an explicit cache boundary, for a new
conversation or for a compaction the caller asks for. In the UI, `/new`
replaces the model session. Any future context-budgeting integration must keep
that reset explicit.

The prefix promise covers only successful turns in a row. Each of these starts
fresh:

- `/new`
- `/undo`
- a failed or cancelled direct turn (a failed direct turn also discards its
  unfinished exchange)
- delegation state held by the provider

### Gateway

The gateway keeps the caller's request bytes, with one exception: a request
with no sampling fields gets the [shared greedy defaults](passthrough-matrix.md).
Naming any sampling field, even as `null`, keeps the caller's body as it is.
Stable-prefix cache decoration is an explicit opt-in, not a way to repair a
prefix.

### Delegation to Codex

Delegation sends only the exact current prompt, on the same client and thread.
A three-turn first-party app-server mock checks:

- whitespace and Unicode
- that the input shape does not change
- that no history or instructions are injected
- that the mock owns the tool history

This proves the adapter boundary, not the prompt the installed Codex model
sees. Codex's real token-prefix behavior is still unverified.

### What the tests prove

- The three-request Responses mock includes function calls and results, and
  proves that unchanged source prefixes reach the upstream.
- Native tests capture six real HTTP requests on one socket: three user turns
  and three tool continuations, including reasoning and provider extensions.
- Real CLI integration captures three user turns on one socket, in both direct
  and native modes.
- The newer [Engine contract client](engine-contract.md) adds negotiated fixed
  decoration and exact admission, with fixtures for the complete decorated
  prefix and for tool continuations.

Those mocks do not run the engine.

## Connections, streaming and the display

### Reusing connections

`ChatClient` and `NativeModel` keep idle HTTP/1.1 connections across successful
turns. The gateway shares a bounded upstream pool across downstream requests.
The pool keeps at most 16 idle sockets and expires them after 15 seconds.

A nonblocking probe throws away idle sockets that are already closed or
unexpectedly readable. A peer can still close after the probe; that POST then
fails and is never replayed.

Optional [cluster routing](gateway-routing.md) adds endpoint affinity checked
against capabilities, local in-flight load balancing, and passive metrics from
reported usage. The three-endpoint mock counts in that guide show routing, not
engine speed.

A socket goes back to the pool only after a fully consumed, self-delimited
response. Any of these retires the socket instead:

- HTTP/1.0 or `Connection: close`
- a close-delimited body
- an error, a cancellation or truncation
- extra response bytes left in the buffer

Chunk terminators and trailers are consumed before reuse, and `[DONE]` never
stands in for complete HTTP framing.

Some things are not supported. The downstream gateway still closes after one
request, and there is no HTTP pipelining, WebSocket, TLS or compression
support. Unsupported TLS fails explicitly. Gateway binding stays
loopback-only.

### Parsing streamed text without copying

Most streamed text is parsed straight from the socket buffer, without copying
it. `BodyReader::fill_buf` lends that buffer to SSE parsing (server-sent
events, the streaming format) and to the gateway relay. A complete SSE event
on a single data line borrows it directly. JSON views check every field and
catch duplicate keys without building a value tree, and unescaped text reaches
the display callback still borrowed.

Some cases still copy:

- escaped strings, which need decoding
- SSE events split across reads or lines, which use reusable bounded scratch
  space
- the kept conversation and the messages sent to the display thread, which
  need their own copies

So this is a borrowed common path, not zero allocation or end-to-end zero copy.
Tool calls and unfamiliar extension deltas still use the full owned parser.

### Display updates without polling

The display no longer polls. Typed input and model output wake the UI thread
directly, so the old 20 ms polling delay and the forced 5 ms producer sleeps
are gone from normal delivery.

Separate channels carry control messages, receipts, finished answers and render
updates. Small pieces of text that arrive together merge into a 64 KiB render
buffer with no added wait, and each redraw takes at most 16 KiB. If the buffer
fills, the live display pauses at the gap and the finished answer fills it in.
Workers never wait for the screen.

In the code:

- Receipt and completion channels use `wake::channel`; render, notice and
  state changes unpark the same receiver.
- Input is read with `try_recv` in bounded batches, and leftover input or
  render work schedules another pass right away.
- `park_timeout` only bounds the spinner, resize and signal checks; it does
  not hold up delivery.
- Follow-up work and steering keep their own bounds.

### Cancelling with Esc

DNS, connect and reads stay on worker threads. Raw-terminal integration checks
send Esc while an upstream sends no response bytes. They see the cancellation
notice within 1 s and confirm that the socket closes.

Two limits remain. Blocking terminal output still cannot be preempted, and OS
DNS and a connect already in progress keep their existing cancellation limits.

## Reproduce locally

Test, build and run both benchmarks with:

```sh
cargo test --offline --all-targets
RUSTFLAGS='-D warnings' cargo check --offline --all-targets
cargo build --release --offline
cargo run --release --offline --example perf_mock -- target/release/vade-harness 3
cargo run --release --offline --example prompt_memory
```

### What `perf_mock` runs

`perf_mock` is macOS-only. It binds fresh ephemeral loopback ports and runs a
mock in the benchmark parent, so CPU and RSS (resident memory) accounting
covers only the harness child.

Each run renders 100,000 synthetic `x` characters:

- **Direct mode** sends one character per SSE event.
- **Native mode** sends eight per event (12,500 events). The native host prose
  budget is turned off explicitly with `--reply-tokens 0`, so this
  deliberately long fixture finishes in one model call.

These are characters, not tokenizer tokens, and no model is loaded. The fixture
exercises the production HTTP, parsing, queue, sanitization, native journaling
and output paths.

In the result tables, **agent** rows are native mode. **Pipe** rows send output
to a pipe, and **PTY** rows run the raw terminal UI on a pseudo-terminal.

### How the benchmark measures

- **Markers:** each stream starts and ends with unique text markers.
- **First output** (`first_output_ms`): from just before the mock writes the
  first marker to when the parent sees it. Raw PTY rows wait for the
  synchronized terminal frame to close, because a read boundary alone does not
  mean the frame is complete.
- **Streaming:** direct pipe output and both raw terminal modes must show the
  first marker before the mock sends the rest, which proves the output arrives
  progressively. Native redirected output deliberately prints only the
  finished answer. Its `progressive=false` row therefore measures time to the
  completed answer, **not** per-token display overhead.
- **Correct output:** plain rows must contain the entire exact answer, and raw
  rows check complete start and end frames. Physical display paint and engine
  computation are not included.
- **Wall time** (`wall_s`): until the final marker is seen, including the
  fixture's progress handshake and the parent's scheduling.
- **CPU:** sampled live at both ends of that interval with `proc_pid_rusage`,
  with Mach ticks converted by `mach_timebase_info` (100% = one core). The live
  totals are checked against the `wait4` whole-process totals and may not
  exceed them by more than 2 ms.
- **Memory:** resident bytes at both ends, plus peak RSS through exit from
  `wait4`.
- **Startup:** from spawn to a complete, ready raw demo frame, with no request
  sent. `NO_COLOR=1` keeps the raw TUI.
- **Shutdown:** drains the PTY, so a final redraw cannot block a child that
  has otherwise finished.
- **Limits:** 240 s overall, 10 s startup, 30 s per stream, 1-31 repetitions.
  It cleans up only its own child and temporary workspace, and `TMPDIR` sets
  where that workspace goes.
- **Failures:** a missing marker reports the escaped tail of the output.

### More `perf_mock` runs

Four example tests cover split-frame receipts, shutdown output the size of a
terminal buffer, and owned-child timeout and reaping. The `pty` argument picks
just the two raw rows. The `workload NEW_JSON_FILE` argument runs the separate
six-session, three-turn synthetic contract fixture:

```sh
cargo test --offline --example perf_mock
cargo run --release --offline --example perf_mock -- target/release/vade-harness 7 pty
cargo run --release --offline --example perf_mock -- target/release/vade-harness 1 workload target/workload-mock.json
```

### What `prompt_memory` measures

`prompt_memory` measures the live bytes requested from the allocator for a
native conversation and its append-only request builder. The conversation has
100 user turns, 100 mock tool results of 1,024 bytes each, and user messages of
256 bytes each.

It leaves out allocator metadata, fragmentation, OS and socket buffers, and
thread stacks. It also times 1,000 preparations of the retained request against
reserializing cached records or full owned values. Its counting allocator adds
overhead to those microbenchmarks. Whole-process RSS in `perf_mock` is a
separate measurement.

## Current host baseline (2026-10-23)

The host was an Apple M4 Pro on macOS 27.0.1 ARM64, with Rust 1.99.0 and a
release build under `-D warnings`. Each row has seven repetitions with the
current marker/frame protocol. The raw samples and source identity are in
[performance-current-samples.txt](performance-current-samples.txt).

These measurements found a first-frame redraw wait in the current TUI; the
next section covers the fix. They do not show any change in engine or
tokenizer speed. The agent / pipe row waits for the completed answer by design,
so its first output is the time to the whole answer.

| Mode and output | Output policy | First output median (ms) | Stream wall median (s) |
| --- | --- | ---: | ---: |
| direct / pipe | progressive | 0.134 | 0.168485 |
| agent / pipe | completed answer | 31.092 | 0.031270 |
| direct / PTY | progressive, complete frame | 35.428 | 0.192408 |
| agent / PTY | progressive, complete frame | 33.472 | 0.066939 |

## First visible frame (2026-10-23)

The first visible answer now triggers an immediate raw-terminal redraw. Later
stream frames keep the 33 ms budget, and redirected native output still waits
for the completed answer.

We compared the build before the change (A) with the build after it (X), in the
alternating order `A X X A X A A X`. Each batch had four repetitions, for 16
samples per row for each build. Every plain answer matched the full
100,000-character fixture. Every raw stream passed its start/end-frame receipts
and the progressive-output handshake.

| Raw terminal | Before median (ms) | After median (ms) | After min–max (ms) |
| --- | ---: | ---: | ---: |
| direct | 34.870 | 0.158 | 0.049–0.782 |
| agent | 34.924 | 0.133 | 0.060–5.525 |

This removes over 99.5% of the measured median first-frame delay. These are
observations from loopback to a complete frame, not GPU or physical-paint
timings, and scheduler and load variation still shows in the ranges.
Whole-stream CPU and pipe measurements were noisy, so we claim no throughput or
CPU reduction. All samples, binary hashes and source identities are in
[performance-first-frame-samples.txt](performance-first-frame-samples.txt).

A separate opt-in regression test checks seven trials per mode against a 20 ms
median bound. That leaves room for scheduling while still catching the old
33 ms wait. It is kept out of the ordinary CI timing assertions. Run it
serially:

```sh
cargo test --release --offline --test snappy first_answer_frame_is_immediate -- --ignored --test-threads=1 --nocapture
```

## Historical measured results (2026-10-08)

These numbers come from an earlier version of the benchmark. The current
marker/frame protocol differs, and the older first-`x` samples do not show
current terminal-frame latency. Because the benchmark itself changed, do not
compare them with the new numbers as a before/after speedup.

The host was an Apple M4 Pro on macOS 27.0.1 ARM64, with rustc 1.98.1 and
release builds. Every row has n=3. Times are medians, with the final latency's
min–max in parentheses. CPU is total user + system time during the stream.
Peak RSS is the largest of the three whole-process peaks, in bytes. Every row
renders 100,000 synthetic characters, and the native rows have one eighth as
many SSE/JSON events as the direct rows.

| Mode and output | Baseline first output (ms) | Final first output (ms) | Baseline → final stream wall (s) | Final streaming CPU (s) | Final max peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct / pipe | 22.561 | 0.103 (0.097–0.332) | 80.784015 → 0.090922 | 0.157566 | 3,276,800 |
| agent / pipe | 23.241 | 0.059 (0.056–0.106) | 10.233990 → 0.012582 | 0.021710 | 3,194,880 |
| direct / PTY | 28.207 | 0.093 (0.078–0.098) | 83.233605 → 0.075459 | 0.149030 | 3,014,656 |
| agent / PTY | 23.719 | 0.092 (0.091–0.096) | 10.361180 → 0.010939 | 0.021291 | 3,276,800 |

The final raw demo ready prompt took 2.791, 11.969 and 12.772 ms (n=3; median
11.969 ms). An earlier first launch after a rebuild took 76.705 ms (n=1).
Caches and scheduler load were not controlled, so we make no startup speedup
claim.

The 100-turn native session kept 685,662 heap bytes (n=3, identical),
including 215,808 bytes of request-buffer capacity, for 152,160 bytes of
serialized history. Its empty conversation and mock builder used 2,106 heap
bytes.

The median request preparation was 4.643 µs, cached-record serialization
5.757 µs, and reserialization of owned message values 318.712 µs (n=3 batches
of 1,000 iterations, with the counting allocator on). These are harness
serialization costs, not engine cache metrics.

Raw samples, units, sample counts and host and toolchain details are in
[performance-samples.txt](performance-samples.txt). Small sample sizes,
scheduler and load variation, and the first launch after a rebuild all limit
comparisons. We claim no startup speed improvement.

## Historical contract telemetry check (2026-10-08)

After we added capability, admission and usage handling, we ran the same
release mock benchmark again: n=3 per row, on the same host and toolchain, with
the same 100,000-character workloads.

Discovery gets a mock 404, so these samples exercise the ordinary-chat
fallback. Stream timing starts at the first SSE write, so it leaves out
discovery and admission latency. Engine count and compute performance are not
measured.

| Mode and output | First output median (ms; min–max) | Stream wall median (s) | Streaming CPU median (s) | Max peak RSS (bytes) |
| --- | ---: | ---: | ---: | ---: |
| direct / pipe | 0.079 (0.034–0.081) | 0.096153 | 0.169356 | 3,293,184 |
| agent / pipe | 0.040 (0.031–0.042) | 0.012840 | 0.022873 | 3,293,184 |
| direct / PTY | 0.047 (0.041–0.053) | 0.085006 | 0.169691 | 3,293,184 |
| agent / PTY | 0.041 (0.034–0.042) | 0.012031 | 0.023899 | 3,375,104 |

Demo ready-prompt samples were 90.467, 10.713 and 7.670 ms, including the first
launch after a rebuild. We claim no startup improvement. With small samples and
uncontrolled load, not every difference from the initial samples can be put
down to code. These checks keep the prompt responsive; they do not claim a new
speedup.

Usage collection reuses the validated JSON views of direct events. It skips
rescanning ordinary native deltas that cannot contain a usage key (escaped keys
take the full path). All raw samples are in
[performance-contract-samples.txt](performance-contract-samples.txt).

## Sources

Protocol ideas follow RFC 9112 (HTTP/1.1), RFC 8259 (JSON), and the WHATWG SSE
format already cited by the core. For accounting, the macOS examples use the
Darwin SDK's `sys/resource.h`, `libproc.h` and Mach timebase ABI declarations,
and `openpty` for a local terminal. These are OS interfaces, not Rust
dependencies. All implementation and fixture code is first-party. No code from
competing projects was copied.
