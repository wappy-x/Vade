# Delegated agent contract

This is first-party Rust using only the standard library. The executable fixtures
are handwritten protocol peers, not copied agent code. No installed agent,
account files, Engine, Spark, or GPU is used by these checks.

## Codex subset

The reference snapshot is the research round's
[Codex revision 2fdf047](https://github.com/openai/codex/tree/2fdf047c9631c9ed01a31b62efb7891718a931a8/codex-rs/app-server-protocol),
with the [official lifecycle documentation](https://learn.chatgpt.com/docs/app-server)
reviewed on 2026-10-08. These sources supply interface ideas and field meanings;
no implementation, schema bundle, prompts, or upstream tests are copied.
`tests/fixtures/codex-v2-items.jsonl` pins a small invented event exchange and
`tests/fixtures/codex_child.rs` is the executable compatibility/failure fixture.

The outbound allowlist is initialize, thread/start, thread/resume, turn/start,
turn/interrupt, and turn/steer (active-turn terminal control only).
There is no generic public RPC function, shellCommand, command/exec, filesystem,
login, or configuration-mutation operation. Initialization explicitly declines
experimental APIs. A valid userAgent string confirms the declared initialize
shape; it does **not** negotiate capabilities or prove an installed version is
compatible. No semantic-version inference or automatic fallback is performed.

Thread setup verifies the exact requested cwd and approval policy, sandbox type,
and resumed identity. Workspace-write additionally requires explicit disabled
network access, excluded temporary roots, and at most 32 canonical writable roots
inside the selected workspace. Unknown sandbox fields fail closed. This validates
the external policy response; it is not Vade enforcing an OS sandbox. Codex's
selected read scope is not confined by this adapter.

Only the explicit prompt goes to turn/start. Provider/thread identity stays
separate from native conversation and Engine state. A transport reconnection may
resume the known thread on the next explicit prompt; it never replays a failed
request. External authentication and tool execution remain external.

Items are tracked by ID in encounter order. Deltas are provisional. Completed
agent items supply the completed text; disagreement with a delta is legal and
the final item wins. Repeating an identical completed item is idempotent; changing
it, changing its type, sending a later delta, or completing a successful turn
with an unfinished item fails explicitly. Nonempty terminal item snapshots are
validated against item events; empty snapshots do not erase completed items.
At most 256 items and 8 MiB of complete item JSON are retained per turn. No item
body is logged. Completed text is available only after validated turn success.
Plain CLI output prints only that answer. Raw UI output is marked provisional,
and a differing final answer is shown explicitly as a replacement.

Known informational notifications are bounded and ignored; supported lifecycle
notifications are validated and scoped by thread/turn. Stale foreign-turn events
cannot supply active output. Unknown notifications/item types invalidate the
connection with an Unsupported error; they never silently become success. A
future protocol revision needs an intentional contract/fixture update. Known
approval callbacks decline, permission grants are empty, and all other callbacks
receive a method-not-supported error. There is no approval UI or auto-approval.

| Condition | Result |
| --- | --- |
| Valid completed terminal plus closed items | Completed; authoritative text available |
| Valid interrupted terminal | Interrupted acknowledged by external provider |
| Cancel before dispatch | NotStarted; no remote stop claim |
| Provider completes during cancellation race | Completed remotely; UI discards local result |
| Provider turn failure | Failed, no successful local turn |
| Interrupt rejection or timeout | Error; remote outcome unknown, child cleanup attempted |
| Child crash, malformed/partial stream, unknown event | Error; no final text, no replay |
| Callback during cancellation/tool execution | Deny; continue bounded interruption, or fail explicitly |

Initialization is cancellation-aware, including waits for stdin acknowledgment.
Stdio uses Unix socket pairs with 50 ms I/O polls, 1 MiB JSONL records, eight
inbound records, four queued writes, 32 pre-response events/4 MiB, 16,384 events,
and 64 MiB total wire data per turn/handshake. Budgets remain 20 seconds for RPC,
300 seconds absolute turn, 60 seconds idle, and 5 seconds interrupt grace. Callback
replies and interrupts share those bounded writes. Cancellation during a partial
write is ambiguous, so it cannot be reported as acknowledged interruption.

Cleanup stops I/O, disconnects bounded queues, kills/waits the owned child, and
joins both I/O workers. `Client::cleanup()` reports actual reap/join results after
failure. Writer-thread creation failure also releases and joins the reader.
Inherited descriptors cannot hold the I/O workers open. This is **not** process
tree containment: external descendants and remote computation remain the
external harness's responsibility. OS process spawn/wait, filesystem operations,
and arbitrary client event callbacks cannot be preempted by these deadlines.

## Verification

Use `cargo test --offline` with Rust 2024 edition tooling. The app-server tests
compile only the first-party fixture via rustc (optionally `VADE_TEST_RUSTC`). All
fixtures run in generated target workspaces. They exercise exact prompt forwarding,
resume/reuse, fragmented Unicode, final/delta disagreement, stale IDs, policy
mismatches, known/unknown callback denial, unknown notifications, malformed,
partial and oversized streams, child crash/hang, output bursts, blocked stdin,
handshake cancellation, tool cancellation, interrupt rejection/timeout, and the
completion/cancel race. No live interoperability claim follows from mock success.

## Terminal control

`interaction.rs` is a binary-owned module; shared protocol/native exports are
unchanged. The UI consumes up to 16 ready input events before rendering, using
`try_recv` and wake-driven `park_timeout`, with no blocking input wait or token
polling delay. Every worker lane wakes the UI; retained input/render batches
schedule another immediate pass. Each frame takes at most 16 KiB of provisional
text. It has an eight-entry/64 KiB work
queue, an eight-entry steering control channel, a 16-entry receipt lane, a
reserved terminal-result slot, atomic cancellation/state, and a separate 64 KiB
coalescing render buffer. A full render buffer ends the provisional stream at
its first gap; completed output repairs display. It cannot drop a receipt or
block the model worker. Native complete history still returns through its final
result. Native and delegated plain output use only completed answer text.
Completion, `/new`, and `/undo` invalidate the recall cursor without overwriting
the editor. Up-arrow also clamps stale indices to current history. A regression
checks a completion that evicts ten old turns while recall is active, preventing
an out-of-bounds panic.

An ordinary prompt during work or `/queue TEXT` creates a separate follow-up.
Follow-ups run after success. Failure/cancellation holds them for `/next`, and
EOF reports held prompts as unsent. `/queue-clear` discards them explicitly.
`/steer TEXT` is Codex-only and never aliases a new turn: it sends turn/steer with
the expected active turn ID and exact input, with at most one RPC in flight.
Local queue acknowledgment, remote acceptance/rejection, and an unconfirmed
outcome are separate messages. Each turn permits 32 steering requests/64 KiB
aggregate. Cancellation takes priority over unsent steering. Pending controls
are never retried in another turn or connection.

The raw editor stays usable during work, with UTF-8 scalar cursor movement,
line Home/End, Delete, word deletion, and Up/Down message recall.
Large paste is bounded and cannot submit a command until Enter. Oversized input
is discarded as a whole. Resize starts a fresh frame because a terminal may have
reflowed the previous frame's rows. Cell widths are approximations; full Unicode
grapheme editing, an emulator-independent full-screen layout, and diff views are
not implemented. The visible phase comes from local execution or supported
external events, never guessed inference telemetry. Permission callbacks are
denied, so there is no unresolved approval UI to block the control path.

The Mac PTY fixture checks active editing, queued/accepted steering, two width
changes, Ctrl-C, oversized bracketed paste, exit, and exact termios restoration.
The delayed child fixture checks cancellation while steering is in flight;
the burst fixture emits 2,000 x 128-byte deltas before an interrupted terminal.
Tests print their actual acknowledgment measurements with fixture conditions.
These are debug-build local observations, not latency guarantees. A blocked OS
terminal write, filesystem operation, process spawn/wait, or arbitrary external
callback still cannot be preempted by this UI. Native adapters do not expose all
model/compaction phases, so those phases are not fabricated.

Conceptual credit: [Pi's TUI design](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/tui/README.md),
[OpenCode's UI/service separation](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/packages/web/src/content/docs/server.mdx),
and [Hermes' merged control-path fix](https://github.com/NousResearch/hermes-agent/pull/12560).
No code or tests from those projects were copied.

Broader provider compatibility and process containment remain later work.

## Optional Hermes ACP v1 subset

Interface sources are [Hermes ACP internals at 25a71a7](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/website/docs/developer-guide/acp-internals.md)
and ACP's [initialization](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/initialization.mdx),
[prompt lifecycle](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/prompt-turn.mdx),
and [session setup](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/session-setup.mdx)
at revision `5ed386e`. Sources define field meanings and lifecycle expectations;
`tests/fixtures/hermes_child.rs` is an original executable protocol fixture.
The binary's `delegation::Backend` owns an entire external turn and is separate
from the native model/tool loop. The adapter lives under the existing app-server
export; no shared protocol or library export change is needed. Codex and Hermes
share only the extracted, bounded first-party stdio child transport.

The selected executable receives exactly one argument, `acp`, and the canonical
selected workspace as cwd. Initialization declares ACP version 1, disabled client
filesystem read/write, disabled terminal, and disabled terminal authentication.
There are no MCP servers or additional workspace roots in session setup. An
incompatible major version is an explicit error. In particular ACP v2 prompt
responses have different terminal semantics and are not accepted as v1 success.

The private outbound allowlist is initialize, session/new, session/load,
session/resume, session/prompt, and session/cancel. An existing session uses
resume only when advertised, otherwise load only when advertised. The adapter
records advertised fork/list/close extras but does not expose or invoke them.
No authentication, host terminal/filesystem execution, generic RPC, configuration
write, or fallback to another backend exists. Local shutdown closes I/O and
kills/reaps the owned child; it does not invent a shutdown RPC or claim a remote
session was closed/deleted. Child descendants remain an external responsibility.

Every permission request selects a valid reject_once option if present, or
returns cancelled. While cancelling, permission replies always return cancelled.
No allow option is selected. All other host callbacks, including filesystem,
terminal, authentication, and unknown extensions, get method-not-supported.
Malformed envelopes fail closed. The required CLI policy is named deny-callbacks
because it does **not** establish an OS sandbox for the external runtime's tools.
Neither workspace read confinement nor external tool/network isolation is claimed.

ACP v1 output is a projection of the provider stream, staged until the pending
prompt response. Text messages group by optional message ID; interleaved tool
calls require known IDs and cannot change a terminal status. Successful end_turn
requires all observed tool calls closed. max_tokens, max_turn_requests, and
refusal are explicit failed turns with no completed answer. Only a cancelled
prompt response acknowledges interruption; a concurrent end_turn remains remote
completion, and the UI discards it when locally cancelled. RPC errors invalidate
the client with unknown external outcome. Unknown update variants, stop reasons,
responses, session IDs, tool IDs/statuses, partial JSONL, and transport loss fail
explicitly. No failed or ambiguous prompt is automatically replayed. A later
explicit prompt can create a new connection and load/resume the known session.
ACP v1 has no turn ID in session/update: correct attribution relies on its ordered
prompt-response boundary. It cannot prove that a nonconforming peer's delayed
same-session chunk belongs to a previous turn.

Initialize/session metadata and bounded loaded history remain opaque provider
values in memory. Loaded history is validated but not displayed as a new answer
or inserted into native context. Updates and the terminal response for the latest
attempt are retained separately, including partial/error attempts; only
completed_text() identifies a successful answer. No persistent transcript or
credential/memory import is introduced. External diagnostics are suppressed in
the CLI, and child stderr never shares the JSONL stream.

The transport limits above apply unchanged. ACP setup and individual writes have
a 20-second budget; the pending prompt uses the 300-second absolute and 60-second
idle turn deadlines, with 5 seconds for cancellation acknowledgment. Additional
ACP limits are 64 KiB per
prompt, 256 message IDs and tool calls, 8 MiB answer text, 4,096 retained updates /
8 MiB per attempt, and 32 updates / 4 MiB for loaded-session replay. Overflow is
an explicit failure. Large existing histories can exceed this deliberately narrow
load limit; negotiated resume avoids replay when available. Only text output is
supported. Arbitrary callbacks and OS operations remain outside preemption.

Mock coverage includes new/load/resume, reuse, exact Unicode/multiline prompts,
interleaved message IDs and tools, opaque extensions, known and unknown denied
callbacks, delayed permission during cancellation, actual cancellation receipt,
completion/cancellation races, version mismatch, unadvertised session operations,
explicit reconnect, malformed/partial/oversized streams, crashes, hanging setup,
hanging prompts, and missing cancellation acknowledgment. CLI tests check that
plain stdout contains completed answers and that control receipts go to stderr.
These fixtures do not establish compatibility with an installed Hermes version.

## Session-lock release

One full parallel run failed the unchanged session test's immediate lock
reacquisition assertion after dropping its session. Serial execution passed.
The standalone first-party `tests/fixtures/lock_spawn_probe.rs` reproduced
transient flock retention while other threads spawn children despite verified
FD_CLOEXEC: 59 immediate reacquisition failures in 99,699 successful initial
acquisitions with 1,475 self-mock spawns on this Mac. This supports a spawn/lock
lifetime race; it is not evidence that a post-exec child retains the descriptor.
A subsequent reproduction on
this Mac found 53 immediate reacquisition failures in 99,686 acquisitions with
1,731 self-mock spawns. It is a real, fail-closed product race: a caller can get
a transient "session name is already in use" after the previous owner drops.

`anchored::Lock` now owns each successful acquisition and issues `LOCK_UN` before
closing its descriptor. It retries an interrupted unlock; final close remains
the fallback on another OS error or abrupt process exit. The stable private lock
file is never removed or replaced. Failed contenders do not get an unlocking
guard, and the workspace boundary, permissions, and atomic metadata replacement
are unchanged.

The session regression keeps a duplicated descriptor open, which deterministically
models the shared open file description inherited before exec. It failed before
adding the guard and passes afterward without sleeps or retries. It also checks
that active owners exclude contenders, metadata survives reacquisition, and
closing an old duplicate cannot unlock a new owner. The self-spawn diagnostic's
`--explicit-unlock` mode observed 100,000 acquisitions, zero immediate reacquisition
failures, and 1,778 mock spawns. This is macOS-only validation; other-platform
execution and live external-agent compatibility remain unverified.
