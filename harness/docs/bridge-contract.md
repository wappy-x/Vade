# Handing turns to Codex or Hermes

In `codex` and `hermes` modes, the harness hands each turn to an external
coding agent you installed yourself, and shows that agent's answer. This page
describes that hand-off: which protocol messages the harness sends, what it
accepts back, its limits, and how it handles cancellation and failure. The
[harness README](../README.md) shows the launch commands.

The delegated-agent adapter is our own Rust, using only the standard library.
The executable fixtures are handwritten protocol peers, not copied agent code.
The checks on this page use no installed agent, account files, Engine, Spark
or GPU.

## Codex: the part of the protocol we use

The reference snapshot is
[Codex revision 2fdf047](https://github.com/openai/codex/tree/2fdf047c9631c9ed01a31b62efb7891718a931a8/codex-rs/app-server-protocol),
taken during our research, with the
[official lifecycle documentation](https://learn.chatgpt.com/docs/app-server)
reviewed on 2026-10-08. These sources supply interface ideas and field meanings;
no implementation, schema bundle, prompts, or upstream tests are copied.
`tests/fixtures/codex-v2-items.jsonl` pins a small invented event exchange and
`tests/fixtures/codex_child.rs` is the executable compatibility/failure fixture.

The harness sends only these methods (its outbound allowlist):

- initialize
- thread/start, thread/resume
- turn/start, turn/interrupt
- turn/steer (active-turn terminal control only)

Nothing else is exposed: no generic public RPC function, shellCommand,
command/exec, filesystem, login or configuration-changing operation.
Initialization explicitly declines experimental APIs. A valid userAgent string
confirms the declared initialize shape; it does **not** negotiate capabilities
or prove an installed version is compatible. The harness infers nothing from
semantic versions and has no automatic fallback.

Thread setup checks the exact requested cwd, approval policy and sandbox type,
and the resumed identity. Workspace-write also requires network access to be
explicitly disabled, temporary roots to be excluded, and at most 32 canonical
writable roots, all inside the selected workspace. Unknown sandbox fields fail
closed. This validates the external agent's policy response; it is not Vade
enforcing an OS sandbox, and this adapter does not confine the read scope Codex
selects.

Only the explicit prompt goes to turn/start. The provider and thread identity
stay separate from the native conversation and Engine state. A transport reconnection may
resume the known thread on the next explicit prompt; it never replays a failed
request. External authentication and tool execution remain external.

Codex streams its answer in small pieces (deltas) and then sends the finished
item. Vade tracks items by ID in the order they arrive, shows the pieces as they
come, but treats only the finished item as the real answer. If the two
disagree, the finished item wins and the screen shows the replacement. Sending
the same finished item again changes nothing. Anything that breaks the order is
an error, never a quiet success: a changed finished item (or item type), a
piece arriving after the finish, or a successful turn that ends with an item
still open. A final snapshot of the items is checked against the item events,
and an empty snapshot does not erase finished items.

At most 256 items and 8 MiB of complete item JSON are retained per turn. No item
body is logged. Completed text is available only after validated turn success.
Plain CLI output prints only that answer. Raw UI output is marked provisional,
and a differing final answer is shown explicitly as a replacement.

Messages Vade knows but does not need are ignored, within limits. Lifecycle
messages it uses are checked and tied to their thread and turn, so a late event
from another turn cannot add to the current answer. Messages or item types it
does not recognize end the connection with an Unsupported error, so a future
Codex protocol change needs a deliberate update here.

Known approval callbacks are declined and permission grants are empty. Every
other callback gets a method-not-supported error. There is no approval UI and
no auto-approval.

| Condition | Result |
| --- | --- |
| Valid completed terminal plus closed items | Completed; authoritative text available |
| Valid interrupted terminal | Interrupted, acknowledged by the external provider |
| Cancel before dispatch | NotStarted; no remote stop claim |
| Provider completes during cancellation race | Completed remotely; UI discards local result |
| Provider turn failure | Failed, no successful local turn |
| Interrupt rejection or timeout | Error; remote outcome unknown, child cleanup attempted |
| Child crash, malformed/partial stream, unknown event | Error; no final text, no replay |
| Callback during cancellation/tool execution | Deny; continue bounded interruption, or fail explicitly |

Initialization can be cancelled, including while it waits for stdin
acknowledgment. Stdio runs over Unix socket pairs, polled every 50 ms, with
these limits and time budgets:

| Limit | Value |
| --- | ---: |
| JSONL record | 1 MiB |
| Inbound records | 8 |
| Queued writes | 4 |
| Pre-response events | 32 / 4 MiB |
| Events per turn or handshake | 16,384 |
| Total wire data per turn or handshake | 64 MiB |
| RPC | 20 s |
| Turn, absolute | 300 s |
| Turn, idle | 60 s |
| Interrupt grace | 5 s |

Callback replies and interrupts share those bounded writes. Cancellation during
a partial write is ambiguous, so it cannot be reported as an acknowledged
interruption.

Cleanup stops I/O, disconnects the bounded queues, kills and waits for the
owned child, and joins both I/O workers. After a failure, `Client::cleanup()`
reports the actual reap and join results. If creating the writer thread fails,
the reader is also released and joined.
Inherited descriptors cannot hold the I/O workers open. This is **not** process
tree containment: external descendants and remote computation remain the
external agent's responsibility. OS process spawn/wait, filesystem operations,
and arbitrary client event callbacks cannot be preempted by these deadlines.

## Hermes (optional): the part of ACP v1 we use

Interface sources are [Hermes ACP internals at 25a71a7](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/website/docs/developer-guide/acp-internals.md)
and the Agent Client Protocol's (ACP) [initialization](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/initialization.mdx),
[prompt lifecycle](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/prompt-turn.mdx),
and [session setup](https://github.com/agentclientprotocol/agent-client-protocol/blob/5ed386e81033918fe5e3a0503beaa6030280666e/docs/protocol/v1/session-setup.mdx)
at revision `5ed386e`. Sources define field meanings and lifecycle expectations;
`tests/fixtures/hermes_child.rs` is an original executable protocol fixture.

Hermes mode works like Codex mode: Hermes runs the whole turn itself, separate
from Vade's own agent loop. The two adapters share only the code that talks to a
child process over stdin and stdout.

The selected executable receives exactly one argument, `acp`, and the canonical
selected workspace as cwd. Initialization declares ACP version 1, disabled client
filesystem read/write, disabled terminal, and disabled terminal authentication.
There are no MCP (Model Context Protocol) servers or additional workspace
roots in session setup. An incompatible major version is an explicit error. In
particular, ACP v2 prompt responses have different terminal semantics and are
not accepted as v1 success.

The harness sends only these methods (its private outbound allowlist):

- initialize
- session/new, session/load, session/resume
- session/prompt, session/cancel

For an existing session, the adapter uses resume if it is advertised,
otherwise load if that is advertised. It records advertised fork, list and
close extras but does not expose or invoke them. There is no authentication,
host terminal or filesystem execution, generic RPC, configuration write, or
fallback to another backend. Local shutdown closes I/O and kills and reaps the
owned child; it does not invent a shutdown RPC or claim that a remote session
was closed or deleted. Child descendants remain an external responsibility.

Every permission request selects a valid reject_once option if there is one,
or returns cancelled. While cancelling, permission replies always return cancelled.
No allow option is selected. All other host callbacks, including filesystem,
terminal, authentication, and unknown extensions, get method-not-supported.
Malformed envelopes fail closed. The required CLI policy is called
deny-callbacks because it does **not** set up an OS sandbox for the external
runtime's tools. The adapter claims neither workspace read confinement nor
isolation of the external tools or network.

ACP v1 output is a projection of the provider stream, staged until the pending
prompt response. Text messages group by optional message ID; interleaved tool
calls require known IDs and cannot change a terminal status. Successful end_turn
requires all observed tool calls to be closed. The stop reasons max_tokens,
max_turn_requests and refusal are explicitly failed turns with no completed
answer. Only a cancelled prompt response acknowledges interruption. A
concurrent end_turn still counts as remote completion, and the UI discards it
when the turn was cancelled locally.

RPC errors invalidate the client, with the external outcome unknown. Unknown
update variants, stop reasons, responses, session IDs, tool IDs or statuses,
partial JSONL, and transport loss all fail explicitly. No failed or ambiguous
prompt is replayed automatically. A later explicit prompt can open a new
connection and load or resume the known session.

ACP v1 has no turn ID in session/update: correct attribution relies on its ordered
prompt-response boundary. It cannot prove that a nonconforming peer's delayed
same-session chunk belongs to a previous turn.

Initialize and session metadata, and the bounded loaded history, stay in
memory as opaque provider values. Loaded history is validated but never shown
as a new answer or inserted into native context. Updates and the terminal
response for the latest attempt are kept separately, including partial and
failed attempts; only completed_text() identifies a successful answer. Nothing
creates a persistent transcript or imports credentials or memory. External diagnostics are suppressed in
the CLI, and child stderr never shares the JSONL stream.

The transport limits above apply unchanged. ACP setup and individual writes have
a 20-second budget; the pending prompt uses the 300-second absolute and 60-second
idle turn deadlines, with 5 seconds for cancellation acknowledgment. ACP adds
these limits:

| Limit | Value |
| --- | ---: |
| Prompt | 64 KiB |
| Message IDs and tool calls | 256 |
| Answer text | 8 MiB |
| Retained updates per attempt | 4,096 / 8 MiB |
| Loaded-session replay | 32 updates / 4 MiB |

Overflow is an explicit failure. Large existing histories can exceed this
deliberately narrow load limit; where resume is negotiated, it avoids the
replay. Only text output is supported. Arbitrary callbacks and OS operations
still cannot be preempted.

Mock coverage includes new/load/resume, reusing one session across turns,
exact Unicode/multiline prompts,
interleaved message IDs and tools, opaque extensions, known and unknown denied
callbacks, delayed permission during cancellation, actual cancellation receipt,
completion/cancellation races, version mismatch, unadvertised session operations,
explicit reconnect, malformed/partial/oversized streams, crashes, hanging setup,
hanging prompts, and missing cancellation acknowledgment. CLI tests check that
plain stdout contains completed answers and that control receipts go to stderr.
These fixtures do not establish compatibility with an installed Hermes version.

## Terminal control while an agent works

While Codex or Hermes is working, you can keep typing, queue follow-ups, steer
a Codex turn, or cancel, and the screen stays responsive. To do that, the
terminal never blocks waiting for input: it handles up to 16 keypresses per
redraw and shows at most 16 KiB of streamed text per frame. Every worker wakes
the screen as soon as it has something new, so there is no polling delay.

Follow-ups wait in a queue of up to eight prompts (64 KiB). Steering messages
have their own eight-slot queue, and acknowledgments have a 16-slot queue, so a
flood of output can never drop a cancel or a receipt. The final result has a
reserved slot of its own, and streamed text is merged in a separate 64 KiB
buffer.

A full render buffer ends the provisional stream at its first gap, and the
completed output repairs the display. It cannot drop a receipt or block the
model worker. Native complete history still returns through its final
result. Native and delegated plain output use only completed answer text.

Completion, `/new`, and `/undo` invalidate the recall cursor without
overwriting the editor. Up-arrow also clamps stale indices to current history.
A regression test checks a completion that evicts ten old turns while recall is
active, preventing an out-of-bounds panic.

An ordinary prompt during work or `/queue TEXT` creates a separate follow-up.
Follow-ups run after success. Failure/cancellation holds them for `/next`, and
EOF reports held prompts as unsent. `/queue-clear` discards them explicitly.

`/steer TEXT` is Codex-only and never stands in for a new turn: it sends
turn/steer with the expected active turn ID and exact input, with at most one
RPC in flight. Local queue acknowledgment, remote acceptance or rejection, and
an unconfirmed outcome are separate messages. Each turn allows 32 steering
requests, 64 KiB in total. Cancellation takes priority over unsent steering. Pending controls
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

Conceptual credit:
[Pi's TUI design](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/tui/README.md),
[OpenCode's UI/service separation](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/packages/web/src/content/docs/server.mdx),
and [Hermes' merged control-path fix](https://github.com/NousResearch/hermes-agent/pull/12560).
No code or tests from those projects were copied.

Broader provider compatibility and process containment remain later work.

## Fixed: a session could not be reopened at once

In one full parallel test run, a session could not be reopened straight after
it closed: the lock was still briefly held while other threads were starting
child processes. Before the fix, you might have seen a brief "session name is
already in use" error. Our probe (`tests/fixtures/lock_spawn_probe.rs`)
reproduced it 59 and 53 times in about 100,000 tries:

| Run | Immediate reacquisition failures | Acquisitions | Self-mock spawns |
| --- | ---: | ---: | ---: |
| First reproduction | 59 | 99,699 (successful initial) | 1,475 |
| Second reproduction | 53 | 99,686 | 1,731 |
| `--explicit-unlock` mode (the fix) | 0 | 100,000 | 1,778 |

The fix is that the lock now releases itself explicitly (`LOCK_UN`) before its
file is closed, and with that change the probe saw no failures in 100,000
tries. Closing the file stays the fallback, and the lock file itself is never
removed or replaced. A regression test keeps a duplicate descriptor open to
model the race; it failed before the fix and passes after it, without sleeps or
retries. This was checked on macOS only.

## Verification

From the `harness/` folder, use `cargo test --offline` with Rust 2024 edition
tooling. The app-server tests compile only the first-party fixture via rustc
(optionally `VADE_TEST_RUSTC`). All fixtures run in generated target
workspaces. They exercise exact prompt forwarding, resume, reusing one thread
across turns, fragmented
Unicode, final/delta disagreement, stale IDs, policy
mismatches, known/unknown callback denial, unknown notifications, malformed,
partial and oversized streams, child crash/hang, output bursts, blocked stdin,
handshake cancellation, tool cancellation, interrupt rejection/timeout, and the
completion/cancel race. No live interoperability claim follows from mock success.
