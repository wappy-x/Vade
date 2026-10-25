# Engine input checkpoints (optional)

An input checkpoint is a prompt state the engine saves on request, so that a
later request with the same start can skip reading that part again. You only
need this page if your engine advertises `kv_branch_handles`; without that, the
harness sends no checkpoint fields and no release requests.

The checkpoint client follows the scoped KV wire contract on the engine's
`kv-handles` branch, read as source on 2026-10-08 (`crates/serve/README.md`,
`protocol.rs`, `usage.rs`, `contract.rs`, and the renderer/cache boundary
checks).

Verification uses only first-party Rust loopback mocks on macOS. No Engine
binary, model, Spark, GPU, or live delegated agent was connected or tested.
There is no claim of measured Engine speed, memory saving, or live
interoperability.

## What a checkpoint saves, and what it does not

`kv_checkpoint:true` retains the entire canonical **input prompt** before decode.
It does not save:

- the generated answer;
- an internal message boundary you choose;
- a separately reusable system prompt, tools and project-rules prefix.

The reason is that the Chat renderer adds an assistant-generation suffix, so a
system-only or post-answer side request is usually not a token prefix of the
next user or child request. `max_tokens:0` is rejected. Sending an invented
field or a one-token warm-up would not implement those missing boundaries.

The harness therefore requests a checkpoint on each actual native model request,
including closed tool continuations. Only validated terminal JSON/SSE responses
publish hints. Each hint records the number of input messages it actually covered.
At completed turns the journal keeps the earliest eligible input and the
latest input (only one when the quota allows a single handle). Generated tails
remain transcript data and are read again by the next real request. There is no
background or warm-up inference, extra generation, local tokenizer, raw KV
storage, or prompt truncation.

What the engine cannot do yet: save a checkpoint at a point you choose, such
as right after the system prompt or right after an answer, without generating
anything. Until it can, a child that shares only the system prompt, tools and
project rules with its parent cannot use the parent's checkpoint, because that
checkpoint also contains the parent's task, so the child gets none. A child
that starts from the parent's complete saved input can use it. The mock models
the generation suffix on purpose, to keep this difference visible in tests.

## Negotiation, requests and fallback

The engine must advertise `kv_branch_handles:true` together with
`kv_checkpoints` containing:

- an opaque `cache_domain`;
- `scope:"authorization"`;
- positive `max_handles_per_scope`, `max_bytes_per_scope` and `ttl_ms`;
- `release:"/v1/kv/release"`.

This scope domain may differ from ordinary shared prefix-cache identity.
Metadata is bounded, and capability metadata that says yes but is malformed
fails negotiation. Remote metadata cannot redirect release to another host or
path; the route is a sibling of the configured Chat endpoint.

Configure native authentication with `--api-key-file PATH`
or `VADE_API_KEY_FILE` (the explicit flag takes precedence). The common transport
loads the file and adds `Authorization: Bearer <key>` to native metadata,
admission, generation and checkpoint-release requests. It does not discover
credentials from other clients; checkpoint hints contain no credential.

Checkpoint hints are added **after** the immutable options and messages. They
never enter the append-only prefix check, tool definitions, messages, or provider
options persisted with the transcript. Exact admission receives the complete
final generation bytes, including hints and output reservation. Full prompt and
output accounting remain mandatory; cached input is not subtracted from context.
Explicit `cache:false` disables automatic use and issuance. Explicit caller-owned
`kv_handle` or `kv_checkpoint` options opt out of lifecycle management.

A hint must match the freshly negotiated authorization domain, be locally
unexpired, fit advertised byte bounds, and refer to a preserved message prefix.
The Engine still compares every saved rendered token. A changed domain
suppresses the handle. An expired, foreign, evicted, unavailable or mismatched
handle never triggers a retry or changes the transcript; the Engine's normal
full-prompt fallback supplies the answer.

A rejected hint is removed from the next branch snapshot. Retention
failures such as `capacity` leave a successful answer successful. Unknown status
labels display as unknown; malformed or conflicting KV telemetry is not published
as a checkpoint or hit. Standard cached-token telemetry remains independently
collected.

No advertisement means no automatic checkpoint fields or release requests.
Custom routes and foreign Chat servers keep existing behavior. The direct text
Chat client observes telemetry but does not manage transcript-branch checkpoints;
the native tool-aware agent owns this lifecycle.

## Quotas, persistence and ownership

A **handle** is the engine's ID for one saved checkpoint. The harness keeps
each handle in a *hint*, together with its domain, token lengths, expiry and
byte counts at issuance. It keeps at most two hints for the active branch,
stored in the private (0600) journal next to the conversation, not as messages.
Forks copy them, normal turns keep them, and compaction clears them. A broken
hint record never stops the conversation from loading, and hints have no say
over files or tool execution.

Handles never show up in `/context`, `/branch`, `/journal` or debug output;
`/journal` masks their bytes without changing offsets. The raw journal files do
contain them, along with the conversation, and are not encrypted, so keep those
files private and out of logs. Older readers ignore these optional records and
still replay the conversation.

Across branches, the harness tracks up to 128 branch names and up to 16
handles, or fewer if the engine's `max_handles_per_scope` is lower. Each branch
has its own prompt, contract state and HTTP pool. Past 128 branches, new
branches just send the full prompt, and handles beyond the count budget are
retired. Deleting a branch keeps any handle another branch still uses. The
engine enforces the real byte and count limits, also across concurrent
children and other processes, and may drop any handle early when it is under
pressure.

Standalone `retained_bytes` are **never summed** as a shared-memory total;
`scope_retained_bytes` is a snapshot at issuance, not a live reservation.

Superseded or unreferenced handles are released at worker boundaries. When the
final owner is dropped normally, or on explicit shutdown, the harness tries to
release every known handle. Release uses its own bounded HTTP pool: at most 16 handles per normal batch,
4 KiB response bounds and a two-second total socket budget. OS DNS resolution cannot
be cancelled through std. Cleanup is synchronous, so a final-owner drop during
`/new` or exit can wait for this bounded release batch. To avoid that, the
child/UI coordinator can call the explicit shutdown hook first, in its joined
cleanup worker.

A release failure is nonfatal and not retried; Engine expiry or eviction
remains the fallback. Early remote release cannot be guaranteed after abrupt
process death, when terminal metadata is missing or invalid, for untracked
external journal copies, or for handles another process owns. Separate processes
may invalidate each other's optional hints; full transcripts remain usable.

## Wiring it into a host program

The native CLI binds the worker client to its journal name. `/fork` and `/switch`
use a fresh `NativeModel::branch_client()` while retaining the shared owner.
`/new` and normal exit release that session's known handles. `/context` reports
`hit`, `replayed`, or the actual miss/retention reason, reused tokens, input and
reusable lengths, expiry and byte snapshots. Ordinary cached-token counts can
also include fallback-prefix hits and are reported separately.

The child scheduler should construct each child with:

```rust,ignore
let mut child_model = parent_model.branch_client();
child.inherit_checkpoint_hints(&parent_conversation);
child_model.bind_branch(child_journal.name())?;
```

Inheritance requires identical serialized options and an identical full saved
message prefix; it is conservative when tools/options change. `branch_client()`
can be created before capability discovery. It shares ownership, not mutable
prompt state. Run the usual `agent::run_observed` with the child's journal; its
validated hints are recorded automatically. Do not put handles in model options,
child-task text, tool results, event logs or UI rows. Ending or cancelling a
child must not call the whole owner's shutdown hook.

After joining a deleted branch's worker and dropping its active history, call
`NativeModel::delete_branch(journal, workspace)`. It deletes only that locked
journal and releases unshared hints, returning attempted/confirmed counts.
`Journal::delete` and `NativeModel::release_branch` are the separate lower-level
operations. The harness has no delete command in its UI; a host that adds one
should use this combined API. Join all child workers before explicit
`release_checkpoints()` on owner shutdown; last shared-owner drop also runs it.
Resume/fork alone never connects, admits a request, generates, or replays a tool.

The gateway currently does not route the release endpoint or provide KV-handle
cluster affinity. Use a directly configured Engine endpoint for this native
lifecycle. Gateway routing of releases and handle affinity is separate work;
an unconfirmed release still has Engine TTL as fallback.

## Local evidence and design credits

`tests/kv_mock.rs` exercises JSON and SSE issuance, exact admitted request bytes,
full-prompt reuse, journal forks/switch/resume, concurrent independent children,
parent-reference preservation, deletion and process-exit release, expiry,
foreign handles, count/byte quota eviction, restart/domain changes, capacity
refusal, prompt mismatch, cancellation, malformed telemetry and truncated streams.
It also runs the native CLI through fork/switch/context/exit. Unit tests cover
local expiry, bounded metadata, conflicting snapshots, redaction, compaction,
and optional-metadata recovery. Existing no-capability fixtures remain unchanged.

These test the wire format and state against a mock we wrote. They are not
evidence of native engine rendering parity or cache performance.

Design references are credited in [Credits](../../CREDITS.md#harness).
Implementation is first-party Rust using only the standard library; no competitor
code or tests were copied.
