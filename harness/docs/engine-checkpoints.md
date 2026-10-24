# Optional Engine input checkpoints

The checkpoint client follows the scoped KV wire contract on the separate Engine's `kv-handles`
branch, inspected as source on 2026-10-08 (`crates/serve/README.md`, `protocol.rs`,
`usage.rs`, `contract.rs`, and the renderer/cache boundary checks). Verification
uses only first-party Rust loopback mocks on macOS. No Engine binary, model,
Spark, GPU, or live delegated agent was connected or tested. There is no claim
of measured Engine speed, memory saving, or live interoperability.

## The current boundary matters

`kv_checkpoint:true` retains the entire canonical **input prompt** before decode.
It does not retain the generated answer, select an internal message boundary, or
return a separately reusable system/tools/project-rules prefix. The Chat renderer
adds an assistant-generation suffix; a system-only or post-answer auxiliary
request is not generally a token prefix of the next user/child request.
`max_tokens:0` is rejected. Sending an invented field or a one-token warm-up would
not implement those missing boundaries.

The harness therefore requests a checkpoint on each actual native model request,
including closed tool continuations. Only validated terminal JSON/SSE responses
publish hints. Each hint records the number of input messages it actually covered.
At completed turns the journal retains the earliest eligible input and latest
input (one under a one-handle quota). Generated tails remain transcript data and
are read again by the next real request. There is no background/warm-up inference,
extra generation, local tokenizer, raw KV storage, or prompt truncation.

**Still needed from Engine:** a negotiated operation for a selected canonical
message boundary, usable without decoding, to create exact system-prefix and
completed-answer checkpoints. A child inheriting only system/tools/project rules
cannot use a handle whose saved input also contains a different task. It receives
no such handle. Children inheriting the complete saved input can use its hint.
The mock deliberately models a generation suffix to preserve this distinction.

## Negotiation, requests and fallback

`kv_branch_handles:true` must accompany `kv_checkpoints` with an opaque
`cache_domain`, `scope:"authorization"`, positive `max_handles_per_scope`,
`max_bytes_per_scope`, `ttl_ms`, and `release:"/v1/kv/release"`. This scope domain
may differ from ordinary shared prefix-cache identity. Metadata is bounded and
malformed affirmative capability metadata fails negotiation. Remote metadata
cannot redirect release to another host/path; the route is a sibling of the
configured Chat endpoint. Configure native authentication with `--api-key-file PATH`
or `VADE_API_KEY_FILE` (the explicit flag takes precedence). The common transport
loads the file and adds `Authorization: Bearer <key>` to native metadata,
admission, generation and checkpoint-release requests. It does not discover
credentials from other clients; checkpoint hints contain no credential.

Checkpoint hints are added **after** the immutable options and messages. They
never enter the append-only prefix check, tool definitions, messages, or provider
options persisted with the transcript. Exact admission receives the complete
final generation bytes, including hints and output reservation. Full prompt and
output accounting remain mandatory; cached input is not subtracted from context.
Explicit `cache:false` disables automatic use/issuance. Explicit caller-owned
`kv_handle` or `kv_checkpoint` options opt out of lifecycle management.

A hint must match the freshly negotiated authorization domain, be locally
unexpired, fit advertised byte bounds, and refer to a preserved message prefix.
The Engine still compares every saved rendered token. Changed domains suppress
the handle; expired/foreign/evicted/unavailable/mismatched handles never trigger a
retry or change the transcript. The Engine's normal full-prompt fallback supplies
the answer. A rejected hint is removed from the next branch snapshot. Retention
failures such as `capacity` leave a successful answer successful. Unknown status
labels display as unknown; malformed or conflicting KV telemetry is not published
as a checkpoint or hit. Standard cached-token telemetry remains independently
collected.

No advertisement means no automatic checkpoint fields or release requests.
Custom routes and foreign Chat servers keep existing behavior. The direct text
Chat client observes telemetry but does not manage transcript-branch checkpoints;
the native tool-aware agent owns this lifecycle.

## Quotas, persistence and ownership

The active branch stores at most two hints, each including the opaque handle,
authorization domain, canonical input/reusable token lengths, expiry, standalone
retained bytes, and scope bytes at issuance. Records are private 0600 journal
metadata, not model messages. Forked journals inherit them; ordinary appends keep
them; compaction clears them. An invalid optional hint cannot prevent transcript
recovery. Hints have no authority over file state or tool execution.

Debug output, `/context`, `/branch`, and `/journal` do not display the handle.
Journal range inspection masks handle bytes without changing offsets, including
invalid hint metadata. Raw journal files contain private scoped capabilities and
transcript content. They are not encrypted, and must not be copied into logs.
Older readers can ignore the new optional records and replay the transcript.

A shared owner tracks at most 128 branch names and at most
`min(16, max_handles_per_scope)` handles. Each branch has independent prompt and
contract state and HTTP pooling. At the branch bookkeeping limit, new branches
use ordinary full-prompt generation. The owner preserves known shared references
when a branch is deleted. Handles beyond its count budget are retired; Engine
pressure can independently evict any handle earlier. The Engine enforces actual
byte/count quotas and issuance atomicity, including concurrent children and other
processes. Standalone `retained_bytes` are **never summed** as a shared-memory total;
`scope_retained_bytes` is a snapshot at issuance, not a live reservation.

Superseded/unreferenced handles are released at worker boundaries. Normal final
owner drop or explicit shutdown attempts release of all known handles. Release
uses a dedicated bounded HTTP pool, at most 16 handles per normal batch, 4 KiB
response bounds and a two-second aggregate socket budget; OS DNS resolution is
not cancellable through std. Cleanup is synchronous: a final-owner drop during
`/new` or exit can wait for this bounded release batch; the child/UI coordinator
can call the explicit shutdown hook in its joined cleanup worker first.
A release failure is nonfatal and not retried;
Engine expiry/eviction remains the fallback. Abrupt process death, unreceived or
invalid terminal metadata, untracked external journal copies, and another
process's ownership cannot guarantee early remote release. Separate processes
may invalidate each other's optional hints; full transcripts remain usable.

## Integration seams

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
child-task text, tool results, event logs or UI rows. Ending/cancelling a child
must not call the entire owner's shutdown hook.

After joining a deleted branch's worker and dropping its active history, call
`NativeModel::delete_branch(journal, workspace)`. It deletes only that locked
journal and releases unshared hints, returning attempted/confirmed counts.
`Journal::delete` and `NativeModel::release_branch` are the separate lower-level
operations. No delete UI command existed at this baseline; a coordinator-added
command should use this combined API. Join all child workers before explicit
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

These are wire/state tests of an authored mock, not evidence of native engine rendering parity or cache performance.

Design references are credited in [Credits](../../CREDITS.md#harness).
Implementation is first-party Rust using only the standard library; no competitor
code or tests were copied.
