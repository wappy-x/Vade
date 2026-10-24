# Bounded native children

The new `child_agents` library module runs independent children through the
existing `agent::run` loop. A native parent can call `child_spawn`, `child_list`,
`child_wait`, and `child_cancel`. Each accepted child has a scheduler-scoped task
ID, its own typed conversation, a finite resource allocation and a terminal
result. No child receives a mutable parent conversation or appends background
messages to it. The parent runtime alone commits bounded tool receipts against
its original call IDs; the parent decides how to integrate a finding.

The library is wired into the real native CLI/TUI behind explicit opt-in. Add
`--children` to an otherwise configured `agent` command, or enter `/children on`
before the first prompt of a fresh native conversation. It is off by default and
the flag is refused in other modes. Children use the parent's selected endpoint,
model, frozen project rules/catalog and workspace through the existing
contract/HTTP client and native `CodingTools` executor. No crates or second tool
implementation are added. Read-only is the default; host launch options can
explicitly grant native patches or sandboxed commands as described below.

| Native command | Effect |
| --- | --- |
| `/children` or `/children list` | List current/last activation's IDs, states, elapsed time and reservations |
| `/children on` / `/children off` | Change opt-in before the first prompt; after history exists, use `/new` first |
| `/children cancel ID` | Cancel just that active child; siblings and parent continue |
| `/children cancel all` | Cancel active children and close new spawns for this activation; parent continues |
| `/context` | Parent context plus per-child last usage and cumulative reported cached tokens |

The raw TUI draws a bounded child panel above the editable prompt, adapting to
terminal width and height. Lifecycle/usage revisions wake the existing UI thread;
ordinary elapsed-time ticks update durations. There is no child-token render
queue or extra UI watcher thread. Commands remain available while parent HTTP
is stalled. Finished states are published only after workers join; a cancellation
request does not fabricate a terminal outcome or remote acknowledgment.

`/new` preserves the opt-in setting but clears the child display/results after
cleanup. Every parent activation owns a new bounded scheduler. The UI retains
only the current/last activation's full in-memory results and metadata; older
parent tool receipts remain in its typed journal. `run_parent_observed` preserves
all parent journal hooks, including intent/result records around scheduler tools.
Native children also have separate private journals: the initial task is recorded
before generation, intent is synced before every tool, and results are synced
before the next request. `child_wait` includes the child journal name; `/sessions`
lists these `child-...` journals. Journals retain the existing private permissions,
size limits, artifact quotas, and refusal of redirects. Each child owns a separate
build area, discarded with worker cleanup.

The in-memory scheduler is not crash-restored. Pending approvals, live tasks,
leases and collection state are never recreated from a transcript or automatically
replayed. If a crash prevents result collection, inspect the child journals from
`/sessions`; an unknown tool outcome still requires workspace reconciliation.
The parent records completed spawn/wait receipts, but there is no atomic journal
transaction across parent and children or complete durable scheduler event log.
Task IDs are process-local; child journal names supply durable transcript references.
Resume validates the same explicit host child flags/scopes and requires a new
launch approval policy. Saved scope descriptions never authorize tools.

```sh
cargo run --offline --example child_agents_mock
cargo test --offline child_agents
cargo test --offline --test contract_mock children_contract
```

The example uses a first-party mock parent and three mock children. It displays
running, completed, deadline cancellation, token quota, and collected outcomes.
It opens no files or sockets. HTTP acceptance tests bind their own loopback mock
listeners. No Engine, Spark, GPU, live model, Codex, or Hermes is contacted.
Actual native CLI/TUI acceptance uses that existing v1 capability fixture and
loopback contract mock helpers: the parent spawns three children, one is cancelled,
two findings return through original tool IDs, and the parent integrates them.
One child performs a real bounded read in a temporary fixture workspace. Separate
checks cover UI cancel-one/all, parent cancellation, resize and disabled tools.

## Resource and lifecycle contract

| Resource | Default per child | Default whole parent activation |
| --- | --- | --- |
| Concurrent children | — | 4 (hard ceiling 8) |
| Accepted tasks | — | 16 (hard ceiling 32) |
| Model turns | 8 | 64 allocated |
| Tool invocations/proposals | 16 | 128 allocated |
| Exact input + reserved output tokens | 32,768 | 262,144 allocated |
| Output reservation per model turn | 1,024 | Included above |
| Wall lifetime | 120 seconds | 300 seconds |
| Serialized history | 512 KiB | 8 MiB allocated |
| Tool result bytes | 128 KiB total; at most 64 KiB each | 1 MiB allocated |

Task prose is capped at 16 KiB. Summaries are capped at 4 KiB on a UTF-8
boundary with an explicit truncation flag. Full typed transcripts and tool
receipts are returned in `ParentRun.children` for inspection. The native host
journals them as described above; custom/mock factories may omit the observer.
Each child also retains the ordinary runtime's response, argument,
event and JSON-complexity bounds. The table's byte counts are serialized data
budgets, not claims about exact process RSS or Rust AST overhead.

There is no hidden task queue. A full concurrent set returns `Capacity`. Global
allocations are reserved before spawning and are never replenished during that
activation, even after collection; this bounds the total work, not just currently
active work. A new explicit parent activation creates a new scheduler and fresh
finite allocations. Model tool arguments can lower the configured default
turn/token/wall limits; they cannot increase them or change global limits.

A monitor observes the parent's shared cancellation flag and deadlines on a
10 ms polling interval while it owns workers (subject to OS scheduling). It stops further admissions and
signals affected children. Individual cancellation leaves siblings running.
Running workers remain in `Stopping` until their threads have actually returned
and been joined. A request and an outcome are separate fields. A completion that
races with a stop request remains a completion; a failed operation is not relabeled
as a successful interruption. HTTP cancellation means local cleanup only. There
is no remote compute cancellation acknowledgment in this transport contract.

`run_parent` cancels and joins outstanding children on parent completion,
failure, cancellation, and unwinding through a panic. `Scheduler::drop` also
cancels and joins. It never detaches a running tool and claims cleanup succeeded.
Deadlines are cooperative cancellation deadlines, not OS process termination.
As in the existing transport, std DNS and an in-progress connect are not
interruptible. An arbitrary host callback that never returns cannot be
preempted by std threads and keeps shutdown pending. There is no unbounded
subprocess or shell implementation in this module.

## Tool scopes and write ownership

The CLI registers every native tool with `ToolPolicy::native()` and binds
`native_http_factory`. Unnamed children receive only `workspace_read`,
`workspace_list`, `tool_output_read`, `skill_read`, `workspace_manifest`, and
`workspace_symbols`. The original `read_only_http_factory` remains available for
hosts that only need read/list. No child can call `child_*` tools, even though
its frozen schemas match the parent. Project rules, skill text and model arguments
cannot widen the executor policy.

Named scopes come only from trusted host configuration. The CLI supplies these
through explicit launch flags (both require `--children`):

```sh
# Usage example for a separately supplied mock; does not start or contact it here.
vade-harness agent --endpoint http://127.0.0.1:18080 --model mock --workspace . \
  --children --child-edit edits=src/example.rs --child-edit edits=src/renamed.rs
# Shell is a separate explicit host capability; approval still applies.
vade-harness agent --endpoint http://127.0.0.1:18080 --model mock --workspace . \
  --children --child-shell checks
```

A model chooses `{"task":"...","grant":"edits"}` in `child_spawn`; omitting `grant`
always selects the read-only scope. Names and exact files are shown in the frozen
parent system prefix. At most 32 grants and 32 file paths per grant are accepted.
A scope can combine edit and shell flags, but a file-edit scope never implicitly
grants shell. `--approval auto` does not grant any tool capability by itself.

`apply_patch` has a distinct multi-path effect classification. Before dispatch,
the scheduler extracts every add/update/delete source and move destination from
the bounded native patch grammar. Every path must be an exact member of the
child's file lease; the native executor still validates full patch syntax,
directory handles, links, preconditions and per-file atomic updates. Absolute,
parent, dot, hidden, credential, backslash and non-ASCII lease paths are refused.
ASCII case-folding and ancestor checks reject overlapping leases on macOS. Parent
patches also use strict canonical ASCII paths while a child write lease exists,
so parent spelling aliases cannot bypass ownership. Without child write leases,
the native parent's ordinary path grammar remains available.

`workspace_shell` has a distinct workspace-wide effect classification. It is
**never read-only or a single-file edit**, regardless of argv. A host shell scope
reserves an exclusive workspace write lease, because the native sandbox permits
visible workspace writes. Parent commands and every other write scope are blocked
until this child finishes **and** its result is collected. Read-only children may
continue. Patch tools still require explicit file paths even inside a shell scope.
Shell commands retain the existing macOS sandbox, no-network policy, scan
exclusions, process/output/deadline bounds and owner-selected read-only toolchain.
Each child gets its own private build/cache/temp area; it never adopts the parent's.

Child edits and commands still pass through the parent's `ApprovalPolicy` and
host preview callback. `never` denies them; `ask` needs the terminal's exact
one-time approval for edits and commands outside the
[native read-only allowlist](agent-tools.md#commands-and-approvals). Allowlisted
commands run after preview without asking in both `ask` and `auto`; `auto` also
applies validated exact patches and runs all other validated commands inside the
same macOS sandbox after preview. It never grants broader filesystem or network
access. Noninteractive runs deny operations
that need approval. Parent and child previews share a cancellable
serialized lane with one aggregate 1 MiB display budget, so concurrent requests
cannot replace another operation's pending approval. Child previews identify the
task ID. `/children cancel ID|all` remains available during an approval and clears
cancelled requests; a stale answer cannot approve later work.

All file leases and shell workspace leases are reserved before submission and
held through completion until explicit collection. The native parent wrapper
checks all affected paths, including move destinations, and shell ownership before
calling its usual executor. Host code outside this wrapper and unrelated editors
are not coordinated; these are in-process ownership rules, not new OS isolation
or atomic multi-file transactions. Disjoint files can still conflict semantically.
The parent reviews and integrates results.

Tool execution records `Started`, `Completed`, `Uncertain`, denial, cancellation,
failure or output-limit status. Partial native patch results and unsuccessful
commands set `Uncertain` even when their detailed receipt is returned successfully;
`child_wait` then reports `uncertain_writes: true`. The result describes what
happened; it is never replaced with a false claim that execution was denied. Completed receipts survive cancellation that races with
return. A write that was started but panicked, failed, or was canceled is marked
uncertain, because an error does not prove no edit happened. The scheduler never
automatically replays such operations or rolls back files. The agent lane's
transactional receipts and durable journal remain authoritative for actual edits.

## Prompt and Engine boundary

`StablePrefix::from_parent` copies the parent's frozen system/developer messages
(including project rules) and complete tool schemas exactly. Every child starts
with that prefix plus a fresh task/scope user message, without the parent's chat
history. Schemas remain identical while execution masks narrow actual authority.
This avoids shifting the shared model-visible prefix whenever permissions differ.
The permission envelope is explanatory data; enforcement lives in the executor.

Child-specific cache affinity is not inherited from parent request options. Each
`HttpChildModel` has its own `Contract`, which generates a distinct scheduling
key only when `prompt_cache_key: scheduling_hint` is advertised. Repeated turns
of one child retain its key and append-only serialized history. Tests compare
parent/child system and tools and successive child wire prefixes. Cache-key,
output-limit and stream options may differ between parent and child requests;
whole-HTTP-body prefix equality is not asserted across those distinct sessions.
Actual rendered token-prefix equality and KV sharing belong to the Engine and
have not been tested. A key or byte equality is not evidence of a cache hit.

Children request bounded JSON responses (`stream: false`), because their UI
displays lifecycle states rather than token deltas. The shared typed response
parser validates the entire response before tools can run. No second agent loop
or inference implementation is introduced.

The native child transport requires negotiated **exact admission**. It counts
the exact fully decorated request bytes, reserves exact prompt tokens plus the
full output cap from that admission, checks the child's remaining allocation,
then sends those same bytes. Repeated prefix tokens count on every request,
including cached input. Reservations are not refunded on short outputs, errors
or cancellation, making exhaustion conservative and predictable. Missing exact
admission stops the child without generation; no byte estimate is presented as
a hard token quota. The parent transport can retain its ordinary fallback.

Reported usage remains separate from reservations. Missing usage is unknown.
Contradictory prompt/output telemetry rejects the response before tool execution.
The live `ChildView` publishes usage only after a validated response and exact
reservation succeed. `/context` labels the last response and the sum across
completed responses with cached-token telemetry; partial coverage is explicit.
A reported zero stays zero, while no reported value stays unknown. Cancelled
requests without a completed response do not contribute invented cache counts.
The provider is responsible for honoring its exact count and output cap; without
an upstream tokenizer the harness cannot independently enforce dishonest token
telemetry. The client independently retains byte, turn and wall bounds.

Parent cache/checkpoint/continuation handles are never forwarded. Unknown
provider root options fail closed when creating a stable child prefix, rather
than accidentally sharing state. `kv_branch_handles` is unused. Cache-domain
changes use the shared contract behavior; there is no invented KV fork or cache
hit claim. All inference, admission capacity and KV ownership stay in the Engine.

## Integration contract and remaining boundaries

The main merge retains native journals, budgets, compaction, frozen rules/skills,
transcript forks, build sessions/toolchain selection, setup/doctor, workload and
gateway routing. Child schemas remain frozen with the parent. Enabling/disabling
children before a first prompt creates a fresh journal; `/new` preserves launch
scopes, and resume refuses mismatched scopes instead of silently restoring grants.
Session switches/forks clear the old child panel after cleanup. Compaction keeps
completed metadata but starts no child work.

Library hosts must pair `ToolPolicy::native()` with `native_http_factory` and
supply the parent's real approval callback in `NativeToolHost`. Call
`run_parent_observed` when using a durable parent journal. Custom factories remain
trusted host code and must supply honest effect classifications, observers and
boundary-enforcing executors; the scheduler is not a plugin sandbox. The native
registry refuses reclassifying `apply_patch` or `workspace_shell` as read-only.

Full crash recovery of scheduler lifecycle/collection remains future work; the
separate child journals preserve tool intents/results without automatically
resuming tasks. Journal limits are per session, not a global disk quota. Shell
ownership is deliberately exclusive for writes across the whole workspace, not
an argv-dependent guess at affected paths. No live Engine integration, KV sharing,
recursive children or automatic merge is implemented. The coordinator owns
integration and any push; this builder does not merge into main or push.

## Evidence and credits

Tests cover actual native parent spawn/wait/integration; exact shared rules/tools;
distinct capability-gated keys; exact admitted versus dispatched bytes; missing
admission; contradictory usage; simultaneous completion; resource exhaustion;
disjoint and overlapping scopes; parent write exclusion; recursive/unknown grant
refusal; cancellation in HTTP and during a write; completed versus uncertain
tool outcomes; lifecycle display bounds; setup failure; and teardown on normal
return, error, panic and drop. Tests run on macOS only in this round.
Native CLI/TUI tests also prove explicit opt-in, denial of unsolicited spawn
when disabled, missing exact admission without child generation, independent
parent/sibling cache keys, reported versus unknown child usage, controls while
parent I/O is blocked, cancel-all closing admission, cleanup before `/new`, and
bounded raw-terminal resize behavior. Integration tests also exercise the real
native edit/shell executors, all patch paths and move destinations, exclusive shell
leases, failed journal initialization, parent observer refusal, actual terminal
approval/denial and cancellation with two concurrent children, and private
parent/child transcript replay without re-executing tools.

Design references are credited in [Credits](../../CREDITS.md#harness).
Implementation is first-party Rust using only the standard library; no competitor
code or tests were copied.
