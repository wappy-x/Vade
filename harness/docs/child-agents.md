# Child agents

Child agents are helpers that the native agent can start for side tasks. Each
child works on its own task and reports what it found to the agent that started
it, called the *parent*.

Children are off by default. To turn them on, do one of these:

- add `--children` to an `agent` command
- type `/children on` before the first prompt of a fresh native conversation

Other modes refuse the flag.

By default a child can only read. Your launch flags can grant more: a named
scope can give a child exact file edits or sandboxed commands, as described
under [Tool scopes and write ownership](#tool-scopes-and-write-ownership).

One term comes up all through this page. An *activation* is one run of the
parent agent, from your prompt to its answer.

## Commands

| Native command | What it does |
| --- | --- |
| `/children` or `/children list` | List IDs, states, elapsed time and reservations for the current or last activation |
| `/children on` / `/children off` | Turn children on or off before the first prompt; once the conversation has history, use `/new` first |
| `/children cancel ID` | Cancel just that active child; its siblings and the parent keep going |
| `/children cancel all` | Cancel active children and close new spawns for this activation; the parent keeps going |
| `/context` | Show parent context, plus each child's last usage and cumulative reported cached tokens |

## How children run

The parent gets four tools: `child_spawn`, `child_list`, `child_wait` and
`child_cancel`.

Each child runs the same agent loop as the parent. It has its own task ID, its
own conversation, its own limits and a final result. It uses the parent's
engine, model, workspace, project rules and skills. It also runs the same tool
code, with fewer tools by default. No crates or second tool implementation are
added.

A child cannot write into the parent's conversation. Instead, the parent
records each finding against its own tool call and decides what to do with it.

### The child panel

While children run, a small panel above your prompt lists each one with its
state and elapsed time. The panel is sized to fit your terminal. Commands keep
working even while the parent is waiting on the engine.

The panel reports only what really happened. A child is shown as finished only
once its worker has really stopped, and asking to cancel a child never invents
an outcome for it.

`/new` keeps your on/off setting but clears the panel and results. Each parent
activation gets a fresh scheduler, the in-memory part that admits and tracks
children. Only the current or last activation's child results are kept in
memory. The parent's [journal](agent-tools.md#journal-recovery) keeps its own
record of each spawn and wait.

## Tool scopes and write ownership

A child without a named scope only gets these read tools: `workspace_read`,
`workspace_list`, `tool_output_read`, `skill_read`, `workspace_manifest` and
`workspace_symbols`.

No child can call `child_*` tools, even though its frozen schemas match the
parent's. (*Frozen* means fixed for the whole session; see [Shared prompt
prefix](#shared-prompt-prefix).) Project rules, skill text and model arguments
cannot widen the executor policy.

### Grant edits or commands with a named scope

Named scopes come only from trusted host configuration. The CLI provides them
through explicit launch flags, and both flags need `--children`:

```sh
# Usage example for a separately supplied mock; does not start or contact it here.
vade-harness agent --endpoint http://127.0.0.1:18080 --model mock --workspace . \
  --children --child-edit edits=src/example.rs --child-edit edits=src/renamed.rs
# Shell is a separate explicit host capability; approval still applies.
vade-harness agent --endpoint http://127.0.0.1:18080 --model mock --workspace . \
  --children --child-shell checks
```

The model picks a scope by name in `child_spawn`, for example
`{"task":"...","grant":"edits"}`. Leaving out `grant` always selects the
read-only scope. Scope names and their exact files are shown in the frozen
parent system prefix.

- At most 32 grants, and 32 file paths per grant, are accepted.
- A scope can combine edit and shell flags, but a file-edit scope never grants
  shell on its own.
- `--approval auto` does not grant any tool capability by itself.

### Leases: who may write which file

A **lease** is a reservation: while a child holds a lease on a file, nobody
else (the parent or another child) may write it. Before a child's patch runs,
the harness lists every file the patch would add, change, delete or move to,
and each one must be in that child's lease. The usual patch checks (syntax,
directory handles, links, preconditions and per-file atomic updates) still
apply.

- Lease paths must be plain relative ASCII paths, not hidden, credential,
  absolute, `..`, `.`, backslash or non-ASCII. Two leases that overlap,
  including by letter case on macOS, are refused.
- While any child holds a write lease, the parent's own patches must also use
  exact ASCII paths, so it cannot get around a lease with a different spelling.
  With no child write leases, the parent's ordinary paths work as usual.
- A shell scope (`--child-shell`) takes the whole workspace, because a command
  can write anywhere in it, whatever its argv. The parent and other writing
  children wait until that child finishes **and** its result is collected;
  read-only children keep going.
- Patches inside a shell scope still need explicit file paths. Shell keeps the
  usual sandbox, no network, exclusions, process, output and deadline limits,
  and your chosen toolchain, and each child gets its own private build folder,
  never the parent's.

All file leases and shell workspace leases are reserved before submission. They
are held through completion, until the results are explicitly collected. Before
calling its usual executor, the native parent wrapper checks every affected
path, move destinations included, and shell ownership.

These are in-process ownership rules, not new OS isolation or atomic multi-file
transactions. Host code outside this wrapper and unrelated editors are not
coordinated. Separate files can still conflict in meaning, so the parent
reviews and folds in the results.

### Approvals

Child edits and commands still go through the parent's approval policy
(`--approval`) and preview:

| `--approval` | Child edits and commands |
| --- | --- |
| `never` | Denied. |
| `ask` | Edits need the terminal's exact one-time approval. Commands that an allow-list rule matches run after preview; every other command needs approval (see [shell safety](../../docs/shell-safety.md)). Blocked commands are refused. |
| `auto` | Validated exact patches are applied, and commands that are not blocked run inside the same macOS sandbox, after preview. Blocked commands are refused. It never grants broader filesystem or network access. |

Noninteractive runs deny operations that need approval, unless an allow
list approves the command or prompts are off (`--yes`).

Parent and child previews wait in one cancellable queue and are shown one at a
time, with a single combined 1 MiB display budget. That way concurrent requests
cannot replace another operation's pending approval. Child previews show the
task ID. `/children cancel ID|all` stays available during an approval and
clears cancelled requests, and a stale answer cannot approve later work.

### How tool outcomes are recorded

Tool execution records `Started`, `Completed`, `Uncertain`, denial,
cancellation, failure or output-limit status. A result describes what
happened, and is never replaced with a false claim that execution was denied.
Completed receipts survive a cancellation that races with the return.

Partial native patch results and unsuccessful commands set `Uncertain`, even
when their detailed receipt comes back successfully, and `child_wait` then
reports `uncertain_writes: true`. A write that started but then panicked,
failed or was canceled is also marked uncertain, because an error does not
prove that no edit happened.

The scheduler never replays such operations automatically or rolls back files.
The native agent's transactional receipts and durable journal remain the
authority on actual edits.

## Budgets and cancellation

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

The byte counts in the table are budgets for serialized data, not claims about
exact process RSS or Rust AST overhead. A few more limits apply:

- Task prose is capped at 16 KiB.
- Summaries are capped at 4 KiB, cut on a UTF-8 boundary, with an explicit
  truncation flag. Full transcripts and tool receipts are kept in each child's
  journal.
- Each child also keeps the ordinary runtime's response, argument, event and
  JSON-complexity bounds.

### When the budget is used up

There is no hidden task queue. When the concurrent set is full, the call
returns `Capacity`.

Global allocations are reserved before spawning. They are never topped up
during that activation, even after collection, so they bound the total work
and not just the work active right now. A new explicit parent activation
creates a new scheduler with fresh finite allocations.

Model tool arguments can lower the configured default turn, token and wall
limits. They cannot raise them or change the global limits.

### How cancellation works

While it owns workers, a monitor watches the parent's shared cancellation flag
and deadlines, polling every 10 ms (subject to OS scheduling). It stops further
admissions and signals the affected children.

- Cancelling one child leaves its siblings running.
- Running workers stay in `Stopping` until their threads have actually returned
  and been joined.
- A request and an outcome are separate fields. A completion that races with a
  stop request is still a completion, and a failed operation is not relabelled
  as a successful interruption.
- HTTP cancellation means local cleanup only. This transport contract has no
  acknowledgment that remote compute was cancelled.

When the parent finishes, fails, is cancelled or panics, any children still
running are cancelled and waited for. A running tool is never left behind with
a claim that cleanup succeeded.

Cancellation has limits of its own:

- Deadlines are cooperative cancellation deadlines, not OS process termination.
- As in the existing transport, standard-library DNS and a connect already in
  progress cannot be interrupted.
- A host callback that never returns cannot be preempted by standard-library
  threads, and keeps shutdown waiting.

This module has no unbounded subprocess or shell implementation.

## Journals, crashes and resume

Native children also have their own private journals:

- the initial task is recorded before generation
- intent is synced before every tool
- results are synced before the next request

`child_wait` includes the child journal name, and `/sessions` lists these
`child-...` journals. Journals keep the existing private permissions, size
limits and artifact quotas, and still refuse redirects. Each child owns a
separate build area, which is thrown away with worker cleanup.

### After a crash

The in-memory scheduler is not restored after a crash. Pending approvals, live
tasks, leases and collection state are never rebuilt from a transcript or
replayed automatically.

If a crash prevents result collection, inspect the child journals from
`/sessions`. An unknown tool outcome still means you must reconcile the
workspace.

The parent records completed spawn and wait receipts. There is no atomic
journal transaction across parent and children, though, and no complete
durable log of scheduler events. Task IDs only exist within the process, and
child journal names are the durable references to transcripts.

### Resuming a session

To resume a session that used children, pass the same `--children`,
`--child-edit` and `--child-shell` flags again; a mismatch is refused.
Approvals are never saved, so the approval policy is the `--approval` you pass
at that launch. Saved scope descriptions never authorize tools.

## Shared prompt prefix

Every child starts with an exact copy of the parent's frozen system and
developer messages (project rules included) and its complete tool schemas. On
top of that it gets a fresh task and scope message, without the parent's chat
history.

The schemas stay identical even when a child is allowed less; execution masks
narrow the actual authority. That way the shared prefix the model sees does not
shift whenever permissions differ. The permission envelope is explanatory data,
and enforcement lives in the executor.

### Cache keys

Child cache affinity is not inherited from the parent's request options. Each
child has its own engine client, which generates a distinct scheduling key only
when the engine advertises `prompt_cache_key: scheduling_hint`. Repeated turns
of one child keep its key and its append-only serialized history.

Tests compare the parent's and child's system and tools, and successive child
wire prefixes. Here is what that does and does not show:

- Cache-key, output-limit and stream options may differ between parent and
  child requests, so whole-HTTP-body prefix equality is not asserted across
  those separate sessions.
- Real rendered token-prefix equality and KV sharing belong to the engine and
  have not been tested.
- A key or byte equality is not evidence of a cache hit.

Parent cache, checkpoint and continuation handles are never forwarded. Unknown
provider root options fail closed when a stable child prefix is created, rather
than accidentally sharing state. `kv_branch_handles` is unused. Cache-domain
changes follow the shared contract behavior; there is no invented KV fork and
no cache hit claim. All inference, admission capacity and KV ownership stay in
the engine.

### Responses

Children ask for bounded JSON responses (`stream: false`), because their UI
shows lifecycle states rather than token deltas. The shared typed response
parser validates the whole response before any tool can run. No second agent
loop or inference implementation is introduced.

## Why children need exact token counts

Each child has a hard token budget (32,768 by default), so the harness needs
the engine to count tokens exactly. Before every child request, the harness
asks the engine how many prompt tokens the request uses. This step is called
exact admission. The harness charges that count, plus the full output
allowance, to the child's budget. Then it sends exactly the bytes that were
counted.

The charge is strict on purpose, which keeps the budget predictable:

- Repeated context is charged every time, even if the engine serves it from
  cache.
- Nothing is refunded for short answers, errors or cancels.

If the engine cannot count exactly, children do not run. The parent can still
fall back to ordinary chat.

### Reported usage

Reported usage stays separate from reservations.

- Missing usage is unknown. A reported zero stays zero, while a value that was
  never reported stays unknown.
- Contradictory prompt or output telemetry rejects the response before any tool
  runs.
- The child panel shows usage only after a validated response and a successful
  exact reservation.
- `/context` labels the last response and the sum across completed responses
  with cached-token telemetry, and partial coverage is stated.
- Cancelled requests without a completed response add no invented cache counts.

The provider is responsible for honoring its exact count and output cap.
Without an upstream tokenizer, the harness cannot independently police
dishonest token telemetry. The client still keeps its own byte, turn and wall
bounds.

## Fit with the rest of the harness, and gaps

Children work alongside the rest of the harness. Native journals, budgets,
compaction, frozen rules and skills, transcript forks, build sessions and
toolchain selection, setup/doctor, workload and gateway routing all stay.

- Child schemas stay frozen with the parent.
- Turning children on or off before a first prompt creates a fresh journal.
- `/new` keeps the launch scopes, and resume refuses mismatched scopes instead
  of quietly restoring grants.
- Session switches and forks clear the old child panel after cleanup.
- Compaction keeps completed metadata but starts no child work.

Known gaps:

- Full crash recovery of scheduler lifecycle and collection is future work. The
  separate child journals keep tool intents and results without resuming tasks
  automatically.
- Journal limits are per session, not a global disk quota.
- Shell ownership is deliberately exclusive for writes across the whole
  workspace, rather than a guess at affected paths based on argv.
- No live engine integration, KV sharing, recursive children or automatic
  merging of child results is implemented.

## Try it against mocks

```sh
cargo run --offline --example child_agents_mock
cargo test --offline child_agents
cargo test --offline --test contract_mock children_contract
```

The example uses a first-party mock parent and three mock children. It shows
running, completed, deadline cancellation, token quota and collected outcomes.
It opens no files or sockets. HTTP acceptance tests bind their own loopback
mock listeners. No engine, Spark, GPU, live model, Codex or Hermes is
contacted.

Real native CLI/TUI acceptance uses the existing v1 capability fixture and
loopback contract mock helpers. The parent spawns three children, and one is
cancelled. Two findings come back through the original tool IDs, and the
parent folds them in. One child does a real bounded read in a temporary
fixture workspace. Separate checks cover cancelling one or all children from
the UI, parent cancellation, resize and disabled tools.

## Evidence and credits

Tests cover:

- real native parent spawn, wait and integration
- exact shared rules and tools
- distinct keys gated on capability
- exact admitted versus dispatched bytes
- missing admission and contradictory usage
- simultaneous completion and resource exhaustion
- separate and overlapping scopes, and parent write exclusion
- refusal of recursive or unknown grants
- cancellation in HTTP and during a write
- completed versus uncertain tool outcomes
- lifecycle display bounds and setup failure
- teardown on normal return, error, panic and drop

So far these tests have run on macOS only.

Native CLI/TUI tests also prove:

- explicit opt-in
- denial of a spawn nobody asked for when children are off
- no child generation when exact admission is missing
- independent parent and sibling cache keys
- reported versus unknown child usage
- controls while parent I/O is blocked
- cancel-all closing admission
- cleanup before `/new`
- bounded raw-terminal resize behavior

Integration tests also exercise:

- the real native edit and shell executors
- all patch paths and move destinations
- exclusive shell leases
- failed journal initialization
- parent observer refusal
- real terminal approval and denial and cancellation with two concurrent
  children
- private parent/child transcript replay without running tools again

Design references are credited in [Credits](../../docs/CREDITS.md#harness). The
implementation is first-party Rust using only the standard library; no code
or tests from competing projects were copied.

## For library hosts

Skip this if you only use the CLI.

- The `child_agents` library module runs each child through the same
  `agent::run` loop the parent uses, with the same contract/HTTP client and the
  native `CodingTools` executor.
- The CLI registers every native tool with `ToolPolicy::native()` and binds
  `native_http_factory`. The original `read_only_http_factory` is still
  available for hosts that only need read and list.
- Library hosts must pair `ToolPolicy::native()` with `native_http_factory` and
  supply the parent's real approval callback in `NativeToolHost`.
- Call `run_parent_observed` when using a durable parent journal. It keeps all
  parent journal hooks, including intent and result records around scheduler
  tools.
- Full typed transcripts and tool receipts come back in `ParentRun.children`
  for inspection. Custom or mock factories may leave out the journal observer.
- `run_parent` cancels and joins outstanding children when the parent
  completes, fails, is cancelled or unwinds through a panic, and
  `Scheduler::drop` also cancels and joins.
- `StablePrefix::from_parent` builds the shared child prefix, and each
  `HttpChildModel` has its own `Contract`. The live `ChildView` publishes child
  usage.
- `apply_patch` and `workspace_shell` each have their own effect
  classification (the record of what a tool can change). The native registry
  refuses to reclassify either one as read-only.
- Custom factories are still trusted host code and must supply honest effect
  classifications, observers and executors that enforce the boundary; the
  scheduler is not a plugin sandbox.
