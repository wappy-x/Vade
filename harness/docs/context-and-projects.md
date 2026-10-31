# Context, project rules and branches

This page is for developers who run the harness's native agent (its built-in
coding agent) on a real project. It covers four things:

- which project rules and skills a session loads;
- how the agent checks that each request fits the model's context window;
- how compaction shrinks older turns when space runs out;
- how forks, branches and resume work.

The context window is the most tokens the model can consider at once. When the
engine supports exact counting (`/v1/admission`), the harness asks the engine to
count each request with its own tokenizer. Otherwise the harness uses a local
estimate, and the screen says so.

The harness never runs the model, and it never holds the model's attention cache
(the [KV cache](../../docs/GLOSSARY.md#ho)). Both stay in the engine. Everything
here is written in Rust using only the standard library, and it is tested
against local mocks.

## Project rules and skills

### Rules files (`AGENTS.md`)

The root `AGENTS.md` goes into the system prompt with the other [project
instructions](project-instructions.md), and Vade reads it again when it changes.
To add rules from deeper folders, pass `--rules-scope PATH`, once per folder.
Each one adds that directory's chain of ancestors, from the first folder below
the root down.

- You can select up to eight paths, each with at most eight components.
- The harness does not discover other subtrees recursively.
- Only the scopes you select are active, so pick the relevant subdirectories
  before you start work.

Root rules apply to the whole workspace. Deeper rules take precedence, but only
within their own subtree. Explicit user requests and the executor/system policy
stay above project rules.

Rules are ordered by depth, then by relative path. Each file is capped at
16 KiB, with 16 files and 24 KiB in total. A selection that is incomplete or too
large fails visibly.

### The skills catalog

A skill is a `SKILL.md` file that the agent reads only when it needs it. You
list skills in an optional, **visible** `skills/catalog.json`, and the format is
deliberately small:

```json
{
  "skills": [
    {
      "name": "review-rust",
      "description": "Review a Rust change using the project checklist",
      "path": "skills/review-rust/SKILL.md"
    }
  ]
}
```

The catalog follows a few strict rules:

- It accepts only these fields, with up to 16 entries and 8 KiB.
- Names have 1–48 ASCII letters, digits, hyphens or underscores.
- Descriptions have at most 240 UTF-8 bytes.
- Paths must stay under the visible `skills/` folder and end in `/SKILL.md`.
- Duplicate names, unsafe paths, script or installer fields, and oversized
  bodies are refused.
- The catalog is always ordered by name.

The harness checks each entry's metadata when the session is created. It does
**not read** a skill's body until `skill_read` or a local `/skill NAME` asks for
it. Bodies are capped at 16 KiB, and they only need to be UTF-8 when they are
loaded.

### What the session sees, and when it refreshes

Rules and the catalog are snapshotted in a single developer message. Each item
is attributed by its workspace-relative path and the selected scopes. The
message is at most 48 KiB serialized, and it counts toward full-request
[admission](#how-the-agent-checks-that-a-request-fits).

- `/project` explains why each item was included.
- `/project-rules` shows the frozen text.
- Resume does not reread the frozen scope rules or the catalog. The [project
  instructions](project-instructions.md) section, with the root `AGENTS.md`, is
  read again before each turn.
- `/new` deliberately refreshes them, and starts a new prefix (the fixed start
  of every request) and a new session.

### How skill reads are guarded

A skill's recorded local revision is its device/inode, byte size and
modification/change timestamps. This guards against stale reads. It is **not a
content hash or security identity**.

Opening and reading a skill use the existing anchored workspace boundary. Links,
special files, and credential and hidden paths are refused. The harness compares
the opened file's revision before and after the bounded read. If it differs from
the frozen catalog, loading is refused with an instruction to refresh.

The exact loaded body bytes, path and revision become a normal tool result (or a
retrievable artifact), and they are journaled. As with editing, a hostile writer
changing the filesystem at the same time is outside the guarantee.

No body, rule, catalog entry or included script changes the approval or sandbox
policy. There is no automatic execution, installer, plugin or dependency loading
here.

### Listing files and finding declarations

Two on-demand views help the agent find its way around a project:

| View | What it gives |
| --- | --- |
| `workspace_manifest` / `/map` | A file list of at most 256 entries and 32 directories, depth 0–3. It leaves out hidden and credential paths, links, `target`, `node_modules` and `vendor`. An incomplete list says so. |
| `workspace_symbols` / `/symbols` | Up to 64 lines that look like declarations, from a single UTF-8 file of at most 64 KiB, with 240 bytes per hint. |

The symbols view is lexical string matching, so comments, strings and language
syntax can cause false positives or misses. Neither view is semantic indexing, a
file-revision guarantee, or an automatic prefix update.

## How the agent checks that a request fits

Before a request is sent, the agent runs **admission**: the check that the whole
request fits the context window. Admission counts:

- every serialized option and message field;
- the tool schemas and the request syntax;
- a framing allowance and a safety margin;
- the output it reserves for the answer.

Admission is deliberately strict. Nothing is subtracted for cached input,
cumulative usage counters are not trusted, and nothing is retried implicitly.
Known text parts and their extensions stay exact. Unsupported media and unknown
content types fail admission, because byte length cannot estimate the rendered
cost of media.

When the engine advertises exact admission, that count replaces the local
heuristic before the request goes to the model, and it uses the engine's actual
output reservation. If the tokenizer or the HTTP call fails, the request is
stopped instead of quietly downgrading to an estimate. The independent
byte/structure limits and tool-receipt safety limits still apply. (A tool
receipt is the recorded call and outcome of one tool run.)

### Large tool output

Large tool output has its own limits:

- **Size.** Each capture holds at most 64 KiB, and a session keeps at most 128
  captures and 4 MiB in total.
- **Excerpts.** A large result reaches the model as a labelled, deterministic
  excerpt. `tool_output_read` fetches UTF-8 byte ranges of the full capture by
  session-local ID, and never runs the tool again.
- **Lost bytes.** Bytes that were never captured (for example past a command's
  output cap) are not available.
- **Expiry.** Range retrieval expires after 24 hours. Raw journal records stay
  under the 64 MiB / 8,192-record quota until you remove them.
- **Order of checks.** Capacity is checked before any side effect, and source
  limits still apply before artifact storage.

## Compaction: shrinking older turns

Compaction replaces older parts of the conversation with something shorter, so
the next request fits. You can run it by hand, or let it run
[automatically](#automatic-compaction-during-a-task) when the window fills up.

### Compacting by hand

| Command | What it does |
| --- | --- |
| `/compact-preview KEEP SUMMARY` | Shows the plan and changes nothing. |
| `/compact KEEP SUMMARY` | Applies the plan while the agent is idle. |
| `/compact` on its own | The model summarizes older turns; the automatic planner is the fallback. |

`KEEP` is how many of the newest user turns to keep, and `SUMMARY` is your own
text. No model writes the summary, and there are no retries or scripts. Your
summary is stored as a user message, never as a system instruction.

A compaction must make the request smaller in both estimated tokens and bytes,
and the result must still fit. The offline `demo-compaction` command reports
only diagnostic hashes and explicit estimates.

### What a compaction with your summary keeps

With a summary you supply, the planner keeps:

- **all user messages**;
- all system and developer messages;
- every assistant tool call and its matching result;
- the latest assistant answer;
- the newest KEEP user turns.

It removes only older standalone assistant messages. This careful policy keeps
the original constraints, unresolved user questions, tool receipts and any file
revisions recorded in them.

That puts one job on you. Your summary must keep the important decisions from
the assistant prose that gets removed, because structure tests cannot judge
semantic quality.

Compaction refuses rather than lose what it must keep:

- If those exact retained facts take up most of the window, compaction refuses
  instead of deleting them.
- A single oversized message is not split.
- Receipts with an unknown outcome or a failed tool block compaction.

### How a plan is checked and recorded

Every session has a journal: its private, append-only record on disk. Before it
applies a plan, the journal rebuilds its current view and compares the complete
request bytes with the planned source. A stale plan is refused even when the
message counts happen to match.

The journal then appends a new **context generation**, which is a new version of
the context the model sees. The record holds:

- the source record range;
- the kept and removed message indices;
- the exact summary;
- the before/after bytes;
- the length of the unchanged prefix.

Existing raw bytes do not change. Two commands let you inspect the result:

- `/compaction` shows the last record's explanation.
- `/journal RECORD OFFSET LENGTH` returns up to 4,096 source bytes, with
  `next_offset` for the next read.

Inspection uses positioned reads, so it cannot move the append cursor. The
terminal transcript is not redrawn as the compacted view.

### What compaction does to the cache

An automatic summary comes after the frozen system and developer messages and
the user messages that directly follow them (the task); those keep their bytes
and indices, so a KV checkpoint up to there stays valid. A summary you supply
comes directly after the frozen messages. After the summary, three things stay
exact: summaries you supplied, summaries of older turns, and the original user
instructions. Generated in-task summaries fold forward into a single summary
instead of piling up.

Compaction never touches the start of the request. The options, tool definitions
and system text stay byte for byte the same, and only what comes after them is
rewritten. If those would change, compaction fails.

Even so, expect fewer cached tokens on the next turn. The engine's cache holds
the conversation as one unbroken sequence. Once something in the middle changes,
everything after it must be read again, even text that stayed the same. Only
that next turn's reported usage tells you how many tokens came from cache.

Compaction clears the last request's numbers. It keeps the same session for
cache routing, and it keeps your running usage totals. The next prompt sends the
complete rebuilt request, counted by exact admission when the engine offers it.
Local context generations are bookkeeping, not engine cache identities.

### Warming the cache after a manual compaction

`--prewarm-compaction` is off by default. When you turn it on, the harness asks
the engine for one throwaway token right after a manual compaction, so the new
context has a chance to be in the cache before your next prompt.

- It needs exact admission and prefix caching from the engine.
- Nothing is pinned, and the engine may still evict it.
- It runs no tools, its output is discarded, and its usage is shown separately.
- A failure is not retried.

Automatic compaction skips the warm-up and goes straight to the real request.
The kept instructions and recent turns still have to be read along with the
summary, and cache eviction or block boundaries can add work.

## Automatic compaction during a task

### When it runs

Automatic compaction triggers when the conversation reaches 80% of the context
window. After that, optional rewrites wait for the 60% low watermark. A next
request that really does not fit tries another safe plan, even above that
watermark.

Automatic compaction uses only the local planner on this page; it never asks
the model for a summary. `/compact` asks the model first;
[context and compaction](../../docs/context-and-compaction.md) describes that
rollover and the budget table.

Each attempt is checked before anything is saved or sent:

- The automatic compaction worker counts the original and the candidate, and
  requires a real token reduction. When admission is negotiated, that means a
  smaller exact input.
- It saves, and dispatches, only a replacement that fits.
- If planning or counting fails, the source view is kept and a failure is
  recorded, and the oversized request is not sent.

Each successful rewrite appends a generation with the exact new request and the
source mapping. Original records stay unchanged. Compaction never resets the
60-round, 240-call, 20-minute or output quotas.

### Closed tool rounds inside a task

A tool round is one model reply that asks for tools, together with the results
of those calls. The automatic planner first tries to condense older complete
tool rounds, including rounds inside a single user request. That planner and the
older-user-turn fallback keep exactly the same things:

- the latest tool rounds: 4 at a 32,768-token window, 8 at 65,536 tokens and
  above, never fewer than 3
- every original user, system and developer message
- the latest prose answer
- the latest patch attempt and successful applied edit per path
- the latest command/check attempt and latest executed receipt per distinct argv
  or check profile
- every uncertain group

They remove calls and their results together. A few rules decide which receipt
counts as the last executed one:

- A denied repeat, invalid arguments or a profile preflight refusal does not
  replace the last executed receipt.
- A known exit, including a nonzero failure, confirms execution from the
  matching result, an output-artifact envelope included.
- Partial and unknown outcomes stay exact on their own, and do not push out the
  last confirmed execution.
- A timeout is never a passing check.

The fallback also keeps the newest KEEP user turns. Several user turns do not
loosen any protected tool-round or receipt rule. If keeping these facts stands
in the way of a real input reduction or a replacement that fits, compaction
refuses and keeps the source view.

### The summary it writes

The automatic planner writes a local extractive summary, built from facts it
copies out. It records call IDs, relative paths, requested line or byte ranges,
search terms, result metadata, and bounded facts from the head and tail.

It starts with a ledger of the excerpted calls: files with the ranges read and
edits applied, commands with their status line and up to 3 error lines, and
other tools (web, memory, MCP) with their arguments and first result line. Its
budget is 6 KiB at a 32,768-token window and 12 KiB at 65,536 and above; the
ledger takes at most half. The budget favors recent excerpts, and it
explicitly counts the old excerpts it left out. Generated task summaries fold forward instead of piling up
another full summary on each rewrite. Original user instructions and earlier
summaries you supplied stay exact.

This is lossy context, not verified semantic understanding. Fetch missing source
or raw journal ranges only when you need them, and re-read files before editing
them.

## Forks, branches and resume

A fork copies the conversation, not your files: it does not restore the
workspace. Every branch works in the same workspace, and switching branches
never undoes edits or reruns tools.

### Making a fork

`/fork NAME [TASK_LABEL]` works only while the agent is idle, and it refuses if
any tool outcome is unknown. Otherwise it:

- records a snapshot of file metadata in the parent;
- copies the journal into a new private session, noting the parent, the context
  version and your optional label;
- never overwrites an existing session.

The fork keeps the original saved tool outputs, with their IDs, quotas and
expiry. After the fork, the two conversations are independent.

A few limits apply:

- Creating a fork copies at most 64 MiB of raw journal bytes and keeps the same
  8,192-record limit.
- It uses one bounded copy buffer. It does not share disk or KV.
- Sessions you create by hand have their own per-file quotas. There is no global
  disk quota, and no automatic cleanup of old branches.
- Raw history includes private transcript data.
- A task label does not start a model request, choose a new model, grant
  permissions, or isolate a workspace.

### Resuming, switching and checkpoints

`/resume NAME` and `/switch NAME` select a journal after cleanup. `/branch`
shows the ancestry and the current checkpoint checks.

A checkpoint samples at most 128 safe relative paths from earlier read, list,
manifest and symbol calls and from patch paths. It labels incomplete sets, and
command effects it could not track. It only compares local metadata. Source
facts may already be stale when the checkpoint is taken, and matching metadata
does not prove matching contents.

Each resume or fork appends a short notice to the model's view, telling it to
read files fresh before acting on old file facts. The notice is durable but
triggers no inference.

### What the journal records about the workspace and engine

New journal headers bind to the device/inode identity of the selected directory,
so copying one into a different workspace is refused. Legacy headers without
this field keep their old behavior. No absolute workspace path, endpoint or
transport credential is stored automatically.

Optional engine checkpoint handles are private, scoped capabilities. They are
recorded separately from messages, in `engine_checkpoints` journal records.

- At most two input-boundary hints survive in the active view.
- A fork copies them, an ordinary append keeps them, and compaction or a changed
  source prefix invalidates them.
- Malformed optional hints are ignored without losing the transcript.
- `/journal` redacts handle bytes but keeps the inspection offsets.

Raw journal files stay private and must not be shared.

### What happens on the next prompt

Fork, resume and switch create a fresh contract client and affinity, and share
bounded handle ownership within the running native session. They clear the old
append-only bytes and mark the prefix as pending. The action itself sends no
model request.

The next explicit prompt negotiates and admits the full replay, and supplies an
eligible handle if one is available. How much of the prompt the engine restores
from its cache, and what that costs, stays unknown until the next response
reports it.

See [Engine checkpoint boundaries and lifecycle](engine-checkpoints.md) for the
input-only contract limitation and the child and deletion integration APIs.
Local generations and journal record numbers are never engine cache identities.

## Conceptual references

We studied these ideas in the project's landscape and backlog research. We
copied no implementation, prompt, format or test code from these or any other
competing projects. The relevant conceptual credits are:

- [Pi's context compaction and legal
  cuts](https://github.com/earendil-works/pi)
- [Hermes' separation of durable history and compressed
  context](https://github.com/NousResearch/hermes-agent)
- [DeepSeek Harness' ordered tool results and
  persistence](https://github.com/deepseek-ai/deepseek-harness)
- [Codex's durable histories and compaction](https://github.com/openai/codex)
- [DeepSeek's declarative skill
  catalog](https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/skill)
- [Aider's repository views](https://aider.chat/docs/repomap.html)
