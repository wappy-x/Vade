# Native context controls

This is a first-party Rust/std implementation. Negotiated `/v1/admission` counts
native model inputs with the upstream renderer/tokenizer; local per-message
figures and unsupported-provider fallback remain labeled heuristics. Engine inference, tokenization and KV
remain separate. Verification uses local fixtures and mock upstreams only.

## Full request admission and output

Admission includes every serialized option/message field, tool schemas, request
syntax, a framing allowance, safety margin and reserved output. There is no
subtraction for cached input, no trust in cumulative usage counters, and no
implicit retry. Known text parts and their extensions remain exact; unsupported
media/unknown content types fail admission because byte length cannot estimate
rendered media cost. Before native model dispatch, advertised exact admission
replaces heuristic context fit and uses its actual output reservation. The
automatic compaction worker counts the original and candidate, requires a real
token reduction, and persists only a fitting candidate. Tokenizer/HTTP failures
stop dispatch rather than silently downgrading. Cached input is not subtracted.
Independent byte/structure and tool-receipt safety limits remain.

Capture, model input, display and journal storage have separate limits. A large
captured result becomes a labeled deterministic excerpt; `tool_output_read`
uses session-local IDs and UTF-8 byte ranges. It never executes the original
tool. Bytes that were never captured (for example after a command hits its
source output cap) are unavailable. Captures are at most 64 KiB each, 128 per
session, 4 MiB total; range retrieval expires after 24 hours. Raw journal records
remain under the 64 MiB/8,192-record quota until owner removal. Capacity is checked
before side effects; source limits remain in force before artifact storage.

## Compaction and inspection

`/compact-preview KEEP SUMMARY` creates only a reviewable plan. `/compact KEEP
SUMMARY` applies that same operation while idle; bare `/compact` uses the
automatic excerpt planner described below. There is no model-authored summary,
summary retry, or script. Summary text is a user-role data message, never elevated
to a new system/developer instruction. The whole input/output budget must still
fit, and both estimated input and request bytes must strictly decrease.

With a supplied summary, the planner retains **all user messages**, all system/developer messages, every
assistant tool call and matching result, the latest assistant answer, and the
newest KEEP user turns. It removes only older standalone assistant messages.
This conservative policy preserves original constraints, unresolved user
questions, tool receipts and any file revisions recorded in them. User summaries
must preserve important decisions in removed assistant prose; structure tests
cannot establish semantic quality. If those exact retained facts dominate the
window, compaction refuses rather than deleting them. A single oversized message
is not split. Unknown-outcome/tool-failed receipts block compaction.

Before applying a plan, the journal reconstructs its current view and compares
the complete request bytes to the planned source. A stale plan is refused even
when message counts happen to match. The journal appends a new context generation
with source record range, retained
and removed message indices, exact summary, before/after bytes and unchanged
prefix length. Existing raw bytes do not change. `/compaction` displays the last
record's explanation; `/journal RECORD OFFSET LENGTH` returns up to 4,096 source
bytes with `next_offset`. Inspection uses positioned reads and cannot move the
append cursor. Terminal transcript display is not rewritten as the compact view.

The summary is the first user message after the frozen system/developer messages;
user-supplied/older-turn summaries and original user instructions remain exact
after it. Generated in-task summaries fold forward. A changed middle invalidates **all
later causal KV**. Equal tail text does not make its old suffix KV reusable.
Compaction verifies and retains the options/tools and system wire bytes, truncating
only the wire-builder suffix. Dropping/changing frozen messages or schemas fails
explicitly. It clears
stale last-request and per-response telemetry while retaining same-session affinity
and labeled cumulative usage. The next explicit prompt re-establishes the complete
prefix through negotiation and advertised exact admission. Cached tokens may drop;
only the new response's reported usage describes reuse. Reusable token prefix,
remote checkpoints and re-prefill cost remain response-reported or unknown. Local context generations
are bookkeeping, not Engine cache identities.

`--prewarm-compaction` is off by default. When selected, manual compaction starts
an idle cancellable worker; automatic compaction proceeds directly to its real request.
An admitted one-token completion retains raw tools/messages and rendering options,
discards all generated output, and executes no tools. The warm requires advertised
prefix caching and exact admission; zero-token generation is unsupported. Its
usage/wall time is separate from answer usage, and failures are not retried.
Retained instructions and tail turns must prefill along with the summary after
the unchanged prefix; cache eviction/block boundaries can add work. The offline
`demo-compaction` command reports diagnostic hashes and explicit estimates only.

### Closed tool rounds inside a task

The automatic planner first tries to condense older complete tool rounds,
including rounds inside a single user request. Both that planner and the
older-user-turn fallback share the same exact retention: at least the latest
three tool rounds, every original user/system/developer message, the latest
prose answer, the latest patch attempt and successful applied edit per path,
the latest command/check attempt and latest executed receipt per distinct argv
or check profile, and every uncertain group. They remove calls and results
together. A denied repeat, invalid arguments or profile preflight refusal does
not replace the last executed receipt. A known exit, including a nonzero failure,
confirms execution from the matching result, including an output-artifact
envelope. Partial and unknown outcomes remain exact independently and do not
displace the last confirmed execution; a timeout is never a check pass.

The fallback also keeps the newest KEEP user turns. Multiple user turns do not
relax any protected tool-round or receipt invariant. If preserving these facts
prevents a real input reduction or a fitting replacement, compaction refuses
and keeps the source view.

A local extractive summary records call IDs, relative paths, requested line or
byte ranges, search terms, result metadata and bounded head/tail facts. Its
12 KiB budget favors recent excerpts and explicitly counts omitted old excerpts.
Generated task summaries fold forward instead of accumulating another full
summary on each rewrite. Original user instructions and earlier user-supplied
summaries stay exact. This is lossy context, not verified semantic understanding;
retrieve missing source or raw journal ranges only when needed and re-read files
before editing them.

The 80% trigger retains its 60% low watermark for optional rewrites. An actual
non-fitting next request attempts another safe plan even above that watermark.
It counts the replacement before saving, requires reduced exact input when
admission is negotiated, and dispatches only a fitting replacement. Failed
planning/counting preserves the source view and records a failure; it does not
send the oversized request. Each successful rewrite appends a generation with
the exact new request and source mapping. Original records remain unchanged.
Compaction never resets the 60-round, 240-call, 20-minute or output quotas.

## Frozen project rules and lazy skills

A new native session loads root `AGENTS.md`. Repeated `--rules-scope PATH` selects
additional directory ancestor chains, from the root down, with a maximum of
eight selected paths and eight components each. It does not recursively discover
other subtrees or read a home directory. Only these selected scopes are active;
select the relevant subdirectories before starting work. Deeper rules take
precedence only for their subtree. Root rules apply to the whole workspace;
explicit user requests and executor/system policy remain above project rules.
Rules are ordered by depth then relative path. Files are bounded at 16 KiB, with
16 files/24 KiB total; an incomplete or oversized selection fails visibly.

The optional **visible** `skills/catalog.json` is deliberately small:

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

The catalog accepts only these fields (up to 16 entries/8 KiB); duplicate names,
unsafe paths, script/installer fields and oversized bodies are refused. Names
have 1–48 ASCII letters, digits, hyphens or underscores; descriptions have at most
240 UTF-8 bytes. Paths must stay under visible `skills/` and end in `/SKILL.md`.
Descriptor metadata is inspected at session creation, but body text is **not
read** until `skill_read` or local `/skill NAME` requests it. Bodies are bounded
at 16 KiB and need UTF-8 only when loaded. Catalog order is canonical by name.

Rules and catalog are snapshotted in a single developer message, attributed by
workspace-relative paths and selected scopes. It is at most 48 KiB serialized
and counts toward full-request admission. `/project` explains each inclusion;
`/project-rules` displays frozen text. Resume does not reread instructions or the
catalog. `/new` deliberately refreshes them and begins a new prefix/session.

A skill's recorded local revision consists of device/inode, byte size and
modification/change timestamps; this is a stale-read guard, **not a content hash
or security identity**. Opening and reading use the existing anchored workspace
boundary, refuse links/special/credential/hidden paths, and compare the opened
file's revision around the bounded read. If it differs from the frozen catalog,
loading is refused with a refresh instruction. Exact loaded body bytes, path and
revision become a normal tool result (or retrievable artifact) and are journaled.
As with editing, a hostile concurrent filesystem writer is outside the guarantee.
No body, rule, catalog entry or included script changes approval/sandbox policy;
no automatic execution, installer, plugin or dependency loading exists here.

`workspace_manifest` / `/map` enumerate on demand, at most 256 entries and 32
directories, depth 0–3, excluding hidden/credential paths, links, `target`,
`node_modules` and `vendor`. Incomplete lists say so. `workspace_symbols` /
`/symbols` scan a single ≤64 KiB UTF-8 file for up to 64 declaration-like lines,
240 bytes per hint. This is lexical string matching; comments, strings and
language syntax can produce false positives or omissions. Neither view is
semantic indexing, a file-revision guarantee, or an automatic prefix update.

## Transcript branches and workspace checks

`/fork NAME [TASK_LABEL]` is an idle-only explicit operation. It reconstructs the
last closed view from the source journal, refuses unknown tool outcomes, appends
a filesystem checkpoint to the parent, then copies the bounded raw journal and
adds parent session/record, context generation, optional task label and active
user-message reference. The new file is installed atomically and exclusively
under the private sessions directory. Existing destinations are refused. Forks
inherit original artifact IDs, count/byte quotas and expiry without resetting
them. Existing source records are unchanged; later messages are independent.

Fork creation copies at most 64 MiB of raw journal bytes and retains the same
8,192-record limit. It uses one bounded copy buffer; it is not a disk-sharing or
KV-sharing implementation. Manually created sessions have independent per-file
quotas; there is no global disk quota or automatic branch garbage collector.
Raw history includes private transcript data. A task label does not start a
model request, choose a new model, grant permissions, or isolate a workspace.

`/resume NAME` and `/switch NAME` select a journal after cleanup. `/branch` shows
ancestry and current checkpoint checks. A checkpoint samples at most 128 safe
relative paths from prior read/list/manifest/symbol calls and patch paths; it
labels incomplete sets and untracked command effects. It compares local metadata
only. Source facts may already be stale at checkpoint time, and matching metadata
is not proof of matching contents. Each resume/fork appends a short notice to the
model view requiring fresh reads before acting on historical file facts. The
notice is durable but causes no inference. Switching never undoes filesystem
edits and never replays any tool. It is a transcript fork, not a workspace restore.

New journal headers bind to the selected directory's device/inode identity;
copying one into a different workspace is refused. Legacy headers without this
field retain their old behavior. No absolute workspace path, endpoint or
transport credential is stored automatically. Optional Engine checkpoint handles
are private scoped capabilities, recorded separately from messages in
`engine_checkpoints` journal records. At most two input-boundary hints survive in
the active view; fork copies them, ordinary append preserves them, and compaction
or a changed source prefix invalidates them. Malformed optional hints are ignored
without losing the transcript. `/journal` redacts handle bytes while preserving
inspection offsets. Raw journal files remain private and must not be shared.

Fork/resume/switch create a fresh contract client and affinity, sharing bounded
handle ownership within the running native session. They clear old append-only
bytes and mark the prefix pending. The next explicit prompt negotiates and admits
the full replay, supplying an eligible handle if available. The action itself
sends no model request. Cache reuse and cost remain unknown until reported by a
new response. See [Engine checkpoint boundaries and lifecycle](engine-checkpoints.md)
for the input-only contract limitation and child/deletion integration APIs. Local
generations and journal record numbers are never Engine cache identities.

## Conceptual references

Ideas were studied in the project's landscape/backlog research; no competitor
implementation, prompt, format or test code was copied. Relevant conceptual
credits are [Pi's context compaction and legal cuts](https://github.com/earendil-works/pi),
[Hermes' separation of durable history and compressed context](https://github.com/NousResearch/hermes-agent),
[DeepSeek Harness' ordered tool results and persistence](https://github.com/deepseek-ai/deepseek-harness),
[Codex's durable histories and compaction](https://github.com/openai/codex),
[DeepSeek's declarative skill catalog](https://github.com/deepseek-ai/deepseek-harness/tree/master/packages/skill),
and [Aider's repository views](https://aider.chat/docs/repomap.html).
