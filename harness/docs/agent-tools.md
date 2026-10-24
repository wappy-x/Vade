# Native tool and session contract

These tools use first-party Rust with the standard library only. It leaves
HTTP, streaming, prompt serialization, inference, and KV ownership with their
existing modules/projects. No competitor implementation code is included.

## Request limits

Each native user request defaults to 60 tool rounds, 61 model turns and 240
proposed tool calls. One model reply requesting tools is one tool round, even
when it proposes several calls. The default time limit is 20 minutes of active
agent work; time waiting for your approval is excluded. On expiry, the agent
finishes or aborts the current tool cleanly and keeps the conversation and
journal. It says "Stopped after 20 minutes; say continue to keep going". Say
`continue` to keep working in the same conversation with a fresh request budget.
Context, history and output bounds still apply and can stop work earlier.

Use `--agent-time-limit-seconds N` to change the time limit in seconds; `0`
disables it. Native `agent` mode also accepts an explicitly selected JSON config
file through `--config PATH`:

```json
{"agent_time_limit_seconds":1200}
```

The key accepts a nonnegative integer number of seconds. An omitted key uses
1,200 seconds, and the CLI time-limit flag overrides the file. Configuration
files are never discovered automatically. Normal and child-enabled parent runs
use these same request limits; child tasks retain their separate quotas.

Native replies also have a local budget: measured decode speed times the smaller
of 120 seconds and one eighth of remaining active task time, clamped to 512–4096
estimated tokens. Unknown speed uses 2048 tokens per 120 seconds. `--reply-tokens N`
overrides the token estimate; `0` disables both local token and time budgets.
The time bound includes dispatch, prefill and response reads. The token cap counts
content prose plus tool-argument bytes (three bytes per token); reasoning is
excluded and bounded only by the time deadline. One complete SSE event can
overshoot the estimate. On exhaustion the harness closes the request
and rolls back the entire uncommitted tool group, then asks for smaller steps
under the same task deadline and turn quotas. It never executes a partial call
or invents closing arguments. Previously completed tool rounds remain committed.
Closing the connection requests upstream cancellation; engine stop is best effort.

## Exact patch format

`apply_patch` accepts one JSON property, `patch`, containing the following marker
format. Markers use LF. Payload lines preserve their bytes, including CR in CRLF
file content. Paths are relative and pass the same hidden/credential/traversal
filter as workspace reads. Parent directories must already exist.

```text
*** Begin Patch
*** Add File: example.txt
+first line
+second line
*** Update File: existing.txt
@@
 exact context
-old line
+new line
*** Update File: source.txt
*** Move to: destination.txt
*** Delete File: obsolete.txt
-entire old content
*** End Patch
```

Update hunks use **bare `@@`**, in file order. The complete old hunk (space and
minus lines) must match exactly once at line boundaries. There is no whitespace
normalization, fuzzy matching, line-number search, or implicit replacement of all
occurrences. An empty existing file accepts an insertion-only hunk. Add lines
must begin `+`; delete lines must reproduce the entire old file with `-` prefixes.
Immediately after an unterminated payload line, use the literal marker
`\ No newline at end of file`. A move can also contain update hunks.

Patch text is limited to 64 KiB, targets and results to 4 MiB per file, and a
batch to 32 files. The unified approval preview shows exact changed hunks with
three context lines, merging overlapping context windows. The 192 KiB preview
cap truncates display with an explicit notice; it does not prevent an otherwise
valid edit. Reads and edits reject symlinks, hardlinks, and special files.
Workspace path components must be ASCII and exactly match the real directory
entry spelling, including case. Every opened component is checked through its
anchored parent handle before content is read or an edit is prepared; Unicode
and case aliases are refused. Lookups fail closed if the entry cannot be verified
within 16,384 entries. Unicode file contents remain supported.
Preflight prepares all operations before approval. The tool sends a short
success/error result back to the model.

`workspace_read` accepts `path` plus optional zero-based byte `offset` (default 0)
and byte `limit` (1–4096, default bounded by the 4 KiB inline output budget).
Sources may be up to 4 MiB. Partial reads identify the actual half-open range,
source size and next offset. The offset must be a UTF-8 boundary; a limit ending
inside a multibyte character stops before it. Whole small-file reads keep their
plain text result. Hidden/credential paths and directory-handle protections
apply to every range. Whole-file local previews, project rules, skills and
symbol scans retain their existing independent read bounds. Limit errors name
the actual size and the applicable bound.

Each file uses a same-directory exclusive temporary file, fsync, and atomic rename.
An existing file's mode bits are retained, but ACLs, extended attributes, and old
file identity are not retained. Add and move destinations require an exclusive
rename. Pure moves use one rename. Move+update installs the destination then
removes the old source. Batches and move+update are not crash-atomic transactions;
errors report partial operations, and a journaled unknown outcome after a crash
requires inspection. No silent overwrite is attempted after a detected conflict.
Checks are optimistic; other processes must not maliciously race workspace edits.
Temporary files from an abrupt crash can remain as hidden `.vade-edit-*` files;
they are not considered completed edits or automatically reused.

## Commands and approvals

`workspace_shell` accepts exactly one of `argv` (an array of 1–128 strings, total
at most 16 KiB) or `check_profile`, plus `timeout_ms` (1–120000). Example arguments:

```json
{"argv":["/usr/bin/true"],"timeout_ms":1000}
```

For this monorepo, profiles `engine-host`, `engine-bench-host`, `harness-host`
and `cli-host` run the existing offline Cargo host checks in their separate
workspaces. They require an explicit `--shell-toolchain` distribution and normal
command approval (`--approval auto` runs them in the sandbox without asking). For example:

```json
{"check_profile":"harness-host","timeout_ms":120000}
```

After an edit, check the affected component, use failure diagnostics to repair,
then rerun. Output overflow keeps bounded head/tail diagnostics while draining
the process to completion, with observed/omitted byte counts, elapsed time,
exit code and signal in the receipt. Discarded middle bytes are unavailable.
Timeout/cancellation remains partial. Checks are synchronous, limited to 120
seconds, and do not qualify GPU numerics; narrow a timed-out check before retrying.

A user check (`--check COMMAND` or `.vade/check`) runs automatically after a
tool round that changed files, as a harness-authored `workspace_shell` call
(`vade-check-N`) with the same sandbox and approval policy; see the
[harness README](../README.md).

There is no implicit shell parsing. `/bin/sh -c SCRIPT` is explicit and outside
the read-only command allowlist, so it requires approval in `ask`. In `auto`, it
runs inside the same OS sandbox without asking, like other validated commands.
Commands remain subject to the same OS sandbox. The allowlist accepts only these
complete argument forms, with either the bare name or its listed system path:

| Command | Fixed executable | Accepted arguments |
| --- | --- | --- |
| `pwd` | `/bin/pwd` | None |
| `true` | `/usr/bin/true` | None |
| `ls` | `/bin/ls` | Optional `-1` first, then zero or more workspace-relative paths |
| `cat` | `/bin/cat` | One or more workspace-relative paths; no options |
| `git status` | `/usr/bin/git` | No option or `--short` |
| `git diff` | `/usr/bin/git` | No option or `--stat` |
| `git log` | `/usr/bin/git` | No options, `--oneline`, or `-n N` with N from 1 to 100; options cannot be combined |

Bare names resolve to these fixed executables, never through PATH. Path arguments
may be `.`; they cannot start with `-` or contain parent traversal, hidden or
credential-related components. In `ask`, other arguments require approval even
for a listed executable. An executable in the workspace, shell metacharacters, PATH
manipulation, a journal, or a previous approval cannot extend the allowlist.

| Policy | Exact patch | Allowlisted command | Other sandboxed command |
| --- | --- | --- | --- |
| `ask` (default) | Ask once | Execute after preview | Ask once |
| `auto` | Apply after preview | Execute after preview | Execute after preview |
| `never` | Deny | Deny | Deny |

`auto` includes workspace executables such as `./tools/rgbds/rgbasm`, system
tools, and explicit shell commands. Every command still uses the macOS OS
sandbox: network is denied, writes stay inside the selected workspace and the
private session build/cache/temp area, and hidden, credential and excluded paths
remain inaccessible. The policy does not authorize commands outside that
sandbox. This harness has no unsandboxed execution route; a request needing
broader access is refused rather than retried with broader permissions.

UI approvals bind to one prepared patch or one immutable argv request. They expire
after 120 seconds; cancellation or a disconnected UI denies them. Noninteractive
runs never consume piped prompts as approvals. Ask/deny is an operational policy,
not a substitute for sandbox boundaries. Unsupported sandbox startup fails closed.
Journals from the immediately preceding approval policy can still resume with
their original tool descriptions and exact prompt bytes. The selected launch
policy controls approval; saved descriptions never grant additional permissions.

The macOS profile uses default-deny Seatbelt rules. Reads allow the selected root
and explicit system/toolchain locations; writes allow the root, a private session
build area, and `/dev/null`.
Network and hardlink creation are denied. Sysctl reads are limited to named
OS/version, argument-size, CPU-count, memory-size and page-size values, including
`hw.pagesize_compat` for Rust runtime startup. Cargo's system library startup also
requires the single public `/private/etc/ssl/openssl.cnf` file; this grants neither
TLS/network access nor general `/etc` reads. Process
arguments/environment and hostname sysctls are not allowed. Foreign
`process-info*` and `kern.procargs*` reads are explicitly denied; self queries
needed for executable startup stay under the existing OS policy. A compiled Rust
fixture probes `KERN_PROCARGS2` directly against a parent with an otherwise-cleared
environment and a synthetic marker. A failing `ps` alone is insufficient evidence
because it may stop at a different metadata query first. Homebrew reads are
limited to its tool/library subdirectories; its `etc` and `var` trees are excluded.
An explicit owner `--shell-toolchain DIR` selects a Rust distribution, including
a home-installed distribution. Its canonical bin/lib trees alone become readable
and are explicitly write-denied. Required bin/cargo, bin/rustc and bin/rustdoc
files and bin/lib directories must be real, not symlinks; the distribution must
be separate from the workspace. The owner vouches for public compiler/runtime
contents. There is no credential, registry or toolchain discovery, no rustup shim
execution, and no reading the rest of the distribution or home. This option is
launch policy, never a model tool parameter or a journal-restored capability.
System/Homebrew toolchains and workspace-local executables remain
subject to approval and the same sandbox. File metadata lookup is broader to
support OS path traversal. `/System/Volumes/Data` is deliberately not an allowed
read root. Hidden and credential-related paths are denied even within the root.
Changed workspace-root identity is refused before launch. Workspace verification
skips `target` and `node_modules` directories at every depth, while Seatbelt denies
all reads (including executable mapping) and writes to these names and descendants.
Removal and rename are denied too, including macOS Data-volume aliases. Thus Cargo
or pnpm hardlinks and large dependency trees do not disable unrelated commands;
uninspected aliases cannot reveal or change outside files. A visible alias in the
scanned source tree still fails the hardlink check, and an excluded root that is
itself a symlink is refused. Hardlink creation remains denied.

The owner can add at most eight `--shell-exclude NAME` options at launch. Each is
one visible ASCII name, at most 64 bytes, using letters, digits, `_`, `-` or `.`
(no leading dot). Names match without ASCII case distinctions (including macOS
case aliases) at every depth, even for newly created paths, and
also block same-named files conservatively. Defaults cannot be removed; model
arguments and journals cannot change the policy. Approval previews list all
exclusions. A workspace located inside a matching named component is likewise
inaccessible. This is deliberately not a recursive `.gitignore` parser.

The remaining tree is scanned without following links, at most 20,000 entries;
excluded descendants consume no scan budget. Links, special files, unreadable
entries and limit/cancellation refusals identify the offending relative path.
Commands can read/write ordinary source files after preflight, but cannot use
dependencies or artifacts in excluded trees. New build artifacts use the private
area described below. This does not grant package-install network access.
Tool-created escaping symlinks still cannot bypass the sandbox.

### Session build area

Only after approval and workspace preflight, the harness exclusively creates a
new `vade-build-*` directory in the canonical OS temporary directory, outside the
workspace, using `create_dir` with mode 0700. Existing paths are never adopted.
It verifies directory type, owner, mode and device/inode identity before reuse.
The profile allows that exact tree, while denying hardlink creation and root
rename/removal/mode changes. Other sessions' areas remain inaccessible. A fresh
empty tree plus denied hardlink creation prevents pre-existing foreign aliases;
workspace `target` and `node_modules` stay completely denied.

The environment is cleared, then rebuilt with a fixed PATH (optionally prefixed
by the selected Rust bin), locale and these harness-owned locations:

| Variable | Private build-area location / value |
| --- | --- |
| `CARGO_TARGET_DIR`, `CARGO_HOME` | `cargo-target`, `cargo-home` |
| `TMPDIR`, `HOME` | `tmp`, `home` |
| `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` | `cache`, `config` |
| `npm_config_cache` | `npm-cache` |
| `npm_config_userconfig`, `npm_config_globalconfig` | Separate empty config files |
| `CARGO_NET_OFFLINE`, `npm_config_offline` | `true` |
| `CARGO_BUILD_JOBS`, `CARGO_INCREMENTAL` | `2`, `0` |
| `RUSTC`, `RUSTDOC` (selected distribution only) | Its real bin executables |

Hidden generated files in this area are permitted, including Cargo fingerprints.
Workspace hidden files/journals and credential-name rules stay denied. Scripts
may override environment variables or arguments, but OS rules still deny excluded
trees, unrelated reads, writes outside the allowed roots, and all network. Neither
an `env` command nor `--target-dir` can expand these capabilities.

The area persists across commands and prompts in one active native session.
Compaction preserves it. Exit, `/new`, resume/switch or fork selection releases
it; cleanup is best-effort and never follows a replaced root or symlink entries.
Abrupt termination or unreadable tool-created subdirectories may leave an orphan,
which is never adopted automatically. There is no persistent build cache.
Between commands, the harness bounds retained state to 1 GiB and 100,000 entries
without following symlinks. Exceeding it yields `BuildLimit` and refuses further
commands for that build session; use `/new`. This is not an instantaneous disk
quota: a running command can exceed it before exit, under the existing deadline
and per-file limits. Limit scans never inspect pre-existing workspace artifacts.

Real local acceptance runs `cargo build --offline`, `cargo test --offline`, and
a repeated build through `workspace_shell` for a temporary dependency-free crate
with a build script and unit test. A second fixture exercises CLI toolchain
selection and interactive approval. No host Cargo config, credentials or registry
is imported. Missing dependencies and hidden workspace config remain unsupported.
npm cache/config locations are provisioned, but existing `node_modules` remains
denied; no actual Node/npm installation or third-party package was executed.

Children have null stdin and nonblocking captured pipes. Kernel limits are 120 CPU seconds,
64 MiB per output file, 1,024 processes for the user as evaluated by the OS, and
128 descriptors. These are inherited kernel limits, not a per-command memory or
process-tree quota. Command wall time and aggregate stdout/stderr are polled every
10 ms; each drain iteration is bounded. Outputs retain partial data on timeout,
cancellation, nonzero exit, or cap exhaustion. The native adapter uses a smaller
byte capture cap to reserve room for result framing and worst-case UTF-8 repair.
The UI displays capped command results. Local I/O failures stop the activation as
`tool_failed`; the UI also shows the refusal/error detail, including scan paths.
Inspect state before retrying because execution may have started.
The process group is killed and direct child reaped on all return/error paths.
Abrupt harness termination cannot run process cleanup. A descendant that
deliberately changes process group is outside the cleanup
promise; OS sandbox restrictions still inherit. There is no claim of security
against a hostile same-user process concurrently changing the workspace.

## Journal recovery

Native journals are private local transcripts, separate from the legacy
metadata-only `Session` store used by direct/demo/Codex. Every record contains
an eight-byte little-endian JSON payload length, the JSON payload, and the same
eight-byte length as commit trailer. There is no custom cryptography or checksum.
JSON records include version, monotonic sequence, a retained message-prefix
length, and the appended/replacement tail. Only the first record contains the
stable request options. Writes append and fsync; subsequent tool-stage revisions
replace only the logical tail during replay, leaving old records on disk.

A before-tool checkpoint includes the validated assistant call and closed results
for every proposed ID. The imminent call is marked `unknown_outcome`; other pending
calls are `not_executed`. This record must sync before executor entry. Each result
syncs before the runtime continues. If an intent sync fails, the tool does not run.
If a result sync fails, further execution stops and resume may show unknown outcome.
The UI blocks further prompts against a failed journal.

Resume opens the owned 0600 file under an exclusive kernel lock, validates framing,
sequence, bounded JSON/history, and closed tool pairing, and removes only a torn
final record. A malformed complete record is an error. Replay never executes a
tool or contacts an upstream. Limits are 64 MiB, 8,192 records, and the existing
4 MiB active-history bound. `/new` creates a separate session rather than appending
a reset that could erase the prior session's identity. `/compact` appends a logical
replacement; the original records remain available on disk. No encryption or
content-based credential scanner is claimed; users should not put credentials in
prompts or workspace content sent to the model.

## Context accounting

The current estimate counts every serialized request component at three UTF-8
JSON bytes per token, adds 256 + 16/message + 32/tool framing units and a 12.5%
(minimum 256) safety margin. The default output reserve is 4,096 tokens;
`--max-output-tokens` replaces it, including with a smaller value. Negotiated
defaults respect the advertised output limit, and explicit request output limits
are counted as well. These are heuristics, not exact counts or a
model-independent upper bound. The byte figure is the actual serialized request
size. Media/unknown content parts are refused without exact provider admission.
Cached input is never subtracted. Native/direct requests use negotiated output
defaults unless the caller selects an output limit.

Negotiated tokenizer admission drives native pre-generation fit and automatic
compaction thresholds; the candidate must reduce actual input tokens and fit
before persistence. Without that capability, labeled estimates still apply.
The runtime checks before every model call and before tool execution. Large
results become labeled excerpts backed by exact captured bytes in capped private
journal artifacts. `tool_output_read` provides bounded UTF-8 ranges; `/output`
inspects them locally. A result that exceeds remaining estimated context stays
durable and closed; the next model request stops. Hard byte/JSON limits remain
independent. There is no automatic identical retry.

`/compact-preview KEEP SUMMARY` previews; `/compact` deliberately replaces old
standalone assistant prose while keeping every user message, instruction, tool
pair/receipt, last meaningful answer and recent turns. `/compaction` and `/journal`
show exact summaries and source records. The operation must reduce bytes and
estimated tokens and still fit admission; unknown tool outcomes are refused.
Raw records are retained. A changed middle invalidates all subsequent KV,
including unchanged tail text. The frozen system/tools wire prefix stays exact;
the summary immediately follows it. `--prewarm-compaction` optionally sends one
discarded token on a cancellable worker, executing no tools and leaving raw
records unchanged. Actual cache reuse/cost requires upstream response telemetry.

See [context, project rules, skills and transcript branches](context-and-projects.md)
for exact storage limits, catalog format, snapshot/revision semantics and local
inspection commands. All project tools are read-only data operations: no skill
scripts or installers run automatically, and project text cannot grant tool
permissions. Fork/switch changes transcripts, never the workspace's files.

## Conceptual references

The existing README credits Codex, Pi, and Hermes for high-level coding-agent and
terminal interaction concepts. The familiar apply-patch marker vocabulary is an
interface idea, implemented here without copying an implementation. macOS's
`sandbox-exec(1)` facility and installed OS sandbox profile declarations were
inspected to understand the platform interface and dynamic-loader requirements;
this project's minimal profile is first-party. No engine or provider was contacted.

Patch approval fidelity: changed lines, unchanged context, and path headers
visibly escape nonprinting Unicode, combining marks, directional controls,
variation selectors, and non-ASCII whitespace as `\u{hex}`. Tabs are `\t`,
carriage returns are `\r`, and literal backslashes are doubled. Line feeds
separate preview lines. Classification uses Rust's standard-library Unicode
printability tables, with an explicit escape for the printable braille blank.
Escaping counts toward the 192 KiB limit; an oversized diff is refused before
approval. The frozen approval remains bound to the original patch bytes and
write plan. Workspace path controls remain prohibited.

Command approval previews are JSON argument arrays. Nonprinting Unicode is
additionally represented as visible JSON Unicode escapes, including UTF-16
surrogate pairs for supplementary code points. Literal escapes remain distinct,
and the displayed JSON decodes to the original argv. The terminal refuses any
approval preview with unescaped nonprinting characters; argv and one-time
approval binding retain the original values.
