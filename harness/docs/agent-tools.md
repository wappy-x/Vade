# Native agent tools and limits

This is the reference for the native agent's tools. It covers how much work one
request may do, the exact patch format, file reads and writes, shell commands
and approvals, the macOS sandbox, journal recovery and context accounting. The
[harness README](../README.md#native-coding-agent) gives the short version.

We wrote these tools ourselves in Rust, using only the standard library. HTTP,
streaming, prompt serialization, inference and KV ownership stay with the
modules and projects that already handle them. No implementation code from
competing coding agents is included.

**In short.** The agent works inside one workspace folder you pick. It can
search, read and patch files there. Under the default approval policy,
`--approval ask`, it asks before it applies each patch, and the patch preview
shows the exact diff. On macOS it can also run commands in a sandbox; on Linux
it has no command tool. A command's approval shows its arguments and sandbox
settings, the folder and every environment variable, not the files it may
change. Every command asks unless an allow-list rule matches its argv. `--approval auto` applies patches and runs commands without asking,
still inside the sandbox. Each request runs under fixed limits
on rounds, calls and time, and a stopped request keeps its conversation, so
`continue` picks up where it left off. The sections below give every limit and
rule.

## How much one request may do

Each native user request gets, by default:

- **60 tool rounds.** One model reply that asks for tools counts as one round,
  even when it proposes several calls.
- **61 model turns** and **240 proposed tool calls**.
- **20 minutes of active agent work.** Time spent waiting for your approval
  does not count.

When time runs out, the agent finishes or cleanly aborts the current tool and
keeps the conversation and journal. It says "Stopped after 20 minutes; say
continue to keep going". Say `continue` to keep working in the same
conversation with a fresh request budget. Context, history and output bounds
still apply and can stop work earlier.

Use `--agent-time-limit-seconds N` to change the time limit in seconds; `0`
turns it off. Native `agent` mode also accepts a JSON config file that you pick
explicitly with `--config PATH`:

```json
{"agent_time_limit_seconds":1200}
```

The key takes a nonnegative whole number of seconds. If the key is missing, the
limit is 1,200 seconds, and the CLI time-limit flag wins over the file.
Configuration files are never discovered automatically. Normal parent runs and
parent runs with children enabled use these same request limits, and child tasks
keep their own quotas.

### Reply budget

Each native reply also has a local budget:

- **Time.** The smaller of 120 seconds and one eighth of the remaining active
  task time, but never less than 30 seconds (or the remaining time minus 10
  seconds, when that is shorter). It covers dispatch, prefill and response
  reads.
- **Tokens.** The measured decode speed times that time, clamped to 512–4096
  estimated tokens. Before the speed is known, the rate is 2048 tokens per 120
  seconds.
- **What counts.** Content prose plus tool-argument bytes, at three bytes per
  token. Reasoning is not counted; only the time limit bounds it. One complete
  SSE event can overshoot the estimate.
- **Override.** `--reply-tokens N` sets the token estimate, and `0` turns off
  both the token and the time budget.

When the budget runs out, the harness closes the request and rolls back the
whole uncommitted tool group, then asks the model for smaller steps under the
same task deadline and turn quotas. It never runs a partial call or makes up
closing arguments. Tool rounds that already completed stay committed. Closing
the connection asks the upstream to cancel; stopping the engine is best effort.

## Exact patch format

`apply_patch` takes one JSON property, `patch`, that holds the marker format
below. Markers use LF. Payload lines keep their bytes, including CR in CRLF file
content. Paths are relative and go through the same hidden/credential/traversal
filter as workspace reads. Missing parent folders inside the workspace are
created.

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
*** End Patch
```

The old text, not a line number, decides where a hunk goes:

- Update hunks start with `@@`, in file order. A header with line numbers or an
  annotation, such as `@@ -12,4 +12,4 @@ fn main`, is accepted; the numbers are
  ignored.
- The complete old hunk (the space and minus lines) must match exactly once, at
  line boundaries. When there is no exact match, a unique match that differs
  only in spaces or tabs is accepted, keeps the file's unchanged lines and is
  reported in the result. Text that matches more than once is refused. There is
  no other fuzzy matching, line-number search or implicit replace-all.
- A bare empty line inside a hunk counts as an empty context line.
- An empty existing file accepts a hunk that only inserts.
- Add lines start with `+`. An Add body with no `+` prefixes at all is taken as
  the whole new file; a body that mixes the two is refused. Add requires an
  absent file, and Update an existing one.
- A Delete needs no body. A body of `-` lines, if given, must reproduce the
  entire old file.
- The `*** Begin Patch` and `*** End Patch` markers may be left out, and one
  enclosing plain, `diff` or `patch` code fence is accepted.
- Right after a payload line with no trailing newline, use the literal marker
  `\ No newline at end of file`.
- A move can also contain update hunks.

Size limits: patch text up to 64 KiB, targets and results up to 4 MiB per
file, and up to 32 files in a batch. Reads and edits reject symlinks,
hardlinks and special files.

The unified approval preview shows the exact changed hunks with three context
lines, and merges context windows that overlap. It is capped at 192 KiB. A
patch whose diff would go past that is refused before approval, because you
must see every changed line before you approve it.

To keep the approval preview faithful to the patch, changed lines, unchanged
context and path headers visibly escape nonprinting Unicode, combining marks,
directional controls, variation selectors and non-ASCII whitespace as `\u{hex}`.
Tabs show as `\t`, carriage returns as `\r`, and literal backslashes are
doubled. Line feeds separate preview lines. Classification uses Rust's
standard-library Unicode printability tables, with an explicit escape for the
printable braille blank. Escaped characters count toward that 192 KiB cap. The
frozen approval stays bound to
the original patch bytes and write plan. Workspace path controls are still
prohibited.

## Paths, reads and writes

Workspace path components must be ASCII and exactly match the spelling of the
real directory entry, case included. Every opened component is checked through
its anchored parent handle before content is read or an edit is prepared, and
Unicode and case aliases are refused. Lookups fail closed if the entry cannot be
verified within 16,384 entries. Unicode file contents are still supported.
Preflight prepares every operation before approval. The tool sends a short
success or error result back to the model.

`workspace_read` takes `path`, plus an optional zero-based byte `offset`
(default 0) and byte `limit` (1–4096, by default bounded by the 4 KiB inline
output budget). Sources may be up to 4 MiB. Partial reads give the actual
half-open range, the source size and the next offset. The offset must fall on a
UTF-8 boundary, and a limit that ends inside a multibyte character stops before
it. Whole reads of small files keep their plain text result. The
hidden/credential path and directory-handle protections apply to every range.
Whole-file local previews, project rules, skills and symbol scans keep their own
independent read bounds. Limit errors name the actual size and the bound that
applies.

A file that starts with `%PDF-` reads as extracted text. PDF files may be up to
50 MiB. The text starts with a summary line, then gives each page as a
`page N of M` line and the page text, with one blank line between pages. Text
follows the order of the content stream, so multi-column pages and tables can
interleave. `offset` and `limit` index the extracted text, and partial reads
name its size and the source PDF size. The reader reads at most 2,000 pages and
4 MiB of text, and a note names the page where it stopped. Encrypted files are
refused. A file with no text layer, such as a scan, gives one line that says so.
When the page limit stops the read first, that line covers only the pages read
and keeps the page-limit note. When a work limit skipped content and no text
was found, the line says that the PDF may still have a text layer. The last
extracted text is kept for the same path and file revision, so paging through a
PDF parses it once.

Each PDF read has one work budget, and every parser and decoder charges it
before the work. The budget allows 1 GiB of bytes lexed or scanned, 512 MiB of
stream data decoded or copied (every use of a stream counts), 64 Mi parsed
objects, 256 MiB of cached data, 1 Mi built tables (Huffman tables,
cross-reference sections, CMaps), 4 Mi font entries, 256 Mi other work steps and
a nesting depth of 64. When one of these runs out, the read stops, and a note
names the limit and the page. Stream bytes are not copied until a charged
decode reads them. Some limits apply to one page and skip only part of it: one
page's content may join up to 128 MiB from at most 4,096 `/Contents` entries,
one content-stream operand may hold 65,536 objects, and Form XObjects may run
2,000 times and 32 MiB of content on one page, and 100,000 times in one file. A
note names the pages where such a limit skipped content. An inline image is
skipped by its own data size or encoding; when its end cannot be found, the rest
of that content stream is skipped and the page is noted. A cancelled or
timed-out tool call stops the parse, also in a child agent's read.

Each file is written through an exclusive temporary file in the same directory,
fsync, and an atomic rename. An existing file keeps its mode bits, but its ACLs,
extended attributes and old file identity are not kept. Add and move
destinations need an exclusive rename. A pure move is one rename. A move with an
update installs the destination, then removes the old source. Batches and
move+update are not crash-atomic transactions: errors report the partial
operations, and a journaled unknown outcome after a crash needs inspection. Once
a conflict is detected, nothing is silently overwritten. The checks are
optimistic, so other processes must not maliciously race workspace edits.
Temporary files from an abrupt crash can be left behind as hidden `.vade-edit-*`
files; they are not treated as completed edits and are never reused
automatically.

## Web reads and search

`web_fetch` reads a public HTTP or HTTPS URL. Use it for a URL from the user,
current documentation, or facts after the model's training. It takes `url`,
plus optional `max_bytes` and `timeout_s`. The limits in the tool schema are
262144 bytes and 20 seconds by default. The hard limits are 2097152 bytes and
60 seconds. Values must be positive whole numbers within those limits.

The result gives the final URL, content type and captured byte count before
the body. HTML goes through the document reader to Markdown. Plain text,
Markdown and JSON pass through. PDF and other formats with a registered reader
use that reader. Other types give `binary <type>, <bytes> bytes`.
The byte limit applies to both the downloaded body and the converted text.
A cut result ends with `[truncated at N bytes]`. A reader can refuse an
incomplete document. When a download cuts a UTF-8 character, HTML and table
readers get the valid prefix. Earlier characters keep their encoding. A
complete response with a damaged UTF-8 byte order mark tail gives an error.
The conversion character cap is set before the reader runs. It uses the
smaller of the download limit and the retained result room. The calculation
in `web.rs` reserves the response header and 128 bytes for cut markers.

`web_search` takes `query` and optional `count`. Its schema sets the default
count to 5 and the hard limit to 10. It reads the DuckDuckGo HTML search page
through the same fetch path. It returns a numbered list of source titles,
URLs and snippets. It unwraps DuckDuckGo result links. An empty search or
failed request gives a short line. The harness does not make up results.

Both tools are on by default. Launch the harness with `--no-web`, or set
`VADE_WEB=0`, to turn both off. The setting also blocks execution from a saved
session that still lists the tools. These are read tools, like workspace
reads. The patch and shell approval policy does not control them. Shell
commands still have no network access. Fetched text is source data, not
instructions or permission to run another tool.

The runner uses system curl with no config, credentials, cookies or proxy.
It allows only HTTP and HTTPS. It checks each redirect before it sends the
next request. The runner limit is 5 redirects per call. It refuses local
hosts, loopback, private and link-local address literals, URL credentials,
and ambiguous numeric host spellings. DNS rebinding is out of scope in r1.
Missing curl, timeout, HTTP failure and reader failure give explicit errors.
Cancellation stops and reaps the child process.

Results use the normal retained output path. The existing artifact limits
in `artifacts.rs` allow 65536 bytes per result and a 4096-byte inline excerpt.
A cut context result has its own byte marker. Use `tool_output_read` to read
retained text. Bytes past either cap are not retained. Search queries are
limited to 2048 bytes, and URLs to 8192 bytes, by `web.rs`.

## Commands and approvals

`workspace_shell` takes exactly one of `argv` (an array of 1–128 strings, at
most 16 KiB in total) or `check_profile`, plus `timeout_ms` (1–120000). Example
arguments:

```json
{"argv":["/usr/bin/true"],"timeout_ms":1000}
```

For this monorepo, the profiles `engine-host`, `engine-bench-host`,
`harness-host` and `cli-host` run the existing offline Cargo host checks in
their separate workspaces. They need an explicit `--shell-toolchain`
distribution and normal command approval (`--approval auto` runs them in the
sandbox without asking). For example:

```json
{"check_profile":"harness-host","timeout_ms":120000}
```

After an edit, check the component you touched, use the failure diagnostics to
repair it, then run the check again. When output overflows, the harness keeps
bounded head and tail diagnostics while it drains the process to completion, and
the receipt (the tool's result record) carries the observed and omitted byte
counts, elapsed time, exit code and signal. The discarded middle bytes are not
available. A timeout or cancellation stays partial. Checks are synchronous,
limited to 120 seconds, and do not qualify GPU numerics; narrow a check that
timed out before you retry it.

A user check (`--check COMMAND` or `.vade/check`) runs automatically after a
tool round that changed files. It runs as a `workspace_shell` call written by
the harness (`vade-check-N`), with the same sandbox and approval policy; see the
[harness README](../README.md).

There is no implicit shell: `/bin/sh -c SCRIPT` is explicit. Before any command
runs, [shell safety](../../docs/shell-safety.md) classes it as safe, needs
approval or blocked. It parses `sh -c` scripts into simple commands for this.
In `ask`, every command waits for `y`, `s`, `p` or `n`, unless an allow-list
rule matches its argv of plain literal words (word for word, with `<path>`
slots for workspace paths); then it runs after preview. The
allow lists are empty by default. The checker's safe class counts only with the
experimental `VADE_SHELL_CHECKER=1`. In `auto`, every command that is not
blocked runs inside the same OS sandbox without asking. Blocked commands never
run.

The read-only list below picks the sandbox only; it never skips a prompt.
These forms run in a read-only sandbox with fixed executables, with either
the bare name or the listed system path:

| Command | Fixed executable | Accepted arguments |
| --- | --- | --- |
| `pwd` | `/bin/pwd` | None |
| `true` | `/usr/bin/true` | None |
| `ls` | `/bin/ls` | Optional `-1` first, then zero or more workspace-relative paths |
| `cat` | `/bin/cat` | One or more workspace-relative paths; no options |
| `git status` | `/usr/bin/git` | No option or `--short` |
| `git diff` | `/usr/bin/git` | No option or `--stat` |
| `git log` | `/usr/bin/git` | No options, `--oneline`, or `-n N` with N from 1 to 100; options cannot be combined |

Bare names resolve to these fixed executables, never through PATH. Path
arguments may be `.`; they cannot start with `-` or contain parent traversal or
hidden or credential-related components. Other arguments do not get the read-only
sandbox. Approval is the same for every command. An executable in the workspace, shell
metacharacters, PATH tricks, a journal or an earlier approval cannot extend the
allowlist.

| Policy | Exact patch | Command on an allow list | Any other command | Blocked command |
| --- | --- | --- | --- | --- |
| `ask` (default) | Ask once | Execute after preview | Ask (`y`, `s`, `p` or `n`) | Refuse |
| `auto` (`--yes`) | Apply after preview | Execute after preview | Execute after preview | Refuse |
| `never` | Deny | Deny | Deny | Refuse |

`auto` covers workspace executables such as `./tools/rgbds/rgbasm`, system
tools, and explicit shell commands. Every command still runs in the macOS OS
sandbox: network is denied, writes stay inside the selected workspace and the
private session build/cache/temp area, and hidden, credential and excluded paths
stay out of reach. The policy does not authorize commands outside that sandbox.
This harness has no unsandboxed way to run anything; a request that needs
broader access is refused, not retried with broader permissions.

A UI approval binds to one prepared patch or one fixed argv request. Approvals
expire after 120 seconds, and cancellation or a disconnected UI denies them.
Noninteractive runs never treat piped prompts as approvals. Ask/deny is an
operational policy, not a replacement for sandbox boundaries. If the sandbox
cannot start, the command fails closed. Journals saved under the previous
version of the approval policy can still resume with their original tool
descriptions and exact prompt bytes. The policy you launch with controls
approval; saved descriptions never grant extra permissions.

Command approval previews are JSON argument arrays. Nonprinting Unicode is also
shown as visible JSON Unicode escapes, including UTF-16 surrogate pairs for
supplementary code points. Literal escapes stay distinct, and the displayed JSON
decodes to the original argv. The terminal refuses any approval preview that has
unescaped nonprinting characters, and argv and the one-time approval binding
keep the original values.

### The macOS sandbox

Commands run under macOS's built-in sandbox (Seatbelt, the system behind
`sandbox-exec`). It denies everything by default and then allows only:

- reads of the workspace and of named system and toolchain folders (for
  Homebrew, only its tool and library folders, not `etc` or `var`);
- writes to the workspace, the private session build folder and `/dev/null`;
- a few system values (sysctls): OS version, argument size, CPU count, memory
  size and page size, including `hw.pagesize_compat`, which Rust programs need
  to start;
- the one public file `/private/etc/ssl/openssl.cnf`, which Cargo needs to
  start. It grants no network access and no other `/etc` reads.

Network access and hard-link creation are denied. Commands cannot read other
processes' arguments or environment (`process-info*`, `kern.procargs*`) or the
hostname; the queries a program makes about itself at startup stay under the
normal OS policy.

How we tested this: a compiled Rust fixture probes `KERN_PROCARGS2` directly
against a parent with an otherwise-cleared environment and a synthetic marker. A
failing `ps` on its own is not enough evidence, because it may stop at a
different metadata query first.

### Toolchains and excluded folders

You can pass `--shell-toolchain DIR` at launch to pick a Rust distribution,
including one installed in a home directory. Only its canonical bin and lib
trees become readable, and they are explicitly write-denied. The required
bin/cargo, bin/rustc and bin/rustdoc files and the bin/lib directories must be
real, not symlinks, and the distribution must be separate from the workspace.

By passing it, you vouch for its public compiler and runtime contents. There
is no credential, registry or toolchain discovery, no running of rustup shims,
and no reading of the rest of the distribution or home. The option is launch
policy: never a model tool parameter, and never a capability restored from a
journal. System and Homebrew toolchains and workspace-local executables stay
subject to approval and the same sandbox.

File metadata lookup is broader, to support OS path traversal.
`/System/Volumes/Data` is deliberately not an allowed read root. Hidden and
credential-related paths are denied even inside the root. A changed
workspace-root identity is refused before launch. Workspace verification skips
`target` and `node_modules` directories at every depth, while Seatbelt denies
all reads (executable mapping included) and writes to these names and everything
below them. Removal and rename are denied too, including macOS Data-volume
aliases. So Cargo or pnpm hardlinks and large dependency trees do not break
unrelated commands, and aliases that were not inspected cannot reveal or change
files outside. A visible alias in the scanned source tree still fails the
hardlink check, and an excluded root that is itself a symlink is refused.
Hardlink creation stays denied.

You can add at most eight `--shell-exclude NAME` options at launch. Each one is
a single visible ASCII name of at most 64 bytes, using letters, digits, `_`, `-`
or `.` (no leading dot). Names match without regard to ASCII case (macOS case
aliases included) at every depth, even for newly created paths, and they also
block files with the same name, to be safe. The defaults cannot be removed, and
model arguments and journals cannot change the policy. Approval previews list
every exclusion. A workspace that sits inside a matching named component is also
inaccessible. This is deliberately not a recursive `.gitignore` parser.

The rest of the tree is scanned without following links, up to 20,000 entries;
excluded descendants use none of that scan budget. Links, special files,
unreadable entries and limit or cancellation refusals name the offending
relative path. After preflight, commands can read and write ordinary source
files, but cannot use dependencies or artifacts in excluded trees. New build
artifacts go to the private area described below. This does not grant network
access for package installs. Symlinks that a tool creates to escape still cannot
get around the sandbox.

### Session build area

Each session gets its own private build folder, so Cargo and npm can write their
output and caches without touching your workspace or home. After a command is
approved and the workspace check passes, the harness creates a brand-new
`vade-build-*` folder in the system temp directory, outside the workspace,
readable only by you (mode 0700). It never reuses an existing folder, and it
checks the folder's type, owner, mode and identity before each use. The sandbox
allows that one folder, but a command cannot rename, remove or re-permission the
folder itself, or create hard links. Because the folder starts empty and hard
links are denied, nobody can plant links to outside files in it beforehand.
Other sessions' folders stay out of reach, and `target` and `node_modules` in
the workspace stay blocked.

The environment is cleared, then rebuilt with a fixed PATH (optionally prefixed
by the selected Rust bin), a locale, and these locations owned by the harness:

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

Hidden generated files are allowed in this area, Cargo fingerprints included.
The workspace's hidden files and journals, and the credential-name rules, stay
denied. Scripts may override environment variables or arguments, but the OS
rules still deny excluded trees, unrelated reads, writes outside the allowed
roots, and all network. Neither an `env` command nor `--target-dir` can widen
these capabilities.

The area lasts across commands and prompts in one active native session, and
compaction keeps it. Exit, `/new`, resume/switch or fork selection releases it.
Cleanup is best effort and never follows a replaced root or symlink entries. An
abrupt termination or unreadable subdirectories created by a tool may leave an
orphan behind, which is never adopted automatically. There is no persistent
build cache. Between commands, the harness caps retained state at 1 GiB and
100,000 entries, without following symlinks. Going over yields `BuildLimit` and
refuses further commands for that build session; use `/new`. This is not an
instant disk quota: a running command can go over before it exits, within the
existing deadline and per-file limits. Limit scans never look at workspace
artifacts that were already there.

Our real local acceptance test runs `cargo build --offline`,
`cargo test --offline` and a repeated build through `workspace_shell` on a
temporary crate with no dependencies, a build script and a unit test. A second
fixture exercises CLI toolchain selection and interactive approval. No host
Cargo config, credentials or registry is imported. Missing dependencies and
hidden workspace config are still unsupported. npm cache and config locations
are set up, but an existing `node_modules` stays denied, and we did not run a
real Node/npm installation or any third-party package.

### Process limits

Command processes get null stdin and nonblocking captured pipes. The kernel
limits are
120 CPU seconds, 64 MiB per output file, 1,024 processes for the user as the OS
counts them, and 128 descriptors. These are inherited kernel limits, not a
per-command memory or process-tree quota. Command wall time and combined
stdout/stderr are polled every 10 ms, and each drain iteration is bounded.
Outputs keep partial data on timeout, cancellation, nonzero exit or cap
exhaustion. The native adapter uses a smaller byte capture cap, to leave room
for result framing and worst-case UTF-8 repair. The UI shows capped command
results. Local I/O failures stop the current request as `tool_failed`, and the
UI also shows the refusal or error detail, scan paths included. Inspect the
state before retrying, because execution may have started.

The process group is killed and the direct child reaped on every return and
error path. An abrupt termination of the harness cannot run process cleanup. A
descendant that deliberately changes its process group is outside the cleanup
promise, though it still inherits the OS sandbox restrictions. We make no claim
of security against a hostile process of the same user changing the workspace at
the same time.

## Journal recovery

If the harness crashes, a resume picks up from the last completed step, never
reruns a tool, and tells you plainly when a tool's outcome is unknown.

Native journals are private local transcripts, separate from the older
metadata-only `Session` store that direct, demo and Codex modes use. Before
each tool runs, the journal saves a record with the model's validated call and
a closed result for every proposed call: the one about to run is marked
`unknown_outcome`, and the other pending calls `not_executed`. That record must
reach the disk before the tool starts, and each result is saved before the
agent goes on. If the first save fails, the tool does not run. If a result save
fails, the agent stops, and a resume may show an unknown outcome. The terminal
refuses further prompts against a journal that failed.

On resume, the harness opens the private (0600) file under an exclusive lock
and checks every record: framing, order, JSON size, history and that each tool
call has its result. It drops only a half-written last record and refuses any
complete record that is damaged. Replay never runs a tool or contacts the
engine. A journal is limited to 64 MiB, 8,192 records and the usual 4 MiB of
active history. `/new` starts a separate session instead of appending a reset
that could erase the earlier one. `/compact` appends a replacement, and the
original records stay on disk. There is no encryption and no scanner for
credentials in the content, so do not put credentials in prompts or in
workspace files that are sent to the model.

File format: each record is an eight-byte little-endian length, the JSON
payload, and the same eight-byte length again as a commit trailer, with no
custom cryptography or checksum. A record holds a version, a sequence number,
how many earlier messages it keeps, and the tail it appends or replaces. Only
the first record holds the stable request options. Writes append and fsync.
Later revisions of a tool stage replace only the logical tail during replay,
and the old records stay on disk.

## Context accounting

Without an exact count from the engine, the harness estimates how much of the
context a request uses. The input estimate and the reserve come from the
context estimator ([`src/context.rs`](../src/context.rs)), and the output caps
from `AUTOMATIC_OUTPUT_TOKENS` and `DEFAULT_OUTPUT_TOKENS` in
[`src/contract.rs`](../src/contract.rs):

- **Input.** Every serialized part of the request at three UTF-8 JSON bytes per
  token, plus 256 + 16 per message + 32 per tool for framing, plus a 12.5 %
  safety margin (at least 256).
- **Output reserve.** 4,096 tokens by default (`DEFAULT_RESERVE`).
  `--max-output-tokens` replaces it, even with a smaller value.
- **Automatic output cap.** Once the client has picked one for a request (at
  most 16,384 tokens, or 8,192 without advertised capabilities), the estimate
  reserves that cap instead. It never reserves less than the smallest of the
  advertised output limit, 8,192 tokens and a quarter of the context.

Negotiated defaults respect the advertised output limit, and explicit request
output limits are counted too. Native and direct requests use the negotiated
output defaults unless the caller picks an output limit.

These are heuristics, not exact counts or an upper bound that holds for every
model. The byte figure is the actual serialized request size. Cached input is
never subtracted. Media and unknown content parts are refused without exact
provider admission.

When the engine offers tokenizer admission (an exact count of the rendered
request), that count drives the native fit check before generation and the
automatic compaction thresholds; the candidate must reduce actual input tokens
and fit before it is saved. Without that capability, labelled estimates still
apply. The runtime checks before every model call and before tool execution.
Large results become labelled excerpts, backed by the exact captured bytes in
capped private journal artifacts. `tool_output_read` gives bounded UTF-8 ranges,
and `/output` inspects them locally. A result that exceeds the remaining
estimated context stays durable and closed, and the next model request stops.
Hard byte and JSON limits stay independent. There is no automatic identical
retry.

`/compact-preview KEEP SUMMARY` shows a preview. `/compact` deliberately
replaces old standalone assistant prose, while keeping every user message,
instruction, tool pair and receipt, the last meaningful answer, and recent
turns. `/compaction` and `/journal` show the exact summaries and source records.
The operation must reduce bytes and estimated tokens and still fit admission,
and unknown tool outcomes are refused. Raw records are kept. A changed middle
invalidates all later KV, including unchanged tail text. The frozen system/tools
wire prefix stays exact, and the summary comes right after it.
`--prewarm-compaction` can send one discarded token on a cancellable worker; it
runs no tools and leaves raw records unchanged. Actual cache reuse and cost need
upstream response telemetry.

See [context, project rules, skills and transcript
branches](context-and-projects.md) for exact storage limits, the catalog format,
snapshot and revision semantics, and local inspection commands. All project
tools are read-only data operations: no skill scripts or installers run
automatically, and project text cannot grant tool permissions. Fork and switch
change transcripts, never the files in the workspace.

## Project memory tools

`memory_save {key, text}` saves or replaces a durable project note.
`memory_forget {key}` removes it. Each returns the entry count after the change.
Keep user facts and preferences, decisions, and corrections from the user.
Do not save code structure, repo facts, secrets, or one-off details.
These tools are on in native chat unless memory is off. They write private
notes without a workspace edit approval. Cards and library tools keep them
off. See [project memory](memory.md) for the file format and switches.

## Conceptual references

The harness README credits Codex, Pi and Hermes for high-level coding-agent and
terminal interaction concepts. The familiar apply-patch marker vocabulary is an
interface idea, implemented here without copying an implementation. macOS's
`sandbox-exec(1)` facility and installed OS sandbox profile declarations were
inspected to understand the platform interface and dynamic-loader requirements;
this project's minimal profile is first-party. We did this without contacting
any engine or provider.
