<p align="center"><img src="docs/assets/vade-harness-banner.svg" width="100%" alt="VADE Harness: a coding-agent harness, written from scratch in Rust"></p>

# Vade Harness

A first-party Rust terminal harness with no third-party crates. Inference and KV
state belong to the [engine workspace](../engine/README.md) in this monorepo.

The UI provides a native **tool-aware agent**, an offline demo, direct streaming
**text chat**, a gateway command, and optional delegation to an installed Codex
app-server or Hermes ACP v1 agent. Native agent mode runs the bounded runtime with read/list, exact patch edits,
and approved sandboxed commands inside an explicitly selected workspace. Model-requested tool results are
sent to the selected model. Direct chat keeps its manual local tools.
Validation scope and historical measurement references are documented below.

The native harness remains minimal: automatic limits, greedy/cache-friendly
requests and bug fixes. The experimental web, PDF/image-text and launcher/model-
discovery additions have been removed. The separate gateway provides client integration.
See the [client proof and setup guide](docs/passthrough-matrix.md) for compatible API clients.

The harness requests caching when the engine supports it; current FP4-Local
retention remains off by default. See [cache defaults and measured tradeoffs](docs/cache-defaults.md)
for the request/server distinction and the evidence behind that recommendation.

For clients that change only their base URL, build the separate
[`vade-gateway` binary](docs/invisible-gateway.md). It forwards Chat Completions,
Responses and arbitrary HTTP paths, with optional stats, recording, replay and
explicit optimization flags. Repeat `--upstream` for **replicas**: independent
servers of the same one-Spark model (for example from `vade up --replicas 2`).
Each request goes to the upstream with the fewest in-flight requests, ties
rotate, and requests whose `messages` (or Responses `input` array) share their
first 2048 bytes through the last user message prefer the upstream that served
the previous one while it is among the least busy (`--no-sticky` turns that
off); FP4-Local prefix retention is OFF by default, so cache hits from this are
unverified. `GET /health` then reports
every upstream and fails when their model IDs differ. A failed upstream is
retried on another one only while nothing has been streamed to the client.
Use replicas for a model that fits on one Spark and many concurrent chats; use
the multi-Spark chain when the model is too large for one Spark. Identical
replicas return identical token IDs, so routing never changes an answer.
The [historical measurements](docs/gateway-measurements.md)
record exact greedy answer comparisons and measured latency limits. The
[historical gateway card](docs/harness-card.md) measures installed Codex directly
and through this gateway at 10k, 50k and 100k prompt tokens.

## Validation scope

The doctor checks advertised capabilities, admission and response telemetry.
A server declaring `prefix_cache:false` does not fail solely for absent cache
reuse. If caching is advertised, missing or zero reuse fails the cache check;
malformed telemetry and other probe failures still fail.

Historical summaries and their limits remain in the linked gateway measurement
records. Host fixtures exercise the protocol and tools; they do not establish
current native model performance or a current installed-client qualification.

## Four ways to run Vade

Examples assume a separately started HTTP Engine at `http://127.0.0.1:18080`;
replace `MODEL` with its exact model ID. These are usage examples, not live tests.

For the FP4-Local server installed by [INSTALL.md](../INSTALL.md), follow its
[Route B controller commands](../INSTALL.md#5-start-the-two-spark-server) and use
`qwen3.8-flash-next-fp4-local`. The engine, and on a controller its tunnel,
stay up until `vade down`; direct and gateway clients connect in between. The [legacy FP8 connection guide](../engine/docs/flash-server.md#run-the-native-coding-agent)
uses its separate model ID and profile.

1. **Vade Engine + Vade Harness:** native UI and bounded workspace tools.
   `vade-harness agent --endpoint http://127.0.0.1:18080 --model MODEL --workspace .`
2. **Your harness through Vade's gateway:** point your client's provider at port 8787.
   `vade-gateway --upstream http://127.0.0.1:18080/v1` (as in INSTALL) forwards every
   path, including `GET /v1/models`. `vade-harness gateway --upstream http://127.0.0.1:18080`
   serves only the API routes listed under [Two bridge directions](#two-bridge-directions)
   and answers 404 to `/v1/models`, so prefer `vade-gateway` for clients that list models.
3. **Vade UI delegating to Codex/Hermes:** the installed agent owns the turn.
   `vade-harness codex --workspace . --permission read-only`
   For Hermes, use `hermes` mode and `--permission deny-callbacks` (not an OS sandbox).
4. **Vade Engine alone with any compatible client:** no Harness process required.
   `curl -q http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MODEL","messages":[{"role":"user","content":"Hello"}]}'`
   FP4-Local thinks by default (`default_thinking: true` in `/v1/capabilities`), so a plain
   client first receives `reasoning_content`, then `content`. Add `"enable_thinking": false`
   to the request for the answer alone; the Vade harness sends that unless thinking is on.

`vade-harness setup codex --endpoint http://127.0.0.1:18080 --model MODEL`
prints both direct and gateway configurations; also supports `hermes`, `opencode`,
`pi`, and `generic`. It is offline and writes only with `--write PATH` and a
selected `--target direct|gateway`. Codex requires Responses support on the
upstream; the gateway does not translate APIs.
For a complete isolated configuration, add `--dedicated-home --write DIR`,
`--context-window N` and `--max-output-tokens N` using verified upstream limits.
Setup creates a new private directory with the client's own instruction file and
`launch.sh`; the launcher keeps HOME, XDG and client state inside that directory.
Existing homes are refused. Run the launcher from the workspace you want the
client to use. See the [client matrix and launch commands](docs/passthrough-matrix.md).

Vade Engine serves `/v1/responses`, so Codex works directly and through the gateway.
The generated Codex home selects Codex's read-only sandbox; for a task that edits files
and runs commands, pass `-s workspace-write`, for example
`/path/to/home/launch.sh exec -s workspace-write --json "Add a test and run cargo test"`.
The launcher clears the environment and gives Codex a private HOME, so rustup's
`~/.cargo/bin` shims cannot find the owner's toolchains there (they try to download a new
one). Put the toolchain's own `bin` directory first on PATH when launching, for example
`PATH="$HOME/.rustup/toolchains/1.99.0-aarch64-apple-darwin/bin:$PATH" .../launch.sh exec ...`.
Setup has no gateway token option. With `vade-gateway --token` (or `VADE_GATEWAY_TOKEN`),
pass the token to Codex as
`-c 'model_providers.vade.experimental_bearer_token="TOKEN"'`; without it Codex stops
with `401 Unauthorized: gateway token required`. The token is then visible in the process list.
`vade-harness doctor http://127.0.0.1:18080` actively checks capabilities, admission
and two synthetic chat turns. Gateway stats require its response marker; the
cache prefix exceeds two advertised cache blocks, or 8192 tokens by default,
verified with admission when available. It prints turn-2 reuse as a percentage;
only reported reuse passes the cache check.
See [setup and doctor details, sources, and mock coverage](docs/setup-doctor.md).

The harness has no Claude Code delegation or setup mode. For supported client
setup commands and their validation scope, use the
[client matrix](docs/passthrough-matrix.md). The CLI's separate single-Spark
workflow prints a Claude Code environment command, but live Claude Code chat and
tool compatibility remain unqualified there; see the [CLI guide](../cli/README.md).

## Run

Attach workspace text files with `@path` (quote or backslash-escape spaces), or
drop/paste an absolute file path as the whole input. Absolute paths inside logs
stay text. Tab completes a simple `@path` common prefix; a second Tab lists matches
from a bounded directory listing. Names must be visible ASCII; symlinks, binary
files, images and PDFs cannot be attached. Messages including files are capped at
64 KiB. Refusals keep the draft; recall stores typed names, while the journal saves
the exact file bytes once. Clipboard images and PDF extraction are not supported.

From `harness/` in this monorepo, use the pinned Rust `1.99.0` compiler:

```sh
cargo build --offline
cargo run --offline -- demo
cargo run --offline -- --demo --prompt "Show me the local demo"
cargo run --offline -- --help
```

No mode argument defaults to **demo**, which produces a clearly labeled scripted
response and makes no network requests. Direct mode requires both an endpoint and
a model. Native agent mode also requires an explicit workspace. These commands
illustrate configuration for a separately provided mock; they do not start a mock
or connect automatically during setup:

```sh
vade-harness direct --endpoint http://127.0.0.1:18080/v1/chat/completions --model mock
vade-harness agent --endpoint http://127.0.0.1:18080/v1/chat/completions --model mock --workspace .
vade-harness gateway --upstream http://127.0.0.1:18080 --bind 127.0.0.1:8787
```

`vade-harness workload --endpoint URL --model NAME --json FILE` measures concurrent
native sessions with deterministic synthetic observations and growing histories.
Defaults are one main plus five subordinate sessions, eight turns each; no real
tools run. It prints a role/session table and writes a new private JSON metrics
file. Turn-2-onward results include per-role cache ratios, TTFT percentiles,
output throughput, admissions and overload/errors. It also prints every turn's
C/U and count basis. Exact admission separately counts the retained history with
the full assistant reply to avoid using generated-output counts as rendered
input; the token-LCP denominator remains explicitly **estimated** because
the current contract does not expose matching token IDs. See the
[workload guide and coordinator command](docs/workload.md) before interpreting the
0.98 target. `workload --help` lists size, concurrency, output and deadline flags.

Repeat `--upstream URL` to enable **cache-aware cluster routing** across compatible
endpoints. The gateway discovers each endpoint's capabilities, retains bounded
conversation affinity, and uses least-loaded compatible fallback for unhealthy
or saturated endpoints. Before forwarding any response bytes, a fresh connection
failure or a 429/503 refusal permits one retry on another verified healthy,
compatible upstream. Explicit payload fields retain their bytes. Once response forwarding
starts, there is no replay; write/read failures after dispatch are also not retried.
Cluster mode only (multiple `--upstream` options): `GET /_vade/gateway/stats`
exposes per-endpoint requests, failures and reported cached/prompt-token ratios.
The single-upstream relay does not expose this route (HTTP 404).
See [routing configuration, mock evidence and
limits](docs/gateway-routing.md). A single upstream retains the original relay.

Both `vade-gateway` and `vade-harness gateway` answer 421 when Host is not
`127.0.0.1`, `localhost` or `[::1]`, bare or with the listener port, and 403 when a request
carries any `Origin` header, so browser pages and DNS rebinding cannot reach the
engine through them. To allow a browser client deliberately, pass its exact origin
with `--allow-origin` (repeatable; it then gets CORS headers and preflight answers,
and `--allow-host` admits another Host name) and set `--token TOKEN` or
`VADE_GATEWAY_TOKEN` so every request needs `Authorization: Bearer TOKEN`, which is
not forwarded upstream.

The gateway defaults to `127.0.0.1:8787`. The core owns HTTP routing, forwarding,
response parsing, and upstream timeouts. The CLI refuses HTTPS explicitly: there
is no TLS implementation or custom cryptography. URL credentials, query strings,
fragments, and control characters are rejected. Direct and native clients accept
`--api-key-file PATH` or `VADE_API_KEY_FILE` for Bearer authentication.

When capabilities advertise `prefix_cache` or the `scheduling_hint` cache key, agent, direct, child and doctor
requests default to `"cache": true` and retain `prompt_cache_key`. Caller-supplied
`cache` values are preserved. Set `VADE_HARNESS_PREFIX_CACHE=0` to send
`"cache": false` instead, including in children and doctor.

Direct/native workers negotiate the optional **vade.engine.v1** contract before
their first model request. Advertised exact admission checks the complete final
request and output reservation before every generation, including tool rounds.
Without the contract, requests use ordinary chat behavior and explicitly labeled
estimates. A generated session cache key is sent only when its scheduling-hint
feature is advertised; affinity alone never implies a hit.
Per-response input/output and actual reported cached-token counts appear after
each run and in **/context**. Missing telemetry stays unknown. See the
[capability, admission and gateway contract](docs/engine-contract.md) for schema,
fallback behavior, limits and coordinator integration.

A fresh direct or native session without sampling flags sends no sampling fields.
Vade Engine applies its shared greedy default; another upstream retains its own
default policy. An internal retry counter does not silently opt into sampling.
`/retry` keeps this absence intact, so a greedy answer may repeat. The CLI never
invents or increments a seed: `--temperature` and `--greedy` without `--seed`
leave the seed unspecified. An explicit or saved seed stays unchanged across
retry, resume, tool continuations and ordinary follow-ups. To choose a different
seed, restart with a different `--seed`; `/retry` reminds you when one is pinned.

New default native journals remain unseeded across retry and resume. Existing
journals with a recorded seed retain it, including older sessions that recorded
the former implicit seed. Use `/new` or a new session to adopt the no-field policy;
other generation flags remain per-launch settings.

Direct and native agent modes accept `--greedy`: every model request uses
`temperature: 0`, including tool turns and continuations. This permits strict
greedy drafting on engines that support it; drafting preserves the plain greedy
answer bytes. It does not enable sampled drafting. `--temperature` and `--greedy`
are mutually exclusive. The per-reply footer reports cache reuse as restored
prompt tokens / prompt tokens, summed across that reply's model calls. Missing
telemetry remains unknown. System instructions, tool definitions, ordered history
and the per-conversation cache key stay stable across ordinary turns.

Direct and native agent modes accept `--max-output-tokens N` and
`--thinking on|off`. The output budget must be positive and no greater than the
Engine's advertised output limit; an oversized budget is refused before
admission or generation. Without an advertised limit, the upstream validates
the upper bound. By default, each reply reserves the advertised output limit or
the remaining context space minus 256 safety tokens, whichever is smaller.
An explicit `--max-output-tokens` keeps the chosen reservation and must still fit
the context. Explicit thinking flags override `VADE_HARNESS_THINKING=1`;
thinking defaults off. Explicit output and thinking settings remain fixed for the
launched session. When the Engine advertises thinking, the harness sends
`enable_thinking: false` unless thinking is turned on. With thinking on and no
effort, the Engine uses the model's default effort, which is `xhigh` for
Qwen3.8 and can spend most of a reply on hidden reasoning. `--effort
low|medium|high|xhigh` (or `VADE_HARNESS_EFFORT`) sends that `reasoning_effort`
and turns thinking on; it cannot be combined with `--thinking off`.

Direct chat and the native agent automatically continue answers cut off at the
output limit. Each continuation sends the same user message as `/continue`,
keeps the seed and prior answer, and shows a short notice. Use
`--no-auto-continue` to wait for a manual `/continue` instead. At most 32
consecutive continuations are sent for one answer; they share the original
request's active-work budget, which defaults to 20 minutes. Native continuations
also keep the original 61-model-turn, 60-tool-round and 240-proposed-tool-call
limits. A saved answer remains visible when a cutoff reaches one of those
limits; a manual `/continue` starts a fresh work budget with the prior
conversation and seed. The native JSON config accepts `"auto_continue": false`
for the same opt-out. If context cannot fit another reply, native mode tries its
existing in-task compaction and direct chat stops with a plain message.
Continuations stay in the exact conversation; native journals retain them too.

A native run never ends silently. A finished reply with no visible text and no
tool call (hidden reasoning only) gets a short recorded reminder, at most twice
in a row; a third empty reply stops the run with a notice saying why. Reminders
use the same request limits.

`--check COMMAND` (or the first nonblank, non-`#` line of the workspace file
`.vade/check`; `--no-check` ignores the file) closes the build loop. After any
tool round whose patches changed files, the harness runs `/bin/sh -c COMMAND` as
a journaled `workspace_shell` call with ID prefix `vade-check-`, under the same
approval policy and macOS sandbox (no prompt in `--approval auto`). The model
receives the exit status, matching error lines first and then the last lines,
within about 3.5 KiB; longer raw output stays in a session artifact. A final
answer while the latest check fails gets up to two reminders. The model also sees
a note when it reruns an identical failing command without changing files, or
applies a patch with more than 150 changed lines.

Each native run prints one speed summary; `/status` shows per-response TTFT,
output tok/s, and prompt/cached tokens. Direct mode reports each validated response.
Decode speed uses `(output tokens - 1) / (last generated delta - first generated
delta)` from the local SSE stream, including reasoning and tool function data.
Usage trailers and response completion waits are excluded. Buffered chunks limit
timing precision; JSON, missing usage, or a single generated delta cannot yield
a decode rate. `/status` includes mean TTFT and a decode rate weighted by decode
time across the session. Missing measurements stay unknown.

## Native coding agent

For two-Spark Flash-Next, start with the
[Mac connection and run recipe](../engine/docs/flash-server.md#run-the-native-coding-agent).
The tools below execute in the selected Mac workspace while inference runs on
the Sparks.

`agent` (or `--mode agent`) invokes `transport::NativeModel` through the bounded
`agent` runtime. It requires `--endpoint`, `--model`, and **explicit `--workspace`**.
Tools are `workspace_read`, `workspace_list`, `apply_patch`, `workspace_shell`,
`tool_output_read`, `skill_read`, `workspace_manifest`, `workspace_symbols`,
`workspace_search`, and `workspace_read_lines`.
Start discovery with `workspace_search` (`path`, `literal` or related `literals`); it returns relative
paths and line numbers in a complete inline JSON result. Follow with
`workspace_read_lines` (`path`, `start_line`, `end_line`, or disjoint `ranges`) for exact context.
Search is literal, case-sensitive, capped at 512 files/8 MiB per call, and skips
hidden, credential, link, and dependency/build paths. Pagination is a live view;
copy `next_offset`, or narrow the path when the scan reports omitted input.
Line reads select up to 200 inclusive lines from a source up to 4 MiB, with
path/range provenance and a 4 KiB inline budget.
Partial patch results carry affected paths, committed namespace operations and
`uncertain_writes`; the parent stops further calls and preserves that receipt in
the journal. Inspect current state before resuming a side effect.
Tool results return to the selected model with their original call IDs. Text
deltas remain provisional; only validated full messages enter typed history.
Reasoning-only or empty replies without tool calls receive `content: ""` before
entering history. Reasoning stays in its upstream field, including
`reasoning_content`; subsequent requests preserve those serialized bytes.

Length-limited replies remain visible and are saved as schema-valid assistant
messages. Cutoff metadata stays in the private journal, outside upstream requests,
and is shown again on resume. Cutoff notices identify the limit and show the
output-token count when available. The agent continues ordinary cut-off answer text
automatically, within the same round, model-turn and active-work limits. A reply
cut off inside a tool proposal stops with a plain message: the incomplete call
is discarded and never executed. A stopped turn pauses queued work until an
explicit follow-up or `/next`. Terminal status ticks update in place;
plain interactive status prints state changes until reply text starts, then stays
silent for that run. Notices finish the current text line, and redirected output
contains no ticking status lines.

After dispatch, native generation waits show **queued / awaiting upstream** until
generated text, reasoning, or tool data arrives. Reasoning streams show **thinking**;
the raw UI starts reasoning collapsed behind a labelled Thinking row. Ctrl+T
expands or collapses it; the reply stays visible. Token counts use provider
reasoning usage when supplied, otherwise a clearly marked estimate. Visible
answer/tool data switches to **generating**. Leading blank lines from hidden
reasoning are suppressed only in the display; transcript/wire content stays exact.
This is a local waiting label,
not an Engine queue-position receipt. Silent queue/prefill waits retain the
connection up to the 3,600-second HTTP lifetime, with cancellation polled every
250 ms. Native generation response reads use that lifetime instead of the
ordinary 600-second idle timeout. The request's active-work budget and Esc can
cancel sooner, as can native child/workload deadlines. Discovery, admission,
request writes and direct chat use the ordinary idle timeout; gateway generation
responses also allow the full HTTP lifetime. There is no automatic replay.

`apply_patch` adds, updates, deletes, and moves bounded UTF-8 files. It requires
exact, unique old text and existing parent directories; there is no fuzzy match.
Patch text is limited to 64 KiB; each target and result may be up to 4 MiB.
Native `workspace_read` accepts zero-based byte `offset` and byte `limit` (1–4096)
for sources up to 4 MiB. Partial results identify the actual range, total size and next
offset. Reads retain the hidden/credential path and directory-handle protections.
It preflights all files and displays unified diff hunks with three context lines
before approval. Overlapping context is merged. Nonprinting Unicode, tabs,
carriage returns, and literal backslashes are visibly escaped. The 192 KiB cap
counts the escaped display bytes; an oversized diff is refused before approval.
Updates stage a same-directory temporary file, sync it, recheck the original
metadata and bytes, then atomically rename. Adds and moves never replace existing
destinations. Delete sections must contain the entire exact old text. Existing
permission bits are preserved. Atomicity is **per file**, not a batch transaction;
a combined move+edit can leave both paths if interrupted between installation and
source removal. Partial results and conflicts are explicit. See
[tool grammar and limits](docs/agent-tools.md).

**Native children are opt-in:** add `--children`, or enter `/children on` before
the first prompt of a fresh native conversation. This adds `child_spawn`,
`child_list`, `child_wait`, and `child_cancel` for the parent. Children use the
same endpoint, model, workspace boundary, contract client and native loop, with
fresh task histories and independent cache affinity. Children default to the six
native read/context tools and cannot spawn children. Exact admission is required
for child generation; ordinary chat fallback remains available to the parent only.

Host launch flags can declare named scopes: `--child-edit edits=src/example.rs`
leases one exact file to the scope `edits`; repeat for each file, including both
ends of a move. `--child-shell checks` explicitly grants sandboxed shell commands
to scope `checks`, with an exclusive workspace write lease. The model can select
an existing scope through `child_spawn`'s `grant`, but cannot create or extend it.
Uncollected write leases block conflicting parent and sibling writes. Every child
edit/command still follows the parent's `--approval` policy; noninteractive runs
deny operations that need approval. Approval prompts are serialized and identify
the child; cancelling a child clears its pending approval. Shell is never
inherited merely because the parent advertises it. See
[scope examples and limits](docs/child-agents.md).

The raw TUI shows a live, size-bounded child list. `/children` lists IDs and
actual states; `/children cancel ID` requests cancellation of one child, while
`/children cancel all` also closes further spawns for the current activation.
These commands leave the parent running so it can collect results and integrate
findings. `/cancel`, exit, or parent completion cancels and joins remaining
children. Cancellation reports local cleanup, never confirmed remote compute stop.
`/context` includes each child's last reported cached-token count and cumulative
reported counts, separately from token reservations; absent telemetry is unknown.

Child tool schemas stay frozen with the parent prefix. To change the setting
after a prompt, wait for cleanup, use `/new`, then `/children on` or `/children
off`. `/new` retains the current setting. Each activation has fresh finite child
budgets; the UI retains only the current/last activation's child metadata and
in-memory results. Parent tool receipts remain in its typed journal. Native children
have separate private journals and build areas; `child_wait` returns the child
journal name for inspection. Child scheduler state itself is not restored or
restarted after a crash. Resume requires the same explicit child flags/scopes;
stored descriptions never grant permission. See the
[child lifecycle, quotas and integration notes](docs/child-agents.md).

`workspace_shell` takes explicit `argv` and `timeout_ms`; there is no implicit
shell. A shell script requires an explicit `/bin/sh -c` request and follows the
selected approval policy.
On macOS it runs through the OS `sandbox-exec` facility with a cleared environment,
workspace cwd, system/toolchain reads, workspace and private build-area writes,
and no network. Workspace hidden
paths, common credential files, and hardlink creation are denied. Preflight skips
`target/` and `node_modules/` at every depth; the sandbox denies reads, execution,
writes, removal and rename of these excluded trees. Unscanned hard links cannot
expose or change outside files. Repeated `--shell-exclude NAME` adds up to eight
visible directory names to this policy; the command preview lists them.
Other visible symlinks, hardlinks, special files, and scans above 20,000 entries
still fail closed, with the offending relative path. Excluded descendants do not
consume the scan budget. Existing build output/dependencies in excluded trees
remain inaccessible.

Approved commands get a fresh harness-owned **session build area**, created
outside the workspace with mode 0700. Cargo target/home, temporary files, HOME,
XDG caches/config and npm cache/config use this area; Cargo/npm default to offline.
Generated hidden files there are allowed. Builds reuse this area across prompts;
exit, `/new`, resume/switch and fork selection discard it with best-effort cleanup.
Compaction keeps it. No old cache is adopted or restored from a journal. A 1 GiB /
100,000-entry retained-state limit is checked between commands; it is not an
instantaneous disk quota. Existing process, time, output and per-file limits apply.

System/Homebrew tools remain available. For a Rust distribution installed under
the owner's home, explicitly launch with `--shell-toolchain DIR`, where DIR is the
installed distribution root containing real `bin/cargo`, `bin/rustc`,
`bin/rustdoc` and `lib`. Only its bin/lib trees are readable, always read-only;
the owner must select a trusted distribution of public compiler/runtime files.
Rustup shims, Cargo credentials, registry caches and other home contents are not
imported or discovered. Ordinary `cargo build --offline` and `cargo test --offline`
requests ask in `ask` and run in the sandbox without asking in `auto`.
Local std-only crates and their build scripts
are verified; missing dependencies still fail offline. npm cache writes are
supported, but pre-existing `node_modules` remains inaccessible and real npm
execution has not been tested. This does not enable package installation.
Argument/environment overrides cannot widen the OS filesystem or network policy.
There is no unsandboxed fallback. Other platforms refuse command execution.

`--approval ask` is the default: edits and commands outside the
[read-only allowlist](docs/agent-tools.md#commands-and-approvals) wait for `y` or
`n` on an empty line (or `/approve` / `/deny`) in the terminal. Allowlisted forms
of `pwd`, `true`, `ls`, `cat`, `git status`, `git diff` and `git log` run without
asking. The patch preview shows additions and removals in colour. A `y`/`n`
typed with a draft in the editor, during `/paste`, or within 400 ms of the prompt
appearing is kept as text. `/approve` and `/deny` answer the displayed operation
once, exactly like y/n; sending them never authorizes a later operation.
With a draft already in the input, type `/approve` or `/deny` on its own final
line and press Enter; the earlier draft stays. Pasted or recalled multiline
text cannot trigger this final-line shortcut.
`--approval auto` permits validated exact patches and all validated commands
inside the existing macOS OS sandbox after preview, including workspace programs
such as `./tools/rgbds/rgbasm` and explicit `/bin/sh -c` commands. Writes remain
limited to the workspace and private session build/cache/temp area; network,
hidden paths, credentials and excluded trees remain denied. Auto never grants
execution outside that sandbox. No unsandboxed route exists: broader access is
refused, and unsupported platforms fail closed. `ask` keeps its existing behavior.
`--approval never` denies both writes and commands. The active policy, command,
deadline, and patch diff are shown in the UI. Noninteractive/one-shot requests
that need approval are denied; approvals expire after 120 seconds and never
survive a session resume.

Commands have a maximum 120-second deadline, at most 64 KiB displayed output
(the native adapter reserves result framing/UTF-8 overhead), null stdin, bounded
pipes, and kernel CPU/file-size/process/descriptor limits. Output truncation
retains bounded head/tail diagnostics, drains both pipes and lets the command
reach an exact exit receipt. Observed/omitted byte counts and elapsed time are
explicit; discarded middle bytes are unavailable. Cancellation, timeout,
and normal child exit kill the owned process group and reap the
child. This does not promise cleanup of a hostile descendant that deliberately
escapes that group, or immediate interruption of a blocked local filesystem call.
Tools assume no hostile concurrent mutation of the workspace by another process;
optimistic edit conflict checks cannot provide an OS-level compare-and-swap with
an unrelated writer. The sandbox is a platform interface, not custom isolation or
cryptography. No GPU, Engine, Spark, installed Codex, or live-model test was run.

Native sessions are journaled by default under the private
`VADE_HOME/projects/NAME-HASH/sessions` directory (`~/.vade` without `VADE_HOME`),
one per workspace path, never inside the workspace; input recall sits beside it in
`history`. `--vade-home DIR` picks another root; `--state-in-project` keeps the
legacy `WORKSPACE/.vade` folder, which then receives a `.gitignore` of `*`
(except `check`) so git ignores it. Every state folder gets that `.gitignore`.
Journals written to `WORKSPACE/.vade` by older versions stay there; resume them
with `--state-in-project`.
`--session NAME` creates a named session; otherwise the name is generated.
`--resume NAME` explicitly restores a past session; the model and tool definitions
must match; new journals also bind to the workspace directory's device/inode
identity (no absolute path). `/sessions` lists sessions, `/resume NAME` or
`/switch NAME` switches after cleanup, and
`/new` creates a fresh journal without deleting old sessions. `/undo` remains
unavailable for tool histories. Resume never sends a prompt or replays a tool by
itself. Policy and endpoint come from the new launch, never from stored approvals.

`/fork NAME [TASK_LABEL]` creates and selects a journal-only branch while idle.
It copies the bounded raw journal so inherited artifact IDs, expiry, summaries
and receipts remain inspectable. Atomic exclusive installation never overwrites
an existing destination. Parent records remain available; subsequent messages
are independent. The label is bookkeeping, not a submitted model prompt.
`/branch` displays parent checkpoint, context generation and bounded metadata
checks for up to 128 explicitly referenced paths. These checks are not content
verification or a complete repository snapshot; commands may change untracked
files. Resume/fork appends a revalidation notice before any future model request.
**Branches share the workspace: switching never restores files or replays tools.**
Unknown tool outcomes block new forks. When the negotiated Engine advertises
`kv_branch_handles`, native requests ask to retain their canonical input. Private,
expiring hints travel with the journal; forks and switches send a matching hint
alongside the **complete prompt**. Expired, evicted, foreign, or mismatched handles
fall back normally. `/context` reports named-handle reuse separately from ordinary
cached tokens. Shared handle ownership survives branch switching; session shutdown
attempts bounded releases. Without the capability, wire behavior is unchanged.

The current Engine contract checkpoints the submitted **input**, not generated
answers or a selected system-only prefix. Exact prefix/completed-answer checkpoints
and prefix-only children need an Engine extension; no hidden warm-up inference is
performed. See [checkpoint boundaries, quotas, lifecycle and child integration](docs/engine-checkpoints.md).
Forks reset the append-only request and contract client; the next explicit prompt
negotiates and admits the complete transcript. Only response telemetry proves reuse.
A fork may copy up to the journal's 64 MiB cap, including private transcript
content; there is no
copy-on-write disk sharing or global quota across manually created sessions.

The append-only journal keeps exact typed messages, reasoning, tool arguments,
results, and stable model/tool options. It does not store endpoints, inherited
environment, or automatic workspace-path metadata. Explicit prompt/file/command
content can contain sensitive information; journals are private transcripts, not
a secret scanner or encrypted storage. Direct/demo/Codex retain their separate
metadata-counter behavior. Locks are kernel-owned and release on process exit;
files use 0600 and owned directories use 0700, with redirects refused.

Before a tool can run, a synced record marks its outcome unknown. Completed
results are synced before continuing. On a crash, explicit resume discards only
an incomplete final record and restores the last closed call/result history.
An `unknown_outcome` requires inspecting the workspace before any repeated side
effect. Complete corrupt records are refused. Journal failure stops execution;
no further prompt proceeds against a failed journal. Journals are capped at
64 MiB/8,192 records; start a new session when full. The journal preserves old
records even after deliberate context compaction.

`/context` shows per-message and tool/options estimates plus measured request
bytes and the last negotiated tokenizer admission. Local estimation counts all serialized message fields (including reasoning,
arguments, text attachments and provider extensions), immutable rules/tools,
request syntax, a provider-framing allowance, and a safety margin. The heuristic
uses three UTF-8 JSON bytes per token, adds 256 + 16/message + 32/tool framing
units and a 12.5% (minimum 256) margin. **These are estimates, not a tokenizer or a
guaranteed upper bound.** Media/unknown content-part admission is refused because
small URLs cannot estimate rendered image/audio costs. Cached input is never
subtracted. These local estimates remain independent of the contract client's
last-request exact admission and actual reported usage, also shown by `/context`.
Provider token counts and hidden framing remain unknown without that telemetry.

`--context-tokens N` overrides the server context budget. Without the flag,
`/v1/capabilities` supplies the context limit (for example 262,144); only servers
without that contract fall back to 131,072. The default output
reservation uses as much space as fits within the advertised output limit and
remaining context, leaving 256 safety tokens. Without an advertised limit, the
output ceiling is 8,192 tokens and available context uses a labeled estimate.
`--max-output-tokens` replaces
that reservation, including with a smaller value. The UI
warns at 80%; the runtime rechecks before every model request and tool batch.
Automatic context rewrites are visible and journaled. The reservation also sets
the provider generation limit. Native workers negotiate before their context decision;
when `/v1/admission` is advertised, its real rendered input count and actual output
reservation drive the 80%/60% compaction thresholds and pre-generation fit check.
The complete candidate is counted before automatic compaction is persisted; it
must reduce real input tokens and fit both the local and advertised context limits.
Unknown/foreign contracts retain the labeled heuristic. Byte/structure limits and
the independent tool-receipt safety guards still apply. No local tokenizer is loaded.

`/compact-preview KEEP_TURNS SUMMARY` shows an exact proposed compaction without
changing the session. `/compact KEEP_TURNS SUMMARY` explicitly applies it; this
first pass summarizes only older assistant prose. A bare `/compact` (or
`/compact-preview`) applies the same bounded-excerpt plan that automatic
compaction uses, so no summary has to be typed. It can condense older closed tool
rounds inside a single task. Both automatic planners keep at least three recent
tool rounds, exact goals/rules, the latest prose answer, the latest patch attempt
and applied edit per path, the latest command/check attempt and latest executed
receipt per distinct argv or check profile, and uncertain groups. A refused
repeat cannot displace the executed receipt. Its generated task summary
folds forward within 12 KiB with path/range provenance and explicit omissions.
The older-turn fallback also keeps two recent user turns; multiple user turns
do not relax the shared retention rules. If these protected facts prevent
reduced input and a fitting replacement, compaction refuses. An actual admission
overflow attempts another plan even while optional compaction is above its
60% low watermark. With a supplied summary, every original user message,
system/developer instruction, tool call/result pair, the last answer, and newest
KEEP turns remain verbatim. Keep the important assistant decisions in your supplied
summary: the harness cannot verify semantic completeness. No summarizer model is
called. Empty, nonreducing, over-budget summaries and unknown tool outcomes are
refused. If preserved material itself is too large, use a narrower new session.

Search likely files with related `literals`, then read small disjoint `ranges`
from one file snapshot with `workspace_read_lines`. Native code excerpts stay
within 4 KiB inline; avoid mechanical artifact paging. After an edit, run the
relevant existing check and repair from its diagnostics. `workspace_shell`
accepts a named offline `check_profile` (`engine-host`, `engine-bench-host`,
`harness-host`, `cli-host`) instead of argv, with an owner-selected
`--shell-toolchain`. Profiles retain command approval and the 120-second limit;
timeout/cancellation is partial, and host checks do not qualify GPU behavior.

`/compaction` shows the exact summary, context generation, source record range,
byte reduction and unchanged prefix length. `/journal RECORD OFFSET LENGTH`
inspects raw records locally in bounded UTF-8 ranges. A compaction record maps
retained/removed message indices back to the prior view; raw records and terminal
scrollback are unchanged. **Changing the middle invalidates every later causal KV
entry, even for identical retained tail text.** The summary is the first user
message after the exact system/developer prefix. Compaction truncates only the
wire-builder suffix, verifies the frozen options/tools and system bytes, and clears stale request/response telemetry through the contract client,
retaining same-session affinity and separately labeled cumulative usage. The
next explicit prompt submits the complete rebuilt request to
advertised exact admission. Automatic journal records include exact before/after
input counts when negotiated. Manual previews remain local estimates; admission
counts an applied replacement before saving. Native capabilities remain frozen
until the Engine reports a changed identity, cache epoch, or explicit new
`build_hash`; reconciliation visibly rebuilds the request prefix. Cached tokens may drop; only new response telemetry
reports reuse. Compaction waits for worker cleanup and never resolves an uncertain
side effect by summarizing it away.

`--prewarm-compaction` opts into one discarded, greedy output token after
manual compaction; it is off by default. Manual compaction warms on a cancellable worker
immediately; automatic compaction proceeds directly to its real generation. It requires
advertised exact admission and prefix caching. The server has no prefill-only
route and rejects zero output tokens. Warm requests keep raw tools/messages and
rendering options, change only generation controls, and never execute tools or
add a user/assistant turn. `/context` labels their completion wall time and reported
usage separately from answers. Warm failures are logged and ordinary generation
continues; they never fail the user activation or hold the follow-up queue.
Warm and real HTTP attempts are both counted, but answer totals exclude warm output.
This is not a checkpoint: nothing is pinned, the Engine may evict the prefix,
and cancellation only closes the local socket; Engine work may finish anyway.
Restoration remains subject to the Engine's cache capacity, granularity and identity.

`cargo run --offline -- demo-compaction` audits frozen prefix bytes and diagnostic
hashes without a tokenizer or network, and prints explicit wire-byte estimates
for summary-only and full suffix prefill. The latter includes exact retained user
instructions/recent turns: they require fresh causal KV too. This native feature
does not rewrite delegated Codex compactions or byte-preserving gateway traffic.

Tool results larger than 4 KiB become labeled excerpts backed by exact captured
output in the private session journal. `tool_output_read` retrieves UTF-8 byte
ranges, at most 4,096 source bytes and a 4 KiB result envelope; follow the returned
`next_offset`. `/output ID OFFSET LENGTH` inspects a range locally. IDs are scoped
to the active workspace/session, never paths. Source capture remains capped at
64 KiB (commands reserve framing/UTF-8 overhead); truncation at source is explicit
in command status, and uncaptured bytes cannot be retrieved. At most 128 artifacts
and 4 MiB of captured output are retained per session. Quota is checked before
execution; a full quota stops tools and requires a new session. Retrieval expires
after 24 hours by wall clock. Expired captures still consume quota and remain in
the append-only raw journal until the owner removes the session; its independent
64 MiB/8,192-record disk limit still applies. A saved artifact never replays a tool.

Native project context loads root `AGENTS.md` and, for each repeated
`--rules-scope PATH`, only that directory's AGENTS.md ancestor chain inside the
workspace (at most eight scopes, depth eight). Root rules apply everywhere;
deeper rules apply only in their subtree and take precedence there. Discovery is
explicit, not a recursive scan or a search above the workspace. At most 16 rule
files, 8 KiB each/24 KiB total, enter a frozen, deterministically ordered prefix.
Rules never grant executor capabilities or override explicit user requests.

An optional `skills/catalog.json` lists up to 16 names, descriptions, and visible
`skills/.../SKILL.md` paths. It is declarative JSON, at most 8 KiB; scripts and
unknown entry fields are rejected. Bodies are at most 16 KiB and are read only
when `skill_read` requests one. A frozen local metadata revision guard refuses
changed/replaced bodies; exact loaded bytes and source/revision travel through
the normal tool result/artifact journal. No script, installer or dependency runs
automatically. Catalog ordering is sorted by name; edits require `/new` to become
part of a new prefix. Resume preserves the original snapshot.

`/project` shows scopes, reasons and lazy catalog; `/project-rules` shows frozen
rules; `/skill NAME` inspects a body locally. `/map [PATH]` and `/symbols PATH`
provide bounded on-demand names and lexical declaration hints, as do their model
tools. These views are not semantic indexing or automatically inserted context.
See [formats, limits and design credits](docs/context-and-projects.md).

Each explicit user prompt allows at most 60 tool rounds, 61 model turns, 240 proposed
tool calls, 64 KiB per tool result, and 256 KiB total results. A tool round is one
model reply that asks for tools; its calls share that round. The native agent
also stops after 20 minutes of active work per request. Time waiting for your
approval does not count. It cleans up the current tool, retains the conversation
and journal, and says "Stopped after 20 minutes; say continue to keep going".
Say `continue` to resume the conversation with a fresh request budget.
`--agent-time-limit-seconds N` changes the time limit; `0` disables it. To save
the same setting, explicitly load a JSON file with `--config PATH`, containing
`{"agent_time_limit_seconds":1200}`. An omitted key uses the 20-minute default;
the CLI time-limit flag overrides the file. Both options require native `agent`
mode. These limits apply with or without `--children`; child tasks keep their
separate quotas. The wire/history byte and JSON complexity bounds remain in force
(4 MiB request/history). Assistant text
and tool previews each have a 1 MiB per-run display bound. Completed tool results
remain closed history on cancellation/failure, and there is no automatic retry.
New requests, session changes, and compaction wait for worker cleanup. Only a
validated final `stop` answer counts as a successful turn. Exit cancels and joins
the worker; HTTP DNS/connect and terminal writes keep their existing limits.

Native mode uses the existing cancellable HTTP adapter and bounded workspace
callbacks, with independent bounded work, control, and provisional-render queues.
Socket reads poll cancellation. Requests with an absolute deadline, including
cancellation controls, also bound DNS waiting and connects. Ordinary DNS lookup
and an in-progress connect cannot be interrupted; a timed-out OS lookup may
finish later without dispatching a request.
A blocked terminal output write cannot be preempted. Exit cancels and joins the
native worker. These are local lifecycle guarantees, not remote compute guarantees.

## Two bridge directions

**Other client → Vade gateway → upstream engine:** gateway mode delegates to the
core HTTP relay. With the core bridge integrated, it accepts `/v1/responses` and
`/responses` as well as `/v1/completions`, `/completions`, both chat-completions paths, `/v1/messages`, and
`/v1/messages/count_tokens`. Query strings are forwarded unchanged. It preserves original JSON
tool/reasoning/state/model/cache-key fields and explicit sampling values, SSE payloads,
upstream status, and error bodies. It does not prepend prompts or substitute a
model/cache identity. Requests are buffered within an 8 MiB limit; oversized
fixed-length or chunked bodies receive HTTP 413 before any upstream dispatch. Responses
stream within the core's 64 MiB wire-body bound. HTTP transfer framing is
normalized. Generation requests receive `temperature: 0` (greedy),
`seed: 20260905`, `top_p: 1.0`, `top_k: 0`, `repetition_penalty: 1.0`,
`presence_penalty: 0.0`, and `frequency_penalty: 0.0` only when **none** of these sampling fields
is present: `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `seed`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`,
`n`, `best_of`, `sampling`, `generation_config`. Any one present, including
null, suppresses all defaults: the body is forwarded byte for byte and
engine defaults apply. Routing, affinity, and the 8 MiB accepted-input boundary
use the original client bytes. Defaults are inserted only for upstream dispatch;
that body may exceed the client limit by the inserted defaults.
`GET /v1/capabilities` advertises the exact defaults in
`X-Vade-Sampling-Defaults`, their condition in `X-Vade-Sampling-Defaults-When`,
and the recognized fields in `X-Vade-Sampling-Fields`. The optional `GET /v1/capabilities`
and `POST /v1/admission` routes are relayed unchanged; the gateway never generates
a cache key or performs admission on a caller's behalf. Backpressure writes and
flushes each chunk before reading more upstream data, with an 8 KiB read buffer.

Known upstream API suffixes route to the matching completions, Responses, chat, or Messages sibling;
custom exact paths stay exact in single-upstream mode. Cluster mode requires known
API routes and observes usage with a bounded passive tap. Upstream HTTP/1.1 connections are reused after
complete response framing; the bounded pool retains at most 16 idle sockets for
15 seconds. Downstream connections handle one request each. Failed POSTs are
never automatically replayed. There is no WebSocket, TLS, compression, or upgrade support.
After a complete request, downstream read EOF or a socket error closes the upstream
connection, including during silent waits; socket I/O polls every 50 ms.
Write-half-closed callers are unsupported. Pipelined input is buffered up to
8,421,376 bytes on demand, alongside the 8,192-byte downstream read buffer.
Owner cancellation propagates through the core's `run_until` API. Responses
routing and streaming/error preservation passed local mock tests. Installed Codex
0.164.0 completed a file-edit and `cargo test` task over Responses against FP4-Local,
directly and through `vade-gateway` (2026-10-24); cache behavior remains unverified
because FP4-Local retention is off by default.
Local mock transport/render measurements are documented separately. See the official
[provider configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
for `wire_api = "responses"` and HTTP transport configuration.

**Vade UI → Codex delegated agent:** `codex` mode uses the first-party core adapter
for an optional, separately installed executable. No Codex code or Rust dependency
is bundled. The default executable is `codex` on PATH; `--codex-bin PATH` selects
one executable, not a shell command. Explicit relative paths resolve from the
launch directory before Codex changes to the selected workspace. Starting a prompt
launches that executable.
These commands are usage examples, not part of the local verification:

```sh
vade-harness codex --workspace . --permission read-only
vade-harness codex --workspace . --permission workspace-write --model MODEL
vade-harness codex --workspace . --permission read-only --thread THREAD_ID
```

Both `--workspace` and `--permission` are required. The selected policy is displayed
**before launch**; workspace write access is enabled only by explicit selection.
The core adapter denies unsupported approval/tool requests. The installed Codex
read-only preset does **not** confine file reads to this workspace; workspace-write
restricts write roots, while Codex owns sandbox enforcement. The core checks the
returned cwd, approval policy, and sandbox, refusing broader or unconfirmed policy.
Vade does not read or copy the installed provider's credentials, auto-approve
permission escalation, or start login flows.
Installed Codex retains responsibility for its configured authentication,
permissions, tools, and its own data storage.

The adapter contract initializes the stdio JSONL connection, starts or resumes a
thread, starts each turn, and streams provisional agent-message deltas. Completed item
events supply the authoritative answer; unfinished, conflicting, or unknown
lifecycle events fail explicitly. Plain output prints only the completed answer. Only the current exact
user prompt is forwarded: **no Vade system prefix, local history, or local file
preview is inserted**. The same client/thread is reused across successful turns.
`--thread ID` resumes an existing thread; thread identity is reported separately
on stderr or in the terminal and in `/status`. Provider, thread, model override,
and Vade's optional counter are separate fields, not cache identities.

Completed, failed, and interrupted turns have distinct UI outcomes. Failed,
interrupted, and locally canceled turns never increment the success counter or
enter local recall. Codex may still retain such turns in its own thread. `/new`
drops the current client and starts a new thread on the next prompt; it does not
delete the old thread. `/undo` is refused in Codex mode because this adapter API
has no rollback operation. Protocol failure drops the connection; the next
explicit prompt reconnects using the known thread ID. Raw provider notices and
errors are not echoed, since they can contain prompt or other private content.

The wire protocol follows the official
[app-server lifecycle](https://learn.chatgpt.com/docs/app-server). The committed core
adapter and UI have been checked together against a first-party Rust mock child;
no installed Codex or live model has been called during verification. This is a
bounded bridge slice, not complete Codex feature compatibility. The adapter is
Unix-only, with 1 MiB JSONL records, bounded queues, a 20-second RPC budget,
300-second turn lifetime, 60-second idle deadline, and 5-second interruption grace.
It kills/waits its owned child on shutdown; delegated tool-process lifecycle stays
with Codex. Arbitrary callback/output work cannot be preempted by core deadlines.

**Vade UI → Hermes delegated agent (optional):** `hermes` mode uses the
separate `app_server::hermes` adapter and invokes the selected executable with the
fixed `acp` argument. It requires explicit `--workspace PATH` and
`--permission deny-callbacks`; `--hermes-bin PATH` selects the executable.
`--external-session ID` requests negotiated resume, or load when supported.
The default creates a new external session. Model and authentication setup stay
external; Vade does not read login files, import memory, write configuration, or
start messaging channels. No installed Hermes has been run for verification.

This adapter supports ACP major version **1 only**. It disables client filesystem,
terminal, and terminal-authentication capabilities and supplies no MCP servers.
Permission requests are denied; unknown host callbacks receive an unsupported
method error. These callback rules **do not confine Hermes' own tools or establish
an OS sandbox**. The launch notice and `/status` state that limit and show the
external session separately from Vade recall/counters. The same exact-input rule
applies: no Vade system text, conversation replay, native loop, or tool execution.

ACP v1 message chunks are staged by message ID and become completed answer text
only after `end_turn` and closed tool calls. Other stop reasons do not count as
success. A `cancelled` response to the pending prompt is the cancellation
acknowledgment; RPC errors, child exit, partial streams, and missing replies leave
the external outcome unconfirmed. Setup metadata, bounded replay, turn updates,
and the terminal response remain opaque provider-scoped data in memory. They are
not native history, KV handles, or a durable journal. `/new`, `/undo`, reconnect,
follow-up queues, and local child cleanup follow the delegated lifecycle above.
Live steering, session fork/list/close, ACP v2, multimedia, and live compatibility
are outside this slice. See [the pinned bridge contract](docs/bridge-contract.md).

Use `--prompt TEXT` for one response and exit; ordinary piped input accepts control commands while a worker is active; subsequent
Direct/native prompts steer the active request by default: cancel, retain a marked partial reply, and send the new prompt. `/queue on` opts into the bounded follow-up queue. EOF waits for active work and successful
queued work, and reports any held prompts as unsent. `--plain`, redirected input/output, or `TERM=dumb` selects line
input. `NO_COLOR` disables color without disabling raw input.

A native TTY launch creates a fresh interactive journal; only `--resume NAME`
loads an earlier conversation. The input wraps as you type and on resize.

The conversation is built from four kinds of block, each with a one-line label
and one blank line before the next block:

```text
── You · <time> ──
Fix the failing parser test.

▸ Thinking · <tokens> · <elapsed> · Ctrl+T expand

● Read src/parser.rs
  ⎿ <lines> · <elapsed>

── Answer · <time> ──
The test failed because ...
╰ <time> · <elapsed> · <tokens written> · <write rate> · <read rate> · <first token> · <model calls> · <context use>

── Warning · <time> ──
The next reply would not fit in the model's context.
Vade stopped and kept the history.
Use /compact to shrink older turns, or /new to start over.
```

**You** marks your message. **Thinking** is dim and starts collapsed to one
summary row; Ctrl+T expands it without hiding the reply. **Answer** renders
basic Markdown (headings, lists, code, rules, quotes, links) and ends with one
dim footer: finish time, elapsed time, tokens written, write and read speed,
time to first token, model calls (requests sent to the model) and context use.
Missing measurements display `—`; narrow terminals drop the last facts first
and keep elapsed time, tokens and write speed. **Warning** (yellow) always says
what happened, what Vade is doing about it, and what you can do. A run that
does not finish normally ends with one plain line such as
`Canceled · <elapsed> · <edits> · check passing · say continue`. Labels carry the
meaning with `NO_COLOR` and in plain or redirected output, and the UI uses the
terminal's foreground/background so it works in light and dark themes.
`/status` shows the full counters; `--verbose` also prints the contract
details after each run. The terminal owns mouse selection and Cmd+C; Vade does
not enable mouse reporting. PgUp/PgDn scroll the conversation and full approval
previews.

The footer row above the input always carries facts: at rest `model · folder ·
approval · ctx used/limit · session`; while waiting, the spinner, state
(`connecting`, `thinking`, `reasoning`, `running tool`) and clock sit where the
answer will appear; while writing, the clock, estimated tok/s, queued follow-ups
and the cancel hint. Narrow rows drop middle facts first. The input box shows
its newline keys until you send the first message. Tool calls print inside the
answer flow as `● Run: cargo test` rows followed by one indented, dim
`⎿ exit <code> · <lines> · <elapsed>` summary (exit code, line counts, first error line,
applied operations or the written file); full command and patch output needs
`--verbose`, which describes the command sandbox once and again only when it
changes. `/` opens a menu of the
common commands; typing searches every command for the mode. ↑/↓ select, Enter
runs the selection (an ambiguous prefix is completed into the editor first, as
Tab does), and Esc closes the menu.

## Conversation

| Command | Behavior |
| --- | --- |
| `/help` | Commands and keyboard controls |
| `/children` | Native child IDs and states; `on`/`off` before first prompt; `cancel ID`/`cancel all` during work |
| `/status` | Mode, request state, queues, history bytes/turns, and metadata status |
| `/new` | Create a new native journal, reset direct/demo history, or start a fresh external session; wait for cleanup |
| `/sessions`, `/resume NAME`, `/switch NAME` | List or explicitly select native journals |
| `/fork NAME [TASK_LABEL]`, `/branch` | Fork a transcript / inspect ancestry and file changes |
| `/context` | Native full-request estimates, margin, reserve, and bytes; parent/child admission and reported cache usage |
| `/output ID OFFSET LENGTH` | Inspect a retained output range locally |
| `/compact-preview KEEP SUMMARY` | Preview a compaction without changing context |
| `/compact [KEEP SUMMARY]` | Compact older context automatically, or with your own summary; constraints, receipts and recent turns stay verbatim |
| `/compaction` | Inspect summary, source range, generation and KV invalidation |
| `/journal RECORD OFFSET LENGTH` | Read a raw journal range locally |
| `/project`, `/project-rules` | Inspect frozen project context and rules |
| `/skill NAME` | Inspect a catalog skill locally; no scripts execute |
| `/map [PATH]`, `/symbols PATH` | Bounded repository names / lexical hints |
| `/approve`, `/deny` | Respond to a pending one-time native tool approval; `y`/`n` on an empty line do the same |
| `/undo` | Remove the last direct/demo exchange; refused in native/delegated modes |
| `/cancel` | Request cancellation; show acknowledgment only when received |
| `/steer TEXT` | Codex active-turn input; separate queued/accepted/rejected receipts |
| `/queue on`, `/queue off`, `/queue TEXT` | Toggle queueing instead of default steering, or queue one follow-up |
| `/next`, `/queue-clear` | Resume a held queue, or discard queued follow-ups |
| `/ls [path]` | Bounded listing relative to the workspace |
| `/read path` | Local UTF-8 preview, at most 64 KiB; never sent to the model |
| `/paste` | Collect lines until `/end`; `/cancel` discards them |
| `/quit`, `/exit` | Exit |
| `//text` | Send a prompt starting with a slash |

In the raw terminal, **Enter** sends. **Shift+Enter** inserts a newline when the
terminal reports the modifier through CSI u or modifyOtherKeys. Terminal.app
does not report Shift+Enter by default, so use **Option+Enter** or **Ctrl+J** there;
the input box shows the hint. Left/Right, Home/End, Delete, and Backspace edit at
a UTF-8 cursor. Up/Down recall submitted messages in the current session, including
multiline messages; Down past the newest restores your unsent draft. Recall is
bounded and separate from model context and journal history. `/new` or selecting
another session clears it.
Ctrl+Up/Down or Option+Up/Down move between lines when the terminal reports those
keys; Home/End and Left/Right remain available for editing multiline input.
Ctrl+U clears input, Ctrl+K deletes to the line end, Ctrl+W or Alt+D deletes a
word, Alt+B/F or Ctrl+Left/Right move by word, and Ctrl+L repaints. The editor shows the end of long input and
adapts to terminal width; a bracketed multiline paste lands as one editable
message and does not send until Enter. Input remains editable during work. Cursor movement is by Unicode scalar, with
approximate terminal cell widths; grapheme-cluster editing and autocomplete are
not implemented. On resize a fresh frame avoids erasing old reflowed rows.

**Esc** or **Ctrl+C** during streaming interrupts the response and returns to the
prompt. Direct and native modes retain partial prose with an `[interrupted]`
marker; native mode also retains completed tool results. Esc never discards a
draft (Ctrl+U clears it). Direct mode binds the core's cancellable HTTP transport; pending
socket I/O polls cancellation every 250 ms. OS DNS lookup and connection setup
have separate limits, and closing a socket does not guarantee upstream compute
stops. Native mode cancels the runtime. Codex
mode asks the core adapter to interrupt the turn and finish cleanup.
In delegated modes, or with `/queue on`, a prompt entered during work becomes a separate queued follow-up (at most eight
prompts / 64 KiB total). It runs after success; failure or cancellation holds the
queue for explicit `/next`. Local commands and exit remain available. Codex `/new` waits for in-flight cleanup before changing threads.
Failed direct exchanges and canceled or failed demo/Codex exchanges are excluded
from local recall. Interrupted direct exchanges remain in conversation history;
native mode retains its closed typed history, including completed tool results.
Ctrl+C clears idle input, then exits when the prompt is empty. Ctrl+D sends
nonempty input or exits.
In line mode, Ctrl+C exits; native/Codex modes also cancel and join their workers
before exit, and Codex drops its client. Codex initialization observes cancellation
while waiting on child I/O. The UI reports interruption only after an interrupted
terminal event; a rejected interrupt, timeout, or broken stream leaves the remote
outcome unconfirmed. A completion that races with cancellation is reported as a
remote completion with the local result discarded. Raw terminal state is restored on normal exit, I/O
errors, unwinding, SIGINT, SIGHUP, and SIGTERM. Uncatchable termination cannot run
cleanup. These statements describe local cleanup, not remote compute guarantees.

Native messages are serialized once and retain exact bytes, including tool calls,
results and provider fields. An append-only request builder rejects historical
rewrites. Fixed options/tools precede history. The stable canonical JSON prefix
excludes the final `]}` envelope. Optional KV hints follow messages, outside that
prefix. Token-prefix equality inside an engine remains unverified. Direct/native clients persist across user turns and reuse framed HTTP
connections. See [the prefix and performance contract](docs/performance.md).

For direct/demo chat, the fixed system prefix is passed to the core builder unchanged. Successful
user/assistant exchanges retain exact contents. Local recall evicts oldest **whole
exchanges** at 32 turns or 256 KiB; an oversized exchange is excluded. Direct mode
then requires `/new` before sending another request, so evicted history is never
silently resent as a changed prefix. Demo recall remains bounded as before. Prompts are
limited to 64 KiB and response display to 1 MiB. Status reports local bytes and turns. `/context` separately labels exact admission,
rough estimates, and provider-reported usage; it never infers cache hits from
request bytes or affinity.

## Workspace and privacy

`--workspace PATH` selects the root and is required explicitly for native agent
and Codex modes; direct/demo default to the current directory. Read and
list operations use directory handles on macOS and Linux, refusing absolute paths,
parent traversal, symlink traversal, hard-linked reads, non-regular files, hidden
paths, and common credential filenames. Directory enumeration visits at most 256
entries, including filtered entries; a limit marker signals an incomplete list.
Other platforms refuse these operations rather than weakening the boundary.

Vade's explicit `/ls` and `/read` results remain local. The direct/demo UI has no
automatic context discovery, workspace writes, subprocess tools, or model-directed
tool execution. Native model-requested tool results are sent to the selected
model; no root path is automatically inserted, though explicit command output
(such as `pwd`) can contain it. Delegated Codex tools follow its
selected external policy. Terminal output filters control sequences and
directional overrides. File filtering is not a
content secret scanner; explicit previews are visible in terminal scrollback.

Direct/demo recall stays in process memory; delegated agents own their persistence.
For these modes, the metadata store writes no transcript, provider/thread identifier,
endpoint, model name, or file contents. Optional `--session NAME` persists only
a completed-turn count under
`sessions/NAME.state` in the same private state folder. It does **not** restore a conversation. The directories
must be owned by the current user with private permissions; files are created as
0600, updates use atomic replacement, and symlink redirects are refused. An empty
private lock file prevents simultaneous instances from overwriting the same
counter. An owning guard explicitly unlocks on session release, even if a
concurrent spawn briefly inherits a descriptor before exec; the kernel also
releases the lock when its last descriptor closes after process exit. Use a generic
session name, since the name becomes a filename. Default demo mode creates no files. Native agent journals are described above.

## Validation and integration

```sh
cargo test --offline
```

Local tests cover workspace boundaries, crash-tail replay, journal failures,
mock HTTP sessions, subprocess protocol fixtures and terminal approval handling.
The [CI workflow](../.github/workflows/ci.yml) runs host checks with warnings denied.
Use the test output from the current checkout for counts and platform limits.
Host fixtures do not qualify native inference or installed-client behavior.

The session lock uses explicit unlock before descriptor close. The
[bridge contract](docs/bridge-contract.md) explains the race and regression.
Synthetic stream measurements and their limits are retained in
[performance methodology](docs/performance.md) and its linked sample files.

## Design references

These references informed UI and interface choices:

- [Hermes Agent](https://github.com/NousResearch/hermes-agent): visible conversation
  controls, multiline terminal entry, and interruptible interaction.
- [Pi coding agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent):
  a minimal harness with interactive and noninteractive operating modes.
- [Codex CLI](https://github.com/openai/codex): a local terminal entry point for a
  coding assistant.

These projects are conceptual references, not dependencies. No competitor
implementation code was copied. Vade's workspace boundary, session journal, coding tools,
and core/UI contract are first-party implementations for this project.

## Bridge hardening and responsive control

The bridge provides a pinned first-party app-server fixture, authoritative
item completion, strict unknown-event handling, cancellable initialization,
steering receipts, independent control/render capacity, optional Hermes ACP v1
delegation, and Mac pseudo-terminal checks. Workers coalesce provisional text in a 64 KiB render buffer, while final
results and control receipts remain separate. Each UI frame drains at most 16 KiB
of provisional text. An omitted display suffix is repaired from completed output;
it never supplies history. OS-blocked output writes remain non-preemptible.

See [the bridge contract](docs/bridge-contract.md) for exact limits, failure states,
source credits, and mock-only scope. These changes use no third-party crates and
make no live Codex, Hermes, Engine, or Spark compatibility claim.

## API authentication

Use `--api-key-file PATH` on any harness command or `vade-gateway`, or set `VADE_API_KEY_FILE`.
The flag overrides the environment. The file contains the same trimmed,
single-line key of at least 16 bytes as the server, using only visible ASCII
`0x21-0x7E`. Prefer 32 random bytes encoded as hexadecimal and protect remote
HTTP with an SSH tunnel or TLS termination. The harness selects the path at
startup and reads it once before agent/direct UI startup or gateway serving;
other commands read it just before their first upstream HTTP request. Help,
version, setup output, and local workspace commands work even if the file is
missing or unreadable. It sends `Authorization: Bearer <key>` on discovery, admission, generation,
and checkpoint release, including child agents. Gateways use it for upstream
requests and replace incoming credential headers when configured. HTTP 401 is
shown as `HTTP 401 Unauthorized: authentication failed`. Keys are never logged
or written into session state; restart to rotate the file. Server `/health`
remains open.

The standalone gateway also applies the selected key to capability/admission
probes and recorded replay. Without a configured key, it forwards the client's
credential headers unchanged. A configured key replaces both `Authorization`
and `X-Api-Key` on upstream requests; response headers remain opaque. Recordings
exclude credential headers, so authenticated replay requires its own key file.

Native streaming replies have a soft prose/reasoning cap: 120 seconds of measured
model decode (512–4096 estimated tokens; 2048 before speed is known). Configure it
with `agent --reply-tokens N`. The estimate uses three UTF-8 bytes per token and
can overshoot by one SSE event. A started tool group always finishes under the
existing upstream limits and framing checks; the soft cap never cuts arguments.
A host cutoff closes the socket and enters the ordinary bounded continuation
loop. Upstream cancellation is best effort. Nonstreamed JSON keeps upstream limits.

Direct and native chat show `prompt cache: off/on/unknown` from the Engine's
advertised capabilities in reply status. The system prompt includes the configured
model id and its configured catalog family; unknown model names do not invent a
family. Steering through a single-upstream gateway forwards `/v1/cancel`;
cluster gateways reject cancellation until request affinity is available, so the
follow-up stays held. Connect directly to the selected engine to steer there.
