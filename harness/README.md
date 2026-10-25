<p align="center"><img src="docs/assets/vade-harness-banner.svg" width="100%" alt="VADE Harness: a coding-agent harness, written from scratch in Rust"></p>

# Vade Harness

Vade Harness is the terminal program you use to work with the model. It is both
a coding agent and a chat client, written in first-party Rust with no
third-party crates. The model itself runs in the [engine
workspace](../engine/README.md) in this monorepo. The engine does the inference
and keeps the KV state, the attention memory it holds for each conversation
(see [KV cache](../docs/GLOSSARY.md#ho) in the glossary).

It gives you:

- a native **tool-aware agent**. It reads and lists files, makes exact patch
  edits and runs sandboxed commands once they are approved, all inside a
  workspace you pick and within fixed limits;
- direct streaming **text chat**. Its only tools are local commands you type,
  such as `/ls` and `/read`;
- an offline **demo**;
- a **gateway** command that sits between another client and the engine;
- a hand-off of each turn to **Codex** (through its app-server) or to a
  **Hermes** agent (over ACP v1), if you have one installed.

We keep the native harness small on purpose: it only changes for automatic
limits, [greedy](../docs/GLOSSARY.md#dg) and cache-friendly requests, and bug
fixes. Earlier experiments with web access, PDF and image-to-text, and a
launcher with model discovery were taken out.

To use your own client instead, build the [`vade-gateway`
binary](docs/invisible-gateway.md) and put it in front of the engine; your
client only needs to change its base URL. The gateway forwards Chat
Completions, Responses and any other HTTP path. It can optionally add stats,
recording and replay, and it has explicit optimization flags. [Setting up
compatible API clients, and what we tested](docs/passthrough-matrix.md) covers
each client.

What has and has not been checked is near the end, under [What has been
validated](#what-has-been-validated).

## Four ways to run Vade

The examples below assume an HTTP engine that you started separately at
`http://127.0.0.1:18080`. Replace `MODEL` with its exact model ID. They are
usage examples, not live tests.

- For the FP4-Local server that [INSTALL.md](../INSTALL.md) installs, the model
  ID is `qwen3.8-flash-next-fp4-local`.
- Start the engine with INSTALL.md: Route A if you work on the Spark itself,
  or [Route B](../INSTALL.md#5-start-the-two-spark-server) if you drive it from
  another Linux or macOS computer (a controller).
- The engine keeps running until you type `vade down`. On a controller, the
  network tunnel to the Spark stays open until then too. Any client, direct or
  through a gateway, can connect while the engine is up.
- The [legacy FP8 connection
  guide](../engine/docs/flash-server.md#run-the-native-coding-agent) uses its
  own model ID and profile.

Pick the way that fits how you work:

1. **Use ours (Vade Engine + Vade Harness).** You run the Vade harness against
   the Vade engine, with our native UI and its bounded workspace tools.

   `vade-harness agent --endpoint http://127.0.0.1:18080 --model MODEL --workspace .`

2. **Put ours in front (your harness through Vade's gateway).** You keep your
   own harness and point its provider at port 8787. Vade's gateway sits between
   it and the engine. You can use either of two gateways:

   - `vade-gateway --upstream http://127.0.0.1:18080/v1` (as in INSTALL)
     forwards every path, including `GET /v1/models`.
   - `vade-harness gateway --upstream http://127.0.0.1:18080` serves only the
     API routes listed under [Gateway relay, Codex and Hermes in
     detail](#gateway-relay-codex-and-hermes-in-detail), and answers 404 to
     `/v1/models`.

   Prefer `vade-gateway` for clients that list models.

3. **Hand turns to Codex or Hermes (Vade UI + installed agent).** You keep the
   Vade UI, and the installed agent owns each turn.

   `vade-harness codex --workspace . --permission read-only`

   For Hermes, use `hermes` mode with `--permission deny-callbacks`. That
   permission setting is not an OS sandbox.

4. **Bring your own (Vade Engine alone with any compatible client).** You point
   any compatible client straight at the engine. No harness process is needed.

   `curl -q http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"MODEL","messages":[{"role":"user","content":"Hello"}]}'`

   FP4-Local thinks by default (`default_thinking: true` in
   `/v1/capabilities`). A plain client therefore gets `reasoning_content`
   first, then `content`. To get just the answer, add
   `"enable_thinking": false` to the request. The Vade harness sends that for
   you unless thinking is on.

### Set up a client and check the endpoint

`setup` prints a ready configuration for your client, both direct and through
the gateway:

`vade-harness setup codex --endpoint http://127.0.0.1:18080 --model MODEL`

Besides `codex`, it supports `hermes`, `opencode`, `pi` and `generic`. It works
offline, and it writes nothing unless you pass `--write PATH` plus a chosen
`--target direct|gateway`. Codex needs Responses support on the upstream,
because the gateway does not translate between APIs.

There is no Claude Code setup or delegation mode in the harness. The CLI's
single-Spark workflow does print an environment command for Claude Code, but
live Claude Code chat and tool compatibility have not been qualified there yet;
see the [CLI guide](../cli/README.md).

For a complete, isolated configuration, add `--dedicated-home --write DIR`,
`--context-window N` and `--max-output-tokens N`, using upstream limits you have
verified. Setup then creates a new private directory with the client's own
instruction file and a `launch.sh`. The launcher keeps HOME, XDG and client
state inside that directory. Setup refuses existing homes. Run the launcher
from the workspace you want the client to use. The [client matrix and launch
commands](docs/passthrough-matrix.md) lists the supported setup commands and
what each has been validated against.

Vade Engine serves `/v1/responses`, so Codex works both directly and through the
gateway. Three things to know when you run Codex from a dedicated home:

- **Sandbox.** The generated home picks Codex's read-only sandbox. For a task
  that edits files and runs commands, pass `-s workspace-write`, for example
  `/path/to/codex-home/launch.sh exec -s workspace-write --json "Add a test and run cargo test"`.
- **Rust toolchain.** The launcher clears the environment and gives Codex a
  private HOME. Rustup's `~/.cargo/bin` shims then cannot find your toolchains
  (they try to download a new one instead). Put the toolchain's own `bin`
  directory first on PATH when you launch, for example
  `PATH="$HOME/.rustup/toolchains/1.99.0-aarch64-apple-darwin/bin:$PATH" .../launch.sh exec ...`.
- **Gateway token.** Setup has no gateway token option. If you run
  `vade-gateway --token` (or set `VADE_GATEWAY_TOKEN`), pass the token to Codex
  as `-c 'model_providers.vade.experimental_bearer_token="TOKEN"'`. Without it,
  Codex stops with `401 Unauthorized: gateway token required`. The token is
  then visible in the process list.

To check an endpoint before you rely on it, run the doctor:

`vade-harness doctor http://127.0.0.1:18080`

The doctor checks the endpoint's capabilities and admission (the engine's check
that a request fits). It then sends two made-up chat turns in one
conversation. Both turns start with the same text, which is longer than two of
the engine's advertised cache blocks (longer than 8192 tokens if it advertises
no block size). When the engine offers admission, the doctor first uses it to
check that this shared start is long enough. The doctor then prints how much of
the second turn's prompt the engine reports restoring from its cache:

- only a reported hit passes the cache check;
- a server that says its cache is off gets a warning instead.

It reads gateway stats only from a gateway that marks its responses with
`X-Vade-Gateway: v1`. See [setup and doctor details, sources, and mock
coverage](docs/setup-doctor.md).

## Build and run from source

From `harness/` in this monorepo, use the pinned Rust `1.99.0` compiler:

```sh
cargo build --offline
cargo run --offline -- demo
cargo run --offline -- --demo --prompt "Show me the local demo"
cargo run --offline -- --help
```

With no mode argument you get the **demo**. It gives a clearly labelled scripted
response and makes no network requests.

Direct mode needs both an endpoint and a model, and native agent mode also needs
an explicit workspace. The commands below show the configuration for a mock you
provide separately. They do not start a mock or connect anywhere during setup:

```sh
vade-harness direct --endpoint http://127.0.0.1:18080/v1/chat/completions --model mock
vade-harness agent --endpoint http://127.0.0.1:18080/v1/chat/completions --model mock --workspace .
vade-harness gateway --upstream http://127.0.0.1:18080 --bind 127.0.0.1:8787
```

### Attach files

To attach a workspace text file, type `@path` (quote or backslash-escape
spaces). You can also drop or paste an absolute file path as the whole input.
An absolute path inside a pasted log stays text.

- Tab completes a simple `@path` common prefix, and a second Tab lists matches
  from a bounded directory listing.
- Names must be visible ASCII.
- Symlinks, binary files, images and PDFs cannot be attached. Clipboard images
  and PDF extraction are not supported.
- Messages that include files are capped at 64 KiB. A refusal keeps your draft.
- Input recall stores the names you typed, while the journal saves the exact
  file bytes once.

## Gateway relay, Codex and Hermes in detail

### What the gateway changes

The gateway forwards your client's requests to the engine without changing
their content: no extra prompts, no model swap, no new cache key.

`vade-harness gateway` accepts these routes, and forwards query strings
unchanged:

- `/v1/responses` and `/responses`;
- `/v1/completions` and `/completions`;
- both chat-completions paths;
- `/v1/messages` and `/v1/messages/count_tokens`.

Bodies up to 8 MiB are accepted. A larger one gets HTTP 413 before anything is
sent upstream.

The gateway does fill in sampling defaults. If a generation request sets none
of the sampling fields, the gateway adds the greedy defaults: `temperature: 0`,
`seed: 20260905`, `top_p: 1.0`, `top_k: 0`, `repetition_penalty: 1.0`,
`presence_penalty: 0.0` and `frequency_penalty: 0.0`. The sampling fields are
`temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `seed`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`,
`n`, `best_of`, `sampling` and `generation_config` (the response header
`X-Vade-Sampling-Fields` lists them too). If the request sets any one of them,
including null, no default is added and the body goes through byte for byte,
so the engine's own defaults apply.

Installed Codex 0.164.0 completed a file-edit and `cargo test` task over
Responses against FP4-Local, both directly and through `vade-gateway`
(2026-10-24). Full rules: [engine contract](docs/engine-contract.md#opaque-gateway).

### Several servers behind `vade-gateway`

With `vade-gateway`, repeat `--upstream` to add **replicas**: independent
servers running the same one-Spark model (for example from
`vade up --replicas 2`). Routing works like this:

- Each request goes to the upstream with the fewest requests in flight, and
  ties take turns.
- Requests whose `messages` (or Responses `input` array) share their first 2048
  bytes up to the last user message prefer the upstream that served the
  previous one, as long as it is among the least busy. `--no-sticky` turns that
  off. FP4-Local prefix retention is OFF by default, so any cache hits from this
  are unverified.
- `GET /health` then reports every upstream, and fails when their model IDs
  differ.
- A failed upstream is retried on another one only while nothing has been
  streamed to the client yet.

Use replicas when the model fits on one Spark and you run many chats at once.
Use the multi-Spark [chain](../docs/GLOSSARY.md#ac) when the model is too large
for one Spark. Identical replicas return identical token IDs, so routing never
changes an answer.

The [historical measurements](docs/gateway-measurements.md) record exact greedy
answer comparisons and measured latency limits. The [historical gateway
card](docs/harness-card.md), a one-page benchmark summary, measures installed
Codex directly and through this gateway at 10k, 50k and 100k prompt tokens.

### Gateway routing and access

With `vade-harness gateway`, repeat `--upstream URL` to turn on **cache-aware
cluster routing** across compatible endpoints. The gateway:

- discovers each endpoint's capabilities;
- keeps a bounded memory of which conversation went where (affinity);
- falls back to the least-loaded compatible endpoint when one is unhealthy or
  saturated.

If a new connection fails, or the upstream answers 429 or 503, before any
response bytes have been forwarded, the gateway retries once on another
verified healthy, compatible upstream. The retry sends your request unchanged,
including the fields you set. Once a response has started forwarding, nothing
is replayed, and read or write failures after the request was sent are not
retried either.

With several `--upstream` options, `GET /_vade/gateway/stats` shows each
endpoint's requests, failures and reported cached/prompt-token ratios. With one
upstream the gateway stays a plain relay, and that route answers HTTP 404. See
[routing configuration, mock evidence and limits](docs/gateway-routing.md).

**Access from browsers and other hosts.** Both `vade-gateway` and
`vade-harness gateway` answer 421 when Host is not `127.0.0.1`, `localhost` or
`[::1]` (bare or with the listener port). They answer 403 when a request
carries any `Origin` header. That way browser pages and DNS rebinding cannot
reach the engine through them. To allow a browser client on purpose:

- pass its exact origin with `--allow-origin` (repeatable). It then gets CORS
  headers and preflight answers, and `--allow-host` admits another Host name;
- set `--token TOKEN` or `VADE_GATEWAY_TOKEN`, so that every request needs
  `Authorization: Bearer TOKEN`. That header is not forwarded upstream.

**Listener and transport.** The gateway listens on `127.0.0.1:8787` by default.
The core handles HTTP routing, forwarding, response parsing and upstream
timeouts. The CLI refuses HTTPS explicitly: there is no TLS implementation and
no custom cryptography. URL credentials, query strings, fragments and control
characters are rejected. Direct and native clients accept
`--api-key-file PATH` or `VADE_API_KEY_FILE` for Bearer authentication; see
[API authentication](#api-authentication).

### Codex mode

`codex` mode runs your separately installed `codex`, or the executable you name
with `--codex-bin PATH`. Vade sends only your prompt, with no Vade system text
or history. These are usage examples:

```sh
vade-harness codex --workspace . --permission read-only
vade-harness codex --workspace . --permission workspace-write --model MODEL
vade-harness codex --workspace . --permission read-only --thread THREAD_ID
```

Both `--workspace` and `--permission` are required, and the policy is shown
before launch. Note that Codex's read-only preset does not stop it reading
files outside the workspace. Vade never reads Codex's credentials or approves
its permission requests.

### Hermes mode

`hermes` mode runs `hermes acp` (ACP version 1) with an explicit
`--workspace PATH` and `--permission deny-callbacks`. It turns down every
permission request, but that is not a sandbox: Hermes' own tools are not
confined. Protocol, limits and failure handling are in the [bridge
contract](docs/bridge-contract.md).

## How the harness builds requests

These sections cover the requests that direct chat and the native agent send:
what they ask the cache for, how they keep history stable, and which sampling,
output and thinking settings they use.

### Caching and the engine contract

The harness asks for caching whenever the engine supports it, but the current
FP4-Local engine keeps prefix retention off by default: unless the server is
started with retention on, it does not keep the start of earlier prompts cached
for reuse. [Cache defaults and measured tradeoffs](docs/cache-defaults.md)
explains the difference between what a request asks for and what the server
does.

When the capabilities advertise `prefix_cache` or the `scheduling_hint` cache
key, agent, direct, child and doctor requests send `"cache": true` by default
and keep `prompt_cache_key`. A `cache` value the caller supplies is kept as is.
Set `VADE_HARNESS_PREFIX_CACHE=0` to send `"cache": false` instead, in children
and doctor too.

Direct and native chat show `prompt cache: off/on/unknown` in the reply status,
based on the engine's advertised capabilities. The footer after each reply
reports cache reuse as restored prompt tokens / prompt tokens, summed over that
reply's model calls. Missing telemetry stays unknown.

Before their first model request, direct and native workers negotiate the
optional **vade.engine.v1** contract (the Vade engine's capability and admission
interface). When exact admission is advertised, the worker checks the complete
final request and output reservation before every generation, tool rounds
included. Without the contract, requests use ordinary chat behavior and clearly
labelled estimates.

The harness sends its generated session cache key only when the engine
advertises the scheduling-hint feature. Routing a conversation back to the same
server (affinity) is never counted as a cache hit by itself. Per-response input
and output counts, and the cached-token counts actually reported, appear after
each run and in **/context**. See the [capability, admission and gateway
contract](docs/engine-contract.md) for the schema, fallback behavior, limits and
coordinator integration.

### History and the stable prefix

The engine can only reuse its cache when a new request starts the same way as
the last one, so the harness keeps that start stable. System instructions, tool
definitions, ordered history and the per-conversation cache key stay stable
across ordinary turns. The system prompt includes the configured model id and
its configured catalog family, and unknown model names do not get an invented
family.

Each native message is turned into JSON once and then never changes, tool
calls, results and provider fields included. The request builder only appends,
and it refuses any edit to history. Fixed options and tools come first, then
history. Optional KV hints go after the messages, so they never change the
reusable start, which stops just before the closing `]}` of the JSON. We have
not yet verified that the engine sees identical tokens for that start. Direct
and native clients persist across user turns and reuse framed HTTP
connections. See [the prefix and performance contract](docs/performance.md).

For direct and demo chat, the fixed system prefix goes to the core builder
unchanged, and successful user/assistant exchanges keep their exact contents.
Local recall evicts the oldest **whole exchanges** at 32 turns or 256 KiB, and
an oversized exchange is left out. Direct mode then needs `/new` before it sends
another request, so evicted history is never quietly resent as a changed prefix.
Demo recall is bounded too. Prompts are limited to 64 KiB and response display
to 1 MiB. Status reports local bytes and turns. `/context` labels exact
admission, rough estimates and provider-reported usage separately, and never
infers cache hits from request bytes or affinity.

### Sampling and seeds

A fresh direct or native session without sampling flags sends no sampling fields
at all. Vade Engine then applies its shared [greedy](../docs/GLOSSARY.md#dg)
default, and any other upstream keeps its own default policy. An internal retry
counter never quietly opts into sampling. `/retry` keeps the fields absent, so a
greedy answer may simply repeat.

The CLI never invents or increments a seed. `--temperature` and `--greedy`
without `--seed` leave the seed unspecified. An explicit or saved seed stays the
same across retry, resume, tool continuations and ordinary follow-ups. To pick a
different seed, restart with a different `--seed`; `/retry` reminds you when
one is pinned.

New native sessions never get a seed unless you set one, even after retry or
resume. A session that already saved a seed keeps using it. That includes older
sessions whose seed an earlier version picked automatically. Start a new session
(or use `/new`) to get the current behavior, where no sampling fields are sent,
seed included. Other generation flags apply only to the launch that sets them.

Direct and native agent modes accept `--greedy`. Every model request then uses
`temperature: 0`, tool turns and continuations included. On engines that
support it, this allows strict greedy drafting: the model proposes a few tokens
ahead and the engine verifies them. Drafting keeps the same answer bytes as
plain greedy decoding, and `--greedy` does not turn on sampled drafting.
`--temperature` and `--greedy` cannot be used together.

### Output, thinking and continuation

Direct and native agent modes accept `--max-output-tokens N` and
`--thinking on|off`. Both stay fixed for the session you launched.

**Output room.** By default, each reply reserves the smallest of three numbers:
16,384 tokens, the engine's advertised output limit, and the remaining context
minus 256 safety tokens. An explicit `--max-output-tokens` keeps the reservation
you chose. It must be positive, no larger than the advertised output limit, and
still fit the context. An oversized value is refused before admission or
generation. If no limit is advertised, the upstream checks the upper bound.

**Thinking.** Thinking is off by default, and explicit thinking flags override
`VADE_HARNESS_THINKING=1`. When the engine advertises thinking, the harness
sends `enable_thinking: false` unless you turn thinking on. With thinking on and
no effort set, the engine uses the model's default effort. For Qwen3.8 that is
`xhigh`, which can spend most of a reply on hidden reasoning.
`--effort low|medium|high|xhigh` (or `VADE_HARNESS_EFFORT`) sends that
`reasoning_effort` and turns thinking on, so you cannot combine it with
`--thinking off`.

**Continuation.** When an answer is cut off at the output limit, direct chat and
the native agent continue it automatically. Each continuation sends the same
user message as `/continue`, keeps the seed and the earlier answer, and shows a
short notice. If you would rather type `/continue` yourself, use
`--no-auto-continue`. The native JSON config accepts `"auto_continue": false`
for the same opt-out. Automatic continuations have limits:

- at most 32 consecutive continuations are sent for one answer;
- they share the original request's active-work budget, which is 20 minutes by
  default;
- native continuations also keep the original 61-model-turn, 60-tool-round and
  240-proposed-tool-call limits.

A saved answer stays visible when a cutoff hits one of those limits. A manual
`/continue` then starts a fresh work budget with the earlier conversation and
seed. If the context cannot fit another reply, native mode tries its existing
in-task compaction (shrinking older turns), and direct chat stops with a plain
message. Continuations stay in the exact conversation, and native journals keep
them too.

**Empty replies.** A native run never ends silently. A finished reply with no
visible text and no tool call (only hidden reasoning) gets a short recorded
reminder, at most twice in a row. A third empty reply stops the run with a
notice saying why. Reminders use the same request limits.

**Reply size.** Native replies also have a soft size limit, so one runaway reply
cannot eat the whole task. The limit is about 120 seconds of decoding at the
measured speed, kept between 512 and 4,096 estimated tokens (2,048 before the
speed is known). Tool calls get twice the room. The limit counts visible text
and tool-call arguments at three bytes per token, not hidden reasoning, and one
SSE event (one chunk of the streamed reply) can push past it. Change it with
`agent --reply-tokens N`; `0` turns it off. When a reply hits the limit, the
harness:

1. closes the connection;
2. throws away any half-finished tool call without running it;
3. asks the model to go on in smaller steps, under the usual continuation
   limits.

Stopping the work on the engine side is best effort. Non-streamed JSON replies
keep the upstream limits. Details are in [agent
tools](docs/agent-tools.md#reply-budget).

## Native coding agent

`agent` (or `--mode agent`) runs the native agent loop with fixed limits. It
needs `--endpoint`, `--model` and an **explicit `--workspace`**. The model gets
these tools: `workspace_read`, `workspace_list`, `apply_patch`,
`workspace_shell`, `tool_output_read`, `skill_read`, `workspace_manifest`,
`workspace_symbols`, `workspace_search` and `workspace_read_lines`.

The agent's tools always run on the computer where you start the harness, in
the workspace you pick there. When you drive the Spark from a controller
(another computer; see INSTALL's [Route
B](../INSTALL.md#5-start-the-two-spark-server)), inference runs on the Spark and
the tools run on the controller. Shell commands need macOS; other platforms
refuse them. The legacy two-Spark FP8 setup has its own [connection and run
recipe](../engine/docs/flash-server.md#run-the-native-coding-agent).

### Search and read

The approach that works best has three steps:

1. Search likely files with `workspace_search` (`path`, `literal` or related
   `literals`). It returns relative paths and line numbers in a complete inline
   JSON result.
2. Read small disjoint `ranges` from one file snapshot with
   `workspace_read_lines` (`path`, `start_line`, `end_line`, or disjoint
   `ranges`) for exact context.
3. After an edit, run the relevant existing check and repair from what it
   reports.

Search is literal and case-sensitive, capped at 512 files/8 MiB per call. It
skips hidden, credential, link, and dependency/build paths. Pagination is a live
view, so copy `next_offset`, or narrow the path when the scan reports omitted
input.

A line read returns up to 200 lines (an inclusive range) from a file of up to
4 MiB, labelled with its path and range, and at most 4 KiB of text. Because
each excerpt is capped at 4 KiB, searching first and reading targeted ranges
works better than paging through a file from the top.

Native `workspace_read` accepts a zero-based byte `offset` and a byte `limit`
(1–4096) for sources up to 4 MiB. Partial results give the actual range, the
total size and the next offset. Reads keep the hidden/credential path and
directory-handle protections.

`workspace_shell` accepts a named offline `check_profile` (`engine-host`,
`engine-bench-host`, `harness-host`, `cli-host`) instead of argv, with a
`--shell-toolchain` you picked. Profiles keep command approval and the
120-second limit. A timeout or cancellation gives a partial result. Passing
host checks says nothing about GPU behavior.

### Tool results, replies and waits

While the agent waits you will see **queued / awaiting upstream** until the
first text, reasoning or tool data arrives. Then you see **thinking** for
reasoning and **generating** for the answer. These labels are local; they do
not show your place in the engine's queue. In the interactive terminal,
reasoning starts folded behind a Thinking row, and Ctrl+T opens or closes it.

When a reply is cut off:

- at the length limit, it stays on screen and is continued automatically;
- in the middle of a tool call, the run stops with a plain message, and the
  half-written call never runs.

If a patch was only partly applied, the agent stops and keeps a record of what
was written, so check the files before you resume. A stopped turn holds any
queued prompts until you send a follow-up or `/next`.

A silent wait while the engine queues or reads your prompt can last up to the
3,600-second HTTP lifetime. Esc, the active-work budget (20 minutes by
default), or a child or workload deadline can end it sooner. Discovery,
admission, request writes and direct chat use the ordinary 30-second idle
timeout. Nothing is replayed automatically. For history bytes, partial patch
receipts and timeouts, see [how the harness talks to the
engine](docs/engine-contract.md).

### Patches

`apply_patch` adds, updates, deletes and moves bounded UTF-8 files. It needs
exact, unique old text and existing parent directories; there is no fuzzy
matching. Patch text is limited to 64 KiB, and each target and result may be up
to 4 MiB.

**The preview.** `apply_patch` checks all files first and shows unified diff
hunks with three context lines before you approve. Overlapping context is
merged. Nonprinting Unicode, tabs, carriage returns and literal backslashes are
visibly escaped. The 192 KiB cap counts the escaped display bytes, and an
oversized diff is refused before approval.

**How files are written.** Updates stage a temporary file in the same
directory, sync it, recheck the original metadata and bytes, then rename it into
place atomically. Adds and moves never replace an existing destination. Delete
sections must contain the entire exact old text. Existing permission bits are
kept.

Atomicity is **per file**, not a batch transaction. A combined move and edit can
therefore leave both paths if it is interrupted between installing the new file
and removing the source. Partial results and conflicts are reported explicitly.
See [tool grammar and limits](docs/agent-tools.md).

### Build checks and speed reports

A build check closes the build loop. Set it with `--check COMMAND`, or put it in
the workspace file `.vade/check`, where the first nonblank, non-`#` line is
used. `--no-check` ignores the file.

After any tool round whose patches changed files, the harness runs
`/bin/sh -c COMMAND` as a journaled `workspace_shell` call with ID prefix
`vade-check-`. It runs under the same approval policy and macOS sandbox, with no
prompt in `--approval auto`.

- The model gets the exit status, with matching error lines first and then the
  last lines, within about 3.5 KiB. Longer raw output stays in a session
  artifact.
- If the model gives a final answer while the latest check is failing, it gets
  up to two reminders.
- The model also sees a note when it reruns an identical failing command without
  changing files, or applies a patch with more than 150 changed lines.

Each native run prints one speed summary, and `/status` shows per-response TTFT
(time to first token), output tok/s and prompt/cached tokens. Direct mode
reports each validated response. `/status` also includes the mean TTFT and a
decode rate weighted by decode time across the session. Missing measurements
stay unknown.

Decode speed is
`(output tokens - 1) / (last generated delta - first generated delta)` from the
local SSE stream, counting reasoning and tool function data. Usage trailers and
the wait for response completion are left out. Buffered chunks limit timing
precision. JSON, missing usage, or a single generated delta cannot give a decode
rate.

### Child agents

**Native children are opt-in.** Add `--children`, or type `/children on` before
the first prompt of a fresh native conversation. This gives the parent
`child_spawn`, `child_list`, `child_wait` and `child_cancel`.

Children use the same endpoint, model, workspace boundary, contract client and
native loop, with fresh task histories and their own cache affinity. By default
children get the six native read/context tools and cannot spawn children of
their own. Child generation requires exact admission; the ordinary chat
fallback is only available to the parent.

**Scopes.** Host launch flags can declare named scopes:

- `--child-edit edits=src/example.rs` leases one exact file to the scope
  `edits`. Repeat it for each file, including both ends of a move.
- `--child-shell checks` explicitly grants sandboxed shell commands to the scope
  `checks`, with an exclusive workspace write lease.

The model can pick an existing scope through `child_spawn`'s `grant`, but cannot
create or widen one. Write leases that have not been collected block
conflicting writes by the parent and by siblings. A child never gets shell just
because the parent advertises it.

**Approvals.** Every child edit and command still follows the parent's
`--approval` policy, and noninteractive runs deny anything that needs approval.
Approval prompts come one at a time and name the child, and cancelling a child
clears its pending approval. See [scope examples and
limits](docs/child-agents.md).

**Watching and cancelling.** The raw TUI shows a live child list with a bounded
size. `/children` lists IDs and actual states. `/children cancel ID` asks one
child to cancel, while `/children cancel all` also stops further spawns for the
current activation. These commands leave the parent running, so it can collect
results and fold in what the children found. `/cancel`, exit, or the parent
finishing cancels and joins any remaining children. Cancelling reports local
cleanup only; it never confirms that the engine stopped computing. `/context`
includes each
child's last reported cached-token count and its cumulative reported counts,
separate from token reservations; absent telemetry is unknown.

**Changing the setting, and resume.** Child tool schemas are frozen along with
the parent prefix. To change the setting after a prompt, wait for cleanup, use
`/new`, then `/children on` or `/children off`. `/new` keeps the current
setting. Each activation gets fresh, finite child budgets, and the UI only keeps
child metadata and in-memory results for the current or last activation.

The parent's tool receipts stay in its typed journal. Native children have their
own private journals and build areas, and `child_wait` returns the child journal
name so you can inspect it. Child scheduler state is not restored or restarted
after a crash. Resume needs the same explicit child flags and scopes, and stored
descriptions never grant permission. See the [child lifecycle, quotas and
integration notes](docs/child-agents.md).

### Shell commands and the sandbox

On macOS every command runs in the OS sandbox (`sandbox-exec`). Inside it, a
command:

- runs with a cleared environment, the workspace as its working directory and
  no network;
- can read system and toolchain files;
- can write only to the workspace and a private build folder.

`workspace_shell` takes an explicit `argv` and `timeout_ms`; to run a script,
ask for `/bin/sh -c` explicitly. Hidden paths, common credential files,
`target/` and `node_modules/` stay out of reach, and hard links cannot be
created. Add up to eight more folder names with `--shell-exclude NAME`.

Builds and caches (Cargo, npm, HOME, XDG, temp files) go to a fresh private
folder outside the workspace. It lasts for the session and is thrown away on
exit, `/new`, resume, switch or fork. It is capped at 1 GiB / 100,000 entries,
checked between commands. Cargo and npm run offline. System and Homebrew tools
stay available.

If your Rust toolchain lives under your home folder, launch with
`--shell-toolchain DIR` pointing at the real distribution (with `bin/cargo`,
`bin/rustc`, `bin/rustdoc` and `lib`); rustup shims are not used. Package
installs are not possible, there is no unsandboxed fallback, and other
platforms refuse to run commands. Full rules: [agent
tools](docs/agent-tools.md#the-macos-sandbox).

### Approvals and command limits

There are three approval policies:

| Policy | What runs without asking |
| --- | --- |
| `--approval ask` (default) | Allowlisted forms of `pwd`, `true`, `ls`, `cat`, `git status`, `git diff` and `git log`. Edits, and commands outside the [read-only allowlist](docs/agent-tools.md#commands-and-approvals), wait for you. |
| `--approval auto` | Validated exact patches and all validated commands, inside the existing macOS sandbox, after preview. That includes workspace programs such as `./tools/rgbds/rgbasm` and explicit `/bin/sh -c` commands. |
| `--approval never` | Nothing: it denies both writes and commands. |

**Answering a prompt.** Under `ask`, type `y` or `n` on an empty line (or
`/approve` / `/deny`) in the terminal. The patch preview shows additions and
removals in colour. A `y`/`n` typed while there is a draft in the editor, during
`/paste`, or within 400 ms of the prompt appearing is kept as text. `/approve`
and `/deny` answer the displayed operation once, exactly like y/n, and sending
them never authorizes a later operation. With a draft already in the input, type
`/approve` or `/deny` on its own final line and press Enter; the earlier draft
stays. Pasted or recalled multiline text cannot trigger this final-line
shortcut.

**What auto still denies.** Under `auto`, writes stay limited to the workspace
and the private session build/cache/temp area. Network, hidden paths,
credentials and excluded trees stay denied. Auto never allows execution outside
that sandbox. There is no unsandboxed route: broader access is refused, and
unsupported platforms fail closed.

The UI shows the active policy, the command, the deadline and the patch diff.
Noninteractive or one-shot requests that need approval are denied. Approvals
expire after 120 seconds and never carry over into a resumed session.

**Command limits.** Each command gets:

- a 120-second maximum deadline;
- at most 64 KiB of displayed output (the native adapter reserves room for
  result framing and UTF-8 overhead);
- null stdin and bounded pipes;
- kernel limits on CPU, file size, processes and descriptors.

When output is too long, the harness keeps a bounded start and end and drops
the middle. It still drains both pipes, so long output does not stall the
command, and you still get its exact exit status. It reports how many bytes it saw and dropped,
and the elapsed time; the dropped middle cannot be recovered. Cancellation,
timeout and a normal child exit all kill the owned process group and reap the
child.

**What these limits do not promise.** They do not promise cleanup of a hostile
descendant that deliberately escapes that group, or an immediate interrupt of a
blocked local filesystem call. The tools assume no other process is changing the
workspace with hostile intent at the same time. Optimistic edit-conflict checks
cannot give an OS-level compare-and-swap against an unrelated writer. The
sandbox is a platform interface, not custom isolation or cryptography. We
checked these command limits without a GPU, the engine, a Spark, an installed
Codex or a live model.

### Sessions, forks and journals

**Where sessions live.** Native sessions are journaled by default, one per
workspace path, in the private `VADE_HOME/projects/NAME-HASH/sessions`
directory (`~/.vade` without `VADE_HOME`). They are never kept inside the
workspace. Input recall sits next to them in `history`. `--vade-home DIR` picks
another root.

`--state-in-project` keeps the legacy `WORKSPACE/.vade` folder, which then gets
a `.gitignore` of `*` (except `check`) so git ignores it. Every state folder
gets that `.gitignore`. Journals that older versions wrote to `WORKSPACE/.vade`
stay there; resume them with `--state-in-project`.

**Naming, resuming and switching.** `--session NAME` creates a named session;
otherwise a name is generated. `--resume NAME` explicitly restores a past
session, and the model and tool definitions must match. New journals also bind
to the device/inode identity of the workspace directory (no absolute path).

- `/sessions` lists sessions.
- `/resume NAME` or `/switch NAME` switches after cleanup.
- `/new` creates a fresh journal without deleting old sessions.
- `/undo` is still unavailable for tool histories.

Resume never sends a prompt or replays a tool on its own. Policy and endpoint
come from the new launch, never from stored approvals.

**Forks.** `/fork NAME [TASK_LABEL]` creates and selects a journal-only branch
while the agent is idle. It copies the bounded raw journal, so inherited
artifact IDs, expiry, summaries and receipts can still be inspected.
Installation is atomic and exclusive, and never overwrites an existing
destination. Parent records stay available, and messages after the fork are
independent. The label is for bookkeeping, not a prompt sent to the model.

`/branch` shows the parent checkpoint, the context generation and bounded
metadata checks for up to 128 explicitly referenced paths. These checks do not
verify content and are not a complete repository snapshot; commands may change
untracked files. Resume and fork add a revalidation notice before any future
model request. **Branches share the workspace: switching never restores files or
replays tools.** Unknown tool outcomes block new forks.

**Branch handles in the engine.** If the engine advertises
`kv_branch_handles`, native requests ask it to keep each request's input under
a handle. The harness saves each handle in the journal as a private hint that
expires. When you fork or switch branches, the harness sends the **complete
prompt** plus the matching hint, so the engine can reuse what it kept. If a
handle is expired, evicted, foreign or mismatched, the engine simply reads the
full prompt as usual. `/context` reports named-handle reuse separately from
ordinary cached tokens. Shared handle ownership survives branch switching, and
when the session shuts down the harness tries to release its handles, within
bounds. Without the capability, requests on the wire are unchanged.

**What the engine checkpoints.** The current engine contract checkpoints the
submitted **input**, not generated answers or a chosen system-only prefix.
Exact prefix or completed-answer checkpoints, and prefix-only children, need an
engine extension; no hidden warm-up inference runs. See [checkpoint boundaries,
quotas, lifecycle and child integration](docs/engine-checkpoints.md).

Forks reset the append-only request and the contract client, and the next
explicit prompt negotiates and admits the complete transcript. Only a response's
reported usage shows how much of the prompt came from the cache. A fork may copy
up to the journal's 64 MiB cap, private transcript content included; there is
no copy-on-write disk sharing or global quota across sessions you create by
hand.

**What the journal keeps.** The append-only journal keeps the exact typed
messages, reasoning, tool arguments, results, and stable model and tool
options. It does not store endpoints, the inherited environment, or automatic
workspace-path metadata. Prompts, files and commands you send can contain
sensitive information; journals are private transcripts, not a secret scanner
or encrypted storage. Direct, demo and Codex modes keep their separate
metadata-counter behavior (see [Workspace and privacy](#workspace-and-privacy)).
Locks are owned by the kernel and are released when the process exits. Files
use 0600 and owned directories use 0700, and redirects are refused.

**After a crash.** Before a tool runs, a synced record marks its outcome as
unknown. Completed results are synced before the agent continues. After a crash,
an explicit resume throws away only an incomplete final record and restores the
last closed call/result history. An `unknown_outcome` means you must inspect the
workspace before repeating any side effect. Complete but corrupt records are
refused.

A journal failure stops execution, and no further prompt runs against a failed
journal. Journals are capped at 64 MiB/8,192 records; start a new session when
one is full. The journal keeps old records even after you deliberately compact
the context.

### Context and compaction

`/context` shows how much of the context each message, the tools and the
options take. It also shows the real request size in bytes, and the last exact
count from the engine when it offers one. Local figures are labelled estimates,
not a tokenizer. `--context-tokens N` overrides the context size. Without it,
the engine's advertised limit is used (for example 262,144), or 131,072 when the
engine advertises none.

When a conversation gets close to the context limit (80%), the agent compacts
it: older turns are replaced with a shorter version so the next request fits.

- `/compact` does this now.
- `/compact-preview` shows the plan first.
- `/compact KEEP SUMMARY` uses your own summary.

Your messages, instructions, recent tool rounds, the latest edit for each file,
the latest run of each check, and the last answer stay word for word. No model
writes the summary.

Compaction has a cost. The engine's cache holds the conversation as one
unbroken sequence, so changing anything in the middle means everything after it
has to be read again, even text that did not change. `/compaction` and
`/journal` show exactly what changed, and the journal keeps the original
records. Limits, thresholds and `--prewarm-compaction` are in [context, project
rules and branches](docs/context-and-projects.md).

### Tool output, project rules and skills

**Large tool results.** Tool results larger than 4 KiB become labelled
excerpts, backed by the exact captured output in the private session journal.

- `tool_output_read` fetches UTF-8 byte ranges, at most 4,096 source bytes in a
  4 KiB result envelope; follow the returned `next_offset`.
- `/output ID OFFSET LENGTH` inspects a range locally.
- IDs are scoped to the active workspace and session, never paths.

Each saved capture is an **artifact**. Their limits:

- **Capture.** Each capture is capped at 64 KiB (commands reserve room for
  framing and UTF-8 overhead). The command status says when the source was
  truncated, and bytes that were never captured cannot be retrieved.
- **Quota.** A session keeps at most 128 artifacts and 4 MiB of captured
  output. The quota is checked before a tool runs, and a full quota stops tools
  until you start a new session.
- **Expiry.** Retrieval expires after 24 hours of wall-clock time. Expired
  captures still count toward the quota and stay in the append-only raw journal
  until you remove the session, and the separate 64 MiB/8,192-record disk limit
  still applies.

A saved artifact never replays a tool.

**Project rules.** Native project context loads the root `AGENTS.md`. For each
repeated `--rules-scope PATH`, it also loads only that directory's chain of
AGENTS.md ancestors inside the workspace (at most eight scopes, depth eight).
Root rules apply everywhere. Deeper rules apply only in their subtree and take
precedence there. Discovery is explicit: no recursive scan, and no search above
the workspace. At most 16 rule files, 16 KiB each and 24 KiB in total, go into a
frozen prefix in a deterministic order. Rules never grant executor capabilities
or override explicit user requests.

**Skills.** An optional `skills/catalog.json` lists up to 16 names, descriptions
and visible `skills/.../SKILL.md` paths. It is declarative JSON, at most 8 KiB;
scripts and unknown entry fields are rejected. Bodies are at most 16 KiB, and
one is read only when `skill_read` asks for it. A revision check on the local
metadata the session froze refuses a skill file that has changed or been
replaced since then. The exact skill text that is loaded, with its source and revision, is
recorded in the journal like any other tool result. No
script, installer or dependency runs automatically. The catalog is sorted by
name, and edits only become part of a new prefix after `/new`. Resume keeps the
original snapshot.

**Inspecting them.** `/project` shows the scopes, the reasons and the lazy
catalog. `/project-rules` shows the frozen rules, and `/skill NAME` inspects a
body locally. `/map [PATH]` and `/symbols PATH` give bounded, on-demand names
and lexical declaration hints, as do their model tools. These views are not
semantic indexing, and they are not inserted into context automatically. See
[formats, limits and design credits](docs/context-and-projects.md).

### Request limits and cancellation

Each explicit user prompt allows at most:

- 60 tool rounds. A tool round is one model reply that asks for tools, and its
  calls share that round;
- 61 model turns;
- 240 proposed tool calls;
- 64 KiB per tool result, and 256 KiB of results in total.

The native agent also stops after 20 minutes of active work per request; time
spent waiting for your approval does not count. It cleans up the current tool,
keeps the conversation and journal, and says "Stopped after 20 minutes; say
continue to keep going". Say `continue` to pick the conversation back up with a
fresh request budget.

`--agent-time-limit-seconds N` changes the time limit, and `0` turns it off. To
save the same setting, load a JSON file explicitly with `--config PATH`,
containing `{"agent_time_limit_seconds":1200}`. A missing key uses the 20-minute
default, and the CLI time-limit flag wins over the file. Both options need
native `agent` mode.

These limits apply with or without `--children`, and child tasks keep their own
quotas. The wire/history byte and JSON complexity bounds still apply (4 MiB
request/history). Assistant text and tool previews each have a 1 MiB display
bound per run.

**On cancellation or failure,** completed tool results stay as closed history,
and there is no automatic retry. New requests, session changes and compaction
wait for worker cleanup. Only a validated final `stop` answer counts as a
successful turn. Exit cancels and joins the worker.

**How far cancelling reaches.** Native mode keeps work, control and screen
updates in separate bounded queues. Its network reads check for cancellation,
and requests with a deadline (cancellation controls included) also limit DNS
waiting and connect time. A DNS lookup or connect already in progress cannot be
interrupted, and an OS lookup that timed out may finish later without sending a
request. A blocked terminal write cannot be interrupted either. All of this is
about local work: cancelling never guarantees that the engine stops computing.

## Terminal interface

Use `--prompt TEXT` to get one answer and exit.

**Steering.** In a normal session, typing a new message while the model is
still answering **steers** it. Vade cancels the current reply, keeps what was
written so far (marked as partial), and sends your new message. If you would
rather wait your turn, `/queue on` lines up follow-ups instead. Piped input can
still send control commands while a reply is running.

Steering works directly against the engine and through a single-upstream
gateway, which forwards the cancel. A gateway with several upstreams cannot
cancel yet, so your follow-up waits; connect straight to the engine if you need
to steer.

At end of input (EOF), Vade finishes the current work and any queued prompts
that can run, then lists any held prompts as unsent. `--plain`, redirected
input or output, or `TERM=dumb` switch to simple line input; `NO_COLOR` turns
off color only.

**How a conversation looks.** A native TTY launch creates a fresh interactive
journal; only `--resume NAME` loads an earlier conversation. The input wraps as
you type and when you resize. The conversation is built from four kinds of
block, each with a one-line label and one blank line before the next block:

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

- **You** marks your message.
- **Thinking** is dim and starts collapsed to one summary row. Ctrl+T expands it
  without hiding the reply.
- **Answer** renders basic Markdown (headings, lists, code, rules, quotes,
  links). It ends with one dim footer: finish time, elapsed time, tokens
  written, write and read speed, time to first token, model calls (requests sent
  to the model) and context use. Missing measurements show `—`. Narrow
  terminals drop the last facts first and keep elapsed time, tokens and write
  speed.
- **Warning** (yellow) always says what happened, what Vade is doing about it,
  and what you can do.

A run that does not finish normally ends with one plain line such as
`Canceled · <elapsed> · <edits> · check passing · say continue`. The labels
carry the meaning with `NO_COLOR` and in plain or redirected output. The UI uses
the terminal's own foreground and background, so it works in light and dark
themes. `/status` shows the full counters; `--verbose` also prints the contract
details after each run.

The terminal owns mouse selection and Cmd+C, and Vade does not turn on mouse
reporting. PgUp/PgDn scroll the conversation and full approval previews.

**The status row.** The footer row above the input always shows facts. At rest
it shows `model · folder · approval · ctx used/limit · session`. While waiting,
the spinner, the state (`connecting`, `thinking`, `reasoning`, `running tool`)
and the clock sit where the answer will appear. While writing, the row shows the
clock, estimated tok/s, queued follow-ups and the cancel hint. Narrow rows drop
middle facts first. The input box shows its newline keys until you send the
first message.

**Tool rows.** Tool calls print inside the answer flow as `● Run: cargo test`
rows. Each is followed by one indented, dim
`⎿ exit <code> · <lines> · <elapsed>` summary (exit code, line counts, first
error line, applied operations or the written file). Full command and patch
output needs `--verbose`, which describes the command sandbox once, and again
only when it changes.

**The command menu.** `/` opens a menu of the common commands, and typing
searches every command for the mode. ↑/↓ select, and Enter runs the selection
(an ambiguous prefix is completed into the editor first, as Tab does). Esc
closes the menu.

## Commands and keys

| Command | What it does |
| --- | --- |
| `/help` | Show commands and keyboard controls |
| `/children` | Show native child IDs and states; `on`/`off` before the first prompt; `cancel ID`/`cancel all` during work |
| `/status` | Show mode, request state, queues, history bytes/turns, and metadata status |
| `/new` | Create a new native journal, reset direct/demo history, or start a fresh external session; waits for cleanup |
| `/sessions`, `/resume NAME`, `/switch NAME` | List or explicitly select native journals |
| `/fork NAME [TASK_LABEL]`, `/branch` | Fork a transcript, or inspect ancestry and file changes |
| `/context` | Show native full-request estimates, margin, reserve and bytes, plus parent/child admission and reported cache usage |
| `/output ID OFFSET LENGTH` | Inspect a retained output range locally |
| `/compact-preview KEEP SUMMARY` | Preview a compaction without changing context |
| `/compact [KEEP SUMMARY]` | Compact older context automatically, or with your own summary; constraints, receipts and recent turns stay verbatim |
| `/compaction` | Inspect the summary, source range, generation and KV invalidation |
| `/journal RECORD OFFSET LENGTH` | Read a raw journal range locally |
| `/project`, `/project-rules` | Inspect frozen project context and rules |
| `/skill NAME` | Inspect a catalog skill locally; no scripts execute |
| `/map [PATH]`, `/symbols PATH` | Show bounded repository names, or lexical hints |
| `/approve`, `/deny` | Answer a pending one-time native tool approval; `y`/`n` on an empty line do the same |
| `/undo` | Remove the last direct/demo exchange; refused in native/delegated modes |
| `/cancel` | Ask to cancel; an acknowledgment shows only once it is received |
| `/steer TEXT` | Send input to the active Codex turn, with separate queued/accepted/rejected receipts |
| `/queue on`, `/queue off`, `/queue TEXT` | Queue follow-ups instead of the default steering, switch back, or queue one follow-up |
| `/next`, `/queue-clear` | Resume a held queue, or discard queued follow-ups |
| `/ls [path]` | Bounded listing relative to the workspace |
| `/read path` | Local UTF-8 preview, at most 64 KiB; never sent to the model |
| `/paste` | Collect lines until `/end`; `/cancel` discards them |
| `/quit`, `/exit` | Exit |
| `//text` | Send a prompt that starts with a slash |

In the raw terminal, these keys edit and send your input:

| Key | What it does |
| --- | --- |
| **Enter** | Send |
| **Shift+Enter** | Insert a newline, when the terminal reports the modifier through CSI u or modifyOtherKeys |
| **Option+Enter** or **Ctrl+J** | Insert a newline in Terminal.app, which does not report Shift+Enter by default; the input box shows the hint |
| Left/Right, Home/End, Delete, Backspace | Edit at a UTF-8 cursor |
| Up/Down | Recall messages you sent in the current session, multiline ones included; Down past the newest brings back your unsent draft |
| Ctrl+Up/Down or Option+Up/Down | Move between lines, when the terminal reports those keys |
| Ctrl+U | Clear input |
| Ctrl+K | Delete to the end of the line |
| Ctrl+W or Alt+D | Delete a word |
| Alt+B/F or Ctrl+Left/Right | Move by word |
| Ctrl+L | Repaint |

Recall is bounded and separate from model context and journal history; `/new`
or picking another session clears it. Home/End and Left/Right stay available for
editing multiline input.

The editor shows the end of long input and adapts to the terminal width. A
bracketed multiline paste lands as one editable message and does not send until
Enter. Input stays editable while work runs. The cursor moves by Unicode scalar,
with approximate terminal cell widths; grapheme-cluster editing and autocomplete
are not implemented. On resize, a fresh frame avoids erasing old reflowed rows.

**Interrupting.** **Esc** or **Ctrl+C** while a response streams interrupts it
and takes you back to the prompt. Direct and native modes keep the partial prose
with an `[interrupted]` marker, and native mode also keeps completed tool
results. Esc never throws away a draft (Ctrl+U clears it).

Direct mode uses the core's cancellable HTTP transport, and pending socket I/O
polls for cancellation every 250 ms. OS DNS lookup and connection setup have
their own limits, and, as in native mode, closing the socket does not guarantee
that the engine's work stops. Native mode cancels the runtime. Codex mode asks the core
adapter to interrupt the turn and finish cleanup.

**Queued follow-ups.** In delegated modes, or with `/queue on`, a prompt you
type during work becomes a separate queued follow-up (at most eight prompts /
64 KiB total). It runs after success; a failure or cancellation holds the queue
until you type `/next`. Local commands and exit stay available. Codex `/new`
waits for in-flight cleanup before it changes threads.

Failed direct exchanges, and canceled or failed demo/Codex exchanges, are left
out of local recall. Interrupted direct exchanges stay in conversation history,
and native mode keeps its closed typed history, completed tool results included.
Ctrl+C clears idle input, then exits when the prompt is empty. Ctrl+D sends
nonempty input or exits.

**Line mode and exit.** In line mode, Ctrl+C exits. Native and Codex modes also
cancel and join their workers before exit, and Codex drops its client. Codex
initialization watches for cancellation while waiting on child I/O.

The UI reports an interruption only after the stream's final event says the
turn was interrupted. A rejected interrupt, a timeout or a broken stream leaves
the remote outcome unconfirmed. If a reply completes just as you cancel, it is
reported as a remote completion and the local result is discarded. Raw terminal
state is restored on normal exit, I/O errors, unwinding, SIGINT, SIGHUP and
SIGTERM. Termination that cannot be caught cannot run cleanup.

## Workspace and privacy

`--workspace PATH` picks the root. You must pass it explicitly for native agent
and Codex modes; direct and demo default to the current directory.

Read and list operations use directory handles on macOS and Linux. They refuse
absolute paths, parent traversal, symlink traversal, hard-linked reads,
non-regular files, hidden paths and common credential filenames. Directory
listing visits at most 256 entries, filtered entries included, and a limit
marker signals an incomplete list. Other platforms refuse these operations
rather than weaken the boundary.

**What reaches the model.** Vade's explicit `/ls` and `/read` results stay
local. The direct/demo UI has no automatic context discovery, workspace writes,
subprocess tools or model-directed tool execution. Results of the tools the
native model asks for are sent to the selected model. No root path is inserted
automatically, though explicit command output (such as `pwd`) can contain it.
Delegated Codex tools follow the external policy you picked. Terminal output
filters control sequences and directional overrides. File filtering is not a
secret scanner for content, and explicit previews are visible in terminal
scrollback.

**What is stored.** Direct/demo recall stays in process memory, and delegated
agents own their persistence. For these modes, the metadata store writes no
transcript, provider/thread identifier, endpoint, model name or file contents.
The optional `--session NAME` saves only a count of completed turns, in
`sessions/NAME.state` in the same private state folder. It does **not** restore
a conversation. Use a generic session name, since the name becomes a filename.

The directories must be owned by the current user with private permissions.
Files are created as 0600, updates use atomic replacement, and symlink
redirects are refused. A small private lock file stops two copies of the
harness from updating the same counter at once, and the lock is released as
soon as the session ends. Default demo mode creates no files. Native agent
journals are described under [Sessions, forks and
journals](#sessions-forks-and-journals).

## API authentication

Use `--api-key-file PATH` on any harness command or on `vade-gateway`, or set
`VADE_API_KEY_FILE`. The flag wins over the environment.

The file holds the same key as the server: trimmed, single-line, at least 16
bytes, using only visible ASCII `0x21-0x7E`. We suggest 32 random bytes encoded
as hexadecimal. We also suggest protecting remote HTTP with an SSH tunnel or TLS
termination.

**When the file is read.** The harness picks the path at startup and reads it
once before agent/direct UI startup or gateway serving. Other commands read it
just before their first upstream HTTP request. Help, version, setup output and
local workspace commands work even if the file is missing or unreadable.

**Where the key goes.** The harness sends `Authorization: Bearer <key>` on
discovery, admission, generation and checkpoint release, child agents included.
Gateways use it for upstream requests and replace incoming credential headers
when configured. HTTP 401 is shown as
`HTTP 401 Unauthorized: authentication failed`. Keys are never logged or
written into session state; restart to rotate the file. The server's `/health`
stays open.

The standalone gateway also applies the selected key to capability and admission
probes and to recorded replay. Without a configured key, it forwards the
client's credential headers unchanged. A configured key replaces both
`Authorization` and `X-Api-Key` on upstream requests, and response headers stay
opaque. Recordings leave out credential headers, so authenticated replay needs
its own key file.

## Measure concurrent sessions

`vade-harness workload --endpoint URL --model NAME --json FILE` runs several
native sessions at once, to show how the engine holds up under load. By default
you get one main session and five smaller ones, eight turns each. Tool results
are made up and repeatable, histories grow each turn, and no real tools run.
You get a table by role and session, plus a new private JSON metrics file.

From turn 2 on, each role reports:

- cache ratio, TTFT percentiles and output speed;
- admissions, overloads and errors;
- each turn's C and U, with the basis for each count. C is the cached prompt
  tokens the server reports. U is the estimated part of the prompt shared with
  the previous turn.

U is always marked **estimated**. It is the longest common prefix in tokens,
and the current contract does not return token IDs to compare. When the engine
offers exact admission, the kept history (full assistant reply included) is
counted on its own, so generated-output counts are never treated as rendered
input. Read the [workload guide and coordinator command](docs/workload.md)
before you compare results with the 0.98 cache-reuse target. `workload --help`
lists the size, concurrency, output and deadline flags.

## What has been validated

The doctor checks the capabilities the server advertises, admission (the
engine's check that a request fits), and response telemetry. A server that
declares `prefix_cache:false` does not fail just because nothing came from the
cache. If caching is advertised, missing or zero cached tokens on the second
turn fail the cache check. Malformed telemetry and other probe failures still
fail.

Older measurements and their limits are in the [gateway measurement
records](docs/gateway-measurements.md). Host fixtures (tests that run on your
computer without a GPU) cover the protocol and the tools. They say nothing about
current model performance, and they do not prove that a client you have
installed works with Vade.

### Run the tests

```sh
cargo test --offline
```

Local tests cover workspace boundaries, crash-tail replay, journal failures,
mock HTTP sessions, subprocess protocol fixtures and terminal approval handling.
The [CI workflow](../.github/workflows/ci.yml) runs host checks with warnings
denied. Use the test output from your current checkout for counts and platform
limits.

Synthetic stream measurements and their limits are kept in [performance
methodology](docs/performance.md) and its linked sample files.

### Bridge checks

The Codex and Hermes adapters are checked against our own protocol mocks and in
a Mac pseudo-terminal. No installed Codex or Hermes, and no engine or Spark, was
used in those checks, so they do not prove live compatibility; see [the bridge
contract](docs/bridge-contract.md).

## Design references

These projects shaped our UI and interface choices:

- [Hermes Agent](https://github.com/NousResearch/hermes-agent): visible
  conversation controls, multiline terminal entry, and interruptible
  interaction.
- [Pi coding
  agent](https://github.com/badlogic/pi-mono/tree/main/packages/coding-agent):
  a minimal harness with interactive and noninteractive operating modes.
- [Codex CLI](https://github.com/openai/codex): a local terminal entry point for
  a coding assistant.

These projects are conceptual references, not dependencies. No implementation
code from them, or from any other competing project, was copied. Vade's
workspace boundary, session journal, coding tools, and core/UI contract are
first-party implementations for this project.
