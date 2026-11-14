# Other clients through the gateway

This page shows how to use your own coding client with Vade.

Setup can write a complete dedicated home for Claude Code, Codex, Hermes,
OpenCode and Pi. It generates the configuration offline. The gateway relays
the API that you pick. It keeps request bodies, tools, cache identity,
response usage, SSE and errors as they are. The separate engine owns
inference and KV caching.

For a current FP4-Local server, follow [INSTALL.md](../../docs/INSTALL.md):

- **On the Spark itself.** `vade launch` or `vade up` starts an engine that
  keeps running. It opens no tunnel.
- **On a controller.** `vade up` starts the engine, opens a detached SSH
  tunnel to the configured local port (default `18080`), and returns.

The engine, and on a controller its tunnel, stay up until `vade down`. Direct
and gateway clients can connect at any time in between. Read the exact model
ID and limits from the capabilities, as shown below.

If the harness is not on your PATH, use `harness/target/release/vade-harness`
from the repo root instead of `vade-harness`.

## Create a dedicated home

Create an empty scratch parent. Then pick a **new** child directory inside
it.

Both `--context-window` and `--max-output-tokens` must be values that you have
verified on the upstream. Setup does not discover or make up limits. Read them
from the running engine's `/v1/capabilities`, together with its exact model ID.
The example uses `curl` and `jq`.

FP4-Local starts with context 32768 and output `min(context, 8192)`. This is
true when you start it from a controller (on any number of Sparks) and with
`vade launch` on the Spark. `vade launch` lowers a default context only when
the engine refuses it for memory. `--context 262144` picks the single-chat
long-context preset on either route. Do not assume 32768. Read the running
server's capabilities again after every launch change.

```sh
UPSTREAM=http://localhost:18080
CAPABILITIES=$(curl -qfsS "$UPSTREAM/v1/capabilities")
MODEL=$(printf '%s' "$CAPABILITIES" | jq -er '.model')
CONTEXT_WINDOW=$(printf '%s' "$CAPABILITIES" | jq -er '.context_limit | select(. > 0)')
MAX_OUTPUT_TOKENS=$(printf '%s' "$CAPABILITIES" | jq -er '.output_limit | select(. > 0)')
mkdir -m 700 vade-scratch
vade-harness setup pi --endpoint "$UPSTREAM" --model "$MODEL" \
  --target gateway --dedicated-home --write ./vade-scratch/pi \
  --context-window "$CONTEXT_WINDOW" --max-output-tokens "$MAX_OUTPUT_TOKENS"
vade-gateway --upstream "$UPSTREAM"
```

Build the standalone binary as described in [Standalone
gateway](invisible-gateway.md). Use the same advertised limits for the Codex,
Hermes and OpenCode setup homes.

From a separate terminal, change to your workspace. Run the generated
`launch.sh` by its absolute path. It accepts the client's normal arguments.

These setup options change the target and the client:

- **`--target direct`** points the client at the upstream directly.
- **`--gateway URL`** picks a gateway base that you configured separately.
- **`--client-bin PATH`** picks a pinned executable. Without it, the client is
  found on PATH.
- **`--write PATH` without `--dedicated-home`** writes only a fragment, as
  before.

The parent folder must already exist and contain no symlinks. Give it by its
real path (`pwd -P` prints it). Setup does not write over anything that
exists. It does not write through a final symlink, outside the parent (`..`),
or inside `.codex`, `.hermes`, `.pi` or `.config`. Files are created with mode
0600. Folders and the launcher get 0700. Setup never reads or merges your real
client settings. If a write fails, setup tries to remove only what it just
created. Check the half-written folder before you retry at a new path.

The launcher starts the client in a clean environment (`env -i`). It keeps PATH
and your terminal settings and gives the client its own HOME, TMPDIR, XDG
folders and client home. Your credentials, provider overrides and shell config
are not passed in. Where the client allows it, startup update checks are turned
off and the local provider is selected. Each client keeps its own safety rules:

- **Codex** runs in its read-only sandbox.
- **OpenCode** allows read and search tools.
- **Pi** selects read and search tools.
- **Hermes** offers file tools, but it can write only inside the launch
  workspace (`HERMES_WRITE_SAFE_ROOT`).

This does not give every client Vade's strict native read boundary or an OS
sandbox.

Each home contains this single instruction line:

> When a task needs a tool, call the tool immediately; do not end the turn with a promise such as 'Let me check' before calling it.

This is a rule at the prompt level, so it cannot guarantee how the model
behaves. Project instructions and explicit user requests can still shape the
final prompt.

### What setup writes for each client

Each client name below includes the client version that we checked locally.

**Claude Code 2.1.310**

- **Files.** `settings.json`, `.claude.json`, `CLAUDE.md`, `launch.sh`.
- **Environment.** `CLAUDE_CONFIG_DIR` plus the gateway variables below.
- **One-shot.** `-p PROMPT --output-format json`.
- **Interactive.** No extra arguments.
- **API.** Messages at the API root. No Anthropic account or key.

**Codex 0.160.1**

- **Files.** `config.toml`, first-party `models.json`, `AGENTS.md`,
  `launch.sh`.
- **Environment.** `CODEX_HOME`.
- **One-shot.** `exec --skip-git-repo-check --json PROMPT`.
- **Interactive.** No extra arguments.
- **API.** Responses HTTP/SSE. No OpenAI authentication, WebSockets or shared
  daemon.

**Hermes 781334eea4b**

- **Files.** `config.yaml`, `SOUL.md`, `launch.sh`.
- **Environment.** `HERMES_HOME`, `HERMES_WRITE_SAFE_ROOT`.
- **One-shot.** `-z PROMPT`.
- **Interactive.** `chat`.
- **API.** Chat Completions with API streaming enabled.

**OpenCode 1.18.18**

- **Files.** `opencode.json`, `AGENTS.md`, `launch.sh`.
- **Environment.** `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, private XDG state.
- **One-shot.** `run --format json PROMPT`.
- **Interactive.** No extra arguments.
- **API.** `@ai-sdk/openai-compatible`, Chat Completions.

**Pi 0.73.1**

- **Files.** `models.json`, `settings.json`, `AGENTS.md`, `launch.sh`.
- **Environment.** `PI_CODING_AGENT_DIR`.
- **One-shot.** `--mode json --print PROMPT`.
- **Interactive.** No extra arguments.
- **API.** `openai-completions`, with the system role, `max_tokens` and
  streaming usage.

### Limits for Hermes, Codex, Pi and OpenCode

The pinned Hermes custom provider rejects contexts below 64,000 tokens before
it sends a request. For a smaller context that you supply explicitly,
dedicated setup uses Hermes' built-in LM Studio compatibility profile. It
keeps the exact base URL and model. Hermes' initialization code allows this
explicit smaller window. This does not say that the upstream is an LM Studio
server. It only picks a client compatibility profile. Ordinary fragments keep
`provider: custom`.

`--max-output-tokens` sets the Pi and OpenCode model limits and the local
context reservation. Codex and Hermes own their wire-level generation limits.
For this reason, the flag cannot guarantee that their requests carry a matching
output cap.

Codex gets an original short coding prompt and a local model catalog instead
of unknown-model defaults. For Codex, the upstream must implement Responses.

The Codex home sets `model_reasoning_effort = "none"`, so the model answers
without thinking first. A model whose name contains `glm` cannot turn thinking
off, and its engine refuses that effort with HTTP 400 on every request. For
such a model the home names no effort, and the model thinks before it answers.

## Check a client yourself

To check a client yourself, run every check twice: client → upstream, and
client → gateway → the same upstream. Use a separate new home for each route
and session, and an identical scratch workspace. No check needs personal
instructions, credentials, remote tools or project data.

Each check below gives the fixed task or method, then the evidence to keep.

1. **Configuration.** Generate the home and run the pinned client without
   additional provider overrides. Keep the exact pin, the generated files,
   the flags, the API route and the selected model. Verify that the tool rule
   appears in the sent instructions.
2. **Chat.** Send `Reply exactly OK.` Keep the successful terminal answer, the
   finish reason, the streamed request flag and the usage.
3. **Tool use.** Put `TOKEN_R1_314` in `marker.txt`. Ask
   `Read marker.txt using a tool and report its exact contents.` Keep a real
   read-tool invocation, the successful tool result, the next model request
   and the final marker. Prose that promises a tool is a failure.
4. **Follow-up and cache.** Continue the same session with
   `Repeat the exact marker from the previous answer without reading it again.`
   Keep the prompt, output and cached counts of each request, the stable prior
   tool result, and `100 * cached / prompt`. Missing usage is UNKNOWN.
5. **Streaming.** Delay the mock's stream completion after its first answer
   or tool event. The first meaningful answer or tool event must reach the
   client before terminal completion. Role-only events are not TTFT.
6. **Timing.** Discard one warm-up for each profile. Use fresh upstream
   sessions in direct, gateway, gateway, direct order. Keep the median request
   TTFT, the output-token count and the decode-only rate
   `(output_tokens - 1) / (last - first)`. Keep the time from startup to the
   visible answer separate.
7. **Errors.** Make the mock return HTTP 400 with a synthetic structured
   error. Also check truncation and disconnect. Keep the visible client
   failure, the preserved upstream status and body, the bounded retries and
   the absence of a false successful answer.

The cache percentage comes from each completed upstream response. It does
not come from a guess at prompt similarity or from gateway affinity. Some
clients subtract cached input when they display billable input. Add the
cached input back to get the total prompt occupancy. The gateway leaves cache
fields alone unless an explicit optimization flag asks for cache decoration.

Requests without any sampling field get the shared greedy defaults. Any named
sampling field, even null, turns all defaults off. Record the actual request
settings. Sampling that the client chose is not the same as a request that
leaves sampling controls out.

The pinned Pi JSON mode can exit zero while it reports `stopReason: error`. For
this reason, automation must check the terminal result as well as the process
exit code.

The first local pass completed streamed chat and the read-tool round trip in all
eight combinations of client and route, against a first-party Rust mock. Mock
usage is synthetic, so its cache counts and token rate show only that the
transport is compatible. Live interoperability, real prefix reuse and engine
throughput need separate measurements.

The standalone gateway waits for new connections in a blocking accept. This
avoids a 25 ms polling delay. The cancellable API for embedding the gateway in
other programs still polls for connections, with a bound.

The live OpenCode and Pi results date from **2026-10-20 and used no tools**.
They are four direct and gateway smoke tests on the older commit `26b1692`.
Each test is a one-line answer capped at 128 tokens. Hermes could not run
because of missing dependencies. These results show no tool use. They also
show no live run of Claude Code or of dsh (DeepSeek's coding harness). See the
[gateway proof](gateway-measurements.md).

## Offline reference-client proof (2026-10-23)

`tests/passthrough_clients.rs` starts the real `vade-gateway` and
`vade-harness gateway` executables. An independent HTTP decoder sends 27
synthetic cases through the raw upstream and through both processes. That
makes **81 dispatches and 54 paired relay comparisons**. The three dialects
are Chat Completions, Responses and Messages. Each dialect runs the same nine
cases. The tool call and the tool result are two separate cases that share a
row below:

<div align="center">

| Check | Required result |
| :---: | :---: |
| JSON answer | Body and selected end-to-end headers stay exact |
| Fragmented SSE | First bytes arrive before the upstream can finish; a split UTF-8 character remains exact |
| Tool call and tool result | Parse the native schema, read the fixture file, and forward the actual result in the continuation |
| Usage | Preserve the upstream's reported counts without inventing a cache hit |
| HTTP failure | Keep status 429, Retry-After and opaque error body |
| Explicit null | Preserve every request byte, including null and lexical number/string forms |
| No sampling fields | Append only the established greedy defaults |
| Truncated SSE | Preserve delivered bytes and incomplete framing; do not manufacture successful termination |

</div>

The tool checks do **nine real file reads**, one for each API on each route (raw
upstream, `vade-gateway` and `vade-harness gateway`). The test server checks the
path, body and chosen headers of every request. It also checks that each case
was sent exactly once. These are our own reference clients with made-up
responses, so they say nothing about installed vendor clients or engine speed.

A separate test covers the completions endpoint: both aliases, explicit and
default sampling, and token-ID payloads in JSON and SSE. The earlier 27-of-27
live answer comparison still stands for what it covered. This test does not
replace it.

### Installed client headers (2026-10-30)

On 2026-10-30, Claude Code 2.1.314 (Messages), Codex 0.165.0 (Responses) and the
openai Python client 3.41.0 (Chat Completions) sent their requests to a loopback
capture server. The test replays each client's request headers through the raw
upstream and through both gateways. These headers include the `X-Stainless-*`
set, `anthropic-beta`, a JSON-valued `x-codex-turn-metadata` and
`Accept-Encoding`. The IDs in the test are fixed values with the same shape. The
upstream gets every end-to-end header in the same order, and each value is
unchanged byte for byte. The core gateway sends header names in lower case. HTTP
header names are not case-sensitive. The only other headers are `Host` and the
relay's own framing. The request body, the status, five response headers
(request ID, rate limit, timing and `Retry-After`) and the response body are
also unchanged. This test uses the clients' header sets, not the running
clients. A live run of the clients through the gateway to the engine is a
separate check.

From `harness/`, run:

```sh
cargo test --offline --test passthrough_clients -- --nocapture
cargo test --offline --lib recorded_engine_completions
```

## Claude Code

`vade-harness setup claude` prints, or with `--dedicated-home` writes, the
settings from Claude Code's [gateway
protocol](https://code.claude.com/docs/en/llm-gateway-protocol) page. Use the
`MODEL`, `CONTEXT_WINDOW` and `MAX_OUTPUT_TOKENS` you read above:

```sh
vade-harness setup claude --endpoint "$UPSTREAM" --model "$MODEL" \
  --target gateway --dedicated-home --write ./vade-scratch/claude \
  --context-window "$CONTEXT_WINDOW" --max-output-tokens "$MAX_OUTPUT_TOKENS"
```

- `ANTHROPIC_BASE_URL` is the API root. Setup drops the `/v1` that the other
  clients use, and keeps any proxy prefix in front of it.
- `ANTHROPIC_AUTH_TOKEN=no-key-required` is a dummy bearer value, so Claude
  Code never asks for an Anthropic login. The launcher's private
  `CLAUDE_CONFIG_DIR` and `HOME` keep your real `~/.claude` untouched.
- `ANTHROPIC_MODEL`, the `opus`/`sonnet`/`haiku` aliases and
  `CLAUDE_CODE_SUBAGENT_MODEL` all name the one loaded model. This way,
  background tasks and subagents do not ask for a Claude model that the engine
  does not have.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` carry the
  verified limits. Without the output cap, Claude Code assumes 32000 output
  tokens for a model it doesn't know and compacts a 32768-token window early.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `ENABLE_TOOL_SEARCH=false` and
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` keep pre-release request fields,
  MCP tool search, telemetry and update checks out of the session.
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` removes the billing line that Claude Code
  puts first in its system prompt. The last characters of that line change
  with every session. With the line there, a new session shares no prompt
  start with the session before it, so the engine reads the whole prompt
  again. Checked with Claude Code 2.1.322 on 2026-11-11: two sessions shared
  66 of 61,695 rendered prompt bytes with the line and 52,866 of 61,609
  without it.
- The home's `settings.json` allows Read, Grep and Glob and denies shell, edit,
  write and web tools, like the other homes' read-only defaults. Edit it to
  give Claude Code more.
- The same `settings.json` carries every setting above in its `env` block.
  A tool that starts Claude Code with only this home as its config directory
  (`CLAUDE_CONFIG_DIR`), and does not run `launch.sh`, then reaches Vade too.
  Such a tool passes its own environment, so `launch.sh` stays the route that
  keeps your credentials and shell config out.
- Claude Code 2.1.310 sends its environment notes as a `role: "system"` entry
  inside `messages`. The engine accepts only user and assistant turns there
  and answers 400, and Claude Code does not retry. Both gateways relabel only
  those role values as `"user"` on Messages and token-count requests. This is
  the recovery that Claude Code documents, so use `--target gateway`. Direct
  to the engine, only `claude --bare` works until the engine accepts them.
- Images are not supported yet.
- Sampling works as for the other APIs. With no sampling fields, the gateway
  adds greedy defaults. Any field that the client sends stays as is.

## aider, Continue and Open WebUI

`vade-harness setup aider|continue|open-webui` prints each client's documented
OpenAI-compatible settings for the engine or the gateway. They have no dedicated
home. aider and Open WebUI read environment variables. Continue reads the
`config.yaml` that you paste into its settings.

- **aider** gets `OPENAI_API_BASE`, a dummy key and `--model openai/MODEL`.
  aider warns that it does not know the model's limits. That warning is
  expected.
- **Continue** gets one model for chat, edit and apply. Pass `--context-window`
  and `--max-output-tokens` to add `defaultCompletionOptions`. Tab autocomplete
  is left out on purpose. The engine ignores the fill-in-the-middle `suffix`
  field, so autocomplete would only continue the text before the cursor.
- **Open WebUI** gets `OPENAI_API_BASE_URL` and lists models from
  `/v1/models`, which both gateways serve. In Docker, use
  `http://host.docker.internal:8787/v1`, run the container with
  `--add-host=host.docker.internal:host-gateway`, and start the gateway with
  `--allow-host host.docker.internal`. Otherwise, the gateway's Host guard
  answers 421.

None of these three has been run live yet.

## dsh recipe (not live-qualified)

dsh is DeepSeek's coding harness. To try it, follow these steps:

1. With `dsh` already installed, launch `dsh web` in the workspace that you
   want.
2. Open **Settings → Models → Add model provider → Custom model API**. Set
   provider ID `vade`, base URL `http://127.0.0.1:8787/v1`, and API protocol
   **OpenAI Chat Completions**.
3. For a local engine without a key, enter the literal dummy
   `no-key-required`.
4. Add a model by hand: ID = `MODEL`, context window = `CONTEXT_WINDOW`, max
   output tokens = `MAX_OUTPUT_TOKENS`, and input types = **Text** only. Use
   the values of those variables.
5. Save and select that model for a new session.

The [provider
guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md)
describes these fields. Discovery alone does not supply Vade's missing model
limits.

Routes you declare by hand need a nonempty `models` list. The [pi-ai
adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-pi-ai/README.md)
can move later system text into the initial prompt. Verify prefixes before you
claim cache reuse or client exactness. No dsh tool round trip is proven here.

## Sources and credit

Setup writes original Rust templates and a short original tool rule. No
implementation or prompt text from these or other competing clients is copied
into the harness. The pinned clients' local help and documentation establish
their command and configuration contracts.

- [Claude Code gateway
  protocol](https://code.claude.com/docs/en/llm-gateway-protocol) and
  [environment variables](https://code.claude.com/docs/en/env-vars): base URL,
  bearer token, model aliases, context/output limits and beta controls.
- [aider OpenAI-compatible
  APIs](https://aider.chat/docs/llms/openai-compat.html), [Continue
  config.yaml reference](https://docs.continue.dev/reference) and [Open WebUI
  quick start](https://docs.openwebui.com/getting-started/quick-start/).
- [Codex configuration
  schema](https://learn.chatgpt.com/docs/config-file/config-reference):
  provider, Responses, local model catalog and instruction settings.
- [Hermes pinned context
  initialization](https://github.com/NousResearch/hermes-agent/blob/781334eea4b/agent/agent_init.py):
  the custom-provider floor and explicit LM Studio context exception; [file
  safety](https://github.com/NousResearch/hermes-agent/blob/781334eea4b/agent/file_safety.py)
  defines `HERMES_WRITE_SAFE_ROOT`.
- [OpenCode providers](https://opencode.ai/docs/providers/) and
  [permissions](https://opencode.ai/docs/permissions/): compatible providers,
  model limits, explicit instructions and tool permissions.
- [Pi model
  configuration](https://github.com/earendil-works/pi/blob/v0.73.1/packages/coding-agent/docs/models.md)
  and
  [settings](https://github.com/earendil-works/pi/blob/v0.73.1/packages/coding-agent/docs/settings.md):
  compatibility fields, explicit model limits and agent-directory settings.
