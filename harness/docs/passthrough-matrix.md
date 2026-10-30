# Other clients through the gateway

This page is for when you bring your own coding client. Setup can write a
complete dedicated home for Claude Code, Codex, Hermes, OpenCode and Pi, and it
generates the configuration offline. The gateway relays the API you pick and keeps request
bodies, tools, cache identity, response usage, SSE and errors as they are. The
separate engine owns inference and KV caching.

For a current FP4-Local server, follow [INSTALL.md](../../INSTALL.md). On the
Spark itself, `vade launch` or `vade up` starts an engine that keeps running,
and opens no tunnel. On a controller, `vade up` starts the engine, opens a
detached SSH tunnel to the configured local port (default `18080`), and returns.
The engine, and on a controller its tunnel, stay up until `vade down`, so direct
and gateway clients can connect at any time in between. Read the exact model ID
and limits from the capabilities, as shown below. If the harness is not on your
PATH, use `harness/target/release/vade-harness` from the repo root instead of
`vade-harness`.

## Create a dedicated home

Create an empty scratch parent, then pick a **new** child directory inside it.
Both `--context-window` and `--max-output-tokens` must be values you have
verified on the upstream; setup does not discover or make up limits. Read them
from the running engine's `/v1/capabilities`, together with its exact model ID
(the example uses `curl` and `jq`). FP4-Local starts with context 32768 and
output `min(context, 8192)`, whether you start it from a controller (on any
number of Sparks) or with `vade launch` on the Spark. `vade launch` lowers a
default context only when the engine refuses it for memory, and
`--context 262144` picks the single-chat long-context preset on either route.
Do not assume 32768: read the running server's capabilities again after every
launch change.

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

From a separate terminal, change to your workspace and run the generated
`launch.sh` by its absolute path. It accepts the client's normal arguments. Use
`--target direct` to point at the upstream directly, or `--gateway URL` to pick
a gateway base you configured separately. `--client-bin PATH` picks a pinned
executable; otherwise the client is found on PATH. A single-file `--write PATH`
without `--dedicated-home` keeps the existing behavior of writing just a
fragment.

The parent folder must already exist, contain no symlinks, and be given by its
real path (`pwd -P` prints it). Setup will not write over anything that
exists, through a final symlink, outside the parent (`..`), or inside
`.codex`, `.hermes`, `.pi` or `.config`. Files are created mode 0600; folders
and the launcher, 0700. Setup never reads or merges your real client settings.
If a write fails, it tries to remove only what it just created; check the
half-written folder before you retry at a new path.

The launcher starts the client in a clean environment (`env -i`). It keeps
PATH and your terminal settings and gives the client its own HOME, TMPDIR, XDG
folders and client home. Your credentials, provider overrides and shell
config are not passed in. Where the client allows it, startup update checks
are turned off and the local provider is selected. Each client keeps its own
safety rules: Codex runs in its read-only sandbox, OpenCode allows read and
search tools, Pi selects read and search tools, and Hermes offers file tools
but can write only inside the launch workspace (`HERMES_WRITE_SAFE_ROOT`).
This does not give every client Vade's strict native read boundary or an OS
sandbox.

Each home contains this single instruction line:

> When a task needs a tool, call the tool immediately; do not end the turn with a promise such as 'Let me check' before calling it.

This is a rule at the prompt level, so it cannot guarantee how the model
behaves. Project instructions and explicit user requests can still shape the
final prompt.

| Client version checked locally | Files written | Environment and normal launch arguments | API |
| --- | --- | --- | --- |
| Claude Code 2.1.310 | `settings.json`, `.claude.json`, `CLAUDE.md`, `launch.sh` | `CLAUDE_CONFIG_DIR` plus the gateway variables below; one-shot: `-p PROMPT --output-format json`; interactive: no extra arguments | Messages at the API root; no Anthropic account or key |
| Codex 0.160.1 | `config.toml`, first-party `models.json`, `AGENTS.md`, `launch.sh` | `CODEX_HOME`; interactive: no extra arguments; one-shot: `exec --skip-git-repo-check --json PROMPT` | Responses HTTP/SSE; no OpenAI authentication, WebSockets or shared daemon |
| Hermes 781334eea4b | `config.yaml`, `SOUL.md`, `launch.sh` | `HERMES_HOME`, `HERMES_WRITE_SAFE_ROOT`; one-shot: `-z PROMPT`; interactive: `chat` | Chat Completions with API streaming enabled |
| OpenCode 1.18.18 | `opencode.json`, `AGENTS.md`, `launch.sh` | `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, private XDG state; one-shot: `run --format json PROMPT`; interactive: no extra arguments | `@ai-sdk/openai-compatible`, Chat Completions |
| Pi 0.73.1 | `models.json`, `settings.json`, `AGENTS.md`, `launch.sh` | `PI_CODING_AGENT_DIR`; one-shot: `--mode json --print PROMPT`; interactive: no extra arguments | `openai-completions`; system role, `max_tokens`, streaming usage |

The pinned Hermes custom provider rejects contexts below 64,000 tokens before it
sends a request. For a smaller context that you supply explicitly, dedicated
setup uses Hermes' built-in LM Studio compatibility profile, and keeps the exact
base URL and model. Hermes' initialization code allows this explicit smaller
window. This does not say the upstream is an LM Studio server; it only picks a
client compatibility profile. Ordinary fragments keep `provider: custom`.

`--max-output-tokens` sets the Pi and OpenCode model limits and the local
context reservation. Codex and Hermes own their wire-level generation limits, so
this flag cannot guarantee that their requests carry a matching output cap.
Codex gets an original short coding prompt and a local model catalog instead of
unknown-model defaults. The upstream must actually implement Responses for
Codex.

## Check a client yourself

To check a client yourself, run every cell twice: client → upstream,
and client → gateway → the same upstream. Use a separate new home for each route
and session, and an identical scratch workspace. No cell needs personal
instructions, credentials, remote tools or project data.

| Check | Fixed task / method | Evidence to keep |
| --- | --- | --- |
| Configuration | Generate the home and run the pinned client without additional provider overrides | Exact pin, generated files, flags, API route and selected model; verify the tool rule appears in the sent instructions |
| Chat | `Reply exactly OK.` | Successful terminal answer, finish reason, streamed request flag and usage |
| Tool use | Put `TOKEN_R1_314` in `marker.txt`; ask `Read marker.txt using a tool and report its exact contents.` | A real read-tool invocation, successful tool result, next model request and final marker; prose promising a tool is a failure |
| Follow-up/cache | Continue the same session with `Repeat the exact marker from the previous answer without reading it again.` | Per-request prompt/output/cached counts, stable prior tool result, and `100 * cached / prompt`; missing usage is UNKNOWN |
| Streaming | Delay the mock's stream completion after its first answer/tool event | First meaningful answer/tool event reaches the client before terminal completion; role-only events are not TTFT |
| Timing | One discarded warm-up per profile; fresh upstream sessions in direct/gateway/gateway/direct order | Median request TTFT, output-token count and decode-only `(output_tokens - 1) / (last - first)`; keep startup-to-visible-answer separate |
| Errors | Mock returns HTTP 400 with a synthetic structured error; also check truncation/disconnect | Visible client failure, preserved upstream status/body, bounded retries and no false successful answer |

The cache percentage comes from each completed upstream response, not from
guessing at prompt similarity or from gateway affinity. Some clients subtract
cached input when they display billable input; add the cached input back to get
the total prompt occupancy. The gateway leaves cache fields alone unless an
explicit optimization flag asks for cache decoration. Requests without any
sampling field get the shared greedy defaults, and any named sampling field,
even null, turns all defaults off. Record the actual request settings: sampling
the client chose is not the same as a request that leaves sampling controls out.

The pinned Pi JSON mode can exit zero while reporting `stopReason: error`, so
automation must check the terminal result as well as the process exit code.

The first local pass completed streamed chat and the read-tool round trip in all
eight client/route combinations against a first-party Rust mock. Mock usage is
synthetic, so its cache counts and token rate only show that the transport is
compatible. Live interoperability, real prefix reuse and engine throughput need
separate measurements.

The standalone gateway waits for new connections in a blocking accept, which
avoids a 25 ms polling delay. The cancellable API for embedding the gateway in
other programs still polls for connections, with a bound.

The live OpenCode and Pi results date from **2026-10-20 and used no tools**:
four direct and gateway smoke tests, each a one-line answer capped at 128
tokens, on the older commit `26b1692`. Hermes could not run because of missing
dependencies. These results show neither tool use nor a live Claude Code or
dsh (DeepSeek's coding harness) run; see the [gateway proof](gateway-measurements.md).

## Offline reference-client proof (2026-10-23)

`tests/passthrough_clients.rs` starts the real `vade-gateway` and
`vade-harness gateway` executables. An independent HTTP decoder sends 27
synthetic cases through the raw upstream and through both processes: **81
dispatches, 54 paired relay comparisons**. The three dialects are Chat
Completions, Responses and Messages, and each one runs the same nine cases.
The tool call and the tool result are two separate cases that share a row
below:

| Check | Required result |
| --- | --- |
| JSON answer | Body and selected end-to-end headers stay exact |
| Fragmented SSE | First bytes arrive before the upstream can finish; a split UTF-8 character remains exact |
| Tool call and tool result | Parse the native schema, read the fixture file, and forward the actual result in the continuation |
| Usage | Preserve the upstream's reported counts without inventing a cache hit |
| HTTP failure | Keep status 429, Retry-After and opaque error body |
| Explicit null | Preserve every request byte, including null and lexical number/string forms |
| No sampling fields | Append only the established greedy defaults |
| Truncated SSE | Preserve delivered bytes and incomplete framing; do not manufacture successful termination |

The tool checks do **nine real file reads**, one for each API on each route
(raw upstream, `vade-gateway` and `vade-harness gateway`). The test server
checks every request's path, body and chosen headers, and that each case was
sent exactly once. These are our own reference clients with made-up
responses, so they say nothing about installed vendor clients or engine speed.
A separate test covers the completions endpoint: both aliases, explicit and
default sampling, and token-ID payloads in JSON and SSE. The earlier
27-of-27 live answer comparison still stands for what it covered; this test
does not replace it.

### Installed client headers (2026-10-30)

On 2026-10-30, Claude Code 2.1.314 (Messages), Codex 0.165.0 (Responses) and
the openai Python client 3.41.0 (Chat Completions) sent their requests to a
loopback capture server. The test replays each client's request headers through
the raw upstream and through both gateways. These headers include the
`X-Stainless-*` set, `anthropic-beta`, a JSON-valued `x-codex-turn-metadata`
and `Accept-Encoding`. The IDs in the test are fixed values with the same shape.
The upstream gets every end-to-end header in the same order, and each value is
unchanged byte for byte. The core gateway sends header names in lower case;
HTTP header names are not case-sensitive. The only other headers are `Host` and
the relay's own framing. The request body, the status, five response headers
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

- `ANTHROPIC_BASE_URL` is the API root: setup drops the `/v1` that the other
  clients use, and keeps any proxy prefix in front of it.
- `ANTHROPIC_AUTH_TOKEN=no-key-required` is a dummy bearer value, so Claude
  Code never asks for an Anthropic login. The launcher's private
  `CLAUDE_CONFIG_DIR` and `HOME` keep your real `~/.claude` untouched.
- `ANTHROPIC_MODEL`, the `opus`/`sonnet`/`haiku` aliases and
  `CLAUDE_CODE_SUBAGENT_MODEL` all name the one loaded model, so background
  tasks and subagents don't ask for a Claude model the engine doesn't have.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` and `CLAUDE_CODE_MAX_OUTPUT_TOKENS` carry the
  verified limits. Without the output cap, Claude Code assumes 32000 output
  tokens for a model it doesn't know and compacts a 32768-token window early.
- `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`, `ENABLE_TOOL_SEARCH=false` and
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` keep pre-release request fields,
  MCP tool search, telemetry and update checks out of the session.
- The home's `settings.json` allows Read, Grep and Glob and denies shell, edit,
  write and web tools, like the other homes' read-only defaults. Edit it to
  give Claude Code more.
- Claude Code 2.1.310 sends its environment notes as a `role: "system"` entry
  inside `messages`. The engine accepts only user and assistant turns there
  and answers 400, and Claude Code does not retry. Both gateways relabel just
  those role values as `"user"` on Messages and token-count requests, which is
  the recovery Claude Code documents, so use `--target gateway`. Direct to
  the engine, only `claude --bare` works until the engine accepts them.
- Images are not supported yet.
- Sampling works as for the other APIs: with no sampling fields the gateway
  adds greedy defaults, and any field the client sends stays as is.

## aider, Continue and Open WebUI

`vade-harness setup aider|continue|open-webui` prints each client's documented
OpenAI-compatible settings for the engine or the gateway. They have no
dedicated home: aider and Open WebUI read environment variables, and Continue
reads the `config.yaml` you paste into its settings.

- **aider** gets `OPENAI_API_BASE`, a dummy key and `--model openai/MODEL`.
  aider warns that it does not know the model's limits; that warning is
  expected.
- **Continue** gets one model for chat, edit and apply. Pass
  `--context-window` and `--max-output-tokens` to add
  `defaultCompletionOptions`. Tab autocomplete is left out on purpose: the
  engine ignores the fill-in-the-middle `suffix` field, so autocomplete would
  only continue the text before the cursor.
- **Open WebUI** gets `OPENAI_API_BASE_URL` and lists models from
  `/v1/models`, which both gateways serve. In Docker, use
  `http://host.docker.internal:8787/v1`, run the container with
  `--add-host=host.docker.internal:host-gateway`, and start the gateway with
  `--allow-host host.docker.internal`; otherwise the gateway's Host guard
  answers 421.

None of these three has been run live yet.

## dsh recipe (not live-qualified)

dsh is DeepSeek's coding harness. With `dsh` already installed, launch
`dsh web` in the workspace you want. In **Settings → Models → Add model
provider → Custom model API**, set provider ID `vade`, base URL `http://127.0.0.1:8787/v1`, and API protocol **OpenAI Chat
Completions**. Enter the literal dummy `no-key-required` for a local engine
without a key. Add a model by hand: ID = `MODEL`, context window =
`CONTEXT_WINDOW`, max output tokens = `MAX_OUTPUT_TOKENS`, and input types =
**Text** only. Use the values of those variables, then save and select that
model for a new session. The [provider
guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md)
describes these fields; discovery alone does not supply Vade's missing model
limits.

Routes you declare by hand need a nonempty `models` list. The [pi-ai
adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-pi-ai/README.md)
may move later system text into the initial prompt, so verify prefixes before
claiming cache reuse or client exactness. No dsh tool round trip is proven here.

## Sources and credit

These are original Rust templates and a short original tool rule. No
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
