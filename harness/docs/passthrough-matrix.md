# Other clients through the gateway

Setup can write complete dedicated homes for Codex, Hermes, OpenCode and Pi.
Configuration generation is offline. The gateway relays the selected API and
preserves request bodies, tools, cache identity, response usage, SSE and errors.
The separate engine owns inference and KV caching.

For a current FP4-Local server, follow [INSTALL.md](../../INSTALL.md). On the
Spark itself, `vade launch` or `vade up` starts a resident engine and opens no
tunnel. On a controller, `vade up` starts the engine, opens a detached SSH
tunnel to the configured local port (default `18080`) and returns. The engine,
and on a controller its tunnel, stay up until `vade down`, so direct and gateway
clients can connect at any time in between. Read the exact model ID and limits from capabilities below. If the
harness is not on PATH, use `harness/target/release/vade-harness` from the repo
root in place of `vade-harness`.

## Create a dedicated home

Create an empty scratch parent, then choose a **new** child directory. Both
`--context-window` and `--max-output-tokens` must be verified upstream values;
setup does not discover or invent limits. Read them from the running engine's
`/v1/capabilities`, along with its exact model ID (the example uses `curl` and
`jq`). FP4-Local starts with context 32768 and output `min(context, 8192)`
from a controller on any Spark count and with `vade launch` on the Spark, which
lowers a default context only when the engine refuses it for memory;
`--context 262144` selects the single-chat long-context preset.
Read the running server's capabilities after every launch change. Do not
assume 32768.

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

Build the standalone binary as described in [Standalone gateway](invisible-gateway.md).
Use the same advertised limits for Codex, Hermes and OpenCode setup homes.

From a separate terminal, change to your workspace and invoke the generated
`launch.sh` using its absolute path. It accepts normal client arguments. Use
`--target direct` to select the upstream directly, or `--gateway URL` to choose
a separately configured gateway base. `--client-bin PATH` selects a pinned
executable; otherwise the client is found on PATH. Single-file `--write PATH`
without `--dedicated-home` keeps the existing fragment-only behavior.

The parent must exist, contain no symlinks and use its physical path (obtain it
with `pwd -P`). Existing destinations, final symlinks, parent traversal and paths
inside `.codex`, `.hermes`, `.pi` or `.config` are refused. Files are mode 0600;
directories and the launcher are mode 0700. No real client state is read or
merged. A failed write attempts to remove only entries created by this call;
an incomplete destination must be inspected before retrying at a fresh path.

The launcher starts with `env -i`, retains PATH and terminal presentation, and
sets a private HOME, TMPDIR, XDG directories and the client-specific home.
Credentials, provider environment overrides and user shell configuration are
not imported. It disables startup update checks where supported and selects
local provider configuration. Client tools retain their own enforcement:
Codex uses its read-only sandbox, OpenCode permits read/search tools, Pi selects
read/search tools, and Hermes exposes file tools with writes restricted by its
`HERMES_WRITE_SAFE_ROOT` to the physical launch workspace. This does not establish
Vade's native strict read boundary or an OS sandbox for every client.

Each home contains this single instruction line:

> When a task needs a tool, call the tool immediately; do not end the turn with a promise such as 'Let me check' before calling it.

This is a prompt-level rule. It cannot guarantee model behavior. Project
instructions and explicit user requests can still affect the resulting prompt.

| Client pin checked locally | Files written | Environment and normal launch arguments | API |
| --- | --- | --- | --- |
| Codex 0.160.1 | `config.toml`, first-party `models.json`, `AGENTS.md`, `launch.sh` | `CODEX_HOME`; interactive: no extra arguments; one-shot: `exec --skip-git-repo-check --json PROMPT` | Responses HTTP/SSE; no OpenAI authentication, WebSockets or shared daemon |
| Hermes 781334eea4b | `config.yaml`, `SOUL.md`, `launch.sh` | `HERMES_HOME`, `HERMES_WRITE_SAFE_ROOT`; one-shot: `-z PROMPT`; interactive: `chat` | Chat Completions with API streaming enabled |
| OpenCode 1.18.18 | `opencode.json`, `AGENTS.md`, `launch.sh` | `OPENCODE_CONFIG`, `OPENCODE_CONFIG_DIR`, private XDG state; one-shot: `run --format json PROMPT`; interactive: no extra arguments | `@ai-sdk/openai-compatible`, Chat Completions |
| Pi 0.73.1 | `models.json`, `settings.json`, `AGENTS.md`, `launch.sh` | `PI_CODING_AGENT_DIR`; one-shot: `--mode json --print PROMPT`; interactive: no extra arguments | `openai-completions`; system role, `max_tokens`, streaming usage |

The pinned Hermes custom provider rejects contexts below 64,000 tokens before
sending a request. Dedicated setup uses Hermes' built-in LM Studio compatibility
profile for a smaller explicitly supplied context, preserving the exact base URL
and model. Its initialization code permits this explicit smaller window. This
does not identify the upstream as an LM Studio server; it selects a client
compatibility profile. Ordinary fragments retain `provider: custom`.

`--max-output-tokens` sets Pi/OpenCode model limits and local context reservation.
Codex and Hermes own their wire-level generation limits; this flag cannot
guarantee that their requests carry a corresponding output cap. Codex receives
an original short coding prompt and local model catalog, rather than unknown-model
defaults. The upstream must actually implement Responses for Codex.

## Validation matrix

Run every cell twice: client → upstream and client → gateway → the same upstream.
Use a separate new home for each route/session and an identical scratch workspace.
No cell needs personal instructions, credentials, remote tools or project data.

| Check | Fixed task / method | Evidence to retain |
| --- | --- | --- |
| Configuration | Generate the home and run the pinned client without additional provider overrides | Exact pin, generated files, flags, API route and selected model; verify the tool rule appears in the sent instructions |
| Chat | `Reply exactly OK.` | Successful terminal answer, finish reason, streamed request flag and usage |
| Tool use | Put `TOKEN_R1_314` in `marker.txt`; ask `Read marker.txt using a tool and report its exact contents.` | A real read-tool invocation, successful tool result, next model request and final marker; prose promising a tool is a failure |
| Follow-up/cache | Continue the same session with `Repeat the exact marker from the previous answer without reading it again.` | Per-request prompt/output/cached counts, stable prior tool result, and `100 * cached / prompt`; missing usage is UNKNOWN |
| Streaming | Delay the mock's stream completion after its first answer/tool event | First meaningful answer/tool event reaches the client before terminal completion; role-only events are not TTFT |
| Timing | One discarded warm-up per profile; fresh upstream sessions in direct/gateway/gateway/direct order | Median request TTFT, output-token count and decode-only `(output_tokens - 1) / (last - first)`; keep startup-to-visible-answer separate |
| Errors | Mock returns HTTP 400 with a synthetic structured error; also check truncation/disconnect | Visible client failure, preserved upstream status/body, bounded retries and no false successful answer |

Cache percentage comes from each completed upstream response, not inferred prompt
similarity or gateway affinity. Some clients subtract cached input when displaying
billable input; add cached input back to obtain total prompt occupancy. The gateway
leaves cache fields unchanged unless an explicit optimization flag requests
cache decoration. Requests without any sampling field receive the shared greedy
defaults; any named sampling field, including null, suppresses all defaults.
Record actual request settings: client-selected sampling remains distinct from
a request that omits sampling controls.
The pinned Pi JSON mode can exit zero while reporting `stopReason: error`;
automation must inspect the terminal result as well as the process exit code.

The first local pass completed streamed chat and the read-tool round trip in all
eight client/route combinations against a first-party Rust mock. Mock usage is
synthetic, so its cache counts and token rate establish transport compatibility
only. Live interoperability, real prefix reuse and engine throughput require
separate measurements. The standalone gateway blocks in accept to avoid a 25 ms
polling penalty; the cancellable embedding API retains bounded accept polling.

The separate live OpenCode/Pi receipts are **10-20, no tools**: four direct/gateway
smokes, a one-line answer with a 128-token cap, on older main `26b1692`. Hermes
was blocked by missing dependencies. These receipts establish neither tool use
nor a live Claude Code or dsh run; see the [gateway proof](gateway-measurements.md).

## Offline reference-client proof (2026-10-23)

`tests/passthrough_clients.rs` starts the real `vade-gateway` and
`vade-harness gateway` executables. An independent HTTP decoder sends 27
synthetic cases through the raw upstream and both processes: **81 dispatches,
54 paired relay comparisons**. The three dialects are Chat Completions,
Responses and Messages, with nine checks each:

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

The tool checks complete **nine real fixture-file reads**, one per API and arm.
The server verifies every request path, body and selected headers, then checks
that each case was dispatched exactly once. These are first-party reference
clients with synthetic responses, not installed-vendor or engine-performance
proofs. The separate completions conformance fixture covers both aliases,
explicit/default sampling and JSON/SSE token-ID payloads. Historical 27/27 live
answer comparisons retain their original scope; this matrix does not replace them.

From `harness/`, run:

```sh
cargo test --offline --test passthrough_clients -- --nocapture
cargo test --offline --lib recorded_engine_completions
```

## Claude Code recipe (not live-qualified)

With the standalone gateway running, use the `MODEL` and `CONTEXT_WINDOW` read
above. For a keyless local engine, this literal placeholder is sufficient:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8787 \
ANTHROPIC_AUTH_TOKEN=no-key-required \
ANTHROPIC_MODEL="$MODEL" \
CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CONTEXT_WINDOW" \
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 ENABLE_TOOL_SEARCH=false \
claude
```

The [context override](https://code.claude.com/docs/en/model-config#correct-the-window-for-a-gateway-or-custom-model-id)
applies directly to an unrecognized ID without a `claude-` prefix or `[1m]`
suffix, such as Vade's Qwen ID. Messages uses the API root, without `/v1`.
The [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol)
documents these beta controls. Mid-conversation system roles can still fail
until the engine's self-heal wording is fixed. Images remain unsupported;
the gateway applies the same sampling-field presence rule to Messages as to
Chat Completions, Completions and Responses. Client-provided sampling fields
remain unchanged. This recipe alone does not prove which fields a client sends
or establish live client compatibility.

## dsh recipe (not live-qualified)

With an already installed `dsh`, launch `dsh web` in the intended workspace.
In **Settings → Models → Add model provider → Custom model API**, set provider
ID `vade`, base URL `http://127.0.0.1:8787/v1`, and API protocol **OpenAI Chat
Completions**. Enter the literal dummy `no-key-required` for a keyless local
engine. Add a model manually: ID = `MODEL`, context window = `CONTEXT_WINDOW`,
max output tokens = `MAX_OUTPUT_TOKENS`, and input types = **Text** only. Use
the variables' values, then save and select that model for a new session.
The [provider guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/providers.md)
describes these fields; discovery alone does not supply Vade's missing model limits.

Hand-declared routes require a nonempty `models` list. The
[pi-ai adapter](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/llm/llm-pi-ai/README.md)
may move later system text to the initial prompt; verify prefixes before claiming
cache reuse or client exactness. No dsh tool round trip is proven here.

## Sources and credit

These are original Rust templates and a short original tool rule. No competitor
implementation or prompt text is copied into the harness. The pinned clients'
local help and documentation establish their command/configuration contracts.

- [Codex configuration schema](https://learn.chatgpt.com/docs/config-file/config-reference): provider, Responses, local model catalog and instruction settings.
- [Hermes pinned context initialization](https://github.com/NousResearch/hermes-agent/blob/781334eea4b/agent/agent_init.py): the custom-provider floor and explicit LM Studio context exception; [file safety](https://github.com/NousResearch/hermes-agent/blob/781334eea4b/agent/file_safety.py) defines `HERMES_WRITE_SAFE_ROOT`.
- [OpenCode providers](https://opencode.ai/docs/providers/) and [permissions](https://opencode.ai/docs/permissions/): compatible providers, model limits, explicit instructions and tool permissions.
- [Pi model configuration](https://github.com/earendil-works/pi/blob/v0.73.1/packages/coding-agent/docs/models.md) and [settings](https://github.com/earendil-works/pi/blob/v0.73.1/packages/coding-agent/docs/settings.md): compatibility fields, explicit model limits and agent-directory settings.
