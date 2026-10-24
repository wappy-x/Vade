# Client setup and endpoint doctor

Rounds 5–6 used local Rust mocks without installed clients or a live provider.
The passthrough round adds dedicated homes and checks the pinned clients against
a first-party Rust mock on macOS. Product code remains Rust, standard library
only. See the [direct/gateway client matrix](passthrough-matrix.md) for the new
home mode, exact files, launch arguments and interoperability limits.

For INSTALL's current FP4-Local setup, run `vade up` as described in
[INSTALL](../../INSTALL.md) and keep its Mac tunnel open. Replace `MODEL` below
with `qwen3.8-flash-next-fp4-local`, the model ID in
[`cli/examples/fp4-local.conf`](../../cli/examples/fp4-local.conf), or the exact
model ID advertised by your configured endpoint. The older
[manual FP8 tunnel instructions](../../engine/docs/flash-server.md#connect-an-existing-agent)
and `qwen3.8-flash-next-fp8` model ID apply only to legacy FP8 deployments. If the harness is not on PATH, use the built
`harness/target/release/vade-harness` from the monorepo root.

## Setup: offline configuration

```sh
vade-harness setup codex --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup hermes --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup opencode --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup pi --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup generic --endpoint http://127.0.0.1:18080 --model MODEL
```

Choose the Engine URL and exact model ID explicitly. By default stdout contains
two separately labeled, complete configurations: client → Engine, and client →
gateway → Engine. The gateway base defaults to `http://127.0.0.1:8787/v1`;
`--gateway URL` changes the printed client endpoint. Start/configure the gateway
separately. It must forward to the chosen Engine; setup does not launch anything.

HTTP origins, `/v1` bases, and known chat/responses/capabilities suffixes normalize
to a client API base. Reverse-proxy prefixes such as `/proxy/v1` are retained.
Unknown exact routes, URL credentials, queries, fragments, encoded/interpolated
paths, dot segments, and HTTPS are refused rather than guessed. Model IDs must be
literal: letters, digits, `._:/+-`, starting with a letter or digit, up to 512 bytes.
This prevents the clients' environment/file interpolation and command expansion
from turning a generated setting into a different value or an executable command.

Select `--target direct` or `--target gateway` to print **only the client-ready
configuration** on stdout; usage instructions and the source citation go to stderr.
With `--write PATH`, that selected configuration goes into a new file at precisely
the named path. Existing files and final symlinks are refused; parents are not
created. Files use mode 0600 on macOS. No client configuration is searched, read,
merged, or overwritten. No credentials or user configuration are inspected.

```sh
vade-harness setup opencode --endpoint http://127.0.0.1:18080 --model MODEL --target gateway --write ./vade-opencode.json
```

`--write` requires one target so that the file is valid client configuration,
not two concatenated TOML/YAML/JSON documents. Setup prints placement/selection
instructions; the user decides how to apply the fragment to their installed client.
Generic output is a POSIX-shell `curl -q` request, not a universal SDK config.
Generating it sends nothing; executing it sends a synthetic chat request.

To write complete isolated state instead of a fragment, use `--dedicated-home`
with `--write DIR`, a single target, and verified `--context-window N` and
`--max-output-tokens N`. This creates a new private home with native instructions
and a clean-environment launcher. Existing destinations are refused; this does
not merge into installed client state. `--client-bin PATH` selects a pinned
client executable. The [home-mode table](passthrough-matrix.md) lists every file
and the pinned Hermes small-context compatibility profile.

| Client | Documented settings used | API and use |
| --- | --- | --- |
| Codex | `model`, `model_provider`, `model_providers.vade` with `base_url`, `wire_api`, `requires_openai_auth`, `supports_websockets` | TOML provider fragment; Responses over HTTP/SSE; no OpenAI auth |
| Hermes | `model.default`, `model.provider: custom`, `model.base_url`, `model.api_key` | YAML model fragment; Chat Completions |
| OpenCode | `model: vade/MODEL`, `provider.vade.npm`, `options.baseURL`, `options.apiKey`, `models` | JSON config; `@ai-sdk/openai-compatible` selects Chat Completions |
| Pi | `providers.vade.baseUrl`, `api: openai-completions`, `apiKey`, `models[].id` | `models.json` fragment; select with `pi --provider vade --model MODEL` |
| Generic | HTTP JSON `model`, `messages`, `max_tokens`, `stream` | One curl Chat Completions request; no Harness runtime |

`no-key-required` is a literal dummy for keyless endpoints, not a credential.
Setup never reads environment keys. Existing client settings/authentication can
affect an installed client's behavior; generated snippets do not audit them.
No context limit or reasoning/tool capability is invented in client configuration.
The client retains its own defaults unless the user configures verified limits.
The upstream must implement the client's API; the gateway neither converts Chat
Completions to Responses nor adds provider features. HTTP only is intentional.

## Doctor: active, bounded checks

```sh
vade-harness doctor http://127.0.0.1:18080
vade-harness doctor http://127.0.0.1:8787 --max-seconds 30
vade-harness doctor http://127.0.0.1:18080 --model MODEL
```

Doctor first fetches `/v1/capabilities` on the explicitly selected origin/base.
The default whole-probe deadline is 60 seconds: an approximately 11,000-token
cold Flash prompt can take 30 seconds before the cached second turn, so the
budget also leaves room for metadata, exact admission, and generation. Set
`--max-seconds` explicitly for slower endpoints (1–300 seconds).
It fetches `/_vade/gateway/stats` only when that response identifies a Vade gateway
with `X-Vade-Gateway: v1`. The gateway adds this response header to forwarded and
local/error responses without changing their bodies. Direct endpoints receive no
gateway-route probe, including engines that return HTTP 400 for unknown routes.
Older gateways without this marker get an explicit “not probed” stats line; their
chat/cache checks still work. It does not discover other hosts, follow
redirects, launch services, or open a workspace. For an authenticated endpoint,
pass `--api-key-file PATH` or set `VADE_API_KEY_FILE`; the explicit flag takes
precedence. The shared native transport reads that file and sends
`Authorization: Bearer <key>` on metadata, admission and generation requests,
including doctor probes. It does not discover keys from other clients or print
the credential. A recognized
`vade.engine.v1` contract supplies model ID, opaque weight/tokenizer/template
fingerprints, context/input/output limits, tool/thinking flags, API routes,
prefix-cache advertisement, and exact-admission support. Flags are labeled
**advertised**; doctor does not force a tool call or thinking output to test them.
Missing/null flags are unknown. Responses support is advertised, not probed.

Absent or foreign capabilities leave identity/limits/features unknown. Supply
`--model NAME` to try standard Chat Completions in that case. A selected model
must match recognized metadata. Malformed recognized metadata, model/dialect
mismatch, discovery HTTP failure, or exact-admission refusal prevent generation;
there is no silent fallback after a recognized contract changes or disappears.

The cache probe is exactly two synthetic generations in one in-memory conversation.
The optional positive integer `cache_granularity_tokens` capability sizes its
unchanged prefix to at least `2 * granularity + 1` rendered prompt tokens. With
4096-token blocks, that is at least 8193 tokens. Absent/null granularity uses the
same 8193-token fallback. A fixed original synthetic word sequence provides the
padding; bytes or words are never presented as exact tokens.

When exact admission is advertised, up to four admission-only sizing requests
verify the target before inference. An undersized candidate grows from the
observed count; refusal, invalid counts, unfit advertised limits, or the padding
bound stop the check without generation. Each generation then refreshes metadata
and admits its full final body normally. Sizing performs no inference, KV warming
or resource reservation. Without exact admission, the fallback uses 8193 words
and explicitly reports its token size as unverified; a zero hit with a reported
first prompt below target is INCONCLUSIVE, rather than NO HIT.
Each generation requests at most 32 output tokens (or a lower advertised output limit),
with no tools. The second appends the validated full assistant message and a new
user message to unchanged history/options. The existing native client owns
prefix checks, negotiated scheduling hints, per-turn capability refresh, and
exact admission of the complete final request. Successful admission proves
context fit, not resource reservation. Output-token-limit finishes are accepted
for this capped probe and labeled separately from ordinary stop finishes.

Doctor prints both turns' actual prompt/completion/cached counts and turn-2
reuse as `100 * cached_tokens / prompt_tokens` percent. The denominator uses exact
admission when available, otherwise reported prompt usage; absent counts remain
unknown. This percentage measures cached input occupancy, not workload token LCP.
A **PASS** means
the second completed generation reported valid `cached_tokens > 0`. A reported
zero is **NO HIT**. Missing, conflicting, or impossible telemetry is **UNKNOWN**;
failed/truncated streams cannot establish a hit. Usage is cross-checked against
exact admitted counts and the output reservation. No aggregate cluster counter,
affinity key, inferred prefix length, or cache advertisement can substitute for
the second response's usage. The sizing target addresses cache block thresholds;
it does not ensure the upstream retains the prefix under cache pressure. A hit
does not prove causal reuse from turn 1, model correctness, or
that every client/cluster node will hit.

Cluster stats show bounded endpoint indices, compatibility groups, health,
cooldown, in-flight work, and cumulative counts **before** generation; no URLs or
credentials are printed. Two turns exercise the route selected by the gateway,
not every node. Heterogeneous groups can make the gateway's model-less capability
route ambiguous (HTTP 409); doctor reports that limitation rather than choosing
an arbitrary model or bypassing the gateway. Use a homogeneous gateway endpoint
for the full negotiated two-turn check. Unknown/missing stats do not imply a
single healthy node. Recognized degraded stats cause a failing exit status even
when the selected conversation reports a cache hit.

The whole check defaults to 60 seconds, configurable to 1–300 seconds. Metadata
is bounded to 64 KiB; at most 16 cluster nodes are rendered. The native diagnostic
is limited to two model turns total, no tool execution, 65,536 padding words,
512 KiB per message, 1 MiB history, 128 KiB response stream payload, and 2,048 SSE
events per turn. The local context estimate budget is 524,288 with a 32-token
reserve so it cannot reject a large exact-admitted diagnostic at the normal
131,072-token local default; negotiated generation admission still enforces the
upstream input/context/output limits. The shared
HTTP framing limits also apply. Cancellation polls socket I/O; standard-library
DNS and in-progress connection setup can outlast the overall budget. No request
is retried. Remote compute cancellation is never claimed.

Exit status is **0** only when the two-turn cache check passes without a detected
probe/telemetry failure or degraded cluster node; **1** means failed/incomplete
diagnostics (including unknown/zero cache usage); **2** means invalid arguments.
No files are written. Raw remote bodies, generated answers, and transport errors
are not printed.

## Sources and credit

These original templates follow the APIs/settings below, supplementing the
[design references](../../CREDITS.md#harness). No competitor code was
copied, bundled, or added as a dependency. Separately installed pinned clients
are used only in the documented passthrough mock/live checks.

- [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference), reviewed 2026-10-08: custom provider URL, Responses dialect, OpenAI-auth opt-out, HTTP/WebSocket selection.
- [Hermes provider configuration](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/website/docs/integrations/providers.md#custom--self-hosted-llm-providers): the `model` mapping and `custom` provider. The pinned docs explicitly retire `LLM_MODEL` and limit `OPENAI_BASE_URL`; setup uses YAML, not those legacy environment instructions.
- [OpenCode provider configuration](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/packages/web/src/content/docs/providers.mdx#custom-provider): provider/model maps and the Chat Completions adapter.
- [Pi compatible endpoints](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/coding-agent/docs/models.md#configure-a-compatible-endpoint) and [CLI model selection](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/coding-agent/docs/cli.md): `models.json`, dummy key, provider/model flags, interpolation constraints.
- [Chat Completions request reference](https://platform.openai.com/docs/api-reference/chat/create): generic HTTP request shape. Vade capability/admission semantics remain the separate [first-party contract](engine-contract.md).

Mock verification covers all five templates and both targets, offline/no-write
behavior, explicit new-file writes, invalid/interpolated inputs, capabilities and
model selection, successful/refused admission, stable second-turn history, cache
hit/zero/missing/invalid usage and percentages, malformed/truncated responses,
deadlines, direct engines rejecting gateway routes, exact adaptive sizing above
4096-token cache blocks, the >8192-token fallback, unfit/overflowing granularity,
and homogeneous/degraded/ambiguous gateway clusters.

## Disabled cache advertisement

When recognized capabilities explicitly contain `prefix_cache:false`, the cache
row reports measured reuse as `WARN engine-pending`; it cannot alone fail an
otherwise successful doctor. Missing or unknown cache advertisement retains the
ordinary requirement for a reported hit, and invalid telemetry still fails.
This distinguishes a server-declared disabled feature from harness connectivity,
admission and chat failures. It does not establish any cache performance gain.
