# Client setup and endpoint doctor

Two commands help you connect another client to the engine:

- **Setup** writes the client's configuration for you. It works offline and
  launches nothing.
- **Doctor** checks that an endpoint really works. It sends a short two-turn
  conversation and reports how much of the second turn's prompt the engine says
  it restored from its cache.

Both use only Rust's standard library. We tested them against our own Rust
mocks, without installed clients or a live provider. Later we added dedicated
client homes and checked the pinned clients against the same kind of mock on
macOS. The [direct/gateway client matrix](passthrough-matrix.md) lists the home
mode's files, launch arguments and interoperability limits.

For the current FP4-Local setup, run `vade up` as described in
[INSTALL](../../INSTALL.md). If you drive the Spark from a controller (another
computer, INSTALL's Route B), keep the tunnel `vade up` opens; on the Spark
itself there is no tunnel. Replace `MODEL` below with
`qwen3.8-flash-next-fp4-local`, the model ID in
[`cli/examples/fp4-local.conf`](../../cli/examples/fp4-local.conf), or the exact
model ID your configured endpoint advertises. The older [manual FP8 tunnel
instructions](../../engine/docs/flash-server.md#connect-an-existing-agent) and
the `qwen3.8-flash-next-fp8` model ID apply only to FP8 deployments. If
the harness is not on your PATH, use the built
`harness/target/release/vade-harness` from the monorepo root.

## Setup: offline configuration

```sh
vade-harness setup codex --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup hermes --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup opencode --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup pi --endpoint http://127.0.0.1:18080 --model MODEL
vade-harness setup generic --endpoint http://127.0.0.1:18080 --model MODEL
```

Pick the engine URL and the exact model ID explicitly. By default, stdout holds
two complete configurations, each clearly labelled: client → Engine, and client
→ gateway → Engine. The gateway base defaults to `http://127.0.0.1:8787/v1`, and
`--gateway URL` changes the client endpoint that gets printed. Start and
configure the gateway yourself; it must forward to the engine you chose. Setup
does not launch anything.

You can give the engine URL in several forms, such as `http://host:port`,
`.../v1`, or a full `.../v1/chat/completions` (known chat, responses and
capabilities routes all work), and setup cuts it down to the base URL the client
needs. A reverse-proxy prefix such as `/proxy/v1` is kept. Setup refuses
anything it would have to guess at: unknown paths, credentials in the URL,
`?query` or `#fragment` parts, encoded or `${VAR}` paths, `.`/`..` segments, and
HTTPS. Model IDs must be literal: letters, digits and `._:/+-`, starting with a
letter or digit, at most 512 bytes. That way a client's own variable or file
expansion can never turn the setting into a different value or a command.

Pick `--target direct` or `--target gateway` to print **only the client-ready
configuration** on stdout; usage instructions and the source citation go to
stderr. With `--write PATH`, that configuration goes into a new file at exactly
the path you name. Existing files and final symlinks are refused, and parent
directories are not created. Files use mode 0600 on macOS. No client
configuration is searched for, read, merged or overwritten, and no credentials
or user configuration are looked at.

```sh
vade-harness setup opencode --endpoint http://127.0.0.1:18080 --model MODEL --target gateway --write ./vade-opencode.json
```

`--write` needs a single target, so that the file is valid client configuration
and not two TOML/YAML/JSON documents glued together. Setup prints instructions
on where to put the fragment and how to select it, and you decide how to apply
it to your installed client. Generic output is a POSIX-shell `curl -q` request,
not a universal SDK config. Generating it sends nothing; running it sends a
synthetic chat request.

To write a complete, isolated state instead of a fragment, use
`--dedicated-home` with `--write DIR`, a single target, and verified
`--context-window N` and `--max-output-tokens N`. This creates a new private
home with native instructions and a launcher that starts from a clean
environment. Existing destinations are refused, and nothing is merged into
installed client state. `--client-bin PATH` picks a pinned client executable.
The [home-mode table](passthrough-matrix.md) lists every file, and the pinned
Hermes small-context compatibility profile.

| Client | Documented settings used | API and use |
| --- | --- | --- |
| Codex | `model`, `model_provider`, `model_providers.vade` with `base_url`, `wire_api`, `requires_openai_auth`, `supports_websockets` | TOML provider fragment; Responses over HTTP/SSE; no OpenAI auth |
| Hermes | `model.default`, `model.provider: custom`, `model.base_url`, `model.api_key` | YAML model fragment; Chat Completions |
| OpenCode | `model: vade/MODEL`, `provider.vade.npm`, `options.baseURL`, `options.apiKey`, `models` | JSON config; `@ai-sdk/openai-compatible` selects Chat Completions |
| Pi | `providers.vade.baseUrl`, `api: openai-completions`, `apiKey`, `models[].id` | `models.json` fragment; select with `pi --provider vade --model MODEL` |
| Generic | HTTP JSON `model`, `messages`, `max_tokens`, `stream` | One curl Chat Completions request; no Harness runtime |

`no-key-required` is a literal placeholder for endpoints without a key, not a
credential. Setup never reads keys from the environment. Existing client
settings and authentication can change how an installed client behaves, and the
generated snippets do not audit them. No context limit or reasoning/tool
capability is made up in the client configuration, and the client keeps its own
defaults unless you configure verified limits. The upstream must implement the
client's API; the gateway does not convert Chat Completions to Responses, and
does not add provider features. Setup supports plain HTTP only, on purpose.

## Doctor: active, bounded checks

```sh
vade-harness doctor http://127.0.0.1:18080
vade-harness doctor http://127.0.0.1:8787 --max-seconds 30
vade-harness doctor http://127.0.0.1:18080 --model MODEL
```

### What doctor reads first

Doctor first fetches `/v1/capabilities` from the origin or base you selected.
The default deadline for the whole probe is 60 seconds. A cold Flash prompt of
roughly 11,000 tokens can take 30 seconds before the cached second turn, so the
budget also leaves room for metadata, exact admission and generation. Set
`--max-seconds` explicitly for slower endpoints (1–300 seconds).

It fetches `/_vade/gateway/stats` only when that response identifies a Vade
gateway with `X-Vade-Gateway: v1`. The gateway adds this response header to
forwarded, local and error responses without changing their bodies. Direct
endpoints never get a gateway-route probe, including engines that return HTTP
400 for unknown routes. Older gateways without this marker get an explicit “not
probed” stats line, and their chat and cache checks still work. Doctor does not
discover other hosts, follow redirects, launch services or open a workspace.

For an authenticated endpoint, pass `--api-key-file PATH` or set
`VADE_API_KEY_FILE`; the explicit flag wins. The shared native transport reads
that file and sends `Authorization: Bearer <key>` on metadata, admission and
generation requests, doctor probes included. It does not discover keys from
other clients and never prints the credential.

A recognized `vade.engine.v1` contract supplies the model ID, opaque
weight/tokenizer/template fingerprints, context/input/output limits, tool and
thinking flags, API routes, whether the prefix cache is advertised, and
exact-admission support. Flags are labelled **advertised**; doctor does not
force a tool call or thinking output to test them. Missing or null flags are
unknown. Responses support is advertised, not probed.

If capabilities are absent or foreign, identity, limits and features stay
unknown. In that case, pass `--model NAME` to try standard Chat Completions. A
model you pick must match recognized metadata. Malformed recognized metadata, a
model or dialect mismatch, a discovery HTTP failure, or an exact-admission
refusal all prevent generation; there is no silent fallback after a recognized
contract changes or disappears.

### The two-turn cache check

To test the cache, doctor holds a short two-turn conversation in memory. The
first turn sends a long block of made-up filler text; the second turn repeats
it and adds one more message. If the engine's cache works, most of the second
prompt should come back from cache. The filler must cover at least two of the
engine's cache blocks, so doctor makes it at least
`2 × cache_granularity_tokens + 1` tokens: 8,193 tokens for 4,096-token blocks,
and 8,193 when the engine advertises no block size.

If the engine can count tokens exactly, doctor first checks the filler size
with up to four counting-only requests and grows it if it falls short. These
requests generate nothing, warm no cache and reserve nothing. If the size
cannot be confirmed (a refusal, bad counts, limits too small, or filler past its
size cap), doctor stops before generating. Without exact counting, doctor uses
8,193 words and says the token size is unverified. Then a zero-hit result with
a first prompt below the target is reported as INCONCLUSIVE rather than NO
HIT.

Each generation asks for at most 32 output tokens (or a lower advertised output
limit), with no tools. The second one appends the validated full assistant
message and a new user message to the unchanged history and options. The
existing native client owns the prefix checks, negotiated scheduling hints,
per-turn capability refresh, and exact admission of the complete final request.
A successful admission proves the request fits the context, not that resources
are reserved. Finishing at the output-token limit is accepted for this capped
probe, and labelled separately from ordinary stop finishes.

Doctor prints the actual prompt, completion and cached counts for both turns,
and turn-2 reuse as `100 * cached_tokens / prompt_tokens` percent. The
denominator uses exact admission when available, otherwise the reported prompt
usage, and absent counts stay unknown. This percentage measures how much of the
input was cached, not the workload's token LCP (longest common prefix).

### What PASS, NO HIT and UNKNOWN mean

A **PASS** means the second completed generation reported a valid
`cached_tokens > 0`. A reported zero is **NO HIT**. Missing, conflicting or
impossible telemetry is **UNKNOWN**, and failed or truncated streams cannot show
a hit. Usage is cross-checked against the exact admitted counts and the output
reservation. No aggregate cluster counter, affinity key, inferred prefix length
or cache advertisement can stand in for the second response's usage. The sizing
target deals with cache block thresholds; it does not make sure the upstream
keeps the prefix under cache pressure. A hit does not prove causal reuse from
turn 1, model correctness, or that every client and cluster node will hit.

### Through a gateway cluster

Cluster stats show bounded endpoint indices, compatibility groups, health,
cooldown, in-flight work and cumulative counts **before** generation, and no
URLs or credentials are printed. The two turns exercise the route the gateway
selects, not every node. Mixed groups can make the gateway's model-less
capability route ambiguous (HTTP 409); doctor reports that limitation instead of
picking an arbitrary model or going around the gateway. Use a gateway endpoint
whose upstreams all match for the full negotiated two-turn check. Unknown or
missing stats do not mean there is a single healthy node. Recognized degraded
stats give a failing exit status even when the selected conversation reports a
cache hit.

### Limits and exit status

Doctor keeps itself small. It sends two model turns in total and runs no
tools. Metadata may be up to 64 KiB, filler up to 65,536 words, each message up
to 512 KiB, history up to 1 MiB, and each reply stream up to 128 KiB and 2,048
events. At most 16 cluster nodes are shown. Doctor's own local context estimate
allows 524,288 tokens with a 32-token reserve, so the usual 131,072-token local
default does not reject a probe the engine has already accepted; the engine's
own input, context and output limits still apply, and so do the shared HTTP
framing limits. Nothing is retried. Cancelling stops waiting on the socket, but
a DNS lookup or connection already under way can run past the deadline, and
doctor never claims that work on the engine was cancelled.

The exit status is **0** only when the two-turn cache check passes with no
detected probe or telemetry failure and no degraded cluster node. **1** means
the diagnostics failed or are incomplete (unknown or zero cache usage included),
and **2** means invalid arguments. No files are written. Raw remote bodies,
generated answers and transport errors are not printed.

## When the server says its cache is off

When recognized capabilities explicitly contain `prefix_cache:false`, the cache
row reports measured reuse as `WARN engine-pending`, and that alone cannot fail
a doctor run that otherwise succeeds. A missing or unknown cache advertisement
keeps the ordinary requirement for a reported hit, and invalid telemetry still
fails. This separates a feature the server says is disabled from connectivity,
admission and chat failures in the harness. It does not show any cache
performance gain.

## Sources and credit

These original templates follow the APIs and settings below, alongside the
[design references](../../CREDITS.md#harness). No code from these or other
competing clients was copied, bundled, or added as a dependency. Separately
installed pinned clients are used only in the documented passthrough mock and
live checks.

- [Codex configuration
  reference](https://learn.chatgpt.com/docs/config-file/config-reference),
  reviewed 2026-10-08: custom provider URL, Responses dialect, OpenAI-auth
  opt-out, HTTP/WebSocket selection.
- [Hermes provider
  configuration](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/website/docs/integrations/providers.md#custom--self-hosted-llm-providers):
  the `model` mapping and `custom` provider. The pinned docs explicitly retire
  `LLM_MODEL` and limit `OPENAI_BASE_URL`; setup uses YAML, not those legacy
  environment instructions.
- [OpenCode provider
  configuration](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/packages/web/src/content/docs/providers.mdx#custom-provider):
  provider/model maps and the Chat Completions adapter.
- [Pi compatible
  endpoints](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/coding-agent/docs/models.md#configure-a-compatible-endpoint)
  and [CLI model
  selection](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/packages/coding-agent/docs/cli.md):
  `models.json`, dummy key, provider/model flags, interpolation constraints.
- [Chat Completions request
  reference](https://platform.openai.com/docs/api-reference/chat/create):
  generic HTTP request shape. Vade capability/admission semantics remain the
  separate [first-party contract](engine-contract.md).

Mock verification covers all five templates and both targets, offline and
no-write behavior, explicit new-file writes, invalid and interpolated inputs,
capabilities and model selection, successful and refused admission, stable
second-turn history, cache hit/zero/missing/invalid usage and percentages,
malformed and truncated responses, deadlines, direct engines that reject gateway
routes, exact adaptive sizing above 4096-token cache blocks, the >8192-token
fallback, granularity that does not fit or overflows, and gateway clusters that
are homogeneous, degraded or ambiguous.
