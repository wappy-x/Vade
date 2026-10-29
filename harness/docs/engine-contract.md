# How the harness talks to the engine

This page describes how the harness talks to a Vade engine: how it learns what
the engine supports, how it checks that a request fits, how it reports cache
use, and what the gateway relays unchanged. The harness follows the
**vade.engine.v1** contract as the engine implemented it at commit **a211811**
(2026-10-08; an earlier report names fb77f4f). We read that source without
changing it, and our own Rust fixtures check the schema.

On a first read, two sections matter most:

- [Exact admission](#exact-admission): how the harness picks the output room
  for each reply.
- [Prompt-cache fields on the FP4-Local server](#prompt-cache-fields-on-the-fp4-local-server):
  what a request must send to get cache hits.

To check a running endpoint, use [setup and doctor](setup-doctor.md). The host
measurements we kept are in [performance](performance.md); they say nothing
about live cache performance. The [harness README](../README.md) describes the
direct chat, native agent and gateway modes this page mentions.

## How the harness learns what the engine supports

Creating a client does no network I/O. On the first model request, the
worker fetches GET /v1/capabilities before generation. Once negotiated, it
refreshes capabilities before each model request, including tool continuations.
Name lookup, connecting and reading all happen off the thread that draws the
terminal.

| Engine field | What the harness does with it |
| --- | --- |
| object, contract_version | engine.capabilities, vade.engine.v1 |
| model | Must match the selected model; no substitution |
| identity.weights.fingerprint | Opaque checkpoint identity |
| identity.weights.fingerprint_kind | Retained interpretation; currently safetensors_header_metadata, not a full weight-content hash |
| identity.tokenizer_fingerprint, identity.template_fingerprint | Opaque rendering identities |
| context_limit, input_limit, output_limit | Positive integer token limits |
| tools, thinking, default_thinking | Advertised features/default; null or absent support is unknown/unsupported |
| cache_domain, cache_scope, prefix_cache | Engine-owned reporting identity and feature; no proof of a hit |
| cache_granularity_tokens | Optional positive integer cache block size; absent/null unknown. The doctor check sizes its probe above two blocks and confirms it through admission. Advertising a block size does not prove the engine keeps the cache |
| prompt_cache_key | Only "scheduling_hint" enables generated affinity |
| admission, admission_kind | Only "/v1/admission" plus "context_fit" enables exact admission |
| kv_branch_handles, kv_checkpoints | Explicit opt-in to scoped input checkpoints, quotas, TTL and the fixed release route; see [lifecycle](engine-checkpoints.md) |
| endpoints | Must advertise /v1/chat/completions for native/direct mode |

How the harness reads the reply:

- The capabilities reply may be up to 64 KiB, and each kept string up to 512
  bytes. Unknown fields are ignored, and missing identity stays unknown.
- Fingerprints are compared as plain strings; the harness does not interpret
  them. The current engine's weight fingerprint reads only file headers, so it
  will not notice if weights change but the file size stays the same.
- The harness has no tokenizer, cryptography or KV of its own, and does not
  assume the engine supports continuation or remote cancel.

The optional [checkpoint client](engine-checkpoints.md) follows the later
Engine `kv-handles` source contract, using only opaque handles and local mocks.

When the configured chat URL ends in a known path, the client finds the
sibling routes on the same origin and base path. A custom exact path skips
this discovery. An advertised URL cannot send admission to a different origin,
and HTTPS still fails explicitly.

If the first reply is a 404, 405 or 501, comes from a different kind of
server, or names a contract version we do not know, the session falls back to
standard chat and sends no Vade-specific fields or admission calls. Native mode
keeps its standard chat tools; it never switches API or strips tools. If a
known contract has no tool support, a request that carries tools is refused.

Anything else stops the request: other discovery errors, garbled metadata from
a known contract, a model or API mismatch, or a contract that disappears after
it was negotiated. Nothing quietly downgrades after that; `/new` starts fresh.

## Exact admission

Before each negotiated generation, the harness sends POST /v1/admission with
a body like this:

    {"contract_version":"vade.engine.v1","endpoint":"/v1/chat/completions","output_tokens":32,"request":{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":"Hello"}]}}

In a real call, the `request` member embeds the complete final generation
body **byte for byte**: tools and options in order, all messages, results and
reasoning, provider extensions, and affinity. Counting never uses a parsed and
re-serialized copy. Once admission succeeds, generation receives those exact
embedded bytes. Tool results are counted again before their next model turn.

The output reservation is the request's `max_completion_tokens` or
`max_tokens`. The harness picks that number like this:

- **Explicit limits.** Caller options, including `--max-output-tokens`, remain
  fixed and must fit both the advertised output limit and remaining context.
- **No explicit limit.** Direct/native clients insert max_tokens and adapt it
  for each request to the smallest of 16,384 tokens, the advertised output
  limit and remaining context minus 256 safety tokens.
- **Context.** By default, context uses the advertised limit directly. An
  explicit local budget bounds it; without capabilities the fallback is
  131,072 tokens.
- **No advertised output limit.** The fallback output ceiling is 8,192
  tokens.

Exact admission supplies the prompt count. Then:

- If the default reservation needs adjusting, the client changes only the
  max_tokens number it inserted, then admits the final body again before
  generation. Historical messages, tool schemas and explicit options keep
  their exact bytes.
- Without exact admission, the same calculation uses a labeled local estimate.
- Generation is refused if no output room remains or an explicit reservation
  cannot fit. Native mode may use its existing in-task compaction (shortening
  the conversation history).
- Gateway requests keep the caller's bytes.

A reply is accepted only when it:

- identifies itself as engine.admission;
- has the same contract, model and reservation;
- has matching context, input and output limits;
- says admission_kind:"context_fit" and reserves_resources:false.

Counts and fit arithmetic are checked as integers, without subtracting cached
input. A refusal, a non-200 status, changed or malformed counts, truncation or
cancellation all stop generation, and nothing is retried automatically.
Refusals show counts and a local explanation; raw upstream reasons are
suppressed.

Admission does not reserve memory, a scheduler slot, or KV. It is not a
transactional snapshot spanning generation; the Engine can still reject or
change state between requests. Generation usage is shown separately.

A native generation response may stay silent while the request is queued or
the engine is reading the prompt (prefill), so not every connection gets the
ordinary 30-second idle timeout:

| Traffic | Timeout |
| --- | --- |
| Native generation while queued or reading the prompt: before headers, after headers or role-only SSE events, and JSON fallback, including the HTTP child adapter | The 3,600-second absolute HTTP lifetime, or earlier cancellation |
| Gateway generation POST responses | The full HTTP lifetime |
| Discovery, admission, request writes, direct chat and other gateway I/O | The ordinary idle timeout |

Native generation can still end sooner: the original request's active-work
budget, Esc, and child or workload deadlines all cancel it. Nothing is retried.

Until generated data actually arrives (or the JSON reply completes), native UI
and workload progress shows `queued / awaiting upstream`. The contract gives no
authoritative queue position. Cancellation is still polled every 250 ms.

Engine cancellation and generation deadlines end a request with the error
codes `client_cancelled` and `generation_deadline`. As JSON, they fail with
HTTP 409 and 504 respectively. A reply that is already streaming keeps its
deltas and ends with a structured error instead of `finish_reason: "length"`.

Native and direct chat treat these codes as an interruption or a timeout and
stop without automatic continuation. Text already on screen stays visible, but
a failed stream does not commit a partial assistant message or run any proposed
tools. Only a real token limit allows the normal length-driven continuation.
The deadline check also catches model work and final synchronization that
finish after the deadline, even when that work reaches the requested output
token limit.

Without exact admission, "wire bytes + 256" is explicitly labeled a rough
**estimate**, including serialized options/tools/history. It is not a tokenizer,
a guaranteed bound on hidden framing, or an admission decision. Unknown limits
remain unknown; existing independent byte/resource guards still apply.

## Assistant history

The contract review also read `crates/serve/src/protocol.rs` and `tokenizer.rs`
at Engine **d2a95d2**, without changing them, to see how assistant history must
look. At that commit, the parser requires string or text-part content on an
assistant message without tool calls.

So when such a reply arrives with content absent or null, the native adapter
fills in `content: ""`. This includes reasoning-only and length-limited
responses, over SSE or JSON. Qwen reasoning stays in `reasoning_content`, and an
accepted `reasoning` field is kept when provided. No thinking text is moved into
visible content or renamed, and history and options that are already frozen are
never rewritten on a later request. Tool-call messages keep their legal
absent/null content representation.

## Affinity, usage and UI

When the engine advertises the scheduling hint, each in-memory client session
gets one random, non-secret key from the OS, with no private path or model
data in it. A key the caller supplies is kept. The fixed fields the harness
adds (decoration) go before the history, and native requests keep their check
that each request only appends bytes to the previous one.

The key survives normal turns, tools, reconnects, Engine restarts, and
NativeModel::reset_prompt() compaction boundaries. A new client, /new, or
ChatClient::reset_context() gets a fresh key. Keys are never printed or stored
in the session counter. A change to the fingerprints, the domain or scope, or
the thinking default increments the observed cache-domain change counter.
Affinity never proves reuse.

A recognized cache scheduling hint also turns on `cache: true` when
`prefix_cache` is false. Neither the advertisement nor the request proves that
anything was reused. Explicit cache fields from the caller, and the opt-out
environment variable, still take precedence.

JSON and SSE collect usage.prompt_tokens, usage.completion_tokens, and
usage.prompt_tokens_details.cached_tokens, including on foreign servers.
The engine's top-level `vade.prefill_cache_restored_tokens` is a fallback;
conflicting duplicate counts are invalid. Prefill timing accepts top-level
`vade.prefill_ms` as well as `usage.vade` timing.

The per-reply footer sums restored / prompt tokens across completed calls and
prints cache reuse %. A missing count is not the same as zero. Identical
duplicate snapshots count once; conflicting or malformed snapshots become
unknown. Truncated, cancelled or error responses never publish provisional
cache usage; counts appear only after full HTTP and model validation. Cached
tokens still occupy logical context.

After worker cleanup, the UI reports each completed model response in that user
run (up to the runtime's 61 model turns). /context shows the last attempted
request's exact admission or estimate, local wire bytes, capability status
and per-response counts, plus a separate session sum with telemetry coverage.
It does not count an unsent next prompt, and during active work it labels the
snapshot as the previous one. Noninteractive answers stay on stdout, while
automatic telemetry goes to stderr. Demo and delegated modes report this
contract as unavailable.

### For library hosts

These notes matter only if you embed the harness library in your own program.
For context-budget and UI integration, NativeModel::context() and
ChatClient::context() expose bounded, typed metadata. NativeModel::begin_run()
clears the per-run list but keeps the session totals. Compaction, journal and
tool policy are owned elsewhere. After an explicit history rewrite, the owner
must also call reset_prompt() or replace the model; neither one executes or
replays tools.

The native UI performs these resets for `/compact`, `/fork`, `/resume`,
`/switch` and launch-time `--resume`. Compaction preserves the in-memory session
affinity and cumulative usage; selecting a journal creates a fresh client and
affinity, while sharing bounded checkpoint ownership across native branch
clients.

`reestablish_prompt(PromptBoundary)` clears stale last-request/response
telemetry, records the explicit boundary, and forces negotiation on the next
user prompt, even after standard-chat fallback. The complete rebuilt request
passes advertised exact admission before generation. Compaction keeps the
frozen options, tools and system/developer wire bytes and replaces only what
follows them; other boundaries reset the builder.

Native context decisions count tokens through admission, on the worker,
before the compaction and fit checks. When the final submitted body is
byte-identical, that successful count also serves as generation admission,
which saves a round trip. If the KV decoration changed, the actual body still
needs its own admission. Admission is still not transactional.

With opt-in `--prewarm-compaction`, a discarded one-token completion warms a
persisted compacted view; no output or tool execution enters the transcript.
Its reported usage and completion wall time appear separately in `/context`.
Without the flag, local boundary operations send nothing on their own.

A recorded, unexpired input checkpoint may be supplied on the next request;
changing the view clears its hints. The UI tells a pending prefix apart from a
submitted one. Compaction invalidates later causal KV, so cached tokens can
drop, and only the new response's actual counters describe reuse. Local byte
equality, affinity and old usage cannot establish a cache hit for the
replacement context.

Context also exposes cumulative counts of requests, admissions, refusals and
HTTP 429/503 replies, the latest HTTP stage and status, and the timing of the
first generated SSE delta in native mode.

The [workload driver](workload.md) uses per-attempt differences and validates
usage against admitted counts. The `vade.engine.v1` contract gives exact
prompt lengths, but not token IDs or longest-common-prefix (LCP) counts, so
the driver's reusable-prefix denominator (U) is always labeled an estimate, even
when the counts under it are exact.

Warm workload turns also count the retained completed transcript, including
assistant reasoning and provider fields, through a separate admission-only
request. These observations do not change generation telemetry, prefix or KV
state, and their request counts and errors are reported separately. Exact
render lengths improve U without claiming an exact token LCP.

<a id="opaque-gateway"></a>

## What the gateway passes through unchanged

The gateway preserves the engine's reply body. It changes request role values
on specific routes and inserts greedy defaults when sampling fields are absent.
It does not swap the model, add prompt text or make up a cache key. In practice:

- Requests are checked as JSON (up to 8 MiB, with the usual depth, value and
  duplicate-key limits). On POST Messages routes, including token counting,
  `messages[i].role: "system"` becomes `"user"`. The top-level `system` field
  stays unchanged. On POST chat completions routes, `messages[i].role: null`
  becomes `"assistant"`. These changes apply with or without the `/v1` prefix.
  See the [role rules](../src/gateway/system_turns.rs).
- A generation request that sets no sampling field gets the greedy defaults.
  Any explicit sampling field suppresses that insertion, including a field
  set to `null`. The gateway applies the
  [sampling rules](../src/gateway/sampling.rs) after the role changes in
  [`prepare_dispatch`](../src/gateway.rs). All other body bytes stay unchanged,
  including spacing, key order, escapes, unknown fields and number spelling.
- Replies, error bodies and stream events pass through exactly, including
  binary bytes, comments, unknown events, multiline data, event order and
  Unicode split across chunks. Only HTTP framing and header spelling are
  normalized, so an identical payload does not mean identical HTTP wire bytes.
- Every reply carries `X-Vade-Gateway: v1`, local errors included, and any
  upstream copy of that header is replaced. Doctor uses it to recognize a
  gateway. It is a label, not proof of identity.
- Credentials you send go only to the upstream you configured. The gateway
  never picks up credentials from anywhere else and never follows redirects.
  Hop-by-hop headers and trailers are removed.
- If the client disconnects, the upstream request is dropped, even during a
  silent wait. A closed socket does not mean the engine confirmed the cancel.
- If the upstream reply is cut off, the client sees it cut off: the final
  chunk marker is never sent. A POST that fails partway returns a local 502
  saying the outcome may be unknown and the request was not resent.

The single-upstream gateway does no capability negotiation, and neither gateway
mode admits, decorates or tokenizes another client's request. Both relay
`GET /v1/capabilities` and `POST /v1/admission`, so clients can use those
routes themselves. Known API suffixes map to the matching sibling route, and
custom configured paths stay exact in single-upstream mode. A Responses body
sent to a chat route is forwarded unchanged, so the engine's error comes back
as it is. Repeated upstreams turn on [cache-aware routing](gateway-routing.md),
with background capability probes and passive usage metrics; there, a request
the engine refused before any reply byte can be retried once on another
endpoint, and a request with an uncertain outcome is never replayed.

Both modes relay `POST /v1/cancel` and `POST /v1/kv/release`. With one
upstream they are plain relays. With several, the gateway does not record which
endpoint holds a key, so it sends each call to every upstream (the core
gateway at once, within 1.5 seconds; `vade-gateway` one after another) and
answers with the first 2xx reply, in configuration order, that reports
`"cancelled":true` or `"released":true`, otherwise the first reply, or HTTP 502
when no upstream answers. Keys are random capabilities, so an unknown key changes nothing on the
other endpoints. Cluster mode still has no handle-based affinity: a request
that names a checkpoint handle is routed like any other and can reach an
endpoint that does not hold it, so use one upstream or a direct engine endpoint
for checkpoint work.

### Gateway buffers and timeouts

- Relay work uses one 8 KiB upstream read buffer, no response queue and at
  most 16 active requests. Each chunk is written and flushed before the next
  read, so a slow reader slows the upstream down (the OS socket buffers come on
  top). Cluster mode also reads a bounded JSON/SSE copy for usage; if parsing
  fails, only the metrics are lost.
- The 64 MiB response bound, the idle and absolute deadlines and cancellation
  polling still apply. Socket I/O polls every 50 ms. Callers that close their
  write half are not supported.
- Pipelined bytes are buffered on demand, up to 8,421,376 bytes, next to the
  8,192-byte downstream read buffer, so they cannot hide a client disconnect.
- Complete framed responses may reuse sockets, with at most 16 idle sockets
  that expire after 15 seconds. A stale pooled connection never causes an
  accepted request to be resent.
- Unsupported routes or methods, TLS, compression, upgrades, and ambiguous
  framing or representation headers fail explicitly.

## Prompt-cache fields on the FP4-Local server

This section covers the FP4-Local server: `flash-serve ... --fp4-local`, serving
the `nvidia/Qwen3.8-Flash-Next-FP4-Local` checkpoint. The
[cache switch and topology guard](../../engine/crates/serve/src/prefix_switch.rs)
define where prefix reuse is available.

- **Prefix cache.** Started with `VADE_FLASHNEXT_PREFIX_CACHE=1`, the server
  restores exact prompt prefixes from its cache. The default is off: no reuse
  through this cache, and capabilities report `prefix_cache:false`.
- **Two Sparks.** The tensor split allows the prefix cache and restores a
  matching prefix on each rank. The [HTTP adapter](../../engine/crates/serve/src/flashnext.rs)
  checks `tp_prefix_supported` before it allows the switch.
- **Layer chain.** The chain used on three or more Sparks refuses
  `VADE_FLASHNEXT_PREFIX_CACHE=1`. It has no prefix restore contract.
- **Host memory.** When the KV reserve needs host memory for spare session
  pages, saved prefixes give it up before the reserve's pressure rule applies
  (`VADE_FLASHNEXT_CACHE_YIELD`, default on; `0` means only the session pool
  limits it). The limit counts only what the KV reserve keeps.

| Field (chat completions) | Meaning |
| --- | --- |
| request `cache` | `true`: the request may restore and retain prompt prefixes and returns an `X-Session-Capability`; send `session` only with a value you received. `false`: cold, nothing retained. Absent: on this server the same as `false` unless the request has a `prompt_cache_key` (the cache serves only requests that ask; everything else keeps the no-cache path) |
| request `prompt_cache_key` | A hint that keeps related requests together (cohort affinity); on this server it also opts the request into the prefix cache (restore and retain). Keep it stable per conversation |
| `usage.prompt_tokens_details.cached_tokens` | Prompt tokens restored from the cache; equals `vade.prefill_cache_restored_tokens` |
| `vade.prefill_cache_reuse_pct` | Restored / prompt tokens in percent, one decimal: print this as "cache reuse" |
| `vade.prefill_cache_restore_ms` | Wall ms from cache lookup to the restored session; `null` without a hit |
| `vade.prefill_ms`, `vade.admission_to_first_logits_ms` | Engine prefill wall and admission-to-first-logits (engine-side first token) |

`vade` is the top-level object of the JSON body and of the final SSE chunk (the
chunk that carries `usage`); no `stream_options` are needed.

The server restores up to the last whole 64-token tile of a prompt, because
the model's recurrent state is exact only at those boundaries. So a follow-up
turn in the same conversation restores at most its previous prompt, rounded
down to 64 tokens; the previous reply and the new message are read fresh. For
that hit, the start of the request must stay byte-for-byte the same from turn
to turn: system prompt, tool schemas, history order and the rendered previous
turns.

Plain decoding (`draft:"plain"`) and sampled requests read, restore and retain
normally; a plain request that saves a prefix also keeps the draft history that
prefix needs. Outputs are the same bytes as a cold request.

<a id="output-extension-provenance-824ab96f"></a>

## How the output limits changed (October 2026)

In October 2026 we raised the harness's output limits and timeouts. This table
compares the current behavior with the earlier main build `acdff5e1`, which used
an 8,192-token output default, a fixed session reservation and manual
continuation. The new values came from change `824ab96f` and still need to be
checked on live hardware. The comparison is dated October 12, 2026, and was
updated October 13.

| Setting | Main baseline `acdff5e1` | Integrated extension from `824ab96f` |
| --- | --- | --- |
| Flash-Next server and `vade up` output default | 8,192 tokens | `min(65536, context)` tokens |
| `vade chat` output selection | Passes the saved session output limit to the harness | Omits the fixed output argument; harness chooses each turn's reservation |
| Absolute HTTP lifetime | 300 seconds | 3,600 seconds |
| Ordinary HTTP idle timeout | 30 seconds | 30 seconds |
| Native response wait while queued or reading a prompt | Up to the 300-second lifetime | Up to the 3,600-second lifetime |
| Native assistant-message bound | 512 KiB | 4 MiB |
| Native response-stream bound | 2 MiB | 64 MiB |
| Native response-event bound | 16,384 | 262,144 |
| Native tool-argument bound per response | 64 KiB | 256 KiB |

Today's output reservation and admission work as described in [Exact
admission](#exact-admission).

Compaction thresholds count at most 8,192 reserved output tokens. When an
automatic reservation falls below 8,192, native mode must compact before sending
generation, or stop if compaction cannot make room. Smaller advertised ceilings
and explicit output settings retain their own bounds.

The extension enables automatic continuation of length-limited answers in
direct and native agent modes. Continuations share the original request's work
budget; native continuations also share its model-turn and tool budgets. Both
modes allow at most 32 consecutive continuations. Incomplete tool
proposals still stop without execution. `--no-auto-continue` or native config
`"auto_continue": false` opts out. Workload measurements explicitly disable this
behavior and retain one response per turn and a 1–8,192-token output reservation.

Both versions retain the 5-second connect timeout, 250 ms cancellation polling
and 4 MiB native history bound; local work/deadline limits can end a request
sooner. Current contracts and command guides describe the integrated behavior;
this dated comparison preserves its origin without claiming native qualification.

## Verification and credit

From the `harness/` folder, run `cargo test --offline --all-targets`. Local
fixtures cover full admission bytes, tool continuations, decorated prefixes,
resets and restarts, unsupported capabilities, refused or malformed admission,
cancellation, usage and actual CLI /context output.
Gateway cases cover opaque unknown fields/dialects/errors, fragmented streams,
contract routes, blocked writes, disconnects, truncated framing, and uncertain
POSTs on reused sockets. These results cover the harness against mocks only.

The schema is first-party Engine work; no Engine code was copied. Original
schema work used source comparison and first-party fixtures. Conceptual
references include [Codex context/transport separation](https://github.com/openai/codex),
[Hermes request decoration](https://github.com/NousResearch/hermes-agent), and
[Goose prefix invariance](https://github.com/aaif-goose/goose). Protocol
references are in [performance.md](performance.md). No competitor code or
dependency was added.
