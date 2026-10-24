# Harness capability, admission and cache contract

The capability and admission clients match the separate Engine's **vade.engine.v1** implementation at
**a211811** (2026-10-08), read as source only; the earlier report names fb77f4f.
Schema verification uses first-party Rust fixtures. See [setup and doctor](setup-doctor.md)
for endpoint capability checks and [performance](performance.md) for retained
host measurements. Those checks do not establish live cache performance.

The contract review also checked `crates/serve/src/protocol.rs` and `tokenizer.rs` at Engine
**d2a95d2**, read-only, for assistant history shape. The parser requires string or
text-part content on an assistant without tool calls. The native adapter supplies
`content: ""` for absent/null content on these replies at response ingestion,
including reasoning-only, length-limited SSE and JSON responses. Qwen reasoning
remains in `reasoning_content`; an accepted `reasoning` field is retained
when provided. No thinking text is moved into visible content or renamed, and
already-frozen history/options are never rewritten on a subsequent request.
Tool-call messages retain their legal absent/null content representation.

## Negotiation

Client construction performs no network I/O. On the first model request, the
worker fetches GET /v1/capabilities before generation. Once negotiated, it
refreshes capabilities before each model request, including tool continuations.
DNS/connect/reads stay off the terminal thread.

| Engine field | Harness interpretation |
| --- | --- |
| object, contract_version | engine.capabilities, vade.engine.v1 |
| model | Must match the selected model; no substitution |
| identity.weights.fingerprint | Opaque checkpoint identity |
| identity.weights.fingerprint_kind | Retained interpretation; currently safetensors_header_metadata, not a full weight-content hash |
| identity.tokenizer_fingerprint, identity.template_fingerprint | Opaque rendering identities |
| context_limit, input_limit, output_limit | Positive integer token limits |
| tools, thinking, default_thinking | Advertised features/default; null or absent support is unknown/unsupported |
| cache_domain, cache_scope, prefix_cache | Engine-owned reporting identity and feature; no proof of a hit |
| cache_granularity_tokens | Optional positive integer cache block size; absent/null unknown. Doctor sizes above two blocks and verifies with admission; advertisement does not prove retention |
| prompt_cache_key | Only "scheduling_hint" enables generated affinity |
| admission, admission_kind | Only "/v1/admission" plus "context_fit" enables exact admission |
| kv_branch_handles, kv_checkpoints | Explicit opt-in to scoped input checkpoints, quotas, TTL and the fixed release route; see [lifecycle](engine-checkpoints.md) |
| endpoints | Must advertise /v1/chat/completions for native/direct mode |

Capability responses are bounded at 64 KiB; retained strings at 512 bytes each.
Unknown fields are tolerated, missing identity remains unknown, and fingerprints
and their reported kind are compared as opaque bytes. The current Engine's weight
identity does not detect same-size payload changes; the harness makes no independent
content-integrity claim. The harness implements no tokenizer, cryptography or KV
state, and does not assume continuation or remote cancellation.
The optional [checkpoint client](engine-checkpoints.md) follows the later
Engine `kv-handles` source contract, using only opaque handles and local mocks.

Known chat suffixes discover sibling routes on the configured origin/base path.
Custom exact paths skip discovery. An advertised URL cannot redirect admission
to a different origin. HTTPS still fails explicitly.

Initial 404/405/501, foreign metadata, or an unknown contract version selects
standard chat behavior for that session. No guessed custom fields or admission
calls are sent. Explicit native mode retains its configured standard chat tools;
it never changes dialect or strips tools. A recognized contract without tool
support refuses a tool-bearing request. Other discovery errors, malformed
recognized metadata, model/dialect mismatch, or a previously negotiated contract
disappearing stop dispatch. There is no silent downgrade after negotiation.
/new explicitly starts a fresh negotiation.

## Exact admission

Before each negotiated generation, POST /v1/admission contains:

    {"contract_version":"vade.engine.v1","endpoint":"/v1/chat/completions","output_tokens":32,"request":{"model":"mock","max_tokens":32,"messages":[{"role":"user","content":"Hello"}]}}

The real request member embeds the complete final generation body **byte for
byte**, including ordered tools/options, all messages/results/reasoning, provider
extensions, and affinity. No parsed-and-reserialized copy is used for counting.
After successful admission, generation receives those exact embedded bytes.
Tool results are counted again before their next model turn.

The reservation equals the generation request's max_completion_tokens or
max_tokens. Explicit caller options, including `--max-output-tokens`, remain
fixed and must fit both the advertised output limit and remaining context.
Without an explicit limit, direct/native clients insert max_tokens and adapt it
for each request to the smaller of the advertised output limit and remaining
context minus 256 safety tokens. By default, context uses the advertised limit directly. An explicit local
budget bounds it; without capabilities the fallback is 131,072 tokens. The fallback output ceiling is 8,192 tokens when
no contract is available.

Exact admission supplies the prompt count for this calculation. If the default
reservation needs adjustment, the client changes only its inserted max_tokens
number and admits the final body again before generation. Historical messages,
tool schemas and explicit options keep their exact bytes. Without exact
admission, the same calculation uses a labeled local estimate. Generation is
refused if no output room remains or an explicit reservation cannot fit; native
mode may use its existing in-task compaction. Gateway requests retain the
caller's bytes.

Replies must identify engine.admission, the same contract/model/reservation,
matching context/input/output limits, admission_kind:"context_fit", and
reserves_resources:false. Integer counts and fit arithmetic are checked without
subtracting cached input. Refusal, non-200 status, changed/malformed counts,
truncation, or cancellation prevent generation. No automatic retries occur.
Refusals show counts and a local explanation; raw upstream reasons are suppressed.

Admission does not reserve memory, a scheduler slot, or KV. It is not a
transactional snapshot spanning generation; the Engine can still reject or
change state between requests. Generation usage is shown separately.

Native generation responses may remain silent while queued or prefilling. They
keep the dispatched connection until the 3,600-second absolute HTTP lifetime
or earlier cancellation instead of the ordinary 30-second idle timeout. This
applies before headers, after headers/role-only SSE events, and to JSON fallback,
including the HTTP child adapter. The original request's active-work budget,
Esc and child/workload deadlines can cancel sooner. Native UI/workload progress
says `queued / awaiting upstream` until actual generated data arrives (or JSON
completes); the contract does not provide an authoritative queue position.
Cancellation polling remains 250 ms. Discovery, admission, request writes and
direct chat use the ordinary idle timeout. Gateway generation POST responses
also use the full HTTP lifetime; other gateway I/O uses the ordinary idle bound.
No retry is introduced.

Engine cancellation and generation deadlines are terminal errors with codes
`client_cancelled` and `generation_deadline`, respectively. JSON failures use
HTTP 409 and 504; an already streaming reply retains its deltas and ends with a
structured error instead of `finish_reason: "length"`. Native and direct chat
recognize these codes as interruption/timeout and stop without automatic
continuation. Already displayed text remains visible; a failed stream does not
commit a partial assistant or execute any proposed tools. Only a real token
limit permits the normal length-driven continuation policy.
Deadline checks include model work and final synchronization that finish after
the deadline, even when that work reaches the requested output token limit.

Without exact admission, "wire bytes + 256" is explicitly labeled a rough
**estimate**, including serialized options/tools/history. It is not a tokenizer,
a guaranteed bound on hidden framing, or an admission decision. Unknown limits
remain unknown; existing independent byte/resource guards still apply.

## Affinity, usage and UI

An advertised scheduling-hint feature gets one OS-random, non-secret key per
in-memory client session. No private path/model data forms the key.
Caller-supplied keys survive. Fixed decoration precedes history; native requests
retain their append-only byte-prefix check.

The key survives normal turns, tools, reconnects, Engine restarts, and
NativeModel::reset_prompt() compaction boundaries. A new client, /new, or
ChatClient::reset_context() gets a fresh key. Keys are neither printed nor stored
in the session counter. Changes to fingerprints, domain/scope or thinking default
increment the observed cache-domain change counter. Affinity never proves reuse.

Recognized cache scheduling hints also enable `cache: true` when `prefix_cache`
is false; the advertisement and request never establish actual reuse. Explicit
caller cache fields and the opt-out environment variable retain precedence.

JSON and SSE collect usage.prompt_tokens, usage.completion_tokens, and
usage.prompt_tokens_details.cached_tokens, including on foreign servers.
The engine's top-level `vade.prefill_cache_restored_tokens` is a fallback;
conflicting duplicate counts are invalid. Prefill timing accepts top-level
`vade.prefill_ms` as well as `usage.vade` timing. The per-reply footer sums
restored / prompt tokens across completed calls and prints cache reuse %.
Missing differs from zero. Identical duplicate snapshots count once; conflicting
or malformed snapshots become unknown. Truncated, canceled or error responses
never publish provisional cache usage. Complete HTTP/model validation precedes
publication. Cached tokens remain part of logical context occupancy.

After worker cleanup, the UI reports each completed model response in that user
run (up to the runtime's 61 model turns). /context shows the last attempted
request's exact admission or estimate, local wire bytes, capability status,
per-response counts, and a separate session sum with telemetry coverage.
It does not count an unsent next prompt. During active work it identifies the
snapshot as the previous one. Noninteractive answers stay on stdout; automatic
telemetry goes to stderr. Demo/delegated modes report this contract unavailable.

NativeModel::context() and ChatClient::context() expose bounded typed metadata
for context-budget/UI integration. NativeModel::begin_run() clears the per-run
list without clearing session totals. Compaction/journal/tool policy is separately
owned. After explicit history rewrites, the owner must also reset_prompt() or
replace the model. Neither operation executes or replays tools.

The native UI now performs these resets for `/compact`, `/fork`, `/resume`,
`/switch` and launch-time `--resume`. Compaction preserves the in-memory session
affinity and cumulative usage; selecting a journal creates a fresh client and
affinity, while sharing bounded checkpoint ownership across native branch clients.
`reestablish_prompt(PromptBoundary)` clears stale last-request/response telemetry,
records the explicit boundary, and forces
negotiation on the next user prompt, even after standard-chat fallback. The
complete rebuilt request passes advertised exact admission before generation.
Compaction retains the frozen options/tools and system/developer wire bytes and
replaces only their suffix; other boundaries reset the builder. Native context
decisions now count with admission on the worker before compaction/fit checks.
A successful count can supply generation admission when the final submitted body
is byte-identical, avoiding a redundant round trip. Changed KV decoration still
requires admission of the actual body. Admission remains nontransactional.
With opt-in `--prewarm-compaction`, a discarded one-token completion warms a
persisted compacted view; no output/tool execution enters the transcript. Its
reported usage and completion wall time appear separately in `/context`.
Otherwise local boundary operations send nothing by themselves. A recorded, unexpired input
checkpoint may be supplied on the next request; changing the view clears its hints.
The UI distinguishes a pending prefix from a submitted one.
Compaction invalidates later causal KV; cached tokens can drop and only the new
response's actual counters describe reuse. Local byte equality, affinity and old
usage cannot establish a cache hit for the replacement context.

Context also exposes cumulative request/admission/refusal and HTTP 429/503
counts, the latest HTTP stage/status, and native first-generated-SSE-delta timing.
The [workload driver](workload.md) uses per-attempt differences and validates
usage against admitted counts. `vade.engine.v1` supplies exact prompt lengths,
not token IDs or LCP counts; the driver's reusable-prefix denominator remains
an explicitly labeled estimate even when the underlying counts are exact.
Warm workload turns additionally count the retained completed transcript through
a separate admission-only request, including assistant reasoning/provider fields.
These observations do not mutate generation telemetry/prefix/KV state; their
request counts and errors are reported separately. Exact render lengths improve
U without claiming an exact token LCP.

## Opaque gateway

The single-upstream gateway performs no capability negotiation. Neither gateway
mode admits, decorates or tokenizes another client's request. Both relay
GET /v1/capabilities and POST /v1/admission so clients may explicitly use those
routes. Repeated upstreams opt into [cache-aware routing](gateway-routing.md),
with separate background capability probes and passive usage metrics.
The gateway does not yet route `/v1/kv/release` or provide handle-based cluster
affinity. Use a directly selected Engine endpoint for checkpoint lifecycle work;
through the gateway, release attempts are unconfirmed and Engine TTL is the fallback.
Chat and Responses remain opaque. Known API suffixes
map to the matching sibling; custom configured paths stay exact in single-upstream mode. A Responses body
on a chat route is forwarded unchanged, preserving the upstream error. No dialect
repair, model substitution, affinity rewrite, prompt insertion, or retry occurs.

Requests are buffered as bounded JSON objects (8 MiB, with existing depth/value
and duplicate-key limits). A borrowed view validates without normalizing
whitespace, number spelling, key order, escapes or unknown fields. Response
payloads remain opaque for relay: errors, binary bytes, comments, unknown events,
multiline data, event order and fragmented Unicode remain exact. Cluster mode
observes a bounded JSON/SSE copy for usage; parse failure only suppresses metrics.

HTTP transfer framing and header spelling/whitespace are intentionally normalized;
payload equivalence does not mean identical HTTP wire bytes. The gateway adds
`X-Vade-Gateway: v1` to responses (including local errors) and replaces any upstream
copy of that header. Doctor uses this local role marker before probing stats; it
is a diagnostic advertisement, not an authenticated identity. Hop-by-hop headers
and trailers are removed. Explicit client-supplied end-to-end authorization goes
only to the selected explicitly configured origin. Credentials are never scavenged and redirects
are never followed.

One 8 KiB upstream read buffer, no response queue, and at most 16 active requests
bound relay work. Cluster telemetry adds bounded JSON/SSE scratch as documented
in the routing guide. Every downstream chunk is written/flushed before the next read.
A blocked reader applies backpressure; finite OS socket buffers are additional
to this in-process bound. The 64 MiB response bound, idle/absolute deadlines and
cancellation polling remain. After a complete request, downstream read EOF or a
socket error drops the upstream socket, including during silent waits; socket I/O
polls every 50 ms. Write-half-closed callers are unsupported. Pipelined bytes are
buffered up to 8,421,376 bytes on demand, alongside the 8,192-byte downstream read
buffer, so they cannot hide EOF. Socket closure is not remote cancellation acknowledgment.

Upstream truncation leaves the downstream chunk terminator unsent. Uncertain
POST failures return a local 502 explaining that the outcome may be unknown and
the request was not replayed. Complete framed responses may reuse sockets: at
most 16 idle sockets, expiring after 15 seconds. A stale pooled connection cannot
cause an accepted request to be resent. Unsupported routes/methods, TLS,
compression, upgrades and ambiguous framing/representation headers fail explicitly.

## Verification and credit

Run cargo test --offline --all-targets. Local fixtures cover full admission bytes,
tool continuations, decorated prefixes, resets/restarts, unsupported capabilities,
refused/malformed admission, cancellation, usage and actual CLI /context output.
Gateway cases cover opaque unknown fields/dialects/errors, fragmented streams,
contract routes, blocked writes, disconnects, truncated framing, and uncertain
POSTs on reused sockets. These are harness/mock claims only.

The schema is first-party Engine work; no Engine code was copied. Original
schema work used source comparison and first-party fixtures. Conceptual references include [Codex context/transport separation](https://github.com/openai/codex),
[Hermes request decoration](https://github.com/NousResearch/hermes-agent), and
[Goose prefix invariance](https://github.com/aaif-goose/goose) as conceptual
references. Protocol references are in [performance.md](performance.md).
No competitor code or dependency was added.

## Output extension provenance (824ab96f)

**Source comparison dated October 12, 2026, updated October 13.** The big-output
work originated at `824ab96f`. The table compares the current harness behavior
with main baseline `acdff5e1`, whose output default was 8,192 tokens, with fixed
session reservation and manual continuation. The integrated behavior still
requires live hardware qualification; the comparison is source provenance.

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

Automatic output reservation is the smallest of 16,384 tokens, the advertised
output ceiling and the remaining context after the current prompt plus 256 safety
tokens. Context defaults to the advertised limit, bounded only by an explicit local
override. Without an
advertised output ceiling, the fallback is 8,192 tokens. Exact admission supplies
the rendered prompt count when available; otherwise the client uses its labeled
local estimate. If the reservation changes after counting, the client changes
only the inserted output-control number and admits the resulting request before
generation. Historical message and tool bytes remain unchanged. Explicit output
limits stay fixed and must fit; no output room means no generation is sent.
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

## Engine prompt-cache fields (2026-10-19)

Source baseline: `26b1692`. The 1-Spark FP4-Local
server (`flash-serve ... --fp4-local`, world 1) restores exact prompt prefixes
from its cache when started with `VADE_FLASHNEXT_PREFIX_CACHE=1` (default off:
no reuse at all, capabilities `prefix_cache:false`, as on main 26b1692 and earlier).
World 2's layer pipeline (`VADE_W2_PP`) refuses the switch at startup on both
ranks: a restored prefix would hold the front rank's layers only, while the back
rank keeps no prefix checkpoints.
Retained prefixes yield the host memory the KV reserve's spare session pages need
before its pressure rule (`VADE_FLASHNEXT_CACHE_YIELD`, default on; `0` = the
session pool alone bounds it); the bound counts the KV reserve's retention only.

| Field (chat completions) | Meaning |
| --- | --- |
| request `cache` | `true`: the request may restore and retain prompt prefixes and returns an `X-Session-Capability`; send `session` only with a value you received. `false`: cold, nothing retained. Absent: on Flash-Next the same as `false` unless the request has a `prompt_cache_key` (the cache serves only requests that ask; everything else keeps the no-cache path) |
| request `prompt_cache_key` | Cohort affinity hint; on Flash-Next it also opts the request into the prefix cache (restore and retain). Keep it stable per conversation |
| `usage.prompt_tokens_details.cached_tokens` | Prompt tokens restored from the cache; equals `vade.prefill_cache_restored_tokens` |
| `vade.prefill_cache_reuse_pct` | Restored / prompt tokens in percent, one decimal: print this as "cache reuse" |
| `vade.prefill_cache_restore_ms` | Wall ms from cache lookup to the restored session; `null` without a hit |
| `vade.prefill_ms`, `vade.admission_to_first_logits_ms` | Engine prefill wall and admission-to-first-logits (engine-side first token) |

`vade` is the top-level object of the JSON body and of the final SSE chunk (the
chunk that carries `usage`); no `stream_options` are needed. Flash-Next restores
the prompt's last whole 64-token tile boundary (its recurrent state is exact
only there), so a follow-up turn reuses up to the previous prompt rounded down
to 64 tokens; the previous reply and the new message are read fresh. Reuse needs
a byte-stable prefix (system prompt, tool schemas, history order and the
rendered previous turns). Plain decoding (`draft:"plain"`) and sampled requests
read, restore and retain normally (a retaining plain request keeps the draft
history its published prefix needs). Outputs are the same bytes as a cold request.
