# Cache-aware gateway routing

This module provides a first-party, standard-library-only router for external harnesses.
Inference, tokenization, scheduling and KV remain upstream-owned. Verification
used three independent loopback mocks on macOS ARM64, never Vade Engine, Sparks,
GPUs, installed agents or live providers.

## Use

One `--upstream` retains the existing opaque single-endpoint relay, with no
background discovery. Repeating it enables capability-verified cluster routing:

```sh
vade-harness gateway --bind 127.0.0.1:8787 \
  --upstream http://127.0.0.1:18081 \
  --upstream http://127.0.0.1:18082 \
  --upstream http://127.0.0.1:18083
```

These are example addresses for separately provided mocks. Running the command
contacts those configured endpoints; it does not start mocks. Configure an
external client to use the gateway's text Completions (`/v1/completions` or
`/completions`), Chat Completions, Responses, or Anthropic
Messages route (including `/v1/messages/count_tokens`) and
its upstream's model. Nothing is added to its prompt or request body. Each
configured endpoint must be authorized to receive that client's forwarded
end-to-end headers, including any explicitly supplied authorization.

`GET /_vade/gateway/stats` returns local JSON metrics in cluster mode. It never
forwards to an upstream. IDs are zero-based `--upstream` positions, not hostnames.
The `Cluster::stats()` library handle exposes the same snapshot. No body,
credential, client cache key, routing hash, model name or endpoint URL is logged
or included in these metrics. Metrics and affinity live only in memory.

The library API is `Cluster::new(&[String], RoutingConfig)` followed by
`run_cluster(bind, cluster)` or `run_cluster_until(bind, cluster, stop)`.
Construction validates configuration without network I/O. Clones share routing
state and metrics; use one running gateway owner per cluster. `RoutingConfig`
allows bounded overrides of the defaults; the CLI uses these defaults:

| Bound | Default |
| --- | --- |
| Configured endpoints | 1–16 in library; repeated CLI selects cluster mode |
| Affinity entries | 4,096, maximum configurable 65,536 |
| Affinity idle TTL | 15 minutes, maximum configurable 24 hours |
| In-flight requests per endpoint | 8, configurable 1–16 |
| Total gateway client connections | 16 |
| Capability probe interval | 10 seconds after previous probe completes |
| Capability freshness | 30 seconds |
| Probe deadline / body | 5 seconds / 64 KiB |
| Failure / overload cooldown | 5 seconds |
| Upstream idle connection pool | 16 sockets total, 15-second idle expiry |

Bind remains loopback-only. TLS, credentials embedded in URLs, ambiguous HTTP
framing, compressed representations and upgrades fail explicitly. Cluster URLs
must be known API suffixes or `/v1` base paths; custom exact paths remain available
in single-upstream mode. Cluster URLs cannot contain query strings. The caller's
query is still relayed and included in its routing scope.

## Compatibility and discovery

Each configured endpoint gets its own bounded background discovery worker for
`GET /v1/capabilities` (or the equivalent sibling under a configured base path).
Probes carry no caller body, affinity key or caller credentials and follow no
advertised URLs or redirects. They use the configured upstream API key as a
Bearer credential, when present, through the same transport as generation.
Authenticated capabilities are supported; configure an upstream key authorized
to read them when the server requires authentication.
All initial probes must finish or fail before generation routing starts; clients
receive 503 during that interval. One hung probe does not stop other probes, the
listener or the stats route. Owner shutdown cancels and joins discovery workers.
Socket cancellation polls at 250 ms; std OS DNS lookup cannot be interrupted.

Only recognized `vade.engine.v1` capabilities with complete weights, tokenizer
and template fingerprints can join a group. Group equality also requires model
name, weights fingerprint kind, advertised Completions/Chat/Responses/Messages/token-count routes, token limits,
tool/thinking flags and defaults, cache scope/features and admission support to
match. Unknown extension fields are ignored. Fingerprints are opaque advertised
identities; the harness makes no independent weights-integrity claim.

Instance `cache_domain` is deliberately excluded from group equality: replicas
have different KV instances. A changed domain invalidates that endpoint's old
affinity hints. A failed probe invalidates its hints and health. The first valid
compatibility signature for each endpoint is pinned for the gateway lifetime;
later fingerprint, model or contract changes quarantine it until the original
signature returns or the operator explicitly restarts/reconfigures the gateway.
Expired capabilities also make an endpoint unavailable. Selection and discovery
are not an atomic transaction with the upstream; changes between probes or
between selection and dispatch remain possible.

Different model names can form separate groups. A request's `model` chooses its
group; admission uses `request.model`. If the same model name has multiple
incompatible groups, routing returns 409 instead of picking a fingerprint. That
ambiguity remains even when one group is unhealthy. A request without a model,
including capability discovery, requires exactly one discovered group. Thus a
separate gateway per model is the simplest configuration for clients that first
fetch capabilities. A route the group does not advertise in `endpoints`
(including `/v1/completions`) or an unavailable group returns 503. Unknown/foreign capability contracts are never treated as compatible.

## Affinity and fallback

Within a compatible group, the router hashes a decoded client `prompt_cache_key`
when it is a JSON string (including empty). Otherwise it hashes leading **original
wire bytes** from `system`, `instructions`, `tools`, and up to 17 leading messages,
stopping after the first message whose role is neither system nor developer.
Each selected value contributes at most 16 KiB; scalar Responses `input` uses
its first 16 KiB, and a text Completions `prompt` uses its first 16 KiB of wire
bytes. Appending later messages or prompt text beyond that prefix does not change
this hint.
Tool/system edits, different serialization or long shared/truncated prefixes
can change or alias inferred affinity. A stable explicit key is preferable.

Keys also include the API family (Completions, chat, Responses or Messages;
admission uses its target `endpoint`), query and forwarded Authorization,
OpenAI-Organization and OpenAI-Project header values. The process-randomized
standard-library hash is only a scheduling hint. Collisions are possible. It is
not cryptography, an authorization boundary, a durable identifier or a claim
that two prompts have the same tokens. Only hashes and endpoint/expiry metadata
are retained in the bounded table; request text and credentials are not retained
there. Upstream authorization and cache isolation remain upstream responsibilities.

A healthy, unsaturated affine endpoint wins. Otherwise the router atomically
reserves the endpoint with the fewest **gateway in-flight requests** in the same
group, rotating ties. This does not observe work submitted directly to engines
or measure GPU/KV utilization. Reservations last through response delivery and
are released even on cancellation or failure. New affinity is recorded at
selection so simultaneous related requests can use it. TTL refreshes on use;
expired entries are removed on routing/stats reads, and the oldest-used entry is
evicted at capacity. There is no response cache or independent KV store.

A transport failure makes the endpoint unavailable pending another successful
probe. HTTP 429 or any 5xx starts a cooldown; 429/503/529 also increment `overloads`.
A numeric `Retry-After` uses 1–300 seconds; dates/invalid values use the default.
A health probe cannot erase an overload cooldown or a newer transport failure.
When all compatible endpoints are unavailable/saturated, the gateway returns 503.

**At most one refusal retry occurs before response forwarding.** A fresh DNS or
connection failure, or an HTTP 429/503/529 refusal before downstream headers/body
are sent, can retry the exact request once on another verified healthy compatible
endpoint. The failed endpoint is excluded even if a health probe races the retry.
If no alternate is available, the original refusal is relayed unchanged (or a
connection failure returns 502). A second refusal is forwarded without another
retry. Stale pooled sockets, request write failures, response read/parse errors,
and truncated bodies are uncertain outcomes and never trigger replay. No request
is replayed after any response byte has been forwarded. The external harness
owns any further explicit follow-up/retry decision.
Stateful provider continuation handles remain opaque: compatible model metadata
does not establish that a `previous_response_id` or server-side session is shared
between replicas. Use full-history/stateless requests for portable failover;
affinity alone cannot make engine-local continuation state portable.

## Byte preservation and usage

Inbound Host must name `127.0.0.1`, `localhost`, `[::1]` or the listener's loopback
IP, or an `--allow-host` value; an explicit port must match the listener (else 421).
Any Origin, including the gateway's own, is rejected with 403 unless `--allow-origin`
lists it; duplicate Host or Origin headers are rejected with 400. POSTs
require `Content-Type: application/json` and exactly one bounded UTF-8 JSON
object; duplicate keys and trailing JSON values are rejected before dispatch.
Clients that previously omitted Content-Type must add it. Validated body bytes
remain unchanged on the upstream connection.

Request JSON is validated and forwarded byte-for-byte. The gateway does not
insert a cache key, reorder tools, repair dialects, admit requests on behalf of
clients or rewrite models. Capability/admission responses remain upstream payloads.
HTTP transfer framing and hop-by-hop headers are normalized as in the existing
relay; JSON/SSE payload bytes, unknown fields, errors and binary data are preserved.
Downstream write/flush finishes before the next upstream body read.

Cluster mode adds a passive usage tap after each successful downstream write.
It recognizes Chat `usage.prompt_tokens` and
`usage.prompt_tokens_details.cached_tokens`, plus Responses `usage.input_tokens`
and `usage.input_tokens_details.cached_tokens`, including nested completed SSE
responses. Identical snapshots count once. Conflicting, malformed, missing,
oversized or incomplete telemetry remains unknown. HTTP framing must complete,
status must be 2xx, and SSE must include `[DONE]` or `response.completed` with no
pending event data. Failed/incomplete response events suppress metrics. The tap
never rejects or modifies an upstream body because it cannot parse telemetry.

JSON telemetry has a 1 MiB observation limit; SSE reuses the existing bounded
1 MiB line/event decoder. This adds bounded scratch beyond the 8 KiB relay read
buffer, with no response queue or read-ahead. The 8 MiB request and 64 MiB response
bounds remain. Truncation or downstream delivery failure publishes no usage.

Per-endpoint metrics:

- `requests`: forwarded client dispatch attempts, including capability/admission
  requests; background discovery and the local stats route are excluded.
- `failures`: HTTP status >=400 or incomplete/failed transport or delivery,
  counted once per attempt. `probe_failures` is separate.
- `overloads`, `in_flight`, `healthy`, `cooling_down`, `group`: routing observations.
- `affinity_hits`: reused routing hints, **not** cache hits; `fallbacks` counts
  selections away from an existing unusable hint.
- `usage_responses`: completed generation responses with both valid token counts.
  `prompt_tokens` and `cached_tokens` sum only these paired observations.
- `cached_tokens_ratio`: sum(cached)/sum(prompt) across that same subset, or null
  when the denominator is zero. Missing cache usage is never treated as zero.
  Cached tokens still occupy logical context; nothing is subtracted from input.

## Local evidence and credit

Three mock engines maintain independent sets of previously seen prefixes and
report 100 prompt tokens, with 0 cached on first sight and 80 thereafter. Tests
send six conversations (three explicit keys, three inferred prefixes), two turns
each, using JSON and fragmented SSE. Each endpoint receives four requests and
reports 400 prompt / 160 cached tokens, ratio 0.4; every second turn reports 80
cached tokens. These are deterministic fixture counts, not measured Engine
performance or a production hit-rate prediction.

Additional cases cover local concurrency saturation, unchanged 429/503/529 payloads,
uncertain/truncated outcomes with no replay, changed and foreign capabilities,
TTL/eviction/domain invalidation, model isolation/ambiguity, Responses/admission,
opaque unparseable payloads, owner shutdown, scope separation and actual CLI
routing with three mocks. Existing single-upstream conformance still runs.

Design references are credited in [Credits](../../CREDITS.md#harness).
Implementation is first-party Rust using only the standard library; no competitor
code or tests were copied.
