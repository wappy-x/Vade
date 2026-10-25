# Cache-aware gateway routing

This module lets the harness gateway spread requests from external harnesses
across several endpoints (cluster routing). We wrote it in Rust using only the
standard library. Inference, tokenization, scheduling and KV stay with the
upstream. We verified it with three independent loopback mocks on macOS ARM64.
It was never run against Vade Engine, Sparks, GPUs, installed agents or live
providers.

## Use

With one `--upstream`, you keep the existing opaque relay to a single endpoint,
with no background discovery. Repeat it to turn on cluster routing, which checks
each endpoint's capabilities first:

```sh
vade-harness gateway --bind 127.0.0.1:8787 \
  --upstream http://127.0.0.1:18081 \
  --upstream http://127.0.0.1:18082 \
  --upstream http://127.0.0.1:18083
```

These are example addresses for mocks you provide separately. Running the
command contacts those configured endpoints; it does not start any mocks.
Configure your external client to use the gateway's text Completions
(`/v1/completions` or `/completions`), Chat Completions, Responses, or Anthropic
Messages route (including `/v1/messages/count_tokens`), together with its
upstream's model. Nothing is added to the client's prompt or request body. Every
configured endpoint must be allowed to receive the end-to-end headers that
client forwards, including any authorization it supplies.

`GET /_vade/gateway/stats` returns local JSON metrics in cluster mode, and never
forwards to an upstream. IDs are zero-based `--upstream` positions, not
hostnames. The `Cluster::stats()` library handle exposes the same snapshot. No
body, credential, client cache key, routing hash, model name or endpoint URL is
logged or included in these metrics. Metrics and affinity live in memory only.

The library API is `Cluster::new(&[String], RoutingConfig)` followed by
`run_cluster(bind, cluster)` or `run_cluster_until(bind, cluster, stop)`.
Construction checks the configuration without any network I/O. Clones share
routing state and metrics, so run only one gateway per cluster. `RoutingConfig`
allows bounded overrides of the defaults. The CLI uses these defaults:

| Setting | Default |
| --- | --- |
| Configured endpoints | 1–16 (library); repeating `--upstream` selects cluster mode |
| Affinity entries | 4,096, maximum configurable 65,536 |
| Affinity idle TTL | 15 minutes, maximum configurable 24 hours |
| In-flight requests per endpoint | 8, configurable 1–16 |
| Total gateway client connections | 16 |
| Capability probe interval | 10 seconds after previous probe completes |
| Capability freshness | 30 seconds |
| Probe deadline / body | 5 seconds / 64 KiB |
| Failure / overload cooldown | 5 seconds |
| Upstream idle connection pool | 16 sockets total, 15-second idle expiry |

The bind address stays loopback-only. TLS, credentials embedded in URLs,
ambiguous HTTP framing, compressed representations and upgrades fail explicitly.
Cluster URLs must be known API suffixes or `/v1` base paths; custom exact paths
are still available in single-upstream mode. Cluster URLs cannot contain query
strings, but the caller's query is still relayed and is part of its routing
scope.

## Compatibility and discovery

Each configured endpoint gets its own bounded background discovery worker for
`GET /v1/capabilities` (or the matching sibling under a configured base path).
Probes carry no caller body, affinity key or caller credentials, and they follow
no advertised URLs or redirects. When an upstream API key is configured, probes
use it as a Bearer credential, through the same transport as generation.
Authenticated capabilities are supported; if the server requires authentication,
configure an upstream key that is allowed to read them.

All initial probes must finish or fail before generation routing starts, and
clients get 503 during that time. One hung probe does not hold up the other
probes, the listener or the stats route. Shutting the gateway down cancels and
joins the discovery workers. Socket cancellation polls at 250 ms; an OS DNS
lookup through the standard library cannot be interrupted.

Only recognized `vade.engine.v1` capabilities with complete weights, tokenizer
and template fingerprints can join a group. For two endpoints to be in the same
group, these must also match: model name, weights fingerprint kind, the
advertised Completions/Chat/Responses/Messages/token-count routes, token limits,
tool and thinking flags and defaults, cache scope and features, and admission
support. Unknown extension fields are ignored. Fingerprints are opaque
advertised identities; the harness makes no independent claim about weight
integrity.

Instance `cache_domain` is deliberately left out of group equality, because
replicas have different KV instances. A changed domain invalidates that
endpoint's old affinity hints. A failed probe invalidates its hints and health.
The first valid compatibility signature for each endpoint is pinned for the
lifetime of the gateway. A later change of fingerprint, model or contract
quarantines the endpoint until the original signature comes back or you
explicitly restart or reconfigure the gateway. Expired capabilities also make an
endpoint unavailable. Selection and discovery are not an atomic transaction with
the upstream, so changes between probes, or between selection and dispatch, can
still happen.

Different model names can form separate groups. A request's `model` picks its
group, and admission uses `request.model`. If the same model name has several
incompatible groups, routing returns 409 instead of picking a fingerprint, and
that ambiguity remains even when one of the groups is unhealthy. A request
without a model, capability discovery included, needs exactly one discovered
group. So for clients that fetch capabilities first, the simplest setup is a
separate gateway per model. A route the group does not advertise in `endpoints`
(including `/v1/completions`), or an unavailable group, returns 503. Unknown or
foreign capability contracts are never treated as compatible.

## Affinity and fallback

Affinity means sending related requests to the same endpoint, so the engine
there has a chance to reuse what it already computed. Within a compatible group,
the router hashes a decoded client `prompt_cache_key` when it is a JSON string
(empty included). Otherwise it hashes the leading **original wire bytes** from
`system`, `instructions`, `tools`, and up to 17 leading messages, stopping after
the first message whose role is neither system nor developer. Each selected
value contributes at most 16 KiB. A scalar Responses `input` uses its first 16
KiB, and a text Completions `prompt` uses its first 16 KiB of wire bytes.
Appending later messages or prompt text beyond that prefix does not change this
hint. Edits to tools or the system text, different serialization, or long shared
or truncated prefixes can change or alias the inferred affinity, so a stable
explicit key is better.

Keys also include the API family (Completions, chat, Responses or Messages),
the query, and the forwarded Authorization, OpenAI-Organization and
OpenAI-Project header values. For admission, the family is the one named in its
target `endpoint`. The
process-randomized standard-library hash is only a scheduling hint, and
collisions can happen. It is not cryptography, an authorization boundary, a
durable identifier, or a claim that two prompts have the same tokens. The
bounded table keeps only hashes and endpoint/expiry metadata, not request text
or credentials. Upstream authorization and cache isolation remain the upstream's
job.

If the endpoint that affinity ties the request to is healthy and not full, it
wins. Otherwise the router atomically reserves the endpoint in the same group
with the fewest **gateway in-flight requests**, and ties take turns. This does
not see work sent directly to the engines, and it does not measure GPU or KV
use. Reservations last until the response has been delivered, and are released
on cancellation or failure too. New affinity is recorded at selection, so
related requests arriving at the same time can use it. The TTL refreshes on use.
Expired entries are removed on routing and stats reads, and the least recently
used entry is evicted when the table is full. There is no response cache or
independent KV store.

A transport failure makes the endpoint unavailable until another probe succeeds.
HTTP 429 or any 5xx starts a cooldown, and 429/503/529 also increment
`overloads`. A numeric `Retry-After` uses 1–300 seconds; dates and invalid
values use the default. A health probe cannot erase an overload cooldown or a
newer transport failure. When every compatible endpoint is unavailable or
saturated, the gateway returns 503.

**At most one refusal retry happens before response forwarding.** A fresh DNS or
connection failure, or an HTTP 429/503/529 refusal before any downstream headers
or body are sent, can retry the exact request once on another verified healthy,
compatible endpoint. The failed endpoint is excluded even if a health probe
races the retry. If there is no alternative, the original refusal is relayed
unchanged (or a connection failure returns 502). A second refusal is forwarded
without another retry. Stale pooled sockets, request write failures, response
read or parse errors, and truncated bodies are uncertain outcomes and never
trigger a replay. No request is replayed after any response byte has been
forwarded. The external harness owns any further explicit follow-up or retry
decision.

Stateful provider continuation handles stay opaque: matching model metadata does
not mean a `previous_response_id` or a server-side session is shared between
replicas. Use full-history, stateless requests for failover that works across
endpoints; affinity alone cannot make continuation state that lives in one
engine portable.

## Byte preservation and usage

The inbound Host must name `127.0.0.1`, `localhost`, `[::1]` or the listener's
loopback IP, or an `--allow-host` value, and an explicit port must match the
listener (otherwise 421). Any Origin, including the gateway's own, is rejected
with 403 unless `--allow-origin` lists it, and duplicate Host or Origin headers
are rejected with 400. POSTs need `Content-Type: application/json` and exactly
one bounded UTF-8 JSON object; duplicate keys and trailing JSON values are
rejected before dispatch. Clients that used to leave out Content-Type must add
it. Validated body bytes go to the upstream connection unchanged.

Request JSON is validated and forwarded byte for byte. The gateway does not
insert a cache key, reorder tools, repair dialects, admit requests on the
client's behalf or rewrite models. Capability and admission responses stay
upstream payloads. HTTP transfer framing and hop-by-hop headers are normalized
as in the existing relay, while JSON/SSE payload bytes, unknown fields, errors
and binary data are kept as they are. The downstream write and flush finish
before the next upstream body read.

Cluster mode adds a passive usage tap after each successful downstream write. It
recognizes Chat `usage.prompt_tokens` and
`usage.prompt_tokens_details.cached_tokens`, plus Responses `usage.input_tokens`
and `usage.input_tokens_details.cached_tokens`, including nested completed SSE
responses. Identical snapshots count once. Conflicting, malformed, missing,
oversized or incomplete telemetry stays unknown. HTTP framing must complete, the
status must be 2xx, and SSE must include `[DONE]` or `response.completed` with
no pending event data. Failed or incomplete response events suppress metrics.
The tap never rejects or changes an upstream body because it cannot parse the
telemetry.

JSON telemetry has a 1 MiB observation limit, and SSE reuses the existing
bounded 1 MiB line/event decoder. This adds bounded scratch space beyond the 8
KiB relay read buffer, with no response queue or read-ahead. The 8 MiB request
and 64 MiB response bounds remain. Truncation or a failed downstream delivery
publishes no usage.

Per-endpoint metrics:

- `requests`: forwarded client dispatch attempts, capability and admission
  requests included; background discovery and the local stats route are not
  counted.
- `failures`: HTTP status >=400, or incomplete or failed transport or delivery,
  counted once per attempt. `probe_failures` is separate.
- `overloads`, `in_flight`, `healthy`, `cooling_down`, `group`: routing
  observations.
- `affinity_hits`: reused routing hints, **not** cache hits; `fallbacks` counts
  selections that moved away from an existing hint that could not be used.
- `usage_responses`: completed generation responses with both valid token
  counts. `prompt_tokens` and `cached_tokens` sum only these paired
  observations.
- `cached_tokens_ratio`: sum(cached)/sum(prompt) over that same subset, or null
  when the denominator is zero. Missing cache usage is never treated as zero.
  Cached tokens still take up logical context, and nothing is subtracted from
  input.

## Local evidence and credit

Three mock engines each keep their own set of prefixes they have already seen,
and report 100 prompt tokens, with 0 cached the first time and 80 after that.
Tests send six conversations (three with explicit keys, three with inferred
prefixes), two turns each, using JSON and fragmented SSE. Each endpoint gets
four requests and reports 400 prompt / 160 cached tokens, a ratio of 0.4, and
every second turn reports 80 cached tokens. These are deterministic fixture
counts, not measured engine performance or a prediction of production hit rates.

Further cases cover local concurrency saturation, unchanged 429/503/529
payloads, uncertain or truncated outcomes with no replay, changed and foreign
capabilities, TTL, eviction and domain invalidation, model isolation and
ambiguity, Responses and admission, opaque unparseable payloads, gateway
shutdown, scope separation, and real CLI routing with three mocks. The existing
single-upstream conformance tests still run.

Design references are credited in [Credits](../../CREDITS.md#harness). The
implementation is first-party Rust using only the standard library; no code
or tests from competing projects were copied.
