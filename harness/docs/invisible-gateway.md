# Standalone gateway

`vade-gateway` sits between your existing client and the engine, so the client
only has to change its base URL. Build it from the checkout root with
`cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml --bins`,
as in [INSTALL.md](../../INSTALL.md#client-gateway); the binary ends up at
`harness/target/release/vade-gateway`.

## Run it

Run `vade-gateway --upstream HTTP_BASE_URL`. You can also give the upstream
through `VADE_GATEWAY_UPSTREAM`. Point your client's base URL at the gateway's
loopback listener on port 8787, keeping its `/v1` suffix and its original model.
`--bind` accepts another loopback address. The gateway speaks plain HTTP; TLS and
protocol upgrades need a separate transport.

Before it contacts any upstream, the gateway checks three things. A Host that is
not loopback with the listener port gets 421. Any `Origin` header not listed by
`--allow-origin` gets 403. And with `--token` (or `VADE_GATEWAY_TOKEN`), a
missing `Authorization: Bearer` gets 401. Prefer `VADE_GATEWAY_TOKEN`: other
local users can read a `--token` value in the process list. See the
[harness README](../README.md) for how to allow a browser client.

## What it forwards

This standalone binary forwards every HTTP path, including models, capabilities,
admission, Completions, Chat Completions and Responses. It does not translate
between APIs, add instructions or pick a model for you. With one upstream it
never retries a request. Capability negotiation only happens with the explicit
`--clamp-limits` option described below. The existing `vade-harness gateway`
command keeps its own separate behavior.

Use `--api-key-file PATH` or `VADE_API_KEY_FILE` to pick an engine credential;
the command option wins over the environment. The configured key replaces
incoming `Authorization` and `X-Api-Key` headers with an upstream bearer header,
on capability and admission probes and on replay too. Without a configured key,
client credential headers pass through unchanged, with one exception: under
`--token`, the gateway consumes the `Authorization` header and never forwards
it, so use `--api-key-file` to authenticate upstream. This credential choice
does not change response headers or bodies. Recording metadata leaves out keys.

When a client hangs up before its reply ends, the gateway writes one blank line
to the upstream, then closes that connection. HTTP servers ignore a blank line
between requests (RFC 9112, section 2.2), but the Vade engine drops a request
that bytes follow, so a stream still in the engine queue stops at once. With
several upstreams, `POST /v1/cancel` and `/v1/kv/release` go to every upstream
at the same time. The gateway waits at most 1.5 seconds and reads at most 64 KiB
from each one.

Request bodies are capped at 8 MiB on every route. A body over the cap, whether
by its declared length or by its chunked total, gets HTTP 413 with the byte
limit, and Messages routes use `request_too_large`. The whole request is checked
before an upstream connection is opened, so an oversized request is never partly
forwarded. The exact public
`/completions` alias picks `/v1/completions` upstream, including query strings
and a configured API prefix. Other custom paths stay opaque, and this alias does
not translate request or response schemas.

## Sampling defaults

JSON on Completions, Chat Completions, Responses and Messages gets
`temperature: 0` (greedy), `seed: 20260905`, `top_p: 1.0`, `top_k: 0`,
`repetition_penalty: 1.0`, `presence_penalty: 0.0` and `frequency_penalty: 0.0`
only when **none** of these sampling fields is present: `temperature`, `top_p`,
`top_k`, `min_p`, `typical_p`, `seed`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `logit_bias`, `n`, `best_of`, `sampling`,
`generation_config`. If any one of them is present, including null, all defaults
are skipped: the gateway forwards the body byte for byte and the engine's
defaults apply.

The client's original bytes decide routing, affinity, the accepted-input
boundary and the opt-in policy clamps. Sampling defaults are appended after
those decisions, only for the upstream dispatch and for recording the effective
body. A body of exactly 8 MiB is still accepted, and the dispatched body may be
larger than that client limit by the inserted defaults (and any opt-in policy
edits). Other bodies stay opaque, including non-JSON and compressed bodies.
Status lines and the spelling, whitespace and order of end-to-end headers
survive. Host, connection and transfer framing belong to each hop, and
connection-nominated headers and trailers are removed. Capabilities replies add
`X-Vade-Sampling-Defaults` with the exact defaults,
`X-Vade-Sampling-Defaults-When` with the condition, and `X-Vade-Sampling-Fields`
with the recognized fields.

## Streaming and connections

Replies are relayed through a single bounded read buffer that is flushed before
the next read, for SSE, binary data and engine errors alike. A truncated
upstream reply leaves the downstream framing incomplete. A client disconnect
closes the upstream during silent prefill as well as during generation. An
HTTP/1.0 upstream reply keeps its status line and uses close-delimited output.

At most 64 connections are active at once. Connect attempts share a five-second
budget, and generation waits use the existing one-hour HTTP lifetime. Each
client connection carries one request, and downstream clients can reconnect
normally. Loopback mock tests verify byte-identical streamed and nonstreamed
bodies, unchanged error status and headers, and delivery before the reply
finishes. They do not show live engine performance.

## Stats

`--stats FILE` appends one JSON object per request, failures included. Payloads
never go into this log. Missing measurements are `null`, and zero cached tokens
means an observed miss. JSON and the final SSE usage supply normalized prompt,
completion and cached counts, the cache reuse percentage, and prefill and
first-logits timing. Conflicting cache aliases or incomplete replies suppress
token counts. Responses usage may use input/output token names and nest timing
under usage.

`first_byte_ms` measures the first upstream response bytes, counted from when
the gateway accepted the request. `first_token_ms` measures the first generated
text, reasoning or function data flushed downstream, including streamed
Completions `choices[].text`, and leaves out empty, role-only and usage-only
events. These server-side observations leave out the client's initial TCP
connect and its own receiving and application delay. The decode rate is
`(completion_tokens - 1) / (last generated event - first generated event)`; JSON
replies, a single generated event and coalesced deltas may not give a usable
rate.

`added_first_token_ms` subtracts the engine's admission-to-first-logits time
from the gateway's first-token observation. It includes request transport,
differences at the queue boundary and return transport, so it is not an
isolated, network-free relay cost. `relay_overhead_ms_per_token` divides the
measured downstream framing, write and flush time by the completion tokens, and
includes downstream backpressure. `chunks` counts payload writes, `sse_events`
counts decoded events, and bytes count payload only. Logging happens after the
reply is forwarded. Without a log, observation is off.

Some clients stop reading right after the final model event. After that event,
the gateway allows one second to validate the rest of the upstream HTTP framing.
`complete` records a validated upstream completion, and `downstream_complete`
also reports whether the downstream framing finished. A fully validated upstream
reply keeps its usage and recording even when the client has already closed.
Truncated upstream framing still suppresses usage and keeps the recording out of
replay. Local busy refusals also produce a stats row, with unavailable request
fields left null.

## Record and replay

`--record DIR` creates private request directories. Each one holds the exact
`request.body`, the complete transfer-decoded `response.body` (including the
final SSE event), `timings.json` in the stats format, and a versioned
`meta.json`. Reply bytes stream straight to disk after they are forwarded; the
recorder never holds a whole reply in memory. New directories get owner-only
permissions, and files get owner read/write permissions. An existing DIR that
other users can open gets owner-only permissions at startup, with a note on
stderr. Only a
complete request that passed the Host, Origin and `--token` checks gets a
directory, so a refused request leaves nothing on disk. DIR holds at most
100,000 request directories, the replay limit. When it is full, the gateway
says so once on stderr and forwards later requests without recording them.
Recording failures are reported on stderr and forwarding carries on. Incomplete
recordings are left out of replay and do not count against its limit. Bodies
can contain session content, so keep recordings outside the tracked tree.

`vade-gateway replay DIR --upstream HTTP_BASE_URL --concurrency N --stats FILE`
replays complete recordings, prints a stats row per request, and can also append
the same row to a JSONL file. Each keyed session keeps its request order, and up
to N sessions run at the same time (1 through 64). Sessions are identified first
by existing prompt-cache keys, then by explicit session headers. Requests
without a key form one ordered session, so raising concurrency alone cannot run
them in parallel.

Recorded request bodies are sent byte for byte. Only content type and encoding,
accept and named session headers are saved; authorization and cookies are not.
Authenticated replay needs its own `--api-key-file` or `VADE_API_KEY_FILE`, and
credentials are never restored from saved headers. Replay rejects control
characters in allowed saved header values before it sends any request or creates
its stats file. Replay opens fresh connections and does not reproduce the
original think time between requests. It measures the endpoint you pick, which
may itself be the gateway. Stateful upstream response handles must still be
valid, or be replaced by a self-contained client workload. If standard output
closes or fails, replay stops queued work and exits with an error, though
requests already running may finish. Check the optional stats file before you
explicitly replay the remaining requests, since completed requests are not
rolled back.

## Optional optimizations

The sampling-default rule above always applies. The optimizations below are off
by default. Limit changes only touch Chat Completions and Responses requests.
The cache fields of `--stable-prefix` also go on Messages requests. Affinity
routing also applies to the metadata and admission routes.

- `--stable-prefix` adds a missing `prompt_cache_key` and `cache: true`.
  The engine reuses a prefix only for a request with one of them. Claude Code
  and OpenCode send neither, so without the flag each of their turns prefills
  the full prompt again. Explicit values, including `cache: false`, survive. Keys come from explicit
  session headers, Claude Code's `X-Claude-Code-Session-Id` among them, or else
  from a deterministic hash of the model, instructions, the Messages `system`
  field, tools, system/developer messages and the first user message. Turns that send
  the full history keep the same key; clients that send only new input must
  supply a stable key or session header. This non-secret hash is a scheduling
  hint, and the engine validates actual prefix reuse.
- `--affinity` routes a key to the same upstream with rendezvous hashing. Repeat
  `--upstream` for several servers. Reordering the list keeps placement, and
  adding a server moves only the keys assigned to that server. No model
  substitution happens. Configure only compatible upstreams; choosing them is
  your routing decision.
- `--clamp-limits` fetches each upstream's capabilities once at startup. It caps
  positive output budgets at the advertised output limit, and uses exact
  admission to shrink the reservation to the remaining context. A missing budget
  gets the bounded default. Final changed reservations are admitted again.
  Messages and tools keep their exact bytes. Input above `input_limit` passes
  through with sampling defaults so the engine handles the oversized input, and
  the numeric count and limit appear in `policy.oversized_input`. Admission
  failures are logged, and the engine's handling stays authoritative. Admission
  does not reserve resources or prevent concurrent state changes.
- `--greedy-when-unset` is still accepted for compatibility; the
  sampling-default presence rule already supplies `temperature: 0` only when no
  sampling field is present.

Stats include the selected upstream index, every changed budget with its before
and after values, inserted defaults, and oversized-input counts.
`policy.sampling_defaults_added` contains
`{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}`
exactly when these defaults were inserted, and `null` otherwise.
`policy.greedy_added` reports the same insertion as a boolean, even when the
opt-in policies are off. Recordings keep the original client bodies, and changed
requests also get an exact `upstream-request.body` file. Replay sends
`upstream-request.body` when it exists, otherwise `request.body`, so the exact
applied defaults and policy changes survive replay. Cache reuse is always the
actual restored/prompt tokens; neither routing nor a stable key proves a cache
hit.

## Replicas

Repeating `--upstream` without `--affinity` treats the upstreams as identical
replicas (full, separate copies of the same model), such as the one-Spark
servers that `vade up --replicas N` starts. Each request goes to the upstream
with the fewest requests in flight through this gateway, and ties rotate in
configuration order. Work sent straight to an engine is not counted. Requests
whose `messages` (or Responses `input` array) share their first 2048 bytes up to
the last user message form one conversation: a later turn prefers the upstream
that served the previous one while that upstream is among the least busy, and
otherwise load wins. The bounded table keeps hashes and positions only.
`--no-sticky` routes every request by load.

With several upstreams, the gateway answers `GET /health` itself. It returns the
first ready replica's engine fields (`model`, `vade`, `caps`) with `active` and
`queued` summed, and an `upstreams` array with each position's status, HTTP
code, model and load. Status is `ready` (HTTP 200) when every upstream is ready,
`degraded` or `unavailable` (HTTP 503) otherwise, and `error` (HTTP 502) when
model IDs, builds, PTX or checkpoint hashes differ.

A request only moves to another upstream before any reply byte reaches the
client, and only for a connection failure, a failure while sending the request
or reading the reply head, or an HTTP 503 reply. Each upstream is tried at most
once. Timeouts and client disconnects are never retried, and once a reply byte
has been forwarded, the request is not replayed. Identical replicas answer a
greedy request with the same token IDs, so routing does not change answers;
check this with `vade verify` against each replica.

With several upstreams, `POST /v1/cancel` and `POST /v1/kv/release` are not
routed by load. The gateway does not record which replica holds a key, so it
sends each call to every upstream in order. The client gets the first 2xx reply
that reports `"cancelled":true` or `"released":true`, otherwise the first reply,
or HTTP 502 when no upstream answers. Keys are random capability tokens, so an
unknown key changes nothing on the other replicas.
