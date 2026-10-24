# Standalone gateway

Build `vade-gateway` from the checkout root with
`cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml --bins`,
as in [INSTALL.md](../../INSTALL.md#client-gateway); the binary is
`harness/target/release/vade-gateway`.
Run `vade-gateway --upstream HTTP_BASE_URL`. The upstream may also come from
`VADE_GATEWAY_UPSTREAM`. Point the existing client's base URL at the gateway's
loopback listener on port 8787, retaining its `/v1` suffix and original model.
`--bind` accepts another loopback address. HTTP is supported; TLS and protocol
upgrades require a separate transport.
Before any upstream contact, a Host that is not loopback with the listener port
gets 421, any `Origin` header not listed by `--allow-origin` gets 403, and with
`--token` (or `VADE_GATEWAY_TOKEN`) a missing `Authorization: Bearer` gets 401;
see the [harness README](../README.md) for allowing a browser client.

This standalone binary forwards every HTTP path, including models, capabilities,
admission, Completions, Chat Completions and Responses. It does not translate APIs,
insert instructions or select a model. With one upstream it never retries a request.
Capability negotiation
requires the explicit `--clamp-limits` option described below.
The existing `vade-harness gateway` command retains its separate behavior.

Use `--api-key-file PATH` or `VADE_API_KEY_FILE` to select an engine credential;
the command option overrides the environment. The configured key replaces
incoming `Authorization` and `X-Api-Key` headers with an upstream bearer header,
including on capability/admission probes and replay. Without a configured key,
client credential headers pass through unchanged, except that under `--token`
the gateway consumes the `Authorization` header and never forwards it; use
`--api-key-file` to authenticate upstream. Response headers and bodies
are not changed by this credential selection. Recording metadata excludes keys.

Request bodies are bounded at 8 MiB on every route. Declared and cumulative
chunked overflow receive HTTP 413 with the byte limit; Messages routes use
`request_too_large`. The whole request is checked before opening an upstream
connection, so an oversized request is never partially forwarded.
The exact public `/completions` alias selects `/v1/completions` upstream,
including query strings and a configured API prefix. Other custom paths remain
opaque; this alias does not translate request or response schemas.

JSON on Completions, Chat Completions, Responses and Messages receives `temperature: 0`
(greedy), `seed: 20260905`, `top_p: 1.0`, `top_k: 0`, `repetition_penalty: 1.0`,
`presence_penalty: 0.0`, and `frequency_penalty: 0.0` only when **none** of these sampling
fields is present: `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `seed`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`, `n`,
`best_of`, `sampling`, `generation_config`. Any one present, including null,
suppresses all defaults: sampling normalization forwards the body byte for
byte and engine defaults apply. The original client bytes determine routing,
affinity, the accepted-input boundary, and opt-in policy clamps. Sampling defaults
are appended after those decisions, only for upstream dispatch and recording the
effective body. A body exactly at 8 MiB remains accepted; the dispatch body may
exceed that client limit by the inserted defaults (and any opt-in policy edits).
Other bodies remain opaque, including non-JSON and compressed bodies. Status lines and end-to-end header spelling, whitespace and
order survive. Host, connection and transfer framing belong to each hop;
connection-nominated headers and trailers are removed. Capabilities replies add `X-Vade-Sampling-Defaults` with the exact defaults,
`X-Vade-Sampling-Defaults-When` with the condition, and `X-Vade-Sampling-Fields`
with the recognized fields. Replies are relayed with a single bounded read buffer, flushed
before the next read, including SSE, binary data and engine errors. A truncated
upstream reply leaves downstream framing incomplete. A client disconnect closes
the upstream during silent prefill as well as during generation. An HTTP/1.0
upstream reply retains its status line and uses close-delimited output.

At most 64 connections are active. Connect attempts have a shared five-second
budget; generation waits use the existing one-hour HTTP lifetime. Each client
connection carries one request; downstream clients may reconnect normally.
Loopback mock tests verify byte-identical streamed and nonstreamed bodies,
unchanged error status and headers, and delivery before the reply finishes.
They do not establish live engine performance.

`--stats FILE` appends one JSON object per request, including failures. Payloads
are never included in this log. Missing measurements are `null`; zero cached
tokens means an observed miss. JSON and final SSE usage supply normalized
prompt/completion/cached counts, cache reuse percentage, prefill and first-logits
timing. Conflicting cache aliases or incomplete replies suppress token counts.
Responses usage may use input/output token names and nest timing under usage.

`first_byte_ms` measures first upstream response bytes from gateway acceptance.
`first_token_ms` measures the first generated text, reasoning or function data
flushed downstream, including streamed Completions `choices[].text` and excluding
empty, role-only and usage-only events. These server-side observations
exclude the client's initial TCP connect and receiving/application delay. Decode
rate is `(completion_tokens - 1) / (last generated event - first generated event)`;
JSON replies, one generated event and coalesced deltas may lack a usable rate.
`added_first_token_ms` subtracts engine admission-to-first-logits time from the
gateway's first-token observation. It includes request transport, queue boundary
differences and return transport; it is not an isolated network-free relay cost.
`relay_overhead_ms_per_token` divides measured downstream framing/write/flush
time by completion tokens; it includes downstream backpressure. `chunks` counts
payload writes, `sse_events` counts decoded events, and bytes count payload only.
Logging occurs after reply forwarding. Observation is disabled without a log.

Some clients stop reading immediately after the final model event. After that
event, the gateway allows one second to validate the remaining upstream HTTP
framing. `complete` records validated upstream completion; `downstream_complete`
also reports whether downstream framing finished. A fully validated upstream
reply retains its usage and recording when the client has already closed.
Truncated upstream framing still suppresses usage and excludes the recording
from replay. Local busy refusals also produce a stats row; unavailable request
fields remain null.

`--record DIR` creates private request directories. Each contains the exact
`request.body`, the complete transfer-decoded `response.body` (including the final
SSE event), `timings.json` in the stats format, and versioned `meta.json`. Reply
bytes stream directly to disk after being forwarded; the recorder does not
accumulate a whole reply in memory. New directories use owner-only permissions
and files use owner read/write permissions. Recording failures are reported to
stderr and forwarding continues. Incomplete recordings are excluded from replay.
Bodies can contain session content; keep recordings outside the tracked tree.

`vade-gateway replay DIR --upstream HTTP_BASE_URL --concurrency N --stats FILE`
replays complete recordings, printing a stats row per request and optionally
appending the same row to a JSONL file. Each keyed session retains request order;
up to N sessions run concurrently (1 through 64). Existing prompt-cache keys,
then explicit session headers, identify sessions. Unkeyed requests form one
ordered session, so increasing concurrency alone cannot parallelize them.
Recorded request bodies are sent byte for byte. Only content type/encoding,
accept and named session headers are persisted. Authorization and cookies are
excluded. Authenticated replay needs its own `--api-key-file` or
`VADE_API_KEY_FILE`; credentials are never restored from saved headers. Replay
rejects control characters in allowed saved header values before dispatching
any request or creating its stats file. Replay opens fresh connections
and does not reproduce original think time between requests. It measures the
selected endpoint, which may itself be the gateway. Stateful upstream response
handles must remain valid or be replaced by a self-contained client workload.
If standard output closes or fails, replay stops queued work and exits with an
error; requests already running may finish. Inspect the optional stats file
before explicitly replaying remaining requests, since completed requests are
not rolled back.

The sampling-default presence rule always applies; the following optimizations
default off. Request decoration and limit changes affect only Chat Completions
and Responses; affinity also routes metadata and admission:

- `--stable-prefix` adds a missing `prompt_cache_key` and `cache: true`. Explicit
  values, including `cache: false`, survive. Keys use explicit session headers,
  otherwise a deterministic hash of model, instructions, tools, system/developer
  messages and the first user message. Full-history turns retain the same key;
  clients sending only new input must supply a stable key/session header. This
  non-secret hash is a scheduling hint; the engine validates actual prefix reuse.
- `--affinity` routes a key to the same upstream with rendezvous hashing. Repeat
  `--upstream` for several servers. Reordering the list preserves placement;
  adding a server moves only keys assigned to that server. No model substitution
  occurs. Configure compatible upstreams; this is an operator routing decision.
- `--clamp-limits` fetches each upstream's capabilities once at startup. It caps
  positive output budgets at the advertised output limit and uses exact
  admission to reduce the reservation to remaining context. A missing budget
  receives the bounded default. Final changed reservations are admitted again.
  Messages and tools retain their exact bytes. Input above `input_limit` passes
  through with sampling defaults so the engine handles the oversized input;
  the numeric count and limit appear in `policy.oversized_input`. Admission
  failures are logged and engine handling remains authoritative. Admission does
  not reserve resources or prevent concurrent-state changes.
- `--greedy-when-unset` remains accepted for compatibility; the sampling-default
  presence rule already supplies `temperature: 0` only when no sampling field is present.

## Replicas

Repeating `--upstream` without `--affinity` treats the upstreams as identical
replicas, such as the one-Spark servers started by `vade up --replicas N`. Each
request goes to the upstream with the fewest requests in flight through this
gateway; ties rotate in configuration order. Work sent to an engine directly is
not counted. Requests whose `messages` (or Responses `input` array) share their
first 2048 bytes through the last user message form one conversation: a later
turn prefers the upstream that served the previous one while that upstream is
among the least busy; otherwise load wins. The bounded table
keeps hashes and positions only. `--no-sticky` routes every request by load.

With several upstreams `GET /health` is answered by the gateway itself. It
returns the first ready replica's engine fields (`model`, `vade`, `caps`) with
`active` and `queued` summed, and an `upstreams` array with each position's
status, HTTP code, model and load. Status is `ready` (HTTP 200) when every
upstream is ready, `degraded` or `unavailable` (HTTP 503) otherwise, and `error`
(HTTP 502) when model IDs, builds, PTX or checkpoint hashes differ.

A request moves to another upstream only before any reply byte reaches the
client: a connection failure, a failure while sending the request or reading the
reply head, or an HTTP 503 reply. Each upstream is tried at most once, timeouts
and client disconnects are never retried, and once a reply byte is forwarded the
request is not replayed. Identical replicas answer a greedy request with the
same token IDs, so routing does not change answers; check this with
`vade verify` against each replica.

With several upstreams `POST /v1/cancel` and `POST /v1/kv/release` are not
routed by load: the gateway does not record which replica holds a key, so it
sends each call to every upstream in order. The client gets the first 2xx reply
that reports `"cancelled":true` or `"released":true`, otherwise the first reply,
or HTTP 502 when no upstream answers. Keys are random capabilities, so an
unknown key changes nothing on the other replicas.

Stats include selected upstream index, every changed budget with before/after
values, inserted defaults, and oversized-input counts. `policy.sampling_defaults_added`
contains `{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}` exactly when these defaults
were inserted; otherwise it is `null`. `policy.greedy_added` reports the same
insertion as a boolean, including when opt-in policies are disabled. Recordings
retain original client bodies; changed requests also get an exact `upstream-request.body` file.
Replay sends `upstream-request.body` when present, otherwise `request.body`,
so the exact applied defaults and policy changes survive replay. Cache reuse is always actual restored/prompt tokens;
neither routing nor a stable key establishes a cache hit.
