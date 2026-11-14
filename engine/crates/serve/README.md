# spark-serve

**Vade's chat server. The program `spark-serve` serves one model on one
Spark through an OpenAI-compatible HTTP API, so ordinary chat apps and coding
tools can use it.** It uses only Vade's own crates and the Rust standard
library.

This page is for people reading or changing the server code. To run it,
follow [Install](../../docs/install.md). Every flag, request field and limit
is in [Server and API](../../docs/server.md).

<details>
<summary><b>Terms on this page</b>: prefill, decode, KV cache, prefix reuse, SSE, draft decoding, slot</summary>

<dl>
<dt>Prefill</dt>
<dd>Reading the prompt, before the first answer token.</dd>
<dt>Decode</dt>
<dd>Producing the answer, one token per step.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the tokens it has read.</dd>
<dt>Prefix reuse</dt>
<dd>Skipping the part of a prompt the server has already read, by reusing its KV
cache.</dd>
<dt>SSE</dt>
<dd>Server-sent events: the answer streams back as a series of
<code>data:</code> lines.</dd>
<dt>Draft decoding</dt>
<dd>Guessing several next tokens cheaply, then letting the model check them all
in one pass. Also called speculative decoding.</dd>
<dt>Slot</dt>
<dd>One place for an active chat. Slots share one memory pool.</dd>
</dl>

</details>

## What it serves today

**Six official models, each at its maker's precision. Whether it drafts
(guesses tokens ahead) depends on the model.**

| Model | Weights | Drafting |
| --- | --- | --- |
| Qwen3-8B, Qwen3-32B | BF16 | Qwen3-0.6B |
| Qwen3-30B-A3B | BF16 | Off |
| Qwen3.8-27B | BF16 | Sibling or MTP |
| Qwen3.8-27B-FP8 | Official FP8 | Sibling |
| Gemma 4 31B | BF16, text only | Assistant |

- **Qwen3-8B and Qwen3-32B** draft with the official Qwen3-0.6B draft model,
  when it is installed.
- **Qwen3-30B-A3B does not draft:** checking guesses on the expert model was
  slower.
- **Qwen3.8-27B** drafts with the official sibling draft or the built-in MTP
  (multi-token prediction) head, for one active chat.
- **Qwen3.8-27B-FP8**, the official FP8 release with BF16 parts, drafts with
  the sibling for one active chat. Its built-in FP8 draft does not start yet.
- **Gemma 4 31B** drafts with Google's official assistant, for one active
  chat.

[Models](../../docs/models.md) has the supported scope and download steps.

## The promise this code keeps

**Same answer every time, checked against the official model.**

- **Same answer every time.** A prompt gives exactly the same tokens alone or
  batched with other chats, and cached or read fresh.
- **Checked against the official model.** Agreement is measured and not
  perfect; [Validation](../../docs/validation.md) has every figure.
- **Draft decoding never changes the answer** in `auto` and `strict`. The
  experimental `fast` policy can, and is off by default.

## Where it fits

**The server schedules the work; the engine runs the model.**

<dl>
<dt><a href="../engine/README.md">spark-engine</a></dt>
<dd>Runs the model. The server schedules work onto it.</dd>
<dt><a href="../tokenizer/README.md">engine-tokenizer</a></dt>
<dd>Turns chats into token IDs and back.</dd>
<dt><a href="../model/README.md">engine-model</a></dt>
<dd>Reads model folders and configs.</dd>
<dt><a href="../gpu/README.md">spark-gpu-probe</a>,
<a href="../seqpar/README.md">spark-seqpar</a></dt>
<dd>Linux-only dependencies for the GPU and long-prompt helpers.</dd>
<dt><a href="../tp/README.md">spark-tp</a></dt>
<dd><code>tp-serve</code> builds on this crate's library, with its own
multi-Spark engine and smaller feature set.</dd>
</dl>

## Build and start

**Build the kernels, then the server, then wait for the `ready` line.**

### Do it yourself

1. Build the GPU kernels, then the server. The server carries its own copy
   of the kernels.

   ```sh
   # Run from the checkout root.
   cargo kernels
   cargo +1.99.0 build --release -p spark-serve
   ```

2. Start it with a model folder.

   ```sh
   ./target/release/spark-serve models/Qwen/Qwen3-8B
   ```

3. Wait for the line starting with `ready`. It lists every limit the server
   chose and the address it listens on, after `bind=`. The default is
   loopback, port 8080.

4. In another terminal, check that it answers.

   ```sh
   curl -s http://localhost:8080/health
   ```

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, run `cargo kernels`, then
`cargo +1.99.0 build --release -p spark-serve`. Start
`./target/release/spark-serve models/Qwen/Qwen3-8B` and wait for the line
starting with "ready". Then run `curl -s http://localhost:8080/health` and
report the ready line and the health reply. Never use sudo, change the
network or listen beyond loopback. Do not commit or push.
```

## Checks

**Unit tests need no GPU. Live checks run against a started server:
branching, prefix reuse and drafting must match fresh runs, and cancellation
must work.**

### Do it yourself

1. Run the unit tests. They use real local sockets with stand-in engines
   (the shipped program always runs the real GPU engine), so they check the
   HTTP and scheduling code, not GPU speed. On the Spark, build the kernels
   first.

   ```sh
   cargo +1.99.0 test --workspace --exclude spark-rust-kernels --offline
   ```

2. Start `spark-serve`. Set `BIND` to the address shown after `bind=` on its
   `ready` line.

3. Run the live checks. Each comment says what the command proves.

   ```sh
   # Branching matches fresh runs, a main chat stays fast during
   # background prefill, and cancellation works.
   ./target/release/serve-http-check "$BIND" 512

   # The same, with draft decoding.
   VADE_CHECK_DRAFT=strict ./target/release/serve-http-check "$BIND" 512

   # Automatic prefix reuse gives the same tokens as "cache": false.
   # The check reads restored rows from final usage.
   VADE_CHECK_AUTOMATIC=1 ./target/release/serve-http-check "$BIND" 32 \
     fixtures/qwen3/short.json

   # plain, strict and fast stream the same text.
   ./target/release/serve-draft-check "$BIND" PROMPT_DIR
   ```

`serve-draft-check` reads `code.ids`, `json.ids`, `quotes.ids` and
`prose.ids` from `PROMPT_DIR`. Each holds token IDs separated by spaces or
line breaks. The `tokenize` example in the
[tokenizer README](../tokenizer/README.md#test-it) prints the IDs for any
text; copy them into each file without the brackets and commas.

`serve-http-check` compares emitted token IDs. It reads the restored prefix
count from final usage. A recurrent backend can restore a checkpoint below
the prompt end. A cache miss prints `SKIP` for cache coverage after token
equality passes. Use a longer prompt fixture to check actual cache reuse.
A `SKIP` line does not establish cached exactness.
The checker prints each request's prompt IDs and each reply's token IDs.
It omits session capability values from this output.
Automatic reuse can select a longer prefix than the named session provides.
For example, a prior plain request can cache a later strict request's prefix.
Draft mode does not give each request a separate cache scope.
The checker bounds reuse by the request length, since replies do not identify
the selected snapshot. It reports the named parent's shared prefix separately.
`named_parent_source=UNPROVEN` means that reuse does not prove which snapshot
the server selected. A token-ID difference still fails at its first index.
The last `CACHE_COVERAGE` line gives the hit and miss counts for the checked
comparisons. A miss still passes the token comparison when the IDs match.
Use `serve-http-check --health ADDRESS` to save the server's health JSON.

**Reference check, with the server stopped.**
`qwen-reference-check PTX MODEL FIXTURE.json...` compares chat tokenization
and logits with the [Hugging Face reference fixtures](../../fixtures/qwen3/README.md).
It needs the GPU to itself, so
[stop the server](../../docs/server.md#stopping) first.

To test a kernel file without rebuilding the server, pass it before the model
folder: `spark-serve KERNELS.ptx MODEL_FOLDER`.

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, with spark-serve already running on
loopback, read the address after bind= on its ready line and use it as BIND.
Run `./target/release/serve-http-check "$BIND" 512`, then the same with
`VADE_CHECK_DRAFT=strict`, as listed in crates/serve/README.md. Report every
PASS, SKIP, FAIL and CACHE_COVERAGE line. Do not restart services, change
settings, commit or push.
```

## How the code works

**One GPU thread does all model work. Each turn, many chats get their next
token in one pass.**

- **Sharing the GPU.** Long prompts are read in slices between turns, so
  other chats keep streaming.
- **Reusing earlier work.** On by default: a prompt's already-read beginning
  is skipped. Matches are checked token by token, never by hash alone.
- **Draft decoding.** The model checks every guess and keeps only the ones it
  agrees with, so in `auto` and `strict` the answer equals `plain`.
- **Memory.** Planned once, at start. Each request reserves its prompt plus
  `max_tokens`, or waits. [BUDGETS.md](BUDGETS.md) explains admission.

<details>
<summary><b>Sharing the GPU</b>: priority, batching, fairness and row budgets</summary>

All model work runs on one GPU thread. Sockets never block it.

1. **Priority.** Higher runs first. Ordinary clients are capped at 0 by
   default; `--client-priority-cap` can raise their cap to 254. An owner key
   grants priority 255. With that key and at least three active slots, one
   slot and memory for a short request are kept for the owner.
2. **Batching.** Each turn, the next token for many chats is computed in one
   pass. Their output is sent before any long prompt continues.
3. **Fairness.** The most urgent chat decodes every turn. Others decode every
   one to four turns, depending on how long their conversation is.
4. **Long prompts in slices.** A long prompt is read in short slices between
   turns, so other chats keep streaming. While other work waits, a
   high-priority prompt uses slices of about 15 ms and a background prompt
   about 100 ms.
5. **Short prompts together.** For Qwen3, prompts of up to 256 new tokens
   that arrive within 2 ms are read in one shared pass: up to 2,048 rows, or
   256 while a foreground chat decodes. Cached suffixes use the same window.
   A long prompt's short tail can join once its current chunk commits.
6. **Shared beginnings.** Requests waiting for the same prefix let its owner
   finish wide chunks without pausing between layers, until unrelated work
   needs a turn.

Row budgets per step:

| Setting | Default | Range |
| --- | ---: | --- |
| `--foreground-rows` | 8 | 1 to 512 |
| `--background-rows` | 32, 48 or 64 | 1 to 512 |

- `--background-rows` defaults to 48 for Qwen3-8B with a draft model and 64
  for Qwen3-32B with a draft model; otherwise 32, including expert targets.
- Automatic background rows are capped by active slots times `K+1`, where
  `K` is the number of guesses per step.

`--owner-key-file` reads a file with permissions 0600 containing 64 lowercase
hex digits. Requests sending that key as `Authorization: Bearer KEY` get owner
priority. Other clients still need no key unless you also set
`--api-key-file`.

</details>

<details>
<summary><b>Reusing earlier work</b>, and shared memory</summary>

The flags, request fields and per-key scope are in
[Server and API](../../docs/server.md#prompt-caching).

- Requests that share a beginning while it is still being read wait for it,
  then share it.
- Tokens the model generated are read again on the next turn. Their cached
  form can differ slightly from a fresh read, so the server does not reuse
  it.
- The cache holds at most 128 entries. An entry that no request used for
  the server's idle time is dropped, or sooner when memory runs short: five
  minutes for `spark-serve`, `tp-serve` and the two-Spark GLM text service,
  one hour for the `flash-serve` Flash services (FP4-Local and FP8). The time
  is fixed in each server's code. A model swap clears all entries.
- A cache scope keeps prompts apart, not the cache's room. The room is one
  pool for all callers: a prompt one caller saves can push out another
  caller's, who then sees its own request read fresh (its own
  `cached_tokens` is 0, its first token comes later) and can tell that
  someone else used the cache. With private counts (the two-Spark GLM text
  service) another caller's counts, event lines and device bytes are not
  shown; the shared room remains. Where callers do not trust each other,
  turn prompt reuse off (`--prefix-cache off`; for the two-Spark GLM text
  service, run it with prompt keep off).

Shared memory:

- Shared decode reads the original prefix and allocates only private tails.
- Chats sharing a KV head read its tiles together, while keeping each chat's
  attention sums in the same order as ordinary decode.
- Short BF16 suffixes can share one attention pass.
- Ordinary session calls restore contiguous cache views when needed.
- Expert suffixes use the same softmax arithmetic as their cold BF16 prefill,
  including when the original prefix is read directly instead of swizzled.

The bounded KV reserve also keeps wholly private mapped ranges, which avoids
tearing down page tables between turns.

- A new session starts at length zero and overwrites its prompt.
- Cached-prefix reuse still requires the same API key.
- Memory pressure, 30 idle seconds, model swaps and shutdown release the
  reserve.

Completed sessions delay their teardown, to leave room for a client that
continues right away.

- Their CUDA teardown yields to pending first tokens.
- Their resident KV and scratch stay charged to admission.
- Shared pages are counted once.
- Admission blocked on memory, swaps and shutdown reclaim completed sessions
  before going ahead.

</details>

<details>
<summary><b>Draft decoding</b>: how guesses are picked, and when it does not draft</summary>

The policies (`auto`, `strict`, `plain`, `fast`), the draft model and
`--draft-global-bank` are described in
[Server and API](../../docs/server.md#speculative-decoding).

- Qwen3 `auto` uses adaptive strict verification at every chat count. Target
  and draft costs, and each request's acceptance, choose how many guesses to
  make. When guessing costs more than plain decode, it uses one canonical row
  and occasionally probes another guess.
- Requests sharing at least 1,024 prompt tokens in the same cache scope use
  plain decode until they finish. Explicit modes override that choice.
- `--draft-k` accepts 1 to 8. Dense targets default to 2 guesses, or 3 when a
  pass reads at least 32 GiB of weights, which includes Qwen3-32B. Explicit
  values survive model swaps; automatic defaults are recalculated for the new
  model.
- The draft model is picked once, at start, in `select_draft_model` in
  `src/bin/spark-serve.rs`. Its tokenizer files must match byte for byte.
- Expert models always decode plainly. Their sessions return no draft target
  (`draft_target` in `src/qwen.rs`), so `--draft strict` and `--draft-model`
  do not change that.

Single-row decode, shared decode and verification share four fixed ranges of
16-column BF16 multiply tiles. Their FP32 sums merge as
`(range0 + range1) + (range2 + range3)`, whatever the batch width.

</details>

<details>
<summary><b>Memory</b>: the prefix cache, context limits and the per-token formula</summary>

[Server and API](../../docs/server.md#how-it-sizes-itself) walks through the plan.

- The Qwen3 prefix cache borrows idle capacity and evicts least recently used
  entries under pressure.
- A larger memory budget does not raise the model's native context limit.

For Qwen3, the KV cache size per token in BF16 is:

```text
bytes per token = layers × KV heads × head dimension × 2 (keys and values)
                  × 2 bytes
```

[Server and API](../../docs/server.md#how-it-sizes-itself) has the result for each
model. This formula covers Qwen3 only. Hybrid Qwen3.8 also holds FP32
recurrent state, and Gemma holds local and global KV caches.

</details>

<details>
<summary><b>What is in the crate</b>: files, endpoints and request details</summary>

<dl>
<dt><code>src/bin/spark-serve.rs</code></dt>
<dd>Reads the flags, sizes memory, loads the model and starts serving.</dd>
<dt><code>src/bin/spark-serve/</code></dt>
<dd>Model-folder detection (<code>cartridges.rs</code>), Gemma start-up
(<code>gemma4.rs</code>) and process lifetime (<code>lifecycle.rs</code>).</dd>
<dt><code>src/http.rs</code></dt>
<dd>Reads and writes HTTP/1.1.</dd>
<dt><code>src/protocol.rs</code></dt>
<dd>Parses requests and writes JSON and SSE replies.</dd>
<dt><code>src/scheduler.rs</code></dt>
<dd>Runs every chat on one GPU thread, in turns, by priority.</dd>
<dt><code>src/cache.rs</code></dt>
<dd>Reuses the KV cache of prompts it has already read.</dd>
<dt><code>src/budget.rs</code></dt>
<dd>Decides how much memory each request may reserve.</dd>
<dt><code>src/generation.rs</code></dt>
<dd>Steps one request forward and samples its next token.</dd>
<dt><code>src/chat_output.rs</code></dt>
<dd>Splits model output into answer text, thinking and tool calls.</dd>
<dt><code>src/tokenizer.rs</code></dt>
<dd>Connects to the tokenizer and the chat template.</dd>
<dt><code>src/qwen.rs</code></dt>
<dd>Connects the scheduler to the Qwen3 engine.</dd>
<dt><code>src/hybrid.rs</code></dt>
<dd>Connects the scheduler to Qwen3.8-27B.</dd>
<dt><code>src/gemma4.rs</code></dt>
<dd>Connects the scheduler to Gemma 4.</dd>
</dl>

Request fields, defaults, errors and limits are in
[Server and API](../../docs/server.md#use-it-from-an-app).

<dl>
<dt><code>POST /v1/chat/completions</code></dt>
<dd>Chat</dd>
<dt><code>POST /v1/completions</code></dt>
<dd>Plain completion from <code>prompt</code> or
<code>prompt_token_ids</code>.</dd>
<dt><code>GET /v1/models</code>, <code>GET /health</code></dt>
<dd>Model name, state and memory in use.</dd>
<dt><code>GET /v1/events</code></dt>
<dd>Best-effort events of this process: counts by name and the last 64 lines.
See <a href="#best-effort-events">Best-effort events</a>.</dd>
<dt><code>POST /v1/cancel</code></dt>
<dd>Cancel a request.</dd>
<dt><code>POST /v1/admin/swap</code></dt>
<dd>Load another model, on the admin socket only.</dd>
</dl>

- `POST /v1/admin/swap` goes through the private socket enabled by
  `--admin-socket`. It is disabled by default and never on the HTTP port.
- Time spent in the queue does not use up a request's generation budget.
- Only complete, valid tool calls become tool events. Malformed or
  incomplete ones come back as ordinary content with HTTP 200.

</details>

## API authentication

On loopback, authentication is optional: the server checks a key only once
you give it a key file. To let other machines reach it (`--allow-remote`), you
must give it one; without it, the server refuses to start with
`--allow-remote requires --api-key-file PATH`. Remote HTTP still needs a
trusted tunnel or TLS termination.

Both `spark-serve` and `flash-serve` accept `--api-key-file PATH`. The file
holds one key. After trimming, the key must be at least 16 bytes of visible
ASCII (`0x21-0x7E`), and the whole file may be at most 4096 bytes. We suggest
32 random bytes written as hex.

Clients send `Authorization: Bearer <key>` or `x-api-key: <key>`; if they send
both, both must match. A missing or wrong credential gets HTTP 401 with a
generic OpenAI error. Only `GET /health` stays open; OPTIONS needs the key too.
Without the key, `GET /health` gives `status` only: no model name. With the
key it gives the full body.

If you also set an owner key (`spark-serve --owner-key-file`, described above),
the server checks it separately, in constant time, and its requests keep
priority 255. It works alone or alongside the API key sent in `x-api-key`.
The cache scopes for API-key and owner-key callers contain no credentials.

The file is read once, so restart the server to rotate the key. At startup the
server refuses a file that group or others can read or write, and it names
the fix (`chmod 600`). It does not change the permissions or log the contents.

## Engine contract v1

This section is reference material for `vade.engine.v1`: the request rules,
extra endpoints and response fields that tools built on Vade can rely on. In
order, it covers:

- request rules and output limits;
- capabilities, checkpoint identity and admission (`/v1/admission`);
- usage, prefill timing and `prompt_cache_key`;
- scoped KV checkpoints.

Request rules:

- OpenAI Completions accepts `prompt` as text or one array of token IDs, as
  well as the existing `prompt_token_ids`. Token IDs pass through unchanged.
- `POST /tokenize` returns `tokens`, `count` and `max_model_len` for
  `{model,prompt,add_special_tokens:false}`.
- `ignore_eos:true` continues to the token limit.
- `cache_salt` isolates exact cache reuse without changing model inputs.
- Streaming output limits count decoded content rather than SSE overhead, and
  normal streams end with usage and `[DONE]`.

Output limits:

- If a request omits its output limit or asks for too much, Chat,
  Completions, Responses and Messages clamp it to the smaller of the service
  cap and the context left after the complete tokenized prompt.
- Zero or malformed limits still fail, and a prompt must leave room for at
  least one output token.
- `output_limit` in capabilities and `caps.output_limit` in health report the
  ceiling for a nonempty prompt on the loaded engine.
- Admission probes (requests to `/v1/admission`, described below, that ask
  whether a request fits) keep the exact `output_tokens` reservation for their
  fit check.

Chat and Responses preserve Qwen3.8's official effort instructions: low is
brief, medium omits the effort preamble, and high is an alias for xhigh. Chat
accepts `reasoning_effort` at the top level or in `chat_template_kwargs` (which
wins). `vade.reasoning_effort` in the final response echoes the effort level
actually used.

`tool_choice:"none"` hides callable definitions and tool events while retaining
thinking and earlier tool history.

### Best-effort events

A best-effort path is a fallback, a retry, a skipped step, a refused
admission, a pool size decision or a cache miss. The paths that have an event
write one log line. The line starts with a stable name, then the numbers as
`key=value`, for example
`prefix_cache_miss prompt_tokens=812 reason=empty scoped_entries=0`.
Lines that existed before keep their text and are counted under a name. Some
best-effort paths do not have an event yet. The event is counted before the
line is written, and a failed log write is ignored.

- `GET /v1/events` returns `{"counts":{name:count},"recent":[{"unix_ms",
  "name","line"}]}`: the counts since the process started, and its last 64
  lines, oldest first. It uses the same key rules as `GET /v1/models`.
- `GET /health` and the `vade` object of every response carry the counts as
  `"events":{name:count}`, so a benchmark card can show a fallback the same
  day. A response holds the counts from the time the server built it.
- `vade status` prints the counts and the last 20 lines, for a local launch
  and for a controller launch. With replicas, it reads each server and labels
  each section with the machine name. If the events read fails, it keeps the
  counts from `/health` and names the failure.

The events are host bookkeeping only. No decision reads them, so they cannot
change a token.

Both `spark-serve` and `flash-serve` expose `GET /v1/capabilities` and
`POST /v1/admission`. They use the same Host, Origin and caller scope rules
as generation. Existing Chat Completions and Responses requests remain valid.

Capabilities advertise:

- `contract_version: "vade.engine.v1"`;
- model identity;
- context, input and output limits;
- tool and thinking support, and the default thinking mode;
- the cache scope and an opaque `cache_domain`.

The cache domain names one cache namespace. It changes every time the server
loads a model, and by default it differs between authorization scopes. An
explicitly shared cache advertises `cache_scope: "shared"`; anonymous callers
share a namespace of their own. A domain only identifies a namespace; it is
never a credential or proof of a cache hit.

Identity includes SHA-256 fingerprints of checkpoint metadata, tokenizer and
template/renderer.

<details>
<summary><b>What each fingerprint covers</b>, and how startup reads it</summary>

- **Weights.** This fingerprint reports
  `fingerprint_kind:"safetensors_header_metadata"`. It hashes the config and
  the optional index, plus, for each shard in name order, its name, its file
  size and a hash of its 8-byte header length and JSON header.
  That makes it a metadata fingerprint, not a digest of the weights: a change
  to the weights that keeps the file size the same goes unnoticed.
  `artifact_bytes` counts file bytes, not resident GPU memory.
- **Tokenizer.** The fingerprint hashes `tokenizer.json`.
- **Template.** The fingerprint covers canonical tokenizer config with the
  effective template text and authored rendering/translation sources.
  Trailing CR/LF is normalized as in Flash's template equality check; an
  optional duplicate template file does not change identity.

Startup and swaps read only headers and metadata, using 65,536 bytes of hash
scratch; no weight payload is read or retained for identity. Peer payload
agreement still requires model-manifest verification. Checkpoint files must
remain immutable while loaded. Engines embedded without checkpoint identity
report `identity`, `tools` and `thinking` as null (unknown).

</details>

Admission takes the complete candidate request, preserving schema key order:

```json
{"contract_version":"vade.engine.v1","endpoint":"/v1/chat/completions","request":{"messages":[{"role":"user","content":"Hello"}],"tools":[]},"output_tokens":128}
```

`endpoint` also accepts `/v1/responses` and `/v1/completions`. `output_tokens`
overrides the candidate's output limit for this check; use that reservation when
sending it for generation.

The reply gives the exact `prompt_tokens` after rendering, plus
`context_limit`, `input_limit`, `output_limit`, `admitted` and a `reason` that
may be null. Rendering, tools, thinking and token checks run exactly as they
would for generation, including the template's hidden framing tokens. Cached
input still counts against the full context.

- A valid but oversized request returns HTTP 200 with `admitted:false`;
  malformed requests return HTTP 400.
- `admission_kind:"context_fit"` and `reserves_resources:false` mean this does
  not reserve a slot or guarantee immediate memory admission. No model session
  is created.
- Admission uses the same connection cap as generation; an overloaded server
  returns HTTP 503.
- Unsupported contract versions are rejected.

Every terminal usage object includes `prompt_tokens_details.cached_tokens`;
Responses also retains its standard `input_tokens_details.cached_tokens`.
Streaming and nonstreaming share the same accounting. Counts describe prompt
work that was actually reused. They leave out recurrent checkpoint tails that a
backend replays to stay exact.

The older opt-in `X-Prefix-Reused-Tokens` header describes the matched snapshot
at admission and can be larger than final usage when a backend replays a tail.

Every admitted streamed reply also sends a warm-cache estimate in its
response head, before prefill starts. A streamed request that is cancelled
or passes its deadline while it waits in the queue gets no estimate.

- `X-Vade-Prompt-Tokens` and `X-Vade-Cached-Tokens`: the prompt length and
  the cached prefix matched at admission. The cached count is the same
  number as `X-Prefix-Reused-Tokens`, so it can also be larger than final
  usage when a backend replays a tail.
- `X-Vade-Estimated-First-Token-Ms`: about how many ms the first token takes
  from admission, with that prefix.
- `X-Vade-Estimated-Cold-Ms`: the same for the prompt with nothing cached.

The estimate is a hint. It is
`setup + fixed + max(0, U - 256) * row * (1 + (P + C) / 600000)`, with P
prompt tokens, C cached tokens, U = P - C, and the measured setup time since
admission. When U is 0 the prefill part is 0. The `fixed` and `row` rates
start from the one-Spark card in [docs/benchmarks.md](../../../docs/benchmarks.md)
(cold 8K in 3.761 s; a short prefill in about 350 ms). Each finished prompt
updates them: short tails (256 rows or fewer) teach `fixed` and prompts of
2,048 rows or more teach `row`. The rates are a mean of the start value and
the first three prompts, then a moving average with weight 0.25. One server
process serves one model, so it keeps one pair of rates. The estimate makes no
scheduling, cache or sampling decision, so it cannot change token IDs.

Every admitted request also echoes always-on engine prefill timing and counters
in the top-level `vade` sampling object.

<details>
<summary><b>Prefill timing fields</b>, and the opt-in <code>usage.vade</code> timing</summary>

- `prefill_ms`, `prefill_prompt_tokens`, `prefill_chunks`,
  `prefill_rows_per_chunk` and `prefill_cache_restored_tokens`.
- `prefill_cache_reuse_pct`: restored / prompt tokens in percent, one decimal.
  It is the one field a client prints as "cache reuse".
- `prefill_cache_restore_ms`: wall ms from cache lookup to the restored
  session; null without a hit.
- `admission_to_first_logits_ms`.
- `prefill_estimated_ms` and `prefill_cold_estimated_ms`: the warm-cache
  estimate from the response head, so a client can compare it with
  `admission_to_first_logits_ms`. Admitted requests only.

Terminal server logs carry the same object. Chunk timing spans successful
commit, including layer yields; admission timing also includes
restore/allocation and waits before first predictions. No device
synchronization is added. See `docs/server.md` for batch, cache-hit and partial
request semantics. Client TTFT remains a separate measurement.

Set `include_usage_timing:true` to add `usage.vade` with `prefill_ms`,
`decode_ms` and `timing:"admitted_wall"`. This is wall-clock time, not GPU
kernel time:

- it starts at cache selection and session admission;
- it splits into prefill and decode when the prompt's logits are ready;
- it ends when generation finishes.

It includes scheduler waits and streaming backpressure. It leaves out request
parsing, tokenization and queue time. A request that ends before admission
reports zero times and zero processed tokens.

</details>

`prompt_cache_key` takes a nonempty string of at most 256 UTF-8 bytes on both
APIs. Empty, oversized and non-string values are ignored, and the request still
goes through.

The key is a scheduling hint. When the server picks work to backfill, priority
and an eligible queue head still come first; among the remaining candidates,
work matching the same caller's active key (affinity) goes before older work,
within the existing slot and byte limits. The key never decides a cache hit:
changing it does not prevent an exact token-prefix hit, and keeping it does not
create one. `cache:false` disables reuse and this hint.

### Scoped KV checkpoints

Dense Qwen and hybrid `spark-serve` advertise `kv_branch_handles:true` when the
retention budget is nonzero. Other model backends (called cartridges in the
code), including Flash-Next, advertise false. `kv_checkpoints` in capabilities
gives the authorization-specific cache domain, count and byte quotas, how
long a checkpoint lives (TTL) in milliseconds, and the release endpoint.
Handles remain authorization-scoped even if ordinary prefix caching is
explicitly shared. Anonymous callers belong to one anonymous scope.

Add `kv_checkpoint:true` to a complete Chat, Responses or Completions request to
retain the exact point where its **prompt** ends, including the prompt's
next-token logits. Terminal JSON/SSE usage includes `usage.vade.kv` independently of timing opt-in:

```json
{"cache_domain":"DOMAIN","reuse":"not_requested","reused_tokens":0,"retention":"retained","checkpoint":{"handle":"HANDLE","prompt_tokens":4096,"reusable_prefix_tokens":4096,"expires_at_ms":1790000000000,"retained_bytes":123456789,"scope_retained_bytes":123456789}}
```

The example numbers are illustrative. `handle` is a 256-bit random value from
the OS, in lowercase hex. It works like a key, so keep it private. The domain
binds it to this loaded model, tokenizer/template, position/KV semantics and
authorization scope; restart or model replacement invalidates it. No weights or
raw KV leave the engine.

To branch, send `kv_handle:"HANDLE"` **together with the complete prompt**.
Rendering, token checks and context and memory admission work as usual. Every
saved prompt token must match the start of the new rendered prompt. An exact
repeat reuses the saved logits; a longer prompt gets its own writable copy of
the state. Completely filled attention pages never change, so branches share
them, with reference counts. Partly filled pages and hybrid recurrent state
are private to each branch. A hybrid append replays the last unfinished tile
of the saved prompt, so `reusable_prefix_tokens` leaves that tail out, while
`prompt_tokens` names the whole saved boundary. Exact repeats can reuse all
`prompt_tokens`.

`reuse` reports `hit`, `replayed` (matched, but all inherited rows were replayed),
`prompt_mismatch`, `invalid_handle`, `unavailable`, `unsupported`, `cache_disabled`,
`not_attempted` (cancelled before admission), or `not_requested`. Unknown, expired,
evicted and foreign handles all return the same `unavailable` result. Misses use
ordinary prefix matching and prefill. `reused_tokens` counts work actually reused
through the named handle; normal cached-token usage can additionally report a
fallback prefix hit. Neither affinity nor a handle bypasses token comparison.

`retention` is `retained`, `capacity`, `unavailable`, `randomness_unavailable`,
`unsupported`, `cache_disabled`, `prompt_incomplete`, or `not_requested`.
Checkpoint issuance failure does not fail generation. A request can both use a
handle and request a new checkpoint. Generated tails are never published by this
API. `cache:true` is unnecessary for explicit checkpoints; `cache:false` disables
both their use and issuance. The legacy `session` capability remains separate.

Limits and eviction:

- **Per scope:** at most 16 handles (or the configured cache entry limit, if
  smaller) and half the configured retained-byte budget. The existing global
  cache/pool limits also apply.
- **`retained_bytes`** is a standalone accounted bound for one checkpoint,
  including tokens/logits, private state/scratch and referenced pages. **Do not
  sum it across handles.**
- **`scope_retained_bytes`** is the scope's total at issuance, counting shared
  allocations once. Active forks also count shared pages only once.
- **Eviction:** byte/count pressure evicts least-recently-used references;
  reads refresh LRU order but do not extend the fixed expiry.
- **Expiry** is an upper bound on retention: pressure can evict earlier, even
  during a long response.

`POST /v1/kv/release` with `{"kv_handle":"HANDLE"}` drops that scope's reference.
It returns `{"released":true}` idempotently, including for unknown/foreign handles.
Existing children and other handles keep their own references. Cancelling a child
does not release its parent. To have something to fall back on, save the
transcript, not raw KV.

## Safety and known limits

**Plain HTTP with an optional shared API key. It listens on loopback unless
told otherwise.**

- It listens on loopback by default. Any other address needs
  `--allow-remote`, and `--allow-remote` needs `--api-key-file`.
- **The API key is the only credential the server checks.** With
  `--api-key-file`, every request except `GET /health` must carry that key,
  or the owner key if you set one. `GET /health` without the key shows only
  the state. Without a key file it checks no credential
  at all, which is allowed on loopback only. There are no user accounts.
  Details are in [API authentication](#api-authentication).
- **There is no TLS.** The server speaks plain HTTP. To reach it from another
  machine, use an SSH tunnel or a TLS proxy you trust in front of it.
- Separately from the key, it refuses a request whose `Host` header is not
  loopback, its own listen address or an `--allow-host` entry. It also
  refuses an `Origin` header other than its own `http://` address or an
  `--allow-origin` entry.
- The server never runs a tool. Tool calls are returned to the client.
- **Swapping the model is unfinished.** Restarting the server is the reliable
  way to change models today.
- **The long-prompt helper flags (`--seqpar-*`) have no effect today.** To
  spread one model over several Sparks, use `tp-serve` from
  [spark-tp](../tp/README.md).

<details>
<summary><b>The admin socket, secrets and GPU faults</b></summary>

[Server and API](../../docs/server.md#limits-on-every-request) lists the HTTP rules
and limits.

- Model administration uses a separate private Unix socket, disabled unless
  `--admin-socket` is set. The HTTP port rejects that endpoint.
- Cancellation and session secrets work like passwords. Keep them private.
- Invalid numerical rows fail only their own request. A sticky GPU fault
  makes `/health` report `poisoned` with HTTP 503 briefly, then the process
  exits with an error.

</details>

<details>
<summary><b>How the swap works</b>, and why the helpers are unused</summary>

`POST /v1/admin/swap` on the private admin socket loads another Qwen3 model
from under `--models-root` without restarting.
[Server and API](../../docs/server.md#the-swap-endpoint-unfinished) has the command
and the known failures. In the code:

1. The new model's files are checked before the old model is touched.
2. New requests stop, and active ones get one second to finish.
3. The old model is freed completely, then the new one loads.
4. If loading fails, the old model reloads. If that fails too, the server
   stays up without a model and accepts another swap.

Gemma and hybrid models need a fresh launch. The Qwen3 swap keeps the memory
plan from start-up, so a larger model can fail to fit.

The flags `--seqpar-peer`, `--seqpar-listen` and `--seqpar-threshold` connect
helper processes from [spark-seqpar](../seqpar/README.md) that read part of a
long prompt. The helpers need an FP32 KV cache, and the server keeps a BF16
one, so every prompt is read locally.

</details>

**Where next:** [Server and API](../../docs/server.md) · [spark-engine](../engine/README.md) ·
[Budgets](BUDGETS.md)
