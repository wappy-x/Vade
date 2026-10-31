# One-Spark server and API

For the tested two-Spark Qwen3.8-Flash-Next FP8 installation, use
[Two-Spark Flash server](flash-server.md): rank zero, port `18080`, model
`qwen3.8-flash-next-fp8`, and a Mac SSH tunnel. The startup and application
recipes below cover the separate one-Spark `spark-serve` workflow.

**`spark-serve` is Vade's inference server: one model on one Spark, behind an OpenAI-compatible API on port 8080.** For app developers and operators.

- **Start:** `./target/release/spark-serve models/Qwen/Qwen3-8B`, then wait for `ready`.
- **Connect:** base URL `http://localhost:8080/v1`, model `qwen3-8b`.
- **First:** [Part 2 of the one-Spark quickstart](../../docs/quickstart-one-spark.md#qwen3-8b-bf16-ubuntu-to-a-first-answer), the older Qwen3-8B route, builds it. Several Sparks: [Multiple Sparks](multi-spark.md).

**Do:** [Start](#start-it) · [Connect an app](#use-it-from-an-app) · [Remote access](#reach-it-from-another-machine) ·
[Stop](#stopping) · [Swap model](#swap-the-model). **Look up:** [Sizing](#how-it-sizes-itself) ·
[Long chats](#long-conversations) · [Caching](#prompt-caching) · [Drafting](#speculative-decoding) ·
[Flags](#flags) · [Limits](#limits-on-every-request)

<details>
<summary><b>Terms on this page</b>: token, context, KV cache, prefix caching, speculative decoding, loopback, greedy, thinking, slot, BF16, MiB</summary>

<dl>
<dt>Token</dt>
<dd>The unit a model reads and writes, often part of a word. The tokenizer maps
text to tokens and back.</dd>
<dt>Context</dt>
<dd>The whole conversation the model sees: prompt plus answer, in tokens.</dd>
<dt>KV cache</dt>
<dd>The attention keys and values kept for every token already read: the model's
memory of the conversation. It grows with every token.</dd>
<dt>Prefix caching</dt>
<dd>Reusing the KV cache of a prompt beginning that was read before.</dd>
<dt>Speculative decoding</dt>
<dd>Proposing several tokens cheaply and checking them in one pass of the model.
Vade's flags call it drafting.</dd>
<dt>Loopback</dt>
<dd>An address only programs on this Spark can reach, such as
<code>localhost</code>.</dd>
<dt>Greedy</dt>
<dd>Always taking the most likely next token; request <code>temperature: 0</code>.</dd>
<dt>Thinking</dt>
<dd>The model writing out its reasoning, between <code>&lt;think&gt;</code> and
<code>&lt;/think&gt;</code>, before the answer. On by default. It makes answers
longer and slower.</dd>
<dt>Slot</dt>
<dd>One request the server is running at the same moment as others.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
<dt>GB, MiB, GiB, KiB</dt>
<dd>1,000,000,000; 1,048,576; 1,073,741,824; and 1,024 bytes.</dd>
</dl>

</details>

## API authentication

`spark-serve` and `flash-serve` accept `--api-key-file PATH`. The regular file
must contain one trimmed key of at least 16 bytes, using visible ASCII
`0x21-0x7E`; the entire file is limited to 4096 bytes. Prefer 32 random bytes
encoded as hexadecimal and keep the file private. Startup warns once if the
file is group- or world-readable; it never changes permissions or prints the key.

Send `Authorization: Bearer <key>` or `x-api-key: <key>`. If both identify the
API credential, both must match. Missing or incorrect credentials return HTTP
401 with a generic OpenAI authentication error, including on Anthropic routes.
Only `GET /health` stays open; OPTIONS also requires the key. Browser preflight
support with authentication needs an owner decision.

Loopback authentication is optional unless the file is supplied.
`--allow-remote` requires the file and otherwise refuses startup with
`--allow-remote requires --api-key-file PATH`. HTTP has no TLS: use an SSH tunnel
or TLS termination for remote access. Restart to rotate the key. Harness and
CLI clients accept `--api-key-file PATH` or `VADE_API_KEY_FILE`.

An existing `--owner-key-file PATH` keeps working alongside API authentication:
the owner Bearer key is checked separately in constant time and grants priority
255. It may be used alone or with a matching API credential in `x-api-key`.
Owner and API requests have separate cache scopes, neither containing the key.

## Start it

**The model folder is the only required setting; everything else is sized automatically.**

### Do it yourself

1. From the monorepo’s `engine/` workspace, start the server with a model folder:

   ```sh
   ./target/release/spark-serve models/Qwen/Qwen3-8B
   ```

2. Wait for the line that starts with `ready`. It lists every limit the
   server chose and the memory it uses, in bytes.

What it picks when you add no flags:

| Setting | Default |
| --- | --- |
| Address | Loopback |
| Port | 8080 |
| Model ID | Folder name |
| Chats at once | Up to 32 |
| Context | Longest that fits |
| Sampling | Server/model defaults; see the `sampling_defaults` startup log |
| Prompt caching | On, per API key |
| Drafting | `auto` |
| API authentication, encryption | Optional key on loopback; no TLS |

- The model ID is the folder name in lower case, here `qwen3-8b`. Loopback
  means only programs on this Spark can reach it.
- Connections made before `ready` wait until the model is loaded.

For the configured multi-Spark Flash-Next workflow, use
[`vade up`](../../cli/README.md#multi-spark-source-setup). Flash-Next defaults to
131,072 context tokens, a 65,536-token output cap and MTP depth 2. Explicit
`--context` values override the context default. The automatic session pool and
retained cache fund one full-context session, or startup refuses with a plain
memory message; `--chats` reserves explicit concurrent full-context sessions.
Use `vade up --context 32768 --output 8192` for comparisons with the dated timing
receipts.
The new default still needs native startup, cache and capacity qualification.
See [INSTALL.md](../../INSTALL.md#5-start-the-two-spark-server) for the product route.

### Ask your coding agent

```text
Start Vade's server on this DGX Spark from the monorepo's engine/ workspace with
./target/release/spark-serve and the model folder I name (default
models/Qwen/Qwen3-8B). Run as the current user and never use sudo. Add no
flags unless I ask. Wait for the line that starts with "ready", then show it
and the output of curl -sS http://localhost:8080/health.
```

## Use it from an app

**Vade speaks the OpenAI chat API, so any app with an OpenAI base URL setting can use it, as long as it sends plain-text messages.**

### Do it yourself

1. If the app runs on another machine, such as a laptop, open an SSH tunnel
   first, as in [Option A](#option-a-an-ssh-tunnel-recommended).

2. Set the app's base URL to `http://localhost:8080/v1`.

3. If the server uses `--api-key-file`, configure the app with that key.
   Vade validates it; see [API authentication](#api-authentication).

4. Set the model to the model folder's name in lower case, such as `qwen3-8b`.

5. Check the connection from the machine the app runs on. The reply should
   name the model.

   ```sh
   curl -s http://localhost:8080/v1/models
   ```

6. Optional: turn Qwen3's thinking off, for shorter and faster answers. Send
   `"chat_template_kwargs": {"enable_thinking": false}` if the app allows it,
   or add `/no_think` to the last user message. If the app can do neither,
   start the server with `--default-thinking off`.

Whether environment variables work for these settings depends on the app.

### Ask your coding agent

```text
Connect this app to the Vade server on a DGX Spark through its
OpenAI-compatible API: base URL http://localhost:8080/v1, any API key, and
the model ID that /v1/models returns (for example qwen3-8b). If the app runs
on another machine, first open an SSH tunnel with
ssh -N -L 8080:localhost:8080 "$SPARK". Check the connection with
curl -s http://localhost:8080/v1/models. Send message content as plain text,
not content arrays. If the app allows extra request fields, add
"chat_template_kwargs": {"enable_thinking": false}. Do not change anything on
the Spark.
```

<details>
<summary><b>Endpoints</b>: seven, and what is missing</summary>

<dl>
<dt>POST <code>/v1/chat/completions</code></dt>
<dd>Chat. Streamed with <code>"stream":true</code>, otherwise one JSON
object.</dd>
<dt>POST <code>/v1/responses</code></dt>
<dd>Text Responses API, including streamed semantic events and function calls.
Uses the same chat template, sampling, scheduler and prefix cache as chat.</dd>
<dt>POST <code>/v1/completions</code></dt>
<dd>Plain text completion from <code>prompt</code>, or from
<code>prompt_token_ids</code>.</dd>
<dt>GET <code>/v1/models</code></dt>
<dd>The served model ID and its state.</dd>
<dt>GET <code>/health</code></dt>
<dd><code>status</code>, <code>model</code>, <code>active</code>,
<code>queued</code>, <code>device_bytes</code>, and the cached build/checkpoint
receipt in <code>vade</code> (below).</dd>
<dt>POST <code>/v1/cancel</code></dt>
<dd>Cancel a request; see Errors, timeouts and cancelling.</dd>
<dt>POST <code>/v1/admin/swap</code></dt>
<dd>Replace the model; admin socket only.</dd>
</dl>

`/v1/admin/swap` works only through the private Unix socket enabled by
`--admin-socket`. It is disabled by default and unavailable on the HTTP port.

There is no embeddings endpoint. Apps that need other endpoints or structured
output need their own adaptation.

`GET /health` includes serving metadata in `vade`. Its `numerics.draft` and
`numerics.mtp_depth` report the server default mode and depth, with zero depth
for plain decoding (`--mtp 0`). For example (digest values abbreviated here):

```json
{
  "status": "ready", "model": "qwen3.8-flash-next",
  "active": 0, "queued": 0, "device_bytes": 81234567890,
  "vade": {
    "build": {"git_sha": "b193d086499d4b22e3258cbd04cc1687ef99e8cb", "dirty": false, "source": "git"},
    "ptx_sha256": "0123...",
    "layer_ptx_sha256": "89ab...",
    "checkpoint": {"directory": "Qwen3.8-Flash-Next", "sha256": "4567...", "hash_state": "ready", "hash_source": "manifest"},
    "numerics": {"flash_bf16_mma_set": false, "flash_bf16_mma": false,
      "flash_bf16_experts_set": false, "flash_bf16_experts": false,
      "flash_mtp_sampled_set": false, "flash_mtp_sampled": false, "nvfp4": true,
      "draft": "strict", "mtp_depth": 3, "world_size": 1, "topology": "local"},
    "server_start_unix_ms": 1791720000000
  }
}
```

`build.git_sha` and `build.dirty` are embedded when Cargo builds the server
library. Dirty includes staged, unstaged and untracked files that Git does not
ignore. A release export preserves `EXPORT-MANIFEST.json`'s source revision.
Git archives use the export-substituted `REVISION` with `source:archive`.
They report `dirty:true` if a source file is newer than `REVISION` or source
metadata cannot be read. Cargo output, generated PTX and kernel provenance
are excluded from this mtime guard. Unidentified archives and previews report `git_sha:null,
dirty:true`. `build.source` is `git`, `export`, or `archive`.
`ptx_sha256` hashes the exact PTX buffer passed to CUDA, excluding its terminating
NUL, for either embedded kernels or an explicit PTX file. `layer_ptx_sha256`
hashes the embedded layer module CUDA also loads, excluding its NUL.

The checkpoint digest matches the goldens: SHA256 of plain sha256sum records
`<lowercase hex>  <filename>\n`, with the index row first and then unique shards
in lexical filename order. An unsharded checkpoint has only the
`model.safetensors` row. Directory names are reported separately; paths are
never included. The capabilities header fingerprint remains unchanged.

`manifest.sha256` beside the checkpoint supplies expected hashes. It must hash
the current index and every shard. NVFP4 also honors `FLASH_NVFP4_MANIFEST` and
retains its required config/tokenizer rows. The existing private stamp cache
binds each expected hash to file size, device, inode, mtime and ctime.
Cache hits avoid payload reads. A missing manifest or stamp starts up with
`checkpoint.sha256:null, hash_state:pending`; after model loading, up to eight background workers
hash with 65,536 bytes of scratch each. Successful verification publishes
`hash_state:ready` to health and shared request echoes and seeds a missing
manifest. `manifest.sha256.sizes` binds per-file sizes to the manifest digest;
the index hash and file identities are also checked before publishing.
Stale or malformed server-written manifests and sizes never block loading.
They report `hash_state:failed-stale` with a null digest while a fresh background
rehash replaces each of the two files atomically. Same-size payload mismatches discovered
in the background also trigger this refresh. Deleting both `manifest.sha256`
and `manifest.sha256.sizes` forces a full rehash, even if private stamps remain.
The operator-supplied NVFP4 manifest retains its existing strict verification.
Background I/O or concurrent file-change failures report `hash_state:failed`
with a null digest. `hash_source` is `manifest`,
`background`, or `manifest+background`. Read-only checkpoint storage still
permits a verified receipt, but logs persistence failure. A swap refreshes the
receipt and retains the server start time. Files must remain immutable for the
loaded checkpoint's lifetime. Requests never hash payloads.

`numerics.flash_dense_arithmetic` is the fixed string
`"bf16_hi_lo_tensor_core"`: Flash's covered BF16 projections always use the
one tensor-core mode. There are no `flash_bf16_mma_set` or `flash_bf16_mma`
switch fields. `numerics` also records presence and value `1` for
`FLASH_BF16_EXPERTS` and `FLASH_MTP_SAMPLED`, the selected NVFP4 format,
configured MTP depth, world size and actual `local`, `ring` or `path`
topology. `spark-serve` with sequence peers reports their path and total rank
count. `server_start_unix_ms` is the process startup wall clock in Unix
milliseconds. These fields change no generation
defaults. In-process engines without checkpoint metadata and model-swap loading
responses report `vade:null`. A checkpoint pin is available only when
`checkpoint.hash_state` is `ready`, even if the server is already serving.
Receipt identity comparisons include `ptx_sha256`, `layer_ptx_sha256` and
`numerics`; they exclude `hash_source`, which can change across launches.

</details>

<details>
<summary><b>Responses API and Codex CLI</b></summary>

`/v1/responses` accepts `input` as a string or full history of text `message`,
`function_call`, `function_call_output`, `custom_tool_call`,
`custom_tool_call_output` and reasoning-summary items.
`instructions` becomes the first system message. Message roles include
`developer`; content parts may be `input_text` or `output_text`. Contiguous
assistant text, reasoning summaries and calls form one chat turn.
`reasoning.effort` works like chat's `reasoning_effort`: `none` and `minimal`
turn thinking off for that request; `low`, `medium`, `high` and `xhigh` turn it
on with the checkpoint's effort instructions; any other value returns HTTP 400, as does a `reasoning` that is not an
object. Leaving it out uses the server default.

Tools use the Responses shape (`type`, `name`, `description`, `parameters`,
`strict`), with up to 64 tools after flattening namespace groups. Function
argument strings retain their JSON order. `max_output_tokens` sets the shared
answer limit; sampling and Vade cache fields have the same meanings as chat.
`tool_choice` supports `auto` and `none`. An optional field sent as `null`
counts as left out.

**Custom tools.** A custom (freeform) tool takes one piece of plain text
instead of JSON arguments. Codex's file-editing tool, `apply_patch`, is one.
It is sent as `type: "custom"` with `name`, an optional `description` and an
optional `format`: `{"type": "text"}`, or `{"type": "grammar", "syntax": "lark"
or "regex", "definition": ...}`.

- What the model sees: a function with the same name and one required string
  parameter, `input`. Its description is the tool's description, then "Pass
  the tool's text as the `input` string." and, with a grammar, "It must match
  this Lark grammar:" (or "regular expression") and the grammar itself.
- Not enforced: the grammar guides the model, but generation is not
  constrained to it. A model can still write input the grammar rejects; the
  client's own tool then reports the error to the model.
- What the client gets: a `custom_tool_call` item (`id`, `call_id`, `name`,
  `input`, `status`) instead of a `function_call`. `input` is the decoded
  `input` string from the model's arguments `{"input": "..."}`, or a bare JSON
  string. Text after that string is dropped. If the arguments do not start
  either way, `input` is the argument text exactly as the model wrote it.
- Streaming: `response.output_item.added`, then
  `response.custom_tool_call_input.delta` events as the string is written,
  then `response.custom_tool_call_input.done` and `response.output_item.done`.
  The deltas always add up to the final `input`, in both streaming and
  non-streaming mode.
- Cut short: a call cut off by `max_output_tokens` keeps its partial input and
  the response ends `incomplete`. A call that never closes its string or its
  markup ends `completed` with the input written so far. Nothing here is an
  HTTP error.
- History: a `custom_tool_call` goes back to the model as the function call
  `{"input": "..."}` written in the chat template's own JSON style, and a
  `custom_tool_call_output` as a tool result. A custom history therefore
  produces exactly the same prompt as the matching function-call history.

Streaming emits `response.created`, `response.in_progress`, output item and
content-part events, `response.output_text.delta`, reasoning-summary deltas,
and function argument deltas. A call's name is sent as soon as it is complete,
and its arguments stream as the model writes them, just as in chat. `response.completed` includes the final output and
input/output/total token usage, including reused input tokens. A length stop
emits `response.incomplete` with `max_output_tokens`; a generation failure
emits `response.failed`. There is no chat-style `[DONE]` marker.
Nonstreaming returns the same final Response object.

This endpoint is stateless: send full history on every turn, as Codex does.
`previous_response_id` is accepted as metadata alongside full input; it does
not retrieve stored history. `store` is always false. Opaque encrypted reasoning
is ignored; text reasoning summaries are accepted. Hosted tools, images, item
references, background requests and structured output are unsupported and
return HTTP 400. Unknown client metadata is ignored.

With `URL` set to the server base URL through a local forward, Codex CLI
0.160.0 sends requests this endpoint accepts with the settings below. A full
Codex coding session on Vade has not been completed yet. Codex sends
`apply_patch` as a custom tool only when its model catalog entry for the model
sets `apply_patch_tool_type` to `freeform`; for an unknown model name it sends
no file-editing tool at all.

```sh
codex exec -m qwen3-32b \
  -c model_provider=vade \
  -c 'model_providers.vade.name="Vade"' \
  -c "model_providers.vade.base_url=\"$URL/v1\"" \
  -c 'model_providers.vade.wire_api="responses"' \
  -c 'model_providers.vade.requires_openai_auth=false' \
  -c 'model_providers.vade.supports_websockets=false' \
  -c 'web_search="disabled"' \
  'Use the shell tool to calculate 7 * 6, then return its output.'
```

Mac validation used recorded CLI requests and Vade's real scheduler with a
scripted host backend. Model-driven tool use on Qwen3-32B requires Spark
verification: default server launch, an SSH local forward, then this command.
Verify the tool arguments, matching `call_id` on its returned output, a second
request carrying full history, and the final answer and usage. Run the frozen
Qwen3 client (commit `1d80e7c`) and the full exactness gate on a Spark
before merging; include the required Qwen3.8-27B BF16 LONG serve gate.

</details>

<details>
<summary><b>Request fields</b>: the OpenAI fields Vade accepts, and what it rejects</summary>

Message and output fields:

<dl>
<dt><code>messages</code></dt>
<dd>1 to 4,096 messages. Roles <code>system</code>, <code>user</code>,
<code>assistant</code>, <code>tool</code>. Content is a string; assistant
messages with tool calls may leave it out or use <code>null</code>.</dd>
<dt><code>model</code></dt>
<dd>Optional. If present, it must match the served model. Any optional field
sent as <code>null</code> counts as left out, here and below.</dd>
<dt><code>stream</code></dt>
<dd><code>true</code> for server-sent events. Default <code>false</code>.</dd>
<dt><code>max_tokens</code></dt>
<dd>Positive integer. Clamped to the service answer limit and the room left
after the tokenized prompt. Omitted limits use that available room.</dd>
<dt><code>max_completion_tokens</code></dt>
<dd>The same clamp applies; wins if <code>max_tokens</code> is also set.</dd>
<dt><code>stop</code></dt>
<dd>A non-empty string, or a list of up to 4; at most 1,024 bytes each.</dd>
<dt><code>n</code></dt>
<dd>Only 1.</dd>
<dt><code>tools</code></dt>
<dd>Up to 64 function tools: name, description, parameters.</dd>
<dt><code>tool_calls</code></dt>
<dd>Up to 64 function calls per assistant message. Arguments are a JSON object
or a JSON string. Qwen's template shows tool results in order.</dd>
<dt><code>tool_call_id</code></dt>
<dd>A non-empty string on a tool message.</dd>
<dt><code>reasoning_content</code></dt>
<dd>On assistant messages, to pass back earlier thinking.</dd>
<dt><code>enable_thinking</code></dt>
<dd>In <code>chat_template_kwargs</code> or at the top level. <code>true</code>
turns thinking on and <code>false</code> turns it off, whatever the server
default. While thinking is on, <code>/no_think</code> in the last user message
turns it off.</dd>
<dt><code>reasoning_effort</code></dt>
<dd>At the top level or in <code>chat_template_kwargs</code> (the latter wins).
Decides thinking when <code>enable_thinking</code> is left out.
<code>none</code> and <code>minimal</code> turn thinking off; <code>low</code>,
<code>medium</code>, <code>high</code> and <code>xhigh</code> turn it on.
Qwen3.8 uses its official brief-thinking instruction for low, no effort
preamble for medium, and its careful-thinking instruction for xhigh.
OpenAI high aliases the checkpoint's xhigh. Other templates retain their
ordinary thinking behavior. Any other value returns HTTP 400, even when
<code>enable_thinking</code> is also set. With neither field, the server default
applies: the chat template's own default (off for Gemma 4 and the small
Qwen3.5 templates, on otherwise), unless the server was started with
<code>--default-thinking on|off</code>.</dd>
<dt><code>prompt</code></dt>
<dd><code>/v1/completions</code> only: a string or one array of integer token IDs.
IDs pass through unchanged. Exactly one of <code>prompt</code> and
<code>prompt_token_ids</code>.</dd>
<dt><code>prompt_token_ids</code></dt>
<dd><code>/v1/completions</code> only. Exactly one of <code>prompt</code> and
<code>prompt_token_ids</code>.</dd>
<dt><code>add_special_tokens</code></dt>
<dd>Boolean. Qwen plain-text prompts have no automatic BOS/EOS insertion;
token-ID prompts already contain their complete input.</dd>
<dt><code>ignore_eos</code></dt>
<dd>Boolean, default false. When true, model EOS tokens do not stop generation;
the token limit and explicit stop strings still apply.</dd>
<dt><code>cache_salt</code></dt>
<dd>A nonempty string of at most 256 UTF-8 bytes. Reuse requires the same
salt and authorization scope, without changing prompt tokens. Different
salts cannot share automatic prefixes, sessions or KV checkpoint handles.</dd>
</dl>

`POST /tokenize` accepts `model`, a string `prompt`, and optional boolean
`add_special_tokens`. It returns `tokens`, `count`, and `max_model_len` using
the same plain-text encoding as Completions, with no generation or KV allocation.
It uses generation's authentication and request limits; `max_model_len` reports
the configured context limit, as described in [Start it](#start-it).

When a request leaves out its answer limit, see [Long conversations](#long-conversations).

Sampling fields:

Omitted or `null` sampling fields inherit the server's defaults. When the
server loads `generation_config.json`, its settings win over the fallback;
request fields override them individually. For the official
Qwen3.8-Flash-Next-FP8 checkpoint, omission means temperature 1, top-k 20,
top-p 0.95 and penalties 1/0/0: a sampled request. Without checkpoint sampling
defaults, the HTTP fallback has temperature 0, top-k 20 and top-p 0.95 with
thinking or 0.8 without. This behavior is unchanged.

Every benchmark and gate body must explicitly send `"temperature":0`,
`"top_k":0` (or 1), `"top_p":1`, `"repetition_penalty":1`,
`"presence_penalty":0` and `"frequency_penalty":0`.

<dl>
<dt><code>temperature</code></dt>
<dd>0 or more. 0 is greedy; omission uses the server/model default above.</dd>
<dt><code>top_p</code></dt>
<dd>Above 0 and at most 1. Omission uses the server/model default above. Applies only
when <code>temperature</code> is above 0.</dd>
<dt><code>top_k</code></dt>
<dd>A whole number; 0 turns the cap off. Omission uses the server/model default above.
Applies only when <code>temperature</code> is above 0.</dd>
<dt><code>min_p</code></dt>
<dd>0 to 1. Default 0 (off). Keeps the tokens whose probability is at least
<code>min_p</code> times the best token's, after top-k and top-p. Applies only when
<code>temperature</code> is above 0. The echo shows it only when it is above 0.</dd>
<dt><code>seed</code></dt>
<dd>Unsigned 64-bit number. Random if left out; streamed responses report it in
the <code>X-Sampling-Seed</code> header and the <code>vade.sampling.seed</code> field.</dd>
<dt><code>presence_penalty</code></dt>
<dd>-2 to 2. Default 0.</dd>
<dt><code>frequency_penalty</code></dt>
<dd>-2 to 2. Default 0.</dd>
<dt><code>repetition_penalty</code></dt>
<dd>Above 0. Default 1.</dd>
</dl>

Generated responses include a top-level `vade` object beside `usage`:

```json
"vade": {
  "sampling": {
    "temperature": 0, "top_k": 0, "top_p": 1,
    "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
    "seed": 42
  },
  "draft": "plain", "mtp_depth": 0, "thinking": false,
  "draft_used": false, "model_draft_used": false,
  "prefill_ms": 12.345, "prefill_prompt_tokens": 768,
  "prefill_chunks": 2, "prefill_rows_per_chunk": [512, 256],
  "prefill_rows_per_chunk_omitted": 0,
  "prefill_cache_restored_tokens": 0, "admission_to_first_logits_ms": 14.567
}
```

These are the values resolved for this request, including its random seed when
omitted. `draft` is the effective mode after the scheduler resolves the request
and server policy, including concurrency and shared-prefix restrictions. It
can change during execution; the final echo reports the last resolved mode.
`mtp_depth` is the resolved request MTP depth ceiling, zero for plain decoding
or without an MTP head. FP4-Local defaults to the server depth (normally 3);
an explicit supported request depth overrides it. `numerics.draft` and
`numerics.mtp_depth` agree with the request fields; health reports server defaults.
For hybrid `spark-serve --mtp`, the default ceiling is 3. Explicit adaptive
`--draft-depth 0` reports its maximum ceiling of 4 (the typed MoE head caps
at 3). It does not claim that MTP ran for this request.
`draft_used` records verification of
speculative proposals; `model_draft_used` records model proposals, including
MTP. Both can be false even with drafting enabled. `thinking` is the effective chat
template switch after request fields, tool policy and `/no_think`; plain
completions have no chat template and report false.

The prefill fields are always present for admitted requests, including terminal
errors, and are logged in `request_completed`, `request_cancelled` or `request_failed`. `prefill_ms`
adds host wall time from each engine prompt chunk's start to its successful
commit. A resumable chunk keeps one start across layer yields, so its time
includes intervening scheduler work and waits. Batch peers each receive the
whole shared engine-call duration; these times must not be summed across peers.
`prefill_chunks` counts committed chunks, and `prefill_rows_per_chunk` lists
their submitted prompt rows in order, capped at 64 entries in receipts and logs;
the trailing `prefill_rows_per_chunk_omitted` counts omitted chunks (zero without truncation).
Coordinated long prefill is one chunk. Failed or abandoned chunks are excluded.
The row list uses at most 512 host bytes on Spark, plus vector metadata and a
chunk counter; no device memory is added.

`prefill_prompt_tokens` is the full rendered input count;
`prefill_cache_restored_tokens` is actual reused work, with backend replay tails
excluded, matching terminal cached-token usage. `admission_to_first_logits_ms`
starts at cache selection/session admission and ends when final prompt
predictions return, before host sampling or output. It includes cache restore,
session allocation and admitted scheduler waits, excludes HTTP parsing,
tokenization and queue time, and is `null` if prefill never completes. A full
cache hit reports zero chunks and zero `prefill_ms`, with a measured admission
time to restored logits. These are monotonic host timings, with no added device
synchronization; they are not GPU kernel timings. The optional `usage.vade`
admitted-wall timing remains available separately. Client receipts retain the
top-level `vade` object beside their own `ttft_ms`.
`usage.vade.prefill_ms` is opt-in admitted wall time (roughly `admission_to_first_logits_ms`); top-level `vade.prefill_ms` sums committed engine chunks through synchronization, and receipts read the top-level value.
Queued empty completions carry the rendered prompt count, zero prefill time,
zero chunks/restored tokens and `null` admission-to-logits time.

The same `vade` object also includes `build`, `ptx_sha256`, `layer_ptx_sha256`, `checkpoint`,
`numerics` and `server_start_unix_ms` from `/health`, so a response identifies
the binary and weights that served it. The sampling example above omits those
cached fields for space. No files are hashed on the request path.

Non-streaming chat, completions, Responses and Anthropic Messages bodies carry
the object. Streams carry it on the final JSON chunk/event before termination
(on Anthropic's `content_block_stop`, `message_delta` and `message_stop`, and on
the final Responses event). Generation errors and queued empty completions carry it too. Requests
rejected before sampling is resolved have no generation settings to echo.
Health carries the serving receipt without sampling fields. OpenAI-style clients can ignore this
additional field; generated content, tokens and logits are unchanged.

The server writes one `request_sampling id=... vade=...` line at admission,
before generation, so it remains available if the client disconnects. Its
sampling, MTP depth and thinking match the response; its draft mode records
the admission policy and its draft-use flags are initially false. Queued empty
completions log at termination with `draft:plain` because no generation ran.
Each request id gets one terminal log: `request_completed` after a terminal result
is produced, `request_cancelled` for cancellation/deadline/disconnection (including
queued empty responses), or `request_failed` for generation/output/capacity errors.
A later flush failure does not add another terminal log or replace a buffered result.
A `sampling_defaults` line at startup lists chat and completion defaults and
configured MTP depth. Startup uses `seed:null` with `seed_policy=random_per_request`
because
the seed is chosen separately for each request. Benchmark receipts retain
each response's `vade` object; the server determinism gate fails if its echo
is missing or is not explicitly greedy.

What is not supported:

- Fields not listed here, such as `response_format`, `tool_choice` or
  `logprobs`, are ignored. JSON mode is therefore not enforced.
- Content arrays and images are rejected with HTTP 400. A `developer` message
  is rendered as a `system` turn, wherever it appears in the conversation.

</details>

<details>
<summary><b>Vade's own fields</b>, and how priority is granted</summary>

<dl>
<dt><code>priority</code></dt>
<dd>0 to 255. Higher runs first, subject to the server's
<code>--client-priority-cap</code> (default 0). An owner key grants 255.</dd>
<dt><code>draft</code></dt>
<dd><code>auto</code>, <code>plain</code>, <code>strict</code> or
<code>fast</code> for this request.</dd>
<dt><code>cache</code></dt>
<dd><code>false</code> reads this prompt fresh and does not keep it.
<code>true</code> also returns a session secret.</dd>
<dt><code>session</code></dt>
<dd>Continue a saved conversation. Requires <code>"cache": true</code>.</dd>
<dt><code>fork</code></dt>
<dd>Branch a saved conversation. Requires a session.</dd>
<dt><code>cancel</code></dt>
<dd>A 64-hex-digit secret of the client's own, to cancel before the response
starts.</dd>
</dl>

More on these: [Speculative decoding](#speculative-decoding) · [Prompt caching](#prompt-caching).

How priority is granted:

- Ordinary clients get priority 0 unless `--client-priority-cap` allows up to
  254.
- `--owner-key-file` grants priority 255 to requests with the matching
  `Authorization: Bearer KEY` header.
- With at least three active slots (chats running at once), that owner key
  also reserves one slot and memory for a short owner request.
- Ordinary API keys never grant owner priority.

</details>

<details>
<summary><b>Streaming and tool calls</b>: an example request, and what comes back</summary>

```sh
curl -N http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-8b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "Give me three names for a sailboat."}
    ],
    "stream": true,
    "max_tokens": 200,
    "temperature": 0.7,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

What comes back:

- Each event is one `data:` line holding a `chat.completion.chunk`.
- Each chunk's `delta` carries `content`, `reasoning_content` (thinking text)
  or `tool_calls`.
- The last chunk carries `finish_reason` and `usage`. Then `data: [DONE]`
  follows.
- Without `"stream":true`, a finished request returns one `chat.completion`
  object with the same fields. Its message always carries `"refusal": null`,
  as the OpenAI format requires.

How tool calls work:

- The server never runs a tool. It hands each call back to the app.
- Tool calls stream as `tool_calls` once each call is complete and valid. The
  request then finishes with `finish_reason: "tool_calls"`.
- Tool schemas guide the model through its prompt. They do not force the shape
  of its output.
- Malformed or incomplete tool calls come back as ordinary content with HTTP
  200. After a malformed call, the rest of that reply stays ordinary content.
- Qwen3.8-27B does not support tool chats yet. Its chat formatter rejects a
  request that carries `tools`, tool calls or tool messages.

</details>

<details>
<summary><b>Errors, timeouts and cancelling</b></summary>

Errors are JSON, `{"error":{"type":"...","message":"..."}}`, with an HTTP status:

<dl>
<dt>400</dt>
<dd>Type <code>invalid_request_error</code>. Something in the request is outside
the rules above. The message says what.</dd>
<dt>409</dt>
<dd>Type <code>client_cancelled</code>. The client cancelled the request.
The response includes <code>x-should-retry: false</code>.</dd>
<dt>500</dt>
<dd>Type <code>generation_error</code>. Generation failed. An invalid numerical
row fails only its own request.</dd>
<dt>503</dt>
<dd>The model is loading or swapping, or all 32 connections are busy. Retry
shortly.</dd>
<dt>504</dt>
<dd>Type <code>generation_deadline</code>. The request reached its deadline.
The response includes <code>x-should-retry: false</code>.</dd>
</dl>

Cancellation and deadline expiry are terminal errors. A non-streamed request
returns 409 or 504, including while queued, without a successful partial answer.
An open stream keeps deltas already sent, then ends with an error event; Responses
uses `response.failed`. Neither ending is a successful `length` stop.

The generation deadline defaults to 3,600 seconds (one hour), also the maximum;
`--generation-timeout-secs` accepts 1 to 3,600 for `spark-serve`. Queue wait has a
separate bound of the same length; admission starts a fresh generation deadline.
The deadline can stop a reply before its token cap, including a 65,536-token
Flash-Next reply. A higher token cap does not extend the time limit.

A sticky GPU fault makes `/health` return HTTP 503 with `status: "poisoned"`
briefly, before the server exits with an error.

<a id="gpu-wait-limits"></a>

Each CUDA wait is bounded to 120 seconds by default, in servers, probes and tests.
Set `VADE_GPU_WAIT_TIMEOUT_SECS` in the GPU process environment before startup to
a nonnegative whole number of seconds; `0` disables this wait bound. A stalled
wait retires the process after another 4.5 seconds, retaining in-flight
allocations until exit. This override does not disable fatal-error cleanup after
the serving process installs its teardown supervisor.

**Cancelling.** Closing the connection cancels the request. To cancel from
somewhere else:

1. Read the `X-Cancel-Capability` header from a streamed response.

2. Post it back:

   ```sh
   curl -s http://localhost:8080/v1/cancel \
     -H 'Content-Type: application/json' -d '{"cancel":"<capability>"}'
   ```

   The answer is `{"cancelled":true}`.

To be able to cancel before the response starts, put a random 64-hex-digit
secret in the request's `cancel` field. Cancellation takes effect at the next
step.

</details>

## Reach it from another machine

**Use an SSH tunnel: API keys authenticate requests, but HTTP has no TLS (encryption).**

By default only programs on this Spark can reach it, which still includes
every account on this Spark.

### Option A: an SSH tunnel (recommended)

1. On your laptop, open a tunnel. Set `SPARK` to the existing SSH alias.

   ```sh
   ssh -N -L 8080:localhost:8080 "$SPARK"
   ```

2. Leave it open. Your laptop's `http://localhost:8080/v1` now reaches the Spark.

### Ask your coding agent

```text
Make the Vade server on the DGX Spark reachable from this laptop with an SSH
tunnel, as in docs/server.md Option A. Use the existing SSH alias in $SPARK.
Run ssh -N -L 8080:localhost:8080 "$SPARK" and leave it open, then run
curl -s http://localhost:8080/v1/models on the laptop and show the reply.
Do not use --allow-remote. Do not change SSH, firewall or network settings.
```

<details>
<summary><b>Option B</b>, a login-checking proxy on the Spark, and <b>option C</b>, a network address with API authentication</summary>

**Option B: a login-checking proxy on the Spark.** The server stays on
loopback, without `--allow-remote`. A reverse proxy on the same Spark checks
logins, takes outside connections and passes them to loopback port 8080.

1. The proxy must forward an allowed Host. `--allow-host AUTHORITY` adds an
   exact host name, optionally with a port.
2. For browser clients, `--allow-origin ORIGIN` adds an exact HTTP or HTTPS
   origin, including its port, without a trailing slash.
3. Repeat either flag to allow more entries.

Model administration stays on its private socket, and is disabled unless one
is configured. Choosing and setting up the proxy is outside this guide.

**Option C: a network address.** A direct remote listener requires
`--allow-remote --api-key-file PATH`. It has no TLS. Use a trusted tunnel or
TLS termination to protect credentials. This setup
does not change firewall rules or interface settings.

```sh
./target/release/spark-serve "$MODEL" --bind "$REMOTE_BIND" --allow-remote --api-key-file "$API_KEY_FILE"
```

`REMOTE_BIND` is the chosen socket address on that network. Model
administration remains unavailable on the HTTP port.

With `--allow-remote` and `--prefix-cache-scope global` together, the server
prints a warning: cached prompts are then shared across clients.

</details>

## Keep it running

**Run it as a systemd user service or as a `--daemon` background process, not both.**

[Run, update, remove](upkeep.md#keep-it-running) sets up either one.

<details>
<summary><b>Changing the service</b>, and where the daemon writes its files</summary>

- **Service.** To change the model or add flags, edit `ExecStart` in the
  linked local service file, then run `systemctl --user daemon-reload` and
  `systemctl --user restart vade-serve.service`. Follow the log with
  `journalctl --user -u vade-serve.service -f`.
- **Daemon.** `--daemon` detaches from the terminal and writes
  `spark-serve.pid` and `spark-serve.log` in a `vade` folder under the XDG
  state folder: `XDG_STATE_HOME` if set, otherwise the standard default. A
  second daemon is refused while one runs.

</details>

## Stopping

**Each way of running has its own stop command, and all three shut down cleanly.**

<dl>
<dt>In a terminal</dt>
<dd>Ctrl-C</dd>
<dt>As the user service</dt>
<dd><code>systemctl --user stop vade-serve.service</code></dd>
<dt>With <code>--daemon</code></dt>
<dd><code>kill -TERM "$(cat runtime/state/vade/spark-serve.pid)"</code></dd>
</dl>

- The `--daemon` command uses the state folder set up in
  [Run, update, remove](upkeep.md#option-b-a-background-process-without-systemd).
- All three cancel unfinished requests, free the model's memory and log
  `shutdown complete exit=0`.
- A GPU stuck mid-step still exits in about 4.5 s.

## Swap the model

**To change models, restart the server.** Swapping in a running server is unfinished.

### Do it yourself

1. Stop the running server the way it was started; see [Stopping](#stopping).

2. Start it with the new model folder. If it runs as the user service, edit
   `ExecStart` instead, as in [Keep it running](#keep-it-running).

   ```sh
   ./target/release/spark-serve models/Qwen/Qwen3-32B
   ```

3. Wait for `ready`, then check the new model ID with
   `curl -s http://localhost:8080/v1/models`.

### Ask your coding agent

```text
Change the model Vade serves on this DGX Spark by restarting the server, as
docs/server.md "Swap the model" recommends. Stop the running server the way
it was started. Start spark-serve with the new model folder, or edit
ExecStart if it runs as the user service. Wait for "ready" and show the
model ID from /v1/models. Do not use the admin swap socket. Never use sudo.
```

### The swap endpoint, unfinished

**Not ready: a swap keeps the startup memory plan, and a Qwen3-8B to Qwen3-32B swap failed in testing.**

<details>
<summary><b>How the unfinished swap endpoint works</b>, and what does not work yet</summary>

The swap endpoint exists for models under `--models-root`, through a separate
Unix socket:

- Start the server with `--admin-socket PATH` to enable it.
- The socket has permissions 0600, and its parent folder must be private
  (0700).
- A missing parent folder is created. An existing socket is never replaced.
- The HTTP port rejects this route.

```sh
curl -s --unix-socket "$ADMIN_SOCKET" http://localhost/v1/admin/swap \
  -H 'Content-Type: application/json' -d '{"path":"Qwen3-8B"}'
```

Set `ADMIN_SOCKET` to the path passed to `--admin-socket`. The model path is
relative to `--models-root`, or an absolute path beneath it.

What a swap does:

1. Checks the new files before touching the old model.
2. Gives active requests a second to finish.
3. Frees every byte of the old model, then loads the new one.
4. If loading fails, reloads the old model.

Gemma and hybrid models need a fresh server launch to change models. The
limits below describe the Qwen3 swap path.

What does not work yet:

- A swap keeps the memory plan from startup. A replacement of a different
  size can fail, and the server then reloads the old model. A swap from
  Qwen3-8B to Qwen3-32B failed this way in testing.
- Swapping to or from the expert model has not been shown to work.
- If both the swap and the reload fail, the server stays up without a model.
  Requests get HTTP 503 until a later swap succeeds or the server is
  restarted.

</details>

## How it sizes itself

**At start it picks the longest context, the most concurrent requests and a memory ceiling that fit, and prints them on the `ready` line.**

It reads the model's config and the memory Linux reports as available. The
ceiling is a limit, not a reservation: apart from the weights, memory is
taken as requests need it.

<details>
<summary><b>The plan, step by step</b>, and where to read it</summary>

Available memory includes file cache Linux can reclaim.

1. **A ceiling.** It leaves 10 percent of available memory, at least 1 GiB,
   for the system. The rest is the most it will ever allocate.
2. **Context.** It picks the longest conversation that fits, up to the
   model's ceiling in [Long conversations](#long-conversations). Gemma's plan
   reserves its local and global KV cache.
3. **Chats at once.** For Qwen3, with that context, it picks the largest
   number of simultaneous chats that fits, up to 32. Without `--max-active`,
   Qwen3.8-27B uses 16 and Gemma 4 31B uses 32. Chats share one memory pool,
   so each slot does not need room for a full-length conversation.
4. **Prompt and answer limits.** Each defaults to the context minus one
   token. The prompt and answer together must fit within the context. When a
   request leaves out its answer limit, the server uses the space left after
   the prompt, up to the configured answer limit.

Where to read the plan:

- For Qwen3 models, the `ready` line reports `context_tokens`, `input_limit`,
  `output_limit`, `max_active`, `gpu_budget_bytes`, the shared workspace and
  `kv_bytes_per_token`.
- Qwen3.8 and Gemma print a shorter `ready` line. It still includes
  `context_tokens` and `max_active`.
- `/health` reports the memory in use at any moment.

</details>

<details>
<summary><b>A ceiling, not a reservation</b>: how memory is shared, and what one conversation costs</summary>

The server does not grab all of that memory at start:

- Weights load once into GPU memory, through small pinned buffers. Every
  chat shares that one copy.
- Each request reserves room for its own prompt plus its `max_tokens` before
  it starts. If the pool is full, the request waits in the queue.
- Qwen3 and hybrid KV cache grows in blocks as tokens arrive, rather than all
  at startup. It is stored in BF16, the model's own precision.
- The Qwen3 prefix cache can borrow idle session memory. Under pressure, the
  least recently used entries are evicted to admit requests. Shared physical
  pages are charged once. Admission charges real BF16 KV bytes and bounded
  scratch and page slack, not a second FP32-sized KV copy.
- After a burst, the server keeps some freed conversation memory for the next
  one: at most the larger of 2 GB (2,000,000,000 bytes) and four times the
  largest conversation since the reserve was last cleared. Idle pages go back
  to the system after 30 seconds, or sooner under memory pressure.

What one conversation's KV cache costs, worked out from the model's shape
rather than measured on a running process:

| Model | Per token (bytes) | 28,672 tokens (GB) | 40,960 tokens (GB) |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 147,456 | 4.2 | 6.0 |
| Qwen3-32B | 262,144 | 7.5 | 10.7 |
| Qwen3-30B-A3B | 98,304 | 2.8 | 4.0 |

- 40,960 tokens is a full-length conversation.
- The per-token figure is layers × KV heads × head size × 2 (keys and
  values) × 2 bytes. For Qwen3-8B that is 36 × 8 × 128 × 2 × 2.
- Because chats share one pool, a long conversation and many short ones can
  run side by side. Each chat reserves its prompt and answer space within
  that pool.

</details>

## Shared prefill progress

The scheduler caps shared-prefill eligibility at the shared row budget.
A larger remainder continues through ordinary chunked prefill until it fits.
Admission keeps the hybrid adapter's 512-token automatic prefix-cache threshold:
257–512-token prompts neither wait for a prefix owner nor use automatic caching.
A zero shared row budget refuses startup, naming both threshold and budget.
This changes scheduling only, preserving canonical tile boundaries and arithmetic.

## Long conversations

**Context ceilings: 40,960 tokens for Qwen3 (131,072 with YaRN), 262,144 for Qwen3.8-27B, 32,768 for Gemma 4 31B.**

The context is the whole prompt plus the new answer, including earlier
messages and the tokens the chat template adds. Memory can force a shorter
context; the `ready` line shows the actual one. Each model's ceiling is on
[Models](models.md#memory-and-context).

<details>
<summary><b>How long an answer can be</b>, and <b>YaRN</b> for longer Qwen3 chats (off by default; can change answers)</summary>

- When both `max_tokens` and `max_completion_tokens` are left out, the answer
  can run to the end of the remaining context, with thinking on or off.
- The server's `--output-tokens` setting can cap it.
- The model can finish earlier. A stop string, timeout or response-byte limit
  can also end the reply.
- An explicit answer limit that exceeds the service cap or remaining context
  is clamped. A prompt must still leave room for at least one output token.
- A smaller answer limit reserves less conversation memory, which can leave
  room for more concurrent requests.

KV cache and hybrid recurrent state grow on demand. Gemma uses its own local
and global cache layout.

**YaRN, for longer Qwen3 conversations.** `--rope-scaling yarn` turns on
Qwen3's optional YaRN setting: factor 4 with an original context of 32,768
positions, for a ceiling of 131,072 tokens.

- The server still sizes the context to available memory.
- `--context-tokens N` can set a smaller ceiling. On its own, it does not
  enable YaRN.
- YaRN changes how the model handles token positions. It can change answers,
  even on short prompts. It is off by default.
- It applies to Qwen3 only; a Qwen3.8 checkpoint refuses it.

</details>

## Prompt caching

**On by default, with one promise: the same answer as reading the prompt fresh.**

Prompt caching (prefix caching) reuses the KV cache of a prompt beginning
already read.

- To turn it off for the whole server, start with `--prefix-cache off`.
- To run one request fresh and not keep it, send `"cache": false` in that
  request.

<details>
<summary><b>The promise</b>, what is kept, shared servers and saved conversations</summary>

It compares the actual tokens before any reuse.

**The promise:** the same answer as reading the prompt fresh. The merged Qwen3
checks matched cached and fresh replies on the recorded workloads. That is
separate from agreement with the official model; [Validation](validation.md)
lists the measured scope and the remaining disagreements.

What it helps with:

- **Follow-up turns.** Ordinary OpenAI clients send the whole conversation
  again each turn. The earlier prompt is reused. The previous answer and the
  new message are read again.
- **Many chats with the same long system prompt.** Chats that start while
  another is still reading the shared part wait for it, then share its memory
  and read only their own part.
- **Repeated documents or instructions** at the start of a prompt.

What is kept, and for how long:

- The prompt part of each finished request, within the memory budget.
- At most 128 saved entries. The least recently used go first when room is
  needed.
- Anything idle for five minutes is dropped.

On Qwen3.8-27B, the server's code sets the saved-prompt store to one entry of
one byte, so in practice it keeps no finished prompts. Qwen3 and Gemma 4 31B
use the 128-entry store.

`--report-prefix-reuse` adds an `X-Prefix-Reused-Tokens` header to streamed
responses, with the number of tokens the server skipped. It is off by default,
including for saved conversations.

**Shared servers.** With `--api-key-file`, authenticated API requests share
one cache scope; the owner has a separate scope. Credentials never enter caches.
Without that file, `--prefix-cache-scope api-key` uses the complete Authorization
header as a scope label and does not validate credentials. Requests without a
credential header share an anonymous scope. Header values are never logged.

`--prefix-cache-scope global` explicitly shares reuse across clients. A model
swap clears the whole cache in either scope.

**Saved conversations with a secret.** A client can also hold on to one
conversation explicitly:

1. Send a streamed request with `"cache": true`. The response headers include
   `X-Session-Capability`, a secret for this conversation.
2. On the next turn, send the whole conversation again with `"cache": true`
   and `"session": "<that secret>"`. The server reuses the longest beginning
   it holds for that conversation.
3. To branch, add `"fork": true`. The earlier state stays as it was, and the
   response carries a new secret for the branch.

Saved conversations expire after five idle minutes, or sooner when the memory
budget needs room. Keep session secrets private. With the default scope,
continuing one also requires the same `Authorization` header.

</details>

## Speculative decoding

**Drafting proposes a few next tokens cheaply; the model checks them in one pass and keeps only what it would have written itself, so greedy answers do not change.**

This section is the API and the flags. [Speculative decoding](speculative-decoding.md)
explains how it works and the rules for each model.

<dl>
<dt><code>auto</code></dt>
<dd>The default. Strict verification at any chat count, sized by measured cost
and each request's acceptance.</dd>
<dt><code>strict</code></dt>
<dd>The model itself checks every guess. Available draft sources depend on the
model.</dd>
<dt><code>plain</code></dt>
<dd>Never guess. As the server default, it also skips loading any draft
model.</dd>
<dt><code>fast</code></dt>
<dd>Experimental, with a numerical guard. <b>It can change the output.</b></dd>
</dl>

- Under `auto`, requests sharing at least 1,024 cached prompt tokens decode
  plainly. A request's explicit mode overrides this.

<details>
<summary><b>Naming a draft model</b>, and the privacy of proposals</summary>

Which draft each model uses, and when it drafts, is on
[Speculative decoding](speculative-decoding.md#rules-for-each-model). In
short: Qwen3-8B and Qwen3-32B draft from the conversation's own text and the
official `Qwen3-0.6B`; Qwen3-30B-A3B never drafts; Qwen3.8-27B and Gemma 4
31B draft only while one request is active.

For dense Qwen3, the draft's `tokenizer.json`, `tokenizer_config.json`,
`vocab.json` and `merges.txt` must match the main model's byte for byte. A
non-matching automatic candidate is skipped; a non-matching `--draft-model`
is an error. To name the draft explicitly, after the optional download in
[Install](install.md#4-download-qwen3-8b):

```sh
./target/release/spark-serve models/Qwen/Qwen3-8B \
  --draft-model models/Qwen/Qwen3-0.6B
```

[Models](models.md#draft-models) lists the discovery rules.

**Privacy of guesses.** By default, guesses come only from the request's own
conversation. `--draft-global-bank` lets them come from other clients' recent
answers too. That would also let a client infer from timing what others
asked, so it is off by default.

</details>

## Precision and verification

**Weights stay as their makers published them, two opt-in settings can change answers, and agreement with the official model is measured separately and is not perfect.**

- BF16 models multiply in BF16, sum in FP32 and keep the KV cache in BF16.
  Sampling follows the server/model defaults unless request fields override them.
- The two settings, both off by default: the `fast` draft policy and
  [YaRN](#long-conversations). Drafting checks cover `auto` and `strict`
  (not `fast`) against plain decoding, on their supported targets.
- [Validation](validation.md) has the agreement counts, the build each was
  measured on, and every disagreement.

## Flags

**Every flag is optional; set one only when the automatic choice does not fit.**

- The form is `spark-serve [PTX] MODEL_FOLDER [flags]`. Normal use leaves out
  `PTX`, an optional path to an external kernel file used instead of the
  kernels built into the program.
- Each entry gives the flag, what it sets and its default.

<details>
<summary><b>Network and access</b> flags</summary>

**Network**

- `--bind IP:PORT`: where it listens. Default: loopback, port 8080.
- `--allow-remote`: required to listen on anything other than loopback.
  Default: off. Requires `--api-key-file PATH`.
- `--allow-host AUTHORITY`: add an allowed Host; repeat for more entries.
  Default: localhost, loopback and the listening address, with optional
  listening port.
- `--allow-origin ORIGIN`: add an allowed browser origin; repeat for more
  entries. Default: allowed local HTTP origins at the listening port.

**Access**

- `--api-key-file PATH`: authenticate API requests using the file described
  [above](#api-authentication). Default: none on loopback; required with `--allow-remote`.
- `--client-priority-cap N`: highest priority ordinary clients can request,
  0 to 254. Default: 0.
- `--owner-key-file PATH`: owner key, a regular 0600 file containing 64
  lowercase hex digits. Default: none.
- `--admin-socket PATH`: enable model swaps through a private Unix socket.
  Default: disabled.
- `--models-root PATH`: folder that model swaps may load from. Default: the
  model folder's parent.

</details>

<details>
<summary><b>Context and memory</b> flags</summary>

**Context**

- `--context-tokens N`: longest conversation, prompt plus answer. Default:
  model-specific and memory-sized; see [Long conversations](#long-conversations).
- `--rope-scaling yarn`: Qwen3 YaRN, factor 4, up to 131,072 context tokens.
  Default: off; native positions.
- `--input-tokens N`: longest prompt per request. Default: context minus one
  token.
- `--output-tokens N`: longest answer per request; prompt plus answer must fit
  in the context. Default: context minus one token; requests without a limit
  use the remaining context.

**Memory**

- `--max-active N`: chats running at once, 2 to 32. Default: sized at start
  for Qwen3; 16 for Qwen3.8-27B; 32 for Gemma 4 31B.
- `--gpu-budget-mib N`: memory ceiling, in MiB. Default: available memory
  minus 10%, leaving at least 1 GiB.
- `--kv-budget-mib N` and `--session-budget-mib N`: finer memory limits for
  conversation state, in MiB. Default: sized at start.

</details>

<details>
<summary><b>Caching and drafting</b> flags</summary>

**Caching**, see [Prompt caching](#prompt-caching)

- `--prefix-cache on|off`: automatic prompt caching. Default: `on`.
- `--prefix-cache-scope global|api-key`: who shares cached prompts and saved
  conversations. Default: `api-key`.
- `--report-prefix-reuse`: add `X-Prefix-Reused-Tokens` to streamed replies.
  Default: off.

**Drafting**, see [Speculative decoding](#speculative-decoding)

- `--draft auto|plain|strict|fast`: speculative decoding policy. Default:
  `auto`.
- `--draft-model PATH`: select a compatible official draft. Default:
  Qwen3-0.6B, Qwen3.5-0.8B or Google's assistant sibling, depending on the
  target.
- `--mtp`: load Qwen3.8's built-in multi-token prediction head. Default:
  automatic for eligible BF16 without a sibling; the FP8 built-in draft does
  not start yet.

**Draft tuning**

- `--draft-depth N`: hybrid draft ceiling, 0 or 1 to 4. Default: 3;
  0 selects adaptive depth. Copy proposals and confidence gating default on; see
  [Speculative decoding](speculative-decoding.md#hybrid-model-drafts).
- `--draft-k K`: most tokens guessed per step, 1 to 8. Default: 2 for
  Qwen3-8B, 3 for Qwen3-32B, 8 for expert targets (which serve plainly), 4 for
  the Qwen3.8 sibling draft and Gemma's assistant. Hybrid built-in MTP defaults
  to 3; an explicit K overrides that default (built-in capacity is 4). With
  explicit `--draft-depth` and K, the smaller ceiling wins.
- `--draft-global-bank`: allow guesses from other clients' recent answers.
  Default: off.

</details>

<details>
<summary><b>Scheduling and running</b> flags, notes, and two more commands</summary>

**Scheduling**

- `--prefill-chunk N`: prompt tokens read per step while other chats wait,
  1 to 512. Default: 512.
- `--foreground-rows N`: rows one GPU step may compute while a request with
  priority above 0 is running, 1 to 512. Default: 8.
- `--background-rows N`: rows one GPU step may compute otherwise, 1 to 512.
  Default: 48 for Qwen3-8B with a draft, 64 for Qwen3-32B with a draft,
  otherwise 32.

**Running**

- `--default-thinking on|off`: thinking for chat and Responses requests that
  set neither `enable_thinking` nor a reasoning effort. A request's own setting
  always wins. Use `off` for apps and agents that cannot send these fields.
  Default: the chat template's own default (`off` for Gemma 4 and the small
  Qwen3.5 templates, `on` otherwise).
- `--generation-timeout-secs N`: fatal generation deadline, 1 to 3,600 seconds.
  Default and maximum: 3,600. Starts at admission; queue wait has a separate
  bound of the same length. Expiry returns 504 or a terminal stream error.
- `--output-bytes N`: largest response body, 1 KiB to 16 MiB. Default: sized
  from the answer limit, at least 1 MiB.
- `--daemon`: detach and write a PID file and a log; see
  [Keep it running](#keep-it-running). Default: off.
- `--device N`: CUDA device number. Default: 0.

**Notes**

- Without an owner key, all `--max-active` slots are open to ordinary
  clients. With one, see "How priority is granted" under
  [Use it from an app](#use-it-from-an-app).
- Automatic background rows are capped by active slots times `K+1`, where
  `K` is the draft depth.
- Explicit row and draft settings survive model swaps. Automatic defaults are
  recalculated for the new model.
- Gemma 4 31B uses fixed scheduling settings: prefill chunk 512, foreground
  rows 1, background rows 32.
- Qwen3.8 refuses `--draft fast` and `--draft-global-bank`. Gemma refuses
  `--draft fast`. For Qwen3.8, `--draft strict` needs a draft: a sibling
  folder, `--draft-model` or `--mtp`.

**Two more commands**

- `spark-serve --list MODELS_ROOT` lists installed models and estimates
  whether each fits in memory, without loading weights. On the Spark it reads
  free memory through the CUDA driver.
- `spark-serve` with no arguments prints its usage.

The flags `--seqpar-peer`, `--seqpar-listen` and `--seqpar-threshold` exist for
helper Sparks, but this server cannot use them yet; see
[Multiple Sparks](multi-spark.md#not-done-yet).

</details>

## Limits on every request

**Fixed limits protect the server: 4,096 messages, 64 tools and 32 connections, among others.**

<details>
<summary><b>Every limit</b>, and the HTTP rules</summary>

| Limit | Value |
| --- | --- |
| Request headers | 16 KiB |
| Messages | 4,096 |
| Tools | 64 |
| Stop strings | 4 |
| Bytes per stop string | 1,024 |
| Response body, default | At least 1 MiB |
| Time per generation | 3,600 s |
| Idle read or write | 15 s |
| Request connections | 32 |
| Spare connections for health and cancel, once requests fill the 32 | 8 |

- **Request body:** the larger of 256 KiB and 16 bytes per allowed input
  token, capped at 4 MiB.
- **Response body:** `--output-bytes` accepts 1 KiB to 16 MiB. Streaming counts
  decoded output against this limit; SSE envelopes and heartbeats do not consume
  it. Pending socket writes remain bounded to 1 MiB, with terminal space reserved.
- **Time per generation:** `--generation-timeout-secs` sets it, 1 to 3,600
  seconds; default and maximum are 3,600. Expiry is a terminal error.

HTTP rules:

- HTTP 1.1, one request per connection, with a `Content-Length` body.
- POST requests require `Content-Type: application/json`.
- Host must match localhost, a loopback address or the listening address,
  with an optional listening port. Wildcard bind addresses are excluded.
- Browser origins must match an allowed origin. `--allow-host` and
  `--allow-origin` extend those lists.
- Chunked uploads, compressed bodies and `Expect: 100-continue` are rejected.
  curl adds that last header to some large uploads; pass `-H 'Expect:'` to
  turn it off.

</details>

**Where next:** [Models](models.md) · [Speculative decoding](speculative-decoding.md) ·
[Run, update, remove](upkeep.md) · [Documentation index](README.md)
