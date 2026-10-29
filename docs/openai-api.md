# OpenAI-compatible API

Vade serves the OpenAI Chat Completions API, so most OpenAI clients work
without code changes. Set the client's base URL to Vade, and set the model to
the ID that `GET /v1/models` returns. This page tells you which routes and
fields work, what the replies contain, and where Vade differs from OpenAI.

Vade also serves the Responses and Anthropic Messages APIs. The [engine server
reference](../engine/docs/server.md) covers them.

## Base URLs

Use one of these base URLs. Each one listens on loopback only.

| Base URL | What answers | Start it with |
| --- | --- | --- |
| `http://127.0.0.1:18080/v1` | The engine | `vade launch` on the Spark, or `vade up` from a controller ([INSTALL.md](../INSTALL.md)) |
| `http://127.0.0.1:8787/v1` | The core gateway, in front of the engine | `vade-harness gateway --upstream http://127.0.0.1:18080` |
| `http://127.0.0.1:8787/v1` | The standalone gateway, in front of the engine | `vade-gateway --upstream http://127.0.0.1:18080/v1` |

The two gateways are described in the [harness README](../harness/README.md)
and in [Standalone gateway](../harness/docs/invisible-gateway.md). Run only
one gateway on port 8787 at a time.

The FP4-Local server serves the model ID `qwen3.8-flash-next-fp4-local`. Read
the ID from your own server, because another checkpoint has another ID.

## Try it with curl

These commands use the engine address. To go through a gateway, change the
port to 8787.

```sh
BASE=http://127.0.0.1:18080/v1
curl -sS "$BASE/models"
curl -sS "$BASE/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-fp4-local",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "enable_thinking": false
}'
curl -sSN "$BASE/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-fp4-local",
  "messages": [{"role": "user", "content": "Say hello."}],
  "max_tokens": 64,
  "stream": true
}'
```

FP4-Local thinks by default. Without `"enable_thinking": false`, the reply
gives the thinking in `reasoning_content` first, then the answer in `content`.
[examples/curl-openai.sh](../examples/curl-openai.sh) runs the same requests,
and [examples/README.md](../examples/README.md) shows the output of a real run.

## Use the openai Python client

This example is documentation only. The repository contains no Python code.
Install the client in your own environment with `pip install openai`.

```python
from openai import OpenAI

# Vade needs no key on loopback. The client needs a value, so any text works.
client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="unused")
model = client.models.list().data[0].id

# One answer.
reply = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Say hello."}],
    max_completion_tokens=64,
    extra_body={"enable_thinking": False},
)
print(reply.choices[0].message.content, reply.usage.total_tokens)

# A streamed answer. Usage arrives on the last chunk.
for chunk in client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Count to five."}],
    stream=True,
    extra_body={"enable_thinking": False},
):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
print()

# Function calling. Vade never runs a tool; your code runs it.
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
messages = [{"role": "user", "content": "What is the weather in Paris?"}]
reply = client.chat.completions.create(model=model, messages=messages, tools=tools)
message = reply.choices[0].message
if message.tool_calls:
    messages.append(message)
    for call in message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": call.id, "content": '{"temp_c": 18}'})
    reply = client.chat.completions.create(model=model, messages=messages, tools=tools)
print(reply.choices[0].message.content)
```

`extra_body` sends Vade's own fields, such as `enable_thinking`, which the
client does not know.

## Routes

| Route | Engine | Core gateway | Standalone gateway |
| --- | --- | --- | --- |
| `POST /v1/chat/completions` | Served | Forwarded | Forwarded |
| `POST /v1/completions` (legacy) | Served | Forwarded | Forwarded |
| `GET /v1/models` | Served | Forwarded | Forwarded |
| `POST /v1/embeddings` | 404 | 501 with a reason | Forwarded, so 404 |
| `GET /v1/models/{id}` | 404 | 404 | Forwarded, so 404 |
| Audio, images, files, batches, moderations | 404 | 404 | Forwarded, so 404 |

Vade is a text generation server. It has no embedding model, so it cannot
serve `/v1/embeddings`. The core gateway answers that route with HTTP 501 and
this error:

```json
{"error":{"type":"invalid_request_error","message":"embeddings are not implemented: Vade serves text generation only; use a separate embedding server","param":null,"code":"unsupported_endpoint"}}
```

`GET /v1/models` lists one model. Its `id` is the served model ID. It also
gives `context_length`, `max_model_len`, `max_output_tokens` and
`input_modalities`. It has no `created` field.

## Chat request fields

| Field | What Vade does |
| --- | --- |
| `model` | Optional. If you send it, it must equal the served model ID, or the engine returns 400 `unknown model`. |
| `messages` | 1 to 4,096 messages. Roles `system`, `developer` (read as `system`), `user`, `assistant` and `tool`. |
| `content` | A string, or an array of `{"type": "text"}` parts, which Vade joins in order. Image parts return 400. An assistant message with tool calls can send `null`. |
| `name` on a message | Not used by the FP4-Local chat template. |
| `stream` | `true` sends server-sent events. The default is `false`. |
| `stream_options` | Accepted and ignored. Usage is always on the last chunk. |
| `max_tokens`, `max_completion_tokens` | Positive integers. `max_completion_tokens` wins when you send both. Vade clamps the value to the server's output limit and to the room left after the prompt. |
| `stop` | A string, or up to 4 strings. Each string must have 1 to 1,024 bytes. |
| `n` | Only 1. It counts as a sampling field, even as 1; see [Sampling and exactness](#sampling-and-exactness). |
| `tools` | Up to 64 function tools. |
| `tool_choice` | `auto` (the default), `none`, `required`, or `{"type": "function", "function": {"name": ...}}`. |
| `parallel_tool_calls` | `false` allows one call per reply. The default is `true`. |
| `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Used as sent. See [Sampling and exactness](#sampling-and-exactness). |
| `response_format` | Ignored. See [Structured output](#structured-output). |
| `min_p`, `typical_p`, `logit_bias`, `best_of` | Ignored by the engine. They still count as sampling fields. |
| `logprobs`, `top_logprobs`, `user`, `metadata`, `store` | Ignored. |
| `enable_thinking`, `reasoning_effort`, `chat_template_kwargs` | Vade fields that turn thinking on or off. See the [engine server reference](../engine/docs/server.md). |

The engine ignores fields that it does not know. A field that is ignored
changes nothing in the reply.

## Replies

A non-streamed reply is one `chat.completion` object. Its message has
`role: "assistant"`, `content` and `"refusal": null`. It also has
`reasoning_content` when the model thought, and `tool_calls` when the model
called a tool.

A streamed reply is a series of `data:` lines. Each line holds one
`chat.completion.chunk`. Each chunk's `delta` holds `content`,
`reasoning_content` or `tool_calls`. Some deltas are empty. The last chunk has
`finish_reason` and `usage`. Then `data: [DONE]` ends the stream. This is the
end of a real stream, from [examples/README.md](../examples/README.md):

```text
data: {"id":"completion-2","object":"chat.completion.chunk","created":1792935961,"model":"qwen3.8-flash-next-fp4-local","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":24,"completion_tokens":10,"total_tokens":34,"prompt_tokens_details":{"cached_tokens":0}},"vade":{…}}
data: [DONE]
```

Vade's chunks differ from OpenAI's in four ways. Clients that follow the
OpenAI schema read them correctly.

- No delta carries `role`. OpenAI sends `"role": "assistant"` in the first
  delta. See [Streamed tool calls in history](#streamed-tool-calls-in-history).
- Usage is on the last chunk, with its `finish_reason`. OpenAI sends usage in
  an extra chunk with no choices, and only when `stream_options.include_usage`
  is `true`.
- Each choice can carry `token_id`, the ID of the token that produced it.
- Every reply carries a `vade` object with the sampling settings, drafting and
  build identity that served it. Clients can ignore it.

### Usage

`usage` has `prompt_tokens`, `completion_tokens` and `total_tokens`.
`prompt_tokens_details.cached_tokens` counts the prompt tokens that Vade read
from its prefix cache instead of computing them again.

### Finish reasons

| `finish_reason` | Meaning |
| --- | --- |
| `stop` | The model ended its answer, or the reply reached a `stop` string. |
| `length` | The reply reached its token limit. |
| `tool_calls` | The reply ends with one or more tool calls. |

Vade does not send `content_filter` or `function_call`.

## Tool calling

Send function tools in the OpenAI shape. The model decides when to call one,
and Vade returns each call to your client. Vade never runs a tool.

In a stream, a call starts with its index, ID, type and name. Its arguments
then arrive in parts as the model writes them:

```text
data: {…"delta":{"tool_calls":[{"index":0,"id":"call-1234567-0","type":"function","function":{"name":"get_weather","arguments":""}}]}…}
data: {…"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\": "}}]}…}
data: {…"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"Paris\"}"}}]}…}
data: {…"delta":{},"finish_reason":"tool_calls"…}
data: [DONE]
```

- The tool schema guides the model through the prompt. Vade does not force
  the arguments to match the schema, so check them before you run the tool.
- A malformed or incomplete call comes back as ordinary `content`, with HTTP
  200.
- To send the result, add the assistant message with its `tool_calls`, then a
  `tool` message with the same `tool_call_id`.

### Streamed tool calls in history

The openai Python client can build a message from a stream, for example with
`client.chat.completions.stream(...)` and `get_final_completion()`. That
message takes its role from the deltas. Vade's deltas have no role, so the
message has `role: null`. If you add that message to `messages` and send it
back, the engine refuses the request with HTTP 400.

Both gateways repair this. On the chat routes, they change a message role of
`null` to `"assistant"` before they forward the request. When you call the
engine directly, set the role yourself, for example
`{**message.model_dump(), "role": "assistant"}`.

## Structured output

Vade has no constrained decoding mode. The engine ignores `response_format`,
so `{"type": "json_object"}` and `{"type": "json_schema", ...}` do not force
JSON. The request still succeeds, and the reply is ordinary text.

To get JSON, ask for it in the prompt, then parse and check the reply in your
code. Turn thinking off with `"enable_thinking": false` so the answer starts
at once.

## Sampling and exactness

When a request sends no sampling field, Vade decodes greedily: it always picks
the most likely token. Greedy is the default, and it is exact: the same request
gives the same token IDs on every run. Vade's exactness checks use greedy
requests.

The gateways make that default explicit. When a request has none of the
fields `temperature`, `top_p`, `top_k`, `min_p`, `typical_p`, `seed`,
`frequency_penalty`, `presence_penalty`, `repetition_penalty`, `logit_bias`,
`n`, `best_of`, `sampling` or `generation_config`, a gateway adds
`{"temperature":0,"seed":20260905,"top_p":1.0,"top_k":0,"repetition_penalty":1.0,"presence_penalty":0.0,"frequency_penalty":0.0}`
before it forwards the request. It changes no other byte.

When a request sends any one of those fields, the gateways forward it
unchanged. The engine then takes each missing field from the checkpoint's
`generation_config.json`. For FP4-Local that means temperature 1, top-k 20 and
top-p 0.95 (see [the recorded completions
fixture](../harness/tests/fixtures/completions-engine-2026-10-23.md)). So a
client that sends only `top_p` gets a sampled reply.

- To keep replies exact, send `"temperature": 0`, or send no sampling field.
- With `temperature` above 0, the engine draws a random seed for each request
  unless you send `seed`. A streamed reply reports the seed in the
  `X-Sampling-Seed` header. Every reply reports it in `vade.sampling.seed`.
- Many chat apps send `temperature`, and some send `"n": 1`, by default. Set
  `temperature` to 0 in the app if you want exact replies.

## Errors

Errors use the OpenAI shape, `{"error": {"type": ..., "message": ...}}`.
Errors that a gateway writes itself also carry `param` and `code`.

| Status | `type` | When |
| --- | --- | --- |
| 400 | `invalid_request_error` | A field breaks the rules above, or `model` is not the served model ID. The message says what is wrong. |
| 401 | `authentication_error` | A key is required and the request has no key or the wrong key. The `code` is `invalid_api_key`. |
| 403 | `permission_error` | A gateway refused a browser `Origin` that is not on its allow list. |
| 404 | `not_found_error` | The route does not exist. |
| 409 | `client_cancelled` | The client cancelled the request. |
| 413 | `request_too_large` | The request body is larger than 8 MiB (gateway limit). |
| 415 | `invalid_request_error` | The body is not `application/json`, or it is compressed. |
| 421 | `invalid_request_error` | A gateway refused a `Host` header that is not loopback. |
| 500 | `generation_error` | Generation failed. |
| 501 | `invalid_request_error` | The core gateway does not serve this route (`/v1/embeddings`). The `code` is `unsupported_endpoint`. |
| 502 | `server_error` | A gateway could not reach the engine. |
| 503 | `overloaded` | The engine is loading, or all its connections are busy. Try again. Through a gateway, the reply has `Retry-After`. |
| 504 | `generation_deadline` | The request reached its time limit. |

## API keys

Keys are off by default, and every listener binds to loopback only.

- **Engine.** `--api-key-file PATH` makes the engine require
  `Authorization: Bearer KEY` or `X-Api-Key: KEY`. The engine requires a key
  file when you start it with `--allow-remote`.
- **Gateways.** `--token TOKEN`, or the `VADE_GATEWAY_TOKEN` environment
  variable, makes a gateway require `Authorization: Bearer TOKEN`. The gateway
  does not forward that header. Use the environment variable, so the token
  does not show in the process list.
- **No key set.** Clients can send any key, or none. A gateway then forwards
  the client's `Authorization` header unchanged, and an engine without a key
  file ignores it.

The openai Python client always sends `Authorization: Bearer <api_key>`. Set
`api_key` to the gateway token when the gateway requires one.

## Client notes

The conformance test replays requests from these clients (see [How this page
is tested](#how-this-page-is-tested)). It marks each request as captured from
the real client or written from memory.

**openai Python client.** Request bytes were captured from version 3.41.0.
That version read every reply shape on this page, including streamed tool
calls with `client.chat.completions.stream`. It turns the 501 embeddings
error into an `InternalServerError` with the message above.

**Continue and Open WebUI.** Their requests in the fixture are written from
memory, not captured, and neither client has been run against Vade.
`vade-harness setup continue` and `vade-harness setup open-webui` print their
settings; [Other clients through the
gateway](../harness/docs/passthrough-matrix.md#aider-continue-and-open-webui)
explains them.

- Continue can send content as text parts, which Vade accepts. Leave tab
  autocomplete off: the engine ignores the fill-in-the-middle `suffix`, so
  autocomplete would only continue the text before the cursor.
- Open WebUI reads the model list, then streams chats. The gateways accept
  connections from loopback only. On Linux, a container on a bridge network
  reaches the host through the bridge address, which is not loopback; run Open
  WebUI with host networking there, or outside a container. This Linux note
  is not tested.

**Cursor.** Not tested. Cursor sends requests for a custom base URL from its
own servers, so it cannot reach a loopback address.

## Known gaps

- `response_format` is not enforced. There is no constrained decoding mode.
- `/v1/embeddings` is not served.
- Streamed deltas carry no `role`.
- An unknown `model` returns 400 `unknown model`. OpenAI returns 404
  `model_not_found`.
- `GET /v1/models/{id}` is not served, and the model list has no `created`.
- `/v1/completions` takes one string prompt or one array of token IDs. An
  array of strings returns 400. `suffix`, `echo` and `logprobs` are ignored.
- `logprobs` is not returned.

## How this page is tested

[`harness/src/gateway_conformance.rs`](../harness/src/gateway_conformance.rs)
replays the requests in
[`openai-clients.json`](../harness/tests/fixtures/openai-clients.json)
through the core gateway and the standalone gateway, to a mock engine. For
each request it checks that the engine receives the expected bytes, that the
client receives the engine's reply byte for byte, and that the reply gives the
client its text, tool calls, finish reason, usage, model IDs or error.

- Each case is marked `verified`, `from memory` or `from docs`. In this
  version, 9 cases come from openai-python 3.41.0, 5 from memory (Continue and
  Open WebUI), and 1 from the curl example on this page.
- The engine replies in the fixture are written by hand in the engine's wire
  format. They were not recorded from a live engine. The test runs on the host
  and does not run the model.

Run it from `harness/`:

```sh
cargo +1.99.0 test --offline --lib gateway::conformance
```
