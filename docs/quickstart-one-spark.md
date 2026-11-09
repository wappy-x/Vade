# One Spark: a first answer through the gateway

This page takes one DGX Spark from installed to a first answer through the
gateway. You run the commands on a controller. A controller is a second Linux
or macOS computer that drives the Spark over SSH with the `vade` command.

To work on the Spark itself, with no second computer, use the
[quick start on one Spark](INSTALL.md#quick-start-on-one-spark):
`curl -fsSL https://vade.lol/install.sh | sh`, `vade pull`, then `vade up`.

<a id="part-1-fp4-local-from-a-controller"></a>

## FP4-Local from a controller

On this page you start the engine on the Spark, check it, put the gateway in
front of it, and send a first chat through the gateway. Apart from one last
check on the Spark, you run everything from the controller.

### Before you begin

First finish [INSTALL steps 1–3](INSTALL.md) and the setup at the start of
[Route B](INSTALL.md#route-b-from-a-controller). By then you have:

1. checked the Spark and the SSH prerequisites
2. installed the pinned compilers
3. verified the complete checkpoint
4. cloned and built Vade on the controller, and created the private config

Keep that controller shell open. Run the commands below from the repository
root, with `VADE_CONFIG` set to your private config and the built CLI on your
`PATH`.

### Install and start the engine

```sh
vade install --dry-run
vade install
vade up --fp4-local --dry-run
vade up --fp4-local
vade status
vade doctor
```

As in [INSTALL](INSTALL.md#install-and-start), each `--dry-run` line
previews the command after it without changing anything. Here is what each
real command does:

- `vade install` copies your tracked source to the configured Spark and builds
  it there. It does not download weights.
- `vade up --fp4-local` starts the engine on the Spark.
- `vade status` prints how many chats at once the engine accepted.
- `vade doctor` runs the machine checks plus the harness doctor.

From the controller, `vade up` asks for a context of 32768 tokens and up to
min(context, 8192) output tokens, the same defaults as `vade launch` on the
Spark. A controller never lowers that context for you: if the Spark lacks the
memory, the engine refuses it, and you choose a smaller `--context`. Keep it
at 8192 or more if you want `vade chat`. (On the Spark itself, a local launch
lowers a default or saved context, but only when the engine refuses it for
lack of memory.) See
[Defaults on a controller](INSTALL.md#defaults-on-a-controller).

If startup fails, see [launcher recovery](../cli/README.md#recovery).

### Check that the engine is ready

`vade up` opens an SSH tunnel, so the engine answers on your controller at
`http://127.0.0.1:18080/v1`. Check its health, and the model and limits it
reports:

```sh
curl -qfsS http://127.0.0.1:18080/health
curl -qfsS http://127.0.0.1:18080/v1/capabilities
```

Wait until `/health` reports `"status":"ready"` and its checkpoint section shows
`hash_state: ready`, which means the engine has finished hashing the checkpoint
files. Until then, repeat the two commands.

### Check that answers are exact

Exactness means the same prompt, model and settings give the same answer,
token for token, on every run and every Spark. Vade's default is greedy decoding: it
always picks the most likely next token. Drafting (guessing tokens ahead) can
make the answer arrive sooner, but it must not change it.

With the engine still running, run `vade verify`. It:

- sends the short bundled prompt 20 times with drafting off and 20 times
  with drafting on, asking for exactly 128 tokens each time, with the prompt
  cache and thinking off
- compares the token IDs of every run
- prints one result line: `IDENTICAL: ...` when every run matches, or
  `DIFFERENT at run ... token ...` at the first mismatch
- exits 0 only when the runs are identical (1 means different or incomplete)
- saves the receipt `vade-verify.json` in the current folder

The drafting-on runs must actually draft. If drafting is unavailable, the
check fails rather than passing on plain decoding.

Move that receipt out of the checkout before you run `vade install` again,
because install refuses a clone with untracked files.

A pass covers the prompt, model, output length and machines recorded in that
receipt. To test more widely:

| To | Run |
| --- | --- |
| Repeat the prompt more times | `vade verify -n 50` |
| Check longer answers | `vade verify --max-tokens 256` |
| Use a longer bundled prompt | `vade verify --prompt 2k` or `vade verify --prompt 8k` |
| Compare with a second running engine | Add `--machine <name>` (for example `spark-b`) after you give that machine a `verify_url` in `VADE_CONFIG`; see [Compare with a second engine](../cli/README.md#compare-with-a-second-engine) |

### Put the gateway in front

The gateway sits between your client and the engine. Your client only has to
change its base URL, which is why this is called "invisible" mode.
[Standalone gateway](../harness/docs/invisible-gateway.md) covers what it
forwards and the optional `--greedy-when-unset` policy. That policy adds
greedy defaults only when a request sets no sampling field. The FP4-Local
engine already uses greedy decoding for those requests.

Start the gateway in a second controller terminal, from the same checkout. If
`harness/target/release/vade-gateway` is not there yet, first build it as
shown in [Client gateway](INSTALL.md#client-gateway).

```sh
harness/target/release/vade-gateway \
  --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

### Send a first chat

In a third controller terminal, send a chat, then a streaming chat:

```sh
curl -qfsS http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp4-local","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
curl -qfsSN http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp4-local","messages":[{"role":"user","content":"Explain why ice floats in one sentence."}],"max_tokens":96,"chat_template_kwargs":{"enable_thinking":false},"stream":true}'
```

What you should get back:

- The first response has the assistant's message in
  `choices[0].message.content`.
- The stream is a series of `data:` events that ends with `data: [DONE]`.

The Chat Completions endpoint does not remember history for you. A follow-up
must include the previous user message, then the **actual** assistant reply,
then the new user message.

To chat interactively instead, run `vade chat`.
[Steering](#steering-and-keeping-the-engine-ready) explains what happens when
you type while a reply is running.

### Connect Codex (optional)

To set up Codex against Vade, see
[the client setup guide](../harness/docs/passthrough-matrix.md). With
`--target gateway`, the setup writes Codex settings to a separate folder, so
your usual Codex settings are left alone. It fills that folder with the model
and limits the running engine reports, and points Codex at
`http://127.0.0.1:8787/v1`. The gateway forwards requests without converting
one API format into another.

### Stop everything

Keep the checkout and `VADE_CONFIG` in place until all three steps are done.
Shut down in this order:

1. Stop the client.
2. Press **Ctrl-C in the gateway terminal**.
3. In the first controller terminal, where `VADE_CONFIG` is set, run
   `vade down`:

```sh
vade down
```

You should see `Vade is stopped. The engines and the tunnel that vade started are closed.` Then,
on the Spark, `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`
must show no row for that server.

### Steering and keeping the engine ready

`vade chat` attaches to the engine that `vade up` started, and leaving chat
keeps that engine loaded. So you can open another chat without waiting for the
model to load or for the GPU kernels to compile (JIT) again. Use `vade down`
when you want the GPU back.

By default, sending a message while a reply is running interrupts that reply.
This is called steering. Here is what happens:

- Vade keeps the partial reply you saw in the conversation, marked
  `[interrupted]`.
- With Vade's engine, chat sends that request's private cancel token to
  `/v1/cancel`. This works even before the first token appears.
- Once the cancel goes through, chat sends your new message, with the partial
  reply as context.
- If the cancel is not acknowledged, Vade holds your steering message and
  reports the failure.

If chat is pointed at a server other than Vade's engine, that server only sees
the connection close, and it may keep generating the reply anyway.

To change this, use these chat commands:

| Command | What it does |
| --- | --- |
| `/queue on` | Lets replies finish before your next typed messages go out |
| `/queue off` | Goes back to steering |
| `/queue TEXT` | Queues one follow-up |
| `/exit` | Cancels an active reply, then closes chat; the engine keeps running |

`vade chat` opens either the native agent (Vade's built-in coding agent) or a
direct chat with the model. The agent saves interrupted replies in its private
session journal. Direct chat keeps its history in memory only.

### Starting again faster

Running `vade up` again with the same settings checks the healthy server and
reuses it.

Vade keeps the compiled GPU kernels (the CUDA JIT caches) between `vade down`
and `vade up`. There is one cache for each hash of the kernel files (PTX), so a
later start with the same kernels does not compile them again.

To skip compiling on the Spark altogether, point `[global] prebuilt_dir` at a
trusted local bundle root (a folder of prebuilt bundles):

- A bundle is used only if its source revisions and SHA-256 hashes match
  exactly.
- If a bundle is missing, Vade builds from source instead.
- Vade checks the model files while it copies and builds, and skips the check
  when nothing has changed.

[Prebuilt bundles](../cli/README.md#prebuilt-bundles) in the CLI guide
describes the bundle layout.

<a id="qwen3-8b-bf16-ubuntu-to-a-first-answer"></a>
<a id="part-2-older-route-build-from-source-with-qwen3-8b"></a>
<a id="0-prerequisites-before-downloads"></a>
<a id="1-install-the-pinned-rust-tools"></a>
<a id="2-get-the-source-and-build-the-engine"></a>
<a id="3-download-and-verify-the-official-small-model"></a>
<a id="4-start-the-server"></a>
<a id="5-first-chat-then-the-same-answer-twice"></a>
<a id="6-stop-and-record-the-result"></a>
<a id="7-cli-chat-optional"></a>
<a id="prefill-check-and-qualification-limits"></a>

## The older Qwen3-8B route

The route that builds the engine from source and serves Qwen3-8B BF16 is now in
the [history pages](history/quickstart-one-spark-qwen3-8b.md).

## Extra checks and troubleshooting

### Containers: large model refused despite free RAM

In a container or VM (LXC, for example), the server can think memory is short.
The container's MemAvailable can leave out the host's reclaimable cache, and
that makes the server's estimate come out low. This section explains the
refusal message and how to set a budget by hand.

When the server turns down a size, the message starts with
**`Requested size does not fit.`**, followed by `Asked:`, `Available:`, `Gap:`
and `Next:` lines:

- Byte totals are in binary GiB (1 GiB = 1073741824 bytes), with the exact
  bytes beside them.
- The `Asked:` line lists what it knows: weights per rank (one engine process
  per Spark), memory reserved per session and the number of sessions, CONTEXT
  and OUTPUT. When a refusal comes from deeper in the server, the line says
  which of these values it was not given.

For example, when the server refuses the session pool it negotiated, the
message includes:

```text
Requested size does not fit.
Asked: <weights/rank; reservation/session × sessions; CONTEXT; OUTPUT; fixed/cache terms>; total 18.48 GiB (19847557088 bytes).
Available: 6.92 GiB (7425823856 bytes) from <budget source and caps>.
Gap: 11.57 GiB (12421733232 bytes).
Next: 1. Lower <supported controls>. 2. In containers, spark-serve supports --gpu-budget-mib N; size N by rounding up (weights + shared workspace + session reservations + retained cache + headroom) / 1048576, within verified host capacity. Flash uses SEQPAR_MEMORY_BYTES in bytes; its CUDA safety cap still applies. 3. Use more Sparks with supported tensor parallelism.
```

The server prints `Next:` as one long line. It offers three options:

1. Lower the controls it names.
2. In a container, give `spark-serve` a budget with `--gpu-budget-mib N`. Work
   out N by adding up the bytes for weights, shared workspace, session
   reservations, retained cache and headroom, dividing by 1048576 (bytes per
   MiB) and rounding up. Stay within the host capacity you have checked. Flash
   takes `SEQPAR_MEMORY_BYTES` in bytes instead, and its CUDA safety cap still
   applies.
3. Use more Sparks with supported tensor parallelism.

The bracketed fields change from one refusal to the next, and each server
names its own controls. The
[older Qwen3-8B route](history/quickstart-one-spark-qwen3-8b.md) runs
`spark-serve`. The FP4-Local engine on this page is Flash (`flash-serve`), and
`tp-serve` is the tensor-parallel server:

| Refused by | Controls it names |
| --- | --- |
| `spark-serve` | `--context-tokens`, `--output-tokens`, `--max-active` |
| Flash-Next FP4-Local | CLI `--context` and `--output`, or the native server's positional context and `FLASH_SERVE_OUTPUT_TOKENS`; memory admission chooses the chat count |
| Flash-Next FP8 | The same context and output controls, plus `--chats` or `FLASH_SERVE_MAX_ACTIVE` |
| `tp-serve` | Positional `CONTEXT`, `TP_SERVE_MAX_ACTIVE` |
| A logical cap on the KV cache (which holds each chat's attention state) or on sessions | `--kv-budget-mib`, `--session-budget-mib` |

Token refusals use the same first line, but report a token gap and say
`GiB gap is not applicable to a token ceiling.` A request past the configured
context names `vade --context` in that refusal. To fix it, shorten the prompt
or lower `max_tokens`. To raise the server's context, restart it with the flag
for the route you used. On this page's route, run `vade down` and then
`vade up --context 262144` (or choose a smaller supported context). On the
older route, stop the foreground server with Ctrl+C and restart `spark-serve`
with `--context-tokens N`, where N fits that checkpoint and the available
memory.
Memory overrides and extra Sparks cannot raise the checkpoint's token ceiling.

**How the server works out its memory:**

- It counts CUDA's free memory plus what Linux could reclaim from its file
  cache (MemAvailable − MemFree, if positive), capped at MemAvailable and the
  CUDA total.
- `spark-serve` keeps 10% back, at least 1 GiB.
- Flash keeps 10% back, at least 2 GiB, then sets aside memory for shared
  buffers and graphs, plus a 512 MiB reserve for per-step buffers. Across
  ranks it plans with the smallest pool and the largest session reservation.
- Any limit you set explicitly also caps it.

When you supply these inputs, the message prints the actual values.

Check the host's capacity before you choose N, because the override does not
create memory, and Flash still applies its CUDA safety cap. Real CUDA errors
are reported separately. See
[memory limits](../engine/docs/server.md#how-it-sizes-itself).
