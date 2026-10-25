# One Spark: a first answer through the gateway

This page takes one DGX Spark from installed to a first answer. It has two
parts:

| Part | Model | Where you run the commands |
| --- | --- | --- |
| [Part 1](#part-1-fp4-local-from-a-controller) | **Qwen3.8-Flash-Next-FP4-Local** | On a controller: a second Linux or macOS computer that drives the Spark over SSH with the `vade` CLI |
| [Part 2 (older route)](#qwen3-8b-bf16-ubuntu-to-a-first-answer) | Qwen3-8B BF16, a different model | Directly on the Spark |

Working on the Spark itself, with no second computer? Use
[Route A in INSTALL](../INSTALL.md#route-a-on-this-spark) instead of Part 1.

## Part 1: FP4-Local from a controller

In this part you start the engine on the Spark, check it, put the gateway in
front of it, and send a first chat through the gateway. Apart from one last
check on the Spark, you run everything from the controller.

### Before you begin

First finish [INSTALL steps 1–3](../INSTALL.md) and the setup at the start of
[Route B](../INSTALL.md#route-b-from-a-controller). By then you have:

1. checked the Spark and the SSH prerequisites;
2. installed the pinned compilers;
3. verified the complete checkpoint;
4. cloned and built Vade on the controller, and created the private config.

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

As in [INSTALL](../INSTALL.md#install-and-start), each `--dry-run` line
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
memory, the engine refuses it, and you choose a smaller `--context`; keep it
at 8192 or more if you want `vade chat`. (On the Spark itself, a local launch
lowers a default or saved context, but only when the engine refuses it for
lack of memory.) See
[Defaults on a controller](../INSTALL.md#defaults-on-a-controller).

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

Exactness means the same prompt and model give the same answer, token for
token, on every run and every Spark. Vade's default is greedy decoding: it
always picks the most likely next token. Drafting (guessing tokens ahead) can
make the answer arrive sooner, but it must not change it.

With the engine still running, run `vade verify`. It:

- sends the short bundled prompt 20 times with drafting off and 20 times
  with drafting on, asking for exactly 128 tokens each time, with the prompt
  cache and thinking off;
- compares the token IDs of every run;
- prints one result line: `IDENTICAL: ...` when every run matches, or
  `DIFFERENT at run ... token ...` at the first mismatch;
- exits 0 only when the runs are identical (1 means different or incomplete);
- saves the receipt `vade-verify.json` in the current folder.

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
forwards and the greedy defaults it adds when a request sets no sampling field.

Start the gateway in a second controller terminal, from the same checkout. If
`harness/target/release/vade-gateway` is not there yet, first build it as
shown in [Client gateway](../INSTALL.md#client-gateway).

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

To chat interactively instead, run `vade chat`;
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

You should see `Vade: stopped. Its owned ranks and tunnel are closed.` Then,
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

## Part 2 (older route): build from source with Qwen3-8B

This is a **legacy route**. The supported model in v0.1.0 is
nvidia/Qwen3.8-Flash-Next-FP4-Local (Part 1). We keep this part because it
shows how to build the engine from source. It uses the small Qwen3-8B model,
and its download and disk notes apply only to that model. (Looking for
Flash-Next FP8 on two Sparks? See
[the legacy FP8 installation](install-legacy-fp8.md), which has its own
checkpoint helper.)

In this part you build Vade, serve the official Qwen3-8B BF16, and check that
the same request gives the same answer twice.

- Run these commands **on the Spark**, in Bash. A Mac can show you this guide,
  but it does not build or serve the GB10 engine.
- You do not need a second Spark, a cable or RDMA setup.
- Package setup is for an administrator. The remaining steps run as your
  normal user.

GB10 is the GPU in an NVIDIA DGX Spark, and BF16 is the number format of the
weights. See the [Glossary](GLOSSARY.md) for PTX, prefill, MTP and other terms.

As you go, write down your source revision, compiler versions, checkpoint
hashes and result receipts (the saved files that record each answer). The
reference hashes in step 5 come from one named older build, so treat them as
history, not a target. We don't claim a recent run-through of this guide from a
freshly installed OS.

If you are an agent following this page, use persistent Bash sessions, run one
step at a time, check each step's expected result, and stop on failure.

Some blocks wait for you: they ask for input (`read -p`) or open a viewer
(`less`). Run each of those on its own, and answer it or press `q` before you
start the next block.

### 0. Prerequisites before downloads

You need a GB10 with its NVIDIA driver working and enough free memory. Stop
other GPU jobs through their owner or their service controls. Leave disk space
for the complete checkpoint, the builds and download state; we have no measured
minimum for free disk.

The required programs are `cc`, `curl`, `git`, `sha256sum`, `cmp`, `less`,
`jq`, `python3` and `python3-venv`. Vade itself needs neither Python nor the
CUDA toolkit; this guide checks for Python and venv only so that the official
Hugging Face tool is ready if you want it. Rust is pinned to **1.99.0** and the
GPU kernels to **nightly-2026-10-01**; keep both versions.

An administrator installs any missing Ubuntu packages **before** you start the
user-level install. Refresh the package lists first, even on a fresh image:

```sh
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl diffutils git jq less python3 python3-venv
```

Expected: both commands exit 0, and package installation finishes with no
missing candidates.

`python3-venv` can report “no installation candidate” when the package lists
are stale or missing. On Ubuntu 24.04 the versioned dependency is
`python3.12-venv`. If refreshing the lists still leaves no candidate, stop and
ask the administrator to check the Ubuntu package sources. Do not work around
the failure.

Start a Bash session in the parent folder where you will clone Vade (or in an
existing checkout). Run the checks there, so the disk figure is for that
filesystem:

```sh
bash
```

```sh
set -euo pipefail
uname -m
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
command -v cc curl git sha256sum cmp less jq python3
ldconfig -p | grep 'libcuda.so.1'
df -h .
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
VADE_VENV_CHECK=$(mktemp -d)
python3 -m venv "$VADE_VENV_CHECK/check"
"$VADE_VENV_CHECK/check/bin/python" -m pip --version
```

Expected:

- `aarch64`, GB10 and a driver version;
- no compute-process rows;
- all programs found, and a driver library;
- enough free disk and RAM;
- a pip version.

Compare disk `Avail` with the files you will download and build. Memory values
are in kB. There is no measured minimum for free RAM; context is sized at
startup. Inside a container, read the RAM figures together with
[Containers: large model refused despite free RAM](#containers-large-model-refused-despite-free-ram)
below.

The venv check must exit 0; “ensurepip is not available” means the venv
package is missing. This check leaves its small temporary environment in place
so you can inspect it. Keep this Bash session for steps 1–4.

### 1. Install the pinned Rust tools

If `~/.cargo/bin/rustup` does not exist, fetch the official installer and
read it first. (A shell that has not loaded `~/.cargo/env` may not find an
installed `rustup`, so check for the file rather than the command.)

```sh
curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs
less /tmp/vade-rustup-init.sh
```

Once you have read the installer, press `q`, then run:

```sh
sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path
```

The installer should exit 0.

Then install the pinned toolchains. If rustup was already installed, start
here:

```sh
. "$HOME/.cargo/env"
rustup toolchain install 1.99.0 --profile minimal
rustup toolchain install nightly-2026-10-01 --profile minimal \
  --target nvptx64-nvidia-cuda \
  --component rust-src --component llvm-tools --component llvm-bitcode-linker
rustup run 1.99.0 rustc --version
rustup run nightly-2026-10-01 rustc --version
```

Expected: both toolchains installed, and the version checks show **1.99.0**
and the pinned nightly chosen by the
[kernel builder](../engine/crates/gpu/build-kernels/src/main.rs). Do not go on
if a version check fails, even if rustup said “unchanged”.

### 2. Get the source and build the engine

If you are already at the root of a Vade checkout, skip the next two blocks.

Otherwise, clone into a new, empty parent folder. When prompted, enter the
clone URL; for the public release that is
`https://github.com/wappy-x/vade.git`. Keep any access credentials out of the
checkout.

```sh
read -r -p 'Repository clone URL: ' VADE_REPO_URL
```

```sh
git clone "$VADE_REPO_URL" vade
cd vade
```

Then, from the repository root:

```sh
VADE_ROOT="$PWD"
VADE_ENGINE="$VADE_ROOT/engine"
git rev-parse HEAD
cd "$VADE_ENGINE"
unset RUSTUP_TOOLCHAIN CARGO_TARGET_DIR RUSTFLAGS CARGO_ENCODED_RUSTFLAGS \
  CARGO_BUILD_TARGET RUSTC RUSTDOC RUSTC_WRAPPER
cargo +1.99.0 --version
cargo kernels
cargo +1.99.0 build --release --offline -p spark-serve
```

Expected: a full source SHA, Cargo 1.99.0, and Cargo builds that exit
successfully. Write down your source SHA with your results. The first kernel
build can fetch compiler source components.

The build leaves these files, relative to the current engine directory:

- the PTX files (the built GPU kernels):
  `target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx` and
  `target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx`;
- the server: `target/release/spark-serve`.

Every `target/`, `models/` and `fixtures/` path below is relative to the
**engine workspace**, not the repository root. Go back to `"$VADE_ENGINE"`
before each build, download or server start.

### 3. Download and verify the official small model

Use revision `b968826d9c46dd6066d109eabc6255188de91218` of Qwen/Qwen3-8B, in a
new model folder for this revision. Read the maker's terms before you use it.

This downloads five BF16 shards and **seven supporting files**, including
`generation_config.json`. A rerun skips files that are already complete and
downloads any interrupted `.part` file again.

```sh
cd "$VADE_ENGINE"
MODEL=models/Qwen/Qwen3-8B
MODEL_REV=b968826d9c46dd6066d109eabc6255188de91218
mkdir -p "$MODEL"
MODEL_URL="https://huggingface.co/Qwen/Qwen3-8B/resolve/$MODEL_REV"
fetch_qwen_file() {
  printf 'Model file: %s\n' "$1"
  test -f "$MODEL/$1" || {
    curl -fL --retry 3 -o "$MODEL/$1.part" "$MODEL_URL/$1" &&
      mv "$MODEL/$1.part" "$MODEL/$1"
  }
}
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
  vocab.json merges.txt model.safetensors.index.json; do
  fetch_qwen_file "$f"
done
for n in 1 2 3 4 5; do
  f=$(printf 'model-%05d-of-00005.safetensors' "$n")
  fetch_qwen_file "$f"
done
```

Expected: each curl finishes with exit 0 and no missing files.

Now check **every file**, supporting files included, against the pinned
SHA-256 list kept in [the CLI catalog](../cli/src/catalog.rs):

```sh
cd "$VADE_ENGINE"
(cd "$MODEL" && sha256sum -c - <<'SHA256'
f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30  config.json
2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2  generation_config.json
8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5  merges.txt
31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f  model-00001-of-00005.safetensors
5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282  model-00002-of-00005.safetensors
c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836  model-00003-of-00005.safetensors
b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a  model-00004-of-00005.safetensors
20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff  model-00005-of-00005.safetensors
f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc  model.safetensors.index.json
aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4  tokenizer.json
d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101  tokenizer_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910  vocab.json
SHA256
)
```

Expected: **12 `OK` lines** and exit 0. Stop on any mismatch or missing file,
and continue only when every pinned file passes.

Use this full hash check rather than a tool that only reads metadata and
tensor shapes: such a tool cannot check every weight byte. You don't need the
Hugging Face CLI for it.

### 4. Start the server

```sh
cd "$VADE_ENGINE"
./target/release/spark-serve models/Qwen/Qwen3-8B
```

Keep this terminal open. When the server is ready, it prints a line starting
with `ready` that names `model=qwen3-8b` and the context, input and output
limits it accepted. Write down your actual limits; they depend on available
memory.

By default the server listens for local HTTP on port 8080, as set in
[server options](../engine/crates/serve/src/bin/spark-serve.rs). Wait until it
is ready and reports the checkpoint identity before you send requests.

If you time the start, note that a later start can read the weights from the
warm filesystem cache, so its time is not a cold-start time.

### 5. First chat, then the same answer twice

Open a second Bash terminal **on the Spark**. When prompted, enter `http://`
followed by the `bind=` value the server printed (port 8080, no trailing
slash). Keep that address in your terminal, not in Git:

```sh
bash
```

```sh
set -euo pipefail
read -r -p 'Local server HTTP origin (port 8080, no trailing slash): ' VADE_API_URL
```

```sh
curl -fsS "$VADE_API_URL/health" | jq '{status,model,checkpoint:.vade.checkpoint}'
curl -fsS "$VADE_API_URL/v1/models" | jq .
```

Expected: health `ready`, checkpoint `hash_state: ready`, and `/v1/models`
contains `"id": "qwen3-8b"`. The model ID is the model folder's name,
lowercased. If hashing is still pending, repeat this block before you send a
request.

Then make a new folder for the small response receipts, and send the first
request from there:

```sh
VADE_ANSWERS=$(mktemp -d)
cd "$VADE_ANSWERS"
cat > request.json <<'JSON'
{"model":"qwen3-8b","messages":[{"role":"user","content":"Explain why ice floats."}],"max_tokens":128,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"cache":false,"draft":"plain","enable_thinking":false,"stream":true,"stream_options":{"include_usage":true}}
JSON
curl -fsSN "$VADE_API_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @request.json > first.sse
```

Expected: `first.sse` holds the streamed reply. Write down your response time
and the effective settings the server reports. These numbers only count if the
health check above already showed `ready`, model `qwen3-8b` and checkpoint
`hash_state: ready`.

Next, decode the streamed answer and send the same request a second time. The
`extract_answer` function keeps the answer's UTF-8 bytes exactly, adds no
newline, and checks the effective settings echoed in the final JSON event:

```sh
extract_answer() {
  test "$(tail -n 2 "$1" | tr -d '\r\n')" = 'data: [DONE]'
  sed -n 's/^data: //p' "$1" | sed '/^\[DONE\]$/d' > "$1.events.json"
  jq -se 'length > 0 and all(.[]; .error == null) and
    ([.[] | select(.vade != null)] | length > 0 and all(.[];
    .vade.sampling.temperature == 0 and
    .vade.sampling.top_p == 1 and .vade.sampling.top_k == 0 and
    .vade.sampling.repetition_penalty == 1 and
    .vade.sampling.presence_penalty == 0 and
    .vade.sampling.frequency_penalty == 0 and
    .vade.draft == "plain" and .vade.mtp_depth == 0 and
    .vade.thinking == false and .vade.draft_used == false and
    .vade.model_draft_used == false))' "$1.events.json"
  jq -sj '[.[] | .choices[]? | .delta.content? // empty] | join("")' \
    "$1.events.json" > "$2"
  test -s "$2"
}
extract_answer first.sse first.answer
cat first.answer
printf '\n'
curl -fsSN "$VADE_API_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @request.json > second.sse
extract_answer second.sse second.answer
cmp first.answer second.answer
sha256sum first.answer second.answer
wc -c first.answer second.answer
```

Expected:

- the settings check prints `true` twice;
- `cmp` exits 0 with no output;
- the two answer hashes and byte counts match.

Response IDs and timestamps can differ, and so can the `seed` echoed in the
settings. The server picks a fresh random seed for each request that does not
name one, and greedy decoding never uses it. So compare the decoded answer
bytes, not the whole responses.

For reference, an earlier test on build
`e335236a11d681c473de0dce92dece1562d6efa0` compared the answers from
**two fresh process starts** of the server with these settings. All five pairs
matched. It recorded:

| Prompt | Answer size | SHA-256 |
| --- | --- | --- |
| The ice question | **609 bytes** | `4a55e40cf7906fffab5df6cb8339120faeb2e996327692cf88c598adb9d05873` |
| “Calculate 17 times 23. Give the number and one sentence explaining it.” | **135 bytes** | `59eb112d62542c1919bfea67ccec4ca4667c3c0181508fa1eb1fe205b7130563` |

These hashes belong to that one old build. Yours may differ; what matters is
that your two answers match each other.

To check restarts as well, stop with Ctrl-C in the server terminal and repeat
step 4. Wait for ready health and identity, then send the same saved request
again and compare its decoded answer.

### 6. Stop and record the result

Press **Ctrl-C in the server terminal**. Expected:

- the server stops and you get your shell prompt back;
- `nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`
  shows no row for that server;
- every process this test started has exited 0.

Keep your source SHA, version checks, file-check output, request, effective
settings and answer hashes with your install notes, outside Git.

In the second terminal, print where the receipts are and check shutdown:

```sh
printf 'Receipt directory: %s\n' "$VADE_ANSWERS"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

### 7. CLI chat (optional)

After step 6, go back to the first terminal. This flow also needs `tar` and
clean Git source checkouts (no uncommitted changes). Our timed install
run-throughs did not cover it.

`vade run` needs its own installation. If you already set up FP4-Local on this
Spark with `vade launch`, it refuses to start and tells you to unset
`VADE_CONFIG` and choose a fresh `VADE_HOME`. Do that, and keep the same
`VADE_HOME` for `chat` and `down`.

```sh
cd "$VADE_ROOT"
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
unset VADE_CONFIG
./cli/target/release/vade run qwen3-8b --engine-source "$VADE_ROOT/engine" \
  --harness-source "$VADE_ROOT/harness" --weights-source "$VADE_ENGINE/models/Qwen/Qwen3-8B"
./cli/target/release/vade chat
./cli/target/release/vade down
```

`run` checks the weights, builds, starts its server, answers and runs doctor.
`chat` opens the harness; leave chat before `down`. See
[CLI recovery](../cli/README.md) and [harness usage](../harness/README.md).

Time your own runs rather than relying on the dated timings in these docs.
[Engine checks](../engine/crates/engine/README.md#build-and-run-the-checks)
and [two-Spark installation](fresh-install.md) cover separate qualification
checks.

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
names its own controls. Part 2 of this page runs `spark-serve`. Part 1's
FP4-Local engine is Flash (`flash-serve`), and `tp-serve` is the
tensor-parallel server:

| Refused by | Controls it names |
| --- | --- |
| `spark-serve` | `--context-tokens`, `--output-tokens`, `--max-active` |
| Flash | Launcher `CONTEXT` / `OUTPUT`, positional context, `FLASH_SERVE_OUTPUT_TOKENS`, `FLASH_SERVE_MAX_ACTIVE` |
| `tp-serve` | Positional `CONTEXT`, `TP_SERVE_MAX_ACTIVE` |
| A logical cap on the KV cache (which holds each chat's attention state) or on sessions | `--kv-budget-mib`, `--session-budget-mib` |

Token refusals use the same first line, but report a token gap and say
`GiB gap is not applicable to a token ceiling.` A request past the configured
context names `vade --context` in that refusal. To fix it, shorten the prompt
or lower `max_tokens`. To raise the server's context, restart it with the flag
for the route you used: for Part 1, run `vade down` and then
`vade up --context 262144` (or choose a smaller supported context); for Part 2,
stop the foreground server with Ctrl+C and restart `spark-serve` with
`--context-tokens N`, where N fits that checkpoint and the available memory.
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

### Prefill check and qualification limits

This check tests prefill, the step that reads the whole prompt in one batch.
Stop the server first, then run it from the engine workspace:

```sh
cd "$VADE_ENGINE"
cargo +1.99.0 run --release --offline -p spark-engine --bin batch-bench -- \
  target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx \
  models/Qwen/Qwen3-8B prefill
```

Expected: exit 0 and `PASS actual_checkpoint_prefill logit_comparisons=7`.

The check confirms that skipping the logits for some prompt rows (head
skipping) does not change results within a phase. It does not cover every mix
of batch, cache, chunk and MTP, or model quality. Write down your SHA and
result; old results say nothing about later builds.

Some history: an older version of this check compared prompt logits (the
model's raw scores for each next token) with decode logits. Those use
different fixed matrix-multiply (GEMM) orders on purpose, so its failure was
not a real batched-prefill bug. The corrected check compares within the same
phase. It passed **3 of 3** runs on a cloud GB10 at commit `67089c3` (full ID
`67089c316b05e9c9439aff765a55dc4a538db824`), with seven finite, bit-identical
full rows each.
