# One Spark: build from source with Qwen3-8B

This is a **legacy route**. It is history, and it does not qualify current
main. Vade now runs one model, Qwen3.8 Flash-Next. We keep this page because it shows how to build the engine from source. It uses
the small Qwen3-8B model, and its download and disk notes apply only to that
model. To install Vade today, start at [INSTALL.md](../../INSTALL.md).

On this page you build Vade, serve the official Qwen3-8B BF16, and check that
the same request gives the same answer twice.

- Run these commands **on the Spark**, in Bash. A Mac can show you this guide,
  but it does not build or serve the GB10 engine.
- You do not need a second Spark, a cable or RDMA setup.
- Package setup is for an administrator. The remaining steps run as your
  normal user.

GB10 is the GPU in an NVIDIA DGX Spark, and BF16 is the number format of the
weights. See the [Glossary](../GLOSSARY.md) for PTX, prefill, MTP and other terms.

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

## 0. Prerequisites before downloads

You need a GB10 with its NVIDIA driver working and enough free memory. Stop
other GPU jobs through their owner or their service controls. Leave disk space
for the complete checkpoint, the builds and download state; we have no measured
minimum for free disk.

The required programs are `cc`, `curl`, `git`, `sha256sum`, `cmp`, `less`,
and `jq`. Vade needs no Python or CUDA toolkit. The model download helper uses
Rust and curl. Rust is pinned to **1.99.0** and the
GPU kernels to **nightly-2026-10-01**. Keep both versions.

An administrator installs any missing Ubuntu packages **before** you start the
user-level install. Refresh the package lists first, even on a fresh image:

```sh
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl diffutils git jq less
```

Expected: both commands exit 0, and package installation finishes with no
missing candidates.

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
command -v cc curl git sha256sum cmp less jq
ldconfig -p | grep 'libcuda.so.1'
df -h .
awk '/MemTotal:|MemAvailable:/ {print}' /proc/meminfo
```

Expected:

- `aarch64`, GB10 and a driver version;
- no compute-process rows;
- all programs found, and a driver library;
- enough free disk and RAM.

Compare disk `Avail` with the files you will download and build. Memory values
are in kB. There is no measured minimum for free RAM. Context is sized at
startup. Inside a container, read the RAM figures together with
[Containers: large model refused despite free RAM](../quickstart-one-spark.md#containers-large-model-refused-despite-free-ram)
below.

Keep this Bash session for steps 1–4.

## 1. Install the pinned Rust tools

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
[kernel builder](../../engine/crates/gpu/build-kernels/src/main.rs). Do not go on
if a version check fails, even if rustup said “unchanged”.

## 2. Get the source and build the engine

If you are already at the root of a Vade checkout, skip the next two blocks.

Otherwise, clone into a new, empty parent folder. When prompted, enter the
clone URL. For the public release that is
`https://github.com/wappy-x/Vade.git`. Keep any access credentials out of the
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
cargo +1.99.0 build --release --offline --locked -p spark-serve
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

## 3. Download and verify the official small model

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
SHA-256 list kept in [the CLI catalog](../../cli/src/catalog.rs):

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

## 4. Start the server

```sh
cd "$VADE_ENGINE"
./target/release/spark-serve models/Qwen/Qwen3-8B
```

Keep this terminal open. When the server is ready, it prints a line starting
with `ready` that names `model=qwen3-8b` and the context, input and output
limits it accepted. Write down your actual limits. They depend on available
memory.

By default the server listens for local HTTP on port 8080, as set in
[server options](../../engine/crates/serve/src/bin/spark-serve.rs). Wait until it
is ready and reports the checkpoint identity before you send requests.

If you time the start, note that a later start can read the weights from the
warm filesystem cache, so its time is not a cold-start time.

## 5. First chat, then the same answer twice

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

These hashes belong to that one old build. Yours may differ. What matters is
that your two answers match each other.

To check restarts as well, stop with Ctrl-C in the server terminal and repeat
step 4. Wait for ready health and identity, then send the same saved request
again and compare its decoded answer.

## 6. Stop and record the result

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

## 7. CLI chat (optional)

After step 6, go back to the first terminal. This flow also needs `tar` and
clean Git source checkouts (no uncommitted changes). Our timed install
run-throughs did not cover it.

`vade run` needs its own installation. If you already set up FP4-Local on this
Spark with `vade launch`, it refuses to start and tells you to unset
`VADE_CONFIG` and choose a fresh `VADE_HOME`. Do that, and keep the same
`VADE_HOME` for `chat` and `down`.

```sh
cd "$VADE_ROOT"
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
unset VADE_CONFIG
./cli/target/release/vade run qwen3-8b --engine-source "$VADE_ROOT/engine" \
  --harness-source "$VADE_ROOT/harness" --weights-source "$VADE_ENGINE/models/Qwen/Qwen3-8B"
./cli/target/release/vade chat
./cli/target/release/vade down
```

`run` checks the weights, builds, starts its server, answers and runs doctor.
`chat` opens the harness. Leave chat before `down`. See
[CLI recovery](../../cli/README.md) and [harness usage](../../harness/README.md).

Time your own runs rather than relying on the dated timings in these docs.
[Engine checks](../../engine/crates/engine/README.md#build-and-run-the-checks)
and [two-Spark installation](../fresh-install.md) cover separate qualification
checks.

## Prefill check and qualification limits

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
result. Old results say nothing about later builds.

Some history: an older version of this check compared prompt logits (the
model's raw scores for each next token) with decode logits. Those use
different fixed matrix-multiply (GEMM) orders on purpose, so its failure was
not a real batched-prefill bug. The corrected check compares within the same
phase. It passed **3 of 3** runs on a cloud GB10 at commit `67089c3` (full ID
`67089c316b05e9c9439aff765a55dc4a538db824`), with seven finite, bit-identical
full rows each.
