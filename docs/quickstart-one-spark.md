# One Spark: a first answer through the gateway

For **Qwen3.8-Flash-Next-FP4-Local**, control the Spark from a Linux or macOS controller
with the launcher. Complete [INSTALL steps 1–3](../INSTALL.md): check the Spark and
SSH prerequisites, install the pinned compilers, clone/build on the controller, create
the private config, and verify the complete checkpoint. The older Qwen3-8B
BF16 instructions below use a different model and run directly on the Spark.

Keep the configured controller shell open. From the repository root, with
`VADE_CONFIG` set to your private config and the built CLI on `PATH`:

```sh
vade install --dry-run
vade install
vade up --fp4-local --dry-run
vade up --fp4-local
vade status
vade doctor
```

`install` copies your tracked source and builds on the configured Spark. It
does not download weights. The controller starts the same defaults as
`vade launch` on the Spark, context 32,768 and output 8,192; only `vade launch`
lowers context, and only when the engine refuses it for memory. `vade status`
prints the concurrent chats the engine admitted. The ready endpoint is
`http://127.0.0.1:18080/v1`, reached through the launcher's SSH tunnel.
Check its actual model and limits:

```sh
curl -qfsS http://127.0.0.1:18080/health
curl -qfsS http://127.0.0.1:18080/v1/capabilities
```

Wait for health `ready` and checkpoint `hash_state: ready`.

Exactness means that the same prompt and model produce the same answer, token
for token, on every run and every Spark; Vade's greedy default chooses the most
likely next token. Drafting may make the answer arrive sooner, but must not
change it. With the engine still running, run `vade verify`: it repeats a
bundled prompt 20 times with drafting on and off, compares token IDs, prints
one `IDENTICAL` or `DIFFERENT` line, and saves `vade-verify.json`. Use
`vade verify -n 50` for more repeats, or `vade verify --prompt 2k` and
`vade verify --prompt 8k` for longer bundled prompts. Add `--machine spark-b`
to compare a second running engine configured as described in the
[CLI guide](../cli/README.md#verify-exactness). A pass covers the prompt, model,
output length and machines recorded in that receipt.

Start the invisible gateway in a second controller terminal, from the same checkout:

```sh
harness/target/release/vade-gateway \
  --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

In a third controller terminal, send a chat, then a streaming chat:

```sh
curl -qfsS http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp4-local","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
curl -qfsSN http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp4-local","messages":[{"role":"user","content":"Explain why ice floats in one sentence."}],"max_tokens":96,"chat_template_kwargs":{"enable_thinking":false},"stream":true}'
```

The first response has an assistant message in `choices[0].message.content`.
The stream contains `data:` events and ends with `data: [DONE]`. A follow-up
must include the previous user message and the **actual** assistant reply
before the new user message; the Chat Completions endpoint does not remember
history for you. See [the client setup guide](../harness/docs/passthrough-matrix.md)
to generate an isolated Codex home using the running engine's advertised
model and limits, with `--target gateway`. Its client URL is
`http://127.0.0.1:8787/v1`; the gateway does not translate API dialects.

Stop the client, press **Ctrl-C in the gateway terminal**, then in the
configured launcher terminal run:

```sh
vade down
```

Expected: `Vade: stopped. Its owned ranks and tunnel are closed.` On the Spark,
`nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader`
must have no row for that server. Keep the checkout and `VADE_CONFIG` until
shutdown is complete. For startup failures, use [launcher recovery](../cli/README.md#recovery).

## Qwen3-8B BF16: Ubuntu to a first answer

**Legacy route.** v0.1.0 supports nvidia/Qwen3.8-Flash-Next-FP4-Local; this
section is kept as the engine's from-source build walkthrough.
This page's model download and small-model disk allowance are for Qwen3-8B.

The rest of this page builds Vade and serves official Qwen3-8B BF16,
and checks that the same request gives the same answer twice. Run these commands
**on the Spark**, in Bash. A Mac can display this guide; it does not build or
serve the GB10 engine. No second Spark, cable or RDMA setup is needed.

GB10 is the GPU in an NVIDIA DGX Spark; BF16 is the weights' number format.
See the [Glossary](GLOSSARY.md) for PTX, prefill, MTP and other terms.
For agents: use persistent Bash sessions, execute one step at a time, check
its expected result, and stop on failure. Package setup is an administrator
prerequisite; the remaining steps are user-level. Run prompt and viewer
blocks separately, supplying input or pressing `q` before the next block.

Record your source revision, compiler versions, checkpoint hashes and result
receipts. Historical answer hashes below apply only to their named build; this
guide does not claim a current blank-OS replay.

## 0. Prerequisites before downloads

Use a GB10 with its working NVIDIA driver and sufficient available memory.
Stop other GPU jobs through their normal owner or service controls. Allow disk
space for the complete checkpoint, builds and download state; no measured
minimum free-disk threshold is published here. For Flash-Next FP8 across two
Sparks, use [the legacy FP8 installation](install-legacy-fp8.md) and its separate checkpoint helper.

Required programs: `cc`, `curl`, `git`, `sha256sum`, `cmp`, `less`, `jq`, `python3` and
`python3-venv`. Python/venv prepares the optional official Hugging Face tool;
Vade itself runs without Python or the CUDA toolkit. Rust is pinned to
**1.99.0**, GPU kernels to **nightly-2026-10-01**; keep both versions.

An administrator prepares missing Ubuntu packages **before** you start the
user-level install. Refresh package lists first, including on a fresh image:

```sh
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl diffutils git jq less python3 python3-venv
```

Expected: both commands exit 0; package installation ends without missing
candidates.
`python3-venv` can report “no installation candidate” when package lists are
stale or absent. If refreshing lists still leaves no candidate, stop and ask
the administrator to check the Ubuntu package sources. Do not bypass the
failure. Ubuntu 24.04's versioned dependency is `python3.12-venv`.

Start a Bash session in the parent folder where you will clone Vade (or in
an existing checkout). Check disk space on that same filesystem:

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

Expected: `aarch64`, GB10 and a driver version, no compute-process rows, all
programs found, a driver library, enough free disk/RAM, and a pip version.
Check disk `Avail` against the files you will download and build; memory
values are in kB. There is no measured minimum free-RAM threshold; context
is sized at startup. Container RAM readings need the troubleshooting check below.
The venv check must exit 0; “ensurepip is not available” means the venv package
is missing. This check leaves its small temporary environment for inspection.
Keep this Bash session for steps 1–4.

## 1. Install the pinned Rust tools

If `rustup` is absent, fetch and inspect its official installer:

```sh
curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs
less /tmp/vade-rustup-init.sh
```

After inspecting the installer, press `q`, then run:

```sh
sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path
```

Press `q` to leave `less`. Expect installer exit 0. If rustup already
exists, skip that block. Then:

```sh
. "$HOME/.cargo/env"
rustup toolchain install 1.99.0 --profile minimal
rustup toolchain install nightly-2026-10-01 --profile minimal \
  --target nvptx64-nvidia-cuda \
  --component rust-src --component llvm-tools --component llvm-bitcode-linker
rustup run 1.99.0 rustc --version
rustup run nightly-2026-10-01 rustc --version
```

Expected: both toolchains installed; version checks show **1.99.0** and the
pinned nightly selected by the [kernel builder](../engine/crates/gpu/build-kernels/src/main.rs).
Do not proceed if a version check fails, even if rustup said “unchanged”.

## 2. Get the source and build the engine

Repository access must already work. Use the clone URL of the repository you are installing;
keep access credentials outside the checkout. Use a new, empty parent folder:
If already at the monorepo root, skip the next two blocks.

```sh
read -r -p 'Repository clone URL: ' VADE_REPO_URL
```

```sh
git clone "$VADE_REPO_URL" vade
cd vade
```

From the monorepo root:

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

Expected: a full source SHA, Cargo 1.99.0, and successful Cargo build exits.
Relative to the current engine directory, PTX files are
`target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx` and
`target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx`; the server is
`target/release/spark-serve`. The initial kernel build can fetch compiler source
components. Record your source SHA with your results.

Every `target/`, `models/` and `fixtures/` path below is relative to the
**engine workspace**, not the monorepo root. Return to `"$VADE_ENGINE"` before
each build, download or server start.

## 3. Download and verify the official small model

Revision: `b968826d9c46dd6066d109eabc6255188de91218` of Qwen/Qwen3-8B.
Use a new model directory for this revision. Read the maker's terms before
using it. This downloads five BF16 shards and **seven supporting files**,
including `generation_config.json`; a complete file is skipped on a rerun,
while an interrupted `.part` file is downloaded again.

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

Expected: each curl completes, with exit 0 and no missing files.

Verify **every file**, including the supporting files, against the pinned
SHA-256 inventory retained in [the CLI catalog](../cli/src/catalog.rs):

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

Expected: **12 `OK` lines** and exit 0. Stop on any mismatch or missing file.
Continue only when every pinned file passes verification.
A metadata/shape inspector alone cannot verify all weight
bytes. These checks do not need the Hugging Face CLI.

## 4. Start the server

```sh
cd "$VADE_ENGINE"
./target/release/spark-serve models/Qwen/Qwen3-8B
```

Keep this terminal open. Expected: a line beginning `ready`, identifying
`model=qwen3-8b` and the admitted context/input/output limits. Record your
actual limits; they depend on available memory. The default listener is local
HTTP on port 8080, as defined by [server options](../engine/crates/serve/src/bin/spark-serve.rs).
Wait for ready checkpoint identity before requests. Do not treat a warm
filesystem cache as a cold-start measurement.

## 5. First chat, then the same answer twice

Open a second Bash terminal **on the Spark**. Enter `http://` followed by the
server's printed `bind=` value (port 8080, no trailing slash) when prompted.
Keep that address in your terminal, outside Git.
Use a new folder for the small response receipts:

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
contains `"id": "qwen3-8b"`. If hashing is pending, repeat this block before sending a request.
The model ID is the model directory's basename, lowercased.

```sh
VADE_ANSWERS=$(mktemp -d)
cd "$VADE_ANSWERS"
cat > request.json <<'JSON'
{"model":"qwen3-8b","messages":[{"role":"user","content":"Explain why ice floats."}],"max_tokens":128,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"cache":false,"draft":"plain","enable_thinking":false,"stream":true,"stream_options":{"include_usage":true}}
JSON
curl -fsSN "$VADE_API_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' --data-binary @request.json > first.sse
```

Expected: health `ready`, model `qwen3-8b`, and checkpoint `hash_state: ready`
before recording identity/timing. If hashing is pending, repeat the health
check before the request. Record your own response timing and effective settings.

Decode the streamed answer, preserving its UTF-8 bytes without adding a newline,
and check the effective settings echoed on the terminal JSON event:

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

Expected: settings validation prints `true` twice; `cmp` exits 0 with no output;
the two answer hashes and byte counts match.
Response IDs, timestamps and random greedy seeds can differ; compare the
decoded answer bytes, not whole response envelopes.

The earlier `e335236a11d681c473de0dce92dece1562d6efa0` test compared **two fresh
process starts** with these settings; all five pairs matched. For ice it recorded
**609 bytes**, **128 completion tokens**, and SHA-256
`4a55e40cf7906fffab5df6cb8339120faeb2e996327692cf88c598adb9d05873`.
The arithmetic question “Calculate 17 times 23. Give the number and one sentence
explaining it.” recorded **135 bytes**, **67 tokens**, and SHA-256
`59eb112d62542c1919bfea67ccec4ca4667c3c0181508fa1eb1fe205b7130563`.
These are historical reference hashes for the named build and tested inventory;
your check requires your two answers to match. A later build is not qualified
by a historical hash. To check restarts too, stop with Ctrl-C in the server
terminal, repeat step 4, wait for ready health/identity, and submit the same
saved request again before comparing its decoded answer.

## 6. Stop and record the result

Press **Ctrl-C in the server terminal**. Expected: the foreground process
returns to the shell; `nvidia-smi --query-compute-apps=pid,process_name
--format=csv,noheader` has no row for that server. The test's owned processes
exited 0. Keep your source SHA, version checks,
file-check output, request, effective settings and answer hashes with your
install notes outside Git.

In the second terminal, print the receipt location and check shutdown:

```sh
printf 'Receipt directory: %s\n' "$VADE_ANSWERS"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

## Containers: large model refused despite free RAM

Sizing refusals start with **`Requested size does not fit.`** The following lines
are `Asked:`, `Available:`, `Gap:` and `Next:`. Byte totals use binary GiB
(1 GiB = 1073741824 bytes), with exact bytes alongside them. Asked terms name
weights/rank, session reservations and counts, CONTEXT and OUTPUT where known;
lower-level allocators explicitly identify terms they were not given.
For example, a negotiated session pool refusal includes:

```text
Requested size does not fit.
Asked: <weights/rank; reservation/session × sessions; CONTEXT; OUTPUT; fixed/cache terms>; total 18.48 GiB (19847557088 bytes).
Available: 6.92 GiB (7425823856 bytes) from <budget source and caps>.
Gap: 11.57 GiB (12421733232 bytes).
Next: 1. Lower <supported controls>. 2. In containers, spark-serve supports --gpu-budget-mib N; size N by rounding up (weights + shared workspace + session reservations + retained cache + headroom) / 1048576, within verified host capacity. Flash uses SEQPAR_MEMORY_BYTES in bytes; its CUDA safety cap still applies. 3. Use more Sparks with supported tensor parallelism.
```

Bracketed fields vary with the refusal. `spark-serve` names `--context-tokens`,
`--output-tokens` and `--max-active`. Flash names launcher `CONTEXT` / `OUTPUT`,
positional context, `FLASH_SERVE_OUTPUT_TOKENS` and `FLASH_SERVE_MAX_ACTIVE`.
`tp-serve` names positional `CONTEXT` and `TP_SERVE_MAX_ACTIVE`.
Logical KV and session caps name `--kv-budget-mib` and `--session-budget-mib`.
Token refusals use the same first line but report a token gap and say
`GiB gap is not applicable to a token ceiling.` A request past the configured
context names `vade --context` in that refusal. Lower the prompt, lower
`max_tokens`, or raise context with `--context` up to 262144.
Memory overrides and additional Sparks cannot raise the checkpoint token ceiling.

Unified availability is CUDA free + max(MemAvailable − MemFree, 0), capped by
MemAvailable and CUDA total. Default single-Spark sizing leaves 10%, at least
1 GiB; Flash leaves 10%, at least 2 GiB, then funds shared buffers and graphs
and holds a 512 MiB transient reserve back from sessions for per-tick buffers.
Flash uses the lowest rank pool and largest session reservation. An explicit
ceiling is also a cap. The diagnostic prints actual inputs when supplied.
In LXC or a VM, container MemAvailable can omit host reclaimable cache, leaving
the estimate low. Verify host capacity before choosing N; the override creates
no memory, and Flash still applies its CUDA safety cap. Actual CUDA errors are
reported separately. See [memory limits](../engine/docs/server.md#how-it-sizes-itself).

## Prefill check and qualification limits

The old checker compared prompt and sequential decode logits, whose fixed GEMM orders differ by design; its failure did not establish a batched-prefill regression.
The corrected same-phase checker passed **3/3** cloud GB10 runs at `67089c316b05e9c9439aff765a55dc4a538db824` (seven finite, bit-identical full rows each). After stopping the server, run from the engine workspace:

```sh
cd "$VADE_ENGINE"
cargo +1.99.0 run --release --offline -p spark-engine --bin batch-bench -- \
  target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx \
  models/Qwen/Qwen3-8B prefill
```

Expected: exit 0 and `PASS actual_checkpoint_prefill logit_comparisons=7`.
This checks head skipping within each phase, not every batch/cache/chunk/MTP combination or model quality. Record your current SHA and result; historical results do not qualify later builds.

## 7. CLI chat (optional)

After step 6, use the first terminal. This flow additionally needs `tar` and clean Git source checkouts; it was not part of the install replays.
```sh
cd "$VADE_ROOT"
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
unset VADE_CONFIG
./cli/target/release/vade run qwen3-8b --engine-source "$VADE_ROOT/engine" \
  --harness-source "$VADE_ROOT/harness" --weights-source "$VADE_ENGINE/models/Qwen/Qwen3-8B"
./cli/target/release/vade chat
./cli/target/release/vade down
```

`run` verifies weights, builds, starts its server, answers and runs doctor; `chat` opens the harness. Leave chat before `down`. See [CLI recovery](../cli/README.md) and [harness usage](../harness/README.md).

Record measurements rather than assuming the dated times.
[Engine checks](../engine/crates/engine/README.md#build-and-run-the-checks)
and [two-Spark installation](fresh-install.md) cover separate qualifications.

### Steering and keeping the engine ready

`vade chat` attaches to the engine started by `vade up`; exiting chat leaves that
engine resident. Open another chat without paying model load or JIT again. Use
`vade down` when you want to release the GPU.

While a reply is running, sending another message interrupts it by default. Vade
keeps the visible partial reply marked `[interrupted]` in the conversation and
sends the new message with that context after cancellation. For Vade Engine it
sends the request's private capability to `/v1/cancel`, including before the first
token. A failed cancellation acknowledgment holds the steering message and reports
the failure. Other HTTP servers receive a socket close, which may not stop their work.

Use `/queue on` to let replies finish before sending additional typed messages,
`/queue off` to restore steering, or `/queue TEXT` for one queued follow-up.
`/exit` cancels an active reply before closing chat. Native-agent interrupted
replies are saved in its session journal; direct-chat history remains in memory.

Repeat `vade up` with the same settings to check and reuse the healthy server.
CUDA JIT caches persist per PTX hash across down/up. To avoid compiling on the
Spark, set `[global] prebuilt_dir` to a trusted local bundle root; exact source
revisions and SHA256s must match. Missing bundles fall back to source. Model
inventory checks run alongside copy/build and skip repeated checks when unchanged.
See [the CLI guide](../cli/README.md) for the prebuilt bundle layout.
