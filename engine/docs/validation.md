# Validation

The result tables are historical measurement records for their named builds.
They do not qualify the current checkout; use the documented checks to record
a new result.

**The current exactness bar is identical token IDs across runs and Sparks
with the same setup; official-model agreement is a separate check.** See
[AGENTS.md](../../AGENTS.md) for the binding rules and required evidence.

The evidence for Vade's two promises, every disagreement found so far, and
how to run each check yourself.

Flash's covered BF16 projections now always use one tensor-core mode: BF16
high and residual operands with FP32 accumulation. There is no
`FLASH_BF16_MMA` switch or scalar-FP32 alternative. A fixed build and setup
must preserve the same bytes across batch widths, cache hit/miss, prompt
chunks and greedy MTP. Answers may differ from older builds. Each supported
Spark count must give the same token IDs; FP8 on six Sparks does not yet. The
measurements below belong to their named builds; they do not
establish that the new Flash integration passed a GPU gate.

The one-mode pair handoff compares full output rows with split `23d33f87`
under its `FLASH_BF16_MMA=1` setting, then runs the full determinism and
safety matrix and captures new goldens for the integrated build. Split quality
evidence transfers only if the cross-build bytes match. Mac host tests do not
establish this identity, GPU determinism, long-context quality or speed.

- **Deterministic.** A request gets exactly the same answer alone or in a
  batch of up to 64, with its prompt beginning cached or computed
  fresh. Recorded on Qwen3-8B, Qwen3-32B and Qwen3-30B-A3B.
- **Agreement with the official model.** Vade's next-token choices are
  compared with the makers' own reference code running the official
  weights: **1,523, 1,515 and 1,498 of 1,536** positions match on current
  main, for Qwen3-8B, Qwen3-32B and Qwen3-30B-A3B.
- **Pooled prefill, fixed.** Prompts prefilled together in one pooled pass
  now exactly match the same prompts prefilled one at a time:
  [Pooled prefill (fixed)](#pooled-prefill-fixed).
- Passing the first promise does not establish the second.

<details>
<summary><b>Terms on this page</b>: logits, greedy, reference, teacher-forced, near-tie, BF16 step, fixture, batch width, prefix, pooled prefill</summary>

<dl>
<dt>Logits</dt>
<dd>The model's raw score for every possible next token. The highest score
wins.</dd>
<dt>Top-1 or greedy token</dt>
<dd>The single most likely next token, the one Vade picks with
<code>temperature</code> 0.</dd>
<dt>Reference</dt>
<dd>Hugging Face Transformers, the makers' reference code, running the official
checkpoint.</dd>
<dt>Teacher-forced position</dt>
<dd>One next-token choice made after the reference's own history, so one early
difference does not spoil later positions.</dd>
<dt>Near-tie</dt>
<dd>A disagreement where the reference's top two scores differ by less than one
BF16 step. <b>Near-ties still count as failures</b>.</dd>
<dt>BF16 step</dt>
<dd>The smallest gap between two BF16 numbers of that size. BF16 is the 16-bit
format the weights use.</dd>
<dt>Fixture</dt>
<dd>A saved file of reference inputs and outputs, kept in
<code>fixtures/</code>.</dd>
<dt>Batch width</dt>
<dd>How many requests run through the GPU together.</dd>
<dt>Prefix</dt>
<dd>The opening part of a prompt that is already in the cache.</dd>
<dt>Pooled prefill</dt>
<dd>Prefill is reading the prompt. Pooled prefill reads several prompts together
in one pass.</dd>
<dt>Shared-prefix change</dt>
<dd>The change that gave single requests, batches and cached prefixes one fixed
decode-attention order. Its final gate ran on build <code>39aceb2</code>; it
merged into main as <code>430fdbc</code>.</dd>
</dl>

</details>

## Promise 1: deterministic

**Every check below gave identical results.** They compare Vade with itself.

On the Qwen3 models:

- **Batched against single.** Logits for each prompt alone and inside
  batches of width 1 to 64 were bit-identical on Qwen3-8B, Qwen3-32B and
  Qwen3-30B-A3B (`39aceb2`, and again on main `430fdbc`). Bit-identical
  again on Qwen3-8B and Qwen3-30B-A3B at the Gemma merge gate (`9c3fe9c`).
- **Cached prefixes.** The same comparison after a prefix of 0, 4,090,
  4,096, 16,380, 16,384 or 20,000 tokens, at widths 1, 2, 4, 8, 16, 32 and
  64: bit-identical, 42 of 42 combinations per Qwen3 model (`39aceb2`). The
  full 64-by-6 combination was not run.
- **16 chats against one at a time.** Each of 16 chats replayed alone on a
  fresh server: 16 of 16 replies match on each Qwen3 model (`39aceb2`).
- **Cached against cold.** SHARED replies with the prompt cache on and off:
  16 of 16 replies and 2,048 token IDs per Qwen3 model (`39aceb2` and main
  `430fdbc`).
- **Cache kept per API key.** A different key reuses no cached tokens: zero
  tokens reused across keys on each Qwen3 model (admission gate,
  `3911e2b`).
- **Latest merge gate** (`8b2183c`, merged as main `eb67c8b`). Widths 1 to
  64, the six prefixes, ragged outputs and pooled prefill: 190 of 190
  comparisons per Qwen3 model with zero differing logit bits. 16 of 16
  chats match their replay alone, 2,048 of 2,048 cached SHARED IDs match
  cold, and zero tokens are reused across keys.

**On Qwen3.8 and Gemma,** drafting and batching matched too: Qwen3.8-27B's
built-in multi-token prediction (MTP) drafting, and Gemma 4 31B's batching
and official assistant draft. [Qwen3.8 and Gemma scope](#qwen38-and-gemma-scope)
has the counts.

**A former gap, now fixed.** On Qwen3.8-27B BF16, reading a long prompt in
two pieces split after 255 tokens used to change logit bits (main
`803bd3d`). On main `eb67c8b`, a 28,672-token prompt split at 255, 256,
4,095, 4,096, 16,383 or 16,384 tokens is bit-exact. See
[Qwen3.8 and Gemma scope](#qwen38-and-gemma-scope).

## Pooled prefill (fixed)

**Prompts prefilled together in one pooled pass now exactly match the same
prompts prefilled one at a time, on main `eb67c8b`.**

- **What was wrong.** On main `603f95b`, three short prompts of 13, 17 and
  29 tokens prefilled together and then one at a time differed in 923 logit
  bits on Qwen3-8B and 532 on Qwen3-32B. Prefill computed the vocabulary
  head with decode's four-part reduction.
- **Now.** 36 pooled cases per Qwen3 model, including those lengths and
  12-session mixes, show zero differing logit bits on Qwen3-8B, Qwen3-32B
  and Qwen3-30B-A3B (merge gate `8b2183c`).

## Promise 2: agreement with the official model

**Not perfect: all three Qwen3 models fail strict agreement on both
builds.** The recorded builds match 97.5% to 99.2% of positions.

The broad Qwen3 suites: 24 prompts per model, 64 teacher-forced positions
per prompt, so 1,536 positions per model. These are positions, not complete
answers. Each row counts all 1,536.

| Model and build | Matching | Disagreements | Near-ties |
| --- | ---: | ---: | ---: |
| Qwen3-8B, main `603f95b` | 1,523 | 13 | 5 |
| Qwen3-8B, `80e616c` | 1,522 | 14 | 7 |
| Qwen3-32B, main `603f95b` | 1,515 | 21 | 10 |
| Qwen3-32B, `80e616c` | 1,517 | 19 | 11 |
| Qwen3-30B-A3B, main `603f95b` | 1,498 | 38 | 15 |
| Qwen3-30B-A3B, `80e616c` | 1,502 | 34 | 16 |

- **Disagreements that are not near-ties:** 8, 11 and 23 on current main;
  7, 8 and 18 on `80e616c`. Near-ties still count as failures.
- Current main ran one session per model.
- `80e616c` is **before** the shared-prefix change (shared9). Those counts
  must not be assigned to current main. Between the two builds, Qwen3-8B
  gained one matching position; Qwen3-32B lost two and Qwen3-30B-A3B lost
  four.
- A missing or partial suite for another model establishes no broader
  agreement.

<details>
<summary><b>Which Qwen3-8B disagreements changed</b> on current main</summary>

On Qwen3-8B the set of disagreements changed, not just the count: 4 earlier
ones now match, and 3 are new (`case-19-code-1024` positions 37 and 41,
`case-21-chat-8192` position 23). Disagreements that are not near-ties rose
from 7 to 8. The new ones are not yet explained.

</details>

## Checks and tools

**Host checks run on any computer; GPU checks need a Spark with its GPU
free.**

### On any computer: host checks

Run these commands from **`engine/`**, not the repository root (which has no
`Cargo.toml`). They follow [CI](../../.github/workflows/ci.yml), whose host
steps set `RUSTFLAGS` to `-D warnings`. Install its pinned Rust 1.99.0 and
nightly-2026-10-01 toolchains and NVPTX components first; see
[AGENTS.md](../../AGENTS.md).

Build and stamp the shared kernels **on every OS**, including macOS, before
checking the host crates. Existing stale PTX is rejected even on a Mac;
missing PTX is accepted by the macOS build, but is not a substitute for the
CI kernel compilation check. A first kernel build can need compiler
dependencies from the network; offline work requires those already cached.

```sh
cargo +1.99.0 kernels
RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
  cargo +nightly-2026-10-01 build --offline -Z build-std=core --release \
  --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
```

On a **Linux host without a GPU**, CI builds a fake CUDA library that aborts
if a device call is attempted. From `engine/`, before the host checks:

```sh
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- ci-no-gpu
export LIBRARY_PATH="$PWD/target/ci-no-gpu${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$PWD/target/ci-no-gpu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Then, on macOS or Linux, still from `engine/`:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

The explicit workspace selection matters: `engine/Cargo.toml` has 13 members
but only eight default members. The NVPTX-only `spark-rust-kernels` package is
compiled separately above. Device-only tests report why they are ignored.

CI also checks the independent benchmark, harness and CLI packages. From each
of `engine/tools/bench/`, `harness/` and `cli/`, run:

```sh
cargo +1.99.0 check --offline --all-targets
```

From each of `engine/tools/bench/` and `harness/`, run:

```sh
cargo +1.99.0 test --offline --all-targets
```

From `cli/`, run its tests including the repository ratchet:

```sh
cargo +1.99.0 test --offline --all-targets -- --nocapture
```

From each of `harness/` and `cli/`, CI also checks Linux-only code with the
installed `aarch64-unknown-linux-gnu` Rust target:

```sh
cargo +1.99.0 check --offline --all-targets --target aarch64-unknown-linux-gnu
```

- On macOS, host tests compile no Linux CUDA paths and run nothing on a GPU.
- Host tests alone establish no GPU correctness or speed result.
- CI has no formatting or Clippy gate; preserve the deliberate source style
  and follow the formatting rule in [AGENTS.md](../../AGENTS.md).
- To check a complete repository export, use the same directories inside it.
  [Export](export.md) explains the preview and release modes.

### On a Spark: GPU tools

<details>
<summary><b>The GPU check tools</b>, and what each one checks</summary>

<dl>
<dt><code>batch-bench</code></dt>
<dd>Inference, and bit-identical logits across batch widths and prefix
splits.</dd>
<dt><code>hf-batch-check</code></dt>
<dd>Tokens and logits against saved official references.</dd>
<dt><code>qwen-reference-check</code></dt>
<dd>Tokenizer and chat format against the reference.</dd>
<dt><code>generate-tokens</code></dt>
<dd>Greedy generation; with and without a draft for Qwen3.8 and Gemma.</dd>
<dt><code>serve-draft-check</code></dt>
<dd>Drafting gives the same tokens as plain decoding, through the server.</dd>
<dt><code>serve-http-check</code></dt>
<dd>HTTP behavior.</dd>
<dt><code>load-check</code></dt>
<dd>Loading and unloading a model.</dd>
<dt><code>tools/bench</code></dt>
<dd>Speed, many concurrent requests, and cache behavior over HTTP.</dd>
</dl>

The [engine](../crates/engine/README.md), [server](../crates/serve/README.md)
and [benchmark](../tools/bench/README.md) guides give each tool's arguments.
Earlier builds' speed and memory are in the historical
[performance measurements](history/fp32-cache-measurements.md).

</details>

### Run the determinism checks

Use goldens captured for the current arithmetic and pinned setup. Older
scalar-build goldens are historical comparison data, not the one-mode byte
baseline. Keep the same checkpoint, topology, rank count and settings when
comparing a run with its goldens.

The Flash pair matrix in `vade test determinism --help` reports
`poison=SKIP|PASS|FAIL` in its final summary. Default builds of `flash-bench`
and `flash-serve` enable `spark-engine/flash-routing-per-row` without
`--features`. The native gate expects the poison cell to run and compare
720 survivor rows (715,161,600 bytes/rank). The merge-receipt form is
`vade test determinism --require-poison` with the normal rank arguments
from `--help`: a missing or skipped poison cell becomes `poison=FAIL` and
exit 1, including a server-only run.

`--no-default-features` (on `cargo build --release --offline --workspace --exclude spark-rust-kernels`
or `vade test build`) turns off only the gate's per-row capability: the
poison cell and the 8-slot matrix. Per-row rejection in the engine and server,
the padded-slot API and the +4-byte gather word are unconditional and stay on.
Server, TP and seqpar dependencies forward the feature without re-enabling
defaults. In such a build the capability is `routing_reject_per_row=0` and
poison reports `0 0 0 SKIP`; SKIP means the gate capability was explicitly
disabled and supplies no evidence of row isolation. It is NOT an aggregate
comparison arm: for a before/after timing of per-row routing, the aggregate
arm is main before isolation (7a26f6ae), never this branch with the feature off.
Default builds report `routing_reject_per_row=1`; an API
rename fails the Linux build instead of silently selecting SKIP.
`--gate-capabilities` prints the feature detector and the GPU crate's
`TEST_ROUTING_REJECT_WIDTH` constant. No source-name search selects SKIP.

On batched decode, the existing output gather carries one extra opaque
32-bit routing/survivor word per rank: +4 bytes/rank per call, with no extra
collective. Candidate token/logit pairs and optional vocabulary shards keep
their original bytes and order; the host strips the word before returning
outputs. Token IDs and returned FP32 logit bytes are unchanged. The word is
already present in `7bc7c49`, including comparison builds; enabling this
host-side feature by default adds no further wire bytes or GPU allocation,
changes no PTX, and keeps one copy of the weights.

The startup-only hook accepts legacy `VADE_TEST_ROUTING_REJECT=N:ROW` or
width-aware `N:ROW:ROWS[:RANK_MASK]`. N counts matching verdicts from 1,
ROWS=0 matches any width, and the optional nonzero decimal rank mask selects
global ranks. These force a status word; they do not inject a NaN. The Flash
distinct-row matrix chooses `1:3` for a legacy hook or `1:3:16` when width
matching is supported. It always needs all three fixtures, including with
the hook set. References and setup remain deferred. For the width-aware
poison step it rejects row 3 inside one sixteen-row call and compares the
other 15 rows (14,899,200 bytes/rank); row 3 is then retired for 47
sixteen-slot calls, comparing another 705 rows (700,262,400 bytes/rank).
All 720 survivor rows (715,161,600 bytes/rank) compare full FP32 bytes against
the same lanes in the no-hook sixteen-row captures. Receipts count 48 actual
batch calls for either hook form and subtract the earlier device peak cap
from the measured peak. Input errors retain exit 2 in the final summary;
gate failures exit 1.

When reconciling the per-row routing feature, retain this three-fixture gate's
validation and deferred reference order. Its separate one-fixture `N:3:8`
safety matrix must not replace them. Keep the `ROUTING_DEFERRED` early return inside
`forced_routing_rejection(rows, rank)`, before hook initialization/counting,
and retain the width capability constant and explicit engine feature.
The distinct-row gate checks status rejection; NaN containment remains
unverified. Routed input is produced inside the captured MoE segment, so a
NaN write before the router needs another default-path check or a kernel
change and is omitted under the gate's zero-cost constraint.

#### Do it yourself

Build the release tools, then compare batched with single-row logits at
every width and six prefix lengths. Any mismatch is a failure.

1. Stop any running server, so the GPU is free.

2. Build the kernels and the release tools.

   ```sh
   cargo kernels
   cargo build --release --offline --workspace --exclude spark-rust-kernels
   ```

3. Set the kernel file and the model.

   ```sh
   PTX=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   MODEL=models/Qwen/Qwen3-8B
   ```

4. Compare batched with single-row logits at prefix 0. `verify` checks every
   batch width from 1 to 64 unless `--verify-widths` narrows it.

   ```sh
   ./target/release/batch-bench "$PTX" "$MODEL" 16 0 verify bf16 \
     --verify-prefixes 0
   ```

5. Repeat across the six prefix lengths, at seven widths.

   ```sh
   ./target/release/batch-bench "$PTX" "$MODEL" 20000 0 verify bf16 \
     --verify-prefixes 0,4090,4096,16380,16384,20000 \
     --verify-widths 1,2,4,8,16,32,64
   ```

#### Ask your coding agent

```text
On this DGX Spark, run Vade's determinism checks from the "Run the
determinism checks" section of docs/validation.md for the model in
models/Qwen/Qwen3-8B. Build the kernels and the release workspace first.
Stop any running Vade server so the GPU is free. Report every mismatch
exactly as printed. Run as the current user, never use sudo, and do not
change fixtures, tolerances or code.
```

## Qwen3 CUDA acceptance suite

**The broad check behind
[Promise 2](#promise-2-agreement-with-the-official-model): 24 prompts per
model, 64 positions each, against the official model run on CUDA.** Each
model has its own folder: `fixtures/qwen3-8b-cuda-suite`,
`fixtures/qwen3-32b-cuda-suite` and `fixtures/qwen3-30b-a3b-cuda-suite`.

<details>
<summary><b>How a suite is built</b>: what it contains, and how the reference was made</summary>

**What a suite contains**

- **24 prompts** per model, with exact lengths from 32 to 28,672 tokens:
  prose, code, math, chat, multi-turn and reasoning, built deterministically
  by the privately retained official-model oracle.
- **64 teacher-forced positions** per prompt.
- The stored input token IDs are authoritative. The generator fills the
  user passage in token space and keeps the chat scaffold, including the
  generation prefix.
- At every position: the reference's greedy token, its top-20 token IDs,
  and their BF16 logits stored as FP32.
- The reference token is fed into the next step even after the end-of-text
  token, so every prompt keeps exactly 64 positions.

**How the reference was made**

<dl>
<dt>Software</dt>
<dd>Transformers 5.18.0, PyTorch 2.14.1+cu130.</dd>
<dt>Weights</dt>
<dd>Official checkpoint, resident in BF16 on CUDA.</dd>
<dt>Mode</dt>
<dd>Batch 1, <code>eval()</code>, <code>inference_mode()</code>, greedy with
lowest-ID tie-break, seed 42, TF32 off.</dd>
<dt>Attention</dt>
<dd>Transformers <code>sdpa</code>, restricted to PyTorch's
<code>FLASH_ATTENTION</code> backend on CUDA.</dd>
<dt>Experts (Qwen3-30B-A3B)</dt>
<dd><code>eager</code> dispatch.</dd>
<dt>Operator overrides</dt>
<dd>PyTorch's optional Python-native CPU/CUDA overrides turned off through its
registry, so built-in ATen operators run without Triton development
headers.</dd>
</dl>

No inference library code is added to Vade. The existing Python environment
is used read-only and offline, with no package installs.

**CPU second opinion.** After the CUDA run, the same checkpoint runs on CPU
with SDPA `MATH` attention and 12 threads.

- It replays the first eight positions of every prompt up to 128 tokens
  long: five prompts, 40 positions per model.
- The CPU receives the CUDA history, not its own, and stores its own greedy
  token and top-20 logits. This is a short-history second opinion, not full
  CPU coverage.
- Qwen3-30B-A3B reloads the checkpoint on CPU after freeing CUDA memory.
  The Qwen3-8B and Qwen3-32B runs moved the weights to CPU one piece at a
  time; their manifests say so. Either way, only one copy is resident at a
  time.

**Manifest.** Each folder's `manifest.json` records framework versions,
attention backends, checkpoint file identities, the documented revision
pins and any local download revision. Where download metadata is missing,
every weight file's SHA256 is recorded instead; a missing revision is
marked unavailable, never guessed. A manifest is marked `complete` only
after both the CUDA and CPU phases finish.

</details>

<details>
<summary><b>What the checker reports</b>, and the full results on <code>80e616c</code></summary>

**What the checker reports**

`hf-batch-check --suite` runs every saved history through Vade's normal
BF16 path: canonical BF16 KV cache, ordinary prefill and decode, no draft
and no fast-prefill option.

- It sizes context and memory from the saved histories, requires 64 CUDA
  rows per prompt, and validates the top-20 arrays.
- It reports agreement at each position (counted from zero), the
  reference's top-1 to top-2 margin at every disagreement, and CPU/CUDA
  differences where CPU rows exist.
- A margin strictly smaller than one BF16 step at the winning logit's size
  is labeled `near_tie=true`. It still fails; no alternate token is
  accepted.
- Top-20 logit error is information only. It is not a tolerance, and it
  says nothing about the full vocabulary.
- Any non-finite Vade logit fails. Peak device bytes include the model and
  the session's allocations.

**Full results**

Measured on main `80e616c`, before the shared-prefix change. The suite merge
added the generator, checker and fixtures without changing inference code.
Matching positions, disagreements and near-ties for this build are the
`80e616c` rows in [Promise 2](#promise-2-agreement-with-the-official-model),
next to the re-run on current main `603f95b`.

*Long prompts* counts matching positions in the two 28,672-token prompts,
out of 128. *CPU vs CUDA* counts positions where the CPU and CUDA reference
rows differ, out of the 40 that have a CPU row. *Peak* is the checker's peak
device memory.

| `80e616c` | Long prompts | CPU vs CUDA | Peak (bytes) |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 123 of 128 | 0 of 40 | 21,063,679,504 |
| Qwen3-32B | 126 of 128 | 0 of 40 | 73,925,135,824 |
| Qwen3-30B-A3B | 127 of 128 | 0 of 40 | 66,248,008,784 |

The first disagreement on each model that is **not** a near-tie:

- **Qwen3-8B:** `case-04-multi-turn-512`, generated position 5. Official
  token 448, Vade token 279. Reference margin 0.125; BF16 step 0.125.
- **Qwen3-32B:** `case-01-code-64`, generated position 27 (input position
  90). Official token 14374, Vade token 8420. Reference margin 0.25; BF16
  step 0.25.
- **Qwen3-30B-A3B:** `case-00-prose-32`, generated position 50 (input
  position 81). Official token 66963, Vade token 320. Reference margin 1.0;
  BF16 step 0.25.

What these show:

- In each case Vade picked the reference's second choice.
- The CPU was not run at the Qwen3-8B prompt's length, and covers only the
  first eight positions of the Qwen3-32B prompt.
- CPU and CUDA agree at all 40 replayed short positions on each model. No
  long CPU history was measured.
- The older short fixtures below use different histories. Do not read one
  set of counts as an improvement or regression of the other.

</details>

<details>
<summary><b>Suite size, reference memory</b> and which weights were used</summary>

**Size and reference memory**

- All 75 suite JSON files total **3,986,384 bytes**.
- Each model stores 203,414 input token IDs. The raw 32-bit prompt IDs,
  reference tokens and top-20 IDs and logits need at least 1,065,560 bytes
  per model, before CPU rows and metadata.
- Full-vocabulary logits are not kept.

Peak CUDA allocation and peak process RSS, in bytes:

| Reference run | CUDA (bytes) | RSS (bytes) |
| --- | ---: | ---: |
| Qwen3-8B | 23,713,592,320 | 19,676,176,384 |
| Qwen3-32B | 78,668,114,944 | 69,438,963,712 |
| Qwen3-30B-A3B | 65,407,931,392 | 114,973,020,160 |

The CUDA figures are PyTorch allocation peaks, not physical unified-memory
totals. RSS (CPU memory in use) covers the whole reference process.

**Which weights were used**

- **Qwen3-8B.** Documented revision
  `b968826d9c46dd6066d109eabc6255188de91218`. The local download revision
  was not kept in the local copy. Every weight file's SHA256 is recorded;
  the link to the documented revision is **unverified**.
- **Qwen3-32B.** Documented revision
  `9216db5781bf21249d130ec9da846c4624c16137`. The local download revision
  was not kept in the local copy. Every weight file's SHA256 is recorded;
  the link to the documented revision is **unverified**.
- **Qwen3-30B-A3B.** Documented revision and local download revision are
  both `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, recorded from download
  metadata.

Manifests: [Qwen3-8B](../fixtures/qwen3-8b-cuda-suite/manifest.json),
[Qwen3-32B](../fixtures/qwen3-32b-cuda-suite/manifest.json),
[Qwen3-30B-A3B](../fixtures/qwen3-30b-a3b-cuda-suite/manifest.json).

</details>

<details>
<summary><b>Vade's serving speed on the suite build</b> (no rival)</summary>

Vade only, on the suite build `80e616c`. ONE is one chat with up to 128
answer tokens; LONG is one prompt of exactly 28,672 tokens with up to 16
answer tokens. [Benchmarks](benchmarks.md) defines the workloads and the
rule for comparing engines.

- Frozen client `1d80e7c`, three fresh default-server sessions per model,
  on the same Spark, ONE before LONG.
- All 18 requests succeeded, each with the expected lengths.
- Values are medians. Peaks are the maximum across the three sessions.

ONE, one chat. Peak is device memory:

| `80e616c` | ONE (tok/s) | Peak (bytes) |
| --- | ---: | ---: |
| Qwen3-8B | 23.643 | 18,694,787,096 |
| Qwen3-32B | 6.807 | 68,314,956,504 |
| Qwen3-30B-A3B | 32.859 | 61,785,559,132 |

LONG, the 28,672-token prompt. Peak is device memory:

| `80e616c` | First text (ms) | Full time (ms) | Peak (bytes) |
| --- | ---: | ---: | ---: |
| Qwen3-8B | 8,274.329 | 9,654.912 | 23,435,127,504 |
| Qwen3-32B | 32,777.194 | 37,954.104 | 76,831,332,112 |
| Qwen3-30B-A3B | 7,846.095 | 8,564.498 | 67,104,091,984 |

**Memory scopes.** The device peaks come from the server log's
`engine_memory` allocation ledger, a different scope from the `nvidia-smi`
samples on [Benchmarks](benchmarks.md). The client also samples RSS, which
leaves out CUDA allocations. Do not add the scopes together or call any of
them a physical unified-memory peak.

Peak RSS reported by the client, for ONE and for LONG:

| `80e616c` | ONE (bytes) | LONG (bytes) |
| --- | ---: | ---: |
| Qwen3-8B | 220,545,024 | 221,655,040 |
| Qwen3-32B | 228,220,928 | 237,252,608 |
| Qwen3-30B-A3B | 207,798,272 | 219,836,416 |

Commands, with `MODEL`, `ID`, `URL` and `BENCH` set as in
[Benchmarks](benchmarks.md#reproduce) and a fresh server for each session:

```sh
./target/release/spark-serve "$MODEL" & PID=$!
until curl -sf "$URL/health" | grep -q '"status":"ready"'; do sleep 0.1; done
"$BENCH" "$URL" --model "$ID" --tokenizer "$MODEL" --repeats 1 \
  --timeout-secs 900 --budget-secs 1200 --server-pid "$PID" \
  --only concurrency --concurrency 1 --max-tokens 128
"$BENCH" "$URL" --model "$ID" --tokenizer "$MODEL" --repeats 1 \
  --timeout-secs 900 --budget-secs 1200 --server-pid "$PID" \
  --only long --long-target 28672 --long-output 16
```

</details>

### Run the suite yourself

#### Do it yourself

Build the checker, run it on the 24 saved histories, and compare the totals
with Promise 2. Every disagreement, old or new, is a failure.

1. Stop any running server, so the GPU is free.

2. Build the kernels and the checker.

   ```sh
   cargo kernels
   cargo build --release --offline -p spark-engine --bin hf-batch-check
   ```

3. Set the kernel file, model and suite. Change the last two lines for
   another model.

   ```sh
   PTX=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
   MODEL=models/Qwen/Qwen3-8B
   FIXTURES=fixtures/qwen3-8b-cuda-suite
   ```

4. Run the checker on all 24 saved histories.

   ```sh
   ./target/release/hf-batch-check "$PTX" "$MODEL" \
     "$FIXTURES"/case-*.json --suite
   ```

5. Compare the totals with
   [Promise 2](#promise-2-agreement-with-the-official-model), using the row
   for the build you ran. Report every difference. Every disagreement, old
   or new, is a failure.

#### Ask your coding agent

```text
On this DGX Spark, check Vade against the saved Qwen3 CUDA reference suite
by following "Run the suite yourself" in docs/validation.md. Use
models/Qwen/Qwen3-8B with fixtures/qwen3-8b-cuda-suite. Stop any running
Vade server first. Report the matching-position count, every disagreement
with its margin and near-tie label, and the peak device bytes. Do not
regenerate the references, install packages, change fixtures or use sudo.
```

<details>
<summary><b>Regenerating the references</b>: only to rebuild a suite</summary>

**Regenerating the references** requires the official-model oracle and its
pinned Python environment, both retained privately outside this repository.
Use a new empty output directory under an assigned owner task; do not replace
the committed fixtures during an ordinary engine check.
[Reference provenance](../tools/truth/README.md) records where the oracle fits.

</details>

## Older short fixtures

**Narrower checks, still run: 55 of 56, 56 of 56 and 50 of 50 on the saved
greedy positions, and the expert model's long-prompt routing bound still
fails.**

Before the suites, each Qwen3 model had three short saved references in
`fixtures/qwen3`, `fixtures/qwen3-32b` and `fixtures/qwen3-30b-a3b`. They
are kept and still run. These checks are narrower than full agreement with
the official model.

| Check | Qwen3-8B | Qwen3-32B | Qwen3-30B-A3B |
| --- | ---: | ---: | ---: |
| 16 requests, batched vs alone | 16 of 16 | 16 of 16 | 16 of 16 |
| Greedy positions on the fixtures | 55 of 56 | 56 of 56 | 50 of 50 |
| Reference precision | FP32 | FP32 | BF16 |

- **16 requests:** replies match. **Greedy positions:** on Qwen3-30B-A3B
  they are 50 of 50 token IDs, and the long-prompt routing bound fails.
- **Reference precision:** the FP32 references were computed from the BF16
  checkpoint. The Qwen3-30B-A3B reference is the checkpoint's own BF16, run
  on CPU.
- Results are from the final gate of the shared-prompt attention merge.
  Before it, the mixture-of-experts fixture gave 49 of 50.

<details>
<summary><b>What the older fixtures cover</b>, and what they do not prove</summary>

**Scope**

- Fixture agreement covers only the recorded model, precision, prompts and
  output lengths.
- These counts include disagreements. They do not establish exact agreement
  with the official CUDA model.
- The mixture-of-experts long-prompt routing bound still fails: 25 changed
  expert choices, against 23 allowed.
- The one differing Qwen3-8B long-fixture position has been described as a
  near-tie. That does not waive the mismatch.

**Tolerances and precision**

- The older checker's numerical tolerance is looser than the expert
  fixture's stored bounds. A successful exit means the tokens matched, not
  that those bounds held. The CUDA suites require exact top-1 agreement at
  every saved position.
- The Qwen3-30B-A3B fixtures come from the official model at its own
  precision, BF16. `fixtures/qwen3-30b-a3b-fp32` keeps an earlier FP32 run
  as a debugging reference. The `fixtures/qwen3-30b-a3b` README records the
  difference between the two runs, which sets the check's tolerances.
- The dense FP32 fixtures were made from the BF16 checkpoint and remain as
  debugging references.

</details>

### Run the short check yourself

#### Do it yourself

Build the checker and run it on the bundled Qwen3-8B fixtures. One known
miss, on the long case, is expected and still counts as a failure.

1. Stop the server, so the GPU is free. [Server and API](server.md#stopping)
   shows how.

2. Build the kernels and the checker, from the checkout root.

   ```sh
   cargo kernels
   cargo +1.99.0 build --release -p spark-engine --bin hf-batch-check
   ```

3. Run it on the bundled Qwen3-8B fixtures.

   ```sh
   ./target/release/hf-batch-check \
     target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx \
     models/Qwen/Qwen3-8B \
     fixtures/qwen3/short.json fixtures/qwen3/medium.json \
     fixtures/qwen3/long.json --bf16-kv
   ```

   It prints one `HF fixture=` line per file, with `top1` (matching
   positions), the logit errors, KV cache bytes and peak GPU memory.

4. Read the result. On these Qwen3-8B files the checker may end with
   `FAIL 1 HF comparisons failed`. If the only miss is the long case
   (`top1=23/24`), that is the known one. It stays an exact-agreement
   failure, even when its reference margin makes it a near-tie.

For Qwen3-32B, use its model folder and the files in `fixtures/qwen3-32b/`.
For the mixture-of-experts model, use `fixtures/qwen3-30b-a3b/`.

#### Ask your coding agent

```text
Run Vade's short reference check for Qwen3-8B by following "Run the short
check yourself" in docs/validation.md. Stop the Vade server first, build the
kernels and the checker, and run it on the bundled fixtures. Report each "HF
fixture=" line and the final result exactly as printed. Do not change
fixtures, tolerances or code, and never use sudo.
```

### Expert model: CPU and CUDA references disagree

**At one Qwen3-30B-A3B history, the official model itself picks a different
token on CPU than on CUDA.** Fixtures and tolerances were not changed.

<details>
<summary><b>The full record</b>, and later results on the same fixture</summary>

Recorded before the shared-prefix change.

Vade's canonical expert decode was compared with the same official
checkpoint on CUDA (PyTorch 2.14.1+cu130, Transformers 5.18.0, resident BF16
weights, eager attention and eager experts).

- The short and long greedy tokens matched the CPU fixtures.
- Medium free generation changed at generated index 10, so the CPU history
  was also replayed one cached token at a time on CUDA.
- At generated index 14, official CUDA chose token 264, as Vade did then:
  logits 36.25 for 264 and 36.00 for 2441. The CPU chose 2441, with logits
  37.00 and 35.25.
- CUDA replay also changed index 10, from token 6983 to 31191.
- The replay peaked at 61,890,547,712 CUDA allocated bytes; free generation
  at 62,658,961,408 bytes. These are framework allocation peaks, not
  physical unified-memory measurements, and not engine speed results.

So the official model itself varies between CPU and CUDA at that history.
Fixtures and tolerances were not changed.

Later results on the same fixture: 50 of 50 token IDs on the shared-prefix
build `39aceb2` and again on main `803bd3d`, against 49 of 50 on main
`03068e9` before it. The long-prompt routing bound still fails on all three
builds: 25 changed expert choices against 23 allowed.

</details>

## Qwen3.8 and Gemma scope

**Qwen3.8-27B and Gemma 4 31B have narrower checks than the Qwen3 suites.
Qwen3.8-27B BF16's two earlier gaps are fixed on main `eb67c8b`.**

<details>
<summary><b>Every recorded Qwen3.8 and Gemma check</b></summary>

Qwen3.8 keeps BF16 operation boundaries and FP32 recurrent state (the
running memory its linear-attention layers carry). Official FP8 weights stay
at checkpoint precision and use the checkpoint's BF16 scale conversion.

**Qwen3.8-27B BF16**

- **Earlier saved official IDs:** 224 of 224 at the 27B merge gate.
- **Official reference rebuilt** with Transformers 5.18.0 and PyTorch
  2.14.1+cu130, four prompts, 224 IDs: code, JSON and quotes match. **Prose
  diverges at generated index 27**: Vade scores tokens 264 and 1040 equal at
  21.5 and picks 264; the official model scores 1040 at 21.625 and 264 at
  21.5, a margin of 0.125.
- **A 28,672-token prompt read in one pass against split after 255
  tokens:** the same 16 generated IDs, but **240,741 logit bits differ**
  (largest difference 0.625). Splits at 256 and 4,096 are bit-exact.
- **Built-in MTP drafting against plain:** 16 of 16 replies and 2,048 IDs
  equal (`39aceb2`).
- **Merged build (`8b2183c`, main `eb67c8b`):** 224 of 224 official IDs,
  plain and with MTP drafting, prose included. A 28,672-token prompt split
  at any of six points is bit-exact, and decoding then prefilling matches
  cold.

**Qwen3.8-27B FP8**

- **Official reference IDs:** 31 of 31 at the 27B merge gate.
- **16-chat workload IDs:** 2,048 of 2,048 match at the 27B merge gate.

**Gemma 4 31B**

- **Official saved token IDs:** 96 of 96.
- **Official assistant draft against plain:** 96 of 96 IDs (Gemma merge
  gate).
- **Saved long-prompt IDs:** 16 of 16.
- **Batch of 16 against single row:** equal logits, three prompts for eight
  steps (Gemma merge gate).

**Limits**

- The two Qwen3.8-27B BF16 gaps above were measured on main `803bd3d`;
  both are fixed on main `eb67c8b`.
- Greedy agreement applies only to the recorded model, prompt and token
  count.
- A change to shared kernels needs both the Qwen3 and the Qwen3.8 gates on
  the same Spark.
- Gemma's merge-gate speed is on
  [Benchmarks](benchmarks.md#gemma-4-31b-and-qwen38-27b-vade-only). No rival
  comparison follows from it.

</details>

## Not proven yet

**Five gaps are open, and none is hidden in the counts above.**

Determinism:

- **The prefix sweep** covers seven widths at six splits, not every width
  at every split.

Agreement with the official model:

- **Official agreement is not perfect.** On main `603f95b` (unchanged on
  `eb67c8b`) the
  suites record 72 disagreements across three Qwen3 models; 42 are not
  near-ties. Three Qwen3-8B disagreements that are new since `80e616c` are
  unexplained.
- **Qwen3.8 and Gemma** have no broad suite like the Qwen3 one.
- **Download provenance.** The Qwen3-8B and Qwen3-32B local weights are not
  tied to their documented revisions.
- **Long CPU histories** were not measured.

**Where next:** [Benchmarks](benchmarks.md) · [Models](models.md) ·
[Documentation index](README.md)
