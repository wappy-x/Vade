# spark-tp

**This crate runs one dense Qwen3 model across two or more cabled Sparks.
Each Spark holds a share of every layer, and all of them work on every token
together.** This is called tensor parallelism (TP): one model uses the memory
and bandwidth of several Sparks at once.

This page is for developers. The user guide, with cabling, SSH and the full
serving recipe, is [Multiple Sparks](../../docs/multi-spark.md).

<details>
<summary><b>Terms on this page</b>: rank, ring, path, QSFP, RDMA, KV cache, prefill, decode</summary>

<dl>
<dt>Rank</dt>
<dd>One Spark's process in the group. Rank 0 serves the API.</dd>
<dt>Ring, path</dt>
<dd>How the Sparks are cabled. A ring closes back on itself; a path, called a
line in the user guide, does not.</dd>
<dt>QSFP</dt>
<dd>The Spark's high-speed network ports.</dd>
<dt>RDMA</dt>
<dd>Copying memory straight between machines over the cable, without the usual
network stack. TCP is the ordinary network fallback.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the tokens it has read.</dd>
<dt>Prefill, decode</dt>
<dd>Reading the prompt, then producing the answer one token at a time.</dd>
</dl>

</details>

## Status at a glance

**With `tp-serve`, Qwen3-8B and Qwen3-32B run across Sparks. For Qwen3-8B we
have recorded the same tokens as on one Spark; speed across Sparks is not
measured on this build.**

| Item | Status |
| --- | --- |
| Qwen3-8B, Qwen3-32B (BF16) on `tp-serve` | Supported |
| Other models on `tp-serve` | Not supported |
| Qwen3.8-Flash-Next across Sparks | `flash-serve`, [below](#flash-serve-for-qwen38-flash-next) |
| Rank counts | Any from 2 to 64 |
| Transport | RDMA or TCP |
| Serving | `tp-serve` (rank 0) |
| Same tokens as one Spark | Recorded, Qwen3-8B |
| Speed across Sparks | Not measured |
| Three or more physical Sparks | Not re-checked |
| Qwen3-32B across ranks | Not re-checked |

- **Models:** `tp-serve` supports dense Qwen3-8B and Qwen3-32B, official
  BF16. Its typed FP8 route for hybrid Qwen3.5, 3.6 and 3.8 checkpoints is
  unchecked and not supported. Qwen3-30B-A3B, Qwen3.8-27B and Gemma 4 31B run
  on one Spark with [spark-serve](../serve/README.md). Qwen3.8-Flash-Next runs
  across Sparks with this crate's other server, `flash-serve`.
- **Transport and serving:** RDMA over the QSFP cables, or TCP; `tp-serve`
  serves the chat and completion APIs on rank 0.
- **Same tokens:** recorded with one and two ranks, on the short, medium and
  long checks. Earlier speed results are [historical](MEASUREMENTS.md).
- **Not re-checked on this build:** three or more physical Sparks (the
  launcher supports them), and Qwen3-32B across ranks (the loader and TP path
  exist). Details: [what has been tested](../../docs/multi-spark.md#what-has-been-tested).

The `flash-serve` server for Qwen3.8-Flash-Next is covered
[at the end of this page](#flash-serve-for-qwen38-flash-next).

## Where it fits

**TP reuses the one-Spark kernels, GPU layer and server library.**

<dl>
<dt><a href="../engine/README.md">spark-engine</a>,
<a href="../gpu/README.md">spark-gpu-probe</a></dt>
<dd>The same BF16 kernels and GPU layer as one Spark.</dd>
<dt><a href="../serve/README.md">spark-serve</a></dt>
<dd><code>tp-serve</code> builds on its library.</dd>
<dt><a href="../seqpar/README.md">spark-seqpar</a></dt>
<dd>Provides the sequence-parallel prefill for hybrid mode.</dd>
<dt><a href="../model/README.md">engine-model</a>,
<a href="../tokenizer/README.md">engine-tokenizer</a></dt>
<dd>Checkpoint reading and chat tokenization.</dd>
</dl>

## Build

**Three steps on every Spark: shared kernels and TP programs, TP kernels,
then one combined kernel file.** Run them from `engine/`, after
[Install](../../docs/install.md).

### Do it yourself

1. Build the shared kernels and the TP programs.

   ```sh
   cargo +1.99.0 kernels
   cargo +1.99.0 build --release --offline -p spark-tp
   ```

2. Build the TP kernels.

   ```sh
   RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
     cargo +nightly-2026-10-01 build --offline -Z build-std=core --release \
     --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
   ```

3. Combine both kernel files (PTX, NVIDIA's text format for GPU code) into
   `target/tp.ptx`.

   ```sh
   ./target/release/tp-ptx \
     target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx \
     crates/tp/kernels/target/nvptx64-nvidia-cuda/release/spark_tp_kernels.ptx \
     target/tp.ptx
   ```

<details>
<summary><b>Build notes, and host-only checks</b></summary>

- The launcher starts `target/release/<program>` inside the checkout on each
  Spark, so every Spark needs the same release build.
- The TP kernel file also imports the shared canonical multiply helper and
  blocked weight layout, for its debugging matrix-vector path.
- Native BF16 decode loads from `target/tp.ptx`. Layer kernels are embedded at
  build time from the shared kernel build.

Host-only checks test no GPU or cable. Run them from `engine/` after building
the shared kernels, on either macOS or Linux. They match
[CI](../../../.github/workflows/ci.yml); use its `RUSTFLAGS=-D warnings` setting
and the [Linux fake-CUDA setup](../../docs/validation.md#on-any-computer-host-checks)
when no CUDA library is available:

```sh
cargo +1.99.0 check --offline --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
```

</details>

## Serve a model across Sparks

**One launcher command starts every rank; rank 0 serves the API.**
[Multiple Sparks](../../docs/multi-spark.md#serve-one-model-from-a-group)
is the step-by-step guide, with a coding-agent prompt. The core of it, from
the launching Spark, with a fresh `SEQPAR_CONTROL_SECRET` exported:

```sh
./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
  tp-serve target/tp.ptx 'models/Qwen/Qwen3-8B' 8192 "$BIND" 0
```

`SPARKS` lists existing SSH aliases in cable order. `REMOTE_DIRECTORY` is the
checkout on each Spark. `BIND` is rank 0's loopback HTTP socket. Wait for
`TP service ready`; Ctrl-C on the launcher stops every rank.

<details>
<summary><b>The <code>tp-serve</code> arguments, in order</b></summary>

1. **Kernel file**, `target/tp.ptx`: the combined kernels from the build.
2. **Model folder**, `'models/Qwen/Qwen3-8B'`: read on each Spark, relative
   to its checkout unless absolute.
3. **Context length**, `8192`: prompt plus answer, in tokens.
4. **HTTP address**, `"$BIND"`: rank 0's loopback socket.
5. **Request count**, `0`: `0` serves until stopped.

</details>

<details>
<summary><b>Other programs</b>: generate, benchmark, reference check, network plan</summary>

All of them run through the launcher.

<dl>
<dt><code>tp-generate</code></dt>
<dd>Generates tokens from a prompt of token IDs and reports speed and
memory.</dd>
<dt><code>tp-bench</code></dt>
<dd><code>check</code> compares batched and one-at-a-time results;
<code>bench</code> measures speed.</dd>
<dt><code>tp-hf-check</code></dt>
<dd>Compares with the <a href="../../fixtures/qwen3/README.md">Hugging Face
reference fixtures</a>.</dd>
<dt><code>tp-ptx</code></dt>
<dd>Combines the shared and TP kernel files.</dd>
<dt><code>tp-launch net-plan</code></dt>
<dd>Inspects cabled Sparks over SSH and prints a network plan. It changes
nothing.</dd>
</dl>

Examples, from the checkout root on the launching Spark. Replace the aliases
and paths with the real ones.

```sh
./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
  tp-generate target/tp.ptx 'models/Qwen/Qwen3-8B' 16 1 --tokens-file prompt.ids

./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
  tp-bench target/tp.ptx 'models/Qwen/Qwen3-8B' 1024 32 3 check

./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
  tp-hf-check target/tp.ptx 'models/Qwen/Qwen3-8B' \
  fixtures/qwen3/short.json fixtures/qwen3/medium.json fixtures/qwen3/long.json
```

- **`tp-generate`:** the token file holds one ID per line. Only rank 0 reads
  it, from its checkout. `seq 1 2048 > prompt.ids` makes a test file. Add
  `--seqpar` to use hybrid prefill; it needs at least 128 prompt tokens per
  rank.
- **`tp-bench`:** takes the longest prompt, the answer length and the repeat
  count. It warms up, then reports the slower rank's median time and each
  rank's memory.
- **Two ranks on one Spark, for testing:**
  `tp-launch --dir "$REMOTE_DIRECTORY" local "$SPARK" -- 2 PROGRAM ARGS...`

</details>

## How `tp-serve` differs from `spark-serve`

**`tp-serve` is the smaller server: short answers, loopback only, no
drafting and no model swap.**

| Item | `tp-serve` | `spark-serve` |
| --- | --- | --- |
| Answer length (tokens) | At most 256 | Remaining context |
| Prompt length | Context minus one | Sized at start |
| Listens on | Loopback only | Loopback, or any |
| Draft decoding | No | Yes |
| Model swap | No | Unfinished |
| A failed rank | Stops the group | Single process |

- `tp-serve` answers are also no longer than the context. `spark-serve`
  answers are capped by an explicit request or server limit.
- `spark-serve` listens beyond loopback only with `--allow-remote`.
- `spark-serve` model swap is unfinished; restart to change models.
- In `tp-serve`, a failed rank stops the group. Relaunch it.

## How it works

**Each rank keeps one share of the weights, and the KV cache for its own
heads. Every sum still runs over its full length, in the same order as on one
Spark.** The kernels are the same BF16 kernels as on one Spark.

Hybrid mode keeps a full copy of the weights on each rank. With
`TP_SERVE_SEQUENCE_PREFILL=1` it reads long prompts with sequence-parallel
prefill from [spark-seqpar](../seqpar/README.md), then decodes with tensor
parallelism. That hand-off keeps FP32 KV, so its answers can differ from one
Spark's. Without the opt-in, hybrid keeps BF16 KV and does not split prompts.

<details>
<summary><b>How the model is split, and why the answer matches one Spark</b></summary>

<dl>
<dt>Query, key and value projections</dt>
<dd>By whole groups of heads that share a KV head.</dd>
<dt>Gate and up projections</dt>
<dd>By rows of the feed-forward layer.</dd>
<dt>Output and down projections</dt>
<dd>By rows of their output.</dd>
</dl>

Embeddings are read from disk when needed.

**Why the answer matches one Spark.** Before the output and down projections,
ranks gather their inputs. Every sum then runs over the full length, in the
same order as on one Spark.

1. The 16-column BF16 multiply tiles are split into four fixed ranges, with
   boundary `i` at `floor(i * tile_count / 4)`.
2. FP32 partial sums merge as `(range0 + range1) + (range2 + range3)`.
3. Ranks never add up separate column shares, which would change that order.

TP uses the same BF16 kernels as one Spark: BF16 inputs with FP32 sums. The
final scores are rounded to BF16 before a token is chosen, as on one Spark.

</details>

<details>
<summary><b>Hybrid mode, step by step</b></summary>

1. Each rank keeps one full copy of the weights, with views of its share.
2. Prompts and answers run on the shares, as in shard-only mode (where each
   rank loads only its share), with a BF16 KV cache. Answers match one Spark
   token for token.

**Prompt splitting is opt-in** (`TP_SERVE_SEQUENCE_PREFILL=1`, hybrid only):

1. Each rank reads one slice of a prompt of 6,144 tokens or more.
2. The first token goes out straight away, while the KV cache is handed over
   in pieces.
3. Decoding continues with each rank reading its share of the weights.

With the opt-in, every hybrid session keeps an FP32 KV cache for that
hand-off, not the model's BF16, so answers can differ from one Spark. That is
why it is off by default.

Without the opt-in, long prompts are read with TP alone. Before you turn it
on, check the output tokens for your exact model and requests; fast historical
results alone are not enough.

Without prompt splitting, hybrid mode only costs memory: its chats do not
share a pass. So `TP_SERVE_HYBRID=auto`, the default, picks it only with
`TP_SERVE_SEQUENCE_PREFILL=1`, a context of at least 6,144 tokens and room on
every rank for a full model plus the serving cache. Otherwise the ranks load
only their weight shares.

</details>

## Settings

**Set these on the launching Spark; the launcher passes them to every rank.**
If two neighbouring ranks share a machine, the whole run uses TCP.

<details>
<summary><b>All settings, with defaults</b></summary>

<dl>
<dt><code>SEQPAR_CONTROL_SECRET</code></dt>
<dd>Shared secret. Required across machines.</dd>
<dt><code>TP_REMOTE_DIR</code></dt>
<dd>Checkout on each Spark. <code>--dir</code> before <code>run</code> overrides
it. Default: none.</dd>
<dt><code>TP_SERVE_MAX_ACTIVE</code></dt>
<dd>Chats at once, 2 to 32. Clients share priority 0. Default: 2.</dd>
<dt><code>TP_SERVE_HYBRID</code></dt>
<dd><code>auto</code>, <code>0</code> (shares only) or <code>1</code> (require
hybrid). Default: <code>auto</code> (hybrid only with
<code>TP_SERVE_SEQUENCE_PREFILL=1</code>).</dd>
<dt><code>TP_SERVE_SEQUENCE_PREFILL</code></dt>
<dd><code>1</code> splits long prompts across ranks in hybrid mode. Its FP32
KV changes answers. Default: <code>0</code>.</dd>
<dt><code>TP_SERVE_GENERATION_TIMEOUT_MS</code></dt>
<dd>Generation deadline in milliseconds, reset on admission. Default:
3,600,000.</dd>
<dt><code>TP_RDMA</code></dt>
<dd><code>0</code> uses TCP. Also covers sequence prefill unless
<code>SEQPAR_RDMA</code> is set. Default: RDMA.</dd>
<dt><code>TP_DUAL_RAIL</code></dt>
<dd><code>0</code> uses one network link (rail) per cable when a cable carries
two. Default: both.</dd>
<dt><code>TP_RDMA_SMALL</code></dt>
<dd><code>0</code> turns off the small preposted message buffers. Default:
on.</dd>
<dt><code>TP_TOPOLOGY</code></dt>
<dd><code>path</code> forces a path even when a ring is cabled. Default: a ring
if possible.</dd>
<dt><code>TP_PORT</code></dt>
<dd>First port for rank connections. Default: 29470.</dd>
<dt><code>TP_CPU</code></dt>
<dd>Pins each rank's main thread to this CPU core. Default: not pinned.</dd>
<dt><code>TP_BUSY_POLL_US</code></dt>
<dd>Socket busy-poll time in microseconds, for TCP. Default: off.</dd>
<dt><code>TP_DEADLINE_UNIX</code></dt>
<dd>Stops every rank at this Unix time. Default: none.</dd>
</dl>

</details>

## Security

**Every connection between machines needs the same `SEQPAR_CONTROL_SECRET`.
Traffic after that check is not encrypted, so use a private, trusted network
only.**

<details>
<summary><b>How the secret and the start-up check work</b></summary>

- The launcher sends the secret over SSH input, never on a command line.
- Ranks prove they know the secret with a challenge. The secret is never sent
  between ranks. A wrong or missing secret is rejected within 800 ms.

Before loading the model, ranks also check that they agree on the program,
its arguments, the topology and the transport options. They do not compare
the model files themselves.

</details>

## flash-serve for Qwen3.8-Flash-Next

`flash-serve` serves Qwen3.8-Flash-Next. It is also the server behind the
release default, the FP4-Local checkpoint that `vade up` starts with a
32,768-token context. This section covers the official FP8 checkpoint across
ranks, which `vade up --fp8-legacy` selects on two Sparks.

By default, a cold chat (nothing cached) running on its own uses strict
multi-token prediction (MTP) at depth 2: it guesses up to two tokens, then the
main model checks them. Strict drafting keeps the main model's token choices,
including for fixed-seed sampled requests. Two cases decode plainly instead:

- Chats that overlap with others, until they finish, even after the other
  chats end.
- Cached continuations without retained draft history. They reuse the main
  model's state without rebuilding draft history.

Pass `--mtp 0` on every rank to keep plain decoding, or choose a depth with
`--mtp 1..3`. `vade up --fp8-legacy` uses the same depth-2 default; `MTP=0`
or `vade up --fp8-legacy --mtp 0` keeps it plain. A request can also select
`"draft":"plain"`. The response's `model_draft_used` says whether model drafting
ran; the configured `mtp_depth` is only its ceiling.

Automatic memory sizing includes the MTP weights and the shared drafting
workspace before assigning room to sessions and the prompt cache. Both ranks
must retain their model weights in device memory. The depth-2 policy was chosen
from earlier single-chat measurements; the new default's startup, token
identity and timing still need a new check on two real Sparks. A check run on a
Mac does not count.

In the historical memory measurement (two ranks, 8192-token context), each
rank needed:

- 1,383,226,112 extra bytes of resident weights;
- a 191,993,600-byte shared workspace.

The default prefix-cache budget fell from 2,338,127,184 to 414,558,032 bytes
per rank, so long multi-turn histories could reuse less. These figures
describe that earlier build, not the current automatic memory plan.

**The FP8 checkpoint defaults to a context of 131,072 tokens.** To choose
another, give a context length as the `flash-serve` argument right after the
model folder. When you leave it out, the HTTP bind address takes that place.
`vade up --fp8-legacy` uses the same `131072` default, and an explicit context
value still overrides it. This default has not yet been checked on a real
Spark; older timing comparisons use an explicit 32,768-token context. See
[Server and API](../../docs/server.md#start-it).

The output cap defaults to 65,536 tokens, bounded by context. Explicit limits
may equal context, and replies are capped at the room left after the tokenized
prompt. Automatic pool and cache sizing makes room for one full-context
session, or refuses to start with a plain memory message.

**Where next:** [Multiple Sparks](../../docs/multi-spark.md) ·
[Historical scale measurements](MEASUREMENTS.md) ·
[Shared multi-Spark hooks](MULTI-SPARK.md)
