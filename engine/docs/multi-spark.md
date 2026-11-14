# Multiple Sparks

This page covers the older BF16 `tp-serve` path. FP4-Local uses `flash-serve`
instead. The [launcher](../crates/tp/src/bin/flash-serve.rs) selects the tensor
split on two Sparks, the split in parts on more Sparks over RDMA when every
split width cuts into whole parts and the mailbox has room for that many ranks
(four for the shipped checkpoint), and the layer chain otherwise. The
[`vade up` configuration limit](../../cli/src/config.rs) allows up to 24 Sparks.
These are code paths and launch limits, not device test results. See
[FP4-Local serving](next-http.md) for that path's settings.

**Qwen3-8B and Qwen3-32B, official BF16, can be split across several DGX Sparks joined by their QSFP network cables.**

Checked on this build: Qwen3-8B gives the same tokens on one and on two
Sparks. Speed across Sparks is not measured yet.

This is tensor parallelism (TP): each Spark holds a share of every layer, and
all of them compute every token together. For anyone with two or more Sparks,
by hand or with a coding agent; start with [Install](install.md) on each.

- **Server:** `tp-serve`, separate from the single-Spark `spark-serve`.
- **Groups:** lines, rings and odd counts, up to 64 ranks.
- **Transport:** RDMA over the QSFP cables, or TCP.
- **Outside this `tp-serve` path:** Qwen3-30B-A3B, Qwen3.8-27B (BF16 and FP8)
  and Gemma 4 31B.

<details>
<summary><b>Terms on this page</b>: dense model, tensor parallelism, rank, QSFP, rail, line, ring, RDMA, PTX, KV cache, prefill</summary>

<dl>
<dt>Dense model</dt>
<dd>A model where every token uses all of the weights: Qwen3-8B and
Qwen3-32B.</dd>
<dt>Tensor parallelism (TP)</dt>
<dd>Splitting each layer's weights across Sparks, so they compute every token
together.</dd>
<dt>Rank</dt>
<dd>One Spark's process in the group, numbered from 0 in cable order. Rank 0
runs the HTTP server.</dd>
<dt>QSFP</dt>
<dd>The Spark's high-speed network ports.</dd>
<dt>Rail</dt>
<dd>One of the two network interfaces each QSFP port presents.</dd>
<dt>Line, ring</dt>
<dd>A line joins each rank to the next. A ring adds a cable from the last rank
back to rank 0.</dd>
<dt>RDMA</dt>
<dd>The network card copies memory straight between machines, without the usual
network stack.</dd>
<dt>PTX</dt>
<dd>NVIDIA's portable GPU instruction format.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Prefill, decode</dt>
<dd>Reading the prompt; then writing the answer, one token per step.</dd>
</dl>

</details>

## What has been tested

**The checks below apply to the older BF16 `tp-serve` path.**

| Item | On this build |
| --- | --- |
| Qwen3-8B, one and two ranks | Same tokens |
| Qwen3-32B across ranks | Not requalified |
| More than two physical ranks | Not requalified |
| Speed across ranks | Not measured |
| Expert, Qwen3.8, Gemma models | Not supported by these `tp-serve` adapters; FP4-Local uses `flash-serve` |

- **Same tokens:** on the retained short, medium and long checks.
- **Qwen3-32B:** the loader and TP path exist. **More than two ranks:**
  supported by the launcher.
- **Speed:** earlier results are [historical](../crates/tp/MEASUREMENTS.md).
  More ranks mean fewer weight bytes per rank, but more communication. Any
  speedup must be measured on the actual group.
- **Other models:** these `tp-serve` adapters do not serve expert, Qwen3.8 or
  Gemma models across ranks. FP4-Local uses the separate path described above.

## What you need

**The same software on every Spark, passwordless SSH, healthy direct cables and an idle GPU.**

- **The same software everywhere.** The same built Vade revision, PTX
  kernels, official model files and settings on every Spark.
- **SSH.** Passwordless SSH from the launching Spark to every rank, including
  itself, through configured aliases. These docs contain no host identities.
- **Cables and links.** Direct QSFP cables, healthy links and existing link
  addresses. Network setup is outside Vade; these steps change nothing.
- **An idle GPU.** No other GPU users during loading and measurement.

At startup, ranks compare their programs and settings, but not every byte of
the model files. Keep the same verified model revision on each rank.

## Cabling

**A pair needs one cable; a line joins each rank to the next, and a ring adds a cable from the last rank back to rank 0.**

If link state is missing or degraded, stop and use the Spark's existing
administration process: Vade does not repair links.

<details>
<summary><b>Rails, subnets and the two read-only network tools</b></summary>

Each port presents two rails. Vade uses both when both have matching link
addresses, and one otherwise. Each rail needs its own subnet and a matching
maximum packet size (MTU) at both ends. Those settings must already exist.

- `tp-launch discover` reports link state without changing it.
- `tp-launch net-plan` prints a proposed network plan. It does not apply one.

</details>

## Serve one model from a group

<a id="run-a-dense-model-across-neighbours"></a>

**Build on every rank, check SSH and links, start the group from one Spark, and talk to rank 0 like a single server.**

### Do it yourself

Set these shell variables on the launching Spark first. They are operator
configuration; never commit their values.

<dl>
<dt><code>SPARKS</code></dt>
<dd>A shell array of existing SSH aliases, in cable order.</dd>
<dt><code>REMOTE_DIRECTORY</code></dt>
<dd>The checkout path on each rank.</dd>
<dt><code>REMOTE_MODEL</code></dt>
<dd>The model path on each rank, relative to that checkout if you like.</dd>
<dt><code>BIND</code></dt>
<dd>Rank 0's loopback socket for HTTP.</dd>
<dt><code>URL</code></dt>
<dd>The matching HTTP origin.</dd>
</dl>

1. **Prepare every Spark.** Install the same revision and model on every
   rank, and stop any running Vade instance. Then build the shared kernels and
   the TP programs:

   ```sh
   # Run from the checkout root, on every rank.
   cargo kernels
   cargo +1.99.0 build --release -p spark-tp
   ```

2. **Build and combine the TP kernels** on each rank. Each checkout needs the
   same release programs and both kernel files.

   ```sh
   RUSTFLAGS='-C target-cpu=sm_121 -C overflow-checks=off' \
     cargo +nightly-2026-10-01 build -Z build-std=core --release \
     --target nvptx64-nvidia-cuda --manifest-path crates/tp/kernels/Cargo.toml
   ./target/release/tp-ptx \
     target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx \
     crates/tp/kernels/target/nvptx64-nvidia-cuda/release/spark_tp_kernels.ptx \
     target/tp.ptx
   ```

3. **Check SSH** from the launching rank. Every command must finish without a
   password prompt; stop if access is missing. These steps do not create
   accounts or change SSH configuration.

   ```sh
   for rank in "${SPARKS[@]}"; do ssh "$rank" true || break; done
   ```

4. **Check the links.** On every rank, run the command below. Each line shows
   a fast network interface, its address with prefix length, and its speed in
   Mb/s. Confirm a healthy direct link between each pair of neighbours; a
   missing link stops the launcher.

   ```sh
   ./target/release/tp-launch discover
   ```

   For a read-only topology plan, run
   `./target/release/tp-launch net-plan "${SPARKS[@]}"` on the launching rank.
   Inspect it; do not apply its system-configuration commands as part of
   engine setup.

5. **Start the group** from the launching rank, with a fresh secret. Wait for
   `TP service ready` on rank 0.

   ```sh
   export SEQPAR_CONTROL_SECRET="$(openssl rand -hex 32)"
   ./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
     tp-serve target/tp.ptx "$REMOTE_MODEL" 8192 "$BIND" 0
   ```

   The arguments after `tp-serve` are: kernel file, model, context tokens,
   HTTP socket and request count. A count of 0 serves until stopped.

   The secret authenticates control messages between ranks; do not print it
   or put it in command-line arguments.

6. **Send one request** from a client that can reach rank 0. Use the actual
   model ID from `/v1/models`. A remote client can use an
   [SSH tunnel](server.md#option-a-an-ssh-tunnel-recommended).

   ```sh
   curl -sS "$URL/v1/chat/completions" \
     -H 'Content-Type: application/json' -d @- <<'EOF'
   {"model":"qwen3-8b","messages":[{"role":"user","content":"Explain why ice floats."}],
    "max_tokens":128,"temperature":0,"enable_thinking":false}
   EOF
   ```

Keep the launcher running while serving. Ctrl-C stops every rank. If a rank
fails, the launcher stops the group; it does not retry or recover a
conversation.

### Ask your coding agent

```text
Serve the installed dense Qwen3 model across the configured Sparks in cable
order using docs/multi-spark.md. Use the existing SSH aliases, the same Vade
revision and official model files everywhere. Stop only the current task's
Vade servers. Check passwordless SSH and link discovery; stop if either is
missing. Never use sudo or change network, OS or driver settings. Build the
TP kernels, create a fresh control secret without printing it, and launch
rank 0 on loopback. Wait for "TP service ready", send one text chat and show
the answer. Do not commit, push or publish.
```

<details>
<summary><b>Settings</b>, and the limits of the multi-Spark server</summary>

<dl>
<dt><code>SEQPAR_CONTROL_SECRET</code></dt>
<dd>Shared fresh control secret of at least 16 bytes, passed through the launch
environment.</dd>
<dt><code>TP_RDMA=0</code></dt>
<dd>Use TCP instead of RDMA. <code>SEQPAR_RDMA</code> can override the
prompt-splitting transport.</dd>
<dt><code>TP_SERVE_HYBRID=0</code></dt>
<dd>Force shard-only mode.</dd>
<dt><code>TP_SERVE_HYBRID=1</code></dt>
<dd>Force hybrid mode, if the full model fits.</dd>
<dt><code>TP_SERVE_SEQUENCE_PREFILL=1</code></dt>
<dd>Split long prompts across ranks in hybrid mode. Its FP32 KV can change
answers, so it is off by default. Without it, the default reads long prompts
with TP alone: exact, but slower to the first token (8B, 28,672 tokens, two
Sparks: 33.9 s vs 5.8 s with main's old FP32 split; n=1 each). Per-rank device
peaks were 12,182,142,668 B and 33,964,394,352 B, respectively. The opt-in
restores the old speed, but that split gave wrong answers on SIXTEEN and
FOLLOWUP.</dd>
</dl>

The full list of settings is in [the TP crate](../crates/tp/README.md#settings).

Limits of the multi-Spark server:

- `tp-serve` is separate from `spark-serve`. Its supported API and
  concurrency limits are in
  [the TP crate](../crates/tp/README.md#how-tp-serve-differs-from-spark-serve).
- It uses the same request scheduler as `spark-serve`, with a prompt cache
  that keeps one prompt. It does not use the single-Spark draft model.
- A failed rank ends the group. There is no distributed failover or load
  balancer.

</details>

## Measure decode speed

**Use the existing `tp-generate` program; no new benchmark harness is needed.** It is a direct generation measurement, not the HTTP rival scoreboard on [Benchmarks](benchmarks.md).

### Do it yourself

1. Build and check the group as in steps 1 to 4 above, and set the same shell
   variables.

2. Make a prompt file of token IDs, one per line, in rank 0's checkout. Rank 0
   is the first alias in `SPARKS`. Only rank 0 reads the file; a relative path
   is read from its checkout. For a test file:

   ```sh
   seq 1 2048 > prompt.ids
   PROMPT=prompt.ids
   ```

3. With a fresh control secret exported, run the command below. `32 1` asks
   for 32 generated tokens, once. `--seqpar` reads the prompt in hybrid mode
   and needs at least 128 prompt tokens per rank.

   ```sh
   ./target/release/tp-launch --dir "$REMOTE_DIRECTORY" run "${SPARKS[@]}" -- \
     tp-generate target/tp.ptx "$REMOTE_MODEL" 32 1 \
     --seqpar --ignore-eos --tokens-file "$PROMPT"
   ```

4. Report the actual prompt and output lengths, tokens per second, per-rank
   memory bytes and transport.

### Ask your coding agent

```text
Measure tensor-parallel decode speed for the installed dense Qwen3 model
across the configured Sparks, following the "Measure decode speed" section
of docs/multi-spark.md. Use the same Vade build and model on every rank.
Check SSH and link discovery first; stop if either fails. Never use sudo or
change network, OS or driver settings. Export a fresh SEQPAR_CONTROL_SECRET
without printing it. Report the prompt and output lengths, tokens per
second, per-rank memory bytes and the transport. Label the result as a
direct generation measurement of this commit. Do not commit, push or publish.
```

## How it works

**`tp-serve` runs in one of two modes: shard-only keeps just each rank's share; hybrid keeps a full copy and, only with `TP_SERVE_SEQUENCE_PREFILL=1`, splits prompt reading across ranks.**

<details>
<summary><b>Modes</b>, and how <code>tp-serve</code> chooses</summary>

- **Shard-only TP.** Each Spark holds its share of the weights, and every
  rank computes its share for every token. Conversation memory: BF16, for
  its own attention heads.
- **Hybrid.** Each Spark holds one full copy, with views of each share.
  Prompts and answers run on the shares with BF16 conversation memory, as in
  shard-only mode. With `TP_SERVE_SEQUENCE_PREFILL=1` the prompt is split in
  slices across ranks, then TP decode; conversation memory is then FP32, so
  answers can differ from one Spark.
- **Independent servers.** One full copy per server, separate requests,
  normal single-Spark state.

How `tp-serve` chooses:

- **Hybrid** only with `TP_SERVE_SEQUENCE_PREFILL=1`, when the context is at
  least 6,144 tokens and the full model fits on every rank. It uses the rank
  with the least free memory for this test.
- **Shard-only** otherwise.
- An explicit `TP_SERVE_HYBRID` overrides the choice.

A full copy with share views is still one resident copy. It does cost more
memory per rank than shard-only mode.

</details>

<details>
<summary><b>How splitting works</b>, and why it keeps one Spark's arithmetic</summary>

- **What is split.** Ranks split whole groups of query and KV heads, and the
  output rows of each matrix.
- **Keeping one Spark's arithmetic.** Before the output and down projections,
  ranks gather the input slices they need. Each result then uses the same
  full-length summing order as single-Spark decode. Adding up separate column
  shares would change that order, so Vade does not do it.
- **Prompt splitting is different.** In the long-prompt helper, every rank
  holds the full model, reads one slice and exchanges an FP32 KV cache.

The helper transport and the TP transport are separate protocols.
[Sequence parallelism](../crates/seqpar/README.md) and
[the design note](../crates/tp/MULTI-SPARK.md) explain that boundary.

</details>

<details>
<summary><b>Transport on GB10</b>: staged through CPU memory, and not encrypted</summary>

- The GB10's network card cannot read GPU memory directly, so RDMA data is
  staged through pinned CPU-side buffers.
- Both cable rails can be used. TCP is the explicit fallback.
- NVIDIA's OS, driver and link configuration stay unchanged.
- Traffic after the secret check is not encrypted. Use the cables only on a
  private, trusted network.

</details>

## Not done yet

**Four pieces of multi-Spark work remain.**

- Physical qualification of larger groups on current main.
- A BF16 hand-off for prompt splitting (hybrid's opt-in), so it can be on by
  default.
- Long-prompt helpers for the normal BF16 `spark-serve` path.
- Distributed conversation recovery and request load balancing.

**Where next:** [Architecture](architecture.md#several-sparks-on-one-answer) ·
[spark-tp crate](../crates/tp/README.md) · [Credits](../../docs/CREDITS.md)
