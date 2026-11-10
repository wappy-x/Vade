# spark-seqpar

**This crate splits one long prompt across several processes, on one Spark or
on several cabled Sparks. Each process, called a rank, reads one slice of the
prompt.** Layer by layer, ranks pass the model's memory (the KV cache)
forward, because a token looks back, never ahead. At the end, the last rank
holds the memory of the whole prompt and writes the answer on its own.

This is called sequence-parallel prefill; prefill means reading the prompt.
It helps only when the compute it saves outweighs the cost of moving data
between Sparks, and that must be measured on the actual group.

This page is for developers. To serve one model across Sparks, use
`tp-serve` from [Multiple Sparks](../../docs/multi-spark.md).

<details>
<summary><b>Terms on this page</b>: prefill, decode, KV cache, rank, ring, path, QSFP, RDMA, TCP</summary>

<dl>
<dt>Prefill</dt>
<dd>Reading the prompt, before the first answer token.</dd>
<dt>Decode</dt>
<dd>Producing the answer, one token per step.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the tokens it has read.</dd>
<dt>Rank</dt>
<dd>One process in the group. Ranks are numbered from 0.</dd>
<dt>Ring, path</dt>
<dd>How the ranks are cabled. A ring closes back on itself; a path, called a
line in the user guide, does not.</dd>
<dt>QSFP</dt>
<dd>The Spark's high-speed network ports.</dd>
<dt>RDMA</dt>
<dd>Copying memory straight between machines over the cable, without the usual
network stack.</dd>
<dt>TCP</dt>
<dd>The ordinary network transport, used as the fallback.</dd>
</dl>

</details>

## Status at a glance

**Three ways to run it work today. The chat server's helper flags are not used.**

| Program or mode | Status |
| --- | --- |
| `seqpar-launch`, `seqpar-generate` | Works |
| `tp-serve` hybrid mode | Works |
| `seqpar-worker`, coordinator API | Works (FP32 KV) |
| `spark-serve --seqpar-peer` | Not used today |

- **`seqpar-launch` with `seqpar-generate`** reads a file of token IDs,
  splits the prompt and prints the generated IDs. Decode (writing the
  answer) runs on the final rank only.
- **`tp-generate --seqpar`, and `tp-serve` hybrid mode with
  `TP_SERVE_SEQUENCE_PREFILL=1`,** run sequence-parallel prefill, then
  tensor-parallel decode. See
  [spark-tp](../tp/README.md).
- **`seqpar-worker` with the coordinator API** works for programs that keep
  an FP32 KV cache.
- **`spark-serve --seqpar-peer`:** the flags exist, but the chat server keeps
  a BF16 KV cache and the helpers need FP32, so helpers never take part.

## Where it fits

**It drives the engine's sequence-prefill hooks, and spark-tp uses it for
hybrid prefill.**

<dl>
<dt><a href="../engine/README.md">spark-engine</a></dt>
<dd>Provides the sequence-prefill hooks this crate drives.</dd>
<dt><a href="../gpu/README.md">spark-gpu-probe</a></dt>
<dd>GPU buffers and host transfers.</dd>
<dt><a href="../tp/README.md">spark-tp</a></dt>
<dd>Uses this crate for hybrid prefill.</dd>
<dt><a href="../serve/README.md">spark-serve</a></dt>
<dd>Has helper flags, unused today (see the status above).</dd>
</dl>

This crate owns prompt splitting, peer admission and the ring transport.
[engine-fabric](../fabric/README.md) owns shared peer discovery and RDMA verbs;
this crate re-exports `discovery` and, on Linux, `verbs`, for existing callers.

<details>
<summary><b>How it works, step by step, and how slices are sized</b></summary>

1. Every rank loads the full model.
2. The prompt is cut into slices, one per rank, in order. Rank 0 gets the
   start of the prompt and the last rank gets the end.
3. For each layer, every rank computes the KV cache for its own slice.
4. Ranks send their KV cache forward. Each rank receives only the earlier
   tokens it needs.
5. Each rank finishes the layer for its slice, then moves to the next layer.
6. After the last layer, the final rank holds the KV cache for the whole
   prompt. It writes the answer alone. The other ranks free their memory.

Slices are cut on 128-token boundaries. Later slices get fewer tokens,
because tokens near the end of a prompt look back over more text.

</details>

## Try it on one Spark

**Two or three ranks on one GPU, over local TCP (the ordinary network).** It
checks that everything works, not speed: the ranks share one GPU.

### Do it yourself

1. Build, from the checkout root, after [Install](../../docs/install.md).
   This builds `seqpar-launch`, `seqpar-generate` and `seqpar-worker`.

   ```sh
   cargo kernels
   cargo +1.99.0 build --release -p spark-seqpar
   ```

2. Make a test prompt of 8,192 token IDs, then run it with two ranks and 8
   answer tokens.

   ```sh
   seq 1 8192 > prompt.ids
   SEQPAR_MODEL_DIR=models/Qwen/Qwen3-8B \
     ./target/release/seqpar-launch --local 2 -- prompt.ids 8
   ```

3. Optional: try three ranks by changing `2` to `3`.

The final rank prints one token ID per line. Each rank loads its own copy of
the model, so two ranks need about twice the memory. A prompt needs at least
128 tokens per rank.

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, run `cargo kernels`, then
`cargo +1.99.0 build --release -p spark-seqpar`. Write the IDs 1 to 8192
with `seq 1 8192 > prompt.ids`. Then run this, as shown in
crates/seqpar/README.md:
SEQPAR_MODEL_DIR=models/Qwen/Qwen3-8B \
  ./target/release/seqpar-launch --local 2 -- prompt.ids 8
Report the printed IDs and any error.
Do not change code, network settings, commit or push.
```

## Run it across Sparks

**Two or more cabled Sparks, an SSH alias for each, and the same build and
model on all of them.** [Multiple Sparks](../../docs/multi-spark.md) covers
cabling and SSH. Ranks check a shared secret, but **traffic after that check
is not encrypted**, so use only a private, trusted network.

### Do it yourself

1. Check the links on each Spark. Each line shows one of the Spark's
   high-speed QSFP network ports, its address and its speed.

   ```sh
   ./target/release/seqpar-launch discover
   ```

2. On the Spark you launch from, create a shared secret.

   ```sh
   export SEQPAR_CONTROL_SECRET="$(openssl rand -hex 32)"
   ```

3. Copy the token file to the same path on every Spark.

4. Set these values for your group. Do not commit them.

   - `SPARKS`: a shell array of existing SSH aliases, in cable order. The
     last one writes the answer.
   - `REMOTE_DIRECTORY`: the checkout folder on each Spark.
   - `REMOTE_MODEL`: the model folder on each Spark. Relative paths start in
     that checkout.
   - `PROMPT`: the token file path on each Spark. Relative paths start in
     that checkout.

5. Launch.

   ```sh
   SEQPAR_REMOTE_DIR="$REMOTE_DIRECTORY" SEQPAR_MODEL_DIR="$REMOTE_MODEL" \
     ./target/release/seqpar-launch "${SPARKS[@]}" -- "$PROMPT" 8
   ```

### Ask your coding agent

```text
Run sequence-parallel prefill across the configured Sparks with
crates/seqpar/README.md, "Run it across Sparks". Use the existing SSH
aliases in cable order and the same Vade build and model on every Spark.
Run seqpar-launch discover on each Spark first; stop if a link between
neighbours is missing. Export a fresh SEQPAR_CONTROL_SECRET without printing
it. Report the generated IDs and each rank's timings. Never use sudo or
change network or system settings. Do not commit or push.
```

<details>
<summary><b>What the launcher does, and how the secret is checked</b></summary>

- It finds the direct cable between each pair of neighbours.
- It uses a ring when the last Spark is cabled back to the first, and a path
  (a line) otherwise.
- A missing link between two neighbours stops the launch.
- Each rank runs one job, a single prompt and its answer, and exits when it
  is done. The launcher returns once every rank has exited.
- If one rank fails, the launcher closes every rank's input, and a rank exits
  as soon as its input closes. Any rank still running two seconds later is
  killed; for a remote rank, that means its SSH session.

Security:

- Every rank must have the same `SEQPAR_CONTROL_SECRET`. Across machines it
  is required; a worker refuses to listen beyond loopback without it.
- The launcher passes the secret over SSH input, never on a command line.
- Ranks prove they know the secret with a challenge. The secret itself is
  never sent. A wrong or missing secret is rejected within 800 ms.
- **Traffic after that check is not encrypted.** Use these ports only on a
  private, trusted network.
- All ranks must run the same build. The handshake checks the protocol
  version, prompt and split, and rejects a mismatch before loading the model.

</details>

<details>
<summary><b>All settings, with defaults</b></summary>

<dl>
<dt><code>SEQPAR_MODEL_DIR</code></dt>
<dd>Model folder. Required.</dd>
<dt><code>SEQPAR_REMOTE_DIR</code></dt>
<dd>Checkout folder on each remote Spark. Required for remote runs.</dd>
<dt><code>SEQPAR_CONTROL_SECRET</code></dt>
<dd>Shared secret. Required across machines.</dd>
<dt><code>SEQPAR_PTX</code></dt>
<dd>Kernel file. Default: the main kernel file, named below.</dd>
<dt><code>SEQPAR_PORT</code></dt>
<dd>First rank's port. Each next rank adds one. Default: 29580.</dd>
<dt><code>SEQPAR_TOPOLOGY</code></dt>
<dd><code>auto</code>, <code>ring</code> or <code>path</code>. Default:
<code>auto</code>.</dd>
<dt><code>SEQPAR_RDMA</code></dt>
<dd><code>0</code> forces TCP, <code>1</code> requires RDMA. Default:
automatic.</dd>
<dt><code>SEQPAR_MEMORY_BYTES</code></dt>
<dd>Memory ceiling per process, in bytes. Default: 90% of Linux
<code>MemAvailable</code>.</dd>
<dt><code>SEQPAR_OVERLAP</code></dt>
<dd><code>0</code> turns off overlapping compute with transfers. Default:
on.</dd>
<dt><code>SEQPAR_ATTENTION_WEIGHT</code></dt>
<dd>Changes how slice sizes are balanced. Default: computed from prompt
length.</dd>
<dt><code>SEQPAR_SERIAL_COMPUTE</code></dt>
<dd><code>1</code> makes ranks take turns on the GPU, for profiling. Default:
off.</dd>
<dt><code>SPARK_PROFILE</code></dt>
<dd><code>1</code> prints the GPU operation profile. Default: off.</dd>
</dl>

- `SEQPAR_PTX` defaults to
  `target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx`.
- When several ranks share one Spark, set `SEQPAR_MEMORY_BYTES` for each so
  their total leaves room for the system.
- In automatic mode, RDMA is used only if every rank can open it. Otherwise
  all ranks use TCP.
- The GB10 has no GPUDirect RDMA, so data passes through host memory on each
  side.

</details>

<details>
<summary><b>Profiling on one Spark, and how much data moves</b></summary>

```sh
SEQPAR_MODEL_DIR=models/Qwen/Qwen3-8B SEQPAR_SERIAL_COMPUTE=1 SPARK_PROFILE=1 \
  ./target/release/seqpar-launch --local 2 -- prompt.ids 8
```

- With serial compute, each rank's GPU time is measured while the others
  wait. Use those per-rank times to estimate compute on separate Sparks.
- One-GPU wall time is not a multi-Spark speed-up. Real multi-Spark time also
  includes transfers, copies and uneven slices.
- Each run reports phase times, bytes sent and received, and memory for the
  model, KV cache, scratch and host buffers.
- Prefill time does not include loading the model.

**How much data moves**

For Qwen3-8B with an FP32 cache, one token's KV across all 36 layers is
294,912 bytes. Each rank receives only the earlier slices it needs.

Planned payload for a 28,672-token prompt:

| Ranks | Topology | Total bytes sent |
| ---: | --- | ---: |
| 2 | Ring | 5,058,330,624 |
| 3 | Path | 9,965,666,304 |
| 4 | Ring | 14,835,253,248 |

Slice boundaries, in tokens:

- 2 ranks: 0, 17152, 28672.
- 3 ranks: 0, 12416, 21376, 28672.
- 4 ranks: 0, 9856, 17152, 23296, 28672.

These are calculations, not measurements. For this prompt, each rank also
keeps a host buffer of 234,881,024 bytes, one layer's worth of KV cache.

</details>

## For developers: the coordinator API

**A server can act as the final rank, with warm helper processes reading the
earlier slices.** It drives the
[engine's hooks](../engine/README.md#hooks-for-sequence-parallel-prefill).

<details>
<summary><b>Helpers, the server-side API, its rules and limits</b></summary>

**Start a helper on each idle Spark**

1. Set the same `SEQPAR_CONTROL_SECRET` on every helper and on the server.

2. Set `CONTROL_LISTEN` to a socket on the existing restricted QSFP link.
   Traffic after the handshake is not encrypted; avoid a wildcard listener.

3. Start the helper.

   ```sh
   ./target/release/seqpar-worker "$PTX" "$MODEL" "$CONTROL_LISTEN" \
     "$MAX_TOKENS"
   ```

The helper loads the model once and keeps it loaded. After each job it frees
the request's memory. An existing process can call
`coordinator::serve_peer(&model, listener)` instead, to reuse a model it has
already loaded.

**The server side**

```rust,ignore
use spark_seqpar::coordinator::{Coordinator, Endpoint, Peer, Topology};

let coordinator = Coordinator::new(local_endpoint, Topology::Ring);
let mut session = model.session();
session.configure_kv_bf16(false)?;
let output = coordinator.prefill(&mut session, tokens, &peers)?;
let first = spark_engine::greedy_token(&output.logits)?;
let second = session.step_greedy(first)?;
```

<dl>
<dt><code>Endpoint { listen, next }</code></dt>
<dd>This rank's data address and the next rank's.</dd>
<dt><code>Peer { control, endpoint }</code></dt>
<dd>A helper's control address and data endpoint.</dd>
<dt><code>Topology</code></dt>
<dd><code>Ring</code>, or <code>Path</code> when the last link is missing.</dd>
</dl>

Rules:

- List peers in cable order. The server comes last.
- The session must be fresh and use an FP32 cache.
- With no peers, `prefill` falls back to ordinary local prefill.
- The coordinator reserves every peer before starting. A busy peer answers
  `peer busy` at once.
- After an error, throw the session away before retrying.

Time limits:

| Limit | Value |
| --- | ---: |
| Reserving peers | 1.5 s |
| Connection set-up | 10 s |
| A stuck transfer | 120 s |

Each result reports local timings, each peer's time and peak memory, and the
total time including reservation and release.

The lower-level loop is `spark_seqpar::prefill::prefill`.
`prefill_with_options` adds the overlap and serial-compute options.

Helpers do not keep their slice after prefill. Keeping it, to check proposed
tokens across ranks, is not implemented.

</details>

**Where next:** [spark-tp](../tp/README.md) · [Multiple Sparks](../../docs/multi-spark.md) ·
[Engine hooks](../engine/README.md#hooks-for-sequence-parallel-prefill)
