# spark-gpu-probe

**Vade's GPU layer for the GB10. Rust talks straight to NVIDIA's CUDA Driver
API, with no CUDA toolkit and no outside Rust crates.** It owns the GPU
memory, the weight storage and the compiled kernels.

This page is for people reading or changing the GPU layer. To just run Vade,
start with [Install](../../docs/install.md).

<details>
<summary><b>Terms on this page</b>: PTX, kernel, stream, pinned memory, CUDA graph, KV cache, tensor-core multiply, BF16, FP8, FP32</summary>

<dl>
<dt>PTX</dt>
<dd>NVIDIA's portable GPU instruction format, written as text. The driver
compiles it for the GB10 when a program starts.</dd>
<dt>Kernel</dt>
<dd>A function that runs on the GPU.</dd>
<dt>Stream</dt>
<dd>A queue of GPU work that runs in order.</dd>
<dt>Pinned memory</dt>
<dd>Host memory the GPU can copy from directly.</dd>
<dt>CUDA graph</dt>
<dd>A recorded sequence of kernel launches that can be replayed with one
call.</dd>
<dt>KV cache</dt>
<dd>The model's memory of the conversation so far.</dd>
<dt>Tensor-core multiply</dt>
<dd>The GPU's matrix-multiply instruction, here on BF16 inputs with FP32
sums.</dd>
<dt>BF16, FP8, FP32</dt>
<dd>16-bit, 8-bit and 32-bit number formats.</dd>
</dl>

</details>

## Where it fits

**Every crate that touches the GPU goes through this one.**

<dl>
<dt><a href="../engine/README.md">spark-engine</a></dt>
<dd>Builds models and sessions on top of this crate.</dd>
<dt><a href="../serve/README.md">spark-serve</a></dt>
<dd>Uses it through the engine, and directly for buffers and transfers.</dd>
<dt><a href="../tp/README.md">spark-tp</a>,
<a href="../seqpar/README.md">spark-seqpar</a></dt>
<dd>The same: through the engine, and directly for buffers and transfers.</dd>
<dt><code>kernels/</code> (<code>spark-rust-kernels</code>)</dt>
<dd>The GPU kernels, also written in Rust and compiled to PTX, built separately
for the GPU target.</dd>
<dt><code>build-kernels/</code></dt>
<dd><code>spark-build-kernels</code>, the kernel build tool behind
<code>cargo kernels</code>.</dd>
</dl>

## Build

**Build the kernels first. The host build embeds them and stops if they are
missing.**

### Do it yourself

1. Build the kernels, from the checkout root:

   ```sh
   cargo kernels
   ```

2. Build this crate and its loader check:

   ```sh
   cargo +1.99.0 build --release -p spark-gpu-probe
   ```

### Ask your coding agent

```text
In the Vade checkout on this DGX Spark, run `cargo kernels`, then
`cargo +1.99.0 build --release -p spark-gpu-probe`. If the kernel build
says a toolchain is missing, follow docs/install.md step 1 as the current
user, without sudo. Report the last lines of both builds. Do not commit.
```

<details>
<summary><b>What <code>cargo kernels</code> does</b></summary>

`cargo kernels` is a Cargo alias, defined in `.cargo/config.toml`. It runs
`build-kernels/`, a small Rust program on the pinned stable toolchain.

1. It checks that the pinned nightly Rust and its NVPTX target are
   installed. If not, it prints the `rustup` command to run.
2. It builds the kernels for the GB10 (`sm_121`) with tensor cores on.
3. It writes `spark_rust_kernels.ptx` and `spark_layer_kernels.ptx` to
   `target/nvptx64-nvidia-cuda/release/`.
4. It stamps both files against the kernel sources, shared launch types in
   `kernel_params.rs`, features, target CPU, PTX version and Rust flags.

`cargo kernels --fp4` builds `sm_121a` separately under `target/fp4/`, with
`spark_rust_kernels_fp4.ptx` and `spark_layer_kernels_fp4.ptx` names. Host
builds keep selecting production PTX unless `SPARK_PTX_VARIANT=fp4` is set.
See [the FP4 kernel contract](../../docs/fp4-kernels.md).

[Install](../../docs/install.md) sets up both Rust toolchains.

Host builds reject a missing or stale stamp when a PTX file is present.
Linux builds also require the PTX files. So rebuild the kernels after any
change to their source, comments included, or to the launch types.

<dl>
<dt><code>KERNEL_BIN</code></dt>
<dd>Use the <code>cargo</code> in this folder instead of the pinned
nightly.</dd>
<dt><code>SPARK_LAYER_PTX</code></dt>
<dd>Embed a different layer kernel file at build time. Its stamp and header
must match the selected variant.</dd>
<dt><code>SPARK_PTX_VARIANT</code></dt>
<dd>Select <code>fp4</code> explicitly to embed FP4 PTX in both host crates.
Unset, or <code>production</code>, selects the ordinary <code>sm_121</code> PTX.</dd>
</dl>

</details>

## Checks

`load-range-check` checks row selection, and clean-up after errors, in the
weight loader. It writes one test file to the system's temporary folder
and deletes it. Pass a folder as a second argument to choose where.

```sh
PTX=target/nvptx64-nvidia-cuda/release/spark_rust_kernels.ptx
./target/release/load-range-check "$PTX"
```

Model-level correctness uses the [engine's checks](../engine/README.md#build-and-run-the-checks)
and the [Hugging Face reference fixtures](../../fixtures/qwen3/README.md).

## How it works

**One stream, owned buffers, pinned staging, growable memory and recorded
launches.**

- **One stream.** All GPU work goes into one queue, in order. Reading a
  result back to the host waits for the work before it.
- **Owned buffers.** Each GPU buffer belongs to one CUDA context and cannot
  move between threads. Freeing a buffer first waits for any work that still
  uses it.
- **Pinned staging.** Uploads go through pinned host memory. A staging buffer
  is reused only after its copy has finished.
- **Growing memory.** `segmented_vector` and `grow_vector` add 2 MiB pages
  behind an existing allocation at a fixed address. The KV cache uses this
  to grow without copying.
- **Fewer launches.** Eligible dense decode steps replay a recorded CUDA
  graph. Other shapes launch kernels one by one.

<details>
<summary><b>Weight loading</b></summary>

`Backend::load_bf16_shards` takes open weight files and a list of row ranges.
It loads only those rows, straight into GPU memory, and keeps no host copy.

- Four reader threads share 32 MiB of pinned staging.
- They use direct file reads, and fall back to ordinary reads where the file
  system needs it.
- Packed rows come in groups of 8 rows and 16 columns.
- If a load fails, the readers stop and partial memory is freed.

</details>

## Precision

**Weights retain checkpoint precision; tensor-core products accumulate in FP32.**

| Checkpoint | Weights on the GPU | Multiply and add |
| --- | --- | --- |
| BF16 | BF16 | BF16 in, FP32 sums |
| Qwen3.8-27B official FP8 | FP8, with scales | BF16 in, FP32 sums |
| Flash-Next routed experts | FP8, with scales | W8A8, FP32 sums |
| Flash-Next covered BF16 projections | BF16 | BF16 high + residual parts of FP32 inputs (see below), FP32 sums |

- **BF16 checkpoints:** Qwen3, Qwen3.8-27B and Gemma 4 31B.
- **Official FP8 checkpoints:** Qwen3.8-27B-FP8 converts weights to BF16 when
  used, with the checkpoint's scales. In Flash-Next (Qwen3.8-Flash-Next),
  the routed experts use W8A8 (8-bit weights and 8-bit activations).
  Both retain their published FP8 weights with no second BF16 weight copy.
- **Flash-Next BF16 projections:** each covered projection splits its FP32
  activations into BF16 high and residual parts, multiplies both on the tensor
  cores and accumulates in FP32 in one fixed order. Other models keep their
  native BF16 paths. With the same machine, build and setup, this one
  Flash-Next mode must give the same answer whether a chat runs alone or
  batched, cached or chunked, or with greedy MTP drafting (multi-token
  prediction). Answers may differ from older builds. The scalar-FP32
  alternative and the `FLASH_BF16_MMA` switch have been removed.
- **FP4-Local checkpoint:** `nvidia/Qwen3.8-Flash-Next-FP4-Local` runs on one
  Spark. Its quantization map stores each module as NVFP4, MXFP8 or plain
  tensors; the loader reads that map and checks the tensor list. The NVFP4
  kernels decode the checkpoint's encoded weight scales themselves, without
  loss. See the
  [mixed-storage loader](../model/src/flash/mixed.rs) and the
  [FP4 kernel contract](../../docs/fp4-kernels.md).
- Anything below the checkpoint's precision has to be turned on explicitly.
  `SPARK_ADAPTIVE_PREFILL=1`, a faster, approximate grouped attention for long
  prompts, can change output and is off by default.

## Decode attention order

**Every BF16 decode path shares one attention order, so batched equals alone
and cached equals fresh, exactly.**

While writing an answer, each new token's attention reads the whole KV cache.
Every BF16 decode path runs that work through one function, `split_mma` in
`kernels/src/decode_attention.rs`: one chat alone, many chats batched, guessed
tokens being checked, and chats sharing a cached prompt beginning.

That is not the same as full agreement with the official model;
[Validation](../../docs/validation.md) records the remaining Qwen3
disagreements.

<details>
<summary><b>The order, step by step</b></summary>

For head size 128 and a BF16 KV cache:

1. **Splits.** The cached keys are cut into splits of 64 tokens up to 4,096
   positions, 256 up to 16,384, and 1,024 beyond.
2. **Tiles.** Inside a split, keys go in 64-key tiles from the split start.
   The query is rounded to BF16 once. Each pair of chat row and query head is
   one row of the tensor-core multiply, so its result never depends on the
   other rows or on padding.
3. **Scores.** For each block of 8 keys: 8 chained multiplies over head
   dimensions 0-15, 16-31, and so on to 112-127, summed in FP32, then divided
   by the square root of 128. Keys past the split end score minus infinity.
4. **Softmax.** Per tile, in FP32, with a running maximum and a running
   total. Each GPU lane adds its 16 values in key-block order, then lanes
   combine in a fixed pairing.
5. **Values.** The running output is rescaled, then 4 chained multiplies
   cover keys 0-15, 16-31, 32-47 and 48-63 of the tile, with the softmax
   weights rounded to BF16.
6. **Merge.** Each split leaves a summary: its maximum, its total and 128
   outputs. Summaries merge in a fixed order. A single split's direct result
   exactly equals a one-summary merge.

FP32 and scaled-integer KV caches, and the Qwen3-0.6B draft (two query heads
per KV head), keep the older scalar order. Prompt reading (prefill) has its
own fixed order. In the [Qwen3 session contract](../engine/README.md#qwen3-phase-order),
only `verify_batch` promises the exact same bits as sequential decode. The
[head-skipping check](../engine/README.md#build-and-run-the-checks) compares
asking for logits on all prompt rows, only the last, or none (All/Last/None),
bit for bit, within a phase; it does not equate prompt logits with sequential
decode logits.

</details>

## Precompile kernels for a release

This section is only for people building release binaries; most readers can
skip it.

If you build a release, you can spare owners the first PTX compile by
embedding a trusted cubin (a compiled GPU image) in the engine. This is
optional. It needs a CUDA toolkit on the machine that builds the release;
owners still need only the CUDA driver.

Compile the exact PTX produced by `cargo kernels`, without arithmetic overrides:

```sh
ptxas --gpu-name=sm_121 spark_rust_kernels.ptx --output-file=flash.cubin
```

Next to the image, create `flash.cubin.conf`. It records the SHA256 digests of
both files and the lowest driver API version the build toolkit needs (CUDA 13.0
uses `13000`):

```ini
format=vade-cubin-v1
target=sm_121
ptx_sha256=<SHA256 of spark_rust_kernels.ptx>
cubin_sha256=<SHA256 of flash.cubin>
min_driver_api=13000
```

Set `SPARK_CUBIN=/absolute/path/flash.cubin` only for the engine build,
including a Mac cross-build. The build refuses a changed PTX, image digest or
CUDA ELF target. If validation fails on the build machine, it prints one plain
error and exits with a failure.

Accepted images use CUDA ELF64 OSABI `0x41`, ABI version `8`, executable type
`2`, ELF version `1`, and exact target flags `0x06007902` for `sm_121`. The
`sm_121a` encoding (`0x0a007902`) and other ABI/feature encodings are refused,
even when their numeric compute capability is 12.1 and their digests match.
[Compiler-produced header fixtures](fixtures/README.md) record those rules; a
new compiler encoding needs review.

The engine binary then contains the image, so its existing release digest
covers it. Keep the compiler version, command and hashes with the release's
build records (receipts), separate from measurements of owners' installs.

At run time, the engine uses the compiled image when the PTX is
byte-identical, the GPU has compute capability 12.1 and the driver is new
enough. An incompatible image falls back to PTX. The PTX path is also used
for any other PTX or device, when `CUDA_FORCE_PTX_JIT=1` is set, and by the
probes that record just-in-time (JIT) compile receipts. Owners have no setting
for this. Without `SPARK_CUBIN`, builds load PTX as usual.

We test the shared loader's control flow on the host with a deterministic
stand-in driver: image selection, forced PTX, receipt loads, every allowed
fallback code, fatal errors, cleanup failures, and the exact original PTX passed
after fallback. These tests do not prove the GPU runs the code correctly. The
release gate (the pass-or-fail check before a release) must run the actual
embedded payload, and an explicit PTX fallback, on the GPU.

**Where next:** [spark-engine](../engine/README.md) ·
[Credits](../../docs/CREDITS.md) · [Benchmarks](../../docs/benchmarks.md)
