# Single-device kernel bench

This fixture screens kernels without loading or running the decoder. It is not
a model exactness or speed gate. The `kernel-bench` feature is off by default.
No product request path calls it. It uses the standard library and the existing
first-party crates. It adds no unsafe block or dependency.

## Build and prepare

1. Build the ordinary layer kernels and the runner with the same release setup
   used for `resident-decode-check`:

   ```sh
   cargo kernels
   cargo build --release -p spark-tp --features kernel-bench --bin kernel-bench
   ```

2. Stage the runner, images, command file, and this script on one Spark before
   claiming it. Use each lane's own directory. Build new images off the GPU.
   Candidate images need only the selected entry ABIs. Unrelated layer and
   matvec tables continue to reference the retained baseline module. A missing
   selected entry is an error, not a fallback to baseline arithmetic.
   An external baseline image must contain the full production module ABI.
   PTX and complete CUDA ELF64 cubins are accepted. A cubin avoids PTX JIT time.
   Image load failures do not fall back to the embedded baseline.

3. Inspect the header-only catalog if needed:

   ```sh
   ./kernel-bench catalog "$MODEL" --parts 2 --per-kind 3
   ```

   The default selects three distinct local matrix shapes of each
   storage family: NVFP4, MXFP8, block FP8, and plain weights computed in FP32.
   Shapes rank by matrix multiply-add count times their decoder-layer inventory
   count, so a frequent shared-expert shape precedes a rare large projection.
   Routed-expert inventory counts are not runtime expert-selection counts.
   Plain BF16 checkpoint bytes are widened exactly as in the resident decoder.
   Only decoder-layer matrices are selected. Embedding, head, stored draft
   layers, and mHC mixing matrices are excluded from projection cases.
   Each matrix is real checkpoint data. Only selected matrices and their scale
   planes are read, not the full checkpoint. Full source payloads are limited
   to 512 MiB by default. Set `--memory-mib` or `--per-kind` explicitly to change
   this bound. The actual selected names and dimensions are printed.

## Run one session

Prepare a command file. Paths and labels must have no whitespace:

```text
list
run same-image ./baseline.cubin all 5
run row-warps ./row-warps.cubin p0,p1 5 kernel=expert_group grid_scale=32 threads=128
sweep ./variants.tsv p0,p1 5
quit
```

Run only on an actually RELEASED Spark with no GPU process. Promotion gates
take priority. The wrapper holds the box's lock, appends its claim, starts the
runner immediately, and releases only after the process exits and the GPU is
empty. Copy its claim and release receipt into the day Log.

```sh
bash kernel-bench-one.sh ./kernel-bench "$MODEL" ./evidence commands.txt \
  --baseline ./baseline.cubin --parts 2 --seconds 55
```

`--baseline` is optional. Without it the runner uses its exact embedded image.
Both baseline and candidate images load in one CUDA context. Weights, inputs,
and the decode lookup table remain resident across all submitted images.
There is no transport, pair, tokenizer, prompt prefill, or model load.
The runner ends at EOF, `quit`, or its session budget. The budget includes
header admission and selected-weight loading. It checks the budget before each
new sample. An already submitted CUDA call cannot be canceled safely, so the
budget is not a hard driver-hang timeout. The limit is at most 59 seconds.
An idle process does not keep a box indefinitely.

Optional `kernel=ENTRY grid_scale=N threads=N` words apply only to the candidate
arm and that entry. The baseline keeps its original launch geometry.
`grid_scale` multiplies the original x-grid; y/z grids and shared memory stay
unchanged. It must be 1..1024. `threads` must be complete warps through 1024;
zero keeps the original block size. The candidate must implement the existing
entry ABI and bounds for that geometry. Omitting these words keeps all launch
dimensions unchanged. Invalid words and overflowing grids are rejected.

The case IDs are `p0`, `p0-scalar`, and so on, plus `mhc`:

- `pN` uses the resident grouped projection entry. Packed formats use
  `expert_group`; MXFP8 and plain use `flash_resident_projection_batch`.
- `pN-scalar` uses the ordinary packed, MXFP8 M=1, or plain primitive.
- `mhc` uses one real checkpoint boundary's parameters for three deferred
  post/pre sites. It checks all five planes per site, not only the final output.

`--packed-tokens N` selects 1..1024 token rows for each NVFP4 and block-FP8
case. The default is1. Each case still uses one real matrix, not N distinct
matrices. Inputs are token-major. Output is weight-row-major, with N token
values per weight row. All output bytes are checked, including partial tiles.
MAC counts are rows times columns times N. The case log and result name both
token rows and matrix count. Scalar packed cases are omitted when N exceeds1;
their ABI processes only one row. MXFP8, plain and mHC cases remain single-row.
The per-case FP32 input/output staging budget is128 MiB. Invalid counts or
larger buffers fail before a device context is opened.

For example, add `--packed-tokens 128` to `serve`, then submit only the
desired packed `pN` cases with `run`. The same loaded matrix and activation
rows are retained across all submitted images. Activation preparation is
outside timing. This measures token-row reuse, not the referee's counts of
distinct matrices. Neither result substitutes for a model gate.

Use the existing entry argument layouts in `gpu/src/expert_ops.rs` and
`gpu/src/next_params.rs` when writing a replacement image. The caller and
bounds checks are in `gpu/src/backend/expert_ops.rs` and
`gpu/src/backend/mxfp8_projection.rs`. Paths are relative to `engine/crates`.
The baseline must contain the full production module. A candidate needs all
entries used by its selected cases; missing entries fail instead of falling
back. A CUDA image is trusted executable code, not sandboxed input. The image
validator checks its container, not the safety of its device memory accesses.

Activations are deterministic synthetic finite rows, including negative zero.
They are prepared once. NVFP4 and MXFP8 activation quantization is outside the
timed projection. Projection shape and local row count come from admitted
metadata and `--parts`; this remains a one-device run.

Each trial warms both arms, then alternates baseline/candidate order across
five repeats by default. Every output byte of every warmup and measured sample
must equal the pinned baseline reference. Baseline repeats are also checked
against that reference, so candidate damage or drift is not silently accepted.
Signed zero and NaN payloads are compared as bytes. A failure saves the actual
plane and prints the first differing byte, plane, case, repeat, and arm.

GPU events supply kernel-only nanoseconds. Wall time and launch count are
reported separately. Medians and multiply-adds per second are printed for both
arms. For mHC the operation count covers coefficient dots only, not Sinkhorn
or normalization. Sweeps rank only byte-equal variants by the geometric mean
of selected-case speed ratios. This is not a weighted model-speed estimate.
Repeated operands can be cache-resident. These timings are not a cold-memory
bandwidth measurement or a substitute for the full-model trace and gate.
For an additional cache-pressure screen, use `--cold-mib 256`. It allocates
256 MiB of source and 256 MiB of destination scratch once, then reads and
writes them before every sample. Pressure runs on the baseline arm and stays
outside reported sample timing. The default is zero; the maximum is 1024 MiB
per plane. This is not proof that every timed load reaches device memory.
`COMPLETE` means the session ended cleanly, not that every variant passed.
Inspect `*.REJECT`, per-variant reports, and `ranking.txt`.

A new fixed summation order will correctly fail against today's baseline.
It needs its separately approved host-reference and topology proof. Do not
interpret that failure as a permitted model change, or relabel an old image.

## Build a parameter sweep

Make a specification. The builder defines the parameter names:

```text
warp8 rows_per_warp=8 strip=8 unroll=1 table=768
warp4 rows_per_warp=4 strip=16 unroll=2 table=768
warp1 rows_per_warp=1 strip=32 unroll=4 table=768 kernel=expert_group grid_scale=32 threads=128
```

The explicitly supplied builder is an executable. It receives the new image
path as argument 1, followed by the parameter words. It must produce a complete
PTX or cubin image there and exit zero. Partial candidate images are allowed
when they contain all entries selected by the case list. It can use the lane's existing compiler
setup or an isolated template; the bench does not edit production source.
Run long builds in the normal build worker, with no GPU claim:

```sh
./kernel-bench build-sweep specs.txt ./build-variant ./new-sweep cubin
```

This builds all N variants, keeps each build log, hashes and validates each
image, and writes `new-sweep/variants.tsv`. Submit that manifest with `sweep`.
The builder receives geometry words too. They are retained in the manifest;
all parameter words are saved in the variant's `.params` receipt.
Generated image paths are relative to the manifest's folder. Directory names
can contain spaces. Copy the complete built folder to a Spark without changing
its manifest. The builder still receives an absolute image output path.
The runner hot-swaps each image without reloading weights. Large sweeps should
be split across sub-minute sessions or independent Sparks, never hold a pair.
