# Plane-adjacent MoE prefill rasterization

The host oracle compares the pinned baseline and rasterization trees named in
its source. It checks the grid mapping independently of kernel arithmetic.
The M32/N128/K64 CTA, register accumulators, ascending K16 chain and epilogue
are unchanged. The four M32 planes move from grid Z into adjacent X blocks:
`mb = first + (bx % 4) * 32`, `nb = (bx / 4) * 128`.
Gate launches `(20, tasks, 1)` and down `(80, tasks, 1)`. Decode/masked launches
retain their original geometry. This follows the supplied MoE review's L2 reuse
proposal; all implementation and tools are first-party, with no dependencies.
This historical source-comparison oracle requires both pinned Git objects
listed in its source. A squash-only checkout does not contain those objects.
Use a history-bearing checkout for that comparison; it is not part of the
standard host test suite. From that checkout's root:

```sh
rustc --edition 2021 -O --test engine/tools/oracles/moe_raster.rs -o /tmp/moe-raster
/tmp/moe-raster
clippy-driver --edition 2021 --test engine/tools/oracles/moe_raster.rs -D warnings
```

It checks source equality of the CTA body between two pinned trees, base a116627 and result 500c408, after exactly
two coordinate substitutions. It then compares every output byte owner for an
8K routed synthetic prompt at chunks 512/2048, full gate/down widths, skewed
expert counts, empty experts, all M32 tails and multiple M128 tasks. Unique
arbitrary output bits prove the permutation cannot change routed bytes because
the CTA body is identical and no two CTAs share an output. This is an ownership
proof, not a scalar or CUDA arithmetic emulator or an official-weight receipt.

`moe_raster_gpu.rs` is a standalone microbench/byte entry, following the existing
`flashnext_prefill` example and the historical fresh-session adapter. Compile
it against **each arm's own** release rlibs and production PTX. On Linux, after
`cargo kernels` and the workspace release build, run from `engine/`:

```sh
rustc --edition 2024 -O -C panic=abort tools/oracles/moe_raster_gpu.rs \
  -L dependency=target/release/deps \
  --extern spark_engine=target/release/libspark_engine.rlib \
  --extern spark_gpu_probe=target/release/libspark_gpu_probe.rlib -o moe-raster
```

It loads once per process and creates/drops new sessions per TSV cell. It writes
full vocabulary logits for the prompt's first prediction and all 63 subsequent
predictions, plus 64 greedy tokens per chat. Every run prints its explicit
sampling settings (temperature 0, top_p/top_k 1, repetition 1, presence/frequency
0, MTP 0). A TSV row contains tag, chunk, decode count, chat count, profiler flag
and comma-separated token IDs. The helper rejects invalid shapes and IDs.

The historical Python measurement controller is not distributed. Drive the
helper from a controller that records the complete invocation and output. The
route-count diagnostic readback was removed; timing needs no environment
settings. Qualification uses the production server, pinned answer bytes,
and alternating fresh server starts. Native and Nsight timings are attribution
only; the fold gate is the paired cold 8K client TTFT median.

A later build changed both files again (A staging in fp4_prefill_a.rs; one CTA per token
for flash_moe_route). Their own byte gates prove those changes, not this oracle.
