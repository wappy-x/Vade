# FP32 QSA indexer contract

`spark_gpu_probe::qsa` provides `IndexCache::{append,decode,prefill}`,
`prepare_query`, `select_blocks`, `sparse_attention` and `CacheBytes`.
Linux `Backend::{qsa_prepare,qsa_select,qsa_attention}` accepts the typed
`qsa::contract::{Prepare,Select,Attention}` descriptors. Decode uses rows=1,
past=context-1; prefill uses 1..=1024 consecutive rows. Context is at most 1M.
All addresses are device addresses on that backend; layouts are in the types.
Caller-owned buffers survive stream synchronization. Scratch must be disjoint
from inputs; immutable K and V may share storage. Inputs must be finite and
small enough that FP32 norms, dots and weighted sums do not overflow.

The architecture layer supplies the 640-wide index projection, checkpoint norm
weights, epsilon and FP32 partial-RoPE coefficients (32 cos then 32 sin).
Group RoPE uses the first absolute token of each newly completed group,
including a group started in the preceding chunk. Keep the four-slot raw ring
between calls. `qsa_prepare` reads it before updating it on the same stream.
The cache contains only completed groups. Selection masks future groups per row,
keeps 512 blocks, prefers lower block indices on ties, expands four tokens per
block, and appends every incomplete-tail token. Positions are emitted ascending
to match a mask on ordinary causal attention. Separate u32 counts exclude padding.
The main attention Q/K already have their own normalization and RoPE applied;
the caller applies the output sigmoid gate and projection after this operation.

Every FP32 sum follows ascending feature/head/token order, without FMA or atomic
reductions. Softmax uses a shared range-reduced degree-12 exponential polynomial,
tested against CPU exp over [-104,0] within 0.2 ppm plus two subnormal units.
This defines our CPU/CUDA parity contract; it does not reproduce another engine's
BF16/FP8 rounding or reduction tree. GPU byte parity requires the ignored test.
Text-only contiguous causal visibility is supported; padding, arbitrary masks,
paged allocations, cache rollback and MTP index reuse belong to the caller.

Logical payload: per layer, compressed FP32 index keys cost 128 bytes/token
(floor(context/4)*512 bytes), plus a fixed conservative 2312-byte tail/cursor.
Across 12 layers that is 1536 bytes/token + 27,744 fixed bytes. Main FP32 K/V
add 49,152 bytes/token (BF16 would add 24,576; these kernels consume FP32).
Thus FP32 KV+index payload is 50,688 bytes/token: 6.1875 GiB at 128K and
49.5 GiB at 1M, plus fixed bytes. Allocator capacity and weights are excluded.
Per-row scratch is 4*floor(context/4) + 4*2051 + 4 + 24*2051*4 bytes,
excluding projected/prepared queries, RoPE tables and attention output.
At an assumed 273 GB/s, reading one index cache plus the unique selected FP32
K/V gives bandwidth floors of about 92/522 us per layer at 128K/1M. An unmeasured
planning estimate for this scalar baseline is 1000/5000 us per layer, or
12,000/60,000 us for all twelve QSA layers. Radix passes, cache reuse, serial
dependencies and seven launches per layer make this uncertain. These estimates
exclude all other model work and are not measured end-to-end decode latency.

Fresh CUDA box, from the engine directory, with the pinned toolchains installed:
```sh
cargo kernels
cargo test --release --offline -p spark-gpu-probe --test qsa cpu_contract -- --exact
for n in 8192 131072 1048576; do
  QSA_CONTEXT=$n cargo test --release --offline -p spark-gpu-probe --test qsa gpu_exact -- --ignored --exact --nocapture || break
done
```
The GPU test checks chunked compression/query preparation, scores, stable ties,
padding/counts and two causal rows of sparse GQA against CPU bytes. It prints
attention-only wall time, which includes launch overhead and is not a decode benchmark.
For the isolated FP4 build use `cargo kernels --fp4` followed by the same tests
with `SPARK_PTX_VARIANT=fp4`. This implementation is a qualification baseline.

Study credits: transformers b0cdc61 Qwen4Exp indexer lines 664-770; vLLM
355553a indexer/cache and sparse GQA; b12x 2cc7f66 QSA reference, score, stable
selection and sparse GQA kernels. Implemented independently; no reference code
was executed or copied. The supplied FlashInfer 5a894af snapshot contains no
QSA attention files; its available files cover MoE/quantization instead.
