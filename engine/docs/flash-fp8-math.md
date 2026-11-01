# Flash Next arithmetic

Flash's covered BF16 projections have one arithmetic mode: BF16 high and residual
operands on tensor cores, with FP32 accumulation. This is always enabled; there
is no scalar-FP32 alternative or `FLASH_BF16_MMA` setting. On the same machine,
build and setup, the same request must give the same bytes alone, batched,
cached, chunked or with greedy MTP. Answers may differ from older builds.

The FP8 expert contract below describes official checkpoint revision `236dfdf`.
Its reference details come from local source inspection, not a CUDA parity or
performance claim. The checkpoint declares `quant_method=fp8`,
`activation_scheme=dynamic`, `act_per_tensor=false`, `weight_per_tensor=false`,
`weight_block_size=[128,128]`, and BF16 model compute dtype.

## Official execution

Transformers 5.18.0 `integrations/finegrained_fp8.py` was inspected from a local
source snapshot. Its SHA256 is
`e86cb993a148ba7e1baffb2f8c892fa5afb31504e81992a27da75a505c914624`. The inspected `modeling_qwen4_exp.py`
SHA256 is `b4649e4d0a709d46bc3eeae4b5a5bd2854c8c80404e4fe0a310aef4acd9da1c1`.
The reference record names PyTorch 2.14.1+cu130, kernels 0.17.2, Triton 3.8.0,
and finegrained-fp8 revision `8c178950e8710e26a2210e4e909cd02dc16c8715`.

`FP8Experts.forward` (lines 712–771) loops over selected experts and invokes
FP8 linear twice: combined gate/up, then down. Dynamic activation scales are
passed as `None`, selecting inline activation quantization. It does **not**
dequantize checkpoint FP8 experts to BF16 first. The `weight.element_size()>1`
branch is for already-dequantized weights. `fp8_grouped_mm_experts_forward`
(lines 514–606) instead sorts token/expert pairs, uses grouped FP8 matmul for
both projections, and restores routing order. Both are W8A8. Eager expert
reduction uses FP32 accumulation in expert-ID order; grouped dispatch uses
its own reshape/sum reduction. Vade retains ascending expert-ID reduction.
The fp8 linear dispatcher can also select DeepGEMM when available; this port
specifically follows the finegrained-fp8 reference, not that other backend.

For each token and contiguous K128 group, the finegrained kernel computes
`scale = max(abs(BF16_input)) / 448` in FP32. Quantized bytes are
`E4M3_RNE_satfinite(input / max(scale, 1e-12))`. The stored scale is not floored:
a zero group stores zero and contributes zero. The rounding is nearest even.
The inspected
matmul source promotes each K128 dot with activation and weight scales.
Vade reuses its existing helper exactly: four ascending K32 E4M3 MMAs form an
FP32 block dot, multiplied by `(activation_scale * BF16_weight_scale)`, then
added to the FP32 accumulator in ascending K128 order. This scale association
is retained from the existing FP8 implementation; bit equality to the official kernel is not asserted.

## Shards and boundaries

Each expert has gate/up `[640,2560]` and down `[2560,640]`, with BF16 scale
grids `[5,20]`, `[5,20]`, `[20,5]`. Gate and up are stored separately but have
the same token operand. Two ranks retain contiguous whole 128-channel blocks:
even experts split 3/2 (384/256 channels), odd experts 2/3. Gate/up slice
weight rows and scale rows; down slices weight columns and scale columns.
The local activation groups therefore have the same maxima as their full
expert groups. No scale recomputation or weight padding is needed.

Official BF16 compute boundaries are: incoming hidden state; gate/up outputs;
SiLU output; SiLU times up; down output; routing weights cast to output dtype;
weighted expert output; final expert reduction. Eager expert sums accumulate
weighted BF16 values in FP32 before returning BF16. Vade quantizes the original
hidden row once for all selected gate/up projections, rounds their outputs,
rounds SiLU and its product separately, then quantizes each local intermediate
K128 group once. Down rank partials remain FP32 until the existing ordered
all-reduce; the reduced down output and weighted product are rounded to BF16.
Decode uses M16/N64 tiles; prompt experts with at least 32 routed rows reuse
each weight tile across M32/N64. Both retain the same four K32 MMAs per K128
dot and the same scale association and ascending block sums. Prompt entry
points keep the larger tile's register allocation out of decode. Decode W8A8
expert tiles stream their K128 blocks through three cp.async stages, so two
blocks are in flight while one multiplies; every output keeps the same order. Rank-local
partial summation differs from an unsplit full-K reference, so official logit
agreement still requires measurement. The one-device anchor uses the same two
logical ranks and can test transport-independent equality without changing this
summation structure.

`FP8Embedding.forward` (lines 344–365) gathers n-gram FP8 rows, casts to the
single `weight_scale` tensor's dtype, and multiplies by that scale. With the
checkpoint BF16 scale this returns BF16. It is a lookup, not W8A8 GEMM; the scale
is a multiplier, not a reciprocal. The table is never expanded at load time.
Ordinary BF16 modules return BF16 in the official model, with explicit FP32
router softmax, normalization and recurrent-state operations where prescribed.

Flash retains FP32 activation storage, normalization, residual mixtures and
recurrent state. Covered BF16 projections use the split tensor-core arithmetic
below. This is not a claim of full-model official BF16 parity.
The existing FP32 and BF16 fixtures are not a W8A8 reference. Their strict
comparison remains enabled and can fail after this precision change.

## Rank counts

The FP8 split is built so that two and four Sparks give the same sums. Every
sum that crosses ranks keeps the grouping of the two-rank split. A device run
on each Spark count must confirm that the token IDs are identical.

- Attention `o_proj`, GDN `out_proj` and the shared expert down split K. Two
  ranks hold four canonical K ranges each. On four ranks a shard holds two of
  those ranges, and the generic projection entries compute only those
  ([`canonical_span`](../crates/gpu/kernel_params.rs)). A part that the shard
  does not hold adds -0.0, the exact FP32 identity.
- Rank partials add as a pair tree, `(r0 + r1) + (r2 + r3)`, in the host
  all-reduce and in the shared expert fold
  ([`fold_ranks`](../crates/gpu/kernel_params.rs)). Two and three ranks add
  rank by rank, as before. On a path of three or four Sparks, large sums
  (prompt rows) pipeline through the neighbours in this order
  ([`path_reduce.rs`](../crates/seqpar/src/transport/path_reduce.rs)); other
  rank counts gather all partials first.
- The routed down sum of the 48 decoder layers groups its five K128 blocks as
  the two-rank split does, at every rank count (`EXPERT_FOLD_RANKS`).
- The MTP draft layer (`mtp.layers.0.mlp`) still splits each routed expert's
  blocks by rank, and that sum is not covered. Its drafts can differ between
  two and four Sparks. With Strict verification this changes speed, not the
  token IDs. Sampled verification (`"draft":"sampled"`) can choose other
  tokens for the same seed.

The loader registers a four-rank shard by weight address
([`split_ranges.rs`](../crates/engine/src/flash/split_ranges.rs)), and
`project_rows` puts its ranges in the high bits of the launch's `cols` word
(`bf16_cols_word`). Such a shard takes the generic entries and the 64-row
prompt tiles, which keep the same canonical chains. Two ranks register nothing: each launch has the same
descriptor bytes and kernel as before, and the fixed-shape decode entries keep
their code. One Spark cannot hold the FP8 checkpoint. Six and eight ranks keep
four ranges per shard and give their own IDs. `FLASH_BF16_EXPERTS=1` keeps
rank-split routed down sums and is not covered. The host model
[`split_order_tests.rs`](../crates/gpu/src/split_order_tests.rs) repeats
these addition orders on the CPU for two and four ranks.

## Product controls and memory

W8A8 is the default. `FLASH_BF16_EXPERTS=1` explicitly selects the previous
BF16 expert operands; invalid values error. Both ranks must use the same value.
This existing expert control is separate from the fixed BF16 projection mode;
activation/cache precision and approximate verification controls stay explicit.
For an FP32 activation `x`, the BF16 projection operands are
`hi = BF16_RNE(x)` and `lo = BF16_RNE(x - FP32(hi))`. Each is multiplied by
the unchanged checkpoint BF16 weights, accumulated in FP32 and combined in the
fixed projection order. This changes output bits from older scalar builds.
Decode and prompt tiles use the same four fixed K ranges on every rank. In
each range the high and residual chains accumulate separately in FP32, then
form `P_i = H_i + L_i`. The final merge is `(P_0 + P_1) + (P_2 + P_3)`,
independent of batch width or tile size. Eligible dense matrices are permuted
in place into the dense engine's 64x64 MMA layout when their row count is a
multiple of 64; smaller matrices retain four-row fragments. A bounded 64-row
device tile stages the permutation and is released before inference. Weights
retain one resident copy. Shards whose K is a multiple of eight but not sixteen
retain row-major weights; operands outside the final K16 tile are zero. The
four canonical ranges count the rounded-up K16 tiles at every row width, with
no resident weight padding or alternate arithmetic fallback.
Prompt inputs use one reusable 167,772,160-byte BF16 high/residual
operand tile on the backend's ordered stream, reserved before graph capture.
MMA prompt groups hold up to 4096 tokens in one model-owned workspace; attention
retains its canonical 64-token chunks. Serving funds that workspace once,
including routed-expert buffers allocated before admission. Each session keeps one row for
decode, or up to four with MTP verification, independently of its KV context
capacity; its expert padding follows that row bound.
Prompt execution temporarily borrows the shared workspace; decode graphs only
capture session-owned buffers and the stable shared operand tile. HC decode
normalization and SiLU emit both BF16 operand terms directly while retaining the FP32
values needed by the residual mixture. Decode partial sums use only shared memory.
`FLASH_DUMP_LOGITS=DIR` writes every generated-position row as little-endian
FP32 in `position-NNNNN-rank-R.f32`. It is off by default and does not change
token selection. Compare local-anchor rank 0 against distributed rank 0;
rank suffixes prevent the two processes from overwriting each other's files.

Weights remain a single resident checkpoint copy. For T tokens, P=10T,
R=P+15*min(P,512), and C=ceil(P/128)+min(P,512), grouped scratch reserves:
- routing: `4*(512+513+2P+T+1+2C+R)` bytes;
- W8A8 activations: `16020*R + 2640*T` bytes;
- legacy activations: `11520*R` bytes.
For T=1 the W8A8 reservation is 2,570,756 bytes; for the session's T=64 tile,
it is 133,502,256 bytes. These are requested grouped scratch bytes, not measured
peak device allocations; the backend may align allocations. Scratch is reused
across layers and contains no expanded weights.

Routed weight reads across both ranks total `48*10*3*640*2560 = 2,359,296,000`
FP8 bytes per token plus 288,000 scale bytes. Each 128-channel block contributes
983,160 bytes including scales. Per-rank bytes depend on selected expert parity;
the balanced mean is 1,179,792,000 bytes. These are geometry-derived operand
counts, not measured memory traffic or a throughput claim. Cache behavior,
allocation and collective costs require native measurement.

The official reference must escape and anchor `modules_to_not_convert` names:
unanchored `mlp.gate` also matches `mlp.gate_proj` and discards its scales.
That correction belongs solely in the reference loader; Vade uses checkpoint
names and scale grids directly. No external implementation code was copied.
