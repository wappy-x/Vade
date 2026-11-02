# Ideas and reference behavior

This page lists the papers, official model sources, other projects and earlier
Vade work whose ideas or documented behavior Vade's engine uses. Each entry
says what was used and which part is Vade's own.

**No external implementation code was copied into the engine.** The engine is
Vade's own Rust, and its GPU code is Rust, with inline PTX assembly where
needed, compiled to PTX. When an entry names another project, Vade uses an
idea, a published equation or a documented behavior from it, or ran it
separately as a reference to compare against. The third-party files the repository does keep, such as official model
configurations, chat templates and Unicode data, are listed with their
licenses in [NOTICE](NOTICE).

Short commit hashes and branch names, such as 8e9657e or `work/m32`, name
earlier Vade work from before this repository's history begins, so you will
not find them here.

<details>
<summary><b>Terms on this page</b>: number formats, GPU words and model parts</summary>

<dl>
<dt>BF16, FP32</dt>
<dd>16-bit and 32-bit floating-point formats.</dd>
<dt>E4M3, E2M1</dt>
<dd>8-bit (FP8) and 4-bit (FP4) floating-point formats.</dd>
<dt>NVFP4</dt>
<dd>NVIDIA's 4-bit format: E2M1 values with an E4M3 scale per 16 values and
one FP32 global scale.</dd>
<dt>W4A16, W4A4</dt>
<dd>4-bit weights with 16-bit activations, and 4-bit weights with 4-bit
activations.</dd>
<dt>PTX</dt>
<dd>NVIDIA's GPU assembly language, which the driver compiles for the
chip.</dd>
<dt>MMA</dt>
<dd>A tensor-core matrix multiply instruction.</dd>
<dt>CTA</dt>
<dd>A block of GPU threads that runs together and shares on-chip
memory.</dd>
<dt>KV, KV cache</dt>
<dd>The saved keys and values of tokens already read, so they are not
recomputed.</dd>
<dt>GQA</dt>
<dd>Grouped-query attention: several query heads share one key/value head.
GQA8 means eight query heads per KV head, GQA4 four.</dd>
<dt>MTP</dt>
<dd>Multi-token prediction: a built-in head that guesses tokens ahead.</dd>
<dt>GDN</dt>
<dd>GatedDeltaNet: a linear-attention layer that keeps a fixed-size recurrent
state instead of a growing history.</dd>
<dt>HC</dt>
<dd>Hyper-connection: a learned mix that keeps four residual streams and
decides how much each block adds to each stream.</dd>
<dt>PLE</dt>
<dd>The n-gram embedding: an extra embedding of hashed pairs and triples of
recent tokens, added before decoder 1's attention.</dd>
<dt>QSA</dt>
<dd>Indexed attention: softmax attention limited to the token blocks that a
small indexer selects.</dd>
<dt>packed8</dt>
<dd>Vade's layout for BF16 weights: tiles of eight rows, rearranged so each
GPU lane loads its matrix-multiply fragment directly.</dd>
<dt>Canonical</dt>
<dd>Vade's fixed reference arithmetic, with a set order of operations. A
faster path must give the same bits.</dd>
<dt>Two-component</dt>
<dd>An FP32 value carried as two BF16 numbers: the rounded value and its
rounded remainder.</dd>
</dl>

</details>

## Papers, model sources and other projects

- Yaniv Leviathan, Matan Kalman and Yossi Matias, *Fast Inference from
  Transformers via Speculative Decoding* (2023), https://arxiv.org/abs/2211.17192;
  Charlie Chen et al., *Accelerating Large Language Model Decoding with
  Speculative Sampling* (2023), https://arxiv.org/abs/2302.01318: sampled draft
  proposals, acceptance by min(1, p/q), positive-residual correction and a target
  bonus token. Vade uses these as mathematical ideas for the opt-in Flash
  Sampled verifier, built on Vade's own sampler and GPU kernels.

- Qwen3 official model card (https://huggingface.co/Qwen/Qwen3-8B): the
  thinking sampling defaults (temperature 0.6, top-p 0.95, top-k 20), the
  non-thinking defaults (0.7, 0.8, 20) and the optional presence penalty for
  repetition. The same card gives the opt-in YaRN settings: factor 4,
  original_max_position_embeddings 32768, maximum 131072. The YaRN frequency
  ramp and attention scaling follow the published YaRN method and the
  Transformers RoPE parameter definitions
  (https://huggingface.co/docs/transformers/main/en/internal/rope_utils).

- Hugging Face Transformers: standard Qwen3 reference inference, tokenizer and
  chat template, used only to generate comparison fixtures, and the sign
  convention for the repetition penalty.

- SplitMix64 (Guy L. Steele, Doug Lea and Christine H. Flood, "Fast splittable
  pseudorandom number generators", https://doi.org/10.1145/2660193.2660195;
  reference code on Sebastiano Vigna's page,
  https://prng.di.unimi.it/splitmix64.c): integer finalizer constants for
  stateless seeded sampling. This is not a cryptographic RNG.

- TensorFold: conversation state snapshots and independent branches at message
  boundaries. Vade builds them on the engine's public fork/truncate API.

- TensorFold and vcruz305: keep the complete decode pass and KV cache on
  device, retain activation buffers, resolve kernels once, and queue launches
  until the output is needed.

- FlashAttention: Vade's attention kernels use its online
  maximum/denominator recurrence, query tiles and on-chip QK/PV products.
  NVIDIA's PTX `ldmatrix` mapping informs the V transpose. Retained-FP32
  attention extends this to two-component views of retained FP32 KV. The
  shared-input GEMM and exact verifier staging come from Vade's own measured
  kernels.

  Decode and verify attention add Vade's grouped split mapping: four query
  heads share K/V reads, and fixed row-local partitions preserve the
  decode/verify arithmetic. Long-context attention keeps the same recurrence,
  with Vade's own Rust shared-memory swizzling, register probability fragments
  and on-chip FP32 splitting. Shared KV prefixes use the installed CUDA
  header's allocation-handle mapping contract; GB10 whole/partial alias
  behavior was measured directly.

- The serving scheduler follows SGLang mixed-chunk scheduling and per-row
  attention sizing, and vLLM token-budget scheduling and abort-before-close.
  Context-timed prefill and capability-scoped cancellation are Vade's own;
  streaming attention and shared KV pages come from Vade's existing code.

- Token Recycling (https://arxiv.org/abs/2408.08696): retain token successors
  from target logits, including rejected rows. Vade drafts linear chains; its
  greedy HTTP path learns the observed best successor without downloading full
  logits.

- REST (https://arxiv.org/abs/2311.08252) and CREST
  (https://arxiv.org/abs/2408.04678): retrieve and rank several suffix
  continuations. Vade uses bounded, request-local token history.

- Lookahead Decoding (https://arxiv.org/abs/2402.02057): reuse wrong-prefix
  predictions as guesses in a later verification cycle. Vade shifts
  already-computed rejected rows without extra target passes.

- Grouped-head prefill attention, studied in TensorFold and recorded in Vade's
  internal notes: one block shares K/V across a group of query heads,
  asynchronously stages swizzled tiles, and keeps probabilities in registers at
  absolute key tiles. Swizzled shared-memory tiles themselves are older
  practice from CUTLASS and GPU textbooks.
  Vade's own Rust attention kernels now use single BF16 operands and share
  each K/V tile across its query heads.

- SGLang's GB10 unified-memory sizing, as recorded in our hardware
  measurements: size from Linux MemAvailable, so reclaimable file cache counts
  as capacity.

- SGLang RadixAttention and vLLM automatic prefix caching: reuse committed
  token prefixes independently of client session capabilities. The
  block-hash index, exact comparisons, API-key scopes, live-prefill sharing and
  LRU accounting are Vade's own, built on its shared KV pages.

- OpenAI Codex CLI 0.160.0 (https://github.com/openai/codex): Responses request
  shapes and the semantic SSE event contract were established with local wire
  captures against a mock and Vade's host scheduler. The stateless
  Responses-to-chat translation and event encoder are Vade's own. Presenting a
  custom (freeform) tool to the model as a function with one required `input`
  string follows the JSON form of Codex's own `apply_patch` tool (idea only).

- Qwen3.5-family XML tool calls: typing each `<parameter>` value by the tool's
  declared JSON schema follows the idea in the Qwen3-Coder tool parsers of vLLM
  and SGLang. The streaming parser is Vade's own.

## Hardware documentation and design ideas

- NVFP4 GB10 primitives follow NVIDIA's public PTX block-scale instruction and
  fragment contracts. TensorFold's native architecture target and packed FP4
  MMA choice informed the approach. Pinned source links, layouts and arithmetic
  limits are in `docs/fp4-kernels.md`.

- Vade's own design: sampling uses a counter keyed by each token's absolute
  position, so a speculative step can be replayed exactly.

- Earlier Spark/SGLang experiments: foreground chat preemption between
  background prefill chunks. Vade implements it as a bounded single-thread
  priority scheduler with nonblocking sockets, with no external crates.

- Earlier Vade measurements on DGX Spark: avoid CUDA copies from mmap-backed
  pageable checkpoint pages; read into pinned or unified memory. This loader
  applies that idea with its own Rust pread code, bounded transient staging and
  device arenas, with no host weight copy.

- NVIDIA CUDA Driver API and CUDA for Tegra documentation: allocation ownership,
  managed/pinned/system-memory access contracts and different caching behavior.
  https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
  The installed CUDA 13.0 cuda.h is the ABI source.

- Earlier Vade measurements: prioritize weight bandwidth and remove host
  synchronization/allocation from decode. This build keeps the existing Rust
  warp32 matvec while moving surrounding operations onto the device.

- The kernel implementation starts from Vade's original Rust matvec/GEMM
  code. The recorded ideas it uses were studied in TensorFold and jayleaton
  ROOFLINE (wide coalesced loads and warp-cooperative rows), the LLVM NVPTX
  intrinsic definitions and NVIDIA's PTX MMA fragment-layout specifications.
  Following a precision analysis, the integration adds three-product,
  two-component attention and removes unused variants. Batched scheduling,
  causal attention and storage integration are Vade's own Rust.

- The draft implementation integrates Vade's original Rust draft2 suffix lookup
  and guarded verification. Prompt lookup is Apoorv Saxena's Prompt Lookup
  Decoding (https://github.com/apoorvumang/prompt-lookup-decoding). We first
  saw it, and position-keyed sampling, in TensorFold and vcruz305's work. No
  external model is used.

- Draft-model decoding uses the standard speculative-decoding idea: a smaller
  official Qwen3 model proposes a chain, the target verifies consecutive rows,
  and only its matching prefix commits. Vade extends this to sampled requests:
  the draft proposes greedily, and the target's own position-keyed sample
  decides every accepted token and correction, so drafting never changes a
  sampled output. Proposal generation, KV rollback, tied-weight views and
  canonical acceptance are Vade's own.

- The CUDA virtual memory allocation/mapping ABI is taken from the installed
  NVIDIA CUDA 13 header. Fixed virtual KV addresses with appended physical pages
  are Vade's own.

- Tensor parallelism uses the standard Megatron decomposition: head-shard Q/K/V,
  row-shard gate/up, column-shard O/down and sum in fixed rank order. Direct RC
  SEND/RECV, preposted receives and striped physical links are standard verbs
  techniques, and the declarations follow the rdma-core headers. The optional TP packed8 kernel is Vade's own Rust
  arithmetic with unrolled cache-global loads. The engine's layer hooks for
  tensor parallelism use the engine's own numeric operations and kernels.

- Multi-session design: share exact projection rows across unrelated chats
  while attention and KV stay per session. The multi-session code extends Vade's
  Rust kernels for mixed decode, prompt and draft rows; priority prompt chunking
  follows the earlier Spark/SGLang experiments credited above.

- Sequence-parallel design: additive ordered layer hooks share immutable weights
  and exchange FP32 KV between sequence slices. This integration keeps the
  existing Rust arithmetic and leaves transport to the caller.

- Decode fusions follow Vade's shared-input QKV and gate/up epilogue ideas.
  These Rust kernels keep separate existing weight allocations and canonical
  packed8 arithmetic; single-CTA RMS keeps the original two-stage reduction
  tree.

- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/):
  native m16n8k32 E4M3 fragment mapping and packed E4M3 conversion operand
  order. FP8 quantization, block-scaled GEMM, shared-memory staging and
  dispatch are Vade's own Rust.

- Resident-FP8/BF16 GEMM uses NVIDIA PTX's exact E4M3-to-FP16 conversion,
  packed BF16 multiply, and BF16 MMA. The exponent transfer into the block
  scale, tile-local promotion, and warp-private decode pipeline are Vade's own.

- Prefix-shared/cascade attention: stage common physical K/V tiles across
  independent chat queries. This implementation extends Vade's own
  decode/verify reduction kernel and physical-page aliases.

## Kernels carried over from earlier Vade work

- The shared-attention work brings in two earlier Vade kernels: shared GQA8
  attention (8e9657e), and the two-stage 256x128x64 prefill GEMM with wide QKV
  (7ae8d3d, 570d94d). It adapts the dense QKV and O staging to this engine's
  existing packed weight fragments, so the weights still live in one resident
  BF16 arena. It also keeps the newer m32 64x32 warp tiles (c10db29) and the
  paired attention loads. The only new parts are packed-weight addressing and
  expert dispatch.

- The dense/expert GQA8 body folds in Vade's own expert code, commit 59b8a3c:
  immediate ldmatrix offsets, swizzled BF16 cache tiles and overlapping
  cp.async refills with transient addresses. Base-2 softmax combines the
  head-dimension and exponent scales.

- Prefill tiles use NVIDIA PTX's asynchronous L2-only global-to-shared copies
  and native BF16 MMA. The 256-row CTA, resident tiled-weight addressing and
  fused SwiGLU BF16 fragment epilogue are Vade's own. Bulk-copy candidates
  were rejected. Instruction reference:
  https://docs.nvidia.com/cuda/parallel-thread-execution/ .

- Vade's GQA8 attention design (commit 8e9657e) informs the GQA4 port in
  2de67a9: immediate ldmatrix offsets, fewer live QK fragments, and K/V refills
  after their last readers. GQA4 refills directly from resident BF16 KV, pairs
  matrix loads, and keeps the existing exponential and reduction order.

- Vade's own earlier GEMM work supplies the Qwen3-8B output/down two-stage
  residual/RMS tile and GQA4 fragment output/refill schedule. The existing
  shared prefill helper keeps the existing Qwen3-32B and expert paths.

- Vade's own branches `work/m32` (commit c10db29) and `work/gemm` (commit
  8b2a874) informed the two-stage 128x128x64 grouped expert gate/up tile in the
  MoE work (commit dec2a09). The Rust kernel reuses resident packed weights
  without repacking.

- Dense hybrid prefill reuses Vade's own m32 two-stage 256x128x64 GEMM and fused
  SwiGLU/QKV work (7ae8d3d, 570d94d, c10db29), adapted to the existing one-copy
  packed weights. Its BF16 tensor chunk products, solve and scan keep FP32
  accumulation and state. The makers' normalized chunk products and triangular
  solve run in FP32, and the implementation restores Vade's own parallel FP32
  block inverse/solve, with ascending FP32 products and scan. BF16 tensor
  products stay an explicit opt-in, because they failed the pass-or-fail chat
  check on the native CUDA path.

- The FP8 path ports Vade's own FP8 commit f0c36c8: warp-private cached
  short-attention scores, register-held queries and ordered normalization
  read-ahead. BF16 activation packing was adapted from the existing prefill
  operand and evaluated with the shared canonical staged helper; projection
  experiments were removed. No dependencies or lower-precision operands were
  added.

## Later models and kernels

- Qwen3.6's official config and installed Transformers Qwen3_5Moe model define
  fused expert tensor shapes, normalized top-8 BF16 routing, ascending expert
  accumulation, and the sigmoid-gated shared expert. Vade uses them as a behavior
  reference; the Rust implementation reuses Vade's hybrid and expert kernels.

- Qwen3.8-27B's official BF16 config and 15 MTP tensor headers define the shared
  embedding/head, fusion and one full-attention layer. The installed official
  Transformers qwen3_5 source ignores mtp.*: it gives the decoder/norm/RoPE
  precision contract, but no MTP forward pass. Transformers issue 48913 records this
  topology mismatch: https://github.com/huggingface/transformers/issues/48913.
  VeOmni's MTP architecture note supplies embedding-first fusion, final-normalized
  trunk features and unshifted rotary positions:
  https://github.com/bytedance-seed/veomni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/docs/key_features/mtp.md.
  No other inference engine's source was read. Parity with the maker's MTP forward
  pass is still unverified; the target model's canonical verification is what
  keeps the output exact.

- Qwen's official Qwen3.8-27B checkpoint metadata defines the text tensor names,
  hybrid layer schedule, zero-centered norms, partial RoPE and text chat
  protocol. The adapter and Unicode-mark tokenizer extension reuse Vade's own
  Flash delta-net/gated attention and dense BF16 loader/GEMMs. Native BF16
  operation boundaries follow Transformers 5.18.0's Qwen3_5 model: BF16 beta and gated
  norm, SiLU for GDN and sigmoid for full attention, even though the checkpoint
  also carries output_gate_type=swish.
  https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py

- Qwen3.8-27B-FP8 revision 017b9c7a defines E4M3 weights, BF16 128x128
  block scales and the unconverted module list. Dense FP8 projections reuse
  Vade's own Flash FP8-to-BF16 promotion and tensor-core reductions. The gate
  uses Transformers 5.18.0's official FineGrainedFP8Config(dequantize=True)
  operation: multiply decoded weights by checkpoint scales in FP32, then round
  to BF16.
  https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/finegrained_fp8.py

- Qwen3.8-Flash-Next official checkpoint and Transformers FP8 conversion:
  E4M3FN values multiplied by promoted BF16 128x128 block scales, and one scalar
  scale for PLE embeddings. The shape/ownership contract and upstream source
  links are in docs/history/flash-next-architecture.md. Flash model metadata was
  adapted from an earlier, parked Vade implementation; execution and transport
  remain Vade's own Rust.

- Flash HC/PLE integration reuses Vade's parked HC/PLE algebra and the existing
  Vade rank collectives. N-gram hashing, EOS boundaries, four-stream expansion
  and grouped norms follow the official model source cited in
  docs/history/flash-next-architecture.md; Rust kernels and bounded
  resident-row access are Vade's own.

- Flash GDN integration follows the official Transformers Qwen4Exp chunk64
  triangular-solve equations and cached gated-delta recurrence (source pinned in
  docs/history/flash-next-architecture.md). Head-sharded projections,
  convolution/cache ownership, fixed-order Rust GPU arithmetic and shared
  workspace are Vade's own. The short sublayer snapshot uses hooks around
  that official CPU implementation, without changing model arithmetic.

- Flash indexed attention follows the official Qwen4Exp main-attention, raw-key
  pooling, partial RoPE and QSA selection equations cited in
  docs/history/flash-next-architecture.md. Sharded indexer gathering, causal
  Rust kernels and layer-local caches are Vade's own; BF16 projections reuse
  Vade's GDN kernel. The short oracle adds hooks only, keeping official FP32
  arithmetic.

- Flash sharded MLP follows the official Qwen4Exp full-softmax/top-10 router,
  ascending expert-ID accumulation, shared expert and HC equations cited in
  docs/history/flash-next-architecture.md. FP8 block products are rounded to
  BF16 by the official dequantization policy before FP32 promotion.
  Expert/channel ownership, one-allocation views, Rust kernels and staged rank
  reduction are Vade's own; BF16 projection/HC and TCP transport reuse Vade
  code.

- Google DeepMind Gemma 4 (Apache-2.0), official gemma-4-31B-it config,
  tokenizer artifacts and canonical chat template dated 2026-07-09:
  https://huggingface.co/google/gemma-4-31B-it . The template is retained verbatim
  under crates/tokenizer/templates/gemma4.jinja to validate model folders.
  Rust config parsing, shared BPE extension and text formatter are Vade's own.
  Serving follows the tokenizer config response_template (thought channels and
  raw-string tool arguments); tool dictionaries use Jinja stable case-insensitive
  dictsort semantics. The Rust parser and ordered JSON view are Vade's own.
  Local Hugging Face Transformers Gemma4TextAttention, Gemma4RMSNorm,
  Gemma4TextDecoderLayer and proportional RoPE were read only to establish
  behavior and compare CPU tokenization/template output.

- Gemma 4 text GPU arithmetic follows Google's official Transformers
  Gemma4TextAttention, Gemma4RMSNorm, Gemma4TextDecoderLayer and proportional
  RoPE behavior (tooling reference only). The Rust implementation reuses Vade
  projection and attention helpers; no reference runtime code is linked.

- Hugging Face Transformers / Google Gemma 4 assistant reference: query-only
  decoder over the target's last local/global KV, concatenated scaled target
  embedding and normalized hidden state, pre/post projection recurrence at a
  fixed position, and full tied head when ordered embeddings are disabled. Vade
  implements it with its shared Gemma layers and strict speculator.

- Qwen3.8-27B decode reuses Vade's own packed BF16 asynchronous staging and CUDA
  graph capture/replay from decode commits d4e74a7 and 4b7d54e. Native BF16
  gate/up, projection and residual rounding follow the official Transformers
  precision boundaries already credited above. The cooperative norm preserves
  Vade's ascending 32-lane reduction.

- Independent hybrid prompts reuse Vade's own single-session projection/MLP
  path, per-session Flash convolution/FP32 inverse/solve and causal attention,
  and shared packed8 decode. Session state arenas, stream-ordered device clears
  and the persistent FP32 scan are Vade's own, keeping the existing operation
  order. The device clear ABI follows NVIDIA's CUDA Driver API.

- Independent prompt descriptors reuse Vade's own tiled FP32 GDN kernels and
  cached state arenas. Short prompts preserve separate ascending multiply/add
  reductions; long prompts keep the incoming fused FP32 core. Native shared
  decode fuses products of promoted BF16 operands without changing packed8
  accumulator boundaries.

- The projection implementation reuses Vade's own device argmax, short-prompt
  MMA reduction and vector arenas to remove full prompt-logit readback, pool
  narrow projections and grow all full-attention layers together. The compact
  scalar decode loop keeps the eight packed8 accumulator streams. No new
  arithmetic approximation is used.

- The packed8 implementation expands Vade's existing canonical staged packed8
  helper to 64 outputs, sharing each staged activation group across eight
  output groups. Weight layout, BF16 products, accumulator streams and ascending
  K order remain unchanged; host and kernel use a shared repr(C) launch ABI.
  Direct-load and alternative scheduling attempts were discarded.

- Qwen3.8-27B prefill reuses Vade's packed BF16 GEMM and shared online attention
  at 256-wide heads. Gated-delta chunk equations follow the official
  Transformers implementation credited above; the parallel block inverse,
  independent chunk preparation and convolution kernels are Vade's own Rust.
  Recurrent state remains FP32, with the official BF16 projection, activation
  and gating boundaries.

- Staging, two-chat warp and pinned-readback experiments reused Vade's own
  canonical projection and pinned argmax/exchange designs. All performance
  variants were discarded; the K256 launch precondition is enforced. No
  approximation was added.

- Phi-4: official Microsoft BF16 configuration and tokenizer artifacts; fused
  projection row loading and BF16 boundaries reuse Vade Phi-4 3315d8d and the
  shared dense kernels.

- Llama 3.1/3.2: Meta official BF16 configs, frequency-dependent Llama 3 RoPE
  scaling and text chat templates. Rust configuration, rendering and frequency
  table logic are Vade's own.

- The long-context path uses Vade's own canonical FP32 chunk scan and BF16 MMA
  A-fragment packing: both row halves reuse each prior-state load, K staging
  overlaps intra-chunk output, and the RMS/gate epilogue emits the existing
  packed operand. SGLang 0.5.21's public FLA chunk decomposition and cuBLAS
  kernel names informed phase attribution; its narrower dot operands were not
  adopted.

- The recurrent scan follows the installed official Transformers 5.18.0 Qwen3.5
  chunk-gated-delta-rule arithmetic: FP32 beta-scaled products, pairwise decay,
  then a unit-lower triangular solve. Vade's solve keeps its canonical blocked
  inverse/product order and keeps that inverse on chip. Direct substitution was
  rejected because it changed greedy IDs. The scan reads ahead into dead
  attention storage, and the full-attention gate reuses Vade's existing BF16
  operand layout. No NVIDIA libraries enter the engine.

- The normalization follows PyTorch CUDA Reduce.cuh mean arithmetic: four
  adjacent FP32 accumulators, descending warp reduction and reciprocal-count
  multiplication. The Rust normalization fixes short-prefill and batch-one
  decode accumulator layouts, each independent of pooled chat width. Bounded
  partial-prefill replay reuses Vade state copies and canonical tile execution.

- FP8 activation round follows the official checkpoint's dynamic E4M3,
  per-token 128-element activation groups and BF16 128x128 weight scales.
  The installed Transformers 5.18.0 finegrained_fp8 dispatcher consumes those
  scales through its finegrained FP8 matmul, returning the input BF16 dtype.

  Vade extends its own m16n8k32 Rust/PTX GEMM. Its arithmetic, in order:

  1. BF16-round each input.
  2. Find maxabs in each group and set activation scale=maxabs/448.
  3. Divide by max(scale,1e-12).
  4. Convert with round-to-nearest-even/saturate-finite E4M3.
  5. Form an FP32 K128 block dot from four ascending K32 MMAs.
  6. Multiply that dot by (activation_scale * checkpoint_weight_scale) and add
     it to the FP32 running sum in ascending K128 order.
  7. BF16-round only the output.

  Zero groups have scale zero and contribute zero; this zero/tiny-group scale
  convention reuses Vade's existing quantizer. Resident weights keep the
  checkpoint's row-major bytes, and no weight dequantization copy is retained.
  Vade's larger token tile, PTX ldmatrix fragment loads and separate decode
  entry reuse the same original Rust helper and K128 sum order.

  The same convention appears in Hugging Face finegrained-fp8 v4
  ([pinned source](https://huggingface.co/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py#L440)):
  a per-128 activation maximum, a raw max/448 scale, and division by
  max(scale, 1e-12) before E4M3 rounding.

  The separate reference environment installs Hugging Face kernels 0.17.2 and
  finegrained-fp8 revision 8c178950e8710e26a2210e4e909cd02dc16c8715. Its inline
  quantizer uses the same maxabs/448 convention. Its matmul scales each K128
  dot by the activation scale first, then the weight scale.

  One bug had to be worked around. Transformers 5.18.0's exclusion regex is
  unanchored, so it wrongly matches `mlp.gate_proj` against `mlp.gate` and
  discards 64 checkpoint scale tensors. The reference run anchors and escapes
  the exclusion names before loading. The model forward pass and the official
  kernels stay unchanged. Raw source hashes and both loader results are kept
  with the records from that measurement round.

- Flash-Next W8A8 follows Transformers 5.18.0 `FP8Experts` and the same
  finegrained-fp8 revision recorded above. The grouped path reuses Vade's dense
  K128 helper and quantizer, with BF16 projection, SiLU, product and
  post-collective boundaries. Source provenance, reference dispatch differences
  and remaining FP32 operations are documented in `docs/flash-fp8-math.md`.

- Long-context decode attention for Qwen3.5-family full-attention layers:
  loading each cached K/V tile once per key head for all of its query heads
  (grouped-query sharing) and double-buffering the V tiles into shared memory
  with cp.async follow widely used decode-attention practice (FlashDecoding,
  FlashInfer). Vade's kernels keep the same sequential softmax and value sums as
  the single-chat walk.

- Flash-Next MTP fusion follows the model semantics documented by SGLang's
  installed `qwen4_exp_mtp.py`: whole-residual normalization, one shared
  projection per residual stream, and a broadcast projected token embedding.
  Vade's implementation uses its own sharded weights, attention and MoE kernels.

- NVIDIA Model Optimizer, the official
  [Qwen3.8-Flash-Next-NVFP4 model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4),
  and that checkpoint's headers and quantization metadata: mixed NVFP4 main
  experts, FP8 MTP/PLE and BF16 dense storage. Values are packed E2M1, with an
  E4M3 scale per 16 values and one FP32 global scale. The packed loader and CPU
  dequantization (std-only Rust), and the W4A16 register dequantization, lane
  redistribution and canonical BF16 MMA path (Rust/PTX), are Vade's own. This
  explicit W4A16 operand mode does not implement the checkpoint's W4A4
  quantizer.

- TensorFold's per-stream state-pointer tables informed batching independent
  Flash-Next GDN recurrences. Vade extends its own Rust kernels and keeps each
  session's original arithmetic and private state.

- vLLM issue #59985 shows a class of failure where a tokens == 1 kernel special
  case breaks batch invariance. Vade routes its indexed attention and MoE
  router/shared-expert projections through one kernel per width, and checks
  one-row launches against 4-, 8- and 16-row launches. The kernels and that
  check are Vade's own.
