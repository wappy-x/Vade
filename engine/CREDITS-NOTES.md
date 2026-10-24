# Ideas and reference behavior

- Yaniv Leviathan, Matan Kalman and Yossi Matias, *Fast Inference from
  Transformers via Speculative Decoding* (2023), https://arxiv.org/abs/2211.17192;
  Charlie Chen et al., *Accelerating Large Language Model Decoding with
  Speculative Sampling* (2023), https://arxiv.org/abs/2302.01318: sampled draft
  proposals, acceptance by min(1, p/q), positive-residual correction and a target
  bonus token. Used as mathematical ideas for the opt-in Flash Sampled verifier.
  Implementation is original Rust using Vade's sampler and GPU kernels; no code
  was copied.

- NVFP4 GB10 primitives follow NVIDIA's public PTX block-scale instruction and
  fragment contracts. TensorFold's native architecture target and packed FP4
  MMA choice informed the approach; the Rust implementation is original.
  Pinned source links, layouts and arithmetic limits: `docs/fp4-kernels.md`.

- Qwen3.6's official config and installed Transformers Qwen3_5Moe model define
  fused expert tensor shapes, normalized top-8 BF16 routing, ascending expert
  accumulation, and the sigmoid-gated shared expert. Used as a behavior reference;
  the Rust implementation reuses Vade's hybrid and expert kernels.

- The shared-attention integration imports Vade's shared GQA8 attention (8e9657e) and
  two-stage 256x128x64 prefill GEMM/wide QKV (7ae8d3d, 570d94d). Dense QKV/O
  staging is adapted to this engine's existing packed weight fragments, retaining
  one resident BF16 arena. Integration retains the newer m32 c10db29 64x32 warp
  tiles and paired attention loads, adding only packed-weight addressing and
  expert dispatch. These are Vade's own Rust kernels.
- Qwen3.8-27B's official BF16 config and 15 MTP tensor headers define the shared
  embedding/head, fusion and one full-attention layer. The installed official
  Transformers qwen3_5 source ignores mtp.*; it supplies the decoder/norm/RoPE
  precision contract but no MTP forward. Transformers issue 48913 records this
  topology mismatch: https://github.com/huggingface/transformers/issues/48913.
  VeOmni's MTP architecture note supplies embedding-first fusion, final-normalized
  trunk features and unshifted rotary positions:
  https://github.com/bytedance-seed/veomni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/docs/key_features/mtp.md.
  All engine code is original Rust; no stock-engine source was read or copied.
  MTP maker-forward parity remains unverified; canonical target verification is
  the output exactness gate.

- Qwen3 model card: thinking sampling defaults (temperature 0.6, top-p 0.95, top-k 20), non-thinking defaults (0.7, 0.8, 20), and optional presence penalty for repetition. https://huggingface.co/Qwen/Qwen3-8B
- Hugging Face Transformers: standard Qwen3 reference inference, tokenizer and chat template used only to generate comparison fixtures; repetition penalty sign convention. No inference implementation copied into Vade.
- SplitMix64 (Sebastiano Vigna): integer finalizer constants for stateless seeded sampling, freshly implemented in Rust; this is not a cryptographic RNG.
- Counter-based sampling keyed by absolute position supports exact speculative replay (Vade design note).
- TensorFold: conversation state snapshots and independent branches at message boundaries. Implementation here is fresh Rust using the engine's public fork/truncate API.
- Earlier Spark/SGLang experiments: foreground chat preemption between background prefill chunks. Implemented as a bounded single-thread priority scheduler with nonblocking sockets; no external code or crates.
## Kernel references

- Earlier Vade measurements on DGX Spark: avoid
  CUDA copies from mmap-backed pageable checkpoint pages; read into pinned or
  unified memory. This loader uses that idea with freshly authored Rust pread
  code, bounded transient staging and device arenas with no host weight copy.
- NVIDIA CUDA Driver API and CUDA for Tegra documentation: allocation ownership,
  managed/pinned/system-memory access contracts and different caching behavior.
  https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html
  The installed CUDA 13.0 cuda.h is the ABI source. No NVIDIA implementation
  code or third-party loader code is included.
# Ideas consulted

- Qwen's official Qwen3.8-27B checkpoint metadata defines the text tensor names,
  hybrid layer schedule, zero-centered norms, partial RoPE and text
  chat protocol. The adapter and Unicode-mark tokenizer extension are authored
  in Rust and reuse Vade's own Flash delta-net/gated attention and dense BF16
  loader/GEMMs. No Transformers, vLLM, SGLang or TensorFold code is incorporated.
  Native BF16 operation boundaries follow Transformers 5.18.0's Qwen3_5 model:
  BF16 beta and gated norm, SiLU for GDN and sigmoid for full attention, even
  though the checkpoint also carries output_gate_type=swish.
  https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- Qwen3.8-27B-FP8 revision 017b9c7a defines E4M3 weights, BF16 128x128
  block scales and the unconverted module list. Dense FP8 projections reuse
  Vade's own Flash FP8-to-BF16 promotion and tensor-core reductions. The gate
  uses Transformers 5.18.0's official FineGrainedFP8Config(dequantize=True)
  operation: multiply decoded weights by checkpoint scales in FP32, then round
  to BF16. No external implementation code is incorporated.
  https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/finegrained_fp8.py

- TensorFold and vcruz305: keep the complete decode pass and KV cache on device, retain activation buffers, resolve kernels once, and queue launches until the output is needed. Implemented independently in Rust; no project code copied.
- Earlier Vade measurements: prioritize weight bandwidth and remove host synchronization/allocation from decode. This build retains the existing Rust warp32 matvec while moving surrounding operations onto the device.
- The kernel implementation starts from Vade's original Rust matvec/GEMM implementations. Its recorded ideas: TensorFold and jayleaton ROOFLINE (wide coalesced loads and warp-cooperative rows), LLVM NVPTX intrinsic definitions, and NVIDIA PTX MMA fragment-layout specifications. The integration adds three-product, two-component attention following a precision analysis and removes unused variants. No upstream implementation code was copied. Batched scheduling, causal attention and storage integration are original Rust.

- The draft implementation integrates Vade's original Rust draft2 suffix lookup and guarded verification. The prompt lookup and position-keyed sampling ideas are credited to TensorFold/vcruz305; no external model or implementation is used.
- Draft-model decoding uses the standard speculative-decoding idea: a smaller official Qwen3 model proposes a chain, the target verifies consecutive rows, and only its matching prefix commits. Vade extends this to sampled requests: the draft proposes greedily, and the target's own position-keyed sample decides every accepted token and correction, so drafting never changes a sampled output. Proposal generation, KV rollback, tied-weight views and canonical acceptance are independently authored Rust; no stock-engine implementation was copied.
- Vade attention uses FlashAttention's online maximum/denominator recurrence, query tiles and on-chip QK/PV products. NVIDIA's PTX `ldmatrix` mapping informs the V transpose. Retained-FP32 attention extends this to two-component views of retained FP32 KV. Shared-input GEMM and exact verifier staging come from Vade's own measured kernels.
- CUDA virtual memory allocation/mapping ABI is taken from the installed NVIDIA CUDA 13 header. Fixed virtual KV addresses with appended physical pages are independently implemented in Rust; no CUDA sample code is copied.

- Tensor parallelism uses the standard Megatron decomposition: head-shard Q/K/V,
  row-shard gate/up, column-shard O/down and sum in fixed rank order. No Megatron
  implementation code is used. Direct RC SEND/RECV, preposted receives and striped
  physical links are standard verbs techniques; declarations follow rdma-core
  headers. The optional TP packed8 kernel is Vade's original Rust arithmetic with
  unrolled cache-global loads. Only scale-owned TP code is integrated here; engine
  layer hooks use main's own numeric operations and kernels.
- Decode/verify attention uses FlashAttention's online maximum/denominator recurrence and Vade's grouped split mapping: four query heads share K/V reads, and fixed row-local partitions preserve decode/verify arithmetic. The implementation is original Rust.
- Multi-session design: share exact projection rows across unrelated chats while attention and KV stay per session. Multi extends our Rust kernels for mixed decode, prompt and draft rows; priority prompt chunking follows the earlier Spark/SGLang experiments credited above. No external code copied.
- Main4's scheduler follows SGLang mixed-chunk scheduling and per-row attention sizing, and vLLM token-budget scheduling and abort-before-close. Context-timed prefill and capability-scoped cancellation are independently authored Rust; streaming attention and shared KV pages come from Vade's own code.
- Sequence-parallel design: additive ordered layer hooks share immutable weights and exchange FP32 KV between sequence slices. This integration retains the existing Rust arithmetic and leaves transport to the caller.
- Token Recycling (https://arxiv.org/abs/2408.08696): retain token successors from target logits, including rejected rows. Vade drafts linear chains; its greedy HTTP path learns the observed best successor without downloading full logits.
- REST (https://arxiv.org/abs/2311.08252) and CREST (https://arxiv.org/abs/2408.04678): retrieve and rank several suffix continuations. Vade uses bounded, request-local token history.
- Lookahead Decoding (https://arxiv.org/abs/2402.02057): reuse wrong-prefix predictions as guesses in a later verification cycle. Vade shifts already-computed rejected rows without extra target passes. All drafter code is independently authored Rust; no source was copied.
- Long-context attention retains the existing FlashAttention online recurrence,
  with original Rust shared-memory swizzling, register probability fragments,
  and on-chip FP32 splitting. Shared KV prefixes use the installed CUDA header's
  allocation-handle mapping contract; GB10 whole/partial alias behavior was
  measured directly. No external implementation code was used.
- Decode fusions follow Vade's shared-input QKV and gate/up epilogue ideas. These Rust kernels keep separate existing weight allocations and canonical packed8 arithmetic; single-CTA RMS retains the original two-stage reduction tree.

- TensorFold prefill attention, as described in the campaign backlog: one
  block shares K/V across a group of query heads, asynchronously stages
  swizzled tiles, and keeps probabilities in registers at absolute key tiles.
  The original Rust attention kernels now use single BF16 operands and share
  each K/V tile across its query heads; no source was copied.
- Hugging Face finegrained-fp8 v4 ([pinned source](https://huggingface.co/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py#L440)): per-128 activation maximum, raw max/448 scale, and division by max(scale, 1e-12) before E4M3 rounding.
- [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/): native m16n8k32 E4M3 fragment mapping and packed E4M3 conversion operand order.
- FP8 quantization, block-scaled GEMM, shared-memory staging and dispatch are independently authored Rust; no external implementation code was copied.
- Resident-FP8/BF16 GEMM uses NVIDIA PTX's exact E4M3-to-FP16 conversion, packed BF16 multiply, and BF16 MMA. The exponent transfer into the block scale, tile-local promotion, and warp-private decode pipeline are original Rust implementations; no external kernel code was copied.
- Qwen3.8-Flash-Next official checkpoint and Transformers FP8 conversion: E4M3FN values multiplied by promoted BF16 128x128 block scales, and one scalar scale for PLE embeddings. Shape/ownership contract and upstream source links are in docs/history/flash-next-architecture.md. Flash model metadata was adapted from our parked Vade implementation; execution and transport remain authored Rust.
- Flash HC/PLE integration reuses Vade's parked HC/PLE algebra and the existing Vade rank collectives. N-gram hashing, EOS boundaries, four-stream expansion and grouped norms follow the official model source cited in docs/history/flash-next-architecture.md; Rust kernels and bounded resident-row access are authored here.
- Flash GDN integration follows the official Transformers Qwen4Exp chunk64 triangular-solve equations and cached gated-delta recurrence (source pinned in docs/history/flash-next-architecture.md). Head-sharded projections, convolution/cache ownership, fixed-order Rust GPU arithmetic and shared workspace are authored here. The short sublayer snapshot uses hooks around that official CPU implementation, without changing model arithmetic.
- Flash indexed attention follows the official Qwen4Exp main-attention, raw-key pooling, partial RoPE and QSA selection equations cited in docs/history/flash-next-architecture.md. Sharded indexer gathering, causal Rust kernels and layer-local caches are authored here; BF16 projections reuse Vade's GDN kernel. The short oracle adds hooks only, retaining official FP32 arithmetic.
- Flash sharded MLP follows the official Qwen4Exp full-softmax/top-10 router, ascending expert-ID accumulation, shared expert and HC equations cited in docs/history/flash-next-architecture.md. FP8 block products are rounded to BF16 by the official dequantization policy before FP32 promotion. Expert/channel ownership, one-allocation views, Rust kernels and staged rank reduction are authored here; BF16 projection/HC and TCP transport reuse Vade code. No external implementation code copied.
- SGLang's GB10 unified-memory sizing, as recorded in our hardware measurements: use Linux MemAvailable so reclaimable file cache contributes to capacity. The serving implementation is original Rust; no upstream code was copied.
- SGLang RadixAttention and vLLM automatic prefix caching: reuse committed token prefixes independently of client session capabilities. Vade's block-hash index, exact comparisons, API-key scopes, live-prefill sharing and LRU accounting are independently authored Rust on Vade's shared KV pages; no external implementation code was copied.
- Prefix-shared/cascade attention: stage common physical K/V tiles across independent chat queries. This implementation extends Vade's own decode/verify reduction kernel and physical-page aliases; no external implementation code was copied.
# Shared GQA8 attention

The dense/expert GQA8 body folds in Vade's own expert code, commit 59b8a3c:
immediate ldmatrix offsets, swizzled BF16 cache tiles and overlapping cp.async
refills with transient addresses. Base-2 softmax combines the head-dimension
and exponent scales. No external implementation code was copied.

- Prefill tiles use NVIDIA PTX's asynchronous L2-only global-to-shared copies and native BF16 MMA. The 256-row CTA, resident tiled-weight addressing and fused SwiGLU BF16 fragment epilogue are authored here. Bulk-copy candidates were rejected; no external implementation code was copied. Instruction reference: https://docs.nvidia.com/cuda/parallel-thread-execution/ .
- Vade's GQA8 attention design (commit 8e9657e) informs the GQA4 port in 2de67a9: immediate ldmatrix offsets, fewer live QK fragments, and K/V refills after their last readers. GQA4 refills directly from resident BF16 KV, pairs matrix loads, and retains the existing exponential and reduction order.

- Vade gemm r8/r9 supplies the 8B O/down two-stage residual/RMS tile and GQA4 fragment output/refill schedule. The existing shared prefill helper retains main's 32B/expert paths; no external implementation code copied.

- Vade work/m32 c10db29 and work/gemm 8b2a874 informed the two-stage 128x128x64 grouped expert gate/up tile in moe dec2a09. The Rust kernel reuses resident packed weights without repacking; no external implementation code was copied.

- Google DeepMind Gemma 4 (Apache-2.0), official gemma-4-31B-it config,
  tokenizer artifacts and canonical chat template dated 2026-07-09:
  https://huggingface.co/google/gemma-4-31B-it . The template is retained verbatim
  under crates/tokenizer/templates/gemma4.jinja to validate model folders.
  Rust config parsing, shared BPE extension and text formatter are authored here.
  Serving follows the tokenizer config response_template (thought channels and
  raw-string tool arguments); tool dictionaries use Jinja stable case-insensitive
  dictsort semantics. The Rust parser and ordered JSON view are authored here.
  Local Hugging Face Transformers Gemma4TextAttention, Gemma4RMSNorm,
  Gemma4TextDecoderLayer and proportional RoPE were read only to establish
  behavior and compare CPU tokenization/template output; no inference code copied.

- Gemma 4 text GPU arithmetic follows Google's official Transformers Gemma4TextAttention, Gemma4RMSNorm, Gemma4TextDecoderLayer and proportional RoPE behavior (tooling reference only). Rust implementation reuses Vade projection and attention helpers; no reference runtime code is linked.

- Hugging Face Transformers / Google Gemma 4 assistant reference: query-only decoder over the target's last local/global KV, concatenated scaled target embedding and normalized hidden state, pre/post projection recurrence at a fixed position, and full tied head when ordered embeddings are disabled. Implemented in Rust using Vade's shared Gemma layers and strict speculator; no reference code copied.
- Qwen3.8-27B decode reuses Vade's own packed BF16 asynchronous staging and CUDA graph capture/replay from decode commits d4e74a7 and 4b7d54e. Native BF16 gate/up, projection and residual rounding follow the official Transformers precision boundaries already credited above. The cooperative norm preserves Vade's ascending 32-lane reduction; no external kernel code was copied.

- Independent hybrid prompts reuse Vade's own single-session projection/MLP path, per-session Flash convolution/FP32 inverse/solve and causal attention, and shared packed8 decode. Session state arenas, stream-ordered device clears and the persistent FP32 scan are authored here, retaining the existing operation order. Device clear ABI follows NVIDIA's CUDA Driver API; no external implementation code was copied.
- Independent prompt descriptors reuse Vade's own tiled FP32 GDN kernels and cached state arenas. Short prompts preserve separate ascending multiply/add reductions; long prompts retain the incoming fused FP32 core. Native shared decode fuses products of promoted BF16 operands without changing packed8 accumulator boundaries. No external implementation code was copied.
- The projection implementation reuses Vade's own device argmax, short-prompt MMA reduction and vector arenas to remove full prompt-logit readback, pool narrow projections and grow all full-attention layers together. The compact scalar decode loop retains the eight packed8 accumulator streams. No external code or new arithmetic approximation is used.
- The packed8 implementation expands Vade's existing canonical staged packed8 helper to 64 outputs, sharing each staged activation group across eight output groups. Weight layout, BF16 products, accumulator streams and ascending K order remain unchanged; host and kernel use a shared repr(C) launch ABI. Direct-load and alternative scheduling attempts were discarded. No external implementation code was copied.
- Qwen3.8-27B prefill reuses Vade's packed BF16 GEMM and shared online attention at 256-wide heads. Gated-delta chunk equations follow the official Transformers implementation credited above; the parallel block inverse, independent chunk preparation and convolution kernels are original Rust. Recurrent state remains FP32, with the official BF16 projection, activation and gating boundaries.
- Dense hybrid prefill reuses Vade's own m32 two-stage 256x128x64 GEMM and fused SwiGLU/QKV work (7ae8d3d, 570d94d, c10db29), adapted to the existing one-copy packed weights. Its BF16 tensor chunk products, solve and scan retain FP32 accumulation and state. The makers' normalized chunk products and triangular solve operate in FP32. The implementation restores Vade's own parallel FP32 block inverse/solve with ascending FP32 products and scan; BF16 tensor products remain explicit opt-in after failing the native CUDA chat gate. All Rust kernels are authored here; no Transformers implementation code was copied.
- Staging, two-chat warp and pinned-readback experiments reused Vade's own canonical projection and pinned argmax/exchange designs. All performance variants were discarded; the K256 launch precondition is enforced. No external code or approximation was added.
- The FP8 path ports Vade's own FP8 commit f0c36c8: warp-private cached short-attention scores, register-held queries and ordered normalization read-ahead. BF16 activation packing was adapted from the existing prefill operand and evaluated with the shared canonical staged helper; projection experiments were removed. No external implementation code, dependencies or lower-precision operands were added.

- Qwen3 official model card (https://huggingface.co/Qwen/Qwen3-8B): opt-in YaRN factor 4, original_max_position_embeddings 32768, maximum 131072. YaRN frequency ramp and attention scaling follow the published YaRN method and Transformers RoPE parameter definitions (https://huggingface.co/docs/transformers/main/en/internal/rope_utils); the Rust table generation is independently authored, with no upstream code copied.

- Phi-4: official Microsoft BF16 configuration and tokenizer artifacts; fused projection row loading and BF16 boundaries reuse Vade Phi-4 3315d8d and the shared dense kernels. No external runtime code copied.

- Llama 3.1/3.2: Meta official BF16 configs, frequency-dependent Llama 3 RoPE scaling and text chat templates. Rust configuration, rendering and frequency table logic authored here; no external inference implementation copied.
- The long-context path uses Vade's own canonical FP32 chunk scan and BF16 MMA A-fragment packing: both row halves reuse each prior-state load, K staging overlaps intra-chunk output, and the RMS/gate epilogue emits the existing packed operand. SGLang 0.5.21's public FLA chunk decomposition and cuBLAS kernel names informed phase attribution; its narrower dot operands were not adopted. All implementation code remains original Rust/PTX.

- The recurrent scan follows the installed official Transformers 5.18.0 Qwen3.5 chunk-gated-delta-rule arithmetic: FP32 beta-scaled products, pairwise decay, then a unit-lower triangular solve. The independently authored Rust solve retains Vade’s canonical blocked inverse/product order and keeps that inverse on chip (direct substitution was rejected for changing greedy IDs); the scan reads ahead into dead attention storage. The full-attention gate reuses Vade’s existing BF16 operand layout. No external implementation code or NVIDIA libraries enter the engine.

- The normalization follows PyTorch CUDA Reduce.cuh mean arithmetic: four adjacent FP32 accumulators, descending warp reduction and reciprocal-count multiplication. The Rust normalization fixes short-prefill and batch-one decode accumulator layouts, each independent of pooled chat width. Bounded partial-prefill replay reuses Vade state copies and canonical tile execution. No upstream implementation code was copied.

- FP8 activation round follows the official checkpoint's dynamic E4M3,
  per-token 128-element activation groups and BF16 128x128 weight scales.
  The installed Transformers 5.18.0 finegrained_fp8 dispatcher consumes those
  scales through its finegrained FP8 matmul, returning the input BF16 dtype.
  Vade extends its own m16n8k32 Rust/PTX GEMM: BF16-round each input, find
  maxabs in each group, set activation scale=maxabs/448, divide by
  max(scale,1e-12), then convert with round-to-nearest-even/saturate-finite
  E4M3. Four ascending K32 MMAs form an FP32 K128 block dot. Multiply that
  dot by (activation_scale * checkpoint_weight_scale), add to the FP32
  running sum in ascending K128 order, and BF16-round only the output.
  Zero groups have scale zero and contribute zero. Resident weights retain
  checkpoint row-major bytes; no weight dequantization copy is retained.
  The zero/tiny-group scale convention reuses Vade's existing quantizer.
  The reference environment installs Hugging Face kernels 0.17.2 and finegrained-fp8 revision
  8c178950e8710e26a2210e4e909cd02dc16c8715 in the separate reference environment.
  Its inline quantizer uses the same maxabs/448 convention; its matmul source
  scales each K128 dot by the activation scale and then the weight scale.
  Transformers 5.18.0's unanchored exclusion regex incorrectly matches
  `mlp.gate_proj` against `mlp.gate`, discarding 64 checkpoint scale tensors.
  The reference run anchors and escapes exclusion names before loading;
  model forward and official kernels remain unchanged. Raw source hashes
  and both loader results are retained with round evidence. Vade's larger
  token tile, PTX ldmatrix fragment loads and separate decode entry reuse
  the same original Rust helper and K128 sum order. No external code copied.

- Flash Next W8A8 follows Transformers 5.18.0 `FP8Experts` and the same
  finegrained-fp8 revision recorded above. The independently authored grouped
  path reuses Vade's dense K128 helper and quantizer, with BF16 projection,
  SiLU, product and post-collective boundaries. Source provenance, reference
  dispatch differences and remaining FP32 operations are documented in
  `docs/flash-fp8-math.md`; no upstream implementation code was copied.
- OpenAI Codex CLI 0.160.0 (https://github.com/openai/codex): Responses request shapes and semantic SSE event contract were established with local wire captures against a mock and Vade's host scheduler. The stateless Responses-to-chat translation and event encoder are original Rust; no upstream implementation code was copied. Presenting a custom (freeform) tool to the model as a function with one required `input` string follows the JSON form of Codex's own `apply_patch` tool (idea only).

- Qwen3.5-family XML tool calls: typing each `<parameter>` value by the tool's declared JSON schema follows the idea in the Qwen3-Coder tool parsers of vLLM and SGLang. Vade's streaming parser is its own code; no implementation code was copied.

- Long-context decode attention for Qwen3.5-family full-attention layers: loading each cached K/V tile once per key head for all of its query heads (grouped-query sharing) and double-buffering the V tiles into shared memory with cp.async follow widely used decode-attention practice (FlashDecoding, FlashInfer). Vade's kernels are original Rust/PTX and keep the solo walk's sequential softmax and value sums; no implementation code was copied.

- Flash-Next MTP fusion follows the model semantics documented by SGLang's installed `qwen4_exp_mtp.py`: whole-residual normalization, one shared projection per residual stream, and a broadcast projected token embedding. Vade's implementation is original Rust using its own sharded weights, attention and MoE kernels; no upstream code was copied.

- NVIDIA Model Optimizer and the official [Qwen3.8-Flash-Next-NVFP4 model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): mixed NVFP4 main experts, FP8 MTP/PLE and BF16 dense storage; E2M1 pairs with E4M3 scales per 16 values and an FP32 global scale. The packed loader and CPU dequantization are original std-only Rust; no upstream implementation code was copied.
- NVFP4 expert storage follows the official `nvidia/Qwen3.8-Flash-Next-NVFP4`
  checkpoint headers and quantization metadata: packed E2M1, E4M3 scales per
  sixteen values, and a global FP32 scale. Vade's W4A16 register dequantization,
  lane redistribution and canonical BF16 MMA path are original Rust/PTX. This
  explicit operand mode does not implement the checkpoint's W4A4 quantizer.

- TensorFold's per-stream state-pointer tables informed batching independent Flash-Next GDN recurrences. Vade extends its own Rust kernels and keeps each session's original arithmetic and private state; no upstream code was copied.

- vLLM issue #59985, shows the failure class of a tokens == 1 kernel special case breaking batch invariance. Vade routes its indexed attention and MoE router/shared-expert projections through one kernel per width and gates one-row against 4-, 8- and 16-row launches; the kernels and the gate are original Rust; no vLLM code was copied.
