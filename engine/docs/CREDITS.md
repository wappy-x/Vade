# Credits

**This page names the people, papers and projects whose ideas or reference behavior Vade uses, and the part of Vade that uses each one.** No source code was copied from any of them. One file follows a reference closely: the FP32 CPU reference [refcpu.rs](../crates/model/src/flash/refcpu.rs) computes the formulas of the Transformers file `modeling_qwen4_exp.py` (revision `b0cdc61`) in the same order and cites its lines ([Reference code](#reference-code)). Vade's code is written here, in Rust; the parts designed and built in this project are listed under [Original work](../../CREDITS.md#original-work).

A credit does not mean their code runs in Vade, or that their speed or correctness claims apply to it. Every profile link was checked against the person's own GitHub or Hugging Face profile, as [Link checks](link-checks.md) records.

**On this page:** [Model makers](#model-makers) · [Reference code](#reference-code) · [Attention](#attention-and-determinism) · [Serving](#serving-many-requests) · [Speculative decoding](#speculative-decoding) · [Building blocks](#model-building-blocks) · [Spark community](#dgx-spark-community) · [NVIDIA and transport](#nvidia-and-transport) · [Compiler and standards](#compiler-and-standards) · [Studied, not used](#studied-not-used) · [Reused data](#reused-data) · [Corrections](#corrections)

<details>
<summary><b>Terms on this page</b>: KV cache, tile, main, research engine; how entries are written</summary>

<dl>
<dt>KV cache</dt>
<dd>The attention keys and values kept for every token already read: the model's memory of the conversation.</dd>
<dt>Tile</dt>
<dd>A small block of a larger matrix, sized to fit in fast on-chip memory.</dd>
<dt>Main</dt>
<dd>The merged code on the main branch. Work outside main is labelled <b>development only</b>.</dd>
<dt>Research engine</dt>
<dd>The separate two-Spark engine for Qwen3.8-Flash-Next-FP8. The chat server cannot load it.</dd>
</dl>

- Vade's own code covers the engine, the GPU kernels, the tokenizer, the Spark-to-Spark transport, the server and the tools.
- Each entry says which idea or behavior Vade uses. Where a second line follows, it names the people or gives the links.
- Papers list every author when there are nine or fewer. Longer lists show the first three names and the total; the paper link has them all.
- Paper authors' profile links sit in a fold at the end of their section. An X account or website appears only when the person's own profile lists it; an author missing from the section's profile fold is credited by name and paper only.

</details>

## Model makers

**Vade runs official releases only, so these teams define what it must compute.**

- **Qwen team, Alibaba Cloud.** Model definitions, official BF16 and FP8 weights and scales, tokenizers and chat templates.\
  [GitHub](https://github.com/QwenLM) · [Hugging Face](https://huggingface.co/Qwen) · [X](https://x.com/Alibaba_Qwen) · [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- **Google DeepMind, Gemma team.** Main's Gemma text model: local and global attention, proportional rotary positions, normalization, tokenizer and the canonical chat template.\
  [GitHub](https://github.com/google-deepmind) · [Hugging Face](https://huggingface.co/google) · [Site](https://deepmind.google) · [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it)
- **OpenAI, gpt-oss team.** Development only: the official MXFP4 checkpoint, model equations and the Harmony conversation format.\
  [GitHub](https://github.com/openai) · [Hugging Face](https://huggingface.co/openai) · [Site](https://openai.com/) · [gpt-oss](https://github.com/openai/gpt-oss) · [Harmony](https://github.com/openai/harmony)
- **Mistral AI.** Development only: Mistral Small 3.2's model contract, Tekken tokenizer and instruction format.\
  [GitHub](https://github.com/mistralai) · [Hugging Face](https://huggingface.co/mistralai) · [Site](https://mistral.ai) · [mistral-common](https://github.com/mistralai/mistral-common)
- **Microsoft, Phi team.** Development only: the official BF16 checkpoint, architecture, tokenizer and chat format.\
  [GitHub](https://github.com/microsoft) · [Hugging Face](https://huggingface.co/microsoft) · [X](https://x.com/OpenAtMicrosoft) · [Phi-4](https://huggingface.co/microsoft/phi-4)

<details>
<summary><b>Models in detail</b>: what main serves, the draft models, Qwen's sampling defaults</summary>

- Main serves Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B and Qwen3.8-27B (BF16 and official FP8), and Gemma 4 31B in BF16. Gemma 4 26B-A4B is development only.
- Draft models: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) for Qwen3.8-27B; Qwen3-0.6B for Qwen3-8B and Qwen3-32B. Gemma's official assistant checkpoint defines the draft Vade uses for one active request.
- The [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) sets Vade's default top-p and top-k, and the optional YaRN settings (factor 4, 32,768 original positions).
- The research engine runs [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13).

</details>

## Reference code

**Vade checks its answers against the makers' official code.** The agreement is measured and is not perfect; [Validation](validation.md) has the counts. Determinism is a separate promise: the same request gives the same answer, alone or batched, cached or fresh.

Hugging Face: [GitHub](https://github.com/huggingface) · [Hugging Face](https://huggingface.co/huggingface) · [X](https://x.com/huggingface) · [Site](https://huggingface.co/)

- **[Transformers](https://github.com/huggingface/transformers) contributors and model authors.** The official reference behind Vade's saved CPU and CUDA fixtures, the repetition-penalty direction, and the exact order and rounding of each operation. Reference tooling only; never linked into Vade.
- **[Tokenizers](https://github.com/huggingface/tokenizers) contributors.** The `tokenizer.json` format, the reference tokenization, and the Unicode 12.1 behavior of version 0.23.2 that Vade's tokenizer reproduces for Qwen3.
- **[safetensors](https://huggingface.co/docs/safetensors) and [huggingface_hub](https://github.com/huggingface/huggingface_hub) contributors.** The checkpoint file and index format, read by Vade's own bounded reader, and the official download tooling.

Other reference tools:

- **PyTorch contributors.** Runs the official reference models that produce the saved fixtures. Reference tooling only.\
  [GitHub](https://github.com/pytorch) · [Hugging Face](https://huggingface.co/pytorch) · [Site](https://pytorch.org) · [PyTorch](https://github.com/pytorch/pytorch)
- **Python and NumPy contributors.** The privately retained reference-data generators that produced the saved fixtures. Kept outside this repository; never run in the engine.\
  [python.org](https://www.python.org/) · [numpy.org](https://numpy.org)
- **ByteDance Seed and the VeOmni contributors.** Embedding-first fusion, final-normalized target features and unshifted rotary positions in the multi-token prediction (MTP) draft head for Qwen3.8-27B. Agreement with a maker MTP forward is unverified; the target model checks every drafted token.\
  [GitHub](https://github.com/ByteDance-Seed) · [Hugging Face](https://huggingface.co/ByteDance-Seed) · [Site](https://seed.bytedance.com/) · [VeOmni MTP note](https://github.com/bytedance-seed/veomni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/docs/key_features/mtp.md)
- **kernels-community.** Experimental FP8 activation path only: a per-128 maximum, a scale of maximum / 448, and guarded division before E4M3 rounding. The default keeps BF16 activations.\
  [Hugging Face](https://huggingface.co/kernels-community) · [finegrained-fp8 recipe](https://huggingface.co/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py#L440)

<details>
<summary><b>Pinned sources</b>: the exact versions that settled each detail</summary>

Model code:

- [Transformers Qwen3, v4.51.3](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py): Qwen3 layer behavior used for the earlier fixtures.
- [Transformers Qwen3.5 text, 5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py): for Qwen3.8-27B, the hybrid layer schedule, zero-centered norms, partial rotary positions, BF16 gates, and the linear-attention recurrence with its FP32 chunk equations.
- [Transformers Flash Next](https://github.com/huggingface/transformers/blob/9d8c3a24d434d4a958a23562f576126924358b7f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py): research engine only. Hyper-connections, n-gram and per-layer embeddings, the chunk-64 delta rule, indexed attention and pooling, full-softmax top-10 expert routing, ascending expert accumulation and scaled BF16 weight conversion.

Precision and settings:

- [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), revision `017b9c7a` as recorded at download: E4M3 weights, BF16 128 × 128 block scales, and the list of modules left unconverted.
- [Transformers fine-grained FP8, 5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/finegrained_fp8.py): decode E4M3 weights, multiply by the checkpoint's scales in FP32, then round to BF16. FP8 weights stay resident.
- [Transformers issue 48913](https://github.com/huggingface/transformers/issues/48913): records that the installed reference forward ignores Qwen3.8-27B's built-in MTP tensors.
- [Transformers RoPE definitions](https://huggingface.co/docs/transformers/main/en/internal/rope_utils): the YaRN parameters behind Vade's optional Qwen3 context extension.

</details>

## Attention and determinism

**Each kernel keeps one fixed order of additions, so the same prompt always gives the same answer.** These papers are where the attention algorithms come from.

Attention:

- **[FlashAttention](https://arxiv.org/abs/2205.14135).** Attention in tiles with a running maximum and running total, instead of a full score matrix.\
  Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691).** Splitting attention work across GPU blocks and warps, used to organize prefill attention.\
  Tri Dao
- **[FlashAttention-3](https://arxiv.org/abs/2407.08608).** Overlapping background data movement with matrix work, applied with GB10 instructions. No Hopper kernels imported; no lower precision enabled by default.\
  Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani and Tri Dao
- **[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).** The running softmax total that tiled attention is built on.\
  Maxim Milakov and Natalia Gimelshein
- **[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).** Splitting a long KV cache into parts, summarizing each, then merging them.\
  Tri Dao, Daniel Haziza, Francisco Massa and Grigory Sizov

Determinism:

- **[Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).** Batch-invariant arithmetic: sums that do not change when the batch changes. Vade keeps one order for alone, batched, verification and cached paths.\
  Horace He and Thinking Machines Lab

<details>
<summary><b>Author profiles</b>: GitHub, Hugging Face, X and websites for the people above</summary>

**Tri Dao**: [GitHub](https://github.com/tridao) · [Hugging Face](https://huggingface.co/tridao) · [Site](https://tridao.me)\
**Daniel Y. Fu**: [GitHub](https://github.com/DanFu09) · [Hugging Face](https://huggingface.co/danfu09) · [X](https://x.com/realDanFu) · [Site](https://www.danfu.org)\
**Vijay Thakkar**: [GitHub](https://github.com/thakkarV)\
**Maxim Milakov**: [GitHub](https://github.com/milakov) · [Site](https://milakov.org)\
**Natalia Gimelshein**: [GitHub](https://github.com/ngimel)\
**Francisco Massa**: [GitHub](https://github.com/fmassa)\
**Grigory Sizov**: [GitHub](https://github.com/sgrigory)\
**Horace He**: [GitHub](https://github.com/Chillee) · [Hugging Face](https://huggingface.co/Chillee) · [X](https://x.com/cHHillee) · [Site](http://horace.io/)\
**Thinking Machines Lab**: [GitHub](https://github.com/thinking-machines-lab) · [Site](https://thinkingmachines.ai)

</details>

## Serving many requests

**Continuous batching, shared memory pages, prefix reuse and scheduling, plus the rival engines in the [benchmark tables](benchmarks.md).**

Scheduling and memory:

- **[Orca](https://www.usenix.org/conference/osdi22/presentation/yu).** Continuous batching: requests join and leave between generation steps.\
  Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim and Byung-Gon Chun
- **[PagedAttention](https://arxiv.org/abs/2309.06180).** KV memory in shared, read-only pages, copied only when a request diverges. Vade maps pages with CUDA virtual memory instead of block tables.\
  Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang and Ion Stoica
- **[vLLM](https://github.com/vllm-project/vllm).** From v0.30.0's [scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py), [cancellation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py) and [block pool](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py): a token budget per step, stopping a cancelled request's work before closing it, and automatic prefix caching that finds reusable blocks by hash. Benchmarked at 0.31.0 and as NGC vLLM 25.09 (vLLM v0.10.1.1).\
  vLLM team: [GitHub](https://github.com/vllm-project) · [Hugging Face](https://huggingface.co/vllm-project) · [Docs](https://docs.vllm.ai)
- **[SGLang](https://github.com/sgl-project/sglang).** Mixing prompt chunks with decode steps, sizing attention per request, and sizing GB10 memory from Linux `MemAvailable`. The project's earlier SGLang work on Sparks produced the rule that a live request runs between chunks of a long background prefill. Benchmarked at 0.5.21.\
  SGLang team: [GitHub](https://github.com/sgl-project) · [Docs](https://docs.sglang.ai)
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).** Design references for sharing KV blocks, replaying CUDA graphs over fixed buffers, and mixing prefill and decode work, from its [KV cache notes](https://github.com/NVIDIA/TensorRT-LLM/blob/a81da8a5a8380c4ec8c2d3eafabb8cfc55d7ad56/docs/source/features/kvcache.md). Benchmarked at 1.3.0rc13.\
  NVIDIA TensorRT-LLM team: [GitHub](https://github.com/NVIDIA)

Shared prompt beginnings:

- **[SGLang paper](https://arxiv.org/abs/2312.07104).** RadixAttention: automatic reuse of any matching prompt beginning.\
  Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie and others (12 authors)
- **[FlashInfer paper](https://arxiv.org/abs/2501.01005), [project](https://github.com/flashinfer-ai/flashinfer), [cascade attention](https://flashinfer.ai/2024/02/02/cascade-inference.html).** Reading KV tiles that many requests share only once. Vade keeps each request's own fixed order of sums.\
  Zihao Ye, Lequn Chen, Ruihang Lai and others (11 authors); FlashInfer team
- **[Hydragen](https://arxiv.org/abs/2402.05099).** Shared-prefix attention: avoid rereading a common prompt beginning for each request.\
  Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christopher Ré and Azalia Mirhoseini; Scaling Intelligence Lab

<details>
<summary><b>Author profiles</b>: GitHub, Hugging Face, X and websites for the people above</summary>

**Gyeong-In Yu**: [GitHub](https://github.com/gyeongin) · [Site](https://gyeongin.github.io)\
**Woosuk Kwon**: [GitHub](https://github.com/WoosukKwon) · [X](https://x.com/woosuk_k) · [Site](https://woosuk.me)\
**Zhuohan Li**: [GitHub](https://github.com/zhuohan123) · [X](https://x.com/zhuohan123) · [Site](https://zhuohan.li)\
**Siyuan Zhuang**: [GitHub](https://github.com/suquark)\
**Ying Sheng**: [GitHub](https://github.com/Ying1123)\
**Lianmin Zheng**: [GitHub](https://github.com/merrymercy) · [Hugging Face](https://huggingface.co/lmzheng) · [X](https://x.com/lm_zheng) · [Site](http://lmzheng.net)\
**Cody Hao Yu**: [GitHub](https://github.com/comaniac) · [Site](https://comaniac.github.io)\
**Joseph E. Gonzalez**: [GitHub](https://github.com/jegonzal) · [X](https://x.com/profjoeyg) · [Site](http://eecs.berkeley.edu/~jegonzal)\
**Hao Zhang**: [GitHub](https://github.com/zhisbug) · [X](https://x.com/haozhangml) · [Site](https://haozhang.ai/)\
**Ion Stoica**: [GitHub](https://github.com/istoica) · [Site](http://www.cs.berkeley.edu/~istoica/)\
**Liangsheng Yin**: [GitHub](https://github.com/hnyls2002) · [X](https://x.com/lsyincs)\
**Zhiqiang Xie**: [GitHub](https://github.com/xiezhq-hermann) · [X](https://x.com/Zhiqiang_Xie) · [Site](https://zhiqiangxie.com)\
**Zihao Ye**: [GitHub](https://github.com/yzh119) · [Site](https://expye.com/about)\
**Lequn Chen**: [GitHub](https://github.com/abcdabcd987) · [Site](https://abcdabcd987.com)\
**Ruihang Lai**: [GitHub](https://github.com/MasterJH5574) · [X](https://x.com/ruihanglai) · [Site](https://ruihanglai.com)\
**FlashInfer team**: [GitHub](https://github.com/flashinfer-ai) · [Site](https://flashinfer.ai/)\
**Daniel Y. Fu**: [GitHub](https://github.com/DanFu09) · [Hugging Face](https://huggingface.co/danfu09) · [X](https://x.com/realDanFu) · [Site](https://www.danfu.org)\
**Scaling Intelligence Lab**: [GitHub](https://github.com/ScalingIntelligence) · [Site](https://scalingintelligence.stanford.edu/)

</details>

## Speculative decoding

**A cheap guess proposes several tokens, and the full model checks them in one pass, keeping only the ones it would have chosen itself.** Vade's flags call it drafting.

Draft and verify:

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).** Draft-model guesses checked in one pass of the full model.\
  Yaniv Leviathan, Matan Kalman and Yossi Matias
- **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318).** The verification principle. Vade's default promise is exact greedy output; this credit does not claim the paper's sampling algorithm is implemented.\
  Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre and John Jumper
- **[SpecInfer](https://arxiv.org/abs/2305.09781).** Checking a tree of guesses in one pass, each guess seeing only its ancestors. Development only: draft trees live on unmerged branches, and main drafts single chains.\
  Xupeng Miao, Gabriele Oliaro, Zhihao Zhang and others (15 authors); FlexFlow project
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).** Multi-token prediction as a draft mechanism. Qwen3.8-27B's actual MTP layout comes from Qwen's checkpoint and the [reference code](#reference-code).\
  DeepSeek-AI: [GitHub](https://github.com/deepseek-ai) · [Hugging Face](https://huggingface.co/deepseek-ai) · [Site](https://www.deepseek.com/)
- **[vLLM Ascend pull request 7374](https://github.com/vllm-project/vllm-ascend/pull/7374).** Published Qwen3-8B with Qwen3-0.6B acceptance numbers, which made a small official Qwen3 model the first draft-model choice. Their speeds are not Vade measurements.\
  KlyzhenkoVadim and the vLLM Ascend contributors

Guessing without a draft model:

- **[Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding).** Guessing continuations from text already in the prompt. This is the original published source; we first saw the idea in TensorFold.\
  Apoorv Saxena
- **[Token Recycling](https://arxiv.org/abs/2408.08696).** Keeping the model's predictions, even from rejected rows, as future guesses. Vade keeps the best follower of each token.\
  Xianzhen Luo, Yixuan Wang, Qingfu Zhu, Zhiming Zhang, Xuanyu Zhang, Qing Yang and Dongliang Xu
- **[REST](https://arxiv.org/abs/2311.08252).** Guessing from how earlier text continued. Vade looks only in the request's own recent history.\
  Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee and Di He
- **[CREST](https://arxiv.org/abs/2408.04678).** Ranking several retrieved continuations. Tried and removed; bounded suffix lookup remains.\
  Sophia Ho, Jinsol Park and Patrick Wang
- **[Lookahead Decoding](https://arxiv.org/abs/2402.02057).** Reusing predictions computed after a wrong guess, by shifting rows already computed.\
  Yichao Fu, Peter Bailis, Ion Stoica and Hao Zhang

Choosing how far ahead to guess:

- **[DS41RT draft policy](https://github.com/tpurtell/ds41rt/blob/main/docs/dspark-bandwidth-policy.md).** Choosing the guess depth from measured cost.\
  T.J. Purtell
- **[Flash Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe).** Changing the guess depth according to whether guessing is paying off.\
  Victor Cruz

<details>
<summary><b>Author profiles</b>: GitHub, Hugging Face, X and websites for the people above</summary>

**Gabriele Oliaro**: [GitHub](https://github.com/goliaro)\
**FlexFlow project**: [GitHub](https://github.com/flexflow) · [Site](https://flexflow.ai/)\
**KlyzhenkoVadim**: [GitHub](https://github.com/KlyzhenkoVadim)\
**Apoorv Saxena**: [GitHub](https://github.com/apoorvumang) · [Hugging Face](https://huggingface.co/apoorvumang) · [X](https://x.com/apoorv_umang) · [Site](https://apoorvumang.github.io)\
**Xianzhen Luo**: [GitHub](https://github.com/Luowaterbi) · [Site](https://luowaterbi.github.io)\
**Zhenyu He**: [GitHub](https://github.com/zhenyuhe00) · [Site](https://zhenyuhe00.github.io)\
**Tianle Cai**: [GitHub](https://github.com/ctlllll) · [X](https://x.com/tianle_cai) · [Site](https://tianle.website)\
**Yichao Fu**: [GitHub](https://github.com/Viol2000) · [X](https://x.com/fuyichao123) · [Site](https://viol2000.github.io/)\
**Peter Bailis**: [GitHub](https://github.com/pbailis) · [X](https://x.com/pbailis) · [Site](http://bailis.org)\
**Ion Stoica**: [GitHub](https://github.com/istoica) · [Site](http://www.cs.berkeley.edu/~istoica/)\
**Hao Zhang**: [GitHub](https://github.com/zhisbug) · [X](https://x.com/haozhangml) · [Site](https://haozhang.ai/)\
**T.J. Purtell**: [GitHub](https://github.com/tpurtell)\
**Victor Cruz**: [GitHub](https://github.com/vcruz305) · [Hugging Face](https://huggingface.co/vcruz305) · [X](https://x.com/ViC305)

</details>

## Model building blocks

**Vade computes these because the official models are built from them.** The makers' reference code settles the exact operation order and precision.

Transformer layers:

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762).** The Transformer: attention and feed-forward layers.\
  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin
- **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).** RMSNorm: scaling a layer's values by their root mean square.\
  Biao Zhang and Rico Sennrich
- **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).** SwiGLU, the gated feed-forward activation.\
  Noam Shazeer
- **[GQA](https://arxiv.org/abs/2305.13245).** Grouped-query attention: several query heads share one KV head.\
  Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón and Sumit Sanghai
- **[Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538).** Expert layers: each token uses a few of many small networks. Qwen3-30B-A3B is built this way.\
  Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton and Jeff Dean

Positions:

- **[RoFormer](https://arxiv.org/abs/2104.09864).** Rotary position embeddings (RoPE): rotate query and key values by token position.\
  Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen and Yunfeng Liu
- **[YaRN](https://arxiv.org/abs/2309.00071).** The optional Qwen3 context extension: frequency ramp and attention scaling.\
  Bowen Peng, Jeffrey Quesnelle, Honglu Fan and Enrico Shippole; Nous Research
- **[Qwen2-VL](https://arxiv.org/abs/2409.12191).** Research engine only: the multimodal rotary positions behind Flash Next's text positions. Not multimodal serving.\
  Peng Wang, Shuai Bai, Sinan Tan and others (19 authors)

Linear attention and gates, used by the Qwen3.8 models:

- **[Gated Delta Networks](https://arxiv.org/abs/2412.06464).** The gated recurrent linear-attention layers in the Qwen3.8 models.\
  Songlin Yang, Jan Kautz and Ali Hatamizadeh
- **[Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484).** The chunked delta rule and its triangular solve.\
  Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen and Yoon Kim
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).** The algorithm behind the official Transformers delta-rule code. Vade writes its own Rust kernels.\
  Songlin Yang, Yu Zhang and the flash-linear-attention contributors
- **[Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708).** The output gate on Qwen3.8-27B's full-attention layers.\
  Zihan Qiu, Zekun Wang, Bo Zheng and others (13 authors)
- **[Hyper-Connections](https://arxiv.org/abs/2409.19606).** Research engine only: the parallel residual streams in Flash Next.\
  Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min and Xun Zhou

Tokens, number formats and sampling:

- **[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909).** Byte-pair encoding (BPE): tokenizing by ranked pair merges, in Vade's own merge engine.\
  Rico Sennrich, Barry Haddow and Alexandra Birch
- **[GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-2 code](https://github.com/openai/gpt-2), [tiktoken](https://github.com/openai/tiktoken).** Byte-level tokenization used by Qwen: every byte maps to a printable character before merging. Qwen's pre-tokenizer pattern is a close variant of one tiktoken publishes.\
  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever; OpenAI
- **[OCP microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537).** Development only: MXFP4 values with shared block exponents in the official gpt-oss checkpoint. No extra quantization is added or enabled.\
  Open Compute Project contributors; Bita Darvish Rouhani, Ritchie Zhao and others (33 authors)
- **[Fast splittable pseudorandom number generators](https://doi.org/10.1145/2660193.2660195), [SplitMix64 reference](https://prng.di.unimi.it/splitmix64.c).** SplitMix64's mixing constants turn a seed and a token position into a sampling value, so a guess and its check draw the same number. Written in Rust; not for cryptography.\
  Guy L. Steele, Doug Lea and Christine H. Flood; reference code by Sebastiano Vigna

<details>
<summary><b>Author profiles</b>: GitHub, Hugging Face, X and websites for the people above</summary>

**Łukasz Kaiser**: [GitHub](https://github.com/lukaszkaiser)\
**Biao Zhang**: [GitHub](https://github.com/bzhangGo)\
**Rico Sennrich**: [GitHub](https://github.com/rsennrich)\
**Jianlin Su**: [GitHub](https://github.com/bojone) · [Site](https://kexue.fm)\
**Bowen Peng**: [Hugging Face](https://huggingface.co/bloc97)\
**Jeffrey Quesnelle**: [GitHub](https://github.com/jquesnelle) · [X](https://x.com/theemozilla) · [Site](http://jeffq.com)\
**Honglu Fan**: [GitHub](https://github.com/honglu2875) · [Site](https://honglu.fan)\
**Enrico Shippole**: [GitHub](https://github.com/conceptofmind) · [Hugging Face](https://huggingface.co/conceptofmind) · [X](https://x.com/EnricoShippole) · [Site](https://www.teraflopai.com/)\
**Nous Research**: [GitHub](https://github.com/NousResearch) · [Hugging Face](https://huggingface.co/NousResearch) · [X](https://x.com/nousresearch) · [Site](https://nousresearch.com)\
**Songlin Yang**: [GitHub](https://github.com/sustcsonglin) · [X](https://x.com/SonglinYang4) · [Site](https://sustcsonglin.github.io/)\
**Ali Hatamizadeh**: [GitHub](https://github.com/ahatamiz) · [X](https://x.com/ahatamiz1) · [Site](https://ahatamiz.github.io/)\
**Bailin Wang**: [GitHub](https://github.com/berlino) · [X](https://x.com/bailin_28) · [Site](https://berlino.github.io/)\
**Yu Zhang**: [GitHub](https://github.com/yzhangcs) · [X](https://x.com/yzhang_cs) · [Site](https://yzhang.site)\
**Yikang Shen**: [GitHub](https://github.com/yikangshen)\
**Yoon Kim**: [GitHub](https://github.com/yoonkim) · [Site](https://people.csail.mit.edu/yoonkim/)\
**flash-linear-attention contributors**: [GitHub](https://github.com/fla-org) · [Hugging Face](https://huggingface.co/fla-hub)\
**Barry Haddow**: [GitHub](https://github.com/bhaddow)\
**OpenAI**: [GitHub](https://github.com/openai)\
**Sebastiano Vigna**: [GitHub](https://github.com/vigna) · [Site](http://vigna.di.unimi.it/)

</details>

## DGX Spark community

**These people and projects published their work on the same hardware. Vade uses some of their ideas, never their code.**

**[TensorFold](https://github.com/ashhart/TensorFold), by Ash Hart and the TensorFold contributors.** We studied eleven of its ideas; the fold at the end of this section lists them. Vade's kernel comments and build notes name TensorFold where one of these ideas is used, and each implementation is Vade's own Rust.\
Ash Hart: [GitHub](https://github.com/ashhart) · [X](https://x.com/ashxhart)

Engines and recipes:

- **Mia, Mia's AI Lab.** Every rank reaches transport decisions from shared state; BF16 KV cache experiments; fetching the next weights while waiting.\
  [GitHub](https://github.com/MiaAI-Lab) · [X](https://x.com/MiaAI_lab) · [TensorFold fork](https://github.com/MiaAI-Lab/TensorFold) · [transport deadlock fix](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/92bf731c3aac61927726ef0c422b21b42111f2c4/patches/0040-glm-parallel-deadlocks.patch)
- **Victor Cruz.** Keeping the decode loop on the GPU with few readbacks, and optional placement of the launch thread on a chosen CPU core.\
  [GitHub](https://github.com/vcruz305) · [Hugging Face](https://huggingface.co/vcruz305) · [X](https://x.com/ViC305) · [Flash Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe)
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, and [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by The Royal Lab.** The runtimes Victor Cruz's recipe builds on; their contributors are credited as part of that chain. Vade uses neither.\
  turboderp: [GitHub](https://github.com/turboderp-org) · [X](https://x.com/turboderp_)\
  The Royal Lab: [GitHub](https://github.com/theroyallab)
- **Local Inference Lab.** Clear ownership of every scratch buffer, testing small RDMA messages on their own, and keeping correctness evidence separate from speed measurements.\
  [GitHub](https://github.com/local-inference-lab) · [b12x](https://github.com/local-inference-lab/b12x)
- **Kindling AI.** Pinning exact software versions and recording system state. Vade adopts none of its OS or memory changes.\
  [GitHub](https://github.com/kindlingai) · [Site](https://kindlingai.dev) · [Kindling Spark OS](https://github.com/kindlingai/kindling-spark-os)

Transport and measurement:

- **JJ Eaton.** Measure the Spark's memory bandwidth first and judge every kernel against it; wide neighbouring reads with one warp per row block; the cost of small messages in two-Spark transport.\
  [GitHub](https://github.com/jayleaton) · [X](https://x.com/jayleaton) · [Site](https://www.jayleaton.com/) · [Spark roofline analysis](https://github.com/jayleaton/glm53-tensorfold-spark/blob/main/docs/ROOFLINE.md)
- **hughmadden, Turquoise Bay AI.** RDMA from Rust needs no outside crates; sums across machines in a fixed order.\
  [GitHub](https://github.com/hughmadden) · [glm53f-afd](https://github.com/hughmadden/glm53f-afd)
- **T.J. Purtell.** RDMA receive slots posted in advance, and polling for completions; both are also standard verbs practice.\
  [GitHub](https://github.com/tpurtell) · [DS41RT](https://github.com/tpurtell/ds41rt)
- **Jürgen Schmied.** Evidence that storing recurrent state in reduced precision changes predictions, which supports keeping Gated DeltaNet state in FP32.\
  [GitHub](https://github.com/jschmied) · [GB10 recurrence measurements](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/what-generalises.md)
- **Artem Matskevych.** Which points in a conversation a prefix cache can really resume from.\
  [GitHub](https://github.com/Reederey87) · [X](https://x.com/Reederey) · [Cache-tail analysis](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/blob/fd87969cb43a5ba66573bd2508bc05651fc1579c/docs/17-apc-tail-floor.md)

From the NVIDIA developer forum. Its profile pages refuse automated requests, so these members are credited by name only, with their posts.

- **[ConnectX-7 link capped at 13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461)**, by Ank-Chy, andrewc_actual, mashie and ccatzin. Evidence for diagnosing degraded link states, and why link health must be checked before any transport measurement. No system-change procedure is part of Vade.
- **[Direct QSFP link at 13-16 Gbps](https://forums.developer.nvidia.com/t/dgx-spark-direct-qsfp-connection-only-getting-13-16-gbps-instead-of-expected-200g-performance/370035)**, by ugurbas and robert287. Independent evidence about hot-plug and low-power link behavior.
- **[DGX Spark / GB10 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344)**, by NVIDIA forum staff. Hardware limits, including that the network card cannot reach GPU memory directly.
- **[GB10 NVFP4 thread](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618)**, by adg1, with a device-query result from pontostroy, and **[nvfp4bench](https://github.com/secYOUre/nvfp4bench)** by Alfonso De Gregorio, secYOUre. The GB10's reported 24 MB L2 cache size, used in Vade's research on cache hints for weight reads.\
  secYOUre: [GitHub](https://github.com/secYOUre) · [Site](https://www.pwnshow.com/)

Christopher Owen and FujitsuPolycom are credited under [Studied, not used](#studied-not-used).

<details>
<summary><b>Ideas studied in TensorFold</b>: eleven ideas, in three groups</summary>

On the GPU:

- The whole decode step stays on the GPU: KV cache, activations and buffers stay resident.
- Kernels are looked up once, and launches are queued until a result is needed.
- Prefill attention where one GPU block serves every query head that shares a KV head, with staged, swizzled tiles. Swizzled shared-memory tiles themselves are older practice from CUTLASS and GPU textbooks ([NVIDIA and transport](#nvidia-and-transport)).
- Wide, neighbouring memory reads for matrix-vector kernels.
- Computing output scores only for the final prompt position ([source](https://github.com/ashhart/TensorFold/blob/17c73e189f5e6a5304cda7ea37f086f9c49b4788/src/tensorfold/families/qwen4_exp/cuda/decode.py)).

Drafting, sampling and conversations:

- Exact draft verification: keep only guesses the model itself would choose (original source: the papers under [Speculative decoding](#speculative-decoding)).
- Guessing from text already in the prompt (original source: Apoorv Saxena, under [Speculative decoding](#speculative-decoding)).
- Sampling keyed by seed and token position.
- Conversation snapshots at message boundaries that can branch.

Across Sparks, and testing:

- Sums across Sparks in a fixed rank order.
- Evaluation habits: comparing against plain one-at-a-time runs, bounded draft work and per-stream state.

</details>

## NVIDIA and transport

- **Vade GEMM review (MoE locality recommendation).** Routed FP8 expert launches put row planes next to each other within an expert/output tile to encourage L2 weight reuse. Only launch indexing changes; no source code was copied and CUDA scheduling does not guarantee reuse.

**NVIDIA's documentation and libraries, studied as references, and the papers behind the multi-Spark transport.**

NVIDIA CUDA documentation teams and the NVIDIA DGX Spark team: [GitHub](https://github.com/NVIDIA) · [Hugging Face](https://huggingface.co/nvidia) · [Site](https://nvidia.com)

- **[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).** Instruction and register layouts, tensor-core fragments (including FP8), BF16 and FP8 conversion, and background copies. Vade's small inline PTX blocks follow it.
- **[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [virtual memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html), [graphs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html).** Loading kernels, managing memory, a growable KV cache at a fixed address, shared page mappings, and replaying recorded decode steps. The installed CUDA 13 `cuda.h` header defines Vade's interface declarations.
- **[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html), [graph launch measurements](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/).** How CPU and GPU share memory and caches on one chip, and measured graph launch costs.
- **DGX Spark team: [Clustering guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html), [porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html), [GPUDirect RDMA answer](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark).** GB10 unified memory, link limits, and staging through pinned CPU memory because the network card cannot reach GPU memory.

Libraries and papers:

- **[CUTLASS](https://github.com/NVIDIA/cutlass), [Blackwell notes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).** Block, warp and tensor-core tiles with staged background copies, studied as techniques, and which tensor-core instructions GB10 has. No CUTLASS code or NVIDIA math library runs in Vade.
- **[NCCL](https://github.com/NVIDIA/nccl), [nccl-tests](https://github.com/NVIDIA/nccl-tests).** Earlier lab routing and link observations that informed one subnet per cable. Vade does not link NCCL.
- **[rdma-core](https://github.com/linux-rdma/rdma-core), [perftest](https://github.com/linux-rdma/perftest).** The `verbs.h` interface Vade declares in Rust, standard send and receive practice, and independent link measurements.\
  linux-rdma: [GitHub](https://github.com/linux-rdma)
- **[Megatron-LM](https://arxiv.org/abs/1909.08053).** Tensor parallelism: splitting each layer by heads and matrix rows. Vade keeps a fixed order of additions in its own rank code.\
  Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper and Bryan Catanzaro
- **[Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521).** The ring all-gather with balanced slices, for any number of Sparks.\
  Rajeev Thakur, Rolf Rabenseifner and William Gropp

<details>
<summary><b>Author profiles</b>: GitHub links for the people above</summary>

**Mostofa Patwary**: [GitHub](https://github.com/mpatwary)\
**Jared Casper**: [GitHub](https://github.com/jaredcasper)

</details>

## Compiler and standards

**The compiler and system tools Vade is built with, and the file and wire formats it implements.**

Compiler and system:

- **[Rust](https://github.com/rust-lang/rust), [NVPTX target](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html), [rustup](https://rustup.rs).** The compiler, Cargo, the standard library and GPU `core` support. External build tools, not Vade code.\
  [GitHub](https://github.com/rust-lang) · [Site](https://www.rust-lang.org)
- **[LLVM NVPTX](https://llvm.org/docs/NVPTXUsage.html), [NVVM intrinsics](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsNVVM.td).** Rust's GPU backend, and the exact names and arguments of GPU intrinsics and tensor-core register layouts.\
  [GitHub](https://github.com/llvm) · [Site](https://llvm.org)
- **[NetworkManager](https://networkmanager.dev), [CPU idle states](https://docs.kernel.org/admin-guide/pm/cpuidle.html).** The format of the network plan Vade prints, and CPU wake-up latency research. Vade makes no network or OS changes.

Formats and standards:

- **Unicode Consortium: [Unicode Character Database](https://www.unicode.org/ucd/), [license](https://www.unicode.org/license.txt).** Character categories and normalization tables. Versions are recorded in [Provenance](provenance.md).
- **OpenAI API contributors: [OpenAI API specification](https://github.com/openai/openai-openapi).** Chat and text completion request and reply shapes. Supported fields are in [Server and API](server.md).
- **IETF and WHATWG contributors: [JSON, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259), [HTTP/1.1, RFC 9112](https://www.rfc-editor.org/rfc/rfc9112), [Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html).** Wire formats in Vade's own HTTP parser and streaming writer.
- **NIST and IETF contributors: [SHA-256, FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final), [HMAC, RFC 2104](https://www.rfc-editor.org/rfc/rfc2104), [RFC 4231 test vectors](https://www.rfc-editor.org/rfc/rfc4231).** Hashing, and authenticated control messages between Sparks.

## Studied, not used

**The project examined these closely and chose not to adopt them.** They are listed because they shaped decisions.

Hardware and system:

- **[SparkRing](https://github.com/FujitsuPolycom/sparkring), by FujitsuPolycom.** Uses ConnectX hardware forwarding so Sparks in a ring reach non-neighbours. Vade's rings pass data between neighbours in software.\
  [GitHub](https://github.com/FujitsuPolycom)
- **[DGX Spark Memory Saver](https://github.com/christopherowen/dgx-spark-memory-saver), by Christopher Owen.** Recovers memory lost to GPU page tables, but needs a different kernel, a rebuilt driver module and a reboot. Vade changes nothing in the OS or driver.\
  [GitHub](https://github.com/christopherowen) · [X](https://x.com/christopherowen)

Draft heads and self-drafting:

- **[Medusa](https://arxiv.org/abs/2401.10774).** Needs extra trained draft heads that the model makers do not ship.\
  Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen and Tri Dao; FasterDecoding
- **[EAGLE](https://arxiv.org/abs/2401.15077).** Needs an extra trained draft head that the model makers do not ship.\
  Yuhui Li, Fangyun Wei, Chao Zhang and Hongyang Zhang; SafeAI Lab
- **[LayerSkip](https://arxiv.org/abs/2404.16710).** Its early-exit checkpoints change the official weights.\
  Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich and others (13 authors); Meta Research
- **[Draft & Verify](https://arxiv.org/abs/2309.08168).** Self-drafting by skipping layers was weighed for prose and not built.\
  Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen and Sharad Mehrotra
- **A smaller-vocabulary draft head**, from Victor Cruz's [Flash Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe). Studied; not used today.

<details>
<summary><b>Author profiles</b>: GitHub, Hugging Face, X and websites for the people above</summary>

**Tianle Cai**: [GitHub](https://github.com/ctlllll) · [X](https://x.com/tianle_cai) · [Site](https://tianle.website)\
**Yuhong Li**: [GitHub](https://github.com/leeyeehoo) · [Site](http://leeyeehoo.github.io/)\
**Hongwu Peng**: [GitHub](https://github.com/harveyp123) · [Site](https://harveyp123.github.io)\
**Tri Dao**: [GitHub](https://github.com/tridao) · [Hugging Face](https://huggingface.co/tridao) · [Site](https://tridao.me)\
**FasterDecoding**: [GitHub](https://github.com/FasterDecoding)\
**Yuhui Li**: [GitHub](https://github.com/Liyuhui-12)\
**Hongyang Zhang**: [GitHub](https://github.com/hongyanz) · [X](https://x.com/hongyangzh) · [Site](https://hongyanz.github.io/)\
**SafeAI Lab**: [GitHub](https://github.com/SafeAILab)\
**Mostafa Elhoushi**: [GitHub](https://github.com/mostafaelhoushi) · [X](https://x.com/m_elhoushi)\
**Meta Research**: [GitHub](https://github.com/facebookresearch)

</details>

## Reused data

**A few files are kept unchanged, as data, with their licenses.**

- Qwen3-8B's configuration, weight index and chat template (Apache-2.0).
- Unicode character tables (Unicode License V3).
- Gemma 4's canonical chat template, used to validate model folders.
- Reference outputs in `fixtures/`, made by running the official models in Hugging Face Transformers.

[Provenance](provenance.md) lists the files, checksums and versions, and [NOTICE](../NOTICE) carries the attributions.

Vade also declares the CUDA driver and RDMA functions it calls, written against NVIDIA's `cuda.h` and rdma-core's `verbs.h`: only names, types and constants, as any program calling these libraries must. Vade's reuse of its own earlier kernels is not an outside source, so it is not listed.

## Corrections

**To report a missing or wrong credit, open an issue in the repository's issue tracker.** Name the idea, its original source and the part of Vade that uses it.

A credit names the original source of an idea and, if different, the project where we first saw it. An earlier original source is always welcome.

**Where next:** [Link checks](link-checks.md) · [Provenance](provenance.md) · [Documentation index](README.md)

## Per-row numerical rejection

The outside Numerics advisory drafts dated 2026-10-10 ("Router NaN per-row status"
and "BF16 NaN-safe rounding helper") informed the row-isolation and shared-conversion
contracts. Implementation is original Vade Rust; no advisory code was copied. The
BF16 NaN value 0x7FFF follows Vade's GB10 tensor-conversion measurements.
