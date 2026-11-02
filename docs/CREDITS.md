# Credits

This page names everyone and everything that Vade builds on, in one place.

Vade's code is written from scratch in Rust. No source code was copied from any project on this page. A credit names an idea or a reference behavior and the part of Vade that uses it.

To report a missing or wrong credit, open an issue in the repository's issue tracker. Name the idea, its original source and the part of Vade that uses it.

<br>

**On this page:**

- [Models](#models): the checkpoints and their makers
- [Engine](#engine): the projects, papers and ideas that the engine uses
- [Harness](#harness): the agent and its tools
- [Tooling and docs](#tooling-and-docs): the compiler, the standards and the files kept from others
- [Terms used on this page](#terms-used-on-this-page)
- [Pinned sources](#pinned-sources)

The [NOTICE](../NOTICE) file lists the few files that Vade ships from others, with their licenses. The [engine NOTICE](../engine/NOTICE) keeps the same list with the engine.

<br>

---

## Models

<a id="model-makers"></a>

### Model makers

**Vade runs official releases only, so these teams define what it must compute.**

- **Qwen team, Alibaba Cloud.** Model definitions, official BF16 and FP8 weights and scales, tokenizers and chat templates.
  - [GitHub](https://github.com/QwenLM)
  - [Hugging Face](https://huggingface.co/Qwen)
  - [X](https://x.com/Alibaba_Qwen)
  - [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- **Google DeepMind, Gemma team.** Main's Gemma text model: local and global attention, proportional rotary positions, normalization, tokenizer and the canonical chat template.
  - [GitHub](https://github.com/google-deepmind)
  - [Hugging Face](https://huggingface.co/google)
  - [Site](https://deepmind.google)
  - [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it)
- **OpenAI, gpt-oss team.** Development only: the official MXFP4 checkpoint, model equations and the Harmony conversation format.
  - [GitHub](https://github.com/openai)
  - [Hugging Face](https://huggingface.co/openai)
  - [Site](https://openai.com/)
  - [gpt-oss](https://github.com/openai/gpt-oss)
  - [Harmony](https://github.com/openai/harmony)
- **Mistral AI.** Development only: Mistral Small 3.2's model contract, Tekken tokenizer and instruction format.
  - [GitHub](https://github.com/mistralai)
  - [Hugging Face](https://huggingface.co/mistralai)
  - [Site](https://mistral.ai)
  - [mistral-common](https://github.com/mistralai/mistral-common)
- **Microsoft, Phi team.** Development only: the official BF16 checkpoint, architecture, tokenizer and chat format.
  - [GitHub](https://github.com/microsoft)
  - [Hugging Face](https://huggingface.co/microsoft)
  - [X](https://x.com/OpenAtMicrosoft)
  - [Phi-4](https://huggingface.co/microsoft/phi-4)
- **Meta.** The official Llama 3.1 and 3.2 BF16 configurations, the Llama 3 rotary-position scaling by frequency, and the text chat templates. The configuration reader, the renderer and the frequency table are Vade's own Rust.
- **NVIDIA Model Optimizer.** The official [Qwen3.8-Flash-Next-NVFP4 model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), and that checkpoint's headers and quantization metadata.

### What main serves

- Main serves Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B and Qwen3.8-27B (BF16 and official FP8), and Gemma 4 31B in BF16. Gemma 4 26B-A4B is development only.
- The research engine runs [Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13).
- Draft models: [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) for Qwen3.8-27B, and Qwen3-0.6B for Qwen3-8B and Qwen3-32B. Gemma's official assistant checkpoint defines the draft that Vade uses for one active request.

### What each checkpoint defines

- **Qwen3-8B model card.** The [model card](https://huggingface.co/Qwen/Qwen3-8B) sets Vade's sampling defaults. Thinking mode uses temperature 0.6, top-p 0.95 and top-k 20. Non-thinking mode uses 0.7, 0.8 and 20. The card also gives the optional presence penalty for repetition. It gives the optional YaRN settings too: factor 4, 32,768 original positions, and a maximum of 131,072.
- **Qwen3.6.** The official configuration and the installed Transformers `Qwen3_5Moe` model define the fused expert tensor shapes, the normalized top-8 BF16 routing, the ascending expert sums and the sigmoid-gated shared expert. Vade uses them as a behavior reference. The Rust code reuses Vade's hybrid and expert kernels.
- **Qwen3.8-27B.** The official BF16 configuration and checkpoint metadata define the text tensor names, the hybrid layer schedule, the zero-centered norms, the partial rotary positions and the text chat protocol. Its 15 MTP tensor headers define the shared embedding and head, the fusion and one full-attention layer.
- **Qwen3.8-27B-FP8.** Revision `017b9c7a` defines E4M3 weights, BF16 128 × 128 block scales and the list of modules left unconverted. Dense FP8 projections reuse Vade's own Flash FP8-to-BF16 promotion and tensor-core sums.
- **Qwen3.8-Flash-Next.** The official checkpoint and the Transformers FP8 conversion define E4M3FN values multiplied by promoted BF16 128 × 128 block scales, and one scalar scale for the n-gram embeddings. The shape and ownership contract and the upstream source links are in [the Flash-Next architecture note](../engine/docs/history/flash-next-architecture.md).
- **Qwen3.8-Flash-Next-NVFP4.** The NVIDIA checkpoint mixes NVFP4 main experts, FP8 MTP and n-gram embeddings, and BF16 dense storage. Values are packed E2M1, with an E4M3 scale per 16 values and one FP32 global scale. The packed loader, the CPU dequantization, the W4A16 register dequantization and the BF16 MMA path are Vade's own Rust and PTX. This W4A16 mode does not implement the checkpoint's W4A4 quantizer.
- **Gemma 4 31B.** Google DeepMind's official configuration, tokenizer files and canonical chat template, dated 2026-07-09, under Apache-2.0. The template is kept unchanged in `engine/crates/tokenizer/templates/gemma4.jinja` to check model folders. Serving follows the tokenizer's `response_template`: thought channels and raw-string tool arguments. Tool dictionaries follow the Jinja `dictsort` rule: stable and case-insensitive. The parser, the ordered JSON view, the configuration reader, the BPE extension and the text formatter are Vade's own Rust.
- **Gemma 4 assistant.** The Hugging Face Transformers and Google Gemma 4 assistant reference defines the draft model. It is a query-only decoder over the target's last local and global KV. It joins the scaled target embedding and the normalized hidden state. It runs the projection steps at a fixed position, and it uses the full tied head when ordered embeddings are off. Vade builds it from its shared Gemma layers and its strict verifier.
- **Phi-4.** Microsoft's official BF16 configuration and tokenizer files. The fused projection row loading and the BF16 boundaries reuse Vade's earlier Phi-4 work (3315d8d) and the shared dense kernels.

<br>

---

## Engine

<a id="original-work"></a>

### What Vade built itself

**These parts were designed and built in this project.** Each item names where to check it in the tree. Where an item uses a published idea, this page names that idea and its source.

- **The exactness law and its gates.** The same request with the same setup must give identical token IDs on every run. This holds alone or batched, cached or fresh, and with drafting on or off. The same request must also give the same IDs across supported Spark counts. The rule and the merge gate are in [AGENTS.md](../AGENTS.md#exactness-law-and-merge-gate) and [architecture](architecture.md#exactness-and-model-quality). Batch-invariant arithmetic is a published idea, credited under [Attention and determinism](#attention-and-determinism). The rule does not hold everywhere yet. On FP8, six Sparks give other token IDs than two Sparks ([changelog](../CHANGELOG.md), [Exactness](../README.md#exactness)).
- **Standard-library Rust, GPU kernels included.** The engine, the CLI, the harness and the repository tools use no third-party crates. The GPU kernels are Rust compiled for NVPTX, in [engine/crates/gpu/kernels](../engine/crates/gpu/kernels) ([What you get](../README.md#what-you-get)).
- **The tensor split and the RDMA fabric.** The split runs one model across two or four Sparks. The Sparks send data to each other over RDMA ([engine/crates/tp](../engine/crates/tp/README.md), [verbs.rs](../engine/crates/fabric/src/verbs.rs)). Tensor parallelism follows the Megatron-LM paper. The transport ideas that we studied are credited under [DGX Spark community](#dgx-spark-community) and [NVIDIA and transport](#nvidia-and-transport). The rank code and the transport are our own Rust.
- **The Flash-Next MTP drafting path.** The checkpoint's own multi-token prediction (MTP) head proposes tokens. A batched check keeps only the tokens that the model would have chosen ([next_mtp.rs](../engine/crates/engine/src/flash/next_mtp.rs)). Under greedy decoding, drafting on and off give identical token IDs ([Drafting](architecture.md#drafting)). Draft-and-verify and MTP are published ideas, credited under [Speculative decoding](#speculative-decoding).
- **Cards, gates and receipts.** `vade verify` sends the same requests with drafting on and off. Then it checks that the token IDs match ([Verify exactness](../cli/README.md#verify-exactness)). `vade bench` measures the running engine and writes a receipt. A change to engine behavior must pass a device gate on real Sparks before it merges. Its card joins the exactness and measurement results ([Exactness and evidence](../.github/CONTRIBUTING.md#exactness-and-evidence), [The measurement card](../.github/CONTRIBUTING.md#the-measurement-card)). A gated change that was rebased onto a moved main passes a screen on the Sparks first.

### Reference code

**Vade checks its answers against the makers' official code.** The agreement is measured and is not perfect. [Validation](../engine/docs/validation.md) has the counts. Determinism is a separate promise: the same request gives the same answer, alone or batched, cached or fresh.

One file follows a credited reference closely. The FP32 CPU reference [refcpu.rs](../engine/crates/model/src/flash/refcpu.rs) computes the formulas of the Transformers file `modeling_qwen4_exp.py` (revision `b0cdc61`) in the same order. Its comments cite the lines of that file.

- **Hugging Face.** The team behind the reference libraries below.
  - [GitHub](https://github.com/huggingface)
  - [Hugging Face](https://huggingface.co/huggingface)
  - [X](https://x.com/huggingface)
  - [Site](https://huggingface.co/)
- **[Transformers](https://github.com/huggingface/transformers) contributors and model authors.** The official reference behind Vade's saved CPU and CUDA fixtures, the sign of the repetition penalty, and the exact order and rounding of each operation. Reference tooling only. It is never linked into Vade.
- **[Tokenizers](https://github.com/huggingface/tokenizers) contributors.** The `tokenizer.json` format, the reference tokenization, and the Unicode 12.1 behavior of version 0.23.2 that Vade's tokenizer reproduces for Qwen3.
- **[safetensors](https://huggingface.co/docs/safetensors) and [huggingface_hub](https://github.com/huggingface/huggingface_hub) contributors.** The checkpoint file and index format, read by Vade's own bounded reader, and the official download tooling.
- **PyTorch contributors.** PyTorch runs the official reference models that produce the saved fixtures. Reference tooling only. Vade's normalization follows the mean arithmetic in PyTorch's CUDA `Reduce.cuh`: four neighboring FP32 sums, a descending warp reduction and a multiply by the reciprocal count.
  - [GitHub](https://github.com/pytorch)
  - [Hugging Face](https://huggingface.co/pytorch)
  - [Site](https://pytorch.org)
  - [PyTorch](https://github.com/pytorch/pytorch)
- **Python and NumPy contributors.** The privately kept generators that produced the saved reference data. They are kept outside this repository and never run in the engine.
  - [python.org](https://www.python.org/)
  - [numpy.org](https://numpy.org)
- **ByteDance Seed and the VeOmni contributors.** Embedding-first fusion, final-normalized target features and unshifted rotary positions in the multi-token prediction (MTP) draft head for Qwen3.8-27B. Agreement with a maker MTP forward pass is not verified. The target model checks every drafted token.
  - [GitHub](https://github.com/ByteDance-Seed)
  - [Hugging Face](https://huggingface.co/ByteDance-Seed)
  - [Site](https://seed.bytedance.com/)
  - [VeOmni MTP note](https://github.com/bytedance-seed/veomni/blob/86c5ec1dbba8270206b1ab7f439285215ae4ec92/docs/key_features/mtp.md)
- **kernels-community.** The FP8 activation rule: a per-128 maximum, a scale of maximum / 448, and division by max(scale, 1e-12) before E4M3 rounding. The default keeps BF16 activations.
  - [Hugging Face](https://huggingface.co/kernels-community)
  - [finegrained-fp8 recipe](https://huggingface.co/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py#L440)

The exact versions that settled each detail are under [Pinned sources](#pinned-sources).

### Attention and determinism

**Each kernel keeps one fixed order of additions, so the same prompt always gives the same answer.** These papers are where the attention algorithms come from.

Attention:

- **[FlashAttention](https://arxiv.org/abs/2205.14135).** Attention in tiles with a running maximum and running total, instead of a full score matrix.
  - Tri Dao: [GitHub](https://github.com/tridao), [Hugging Face](https://huggingface.co/tridao), [Site](https://tridao.me)
  - Daniel Y. Fu: [GitHub](https://github.com/DanFu09), [Hugging Face](https://huggingface.co/danfu09), [X](https://x.com/realDanFu), [Site](https://www.danfu.org)
  - Stefano Ermon
  - Atri Rudra
  - Christopher Ré
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691).** Splitting attention work across GPU blocks and warps, used to organize prefill attention.
  - Tri Dao (links above)
- **[FlashAttention-3](https://arxiv.org/abs/2407.08608).** Overlapping background data movement with matrix work, applied with GB10 instructions. No Hopper kernels are imported. No lower precision is on by default.
  - Jay Shah
  - Ganesh Bikshandi
  - Ying Zhang
  - Vijay Thakkar: [GitHub](https://github.com/thakkarV)
  - Pradeep Ramani
  - Tri Dao (links above)
- **[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867).** The running softmax total that tiled attention is built on.
  - Maxim Milakov: [GitHub](https://github.com/milakov), [Site](https://milakov.org)
  - Natalia Gimelshein: [GitHub](https://github.com/ngimel)
- **[Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html).** Splitting a long KV cache into parts, summarizing each, then merging them.
  - Tri Dao (links above)
  - Daniel Haziza
  - Francisco Massa: [GitHub](https://github.com/fmassa)
  - Grigory Sizov: [GitHub](https://github.com/sgrigory)

Determinism:

- **[Defeating nondeterminism in LLM inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).** Batch-invariant arithmetic: sums that do not change when the batch changes. Vade keeps one order for alone, batched, verification and cached paths.
  - Horace He: [GitHub](https://github.com/Chillee), [Hugging Face](https://huggingface.co/Chillee), [X](https://x.com/cHHillee), [Site](http://horace.io/)
  - Thinking Machines Lab: [GitHub](https://github.com/thinking-machines-lab), [Site](https://thinkingmachines.ai)
- **vLLM issue #59985.** It shows a class of failure where a special case for one token breaks batch invariance. Vade sends its indexed attention and its MoE router and shared-expert projections through one kernel per width. It checks one-row launches against 4-, 8- and 16-row launches. The kernels and that check are Vade's own.

### Serving many requests

**Continuous batching, shared memory pages, prefix reuse and scheduling, plus the other engines in the [benchmark tables](../engine/docs/benchmarks.md).**

Scheduling and memory:

- **[Orca](https://www.usenix.org/conference/osdi22/presentation/yu).** Continuous batching: requests join and leave between generation steps.
  - Gyeong-In Yu: [GitHub](https://github.com/gyeongin), [Site](https://gyeongin.github.io)
  - Joo Seong Jeong
  - Geon-Woo Kim
  - Soojeong Kim
  - Byung-Gon Chun
- **[PagedAttention](https://arxiv.org/abs/2309.06180).** KV memory in shared, read-only pages, copied only when a request diverges. Vade maps pages with CUDA virtual memory instead of block tables.
  - Woosuk Kwon: [GitHub](https://github.com/WoosukKwon), [X](https://x.com/woosuk_k), [Site](https://woosuk.me)
  - Zhuohan Li: [GitHub](https://github.com/zhuohan123), [X](https://x.com/zhuohan123), [Site](https://zhuohan.li)
  - Siyuan Zhuang: [GitHub](https://github.com/suquark)
  - Ying Sheng: [GitHub](https://github.com/Ying1123)
  - Lianmin Zheng: [GitHub](https://github.com/merrymercy), [Hugging Face](https://huggingface.co/lmzheng), [X](https://x.com/lm_zheng), [Site](http://lmzheng.net)
  - Cody Hao Yu: [GitHub](https://github.com/comaniac), [Site](https://comaniac.github.io)
  - Joseph E. Gonzalez: [GitHub](https://github.com/jegonzal), [X](https://x.com/profjoeyg), [Site](http://eecs.berkeley.edu/~jegonzal)
  - Hao Zhang: [GitHub](https://github.com/zhisbug), [X](https://x.com/haozhangml), [Site](https://haozhang.ai/)
  - Ion Stoica: [GitHub](https://github.com/istoica), [Site](http://www.cs.berkeley.edu/~istoica/)
- **[vLLM](https://github.com/vllm-project/vllm).** From v0.30.0's [scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py), [cancellation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/engine/async_llm.py) and [block pool](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/block_pool.py): a token budget per step, stopping a cancelled request's work before closing it, and automatic prefix caching that finds reusable blocks by hash. Benchmarked at 0.31.0 and as NGC vLLM 25.09 (vLLM v0.10.1.1).
  - vLLM team: [GitHub](https://github.com/vllm-project), [Hugging Face](https://huggingface.co/vllm-project), [Docs](https://docs.vllm.ai)
- **[SGLang](https://github.com/sgl-project/sglang).** Mixing prompt chunks with decode steps, sizing attention per request, and sizing GB10 memory from Linux `MemAvailable`, so reclaimable file cache counts as capacity. The project's earlier SGLang work on Sparks produced the rule that a live request runs between chunks of a long background prefill. Benchmarked at 0.5.21.
  - SGLang team: [GitHub](https://github.com/sgl-project), [Docs](https://docs.sglang.ai)
- **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM).** Design references for sharing KV blocks, replaying CUDA graphs over fixed buffers, and mixing prefill and decode work, from its [KV cache notes](https://github.com/NVIDIA/TensorRT-LLM/blob/a81da8a5a8380c4ec8c2d3eafabb8cfc55d7ad56/docs/source/features/kvcache.md). Benchmarked at 1.3.0rc13.
  - NVIDIA TensorRT-LLM team: [GitHub](https://github.com/NVIDIA)

Shared prompt beginnings:

- **[SGLang paper](https://arxiv.org/abs/2312.07104).** RadixAttention: automatic reuse of any matching prompt beginning.
  - Lianmin Zheng (links above)
  - Liangsheng Yin: [GitHub](https://github.com/hnyls2002), [X](https://x.com/lsyincs)
  - Zhiqiang Xie: [GitHub](https://github.com/xiezhq-hermann), [X](https://x.com/Zhiqiang_Xie), [Site](https://zhiqiangxie.com)
  - and 9 more authors, named on the paper
- **[FlashInfer paper](https://arxiv.org/abs/2501.01005), [project](https://github.com/flashinfer-ai/flashinfer), [cascade attention](https://flashinfer.ai/2024/02/02/cascade-inference.html).** Reading KV tiles that many requests share only once. Vade keeps each request's own fixed order of sums.
  - Zihao Ye: [GitHub](https://github.com/yzh119), [Site](https://expye.com/about)
  - Lequn Chen: [GitHub](https://github.com/abcdabcd987), [Site](https://abcdabcd987.com)
  - Ruihang Lai: [GitHub](https://github.com/MasterJH5574), [X](https://x.com/ruihanglai), [Site](https://ruihanglai.com)
  - and 8 more authors, named on the paper
  - FlashInfer team: [GitHub](https://github.com/flashinfer-ai), [Site](https://flashinfer.ai/)
- **[Hydragen](https://arxiv.org/abs/2402.05099).** Shared-prefix attention: avoid rereading a common prompt beginning for each request.
  - Jordan Juravsky
  - Bradley Brown
  - Ryan Ehrlich
  - Daniel Y. Fu (links above)
  - Christopher Ré
  - Azalia Mirhoseini
  - Scaling Intelligence Lab: [GitHub](https://github.com/ScalingIntelligence), [Site](https://scalingintelligence.stanford.edu/)

### Speculative decoding

**A cheap guess proposes several tokens, and the full model checks them in one pass, keeping only the ones it would have chosen itself.** Vade's flags call it drafting.

Draft and verify:

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192).** Draft-model guesses checked in one pass of the full model.
  - Yaniv Leviathan
  - Matan Kalman
  - Yossi Matias
- **[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318).** The verification principle. Vade's default promise is exact greedy output. The optional Flash sampled verifier uses the papers' math as ideas: sampled draft proposals, acceptance by min(1, p/q), a correction from the positive remainder, and a bonus token from the target. It is built on Vade's own sampler and GPU kernels.
  - Charlie Chen
  - Sebastian Borgeaud
  - Geoffrey Irving
  - Jean-Baptiste Lespiau
  - Laurent Sifre
  - John Jumper
- **[SpecInfer](https://arxiv.org/abs/2305.09781).** Checking a tree of guesses in one pass, each guess seeing only its ancestors. Development only: draft trees live on unmerged branches, and main drafts single chains.
  - Xupeng Miao
  - Gabriele Oliaro: [GitHub](https://github.com/goliaro)
  - Zhihao Zhang
  - and 12 more authors, named on the paper
  - FlexFlow project: [GitHub](https://github.com/flexflow), [Site](https://flexflow.ai/)
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437).** Multi-token prediction as a draft mechanism. Qwen3.8-27B's actual MTP layout comes from Qwen's checkpoint and the [reference code](#reference-code).
  - DeepSeek-AI: [GitHub](https://github.com/deepseek-ai), [Hugging Face](https://huggingface.co/deepseek-ai), [Site](https://www.deepseek.com/)
- **[vLLM Ascend pull request 7374](https://github.com/vllm-project/vllm-ascend/pull/7374).** Published Qwen3-8B with Qwen3-0.6B acceptance numbers, which made a small official Qwen3 model the first draft-model choice. Their speeds are not Vade measurements.
  - KlyzhenkoVadim: [GitHub](https://github.com/KlyzhenkoVadim)
  - the vLLM Ascend contributors

Guessing without a draft model:

- **[Prompt Lookup Decoding](https://github.com/apoorvumang/prompt-lookup-decoding).** Guessing continuations from text already in the prompt. This is the original published source. We first saw the idea in TensorFold.
  - Apoorv Saxena: [GitHub](https://github.com/apoorvumang), [Hugging Face](https://huggingface.co/apoorvumang), [X](https://x.com/apoorv_umang), [Site](https://apoorvumang.github.io)
- **[Token Recycling](https://arxiv.org/abs/2408.08696).** Keeping the model's predictions, even from rejected rows, as future guesses. Vade drafts single chains. Its greedy HTTP path learns the best follower of each token without downloading the full scores.
  - Xianzhen Luo: [GitHub](https://github.com/Luowaterbi), [Site](https://luowaterbi.github.io)
  - Yixuan Wang
  - Qingfu Zhu
  - Zhiming Zhang
  - Xuanyu Zhang
  - Qing Yang
  - Dongliang Xu
- **[REST](https://arxiv.org/abs/2311.08252).** Guessing from how earlier text continued. Vade looks only in the request's own recent history.
  - Zhenyu He: [GitHub](https://github.com/zhenyuhe00), [Site](https://zhenyuhe00.github.io)
  - Zexuan Zhong
  - Tianle Cai: [GitHub](https://github.com/ctlllll), [X](https://x.com/tianle_cai), [Site](https://tianle.website)
  - Jason D. Lee
  - Di He
- **[CREST](https://arxiv.org/abs/2408.04678).** Ranking several retrieved continuations. Tried and removed. Bounded suffix lookup remains.
  - Sophia Ho
  - Jinsol Park
  - Patrick Wang
- **[Lookahead Decoding](https://arxiv.org/abs/2402.02057).** Reusing predictions computed after a wrong guess, by shifting rows already computed. Vade needs no extra passes of the target model for this.
  - Yichao Fu: [GitHub](https://github.com/Viol2000), [X](https://x.com/fuyichao123), [Site](https://viol2000.github.io/)
  - Peter Bailis: [GitHub](https://github.com/pbailis), [X](https://x.com/pbailis), [Site](http://bailis.org)
  - Ion Stoica (links above)
  - Hao Zhang (links above)

Choosing how far ahead to guess:

- **[DS41RT draft policy](https://github.com/tpurtell/ds41rt/blob/main/docs/dspark-bandwidth-policy.md).** Choosing the guess depth from measured cost.
  - T.J. Purtell: [GitHub](https://github.com/tpurtell)
- **[Flash-Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe).** Changing the guess depth according to whether guessing is paying off.
  - Victor Cruz: [GitHub](https://github.com/vcruz305), [Hugging Face](https://huggingface.co/vcruz305), [X](https://x.com/ViC305)

### Model building blocks

**Vade computes these because the official models are built from them.** The makers' reference code settles the exact operation order and precision.

Transformer layers:

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762).** The Transformer: attention and feed-forward layers.
  - Ashish Vaswani
  - Noam Shazeer
  - Niki Parmar
  - Jakob Uszkoreit
  - Llion Jones
  - Aidan N. Gomez
  - Łukasz Kaiser: [GitHub](https://github.com/lukaszkaiser)
  - Illia Polosukhin
- **[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).** RMSNorm: scaling a layer's values by their root mean square.
  - Biao Zhang: [GitHub](https://github.com/bzhangGo)
  - Rico Sennrich: [GitHub](https://github.com/rsennrich)
- **[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).** SwiGLU, the gated feed-forward activation.
  - Noam Shazeer
- **[GQA](https://arxiv.org/abs/2305.13245).** Grouped-query attention: several query heads share one KV head.
  - Joshua Ainslie
  - James Lee-Thorp
  - Michiel de Jong
  - Yury Zemlyanskiy
  - Federico Lebrón
  - Sumit Sanghai
- **[Sparsely-Gated Mixture-of-Experts](https://arxiv.org/abs/1701.06538).** Expert layers: each token uses a few of many small networks. Qwen3-30B-A3B is built this way.
  - Noam Shazeer
  - Azalia Mirhoseini
  - Krzysztof Maziarz
  - Andy Davis
  - Quoc Le
  - Geoffrey Hinton
  - Jeff Dean

Positions:

- **[RoFormer](https://arxiv.org/abs/2104.09864).** Rotary position embeddings (RoPE): rotate query and key values by token position.
  - Jianlin Su: [GitHub](https://github.com/bojone), [Site](https://kexue.fm)
  - Yu Lu
  - Shengfeng Pan
  - Ahmed Murtadha
  - Bo Wen
  - Yunfeng Liu
- **[YaRN](https://arxiv.org/abs/2309.00071).** The optional Qwen3 context extension: frequency ramp and attention scaling.
  - Bowen Peng: [Hugging Face](https://huggingface.co/bloc97)
  - Jeffrey Quesnelle: [GitHub](https://github.com/jquesnelle), [X](https://x.com/theemozilla), [Site](http://jeffq.com)
  - Honglu Fan: [GitHub](https://github.com/honglu2875), [Site](https://honglu.fan)
  - Enrico Shippole: [GitHub](https://github.com/conceptofmind), [Hugging Face](https://huggingface.co/conceptofmind), [X](https://x.com/EnricoShippole), [Site](https://www.teraflopai.com/)
  - Nous Research: [GitHub](https://github.com/NousResearch), [Hugging Face](https://huggingface.co/NousResearch), [X](https://x.com/nousresearch), [Site](https://nousresearch.com)
- **[Qwen2-VL](https://arxiv.org/abs/2409.12191).** Research engine only: the multimodal rotary positions behind Flash-Next's text positions. Not multimodal serving.
  - Peng Wang
  - Shuai Bai
  - Sinan Tan
  - and 16 more authors, named on the paper

Linear attention and gates, used by the Qwen3.8 models:

- **[Gated Delta Networks](https://arxiv.org/abs/2412.06464).** The gated recurrent linear-attention layers in the Qwen3.8 models.
  - Songlin Yang: [GitHub](https://github.com/sustcsonglin), [X](https://x.com/SonglinYang4), [Site](https://sustcsonglin.github.io/)
  - Jan Kautz
  - Ali Hatamizadeh: [GitHub](https://github.com/ahatamiz), [X](https://x.com/ahatamiz1), [Site](https://ahatamiz.github.io/)
- **[Parallelizing Linear Transformers with the Delta Rule](https://arxiv.org/abs/2406.06484).** The chunked delta rule and its triangular solve.
  - Songlin Yang (links above)
  - Bailin Wang: [GitHub](https://github.com/berlino), [X](https://x.com/bailin_28), [Site](https://berlino.github.io/)
  - Yu Zhang: [GitHub](https://github.com/yzhangcs), [X](https://x.com/yzhang_cs), [Site](https://yzhang.site)
  - Yikang Shen: [GitHub](https://github.com/yikangshen)
  - Yoon Kim: [GitHub](https://github.com/yoonkim), [Site](https://people.csail.mit.edu/yoonkim/)
- **[flash-linear-attention](https://github.com/fla-org/flash-linear-attention).** The algorithm behind the official Transformers delta-rule code. Vade writes its own Rust kernels. SGLang 0.5.21's public split of this chunk work into phases, and its cuBLAS kernel names, helped Vade name the phases in its profiles. Its narrower dot operands were not adopted.
  - Songlin Yang (links above)
  - Yu Zhang (links above)
  - flash-linear-attention contributors: [GitHub](https://github.com/fla-org), [Hugging Face](https://huggingface.co/fla-hub)
- **[Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708).** The output gate on Qwen3.8-27B's full-attention layers.
  - Zihan Qiu
  - Zekun Wang
  - Bo Zheng
  - and 10 more authors, named on the paper
- **[Hyper-Connections](https://arxiv.org/abs/2409.19606).** Research engine only: the parallel residual streams in Flash-Next.
  - Defa Zhu
  - Hongzhi Huang
  - Zihao Huang
  - Yutao Zeng
  - Yunyao Mao
  - Banggu Wu
  - Qiyang Min
  - Xun Zhou

Tokens, number formats and sampling:

- **[Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909).** Byte-pair encoding (BPE): tokenizing by ranked pair merges, in Vade's own merge engine.
  - Rico Sennrich (links above)
  - Barry Haddow: [GitHub](https://github.com/bhaddow)
  - Alexandra Birch
- **[GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [GPT-2 code](https://github.com/openai/gpt-2), [tiktoken](https://github.com/openai/tiktoken).** Byte-level tokenization used by Qwen: every byte maps to a printable character before merging. Qwen's pre-tokenizer pattern is a close variant of one that tiktoken publishes.
  - Alec Radford
  - Jeffrey Wu
  - Rewon Child
  - David Luan
  - Dario Amodei
  - Ilya Sutskever
  - OpenAI (links under [Model makers](#model-makers))
- **[OCP microscaling specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537).** Development only: MXFP4 values with shared block exponents in the official gpt-oss checkpoint. No extra quantization is added or turned on.
  - Open Compute Project contributors
  - Bita Darvish Rouhani
  - Ritchie Zhao
  - and 31 more authors, named on the paper
- **[Fast splittable pseudorandom number generators](https://doi.org/10.1145/2660193.2660195), [SplitMix64 reference](https://prng.di.unimi.it/splitmix64.c).** SplitMix64's mixing constants turn a seed and a token position into a sampling value, so a guess and its check draw the same number. Written in Rust. It is not for cryptography.
  - Guy L. Steele
  - Doug Lea
  - Christine H. Flood
  - Sebastiano Vigna, reference code: [GitHub](https://github.com/vigna), [Site](http://vigna.di.unimi.it/)

### DGX Spark community

**These people and projects published their work on the same hardware. Vade uses some of their ideas, never their code.**

<a id="tensorfold"></a>

**[TensorFold](https://github.com/ashhart/TensorFold), by Ash Hart and the TensorFold contributors.** We studied eleven of its ideas, listed below. Vade's kernel comments and build notes name TensorFold where one of these ideas is used. Each implementation is Vade's own Rust.

- Ash Hart: [GitHub](https://github.com/ashhart), [X](https://x.com/ashxhart)

Ideas studied in TensorFold, on the GPU:

- The whole decode step stays on the GPU: KV cache, activations and buffers stay resident.
- Kernels are looked up once, and launches are queued until a result is needed.
- Prefill attention where one GPU block serves every query head that shares a KV head, with staged, swizzled tiles. Swizzled shared-memory tiles themselves are older practice from CUTLASS and GPU textbooks ([NVIDIA and transport](#nvidia-and-transport)).
- Wide, neighboring memory reads for matrix-vector kernels.
- Computing output scores only for the final prompt position ([source](https://github.com/ashhart/TensorFold/blob/17c73e189f5e6a5304cda7ea37f086f9c49b4788/src/tensorfold/families/qwen4_exp/cuda/decode.py)).

Ideas studied in TensorFold, for drafting, sampling and conversations:

- Exact draft verification: keep only guesses the model itself would choose. The original source is the papers under [Speculative decoding](#speculative-decoding).
- Guessing from text already in the prompt. The original source is Apoorv Saxena, under [Speculative decoding](#speculative-decoding).
- Sampling keyed by seed and token position.
- Conversation snapshots at message boundaries that can branch.

Ideas studied in TensorFold, across Sparks and for testing:

- Sums across Sparks in a fixed rank order.
- Evaluation habits: comparing against plain one-at-a-time runs, bounded draft work and per-stream state.

Engines and recipes:

- **Mia, Mia's AI Lab.** Every rank reaches transport decisions from shared state. BF16 KV cache experiments. Fetching the next weights while waiting.
  - [GitHub](https://github.com/MiaAI-Lab)
  - [X](https://x.com/MiaAI_lab)
  - [TensorFold fork](https://github.com/MiaAI-Lab/TensorFold)
  - [transport deadlock fix](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/92bf731c3aac61927726ef0c422b21b42111f2c4/patches/0040-glm-parallel-deadlocks.patch)
- **Victor Cruz (links above).** Keeping the decode loop on the GPU with few readbacks, and optional placement of the launch thread on a chosen CPU core. His [Flash-Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) is also credited under [Speculative decoding](#speculative-decoding).
- **[ExLlamaV3](https://github.com/turboderp-org/exllamav3) by turboderp, and [TabbyAPI](https://github.com/theroyallab/tabbyAPI) by The Royal Lab.** The runtimes that Victor Cruz's recipe builds on. Their contributors are credited as part of that chain. Vade uses neither.
  - turboderp: [GitHub](https://github.com/turboderp-org), [X](https://x.com/turboderp_)
  - The Royal Lab: [GitHub](https://github.com/theroyallab)
- **Local Inference Lab.** Clear ownership of every scratch buffer, testing small RDMA messages on their own, and keeping correctness evidence separate from speed measurements.
  - [GitHub](https://github.com/local-inference-lab)
  - [b12x](https://github.com/local-inference-lab/b12x)
- **Kindling AI.** Pinning exact software versions and recording system state. Vade adopts none of its OS or memory changes.
  - [GitHub](https://github.com/kindlingai)
  - [Site](https://kindlingai.dev)
  - [Kindling Spark OS](https://github.com/kindlingai/kindling-spark-os)

Transport and measurement:

- **JJ Eaton.** Measure the Spark's memory bandwidth first and judge every kernel against it. Wide neighboring reads with one warp per row block. The cost of small messages in two-Spark transport.
  - [GitHub](https://github.com/jayleaton)
  - [X](https://x.com/jayleaton)
  - [Site](https://www.jayleaton.com/)
  - [Spark roofline analysis](https://github.com/jayleaton/glm53-tensorfold-spark/blob/main/docs/ROOFLINE.md)
- **hughmadden, Turquoise Bay AI.** RDMA from Rust needs no outside crates. Sums across machines in a fixed order.
  - [GitHub](https://github.com/hughmadden)
  - [glm53f-afd](https://github.com/hughmadden/glm53f-afd)
- **T.J. Purtell (links above).** RDMA receive slots posted in advance, and polling for completions. Both are also standard verbs practice. See [DS41RT](https://github.com/tpurtell/ds41rt).
- **Jürgen Schmied.** Evidence that storing recurrent state in reduced precision changes predictions. This supports keeping Gated DeltaNet state in FP32.
  - [GitHub](https://github.com/jschmied)
  - [GB10 recurrence measurements](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/what-generalises.md)
- **Artem Matskevych.** Which points in a conversation a prefix cache can really resume from.
  - [GitHub](https://github.com/Reederey87)
  - [X](https://x.com/Reederey)
  - [Cache-tail analysis](https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark/blob/fd87969cb43a5ba66573bd2508bc05651fc1579c/docs/17-apc-tail-floor.md)

From the NVIDIA developer forum. Its profile pages refuse automated requests, so these members are credited by name only, with their posts.

- **[ConnectX-7 link capped at 13 Gbps](https://forums.developer.nvidia.com/t/connectx-7-inter-spark-link-capped-at-13-gbps-expected-200-gbps-pcie-power-throttling-27w/363461)**, by Ank-Chy, andrewc_actual, mashie and ccatzin. Evidence for diagnosing degraded link states, and why link health must be checked before any transport measurement. No system-change procedure is part of Vade.
- **[Direct QSFP link at 13-16 Gbps](https://forums.developer.nvidia.com/t/dgx-spark-direct-qsfp-connection-only-getting-13-16-gbps-instead-of-expected-200g-performance/370035)**, by ugurbas and robert287. Independent evidence about hot-plug and low-power link behavior.
- **[DGX Spark / GB10 FAQ](https://forums.developer.nvidia.com/t/dgx-spark-gb10-faq/347344)**, by NVIDIA forum staff. Hardware limits, including that the network card cannot reach GPU memory directly.
- **[GB10 NVFP4 thread](https://forums.developer.nvidia.com/t/gb10-really-does-hit-1-pflop-nvfp4-2-4-sparse-measured-with-an-open-source-tool-to-reproduce-it/373618)**, by adg1, with a device-query result from pontostroy, and **[nvfp4bench](https://github.com/secYOUre/nvfp4bench)** by Alfonso De Gregorio, secYOUre. The GB10's reported 24 MB L2 cache size, used in Vade's research on cache hints for weight reads.
  - secYOUre: [GitHub](https://github.com/secYOUre), [Site](https://www.pwnshow.com/)

Christopher Owen and FujitsuPolycom are credited under [Studied, not used](#studied-not-used).

### NVIDIA and transport

**NVIDIA's documentation and libraries, studied as references, and the papers behind the multi-Spark transport.**

- **NVIDIA CUDA documentation teams and the NVIDIA DGX Spark team.**
  - [GitHub](https://github.com/NVIDIA)
  - [Hugging Face](https://huggingface.co/nvidia)
  - [Site](https://nvidia.com)
- **[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).** Instruction and register layouts, tensor-core fragments (including FP8 m16n8k32 E4M3), BF16 and FP8 conversion, and background copies. Vade's small inline PTX blocks follow it. The `ldmatrix` mapping informs the V transpose in attention.
- **[CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/), [virtual memory](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html), [graphs](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GRAPH.html).** Loading kernels, managing memory, a growable KV cache at a fixed address, shared page mappings, device clears, and replaying recorded decode steps. The installed CUDA 13 `cuda.h` header defines Vade's interface declarations.
- **[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html), [graph launch measurements](https://developer.nvidia.com/blog/constant-time-launch-for-straight-line-cuda-graphs-and-other-performance-enhancements/).** How CPU and GPU share memory and caches on one chip, and measured graph launch costs.
- **DGX Spark team: [Clustering guide](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html), [porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/cuda.html), [GPUDirect RDMA answer](https://nvidia.custhelp.com/app/answers/detail/a_id/5780/~/is-gpudirect-rdma-supported-on-dgx-spark).** GB10 unified memory, link limits, and staging through pinned CPU memory because the network card cannot reach GPU memory.

Libraries and papers:

- **[CUTLASS](https://github.com/NVIDIA/cutlass), [Blackwell notes](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html).** Block, warp and tensor-core tiles with staged background copies, studied as techniques, and which tensor-core instructions GB10 has. No CUTLASS code or NVIDIA math library runs in Vade.
- **[NCCL](https://github.com/NVIDIA/nccl), [nccl-tests](https://github.com/NVIDIA/nccl-tests).** Earlier lab routing and link observations that informed one subnet per cable. Vade does not link NCCL.
- **[rdma-core](https://github.com/linux-rdma/rdma-core), [perftest](https://github.com/linux-rdma/perftest).** The `verbs.h` interface that Vade declares in Rust, standard send and receive practice, and independent link measurements.
  - linux-rdma: [GitHub](https://github.com/linux-rdma)
- **[Megatron-LM](https://arxiv.org/abs/1909.08053).** Tensor parallelism: splitting each layer by heads and matrix rows. Vade keeps a fixed order of additions in its own rank code.
  - Mohammad Shoeybi
  - Mostofa Patwary: [GitHub](https://github.com/mpatwary)
  - Raul Puri
  - Patrick LeGresley
  - Jared Casper: [GitHub](https://github.com/jaredcasper)
  - Bryan Catanzaro
- **[Optimization of Collective Communication Operations in MPICH](https://doi.org/10.1177/1094342005051521).** The ring all-gather with balanced slices, for any number of Sparks.
  - Rajeev Thakur
  - Rolf Rabenseifner
  - William Gropp

### Studied, not used

**The project examined these closely and chose not to adopt them.** They are listed because they shaped decisions.

Hardware and system:

- **[SparkRing](https://github.com/FujitsuPolycom/sparkring), by FujitsuPolycom.** It uses ConnectX hardware forwarding so Sparks in a ring reach non-neighbors. Vade's rings pass data between neighbors in software.
  - FujitsuPolycom: [GitHub](https://github.com/FujitsuPolycom)
- **[DGX Spark Memory Saver](https://github.com/christopherowen/dgx-spark-memory-saver), by Christopher Owen.** It recovers memory lost to GPU page tables, but it needs a different kernel, a rebuilt driver module and a reboot. Vade changes nothing in the OS or driver.
  - Christopher Owen: [GitHub](https://github.com/christopherowen), [X](https://x.com/christopherowen)

Draft heads and self-drafting:

- **[Medusa](https://arxiv.org/abs/2401.10774).** It needs extra trained draft heads that the model makers do not ship.
  - Tianle Cai (links above)
  - Yuhong Li: [GitHub](https://github.com/leeyeehoo), [Site](http://leeyeehoo.github.io/)
  - Zhengyang Geng
  - Hongwu Peng: [GitHub](https://github.com/harveyp123), [Site](https://harveyp123.github.io)
  - Jason D. Lee
  - Deming Chen
  - Tri Dao (links above)
  - FasterDecoding: [GitHub](https://github.com/FasterDecoding)
- **[EAGLE](https://arxiv.org/abs/2401.15077).** It needs an extra trained draft head that the model makers do not ship.
  - Yuhui Li: [GitHub](https://github.com/Liyuhui-12)
  - Fangyun Wei
  - Chao Zhang
  - Hongyang Zhang: [GitHub](https://github.com/hongyanz), [X](https://x.com/hongyangzh), [Site](https://hongyanz.github.io/)
  - SafeAI Lab: [GitHub](https://github.com/SafeAILab)
- **[LayerSkip](https://arxiv.org/abs/2404.16710).** Its early-exit checkpoints change the official weights.
  - Mostafa Elhoushi: [GitHub](https://github.com/mostafaelhoushi), [X](https://x.com/m_elhoushi)
  - Akshat Shrivastava
  - Diana Liskovich
  - and 10 more authors, named on the paper
  - Meta Research: [GitHub](https://github.com/facebookresearch)
- **[Draft & Verify](https://arxiv.org/abs/2309.08168).** Self-drafting by skipping layers was weighed for prose and not built.
  - Jun Zhang
  - Jue Wang
  - Huan Li
  - Lidan Shou
  - Ke Chen
  - Gang Chen
  - Sharad Mehrotra
- **A smaller-vocabulary draft head**, from Victor Cruz's [Flash-Next recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe). Studied. Not used today.

<a id="ideas-and-reference-behavior"></a>

### Where each idea is used in the engine

**This list says, part by part, what the engine took from each source and what is Vade's own.** No outside implementation code was copied into the engine. The GPU code is Rust, with inline PTX assembly where needed, compiled to PTX.

Short commit hashes and branch names, such as 8e9657e or `work/m32`, name earlier Vade work from before this repository's history begins. You will not find them in this repository.

Serving and the API:

- **Scheduler.** It follows SGLang's mixed-chunk scheduling and per-row attention sizing, and vLLM's token budget and cancel-before-close. Context-timed prefill and capability-scoped cancellation are Vade's own. Streaming attention and shared KV pages come from Vade's existing code.
- **Chat preemption.** Earlier Spark and SGLang experiments showed that a foreground chat can run between chunks of a background prefill. Vade does this with a bounded single-thread priority scheduler and nonblocking sockets, with no outside crates.
- **Prefix cache.** SGLang RadixAttention and vLLM automatic prefix caching reuse committed token prefixes, separate from client session rights. The block-hash index, the exact comparisons, the API-key scopes, live-prefill sharing and LRU accounting are Vade's own, built on its shared KV pages.
- **Responses API.** The request shapes and the SSE event contract of [OpenAI Codex CLI](https://github.com/openai/codex) 0.160.0 were found with local wire captures against a mock and Vade's scheduler. The translation from Responses to chat and the event encoder are Vade's own. A custom freeform tool reaches the model as a function with one required `input` string. This follows the JSON form of Codex's own `apply_patch` tool (idea only).
- **XML tool calls for the Qwen3.5 family.** Each `<parameter>` value is typed by the tool's declared JSON schema. This follows the idea in the Qwen3-Coder tool parsers of vLLM and SGLang. The streaming parser is Vade's own.

Attention and kernels:

- **Attention kernels.** They use FlashAttention's running maximum and total, query tiles and on-chip QK and PV products. Retained-FP32 attention extends this to two-component views of FP32 KV. The shared-input GEMM and the exact verifier staging come from Vade's own measured kernels.
- **Decode and verify attention.** Four query heads share K and V reads, and fixed row-local parts keep the decode and verify arithmetic the same. Long-context attention keeps the same running sums, with Vade's own Rust shared-memory swizzling, register probability fragments and on-chip FP32 splitting. Shared KV prefixes use the CUDA header's allocation-handle mapping. GB10 alias behavior was measured directly.
- **Grouped-head prefill attention.** It was studied in TensorFold. One block shares K and V across a group of query heads, stages swizzled tiles in the background, and keeps probabilities in registers. Vade's own Rust kernels use single BF16 operands and share each K and V tile across its query heads.
- **Long-context decode attention for Qwen3.5-family layers.** Each cached K and V tile loads once per key head for all of its query heads, and V tiles are double-buffered with `cp.async`. This follows common decode-attention practice (Flash-Decoding, FlashInfer). Vade keeps the same softmax and value sums as the single-chat path.
- **Prefix-shared attention.** Common K and V tiles are staged once for independent chats. This extends Vade's own decode and verify reduction kernel and its page aliases.
- **Matrix kernels.** They start from Vade's own Rust matrix-vector and GEMM code. The ideas they use were studied in TensorFold and in JJ Eaton's roofline analysis (wide neighboring loads, rows shared by a warp), the LLVM NVPTX intrinsic definitions and NVIDIA's PTX MMA fragment layouts. Batched scheduling, causal attention and storage are Vade's own Rust.
- **Prefill tiles.** They use PTX's background L2-only copies and native BF16 MMA. The 256-row block, the resident tiled-weight addressing and the fused SwiGLU epilogue are Vade's own.
- **On-device decode.** TensorFold and Victor Cruz showed the idea: keep the whole decode pass and the KV cache on the device, keep activation buffers, look up kernels once, and queue launches until the output is needed. Earlier Vade measurements also said: favor weight bandwidth, and remove host waits and allocations from decode.
- **Checkpoint loading.** Earlier Vade measurements on the Spark showed that CUDA copies from mmap-backed pageable pages are slow. The loader reads into pinned or unified memory with its own Rust `pread` code, bounded staging and device arenas, with no host copy of the weights. The CUDA Driver API and CUDA for Tegra documents define the memory access rules.
- **Batched recurrences.** TensorFold's per-stream state-pointer tables informed batching independent Flash-Next GDN recurrences. Vade extends its own Rust kernels and keeps each session's arithmetic and private state.
- **Tensor parallelism.** It uses the standard Megatron split: Q, K and V split by head, gate and up split by row, O and down split by column, and sums in fixed rank order. Direct RC SEND and RECV, receives posted in advance, and striped links are standard verbs practice. The declarations follow the rdma-core headers.
- **Several sessions.** Projection rows are shared across unrelated chats, while attention and KV stay per session. Sequence slices share fixed weights and exchange FP32 KV.

Drafting and sampling:

- **Prompt lookup.** It is Apoorv Saxena's Prompt Lookup Decoding. We first saw it, and position-keyed sampling, in TensorFold and in Victor Cruz's work. No outside model is used.
- **Draft-model decoding.** A smaller official Qwen3 model proposes a chain. The target checks the rows, and only its matching prefix commits. For sampled requests, the draft proposes greedily, and the target's own position-keyed sample decides every token. So drafting never changes a sampled output. Proposal, KV rollback, tied-weight views and acceptance are Vade's own.
- **Position-keyed sampling.** Vade's own design: a counter keyed by each token's position, so a draft step can be replayed exactly.
- **Conversation branches.** TensorFold's snapshots and independent branches at message boundaries. Vade builds them on the engine's own fork and truncate API.
- **Qwen3.8-27B MTP.** The installed Transformers `qwen3_5` source ignores the `mtp.*` tensors. It gives the decoder, norm and RoPE precision rules, but no MTP forward pass. Transformers issue 48913 records this. VeOmni's MTP note supplies the fusion order. No other inference engine's source was read for this. The target model's check keeps the output exact.
- **Flash-Next MTP fusion.** It follows the model meaning in SGLang's installed `qwen4_exp_mtp.py`: whole-residual normalization, one shared projection per residual stream, and a broadcast projected token embedding. Vade uses its own sharded weights, attention and MoE kernels.

Models and number formats:

- **Qwen3.8-27B layers.** The BF16 operation boundaries follow Transformers 5.18.0's Qwen3_5 model: BF16 beta and gated norm, SiLU for GDN and sigmoid for full attention, even though the checkpoint also carries `output_gate_type=swish`.
- **Gated delta chunks.** The scan follows the official Transformers 5.18.0 chunk arithmetic: FP32 beta-scaled products, pairwise decay, then a unit-lower triangular solve. Vade keeps its own blocked inverse order on chip. Direct substitution was rejected because it changed greedy IDs. Recurrent state stays in FP32.
- **Flash-Next layers.** Hyper-connections, n-gram embeddings, the chunk-64 delta rule, indexed attention and the sharded MLP follow the official Qwen4Exp source cited in [the Flash-Next architecture note](../engine/docs/history/flash-next-architecture.md). The router is a full softmax with the top 10 experts, summed in ascending expert order. Rust kernels, ownership and transport are Vade's own.
- **NVFP4.** GB10 primitives follow NVIDIA's public PTX block-scale instruction and fragment rules. TensorFold's native target and packed FP4 MMA choice informed the approach. Pinned links, layouts and limits are in [FP4 kernels](../engine/docs/fp4-kernels.md).
- **Gemma 4 text.** The GPU arithmetic follows Google's official Transformers attention, RMSNorm, decoder layer and proportional RoPE behavior. They were read only to fix behavior and to compare tokenization and template output.
- **FP8 GEMM.** It follows the checkpoint's dynamic E4M3 activations in per-token groups of 128 and its BF16 128 × 128 weight scales. Vade extends its own m16n8k32 Rust and PTX GEMM. The arithmetic is these steps, in order:
  1. Round each input to BF16.
  2. Find the largest absolute value in each group. Set the scale to that value / 448.
  3. Divide by max(scale, 1e-12).
  4. Convert to E4M3 with round-to-nearest-even and saturation.
  5. Form an FP32 dot over 128 values from four ascending 32-value MMAs.
  6. Multiply that dot by the activation scale times the weight scale, and add it to the FP32 sum in ascending order.
  7. Round only the output to BF16.
- **The FP8 reference run.** The separate reference setup installs Hugging Face kernels 0.17.2 and finegrained-fp8 revision `8c178950e8710e26a2210e4e909cd02dc16c8715`. Its quantizer uses the same max / 448 rule. Transformers 5.18.0 has one bug here: its exclusion pattern is not anchored, so `mlp.gate_proj` matches `mlp.gate` and 64 checkpoint scale tensors are dropped. The reference run anchors the names before loading. The model and the official kernels stay unchanged.
- **Flash-Next W8A8.** It follows Transformers 5.18.0 `FP8Experts` and the same finegrained-fp8 revision. Details are in [Flash FP8 math](../engine/docs/flash-fp8-math.md).
- **Row rejection.** Outside Numerics advisory drafts dated 2026-10-10 ("Router NaN per-row status" and "BF16 NaN-safe rounding helper") informed the row-isolation and shared-conversion rules. The code is Vade's own Rust. No advisory code was copied. The BF16 NaN value 0x7FFF follows Vade's GB10 measurements.
- **MoE locality.** A Vade GEMM review recommended that routed FP8 expert launches put row planes next to each other within a tile, to encourage L2 weight reuse. Only launch indexing changes. CUDA scheduling does not guarantee the reuse.

Earlier Vade work carried over:

- Shared GQA8 attention (8e9657e), and the two-stage 256 × 128 × 64 prefill GEMM with wide QKV (7ae8d3d, 570d94d).
- The m32 64 × 32 warp tiles (c10db29), and the grouped expert gate and up tile from `work/m32` (c10db29) and `work/gemm` (8b2a874), used in the MoE work (dec2a09).
- The dense and expert GQA8 body from 59b8a3c, and the GQA4 port in 2de67a9.
- The FP8 attention work from f0c36c8.
- Packed BF16 background staging and CUDA graph replay from d4e74a7 and 4b7d54e.
- Earlier, parked Flash-Next model metadata and hyper-connection algebra.

<br>

---

## Harness

**This section names the ideas that the harness studied, and their sources.**

- **What Vade built.** `vade chat` opens a native coding agent ([harness](../harness/README.md)). The `vade` CLI installs, starts, checks and stops the engine ([cli](../cli/README.md), [What you get](../README.md#what-you-get)). On Linux, the Spark included, the agent runs no shell commands. Its shell tool needs the macOS sandbox ([What runs where](../README.md#what-runs-where)).
- **Client setup.** For Claude Code, Codex, Hermes, OpenCode and Pi, client setup follows the pinned command-line help and configuration docs of each client. The Hermes context exception and file-write limits come from its pinned initialization and file-safety modules. The templates, the model catalog prompt and the instruction to call a tool right away instead of promising to call it are Vade's own Rust output. No client code or prompt was copied. The source links and pins are in [the client setup guide](../harness/docs/passthrough-matrix.md#sources-and-credit).
- **Terminal presentation.** We compared Claude Code and Codex CLI. Then we set three clarity goals for the native terminal: a quiet answer footer, a highlighted command picker and readable Markdown. The renderer, the cell layout and the PTY regression tests are original Rust that uses only the standard library. No competitor source, assets or dependencies were used.
- **Context compaction.** Stable-prefix compaction builds on the harness's own session journal and the points where it can safely cut history. Prefix continuity and rebuilding the history after the cut follow the ideas of Codex, Pi and Hermes ([context controls](../harness/docs/context-and-projects.md)). The prefix builder, the admission coordination and the optional warm-up of discarded history are original Rust. Tokenization and cache identity stay with the engine. No tokenizer code, competitor code or cryptography was imported.
- **Design references.** Hermes Agent, the Pi coding agent and Codex CLI informed the interface choices of the harness. They are references for ideas, not dependencies. See [Design references](../harness/README.md#design-references) in the harness README.

**Design review.** The harness design review also studied stable prefixes, continuation identity, session trees, cancellation and isolated child contexts. It used these pinned sources:

- [Codex](https://github.com/openai/codex/blob/2fdf047c9631c9ed01a31b62efb7891718a931a8/README.md)
- [Pi](https://github.com/earendil-works/pi/blob/ce950d78f424dcaf9f5d6a03ce80ab141130eb1d/README.md)
- [Hermes](https://github.com/NousResearch/hermes-agent/blob/25a71a744cb9ef06950a91638e6229b4f808d461/README.md)
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/5badb15009ae1756c3afe0ae0cef1faafc290ccc/README.md)
- [OpenCode](https://github.com/anomalyco/opencode/blob/5a8c1d81b7ba527e580191e1d9638c09c6f79419/README.md)
- [Aider](https://github.com/Aider-AI/aider/blob/5dc9490bb35f9729ef2c95d00a19ccd30c26339c/README.md)
- [Goose](https://github.com/aaif-goose/goose/blob/9560429ff982bbfecec5094963e5f34696b63a1c/README.md)
- [Claude Code](https://code.claude.com/docs/en/how-claude-code-works)

These were studies of source code and documentation. They were not interoperability measurements.

<br>

---

## Tooling and docs

### Compiler and standards

**The compiler and system tools Vade is built with, and the file and wire formats it implements.**

Compiler and system:

- **[Rust](https://github.com/rust-lang/rust), [NVPTX target](https://doc.rust-lang.org/rustc/platform-support/nvptx64-nvidia-cuda.html), [rustup](https://rustup.rs).** The compiler, Cargo, the standard library and GPU `core` support. External build tools, not Vade code.
  - [GitHub](https://github.com/rust-lang)
  - [Site](https://www.rust-lang.org)
- **[LLVM NVPTX](https://llvm.org/docs/NVPTXUsage.html), [NVVM intrinsics](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsNVVM.td).** Rust's GPU backend, and the exact names and arguments of GPU intrinsics and tensor-core register layouts.
  - [GitHub](https://github.com/llvm)
  - [Site](https://llvm.org)
- **[NetworkManager](https://networkmanager.dev), [CPU idle states](https://docs.kernel.org/admin-guide/pm/cpuidle.html).** The format of the network plan that Vade prints, and CPU wake-up latency research. Vade makes no network or OS changes.

Formats and standards:

- **Unicode Consortium: [Unicode Character Database](https://www.unicode.org/ucd/), [license](https://www.unicode.org/license.txt).** Character categories and normalization tables. Versions are recorded in [Provenance](../engine/docs/provenance.md).
- **OpenAI API contributors: [OpenAI API specification](https://github.com/openai/openai-openapi).** Chat and text completion request and reply shapes. Supported fields are in [Server and API](../engine/docs/server.md).
- **IETF and WHATWG contributors: [JSON, RFC 8259](https://www.rfc-editor.org/rfc/rfc8259), [HTTP/1.1, RFC 9112](https://www.rfc-editor.org/rfc/rfc9112), [Server-Sent Events](https://html.spec.whatwg.org/multipage/server-sent-events.html).** Wire formats in Vade's own HTTP parser and streaming writer.
- **NIST and IETF contributors: [SHA-256, FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final), [HMAC, RFC 2104](https://www.rfc-editor.org/rfc/rfc2104), [RFC 4231 test vectors](https://www.rfc-editor.org/rfc/rfc4231).** Hashing, and authenticated control messages between Sparks.

### Reused data

**A few files are kept unchanged, as data, with their licenses.**

- Qwen3-8B's configuration, weight index and chat template (Apache-2.0).
- Unicode character tables (Unicode License V3).
- Gemma 4's canonical chat template, used to check model folders.
- Reference outputs in `fixtures/`, made by running the official models in Hugging Face Transformers.

[Provenance](../engine/docs/provenance.md) lists the files, checksums and versions. [NOTICE](../NOTICE) and the [engine NOTICE](../engine/NOTICE) carry the attributions.

Vade also declares the CUDA driver and RDMA functions it calls, written against NVIDIA's `cuda.h` and rdma-core's `verbs.h`. These are only names, types and constants, as any program that calls these libraries must declare. Vade's reuse of its own earlier kernels is not an outside source, so it is listed only under [Where each idea is used in the engine](#where-each-idea-is-used-in-the-engine).

<a id="corrections"></a>

### Link checks and corrections

Every profile link on this page was checked against the person's own GitHub or Hugging Face profile. [Link checks](../engine/docs/link-checks.md) records how.

A credit names the original source of an idea and, if different, the project where we first saw it. An earlier original source is always welcome. To report one, open an issue in the repository's issue tracker.

<br>

---

## Terms used on this page

**How entries are written:**

- Each entry says which idea or behavior Vade uses. The lines under it name the people or give the links.
- A credit does not mean that their code runs in Vade, or that their speed or correctness claims apply to it.
- Papers list every author when there are nine or fewer. Longer lists show the first three names and the total. The paper link has them all.
- A person's profile links appear once, at their first entry. Later entries say "links above".
- An X account or website appears only when the person's own profile lists it. An author with no links is credited by name and paper only.
- "Development only" means the work is outside main. "Main" is the merged code on the main branch.

**Terms:**

- **BF16, FP32.** 16-bit and 32-bit floating-point formats.
- **E4M3, E2M1.** 8-bit (FP8) and 4-bit (FP4) floating-point formats.
- **NVFP4.** NVIDIA's 4-bit format: E2M1 values with an E4M3 scale per 16 values and one FP32 global scale.
- **W4A16, W4A4.** 4-bit weights with 16-bit activations, and 4-bit weights with 4-bit activations.
- **PTX.** NVIDIA's GPU assembly language, which the driver compiles for the chip.
- **MMA.** A tensor-core matrix multiply instruction.
- **CTA.** A block of GPU threads that runs together and shares on-chip memory.
- **Tile.** A small block of a larger matrix, sized to fit in fast on-chip memory.
- **KV cache.** The attention keys and values kept for every token already read: the model's memory of the conversation.
- **GQA.** Grouped-query attention: several query heads share one key and value head. GQA8 means eight query heads per KV head, GQA4 four.
- **MTP.** Multi-token prediction: a built-in head that guesses tokens ahead.
- **GDN.** GatedDeltaNet: a linear-attention layer that keeps a fixed-size recurrent state instead of a growing history.
- **HC.** Hyper-connection: a learned mix that keeps four residual streams and decides how much each block adds to each stream.
- **PLE.** The n-gram embedding: an extra embedding of hashed pairs and triples of recent tokens, added before decoder 1's attention.
- **QSA.** Indexed attention: softmax attention limited to the token blocks that a small indexer selects.
- **packed8.** Vade's layout for BF16 weights: tiles of eight rows, arranged so each GPU lane loads its matrix-multiply fragment directly.
- **Canonical.** Vade's fixed reference arithmetic, with a set order of operations. A faster path must give the same bits.
- **Two-component.** An FP32 value carried as two BF16 numbers: the rounded value and its rounded remainder.
- **Research engine.** The Flash-Next engine path for Qwen3.8-Flash-Next-FP8. It began as a separate two-Spark engine. Today `flash-serve` serves FP8 on 2, 4 or 6 Sparks.

<br>

---

## Pinned sources

**The exact versions that settled each detail.**

Model code:

- [Transformers Qwen3, v4.51.3](https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/models/qwen3/modeling_qwen3.py): Qwen3 layer behavior used for the earlier fixtures.
- [Transformers Qwen3.5 text, 5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py): for Qwen3.8-27B, the hybrid layer schedule, zero-centered norms, partial rotary positions, BF16 gates, and the linear-attention recurrence with its FP32 chunk equations.
- [Transformers Flash-Next](https://github.com/huggingface/transformers/blob/9d8c3a24d434d4a958a23562f576126924358b7f/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py): research engine only. Hyper-connections, n-gram and per-layer embeddings, the chunk-64 delta rule, indexed attention and pooling, full-softmax top-10 expert routing, ascending expert sums and scaled BF16 weight conversion.

Precision and settings:

- [Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8), revision `017b9c7a` as recorded at download: E4M3 weights, BF16 128 × 128 block scales, and the list of modules left unconverted.
- [Transformers fine-grained FP8, 5.18.0](https://github.com/huggingface/transformers/blob/v5.18.0/src/transformers/integrations/finegrained_fp8.py): decode E4M3 weights, multiply by the checkpoint's scales in FP32, then round to BF16. FP8 weights stay resident.
- [finegrained-fp8, pinned recipe](https://huggingface.co/kernels-community/finegrained-fp8/blob/3c8fcc86c0e42b7abe5bcf666d3a3afdce93a0e8/build/torch-cuda/utils.py#L440): the per-128 activation maximum and the max / 448 scale.
- [Transformers issue 48913](https://github.com/huggingface/transformers/issues/48913): records that the installed reference forward pass ignores Qwen3.8-27B's built-in MTP tensors.
- [Transformers RoPE definitions](https://huggingface.co/docs/transformers/main/en/internal/rope_utils): the YaRN parameters behind Vade's optional Qwen3 context extension.

<br>

- **Back:** [README](../README.md)
- **Next:** [Documentation index](README.md)
