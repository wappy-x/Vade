# Glossary

**Every term used across this repository, in plain words.** Listed A to Z. Each entry is one or two sentences.

Not here? The engine has its own term boxes at the top of each page in [`engine/docs/`](../engine/docs/README.md).

**Quick picks for newcomers:** [Token](#sz) · [Prefill and decode](#pr) · [KV cache](#ho) · [Tensor parallelism](#sz) · [RDMA and RoCE](#pr) · [QSFP and rails](#pr) · [MTU](#ho)

<br>

## A–C

**aarch64 (Arm64).** The processor family of the Spark's CPU. Linux reports it with `uname -m`.

**Allowlist.** A list of the only files allowed to be copied. Anything not named stays behind.

**API (application programming interface).** The agreed format two programs use to talk. Vade's engine speaks the OpenAI-style chat API.

**Apache License 2.0.** A permissive open-source license: use, change and share the code, keep the notices.

**Bandwidth, Gb/s and GB/s.** How much data a link moves per second. Lowercase **b** is bits, uppercase **B** is bytes: 196 Gb/s ≈ 24.5 GB/s.

**Batching.** Running several requests through the model together, so the GPU does more useful work per step.

**BF16 (bfloat16).** A 16-bit number format for model weights. It keeps the range of 32-bit numbers with less precision.

**Bundle.** A prepared set of engine, GPU kernel and harness artifacts with a manifest and checksums, used instead of rebuilding locally. The launcher reads `launcher-bundle.conf`; see [bundle loading](../cli/src/launch/install.rs) and [controller prebuilt bundles](../cli/docs/prebuilt.md).

**Byte-identical.** Exactly the same, down to every byte of the answer text.

**Cache, prompt.** Saved work from reading a prompt. If a new request starts the same way, that part is not read again.

**Cargo.** Rust's build tool and package manager. `cargo build` compiles; `cargo run` compiles and runs.

**Chain.** A layer pipeline across N Sparks, each owning a contiguous range of layers and passing intermediate state onward. More than two Sparks require the `VADE_W2_PP=1` opt-in, which the controller sets for `vade up --fp4-local --sparks N`; see [architecture](architecture.md#ranks-memory-and-ownership) and [controller selection](../cli/src/app.rs).

**Checkpoint.** One published set of a model's files: weights, configuration and tokenizer, at one exact revision.

**Chunked prefill.** Reading a long prompt in pieces instead of all at once, so other chats are not stuck waiting.

**CLI (command-line interface).** A program you use by typing commands in a terminal, such as `vade` or `vade-harness`.

**Client-measured.** Timed by the program sending the requests, the way an app experiences it.

**Cohort, C1..C16.** A group of concurrent request streams in a benchmark: C1 means one stream, C8 means eight, and C16 means sixteen. These are requested concurrency levels, not a promise that every stream occupies a GPU slot at once; see [benchmark definitions](benchmarks.md).

**ConnectX-7 (CX7).** NVIDIA's high-speed network chip inside each Spark. It provides the two QSFP ports and RDMA.

**Context window.** The most tokens a model can consider at once: the prompt plus the answer so far. The configured limit depends on the launch route and available memory; see [installation settings](../INSTALL.md).

**Contract.** The agreed rules two parts follow when they talk. The engine and the harness share one: what the engine can do, whether it has room for a request, and how its prompt cache is reused. See the [harness engine contract](../harness/docs/engine-contract.md).

**Crate.** One Rust package: a library or a program. The engine is a workspace of several crates.

**CUDA, CUDA Driver API, CUDA toolkit.** CUDA is NVIDIA's way of running programs on its GPUs. The Driver API is the low-level layer every NVIDIA GPU system already has. The toolkit is NVIDIA's separate compiler and libraries; Vade does not need it.

<br>

## D–G

**DAC (direct-attach copper).** A cable with QSFP plugs fixed to both ends of a copper cable. Cheap and reliable for short runs between neighbors.

**Decode.** Writing the answer, one token per step, after the prompt has been read. It sets the writing speed (tokens per second).

**Denylist.** A private list of names that must never appear in an exported file. It is kept outside every repository.

**Determinism.** Same input, same output, every time, however the request is run.

**Determinism gate.** The engine's check that runs the same requests alone, cached, chunked and batched, and requires byte-identical answers.

**DGX OS.** NVIDIA's Ubuntu-based operating system for DGX machines, with the GPU driver and network drivers included.

**DGX Spark.** NVIDIA's desktop AI computer, built around the GB10 chip, with 128 GB of memory shared by CPU and GPU.

**Draft model, speculative decoding.** A small, fast model guesses a few tokens ahead and the main model checks them all in one step. Vade keeps only guesses the main model agrees with, so answers do not change.

**Dry run.** A preview: the command prints what it would do and changes nothing. Every `vade` lifecycle command accepts `--dry-run`.

**Endpoint.** The web address a program sends requests to, such as `http://localhost:18080/v1/chat/completions`.

**Fabric.** The private QSFP network between Sparks, separate from the ordinary LAN. In the Vade config, `fabric.<peer>` is a machine's own address on its cable to that peer.

**flash-serve / next-http / next-tp.** `flash-serve` is the current HTTP server for FP4-Local, including world 1 and the layer pipeline; `next-http` is a compatibility launcher for the separate tensor-map rank path, and `next-tp` captures native rank qualification cases. See their [serving](../engine/crates/tp/src/bin/flash-serve.rs), [compatibility](../engine/crates/tp/src/bin/next-http.rs) and [qualification](../engine/crates/tp/src/bin/next-tp.rs) entry points.

**FP4-Local.** The current one-Spark release profile, using `nvidia/Qwen3.8-Flash-Next-FP4-Local` and checkpoint-declared mixed NVFP4, MXFP8 and plain tensor storage. “Local” does not rule out the layer pipeline across multiple Sparks; see [architecture](architecture.md#what-the-current-flash-next-path-runs).

**FP8, FP4 (NVFP4).** 8-bit and 4-bit number formats. Smaller weights need less memory and move faster, at some cost in precision.

**Gateway.** The harness mode that relays requests from other coding tools to one or more engines. With several engines it acts as a router.

**GB10.** The chip inside the DGX Spark: an NVIDIA Blackwell GPU and an Arm CPU in one package.

**GID (global identifier).** An address entry a RoCE device uses to reach its peer. Each IPv4 address on a fabric interface gets a "RoCE v2" GID; Vade finds the right one automatically.

**Golden.** A saved reference answer, with its full logits, from a qualified build at one setup. Exactness checks compare new runs with it byte for byte. Vade keeps one set per Spark count.

**GPU (graphics processing unit).** The processor that does the model's heavy math, many small calculations at once.

**GPU kernel.** A small program that runs on the GPU, such as one matrix multiply. Vade writes its own, in Rust.

**Greedy decoding.** Always taking the model's single most likely next token. This is FP4-Local's default when the request names no sampling fields; see the [request parser](../engine/crates/serve/src/protocol.rs).

<br>

## H–O

**Harness.** The program around a model that turns it into a working assistant: the interface, the tools, the conversation history.

**HTTP.** The ordinary protocol of the web. The engine listens for HTTP requests on a port.

**Hugging Face.** The website where model makers publish their weights. Its `hf` command-line tool downloads them.

**ib_write_bw, perftest.** `perftest` is the standard RDMA test kit; `ib_write_bw` in it measures RDMA write bandwidth between two machines.

**Inference.** Using a trained model to produce output. Training builds the model; inference runs it.

**Inference engine.** The program that loads a model's weights and runs inference efficiently, often for many users at once.

**Jumbo frames.** Network packets larger than the usual 1,500 bytes; here, MTU 9000. Fewer, bigger packets move data faster.

**KV cache.** The model's working memory for one conversation: saved results for every token read so far, so they are not recomputed.

**Launcher.** The on-Spark `vade launch` workflow that finds an existing checkpoint, prepares local artifacts, starts or attaches to its owned engine and opens chat. See [first run](../harness/docs/first-run.md) and the [launcher implementation](../cli/src/launch.rs).

**LLM (large language model).** A model trained on large amounts of text to predict the next token. Chat assistants are built on them.

**Locked memory (memlock).** Memory the OS promises never to move or swap out. RDMA needs it; Vade requires the limit to be `unlimited` in the session that starts it.

**Loopback, localhost.** The address a machine uses to talk to itself (`127.0.0.1`). The launcher binds HTTP there; the configured controller route uses an SSH tunnel to reach the engine from another computer.

**Mixture of experts (MoE).** A model made of many small expert networks; each token visits only a few of them.

**Model plan.** Vade's typed description of a checkpoint, read from `config.json` and the safetensors headers: layers, sizes and each tensor's role. See [Model plan](model-plan.md).

**Monorepo.** One repository holding several projects side by side, like `engine/` and `harness/` here.

**MTP (multi-token prediction), drafting ON.** An extra model head proposes tokens that the target model verifies: FP4-Local defaults to strict drafting and request depth 3, or a request can name `"draft":"strict"`. The adaptive chooser can reach resident depth 5 on one Spark or 3 on the pipeline; prompt lookup and world-one cohort drafting are also enabled by default, while actual use follows the [drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs) and [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**MTP, drafting OFF.** `"draft":"plain"` selects ordinary decoding for that request; `"mtp_depth":0` does the same when no `draft` mode is named, and server `--mtp 0` sets the plain default. Under greedy decoding, ON and OFF must return the same token IDs; see [request controls](../engine/crates/serve/src/protocol.rs), [server defaults](../engine/crates/tp/src/flash/next_batched_service.rs) and [`vade verify`](../cli/src/verify/mod.rs).

**MTU (maximum transmission unit).** The largest packet a network interface sends. Vade's fabric uses 9000 at both ends of every cable; a mismatch drops large packets.

**Netplan.** Ubuntu's network configuration system. The OS-setup scripts write one netplan file for the fabric interfaces.

**Official weights.** Model files exactly as the maker published them, with no community changes.

**OpenAI-compatible API.** Requests and replies in the format OpenAI's API uses. Most chat apps can connect with just a base-URL change.

<br>

## P–R

**Path (topology).** Ranks connected with two open ends rather than a closed ring. An FP4-Local chain relays work through its stages in order; see [pipeline stage geometry](../engine/crates/engine/src/flash/next_pp.rs).

**Poison.** Marking execution state unusable after a partial failure, so later work cannot silently continue from incomplete state. The [native rank session](../engine/crates/tp/src/next.rs) stays failed until a collective reset succeeds.

**Prefill.** Reading the whole prompt once, before the first answer token. It sets the time to first token.

**Prompt.** Everything sent to the model for one turn: instructions, conversation so far, and the new question.

**PTX.** NVIDIA's portable instruction format for GPU code. The driver turns it into machine code for the exact GPU.

**QSFP.** The high-speed network ports and cables that join Sparks directly to each other. Each Spark has two QSFP ports; Vade uses 200 Gb/s cables.

**Quantization.** Storing weights in fewer bits (such as FP8 or FP4) to save memory and time.

**Rail.** One half of a QSFP port. Linux shows each port as two network interfaces on separate PCIe paths; each rail carries about half the cable's speed. Vade uses both when each rail has its own subnet.

**Rank.** One process in a group that shares a job, one per Spark. **Rank 0** is the one that answers HTTP.

**Ratchet.** A repository check that rejects growth beyond recorded limits, including external dependencies, large Rust files and branches tied to specific box counts. See the [ratchet test](../cli/tests/ratchet.rs) and its [baseline](../cli/ratchet-baseline.txt).

**RDMA (remote direct memory access).** One computer writes straight into another's memory over the network, without going through the operating system.

**Receipt.** A saved record of a run's inputs, identity, result and measured evidence; its scope is only the checks that run performed. See the [`vade verify` receipt writer](../cli/src/verify/mod.rs) and [benchmark definitions](benchmarks.md).

**Replica.** A complete, independent copy of the model serving its own requests, on one Spark or a group of Sparks. A router can spread users across several replicas.

**RigMark.** A benchmark run and its recorded results; a published card states the recipe, measurements and qualification of that run. Timing-only runs are not cards; see [Benchmarks](benchmarks.md).

**Ring (topology).** Ranks connected in a loop, each passing data to the next. This remains part of the legacy rank transport; the current FP4-Local pair defaults to a layer pipeline.

**RoCE (RDMA over Converged Ethernet).** RDMA carried over ordinary Ethernet. The Spark's QSFP links use RoCE v2, which travels inside UDP on port 4791.

**Router.** A program that receives requests and forwards each one to one of several servers. In Vade, that is the harness gateway with several `--upstream` engines.

**Rust.** A programming language that is fast like C and checks memory safety when the code compiles.

<br>

## S–Z

**Safetensors.** The file format of the model's weights: plain tensors with a small header, safe to load.

**Same-setup exact.** A label for measured numbers: one machine setup, one set of model files, one set of settings, and answers checked byte-identical across the compared runs.

**Sampling.** Picking the next token with some randomness, controlled by settings like temperature. Turning it off gives greedy decoding.

**Scrub.** Removing private details from files before they are shared.

**Splitter.** The planner that decides which Spark owns which part of each layer, what the Sparks must add up or gather, and whether each one has the memory. See [Splitter](splitter.md).

**SSH alias.** A short name in `~/.ssh/config` that stands for a machine's address, user and key. The Vade config's `ssh` entries may use these aliases or a plain `user@host`.

**SSH tunnel.** A private pipe through SSH that makes a port on a remote machine appear on your own. The configured controller's `vade up` opens one to rank 0.

**Streaming.** Sending the answer as it is written, token by token, instead of all at the end.

**Subnet, /24.** A block of network addresses that share a prefix. A `/24` such as `10.77.0.0/24` holds addresses `10.77.0.1` to `10.77.0.254`; each fabric rail gets its own.

**Sum order.** The fixed order in which partial results from different Sparks are added. Floating-point addition rounds, so another order can change the answer's bytes.

**Tensor parallelism (TP).** Splitting a layer's tensors across several GPUs so they work on a token together. FP4-Local's default two-Spark layer pipeline instead assigns whole layer ranges to stages; see [architecture](architecture.md#ranks-memory-and-ownership).

**Time to first token (TTFT).** Seconds from sending a request to receiving the first piece of the answer.

**Token.** The small unit of text a model reads and writes, often a word or part of a word.

**Tokenizer.** The part that turns text into token numbers and back.

**Tokens per second (tok/s).** How fast answer text comes out.

**Tool calls.** The model asks the harness to run a tool, such as reading a file, and gets the result back.

**VADE_W2_PP pipeline.** The default two-Spark FP4-Local execution path, using the world-one engine's arithmetic in separate layer stages, with draft verification across the stages. `VADE_W2_PP=0` selects the tensor-map rank alternative; explicit `=1` opts larger groups into a chain, as defined by [pipeline selection](../engine/crates/tp/src/flash/settings.rs).

**Weights.** The learned numbers that make up a model. Loading a model means loading its weights into memory.

**Workspace.** In the harness: the one folder an agent may read and change. In Cargo: a group of crates built together.

**YaRN.** A method that stretches a model's context window beyond what it was trained for.

<br>

**Back:** [Architecture](architecture.md) · **Next:** [Benchmarks](benchmarks.md) →
