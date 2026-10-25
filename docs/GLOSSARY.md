# Glossary

Plain-word definitions of the terms used across this repository, from A to Z.
A link to a term lands on the letter section that holds it; scan down for the
bold name.

Missing a term? Each page in [`engine/docs/`](../engine/docs/README.md) starts
with a box that explains its own engine terms.

**New to model serving? Start with these:** [Token](#sz) · [Prefill](#pr) and [decode](#dg) · [KV cache](#ho) · [Drafting](#dg)

**Joining two or more Sparks? Then these:** [Layer pipeline](#ho) · [Tensor parallelism](#sz) · [RDMA and RoCE](#pr) · [QSFP and rails](#pr) · [MTU](#ho)

<br>

## A–C

**aarch64 (Arm64).** The processor family of the Spark's CPU. On Linux, `uname -m` reports it.

**Allowlist.** A list of the only files that may be copied. Anything not on it stays behind.

**API (application programming interface).** The agreed format two programs use to talk to each other. Vade's engine speaks the OpenAI-style chat API.

**Apache License 2.0.** A permissive open-source license: use, change and share the code, keep the notices.

**Bandwidth, Gb/s and GB/s.** How much data a link moves per second. A lowercase **b** means bits and an uppercase **B** means bytes, so 196 Gb/s ≈ 24.5 GB/s.

**Batching.** Running several requests through the model together, so the GPU does more useful work per step.

**BF16 (bfloat16).** A 16-bit number format for model weights. It keeps the range of 32-bit numbers but has less precision.

**Bundle.** A ready-made set of engine, GPU kernel and harness artifacts, with a manifest and checksums. You use it instead of rebuilding locally. There are two formats. The launcher bundle (`vade-launch-bundle-v1`, with `launcher-bundle.conf`) is for running on one Spark; see [bundle loading](../cli/src/launch/install.rs). A controller's `vade install` reads only the separate `vade-prebuilt-v1` format; see [controller prebuilt bundles](../cli/docs/prebuilt.md#prebuilt-binaries-for-a-controller-install).

**Byte-identical.** Exactly the same, down to every byte of the answer text.

**Cache, prompt.** Saved work from reading a prompt. When a new request starts the same way, that part is not read again.

**Cargo.** Rust's build tool and package manager. `cargo build` compiles; `cargo run` compiles and runs.

**Chain.** A [layer pipeline](#ho) stretched across more than two Sparks. Each Spark owns a block of consecutive layers and passes the intermediate state on to the next. Groups of more than two Sparks need the `VADE_W2_PP=1` opt-in, which the controller sets for you when you run `vade up --fp4-local --sparks N`. See [architecture](architecture.md#ranks-memory-and-ownership) and [controller selection](../cli/src/app.rs).

**Checkpoint.** One published set of a model's files (weights, configuration and tokenizer) at one exact revision.

**Chunked prefill.** Reading a long prompt in pieces instead of all at once, so other chats are not stuck waiting.

**CLI (command-line interface).** A program you use by typing commands in a terminal, such as `vade` or `vade-harness`.

**Client-measured.** Timed by the program that sends the requests, which is how an app experiences it.

**Cohort, C1..C16.** A group of request streams running at the same time in a benchmark. C1 means one stream, C8 means eight and C16 means sixteen. These are the concurrency levels requested. They do not promise that every stream holds a GPU slot at once; see [benchmark definitions](benchmarks.md).

**Cohort drafting.** [Drafting](#dg) for several chats at once. The engine drafts for up to eight chats together and checks all their guesses in one shared pass, and each chat still gets the same bytes it would get running alone. It is on by default on one Spark and on the two-Spark layer pipeline; see the [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**ConnectX-7 (CX7).** NVIDIA's high-speed network chip inside each Spark. It provides the two QSFP ports and RDMA.

**Context window.** The most tokens a model can consider at once: the prompt plus the answer so far. The configured limit depends on the launch route and the memory available; see [installation settings](../INSTALL.md).

**Contract.** The rules two parts agree on when they talk to each other. The engine and the harness share one. It covers what the engine can do, whether it has room for a request, and how its prompt cache is reused. See the [harness engine contract](../harness/docs/engine-contract.md).

**Controller.** A computer that drives the Sparks over SSH with the `vade` CLI. It can be a second Linux or macOS computer, or a Spark acting as its own controller; see [Route B](../INSTALL.md#route-b-from-a-controller).

**Crate.** One Rust package, either a library or a program. The engine is a workspace of several crates.

**CUDA, CUDA Driver API, CUDA toolkit.** CUDA is NVIDIA's way of running programs on its GPUs. Two parts of it come up in these docs:

- **Driver API:** the low-level layer every NVIDIA GPU system already has.
- **Toolkit:** NVIDIA's separate compiler and libraries. Vade does not need it.

<br>

## D–G

**DAC (direct-attach copper).** A copper cable with QSFP plugs fixed to both ends. It is cheap and reliable for short runs between neighbors.

**Decode.** Writing the answer, one token per step, after the prompt has been read. It sets the writing speed (tokens per second).

**Denylist.** A private list of names that must never appear in an exported file. It is kept outside every repository.

**Determinism.** Same input, same output, every time, however the request is run.

**Determinism gate.** The engine's check for determinism. It runs the same requests alone, cached, chunked and batched, and passes only if the answers are byte-identical.

**DGX OS.** NVIDIA's Ubuntu-based operating system for DGX machines, with the GPU and network drivers included.

**DGX Spark.** NVIDIA's desktop AI computer, built around the GB10 chip, with 128 GB of memory shared by CPU and GPU.

**Drafting, speculative decoding.** Something cheap guesses a few tokens ahead, and the main model checks them all in one step. Vade keeps only the guesses the main model would have picked itself, so answers do not change.

- **Flash-Next** has no separate draft model. Its guesses come from the checkpoint's own MTP heads (see [MTP](#ho)) and from prompt lookup.
- **Gemma 4 31B** is the exception. It can use Google's separate assistant draft model; see [its engine code](../engine/crates/engine/src/gemma4/assistant.rs).

**Dry run.** A preview: the command prints what it would do and changes nothing. Every `vade` lifecycle command accepts `--dry-run`.

**Endpoint.** The web address a program sends requests to, such as `http://localhost:18080/v1/chat/completions`.

**Fabric.** The private QSFP network between Sparks, kept separate from the ordinary LAN. In the Vade config, `fabric.<peer>` is a machine's own address on its cable to that peer.

**flash-serve, next-http, next-tp.** Three engine programs:

- `flash-serve` is the HTTP server FP4-Local uses today, on one Spark and on the layer pipeline ([source](../engine/crates/tp/src/bin/flash-serve.rs)).
- `next-http` is a compatibility launcher for the older [tensor-map rank path](#sz) ([source](../engine/crates/tp/src/bin/next-http.rs)).
- `next-tp` runs the test cases that qualify the rank code behind that older path. It does not serve chat ([source](../engine/crates/tp/src/bin/next-tp.rs)).

**FP4-Local.** The current one-Spark release profile. It runs the model `nvidia/Qwen3.8-Flash-Next-FP4-Local`, whose checkpoint declares a mix of NVFP4, MXFP8 and plain tensor storage (see [FP8, FP4](#dg)). Despite the name, “Local” does not rule out running the layer pipeline across several Sparks; see [architecture](architecture.md#what-the-current-flash-next-path-runs).

**FP8, FP4 (NVFP4).** 8-bit and 4-bit number formats. Smaller weights need less memory and move faster, at some cost in precision.

**Gateway.** The harness mode that relays requests from other coding tools to one or more engines. With several engines it acts as a [router](#pr).

**GB10.** The chip inside the DGX Spark: an NVIDIA Blackwell GPU and an Arm CPU in one package.

**GID (global identifier).** An address entry a RoCE device uses to reach its peer. Each IPv4 address on a fabric interface gets a "RoCE v2" GID, and Vade finds the right one automatically.

**Golden.** A saved reference answer from a qualified build at one setup. It includes the full logits: the model's raw scores for each possible next token. Exactness checks compare new runs with it byte for byte. Vade keeps one set per Spark count.

**GPU (graphics processing unit).** The processor that does the model's heavy math, as many small calculations at once.

**GPU kernel.** A small program that runs on the GPU, such as one matrix multiply. Vade writes its own, in Rust.

**Greedy decoding.** Always taking the model's single most likely next token. This is FP4-Local's default when the request names no sampling fields; see the [request parser](../engine/crates/serve/src/protocol.rs).

<br>

## H–O

**Harness.** The program around a model that turns it into a working assistant: the interface, the tools and the conversation history.

**HTTP.** The ordinary protocol of the web. The engine listens for HTTP requests on a port.

**Hugging Face.** The website where model makers publish their weights. Its `hf` command-line tool downloads them.

**ib_write_bw, perftest.** `perftest` is the standard RDMA test kit. `ib_write_bw` is the tool in it that measures RDMA write bandwidth between two machines.

**Inference.** Using a trained model to produce output. Training builds the model; inference runs it.

**Inference engine.** The program that loads a model's weights and runs inference efficiently, often for many users at once.

**Jumbo frames.** Network packets larger than the usual 1,500 bytes; here, MTU 9000. Fewer, bigger packets move data faster.

**KV cache.** The model's working memory for one conversation: saved results for every token read so far, so they are not recomputed.

**Launcher.** The `vade launch` workflow you run on the Spark. It finds an existing checkpoint, prepares local artifacts, starts or attaches to the engine it owns, and opens chat. See [first run](../harness/docs/first-run.md) and the [launcher implementation](../cli/src/launch.rs).

**Layer pipeline.** FP4-Local's default way of running on two Sparks. The model's layers are split into two blocks, and each Spark runs its own block with the same arithmetic as the one-Spark engine, passing its result on to the other. Drafted tokens are checked across both stages. The same idea across more Sparks is a [chain](#ac), and the switch behind both is [VADE_W2_PP](#sz).

**LLM (large language model).** A model trained on large amounts of text to predict the next token. Chat assistants are built on them.

**Locked memory (memlock).** Memory the OS promises never to move or swap out. RDMA needs it. Vade requires the limit to be `unlimited` in the session that starts it.

**Loopback, localhost.** The address a machine uses to talk to itself (`127.0.0.1`). The launcher binds HTTP there. The configured controller route uses an [SSH tunnel](#sz) to reach the engine from another computer.

**Mixture of experts (MoE).** A model made of many small expert networks, where each token visits only a few of them.

**Model plan.** Vade's typed description of a checkpoint, read from `config.json` and the safetensors headers: its layers, sizes and each tensor's role. See [Model plan](model-plan.md).

**Monorepo.** One repository holding several projects side by side, like `engine/` and `harness/` here.

**MTP (multi-token prediction), drafting ON.** An extra head in the model guesses a few tokens ahead, and the main model checks them. On FP4-Local:

- **Default:** drafting is on, three tokens ahead (request depth 3). A request can also ask for it with `"draft":"strict"`.
- **One chat alone:** the engine adjusts the depth as it goes, up to a ceiling called the resident depth. The ceiling is 7 on one Spark and on the layer pipeline.
- **Prompt lookup:** also on by default, on one Spark and on the pipeline. It copies guesses from matching earlier text in the same chat. It runs only for greedy requests that draft, and `VADE_MTP_LOOKUP=0` turns it off.
- **Several chats:** [cohort drafting](#ac) is on too.

The older tensor-map rank path never drafts or uses lookup. The code is in [drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs) and [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**MTP, drafting OFF.** Ordinary decoding with no proposed tokens. There are three ways to choose it:

- `"draft":"plain"` selects it for one request.
- `"mtp_depth":0` does the same when the request names no `draft` mode.
- Server `--mtp 0` makes plain decoding the default.

Under greedy decoding, ON and OFF must return the same token IDs; see [request controls](../engine/crates/serve/src/protocol.rs), [server defaults](../engine/crates/tp/src/flash/next_batched_service.rs) and [`vade verify`](../cli/src/verify/mod.rs).

**MTU (maximum transmission unit).** The largest packet a network interface sends. Vade's fabric uses 9000 at both ends of every cable. If the two ends differ, large packets are dropped.

**Netplan.** Ubuntu's network configuration system. The OS-setup scripts write one netplan file for the fabric interfaces.

**Official weights.** Model files exactly as the maker published them, with no community changes.

**OpenAI-compatible API.** Requests and replies in the format OpenAI's API uses. Most chat apps can connect by changing only the base URL.

<br>

## P–R

**Path (topology).** Ranks connected in a line with two open ends, rather than a closed ring. An FP4-Local chain relays work through its stages in order; see [pipeline stage geometry](../engine/crates/engine/src/flash/next_pp.rs).

**Poison.** Marking execution state as unusable after a step fails halfway, so later work cannot quietly carry on from half-updated state. On the older [tensor-map rank path](#sz), a poisoned [rank session](../engine/crates/tp/src/next.rs) stays failed until a reset across all ranks succeeds.

**Prefill.** Reading the whole prompt once, before the first answer token. It sets the time to first token.

**Prompt.** Everything sent to the model for one turn: instructions, the conversation so far, and the new question.

**PTX.** NVIDIA's portable instruction format for GPU code. The driver turns it into machine code for the exact GPU.

**QSFP.** The high-speed network ports and cables that join Sparks directly to each other. Each Spark has two QSFP ports, and Vade uses 200 Gb/s cables.

**Quantization.** Storing weights in fewer bits (such as FP8 or FP4) to save memory and time.

**Rail.** One half of a QSFP port. Linux shows each port as two network interfaces on separate PCIe paths, and each of these rails carries about half the cable's speed. Vade uses both rails when each one has its own subnet.

**Rank.** One process in a group that shares a job, one per Spark. **Rank 0** is the one that answers HTTP.

**Ratchet.** A repository check that fails when the code grows beyond recorded limits. The limits cover external dependencies, large Rust files, and code that branches on a fixed Spark count (such as comparing the number of ranks to a literal 2). See the [ratchet test](../cli/tests/ratchet.rs) and its [baseline](../cli/ratchet-baseline.txt).

**RDMA (remote direct memory access).** One computer writes straight into another's memory over the network, without going through the operating system.

**Receipt.** A saved record of a run: its inputs, which engine build and checkpoint answered, the result and the measured evidence. It covers only the checks that run performed. See the [`vade verify` receipt writer](../cli/src/verify/mod.rs) and [benchmark definitions](benchmarks.md).

**Replica.** A complete, independent copy of the model serving its own requests, on one Spark or a group of Sparks. A router can spread users across several replicas.

**RigMark.** Our name for a benchmark run and its recorded results. A published card is the write-up of one such run: its recipe, measurements and qualification. Timing-only runs are not cards. See [Benchmarks](benchmarks.md).

**Ring (topology).** Ranks connected in a loop, each passing data to the next. Rings belong to the transport of the older [tensor-map rank path](#sz); a current FP4-Local pair runs as a [layer pipeline](#ho) by default.

**RoCE (RDMA over Converged Ethernet).** RDMA carried over ordinary Ethernet. The Spark's QSFP links use RoCE v2, which travels inside UDP on port 4791.

**Router.** A program that receives requests and forwards each one to one of several servers. In Vade, that is the harness gateway with several `--upstream` engines.

**Rust.** A programming language that is fast like C and checks memory safety when the code compiles.

<br>

## S–Z

**Safetensors.** The file format of the model's weights: plain tensors with a small header, safe to load.

**Same-setup exact.** A label you will see next to measured numbers. It means one machine setup, one set of model files and one set of settings, with answers checked byte-identical across the runs being compared.

**Sampling.** Picking the next token with some randomness, controlled by settings like temperature. Turning it off gives greedy decoding.

**Scrub.** Removing private details from files before they are shared.

**Splitter.** The planner that decides which Spark owns which part of each layer, what the Sparks must add up or gather, and whether each one has the memory. See [Splitter](splitter.md).

**SSH alias.** A short name in `~/.ssh/config` that stands for a machine's address, user and key. The Vade config's `ssh` entries may use these aliases or a plain `user@host`.

**SSH tunnel.** A private pipe through SSH that makes a port on a remote machine appear on your own. On the configured controller, `vade up` opens one to rank 0.

**Streaming.** Sending the answer as it is written, token by token, instead of all at the end.

**Subnet, /24.** A block of network addresses that share a prefix. A `/24` such as `10.77.0.0/24` holds addresses `10.77.0.1` to `10.77.0.254`. Each fabric rail gets its own.

**Sum order.** The fixed order in which partial results from different Sparks are added. Floating-point addition rounds, so another order can change the answer's bytes.

**Tensor parallelism (TP).** Splitting a layer's tensors across several GPUs so they work on a token together. FP4-Local's default two-Spark layer pipeline works differently: it gives each stage a whole range of layers (see [architecture](architecture.md#ranks-memory-and-ownership)).

**Tensor-map rank path.** The older way to run FP4-Local on several Sparks:

- Every Spark runs every layer but holds only its share of each one: whole experts, plus slices of the attention heads and channels. A tensor-map file lists each share.
- The Sparks gather each other's partial results and add them in a fixed [sum order](#sz).
- Unlike the layer pipeline, it never drafts.

`VADE_W2_PP=0` selects it on two Sparks; see [the rank backend](../engine/docs/next-http.md#native-rank-backend).

**Time to first token (TTFT).** Seconds from sending a request to receiving the first piece of the answer.

**Token.** The small unit of text a model reads and writes, often a word or part of a word.

**Tokenizer.** The part that turns text into token numbers and back.

**Tokens per second (tok/s).** How fast answer text comes out.

**Tool calls.** The model asks the harness to run a tool, such as reading a file, and gets the result back.

**VADE_W2_PP (layer pipeline).** The environment switch that picks how FP4-Local spreads over several Sparks:

- Unset on two Sparks: the [layer pipeline](#ho), the default.
- `VADE_W2_PP=0` picks the older [tensor-map rank path](#sz) instead.
- For more than two Sparks, `VADE_W2_PP=1` turns on a [chain](#ac), and `vade up --sparks N` sets it for you.

See [pipeline selection](../engine/crates/tp/src/flash/settings.rs).

**Weights.** The learned numbers that make up a model. Loading a model means loading its weights into memory.

**Workspace.** Two meanings. In the harness, it is the one folder an agent may read and change. In Cargo, it is a group of crates built together.

**YaRN.** A method that stretches a model's context window beyond what it was trained for.

<br>

**Back:** [Architecture](architecture.md) · **Next:** [Benchmarks](benchmarks.md) →
