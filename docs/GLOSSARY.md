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

**Agent (coding agent).** The mode of chat that can use tools: it reads, searches and edits files in the [workspace](#sz), and on macOS it can run commands in a [sandbox](#sz). `vade chat` opens it when the context is at least 8192 tokens; see [chat and the coding agent on Linux](../INSTALL.md#chat-and-the-coding-agent-on-linux).

**Allowlist.** A list of the only files that may be copied. Anything not on it stays behind.

**API (application programming interface).** The agreed format two programs use to talk to each other. Vade's engine speaks the OpenAI-style chat API.

**API key file.** A file that holds the secret a client sends to the engine as a bearer credential (an `Authorization: Bearer` header). Pass it with `--api-key-file PATH` or `VADE_API_KEY_FILE`, and keep the file private.

**Apache License 2.0.** A permissive open-source license: use, change and share the code, keep the notices.

**Approval.** The agent asks before it edits a file or runs a command outside its read-only list. `--approval ask` (the default) asks each time, `auto` allows checked patches and sandboxed commands, and `never` refuses them.

**Bandwidth, Gb/s and GB/s.** How much data a link moves per second. A lowercase **b** means bits and an uppercase **B** means bytes, so 196 Gb/s ≈ 24.5 GB/s.

**Batching.** Running several requests through the model together, so the GPU does more useful work per step.

**BF16 (bfloat16).** A 16-bit number format for model weights. It keeps the range of 32-bit numbers but has less precision.

**Bundle.** A ready-made set of engine, GPU kernel and harness artifacts, with a manifest and checksums. You use it instead of rebuilding locally. There are two formats. The launcher bundle (`vade-launch-bundle-v1`, with `launcher-bundle.conf`) supports a local install; see [bundle loading](../cli/src/launch/install.rs). A controller's `vade install` accepts this bundle when its source revisions match, or the separate `vade-prebuilt-v1` format; see [controller prebuilt bundles](../cli/docs/prebuilt.md#prebuilt-binaries-for-a-controller-install).

**Byte-identical.** Exactly the same, down to every byte of the answer text.

**Cache, prompt.** Saved work from reading a prompt. When a new request starts the same way, that part is not read again.

**Cargo.** Rust's build tool and package manager. `cargo build` compiles; `cargo run` compiles and runs.

**Chain.** A [layer pipeline](#ho) stretched across more than two Sparks. Each Spark owns a block of consecutive layers and passes the intermediate state on to the next. The engine runs a chain on its own when FP4-Local runs on three or more Sparks without a tensor map and without the tensor split (over TCP, or when the split does not take that many Sparks: its widths must cut into whole parts and its mailbox must have room; for this checkpoint the split takes two and four). See [architecture](architecture.md#several-sparks-the-tensor-split-layer-pipeline-and-chain) and [controller selection](../cli/src/app/plan.rs).

**Checkpoint.** One published set of a model's files (weights, configuration and tokenizer) at one exact revision.

**Chunked prefill.** Reading a long prompt in pieces instead of all at once, so other chats are not stuck waiting.

**CLI (command-line interface).** A program you use by typing commands in a terminal, such as `vade` or `vade-harness`.

**Client-measured.** Timed by the program that sends the requests, which is how an app experiences it.

**Cohort, C1..C16.** A group of request streams running at the same time in a benchmark. C1 means one stream, C8 means eight and C16 means sixteen. These are the concurrency levels requested. They do not promise that every stream holds a GPU slot at once; see [benchmark definitions](benchmarks.md).

**Cohort drafting.** [Drafting](#dg) for several chats at once. The engine drafts for up to eight chats together and checks all their guesses in one shared pass, and each chat still gets the same bytes it would get running alone. It is on by default on one Spark and on the two-Spark tensor split; see the [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**Compaction.** Making a long chat fit its context again: older answers are replaced by a short summary, and your instructions, tool results and recent turns stay. `/compact` does it by hand, and chat does it by itself when the context is nearly full. The full record stays on disk in the [journal](#ho).

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

**DGX Spark.** NVIDIA's desktop AI computer, built around the GB10 chip, with 128 GB of memory shared by CPU and GPU (NVIDIA's specification).

**Drafting, speculative decoding.** Something cheap guesses a few tokens ahead, and the main model checks them all in one step. Vade keeps only the guesses the main model would have picked itself, so answers do not change. The drafting depth is how many tokens it guesses ahead. `--mtp N` sets it from 0 to 3, and 0 turns drafting off.

- **Flash-Next** has no separate draft model. Its guesses come from the checkpoint's own MTP heads (see [MTP](#ho)) and from prompt lookup.
- **Gemma 4 31B** is the exception. It can use Google's separate assistant draft model; see [its engine code](../engine/crates/engine/src/gemma4/assistant.rs).

**Direct text chat.** Plain chat with no tools. `vade chat` uses it when the local context is below 8192 tokens, too small for the agent.

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

**Golden.** A saved reference answer from a qualified build at one setup. It includes the full logits: the model's raw scores for each possible next token. Exactness checks compare new runs with it byte for byte.

**GPU (graphics processing unit).** The processor that does the model's heavy math, as many small calculations at once.

**GPU kernel.** A small program that runs on the GPU, such as one matrix multiply. Vade writes its own, in Rust.

**Greedy decoding.** Always taking the model's single most likely next token. This is FP4-Local's default when the request names no sampling fields; see the [request parser](../engine/crates/serve/src/protocol.rs).

<br>

## H–O

**Harness.** The program around a model that turns it into a working assistant: the interface, the tools and the conversation history.

**HTTP.** The ordinary protocol of the web. The engine listens for HTTP requests on a port.

**Hugging Face.** The website where model makers publish their weights. `vade pull` downloads a checkpoint from it and checks the files; see [download models](pull-models.md).

**ib_write_bw, perftest.** `perftest` is the standard RDMA test kit. `ib_write_bw` is the tool in it that measures RDMA write bandwidth between two machines.

**Inference.** Using a trained model to produce output. Training builds the model; inference runs it.

**Inference engine.** The program that loads a model's weights and runs inference efficiently, often for many users at once.

**Journal (session journal).** The private record of one agent chat: messages, tool calls and results. It is kept under the [Vade home](#sz) in `projects/`, never in your project folder, so `--resume` can continue the chat later.

**Jumbo frames.** Network packets larger than the usual 1,500 bytes; here, MTU 9000. Fewer, bigger packets move data faster.

**KV cache.** The model's working memory for one conversation: saved results for every token read so far, so they are not recomputed.

**Launcher.** The `vade launch` workflow you run on the Spark. It finds an existing checkpoint, prepares local artifacts, starts or attaches to the engine it owns, and opens chat. See [first run](../harness/docs/first-run.md) and the [launcher implementation](../cli/src/launch.rs).

**Layer pipeline.** A way to run FP4-Local on several Sparks: the model's layers are split into blocks, and each Spark runs its own block with the same arithmetic as the one-Spark engine, passing its result on to the next. Drafted tokens are checked across every stage. It runs as a [chain](#ac) of three or more Sparks when the tensor split does not take that count. Two Sparks no longer run it: they run the [tensor split](#sz), which keeps the pipeline's links and protocol. Four Sparks over RDMA run the split in parts on the same links. On two Sparks, a cold prompt of 2,048 to 8,192 tokens takes the token split, and one of up to 20,480 tokens runs layer by layer. A longer prompt runs as pipeline tiles of 2,048 rows; from 28,672 tokens, the middle tiles take 4,096 rows. See [several Sparks](architecture.md#several-sparks-the-tensor-split-layer-pipeline-and-chain).

**LLM (large language model).** A model trained on large amounts of text to predict the next token. Chat assistants are built on them.

**Locked memory (memlock).** Memory the OS promises never to move or swap out. RDMA needs it. Vade requires the limit to be `unlimited` in the session that starts it.

**Loopback, localhost.** The address a machine uses to talk to itself (`127.0.0.1`). The launcher binds HTTP there. The configured controller route uses an [SSH tunnel](#sz) to reach the engine from another computer.

**Main `abc1234`, #123.** A revision and a pull request of Vade's private development repository. Results, receipts and the changelog name them so a number can be traced to the code that made it. Each public commit holds the files of one private main revision, but the public history does not contain most of those revisions. So `git checkout abc1234` usually does not find the revision here, and pull request `#123` here is a different one or none. The exceptions are the two v0.1.0 release commits, `5a22651` and `99e8f62`: they are public commits.

**Mixture of experts (MoE).** A model made of many small expert networks, where each token visits only a few of them.

**Model plan.** Vade's typed description of a checkpoint, read from `config.json` and the safetensors headers: its layers, sizes and each tensor's role. See [Model plan](model-plan.md).

**Monorepo.** One repository holding several projects side by side, like `engine/` and `harness/` here.

**MTP (multi-token prediction), drafting ON.** An extra head in the model guesses a few tokens ahead, and the main model checks them. On FP4-Local:

- **Default:** drafting is on, three tokens ahead (request depth 3). A request can also ask for it with `"draft":"strict"`.
- **One chat alone:** the engine adjusts the depth as it goes, up to a ceiling called the resident depth. The ceiling is 7 on one Spark and on the layer pipeline.
- **Prompt lookup:** also on by default, on one Spark and on the pipeline. It copies guesses from matching earlier text in the same chat. It runs only for greedy requests that draft, and `VADE_MTP_LOOKUP=0` turns it off ([source](../engine/crates/engine/src/flash/next_lookup.rs)).
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

**Page cache.** Copies of recently read files that Linux keeps in spare memory. Linux gives that memory back when a program needs it, so Vade counts it as available memory. Dropping the page cache does not make more room for the engine.

**Path (topology).** Ranks connected in a line with two open ends, rather than a closed ring. An FP4-Local chain relays work through its stages in order; see [pipeline stage geometry](../engine/crates/engine/src/flash/next_pp.rs).

**Poison.** Marking execution state as unusable after a step fails halfway, so later work cannot quietly carry on from half-updated state. On the older [tensor-map rank path](#sz), a poisoned [rank session](../engine/crates/tp/src/next.rs) stays failed until a reset across all ranks succeeds.

**Prefill.** Reading the whole prompt once, before the first answer token. It sets the time to first token.

**Prefix cache.** Another name for the [prompt cache](#ac): the engine keeps the work from the start of earlier prompts, so a new request that starts the same way reads only the new part. `--cache-bytes 0` turns the retained cache off for a local install, and `vade up --fp8 --cache-bytes 0` turns it off for FP8. From an FP4-Local controller, set `VADE_FLASHNEXT_PREFIX_CACHE=0` on every Spark instead; see [the prefix cache](../INSTALL.md#the-prefix-cache).

**Preflight.** The checks Vade runs before it installs or starts anything: the machine, driver, memory, disk and model files. A failed preflight changes nothing.

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

**RigMark.** A public benchmark suite ([github.com/alexellis/rigmark](https://github.com/alexellis/rigmark)). A card is the result file of one RigMark run: its recipe, measurements and output checks. A timing run with `vade bench` is not a card. See [Benchmarks](benchmarks.md#run-rigmark-against-vade).

**Ring (topology).** Ranks connected in a loop, each passing data to the next. Rings belong to the transport of the older [tensor-map rank path](#sz); a current FP4-Local pair runs the [tensor split](#sz).

**RoCE (RDMA over Converged Ethernet).** RDMA carried over ordinary Ethernet. The Spark's QSFP links use RoCE v2, which travels inside UDP on port 4791.

**Router.** A program that receives requests and forwards each one to one of several servers. In Vade, that is the harness gateway with several `--upstream` engines.

**Rust.** A programming language that is fast like C and checks memory safety when the code compiles.

<br>

## S–Z

**Safetensors.** The file format of the model's weights: plain tensors with a small header, safe to load.

**Sampling.** Picking the next token with some randomness, controlled by settings like temperature. Turning it off gives greedy decoding.

**Sandbox.** The macOS limits around the commands the agent runs: no network, a cleared environment and a 120-second time limit. Linux has no such sandbox yet, so there the agent does not run commands.

**Scrub.** Removing private details from files before they are shared.

**Skill.** A short instruction file that a project offers the agent. The agent sees each skill by name and loads its text only when it needs it. A skill never runs a script by itself.

**Splitter.** The planner that decides which Spark owns which part of each layer, what the Sparks must add up or gather, and whether each one has the memory. See [Splitter](splitter.md).

**SSH alias.** A short name in `~/.ssh/config` that stands for a machine's address, user and key. The Vade config's `ssh` entries may use these aliases or a plain `user@host`.

**SSH tunnel.** A private pipe through SSH that makes a port on a remote machine appear on your own. On the configured controller, `vade up` opens one to rank 0.

**Steering.** Typing while the agent answers. Chat stops the answer, keeps the part already written, and sends your new message. `/queue on` waits for the answer to finish instead, and `/queue off` turns steering back on.

**Streaming.** Sending the answer as it is written, token by token, instead of all at the end.

**Subnet, /24.** A block of network addresses that share a prefix. A `/24` such as `192.0.2.0/24` holds addresses `192.0.2.1` to `192.0.2.254`. Each fabric rail gets its own.

**Sum order.** The fixed order in which partial results from different Sparks are added. Floating-point addition rounds, so another order can change the answer's bytes.

**Tensor parallelism (TP).** Splitting a layer's tensors across several GPUs so they work on a token together. FP4-Local's two-Spark tensor split is the only two-Spark path. The layer pipeline gives each stage a whole range of layers instead; see [architecture](architecture.md#several-sparks-the-layer-pipeline-and-the-chain) for the current default and the one-path rule.

**Tensor-map rank path.** The older way to run FP4-Local on several Sparks:

- Every Spark runs every layer but holds only its share of each one: whole experts, plus slices of the attention heads and channels. A tensor-map file lists each share.
- The Sparks gather each other's partial results and add them in a fixed [sum order](#sz).
- Unlike the layer pipeline, it never drafts.

It runs on four Sparks only; see [the rank backend](../engine/docs/next-http.md#native-rank-backend).

**Thinking.** Reasoning text the model can write before its answer. In `vade chat` it is off by default; `--thinking on` or `--effort LEVEL` turns it on, and Ctrl+T shows or hides it. An API request to FP4-Local thinks first unless it sends `"enable_thinking": false` or `"reasoning_effort": "none"`.

**Time to first token (TTFT).** Seconds from sending a request to receiving the first piece of the answer.

**Token.** The small unit of text a model reads and writes, often a word or part of a word.

**Tokenizer.** The part that turns text into token numbers and back.

**Tokens per second (tok/s).** How fast answer text comes out.

**Tool calls.** The model asks the harness to run a tool, such as reading a file, and gets the result back.

**Vade home (`VADE_HOME`).** The private folder that holds Vade's install, saved defaults, engine state and chat history; `~/.vade` by default. Keep the same Vade home for every command of one installation.

**Weights.** The learned numbers that make up a model. Loading a model means loading its weights into memory.

**Workspace.** Two meanings. In the harness, it is the one folder an agent may read and change. In Cargo, it is a group of crates built together.

**YaRN.** A method that stretches a model's context window beyond what it was trained for.

<br>

**Back:** [Architecture](architecture.md) · **Next:** [Benchmarks](benchmarks.md) →
