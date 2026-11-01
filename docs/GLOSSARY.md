# Glossary

This page defines the terms used across this repository, from A to Z. A link to a term lands on the letter section that holds it. Scan down for the bold name.

Each page in [`engine/docs/`](../engine/docs/README.md) starts with a box that explains its own engine terms.

These terms come up first when you read about how Vade serves a model:

- [Prefill](#pr) and [decode](#dg)
- [KV cache](#ho)
- [Drafting](#dg)

These terms come up when you link two or more Sparks:

- [Layer pipeline](#ho)
- [Tensor parallelism](#sz)
- [RDMA and RoCE](#pr)
- [QSFP and rails](#pr)
- [MTU](#ho)

<br>

## A–C

**aarch64 (Arm64).** aarch64 is the processor family of the Spark's CPU. On Linux, `uname -m` reports it.

**Agent (coding agent).** The agent is the mode of chat that can use tools. It reads, searches and edits files in the [workspace](#sz), and on macOS it can run commands in a [sandbox](#sz). `vade chat` opens it when the context is at least 8192 tokens. See [chat and the coding agent on Linux](../INSTALL.md#chat-and-the-coding-agent-on-linux).

**Allowlist.** An allowlist is a list of the only files that may be copied. Anything not on it stays behind.

**API key file.** An API key file holds the secret that a client sends to the engine as a bearer credential (an `Authorization: Bearer` header). Pass it with `--api-key-file PATH` or `VADE_API_KEY_FILE`, and keep the file private.

**Apache License 2.0.** The Apache License 2.0 is a permissive open-source license. You can use, change and share the code if you keep the notices.

**Approval.** The agent asks before it edits a file or runs a command outside its read-only list. `--approval ask` (the default) asks each time. `auto` allows checked patches and sandboxed commands. `never` refuses them.

**Bandwidth, Gb/s and GB/s.** Bandwidth is how much data a link moves per second. A lowercase **b** means bits and an uppercase **B** means bytes, so 196 Gb/s ≈ 24.5 GB/s.

**Batching.** Batching runs several requests through the model together, so the GPU does more useful work per step.

**BF16 (bfloat16).** BF16 is a 16-bit number format for model weights. It keeps the range of 32-bit numbers but has less precision.

**Bundle.** A bundle is a ready-made set of engine, GPU kernel and harness artifacts, with a manifest and checksums. You use it instead of a local rebuild. There are two formats:

- **Launcher bundle.** The launcher bundle (`vade-launch-bundle-v1`, with `launcher-bundle.conf`) supports a local install. See [bundle loading](../cli/src/launch/install.rs).
- **Prebuilt bundle.** A controller's `vade install` accepts the launcher bundle when its source revisions match. It also accepts the separate `vade-prebuilt-v1` format. See [controller prebuilt bundles](../cli/docs/prebuilt.md#prebuilt-binaries-for-a-controller-install).

**Byte-identical.** Byte-identical means exactly the same, down to every byte of the answer text.

**Cache, prompt.** The prompt cache is saved work from reading a prompt. When a new request starts the same way, Vade does not read that part again.

**Cargo.** Cargo is Rust's build tool and package manager. `cargo build` compiles. `cargo run` compiles and runs.

**Chain.** A chain is a [layer pipeline](#ho) across more than two Sparks. Each Spark owns a block of consecutive layers and passes the intermediate state on to the next. The engine runs a chain on its own when FP4-Local runs on three or more Sparks without a tensor map and without the tensor split. That happens over TCP, or when the split does not take that many Sparks. The split needs widths that cut into whole parts and a mailbox with room. For this checkpoint, the split takes two and four Sparks. See [architecture](architecture.md#several-sparks-the-tensor-split-layer-pipeline-and-chain) and [controller selection](../cli/src/app/plan.rs).

**Checkpoint.** A checkpoint is one published set of a model's files (weights, configuration and tokenizer) at one exact revision.

**Chunked prefill.** Chunked prefill reads a long prompt in pieces instead of all at once, so other chats do not wait.

**CLI (command-line interface).** A CLI is a program you use by typing commands in a terminal, such as `vade` or `vade-harness`.

**Client-measured.** A client-measured time comes from the program that sends the requests. It is the time that an app sees.

**Cohort, C1..C16.** A cohort is a group of request streams that run at the same time in a benchmark. C1 means one stream, C8 means eight and C16 means sixteen. These are the concurrency levels requested. They do not promise that every stream holds a GPU slot at once. See [benchmark definitions](benchmarks.md).

**Cohort drafting.** Cohort drafting is [drafting](#dg) for several chats at once. The engine drafts for up to eight chats together and checks all their guesses in one shared pass. Each chat still gets the same bytes that it would get alone. Cohort drafting is on by default on one Spark and on the two-Spark tensor split. See the [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**Compaction.** Compaction makes a long chat fit its context again. A short summary replaces the older answers. Your instructions, the tool results and the recent turns stay. `/compact` starts it by hand, and chat starts it by itself when the context is nearly full. The full record stays on disk in the [journal](#ho).

**ConnectX-7 (CX7).** ConnectX-7 is NVIDIA's high-speed network chip inside each Spark. It provides the two QSFP ports and RDMA.

**Context window.** The context window is the most tokens a model can consider at once: the prompt plus the answer so far. The configured limit depends on the launch route and the memory available. See [installation settings](../INSTALL.md).

**Contract.** A contract is the set of rules that two parts agree on when they talk to each other. The engine and the harness share one. It covers what the engine can do, whether it has room for a request, and how its prompt cache is reused. See the [harness engine contract](../harness/docs/engine-contract.md).

**Controller.** A controller is a computer that drives the Sparks over SSH with the `vade` CLI. It can be a second Linux or macOS computer, or a Spark that is its own controller. See [Route B](../INSTALL.md#route-b-from-a-controller).

**Crate.** A crate is one Rust package, either a library or a program. The engine is a workspace of several crates.

**CUDA, CUDA Driver API, CUDA toolkit.** CUDA is NVIDIA's way to run programs on its GPUs. Two parts of it come up in these docs:

- **Driver API.** The Driver API is the low-level layer that every NVIDIA GPU system already has.
- **Toolkit.** The toolkit is NVIDIA's separate compiler and libraries. Vade does not need it.

<br>

## D–G

**DAC (direct-attach copper).** A DAC is a copper cable with QSFP plugs fixed to both ends. It is cheap and reliable for short runs between neighbors.

**Decode.** Decode writes the answer, one token per step, after the prompt has been read. It sets the writing speed (tokens per second).

**Denylist.** A denylist is a private list of names that must never appear in an exported file. It is kept outside every repository.

**Determinism.** Determinism means the same input, with the same settings, gives the same output every time, however the request runs.

**Determinism gate.** The determinism gate is the engine's check for determinism. It runs the same requests alone, cached, chunked and batched. It passes only if the answers are byte-identical.

**DGX OS.** DGX OS is NVIDIA's Ubuntu-based operating system for DGX machines, with the GPU and network drivers included.

**Drafting, speculative decoding.** In drafting, something cheap guesses a few tokens ahead, and the main model checks them all in one step. Vade keeps only the guesses that the main model would have picked itself, so the answers do not change. The drafting depth is how many tokens it guesses ahead. `--mtp N` sets it from 0 to 3, and 0 turns drafting off.

- **Flash-Next** has no separate draft model. Its guesses come from the checkpoint's own MTP heads (see [MTP](#ho)) and from prompt lookup.
- **Gemma 4 31B** is the exception. It can use Google's separate assistant draft model. See [its engine code](../engine/crates/engine/src/gemma4/assistant.rs).

**Direct text chat.** Direct text chat is plain chat with no tools. `vade chat` uses it when the local context is below 8192 tokens, which is too small for the agent.

**Dry run.** A dry run is a preview. The command prints what it would do and changes nothing. Every `vade` lifecycle command accepts `--dry-run`.

**Endpoint.** An endpoint is the web address that a program sends requests to, such as `http://localhost:18080/v1/chat/completions`.

**Fabric.** The fabric is the private QSFP network between Sparks. It is separate from the ordinary LAN. In the Vade config, `fabric.<peer>` is a machine's own address on its cable to that peer.

**flash-serve, next-http, next-tp.** These are three engine programs:

- `flash-serve` is the release HTTP server. It serves FP4-Local on one Spark, on the tensor split and on the layer chain, and it serves FP8 on 2, 4 or 6 Sparks ([source](../engine/crates/tp/src/bin/flash-serve.rs)).
- `next-http` is a compatibility launcher for the older [tensor-map rank path](#sz) ([source](../engine/crates/tp/src/bin/next-http.rs)).
- `next-tp` runs the test cases that qualify the rank code behind that older path. It does not serve chat ([source](../engine/crates/tp/src/bin/next-tp.rs)).

**FP4-Local.** FP4-Local is the current one-Spark release profile. It runs the model `nvidia/Qwen3.8-Flash-Next-FP4-Local`. That checkpoint declares a mix of NVFP4, MXFP8 and plain tensor storage (see [FP8, FP4](#dg)). Despite the name, "Local" does not rule out the layer pipeline across several Sparks. See [architecture](architecture.md#what-the-current-flash-next-path-runs).

**FP8, FP4 (NVFP4).** FP8 and FP4 are 8-bit and 4-bit number formats. Smaller weights need less memory and move faster, at some cost in precision.

**Gateway.** The gateway is the harness mode that relays requests from other coding tools to one or more engines. With several engines, it acts as a [router](#pr).

**GB10.** The GB10 is the chip inside the DGX Spark. It has an NVIDIA Blackwell GPU and an Arm CPU in one package.

**GID (global identifier).** A GID is an address entry that a RoCE device uses to reach its peer. Each IPv4 address on a fabric interface gets a "RoCE v2" GID, and Vade finds the right one automatically.

**Golden.** A golden is a saved reference answer from a qualified build at one setup. It includes the full logits: the model's raw scores for each possible next token. Exactness checks compare new runs with it byte for byte.

**GPU kernel.** A GPU kernel is a small program that runs on the GPU, such as one matrix multiply. Vade writes its own kernels in Rust.

**Greedy decoding.** In greedy decoding, the model always takes its single most likely next token. This is FP4-Local's default when the request names no sampling fields. See the [request parser](../engine/crates/serve/src/protocol.rs).

<br>

## H–O

**Harness.** The harness is the program around a model that makes it a working assistant: the interface, the tools and the conversation history.

**HTTP.** HTTP is the ordinary protocol of the web. The engine listens for HTTP requests on a port.

**Hugging Face.** Hugging Face is the website where model makers publish their weights. `vade pull` downloads a checkpoint from it and checks the files. See [download models](pull-models.md).

**ib_write_bw, perftest.** `perftest` is the standard RDMA test kit. `ib_write_bw` is the tool in it that measures RDMA write bandwidth between two machines.

**Inference.** Inference uses a trained model to produce output. Training builds the model, and inference runs it.

**Inference engine.** An inference engine is the program that loads a model's weights and runs inference efficiently, often for many users at once.

**Journal (session journal).** The journal is the private record of one agent chat: messages, tool calls and results. It is kept under the [Vade home](#sz) in `projects/`, never in your project folder. With the journal, `--resume` can continue the chat later.

**Jumbo frames.** Jumbo frames are network packets larger than the usual 1,500 bytes. Here they use MTU 9000. Fewer, bigger packets move data faster.

**KV cache.** The KV cache is the model's working memory for one conversation. It holds saved results for every token read so far, so the model does not compute them again.

**Launcher.** The launcher is the `vade launch` workflow that you run on the Spark. It finds an existing checkpoint and prepares the local artifacts. It starts the engine that it owns, or attaches to it, and opens chat. See [first run](../harness/docs/first-run.md) and the [launcher implementation](../cli/src/launch.rs).

**Layer pipeline.** The layer pipeline is a way to run FP4-Local on several Sparks. It splits the model's layers into blocks. Each Spark runs its own block with the same arithmetic as the one-Spark engine and passes its result on to the next. Drafted tokens are checked across every stage.

- **Chain.** It runs as a [chain](#ac) of three or more Sparks when the tensor split does not take that count.
- **Two Sparks.** Two Sparks no longer run it. They run the [tensor split](#sz), which keeps the pipeline's links and protocol.
- **Four Sparks over RDMA.** They run the split in parts on the same links.
- **Prompts on two Sparks.** A cold prompt of 2,048 to 8,192 tokens takes the token split, and one of up to 20,480 tokens runs layer by layer. A longer prompt runs as pipeline tiles of 2,048 rows. From 28,672 tokens, the middle tiles take 4,096 rows.

See [several Sparks](architecture.md#several-sparks-the-tensor-split-layer-pipeline-and-chain).

**Locked memory (memlock).** Locked memory is memory that the OS promises never to move or swap out. RDMA needs it. Vade requires the limit to be `unlimited` in the session that starts it.

**Loopback, localhost.** Loopback is the address that a machine uses to talk to itself (`127.0.0.1`). The launcher binds HTTP there. The configured controller route uses an [SSH tunnel](#sz) to reach the engine from another computer.

**Main `abc1234`, #123.** These name a revision and a pull request of Vade's private development repository. Results, receipts and the changelog name them, so you can trace a number to the code that made it. Each public commit holds the files of one private main revision. The public history does not contain most of those revisions. So `git checkout abc1234` usually does not find the revision here, and pull request `#123` here is a different one or none. The exceptions are the two v0.1.0 release commits, `5a22651` and `99e8f62`. They are public commits.

**Mixture of experts (MoE).** A mixture of experts is a model made of many small expert networks. Each token visits only a few of them.

**Model plan.** The model plan is Vade's typed description of a checkpoint, read from `config.json` and the safetensors headers. It gives the layers, the sizes and each tensor's role. See [Model plan](model-plan.md).

**Monorepo.** A monorepo is one repository that holds several projects side by side, like `engine/` and `harness/` here.

**MTP (multi-token prediction), drafting on.** In MTP, an extra head in the model guesses a few tokens ahead, and the main model checks them. On FP4-Local:

- **Default.** Drafting is on, three tokens ahead (request depth 3). A request can also ask for it with `"draft":"strict"`.
- **One chat alone.** The engine adjusts the depth as it goes, up to a ceiling called the resident depth. The ceiling is 7 on one Spark and on the layer pipeline.
- **Prompt lookup.** Prompt lookup is also on by default, on one Spark and on the pipeline. It copies guesses from matching earlier text in the same chat. It runs only for greedy requests that draft, and `VADE_MTP_LOOKUP=0` turns it off ([source](../engine/crates/engine/src/flash/next_lookup.rs)).
- **Several chats.** [Cohort drafting](#ac) is on too.

The older tensor-map rank path never drafts or uses lookup. The code is in [drafting defaults](../engine/crates/engine/src/flash/draft_stack.rs) and [cohort policy](../engine/crates/serve/src/mtp_batch.rs).

**MTP, drafting off.** Drafting off is ordinary decoding with no proposed tokens. There are three ways to choose it:

- `"draft":"plain"` selects it for one request.
- `"mtp_depth":0` does the same when the request names no `draft` mode.
- Server `--mtp 0` makes plain decoding the default.

Under greedy decoding, drafting on and off must return the same token IDs. See [request controls](../engine/crates/serve/src/protocol.rs), [server defaults](../engine/crates/tp/src/flash/next_batched_service.rs) and [`vade verify`](../cli/src/verify/mod.rs).

**MTU (maximum transmission unit).** The MTU is the largest packet that a network interface sends. Vade's fabric uses 9000 at both ends of every cable. If the two ends differ, large packets are dropped.

**Netplan.** Netplan is Ubuntu's network configuration system. The OS-setup scripts write one netplan file for the fabric interfaces.

**Official weights.** Official weights are model files exactly as the maker published them, with no community changes.

**OpenAI-compatible API.** An OpenAI-compatible API takes requests and gives replies in the format that OpenAI's API uses. Most chat apps can connect when you change only the base URL.

<br>

## P–R

**Page cache.** The page cache holds copies of recently read files that Linux keeps in spare memory. Linux gives that memory back when a program needs it, so Vade counts it as available memory. A drop of the page cache does not make more room for the engine.

**Path (topology).** A path is a set of ranks connected in a line with two open ends, not a closed ring. An FP4-Local chain relays work through its stages in order. See [pipeline stage geometry](../engine/crates/engine/src/flash/next_pp.rs).

**Poison.** Poison marks execution state as unusable after a step fails halfway. Later work then cannot quietly continue from half-updated state. On the older [tensor-map rank path](#sz), a poisoned [rank session](../engine/crates/tp/src/next.rs) stays failed until a reset across all ranks succeeds.

**Prefill.** Prefill reads the whole prompt once, before the first answer token. It sets the time to the first token.

**Prefix cache.** The prefix cache is another name for the [prompt cache](#ac). The engine keeps the work from the start of earlier prompts, so a new request that starts the same way reads only the new part.

- **Local install.** `--cache-bytes 0` turns the retained cache off.
- **FP8.** `vade up --fp8 --cache-bytes 0` turns it off.
- **FP4-Local from a controller.** Set `VADE_FLASHNEXT_PREFIX_CACHE=0` on every Spark instead.

See [the prefix cache](../INSTALL.md#the-prefix-cache).

**Preflight.** The preflight is the set of checks that Vade runs before it installs or starts anything: the machine, driver, memory, disk and model files. A failed preflight changes nothing.

**Prompt.** The prompt is everything sent to the model for one turn: the instructions, the conversation so far and the new question.

**PTX.** PTX is NVIDIA's portable instruction format for GPU code. The driver turns it into machine code for the exact GPU.

**QSFP.** QSFP is the name of the high-speed network ports and cables that join Sparks directly to each other. Each Spark has two QSFP ports, and Vade uses 200 Gb/s cables.

**Quantization.** Quantization stores weights in fewer bits (such as FP8 or FP4) to save memory and time.

**Rail.** A rail is one half of a QSFP port. Linux shows each port as two network interfaces on separate PCIe paths, and each of these rails carries about half the cable's speed. Vade uses both rails when each one has its own subnet.

**Rank.** A rank is one process in a group that shares a job, one per Spark. **Rank 0** is the one that answers HTTP.

**Ratchet.** The ratchet is a repository check that fails when the code grows beyond recorded limits. The limits cover external dependencies, large Rust files, and code that branches on a fixed Spark count (such as a comparison of the number of ranks with a literal 2). See the [ratchet test](../cli/tests/ratchet.rs) and its [baseline](../cli/ratchet-baseline.txt).

**RDMA (remote direct memory access).** With RDMA, one computer writes straight into the memory of another over the network, without the operating system in between.

**Receipt.** A receipt is a saved record of a run. It holds the inputs, the engine build and checkpoint that answered, the result and the measured evidence. It covers only the checks that the run did. See the [`vade verify` receipt writer](../cli/src/verify/mod.rs) and [benchmark definitions](benchmarks.md).

**Replica.** A replica is a complete, independent copy of the model that serves its own requests, on one Spark or on a group of Sparks. A router can spread users across several replicas.

**RigMark.** RigMark is a public benchmark suite ([github.com/alexellis/rigmark](https://github.com/alexellis/rigmark)). A card is the result file of one RigMark run: its recipe, measurements and output checks. A timing run with `vade bench` is not a card. See [Benchmarks](benchmarks.md#run-rigmark-against-vade).

**Ring (topology).** A ring is a set of ranks connected in a loop, each passing data to the next. Rings belong to the transport of the older [tensor-map rank path](#sz). A current FP4-Local pair runs the [tensor split](#sz).

**RoCE (RDMA over Converged Ethernet).** RoCE carries RDMA over ordinary Ethernet. The Spark's QSFP links use RoCE v2, which travels inside UDP on port 4791.

**Router.** A router is a program that receives requests and forwards each one to one of several servers. In Vade, the router is the harness gateway with several `--upstream` engines.

**Rust.** Rust is a programming language that checks memory safety when the code compiles.

<br>

## S–Z

**Safetensors.** Safetensors is the file format of the model's weights: plain tensors with a small header, safe to load.

**Sampling.** Sampling picks the next token with some randomness. Settings such as temperature control it. With sampling off, decoding is greedy.

**Sandbox.** The sandbox is the set of macOS limits around the commands that the agent runs: no network, a cleared environment and a 120-second time limit. Linux has no such sandbox yet, so on Linux the agent does not run commands.

**Scrub.** To scrub a file is to remove private details from it before it is shared.

**Skill.** A skill is a short instruction file that a project offers the agent. The agent sees each skill by name and loads its text only when it needs it. A skill never runs a script by itself.

**Splitter.** The splitter is the planner that decides which Spark owns which part of each layer and what the Sparks must add up or gather. It also checks whether each Spark has the memory. See [Splitter](splitter.md).

**SSH alias.** An SSH alias is a short name in `~/.ssh/config` that stands for a machine's address, user and key. The Vade config's `ssh` entries may use these aliases or a plain `user@host`.

**SSH tunnel.** An SSH tunnel is a private pipe through SSH that makes a port on a remote machine appear on your own. On the configured controller, `vade up` opens one to rank 0.

**Steering.** Steering means typing while the agent answers. Chat stops the answer, keeps the part already written, and sends your new message. `/queue on` waits for the answer to finish instead, and `/queue off` turns steering back on.

**Streaming.** Streaming sends the answer as it is written, token by token, instead of all at the end.

**Subnet, /24.** A subnet is a block of network addresses that share a prefix. A `/24` such as `192.0.2.0/24` holds the addresses `192.0.2.1` to `192.0.2.254`. Each fabric rail gets its own subnet.

**Sum order.** The sum order is the fixed order in which the engine adds partial results from different Sparks. Floating-point addition rounds, so another order can change the answer's bytes.

**Tensor parallelism (TP).** Tensor parallelism splits a layer's tensors across several GPUs, so they work on a token together. FP4-Local's two-Spark tensor split is the only two-Spark path. The layer pipeline gives each stage a whole range of layers instead. See [architecture](architecture.md#several-sparks-the-layer-pipeline-and-the-chain) for the current default and the one-path rule.

**Tensor-map rank path.** The tensor-map rank path is the older way to run FP4-Local on several Sparks:

- Every Spark runs every layer but holds only its share of each one: whole experts, plus slices of the attention heads and channels. A tensor-map file lists each share.
- The Sparks gather each other's partial results and add them in a fixed [sum order](#sz).
- Unlike the layer pipeline, it never drafts.

It runs on four Sparks only. See [the rank backend](../engine/docs/next-http.md#native-rank-backend).

**Thinking.** Thinking is reasoning text that the model can write before its answer. In `vade chat`, it is off by default. `--thinking on` or `--effort LEVEL` turns it on, and Ctrl+T shows or hides it. An API request to FP4-Local thinks first unless it sends `"enable_thinking": false` or `"reasoning_effort": "none"`.

**Time to first token (TTFT).** TTFT is the time in seconds from sending a request to receiving the first piece of the answer.

**Tokenizer.** The tokenizer is the part that turns text into token numbers and back.

**Tokens per second (tok/s).** Tokens per second is how fast the answer text comes out.

**Tool calls.** In a tool call, the model asks the harness to run a tool, such as a file read, and gets the result back.

**Vade home (`VADE_HOME`).** The Vade home is the private folder that holds Vade's install, saved defaults, engine state and chat history. It is `~/.vade` by default. Keep the same Vade home for every command of one installation.

**Weights.** Weights are the learned numbers that make up a model. To load a model is to load its weights into memory.

**Workspace.** Workspace has two meanings. In the harness, it is the one folder that an agent may read and change. In Cargo, it is a group of crates built together.

**YaRN.** YaRN is a method that stretches a model's context window beyond the length it was trained for.

<br>

- Back: [Architecture](architecture.md)
- Next: [Benchmarks](benchmarks.md)
