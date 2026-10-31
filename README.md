# Vade

Vade runs a large language model on your own NVIDIA DGX Spark, and the same
greedy request on the same setup gives the same tokens every time. To start,
run the one command below on a Spark that has the model in `~/models`. You
get an OpenAI-compatible API and a coding agent in your terminal that reads
and edits files (on the Spark, it does not run commands).

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-static.png">
  <img src="docs/assets/vade-banner.webp" width="100%" alt="VADE drawn in lit monospace characters over a slow field of dots and dashes, as on vade.lol, above one line of text: Pure-Rust engine and harness for the DGX Spark.">
</picture>

On one Spark with the model already in `~/models`, this installs Vade and
opens a chat ([Install](#pick-your-path) has the details):

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade" &&
  sh "$HOME/vade/scripts/install.sh" --launch
```

- **No model yet?** Leave out `--launch`. The script installs Vade without
  the model. Then download the 102.26 GB checkpoint with `vade pull` and run
  `vade launch`. The [quick start](#quick-start-on-one-spark) gives the five
  commands.

[Why Vade](docs/why-vade.md) says why it exists and what "exact" means here.

- **One model, two checkpoints:** Qwen3.8 Flash-Next, text only.
  `nvidia/Qwen3.8-Flash-Next-FP4-Local` runs on one Spark or more.
  `Qwen/Qwen3.8-Flash-Next-FP8` runs on 2, 4 or 6.
- **One Spark or several:** Vade runs on one Spark by default. You can link
  more.
- **Same request, same tokens:** the same greedy request on the same setup
  returns the same token IDs on every run. On FP8, a request with no
  sampling field is sampled, and 2 and 4 Sparks give different IDs today; see
  [exactness](#exactness). `vade verify` checks part of this on your own
  engine.
- **Plain Rust:** the Rust code, GPU kernels included, uses no third-party
  crates. The repository tools also use only Rust and its standard library.

Latest release: [v0.1.0](https://github.com/wappy-x/vade/releases/tag/v0.1.0),
2026-10-24. The [changelog](CHANGELOG.md) lists the later changes under
Unreleased.

## What it is

Vade is an exact inference engine for Qwen3.8 Flash-Next on DGX Spark. It
serves the NVIDIA FP4-Local checkpoint and the publisher's FP8 checkpoint. It
runs on one Spark, or it splits the work across linked Sparks. On a given
Spark count, a greedy request gives identical token IDs with drafting on or
off. The [exactness section](#exactness) gives the rule, the checks we ran and
the FP8 exceptions.

The engine and its GPU kernels use only Rust and its standard library. Vade
has no third-party Rust dependencies and needs no Python.

## What it is not

Vade does not train models. It is not a general server for GPUs from several
vendors. It does not support every checkpoint yet. The supported checkpoints
are the two above.

The engine is not a chat UI. A separate harness supplies the terminal chat
and the coding agent. Both use the engine's API.

<a id="pick-your-path"></a>
## Install

[INSTALL.md](INSTALL.md) is the full guide. If you are an AI agent setting
up Vade for someone, read the [agent quickstart](docs/agents/quickstart.md).

### Quick start on one Spark

You need a DGX Spark with Git, a C linker (`cc`) and curl. Its NVIDIA driver
must support CUDA 13.0 or newer (the `CUDA Version` in the `nvidia-smi`
header), and no other program may use the GPU. INSTALL.md shows how to add
missing tools.

Open a terminal on the Spark and run these five commands:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade launch
```

1. The clone gets the current main, with every change under Unreleased in
   the changelog.
2. The script checks the GPU and driver first. Then it installs the pinned
   Rust tools that are missing and builds `vade`, the GPU kernels, the engine
   and the chat program. It does not need the model. It needs no sudo, and it
   does not change shell profiles or model files.
3. The `export` line puts `cargo` and `vade` on your `PATH` for this shell
   only. Run it again in each new shell, or add it to `~/.profile` yourself.
4. `vade pull` downloads the 102.26 GB model into
   `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local` and checks every file. Skip
   this command if the model is already in `~/models`.
5. `vade launch` finds the model, saves the settings and checks the model
   files. Then it starts the engine and opens the coding agent in the current
   folder. The agent can edit files there, so ask it only to say hello. Type
   `/exit` to leave chat. The engine keeps running until you run `vade down`.
   The API is at `http://127.0.0.1:18080`. If another program has that port,
   `vade launch` names the port it uses instead.

Each step stops at the first problem and prints what to do next. Run the
same command again after the fix; work that is already done is reused. If
you followed the older steps with `vade setup` and `vade chat`, they still
work: on one Spark, `vade setup` skips `vade install` when the installed
files are current.

<a id="no-model-yet"></a>
**About the download.** `vade pull` shows its progress. If it stops, run it
again; it continues from where it stopped. It puts the model in
`~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`, where `vade launch` looks.
Over SSH, run it inside `tmux` or under `nohup`, because a dropped connection
stops it. On 2026-10-24, rented GB10 VMs downloaded the checkpoint at 8 to
22 MB/s: about 80 minutes at the top of that range and over three hours at
the bottom ([benchmarks](docs/benchmarks.md#rented-cloud-gb10-vms)).
[Download models](docs/pull-models.md) explains resume, the file checks,
`vade models` and the commands for the FP8 and Qwen3-8B checkpoints.

**How long it takes.** We timed the manual source steps of INSTALL.md on a
DGX Spark with no Rust, no checkout and no Vade files. The run was on
2026-10-28 at main `285fa8a`. The times come from the timing log of record
`2026-10-28_1744_source-install_285fa8a` and its `Launch ready` line. The
record is not in the repository yet.

| Step | Time (s) |
| --- | ---: |
| Install Rust and the two pinned toolchains ([Install Rust](INSTALL.md#install-rust)) | 16.7 |
| Clone, one kernel build and the CLI build ([Step 2](INSTALL.md#step-2-get-the-code), [Route A](INSTALL.md#route-a-on-this-spark)) | 62.2 |
| First `vade launch`, to `Launch ready in 149.4 s.` | 149.4 |
| Total | 228.2 |

Each row is rounded to 0.1 s, so the rows add up to 228.3 s, not the
228.2 s total. The model was already on the Spark, so the total does not
include the 102.26 GB model download. The clone came from a local copy of the repository,
so it does not include a network fetch of the source. Download times depend
on your network. We have not timed these manual steps on a later main.

**The one-command install** was timed on 2026-10-30 on one DGX Spark with no
Rust and no Vade files, and with the model already on disk. On a clone of
this repository at `82ec816` (the tree of main `0213a82`, which builds the
GPU kernels once), `sh scripts/install.sh` finished in 176.8 s and the first
`vade up` was ready in 67.4 s
([receipt](bench/receipts/2026-10-31/fresh-install-2026-10-30.txt)).

The [INSTALL quick start](INSTALL.md#quick-start-with-an-existing-model)
covers an existing checkout, a checkpoint outside `~/models` and what to do
when a step fails.

**Chat in one step.** When the model is already in `~/models`,
`sh "$HOME/vade/scripts/install.sh" --launch` installs Vade, starts the
engine and opens chat. Later, `vade launch` does the same.

**Optional: the prebuilt bundle.** The
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0)
has a prebuilt launcher bundle and its `SHA256SUMS`. The bundle skips the
Rust builds. It is an older build: v0.1.0, from 2026-10-24. It does not have
the fixes merged since then. On one Spark with drafting ON, the v0.1.0 card
of 2026-10-24 wrote code at 76.7 tok/s and C8 at 173.9 tok/s
([v0.1.0 cards](docs/benchmarks.md#previous-2026-10-24-v010-release-5a22651)).
Main `dc32b7c` gave 81.1 and 179.0 on 2026-10-31 (mean of two starts;
[results](docs/results.md)).
In our replay of the bundle steps, with the model already on the Spark, the
first command to `Launch ready` took
101.262 s (record `2026-10-28_1305_bundle-install_5a22651`, not in the
repository yet). To get the Unreleased changes, build from source.
[INSTALL](INSTALL.md#optional-release-bundle) shows how to use the bundle.

### Two or more Sparks

A controller drives linked Sparks. A controller is a Linux or macOS computer
that reaches each Spark over SSH. One of the Sparks can be the controller.

1. On every Spark, do INSTALL.md
   [steps 1 to 3](INSTALL.md#step-1-prerequisites).
2. Cable the Sparks and give each link an address. See
   [hardware and network setup](docs/network-setup.md).
3. On the controller, build `vade` as in
   [Route B](INSTALL.md#route-b-from-a-controller). Then run `vade setup`. Give
   the Sparks in cable order and each Spark's address on each cable. Setup
   writes the controller config, installs on every Spark and starts them.
   [Two and more Sparks](INSTALL.md#two-and-more-sparks) has the details.

Two Sparks run the tensor split. You set nothing. Both Sparks work on every
token, so one chat writes faster and a long prompt is read faster. Four
Sparks linked by RDMA run the tensor split in parts, and one chat writes
faster again. Three Sparks, five or more, and four over TCP run the layer
chain: they read a long prompt faster than one Spark, but one chat writes at
about the speed of one Spark. See [what works today](#supported-systems).

### FP8 on 2, 4 or 6 Sparks

The FP8 checkpoint does not fit on one Spark, so it always runs on linked
Sparks. It uses the same controller setup as above. Start it with
`vade up --fp8` on two Sparks or `vade up --fp8 --sparks 4` on four. Both
Spark counts run tensor parallel: every Spark works on every token, so four
Sparks write faster than two.
[Install FP8 on 2, 4 or 6 Sparks](docs/install-fp8.md) gives the steps,
the config and the measured speeds.

### From Windows

Windows can't run Vade, but it can drive a Spark. Connect to the Spark with
the OpenSSH client that comes with Windows. Then follow the one-Spark steps
there. See [from a Windows computer](INSTALL.md#from-a-windows-computer).

## What you get

- **Chat in the terminal.** `vade chat` (or `vade launch`) opens Vade's
  coding agent in the folder where you start it. The agent can read, search
  and edit files. On Linux, the Spark included, it can't run shell commands
  yet.
- **Your own tools in chat.** `vade mcp add` connects an MCP server, so
  chat can call the tools you already use. See [MCP tools in chat](docs/mcp.md).
- **An OpenAI-compatible API** at `http://127.0.0.1:18080/v1`. It listens
  only on the Spark's loopback address. If Vade had to pick another port,
  `vade status` prints the address. From another computer, forward the port
  first: `ssh -N -L 18080:127.0.0.1:18080 <user>@<spark>`. From a controller,
  the API is on the `local_port` in your config (18080 unless you changed it).
- **An exactness check.** `vade verify` sends the same requests several
  times, with drafting on and off. It checks that the token IDs match and
  writes `vade-verify.json`.
- **A benchmark client.** `vade bench` measures the running engine and writes
  `vade-bench.json`.
- **Commands to look after the engine.** `vade status` says if the engine is
  ready. `vade doctor` finds problems in your setup. `vade logs` shows the
  engine log. `vade down` stops what Vade started.

**API requests think first.** Unless a request says otherwise, the model
thinks before it answers. The thinking counts against the same `max_tokens`
as the answer. To get only the answer, send `"enable_thinking": false` or
`"reasoning_effort": "none"`. `vade chat`, `vade verify` and `vade bench`
turn thinking off for you.

<a id="defaults-by-route"></a>
### Defaults

The [settings guide](INSTALL.md#settings-you-may-change) lists every default
and how to change it. These are the defaults you meet first:

- **Drafting is ON.** The model's multi-token prediction (MTP) head guesses a
  few tokens ahead. The full model checks each guess, so greedy answers do
  not change.
- **Greedy decoding.** On FP4-Local, a request with no sampling options
  always gets the most likely next token. If you set one option, the engine
  takes the rest from the checkpoint's settings. FP4-Local's own penalties
  are neutral, so `"temperature": 0` alone stays greedy. You can also send
  all of `temperature: 0`, `top_p: 1`, `top_k: 0`, `repetition_penalty: 1`,
  `presence_penalty: 0` and `frequency_penalty: 0`. On FP8, a field that a
  request leaves out takes the checkpoint's sampled value, so a request with
  no sampling options is sampled. Send `temperature: 0` with the three
  neutral penalties, or the full set above. The gateways add the full set,
  plus `seed` 20260905, when a request sends no sampling option.
- **Context 32768 tokens, output up to 8192** on FP4-Local. FP8 starts with
  context 131072 and output 65536
  ([FP8 install](docs/install-fp8.md#defaults-and-drafting)). On the Spark,
  Vade lowers the context (to 16384, then 8192, then 4096) only if the
  engine refuses it for lack of memory. Vade tells you when it does this.
  `--context 262144` asks
  for a larger context. On 2026-10-30 at main `380d8e5`, one Spark served
  `--context 262144`: a 64K-token prompt gave its first token in 32.97 s and
  a 128K-token prompt in 71.53 s (means of two runs), with the same token IDs
  on a repeat ([context on one Spark](docs/benchmarks.md#context-on-one-spark)).
  Two Sparks served `--context 262144` in a test at main `285fa8a`
  ([context windows](docs/benchmarks.md#context-windows-main-285fa8a)).
  The records of both runs are not in this repository.
- **Chats at once** follow from the context and the memory the engine has.

<a id="supported-systems"></a>
## What works today, and what doesn't yet

In short: FP4-Local runs on one Spark or more, and FP8 needs two or more.
Two Sparks make one chat faster, and four Sparks linked by RDMA faster
again. On the Spark,
the agent edits files but does not run commands. The list gives the details.

- **Two checkpoints, both supported.** Each one is used at its pinned
  publisher revision, text only.
  - FP4-Local (`nvidia/Qwen3.8-Flash-Next-FP4-Local`) is the one-Spark route
    and the default. It also runs on linked Sparks; our cards cover two
    and four.
  - FP8 (`Qwen/Qwen3.8-Flash-Next-FP8`) runs on 2, 4 or 6 linked Sparks.
    Our cards cover two and four. Its weights do not fit on one Spark.
    Start it with `vade up --fp8`
    ([FP8 install](docs/install-fp8.md)).

  One older route is still in the tree, but v0.1.0 does not support it: the
  small Qwen3-8B BF16 model (`vade run qwen3-8b`, in the
  [legacy quickstart](docs/quickstart-one-spark.md#qwen3-8b-bf16-ubuntu-to-a-first-answer)).
  [Adding a model](docs/adding-a-model.md) says what a new model must pass.
- **DGX Spark only.** The engine runs only on DGX Spark (NVIDIA GB10) with
  Linux `aarch64`.
- **Controllers run Linux or macOS.** A controller can run Linux (`aarch64`
  or `x86_64`) or macOS. We tested the full controller lifecycle from macOS.
  We also tested it from Linux `aarch64`, with a Spark as its own controller.
  The CLI does not build for Windows. From Windows, connect to the Spark over
  SSH and work there.
- **No shell tool on Linux.** The agent's shell tool and workspace checks
  need the macOS sandbox. So chat on the Spark, or on any Linux computer,
  can't run commands. Reading, searching, line edits and patches work
  everywhere. On macOS, the shell tool sees only system and Homebrew tools.
  If you installed Rust with rustup, see
  [the controller notes](INSTALL.md#route-b-from-a-controller).
- **The agent needs 8192 tokens of context.** Below that, chat on the Spark
  falls back to plain chat. Chat from a controller stops with an error.
- **FP4-Local: two Sparks make one chat faster, and four Sparks faster
  again.** Two Sparks run the tensor split, the only two-Spark path. Both
  Sparks work on every token. Four Sparks linked by RDMA run the tensor
  split in parts: all four work on most layers of every token. In the
  2026-10-31 morning cards at main `dc32b7c`, prose ran at 42.5 tok/s on one
  Spark, 54.9 on two and 63.0 on four ([results](docs/results.md)). Three
  Sparks, five or more, and four over TCP run the layer chain: each Spark
  holds a run of the model's layers and passes its work to the next, so one
  chat writes at about the speed of one Spark. Four Sparks give more total
  output as two separate pairs: on 2026-10-30 at main `51dce02` (engine
  `380d8e5`), two pairs at once gave 465.2 tok/s in total with eight chats on
  each pair (means of two starts), against 245.1 tok/s for one four-Spark
  server with eight chats on 2026-10-31.
- **FP8: four Sparks write faster than two.** FP8 runs tensor parallel on two
  and on four Sparks, so every Spark works on every token. With drafting ON,
  prose ran at 36.6 tok/s on two Sparks and 49.5 on four in the 2026-10-31
  morning cards at main `dc32b7c`. But two and four Sparks give different
  token IDs (see [exactness](#exactness)).
- **Chains are measured up to four machines.** The
  [FP4-Local guide](cli/docs/fp4-local.md) runs 3 to 24 Sparks as a layer
  chain, except four Sparks over RDMA, which run the tensor split in parts.
  We have run at most four: the four-Spark chain on DGX Sparks at `6cb46bf`
  on 2026-10-25, at `76977df` on 2026-10-28 and at `380d8e5` on 2026-10-30,
  and on four cloud GB10 VMs at v0.1.0 on 2026-10-24. No chain of more than
  four has been measured.
- **The prefix cache is on by default.** With the cache, a request that
  starts like an earlier one does not read that start again. The cache works
  on one Spark, on two FP4-Local Sparks (the tensor split) and on FP8.
  FP4-Local on three or more Sparks (the layer chain, or the split in parts on
  four Sparks over RDMA) runs without it. It keeps
  prompts only for requests that send `cache:true` or a
  `prompt_cache_key`. At startup it takes a memory room of its own; on one
  Spark the room never lowers the number of full-context chats, and on two
  Sparks it can lower it by one so a copy of up to 4 GiB fits (a full
  32,768-token context). Set `VADE_FLASHNEXT_PREFIX_CACHE=0`
  to turn it off ([settings](INSTALL.md#the-prefix-cache)). In the
  2026-10-31 morning runs at main `dc32b7c`, the second question on a
  32K-token chat reused 32,064 tokens. It gave its first token in 0.36 s on
  one Spark (15.79 s when the request did not use the cache) and in 0.76 s
  on two Sparks (9.95 s without the cache). With FP8 it took 0.94 s on two
  Sparks and 1.06 s on four. Four FP4-Local Sparks reused nothing and took
  8.72 s ([results](docs/results.md#the-32k-token-chat)).

## Exactness

The rule: the same request on the same build, checkpoint and settings gives
identical token IDs on every run and every Spark. Supported Spark counts
must agree too. The rule must hold when the request runs alone, batched or
chunked, and when it hits or misses the cache. With greedy decoding, it must
also hold with drafting ON or OFF. On FP8, a request that leaves a sampling field out
takes the checkpoint's value for it. A request with no sampling field is
sampled, and without `seed` it gets a new seed. A request with `temperature`
0 and neutral penalties is greedy
([FP8 install](docs/install-fp8.md#defaults-and-drafting)).

`vade verify` checks the drafting part on your own engine. A pass prints
`IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the reply's length
in tokens. It covers only the requests it sent. It does not cover every mix
of batching and caching.

Our latest checks are the 2026-10-31 morning cards at main `dc32b7c`
([results](docs/results.md#token-ids)). Each card sends 66 requests, and its
receipt holds a SHA256 of all 66 token ID arrays:

- FP4-Local on one, two and four Sparks, drafting ON and OFF: all eight cards
  gave the same token IDs in 66 of 66 responses.
- FP8 on two Sparks, and FP8 on four Sparks: drafting ON and OFF gave the
  same token IDs in 66 of 66 responses at each Spark count.
- FP8 on two against four Sparks: only 7 of 66 responses had the same token
  IDs, as on the earlier cards. This breaks the rule above. Until it is
  fixed, compare FP8 results only at the same Spark count
  ([benchmarks](docs/benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).

Earlier exactness comparisons are in the
[benchmark history](docs/benchmarks.md). The card checks compare whole cards.
They do not test one request alone against the same request in a batch. The
32K chat checks a cache hit against a miss for one second question per
setup. In the cards, the 60 chat requests send `cache` false, and the six
8K prompt requests send no cache opt-in.
[Exactness and model quality](docs/architecture.md#exactness-and-model-quality)
explains the full rule and what each check proves.

## Use our agent, add our gateway, or bring your own client

- **Our agent.** `vade chat` opens Vade's agent on the running engine.
- **Our gateway in front of your client.** `vade-gateway` sits between your
  client and the engine, on `http://127.0.0.1:8787/v1`. See
  [the client gateway](INSTALL.md#client-gateway).
- **Your own client.** Point any OpenAI-compatible client at
  `http://127.0.0.1:18080/v1`. The
  [client setup guide](harness/docs/passthrough-matrix.md) covers Claude
  Code, Codex, Hermes, OpenCode and Pi, directly or through the gateway.
  Claude Code needs the gateway, or `claude --bare` direct to the engine.

## Measured results

<!-- bench-docs:start -->

### Current main

These rows measured main `dc32b7c` on our DGX Sparks on the morning of
2026-10-31. Each server start ran one card with drafting ON and one with
drafting OFF. The one-Spark rows are the mean of two server starts; the
other rows are one start each.
[Results and receipts](docs/results.md) links the receipt of every row.

| Model | Sparks | Path | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp4-local | 1 | local | ON | 42.530 | 81.138 | 100.808 | 65.688 | 144.681 | 178.981 | 3.769 |
| fp4-local | 1 | local | OFF | 27.911 | 27.761 | 28.012 | 27.115 | 85.004 | 138.609 | 3.778 |
| fp4-local | 2 | tensor split | ON | 54.878 | 102.314 | 130.800 | 79.632 | 166.796 | 232.198 | 2.153 |
| fp4-local | 2 | tensor split | OFF | 36.047 | 35.840 | 36.142 | 34.375 | 88.178 | 134.414 | 2.131 |
| fp4-local | 4 | tensor split in parts | ON | 63.036 | 113.025 | 145.238 | 86.962 | 177.581 | 245.145 | 1.924 |
| fp4-local | 4 | tensor split in parts | OFF | 42.084 | 41.486 | 42.072 | 38.080 | 95.113 | 143.015 | 1.916 |
| fp8 | 2 | tensor parallel | ON | 36.624 | 51.293 | 52.763 | 45.436 | 66.578 | 98.670 | 5.971 |
| fp8 | 2 | tensor parallel | OFF | 27.072 | 26.913 | 27.114 | 26.089 | 66.273 | 98.059 | 5.952 |
| fp8 | 4 | tensor parallel | ON | 49.508 | 69.340 | 72.248 | 61.435 | 88.017 | 130.999 | 6.923 |
| fp8 | 4 | tensor parallel | OFF | 35.902 | 35.641 | 35.955 | 34.592 | 87.157 | 131.147 | 6.873 |

Rates are in completion tokens per second. Prose, code and structured output
each use one chat. C1, C4 and C8 give the total rates for one, four and eight
concurrent chats. Cold 8K is the time to the first token after an uncached
8192-token prompt, sent with drafting OFF in both cards. A 32K-token prompt
gave its first token in 15.63 s on one Spark, 10.09 s on two Sparks and
8.98 s on four Sparks, and with FP8 in 24.12 s on two Sparks and 27.81 s on
four.

The [receipts page](docs/results.md) gives the build, the machines, the
server commands, the request settings and what you can re-run. The
[benchmark history](docs/benchmarks.md) keeps every earlier card. This
table is written by hand, because the one-Spark rows average two server
starts. `vade-tools bench-docs` writes a simpler table from RigMark card
files: one card per setup, from the newest main only, with no averages.

<!-- bench-docs:end -->

Both checkpoints are supported. The FP4-Local cards used context 32768,
output 8192 and MTP 3; the FP8 cards used context 32768, output 16384 and
MTP 2. FP8 runs tensor parallel on two and on four Sparks, not the layer
chain. The [FP8 install](docs/install-fp8.md#what-to-expect) explains the
FP8 rows.

Earlier results, such as the confirm run at main `285fa8a` and the capacity
test at main `250fb49`, are in the [benchmark history](docs/benchmarks.md).

## License

Vade's code is licensed under [Apache-2.0](LICENSE). The bundled tokenizer and
template fixtures keep their upstream licenses, the
[Code of Conduct](CODE_OF_CONDUCT.md) uses CC BY 4.0, and models keep their own
licenses. [NOTICE](NOTICE) lists the shipped files and their license texts.
[Credits](CREDITS.md) names the papers and projects whose ideas Vade uses.
Its [Original work](CREDITS.md#original-work) section lists what was designed
and built here, with the files to check.

## Documentation map

| Guide | What it covers |
| --- | --- |
| [Why Vade](docs/why-vade.md) | What Vade is, why it exists, what "exact" means and what Vade is not |
| [INSTALL.md](INSTALL.md) | The whole install, starting with one Spark |
| [Agent quickstart](docs/agents/quickstart.md) | For AI agents: exact commands, exit codes, where the files are and what to do on each error |
| [Install and flows](docs/design/install-and-flows.md) | One page on the steps from install to a running engine, and what each prints and how it fails |
| [FP8 install](docs/install-fp8.md) | The FP8 checkpoint on 2, 4 or 6 Sparks: build, weights, config, start and measured speeds |
| [First run](harness/docs/first-run.md) | The launcher on your Spark, saved settings and later runs |
| [Download models](docs/pull-models.md) | `vade pull` and `vade models`: download a checkpoint from Hugging Face with resume and file checks, then see what can serve it |
| [CLI reference](cli/README.md) | Commands, configuration and recovery |
| [Examples](examples/README.md) | Short examples you can run: curl, a Rust API client, Claude Code, Codex, vade bench and a two-Spark launch |
| [Benchmarks](docs/benchmarks.md) | Measured results, receipts and how to reproduce them |
| [Architecture](docs/architecture.md) | How requests reach the engine, and the exactness rule |
| [Glossary](docs/GLOSSARY.md) | The serving terms these docs use |
| [Harness](harness/README.md) | The coding agent, its tools, and the gateway for other clients |
| [Reading files](docs/reading-files.md) | How `vade read` and the agent turn documents into Markdown: formats, the header line and caps |
| [Engine docs](engine/docs/README.md) | How the engine runs, its GPU kernels and how it is validated |
| [All documentation](docs/README.md) | Guides for deploying and developing Vade |
| [Contributing](CONTRIBUTING.md) and [agent rules](AGENTS.md) | How to change Vade: house rules, validation and review |
| [Local CI](docs/ci.md#run-checks-from-a-public-checkout) | Tests run locally with these commands before every merge |
| [Security](SECURITY.md) | How to report a vulnerability privately |
