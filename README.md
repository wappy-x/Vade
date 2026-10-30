# Vade

Vade runs a large language model on your own NVIDIA DGX Spark, and the same
greedy request on the same setup gives the same tokens every time. To start,
run the one command below on a Spark that has the model in `~/models`. You
get an OpenAI-compatible API and a coding agent in your terminal that reads
and edits files (on the Spark, it does not run commands).

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-static.png">
  <img src="docs/assets/vade-banner.webp" width="100%" alt="VADE drawn in lit monospace characters over a slow field of dots and dashes, as on vade.lol. An LLM inference engine, written in Rust, built for the DGX Spark.">
</picture>

On one Spark with the model already in `~/models`, this installs Vade and
opens a chat ([Install](#pick-your-path) has the details):

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade" &&
  sh "$HOME/vade/scripts/install.sh" --launch
```

- **The repository is private for now.** The clone above works only for an
  account with access. Sign in with the [GitHub CLI](https://cli.github.com/)
  (`gh auth login`), then use
  `gh repo clone wappy-x/vade "$HOME/vade" -- --branch main` as the first
  line. [INSTALL step 2](INSTALL.md#step-2-get-the-code) also shows a
  read-only deploy key.
- **No model yet?** The script builds `vade` first, then stops at the model
  check. Download the 102.26 GB checkpoint with `vade pull`, then run the
  script again. [No model yet](#no-model-yet) gives the two commands.

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

> [!IMPORTANT]
> **Private preview.** The repository is private for now. Without access, a
> plain clone and the release downloads fail. Ask the owner for access. Then
> follow the private-preview notes in
> [INSTALL step 2](INSTALL.md#step-2-get-the-code).

[INSTALL.md](INSTALL.md) is the full guide. If you are an AI agent setting
up Vade for someone, read the [agent quickstart](docs/agents/quickstart.md).

### Quick start on one Spark

You need a DGX Spark with Git, a C linker (`cc`) and curl. Its NVIDIA driver
must support CUDA 13.0 or newer (the `CUDA Version` in the `nvidia-smi`
header), and no other program may use the GPU. You also need the complete
102.26 GB checkpoint under `~/models`. INSTALL.md shows how to add missing
tools and how to download the model.

Open a terminal on the Spark and run these five commands:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
vade setup
vade chat
```

1. The clone gets the current main, with every change under Unreleased in
   the changelog.
2. The script checks the GPU and driver first. Then it installs the pinned
   Rust tools that are missing, builds `vade`, checks this Spark and the
   model, and installs the engine, the GPU kernels and the chat program. It needs no sudo, and it does not change shell
   profiles or model files.
3. The `export` line puts `cargo` and `vade` on your `PATH`. The script
   does not edit shell profiles, so the line lasts for this shell only. Run
   it again in each new shell, or add it to your shell profile yourself.
4. `vade setup` uses the model and settings the script saved. It runs
   `vade install` again (nothing is rebuilt), starts the engine with
   `vade up`, then checks everything with `vade doctor`. Without the script,
   or from another computer, it first asks a few questions: which Sparks,
   which checkpoint (for 2, 4 or 6 Sparks) and where the model is.
5. `vade chat` opens the coding agent in the current folder. The agent can
   edit files there, so ask it only to say hello. Type `/exit` to leave. The
   engine keeps running until you run `vade down`.

Each step stops at the first problem and prints what to do next. Run the
same command again after the fix; work that is already done is reused.

<a id="no-model-yet"></a>
**No model yet?** The script builds `vade` first, then stops at the model
check. Run the `export` line, download the model, then run the script again
(not the clone line, because the folder exists now):

```sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
sh "$HOME/vade/scripts/install.sh"
```

`vade pull` shows its progress. If it stops, run it again; it continues from
where it stopped. It puts the model in
`~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`, where the script looks. Over
SSH, run it inside `tmux` or under `nohup`, because a dropped connection
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

The model was already on the Spark, so the total does not include the
102.26 GB model download. The clone came from a local copy of the repository,
so it does not include a network fetch of the source. Download times depend
on your network. We have not timed these manual steps on a later main.

**The one-command install** was timed on 2026-10-30 on one DGX Spark with the
model already on disk. On a Spark with no Rust and no Vade files,
`sh scripts/install.sh` finished in 227.7 s at main `0bad2e6`. With Rust
already installed, `sh scripts/install.sh --launch` printed `Install complete`
at 209.4 s at main `54bcf05`, and chat opened at 228.2 s. At both mains the
install built the GPU kernels twice, about 51 to 56 s each time. Main
`5cb08ee` and later build them once. We have not timed the install on those
mains yet. The run logs are not in the repository yet.

The [INSTALL quick start](INSTALL.md#quick-start-with-an-existing-model)
covers an existing checkout, a checkpoint outside `~/models` and what to do
when a step fails.

**Chat in one step.** `sh "$HOME/vade/scripts/install.sh" --launch`
installs Vade, starts the engine and opens chat. Later, `vade launch` does
the same.

**Optional: the prebuilt bundle.** The
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0)
has a prebuilt launcher bundle and its `SHA256SUMS`. The bundle skips the
Rust builds. It is an older build: v0.1.0, from 2026-10-24. It does not have
the fixes merged since then. On one Spark with drafting ON, the v0.1.0 card
of 2026-10-24 wrote code at 76.7 tok/s and C8 at 173.9 tok/s. Main `380d8e5`
gave 81.4 and 179.5 on 2026-10-30 (mean of two or three starts; see the
[v0.1.0 cards](docs/benchmarks.md#previous-2026-10-24-v010-release-5a22651)).
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
token, so one chat writes faster and a long prompt is read faster. Four or
more Sparks run the layer pipeline. They read a long prompt faster than one
Spark, but one chat writes at about the speed of one Spark, and slower than
on two. See [what works today](#supported-systems).

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
  neutral penalties, or the full set above. The gateways add the full set
  when a request sends no sampling option.
- **Context 32768 tokens, output up to 8192.** On the Spark, Vade lowers the
  context (to 16384, then 8192, then 4096) only if the engine refuses it for
  lack of memory. Vade tells you when it does this. `--context 262144` asks
  for a larger context. On 2026-10-30 at main `380d8e5`, one Spark served
  `--context 262144`: a 64K-token prompt gave its first token in 32.97 s and
  a 128K-token prompt in 71.53 s (means of two runs), with the same token IDs
  on a repeat. Two
  Sparks served `--context 262144` in a test at main `285fa8a`
  ([context windows](docs/benchmarks.md#context-on-one-spark)).
- **Chats at once** follow from the context and the memory the engine has.

<a id="supported-systems"></a>
## What works today, and what doesn't yet

In short: FP4-Local runs on one Spark or more, and FP8 needs two or more.
Two Sparks make one chat faster; four FP4-Local Sparks do not. On the Spark,
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
- **FP4-Local: two Sparks make one chat faster. Four Sparks write at about
  the speed of one.** Two Sparks run the tensor split, the only two-Spark
  path. Both Sparks work on every token. Four or more Sparks run the layer
  pipeline. Each Spark holds a run of the model's layers and passes its work
  to the next. So when one chat writes, each new token passes through the
  Sparks in turn. In the 2026-10-30 morning cards at main `380d8e5`, prose
  ran at 43.1 tok/s on four Sparks, 42.6 on one and 54.9 on two (means of
  two or three server starts). Two or more Sparks read prompts faster than
  one. Four Sparks give more as two separate pairs: on 2026-10-30 at main
  `51dce02` (engine `380d8e5`), two pairs at once wrote prose at 55.8 and
  54.8 tok/s, one chat on each, and 465.2 tok/s in total with eight chats on
  each pair (means of two starts). See
  [measured results](#measured-results).
- **FP8: four Sparks write faster than two.** FP8 runs tensor parallel on two
  and on four Sparks, so every Spark works on every token. With drafting ON,
  prose ran at 36.6 tok/s on two Sparks and 47.3 on four in the 2026-10-30
  morning cards at main `380d8e5`. But two and four Sparks give different
  token IDs (see
  [exactness](#exactness)).
- **Chains are measured up to four machines.** The
  [FP4-Local guide](cli/docs/fp4-local.md) describes chains of 2 to 24
  Sparks. We have run at most four: four DGX Sparks at `6cb46bf` on
  2026-10-25, at `76977df` on 2026-10-28 and at `380d8e5` on 2026-10-30,
  and four cloud GB10 VMs at v0.1.0 on 2026-10-24.
- **The prefix cache is on by default.** With the cache, a request that
  starts like an earlier one does not read that start again. The cache works
  on one Spark, on two FP4-Local Sparks (the tensor split) and on FP8. The
  FP4-Local layer chain (three or more Sparks) runs without it. It keeps
  prompts only for requests that send `cache:true` or a
  `prompt_cache_key`. At startup it takes a memory room of its own; on one
  Spark the room never lowers the number of full-context chats, and on two
  Sparks it can lower it by one so a copy of up to 4 GiB fits (a full
  32,768-token context). Set `VADE_FLASHNEXT_PREFIX_CACHE=0`
  to turn it off ([settings](INSTALL.md#the-prefix-cache)). In the
  2026-10-30 morning cards at main `380d8e5`, the second question on a
  32K-token chat reused 32,064 tokens. It gave its first token in 0.36 s on
  one Spark (15.84 s when the request did not use the cache) and in 0.75 s
  on two Sparks (9.93 s without the cache). With FP8 it took 0.94 s on two
  Sparks and 1.05 s on four
  ([benchmarks](docs/benchmarks.md#a-32k-token-chat-and-its-second-question)).

## Exactness

The rule: the same request on the same build, checkpoint and settings gives
identical token IDs on every run and every Spark. Supported Spark counts
must agree too. This holds when the request runs alone, batched or chunked,
and when it hits or misses the cache. With greedy decoding, it also holds
with drafting ON or OFF. On FP8, a request that leaves a sampling field out
takes the checkpoint's value for it. A request with no sampling field is
sampled, and without `seed` it gets a new seed. A request with `temperature`
0 and neutral penalties is greedy
([FP8 install](docs/install-fp8.md#defaults-and-drafting)).

`vade verify` checks the drafting part on your own engine. A pass prints
`IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the reply's length
in tokens. It covers only the requests it sent. It does not cover every mix
of batching and caching.

Our latest checks compare drafting ON against OFF:

- On main `cbbb11b` (the 2026-10-29 cards), the FP4-Local cards on one Spark
  with drafting ON and OFF, and on two Sparks with drafting ON, all gave the
  same token IDs, in 64 of 64 responses
  ([benchmarks](docs/benchmarks.md#current-2026-10-29-main-cbbb11b-home-dgx-sparks)).
  On main `380d8e5` (the 2026-10-30 morning cards, drafting ON), one, two
  and four Sparks gave the same token IDs in 66 of 66 responses in every
  start ([benchmarks](docs/benchmarks.md#token-ids)).
- On main `cbbb11b`, FP8 on two Sparks matched drafting ON against OFF in 64
  of 64 responses in each of two server starts. On main `380d8e5`, as on
  `76977df`, its two-Spark and four-Spark cards matched each other in only 7
  of 66 responses. This is an exception to the rule above. A host test found the
  cause, the order in which the ranks add their parts, and a fix is in
  screening. Until it is on main, compare FP8 results only at the same Spark
  count
  ([benchmarks](docs/benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).
- On main `285fa8a`, a confirm run gave the same token IDs as the main
  before it, `54392da`, in all four server starts. See
  [measured results](#measured-results).

Earlier exactness comparisons are in the
[benchmark history](docs/benchmarks.md). These checks do not test alone
against batched, or a cache hit against a miss.
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

These rows measured main `380d8e5` on our DGX Sparks on the morning of
2026-10-30, with RigMark cards and drafting ON. Each row is the mean of two
or three server starts.

| Model | Sparks | Path | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp4-local | 1 | local | 42.597 | 81.372 | 100.747 | 65.924 | 144.829 | 179.467 | 3.761 |
| fp4-local | 2 | tensor split | 54.879 | 102.190 | 129.957 | 79.572 | 166.133 | 231.882 | 2.153 |
| fp4-local | 4 | pipeline | 43.099 | 82.173 | 102.364 | 66.123 | 145.608 | 228.526 | 2.178 |
| fp8 | 2 | ring | 36.637 | 51.302 | 52.731 | 45.119 | 65.930 | 98.339 | 5.957 |
| fp8 | 4 | path | 47.321 | 66.450 | 69.085 | 59.242 | 86.447 | 129.360 | 6.854 |

We did not measure drafting OFF on 2026-10-30. The latest drafting OFF
cards are older: the 2026-10-29 runs at main `cbbb11b` and the 2026-10-28
cards at main `76977df`, as the Main column shows.

| Model | Sparks | Path | Main | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp4-local | 1 | local | `cbbb11b` | 27.891 | 27.707 | 27.915 | 27.079 | 85.042 | 138.608 | 3.764 |
| fp4-local | 2 | tensor split | `76977df` | 35.071 | 35.375 | 35.487 | 33.814 | 82.476 | 126.344 | 2.153 |
| fp4-local | 4 | pipeline | `76977df` | 28.157 | 28.177 | 28.503 | 27.642 | 84.974 | 139.335 | 2.177 |
| fp8 | 2 | ring | `cbbb11b` | 26.887 | 26.727 | 26.914 | 26.050 | 66.682 | 97.378 | 5.966 |
| fp8 | 4 | path | `76977df` | 35.266 | 34.722 | 35.284 | 33.9 | 88.201 | 128.203 | 6.948 |

Rates are in completion tokens per second. Prose, code and structured output
each use one chat. C1, C4 and C8 give the total rates for one, four and eight
concurrent chats. Cold 8K is the time to the first token after an uncached
8192-token prompt. At `380d8e5`, a 32K-token prompt gave its first token in
15.66 s on one Spark, 10.13 s on two Sparks and 10.52 s on four Sparks, and
in 24.11 s on two Sparks and 27.49 s on four Sparks with FP8.

The [benchmark history](docs/benchmarks.md#current-2026-10-30-morning-main-380d8e5-home-dgx-sparks)
gives the number of server starts per row, the 32K chat, the token ID checks
and the limits. These tables are written by hand, because they average
several server starts per row and the drafting OFF rows come from two older
dates.
`vade-tools bench-docs` can rewrite this block from RigMark cards.

<!-- bench-docs:end -->

Both checkpoints are supported. The FP8 cards used context 32768, output
16384 and MTP 2. FP8 on four Sparks runs tensor parallel on a path ("path" in
the table), not the layer pipeline. The
[FP8 install](docs/install-fp8.md#what-to-expect) explains the FP8 rows.
Every card sends its cold 8K prompt with drafting OFF
(`"draft": "plain"`), in the drafting ON rows too. So the Cold 8K column
measures drafting OFF.

Earlier results, such as the confirm run at main `285fa8a` and the capacity
test at main `250fb49`, are in the [benchmark history](docs/benchmarks.md).

## License

Vade's code is licensed under [Apache-2.0](LICENSE). The bundled tokenizer and
template fixtures keep their upstream licenses, the
[Code of Conduct](CODE_OF_CONDUCT.md) uses CC BY 4.0, and models keep their own
licenses. [NOTICE](NOTICE) lists the shipped files and their license texts,
and [Credits](CREDITS.md) keeps the source attribution.

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
