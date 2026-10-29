# Vade

CI: local, see [docs/ci.md](docs/ci.md).

Vade runs a large language model on your own NVIDIA DGX Spark. You get a
coding agent in your terminal, a plain chat and an OpenAI-compatible API.

- **One model today:** `nvidia/Qwen3.8-Flash-Next-FP4-Local`, text only.
- **One Spark or several:** Vade runs on one Spark by default. You can link
  more.
- **Exact by design:** the same request on the same setup returns the same
  token IDs on every run. `vade verify` checks part of this on your own
  engine.
- **Plain Rust:** the Rust code, GPU kernels included, uses no third-party
  crates. The repository tools also use only Rust and its standard library.

Latest release: [v0.1.0](https://github.com/wappy-x/vade/releases/tag/v0.1.0),
2026-10-24. The [changelog](CHANGELOG.md) lists the later changes under
Unreleased.

## What it is

Vade is an exact inference engine for NVIDIA FP4-Local checkpoints on DGX
Spark. It runs on one Spark, or it splits the work across linked Sparks. A
request gives identical token IDs with drafting on or off. The
[exactness section](#exactness) gives the rule and the checks we ran.

The engine and its GPU kernels use only Rust and its standard library. Vade
has no third-party Rust dependencies and needs no Python.

## What it is not

Vade does not train models. It is not a general server for GPUs from several
vendors. It does not support every NVIDIA FP4-Local checkpoint yet. The
supported model is the one above.

The engine is not a chat UI. A separate harness supplies the terminal chat
and the coding agent. Both use the engine's API.

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark-static.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-light-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark.svg">
  <img src="docs/assets/vade-banner-light.svg" width="100%" alt="Vade: an inference engine and coding-assistant harness for DGX Spark.">
</picture>

<a id="pick-your-path"></a>
## Install

> [!IMPORTANT]
> **Private preview.** The repository is private for now. Without access, a
> plain clone and the release downloads fail. Ask the owner for access. Then
> follow the private-preview notes in
> [INSTALL step 2](INSTALL.md#step-2-get-the-code).

[INSTALL.md](INSTALL.md) is the full guide.

### One Spark, from source

This is the main route. It builds the current main, with every change under
Unreleased in the changelog.

You need an idle DGX Spark with Git, a C linker (`cc`), curl and a working
NVIDIA driver. You also need the complete 102.26 GB checkpoint under
`~/models`. INSTALL.md shows how to add missing tools and how to download the
model.

Open a terminal on the Spark and paste this block:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade" &&
  sh "$HOME/vade/scripts/install.sh" --launch
```

The script does these steps:

1. It installs the pinned Rust tools that are missing.
2. It builds Vade.
3. It checks this Spark and the model.
4. It starts the engine and opens chat in the current folder.

The script needs no sudo. It does not change shell profiles or model files.
Chat can edit files in the folder where it opens. So ask it only to say
hello, then type `/exit`. The engine keeps running until you run
`vade down`.

In each new shell, put `cargo` and `vade` on your `PATH` first:

```sh
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
```

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
on your network. We did not time `install.sh` itself.

The [INSTALL quick start](INSTALL.md#quick-start-with-an-existing-model)
covers an existing checkout, a checkpoint outside `~/models` and what to do
when a step fails.

**Optional: the prebuilt bundle.** The
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0)
has a prebuilt launcher bundle and its `SHA256SUMS`. The bundle skips the
Rust builds. It is an older build: v0.1.0, from 2026-10-24. It does not have
the fixes merged since then, so it is slower than main and its prefix cache
can miss where main's cache hits. In our replay of the bundle steps, with the
model already on the Spark, the first command to `Launch ready` took
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
3. Set up the controller with [Route B](INSTALL.md#route-b-from-a-controller).
   Then follow [two and more Sparks](INSTALL.md#two-and-more-sparks). The
   start command is `vade up --sparks 2`.

Two Sparks run the tensor split. You set nothing. Both Sparks work on every
token, so one chat writes faster and a long prompt is read faster. Four or
more Sparks run the layer pipeline. They read a long prompt faster, but one
chat does not write faster yet. See
[what works today](#supported-systems).

### From Windows

Windows can't run Vade, but it can drive a Spark. Connect to the Spark with
the OpenSSH client that comes with Windows. Then follow the one-Spark steps
there. See [from a Windows computer](INSTALL.md#from-a-windows-computer).

## What you get

- **Chat in the terminal.** `vade chat` (or `vade launch`) opens Vade's
  coding agent in the folder where you start it. The agent can read, search
  and edit files. On Linux, the Spark included, it can't run shell commands
  yet.
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
- **Greedy decoding.** A request with no sampling options always gets the
  most likely next token. If you set even one option, such as
  `"temperature": 0` alone, the engine takes the rest from the checkpoint's
  settings. Those settings can include penalties that change the answer. For
  greedy answers, send no sampling options. Or send all of `temperature: 0`,
  `top_p: 1`, `top_k: 0`, `repetition_penalty: 1`, `presence_penalty: 0` and
  `frequency_penalty: 0`.
- **Context 32768 tokens, output up to 8192.** On the Spark, Vade lowers the
  context (to 16384, then 8192, then 4096) only if the engine refuses it for
  lack of memory. Vade tells you when it does this. `--context 262144` asks
  for a larger context. We have no published measurement at that size yet.
- **Chats at once** follow from the context and the memory the engine has.

<a id="supported-systems"></a>
## What works today, and what doesn't yet

- **One model.** `nvidia/Qwen3.8-Flash-Next-FP4-Local` at its pinned
  publisher revision, text only. Two older routes are still in the tree, but
  v0.1.0 does not support them: Flash-Next FP8 on two Sparks
  ([legacy install](docs/install-legacy-fp8.md)) and the small Qwen3-8B BF16
  model (`vade run qwen3-8b`, in the
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
- **Two Sparks make one chat faster. Four do not yet.** Two Sparks run the
  tensor split, the only two-Spark path. Both Sparks work on every token.
  Four or more Sparks run the layer pipeline. Each Spark holds a run of the
  model's layers and passes its work to the next. So one chat uses one GPU at
  a time. Two or more Sparks read prompts faster than one. The end-of-day
  cards compare the same measured build on each tested Spark count. See
  [measured results](#measured-results).
- **Chains are measured up to four machines.** The
  [FP4-Local guide](cli/docs/fp4-local.md) describes chains of 2 to 24
  Sparks. We have run at most four: four DGX Sparks at `6cb46bf` on
  2026-10-25 and at `76977df` on 2026-10-28, and four cloud GB10 VMs at
  v0.1.0 on 2026-10-24.
- **The prefix cache is off by default.** With the cache, a request that
  starts like an earlier one does not read that start again. The cache works
  on one Spark and on two Sparks (the tensor split). In the end-of-day runs
  at `76977df`, the second question on a 32K-token chat reused 32,064 tokens.
  It gave its first token in 0.63 s on one Spark (16.03 s with the cache
  off) and in 0.76 s on two Sparks (9.95 s with the cache off). The first
  question takes longer with the cache on: 16.23 s against 16.04 s on one
  Spark, and 11.09 s against 10.31 s on two
  ([benchmarks](docs/benchmarks.md#a-long-chat-and-its-second-question)).
  See [turn on the prefix cache](INSTALL.md#turn-on-the-prefix-cache).

## Exactness

The rule: the same request on the same build, checkpoint and settings gives
identical token IDs on every run and every Spark. Supported Spark counts
must agree too. This holds when the request runs alone, batched or chunked,
and when it hits or misses the cache. With greedy decoding, it also holds
with drafting ON or OFF.

`vade verify` checks the drafting part on your own engine. A pass prints
`IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the reply's length
in tokens. It covers only the requests it sent. It does not cover every mix
of batching and caching.

Our latest checks compare drafting ON against OFF:

- On main `76977df` (the 2026-10-28 end-of-day cards), FP4-Local gave the
  same token IDs on one, two and four Sparks, with drafting ON and OFF. It
  matched in 66 of 66 responses in every compare
  ([benchmarks](docs/benchmarks.md#current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks)).
- On the same main, legacy FP8 matched drafting ON against OFF on each Spark
  count. But its two-Spark and four-Spark cards matched each other in only 7
  of 66 responses. This is a known limit of the legacy FP8 route.
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
  [client setup guide](harness/docs/passthrough-matrix.md) covers Codex,
  Hermes, OpenCode and Pi, directly or through the gateway.

## Measured results

<!-- bench-docs:start -->

### Latest measured main

These cards measured main revision
`76977dff88d36a85a8bfeaff6725d169e8a44845`.
The last card finished on 2026-10-28. They do not measure later engine changes
on main.

| Model | Sparks | Path | Drafting | Prose | Code | Structured | C1 | C4 | C8 | Cold 8K (s) | IDs saved / responses |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3.8-flash-next-fp4-local | 1 | local | OFF | 27.582 | 27.625 | 27.927 | 26.965 | 83.569 | 130.553 | 3.811 | 66/66 |
| qwen3.8-flash-next-fp4-local | 1 | local | ON | 42.039 | 80.458 | 99.93 | 64.074 | 143.929 | 177.496 | 3.759 | 66/66 |
| qwen3.8-flash-next-fp8 | 2 | ring | OFF | 26.823 | 26.709 | 26.856 | 25.839 | 66.618 | 99.298 | 6.005 | 66/66 |
| qwen3.8-flash-next-fp8 | 2 | ring | ON | 36.6 | 50.883 | 52.706 | 45.124 | 66.499 | 98.327 | 5.969 | 66/66 |
| qwen3.8-flash-next-fp4-local | 2 | tensor split | OFF | 35.071 | 35.375 | 35.487 | 33.814 | 82.476 | 126.344 | 2.153 | 66/66 |
| qwen3.8-flash-next-fp4-local | 2 | tensor split | ON | 53.792 | 99.972 | 128.426 | 76.87 | 162.493 | 229.457 | 2.186 | 66/66 |
| qwen3.8-flash-next-fp8 | 4 | path | OFF | 35.266 | 34.722 | 35.284 | 33.9 | 88.201 | 128.203 | 6.948 | 66/66 |
| qwen3.8-flash-next-fp8 | 4 | path | ON | 46.669 | 65.115 | 68.026 | 58.452 | 86.39 | 128.832 | 6.911 | 66/66 |
| qwen3.8-flash-next-fp4-local | 4 | pipeline | OFF | 28.157 | 28.177 | 28.503 | 27.642 | 84.974 | 139.335 | 2.177 | 66/66 |
| qwen3.8-flash-next-fp4-local | 4 | pipeline | ON | 42.758 | 81.287 | 101.673 | 65.82 | 145.757 | 192.895 | 2.176 | 66/66 |

Rates are card medians in completion tokens per second. Prose, code and
structured output each use one chat. C1, C4 and C8 give the total rates for
one, four and eight concurrent chats. Cold 8K is the median time to the first
token after an uncached 8192-token prompt. IDs saved counts responses with
recorded token IDs. This count does not prove that the IDs match.

The [benchmark history](docs/benchmarks.md) gives the card identities,
settings, exactness comparisons, methods, receipt status and limits.

<!-- bench-docs:end -->

FP4-Local is the supported release checkpoint. The FP8 rows are legacy
tests.

**After these cards.** Main `285fa8a` changed the engine after the cards
above. With drafting on, two Sparks now keep more prompt rows for the drafter
on prompts longer than 32K tokens (see the [changelog](CHANGELOG.md)). A
confirm run on 2026-10-28 compared `285fa8a` with the main before it,
`54392da`, on one pair of DGX Sparks with drafting ON (record
`2026-10-28_1713_confirm_285fa8a-vs-54392da`, not in the repository yet). The
token IDs were identical in all four server starts. A 129,220-token prompt
reached its first token in 40.366 s, against 42.090 s before (-4.10 %, mean
of two requests each). Eight chats reached 205.594 tok/s, against 206.735
(-0.55 %), on rounds where all chats started together. On all rounds, one slow
new-main start (195.717 tok/s) made eight chats slower: a mean of 200.656
tok/s against 205.935 (-2.56 %). Prose reached 54.378 and 55.014 tok/s in the
two new starts, against 54.862 and 55.015 (-0.44 %). The first token after a cold 8K prompt
took 2.167 and 2.181 s, against 2.183 and 2.182 s (0.39 % less time).
[The confirm run](docs/benchmarks.md#confirm-run-main-285fa8a) gives the
conditions and the limits.

## License

Vade's code is licensed under [Apache-2.0](LICENSE). The bundled tokenizer and
template fixtures keep their upstream licenses, the
[Code of Conduct](CODE_OF_CONDUCT.md) uses CC BY 4.0, and models keep their own
licenses. [NOTICE](NOTICE) lists the shipped files and their license texts,
and [Credits](CREDITS.md) keeps the source attribution.

## Documentation map

| Guide | What it covers |
| --- | --- |
| [INSTALL.md](INSTALL.md) | The whole install, starting with one Spark |
| [First run](harness/docs/first-run.md) | The launcher on your Spark, saved settings and later runs |
| [CLI reference](cli/README.md) | Commands, configuration and recovery |
| [Examples](examples/README.md) | Short examples you can run: curl, a Rust API client, Claude Code, Codex, vade bench and a two-Spark launch |
| [Benchmarks](docs/benchmarks.md) | Measured results, receipts and how to reproduce them |
| [Architecture](docs/architecture.md) | How requests reach the engine, and the exactness rule |
| [Glossary](docs/GLOSSARY.md) | The serving terms these docs use |
| [Harness](harness/README.md) | The coding agent, its tools, and the gateway for other clients |
| [Engine docs](engine/docs/README.md) | How the engine runs, its GPU kernels and how it is validated |
| [All documentation](docs/README.md) | Guides for deploying and developing Vade |
| [Contributing](CONTRIBUTING.md) and [agent rules](AGENTS.md) | How to change Vade: house rules, validation and review |
| [Security](SECURITY.md) | How to report a vulnerability privately |
