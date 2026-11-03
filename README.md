# Vade

<picture>
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-static.png">
  <img src="docs/assets/vade-banner.webp" width="100%" alt="VADE drawn in lit monospace characters over a slow field of dots and dashes, as on vade.lol, above one line of text: Pure-Rust inference engine for the DGX Spark.">
</picture>

Vade is an inference engine for the NVIDIA DGX Spark. It is written in Rust from the GPU kernels up, and it has zero third-party crates. Its one rule is exactness: the same request, with the same settings, gives the same answer every time, and linking Sparks together does not change that.

The kernels compile to PTX, NVIDIA's portable GPU assembly, and use short inline PTX for the few instructions that have no Rust spelling. The installer and the Spark setup scripts are shell scripts. At run time, Vade needs only the NVIDIA driver. The installer also needs curl, and a source build also needs a C compiler (`cc`) and Rust.

One command installs it. One more pulls the model, 102 GB. One more starts it.

```sh
curl -fsSL https://vade.lol/install.sh | sh
```

If you want to check the script before you run it, its SHA-256 is at https://dl.vade.lol/install.sh.sha256.

Then get the model with `vade pull` and run `vade up`. The engine is ready for any OpenAI client, and for `vade chat`, a coding agent in your terminal.

<a id="what-it-is"></a>

## What you get

- **Exact answers.** The same request and settings give the same token IDs on every run. You can test this yourself with `vade verify`.
- **Speed from more Sparks.** Link two or four Sparks together and one chat gets faster. The answers stay the same.
- **FP4 and FP8.** Vade runs Qwen3.8 Flash-Next in both formats. FP4-Local fits on one Spark. FP8 needs two or more.
- **An OpenAI-compatible API.** Any OpenAI client works. Claude Code, Codex and other coding tools connect with a short setup. `vade chat` opens Vade's own coding agent.
- **Receipts.** Every row of the benchmark table has its records on GitHub.

<a id="pick-your-path"></a>
<a id="install"></a>
<a id="no-model-yet"></a>

## Quick start

You need a DGX Spark with curl, a driver with CUDA 13.0 or newer, and nothing else using the GPU. Only a build from source also needs a C compiler (`cc`).

Run these three commands on the Spark. Allow time for the download and the first full model check.

```sh
curl -fsSL https://vade.lol/install.sh | sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade up
```

1. **Install** checks the GPU and the driver, then installs the prebuilt Vade build. It does not need sudo, and it does not change your shell profile. If your shell cannot find `vade` afterwards, run the `export` line that the installer prints.
2. **Pull** downloads the model (102 GB) and checks every file. The long hash after `@` pins the exact revision we tested. If the download stops, run the same command again. It continues where it stopped.
3. **Up** starts the engine on this Spark. `vade launch` starts it and opens the coding agent in the current folder. `vade down` stops it.

**First-run time.** The [download record](docs/benchmarks.md#rented-cloud-gb10-vms) gives about 80 minutes to over three hours on rented GB10 VMs. With the model on disk, the full check still takes time on each Spark. The [install record for build 5a22651](bench/receipts/2026-10-24/2026-10-24_2014_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/README.txt) gives 74.2 s and 77.9 s for the checks on a pair of Sparks. After the checks and install, `vade up` took 27.0 s. Times depend on the network, disk and build.

The full check runs once per Spark while its files and check cache stay the same. `vade pull` writes a model check record on that Spark. A local `vade install` also writes the record after a check, if the model folder permits writes. Keep that record so later installs can reuse it. A copy to another Spark needs its own check. See [first-run time and records](docs/INSTALL.md#first-run-time-and-records).

Next steps:

- Two or four Sparks: [Two and more Sparks](docs/INSTALL.md#two-and-more-sparks).
- FP8: [FP8 install](docs/install-fp8.md).
- A build from a clone, and every setting: [INSTALL.md](docs/INSTALL.md).
- A coding agent is doing the setup for you: give it the [agent quickstart](docs/agents/quickstart.md).

## Use the API

The engine serves the OpenAI protocol on the Spark at port 18080, so the client you already use can talk to it. Point it at `http://127.0.0.1:18080/v1` with any API key.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3.8-flash-next-fp4-local",
    messages=[{"role": "user", "content": "Name the capital of France in one word."}],
    extra_body={"enable_thinking": False},
)
print(reply.choices[0].message.content)
```

The model thinks before it answers unless you turn that off, as this example does. The [OpenAI API](docs/openai-api.md) page lists every endpoint and field. [INSTALL.md](docs/INSTALL.md) shows how to reach the engine from another computer. The [client setup guide](harness/docs/passthrough-matrix.md) walks through Claude Code, Codex, Hermes, OpenCode and Pi.

<a id="benchmark-results"></a>

## Benchmarks

The numbers are tokens per second (tok/s). Drafting is Vade's speculative decoding: the model's own prediction head guesses a few tokens ahead, and the full model checks each guess. It is on by default. On FP4-Local, the first column is what you get out of the box. The middle column shows the same runs with drafting off.

<!-- bench-docs:start -->

<div align="center">

| Setup | One chat | One chat, drafting off | Eight chats at once (total) |
| :---: | :---: | :---: | :---: |
| FP4-Local · 1 Spark | 42.1 | 27.9 | 179.8 |
| FP4-Local · 2 Sparks | 54.5 | 36.1 | 230.9 |
| FP4-Local · 4 Sparks | 61.9 | 41.8 | 243.5 |
| FP8 · 2 Sparks | 39.5 | 28.4 | 106.4 |
| FP8 · 4 Sparks | 52.6 | 36.9 | 136.2 |

</div>

We measured main `acb49bc` on our own Sparks on the afternoon of 2026-11-01 for the FP8 rows, and main `8976c4f` on the morning of 2026-11-02 for the FP4-Local rows. One chat writes prose of up to 8192 tokens. Every row is one server start. Drafting on means a fixed draft depth: 3 for FP4-Local and 2 for FP8. A default FP8 server chooses the depth itself. [Results](docs/results.md#current-fp8-main-acb49bc-2026-11-01-afternoon) gives those speeds too. The raw record of every run is in the receipts: [FP8](bench/receipts/2026-11-01/README.md) and [FP4-Local](bench/receipts/2026-11-02/README.md).

<!-- bench-docs:end -->

The [2026-11-02 16:39 CST card run](docs/results.md#fp4-local-cached-turn-main-2de87be-2026-11-02-afternoon) measures a 32K-token chat with FP4-Local on two Sparks. Its cached second question has a median first-token time of 0.395 s. The same section gives the cold times and the raw samples.

[Results](docs/results.md) has every cell of these runs, with the build, the server commands and the request settings. [Benchmark history](docs/benchmarks.md) keeps every earlier run.

## Exactness

By default, most inference engines give slightly different answers from one run to the next, and different answers again when you add GPUs. Vade does not.

Exact means the same request with the same settings gives the same token IDs. This holds on every run and on every Spark count under "Covered today" below. It holds for a request alone or in a batch. It holds with or without the prefix cache. For requests that always pick the most likely token, it also holds with drafting on or off.

### Covered today

- FP4-Local on 1, 2 and 4 Sparks.
- FP8 on 2 and 4 Sparks with the default drafting mode, since 2026-10-31. The four-Spark cards of 2026-11-01 match the two-Spark cards of 2026-10-31 in 66 of 66 replies ([FP8 token IDs](docs/results.md#fp8-token-ids)).
- Seeds. A request with a fixed `seed` gets the same sampled reply every time. Tested on FP4-Local on 1 and 2 Sparks, with drafting off.

### Not covered yet

- FP8 on six Sparks.
- The opt-in `fast` and `sampled` draft modes on FP8.
- Seeds with drafting on, on FP8, or on four Sparks.
- Other NVIDIA driver versions.

<a id="defaults-by-route"></a>

**Defaults.** FP4-Local always picks the most likely token unless a request sets sampling fields, so it is exact out of the box. FP8 samples by default, with a new seed for each request. For exact FP8 replies, send `temperature` 0, `repetition_penalty` 1, `presence_penalty` 0 and `frequency_penalty` 0 ([FP8 defaults](docs/install-fp8.md#defaults-and-drafting)).

**Driver.** We tested the same-answer promise on one NVIDIA driver, version 580.178.04 ([receipts](bench/receipts/2026-10-31/README.md#what-was-run)). If NVIDIA ships a new driver, we test again before we say it still holds.

To check your own engine, run `vade verify`. [Token IDs](docs/results.md#token-ids) shows our checks. On main `1a07e3a`, FP8 on two and four Sparks gave the same token IDs ([FP8 token IDs](docs/results.md#fp8-token-ids)). [Architecture](docs/architecture.md#exactness-and-model-quality) gives the full rule and what each check proves.

<a id="supported-systems"></a>
<a id="what-works-today-and-what-doesnt-yet"></a>

## What runs where

- **Model.** Qwen3.8 Flash-Next, text only, at pinned revisions of `nvidia/Qwen3.8-Flash-Next-FP4-Local` and `Qwen/Qwen3.8-Flash-Next-FP8`. To add a model, read [Adding a model](docs/adding-a-model.md).
- **Hardware.** DGX Spark (NVIDIA GB10) on Linux `aarch64`.
- **Sparks.** FP4-Local runs on 1, 2 or 4 Sparks over RDMA, or as a layer chain of 3 to 24 Sparks. We have measured chains up to four Sparks. FP8 runs on 2, 4 or 6 Sparks. We have measured 2 and 4 Sparks, and they give the same token IDs. Six Sparks start, but no benchmark run has measured them, and they give other token IDs. FP8 does not fit on one Spark.
- **Controller.** A controller drives linked Sparks over SSH. It runs on Linux or macOS. From Windows, connect to a Spark and work there ([how](docs/INSTALL.md#from-a-windows-computer)).
- **Coding agent.** The agent reads, searches and edits files anywhere. Its shell tool needs the macOS sandbox, so on the Spark itself it does not run commands.
- **Prefix cache.** On by default on one Spark, on the FP4-Local tensor split (two Sparks, or four over RDMA) and on FP8. Off on the FP4-Local layer chain ([settings](docs/INSTALL.md#the-prefix-cache)).

## Where things are

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [INSTALL.md](docs/INSTALL.md) | The full install, settings and recovery |
| [Agent quickstart](docs/agents/quickstart.md) | Setup steps written for coding agents |
| [FP8 install](docs/install-fp8.md) | FP8 on two or more Sparks |
| [CLI reference](cli/README.md) | Every `vade` command |
| [OpenAI API](docs/openai-api.md) | Endpoints and request fields |
| [Client setup](harness/docs/passthrough-matrix.md) | Claude Code, Codex, Hermes, OpenCode and Pi |
| [Harness](harness/README.md) | The coding agent and its tools |
| [Examples](examples/README.md) | Short scripts against a running engine |
| [Results](docs/results.md) | Every benchmark cell with its settings |
| [Receipts](bench/receipts/README.md) | The records behind the numbers |
| [Benchmarks](docs/benchmarks.md) | All runs, and how to repeat them |
| [Architecture](docs/architecture.md) | How a request flows, and the exactness rule |
| [Why Vade](docs/why-vade.md) | Why Vade exists |
| [Security](.github/SECURITY.md) | How to report a vulnerability privately |
| [Code of conduct](.github/CODE_OF_CONDUCT.md) | How we treat each other, and how to report a problem |
| [Contributing](.github/CONTRIBUTING.md) | House rules, validation and review |
| [Changelog](CHANGELOG.md) | Changes since v0.1.0 |
| [Credits](docs/CREDITS.md) | Everyone Vade builds on, on one page |

</div>

Everything else is in [docs](docs/README.md). Coding agents that change Vade read [AGENTS.md](AGENTS.md) first.

Vade is licensed under [Apache-2.0](LICENSE). The bundled tokenizer and template fixtures keep their own licenses, and so do the models. [NOTICE](NOTICE) lists them. [Credits](docs/CREDITS.md) names everyone Vade builds on, on one page.
