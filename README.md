# Vade

Vade v0.1.0 is available at <https://github.com/wappy-x/vade>.

Vade runs a large language model on your own NVIDIA DGX Spark and gives you a
terminal coding agent, a plain chat and an OpenAI-compatible API, all served
from the Spark. Today it serves one model, `nvidia/Qwen3.8-Flash-Next-FP4-Local`
(text only), on one Spark or on linked Sparks. It is written in Rust using only
the standard library, and it is built so that the same request on the same
setup returns the same token IDs every run; `vade verify` checks that on your
machine.

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark-static.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-light-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark.svg">
  <img src="docs/assets/vade-banner-light.svg" width="100%" alt="Vade: an inference engine and coding-assistant harness for DGX Spark.">
</picture>

Current numbers: [FP4-Local benchmark recipes, results and qualifications](docs/benchmarks.md); [receipt availability](bench/receipts/README.md) identifies the public evidence.

<a id="pick-your-path"></a>
## Install

Follow [INSTALL.md](INSTALL.md). On one Spark you need no other computer: you
install a few Ubuntu packages and the pinned Rust toolchains, clone this
repository, build the GPU kernels once, download and check the 102.26 GB
checkpoint, and then run, from the checkout:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
unset VADE_CONFIG
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
vade launch
```

`vade launch` builds and starts the engine, waits until it is ready and opens
chat in the current folder (here, the checkout; to work on another project,
`cd` there and run `vade chat`). Type `/exit` to leave chat; the engine
keeps running until `vade down`. The checkpoint download and the first build
take most of the time; an end-to-end install time has not been measured.

To drive Sparks from a Linux or macOS computer instead, see
[Route B](INSTALL.md#route-b-from-a-controller); to link Sparks, see
[two and more Sparks](INSTALL.md#two-and-more-sparks).

## What you get

- **Chat in the terminal.** `vade chat` (or `vade launch`) opens Vade's coding
  agent, which can read, search and edit files in the folder you started it
  from. With less than 8192 tokens of context it opens plain chat instead.
  On macOS the agent's shell sees system and Homebrew tools only; for a
  rustup-installed Rust, see [the controller notes in INSTALL](INSTALL.md#route-b-from-a-controller).
- **An OpenAI-compatible API** at `http://127.0.0.1:18080/v1`. If that port was
  busy at first setup, `vade status` prints the server address in use; the API
  is under its `/v1`. Chat requests think by default, and the reasoning counts
  against the same `max_tokens` as the answer; send `"enable_thinking": false`
  (or `"reasoning_effort": "none"`) for the answer alone.
- **An exactness check.** `vade verify` sends the same requests repeatedly with
  drafting on and off, compares the token IDs and writes `vade-verify.json`.
- **A benchmark client.** `vade bench` measures the running engine and writes
  `vade-bench.json`; see [benchmarks](docs/benchmarks.md).
- **Lifecycle commands.** `vade status` reports readiness, `vade doctor`
  diagnoses setup, `vade logs` shows the engine log and `vade down` stops the
  engine this installation started.

<a id="defaults-by-route"></a>
Defaults: drafting (MTP) is ON: the model's multi-token-prediction head
proposes a few tokens ahead and the full model checks them, so greedy answers
stay the same. Requests without sampling settings are decoded greedily (always
the most likely next token). On the Spark and from a controller, Vade asks
for context 32768 tokens and output up to 8192, and both start at 32768. On the
Spark, context is lowered (16384, 8192, then 4096) only if the engine refuses
it for memory, with a one-line notice. The engine admits concurrent chats from
context and memory; on a controller, `vade status` prints the count.
`--context 262144` selects the single-chat long-context preset. The
[settings guide](INSTALL.md#settings-you-may-change) lists every default and
how to change it.

<a id="supported-systems"></a>
## Supported today / not yet

- **One model.** `nvidia/Qwen3.8-Flash-Next-FP4-Local` at its pinned publisher
  revision, text only. Older FP8 routes remain as
  [legacy guides](docs/install-legacy-fp8.md). Other checkpoints do not run;
  [adding a model](docs/adding-a-model.md) is the design note for qualifying one.
- **DGX Spark only.** The engine runs on DGX Spark / NVIDIA GB10 with Linux
  `aarch64`, and nothing else.
- **Controllers.** On the Spark itself, any SSH client works, including from
  Windows. A separate controller can be Linux (`aarch64`, `x86_64`) or macOS;
  the full controller lifecycle was tested from macOS, and a Linux controller
  is untested end to end.
- **No Windows controller.** The CLI does not build for Windows. Work on the
  Spark over SSH, or try WSL 2 (untested).
- **No shell tool on Linux.** The agent's shell tool and workspace checks need
  the macOS sandbox, so in chat on the Spark (or any Linux computer) they are
  unavailable. Reading, searching, line edits and patches work everywhere.
- **Two Sparks read prompts faster; one chat does not write faster.** In the
  published runs a second Spark shortened cold prompt reading and, on cloud
  VMs, raised throughput for many concurrent chats, but a single chat did not
  get faster; see [what the measurements mean](docs/benchmarks.md#what-this-means).
  Published measurements cover one and two Sparks. The
  [FP4-Local guide](cli/docs/fp4-local.md) describes chains of up to 24 Sparks;
  topologies beyond two have no published measurement.
- **Prefix cache is opt-in.** Reusing a cached prompt prefix between requests
  is OFF by default, works on one Spark only, and is enabled with
  `VADE_FLASHNEXT_PREFIX_CACHE=1`; see the
  [settings guide](INSTALL.md#settings-you-may-change).

## Exactness

Vade's rule is that the same request with the same setup (build, checkpoint,
number of Sparks, settings) gives identical token IDs on every run and every
Spark, whether it runs alone or batched, cached or not, and, with greedy
decoding, with drafting ON or OFF. `vade verify` checks the drafting part on
your own engine and prints `IDENTICAL: 20/20 runs, 1 machine, N tokens` on a
pass, where N is the reply's token count; a pass covers the requests it sent,
not every batching or cache combination. The two-Spark home benchmark run
recorded identical token IDs with drafting ON and OFF in 201 of 201 measured
request pairs
([receipt](bench/receipts/2026-10-24/2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/README.txt)).
[Architecture: exactness and model quality](docs/architecture.md#exactness-and-model-quality)
explains the full rule and what each check does and does not prove.

## Use our harness, put it in front, or bring your own

- **Ours:** `vade chat` opens Vade's agent on the running engine.
- **In front:** `vade-gateway` sits between your existing client and the engine on `http://127.0.0.1:8787/v1`; see [the client gateway](INSTALL.md#client-gateway).
- **Your own:** point any OpenAI-compatible client at `http://127.0.0.1:18080/v1`; the [client setup guide](harness/docs/passthrough-matrix.md) covers Codex, Hermes, OpenCode and Pi, direct or through the gateway.

## Measured results

Model `nvidia/Qwen3.8-Flash-Next-FP4-Local`, context 32768, greedy, thinking
OFF, receipts from 2026-10-24. Home rows: release build `73f7a9d`; the v0.1.0
engine differs from it only in how startup admission counts free memory (main
`2c64d0a`). Cloud rows: main `4681d8b` from the same morning; no release-build cloud
cards exist yet. Tok/s means completion tokens per wall second; the 8-stream
rate covers all eight streams together.

| Sparks | Where | Drafting | Code tok/s (single stream) | Prose tok/s (single stream) | 8 streams aggregate tok/s | Cold 8K first token (s) |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | home DGX Spark | OFF | 27.7 | 27.7 | 135.9 | 3.87 |
| 1 | home DGX Spark | ON | 76.7 | 42.0 | 172.6 | 3.85 |
| 2 | home DGX Spark | OFF | not measured | 27.705 † | 133.877 | 2.776 |
| 2 | home DGX Spark | ON | not measured | 38.386 † | 162.558 | 2.853 |
| 1 | rented cloud GB10 VM | OFF | 24.7 | 23.4 | 43.0 | 3.89 |
| 2 | rented cloud GB10 VM | OFF | 25.4 | 23.2 | 70.0 | 3.03 |

† The home 2-Spark rows are custom runner summaries, not RigMark cards: prose
uses 3 samples with a 1024-token cap, so it is not directly comparable with the
other rows, and code was not measured. Cloud drafting ON exists only as timing
runs, not cards, so it has no row here.

Rows, top to bottom, come from
[1444](bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-thinking-off/),
[1438](bench/receipts/2026-10-24/2026-10-24_1438_qwen3.8-flash-next-nvfp4_local-1spark_73f7a9d-drafting-on-thinking-off/),
[1444 pair](bench/receipts/2026-10-24/2026-10-24_1444_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-off-thinking-off/),
[1452 pair](bench/receipts/2026-10-24/2026-10-24_1452_qwen3.8-flash-next-nvfp4_local-2spark_73f7a9d-drafting-on-thinking-off/),
[1046](bench/receipts/2026-10-24/2026-10-24_1046_qwen3.8-flash-next-nvfp4_local-1spark_4681d8b-drafting-off-thinking-off/) and
[1006](bench/receipts/2026-10-24/2026-10-24_1006_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-off-thinking-off/).
Receipts and recipes: [docs/benchmarks.md](docs/benchmarks.md)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmarks-2026-10-24-dark.svg">
  <img src="docs/assets/benchmarks-2026-10-24-light.svg" width="100%" alt="Grouped bar chart of decode speed in tokens per second. On one home DGX Spark, release build 73f7a9d, drafting ON raises single-stream code from 27.7 to 76.7, prose from 27.7 to 42.0, and the 8-stream aggregate from 135.9 to 172.6. Cloud GB10 cards on main 4681d8b with drafting OFF: one VM code 24.7, prose 23.4, 8 streams 43.0; two VMs code 25.4, prose 23.2, 8 streams 70.0. Cloud drafting ON values are timing runs, not cards: one VM code 64.0 and prose 33.6, two VMs code 56.5 and prose 32.0.">
</picture>

The figure draws the same cards as the table's one-Spark rows (1444, 1438)
and cloud rows (1046, 1006); it leaves out the home 2-Spark rows and adds the cloud drafting ON timing runs as
textured bars. Every label is the value printed in a [2026-10-24 receipt](bench/receipts/2026-10-24/README.md). Reproduce the figure: `python3 -I tools/benchmarks-chart.py`.

## Licence

Vade's code is licensed under [Apache-2.0](LICENSE). Bundled tokenizer and
template fixtures keep their upstream licences, the
[Code of Conduct](CODE_OF_CONDUCT.md) uses CC BY 4.0, and models keep their own
licences. [NOTICE](NOTICE) lists the shipped files and their licence texts;
[Credits](CREDITS.md) preserves source attribution.

## Documentation map

| Guide | Purpose |
| --- | --- |
| [INSTALL.md](INSTALL.md) | Complete install path, one Spark first |
| [First run](harness/docs/first-run.md) | Local launcher, saved settings and later runs |
| [CLI reference](cli/README.md) | Commands, configuration and recovery |
| [Benchmarks](docs/benchmarks.md) | Measured results, receipts and how to reproduce them |
| [Architecture](docs/architecture.md) | How requests reach the engine; the exactness rule |
| [Glossary](docs/GLOSSARY.md) | Serving terms |
| [Harness](harness/README.md) | Agent tools and client gateway |
| [Engine docs](engine/docs/README.md) | Execution, kernels and validation |
| [All documentation](docs/README.md) | Deployment and development guides |
| [Contributing](CONTRIBUTING.md) and [agent rules](AGENTS.md) | How to change Vade: house rules, validation, review |
| [Security](SECURITY.md) | Private vulnerability reporting |
