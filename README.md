# Vade

Vade v0.1.0 is available at <https://github.com/wappy-x/vade>.

Vade runs a large language model on your own NVIDIA DGX Spark. It gives you
three things: a coding agent in your terminal, a plain chat, and an
OpenAI-compatible API. Today it serves one model,
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (text only), on one Spark or on linked
Sparks. Vade is written in Rust using only the standard library. It is built so
that the same request on the same setup returns the same token IDs on every
run, and `vade verify` lets you check that on your own machine.

<picture>
  <source media="(prefers-reduced-motion: reduce) and (prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark-static.svg">
  <source media="(prefers-reduced-motion: reduce)" srcset="docs/assets/vade-banner-light-static.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/vade-banner-dark.svg">
  <img src="docs/assets/vade-banner-light.svg" width="100%" alt="Vade: an inference engine and coding-assistant harness for DGX Spark.">
</picture>

For current numbers, how we measured them and their caveats, see the
[benchmarks page](docs/benchmarks.md). We call the saved record of a run a
receipt; [this list](bench/receipts/README.md) shows which receipts are public.

<a id="pick-your-path"></a>
## Install

Follow [INSTALL.md](INSTALL.md). With one Spark you don't need a second
computer: you install a few Ubuntu packages and the pinned Rust toolchains,
clone this repository, run one GPU kernel build while you are online, and
download and check the 102.26 GB checkpoint. Then, from the checkout, run:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
unset VADE_CONFIG
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
vade launch
```

`vade launch` builds and starts the engine, waits until it is ready, and opens
chat in the current folder, which here is the checkout. To work on another
project, `cd` there and run `vade chat`. Type `/exit` to leave chat; the engine
keeps running until you run `vade down`.

Most of the install time goes to the checkpoint download and the first build.
We have not measured an end-to-end install time.

To skip the Rust builds on the Spark, take the prebuilt launcher bundle and its
`SHA256SUMS` from the
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0)
instead; [INSTALL](INSTALL.md#optional-release-bundle) shows how.

If you would rather drive your Sparks from a separate Linux or macOS computer
(we call it a controller), see
[setting up from a controller](INSTALL.md#route-b-from-a-controller). To link
Sparks together, see [two and more Sparks](INSTALL.md#two-and-more-sparks).

## What you get

- **Chat in the terminal.** `vade chat` (or `vade launch`) opens Vade's coding
  agent. It can read, search and edit files in the folder you started it from.
  If your context is set below 8192 tokens, you get plain chat instead.
  On macOS, the agent's shell sees only system and Homebrew tools; if you
  installed Rust with rustup, see [the controller notes in INSTALL](INSTALL.md#route-b-from-a-controller).
- **An OpenAI-compatible API** at `http://127.0.0.1:18080/v1`. If that port was
  busy when you first set up, `vade status` prints the address in use instead,
  and the API is under its `/v1`. By default the model thinks before it
  answers, and the thinking counts against the same `max_tokens` as the answer.
  If you want only the answer, send `"enable_thinking": false` (or `"reasoning_effort": "none"`).
- **An exactness check.** `vade verify` sends the same requests several times
  with drafting on and off (see Defaults below), checks that the token IDs come
  back the same, and writes `vade-verify.json`.
- **A benchmark client.** `vade bench` measures the running engine and writes
  `vade-bench.json`; see [benchmarks](docs/benchmarks.md).
- **Commands to look after the engine.** `vade status` tells you whether it is
  ready, `vade doctor` diagnoses your setup, `vade logs` shows the engine log,
  and `vade down` stops the engine this installation started.

<a id="defaults-by-route"></a>
**Defaults.** The [settings guide](INSTALL.md#settings-you-may-change) lists
every default and how to change it. These are the ones you meet first:

- **Drafting is ON.** The model's multi-token-prediction (MTP) head guesses a
  few tokens ahead, and the full model checks each guess, so greedy answers
  stay the same.
- **Greedy decoding.** A request that sets no sampling options is decoded
  greedily: the engine always picks the most likely next token.
- **Context and output.** Whether you work on the Spark or from a controller,
  Vade starts by asking for a context of 32768 tokens and output of up to 8192
  tokens. On the Spark, Vade lowers the context (to 16384, then 8192, then
  4096) only if the engine refuses it for lack of memory, and it tells you so
  in a one-line notice.
- **Chats at once.** The engine decides how many chats it can run at once from
  the context and the memory it has; on a controller, `vade status` prints that
  count.
- **Long context.** `--context 262144` picks the single-chat long-context
  preset, whether you work on the Spark or from a controller. On the Spark it
  still has to pass the engine's memory check, and a refusal is reported, not
  lowered.

<a id="supported-systems"></a>
## What works today, and what doesn't yet

- **One model.** `nvidia/Qwen3.8-Flash-Next-FP4-Local` at its pinned publisher
  revision, text only. The older FP8 setups still have their own
  [legacy guides](docs/install-legacy-fp8.md). Other checkpoints do not run.
  [Adding a model](docs/adding-a-model.md) is our design note on what a new
  model must pass before we support it.
- **DGX Spark only.** The engine runs on DGX Spark (NVIDIA GB10) with Linux
  `aarch64`, and nothing else.
- **Your terminal and controllers.** To work on the Spark itself, any SSH
  client will do, including one on Windows. A controller can run Linux
  (`aarch64` or `x86_64`) or macOS. We tested the full controller lifecycle
  from macOS, and from Linux `aarch64` with a Spark as its own controller.
- **No Windows controller.** The CLI does not build for Windows. Work on the
  Spark over SSH, or try WSL 2 (untested).
- **No shell tool on Linux.** The agent's shell tool and workspace checks need
  the macOS sandbox, so they are not available in chat on the Spark or on any
  other Linux computer. Reading, searching, line edits and patches work
  everywhere.
- **A second Spark reads prompts faster, but one chat does not write faster.**
  In the published runs, a second Spark cut the time to read a cold prompt
  (one the engine has not seen before), and on rented cloud machines it also
  raised throughput when many chats ran at once. A single chat did not get
  faster; see [what the measurements mean](docs/benchmarks.md#what-this-means).
  The [FP4-Local guide](cli/docs/fp4-local.md) describes chains of up to 24
  Sparks, each holding a run of the model's layers and passing its work to the
  next. Our published measurements cover one and two Sparks, and one, two and
  four cloud GB10 VMs.
- **The prefix cache is opt-in.** The prefix cache reuses the start of an
  earlier prompt, so a new request that begins the same way does not have to
  read it again. It is OFF by default and works on one Spark only. To turn it
  on, set `VADE_FLASHNEXT_PREFIX_CACHE=1`; see the
  [settings guide](INSTALL.md#settings-you-may-change).

## Exactness

Vade's rule: the same request on the same setup (build, checkpoint, number
of Sparks and settings) gives identical token IDs on every run and on every
Spark. That holds whether the request runs alone or batched with others,
and whether or not it hits the cache. With greedy decoding, it also holds with
drafting ON or OFF.

`vade verify` checks the drafting part on your own engine. On a pass it prints
`IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the reply's token
count. A pass covers the requests it sent, not every mix of batching and
caching. In the two-Spark benchmark run on our own Sparks, the token IDs were
identical with drafting ON and OFF in 201 of 201 measured request pairs
([receipt](bench/receipts/2026-10-24/2026-10-24_0711_qwen3.8-flash-next-nvfp4_local-2spark_4681d8b-drafting-on-thinking-off/README.txt)).
[Architecture: exactness and model quality](docs/architecture.md#exactness-and-model-quality)
explains the full rule and what each check does and does not prove.

## Use our agent, add our gateway, or bring your own client

- **Our agent.** `vade chat` opens Vade's agent on the running engine.
- **Our gateway in front of your client.** `vade-gateway` sits between the
  client you already use and the engine, on `http://127.0.0.1:8787/v1`; see
  [the client gateway](INSTALL.md#client-gateway).
- **Your own client.** Point any OpenAI-compatible client at
  `http://127.0.0.1:18080/v1`. The
  [client setup guide](harness/docs/passthrough-matrix.md) covers Codex,
  Hermes, OpenCode and Pi, either directly or through the gateway.

## Measured results

Every row below is a RigMark card for the v0.1.0 release (commit `5a22651`),
measured on 2026-10-24. RigMark is a public benchmark, and a card is the
recorded result of one RigMark run. Each run used the model with a context of
32768 tokens, greedy decoding and thinking OFF. Drafting is ON by default; in
the OFF rows, every request sent `"draft":"plain"`. Code and prose are one
chat at a time, and the 8-chat rate covers all eight chats together. Tok/s
means completion tokens per second of wall-clock time.

| Setup | Drafting | Code tok/s | Prose tok/s | 8 chats tok/s | First token, 8K prompt (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| One DGX Spark | OFF | 27.8 | 27.8 | 136.1 | 3.82 |
| One DGX Spark | ON | 76.7 | 41.7 | 173.9 | 3.86 |
| Two DGX Sparks | OFF | 27.9 | 27.9 | 135.1 | 2.80 |
| Two DGX Sparks | ON | 65.1 | 38.3 | 168.6 | 2.79 |
| One cloud GB10 VM | OFF | 24.4 | 22.5 | 73.3 | 3.99 |
| One cloud GB10 VM | ON | 61.1 | 31.6 | 97.5 | 3.98 |
| Two cloud GB10 VMs | OFF | 24.4 | 22.9 | 112.2 | 2.95 |
| Two cloud GB10 VMs | ON | 54.7 | 30.5 | 138.3 | 2.95 |
| Four cloud GB10 VMs | OFF | 21.9 | 20.7 | 104.2 | 2.38 |
| Four cloud GB10 VMs | ON | 51.6 | 29.0 | 104.2 | 2.42 |

One chat runs at the same speed on every Spark we tried. We measured six of
our own Sparks one at a time: with drafting OFF, prose landed between 27.6 and
27.8 tok/s and code between 27.7 and 27.8; with drafting ON, prose landed
between 41.3 and 41.7 and code between 76.1 and 76.7.

The 8-chat rate moves more. Under that load the engine serves some chats
without drafting (each card's README says how many), so with drafting ON it
ranged from 130.6 to 173.9 tok/s across the six Sparks. The one-Spark ON row
above is the top of that range. One drafting-OFF run, on our Spark home-6,
landed at 86.6, and its receipt does not say why.

The four cloud VMs run as one engine, each VM holding a slice of the model's
layers. That chain is for models too big for one machine; it does not make one
chat faster than one Spark. In its drafting-ON run the engine served all eight
chats without drafting, so both four-VM rows show the same 8-chat rate.

Each row has its own receipt, named by the time of its run:

- One DGX Spark: [OFF (1937)](bench/receipts/2026-10-24/2026-10-24_1937_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off/), [ON (1943)](bench/receipts/2026-10-24/2026-10-24_1943_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off/)
- Two DGX Sparks: [OFF (2012)](bench/receipts/2026-10-24/2026-10-24_2012_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off/), [ON (2019)](bench/receipts/2026-10-24/2026-10-24_2019_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off/)
- One cloud GB10 VM: [OFF (1946)](bench/receipts/2026-10-24/2026-10-24_1946_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-off-thinking-off-cloud-a-vma/), [ON (1954)](bench/receipts/2026-10-24/2026-10-24_1954_qwen3.8-flash-next-nvfp4_local-1spark_5a22651-drafting-on-thinking-off-cloud-a-vma/)
- Two cloud GB10 VMs: [OFF (2027)](bench/receipts/2026-10-24/2026-10-24_2027_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-off-thinking-off-cloud-a/), [ON (2035)](bench/receipts/2026-10-24/2026-10-24_2035_qwen3.8-flash-next-nvfp4_local-2spark_5a22651-drafting-on-thinking-off-cloud-a/)
- Four cloud GB10 VMs: [OFF (2204)](bench/receipts/2026-10-24/2026-10-24_2204_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-off-thinking-off-cloud-a/), [ON (2213)](bench/receipts/2026-10-24/2026-10-24_2213_qwen3.8-flash-next-nvfp4_local-4spark_5a22651-drafting-on-thinking-off-cloud-a/)

The [benchmarks page](docs/benchmarks.md) has the other receipts and the
recipes for reproducing each run.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/benchmarks-2026-10-24-dark.svg">
  <img src="docs/assets/benchmarks-2026-10-24-light.svg" width="100%" alt="Grouped bar chart of decode speed in tokens per second for the v0.1.0 release, drafting OFF and then ON. One DGX Spark: code 27.8 to 76.7, prose 27.8 to 41.7, 8 chats 136.1 to 173.9. Two DGX Sparks: code 27.9 to 65.1, prose 27.9 to 38.3, 8 chats 135.1 to 168.6. One cloud GB10 VM: code 24.4 to 61.1, prose 22.5 to 31.6, 8 chats 73.3 to 97.5. Two cloud GB10 VMs: code 24.4 to 54.7, prose 22.9 to 30.5, 8 chats 112.2 to 138.3. Four cloud GB10 VMs: code 21.9 to 51.6, prose 20.7 to 29.0, 8 chats 104.2 either way.">
</picture>

The chart draws the same ten cards as the table. Every label is the value
printed in a [2026-10-24 receipt](bench/receipts/2026-10-24/README.md).
To redraw the chart, run `python3 -I tools/benchmarks-chart.py`.

## Licence

Vade's code is licensed under [Apache-2.0](LICENSE). The bundled tokenizer and
template fixtures keep their upstream licences, the
[Code of Conduct](CODE_OF_CONDUCT.md) uses CC BY 4.0, and models keep their own
licences. [NOTICE](NOTICE) lists the shipped files and their licence texts,
and [Credits](CREDITS.md) keeps the source attribution.

## Documentation map

| Guide | What it covers |
| --- | --- |
| [INSTALL.md](INSTALL.md) | The whole install, starting with one Spark |
| [First run](harness/docs/first-run.md) | The launcher on your Spark, saved settings and later runs |
| [CLI reference](cli/README.md) | Commands, configuration and recovery |
| [Benchmarks](docs/benchmarks.md) | Measured results, receipts and how to reproduce them |
| [Architecture](docs/architecture.md) | How requests reach the engine, and the exactness rule |
| [Glossary](docs/GLOSSARY.md) | The serving terms these docs use |
| [Harness](harness/README.md) | The coding agent, its tools, and the gateway for other clients |
| [Engine docs](engine/docs/README.md) | How the engine runs, its GPU kernels and how it is validated |
| [All documentation](docs/README.md) | Guides for deploying and developing Vade |
| [Contributing](CONTRIBUTING.md) and [agent rules](AGENTS.md) | How to change Vade: house rules, validation and review |
| [Security](SECURITY.md) | How to report a vulnerability privately |
