# Why Vade

Vade runs a large language model on an NVIDIA DGX Spark that you own. It has three parts: an inference engine, a harness with a coding agent and a chat, and the `vade` command-line tool that installs, starts and checks them. All three are written in Rust and use only its standard library.

## What it is for

Vade turns one Spark, or several linked Sparks, into a private model server. You get these three things:

- **A terminal chat.** `vade chat` opens a coding agent and a plain chat in your terminal.
- **An OpenAI-compatible API.** The engine serves the API on the Spark, so the clients you already use work.
- **A check of exactness.** `vade verify` checks the engine's answers on your own machine.

Vade serves one model, Qwen3.8 Flash-Next, text only. It serves two checkpoints of that model. The NVIDIA FP4-Local checkpoint runs on one Spark or more. The publisher's FP8 checkpoint does not fit on one Spark, so it runs on 2, 4 or 6 linked Sparks. Our cards cover two and four Sparks. A card is the result file of one RigMark run.

## Why it exists

Vade is built on three choices.

**Answers you can repeat.** A floating-point sum can change in its last bits when the order of its terms changes. If that order follows the load, the model's answer can change with the load too. Vade's rules do not allow this. The order in which Vade adds the numbers follows the model's fixed tensor dimensions. It does not follow the batch, the prompt length or the number of chats. So with the same build and settings, you can test a prompt once and get the same reply later.

**A tree you can read.** The engine, its GPU kernels, the CLI, the harness and the repository tools use no third-party crates and no Python. The engine does not use the CUDA toolkit, cuBLAS, CUTLASS or NCCL ([architecture](architecture.md)). Vade writes its own GPU kernels in Rust.

**The hardware you have.** Vade targets one kind of machine. It reads the model's shapes and data types from the checkpoint. Then it plans the work for one Spark, two Sparks or a chain of Sparks.

## What "exact" means

The rule is this: the same request, on the same build, checkpoint and settings, gives identical token IDs on every run and on every Spark. Supported Spark counts must agree too. The Spark count is the number of linked Sparks that serve one model. Similar text is not enough. The token IDs must match.

The rule also covers how a request runs. A request must give the same IDs when it runs alone, in a batch, in chunks or from the prefix cache. With greedy decoding, drafting on and drafting off must give the same IDs too. Drafting is Vade's speculative decoding: a small extra head guesses a few tokens ahead, and the full model checks each guess.

Greedy decoding is the default on FP4-Local. A request with no sampling fields always gets the most likely token. The exactness checks use greedy requests.

Our checks show these results today:

- **FP4-Local.** On main `8976c4f` (2026-11-02 morning), the FP4-Local cards on one, two and four Sparks gave the same token IDs in 66 of 66 responses (64 of 64 on one Spark, whose cards hold 64). Each Spark count had cards with drafting on and with drafting off ([results](results.md#fp4-local-token-ids)).
- **The prefix cache.** In the same runs, the second question on a 32K-token chat gave the same token IDs with the prefix cache and without it.
- **The merge rule.** Before a change to the engine's arithmetic, scheduling, caching or transport merges, it runs at least three times on real Sparks. The results are compared across at least two Sparks ([contributing](../.github/CONTRIBUTING.md)).

Three points apply to FP8:

- **FP8 samples by default.** On FP8, a field that a request leaves out takes the checkpoint's sampled value. So a request with no sampling field is sampled. For greedy replies, send `temperature` 0 with neutral penalties. The gateways add a full greedy set when a request sends none ([FP8 install](install-fp8.md#defaults-and-drafting)).
- **Two and four Sparks in the cards.** In the cards of 2026-10-28, 2026-10-30 and 2026-10-31 (main `dc32b7c`), FP8 on two Sparks and FP8 on four Sparks gave different token IDs. Their cards matched in 7 of 66 responses. Each Spark count was stable on its own: at `dc32b7c`, drafting on and off matched in 66 of 66 responses on two Sparks and on four ([benchmarks](history/benchmarks-2026-10.md#fp8-on-two-and-four-sparks-different-token-ids)).
- **Two and four Sparks since the change.** A later change, approved on 2026-10-31, makes FP8 on four Sparks add its partial sums in the two-Spark order. Four Sparks now give the same token IDs as two. The four-Spark cards of 2026-11-01, with drafting on and off, match the two-Spark cards of 2026-10-31 in 66 of 66 responses ([FP8 token IDs](results.md#fp8-token-ids)). Six Sparks still give other IDs ([changelog](../CHANGELOG.md)).

`vade verify` checks the drafting part of the rule on your own engine. It sends the same requests several times with drafting on and off, and it compares the token IDs. Neither `vade verify` nor our cards check every mix of batching and caching.

## How fast it is

The FP4-Local rows are from the 2026-11-02 morning cards at main `8976c4f`. The FP8 rows are from the 2026-11-01 morning cards at main `1a07e3a`, with a fixed drafting depth 2. Every row is one server start.

Rates are completion tokens per second. Prose, code and C8 ran with drafting on. C8 is the total rate of eight chats at once.

<div align="center">

| Setup | Prose | Code | C8 |
| :---: | :---: | :---: | :---: |
| FP4-Local, one Spark | 42.1 | 81.3 | 179.8 |
| FP4-Local, two Sparks | 54.5 | 101.7 | 230.9 |
| FP4-Local, four Sparks | 61.9 | 112.6 | 243.5 |
| FP8, two Sparks | 37.8 | 53.0 | 98.3 |
| FP8, four Sparks | 51.4 | 72.3 | 131.4 |

</div>

The next table gives the time to the first token for a long prompt. The cards send the cold 8K prompt with drafting off.

<div align="center">

| Setup | Cold 8K prompt | 32K prompt |
| :---: | :---: | :---: |
| FP4-Local, one Spark | 3.77 s | 15.76 s |
| FP4-Local, two Sparks | 2.12 s | 10.14 s |
| FP4-Local, four Sparks | 1.93 s | 9.70 s |
| FP8, two Sparks | 5.96 s | 24.08 s |
| FP8, four Sparks | 6.82 s | 27.38 s |

</div>

[Results and receipts](results.md) gives the receipt of each row, the machines, the commands and what you can run again. With the prefix cache, the second question on a 32K-token chat gave its first token in 0.37 s on one Spark, in 0.77 s on two and in 1.06 s on four.

## What it is not

- **Not generic yet.** The house rules ask the engine to take its plan from the checkpoint's configuration, tensor shapes and data types. The rules allow no check of the model's name, no constants for one checkpoint and no shortcuts for one prompt ([adding a model](adding-a-model.md)). The code does not meet that rule everywhere yet. For example, the hidden width 2560 of the shipped checkpoints is a constant in `engine/crates/engine/src/flash/blocks.rs` and `engine/crates/engine/src/flash/next_prefill.rs`. A checkpoint with other shapes needs engine work first.
- **Not a general GPU server.** The engine runs only on DGX Spark (NVIDIA GB10) with Linux `aarch64`. It does not run on other GPUs.
- **Not a trainer.** Vade serves models. It does not train or fine-tune them.
- **Not for images.** Vade reads and writes text only.
- **Not every checkpoint.** Vade supports two checkpoints of Qwen3.8 Flash-Next: FP4-Local and FP8.
- **Not fast on every Spark count.** Four Sparks linked by RDMA write one chat faster than two (61.9 against 54.5 tok/s prose in the table above). Three Sparks, five or more, and four over TCP run the layer chain. One chat on the chain writes at about the speed of one Spark ([what runs where](../README.md#what-runs-where)). The layer chain keeps no prefix cache.

## Where to go next

<div align="center">

| Page | What it is for |
| :---: | :---: |
| [INSTALL.md](INSTALL.md) | Install Vade on one Spark |
| [Benchmarks](benchmarks.md) | The measured results and how to repeat them |
| [Architecture](architecture.md) | How a request reaches the engine |

</div>
