# Why Vade

Vade runs a large language model on an NVIDIA DGX Spark that you own. It has
three parts: an inference engine, a harness with a coding agent and a chat,
and the `vade` command-line tool that installs, starts and checks them. All
three are written in Rust and use only its standard library.

## What it is for

A DGX Spark is a desktop computer with one NVIDIA GB10 chip. Vade turns one
Spark, or several linked Sparks, into a private model server. You get:

- a coding agent and a plain chat in your terminal (`vade chat`),
- an OpenAI-compatible API on the Spark, for the clients you already use,
- `vade verify`, which checks the engine's answers on your own machine.

Vade serves one model, Qwen3.8 Flash-Next, text only. The NVIDIA FP4-Local
checkpoint runs on one Spark or more. The publisher's FP8 checkpoint does not
fit on one Spark, so it runs on 2, 4 or 6 linked Sparks. Our cards cover two
and four.

## Why it exists

Vade is built on three choices.

**Answers you can repeat.** A floating-point sum can change in its last bits
when the order of its terms changes. If that order follows the load, the
model's answer can change with the load too. Vade's rules do not allow this.
The order of each reduction follows the model's fixed tensor dimensions,
not the batch, the prompt length or the number of chats. So on the same
build and settings, you can test a prompt once and get the same reply
later.

**A tree you can read.** The engine, its GPU kernels, the CLI, the harness
and the repository tools use no third-party crates and no Python. The engine
does not use the CUDA toolkit, cuBLAS, CUTLASS or NCCL
([architecture](architecture.md)). Vade writes its own GPU kernels in Rust.

**The hardware you have.** Vade targets one kind of machine. It reads the
model's shapes and data types from the checkpoint and plans the work for one
Spark, two Sparks or a chain of Sparks.

## What "exact" means

The rule is: the same request, on the same build, checkpoint and settings,
gives identical token IDs on every run and on every Spark. Supported Spark
counts must agree too. Similar text is not enough; the token IDs must match.

The rule covers how a request runs. A request that runs alone, in a batch,
in chunks, or from the prefix cache must give the same IDs. With greedy
decoding, drafting ON and drafting OFF must give the same IDs too. (Drafting
is a small extra head that guesses a few tokens ahead; the full model checks
each guess.)

Greedy decoding is the default on FP4-Local: a request with no sampling
fields always gets the most likely token. The exactness checks use greedy
requests.

What our checks show today:

- On main `dc32b7c` (2026-10-31 morning), the FP4-Local cards on one, two
  and four Sparks, each with drafting ON and OFF, gave the same token IDs in
  66 of 66 responses ([results](results.md#token-ids)).
- In the same runs, the second question on a 32K-token chat gave the same
  token IDs with the prefix cache and without it.
- Before a change to the engine's arithmetic, scheduling, caching or
  transport merges, it runs at least three times on real Sparks and is
  compared across at least two Sparks ([contributing](../CONTRIBUTING.md)).

Two exceptions apply to FP8 today:

- On FP8, a field that a request leaves out takes the checkpoint's sampled
  value. So a request with no sampling field is sampled. Send `temperature`
  0 with neutral penalties for greedy replies. The gateways add a full
  greedy set when a request sends none
  ([FP8 install](install-fp8.md#defaults-and-drafting)).
- FP8 on two Sparks and on four Sparks gives different token IDs. Their
  cards matched in 7 of 66 responses on 2026-10-28, on 2026-10-30 and again
  on 2026-10-31 at main `dc32b7c`. Each Spark count is stable on its own: at
  `dc32b7c`, drafting ON and OFF matched in 66 of 66 responses on two Sparks
  and on four ([benchmarks](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).

`vade verify` checks the drafting part of the rule on your own engine. It
sends the same requests several times with drafting ON and OFF and compares
the token IDs. Neither `vade verify` nor our cards check every mix of
batching and caching.

## How fast it is

These are the 2026-10-31 morning cards at main `dc32b7c`. The one-Spark row
is the mean of two server starts; the other rows are one start each. Prose,
code and C8 ran with drafting ON. The cards send the cold 8K prompt with
drafting OFF. Rates are completion tokens per second. C8 is the total rate
of eight chats at once.

| Setup | Prose | Code | C8 | Cold 8K prompt, first token | 32K prompt, first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP4-Local, one Spark | 42.5 | 81.1 | 179.0 | 3.77 s | 15.63 s |
| FP4-Local, two Sparks | 54.9 | 102.3 | 232.2 | 2.15 s | 10.09 s |
| FP4-Local, four Sparks | 63.0 | 113.0 | 245.1 | 1.92 s | 8.98 s |
| FP8, two Sparks | 36.6 | 51.3 | 98.7 | 5.97 s | 24.12 s |
| FP8, four Sparks | 49.5 | 69.3 | 131.0 | 6.92 s | 27.81 s |

[Results and receipts](results.md) gives the receipt of each row, the
machines, the commands and what you can re-run. With the prefix cache, the
second question on a 32K-token chat gave its first token in 0.36 s on one
Spark and 0.76 s on two. Four FP4-Local Sparks keep no prefix cache.

## What it is not

- **Not generic yet.** The house rules ask the engine to take its plan from
  the checkpoint's configuration, tensor shapes and data types, with no check
  of the model's name, no constants for one checkpoint and no shortcuts for
  one prompt ([adding a model](adding-a-model.md)). The code does not meet
  that rule everywhere yet: for example, the hidden width 2560 of the
  shipped checkpoints is a constant in
  `engine/crates/engine/src/flash/blocks.rs` and
  `engine/crates/engine/src/flash/next_prefill.rs`. A checkpoint with other
  shapes needs engine work first.
- **Not a general GPU server.** The engine runs only on DGX Spark (NVIDIA
  GB10) with Linux `aarch64`. It does not run on other GPUs.
- **Not a trainer.** Vade serves models. It does not train or fine-tune them.
- **Not for images.** Vade reads and writes text only.
- **Not every checkpoint.** Two checkpoints are supported: FP4-Local and FP8
  of Qwen3.8 Flash-Next.
- **Not faster on four Sparks for one chat.** With FP4-Local, four Sparks
  wrote prose at 43.1 tok/s in the 2026-10-30 morning cards at `380d8e5`,
  against 42.6 on one Spark and 54.9 on two. Four Sparks read a cold 8K
  prompt sooner than one.

## Where to go next

- [INSTALL.md](../INSTALL.md) installs Vade on one Spark.
- [Benchmarks](benchmarks.md) has the measured results and how to repeat
  them.
- [Architecture](architecture.md) explains how a request reaches the engine.
