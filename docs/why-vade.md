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
fit on one Spark, so it runs on 2, 4 or 6 linked Sparks.

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

- On main `cbbb11b` (2026-10-29), the FP4-Local cards on one Spark with
  drafting ON and OFF, and on two Sparks with drafting ON, gave the same token
  IDs in 64 of 64 responses
  ([benchmarks](benchmarks.md#current-2026-10-29-main-cbbb11b-home-dgx-sparks)).
- On main `76977df` (2026-10-28), four Sparks gave the same FP4-Local token
  IDs as one and two Sparks, in 66 of 66 responses
  ([benchmarks](benchmarks.md#current-2026-10-28-end-of-day-main-76977df-home-dgx-sparks)).
- In the same cards, the second question on a 32K-token chat gave the same
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
  cards matched in 7 of 66 responses on 2026-10-28. Each Spark count is
  stable on its own: on two Sparks at `cbbb11b`, drafting ON and OFF matched
  in 64 of 64 responses. A fix is in screening
  ([benchmarks](benchmarks.md#fp8-on-two-and-four-sparks-different-token-ids)).

`vade verify` checks the drafting part of the rule on your own engine. It
sends the same requests several times with drafting ON and OFF and compares
the token IDs. Neither `vade verify` nor our cards check every mix of
batching and caching.

## How fast it is

These are the 2026-10-29 cards at main `cbbb11b`. Prose, code and C8 ran
with drafting ON. The cards send the cold 8K prompt with drafting OFF. Rates
are completion tokens per second. C8 is the total rate of eight chats at
once.

| Setup | Prose | Code | C8 | Cold 8K prompt, first token | 32K prompt, first token |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP4-Local, one Spark | 41.9 | 80.2 | 176.7 | 3.79 s | 15.95 s |
| FP4-Local, two Sparks | 54.3 | 100.4 | 228.0 | 2.145 s | 10.09 s |
| FP8, two Sparks | 36.5 | 51.4 | 97.3 | 5.96 s | 24.05 s |

The [benchmarks page](benchmarks.md#current-2026-10-29-main-cbbb11b-home-dgx-sparks)
gives the method, the number of server starts and the limits. With the
prefix cache, the second question on a 32K-token chat gave its first token in
0.53 s on one Spark and 0.75 s on two.

## What it is not

- **Not tuned for one model.** The engine takes its plan from the
  checkpoint's configuration, tensor shapes and data types. The house rules
  do not allow a check of the model's name for speed, constants for one
  checkpoint or shortcuts for one prompt ([adding a model](adding-a-model.md)).
- **Not a general GPU server.** The engine runs only on DGX Spark (NVIDIA
  GB10) with Linux `aarch64`. It does not run on other GPUs.
- **Not a trainer.** Vade serves models. It does not train or fine-tune them.
- **Not for images.** Vade reads and writes text only.
- **Not every checkpoint.** Two checkpoints are supported: FP4-Local and FP8
  of Qwen3.8 Flash-Next.
- **Not faster on four Sparks for one chat.** With FP4-Local, four Sparks
  wrote prose at 42.8 tok/s in the 2026-10-28 cards, against 42.0 on one
  Spark and 53.8 on two. Four Sparks read a cold 8K prompt sooner than one.

## Where to go next

- [INSTALL.md](../INSTALL.md) installs Vade on one Spark.
- [Benchmarks](benchmarks.md) has the measured results and how to repeat
  them.
- [Architecture](architecture.md) explains how a request reaches the engine.
