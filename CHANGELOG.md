# Changelog

This file lists the changes you would notice when you use Vade, newest first.
Changes that are merged but not yet released are under Unreleased.

## Unreleased

### Engine

- On one Spark, when one chat runs alone, the drafting depth can now rise to 7
  instead of 5, as on the layer pipeline. Measured on one Spark: code 75.6 to
  81.6 tokens per second, structured output 88.3 to 99.7, prose unchanged, with
  the same tokens. Batches keep their depth. `VADE_FLASHNEXT_MTP_DEPTH=5`
  restores the previous ceiling.
- On the layer pipeline, the check of drafted tokens now replays each
  Spark's layers as captured GPU work, and when one chat runs alone the
  drafting depth can rise to 7 instead of 3 (the resident depth in the
  [glossary](docs/GLOSSARY.md)). One Spark is unchanged, and
  `VADE_FLASHNEXT_MTP_DEPTH=3` restores the previous ceiling. The change
  targets one-chat speed on two Sparks. The published cards were measured
  before it; the next cards will show its effect.
- On the layer pipeline, each Spark now keeps chat state only for the layers
  it runs, which leaves room for more chats at once: 15 instead of 9 in our
  two-Spark test with a 109 GB memory ceiling (the number depends on the
  memory each Spark has free). In that test, sixteen chats sent at once
  finished 22 % faster with drafting on and 6 % faster with drafting off, with
  the same answers. `VADE_W2_PP_STAGE_STATE=0` on every Spark restores the
  previous behavior.

### CLI

- `vade install` from a controller now builds every Spark at the same time
  instead of one after another. It still prints each machine's output in
  order, and if a machine fails it stops with that machine's error, as before.
- `vade launch` no longer opens a chat when its input isn't a terminal, for
  example under `script` or with input from `/dev/null`. It prints the ready
  line and the chat command, exits successfully and leaves the engine running,
  so you can connect later with `vade chat`.
- When you run Vade by a path, such as `./vade` from the launcher bundle
  folder, the next-step hints from `vade launch` and `vade status` repeat that
  path (for example `./vade chat`) instead of a bare `vade` that your shell
  may not find.

### Documentation

- The README and the [benchmarks page](docs/benchmarks.md) show measured
  v0.1.0 results, and the receipts behind them are published under
  [bench/receipts/2026-10-24](bench/receipts/2026-10-24/README.md).
- The results figure compares one Spark with two, with drafting on. In the
  2026-10-24 runs, a second Spark answered a long prompt sooner but didn't make
  a single chat write faster. Eight chats at once vary from run to run, and the
  figure shows the range we measured.
- [INSTALL](INSTALL.md) covers more situations: an engine that turns away an
  extra chat with HTTP 503, checking each replica with `vade verify --url`,
  keeping the launcher bundle outside your checkout, and what to do when
  `vade install` refuses a checkout with local changes.
- The guides were rewritten in plainer language. Commands, numbers and links
  didn't change.

### Project

- Bug reports and feature requests now use issue forms. The bug form asks for
  the version, model, number of Sparks, how you run Vade, the exact command
  and the full error, plus `vade verify` output when answers are wrong.
- The security policy explains how to report a vulnerability privately, what
  it covers, and what Vade exposes on the network by default.

## v0.1.0 - 2026-10-24

The first public release. Vade runs a large language model on your own NVIDIA
DGX Spark and gives you three things: a coding agent in your terminal, a plain
chat, and an OpenAI-compatible API. It serves one model,
`nvidia/Qwen3.8-Flash-Next-FP4-Local` (text only), on one Spark or on linked
Sparks. It is written in Rust using only the standard library, and it is built
so that the same request on the same setup returns the same token IDs on every
run; `vade verify` lets you check that on your own machine.

### Engine and serving

- An inference engine for the NVIDIA DGX Spark, written in Rust using only the
  standard library. The release path is the text-only Qwen Flash-Next
  FP4-Local model, on one Spark by default. Configured multi-Spark
  deployments run through the exact layer pipeline, which splits the model's
  layers across the Sparks.
- Drafting is on by default: an extra model head proposes tokens and the model
  verifies them. A request that sets no sampling field is decoded greedily, on
  any number of Sparks. A request that sets any sampling field starts from the
  checkpoint's own sampling defaults, and its values replace those defaults.
- Fixed drafting on two Sparks. Both pipeline stages now follow the same
  drafting policy for a cohort (a group of requests running at the same time).
  The two-stage pipeline now allows guarded cohort drafting, which switches a
  cohort back to plain decoding when drafting stops paying off, and the
  worker's decode pools are reclaimed.
- At startup, the pipeline refuses to run if its ranks (one process per
  Spark) have different artifacts, numerical settings or prompt options. It
  also checks the pipeline switches before it chooses a topology.
- The server reports the drafting mode and depth it actually ran, so
  `/health` and receipts tell the truth about drafting instead of echoing the
  request.
- Release builds load verified precompiled kernels. Checkpoint hashing uses a
  safe streaming SHA-256 that does not fill the file cache.
- The startup memory check now counts reclaimable page cache as free (Linux
  `MemAvailable`, capped by CUDA total) instead of using CUDA free memory,
  which leaves out the host page cache inside containers. A memory refusal
  names MemAvailable, the required bytes and the ceiling. Session pools hold
  back a 512 MiB transient reserve, so 8 and 16 concurrent chats at the
  default context no longer stop generating a few megabytes short.
- The Flash-Next prefix cache is off by default. It stays off until tests show
  that serving with the cache is as fast as serving without it in every
  measured case. `VADE_FLASHNEXT_PREFIX_CACHE=1` turns it on, on one Spark
  only.

### Model and installation

- A pinned, verified FP4-Local publisher revision with a committed SHA256
  manifest of the complete snapshot, plus a one-line fetch helper that checks
  the download against that revision.
- A replay-tested local install on one Spark. The launcher reads the Spark's
  identity and resources, finds the models you already have, generates local
  defaults, and runs install, up and chat in one terminal.
- A prebuilt launcher bundle for Linux aarch64. A release workflow builds it on
  version tags, checks every manifest hash, and attaches the bundle and its
  `SHA256SUMS` to the tag's release. If a bundle is rejected, the launcher
  falls back to building from source.
- Entry guides rewritten for first-time readers. README, INSTALL and the
  first-run guide start with the one-Spark launcher, keep the source route and
  the bundle route apart, and name the supported models explicitly.
- A two-Spark route in INSTALL, next to the one-Spark default.
- A supported-systems table in INSTALL. The engine runs on DGX Spark / GB10
  under Linux `aarch64`. For Route B, where you drive the Spark from a
  second computer (the controller), the controller can be Linux `aarch64` or
  `x86_64`, or macOS. For Route A, where the CLI, engine and chat all run on
  the Spark, a Windows computer is supported as the SSH client, but not as a
  controller.
- Source installs also work from a downloaded source archive with no Git
  metadata. The launcher records the archive's stamped commit as the build id,
  or builds as `unknown` when the archive carries none.

### CLI, chat and gateway

- Controller commands for install, launch, up, status, logs, verify and
  shutdown, with bounded JSON status, specific errors that say what to do next,
  and help that stays usable in pipes.
- `vade up` on the controller and `vade launch` on the Spark start the same
  FP4-Local defaults (context 32768, output 8192). If the engine refuses the
  default or saved context for memory, `vade launch` steps it down through
  16384, 8192 and 4096 and saves the result once a start succeeds; that is the
  only case where it steps down. An explicit `--context` that the engine
  refuses is an error.
- `--context 262144` keeps the single-chat long-context preset, and
  `vade status` prints the chats the engine admitted (how many it allows to
  run at once at this context). A controller bench measures concurrent chats
  instead of one queued chat.
- `vade up --replicas N` (2..16, FP4-Local) starts an independent one-Spark
  server on each of N Sparks behind one local `vade-gateway`, which sends each
  request to the least-busy replica. `vade-harness vade-gateway` runs the same
  gateway by hand, with `--upstream` repeated.
- A machine's `ssh` entry can be a `user@host` destination as well as an
  alias, host name or IPv4 address.
- `vade verify` checks that repeated greedy runs against a running server
  return the same tokens, and finds the endpoint of the server it owns
  automatically.
- Chat history lives under the Vade home (`~/.vade/projects`), never in the
  folder you chat from. Set `VADE_CHAT_IN_PROJECT=1` to keep it in a
  git-ignored `./.vade` instead.
- Chat and coding-agent workflows: steer a chat while it runs, keep replies
  that were interrupted, manage sessions, and use an OpenAI-compatible HTTP
  gateway with credential handling and recorded replay.
- Both gateways refuse browser-driven requests by default. They refuse a
  request whose Host header is anything other than loopback with the listener
  port, and any request that carries an `Origin` header. You can allow a
  browser client on purpose with exact allow-lists and an optional bearer
  token.

### Benchmarks

- `vade bench` runs with no flags. It finds the recorded endpoint, reads the
  model from `/health`, uses a bundled offline prompt and measures the
  server's own drafting default.
- Drafting-aware receipts (the saved result records) that keep the requested
  drafting mode apart from the mode and depth the server reports, and withhold
  rates when the two disagree.
- A benchmark page with methods, results and limitations, linking every quoted
  source folder to its published, sanitized receipt.
- A results chart in the README and on the benchmark page, drawn by a
  standard-library script from the committed receipts, so every label is a
  value that a receipt prints.

### Project and community

- Agent rules in AGENTS.md and the human contribution process in
  CONTRIBUTING.md, with a package map, architecture notes and decisions.
- A security policy, Code of Conduct, issue and pull request templates, NOTICE
  and the bundled third-party licence texts.
- A flaky stream-cancellation test is fixed: it now synchronizes
  cancellation and verifies that processes are reaped. The shared
  process-test lock now recovers after it is poisoned.

For measurements, recipes and their limitations, see
[benchmark methods and results](docs/benchmarks.md). See [NOTICE](NOTICE) for
bundled third-party material and [LICENSE](LICENSE) for Vade's Apache-2.0
licence.
