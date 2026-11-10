# Contributing to Vade

Vade is an inference engine for the NVIDIA DGX Spark / GB10. It is written in
Rust and uses only the standard library.

Version 0.1.0 runs one model, `nvidia/Qwen3.8-Flash-Next-FP4-Local`, as text
only. Main also serves `Qwen/Qwen3.8-Flash-Next-FP8`. Vade runs on one Spark by
default, and it can spread across more Sparks when you link them. Drafting is
on by default: an extra model head proposes the next few tokens, and the model
verifies them in one pass.

This page covers four things:

- **Checks.** It shows how to check a change on your own computer.
- **Rules.** It lists the house rules.
- **Exactness.** It says what exactness asks of engine changes.
- **Pull requests.** It shows how to send a pull request.

These pages go with this one:

- **Project rules.** [AGENTS.md](../AGENTS.md) gives the rules for a change to
  Vade's code. This page says the same rules for people.
- **Code of conduct.** Everyone who takes part follows the
  [code of conduct](CODE_OF_CONDUCT.md).
- **Security problems.** Do not open a public issue. Use
  [GitHub private reporting](https://github.com/wappy-x/Vade/security/advisories/new),
  as the [security policy](SECURITY.md) says.
- **Bugs and ideas.** Use the
  [issue forms](https://github.com/wappy-x/Vade/issues/new/choose).
- **CI.** [Local CI](../docs/ci.md) gives the current CI status and the public
  host commands. Run the host checks that your change affects, with doctests
  and Rustdoc. The [workflow](workflows/ci.yml) is the full list.
- **Terms.** The [glossary](../docs/GLOSSARY.md) explains the terms that this page
  uses.

## Build and validate

You can run every host check on this page on macOS or Linux. You only need a
Spark for the device checks described under
[exactness](#exactness-and-evidence).

### Install the toolchains

[CI](workflows/ci.yml) pins two Rust toolchains: 1.99.0 for the host
code and nightly-2026-10-01 for the GPU kernels. Install them once, while
you're online:

```sh
rustup toolchain install 1.99.0 --profile minimal
rustup target add --toolchain 1.99.0 aarch64-unknown-linux-gnu
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

CI denies warnings. To match CI, run `export RUSTFLAGS='-D warnings'` in your
shell. You do not need GPU flags, because the kernel builder adds its own
device flags.

### Check your change

There is no root Cargo workspace, so run these commands from `engine/`.

Build the kernels first, even on a Mac. The host checks fail if the compiled
kernels (PTX files) from an earlier build are stale. The first build also
compiles Rust's core library for the GPU target (the build-std cache). It can
need downloads, so run it once while you are online. After that, everything here
works offline.
[Validation](../engine/docs/validation.md#on-any-computer-host-checks) has the
details.

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- check-doc-links
```

That block checks the CLI, the harness and the benchmark tools. It runs the
ratchet test (explained under [rules](#rules-and-formatting)) and checks the
documentation links. `--locked` makes Cargo use the committed `Cargo.lock`
files as they are, the way CI does.

If you changed the engine, check and test its workspace. On Linux without a
GPU, first prepare the link-only CUDA stub with the commands in
[AGENTS.md](../AGENTS.md#engine-and-full-ci-checks). Then run the block below.
Pass `--workspace` as CI does. A plain `cargo test` covers only the default
members and skips some crates. `--all-targets` does not run doc tests, so CI
runs them in a separate step. That step is the last line here:

```sh
cargo +1.99.0 check --offline --locked --all-targets --workspace --exclude spark-rust-kernels
cargo +1.99.0 test --offline --locked --all-targets --workspace --exclude spark-rust-kernels -- --nocapture
cargo +1.99.0 test --offline --locked --doc --workspace --exclude spark-rust-kernels
```

Run the tests for the workspace that you changed before you open a PR. The
rest of CI's host recipes are in [AGENTS.md](../AGENTS.md) and
[engine validation](../engine/docs/validation.md). They cover the CUDA stub for
Linux machines without a GPU, the separate build of the TP (tensor
parallelism) kernels and the Linux cross-target checks.

These are host checks. Device-only tests are skipped when there is no GPU, so
a host PASS does not prove GPU correctness, exactness, memory use or
performance.

### Check the committed revision

Commit the change before its final checks. Run the public host commands
against the committed revision. Save the commit ID, commands and results with
the change review.

Local checks can cover less than the [workflow](workflows/ci.yml).
They do not replace the Spark gate below. [Local CI](../docs/ci.md) lists the
host commands and what they leave out. If the source changes after a check,
run the affected checks on the new commit.

### Build the CLI

To build the `vade` command yourself, also from `engine/`:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path ../cli/Cargo.toml
```

The build writes the binary to `cli/target/release/vade`. Building it does not
install or start an engine. To install from source on a Spark, follow
[INSTALL.md](../docs/INSTALL.md).

## Rules and formatting

- **Language.** Write code in Rust with the standard library only. Use shell
  only where Rust cannot do the job. Do not use or require Python.
- **Writing.** Use ASD-STE100 Simplified Technical English and American
  spelling. Use short sentences, active voice and plain words. Give each
  sentence one main idea. Do not use em dashes or lab slang. Apply this rule
  to docs, help, errors, reports and messages. Every numerical claim must name
  its card, log or command output. Give raw values with any percentage.
- **Zero third-party dependencies.** Use the Rust standard library and the
  first-party crates already in the tree. Registry, git, off-tree path, build
  and dev dependencies all count, and so do lockfile sources.
- **No new `unsafe`, with one exception.** Don't add unsafe blocks, functions,
  implementations or extern declarations. The existing CUDA, FFI, kernel and
  transport boundaries are not permission to grow them. Keep their safety
  contracts and SAFETY comments intact. The exception: a kernel may add one
  new `unsafe` block in `engine/crates/gpu` or `engine/crates/gpu/kernels`.
  Write its safety contract in a comment directly above the block, and a
  reviewer checks it before the merge.
  The [tools test](../tools/tests/unsafe_sites.rs) checks unsafe source sites during `cargo test` with a [legacy site allowlist](../tools/unsafe-allowlist.txt).
  The allowlist is frozen at merge. Refresh it only on an orchestrator ruling.
  CI never refreshes it. The [tools guide](../tools/README.md#unsafe-source-check)
  gives the command, ruling field and `ptx-kernel` entry exception.
- **Model-generic behavior.** Derive it from the checkpoint configuration and
  tensor metadata, with no dispatch on the model name and no model-specific
  tuning. See [adding a model](../docs/adding-a-model.md).
- **No arithmetic-order switches.** Use one execution path for each topology.
  Do not add a flag or environment variable that selects a different order
  for the math. A speed claim does not make such a switch acceptable.
- **Formatting by hand.** The code is formatted densely on purpose, so match
  the surrounding style and don't run `cargo fmt`, `clippy --fix` or any
  formatting sweep. CI gates on neither the formatter nor clippy.
- **Small files.** Keep new Rust files to 1,500 lines at most.
- **No fixed Spark counts.** Read the number of Sparks from the topology.
  Never branch on a literal world size or rank count.
- **The ratchet only goes down.** The ratchet test counts third-party
  dependencies, Rust files over 1,500 lines and those fixed-count branches.
  It fails when a count rises above its ceiling. Don't raise the
  [ratchet ceilings](../cli/ratchet-baseline.txt). When a count drops, lower its
  ceiling in the same commit.
- **Nothing private.** Never commit secrets, credentials, machine addresses,
  local config, model weights, generated PTX or build output. Keep local
  results outside the tracked tree. Keep license notices intact. If you add non-code third-party
  material (allowed only when its license permits), list it in
  [NOTICE](../NOTICE), or in [engine/NOTICE](../engine/NOTICE) for files under
  `engine/`.

## Exactness and evidence

Exactness means **identical token IDs every run on every Spark with the same
setup**. A request must give the same result whether it runs alone, batched,
cached or chunked, and drafting on must match drafting off. Check the same
request across the supported Spark counts too. Similar answer text is not
enough. With FP4-Local, a request that doesn't specify sampling is decoded
greedily (always the most likely next token). With FP8, a request is greedy
only when it sets `temperature` 0 and leaves the penalties neutral.

To keep that true, don't use floating-point atomics, reductions that depend
on arrival order, or reduction splits that change with batch size, prompt
length or occupancy.

### Which changes need a Spark

Host-only CLI and documentation changes don't need a Spark. Changes to engine
arithmetic, scheduling, caching or transport must pass a gate on real hardware
before they merge. Run the change at least three times. Compare the results
across at least two Sparks with the same setup. If you have one
Spark, run the repeats on it and say so in the PR. The change can't merge
until the comparison on a second Spark is done. There is one exception. A
gated change rebased onto a moved main can merge on its gate with an identical
patch, disjoint files and a passing pre-merge screen on the Sparks (see
[Which proof a change needs](#which-proof-a-change-needs)).

- If your change leaves the arithmetic order alone, it must keep main's
  results: the same token IDs.
- Use one execution path for each topology. The tensor split is the only
  two-Spark path. Change a default only after the full exactness gate and a
  measurement card. Do not keep the old path behind a switch.
- If you change the arithmetic order, also provide a quality comparison
  against main and reviewed updates to any goldens (saved reference answers)
  that change. An ID change needs the project owner's approval before merge.

Run the Spark proof on the exact commit that will merge, except in cases 1, 2
and 3 below. Keep the token ID arrays, the requests and the effective settings
in its receipts. Show the result of each comparison, including failed or
missing checks. The card must cover the change's affected cases as well as the
required exactness cases.

### Which proof a change needs

1. **A docs-only change needs no device proof.** A change is docs-only when
   every file it changes is a Markdown or plain-text page that no build, test,
   CI job or script reads. Files such as `rust-toolchain.toml`, CI files,
   build scripts and shell scripts are executable changes, also when they are
   text.
2. **A rebased engine change can keep the gate that it already passed.** This
   applies when main moved after the gate. All three of these conditions must
   be true:
   - The patch is identical. `git range-diff <gate base>..<proved commit>
     <new main>..<rebased commit>` shows each commit as `=`.
   - The files are disjoint. No file that the change touches is in
     `git diff --name-only <gate base> <new main>`.
   - The rebased commit passes a screen on the Sparks before it merges: the
     first-token cells at 8K and 32K and one round of the card cells. Each
     cell must stay inside the band, and the token IDs must be identical to
     the gated run.

   If the screen fails, run the full gate on the rebased commit. After the
   merge, a full confirm card runs on the merged main within one hour. If the
   confirm card fails, or does not run within the hour, the merge is reverted.
3. **A rebased change can keep the proof of the commit that passed the screen.**
   This applies when its engine patch did not change. Both of these conditions
   must be true:
   - The engine patch is the same. `git diff <base> <commit> -- engine |
     git patch-id --stable` gives the same ID for the screened commit on its
     base and for the rebased commit on the new main. Files outside `engine/`,
     such as `CHANGELOG.md`, can differ.
   - The merge request names the screened commit and its screen receipts:
     each screen of that commit and its result.

   If the patch IDs differ, screen the rebased commit again.
4. **Any other change to engine arithmetic, scheduling, caching or transport
   needs the full device gate.** Rebase it and run that proof again.
   Host-only changes use the host checks described above.

The gate runs main, the change, the change again and main again. The band is
0.8 % per cell against the two main runs. A cell passes when it is at most
0.8 % slower than the main value of that cell. The gate card records the main
value of each cell, so the limits are known before the screen and the confirm
card run. Token IDs must be identical in every run.

The gate base is the main commit that the proof compared against. Each card
and host log names the revision that it tested. Do not relabel old evidence
as a new run. For a merge under case 2, the gate card names the gated
(proved) commit, and the screen and the confirm card name the revisions that
they tested. The merge record lists the proved commit, the gate base, the new
main, the rebased commit, the range-diff output, the two file lists, the gate
card, the screen and the confirm card.

### Tools for the checks

Two tools cover the checks today:

1. Run `vade verify` against a running engine. It compares repeated greedy
   token IDs with drafting off and on, and saves a receipt (a record of what
   it checked).
2. Run `vade test determinism` for the batch, cache and chunk cases. The
   [gate guide](../cli/docs/test.md) shows how. It runs saved test inputs
   (fixtures) through the older `flash-bench` benchmark, so a pass there
   doesn't yet prove that FP4-Local is qualified.

Neither one covers everything yet. A pass from either tool alone is not the
full Spark gate. List in the PR which checks you ran and which you didn't.

### The measurement card

An engine change needs a card that joins its exactness and measurement
results. Name the source commit, checkpoint revision, hardware type and
Spark count. Record the commands, request bodies, effective sampling, cache
state and drafting mode. Include the token ID comparisons and the raw timing
results. Keep failed runs and explain any omitted cases.

For a speed claim, compare the candidate with main on the same setup and
requests. Keep the timing plan fixed before you run it. Give the repeated
results and their full ranges, not just the best run. A host test or a
similar-looking answer cannot support that claim. See the
[benchmark method](../docs/benchmarks.md) for the public results and their limits.

## Commits and pull requests

Make small commits with clear, imperative messages that say what changed and
why. Keep one change per PR. Save unrelated cleanup for another PR. Open the
PR against `main` at
[github.com/wappy-x/Vade](https://github.com/wappy-x/Vade).

In the PR, explain the problem, the resulting behavior, the commands that you
ran and any limits of your validation. For an engine change, attach the Spark
card and its exactness receipts. Include the output of `vade verify` and any
`vade test determinism` checks, with their coverage limits.

If users would notice the change, add one plain line about it to the
Unreleased section of [CHANGELOG.md](../CHANGELOG.md), under the heading for its
area. Put the newest line first.

### How a change merges

Keep your PR small and attach its check results.
The maintainer records acceptance after the required checks and review pass.
A merge train applies accepted changes to current `main` and checks the result.
Keep the commit and proof references current in your PR.

A pile combines several small changes on a branch with one proof set.
The maintainer records the pile commit, its members and its proof.
The maintainer accepts the pile as a unit after its required proofs pass.
Its members need no separate acceptance.

If a pile fails, the maintainer bisects it:

1. Split the pile into halves.
2. Check each half.
3. Split each failed half again.
4. Reject a change that fails alone.
5. Continue with the changes that pass.

The process uses these rules, called bends:

- **Bend 1.** Maintainers screen a pile against `main` with one proof set, then bisect the pile if it fails.
- **Bend 2.** A maintainer can post `READY FOR MERGE` after recorded acceptance with receipts, once the author's automated task stops.
- **Bend 3.** Engine changes outside communication paths need timed 1-Spark proof and 2-Spark and 4-Spark token ID checks.
- **Bend 4.** Maintainers assign routine tasks to a lower-cost API model and keep code, reviews and proof runs with the stronger model.

The ready line names the accepted commit and acceptance time.
Routine tasks include ready lines, CI watches, documentation copies, card copies and pile builds.
The exactness rules and performance limits still apply.

- **TRAIN-UNBLOCK.** If `main` moves while a candidate waits, the maintainer immediately screens its merge commit again on the first free Spark.
- **Engine train order.** Engine trains merge in acceptance order.
  A newer engine train waits while an older train needs its TRAIN-UNBLOCK screen.
  Host-side trains can merge during that wait.

### How a pull request lands

1. **Host checks pass.** Run the checks in [Local CI](../docs/ci.md) before
   every merge. Save the logs for the committed revision. On the public
   repository, the [GitHub workflow](workflows/ci.yml) also runs the
   host checks of the CPU crates on each pull request. A failed build or
   test blocks the change.
2. **A maintainer reviews it.** [CODEOWNERS](CODEOWNERS) requests
   the review automatically. Expect questions about exactness, the house
   rules and anything the PR says it didn't check.
3. **Engine changes pass the device gate.** A change to engine arithmetic,
   scheduling, caching or transport also needs the runs on real Sparks described
   under [exactness](#exactness-and-evidence) before it can land. The commit,
   host logs and Spark card name the same source revision. There is one
   exception. A gated change rebased onto a moved main can merge on its gate
   with an identical patch, disjoint files and a passing pre-merge screen on the
   Sparks. Its cards name the revisions that they tested (see
   [Which proof a change needs](#which-proof-a-change-needs)).
4. **The maintainer applies it.** The maintainers publish `main` here from
   their development tree, one commit per update. For this reason, nobody
   uses the merge button. The maintainer applies the accepted change in the
   development tree and runs local CI again on the result. The change reaches
   `main` in the next published update. Then the PR is closed, not merged.

If you haven't measured performance, don't make a speed claim. A performance
claim needs reproducible receipts that record:

- the source revision and checkpoint identity
- the hardware and topology
- the request and sampling settings
- the drafting and cache modes
- repeated measurements and the baseline comparison

Strip machine details and secrets from them, and don't commit weights or raw
tensor dumps.
