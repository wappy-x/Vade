# Repository tools

The `vade-tools` crate uses only the Rust standard library. It has no Cargo
dependencies and has its own workspace. Run it from the checkout:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-doc-links
cargo +1.99.0 test --offline --locked --all-targets --manifest-path tools/Cargo.toml
```

## Documentation links

`check-doc-links` checks local Markdown and text links, anchors and embedded
asset paths. It reads tracked documents and new documents that are not ignored.
It makes no network requests. CI runs the same command.

## One execution path

`check-one-path` scans tracked Rust source and new, unignored Rust source under
`engine/`. It includes tests, build scripts, examples, kernels and inactive
`cfg` blocks. It does not read build output or private scratch files.
The `engine_source_uses_one_path_allowlist` test runs the same check in the
existing repository CI job. It needs no GPU or network.

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-one-path
cargo +1.99.0 test --offline --locked --manifest-path tools/Cargo.toml -p vade-tools one_path
```

The `env-read` rule flags references to `std::env::{var,var_os,vars,vars_os}`.
It also checks local `use` aliases, grouped imports and glob imports. A function
reference is a finding even if the call occurs later. Comments and literals do
not count.

The `topology-branch` rule flags `if`, `while`, `match`, match guards and
`matches!` that name a topology value. It also flags comparisons outside those
conditions. These comparisons can set a path flag for a later call. Names
include world and rank counts, `rank`, `topology`, `rank_path`, `tensor_split`,
`tp` and `distributed`. Names with a `_world`, `_world_size`, `_rank`, `_ranks`
or `_topology` suffix also count. Field access and method calls count. Match
patterns that name `Topology` count even when the matched value has another
name. Loops and shape calculations that use topology data alone do not count.

Each finding gives the source file, line, Cargo package and qualified item.
The item includes its source module, enclosing type and function name. Trait
implementations use `Type@Trait`. Direct `rustc` tools use their file stem as
the crate name. Source paths distinguish binary and test modules.

The [allowlist](one-path-allowlist.txt) records current sites by rule, crate and
item. Its keys do not use line numbers. Each row gives an exact site count.
New items and extra sites in existing items fail. Removed sites require a
smaller count or removal of the row. Do not expand this list to approve a new
execution path. Review each new finding against [AGENTS.md](../AGENTS.md).

This is a conservative token check, not a Rust type checker. A finding can be
a validation check or an unrelated use of a name such as `rank`. It needs human
review. The check does not expand macros, follow values through aliases or
resolve imports across files. Import shadowing can produce extra findings.
An unchanged site count can hide a replacement in the same item. This check
does not prove that arithmetic has one path. Code review and device evidence
still apply to engine changes.

## Bundle checks

`check-bundle OUTPUT_DIR [FORBIDDEN_PATH ...]` checks the archive checksum,
tar headers, member paths and types, manifest fields, file hashes and executable
modes. It reads archive data without extracting it. It uses the existing safe
SHA-256 implementation and the system `zstd` command for decompression.
The release workflow runs this command on real build output.

The [producer tests](tests/producer.rs) run the real
[`make-bundle.sh`](../scripts/make-bundle.sh) with
[Rust build stubs](src/bin/bundle-fixture.rs). They check publication races,
failure cleanup and compiler path removal. The
[checker tests](tests/bundle.rs) use independent tar fixtures and simulated
compression. These tests need no GPU or model build.

## Example checks

The [two-Spark tests](tests/two_sparks.rs) run the real
[`two-sparks.sh`](../examples/two-sparks.sh) with `vade` and `curl` stubs.
They check paths with `&`, `|`, `\` and spaces, call order, cleanup and exit 42.
They need no Spark, SSH or network.

## CI without a GPU

`ci-no-gpu` runs only on Linux. Build real stamped PTX first. The command
finds CUDA declarations in the engine source and writes an aborting linkage
library to `engine/target/ci-no-gpu`. Rust generates and compiles the same
stand-in symbols as the earlier tool. Each device call aborts. CI adds this
folder to `LIBRARY_PATH` and `LD_LIBRARY_PATH`. Never ship this library.

## Benchmark figure

`benchmarks-chart` draws the light and dark
[release figures](../docs/assets/README.md) from the committed
[receipts](../bench/receipts/2026-10-24/README.md). The Rust SVG writer is under
300 lines. It preserves each printed value and checks the receipt revision,
headline, text margins and label collisions. It reproduces the committed SVGs.

## Model download

`fetch-model REPOSITORY FULL_REVISION MODEL_DIRECTORY` uses `curl` for HTTPS.
Rust checks the pinned remote file list, sizes, LFS SHA-256 hashes and Git blob
hashes. It rejects extra files and links. It resumes partial downloads in the
private sibling `.fetch` directory. It writes a sorted sibling `.sha256`
manifest only after all files pass. The shell entry point is
[`scripts/fetch-model.sh`](../scripts/fetch-model.sh).
An operating system file lock prevents concurrent fetches. The lock file stays
in place. The operating system releases the lock after its owners exit.
The internal [curl guard](../scripts/fetch-curl.sh) stops and waits for curl when
the Rust process exits. Both processes hold the lock until curl has stopped.
Metadata requests have a 120-second deadline. File transfers can run for hours,
but fail if their rate stays below 1 byte per second for 120 seconds.

## API JSON support

`json-get KEY ...` reads a string or number at a JSON path from standard input.
`json-print KEY ...` writes the text without an added newline.
`json-value KEY ...` writes the JSON value. `json-quote` encodes standard input
as a JSON string. The shell API examples use these commands. The
[Rust API example](../examples/openai_client.rs) shares the same JSON parser.
`curl` supplies HTTPS for the shell examples because the standard library has
no TLS client.
`json-stream-check` rejects error envelopes and events without a choices array.
The shell stream example also requires a final `[DONE]` event.

## Timing the installed server

Use `vade bench` for client timing. See [INSTALL.md](../docs/INSTALL.md#run-the-benchmark)
and the [benchmark options](../docs/benchmarks.md#run-vade-bench).
The implementation is the [benchmark library](../engine/tools/bench/README.md).
Its self-test uses a local mock server. It does not check GPU inference.

Use the [Rust gate commands](../cli/docs/test.md) for build and exactness checks.
The frozen [8192-token probe](fixtures/probe-8192.json) has SHA-256
`11869d33b6fe2411447a7c9fd34ef76895ca53017f7505e2d63ac77783df945f`.
