# Repository tools

The `vade-tools` crate uses only the Rust standard library. It has no Cargo
dependencies and has its own workspace. Run it from the checkout:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-doc-links
cargo +1.99.0 test --offline --locked --all-targets --manifest-path tools/Cargo.toml
```

## Unsafe source check

The [unsafe source test](tests/unsafe_sites.rs) runs in the existing CI test
command. It reads tracked Rust files and new Rust files that Git does not
ignore. It skips comments and literals. It does not expand macros.

The [allowlist](unsafe-allowlist.txt) records each site by crate,
source file, item name, site kind and site ordinal. The file and named scopes
qualify the item name. The ordinal counts sites in that item. Keys do not
include source lines, signatures or hashes. A new argument on a legacy unsafe
function does not add a function. An impl key names its trait and type. It
does not include generic bounds. An extra block in the same item needs its
own entry. Review changes to legacy bodies and safety contracts in the diff.
Equal missing and stale block counts can mean that a legacy block moved to
another item or file. The check gives one refresh message for those blocks.
It still fails. Keys do not prove that the block bodies match.

The allowlist is frozen at merge time. Refresh it only on an orchestrator
ruling. CI reads the allowlist. CI never runs the refresh command. Both the
tree-match test and the check command require the keys to equal the current
tree. A stale entry fails. A permitted kernel site still needs an entry.

The optional last field names a site ruling, such as
`ruling 2026-11-02 11:54`. Each legacy row records the freeze as
`frozen ruling 2026-11-02 11:54`. The check accepts only this fixed freeze
field. Do not add frozen rows for new sites.
A new ordinary unsafe function needs its own ruling field even if its key
is already listed. A source comment or a ruling on another row does not
approve it. The check validates the field format. A reviewer must confirm
that the named ruling grants approval for the site. Frozen rows skip the
new-site rules unless the item's block count grows past its frozen row count.
Then every block in that item needs a safety note. An inserted block can take
a frozen ordinal. A possible move still fails and needs a reviewed allowlist
diff. A new kernel still needs its safety notes and block limit when its rows
carry a ruling.

Except for an ordinary unsafe function with a ruling, a new unsafe site
fails outside `engine/crates/gpu` and `engine/crates/gpu/kernels`.
These crates allow a new block in a named kernel wrapper. They also allow an
unsafe `extern "ptx-kernel"` entry with a body, including its
`#[unsafe(no_mangle)]` attribute. Each function may add at most one new block.
Each new block needs a nonempty `SAFETY:` comment directly above its source
line. A new entry also needs a
nonempty `SAFETY:` comment directly above the entry and its attributes.
Adjacent comment lines may continue a contract. Other nested crates have no
exception. A reviewer must still check the wrapper and the safety contracts.

The refresh command requires the ruling as an argument. It writes the current
keys in sorted order and preserves existing ruling fields. It removes stale
rows. It leaves new rows without a ruling. It does not approve new sites.
Review the diff and add only the site rulings that the orchestrator granted.
For an approved legacy move, carry its frozen field to the new key after
review.

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-unsafe-sites
cargo +1.99.0 test --offline --locked --manifest-path tools/Cargo.toml -p vade-tools --test unsafe_sites
```

After an orchestrator ruling, use its date and time. For example, the freeze
ruling uses this command:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- refresh-unsafe-allowlist 'ruling 2026-11-02 11:54'
```

`check-unsafe-sites [ROOT]` checks the Git tree at `ROOT` with that tree's
`tools/unsafe-allowlist.txt`. With no `ROOT`, it checks this checkout.
CI checks the source and allowlist from the same checkout. The command does
not build the engine or kernels.

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
`tp`, `pp`, `sparks` and `distributed`. Names with a `_world`, `_world_size`,
`_rank`, `_ranks` or `_topology` suffix also count. Field access and method
calls count. Match patterns that name `Topology` count even when the matched
value has another name. Loops and shape calculations that use topology data
alone do not count.

The rank-name rule ignores `lora_rank`, `tensor_rank` and names that end in
`_lora_rank` or `_tensor_rank`. The rule ignores letter case. These names
describe tensor dimensions. For example, `q_lora_rank`, `kv_lora_rank` and
`MAX_TENSOR_RANK` do not count as process ranks. This exception applies to
each name, not the whole condition. A condition that also names `rank`,
`world` or another topology value still counts. Other `_rank` names still
count. This rule uses names only. It does not infer a value's type or follow
an alias.

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

## Internal dashboard preview

`rigmark-dashboard TEMPLATE_HTML BUILT_HTML SETS_JSON OUTPUT_DIR` adds columns
for one, two, four and eight Sparks to an existing RigMark dashboard. It adds
a Set column to the card table and set names to chart tips. The filters apply
to the new columns. Each entry shows the latest card for its setup, set,
thinking mode, drafting mode and status. Empty columns show no measured rate.

The input page must come from the current dashboard builder. The command keeps
its embedded card data and build label byte for byte. It does not read a device
or change a measurement. The existing charts keep their groups and values.

The set map is a private JSON object with a `sets` object. Each key in `sets`
is a set name, such as `4H`, `4B`, `2A` or `2B`. Its value is an array of saved
member labels. An optional `cards` object supplies member arrays for cards
whose rows lack this record. Its keys are `folder/card`, as saved in the page.
Use the archived run's member records for these arrays. Keep this file outside
Git. A card gets a set name only when all its saved members match.
Member order and letter case do not affect the match. A rented card or a card
without a matching member list shows `Unlisted`. Spark count alone does not
identify a set.

The command writes `template.html` and `RigMark Dashboard.html` to a new output
directory. It refuses an existing output directory. Check the preview before
installing it. Use the new template for later rebuilds with the existing
builder. Update the private set map and make a fresh preview when sets change.

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
