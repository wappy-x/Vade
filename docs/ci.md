# Local CI

CI currently runs locally. GitHub Actions can refuse to start jobs because of
account billing or spending limits. A refused run has no build or test result.
The README therefore uses a plain local CI line. It does not claim that a
GitHub build passed. Account settings are outside this repository.

The [workflow](../.github/workflows/ci.yml) defines the GitHub host checks.
It also builds GPU kernels without a GPU. This page explains the local checks
and the commands a contributor can run without the private CI script.

## What `ci-local.sh` runs

Maintainers have a private script with this interface:

```sh
ci-local.sh "$WORKTREE" "$COMMIT"
```

`WORKTREE` is the source checkout. `COMMIT` is its full commit ID. Commit the
change first. The script checks out that commit in a separate, detached Git
worktree. It uses the host toolchain pinned in
[rust-toolchain.toml](../rust-toolchain.toml). It denies compiler warnings and
saves a log for each step. It fails if any step fails.

| Workspace | Commands in the private script |
| --- | --- |
| `engine/` | `cargo check --offline --all-targets --workspace --exclude spark-rust-kernels`; `cargo test --offline --all-targets --workspace --exclude spark-rust-kernels` |
| `harness/` | `cargo check --offline --all-targets`; `cargo test --offline --all-targets` |
| `cli/` | `cargo check --offline --all-targets`; `cargo test --offline --all-targets -- --nocapture` |
| `harness/` and `cli/` | `cargo check --offline --all-targets --target aarch64-unknown-linux-gnu` |

The CLI tests include the repository ratchet. The engine commands exclude the
NVPTX kernel crate, which must be built separately. Device tests can remain
ignored. That is not a device pass.

For a change that leaves `engine/` unchanged, maintainers can set
`CI_SKIP_ENGINE=1`. This skips the engine check and engine tests. Name that
skip in the result. Never use it to validate an engine change.

The script does not build kernels, run doctests, build Rustdoc, check
documentation links, test the separate benchmark and tools workspaces, or run
`cargo build` as a separate step. The GitHub workflow includes these additional
checks, except for a separate host `cargo build` step. The script is not the
full GitHub workflow. Neither set of host checks proves GPU exactness or speed.

## Run checks from a public checkout

Use macOS or Linux with Git, a C linker and Rust. Put rustup's bin directory
on `PATH`. Install the pinned toolchains with the commands in
[CONTRIBUTING.md](../CONTRIBUTING.md#install-the-toolchains). The first setup
needs network access. Later checks can run offline when those inputs are
cached. Do not install or start an engine for these checks.

Commit the change first. Run `git rev-parse HEAD` and keep its output with the
logs. Keep the checkout unchanged during the checks. From the repository root,
build and test the CLI and harness:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export RUSTFLAGS='-D warnings'
export RUSTDOCFLAGS='-D warnings'
export RUST_TEST_THREADS=2
export RUSTUP_AUTO_INSTALL=0
cargo +1.99.0 build --offline --locked --manifest-path cli/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path cli/Cargo.toml
cargo +1.99.0 test --offline --locked --all-targets --manifest-path cli/Cargo.toml -- --nocapture
cargo +1.99.0 build --offline --locked --manifest-path harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path harness/Cargo.toml
cargo +1.99.0 test --offline --locked --all-targets --manifest-path harness/Cargo.toml
cargo +1.99.0 test --offline --locked --doc --manifest-path harness/Cargo.toml
cargo +1.99.0 doc --offline --locked --no-deps --manifest-path cli/Cargo.toml
cargo +1.99.0 doc --offline --locked --no-deps --manifest-path harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path cli/Cargo.toml --target aarch64-unknown-linux-gnu
cargo +1.99.0 check --offline --locked --all-targets --manifest-path harness/Cargo.toml --target aarch64-unknown-linux-gnu
```

These workspaces need no GPU or kernel build. The public commands add
`--locked` to keep the committed lockfiles unchanged.

If a repeated run finds an old temporary fixture, use a fresh system temporary
directory. Keep its path short and outside hidden directories. Sandbox tests
can refuse fixtures under a hidden directory. Keep the failed log too.

Also run the repository checks and the separate benchmark tests from the root:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-doc-links
cargo +1.99.0 test --offline --locked --all-targets --manifest-path tools/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path engine/tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --all-targets --manifest-path engine/tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --doc --manifest-path engine/tools/bench/Cargo.toml
```

For engine host checks, first build and stamp the kernels on every host OS.
On Linux without a GPU, prepare the link-only CUDA stub too. Use the commands
in [AGENTS.md](../AGENTS.md#engine-and-full-ci-checks) and
[engine validation](../engine/docs/validation.md#on-any-computer-host-checks).
Always pass `--workspace --exclude spark-rust-kernels` to engine host checks
and tests. The stub aborts if a test calls the device. Do not count ignored
device tests as passed.

After that preparation, also run engine doctests from `engine/`:

```sh
cargo +1.99.0 test --offline --locked --doc --workspace --exclude spark-rust-kernels
```

Report the commit ID, commands, pass and fail totals, ignored tests and skips.
Attach the logs to the change review. A local pass does not make the GitHub
badge green. A new GitHub run must finish before anyone can claim that result.
