# Local CI

Tests run locally before every merge. Use the public commands below.
Vade does not pay for GitHub Actions minutes. GitHub Actions can refuse to
start jobs because of account billing or spending limits. A refused run has
no build or test result.

The [CI workflow](../.github/workflows/ci.yml) runs only when a maintainer
starts it by hand with `workflow_dispatch`. It defines the GitHub host checks
and builds GPU kernels without a GPU. The
[release workflow](../.github/workflows/release.yml) also runs only by hand.
Select an existing tag with the `v` prefix when you start a release run.
The release job rejects other refs before checkout or toolchain setup.
Pushes, pull requests, merge queues and schedules do not start these workflows.

This page explains the local checks and the commands a contributor can run
without the private CI script. There is no `scripts/ci-local.sh` in this
repository.

## What local checks cover and how long they take

The public commands check and test the CLI, harness, engine, benchmark tools
and repository tools. They also check documentation links, run doctests,
build CLI and harness Rustdoc, and check Linux targets. Engine host checks
need a kernel build first. These checks do not require a Spark.

The repository does not publish a measured time for a full local CI run.
Run time depends on the host and cached build files. The first setup also
needs downloads. Record the elapsed time with the command logs for your run.

## What the maintainers' local CI runs

Maintainers run a private script against each committed revision. It is not
in this repository, and contributors do not need it. Contributors run the
public commands in the next section instead.

The script checks out the commit in a separate, detached Git worktree. It
uses the host toolchain pinned in [rust-toolchain.toml](../rust-toolchain.toml).
It denies compiler warnings and saves a log for each step. It fails if any
step fails.

| Workspace | Commands in the private script |
| --- | --- |
| `engine/` | `cargo check --offline --all-targets --workspace --exclude spark-rust-kernels`; `cargo test --offline --all-targets --workspace --exclude spark-rust-kernels` |
| `harness/` | `cargo check --offline --all-targets`; `cargo test --offline --all-targets` |
| `cli/` | `cargo check --offline --all-targets`; `cargo test --offline --all-targets -- --nocapture` |
| `harness/` and `cli/` | `cargo check --offline --all-targets --target aarch64-unknown-linux-gnu` |

The CLI tests include the repository ratchet. The engine commands exclude the
NVPTX kernel crate, which must be built separately. Device tests can remain
ignored. That is not a device pass.

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

The release ref check runs offline. It tests the release workflow's guard
with branch refs and tag refs. From the repository root, run:

```sh
mkdir -p .local
rustc +1.99.0 -D warnings --test .github/workflows/release-ref-tests.rs -o .local/release-ref-tests
.local/release-ref-tests
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
Attach the logs to the change review. A local pass is evidence for the local
checks only. A manual GitHub run must finish before anyone can claim its result.
