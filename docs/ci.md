# Local CI

Tests run locally before every merge. Use the public commands below. Vade does not pay for GitHub Actions minutes.

## What the GitHub workflow runs

The [CI workflow](../.github/workflows/ci.yml) runs on GitHub-hosted standard runners, with no secret. It starts on a push to `main`, on a pull request, or by hand (`workflow_dispatch`).

Every job runs only when the repository is public. While the repository is private, the workflow skips every job, so it uses no paid minutes.

- **Links and tools, on a push or a pull request.** The workflow checks the documentation links. It runs the release ref checks and the repository tools tests. These three steps run on Linux `aarch64` (`ubuntu-24.04-arm`), the Spark's platform.
- **The CPU crates, on a push or a pull request.** The workflow tests the CPU crates on Linux `aarch64` and on Linux `x86_64` (`ubuntu-latest`). Each crate runs the steps that its row in the workflow matrix sets:
  - `cli/` runs the check, the tests with the repository ratchet, and Rustdoc.
  - `harness/` runs the check, the tests, the doctests and Rustdoc.
  - `engine/tools/bench/` runs the check, the tests and the doctests.

  On `x86_64` and on macOS, `cli/` and `harness/` also get a `cargo check` for Linux `aarch64`.
- **The engine and macOS, by hand.** A run started by hand also tests the CPU crates on macOS. It also runs the engine job. The engine job builds the GPU kernels without a GPU. Then it checks and tests the engine host crates on Linux `aarch64` and on macOS.

CI does not run `cargo fmt` or clippy. The Rust is formatted densely on purpose ([AGENTS.md](../AGENTS.md)), and both tools fail on the CPU crates today.

The [release workflow](../.github/workflows/release.yml) runs only by hand. When you start a release run, select an existing tag with the `v` prefix. The release job rejects other refs before checkout or toolchain setup.

This page explains the local checks. It also gives the commands a contributor can run without the private CI script. There is no `scripts/ci-local.sh` in this repository.

## What local checks cover and how long they take

The public commands check and test the CLI, the harness, the engine, the benchmark tools and the repository tools. They also check the documentation links, run the doctests, build the CLI and harness Rustdoc, and check the Linux targets. The engine host checks need a kernel build first. These checks do not need a Spark.

The repository does not publish a measured time for a full local CI run. The run time depends on the host and on the cached build files. The first setup also needs downloads. Record the elapsed time with the command logs for your run.

## What the maintainers' local CI runs

The maintainers run a private script against each committed revision. The script is not in this repository, and contributors do not need it. Contributors run the public commands in the next section instead.

The script checks out the commit in a separate, detached Git worktree. It uses the host toolchain pinned in [rust-toolchain.toml](../rust-toolchain.toml). It denies compiler warnings and saves a log for each step. It fails if any step fails.

These are the commands in the private script:

- **`engine/`.** The script runs `cargo check --offline --all-targets --workspace --exclude spark-rust-kernels` and `cargo test --offline --all-targets --workspace --exclude spark-rust-kernels`.
- **`harness/`.** The script runs `cargo check --offline --all-targets` and `cargo test --offline --all-targets`.
- **`cli/`.** The script runs `cargo check --offline --all-targets` and `cargo test --offline --all-targets -- --nocapture`.
- **`harness/` and `cli/`.** The script runs `cargo check --offline --all-targets --target aarch64-unknown-linux-gnu`.

The CLI tests include the repository ratchet. The engine commands exclude the NVPTX kernel crate, which you must build separately. Device tests can stay ignored. An ignored device test is not a device pass.

The script does not build kernels, run doctests, build Rustdoc or check documentation links. It does not test the separate benchmark and tools workspaces, and it does not run `cargo build` as a separate step. The GitHub workflow has all of these checks except a separate host `cargo build` step. The script is therefore not the full GitHub workflow. Neither set of host checks proves GPU exactness or speed.

## Run checks from a public checkout

Use macOS or Linux with Git, a C linker and Rust. Put rustup's bin directory on `PATH`. Install the pinned toolchains with the commands in [CONTRIBUTING.md](../.github/CONTRIBUTING.md#install-the-toolchains). The first setup needs network access. Later checks can run offline when those inputs are cached. Do not install or start an engine for these checks.

1. Commit the change first.
2. Run `git rev-parse HEAD` and keep its output with the logs.
3. Keep the checkout unchanged during the checks.
4. From the repository root, build and test the CLI and the harness:

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

These workspaces need no GPU and no kernel build. The public commands add `--locked`, so the committed lockfiles do not change.

A repeated run can find an old temporary fixture. If it does, use a new system temporary directory. Keep its path short and outside hidden directories, because sandbox tests can refuse fixtures under a hidden directory. Keep the failed log too.

From the repository root, also run the repository checks and the separate benchmark tests:

```sh
cargo +1.99.0 run --offline --locked --manifest-path tools/Cargo.toml -p vade-tools -- check-doc-links
cargo +1.99.0 test --offline --locked --all-targets --manifest-path tools/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path engine/tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --all-targets --manifest-path engine/tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --doc --manifest-path engine/tools/bench/Cargo.toml
```

The release ref check runs offline. It tests the release workflow's guard with branch refs and tag refs. From the repository root, run:

```sh
mkdir -p .local
rustc +1.99.0 -D warnings --test .github/workflows/release-ref-tests.rs -o .local/release-ref-tests
.local/release-ref-tests
```

For the engine host checks, first build and stamp the kernels on every host OS. On Linux without a GPU, also prepare the link-only CUDA stub. Use the commands in [AGENTS.md](../AGENTS.md#engine-and-full-ci-checks) and [engine validation](../engine/docs/validation.md#on-any-computer-host-checks). Always pass `--workspace --exclude spark-rust-kernels` to the engine host checks and tests. The stub stops the test run if a test calls the device. Do not count ignored device tests as passed.

After that preparation, also run the engine doctests from `engine/`:

```sh
cargo +1.99.0 test --offline --locked --doc --workspace --exclude spark-rust-kernels
```

Report the commit ID, the commands, the pass and fail totals, the ignored tests and the skips. Attach the logs to the change review. A local pass is evidence for the local checks only. A manual GitHub run must finish before anyone can claim its result.
