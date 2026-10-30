# harness/src/shell

**Rust modules for `shell`, part of the `vade-harness` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`build.rs`](build.rs) | Fresh build state, separate from uninspected workspace artifacts. |
| [`permissions.rs`](permissions.rs) | Allow lists of exact commands of plain words: this session (memory only), the project and the user. |
| [`policy.rs`](policy.rs) | The shipped policy: every command asks unless a rule matches its argv (plain words, `<path>` slots); blocked commands never run. |
| [`safety.rs`](safety.rs) | The command checker: blocked, needs approval, or an experimental safe class that counts only with `VADE_SHELL_CHECKER=1`. Its `safety/` folder holds the lexer, the program table and the pattern matcher. |

**Why a folder:** Rust keeps one module per file; this folder groups the modules that belong to `shell`.

`CommandSpec::host_check` selects the `engine-host`, `engine-bench-host`,
`harness-host` or `cli-host` offline Cargo check profile. Each selects its own
manifest; the engine workspace excludes GPU kernels. Named profiles require the
owner-selected real Rust distribution supplied by `--shell-toolchain`; select
the version pinned by the project's `rust-toolchain.toml`. They use the existing
command approval and sandbox. They compile host code only and do not establish
GPU correctness or Spark performance.

After an edit, run the relevant profile, repair reported compiler errors and
repeat that check. Commands still have a maximum 120-second synchronous deadline;
a timed-out check is partial and needs a narrower approved command. There are no
durable start/poll/cancel jobs in this module. CLI installation build supervisors
are a separate lifecycle and cannot grant workspace sandbox permissions.

Output display truncation keeps bounded head and tail excerpts for stdout and
stderr independently. It drains discarded bytes while the command continues,
preserving recent compiler diagnostics. The receipt reports observed pipe byte
counts, omitted middle bytes, elapsed time and the completed child exit code or
signal. A known nonzero exit is a completed failed check that the model can
repair. Timeout, cancellation, pipe failure and build-area exhaustion remain
separate failures. Display truncation cannot turn a failed check into a pass.

**How it fits:** see the [`vade-harness` README](../../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../docs/README.md)
