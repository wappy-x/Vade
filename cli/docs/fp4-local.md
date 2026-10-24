# Launch the FP4-Local model

Start with [INSTALL.md](../../INSTALL.md), the current installation entry.

With a controller config selected through `VADE_CONFIG`, use `vade up`
(`--fp4-local` is optional) for `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one
Spark, or `vade up --fp4-local --sparks N` for a 2–24 Spark layer pipeline.
On any Spark count and with or without `--chats`, this profile defaults to
the `vade launch` preset: context 32,768 and output limit `min(context, 8,192)`.
The engine admits concurrent chats from context and memory; `vade status`
prints the admitted count. An explicit `--context` always wins, through the
engine ceiling 262,144 (`--context 262144` is the single-chat long-context
preset), and explicit `--output` reaches 16,384. A request past the configured context
is refused, and that refusal names `--context`. Direct `flash-serve` with
CONTEXT omitted still defaults to 32,768; `vade up` always passes context.
It omits
`--mtp` so the installed engine selects its current default. An explicit
`--mtp 0..3` or controller `MTP` value is preserved; the command option wins.
The engine's sampling, arithmetic and cache-retention defaults are unchanged.

1. Build the CLI on the controller with
   `cargo build --release --offline --manifest-path cli/Cargo.toml`.
2. Copy [fp4-local.conf](../examples/fp4-local.conf) to a new file and replace
   its placeholders. Set `VADE_CONFIG` to that file's absolute path. Use the
   same verified checkpoint and absolute remote checkpoint/staging paths on
   each selected Spark. See [INSTALL.md model acquisition](../../INSTALL.md#3-fetch-and-verify-the-qualified-checkpoint).
3. For a TCP-only chain, run `export SEQPAR_RDMA=0` in the controller shell
   **before installation** and keep it set for `install`, `up` and `doctor`.
   A chain prepared for RDMA leaves it unset.
   Preview `vade install --dry-run`, then run `vade install` to stage the
   binaries. The command needs existing Rust, SSH and GPU prerequisites and
   verified model files; it does not provision the OS or download weights.
4. Preview `vade up --fp4-local --dry-run`, then run `vade up --fp4-local`.
   A single Spark needs no fabric links or RDMA memlock. For 2–24 Sparks,
   configure both addresses of every adjacent link and select `--sparks N`,
   or list the cable order with `--hosts a,b,c,d`. No closing cable is needed.
   The exported transport setting applies to every pipeline rank.
5. Use `vade status`, `vade chat`, and `vade down`. Update/restart and saved
   ownership records preserve the FP4-Local profile and explicit overrides.

The legacy FP8 profile requires `vade up --fp8-legacy` and its own FP8
controller config; bare `vade up` selects FP4-Local. For the local launcher
without a controller, see [first run](../../harness/docs/first-run.md); its
local defaults and supported options differ from the controller settings above.
Larger counts explicitly select the layer chain. Installation records support
from the engine source used for the staged build; older manifests retain
one/two-Spark support and refuse larger counts before any remote command.
Run `vade install` from current source to stage a chain-capable engine. This launcher
never silently substitutes the legacy rank path or adds a tensor map.

Host proofs cover command construction and ownership round-trip for 1–24 ranks,
explicit budgets, default MTP omission, TCP/RDMA selection on every rank,
cleanup after partial startup, and old-build refusal before network I/O.
The one-Spark path additionally passed live startup, direct/gateway token-ID
comparisons, stream cancellation and owned shutdown with verified existing
weights. Larger chains require their own live GPU exactness qualification.
These host checks do not qualify two-Spark TCP or a fresh download/install.
