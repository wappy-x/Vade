# Launch the FP4-Local model

This page covers starting `nvidia/Qwen3.8-Flash-Next-FP4-Local` from a
controller, on one Spark or on a chain of up to 24. Start with
[INSTALL.md](../../INSTALL.md); on the Spark itself, without a controller, use
the [first-run guide](../../harness/docs/first-run.md) instead.

## Defaults

With a controller config selected through `VADE_CONFIG`, `vade up` starts
FP4-Local on one Spark (`--fp4-local` is optional). `vade up --sparks 2` runs
it as a tensor split: both Sparks work on every token, with every proven level
on (`VADE_W2_PATH=pipeline`, a debug switch, runs the layer pipeline instead).
`vade up --sparks N` with N from 3 to 24 runs it as a layer pipeline: each
Spark holds a slice of the model's layers and passes its result to the next
one. Any other count is
refused with "FP4-Local supports 1..24 Sparks. Choose --sparks 1..24 or list
that many distinct --hosts."

- **Context and output.** However many Sparks you use, and with or without
  `--chats`, the defaults match `vade launch`: a context of 32,768 tokens and
  output of `min(context, 8192)`. The engine decides how many chats run at
  once from the context and its memory, and `vade status` prints that count.
  The controller accepts and saves `--chats` and `--cache-bytes`, but the
  current FP4-Local engine ignores them. Its memory admission and separate
  prefix-cache setting control those limits.
- **Your own limits.** A `--context` you pass always wins, up to the engine's
  ceiling of 262,144. This long-context setting still lets the engine decide
  the chat count from memory. `--output` takes 1,024 to 16,384. A request longer
  than the context is refused, and the error names `--context`. `flash-serve` started by hand
  without a context also defaults to 32,768; `vade up` always passes one.
- **Drafting.** `vade up` leaves out `--mtp`, so the installed engine picks
  its own default. If you pass `--mtp 0..3` (0 turns server drafting off) or set `MTP` in the controller's
  shell, Vade keeps it, and the flag wins over the variable.
- **Everything else.** Sampling, arithmetic and cache defaults are the
  engine's own; `vade up` doesn't change them.

## Steps

1. Build the CLI on the controller:
   `cargo build --release --offline --manifest-path cli/Cargo.toml`.
2. Copy [fp4-local.conf](../examples/fp4-local.conf) to a new file, replace
   its placeholders, and set `VADE_CONFIG` to that file's absolute path. Every
   Spark you use needs the same verified checkpoint, at the same absolute
   checkpoint and staging paths. See
   [getting the model](../../INSTALL.md#step-3-get-the-model).
3. Linked Sparks use RDMA by default. For TCP instead, run
   `export SEQPAR_RDMA=0` in the controller's shell **before you install**, and
   keep it set for `install`, `up` and `doctor`. `SEQPAR_RDMA=1` makes RDMA
   required; leave it unset to keep the RDMA default. Any other value is
   refused. Two Sparks over TCP also need `VADE_W2_PATH=pipeline`: the tensor
   split needs RDMA, so they run the layer pipeline.
4. Run `vade install --dry-run` to preview, then `vade install` to stage the
   programs. It needs Rust, SSH and the GPU already working and the model
   files already verified; it doesn't set up the OS or download weights.
5. Run `vade up --dry-run`, then `vade up`. One Spark needs no fabric links or
   RDMA locked-memory limit. For 2 to 24 Sparks, give both addresses of every
   cable between neighbors, then pass `--sparks N` or list the cable order
   with `--hosts a,b,c,d`. The chain doesn't need a cable that closes it into
   a ring. The transport you chose in step 3 applies to every Spark.
6. Use `vade status`, `vade chat` and `vade down`. Updates and restarts keep
   the FP4-Local profile and any limits you set.

## Chains of more than two Sparks

More than two Sparks always run as a layer chain. When `vade install` stages
the engine, it records whether that engine supports chains. A build staged by
an older install supports one or two Sparks only, and `vade up` refuses a
larger count before it contacts any machine; run `vade install` from current
source to fix that. Vade never falls back to the older tensor-map rank path
on its own.

The legacy FP8 profile needs `vade up --fp8-legacy` and its own FP8 config;
a bare `vade up` always means FP4-Local.

## What has been tested

- **Host tests** (no Spark) cover the commands Vade builds and the records it
  keeps for 1 to 24 Sparks, limits you set, leaving out `--mtp` by default,
  TCP or RDMA on every Spark, cleanup after a start that fails partway, and
  refusing an old build before any network call.
- **On one Spark**, live runs also covered startup, matching token IDs
  directly and through the gateway, canceling a stream, and shutdown, with
  existing verified weights.
- **Larger chains** ran in the published
  [four-VM benchmark cards](../../bench/receipts/2026-10-24/README.md), but
  each new layout still needs its own exactness check on real hardware.
  Neither the host tests nor these runs cover two Sparks over TCP or a fresh
  download and install.
