# Legacy FP8 installation on two DGX Sparks

For the current FP4-Local launcher, start with [INSTALL.md](../INSTALL.md).
This page covers the separate **Qwen/Qwen3.8-Flash-Next-FP8** profile.
It needs the FP8 checkpoint and the `vade up --fp8-legacy` flag.

## Prerequisites and build

Follow INSTALL [Step 1](../INSTALL.md#step-1-prerequisites) and [Step 2](../INSTALL.md#step-2-get-the-code) for machine,
SSH and compiler preparation and the local CLI build. Prepare a pair of Sparks
and use [vade.conf](../cli/examples/vade.conf) as the configuration template.
Keep `model = qwen3.8-flash-next-fp8`; replace SSH aliases, local source paths,
private staging/state paths and reciprocal fabric addresses. Select this
configuration with `VADE_CONFIG`.

Both ranks require the same absolute checkpoint and PTX paths. RDMA is the
default; explicit `SEQPAR_RDMA=0` selects TCP. Follow [network setup](network-setup.md)
for administrator preparation. Vade does not change that configuration.

<a id="4-put-the-official-weights-on-both-sparks"></a>
## Put the official weights on both Sparks

An earlier test install recorded a verified copy of the weights and an
independent check of the official hashes. Its out-of-date receipt has been
[removed](../bench/receipts/README.md). The pinned checkpoint revision is
`236dfdf285828023ca3bcd3f37366c58a3469b13`, also recorded in the
[configuration template](../cli/examples/vade.conf). Preserve the official
precision and the complete snapshot, including configuration, tokenizer, index
and every weight shard. Do not substitute FP4-Local files in this profile.

For a verified copy, follow the source-and-destination hash procedure in
[INSTALL Step 3](../INSTALL.md#step-3-get-the-model), using
the FP8 directory and its own trusted full-file manifest. Check the source
before transferring, then both destinations. Retain the directory basename
`Qwen3.8-Flash-Next-FP8`, which supplies the API model ID.

For a fresh acquisition, from the repository root on a prepared download host:

```sh
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 \
  236dfdf285828023ca3bcd3f37366c58a3469b13 \
  /absolute/path/to/Qwen3.8-Flash-Next-FP8
```

The helper needs Rust 1.99.0, curl and download
access ([ops/fetch-model.sh](../ops/fetch-model.sh)). Continue only after
its strict verification passes, then verify every destination copy. Neither
`vade install` nor `vade up` downloads these weights. Leave disk space for the
complete snapshot, the download state and the builds; we have not measured a
minimum.

## Install, start and stop

For a TCP-only pair, run this in the configured Mac shell **before installation**
and keep it set for subsequent `install`, `up` and `doctor` commands:

```sh
export SEQPAR_RDMA=0
```

A pair prepared for RDMA leaves it unset. From the configured Mac checkout,
with idle GPUs:

```sh
vade install --dry-run
vade install
vade up --fp8-legacy --dry-run
vade up --fp8-legacy
vade status
vade chat
```

Exit the chat, then run `vade down`. Preserve ownership records if cleanup
fails and follow [Recovery](../cli/README.md#recovery).

The legacy CLI defaults ([cli/src/cli.rs:329](../cli/src/cli.rs#L329), [line 100](../cli/src/cli.rs#L100)) are context 131,072, output limit
65,536 and MTP depth 2. Output is also bounded by the remaining context after
tokenizing the prompt. Use `--mtp 0` for plain decoding. These are FP8 settings;
they do not describe the FP4-Local launcher or a direct native server launch.

Nobody has run these steps end to end on a blank machine. The earlier test
install, whose receipt was removed, covered only its own source revision and a
hand-started two-Spark smoke test on Sparks whose OS and network were already
set up. To qualify your own deployment, follow the
[benchmark methods](benchmarks.md) and keep the real responses and token IDs.

More on this route:

- [Fresh install](fresh-install.md) has the longer legacy checklist.
- INSTALL's [two and more Sparks](../INSTALL.md#two-and-more-sparks) and
  [client gateway](../INSTALL.md#client-gateway) sections cover the shared
  harness and gateway commands; use your FP8 model ID in them.
