# Legacy FP8 installation on two DGX Sparks

For the current FP4-Local launcher, start with [INSTALL.md](../INSTALL.md).
This guide retains the separate **Qwen/Qwen3.8-Flash-Next-FP8** profile.
It requires the FP8 checkpoint and the explicit `vade up --fp8-legacy` selector.

## Prerequisites and build

Follow [INSTALL steps 1–2](../INSTALL.md#1-prepare-the-sparks) for machine,
SSH and compiler preparation and the local CLI build. Prepare a pair of Sparks
and use [vade.conf](../cli/examples/vade.conf) as the configuration template.
Keep `model = qwen3.8-flash-next-fp8`; replace SSH aliases, local source paths,
private staging/state paths and reciprocal fabric addresses. Select this
configuration with `VADE_CONFIG`.

Both ranks require the same absolute checkpoint and PTX paths. RDMA is the
default; explicit `SEQPAR_RDMA=0` selects TCP. Follow [network setup](network-setup.md)
for administrator preparation. Vade does not change that configuration.

<a id="4-put-the-official-weights-on-both-sparks"></a>
## 4. Put the official weights on both Sparks

The historical installation canary recorded a verified-copy installation and
independent official hash check; its stale receipt has been
[removed](../bench/receipts/README.md).
The pinned checkpoint revision is
`236dfdf285828023ca3bcd3f37366c58a3469b13`, also recorded in the
[configuration template](../cli/examples/vade.conf). Preserve the official
precision and the complete snapshot, including configuration, tokenizer, index
and every weight shard. Do not substitute FP4-Local files in this profile.

For a verified copy, follow the source-and-destination hash procedure in
[INSTALL model acquisition](../INSTALL.md#3-fetch-and-verify-the-qualified-checkpoint), using
the FP8 directory and its own trusted full-file manifest. Check the source
before transferring, then both destinations. Retain the directory basename
`Qwen3.8-Flash-Next-FP8`, which supplies the API model ID.

For a fresh acquisition, from the repository root on a prepared download host:

```sh
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 \
  236dfdf285828023ca3bcd3f37366c58a3469b13 \
  /absolute/path/to/Qwen3.8-Flash-Next-FP8
```

The helper needs Python with venv/pip and download access. Continue only after
its strict verification passes, then verify every destination copy. Neither
`vade install` nor `vade up` downloads these weights. Allow disk space for the
complete snapshot, download state and builds; no measured minimum is claimed.

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

The [legacy CLI defaults](../cli/src/cli.rs) are context 131,072, output limit
65,536 and MTP depth 2. Output is also bounded by the remaining context after
tokenizing the prompt. Use `--mtp 0` for plain decoding. These are FP8 settings;
they do not describe the FP4-Local launcher or a direct native server launch.

The removed canary receipt applied to its recorded source and manual two-rank
smoke with existing OS/network setup.
It does not qualify this current launcher sequence from a blank machine.
Use [benchmark methods](benchmarks.md) and retain actual response/token evidence
when qualifying a deployment. [Fresh install](fresh-install.md) keeps the longer
legacy checklist; [INSTALL](../INSTALL.md#5-start-the-two-spark-server) describes
the shared harness and gateway commands, with your FP8 model ID substituted.
