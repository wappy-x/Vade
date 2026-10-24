# Product operations

Start directly on one DGX Spark with `vade launch`. The default checkpoint is
**nvidia/Qwen3.8-Flash-Next-FP4-Local**; no controller is needed. Follow
[INSTALL.md](../INSTALL.md) for prerequisites, the pinned model download and
installation, then [first run](../harness/docs/first-run.md) for local operation.
Use `vade status`, `vade logs`, `vade chat` and `vade down` to inspect, use and
stop the owned engine.

## Configure the Rust lifecycle

To control a Spark from another computer, copy
[cli/examples/fp4-local.conf](../cli/examples/fp4-local.conf) to a private local
file, fill its paths and existing SSH alias, and select it with `VADE_CONFIG`.
The publisher revision is already pinned. Then use `vade install` and `vade up`;
keep the same configuration for the entire lifecycle. Real machine identities,
addresses and account paths stay outside Git.

For multiple Sparks, add `[machine]` sections and reciprocal fabric addresses,
then select the configured machines with `--sparks` or `--hosts`. Follow
[INSTALL's topology instructions](../INSTALL.md#two-and-more-sparks).
The controller opens an owned SSH tunnel; `vade down` stops its recorded engines
and that tunnel. The local and controller routes have different context
presets: use the [defaults table](../README.md#defaults-by-route) and
[settings guide](../INSTALL.md#settings-you-may-change), not legacy FP8 values.

The older [vade.conf](../cli/examples/vade.conf) template is for the separate
[FP8 profile](../docs/install-legacy-fp8.md), selected explicitly with
`vade up --fp8-legacy`.

## Setup and qualification

- [Spark system setup](../docs/spark-setup.md): retained OS preparation shell for
  an administrator-managed fresh DGX OS.
- [Rust gate guide](../cli/docs/test.md): `vade test build` and
  `vade test determinism`, with established cases and receipt meanings.
- [Evaluation guide](../cli/evals.md): `vade test longctx`,
  `vade test gdn-state` and `vade test unix-flags-abi`.
- [Measurement reference](scripts/tools/README.md): `vade bench` and preserved
  experiment specifications.

The Rust CLI provides the product lifecycle, benchmark and evaluation commands.
Shell and Python helpers cover system preparation and checkpoint acquisition.
[INSTALL.md](../INSTALL.md) identifies their prerequisites and verification steps.

The [model fetch helper](fetch-model.sh) has a qualified FP4-Local invocation:

```sh
sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"
```

Run it from the checkout root on a prepared Linux download host. It verifies
the pinned snapshot and prints `verified 33 files`; the expected sibling
manifest is [committed here](../cli/examples/qwen3.8-flash-next-fp4-local.sha256).
The payload is 102,262,310,289 bytes (95.24 GiB). See
[INSTALL step 3](../INSTALL.md#step-3-get-the-model)
for disk planning, Python 3/venv prerequisites and publisher
licence (`nvidia-open-model-agreement`). The API reports this revision public
and ungated; no Hugging Face token or click-through acceptance is required
for the download.

Keep machine configuration, credentials, generated logs and raw evidence outside
Git. [Receipt status](../bench/receipts/README.md) explains evidence availability
and the removal of stale summaries. [Serving import hashes](scripts/serve-imports.json)
identify retired artifacts; they are historical provenance, not current-file hashes.
