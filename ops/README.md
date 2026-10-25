# Product operations

This page points you to the right guide for each day-to-day task, from the
first launch to stopping the engine.

`vade` is the Rust CLI that starts, stops, benchmarks and tests Vade. Shell
and Python helpers prepare the system and download the checkpoint.
[INSTALL.md](../INSTALL.md) lists what they need and how to check their
results.

## Start on one Spark

On one DGX Spark, start with `vade launch`. You don't need a second computer.
By default it serves **nvidia/Qwen3.8-Flash-Next-FP4-Local**.

1. Follow [INSTALL.md](../INSTALL.md) for the prerequisites, the pinned model
   download and the install.
2. Then read [first run](../harness/docs/first-run.md) for day-to-day use.

Once Vade has started the engine, four commands cover the rest:

| Command | What it does |
| --- | --- |
| `vade status`, `vade logs` | Check on the engine Vade started. |
| `vade chat` | Chat with it. |
| `vade down` | Stop it. |

## Download the model

The [fetch helper](fetch-model.sh) downloads the model. Run it from the
checkout root on a Linux machine that has Python 3 with venv:

```sh
sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"
```

The long hex string is the pinned revision: one exact version of the model.
The helper checks the download against that revision, comparing each file
with the hashes in
[qwen3.8-flash-next-fp4-local.sha256](../cli/examples/qwen3.8-flash-next-fp4-local.sha256),
which is committed in the repository. It finishes with `verified 33 files`.

- **Size:** the download is 102,262,310,289 bytes (95.24 GiB).
- **Access:** Hugging Face lists this revision as public and ungated, so you
  need no token and no click-through.
- **Licence:** NVIDIA's licence (`nvidia-open-model-agreement`) still applies.

[INSTALL step 3](../INSTALL.md#step-3-get-the-model) covers disk space and the
Python setup.

## Drive Sparks from another computer

You can also run Vade on a Spark from a second computer, called the
controller. It reaches the Spark over SSH.

1. Copy [cli/examples/fp4-local.conf](../cli/examples/fp4-local.conf) to a
   private file on the controller.
2. Fill in its paths and the SSH destination you already use for the Spark.
   The model revision is already pinned in the file.
3. Point `VADE_CONFIG` at the file.
4. Run `vade install`, then `vade up`.

`vade up` opens an [SSH tunnel](../docs/GLOSSARY.md#sz) from the controller
to the Spark. `vade down` closes the tunnel and stops the engines it started.
Leave `VADE_CONFIG` pointing at the same file until then, so that `vade down`
can stop them.

Your config file holds real machine names, addresses and account paths, so
keep it out of Git.

### More than one Spark

The Sparks talk to each other over direct cables, called the
[fabric](../docs/GLOSSARY.md#dg). Add a `[machine]` section to the config for
each Spark. In it, set `fabric.<peer>` to that Spark's own address on its
cable to that peer. Every cable has two ends, so set it on both Sparks.

Then pick the machines to use with `--sparks` or `--hosts`.
[Two and more Sparks](../INSTALL.md#two-and-more-sparks) walks through it.

### Context size

Vade starts with a context window of 32768 tokens, whether you start it on
the Spark or from a controller. When you start on the Spark itself and memory
is short, it steps down to a smaller size. From a controller it doesn't.

The [defaults table](../INSTALL.md#defaults-by-route) and the
[settings guide](../INSTALL.md#settings-you-may-change) list the current
values. The values in the legacy FP8 guide don't apply here.

### The older FP8 profile

The older [vade.conf](../cli/examples/vade.conf) template is for the separate
[FP8 profile](../docs/install-legacy-fp8.md). You pick that profile explicitly
with `vade up --fp8-legacy`.

## Setup and test guides

- [Spark system setup](../docs/spark-setup.md): the older OS preparation
  scripts, for an administrator setting up a fresh DGX OS.
- [Test guide](../cli/docs/test.md): `vade test build` and
  `vade test determinism`, the cases they run, and what their result files
  ([receipts](../docs/GLOSSARY.md#pr)) mean.
- [Evaluation guide](../cli/evals.md): `vade test longctx`,
  `vade test gdn-state` and `vade test unix-flags-abi`.
- [Measurement reference](scripts/tools/README.md): `vade bench`, plus the saved
  specs of past experiments.

## What stays out of Git

Keep machine config, credentials, logs and raw results out of Git.

- [Receipt status](../bench/receipts/README.md) says which result files are in
  this repository and which older summaries were removed.
- [scripts/serve-imports.json](scripts/serve-imports.json) is kept only as
  history. It lists hashes of files Vade no longer uses, so don't use it to
  check current files.
