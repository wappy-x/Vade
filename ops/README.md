# Product operations

This page is a map. For each everyday task, from the first launch to stopping
the engine, it points you to the guide that covers it.

You run Vade through `vade`, a command-line tool written in Rust that starts,
stops, benchmarks and tests the engine. A few shell and Python helpers handle
the jobs around it: preparing the system and downloading the model's files
(the [checkpoint](../docs/GLOSSARY.md#ac)). [INSTALL.md](../INSTALL.md) lists
what those helpers need and how to check that they worked.

## Start on one Spark

With one DGX Spark, everything runs on the Spark itself, so you don't need a
second computer. The command is `vade launch`. It finds the model, saves your
defaults, installs Vade if it is missing, starts the engine and opens a chat.
Run it again later to reattach. By default it serves
**nvidia/Qwen3.8-Flash-Next-FP4-Local**, and it looks for the files under
`~/models` unless your saved config or `VADE_MODELS` points elsewhere.

1. Follow [INSTALL.md](../INSTALL.md) for the prerequisites, the pinned model
   download and the install.
2. Then read [first run](../harness/docs/first-run.md) for day-to-day use.

On this route, leave `VADE_CONFIG` unset. That variable is for driving a Spark
from another computer, described further down.

Once the engine is running, these commands cover the rest:

| Command | What it does |
| --- | --- |
| `vade status`, `vade logs` | Check on the engine Vade started. |
| `vade chat` | Chat with it. |
| `vade down` | Stop it. |

## Download the model

You download the model as part of [INSTALL step 3](../INSTALL.md#step-3-get-the-model).
This is the command on its own. Run the [fetch helper](fetch-model.sh) from the
top folder of your copy of the repository (the checkout root), on a Linux
machine with `python3` (including venv and pip), `sha256sum`, `find` and
`sort`:

```sh
sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"
```

The long hex string is the pinned revision: one exact version of the model.
The helper downloads that revision and checks it with Hugging Face's own
verifier, which fails on any missing, extra or damaged file. It then writes a
SHA-256 hash of every file (a checksum manifest) next to the model folder, as
`<model folder>.sha256`. It prints nothing while it works, and a good run ends
with `verified 33 files`. If a step fails, it prints one line that says what
went wrong and what to do next.

- **Size:** the download is 102,262,310,289 bytes (95.24 GiB).
- **Access:** Hugging Face lists this revision as public and ungated, so you
  need no token and no click-through.
- **License:** NVIDIA's license (`nvidia-open-model-agreement`) still applies.

After the download, compare the files with
[qwen3.8-flash-next-fp4-local.sha256](../cli/examples/qwen3.8-flash-next-fp4-local.sha256),
the list of hashes committed in the repository.
[Check the files](../INSTALL.md#check-the-files) gives the command. INSTALL
step 3 also covers disk space and the Python setup.

## Drive Sparks from another computer

You can also drive a Spark from a second computer, called the
[controller](../docs/GLOSSARY.md#ac). It reaches the Spark over SSH. This is
also the route for using more than one Spark.

1. Copy [cli/examples/fp4-local.conf](../cli/examples/fp4-local.conf) to a
   private file on the controller.
2. Fill in its paths and the SSH destination you already use for the Spark.
   The model revision is already pinned in the file.
3. Set the `VADE_CONFIG` environment variable to the path of that file.
4. Run `vade install`, then `vade up`.

`vade up` opens an [SSH tunnel](../docs/GLOSSARY.md#sz) from the controller
to the Spark. `vade down` closes the tunnel and stops the engines that
`vade up` started. It finds them through the same config file, so keep
`VADE_CONFIG` pointing at that file until you have run `vade down`.

Your config file holds real machine names, addresses and account paths, so
keep it out of Git.

### More than one Spark

Sparks in a group talk to each other over direct cables, called the
[fabric](../docs/GLOSSARY.md#dg). Give each Spark its own `[machine <name>]` section
in the config, such as `[machine spark-a]`. Then, for every cable, add a `fabric.<peer>` line on both of
the Sparks it joins:

- `<peer>` is the name of the Spark at the other end of the cable.
- The value is this Spark's own address on that cable, without a port.

The commented-out second Spark in
[fp4-local.conf](../cli/examples/fp4-local.conf) shows the pattern. Then pick
the machines to use with `--sparks` or `--hosts`.
[Two and more Sparks](../INSTALL.md#two-and-more-sparks) walks through the
whole setup.

### The older FP8 profile

The older [vade.conf](../cli/examples/vade.conf) template is for the separate
[FP8 profile](../docs/install-legacy-fp8.md). Vade uses that profile only when
you ask for it with `vade up --fp8-legacy`.

## Context size

The [context window](../docs/GLOSSARY.md#ac) is the prompt plus the reply,
counted in tokens. Vade starts with 32768 tokens and caps the reply at 8192
(or the context size, if that is smaller), whether you start it on the Spark
or from a controller. The two routes differ only when memory is short:

- **On the Spark itself**, if the engine refuses the default or saved size for
  lack of memory, `vade launch` and `vade up` try 16384, then 8192, then 4096
  tokens, save the first size that starts and tell you which one. A
  `--context` you set yourself is never lowered: if the engine refuses it, you
  get an error.
- **From a controller**, Vade never lowers the context. If the engine refuses
  32768, pass a smaller `--context`. Stay at 8192 or above if you want
  `vade chat`: from a controller it opens the coding agent, which needs at
  least 8192 tokens.

[Context size on one Spark](../INSTALL.md#context-size-on-one-spark),
[Defaults on a controller](../INSTALL.md#defaults-on-a-controller) and the
[settings guide](../INSTALL.md#settings-you-may-change) list the current
values. The values in the legacy FP8 guide don't apply here.

## Setup and test guides

Each of these guides covers one job in depth:

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

Keep machine config, credentials, logs and raw results out of Git. Two files
explain what the repository does hold:

- [Receipt status](../bench/receipts/README.md) says which result files are in
  this repository and which older summaries were removed.
- [scripts/serve-imports.json](scripts/serve-imports.json) is kept only as
  history. It lists hashes of files Vade no longer uses, so don't check
  current files against it.
