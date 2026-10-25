# First chat on one Spark

This page explains the local launcher on one DGX Spark and what happens on
later runs. Use one terminal directly on the Spark, signed in as your normal
user. The default checkpoint is **nvidia/Qwen3.8-Flash-Next-FP4-Local**.

Install with [INSTALL.md](../../INSTALL.md) first. It is the one place to
install the prerequisites, get a runnable copy of Vade, and download the
pinned, verified model.

## Start here

Finish [the on-Spark route](../../INSTALL.md#route-a-on-this-spark); it ends
with your first `vade launch`, and later runs start the same way. Leave
`VADE_CONFIG` unset: that variable points Vade at a controller config, which
this local route does not use. If you use the release bundle, open its folder
and run:

```sh
./vade launch
```

If you built from source and the CLI is on your `PATH`, use `vade launch`.
Either way, it detects the Spark, picks the existing checkpoint, saves local
defaults, installs the engine and harness if needed, starts the engine, and
opens chat in your current folder. Try `Say hello in one sentence.` in chat,
then type `/exit` to get back to the shell; the engine keeps running. Then check
on it or stop it:

```sh
./vade status
./vade verify
./vade down
```

Drop the `./` when the CLI is on your `PATH`. You do not need another
computer, an SSH tunnel or a gateway. The default API address is
`http://127.0.0.1:18080/v1`; if that port is busy, first setup picks another
free loopback port. `vade status` prints the base address, such as
`http://127.0.0.1:18080`; the API lives under its `/v1`. Keep the same
`VADE_HOME` and leave endpoint overrides unset, so bare `vade verify` can find
the saved running port in `~/.vade/state/run.conf`.

With at least 8192 context tokens, chat opens the agent; smaller contexts open
direct text chat. On Linux the agent can read and edit workspace files. Its
shell tool and `--check`/`.vade/check` need the macOS sandbox, so command
execution and automatic shell checks are not available on the Spark. The
launcher turns off automatic checks on Linux, including checks configured in
the workspace. See [harness tools](../README.md) for which tools work on which
platform.

## Bundles and source builds

A launcher bundle holds `vade`, `flash-serve`, `flash.ptx`, `vade-harness`
and `launcher-bundle.conf` together. It needs a working NVIDIA driver and the
system tools listed in INSTALL, but no Rust, because it already ships the built
programs. The v0.1.0 bundle and its `SHA256SUMS` are on the GitHub release
page; follow
[INSTALL's acquisition and verification instructions](../../INSTALL.md#optional-release-bundle)
before you run it. `VADE_BUNDLE` can instead point to an existing bundle folder
by absolute path.

Building from source is the alternative. A source install needs a Git clone
(recommended) or a downloaded source archive, plus the pinned Rust toolchains
and their cached compiler dependencies. An archive (GitHub's Download ZIP or `git archive`) records its
commit as the build id; an archive without that id, or one edited after
download, builds as `unknown`. The launcher finds the checkout through its
executable or the current folder; `VADE_SOURCE` can name an absolute checkout
explicitly. It copies the tracked engine and harness sources into its private
stage, builds the GPU kernels, then builds the engine and harness there. Commit
or restore any changes under `engine/`, `harness/` and `rust-toolchain.toml`
before you install.

Builds run offline. Do the first kernel-build preparation in
[INSTALL](../../INSTALL.md#step-2-get-the-code) before your first source
launch; `vade launch` does not fill a cold compiler cache. Logs from failed
builds stay under `~/.vade/build-jobs/`. The launcher does not install system
packages, compilers or models, and does not change the driver or network.

## Select the existing checkpoint

Use the pinned [model acquisition and manifest check](../../INSTALL.md#step-3-get-the-model)
to put the checkpoint in `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`.
The local launcher serves FP4-Local; the
[legacy FP8 profile](../../docs/install-legacy-fp8.md) uses a separate
controller config and `vade up --fp8-legacy`.

Vade picks the checkpoint in this order:

1. A valid saved `model_path`.
2. On a fresh install, an explicit `VADE_MODELS`.
3. The folder above, or `~/models/Qwen3.8-Flash-Next-FP4-Local`.
4. If neither of those exists, a search of `~/models`.

If it cannot pick one, an interactive launch asks you for an existing
checkpoint folder. Without a terminal, it reports the
problem and asks you to set `VADE_MODELS`. If you name an invalid checkpoint
explicitly, Vade does not quietly fall back to another one.

The folder must hold the configuration, tokenizer, shard index and every weight
shard the index names. A link to the whole folder is fine; links to single
model files must point inside that folder. That rules out Hugging Face cache
snapshots whose shards link out to a shared `blobs` directory: use the
download from INSTALL instead, which writes real files, not links.

To switch away from a valid saved checkpoint, first stop it with its existing
`VADE_HOME`, then pick a fresh absolute `VADE_HOME` and set `VADE_MODELS` for
the new install. Keep that same `VADE_HOME` for all its lifecycle commands.

Two hash lists do different jobs:

| Hash list | Where it comes from | What it tells you |
| --- | --- | --- |
| Publisher manifest | Committed with Vade; INSTALL checks your download against it | Your download matches the publisher's snapshot |
| Vade's list of your files | Kept in Vade's private state: the launcher copies the checkpoint's own `manifest.sha256` there if there is one; otherwise the FP4-Local engine writes one (a local receipt) after hashing | Identifies your files. A list Vade writes itself does not prove where they came from |

The fetch helper's `MODEL_DIR.sha256`, next to the model folder, is not copied
into Vade's private state for you. The original FP4-Local model folder is left
unchanged.

## Defaults and later runs

Local defaults are saved in `~/.vade/config`. They match the
[controller defaults](../../INSTALL.md#route-b-from-a-controller): both start at
32768, and only the local launcher lowers context, when the engine refuses it
for memory. Context covers the prompt and the generated reply; `/health`
reports the effective request limits, and replies must fit in the context that
is left.

| Setting | What happens locally |
| --- | --- |
| Context | 32768, capped by the checkpoint's limit; if the engine refuses it for memory, 16384, 8192 or 4096 with a one-line notice. An explicit `--context` the engine refuses is an error |
| Output | The smaller of the selected context and 8192 tokens |
| Drafting | Engine default, ON, unless an explicit setting or inherited `MTP` overrides it |
| Prefix retention | OFF unless `VADE_FLASHNEXT_PREFIX_CACHE=1` is set at startup; supported on one Spark only |
| Port | Try 18080 on loopback at first setup; otherwise save an available loopback port |

`vade up` and `vade launch` accept `--context`, `--output`, `--mtp`,
`--memory-bytes` and `--cache-bytes 0`. Supported settings are saved once they
pass validation. Explicit context requests must fit the checkpoint and the
measured memory; they are not silently lowered. Explicit output can go up to
16384 tokens if context and memory allow. `--fp4-local` is optional.

Run `vade down` before you change settings on a running engine. Bare `vade up`
reuses the saved settings, except that an `MTP=0..3` environment variable
left set in your shell overrides drafting; an explicit `--mtp` wins. The engine
decides how many chats it can run at once, so local `--chats` is refused.
Positive `--cache-bytes` values are refused too; `--cache-bytes 0` turns off
prefix retention even when the environment opts in. See
[INSTALL settings](../../INSTALL.md#settings-you-may-change).

| Path | What it holds |
| --- | --- |
| `~/.vade/config` | Saved checkpoint, limits, port and private paths |
| `~/.vade/stage/` | Installed engine, kernels, harness and source-build files |
| `~/.vade/state/` | Engine ownership record, runtime manifests and retained log |
| `~/.vade/build-jobs/` | Source-build supervision records and logs |
| `~/.vade/projects/` | Chat journals and input recall, one folder per project path (`VADE_CHAT_IN_PROJECT=1` keeps them in `./.vade`, git-ignored) |

An explicit absolute `VADE_HOME` replaces `~/.vade` for all of these paths.

| Command | What it does |
| --- | --- |
| `vade launch` | Install if needed, start or reuse this installation's engine, open chat |
| `vade install` | Prepare local defaults and artifacts without starting the engine |
| `vade up` | Start or reuse the engine without opening chat |
| `vade chat` | Open the agent or small-context direct chat on the running engine |
| `vade status` / `vade status --json` | Inspect the recorded engine and endpoint |
| `vade logs` | Read the current or last retained engine log |
| `vade doctor` | Check the machine, model, saved limits, artifacts and this installation's endpoint |
| `vade verify` | Compare repeated greedy token IDs with drafting OFF and ON |
| `vade down` | Stop this installation's recorded engine |
| `vade update` | After shutdown, rebuild or restage engine and harness artifacts |

For a separate controller install, set `VADE_CONFIG` and follow
[Route B](../../INSTALL.md#route-b-from-a-controller). That explicit setting
takes priority over local defaults for lifecycle commands; `vade launch`
refuses it. If you also use the older `vade run qwen3-8b` route, give it its own
fresh `VADE_HOME`. If you already have a controller install that uses the
default config, stop it with its own config before you create local defaults
here.

If startup fails, keep the ownership records and use `vade logs` and
`vade doctor`. [INSTALL's first-run problems](../../INSTALL.md#first-run-problems)
maps each launcher message to the next step. Stop other applications with
their original launcher; never kill processes by name.
