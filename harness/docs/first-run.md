# First chat on one Spark

Use one terminal directly on your DGX Spark, signed in as your ordinary user.
The default checkpoint is **nvidia/Qwen3.8-Flash-Next-FP4-Local**. Start with
[INSTALL.md](../../INSTALL.md): it is the single installation entry point for
prerequisites, a runnable Vade copy, and the pinned, verified model download.
This page explains the local launcher and later runs.

## Start here

Complete [the on-Spark route](../../INSTALL.md#route-a-on-this-spark); it ends
with your first `vade launch`. Start later runs the same way, with
`VADE_CONFIG` unset. With a supplied launcher bundle, open its folder and run:

```sh
./vade launch
```

With a source installation and the CLI on your command search path, use
`vade launch`. It detects the Spark, selects the existing checkpoint, saves
local defaults, installs the engine and harness if needed, starts the engine,
and opens chat in your current working folder. Type
`Say hello in one sentence.` in chat. Type `/exit` to return to the shell;
the engine stays running. Then inspect or stop it:

```sh
./vade status
./vade verify
./vade down
```

Omit `./` when using the CLI from your command search path. No other computer,
SSH tunnel or gateway is needed. The default API address is
`http://127.0.0.1:18080/v1`; first setup chooses another free loopback port if
that port is busy. `vade status` prints the base address, such as
`http://127.0.0.1:18080`; the API is under its `/v1`. Bare
`vade verify` discovers the saved running port from `~/.vade/state/run.conf`;
keep the same `VADE_HOME` and leave endpoint overrides unset for that behavior.

With at least 8192 context tokens, chat opens the agent; smaller contexts open
direct text chat. The agent can read and edit workspace files on Linux. Its shell tool and
`--check`/`.vade/check` require the macOS sandbox; command execution and automatic
shell checks are unavailable on the Spark. The launcher disables automatic
checks on Linux, including workspace-configured checks. See [harness tools](../README.md)
for the platform boundary.

## Bundles and source builds

A launcher bundle contains `vade`, `flash-serve`, `flash.ptx`, `vade-harness`
and `launcher-bundle.conf` together. Follow
[INSTALL's acquisition and verification instructions](../../INSTALL.md#optional-release-bundle)
before executing it. `VADE_BUNDLE` can instead name an existing absolute
bundle folder. A bundle needs a working NVIDIA driver and the system tools
listed in INSTALL, but does not need Rust to stage its supplied programs.

Source installation needs a Git clone (recommended) or a downloaded source
archive, plus the pinned Rust toolchains and their cached compiler
dependencies. An archive (GitHub's Download ZIP or `git archive`) records its
commit as the build id; one without that id, or edited after download, builds
as `unknown`. The launcher finds the checkout through its executable or current
folder; `VADE_SOURCE` can select an absolute checkout explicitly. It snapshots
tracked engine and harness sources into its private stage, builds GPU kernels,
then builds the engine and harness there. Commit or restore changes under
`engine/`, `harness/` and `rust-toolchain.toml` before installation.

Builds are offline. Follow the initial kernel-build preparation in
[INSTALL](../../INSTALL.md#step-2-get-the-code) before the first source launch;
a cold compiler cache is not filled by `vade launch`. Failed build logs remain
under `~/.vade/build-jobs/`. The launcher does not install system packages,
compilers or models, and does not change the driver or network.

## Select the existing checkpoint

Use the pinned [model acquisition and manifest check](../../INSTALL.md#step-3-get-the-model)
to place the checkpoint in `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`.
The local launcher serves FP4-Local; the
[legacy FP8 profile](../../docs/install-legacy-fp8.md) uses a separate controller
configuration and `vade up --fp8-legacy`.

A valid saved `model_path` takes priority. For a fresh installation, an explicit
`VADE_MODELS` selects the checkpoint before automatic discovery. Otherwise Vade
checks the directory above or `~/models/Qwen3.8-Flash-Next-FP4-Local`; if neither
exists, it searches `~/models`. If selection fails, an interactive launch asks for an
existing checkpoint folder. Without a terminal, it reports the problem and
asks you to set `VADE_MODELS`. An invalid explicit selection does not silently
fall back to another checkpoint.

The folder must include the configuration, tokenizer, shard index and every
weight shard it names. A link to the whole folder is allowed; individual
payload links must stay inside that folder. Hugging Face cache snapshots with
shards linked out to a shared `blobs` directory are therefore unsuitable: use
the materialized download from INSTALL instead.

To switch away from a valid saved checkpoint, first stop it using its existing
`VADE_HOME`, then select a fresh absolute `VADE_HOME` and set `VADE_MODELS` for
the new installation. Keep that same `VADE_HOME` for its lifecycle commands.

Publisher verification and runtime receipts have different purposes. The
committed publisher manifest in INSTALL verifies the downloaded snapshot.
The launcher copies an existing checkpoint `manifest.sha256` into private
state when present; otherwise the FP4-Local engine creates a local receipt
there after hashing. The fetch helper's sibling `MODEL_DIR.sha256` is not
automatically staged as that runtime manifest. A locally generated receipt
identifies your files; it does not prove publisher provenance. The original
FP4-Local model folder stays unchanged.

## Defaults and later runs

The local defaults are saved in `~/.vade/config`. They match the
[controller defaults](../../INSTALL.md#route-b-from-a-controller): both start at
32768; only the local launcher lowers context, when the engine refuses it for
memory. Context
includes the prompt and generated reply; `/health` reports the effective
request limits, and replies must fit the remaining context.

| Setting | Local behavior |
| --- | --- |
| Context | 32768, capped by the checkpoint's limit; if the engine refuses it for memory, 16384, 8192 or 4096 with a one-line notice. An explicit `--context` the engine refuses is an error |
| Output | The smaller of the selected context and 8192 tokens |
| Drafting | Engine default, ON, unless an explicit setting or inherited `MTP` overrides it |
| Prefix retention | OFF unless `VADE_FLASHNEXT_PREFIX_CACHE=1` is set at startup; supported on one Spark only |
| Port | Try 18080 on loopback at first setup; otherwise save an available loopback port |

`vade up` and `vade launch` accept `--context`, `--output`, `--mtp`,
`--memory-bytes` and `--cache-bytes 0`. Supported settings are saved after
validation. Explicit context requests must fit the checkpoint and measured
memory; they are not silently lowered. Explicit output can reach 16384 tokens
if context and memory permit. `--fp4-local` is optional.

Run `vade down` before changing settings on a running engine. Bare `vade up`
reuses the saved settings, except that inherited `MTP=0..3` supplies a drafting
override; explicit `--mtp` wins. The engine selects its concurrent-chat
capacity: local `--chats` is refused. Positive `--cache-bytes` values are also
refused; `--cache-bytes 0` disables prefix retention even when the environment
opts in. See [INSTALL settings](../../INSTALL.md#settings-you-may-change).

| Path | Contents |
| --- | --- |
| `~/.vade/config` | Saved checkpoint, limits, port and private paths |
| `~/.vade/stage/` | Installed engine, kernels, harness and source-build files |
| `~/.vade/state/` | Engine ownership record, runtime manifests and retained log |
| `~/.vade/build-jobs/` | Source-build supervision records and logs |
| `~/.vade/projects/` | Chat journals and input recall, one folder per project path (`VADE_CHAT_IN_PROJECT=1` keeps them in `./.vade`, git-ignored) |

An explicit absolute `VADE_HOME` replaces `~/.vade` for all these paths.

| Command | Action |
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

For a separate controller installation, set `VADE_CONFIG` and follow
[Route B](../../INSTALL.md#route-b-from-a-controller). That explicit selector
wins over local defaults for lifecycle commands; `vade launch` refuses it.
Use a fresh `VADE_HOME` for the older `vade run qwen3-8b` route. Stop an existing
default controller with its own config before creating local defaults.

If startup fails, keep the ownership records and use `vade logs` and
`vade doctor`. [INSTALL's first-run problems](../../INSTALL.md#first-run-problems)
match the launcher messages to the next action. Stop other applications using
their original launcher; never kill processes by name.
