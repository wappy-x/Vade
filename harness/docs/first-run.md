# First chat on one Spark

This page is for you if you have one DGX Spark and want to chat with the model
right there on it. It covers `vade launch`, how it finds the model, the
settings it saves, and how to check, stop and restart the engine on later runs.

Work in one terminal on the Spark itself, signed in as your normal user. The
default checkpoint is **nvidia/Qwen3.8-Flash-Next-FP4-Local**.

## Start here

Install with [INSTALL.md](../../INSTALL.md) first. It is the one place to
install the prerequisites, get a runnable copy of Vade, and download the
pinned, verified model. Then finish
[the on-Spark route](../../INSTALL.md#route-a-on-this-spark); it ends with your
first `vade launch`, and later runs start the same way.

If you use the release bundle, open its folder and run:

```sh
./vade launch
```

If you built from source and the CLI is on your `PATH`, run `vade launch`.
Either way, `vade launch`:

1. Detects the Spark.
2. Picks the existing checkpoint (see
   [How the launcher finds the model](#how-the-launcher-finds-the-model)).
3. Saves local defaults.
4. Installs the engine and harness, if they are not installed yet.
5. Starts the engine.
6. Opens chat in your current folder.

Try `Say hello in one sentence.` in chat, then type `/exit` to get back to the
shell. The engine keeps running.

The chat window needs an interactive terminal. Over SSH, a normal login is
one. For a single command, add `-t` and give the CLI's path, since that shell
does not have your `PATH` changes: from a source install,
`ssh -t you@spark-address 'cd "$HOME/vade" && ./cli/target/release/vade chat'`;
from the bundle, `ssh -t you@spark-address 'cd "$HOME/vade-launch" && ./vade chat'`.
Without a terminal, chat shows plain lines instead of its window.

Run `vade launch` again later and it reuses the saved settings, and the engine
too if it is still running. To check on the engine or stop it:

```sh
./vade status
./vade verify
./vade down
```

Drop the `./` when the CLI is on your `PATH`.

To check a fresh source install in one go, run
`sh harness/scripts/route-a-check.sh` from the checkout. It builds the CLI,
checks which model `vade launch` will load, and starts an engine in a new
private Vade home of its own, so it never touches `~/.vade`. It sends one chat
request, runs `vade verify` against that engine, tries to stop it again (also
when a step fails or you interrupt the script; a failed stop is a FAIL), and
prints PASS or FAIL with the seconds each step took. See
[Check the whole route in one go](../../INSTALL.md#check-the-whole-route-in-one-go).

### Where the API listens

Everything stays on the Spark: you do not need another computer, an SSH tunnel
or a gateway. The default API address is `http://127.0.0.1:18080/v1`. If that
port is busy, first setup picks another free loopback port and saves it.

`vade status` prints the base address, such as `http://127.0.0.1:18080`; the
API lives under its `/v1`. `vade status` and `vade verify` find the port on
their own, so you don't need to pass it. That holds as long as you keep the
same `VADE_HOME` you launched with and leave `VADE_API_URL` unset.

### The coding agent on the Spark

With at least 8192 context tokens, chat opens the coding agent; smaller
contexts open direct text chat.

On the Spark the agent can read and edit your files, but it can't run shell
commands or checks: those need a sandbox that only macOS has. The launcher
turns checks off for you. See [harness tools](../README.md) for what works
where.

## Release bundle or source build

`vade launch` runs from either a release bundle or a source checkout. Both end up
with the same engine, GPU kernels and chat program; they differ in what the
Spark needs beforehand.

### Release bundle

A launcher bundle is a folder of programs that are already built: `vade`,
`flash-serve` (the engine), `flash.ptx` (its GPU kernels), `vade-harness` (the
chat program) and `launcher-bundle.conf`, which describes them. Because nothing
needs building, it needs no Rust, only a working NVIDIA driver and the system
tools listed in INSTALL.

The v0.1.0 bundle and its `SHA256SUMS` are on the GitHub release page. Follow
[INSTALL's download and checksum steps](../../INSTALL.md#optional-release-bundle)
before you run it. While the repository is private, the plain download links
fail; INSTALL's private-preview note shows the `gh release download` command
that works. The launcher uses the bundle that holds the `vade` you ran.
`VADE_BUNDLE` can instead point to an existing bundle folder by absolute path.

### Source build

A source install needs a Git clone (recommended) or a downloaded source
archive, plus the pinned Rust toolchains and their cached compiler
dependencies.

The launcher finds the checkout through the `vade` program's location or your
current folder; `VADE_SOURCE` can name an absolute checkout explicitly. It
copies the tracked engine and harness sources into its private stage folder,
builds the GPU kernels, then builds the engine and harness there. Before you
install, commit or restore any changes under `engine/`, `harness/` and
`rust-toolchain.toml`; the launcher refuses a Git clone with unsaved changes
there.

Source launches build without the network, so run INSTALL's
[one-time kernel build](../../INSTALL.md#run-one-kernel-build-source-builds-only)
first; it downloads what the build needs. Logs from failed builds stay under
`~/.vade/build-jobs/`.

If the launcher says your tree builds as `unknown`, your source archive was
edited after download or lacks its build id; a Git clone avoids this.

Whichever way you install, the launcher does not install system packages,
compilers or models, and does not change the driver or network.

## How the launcher finds the model

Use INSTALL's [model download and manifest check](../../INSTALL.md#step-3-get-the-model)
to put the checkpoint in `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`. The
engine serves it under the folder's name in lower case,
`qwen3.8-flash-next-fp4-local`.
The local launcher serves FP4-Local only; for older models, see
[INSTALL's legacy models](../../INSTALL.md#legacy-models).

Vade picks the checkpoint in this order:

1. A valid saved `model_path`.
2. On a fresh install, an explicit `VADE_MODELS`.
3. The folder above, or `~/models/Qwen3.8-Flash-Next-FP4-Local`.
4. If neither of those exists, a search of `~/models`.

If it cannot pick one, an interactive launch asks you for an existing
checkpoint folder. Without a terminal, it reports the problem and asks you to
set `VADE_MODELS`. If you name an invalid checkpoint explicitly, Vade does not
quietly fall back to another one.

### What the model folder must hold

The folder must hold the configuration, tokenizer, shard index and every weight
shard the index names. A symbolic link to the whole folder is fine, but links
to individual model files must point inside that folder.

That rules out a Hugging Face cache snapshot, whose shard files are links into
a shared `blobs` directory elsewhere. Use the download from INSTALL instead,
which writes real files, not links.

### Switch to another checkpoint

Vade keeps the checkpoint it saved first. To use a different one, stop the
engine with its existing `VADE_HOME`, then pick a fresh absolute `VADE_HOME`
and set `VADE_MODELS` for the new install. Keep that same `VADE_HOME` for all
its lifecycle commands.

### Checksums

INSTALL checks your download against the publisher's list. After that, Vade
keeps its own list of your files in `~/.vade/state` so it can tell if they
change. It never changes your model folder.

## Saved settings

Local defaults are saved in `~/.vade/config`, and they match the defaults a
[controller](../../INSTALL.md#route-b-from-a-controller) uses.

If you don't pick a context, the launcher starts at 32768 (or the size it saved
last time). When the engine says it lacks the memory, it steps down through
16384, 8192 and 4096, and tells you which size it used. If you pick one with
`--context`, it uses exactly that or stops with an error.

Context is the prompt plus the generated reply, counted in tokens, so a reply
must fit in the context the prompt leaves. The engine's `/health` endpoint
reports the request limits that are in effect.

| Setting | Local default |
| --- | --- |
| Context | 32768, or the checkpoint's own limit if that is lower; stepped down only as described above |
| Output | The smaller of the context and 8192 tokens |
| [Drafting](../../docs/GLOSSARY.md#dg) | On (the engine's default), unless `--mtp` or an `MTP` variable in your shell says otherwise |
| Prefix retention (reusing the start of an earlier prompt) | Off, unless `VADE_FLASHNEXT_PREFIX_CACHE=1` is set when the engine starts; it works on one Spark and on two Sparks (the tensor split) |
| Port | 18080 on loopback at first setup, or another free loopback port if that one is taken; then saved |

### Change a setting

Run `vade down` first if the engine is running. Then pass the new values to
`vade up` or `vade launch`. Settings are saved once they pass validation, and
bare `vade up` reuses them later. These are the ones you are most likely to
change:

- **`--context`** must fit the checkpoint and the measured memory. If the
  engine refuses the size you asked for, you get an error.
- **`--output`** can go up to 16384 tokens if context and memory allow.
- **`--mtp N`** sets how many tokens the engine drafts ahead (0 turns drafting
  off, 3 is the most). Watch out: an `MTP` variable left in your shell also
  changes drafting, even on a bare `vade up`, unless you pass `--mtp`.
- **`--memory-bytes`** sets a memory budget; the engine's memory check still
  applies.

Other flags, such as `--cache-bytes 0` and `--chats`, are covered in
[INSTALL settings](../../INSTALL.md#settings-you-may-change).

## Where Vade keeps its files

| Path | What it holds |
| --- | --- |
| `~/.vade/config` | Saved checkpoint, limits, port and private paths |
| `~/.vade/stage/` | Installed engine, kernels, harness and source-build files |
| `~/.vade/state/` | Engine ownership record, runtime manifests and retained log |
| `~/.vade/build-jobs/` | Source-build supervision records and logs |
| `~/.vade/projects/` | Chat journals and input recall, one folder per project path (`VADE_CHAT_IN_PROJECT=1` keeps them in `./.vade`, git-ignored) |

An explicit absolute `VADE_HOME` replaces `~/.vade` for all of these paths.

## Commands

| Command | What it does |
| --- | --- |
| `vade launch` | Install if needed, start or reuse this installation's engine, open chat |
| `vade install` | Prepare local defaults and artifacts without starting the engine |
| `vade up` | Start or reuse the engine without opening chat |
| `vade chat` | Open the agent or small-context direct chat on the running engine |
| `vade status` / `vade status --json` | Inspect the recorded engine and endpoint |
| `vade logs` | Read the current or last retained engine log |
| `vade doctor` | Check the machine, model, saved limits, artifacts and this installation's endpoint |
| `vade verify` | Check that the engine gives the same answer every time, with drafting on and off |
| `vade down` | Stop this installation's recorded engine |
| `vade update` | After shutdown, rebuild or restage engine and harness artifacts |

## If you also use a controller

- For a separate [controller](../../docs/GLOSSARY.md#ac) install, set
  `VADE_CONFIG` and follow
  [Route B](../../INSTALL.md#route-b-from-a-controller). While `VADE_CONFIG`
  is set, lifecycle commands use the controller install instead of these
  local defaults, and `vade launch` refuses to run.
- If you already have a controller install that uses the default config, stop
  it with its own config before you create local defaults here.

## If startup fails

Run `vade logs` and `vade doctor` first, and don't delete anything under
`~/.vade/state`; Vade uses it to know which engine is its own.
[INSTALL's first-run problems](../../INSTALL.md#first-run-problems) maps each
launcher message to the next step. If another application is in the way, stop
it with the launcher that started it; never kill processes by name.
