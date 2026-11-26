# Install and flows

This page shows how a person or an agent gets from nothing to a running
engine, and back. Both follow the same steps. Each step has one command, one
line that says it worked, an error that names the cause, and a next action
that is safe to run.

Status: built, 2026-10-29. `vade setup` and the `vade install --dry-run`
plan work as this page shows.

## The flow

```text
  get vade            vade setup  (asks: which Sparks, FP4-Local or FP8,
  (clone + script,        |        model folder; --yes takes the defaults)
   or a bundle)           |  writes the config, then runs the three steps
        |                 v
        +-------> vade install --> vade up --> vade doctor --> vade chat
                  bundle or        "ready"     PASS or FAIL    vade read FILE
                  source build        |                        vade tools
                                      |                        API at /v1
                                      v
                  vade update  <-- vade down
                  (then vade up)
```

On one Spark, `vade launch` stays the short way: install if needed, start,
then open chat. The one-Spark quick start is `scripts/install.sh`, then
`vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300` when the model is missing, then `vade launch`. The script needs
no model on disk. Without one, its local `vade install` builds, compiles the
GPU code, saves no settings and prints the `vade pull` line. The first
`vade launch` or `vade up` then saves the settings and checks the model files.

## Each step

| Step | Command | It worked when you see | It fails when | Next safe action |
| --- | --- | --- | --- | --- |
| Get vade | Spark: `sh scripts/install.sh`. Bundle: unpack it. Mac: build the CLI | `Install complete.` and the path to `vade` | Curl is missing, or a source build lacks Rust tools or `cc`; the GPU is not a GB10, its driver is older than CUDA 13.0, or it is busy; no network on the first run | Do what the `Next:` line says, then run the same command again |
| Set up | `vade setup` | One `Setup step` line per step below, then `Setup done.` and exit status 0 | `--yes` is missing a required answer, such as the Sparks on a controller | Add the flag it names, then run it again |
| Install | `vade install` | Exit status 0. Local: `Install done in <time>`, then the GPU startup cache fills. Controller: `Installed build <revision>.` | A toolchain is missing, the checkout has changes, a GPU is busy, the disk is full; from a controller, the model is missing. On a Spark with no model yet, install still passes and prints the `vade pull` line | Do what the `hint:` line says, then `vade install`. A second run is always safe |
| Start | `vade up` | On a Spark: `Up ready in <time>: <address>, context <n> tokens, ...`. From a controller: `Vade is ready on <n> Sparks (<names>) at <address>: context <n> tokens, ...` | The engine refuses the memory, the port is taken, the model load times out | `vade logs`, then `vade doctor`, then `vade up` |
| Check | `vade doctor` | One row per check, then `PASS` | A row says `FAIL` | Run that row's `next:` step, then `vade doctor` |
| Use | `vade chat`, `vade read FILE`, `vade tools`, or the API at `<address>/v1` | Chat opens; the API answers `/v1/models` | The engine is not running | `vade up` |
| Stop | `vade down` | On a Spark: `Down done in <time>`. From a controller: `Vade is stopped.` When nothing runs: `Vade is already stopped.` | Vade cannot confirm which process it started | Keep the state folder; follow CLI recovery |
| Update | `git pull`, then `vade update` | On a Spark: `Local engine and chat updated.` From a controller: the new build, then a restart question or the restart commands | On a Spark, the engine is still running | On a Spark: `vade down`, then `vade update`, then `vade up` |

Exit status is 0 on success and 1 on a failure. A `vade` command writes an
`error:` line and a `hint:` line with the next action to stderr. With
`--json`, the error is in the one JSON object on stdout instead.
`scripts/install.sh` writes `install:` and the reason. Most install
failures then add a `Retry:` line with the command to run again.

## One install command

`vade install` installs for the setup that `vade config` shows. With saved
local defaults, or on a fresh Spark, that is this Spark. With a controller
config, it is each Spark in that config, over SSH.

It takes the engine, the GPU kernels and the chat program from the first
source that applies:

| Where | 1st choice | 2nd choice | Last choice |
| --- | --- | --- | --- |
| This Spark | The bundle that `VADE_BUNDLE` names | A launcher bundle next to the `vade` program | A source build of the checkout that holds `vade` |
| Controller | With `source = installed-bundle`, the bundle next to `vade`, and nothing else. Otherwise, a bundle in `prebuilt_dir` whose engine and CLI revisions match the checkout | | A source build on each Spark |

A source build uses the pinned Rust 1.99.0 and nightly-2026-10-01. Install
checks every bundle file against its SHA-256 before use. A controller bundle
that does not match is named in the output, and install builds from source.

Install keeps all of its files in two places:

| Path | What it holds |
| --- | --- |
| `~/.vade` (or `VADE_HOME`) | Local defaults (`config`), staged builds (`stage`), records of what Vade started and logs (`state`), chat sessions (`projects`). `scripts/install.sh` adds the mark `.vade-home`, the script `bin/vade`, downloaded bundles (`bundles`) and source clones (`source.*`) |
| On each Spark, from a controller | The `stage_dir`, `state_dir` and source folders in the controller config |

Install changes no shell profile, adds no symlink or alias and runs no sudo.
It does not download models. `scripts/install.sh` writes a small script,
`bin/vade`, in the Vade home. With the default home, it also writes
`~/.local/bin/vade`. It skips that file when a file not from Vade is there, when
`~/.local` or `~/.local/bin` is a link, or when it cannot write there. The
stock Ubuntu login profile puts `~/.local/bin` on `PATH` when the folder
exists. Otherwise, add the folder to `PATH` yourself; the docs give the line.

On a Spark, running install again is safe. Staged files that still match
their record are reused, and nothing is rebuilt. Changed or missing files are
staged again.

From a controller, each install builds a new generation, and the controller
switches to it only after every Spark has it. An engine that already runs
keeps the old build, and `vade up` then cannot attach to it. So while an
engine runs, use `vade update`: it stages the new build and asks before it
restarts the engine. Without a terminal it keeps the engine and prints the
`vade down` and `vade up` commands. `vade setup` skips install when the
controller config already has one.

A second lifecycle command on the same home stops with
`Vade is already changing this engine.`

`vade install --dry-run` prints the plan, then changes nothing. On a new
Spark with a source checkout at `/home/you/vade`, it prints:

```text
Plan for vade install:
  Where      this Spark
  Files      a source build of /home/you/vade
  Toolchain  Rust 1.99.0 and nightly-2026-10-01
  Installed  not yet
  Model      the FP4-Local checkpoint under /home/you/models, found at install (optional: without it, install saves no defaults)
  State      /home/you/.vade: config, stage and state
  Steps      detect this Spark and model, save defaults when the model is there, stage the engine, kernels and chat, warm the GPU startup cache
  Next       vade install
Local config: /home/you/.vade/config
```

From a controller, the plan names the Sparks, the files (a prebuilt bundle
or a source build on each Spark), the toolchain, each Spark's model folder
and the state folder. Then it lists the commands that install would run.

On a computer that is not a Spark and has no controller config, install says
so and names `vade setup`. It does not print a Spark plan there.

## The setup walk-through

`vade setup` asks a few questions and shows each default in brackets. Press
Enter to take it.

1. **Which Sparks?** On a Spark: `[this Spark]`. On another computer: the
   names you type after `ssh`, in any order, such as `spark-a,spark-b`. With a
   setup in place: `[keep this setup]`. A list runs the same code as
   `vade up --sparks`: Vade reads the cables and asks for an address only when
   it cannot read one.
2. **Which checkpoint?** `[FP4-Local]`. Setup asks only for 2, 4 or 6
   Sparks, because FP8 runs only there. Otherwise it is FP4-Local.
3. **Where is the model?** On a Spark: `[found under ~/models]`. From a
   controller, for each Spark: `[~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local]`,
   or the FP8 folder. The answer is an absolute path or a path under `~/`;
   setup checks it before the next question.
4. **Linked Sparks only: the address of each Spark on each cable.** There is
   no default.

For this Spark, setup writes nothing itself; install saves `~/.vade/config`.
For a controller, setup writes the controller config at the path that
`vade config` reads by default. It starts from the matching example in
`cli/examples`, points the sources at this checkout, or names the installed
bundle when `vade` runs from one, and keeps the private folders under
`~/.vade`. Only `VADE_SOURCE` makes a `vade` from a bundle use a checkout. A list
of Sparks moves a config aside. Setup then creates the new file only if no file
is there, also when two setups run at once. A run with no list uses the config
it finds. It reads the checkpoint from that config's `model`, so
an FP8 config starts with `vade up --fp8` even if a person wrote it. It
starts the Sparks that setup's first line names; a config without that line
starts every machine it lists, in its order, except `vade verify` engines.
When an engine from the config already runs, setup runs a plain `vade up`,
which attaches with the engine's saved settings. Before it writes a config or
starts an install, setup runs the Spark and cable checks of `vade up`, such
as a different address at each end of a cable.

Then setup runs `vade install`, `vade up` and `vade doctor` in that order. It
stops at the first step that fails, names the step and exits with status 1.
If a controller config already has an install, setup skips `vade install`:
a second install makes a new generation, and `vade up` could then not attach
to an engine that runs from the first one. Run `vade update` after a pull.
On a Spark with saved local settings, setup skips `vade install` when the
installed files were built from the current checkout, for example right
after `scripts/install.sh`. After a pull, they are not current, so setup runs
`vade install`, which builds the new checkout.
Install runs its own SSH, GPU, disk, toolchain, model and network checks on
every Spark before it builds. Doctor runs last, on the running engine, so it
checks the Sparks and cables that `vade up` chose. Before a start, doctor
checks only the Sparks that a plain `vade up` would use.

For agents: `vade setup --yes` never asks. It takes the defaults and these
flags: `--hosts a,b`, `--fp8`, `--model PATH` and `--link FROM:TO=ADDRESS`
(once per cable end). Under `--yes` the steps get no input either, so a step
that would ask stops with its own message. Without a terminal and without
`--yes`, setup stops and names `--yes`. A stop signal to setup (SIGINT or
SIGTERM) goes on to the step that runs, and setup waits for that step to
unwind. `vade setup --dry-run` prints the answers, the config text and
the steps, and writes nothing.

## Decisions

- **A script gets `vade` first.** Before `vade` exists, something must supply
  it. `scripts/install.sh` tries the prebuilt bundle of main. When the bundle
  does not fit, it installs the pinned Rust and builds `vade`. Then it calls
  `vade install`, which holds the rest of the install logic. The script itself
  checks the Spark and the Vade home first, writes the `vade` scripts, and
  runs `--uninstall`.
- **No second install path.** Setup runs the same `vade` program for each
  step, so it shares the locks, checks and messages.
- **Open question for the owner.** The default controller config is outside
  `~/.vade`: `~/.config/vade/config` on Linux,
  `~/Library/Application Support/vade/config` on macOS. Setup writes it there
  because every command already reads it there without `VADE_CONFIG`. The
  other choice is `~/.vade/controller.conf`, which then needs `VADE_CONFIG`
  in every shell.
