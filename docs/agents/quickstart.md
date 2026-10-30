# Quickstart for agents

This page is for an AI agent that sets up or runs Vade for a person, from a
shell. It gives the exact commands, what each one returns, where the files
are and what to do for each kind of error. A person can read it too; the
[install guide](../../INSTALL.md) has the long form.

## Rules

- Run every command as the person's own user. Never use sudo. Never edit a
  shell profile.
- Never kill a process by name, and never stop a process that Vade did not
  start. To stop Vade, run `vade down`.
- Never delete or edit files under `~/.vade/state` or a controller
  `state_dir`. `vade down` reads them to find what to stop.
- Ask the person before you download a model. The FP4-Local checkpoint is
  102.26 GB.
- Ask the person before you discard or stash changes in their checkout.
- The commands on this page exit 0 on success and 1 on a failure. Errors
  come in three forms:
  - A `vade` command writes an `error:` line (what failed) and a `hint:`
    line (what to do next) to stderr.
  - With `--json`, stderr stays empty. The one JSON object on stdout carries
    the error.
  - `scripts/install.sh` writes `install:` and the reason, then a `Retry:`
    line with the command to run again.

  Do what the hint or the reason says, then run the same command again. One
  exception: once a setup exists (a controller config that setup wrote, or
  `~/.vade/config` on a Spark), run `vade setup --yes` again without
  `--hosts`, `--fp8`, `--model` and `--link`. Setup refuses them, because it
  uses the setup as it is.
- Lifecycle commands take `--dry-run`, which prints the plan and changes
  nothing. `status`, `doctor`, `config`, `detect` and `version` take `--json`.

## 1. Find out where you are

```sh
uname -sm
command -v vade || ls "$HOME/vade/cli/target/release/vade"
vade config --json
vade status --json
```

`uname -sm` prints `Linux aarch64` on a DGX Spark. Any other answer means
this computer can only be a controller: it drives Sparks over SSH.

`vade config --json` prints one `vade.config.v1` object. Its `setup` field
tells you what exists:

| `setup` | Meaning | Next |
| --- | --- | --- |
| `none` | Nothing is set up for this user | Section 2 |
| `local` | This Spark has saved defaults in `~/.vade/config` | `vade setup --yes` (it reuses them) |
| `controller` | A controller config exists; `path` names it | `vade setup --yes` (it reuses it) |
| `legacy` | The older `vade run qwen3-8b` setup | Use `vade run qwen3-8b`; do not mix the two |

`vade status --json` prints one `vade.status.v1` object. It exits 0 when
`phase` is `stopped` or the engine is ready, and 1 on an error or an unhealthy
engine. `endpoint` is the API address while the engine runs.

## 2. Install

### On a Spark

Without a `vade` program yet, clone and run the installer. It installs the
pinned Rust toolchains if they are missing, builds `vade`, checks the Spark
and the model, and runs `vade install`. It needs no sudo.

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
vade setup --yes
```

The repository is private for now. Without access, the clone fails with
`could not read Username` or `Repository not found`. Tell the person; do not
try other URLs. [Install step 2](../../INSTALL.md#step-2-get-the-code) gives
the two ways in.

After the installer, `vade setup --yes` reuses the model and settings the
installer saved. It runs `vade install` (nothing is rebuilt), `vade up` and
`vade doctor`, and its last output line before the next steps is
`Setup done.` On a Spark with no saved settings, it uses this Spark and the
FP4-Local checkpoint under `~/models`; add `--model` with an absolute path
or a path under `~/` for another folder.

The `export` line lasts for one shell. Repeat it in each new shell.

### On another computer (a controller)

The controller needs Git, a C linker (`cc`), curl, ssh, rsync and Rust 1.99.0.
It must reach each Spark with `ssh` and no password prompt.

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
cd "$HOME/vade"
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$HOME/.cargo/bin:$PWD/cli/target/release:$PATH"
vade setup --dry-run --yes --hosts spark-a
vade setup --yes --hosts spark-a
```

`--hosts` takes what you type after `ssh`, such as `you@spark-a` or an alias
from `~/.ssh/config`. The dry run prints the config that setup will write
and the steps. Read it before the real run.

Linked Sparks also need each Spark's own address on each cable. Give one
`--link` per cable end, in cable order:

```sh
vade setup --yes --hosts spark-a,spark-b \
  --link spark-a:spark-b=192.0.2.1 --link spark-b:spark-a=192.0.2.2
```

Add `--fp8` for the FP8 checkpoint on 2, 4 or 6 Sparks. Two FP4-Local Sparks
need RDMA; see [network setup](../network-setup.md).

Setup writes the controller config at the path `vade config` shows and never
replaces it. A second `vade setup --yes` uses the file that is there and takes
the checkpoint from its `model`. When that config already has an install,
setup skips `vade install` and runs `vade up` and `vade doctor`. When an
engine from it runs, `vade up` attaches with that engine's saved settings. To change the
setup, ask the person to edit that file. To install new code after a pull, run
`vade down`, then `vade update`.

Under `--yes`, the steps get no input. A step that needs an answer stops with
an `error:` line instead of waiting. To stop setup, send SIGTERM or SIGINT to
it; it passes the stop on to the step that runs, waits for that step and
exits 1.

## 3. Use the engine

| Task | Command |
| --- | --- |
| Check every part | `vade doctor --json`: one `vade.doctor.v1` object; `result` is `pass` or `fail`; each entry in `checks` has `name`, `state` (`ok`, `fail`, `skip`), `detail` and `next` |
| Chat (needs a terminal) | `vade chat` |
| Read a file as Markdown | `vade read FILE` |
| List the chat tools | `vade tools` |
| Call the API | `curl --fail --silent "$ENDPOINT/models"`, with `ENDPOINT` from `vade status --json` (it already ends in `/v1`) |
| Show recent engine logs | `vade logs` |

An API chat request thinks first unless it sends `"enable_thinking": false`.
The reasoning counts against `max_tokens`. See the
[OpenAI API page](../openai-api.md).

## 4. Stop and update

On a Spark:

```sh
vade down
git -C "$HOME/vade" pull
vade update
vade up
```

`vade down` prints `Down done in <time>`, or `Vade is already stopped.` when
nothing runs. `vade update` refuses to run while the engine runs. A plain
`vade up` starts again with the settings saved in `~/.vade/config`.

From a controller, leave the engine running and run:

```sh
vade update
```

It pulls each source checkout and stages the new build. Then it asks
before a restart. Without a terminal it keeps the engine running and prints
`Running engine kept. Restart when ready:` and two commands, `vade down` and
`vade up` with this engine's Sparks and limits. Run those two commands as
they are. Do not use a plain `vade up` instead: after `vade down`, a
controller keeps no record of the last start, and a plain `vade up` starts
one FP4-Local Spark, or two FP8 Sparks. For a config that setup wrote,
`vade setup --yes` also starts the Sparks that the config names, with the
default limits.

From a controller, `vade down` prints `Vade is stopped.`, or
`Vade is already stopped. Nothing to stop.` when nothing runs.

## Where the files are

| Path | What it holds | May you change it? |
| --- | --- | --- |
| `~/.vade/config` | Local defaults on a Spark | Only with `vade down` first, then `vade launch` or `vade up` with new flags |
| `~/.vade/stage` | Staged engine, GPU kernels and chat | No; `vade install` and `vade update` manage it |
| `~/.vade/state` | Records of what Vade started, and logs | Never |
| `~/.vade/projects` | Saved chat sessions | Use `vade sessions` |
| The path in `vade config` | The controller config | Ask the person |
| `~/.vade/controller` | Stage and state of a controller that setup wrote | Never |
| `engine/target/install-kernels.log` in the checkout | Output of the installer's kernel build | Read only |

`VADE_HOME` moves `~/.vade`. `VADE_CONFIG` names a controller config. Keep
both the same for every command in one setup.

## What to do on each kind of error

| You see | Cause | Safe action |
| --- | --- | --- |
| `Vade is not set up yet` | No setup for this user | Section 2 |
| `This computer is not a Spark, and it has no controller config yet.` | `vade install` on a controller before setup | `vade setup --yes --hosts ...` |
| `This command runs only on a DGX Spark (Linux aarch64)` | A local command on another computer | Run it on the Spark, or use a controller |
| `Standard input is not a terminal, so setup cannot ask` | `vade setup` without `--yes` | Add `--yes` and the flags it names |
| `Add --hosts` or `Add --link FROM:TO=ADDRESS` | A required answer is missing | Ask the person for the value; do not guess addresses |
| A `toolchain` or `rustup` hint | A pinned Rust toolchain is missing | Run the `rustup` command in the hint |
| `source has local changes` | The checkout has uncommitted or untracked files | Show `git status` to the person |
| `GPU is in use` | Another program uses the GPU | Tell the person; do not stop it |
| A doctor row `model files` fails | The checkpoint is missing or incomplete | Ask the person, then `vade pull` with the pinned command in [download models](../pull-models.md) |
| `refused --context` or `Not enough memory` | The engine has too little memory | Run `vade up` with a smaller `--context` |
| `Port ... is in use` | Another program has the port | Tell the person |
| `Vade is already changing this engine.` | Another Vade command holds the lock | Wait, then `vade status` |
| `did not reply to the pre-start check` | SSH failed or timed out | `ssh -o BatchMode=yes HOST true`, then run the command again |
| `Vade hit an internal error` | A bug in Vade | Stop. Give the person the output; do not retry in a loop |

The [first-run problems](../../INSTALL.md#first-run-problems) table lists more
messages.
