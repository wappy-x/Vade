# Quickstart for agents

This page is a script for an AI agent that sets up Vade on a DGX Spark for a person. Vade is an inference engine for the Spark, and `vade` is its command.

Run the steps in order. Each step gives the command, the expected output, the exit code, the check that proves the step worked, and what to do on the two most likely failures.

The path is three commands on the Spark:

```sh
curl -fsSL https://vade.lol/install.sh | sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade up
```

FP4-Local is the checkpoint that this path downloads. It fits on one Spark. [INSTALL.md](../INSTALL.md) has the long form for people.

## Rules

- **Your account.** Run every command as the person's own user. Never use sudo. Never edit a shell profile.
- **Processes.** Never kill a process by name. Never stop a process that Vade did not start. To stop Vade, run `vade down`.
- **State.** Never delete or edit files under `~/.vade/state` or a controller `state_dir`. `vade down` reads them to find what to stop.
- **Downloads.** Ask the person before you download a model. The FP4-Local checkpoint is 102.26 GB.
- **Addresses.** Never guess an address or an SSH name. Ask the person.
- **Checkouts.** Ask the person before you discard or stash changes in their checkout.
- **Previews.** Lifecycle commands take `--dry-run`, which prints the plan and changes nothing.
- **Long commands.** The installer, a source build, `vade pull`, `vade up`, `vade setup`, `vade update` and `vade verify` can run longer than your tool's time limit. Use the [background form](#run-a-long-command) for them.
- **JSON.** `status`, `doctor`, `config`, `detect`, `models` and `version` take `--json`.

## How to read an error

Vade commands exit 0 on success and 1 on a failure. Errors come in three forms:

- A `vade` command writes an `error:` line and a `hint:` line to stderr. The first line says what failed. The second line says what to do next.
- With `--json`, stderr stays empty. The one JSON object on stdout carries the failure in its `error` and `next` fields.
- The installer writes a line that starts with `install:`. That line gives the reason and a `Next:` action. Then it usually writes a `Retry:` line. To retry, run that line. If there is no `Retry:` line, run the same install command again.

Do what the hint or the reason says. Then run the same command again.

There is one exception. Once a setup exists, only a list of Sparks changes it: `vade up --sparks` or `vade setup --hosts`. The other answer flags `--fp8`, `--model` and `--link` stop with an error on their own, because setup uses the setup as it is. A setup exists when a controller config or `~/.vade/config` is there.

The [error table](#what-to-do-on-each-kind-of-error) at the end of this page lists the common messages.

## Run a long command

A long command ends on its own with exit 0 or 1. If your tool has a time limit, run the command in this form. Replace `<command>` with the command of the step:

```sh
nohup sh -c '<command>; echo "exit $?"' > ~/vade-step.log 2>&1 < /dev/null &
echo $! > ~/vade-step.pid
```

Read `~/vade-step.log` every few minutes with `tail -n 5 ~/vade-step.log`. Then follow these rules:

- **Done.** The command has ended when the last line of the log is `exit` and a number. That number is the exit code.
- **Still running.** `kill -0 "$(cat ~/vade-step.pid)"` exits 0 while the command runs. Keep waiting.
- **Silent.** If a running command prints no new line for 30 minutes, tell the person. Leave it running.
- **Stopped.** If the command does not run and the log has no `exit` line, run the same command again.
- **Retries.** Retry a failed command at most three times. Then stop, and give the person the output.
- **To stop it.** `pgrep -P "$(cat ~/vade-step.pid)"` prints the process IDs of the command. Send SIGTERM to each ID.

## Before step 1

Run these checks on the Spark. If the person wants linked Sparks, go to [Two or four Sparks](#two-or-four-sparks) now. Step L2 runs these checks on each Spark.

**Check A. Find out which computer this is.**

```sh
uname -sm
```

- **Expected output.** `Linux aarch64` on a DGX Spark.
- **Exit code.** 0.
- **What to do next.** With `Linux aarch64`, continue with Check B. With any other output, this computer cannot run the engine. It can only be a controller, which is a computer that drives Sparks over SSH. Go to [Two or four Sparks](#two-or-four-sparks).

**Check B. Find out what is already set up.**

```sh
vade config --json
```

- **Exit code on a fresh Spark.** 127, because the shell does not find `vade`. Use the exit code, because the shell's message can differ. This is not a failure. Go to Check C.
- **Expected output when `vade` exists.** One `vade.config.v1` object. Its `setup` field tells you what exists.

| `setup` | Meaning | What to do next |
| --- | --- | --- |
| `none` | Nothing is set up for this user | Go to Check C |
| `local` | This Spark has saved defaults in `~/.vade/config` | Run `vade status --json`. If `ready` is true, go to step 5. Otherwise, go to step 3 |
| `controller` | A controller config exists, and `path` names it | Run `vade up`. It starts the Sparks of that setup |
| `legacy` | The older `vade run qwen3-8b` setup | Use `vade run qwen3-8b`. Do not mix the two setups |

**Check C. Find out if the disk has room.**

```sh
df -h "$HOME"
```

- **Expected output.** The `Avail` column shows at least `106G`. `df -h` counts in GiB. That is 95.24 GiB for the model and 10 GiB for the install, rounded up. If `ls ~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local` lists files, the model needs no new room, so `10G` is enough. These are minimums, and we have no measured total for a source build.
- **Exit code.** 0.
- **What to do next.** With enough room, go to step 1. With less room, tell the person and stop.

## Step 1. Install Vade

Run this on the Spark:

```sh
curl -fsSL https://vade.lol/install.sh | sh
```

The installer checks the GPU and the driver. Then it installs the prebuilt bundle of main from dl.vade.lol. It needs no sudo and no model.

- **Expected output.** `Install complete. Next steps:`, then the next steps. Follow this page, not the steps that the installer lists.
- **Exit code.** 0. Exit 0 alone does not prove that the install worked. If `curl` fails, `sh` receives no script and also exits 0.
- **Check.** The output has the line `Install complete. Next steps:`.

The two most likely failures:

- **`curl: (6) Could not resolve host`, or any other `curl:` line.** The Spark has no route to vade.lol. Tell the person. Run the command again when the network works.
- **`install: the GPU is in use.`** Run `vade status --json`. If `vade` exists and `phase` is not `stopped`, Vade holds the GPU. Run `vade down` and retry. In every other case, another program holds the GPU. Tell the person and do not stop it.

When the bundle does not fit this Spark, the installer prints a `Prebuilt:` line with the reason. Then it clones public main into a new folder in `~/.vade` and builds from source there. This needs `git` and `cc`, and it ends with the same `Install complete. Next steps:` line. If the clone or the build fails, follow the `install:` and `Retry:` lines. If it fails again, use [Build from source](#build-from-source).

## Step 2. Put vade on PATH

First, find out whether this shell already finds `vade`:

```sh
vade --version
```

- **Expected output.** A line that starts with `vade` and a version number, such as `vade 0.1.0`.
- **Exit code.** 0 when `vade` is on `PATH`. Then go to step 3.

If `vade` is not found, the shell prints `vade: command not found` and the exit code is 127. Then run the export line that the installer prints last. The command in that line starts with `export PATH=` and looks like this example:

```sh
export PATH="$HOME/.vade/bin:$PATH"
```

The folder in the line can differ between Vade versions, so run the line that the installer printed. Copy that line now, because the next long command replaces `~/vade-step.log`. This command prints it from the log:

```sh
sed -n 's/.*\(export PATH=.*\)/\1/p' ~/vade-step.log | tail -n 1
```

Then run `vade --version` again.

- **Check.** `vade --version` prints a line that starts with `vade` and a version number, and exits 0.

The two most likely failures:

- **`vade: command not found` in a later step.** The `export` line lasts for one shell. If your tool starts a new shell for each command, put the `export` line in front of every `vade` command.
- **The output has no `export PATH=` text.** The install did not finish. Go back to step 1.

Do not add the line to `~/.profile`, unless the person asks you to.

## Step 3. Download the model

Run `vade models` first. Skip this step when it lists the checkpoint with `Supported: Qwen3.8 Flash-Next FP4-Local`. With no model, it prints `No models in ~/models yet.` and exits 0.

Ask the person first, because the checkpoint is 102.26 GB. The command is:

```sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
```

Always use the [background form](#run-a-long-command) here, so that a lost connection or a tool time limit does not stop the download. The log gets one line per file, and a large file can take a long time:

```sh
nohup sh -c 'vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300; echo "exit $?"' > ~/vade-step.log 2>&1 < /dev/null &
echo $! > ~/vade-step.pid
```

- **Expected output.** `Verified 32 files, 102.26 GB in <folder>`, then `Supported: Qwen3.8 Flash-Next FP4-Local` and a `Next:` line.
- **Exit code.** 0, so the last log line is `exit 0`.
- **Check.** The log has the `Verified 32 files` line and the `Supported:` line.

Do not run the `Next:` line. Go to step 4.

The two most likely failures:

- **The last log line is `exit 1`.** Read the `error:` line above it, then run the same command again. It continues from where it stopped.
- **`Not supported yet` or `Incomplete`.** Run the same command again. If the state stays the same, give the person the output. [Download models](../pull-models.md#support-state) explains each state.

## Step 4. Start the engine

```sh
vade up
```

The first `vade up` finds the model under `~/models`, saves the settings in `~/.vade/config`, checks the model files and starts the engine. For a checkpoint in another folder, set `VADE_MODELS` to that folder before the first `vade up`.

`vade up` ends on its own. It took 67.4 s in the [fresh-install receipt](../../bench/receipts/2026-10-31/fresh-install-2026-10-30.txt), where the install had already checked the model files. We have no timed record of a first `vade up` right after `vade pull`. It waits up to 600 s for the model to load, after it checks the files. Use the [background form](#run-a-long-command) if your tool has a time limit.

- **Expected output.** One line in this form:

  ```text
  Up ready in <time>: http://127.0.0.1:18080, context 32768 tokens, output 8192, 1 Spark.
  ```

  Then the output gives the `vade chat` and `vade down` commands.
- **Exit code.** 0.
- **Check.** `vade doctor --json` prints one `vade.doctor.v1` object with `"result":"pass"` and exits 0.

The line gives the true values. The context can be smaller than 32768 on a Spark with less free memory. The port can differ if 18080 was in use at the first start. Use the address in this line in step 5, and add `/v1` to it.

The two most likely failures:

- **`Model folder ... is unavailable` and a `vade pull` line.** The checkpoint is not where Vade looks. Go back to step 3.
- **`GPU is in use`, or `Port ... is in use`.** Another program holds the GPU or the port. Tell the person. Do not stop it.

## Step 5. Send one API request

The engine serves the OpenAI protocol at `http://127.0.0.1:18080/v1`. It needs no API key on this address. The examples use the default port. If the `Up ready` line names another port, use that port.

```sh
curl --fail-with-body -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp4-local","messages":[{"role":"user","content":"Name the capital of France in one word."}],"max_tokens":16,"enable_thinking":false}'
```

- **Expected output.** One JSON object. Its first choice has `"content":"Paris"` and `"finish_reason":"stop"`.
- **Exit code.** 0.
- **Check.** The JSON has `"content":"Paris"`.

The two most likely failures:

- **`curl: (7) Failed to connect`.** The engine is not running on this port. Run `vade status --json`. Use its `endpoint` field as the address, which already ends in `/v1`. If `phase` is `stopped`, go back to step 4.
- **An HTTP 4xx reply with a JSON error.** Read the `message` field. Compare your request with the [OpenAI API page](../openai-api.md).

The model thinks before it answers unless the request sends `"enable_thinking": false`. The reasoning counts against `max_tokens`.

Python is optional. The curl request above is the proof. Run the Python example only if the person asks for it. Install the `openai` package first:

```sh
pip install openai
```

If `pip` prints `externally-managed-environment`, use a virtual environment: run `python3 -m venv ~/vade-venv`, then `~/vade-venv/bin/pip install openai`. If `pip` or `venv` is missing, skip the Python example and tell the person. Installing them needs sudo.

Save this example as `~/hello.py`, and run it from `$HOME`. Run it with `python3 hello.py`, or with `~/vade-venv/bin/python hello.py` in a virtual environment. It prints `Paris`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="unused")
reply = client.chat.completions.create(
    model="qwen3.8-flash-next-fp4-local",
    messages=[{"role": "user", "content": "Name the capital of France in one word."}],
    extra_body={"enable_thinking": False},
)
print(reply.choices[0].message.content)
```

## Step 6. Check exactness

Exactness means that the same request with the same settings gives the same token IDs every time. `vade verify` tests this on the running engine:

```sh
cd "$HOME" && vade verify
```

It sends one prompt 20 times with drafting off and 20 times with drafting on. It prints progress while it waits, and it ends on its own. We have no timed record of this command. Use the [background form](#run-a-long-command) if your tool has a time limit. Drafting is Vade's speculative decoding: the model's prediction head guesses a few tokens ahead, and the full model checks each guess.

- **Expected output.** One line that starts with `IDENTICAL: 20/20 runs, 1 machine,`.
- **Exit code.** 0.
- **Check.** The exit code is 0, the output has the `IDENTICAL` line, and `grep '"result"' "$HOME/vade-verify.json"` prints a line with `identical`. This file is the receipt of the run. Vade also writes it when a run fails.

The two most likely failures:

- **`DIFFERENT at run ... token ...`, exit 1.** The token IDs differ. Stop. Give the person the output and `vade-verify.json`. Do not retry in a loop.
- **`ERROR: Cannot read local owner state`, or `ERROR: The engine is not running on this Spark`.** No engine runs. Go back to step 4.

## Step 7. Stop the engine

The default is to leave the engine running, so that the person can use it. Run this step only when the person asks you to stop the engine:

```sh
vade down
```

- **Expected output.** `Down done in <time>`, or `Vade is already stopped.` when nothing runs.
- **Exit code.** 0.
- **Check.** `vade status --json` prints `"phase":"stopped"` and exits 0.

The two most likely failures:

- **`Vade is already changing this engine.`** Another Vade command holds the lock. Wait, then run `vade status`.
- **`vade down` cannot confirm which process it started.** Keep the state files. Give the person the output and the [CLI recovery](../../cli/README.md#recovery) section.

A plain `vade up` starts the engine again with the settings saved in `~/.vade/config`.

## Where the receipts are

A receipt is the saved record of a run: its build, its settings and what it measured.

- **`vade-verify.json`.** `vade verify` writes it in the current folder.
- **`vade-bench.json` and `vade-bench.txt`.** `vade bench` writes them in the current folder and replaces any older ones.
- **[bench/receipts/README.md](../../bench/receipts/README.md).** This page lists the records behind the published numbers, newest first.
- **[docs/results.md](../results.md).** This page gives every benchmark cell with its build and settings.

## Two or four Sparks

Linked Sparks need a controller, which is a computer that drives the Sparks over SSH. The controller can be one of the Sparks, or another Linux or macOS computer.

The one-line install is enough for this path. Vade copies itself to each other Spark, so a Spark needs no Rust and no checkout. One command does the rest: `vade up --sparks spark-a,spark-b`.

**Ask the person for these answers:**

- which computer is the controller
- what they type after `ssh` for each Spark, in any order
- whether the link checks in [network setup](../network-setup.md#7-check-every-link) pass

If the link checks do not pass, stop. The network is a job for the administrator. Two FP4-Local Sparks need RDMA.

A clear yes from the person is enough for the link checks. `vade doctor` checks each link again in step L5.

Do the steps below in order. Do not run step 4 on a Spark in this path, because `vade setup` starts the engine.

The examples use two names. `spark-a` is a controller Spark, and `spark-b` is the other Spark. The names can come in any order, because Vade reads the cables. The controller does not need to be first.

### Step L1. Check SSH from the controller

Run this once for each Spark that is not the controller, with the name that the person gave you. On a controller Spark, also run the quoted part with no `ssh`.

```sh
ssh -o BatchMode=yes spark-b 'ulimit -Sl; ulimit -Hl; echo "$HOME"'
```

- **Expected output.** `unlimited` twice, then the home folder.
- **Exit code.** 0.
- **Check.** Each Spark prints `unlimited` twice, and every Spark has the same home folder. RDMA needs the unlimited memory lock in this SSH session.

A controller Spark needs no SSH key for itself. Vade finds it by its name in `--sparks`. Check that name with `ssh -G spark-a | grep -i '^hostname '`. The line must name this computer: `localhost`, a loopback address, the output of `hostname`, or an address that `ip -br address` lists. If it does not, ask the person which name to use.

The most likely failures:

- **`Permission denied` or `Host key verification failed`.** The Spark does not trust the controller's key, or the controller does not know the Spark's host key. Ask the person to fix it. Vade never edits SSH settings.
- **A number in place of `unlimited`.** The Spark is not ready for RDMA. Stop and tell the person. The fix is in [network setup](../network-setup.md).
- **The home folders differ.** Vade uses one folder path on every Spark, so `vade up --sparks` stops before it writes anything. Ask the person to use one account name on all Sparks.

### Step L2. Prepare each Spark

Run [Check C](#before-step-1) on each Spark first, and use the [background form](#run-a-long-command). The command for this step is step 1 of this page. Run it on the controller Spark in the background form as it is. For another Spark, wrap the same form in `ssh`:

```sh
ssh -o BatchMode=yes spark-b 'nohup sh -c "curl -fsSL https://vade.lol/install.sh | sh; echo exit \$?" > ~/vade-step.log 2>&1 < /dev/null & echo $! > ~/vade-step.pid'
```

- **Expected output.** `Install complete. Next steps:`, then the next steps.
- **Exit code.** 0.
- **Check.** The output has `Install complete. Next steps:`.

Then run steps 2 and 3 on each Spark. Each Spark downloads its own copy of the checkpoint, and the downloads can overlap. Ask the person once, and name the total: 204.52 GB for two Sparks, or 409.04 GB for four. `vade pull` checks every file, so no other hash check is needed.

Over SSH, the `export` line must be in the same quoted command as `vade`. For example:

```sh
ssh -o BatchMode=yes spark-b 'export PATH="<the folders that the installer printed>:$PATH"; nohup sh -c "vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300; echo exit \$?" > ~/vade-step.log 2>&1 < /dev/null & echo $! > ~/vade-step.pid'
```

The two most likely failures:

- **`install:` and a `Next:` action.** Follow the action. Do not retry in a loop.
- **A driver or GPU error.** Tell the person. Driver updates are an administrator job.

### Step L3. Get vade on the controller

If the controller is a Spark from step L2, skip this step. Its `vade` comes from the install in `~/.vade`.

On another computer, the controller needs Git, a C linker (`cc`), curl, ssh, rsync and `shasum`. Check them with `for tool in git cc curl ssh rsync shasum; do command -v "$tool"; done`, which prints six paths. It needs Rust 1.99.0 too. If `rustup toolchain list` does not show `1.99.0`, run these commands:

```sh
curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs
sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path
"$HOME/.cargo/bin/rustup" toolchain install 1.99.0 --profile minimal --no-self-update
```

Then run:

```sh
git clone --branch main https://github.com/wappy-x/Vade.git "$HOME/vade"
cd "$HOME/vade"
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
```

Run the later steps in this shell. In a new shell, run the two `export` lines again from `$HOME/vade`.

- **Expected output.** Cargo ends with a `Finished` line.
- **Exit code.** 0 for each command.
- **Check.** `vade --version` prints a line that starts with `vade` and a version number.

The two most likely failures:

- **`fatal: destination path ... already exists`.** A checkout is there. Ask the person whether to use it. Do not replace it.
- **A `toolchain` or `rustup` error.** Rust 1.99.0 is missing. Follow [Install Rust](../INSTALL.md#install-rust).

### Step L4. Preview the start

Run this step and step L5 from any folder. Replace the names with the values from the person.

```sh
vade up --sparks spark-a,spark-b --dry-run
```

A dry run opens no connection, so it shows no cable addresses. On a list with no setup yet, Vade uses `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local` on each Spark.

For the FP8 checkpoint, add `--fp8`. FP8 needs two or four Sparks, and it does not fit on one. These values change for FP8:

- **Download.** The [FP8 install](../install-fp8.md) gives its own `vade pull` line, for step L2.
- **Disk.** Check C needs at least `184G` on each Spark: about 173 GiB for the model and 10 GiB for the install.
- **API request.** Use the model `qwen3.8-flash-next-fp8`. FP8 samples by default, so check `"finish_reason":"stop"`, not the word `Paris`.

- **Expected output.** `Dry run: setup writes nothing and starts nothing.`, the config, then `Setup reads the cable addresses from each Spark when it runs.` and a `Steps:` list.
- **Exit code.** 0.
- **Check.** The config names each Spark.

### Step L5. Run the start

Run the same command without `--dry-run`:

```sh
vade up --sparks spark-a,spark-b
```

Vade checks each Spark over SSH and reads its cables. It prints one line such as `Cable order: spark-a - spark-b.`. Then it writes the config and runs `vade install`, `vade up` and `vade doctor`, so the engine is already running at the end. A setup that exists moves aside, and one line such as `Replacing the one-Spark setup with two Sparks.` says so. We have no timed record of this step.

- **Expected output.** The `Cable order:` line, the output of the three steps, then the line `Setup done.`
- **Exit code.** 0.
- **Check.** `vade status --json` prints `"ready":true` and an `endpoint`. Then send the step 5 request to that `endpoint`. It already ends in `/v1`. The command keeps the default ports.

The most likely failures:

- **`ssh spark-b failed:`.** The line shows what ssh said, and the hint gives the one command to run. Vade never edits SSH settings. Ask the person to run it, then run the same command again.
- **`share no cable`.** Two Sparks share no subnet on a fabric port. Ask the person to check that cable. Do not guess addresses. If the person gives the addresses, add `--link spark-a:spark-b=ADDRESS --link spark-b:spark-a=ADDRESS`.
- **`vade install failed, so setup stopped.`** The same message can name `vade up` or `vade doctor`. Fix the error that the step printed just above it. Then run the same command again.

A step gets no input here. A step that needs an answer stops with an `error:` line and does not wait. To stop the command, send SIGTERM or SIGINT to it, as [Run a long command](#run-a-long-command) shows. Vade passes the stop on to the step that runs, waits for that step and exits 1.

### Step L6. Check exactness

Leave the checkout first, because `vade install` refuses untracked files such as the receipt.

```sh
cd "$HOME" && vade verify
```

- **Expected output.** One line that starts with `IDENTICAL: 20/20 runs, 1 machine,`. Linked Sparks count as one machine, because they form one engine.
- **Exit code.** 0.
- **Check.** The same check as step 6.

The failures are the same as in step 6. In this path, "go back to step 4" means `vade up`.

### Step L7. Stop and start again

The default is to leave the engine running. To restart it, run:

```sh
vade down
vade up
```

- **Expected output of `vade down`.** `Vade is stopped. The engines and the tunnel that vade started are closed.`, or `Vade is already stopped. Nothing to stop.`
- **Expected output of `vade up`.** `Starting <model> on 2 Sparks over RDMA. Waiting for the model to load.`, then `Vade is ready on 2 Sparks (spark-a, spark-b) at http://localhost:18080: context 131072 tokens, drafting depth <depth>.`
- **Exit code.** 0 for each command.
- **Check.** `vade status --json` prints `"ready":true`.

`vade up` alone starts the Sparks of the last setup, with its checkpoint and the default limits. `vade status` names them while the engine is stopped. A config written by hand keeps the old rule: a plain `vade up` starts one FP4-Local Spark, or two FP8 Sparks.

The two most likely failures:

- **`ssh spark-b failed:` or `did not reply to the pre-start check`.** SSH failed or timed out. Nothing started. Follow the hint, then run `vade up` again.
- **`refused --context` or `Not enough memory`.** The engine has too little memory. Add `--context 16384` to `vade up`. If it refuses again, use `--context 8192`, then `--context 4096`.

## Build from source

Use this path when step 1 cannot install the bundle. It needs Git and `cc`. The script installs the pinned Rust toolchains if they are missing. Then it builds `vade`, the GPU kernels, the engine and chat.

```sh
git clone --branch main https://github.com/wappy-x/Vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
```

- **Expected output.** `Install complete. Next steps:`, then the next steps.
- **Exit code.** 0 for each command.
- **Check.** The output has an `export PATH=` line. Continue with step 2.

If `git clone` prints `fatal: destination path ... already exists`, a checkout is there. Ask the person whether to use it. Do not replace it.

## Use the engine

| Task | Command | Result |
| --- | --- | --- |
| Check every part | `vade doctor --json` | JSON, exit 0 on a pass |
| Show what runs | `vade status --json` | JSON |
| Chat (needs a terminal) | `vade chat` | Opens chat |
| Read a file as Markdown | `vade read FILE` | Markdown, exit 0 |
| List the chat tools | `vade tools` | A list, exit 0 |
| Show recent engine logs | `vade logs` | Log text, exit 0 |

`vade doctor --json` prints one `vade.doctor.v1` object. `result` is `pass` or `fail`. Each entry in `checks` has `name`, `state` (`ok`, `fail` or `skip`), `detail` and `next`.

`vade status --json` prints one `vade.status.v1` object. It exits 0 when `phase` is `stopped` or the engine is ready. It exits 1 on an error or an unhealthy engine. `endpoint` is the API address while the engine runs.

## Connect Claude Code or Codex

`vade chat` is Vade's own agent. To use Claude Code or Codex with the running engine instead, run the example script from a clone of the repository. A one-line install has no clone, so make one first:

```sh
git clone --branch main https://github.com/wappy-x/Vade.git "$HOME/vade"
cd "$HOME/vade"
```

Each script reads the model and its limits from `/v1/capabilities` first. Without `--interactive`, a script asks one question and prints the answer, so you can check the connection first. The check passes when the script exits 0 and prints an answer.

| Client | Check | Open it |
| --- | --- | --- |
| Claude Code | `sh examples/claude-code.sh` | `sh examples/claude-code.sh --interactive` |
| Codex | `sh examples/codex.sh` | `sh examples/codex.sh --interactive` |

Claude Code needs `claude` on `PATH`. Codex needs `codex` on `PATH` and `vade-harness`. To build `vade-harness`, install Rust 1.99.0 with the commands in step L3, because a one-line install has no Rust. Then run:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --locked --manifest-path harness/Cargo.toml --bins
```

The installer stages its own copy of `vade-harness` for `vade chat`. It does not build `harness/target/release`, which the Codex script uses.

Both scripts use `http://127.0.0.1:18080`. Set `VADE_API_URL` when `vade status --json` shows another `endpoint`. Leave out the `/v1` at the end.

The Claude Code script runs `claude --bare`, without hooks, plugins, memory or `CLAUDE.md`. Without `--bare`, Claude Code sends a `system` role inside `messages`, and the engine answers HTTP 400. For full Claude Code, put the gateway in front of the engine. The [client setup guide](../../harness/docs/passthrough-matrix.md#claude-code) gives the commands. It also covers Hermes, OpenCode and Pi.

The Codex script writes a new Codex home with a read-only sandbox and never touches `~/.codex`. [Examples](../../examples/README.md#claude-code) shows the output of both scripts.

## Update Vade

**After a one-line install.** This case also covers the installer's own source build and the `--source` flag of step L2. Stop the engine, then run the same install command again:

```sh
vade down
curl -fsSL https://vade.lol/install.sh | sh
```

Run the export line that the installer prints last, because the folder can change with the version. Then run `vade up`. The checks are the same as in steps 7, 1 and 4.

**After a build from source in `$HOME/vade` on a Spark.**

```sh
vade down
git -C "$HOME/vade" pull
vade update
vade up
```

`vade update` refuses to run while the engine runs. A plain `vade up` starts again with the settings saved in `~/.vade/config`. The check is the same as in step 4.

**From a controller.** Leave the engine running and run:

```sh
vade update
```

It pulls each source checkout and stages the new build. Then it asks before a restart. Without a terminal, it keeps the engine running and prints `Running engine kept. Restart when ready:` and two commands: `vade down`, and `vade up` with this engine's Sparks and limits. Run those two commands as they are. Do not use a plain `vade up` instead.

## Where the files are

| Path | What it holds | May you change it? |
| --- | --- | --- |
| `~/.vade/config` | Local defaults on a Spark | Only with `vade down` first, then `vade up` with new flags |
| `~/.vade/stage` | Staged engine, GPU kernels and chat | No. `vade install` and `vade update` manage it |
| `~/.vade/state` | Records of what Vade started, and logs | Never |
| `~/.vade/projects` | Saved chat sessions | Use `vade sessions` |
| The path in `vade config` | The controller config | Ask the person |
| `~/.vade/controller` | Stage and state of a controller that setup wrote | Never |

`VADE_HOME` moves `~/.vade`. `VADE_CONFIG` names a controller config. Keep both the same for every command in one setup.

## What to do on each kind of error

| You see | Cause | Safe action |
| --- | --- | --- |
| `vade: command not found` | The `export` line ran in another shell, or the install did not finish | Step 2 |
| `Up ready` is missing after `vade up` | The start failed | Read the `error:` line, then this table |
| `Vade is not set up yet` | No setup for this user | Step 1 |
| `This computer is not a Spark, and it has no controller config yet.` | `vade install` on a controller before setup | Step L4 |
| `This command runs only on a DGX Spark (Linux aarch64)` | A local command on another computer | Run it on the Spark, or use a controller |
| `Standard input is not a terminal, so setup cannot ask` | `vade setup` without `--yes` | Add `--yes` and the flags it names |
| `share no cable` | Two Sparks share no subnet on a fabric port | Ask the person to check that cable. Do not guess addresses |
| `Add --hosts` or `add --link FROM:TO=ADDRESS` | A required answer is missing | Ask the person. Do not guess addresses |
| A `toolchain` or `rustup` hint | A pinned Rust toolchain is missing | Run the `rustup` command in the hint |
| `source has local changes` | The checkout has uncommitted or untracked files | Show `git status` to the person |
| `GPU is in use` | Another program uses the GPU | Tell the person. Do not stop it |
| `needs an NVIDIA GB10` or `CUDA 13.0 or newer` | Not a Spark, or the NVIDIA driver is too old | Tell the person. Driver updates are an administrator job |
| `missing cc` | A build from source needs a C compiler, and the Spark has none | Tell the person. An administrator installs it |
| `no checksum list for this checkpoint` | No `.sha256` list beside the model folder | Tell the person. [Check the files](../INSTALL.md#check-the-files) shows the check |
| `Cannot reach ... over SSH` | A Spark is off, off the network or asks for a password | Step L1, then run the command again |
| `Model folder ... is unavailable` | No checkpoint where Vade looks | Step 3 |
| A doctor row `model files` fails | The checkpoint is missing or incomplete | Step 3 |
| `refused --context` or `Not enough memory` | The engine has too little memory | Run `vade up` with `--context` 16384, then 8192, then 4096 |
| `Port ... is in use` | Another program has the port | Tell the person |
| `Vade is already changing this engine.` | Another Vade command holds the lock | Wait, then run `vade status` |
| `did not reply to the pre-start check` | SSH failed or timed out | Step L1, then run the command again |
| `Vade hit an internal error` | A bug in Vade | Stop. Give the person the output. Do not retry in a loop |

The [first-run problems](../INSTALL.md#first-run-problems) table lists more messages.
