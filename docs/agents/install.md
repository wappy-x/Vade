# Install page for agents

This page is for a program that installs and starts Vade on one DGX Spark with no person at the keyboard. It lists each command, what it prints for a machine, its exit codes, and one action for each failure. The [agent quickstart](quickstart.md) has the long form, linked Sparks and the client setup.

No command on this page asks a question. None needs a terminal. None needs sudo.

## The path

| Step | Command | Works when |
| --- | --- | --- |
| Preview | `sh vade-install.sh --check --json` | exit 0 and the last line has `"ok":true` |
| Install | `sh vade-install.sh --json` | exit 0 and the last line has `"ok":true` |
| Model | `vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300` | exit 0 |
| Start | `vade up` | exit 0 |
| Check | `vade status --json` | exit 0 and `"ready":true` |
| Stop | `vade down` | exit 0 |
| Remove | `sh vade-install.sh --uninstall --json` | exit 0 and the last line has `"ok":true` |

Get the installer as a file first, so that a failed download cannot look like a success:

```sh
curl -fsSL https://vade.lol/install.sh -o vade-install.sh
```

The piped form `curl ... | sh` exits 0 when the download fails, because `sh` then runs an empty script. If you must use it, treat "no result line" as a failure.

After the install, call `vade` by the path in the result (`"vade"`), or put the folder of `"path_line"` on `PATH`. Ask the person before you run `vade pull`: it downloads 103 GB.

## What the installer prints with `--json`

Standard output holds JSON lines and nothing else. Every other line goes to standard error. Read the last line of standard output: it has `"event":"result"`.

```json
{"schema":"vade.install.v1","ok":true,"error":null,"next":"/home/you/.vade/bin/vade up","event":"result","action":"install","check":false,"route":"prebuilt","build":"<40 hex digits>","home":"/home/you/.vade","vade":"/home/you/.vade/bin/vade","path_line":"export PATH=\"$HOME/.vade/bin:$PATH\"","on_login_path":true,"seconds":61}
```

```json
{"schema":"vade.install.v1","ok":false,"error":{"code":"gpu_in_use","message":"the GPU is in use."},"next":"stop that workload through the launcher that started it, then retry.","event":"result","action":"install","check":false,"seconds":1}
```

- `ok` is the answer. `error.code` is a fixed word. `error.message` and `next` are the sentences a person sees.
- `route` is `prebuilt` (the downloaded bundle) or `source` (a build on this Spark). `build` is the commit of the installed `vade`, or `null`.
- `on_login_path` says whether a new login shell finds `vade` without the `path_line`.
- Lines before the last have `"event":"step"` and name the step that starts: `check`, `download`, `clone`, `build`, `install`, `uninstall`.
- With `--check`, the result has `"check":true` and `would`, the plan in words. Nothing is written.
- Fields are only added. A changed shape gets a new schema name.

## Exit codes

| Command | 0 | 1 |
| --- | --- | --- |
| `vade-install.sh --json` | installed, `"ok":true` | not installed, `"ok":false` |
| `vade-install.sh --check --json` | the preview ran; read `ok` to know if the install would work | never |
| `vade-install.sh --uninstall --json` | removed, or nothing was there | nothing was removed; `error` says why |
| Any of the three above, stopped by a signal | never | never: 129 for HUP, 130 for INT, 143 for TERM; the last line is still the result, with `interrupted` |
| `vade up` | the engine is ready and answers | not started; the `error:` line says why |
| `vade status --json` | ready, or stopped: read `ready` and `phase` | an error, or an engine that is not healthy |
| `vade doctor --json` | every check passed or was skipped; a stopped engine is a skip, so read `vade status --json` to know if it answers | a check failed; its `next` field says what to do |
| `vade down` | stopped, or nothing ran | the stop failed; keep the state folder |

## Run it twice

A second install run is safe. It checks the Spark again, reuses the files that match and builds nothing. A second `vade up` attaches to the running engine. A second `vade down` and a second uninstall find nothing to do and exit 0.

## Long commands and no terminal

The install, `vade pull` and `vade up` can run longer than a tool's time limit. Start them detached and read the files:

```sh
nohup sh vade-install.sh --json > install.jsonl 2> install.log < /dev/null &
```

A signal sent to the installer alone waits until the running step (a download or a build) ends; then the result line says `interrupted`. To stop it at once, signal its whole process group. Only do that when you started it with `setsid`, so the installer leads its own group. Run these lines in a shell without job control (a script, or `sh -c '...'`), as an agent's tool usually does; in an interactive shell `$!` names a short-lived `setsid` process instead, and the `kill` reaches nothing:

```sh
setsid nohup sh vade-install.sh --json > install.jsonl 2> install.log < /dev/null &
echo $! > install.pid
kill -TERM -"$(cat install.pid)"
```

Without `setsid`, the installer shares the process group of the shell that started it, and `kill -TERM -<pgid>` stops that shell too. Never use `kill -9`: it leaves no result line and leaves the lock folder `.install-lock` in the Vade home.

Each line reaches `install.log` when it is printed, also with no terminal. A long step writes a `still ...` line with the seconds so far at least every 30 s. The run has ended when the last line of `install.jsonl` has `"event":"result"`.

## Failures and the one action for each

| `error.code` or message | Cause | Action |
| --- | --- | --- |
| `not_a_spark` | Not Linux `aarch64`, or the GPU is not a GB10 | Stop. Tell the person. |
| `driver_too_old`, `driver_failed` | The NVIDIA driver is older than CUDA 13.0, or `nvidia-smi` fails | Stop. Tell the person: a driver update needs an administrator. |
| `gpu_in_use` | A compute process holds the GPU | Run `vade status --json`. If Vade runs, `vade down` and retry. Else tell the person. Never stop another program. |
| `no_disk_space` | Under 9 GB free where the Vade home goes | Tell the person, or set `VADE_HOME` to a folder on a larger disk and retry. |
| `missing_tool` | `curl`, `nvidia-smi`, or for a source build `cc`, `git` or Rust | Tell the person which tool the message names. |
| `bad_download` | The bundle does not match its published hash | Run the same command again, at most three times. |
| `clone_failed`, `build_failed`, `install_failed` | The source route or `vade install` failed | Read the lines above the result in the log, run `vade doctor --json`, then retry once. |
| `engine_running` (uninstall) | `vade down` failed, or a record of a running engine is left | Run `vade down`, then the uninstall again. Never delete the state folder by hand. |
| `bad_arguments` | A flag this installer does not know | Fix the command line. Nothing ran. |
| `bad_home`, `bad_config_home` | `VADE_HOME` (or `XDG_CONFIG_HOME` for `--uninstall --all`) is not a full path of two or more parts, holds `.`, `..` or a link, or is the home folder; or the folder is not a Vade home (no `.vade-home` mark), or it already holds other files at install | Unset it, or set it to the real folder (`pwd -P` prints it), or choose a new folder. Nothing was removed. Never write the mark yourself; ask the person. |
| `busy` | Another install or uninstall holds `.install-lock` in the Vade home | Wait for it to end and retry. If none runs, tell the person; the empty folder can then go. |
| `interrupted` | A signal stopped the installer | Run the same command again. |
| `stopped` | The installer ended early for another reason | Read the lines above the result in the log, then retry once. |
| `vade up` prints `No models yet` | The model is not on this Spark | Ask the person, then run `vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300`. |
| `vade up` prints `refused --context` or `Not enough memory` | Too little free memory | Retry with `--context 16384`, then `8192`, then `4096`. |

## What an agent cannot find out alone

- Whether the person allows the 103 GB download.
- For linked Sparks: what the person types after `ssh` for each Spark. Never guess a name or an address.

## Remove

`--uninstall` runs `vade down`, then removes the programs, builds, caches and state in the Vade home and the script in `~/.local/bin`. It keeps the model files, the Rust toolchains, the person's chat files and the controller config. `--uninstall --all` removes the chat files and the controller config too. With another `VADE_HOME` than `~/.vade`, the uninstall keeps the script in `~/.local/bin` and the controller config, also with `--all`. `--uninstall --check` lists what would go and removes nothing. Run it on each Spark.

The install marks the Vade home with a file `.vade-home`, and refuses a folder that already holds other files. The uninstall works only on a Vade home: a folder with that mark, a home from an older installer (known by its `stage/install.conf`), or a folder that holds only what a stopped install left (Vade bundles in `bundles`, Vade source clones in `source.????????` folders, and `.install-lock`). The install takes such a folder over. From `bin` it removes only `bin/vade` and `bin/vade.new`.
