# What `vade` looks like

Real output, so you can see the [output contract](ux.md) without a Spark.
Everything below was captured on 2026-10-25 on a Mac from this branch's own
debug builds of `vade` and `vade-harness`, with an empty private `HOME` (a
temporary directory, shown here as `/tmp/vade-example`) and no network. The
`vade` commands ran with `TERM=dumb` (so no colour); the `[exit N]` lines are
their exit codes. Chat needs a real terminal (`TERM=dumb` turns its raw mode
off), so it ran differently, as described under [Chat](#chat).

## Chat

One real exchange: `vade-harness agent` (what `vade chat` opens) driven
through a pseudo-terminal with `TERM=xterm-256color` and `COLUMNS`/`LINES` set
to the screen size, against a small offline mock server that streams a short
thought and an answer. The screens were rebuilt from the raw terminal bytes
with a minimal terminal model, then colour was dropped. The counts and
speeds are the mock's, not the engine's: the mock reports 790 prompt and 312
answer tokens for a few short chunks, so `227 tok/s` here says nothing about a
Spark. `VADE_CHAT_SETUP` was set the way `vade chat` sets it.

In colour, the `▌` bar and your words are bold cyan, the thought row is dim
italic, and the footer and status strip are dim.

At 100 columns, after the answer:

```text

Sun 25 Oct 2026
vade agent · / for commands, /help for a guide

▌ Why is the sky blue?

▸ thinking 12:10:36–12:10:38 · ≈42 tokens · 1.5s · Ctrl+T expand

Sunlight is a mix of colours. Air molecules scatter short wavelengths far more than long ones
(about 1/λ⁴), so blue light is redirected across the whole sky and reaches your eye from every
direction.

At sunset the light crosses more air, the blue is scattered away first, and the reds and oranges
are what is left.
╰ 12:10:38 · 1.5s · 312 tokens out · 227 tok/s · prefill ≈253.9k tok/s · first token 0.0s
qwen3.8-flash-next-fp4-local · ctx 1.1k/32.8k · in 790 · out 312 · 227 tok/s · 1 Spark, drafting on
›
```

At 100 columns, while the answer streams (the strip shows the live state,
elapsed time and tokens per second):

```text

Sun 25 Oct 2026
vade agent · / for commands, /help for a guide

▌ Why is the sky blue?

▸ thinking 12:10:52 · ≈42 tokens · Ctrl+T expand

Sunlight is a mix of colours. Air molecules scatter short wavelengths far more than long ones
(about 1/λ⁴), so blue light is redirected across the whole sky
⠹ writing · 1.0s · ≈140 tok/s · 1 Spark, drafting on · round 0/60 · 1s/20m · Esc cancels
›
```

At 64 columns, two Sparks with drafting off. The answer wraps between words,
the hint on the thought row drops whole, and the strip keeps the model name
and the setup:

```text

Sun 25 Oct 2026
vade agent · / for commands, /help for a guide

▌ Why is the sky blue?

▸ thinking 12:10:55–12:10:57 · ≈42 tokens · 1.5s

Sunlight is a mix of colours. Air molecules scatter short
wavelengths far more than long ones (about 1/λ⁴), so blue light
is redirected across the whole sky and reaches your eye from
every direction.

At sunset the light crosses more air, the blue is scattered
away first, and the reds and oranges are what is left.
╰ 12:10:57 · 1.5s · 312 tokens out · 225 tok/s
qwen3.8-flash-next-fp4-local · 2 Sparks, drafting off
›
```

## Help

```text
$ vade --help
Vade runs Qwen3.8 Flash-Next FP4-Local on your DGX Spark.

Examples:
  vade launch  Start on this Spark, then chat
  vade status  See what is running
  vade doctor  Check the setup, one line per check
  vade down    Stop what vade started

Usage: vade <COMMAND> [OPTIONS]

Start and stop:
  launch   Install if missing, start on this Spark, then chat
  up       Start without chat (this Spark, or Sparks from a controller)
  down     Stop what vade started

Use:
  chat     Chat with the running engine

Look:
  status   Show what is running and whether it is healthy
  config   Show which setup is selected and what it holds
  doctor   Check the setup: one line per check, then PASS or FAIL
  detect   Show this Spark's machine, memory, disk, driver and model
  logs     Show recent engine logs

Prove:
  verify   Check exact token IDs across runs, drafting ON and OFF
  bench    Measure a running server from this client
  test     Run the build, determinism and long-context gate checks

Maintain:
  install  Save local defaults and install the engine and chat
  update   Refresh the installed engine and chat
  run      Run the separate legacy Qwen3-8B BF16 workflow
  version  Print the CLI version

Options:
  --json               JSON output for status, doctor, config, detect, version
  --dry-run            Preview a lifecycle command; nothing changes
  --api-key-file PATH  Credential for chat, doctor and local run
  -h, --help           Show help
  -V, --version        Print the CLI version

Environment:
  VADE_HOME          Local install root and saved defaults (default ~/.vade)
  VADE_CONFIG        Controller config file; selects the controller (see Notes)
  VADE_API_KEY_FILE  Credential file when --api-key-file is not given
  NO_COLOR           Plain output without colour

Notes:
  - Explicit VADE_CONFIG selects the controller for up, install, chat, status,
    logs, down and doctor, even after local setup; keep it set to the same file
    throughout that lifecycle. Without VADE_CONFIG, saved local setup wins over
    the user config directory.
  - Global flags such as --dry-run before bench are unsupported.

Learn more: vade <command> --help
[exit 0]
```

```text
$ vade status --help
Show the engine, machines, build, settings and health.

Examples:
  vade status
  vade status --json

Usage: vade status [--json] [--dry-run]

Options:
  --json     Print one vade.status.v1 object; exit 0 for stopped or ready, 1 for
             errors or an unhealthy state
  --dry-run  Read saved state only; unchecked values are null

Notes:
  - Checks share a 15-second deadline.
  - Explicit VADE_CONFIG selects the controller; otherwise generated local setup
    wins. Keep the same VADE_CONFIG and VADE_HOME from start through chat,
    status, logs and down.
[exit 0]
```

## Doctor with nothing set up

There is no config and no Spark here, so the config row fails and names the
next step for both routes; the local tools still get checked.

```text
$ vade doctor
FAIL  config                Config is missing: "/tmp/vade-example/Library/Application Support/vade/config".
      next: On a Spark, run vade launch; for a controller, copy cli/examples/fp4-local.conf, edit it and set VADE_CONFIG (INSTALL.md).
ok    ssh                   found on this computer
ok    rsync                 found on this computer
ok    curl                  found on this computer
FAIL  3 ok, 1 failed, 0 skipped in 178 ms
error: 1 doctor check(s) failed.
 hint: Run each next: command above, then vade doctor.
[exit 1]
```

The same as JSON, for scripts (nothing on stderr):

```text
$ vade doctor --json
{"schema":"vade.doctor.v1","mode":"none","dry_run":false,"result":"fail","ok":3,"failed":1,"skipped":0,"planned":0,"checks":[{"name":"config","state":"fail","detail":"Config is missing: \"/tmp/vade-example/Library/Application Support/vade/config\".","next":"On a Spark, run vade launch; for a controller, copy cli/examples/fp4-local.conf, edit it and set VADE_CONFIG (INSTALL.md)."},{"name":"ssh","state":"ok","detail":"found on this computer","next":null},{"name":"rsync","state":"ok","detail":"found on this computer","next":null},{"name":"curl","state":"ok","detail":"found on this computer","next":null}],"elapsed_ms":172.420292}
[exit 1]
```

## Doctor dry run from a controller

With `VADE_CONFIG` pointing at a copy of
[`cli/examples/fp4-local.conf`](../examples/fp4-local.conf) (one Spark named
`spark-a`). A dry run connects nowhere and lists what it would check:

```text
$ VADE_CONFIG=pair.conf vade doctor --dry-run
Dry run. No commands run, files written, or connections opened.
plan  controller tools      rustc, cargo, rsync, curl, ssh -V
plan  spark-a ssh           ssh spark-a-ssh-alias runs one read-only check script
plan  spark-a nvidia-smi    read by that script
plan  spark-a disk          read by that script
plan  spark-a free memory   read by that script
plan  spark-a model files   read by that script
skip  fabric                one Spark; no fabric
plan  port 18080            free on this computer, or held by the vade tunnel
plan  engine health         ranks, tunnel, /health and two small harness turns
PLAN  0 ok, 0 failed, 1 skipped, 8 planned; dry run, nothing checked
[exit 0]
```

## Which setup is selected

```text
$ vade config
Setup  none yet
  On a Spark: vade launch   From a controller: set VADE_CONFIG (INSTALL.md)
[exit 0]
```

```text
$ VADE_CONFIG=pair.conf vade config
Setup   controller (drives Sparks over SSH)
File    /tmp/vade-example/pair.conf
Model   qwen3.8-flash-next-fp4-local (nvidia/Qwen3.8-Flash-Next-FP4-Local at b9ce18a6887e18c089fdea8509a33e63429a3300)
Sparks  1: spark-a
Port    18080 on this computer
Build   /path/to/vade/engine
State   /tmp/vade-example/.vade-state
[exit 0]
```

## Errors

```text
$ vade frobnicate
error: Unknown command "frobnicate".
 hint: Run vade --help to see the commands.
[exit 1]
```

```text
$ vade up --mtp 9
error: Set --mtp to a whole number from 0 to 3.
[exit 1]
```

## Version as JSON

```text
$ vade --version --json
{"schema":"vade.version.v1","version":"0.1.0"}
[exit 0]
```
