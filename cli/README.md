# vade

`vade` is the command you use to run nvidia/Qwen3.8-Flash-Next-FP4-Local on
your DGX Spark. You start the engine, chat with it, and stop it when you are
done. It is one binary, written in Rust (2024 edition) with only the standard
library.

- **One Spark:** read [Install](#install),
  [Start the engine on your Spark](#start-the-engine-on-your-spark) and
  [Chat, check and stop](#chat-check-and-stop).
- **Two or more Sparks:** read
  [Multi-Spark source setup](#multi-spark-source-setup) after INSTALL's
  [two and more Sparks](../docs/INSTALL.md#two-and-more-sparks).
- **Something broke:** see [Recovery](#recovery).

Later sections cover checking and measuring the engine, the legacy Qwen3-8B
workflow and the gate commands.

This page explains new terms where they first come up. The
[glossary](../docs/GLOSSARY.md) covers the rest.

## What each command does

<div align="center">

| Command | What it does |
| :---: | :---: |
| `vade` | Shows a short status and the next command. On a Spark, that is the same output as `vade status`. From a controller, it lists four choices: Start, Chat, Stop and Check (which runs `vade doctor`) |
| `vade launch` | Starts the engine on the Spark itself |
| `vade up` | From a controller with a config file, runs the model on one or more Sparks. On the Spark itself, it also starts the local engine |
| `vade chat` | Opens chat with the running engine. Add `--resume ID\|NAME\|last` to continue a saved session, or `--name NAME` to name this one |
| `vade sessions` | Lists, exports or deletes saved chat sessions. See [saved chat sessions](../docs/sessions.md) |
| `vade read` | Prints a local file as Markdown with a header line: text, HTML, CSV, RTF, PDF, DOCX, XLSX, PPTX, EPUB, OpenDocument, data and source files, and image sizes. It needs no engine. See [reading files](../docs/reading-files.md) |
| `vade tools` | Lists the tools chat offers the model: the native workspace and memory tools, then the tools of your MCP servers |
| `vade mcp` | Adds, lists, tests, trusts or removes MCP servers for chat. See [MCP tools in chat](../docs/mcp.md) |
| `vade status` | Shows what is running. Add `--json` for scripts |
| `vade config` | Shows which setup the other commands will use (local, controller, legacy or none) and what it holds. Add `--json` for scripts |
| `vade verify` | Runs the same greedy prompt with drafting off and on, and checks that every run returns the same token IDs byte for byte |
| `vade bench` | Measures the wait before output starts and the answer speed |
| `vade doctor` | Prints one line per check, then PASS or FAIL. On the Spark itself, checks the config, `VADE_HOME`, the driver, memory, disk, model, installation, port and engine health. From a controller, checks local tools, then SSH, `nvidia-smi`, GPU use (the busy rule `vade up` applies), disk, memory, model files and the staged build hashes on the active Sparks or the default `vade up` selection (and the pinned Rust toolchains when no `prebuilt_dir` is set), then the fabric, port and engine. Add `--json` for scripts |
| `vade logs` | Shows recent engine logs |
| `vade pull` | Downloads a model from the Hugging Face Hub with resume, size and hash checks and a progress line. See [Download models](../docs/pull-models.md) |
| `vade models` | Lists the models under `~/models` with their size, source and which command can serve them. Add `--json` for scripts |
| `vade detect` | Shows this Spark's memory, disk, driver and model, and changes nothing. Add `--json` for scripts |
| `vade install` | Saves local defaults and installs the engine and chat on this Spark. With `VADE_CONFIG` set, installs on your Sparks from a controller |
| `vade update` | On the Spark itself, restages the current bundle or rebuilds the current source. From a controller, it first updates clean Git checkouts, then rebuilds and restages |
| `vade down` | Stops the engine you own |
| `vade test` | Runs the pass-or-fail checks a build has to clear (the gate checks): build, determinism (the same tokens, with the same settings, on every run), long-context, recurrent-state (GDN) and Unix-flags ABI checks. See [Gate commands](#gate-commands) |
| `vade run qwen3-8b` | A separate, older workflow for the legacy model |
| `vade version` | Prints the CLI version (also `vade --version` or `vade -V`) |

</div>

"The engine you own" is the one Vade started for you. Vade keeps private
records of what it starts, called ownership records, and it stops only what
those records name.

Some commands behave differently on a controller: a computer that drives one or
more Sparks from a config file. It can be a separate Linux or macOS computer, or
a Spark acting as its own controller. When you work on the Spark itself without
a config file, you do not need a controller at all.
[Multi-Spark source setup](#multi-spark-source-setup) explains how to set one
up.

`vade test` has its own [gate guide](docs/test.md) and
[evaluation guide](evals.md).

What every command prints, its exit codes and the help layout are fixed in
the [output contract](docs/ux.md).

## Install

To install `vade` on your Spark, follow [INSTALL.md](../docs/INSTALL.md), the current
FP4-Local route. Other ways in are under [Install from source or a
bundle](#install-from-source-or-a-bundle).

## Start the engine on your Spark

On the Spark itself, with no `VADE_CONFIG` set, run `vade launch` (or
`vade up`). It finds the checkpoint under `~/models`, installs, and serves it
locally. The [on-Spark first-run guide](../harness/docs/first-run.md) walks
through it. Add `--dry-run` to see what it would do, the default limits and
any limits you passed, without changing any files, processes or connections.

FP4-Local is the default, so it needs no flag. `--fp4-local` names it if you
want to be explicit. The FP8 model needs `--fp8`, two or more Sparks and a
controller (see [Multi-Spark source setup](#multi-spark-source-setup)).

MTP (multi-token prediction) is the model's built-in drafting: the model
proposes a few tokens ahead, and the engine checks each one before keeping it.
The engine picks its own MTP setting unless you override it.

One Spark is enough for FP4-Local, but Vade can also spread the model across
several. Each Spark then runs one engine process, called a rank, and FP4-Local
supports 1 to 24 ranks. To do that, run `vade up` from a controller and follow
the [FP4-Local launcher guide](docs/fp4-local.md).

### Context and output limits

The context is the most tokens the model considers at once: the prompt plus the
answer so far. By default the context is 32768 tokens and each answer can be up
to 8192. Those go lower only when the model requires it, or when the engine's
memory check refuses the default or saved context. In that case startup tries
the smaller sizes from 32768, 16384, 8192 and 4096. It saves the first one that
starts and tells you which size it selected.

You can raise the limits with `--context` and `--output`. Stop the engine first
with `vade down`. `--context 262144` selects the largest context. The engine
still decides how many chats can run at once from the available memory.
Output is capped at 16384 tokens and at the configured context. If the engine
refuses a `--context` you passed, you get an error with its memory message.

### Chats at once and the prompt cache

On the Spark itself, the engine works out from the context and its memory how
many chats it can run at once.

The prompt cache is on by default: requests that send `cache:true` or a
`prompt_cache_key` keep the start of their prompt cached between turns. Set
`VADE_FLASHNEXT_PREFIX_CACHE=0` before `vade launch` to turn it off. The cache
takes its own memory room at startup. See
[the prefix cache](../docs/INSTALL.md#the-prefix-cache).

The local launcher sizes both for you, so it refuses `--chats` and any
`--cache-bytes` above zero before it saves anything. `--cache-bytes 0` is
accepted: it is saved and keeps the cache off even when the variable is set.
The controller's sizing flags are under
[Replies, memory and cache](#replies-memory-and-cache).

## Chat, check and stop

`vade launch` opens chat as soon as the engine is ready. `vade up` starts the
engine and returns. Type `/exit` to leave chat. The engine keeps running. From
any terminal on the Spark, `vade chat` talks to the model, `vade status` shows
what is running, and `vade down` stops it when you are done.

If something looks wrong, run `vade doctor`. It checks the driver, resources,
model, installation and engine health. `vade chat` and `vade verify` find the
engine on the saved launcher port, normally 18080.

Chat opens the coding agent in the current folder. When the context is below
8192 (for example, after startup fell back to 4096), `vade chat` opens plain text chat
instead. On Linux the agent reads, searches and edits files but cannot run
shell commands or checks, and a local launch says so before chat opens. Run
builds and tests in your own terminal.

To pick up a new bundle or source build, run `vade update`. It restages the
current bundle or rebuilds the current source, then tells you the restart
command.

The installer writes one small script, `~/.vade/bin/vade`, that runs the build
it just installed. The `export` line that the installer prints puts that folder
on `PATH`, and that line stays the same for every version. For a newer Vade on
a Spark, run the install command again: `curl -fsSL https://vade.lol/install.sh | sh`.

`vade version` (also `vade --version` or `vade -V`) prints the CLI package
version and the short commit of this build, such as `vade 0.1.0 (f7946e5)`. It
says `(unknown commit)` for a build with no commit, and it adds `modified` for a
build from a tree with uncommitted changes. `--json` gives the same facts as
`sha` (40 digits, or null) and `modified`. It does not load your configuration
or connect to a machine. Include it when you report an installation or launcher
problem.

Launcher errors name the command or option to fix. Invalid commands or options
fail before Vade reads your config or touches the network.

## Where Vade keeps your files

Local state lives in `~/.vade`, or in `VADE_HOME` when that is set.

Chat history never goes into your project. `vade chat` and `vade launch` (and the
`vade run qwen3-8b` chat) save your chat history and the input you typed in a
private folder under `~/.vade/projects/NAME-HASH/` (or `$VADE_HOME/projects/`).
Each project folder gets its own private folder. Its name has a hash of the
project's absolute path, and a `workspace` file inside names the project
folder. Because of this, git never sees your conversations in the folder where
you run `vade chat`.

If you want the old in-project location back, set `VADE_CHAT_IN_PROJECT=1`:

- State then goes to `./.vade/`, which gets a `.gitignore` containing `*` (only
  `.vade/check` stays visible to git).
- The harness (`vade-harness`, the chat and coding-agent program behind
  `vade chat`) has matching flags: `--vade-home DIR` and `--state-in-project`.
- Chats that older versions saved in `./.vade` stay there. Resume them with
  `VADE_CHAT_IN_PROJECT=1`.

Each agent chat is a saved session. Direct text chat saves none.
`vade sessions` lists the sessions of the current folder, and
`vade chat --resume last` continues the newest one. See
[saved chat sessions](../docs/sessions.md).

## Install from source or a bundle

You can install from source with a Git clone (recommended) or a downloaded
source archive (GitHub's Download ZIP or `git archive`). The archive records its
commit as the build id in `.vade-build-id`. An archive without that id, or one
edited after download, builds as `unknown` and prints a notice.

A source archive is not a launcher bundle, which holds ready-made programs. To
stage those instead on the Spark itself, set `VADE_BUNDLE` to an existing local
launcher bundle, then run `vade install` with no controller config selected
(no `VADE_CONFIG`, and no default controller config file). Builds of main come
from dl.vade.lol, and GitHub Releases holds only the old v0.1.0 bundle with its
`SHA256SUMS`. [INSTALL.md](../docs/INSTALL.md) shows the download. A controller can also use a
launcher bundle when its engine and CLI revisions match the source checkout.
Set `prebuilt_dir` in the controller config. See
[Prebuilt bundles](#prebuilt-bundles).

## Verify exactness

Exactness means that the same request, with the same settings, gives the same
token IDs every time. `vade verify` checks this on a running engine that
supports drafting (MTP, described above).

Here is what a run does:

- By default it runs the same bundled prompt 20 times. Each run is an on/off
  pair, one request with drafting on and one with it off, so it makes 40
  generation requests per engine.
- Every response must match the first response's token IDs byte for byte,
  including across machines when you pick a second engine.
- It uses explicit greedy sampling (always the most likely next token), turns
  the prompt cache off, and checks 128 generated tokens by default.
- It keeps generating past end-of-answer tokens so it can check that fixed
  length.

The bundled `short`, `2k` and `8k` prompts need no downloads. The longer names
describe rough prompt lengths, which depend on the model's tokenizer.

```sh
vade verify
vade verify -n 50 --prompt 2k
vade verify --url http://127.0.0.1:18080 --prompt 8k --max-tokens 256
```

The command prints one plain result line. It exits 0 for `IDENTICAL`, or 1 for a
difference or an incomplete check. It also writes `vade-verify.json` in the
current directory. This file is the receipt, the JSON record of the run. A
difference report names the run, the token position, the two IDs and their
text.

A pass covers the prompt, model, output length and engines recorded in the
receipt. It does not prove every possible prompt. `vade verify --help` lists the
endpoint and receipt options. The check uses the existing engine connection. It
never starts a server, changes drafting defaults, or downloads models or
prompts.

### Compare with a second engine

To compare against a second engine that runs on its own, add `verify_url` to
that machine's existing section in your `VADE_CONFIG`. Use an HTTP endpoint this
computer can already reach, such as an existing local tunnel:

```ini
[machine spark-b]
ssh = spark-b-ssh-alias
verify_url = http://127.0.0.1:18081
```

Leave the rest of that machine's configuration as it is. Then run
`vade verify --machine spark-b`. Both the primary engine and `spark-b` run every
on/off pair. Adding a rank to one distributed engine does not count as a
comparison with a second, independent engine.

## Measure a running server

`vade bench` measures what the client sees: how long you wait before output
starts, and the total answer tokens per second. After `vade launch` or
`vade up`, run it with no flags in another terminal. Use the same `VADE_HOME`
and `VADE_CONFIG` as the running installation. Like `vade verify`, it finds the
recorded endpoint, reads the model ID from `/health`, and writes
`./vade-bench.json` in the current directory.

```sh
vade bench
vade bench --draft plain --output vade-bench-plain.json
vade bench --self-test
vade bench --help
```

`--self-test` checks the client's local streams and arithmetic, not GPU
performance.

### Drafting and what the receipt records

By default it measures the server as it is served, including the FP4-Local
default of drafting on. Each receipt records:

- the draft mode each request asked for (server default, plain, or strict),
- the build SHA from `/health`, and
- the MTP depth the server reports, verbatim.

To see what each response actually did, check `draft_observations`, which keeps
the draft modes and MTP depths the responses report.

You can override the draft mode, and neither override needs a server restart:

- `--draft plain` sends `"draft":"plain"` in every request. Only receipts where
  every request asked for plain are labeled drafting off.
- `--draft strict` asks for strict drafting.

### Other bench options

- For a server outside the selected installation, pass `--url URL`.
- `--model ID` overrides the model ID from health. If health leaves it out, the
  fallback is `qwen3.8-flash-next-fp4-local`.
- `--output PATH` keeps receipts separate.
- The default uses a bundled offline writing prompt. `--prompts FILE` replaces
  it with saved prose, code and reasoning prompts, one per nonempty UTF-8 line.

Every request sets the greedy sampling fields explicitly and turns the cache and thinking off. It
ignores EOS (the end-of-answer token) so it generates the full fixed output
budget. Completion counts come from the server's usage numbers, never from
counting stream chunks. If one request stops early or one chat fails, the whole
group of requests (the cohort) gets no rate.

Advanced timing setups are in the [timing
options](../docs/benchmarks.md#run-vade-bench): cold 8k probes, measuring two
servers in turn (alternating arms), observer commands and frozen request hashes.

The client speaks plain HTTP. TLS is out of scope for a client built on the
standard library alone, so reach a remote server through an existing local SSH
tunnel.

The older Rust development workloads run through `vade bench --suite`, which is
built into `vade` itself. See [the suite guide](../engine/tools/bench/README.md).

## Status for scripts

`vade status --json` prints one `vade.status.v1` JSON object. It reports the
saved phase, model and build, settings, endpoint, the process identity of each
rank, observed health, and errors. Ownership tokens and credentials are never
status fields. When several Sparks serve one model, `multi_spark_path` gives
the path that the engine chose (`path`), why (`path_reason`) and, for the
tensor split, its levels. It is `null` when the engine does not report a path.

All checks share one 15-second deadline, so an unreachable rank cannot add
another timeout to every later check.

- Exit 0 means stopped or ready.
- Exit 1 means an error, or a recorded engine that needs attention.
- A stopped engine has `ready:false`.

Add `--dry-run` to read saved state without SSH, HTTP or worker probes:

- A recorded running engine then shows `ready:null`, and anything left
  unchecked is `null`, not healthy.
- A valid offline snapshot exits 0.
- Missing config or corrupt records still produce JSON, with
  `phase:"unknown"`, `ready:null`, errors and exit 1.
- Invalid command syntax still prints the normal diagnostic on stderr.

Status never starts, stops or repairs a server, and it keeps ownership records
intact for recovery. Speed appears only when health telemetry provides it;
status never sends a benchmark request.

## Use an engine that needs an API key

If your server requires a key, pass `--api-key-file PATH` to `vade chat`,
`vade doctor` or `vade run`, or set `VADE_API_KEY_FILE`. The flag wins over the
variable.

The file holds the key on one line: at least 16 visible ASCII characters
(`0x21-0x7E`). Spaces and line breaks around it are trimmed. We suggest 32
random bytes written as hex. For a remote server, use an SSH tunnel or TLS.

The CLI notes the path without opening the file. It reads and checks the file
once, when `run`, `chat` or `doctor` starts, before anything else happens:

- `up`, `status` and the menu never send the key (their health and capabilities
  checks go without one), so they never read it.
- Help, local file operations and dry runs work even if the file is missing or
  unreadable.

The harness sends the key as a Bearer token and shows an HTTP 401 plainly. Local
`vade run` also passes the file to `spark-serve`. This option does not copy keys
to remote ranks, so set up the server's key file where the server runs. Keys
stay out of command arguments, previews and saved CLI state.

## Keep local and controller setups apart

To use the separate controller commands below, set `VADE_CONFIG` explicitly.
Keep it pointed at the same file for controller `up`, `chat`, `status`,
`doctor`, `logs` and `down`, even if a local installation exists. Unset it for
the local lifecycle.

A controller config can also name several servers as `[group NAME]` sections.
Set `VADE_GROUP=NAME` to select one: its machines in rank order, its own
ownership record and its own local port. Keep the same `VADE_GROUP` for every
command of that server. A group must be an arc of the cabling: each host is
cabled to the next, in rank order. `vade doctor --fabric` checks every cable,
port and clock in the config, and `--ring` also expects one closed ring. See
[Eight Sparks](../docs/eight-sparks.md).

One `VADE_HOME` holds one local installation: either the default one (started
with plain `up` or `launch`) or the Qwen3-8B one (started with `run qwen3-8b`).
If you try to start the other kind, Vade refuses before it launches anything.
Use a fresh `VADE_HOME` if you want to keep both local installations.

## Multi-Spark source setup

This section is for a controller setup (a controller is described at the top
of this page). If you work on the Spark itself, follow
[INSTALL.md](../docs/INSTALL.md) and the sections above instead.

Choose the checkpoint profile deliberately:

<div align="center">

| Profile | Example config | How to start |
| :---: | :---: | :---: |
| Current FP4-Local model on 1 to 24 Sparks | [fp4-local.conf](examples/fp4-local.conf) | `vade up` (add `--sparks N` or ordered `--hosts` for a chain, or `--replicas N` for independent one-Spark copies) |
| FP8 model on 2 or 4 machines; 6 machines start but are not covered yet ([FP8 install](../docs/install-fp8.md)) | [vade.conf](examples/vade.conf) | `vade up --fp8` (the older flag `--fp8-legacy` also works) |

</div>

A chain is several Sparks linked one after another, serving one model together.
With `--replicas`, each replica is a full, separate copy of the model on its own
Spark.

Start with [INSTALL.md](../docs/INSTALL.md).

With a controller config, FP4-Local asks for the same defaults as `vade launch`
on the Spark, whatever the Spark count and with or without `--chats`: context
32768 and output `min(context, 8192)`. The one difference is that only
`vade launch` lowers the context, and only when the engine refuses it for
memory.

The engine decides from context and memory how many chats can run at once. It
reports that count when it starts, and the controller's `vade status` prints it.

An explicit `--context` wins, up to 262144 on either route. This long-context
setting still lets the engine decide the chat count from memory. An explicit
`--output` reaches 16384. MTP stays engine-selected unless you override it. A request
longer than the configured context is refused, and the refusal names
`--context`.

Before it touches any machine, `vade up` prints its plan: which model and
machines it will use, the context and output limits, your drafting and
concurrency choices, how the machines connect (the transport), and the local
endpoint it will serve on. Run `vade up --dry-run` to review that plan and the
commands without connecting or writing launch state.

Chains of 3 to 24 Sparks need an engine staged from current source. With an older
staged engine, Vade refuses those counts before it runs anything on the Sparks.
Testing so far covers this much:

- On one Spark, a live test on real hardware covers the launcher, direct chat,
  the gateway (the harness's HTTP relay to the engine) and canceling a reply.
- Larger chains also ran in the published
  [four-VM benchmark cards](../bench/receipts/2026-10-24/README.md). Those runs
  do not qualify every layout or recovery case. Each deployment still needs
  its own exactness checks. Two-Spark TCP is covered by host tests only.

### Build and configure

Build the command on the controller (Linux or macOS), from the checkout root:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
./cli/target/release/vade --help
```

Copy the matching example above, fill in its placeholders, and point
`VADE_CONFIG` at that file. The default location is
`~/Library/Application Support/vade/config` on macOS or `~/.config/vade/config`
on Linux. `XDG_CONFIG_HOME` overrides either.

In the config:

- Relative local paths resolve next to the config.
- Machine paths start with `/` or `~/`.
- `ssh` takes an existing alias, a host name or an IPv4 address, optionally as
  `user@host`.

Machine names, fabric addresses and personal paths belong in your local config,
outside the repository. The retired `queue_dir` and `hold_path` keys are refused
as unknown keys, so remove them from an older config before using the current
CLI.

Preview the setup first:

```sh
export VADE_CONFIG=/path/to/vade.conf
./cli/target/release/vade install --dry-run
```

### What `vade install` does

When the Sparks are free for you to use, run `vade install`. It checks the
controller's tools and every configured machine. Then, for the engine and its
helper, it does one of two things:

- uses the optional prebuilt bundle described below (built from exactly the same
  source), or
- copies the tracked source to each machine and builds them there.

For the chat program and CLI on your computer, it either copies a verified
prebuilt pair or builds them. See [prebuilt installation](docs/prebuilt.md),
which also covers how install prepares the Sparks.

Every install builds into new folders, called generations, and saves a private
manifest with SHA256 hashes for each one. When the build is done, Vade switches
a single record (the activation record) to the new generation in one step, so
you never run a half-installed build. Old generations stay around for rollback,
and each command uses one generation from start to finish.

Install prints the path of the CLI inside the new generation. Put that folder on
your PATH, or keep using the CLI you just ran. It checks the activation record
each time you run a command.

Use dedicated private folders for staging, state and the manifest. Vade refuses
existing shared folders and leaves their permissions as they are. Builds on each
machine reuse `build-cache/engine` under `engine_source` and `build-cache/cli`
under `cli_source`. Install never deletes older builds, so a running engine
keeps the build it started from.

On each machine, the engine and CLI source are copied together, as sibling
`engine/` and `cli/` folders under `engine_source/revisions/<snapshot>/`. The
CLI builds against that copy's engine benchmark library. The remote
`cli_source` only says where the CLI build cache goes. It does not hold a
separate CLI source copy.

Folder ownership and permissions are strict:

- Remote source, cache and staging roots must belong to the SSH account and give
  no permissions to group or other.
- Their parent directories must belong to that account or to root, allow no
  group or other writes, and contain no symlinks. Shared temporary directories
  are refused.
- Reused caches also refuse symlinks, entries owned by someone else, and shared
  writes.
- Install keeps existing permissions and creates each source and build
  generation exclusively. An unfinished generation stays unreferenced, and a
  retry uses a fresh generation instead of reusing its contents.

Source installation checks Rust, the CUDA driver, at least 10 GiB of build disk,
and both directions of every configured fabric link (the direct links between
Sparks). RDMA fabric checks use ICMP ping. TCP mode skips them. Engine startup
checks the actual transport and health.

<div align="center">

| Computer | What it needs |
| :---: | :---: |
| Machines | Linux with pidfd support, Rust with the engine's pinned toolchain and NVPTX build prerequisites, OpenSSH, rsync, curl, sha256sum, ping and nvidia-smi |
| Controller | Rust, OpenSSH, rsync, curl and shasum |

</div>

Offline builds need the source toolchains already installed.

To save time, Vade remembers when it last found the model complete, so a repeat
install or startup skips the shard check unless a file in the model folder
changed or went missing. This shortcut does not replace the full checksum check
of the checkpoint.

The Flash-Next weights stay under your control. If weights are missing or
incomplete, Vade tells you the checkpoint, revision and destination it needs.
Follow [step 3 of INSTALL](../docs/INSTALL.md#step-3-get-the-model) to download and
check the weights, then run `vade install` again. Install never downloads
Flash-Next weights itself. A sharded checkpoint must have every nonempty shard
named in its index.

### Prebuilt bundles

For a prebuilt installation, add a single optional key under `[global]`:

```ini
prebuilt_dir = /path/to/prebuilt-bundles
```

`prebuilt_dir` may also be an unpacked release launcher bundle, used when its
`launcher-bundle.conf` names this checkout's engine and CLI commits. See
[prebuilt installation](docs/prebuilt.md#prebuilt-binaries-for-a-controller-install).
Otherwise a bundle is a directory named `<engine-sha40>-<cli-sha40>` inside that
root. It holds `flash-serve`, `flash.ptx`, `vade` (the remote helper) and
`manifest.conf`:

```ini
format=vade-prebuilt-v1
target=aarch64-unknown-linux-gnu
engine_sha=<engine-sha40>
cli_sha=<cli-sha40>
flash-serve.sha256=<sha256-of-flash-serve>
flash.ptx.sha256=<sha256-of-flash.ptx>
vade.sha256=<sha256-of-helper>
```

Supply artifacts built from exactly those source revisions. The directory and
manifest are what you are trusting, so take them from a trusted prebuild or a
verified release.

Vade checks every SHA256 locally and again after transfer, and checks that the
binaries are AArch64 ELF64. If a bundle is absent, incompatible or corrupt, Vade
falls back to a fresh source build after checking the source toolchain. Rejected
bytes are never run, and a failed fallback leaves the active generation as it
was.

With a valid bundle, the machines need the runtime tools above and 1 GiB of free
install space, but no Rust or NVPTX compiler. The helper is verified before it
runs. Shard inventory checks then run alongside the engine and PTX copy (or the
engine compile). A repeat install rechecks the remote hashes and reuses staged
artifacts that have not changed. No release service or daemon is added.

### Controller commands

With `VADE_CONFIG` set, the lifecycle commands work like this:

<div align="center">

| Command | What happens |
| :---: | :---: |
| `vade` | Short status and four choices: start, chat, stop, doctor. |
| `vade up` (also `--fp4-local`) | Starts FP4-Local on one configured Spark. Add `--sparks N` for 2 to 24, or `--sparks a,b` or `--hosts a,b` to name the Sparks in cable order. Larger chains need a current staged build, and each must still pass its own live checks on real Sparks. |
| `vade up --replicas N` | FP4-Local only, N = 2 to 16: one independent one-Spark server on each of the first N configured machines (or the N `--hosts`), behind a local gateway. Details below the table. |
| `vade up --replicas N --pairs` | FP4-Local only, N = 2 to 12: N two-Spark tensor-split servers on 2N machines, in pairs in list order, behind a local gateway. This mode is an option. It is not the four-Spark default ([two pairs behind one address](../docs/four-sparks.md#two-pairs-behind-one-address)). |
| `vade up --fp8` | FP8: starts on the first two configured machines, tensor parallel on a ring. |
| `vade up --fp8 --sparks 4` | FP8: four machines in config order, tensor parallel on a path. It gives the two-Spark token IDs ([benchmarks](../docs/history/benchmarks-2026-10.md#fp8-on-two-and-four-sparks-different-token-ids)). |
| `vade up --hosts spark-a,spark-b` | Selects connected machines of your config in the given order. Give what you type after ssh, as in the `ssh` line of each machine, or its `[machine NAME]` name. `--sparks spark-a,spark-b` is the same. |
| `vade status` | Machines, model, source build, MTP, context, output limit, how many chats the engine allows at once, memory/cache bytes per rank, ranks, tunnel, health, and speed when health reports it. |
| `vade down` | Stops only the ranks Vade recorded and the SSH control connection it opened. |
| `vade logs` | Recent redacted engine logs, including saved logs after shutdown. |
| `vade chat` | Opens the coding agent (`vade-harness agent`) in the current folder, connected to the engine you own, with the recorded context. Details below the table. |
| `vade doctor` | Machine checks plus the harness doctor and its two small cache test turns. |
| `vade install` | Checks, builds, stages, and checks the harness. Without a running engine, it verifies that the harness doctor is available and explains the remaining engine check. |
| `vade update` | Updates clean checkouts with `git pull --ff-only`, rebuilds, restages, and offers a restart. A checkout with no remote stays local. |

</div>

With `--replicas N`, each replica sits behind its own tunnel on the N local
ports after `local_port`, and the local gateway listens on `local_port`. It needs
no `fabric.*` entries and cannot be combined with `--sparks` or `--fp8`.
`vade down` stops the gateway, the tunnels and every replica. See
[Replicas](../docs/INSTALL.md#serve-many-chats-with-replicas). With `--pairs`, each
pair needs the `fabric.*` entries of its own link, only each pair's first
machine gets a tunnel, and the gateway runs with `--stable-prefix`.

Every lifecycle command accepts `--dry-run`, in any position. A dry run prints
every remote command it would run, including worker operations and the stdin
fields that are safe to show, but it runs nothing, writes no files, opens no
connections and prints no secrets. Before a manifest exists, the revision and
hash values in a dry-run startup are marked as placeholders. Help works without
a config.

### Launch, reuse and the JIT cache

`vade up` with no flags keeps the recorded launch options. It reuses a healthy
engine and tunnel only when the current configuration and the activated build
still match. Explicit launch requests and `vade chat` check the same
configuration and build before reusing the server.

- After you change settings or install a new build, run `vade down` before you
  relaunch.
- An older session that lacks the details needed to verify reuse also needs
  `vade down` and a fresh `vade up`.
- With the same configuration and build, you can exit chat and reopen it without
  loading the weights again.

`vade down` stops the engine, but the GPU code cache stays. PTX is the portable
GPU code that the driver compiles for your exact GPU. Each version of the
kernels (each PTX hash) gets its own private CUDA JIT cache in the machine's
state folder, and that cache survives restarts and new installs.

A new PTX hash pays a one-time compile cost unless install already prepared its
cache or a matching trusted compiled image exists. See [prebuilt
installation](docs/prebuilt.md). Memory setup still runs before the engine
reports ready, and no startup or arithmetic step is skipped.

### Chains and fabric

FP4-Local supports 1 to 24 ranks (one Spark per rank). Chains of 3 to 24 need a
current staged build that advertises chain support, so run `vade install` from
current source first. Older staged builds are refused. Every larger layout must
still pass its own exactness check on real GPUs.

How the ranks connect:

- One FP4 rank needs no fabric.
- Two ranks link to each other.
- Larger FP4 chains use an open path in the host order you picked.
- The FP8 profile separately supports 2 or 4 machines (two in a ring, four on
  a path). It also starts on 6 machines on a path, but its token IDs differ
  there, so 6 machines are not covered yet.

Each `fabric.<peer>` value is this machine's own address on its link to that
peer. Define both ends of every link between neighbors. Each rank gets its own
pair of addresses, one to listen on and one to reach the next rank. The two
transport ports are the `tp_port` setting and `tp_port + 1` (for data), and the
serve port must differ from both.

For four Sparks, use [the re-cabling and preparation
guide](../docs/four-sparks.md) and [the separate world-four golden
procedure](docs/test.md#world-four-goldens) (world four means four ranks;
goldens are saved reference outputs). The default four-rank path follows three
consecutive links, even on a physical ring of four cables, so the closing cable
goes unused. Host checks alone do not prove this runtime path. Before it
launches any process, startup rejects selected ranks whose engine or PTX hashes
differ.

All selected ranks must use identical expanded absolute PTX and model arguments.
The staged PTX path is `machine.stage_dir/builds/<generation>/flash.ptx`, so
pick matching remote staging roots as well as matching `model_path` values. The
same `~/` string can expand differently under different remote homes. Before it
launches any rank, `vade up` expands and compares both paths on every selected
machine and rejects a mismatch. Matching artifact hashes are not enough. Dry-run
previews skip this remote check.

### FP8 limits and drafting

For the FP8 profile, the defaults are strict MTP 2 for a cold standalone
chat, context 131072, and output `min(65536, context)`.

- `--context N` accepts 4096..262144.
- `--output N` accepts 1024..262144 and may equal the configured context.
- Each reply is capped at the smaller of the service output limit and the
  context minus the tokenized prompt length, even when the request leaves out
  its output limit.
- Use `--context 32768 --output 8192` to compare with earlier timing receipts.

The legacy `--fast-prompts` flag is still accepted so saved commands keep
working, but it does not change arithmetic: Flash always uses BF16 hi/lo tensor
cores.

Manifests record whether the engine supports `FLASH_SERVE_OUTPUT_TOKENS`, a
settable output limit. Older builds accept only their fixed `min(context,8192)`
limit.

In the FP8 profile, the model proposes tokens ahead and checks each one
against ordinary decoding before keeping it. With no `--mtp`, a chat alone
chooses its depth for each step, up to 7 tokens. With `--mtp 1` to `--mtp 3`,
the depth is fixed. When chats overlap, they
run without drafting (plain) until they finish, even if the other chats end
first. A chat continued from the cache without its saved draft history also runs
plain.

To control drafting depth:

- `MTP=0 vade up --fp8` or `vade up --fp8 --mtp 0` turns drafting
  off, and `--mtp 0..3` overrides the `MTP` environment setting.
- The chosen depth is saved and passed explicitly to every rank, including zero,
  and a restart keeps the saved depth.
- FP4-Local leaves out the depth argument when neither the command nor the
  controller's MTP setting gives one, so the engine default holds through
  restarts.

Check each response's `model_draft_used` to see whether drafting actually ran.
Automatic memory sizing counts the MTP weights and working memory on every rank.

### Replies, memory and cache

Chat tells the harness the recorded context. Each reply sets aside the smaller
of two amounts: the engine's output limit, or what is left of the context minus
256 spare tokens. It uses the engine's exact token counts when they are
available.

If an answer hits its output limit, chat continues it automatically, up to 32
times in a row, as long as the request stays within its turn and work limits. A
manual `/continue` starts a fresh work budget. See the [harness generation
options](../harness/README.md) for direct launches with `--max-output-tokens` or
`--no-auto-continue`.

The harness and engine agree on each reply's output size automatically
([details](../harness/docs/engine-contract.md#how-the-output-limits-changed-october-2026)).
That agreement has host tests only so far. The one-Spark live test above did not
check these limits.

For the FP8 profile, the engine sizes its session pool from the context
and the lowest safe capacity across ranks, after shared workspaces and a safety
reserve are set aside. Startup must have room for at least one full-context
session, and exits with a plain memory message if that reservation cannot fit.

- `--chats 1..32` reserves room for that many full-context requests at once,
  caps the live slots, and persists through status and restart.
- Without it, the pool uses all the capacity the ranks agreed is safe, and the
  number of request slots grows and shrinks as needed.
- By default, the retained prefix cache gets half the pool or the room for one
  full-context request, whichever is larger. Idle cache gives its capacity back
  to live work.
- An explicit `FLASH_SERVE_KV_BYTES` sets retention exactly, including `0` to
  turn retention off, and `SEQPAR_MEMORY_BYTES` sets an explicit memory ceiling.

Startup forwards those overrides when present. Otherwise it clears inherited
remote values so the engine picks its own defaults. It also turns profiling off,
creates a fresh shared run ID and control secret, and clears the launcher's
experimental environment overrides.

Two optional flags set memory and cache sizes:

- `--memory-bytes N` sets a per-rank memory ceiling and overrides
  `SEQPAR_MEMORY_BYTES`.
- In FP8, `--cache-bytes N` sets the retained cache capacity through
  `FLASH_SERVE_KV_BYTES`, including `0` to turn retention off. Local FP4-Local
  supports only `--cache-bytes 0`, which it translates to its own prefix-cache
  switch. A positive capacity is refused.

The controller accepts and saves `--chats` and `--cache-bytes` for FP4-Local,
but the current FP4-Local engine does not use those values. This includes
`--cache-bytes 0`. It chooses the chat count from context and memory, and its
separate prefix-cache setting controls retention. Check the admitted chat
count and cache capability instead of treating these saved flags as limits.

For FP4-Local, the retained prefix cache is on unless it was switched off at
startup (see [the prompt cache](#chats-at-once-and-the-prompt-cache)). A request
that sends `cache:true` does not turn it on by itself. The harness still sends
negotiated cache hints. For FP8, leaving memory and retention unset
means automatic sizing. The engine still checks that the model, shared
workspaces and full-context live reservations fit before it reports ready.

### Ownership, logs and stopping

Vade keeps its ownership records on your computer and on each Spark. Before
launch it checks every file hash, and the helper on the Spark checks the engine
and PTX hashes again.

For RDMA, Vade checks that the SSH session on each Spark has an unlimited
locked-memory (memlock) limit, both soft and hard. Ask an administrator to set
that up for the SSH account Vade uses, because Vade does not change system
limits.

Before it reports ready, and before status, chat or doctor, Vade checks two
things: the loopback HTTP port on the first Spark (rank zero) belongs to the
exact process it recorded, and the model reports healthy. The controller also
hashes the staged copies and checks the installed chat program before every
chat or doctor run.

Logs from failed commands stay in a private temporary folder so you can look at
them. When a command times out, Vade stops that command's own process group
before it collects the command's exit. `vade run`, `vade up` and `vade down`
catch Ctrl-C and TERM.

If you interrupt `vade up`, it cleans up the Sparks and the tunnel it recorded,
using commands that still work after the interrupt. If cleanup does not finish,
the records stay so `vade down` can try again. On Linux, the commands Vade
starts also ask the system to stop them if Vade itself dies. We have not yet
tested on real hardware what a hard kill of the CLI does in the middle of a
compile.

Vade refuses to start, and tells you the next step, if the GPU is busy, records
already exist, a process identity has changed or the machine rebooted.

When you stop, Vade checks every Spark it can reach before sending any signal,
and it uses Linux process handles, not a guessed PID:

- If any process identity does not match, Vade sends no signals at all.
- If a Spark cannot be reached, Vade still stops the others it has verified and
  keeps the records it could not resolve.

The controller's tunnel uses its own private OpenSSH control socket, opened
relative to its private state folder, so long paths work. Vade removes a dead
socket after checking that nothing is listening on it, and leaves alone a
listening socket that fails the SSH check. If a launch reply was lost, Vade can
recover it from the machine's saved run file. If the startup files are
uncertain, you need to check them yourself.

Old ownership records still load their recorded settings, and missing byte
overrides use automatic sizing.

### Chat, doctor and update

`vade chat` uses the harness's normal approval policy and strips inherited
cloud-provider credentials from the command it runs. `vade doctor` needs a
running engine you own for its inference checks.

Update leaves a running engine alone until you accept the restart, and
noninteractive updates print the restart commands. A degraded old engine cannot
stop a finished update from offering that restart.

For how the coding agent handles long tasks (it shortens older finished tool
steps and keeps your goals and recent edits), see [context
controls](../harness/docs/context-and-projects.md). A request stops after 60
tool rounds or 20 minutes of active work. The [native tool
contract](../harness/docs/agent-tools.md) covers the named check profiles. They
need `--shell-toolchain` on a direct harness launch, which `vade chat` does not
add. Passing host checks does not prove the GPU path works.

## Recovery

Try `vade status`, `vade logs`, then `vade down` first. If the controller
crashed or the SSH master died, `vade down` is all you need: it reports
`The controller tunnel was already closed.` and carries on stopping the ranks you own.
If the local engine crashed or was killed, `vade status` says
`the engine is not running (stale record found)`. The next `vade up` or
`vade launch` clears that record in one line, then starts as usual. The crash
log stays in `vade logs`.

The launch helper saves the engine process's exact identity in a journal,
`journal.conf` (see step 1 below), just before it starts that process (exec).
With that record, `vade down` can usually clean up on its own. A rebooted
machine or a launch request that never finished needs the manual steps below.
Do them at a time when nobody else is using that machine.

The local FP4 launcher records its engine in `VADE_HOME/state/run.conf`. The
Qwen3-8B catalog uses `VADE_HOME/run.conf` and `VADE_HOME/worker/`. The identity
checks in step 2 (boot, start time, executable and argv) apply to these files
too.

1. Stop any launch, install or cleanup commands running at the same time. Keep a
   private backup of the controller's `state_dir/session.conf` and of the
   affected machine's `state_dir/active.conf` and `<token>/` directory. Use the
   machine paths and the full token from the saved session, even if the config
   has changed since. The directory holds `journal.conf` (phase and process
   identity), `env.conf` (private environment) and `server.log`. Never print
   `env.conf` or put it in a report. Leave each permanent `.lock` file where it
   is.
2. Inspect every affected machine before you change any files. Compare the
   `boot_id` in `journal.conf` with `/proc/sys/kernel/random/boot_id`. A
   confirmed reboot means that machine's old engine cannot have survived, but
   its saved PID may now belong to someone else. On the same boot, check the
   PID, the `/proc/<pid>/stat` start time, `/proc/<pid>/exe`, and the whole
   NUL-separated `/proc/<pid>/cmdline` against the saved `pid`, `start_ticks`,
   `exe_hex` and `cmdline_hex` (decode the last two from hexadecimal). If the
   identity is missing or uncertain, leave the files in place and work out which
   process the original launch started. A process name, port or GPU entry alone
   is not enough. Never signal a reused PID or another job's process.
3. Once you have confirmed the affected engine is gone, check `nvidia-smi` for
   compute processes and for power draw under 30 W. If the GPU is busy, leave its files
   alone until its current owner finishes. Move only this token's saved
   directory out of `state_dir` into the private backup, and remove
   `active.conf` only when its token matches exactly. Leave other token
   directories, model files, source and build directories, and `.lock`
   untouched. Retry `vade down` from the controller: absent ranks are skipped,
   and the remaining ranks still need their exact identities before any stop.
4. Let a successful `vade down` remove the controller's `session.conf`. If you
   must clean up the controller by hand, first confirm that every rank you own is
   gone, that its machine files are cleared, and that the token-named tunnel has
   closed. Then move `session.conf` aside and remove only its dead
   `tunnel-<token-prefix>.sock`. Do not delete a listening socket that SSH
   cannot verify. Only start again with `vade up` after these checks.

## Legacy Qwen3-8B workflow

For the legacy Qwen3-8B catalog, start with the [Qwen3-8B section of the
one-Spark
quickstart](../docs/quickstart-one-spark.md#part-2-older-route-build-from-source-with-qwen3-8b),
including its optional CLI step. The current FP4-Local route is
[INSTALL.md](../docs/INSTALL.md). On the Spark, with `vade` already installed:

```sh
vade run qwen3-8b
vade chat
vade status
vade down
```

### What `run` checks first

Before it starts any work, `run` checks:

- Linux/aarch64, one GB10 and its driver;
- running compute jobs;
- free GPU and unified RAM, and free disk;
- Linux pidfd support.

It needs curl and sha256sum. On GB10, `nvidia-smi` may report GPU memory as N/A,
so the check uses available unified RAM and reports that number. A new model
needs 25 GiB of free disk and an idle GPU with 24 GiB of free memory. Weights
that are already verified need 1 GiB of disk headroom. Vade changes no OS,
driver, network or sudo configuration.

### Build from source

Source builds also need git, cc, tar, Rust 1.99.0 and nightly-2026-10-01 with
the NVPTX target and the rust-src, llvm-tools and llvm-bitcode-linker
components. Vade reports missing prerequisites before downloading anything, and
it does not install toolchains.

It fetches the Vade monorepo named by [`VADE_REPO_URL`](src/repository.rs) and
takes `engine/` and `harness/`. It saves each full commit before building, and
keeps those commits for retries. Git access must already work. You can also
supply clean local monorepo roots or component directories. Vade does not build
inside them:

```sh
vade run qwen3-8b --engine-source /path/to/engine --harness-source /path/to/harness
```

### Use weights you already have

To use an official BF16 checkpoint you already have, without downloading
weights:

```sh
vade run qwen3-8b --weights-source /path/to/Qwen3-8B
```

Vade only reads the directory you supply. Every pinned payload must be there and
match its SHA-256. Missing or altered files fail, and Vade does not download
replacements. The path is saved in the control home, the private directory
described under [Where the Qwen3-8B workflow keeps
state](#where-the-qwen3-8b-workflow-keeps-state).

- A directory named `Qwen3-8B` is read in place and needs 1 GiB of disk
  headroom.
- A directory with any other name is imported into a verified canonical
  `Qwen3-8B` directory inside the control home, because the engine uses that
  name as its API model ID. This needs 25 GiB at first.

Later runs rehash every payload in the directory the engine uses.

### Where the Qwen3-8B workflow keeps state

This workflow keeps its state apart from the FP4-Local launcher, which uses
`~/.vade` (see [Where Vade keeps your files](#where-vade-keeps-your-files)).
For `run qwen3-8b`, `VADE_HOME` picks a dedicated private directory. The default
is `XDG_DATA_HOME/vade` or the user's `.local/share/vade`. Source snapshots,
generations, revision-specific models, logs and ownership records all stay
there.

Run the same command again to resume a download, build or doctor check. A
running server you own is reused once its process identity, listener, artifacts
and model bytes pass verification. If a startup identity is uncertain, Vade
keeps its records, and you need `vade logs`, `vade down`, or the
[Recovery](#recovery) steps. If a doctor or answer check did not finish, the
server stays recorded so you can retry or shut it down.

### How source builds clean up

Before a local source build starts, Vade saves private records of the job under
`build-jobs`. A small supervisor process holds the build's process group open
until the compiler exits. Before it stops that group, it confirms that the group
is the one it started (same boot, same start time). It also checks that the
group has really stopped before it marks the job done.

A failure, a timeout, Ctrl-C or losing the CLI that started the build all
trigger this cleanup, and a retry or `vade down` finishes it through the
supervisor. If Vade cannot confirm which process it started, or the supervisor is
gone, it sends no signals. A supervisor that crashed or was killed, or a reboot,
leaves a stale record: once the supervisor and every recorded build process
have stopped, the next `vade install`, `launch`, `up`, `update` or `down` marks
the job stopped, prints one line naming its folder and carries on. While any of
them may still run, Vade keeps the records and stops with an error. Keep those
job folders so you can look at them.

Pressing Ctrl-C while you wait for readiness stops a server that this invocation
started, and releases the operation lock after cleanup. A server that was
already running before the invocation stays recorded.

If a saved `--engine-source` or `--harness-source` checkout is missing, the run
fails without fetching, even when its build snapshot is still there. Restore the
original checkout, or deliberately pick a fresh `VADE_HOME` without the local
source options so that Vade fetches from the network into managed storage.

### Pinned model files

The catalog pins the official Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218`. All five BF16 shards, the index and
six tokenizer and configuration files have their SHA-256 digests kept in
`src/catalog.rs`.

Downloads use revision-specific `.part` files and resume with curl ranges. Every
payload is hashed before it is published and again every time it is reused. Old
`.ok` files prove nothing. Extra model payloads and altered published weights
are refused. No draft model or numerical conversion is fetched. The server uses
explicit plain drafting and the engine's default arithmetic.

### Readiness and the first answer

Vade counts the server as ready when the exact process it recorded is listening
on the loopback port and its health check reports qwen3-8b ready. If health
does not report GPU memory use, Vade prints it as unknown. The GPU memory it does
report leaves out memory the driver uses on its own.

Vade then streams a short greedy first answer and prints how long each stage
took: getting the source or build, download, verification, load, first content
and the whole answer. Next it runs the harness doctor, which gets 120 seconds.
On Linux, `vade chat` opens plain text chat.

The last lines print the harness command and an environment command for Claude
Code. Printing them does not mean anyone has tested Claude Code chat or its tools
against this server.

### Use a legacy release directory

If you already have a legacy release directory locally, you can skip all the
Rust and build prerequisites:

```sh
vade run qwen3-8b --release /path/to/release
```

The directory must hold:

- regular `spark-serve`, `vade-harness` and `kernels.ptx` files, with executable
  permissions on the binaries;
- `release.conf` with `version=1`, `target=linux-aarch64-gb10`, the full
  `engine_revision` and `harness_revision` Git SHAs, and the `engine_sha256`,
  `harness_sha256` and `ptx_sha256` SHA-256 digests.

Vade verifies the supplied files and the staged copies. It publishes them as an
immutable generation (a build folder that never changes afterwards), and
switches one generation-ID record in a single atomic step. You are trusting the
manifest you supplied.

The v0.1.0 release does not include this legacy format, and there is no download
for it: without `--release`, `vade run qwen3-8b` builds from source.

### Preview and port

Lifecycle commands accept `--dry-run`. `vade run qwen3-8b --dry-run` previews
the checks and the flow without touching files, running commands or downloading.
`--port` changes the local port (default 8080). Keep the original port when you
resume a server you own.

## Gate commands

The gate commands are the pass-or-fail checks a build has to clear. They work
without `VADE_CONFIG`, `VADE_HOME` or authentication files, and take the
existing gate options and explicit inputs. They do not connect to configured
machines, install toolchains or take charge of starting a server. Name every
input on the command line yourself, and keep receipts (the result files a check
writes) outside the repository.

```sh
vade test build --help
vade test determinism --help
vade test longctx prepare --help
vade test longctx run --help
vade test longctx compare --help
vade test gdn-state --help
vade test unix-flags-abi
vade test unix-flags-abi --target aarch64-unknown-linux-gnu
```

The ABI command compiles twelve assertions about Unix flags without linking or
running any target code. The selected compiler needs that target's Rust
libraries.

Saved long-context and recurrent-state evaluations use
`vade test longctx prepare|run|compare` and `vade test gdn-state`. The
[evaluation guide](evals.md) covers the options, outputs and frozen formats.
These commands work on saved local inputs and need no lifecycle config.

## Safety details

On screen, Vade escapes terminal control characters and text-direction tricks in
errors, command previews and server logs, so a log line cannot take over your
terminal. Normal Unicode and line breaks still show. Saved logs, ownership
records, command arguments and network requests keep their original bytes.

## Working on Vade itself

To build and check a change, see [CONTRIBUTING.md](../.github/CONTRIBUTING.md) and the
house rules in [AGENTS.md](../AGENTS.md). From `engine/`, this block is
identical to the host block in AGENTS.md:

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --locked --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --locked --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
cargo +1.99.0 run --offline --locked --manifest-path ../tools/Cargo.toml -p vade-tools -- check-doc-links
```

The CLI's tests use fake SSH and local temporary files. Live installation, CUDA,
pidfd signalling, RDMA topology, health, tunnel recovery, harness cache checks
and restart all need an authorized session on a Spark, so host tests make no
claim about live performance or GPU exactness.
