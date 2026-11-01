# Install Vade

This guide installs Vade for text-only chat with the supported FP4-Local
checkpoint, **nvidia/Qwen3.8-Flash-Next-FP4-Local**. The installer
downloads the [prebuilt bundle](#prebuilt-bundle) of main from dl.vade.lol
when the bundle fits your Spark. When it does not fit, you build from
source. The source commands build main, including the Unreleased changes in
the [changelog](CHANGELOG.md). You start with one Spark and no other
computer.

Vade also supports the FP8 checkpoint, **Qwen/Qwen3.8-Flash-Next-FP8**, on
2, 4 or 6 linked Sparks. Its weights do not fit on one Spark. FP8 uses
[Route B](#route-b-from-a-controller) and its own config. It starts with
`vade up --fp8`. [Install FP8 on 2, 4 or 6 Sparks](docs/install-fp8.md) has
the steps.

**How long it takes.** We timed the source steps of this page on a DGX Spark
with no Rust, no checkout and no Vade files. The run was on 2026-10-28 at
main `285fa8a`. The model was already on the Spark. The times come from the
timing log of record `2026-10-28_1744_source-install_285fa8a` and its
`Launch ready` line. The record is not in the repository yet.

| Step | Time (s) |
| --- | ---: |
| [Install Rust](#install-rust) and the two pinned toolchains | 16.7 |
| Clone ([step 2](#step-2-get-the-code)), [one kernel build](#run-one-kernel-build-source-builds-only) and the [Route A](#route-a-on-this-spark) CLI build | 62.2 |
| First `vade launch`, to `Launch ready in 149.4 s.` | 149.4 |
| Total | 228.2 |

Each row is rounded to 0.1 s, so the rows add up to 228.3 s, not the
228.2 s total. The total does not include the model download (102.26 GB). It
also does not include a network fetch of the source: the clone came from a
local copy of the repository. Download times depend on your network. We
have not timed these steps on a later main.

On one Spark, use the [quick start](#quick-start-on-one-spark). It also
downloads the model when the Spark does not have it. For the long form, do
steps 1 to 3: prerequisites, the code and the model. Then pick a route:

- **[Route A](#route-a-on-this-spark).** Everything runs on the Spark itself.
  Start here.
- **[Route B](#route-b-from-a-controller).** A controller drives the Spark
  over SSH. The controller is a second Linux or macOS computer.
- **[Linked Sparks](#two-and-more-sparks).** This route builds on Route B.

This guide explains each new term where it first appears. The
[glossary](docs/GLOSSARY.md) has the rest.

On one Spark, there are three steps. Install Vade with the one-line
installer, get the model with `vade pull`, and start the engine with
`vade up`.

From a controller, there are two steps. Get `vade`, and run `vade setup`.
Setup asks a few questions and writes what it needs. Then it runs
`vade install`, `vade up` and `vade doctor`.

Each step prints what to do when it fails. You can run a step again safely.
[Install and flows](docs/design/install-and-flows.md) shows the whole flow on
one page. If you are an AI agent, read the
[agent quickstart](docs/agents/quickstart.md).

## Quick start on one Spark

<a id="quick-start-with-an-existing-model"></a>
You need a DGX Spark with curl and a C compiler (`cc`). Its NVIDIA driver
must support CUDA 13.0 or newer (the `CUDA Version` in the `nvidia-smi`
header). The GPU must be idle: `nvidia-smi` must list no compute processes.

Run these three commands on the Spark. Most of the time goes to the model
download.

```sh
curl -fsSL https://vade.lol/install.sh | sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade up
```

1. **Install** checks the GPU and the driver. Then it downloads the
   [prebuilt bundle](#prebuilt-bundle) of main from dl.vade.lol. The bundle
   holds ready-made programs for the GB10, so the Spark needs no Rust and no
   build. The installer checks the bundle's hashes and installs it into
   `~/.vade`. It does not need sudo or the model, and it does not change your
   shell profile. At the end, it prints an `export` line that puts `vade` on
   your `PATH`. Run that line if your shell cannot find `vade`.
2. **Pull** downloads the model (102.26 GB) and checks every file. The long
   hash after `@` pins the exact revision we tested. If the download stops,
   run the same command again. It continues where it stopped.
3. **Up** starts the engine on `http://127.0.0.1:18080`. `vade launch`
   starts it and opens chat in the current folder. `vade down` stops it.

The installer downloads the build of main from dl.vade.lol. GitHub Releases
holds only the older v0.1.0 bundle.

When the bundle does not fit your Spark, the installer prints one
`Prebuilt:` line with the reason. The one-line install has no copy of the
code to build, so it then stops with an error. In that case, use
[the source build](#build-from-source-on-one-spark) below.

### Build from source on one Spark

Use this route when the prebuilt bundle does not fit your Spark, or when you
want to build main yourself. You also need Git. Run these five commands:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade launch
```

- `scripts/install.sh` checks the Spark first. Then it tries the
  [prebuilt bundle](#prebuilt-bundle), as the one-line install does. When the
  bundle does not fit, the script prints one `Prebuilt:` line with the reason
  and builds from source. The source build installs the pinned Rust toolchains if they are
  missing, and builds `vade`, the GPU kernels, the engine and the chat
  program. It takes about three minutes on a GB10: the `INSTALL` step took
  176.8 s in the
  [fresh-install receipt](bench/receipts/2026-10-31/fresh-install-2026-10-30.txt).
  The script does not need the model. `--source` always builds from source.
  `--check` prints what the script would do, writes nothing and exits 0.
- The `export` line puts `cargo` and `vade` on your `PATH` for this shell
  only, because the installer does not edit shell profiles. Run it again in
  each new shell, or add it to `~/.profile` yourself. This line is for the
  source build. When the script installs the prebuilt bundle, it prints an
  `export` line for the bundle folder instead. Use the line that it prints.
- `vade pull` downloads the 102.26 GB model into
  `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local` and checks every file. Skip
  it if the model is already in `~/models`. To use a checkpoint outside
  `~/models`, set `VADE_MODELS` before the first launch. See
  [use a different checkpoint folder](#use-a-different-checkpoint-folder).
  [Step 3](#download) tells you how to keep a long download running over SSH.
- `vade launch` finds the model, saves the settings in `~/.vade/config` and
  checks the model files. Then it starts the engine on
  `http://127.0.0.1:18080` and opens chat in the current folder. If another
  program has port 18080, it names the port it uses instead. Type `/exit` to
  leave chat. `vade down` stops the engine.

When the model is already in `~/models`, one command can do the last three
steps: `sh "$HOME/vade/scripts/install.sh" --launch` installs, starts and
opens chat. If you already cloned Vade, open its folder and run
`sh scripts/install.sh`, then continue with the `export` line.

**Older steps.** Earlier versions of this guide ran `vade setup` and
`vade chat` after the installer. That still works. On one Spark with saved
settings, `vade setup` skips `vade install` when the installed files are
current, then runs `vade up` and `vade doctor`.

### For a script or an agent

The Vade steps ask no questions. Each command exits 0 on success and 1 on a
failure. A failure prints its cause and a next step. With `--json`, the cause
and the next step are inside the JSON object:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade"
sh "$HOME/vade/scripts/install.sh"
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade up && vade doctor --json
```

`vade up` starts the engine without chat and prints one `Up ready in` line.
`vade doctor --json` prints one `vade.doctor.v1` object. The
[agent quickstart](docs/agents/quickstart.md) gives the rules, the JSON
forms and the controller commands.

The installer runs only on Linux `aarch64`. It does these steps:

1. It checks that the GPU is a GB10, that the driver supports CUDA 13.0 or
   newer and that no compute process uses the GPU. It stops before any
   download if one of these fails.
2. It tries the [prebuilt bundle](#prebuilt-bundle). When the bundle fits and
   its checks pass, it runs `vade install` from the bundle folder. That
   install copies the prebuilt files and builds nothing. Then the script
   goes to step 6. When a check or the download fails, the script prints one
   `Prebuilt:` line with the reason and continues with step 3. With
   `--source`, it skips this step.
3. It finds Rust in `~/.cargo/bin`. It installs any missing pinned
   toolchains and GPU components.
4. It builds the CLI and fetches the Rust `build-std` files that the GPU
   kernel build needs. When they are cached, it downloads nothing.
5. It runs local `vade install`. This checks the Spark once (preflight),
   then builds the GPU kernels, the engine and chat once, offline. It also
   compiles the GPU code into Vade's private startup cache, so the first `up`
   does not have to. When the model is already under `~/models`, it saves
   the settings and checks the model files too. When it is not, it saves no
   settings and prints the `vade pull` command. If `~/.vade` already holds
   files that another checkout or revision built, it builds this checkout and
   prints the generation that is now installed.
6. With `--launch`, it starts the engine and opens chat in the folder where
   you started. Without that flag, it stops after the install.

The installer does not download or edit models. It does not change shell
profiles or run sudo. Every failed stage prints its cause and a next step.
Read the script first if you want to see what it runs.

The first run needs internet access for missing Rust and `build-std` files.
When those are cached, `CARGO_NET_OFFLINE=true sh scripts/install.sh` works
offline from the checkout. It reuses the tools it finds and does not ask
rustup to update them. In record `2026-10-28_1744_source-install_285fa8a`,
the Rust download and install took 16.7 s. We have not timed the model
download. On 2026-10-30, on one Spark with the model on disk and no Rust,
`sh scripts/install.sh` took 176.8 s
([receipt](bench/receipts/2026-10-31/fresh-install-2026-10-30.txt)).

**After launch.** Chat can edit files in the folder where it opens. For this
first test, ask it only to say hello. Then type `/exit`. The engine keeps
running.

Run later commands by their full path, `"$HOME/vade/cli/target/release/vade"`,
or add that folder to `PATH`. `vade up` starts the engine without chat.
`vade chat` connects again. `vade down` stops this installation's engine.
When you run `vade launch` from a script without a terminal, it prints
readiness and exits. The engine keeps running.

**If something goes wrong:**

- **`~/vade` already exists:** use the existing-checkout commands above, or
  clone somewhere else.
- **The clone fails:** check that this connection reaches GitHub.
- **`vade launch` stops at the model check:** the model is not where Vade
  looks. Run the `vade pull` line from the quick start, or set `VADE_MODELS`
  to the checkpoint folder, then run `vade launch` again.
- **`VADE_CONFIG` or `VADE_BUNDLE` is set:** the script tells you which one to
  unset for this local source route.
- **This Spark already has a controller config:** use the manual
  [Route A steps](#route-a-on-this-spark). Without saved local settings,
  the installer's `vade install` can select that controller config.

Keep the same `VADE_HOME` (default `~/.vade`) for install, launch and every
later command.

## Before you start

Once the CLI is built, run `vade doctor` before `vade up`. It checks the
binary, the model files, the memory, the driver, the links, the ports and the
disk. It also shows the four default settings. Each failed check names its
fix. A controller checks the active Sparks, or the Sparks that `vade up` would
select, over the same SSH path.

You need a DGX Spark (NVIDIA GB10) running Linux `aarch64`. `nvidia-smi` must
work, and the GPU must be idle.

We tested on the **580.x driver series**. The v0.1.0 benchmark receipts
record 580.x drivers. The install replay, our scripted run of this page, used
**580.178.04**. That is the version we tested, not a minimum.

One Spark needs no fabric cable and no RDMA setup. The fabric cable is the
direct high-speed cable between linked Sparks.

### Disk space

The model needs **102,262,310,289 bytes** (102.26 GB / 95.24 GiB). That is
the complete checkpoint at one pinned revision, with its configuration and
tokenizer files. The
[committed SHA256 manifest](cli/examples/qwen3.8-flash-next-fp4-local.sha256)
lists every file. We checked its sizes against the
[publisher's file list](https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300?recursive=true)
for that revision.

Beyond the model, leave room for Rust, the source, the build output and
download caches. Before it starts, the launcher checks for free space:

- **8 GiB free where Vade installs**
- **1 GiB free where the model lives**
- **9 GiB free with the model already there**, if both are on the same disk

These checks are minimums. Vade can use more disk than this.

As an example, one of our source installs used **0.72 GiB** for the checkout,
the build output and a fresh `VADE_HOME`. `VADE_HOME` is Vade's private
folder, `~/.vade` by default. The Rust compilers and the model were already
installed. So that figure leaves out Rust and the download caches. It is also
not the most that an install can use at once. We have not measured disk use
with a fresh toolchain and its downloads.

## Supported systems

| What | Supported | Notes |
| --- | --- | --- |
| Runs the engine | DGX Spark / GB10, Linux `aarch64` | No other GPU or OS runs the engine. |
| Route A: your terminal | Any SSH client: Windows, Linux or macOS | CLI, engine and chat all run on the Spark. |
| Route B: controller computer | Linux `aarch64` or `x86_64`, macOS | Needs Rust, `ssh`, `rsync`, `curl` and `shasum`. Tested from macOS, and from Linux `aarch64` with a Spark as its own controller. |
| Windows as controller | Not natively | The CLI does not build for Windows. Use Route A over SSH, or a Linux controller inside WSL 2 (untested). See [From a Windows computer](#from-a-windows-computer). |

Chat includes a coding agent that can read, search and edit files in your
workspace. On Linux, which includes every Route A session, it cannot run shell
commands or its automatic workspace checks, because those need the macOS
sandbox. Reading, searching, line edits and patches still work.
[Chat and the coding agent on Linux](#chat-and-the-coding-agent-on-linux) has
the details.

### From a Windows computer

Windows cannot run the engine or build the `vade` CLI. It can still drive a
Spark in two ways.

**Route A over SSH.** We recommend this way. Windows 10 and 11 include an
OpenSSH client. Open PowerShell or Windows Terminal and run
`ssh you@spark-address`. Then follow [Route A](#route-a-on-this-spark) in that
session. Everything runs on the Spark, and Windows is only the terminal.

To reach the API from a program on Windows, forward the port when you connect:

```powershell
ssh -L 18080:127.0.0.1:18080 you@spark-address
```

While that session stays open, `http://127.0.0.1:18080/v1` on Windows reaches
the engine on the Spark. If `vade status` shows a different port, use that one
on both sides.

**A controller inside WSL 2.** WSL 2 runs a real Linux, usually on `x86_64`.
Route B supports Linux `x86_64` controllers. So far we have only type-checked
this path: `cargo check --all-targets --target x86_64-unknown-linux-gnu`
passes for the CLI and the harness. Nobody has built, linked or run a
controller inside WSL 2 yet. If you try it, follow
[Route B](#route-b-from-a-controller) from the WSL shell. Keep your SSH keys
inside WSL.

**Native Windows** is not supported. The CLI and the harness use the Unix
parts of Rust's standard library: file permissions, process groups and file
descriptors. Two engine crates that the CLI links use them too: the model
reader and the benchmark client. None of these compile for
`x86_64-pc-windows-msvc`. Nobody has tried to start `ssh` and `rsync` from a
native Windows build.

## Step 1: prerequisites

There are two ways to get the Vade programs:

- **Build from source.** Route B needs this for the controller's CLI and chat
  program.
- **Use the prebuilt bundle** (see [Prebuilt bundle](#prebuilt-bundle) in
  step 2). The bundle of main is published on dl.vade.lol, and the installer
  uses it when it fits. The bundle is for running Vade on the Spark itself
  (Route A).

Source builds need both pinned Rust toolchains. The model download helper
needs host Rust 1.99.0. Route A bundle users with an existing verified
checkpoint can skip Rust setup.

<a id="1-prepare-the-sparks"></a>
On the Spark, install any missing Ubuntu packages. This step needs `sudo`:

```sh
sudo apt-get update && sudo apt-get install -y build-essential git curl rsync zstd
```

The model download helper in step 3 uses Rust 1.99.0 and curl. Now check
the driver and the tools you already have:

```sh
uname -m
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
command -v cc git ssh rsync curl sha256sum find sort nvidia-smi
```

The output must show `aarch64`, GB10 and no running compute application. Also
run `nvidia-smi` alone. Its header must show `CUDA Version: 13.0` or newer.

Each of the 9 names in the `command -v` line must print a path. A name with no
path is a missing tool. Install any missing tool with `apt-get`. The `ssh`
command is in the `openssh-client` package.

If something else uses the GPU, stop it through its own launcher first. Vade
does not change the driver or the system. Leave those changes to the Spark's
administrator.

<a id="2-install-the-pinned-rust-compilers"></a>
<a id="install-rust-source-builds-only"></a>
### Install Rust

The engine is written in Rust and has no third-party Rust dependencies.

If `~/.cargo/bin/rustup` already exists, skip to the toolchains below. It can
be there even when a new shell cannot find it, because the rustup installer
leaves your shell startup files alone. Otherwise, download the installer and
read it before you run it:

```sh
curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs
less /tmp/vade-rustup-init.sh
```

Then install it without touching your shell startup files:

```sh
sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path
```

Install the pinned host toolchain on each machine that builds from source or
runs the model download helper:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install 1.99.0 --profile minimal --no-self-update
```

Source builds also need the pinned GPU toolchain:

```sh
rustup toolchain install nightly-2026-10-01 --profile minimal --no-self-update --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

`--no-self-update` keeps rustup itself as it is. `scripts/install.sh` does
the same. The installer runs and prints these same commands.

### Extra tools for a controller or linked Sparks

A controller needs the host toolchain, a native linker, Git, OpenSSH, rsync,
curl and `shasum`. macOS includes `shasum`. On Ubuntu, `shasum` comes with
`perl`.

For linked Sparks, each machine must already have `ping`. On Ubuntu, `ping`
is in the `iputils-ping` package. The network is a separate job for the
administrator. See [network setup](docs/network-setup.md).

## Step 2: get the code

<a id="3-clone-and-build"></a>

On the Spark, clone the repository into your home folder and stay at its root
in this shell. Later steps assume the checkout is `~/vade`:

```sh
cd ~ && git clone --branch main https://github.com/wappy-x/vade.git && cd ~/vade
```

This gives you the latest `main`: v0.1.0 plus the changes listed under
Unreleased in the [changelog](CHANGELOG.md). Build main to use the Unreleased
changes. Clone the v0.1.0 tag (`--branch v0.1.0`) to build the v0.1.0
release.

Once the repository is public, this clone needs no account or token. Do not
put credentials in the clone URL. We recommend a clone. A downloaded source
archive also works, from GitHub's Download ZIP or from `git archive`. The
archive records its commit as the build ID.

### Run one kernel build (source builds only)

If you build from source, run one kernel build now. Run it while you are
online, after you install the pinned toolchains above. Bundle users skip this
build:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
(cd engine && CARGO_NET_OFFLINE=false cargo +1.99.0 kernels)
```

Expect compiler warnings from the kernel crate. The build worked if it ends
with `Finished` and exit status 0. This step needs no GPU.

You run this build for its downloads. The first time it runs, the kernel
compiler downloads its `build-std` dependencies into Cargo's cache. Later,
`vade launch` builds the kernels again, offline, in its own private copy of the
source. It does not reuse this build's output, only those downloads. Without
them, the first launch on a fresh machine fails at the kernel build.
`scripts/install.sh` does not run this build: it fetches only the
`build-std` files, with `cargo fetch`.

### Prebuilt bundle

<a id="optional-release-bundle"></a>
The prebuilt bundle holds `vade`, the engine (`flash-serve`), the GPU kernels
(`flash.ptx`) and the chat program (`vade-harness`), built from one commit of
main for the GB10. With it, a Spark needs no Rust and no build.
`scripts/install.sh` uses it when it fits. You do not need to download it
yourself.

**The bundle of main is published on dl.vade.lol.** The installer downloads
the bundle that `latest.json` names for the `main` channel. The older v0.1.0
launcher bundle on the GitHub release page is not main, and the installer
does not use it.

The installer does these checks in order. When one fails, it prints one
`Prebuilt:` line with the reason and builds from source:

1. The tools it needs (`tar`, `gzip`, `sha256sum` and a few others) are
   there, and the system is Ubuntu 24.04 or newer.
2. It reads the local glibc version (`getconf GNU_LIBC_VERSION`) and the
   NVIDIA driver version (`nvidia-smi`).
3. It downloads `latest.json` for the channel. This record names the commit,
   the file, its SHA-256 hash, the oldest driver and the oldest glibc.
4. It compares the local glibc and driver with the oldest versions in
   `latest.json` (driver 580: the driver branch of CUDA 13.0).
5. It downloads the file and compares its SHA-256 hash with `latest.json`. A
   stalled download (under 1 KB/s for a minute) or a file over 512 MiB stops.
6. It checks that the file has only files and folders inside one folder,
   unpacks it to `~/.vade/bundles/<name>` (under `VADE_HOME` when you set it)
   and checks every file against the hashes in the bundle's `MANIFEST`.
7. It runs `vade install` from that folder. `vade install` checks the hashes
   in `launcher-bundle.conf` again and copies the engine, the kernels and
   chat into `~/.vade`. When an earlier install has other files, from a
   source build or another bundle, it replaces them.

Two settings choose the download. `VADE_DL_BASE` is the download location,
and `VADE_CHANNEL` is the channel (default `main`). The installer reads
`$VADE_DL_BASE/$VADE_CHANNEL/latest.json`, then the file in the same folder.
The default location is in [scripts/install.sh](scripts/install.sh). A
folder that is already in `~/.vade/bundles` and passes the `MANIFEST` check is
used again without downloading the bundle again. The installer still
downloads `latest.json`, so this step needs the network.

The file is named `vade-<version>-<sha7>-linux-aarch64-gb10.tar.gz`, and it
unpacks to one folder with the same name. It is a gzip file. Ubuntu marks
`tar` and `gzip` as essential packages, so every system has them. The folder
has these files:

| File | What it is |
| --- | --- |
| `vade`, `flash-serve`, `vade-harness` | Linux aarch64 programs |
| `flash.ptx` | The GPU kernels as PTX for `sm_121` (the GB10) |
| `launcher-bundle.conf` | The launcher record that `vade install` reads ([bundle details](cli/docs/prebuilt.md)) |
| `MANIFEST` | The commit, its date, the target, the PTX ISA version, the oldest driver and glibc, and the SHA-256 hash of every file |
| `README-BUNDLE` | Short instructions |
| `LICENSE`, `NOTICE`, `engine/` | The license and notice files of the source |

On 2026-10-31, the kernels of main use PTX ISA 8.8 (the `.version` line of
`flash.ptx`, which `MANIFEST` records as `ptx_isa`). CUDA 12.9 introduced it
([PTX ISA 8.8](https://docs.nvidia.com/cuda/archive/12.9.2/parallel-thread-execution/index.html)),
and the oldest Linux driver for CUDA 12.9 is 575.51.03 (Table 4 of the
[CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)).
Vade itself needs a driver for CUDA 13.0, branch R580 (Tables 2 and 3 of the
same page). The driver compiles the PTX for the GPU at the first start, so
newer drivers load it too. An older driver makes the installer build from
source.

The bundle replaces the Rust builds, not the model download. Keep the clone,
because step 3 uses its fetch helper and the committed model manifest.
[Launch from the bundle](#launch-from-the-bundle) in Route A shows how to start
it.

A controller (Route B) uses the bundle only when the controller checkout is at
the bundle's commit. To use it, unpack the bundle on the controller and set
`prebuilt_dir` to the bundle folder. With any other commit, `vade install`
builds from source on each Spark. See [bundle details](cli/docs/prebuilt.md) and
[first run](harness/docs/first-run.md).

## Step 3: get the model

<a id="3-fetch-and-verify-the-qualified-checkpoint"></a>
<a id="4-put-the-official-weights-on-both-sparks"></a>
<a id="4-put-the-official-weights-on-each-spark"></a>
Use publisher revision
[`b9ce18a6887e18c089fdea8509a33e63429a3300`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300).
Keep the checkpoint complete, with the index, the tokenizer and the
configuration. Use the files exactly as published. Do not convert or
re-quantize them.

The files at this revision download **without a Hugging Face token or
click-through access approval**. The publisher's
[NVIDIA Open Model Agreement terms](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/blob/b9ce18a6887e18c089fdea8509a33e63429a3300/README.md)
still apply.

### Download

The checkpoint is 102.26 GB. On 2026-10-24, rented GB10 VMs downloaded it at
8 to 22 MB/s: about 80 minutes at the top of that range and over three hours
at the bottom
([download rates on rented VMs](docs/benchmarks.md#rented-cloud-gb10-vms)).
Over SSH, a dropped connection stops the download. So run it under `nohup` as
below, or inside `tmux` if you have it. If the complete checkpoint is
already in `~/models/nvidia/Qwen3.8-Flash-Next-FP4-Local`, skip the download
and go straight to the manifest check below.

Use `vade pull`. It shows its progress, continues after an interruption and
checks every file.

If you ran `scripts/install.sh`, first run the `export PATH=...` line that it
printed. That line puts `vade` on `PATH`, from the bundle folder in
`~/.vade/bundles` or from `cli/target/release`. After a bundle install there
is no `cargo`, so the second line below works only for a source build. If you
did not run the script, the second line builds `vade` with the host toolchain
from step 1. Run these lines from the checkout root on the Spark:

```sh
export PATH="$HOME/.cargo/bin:$PWD/cli/target/release:$PATH"
command -v vade >/dev/null || cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
nohup vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300 > ~/vade-pull.log 2>&1 &
```

To follow it, run `tail -f ~/vade-pull.log` (`Ctrl-C` stops only `tail`). A
good run prints `Verified 32 files, 102.26 GB in` and the model folder. If
it stops, run the same `vade pull` line again. It continues from where it
stopped. It writes the `.sha256` list beside the model folder, so the first
`vade launch`, `vade up` or `vade install` checks the files.
[Download models](docs/pull-models.md) has the details.

#### Or use the fetch helper

`ops/fetch-model.sh` is the older downloader. It prints nothing until it
finishes, and its output goes to `~/vade-fetch.log`. From the checkout root on
the Spark, run this fetch exactly as written:

```sh
nohup sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" > ~/vade-fetch.log 2>&1 &
```

To follow it, even from a new SSH login, run `tail -f ~/vade-fetch.log`
(`Ctrl-C` stops only `tail`). The helper uses a private sibling folder for
partial downloads and checks the snapshot against the pinned publisher revision. It
then writes a `.sha256` manifest next to the model folder.

A good run ends with `verified 33 files`. If it is interrupted, run the same
command again. Partial downloads stay in the sibling `.fetch` directory.
The helper stops its downloader after an interruption. It keeps an operating
system file lock until the downloader exits. A rerun can report
`another fetch may be running` during that cleanup. Wait for cleanup, then
rerun. A restart releases the lock too. The `.fetch/lock` file stays in place.
Do not remove it. The file alone does not mean that a fetch is running.
Metadata requests have a 120-second deadline. A transfer fails if its rate
stays below 1 byte per second for 120 seconds. Large active downloads can
continue for hours.

Partial downloads can use space in addition to the model files. Leave
headroom beyond the 102.26 GB checkpoint. A complete
snapshot has the byte total given above. A copy of only the tensor shards is
not enough. Vade's `install`, `up` and `launch` do not download these weights.

### Check the files

On Route A, the first `vade launch`, `vade up` or `vade install` checks every
file for you while it compiles the GPU code. It does this when it finds a
checksum list. The list is the `Qwen3.8-Flash-Next-FP4-Local.sha256` file
that `vade pull` or the fetch helper writes beside the checkpoint folder, or a
`manifest.sha256` file inside the folder.

When a file differs, the launch stops and names the file. Vade writes nothing
into the checkpoint. It keeps the result in its private state, so later starts
skip the work.

If you got the checkpoint another way, or you use Route B, check the files
against the committed manifest once yourself. This reads all 102.26 GB of the
checkpoint. From the checkout root:

```sh
MODEL_SHA256_LIST="$PWD/cli/examples/qwen3.8-flash-next-fp4-local.sha256"
(cd "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" && sha256sum -c "$MODEL_SHA256_LIST")
```

Every file must pass. On macOS, use `shasum -a 256 -c` instead.

Keep the checkpoint folder name `Qwen3.8-Flash-Next-FP4-Local`. The engine
serves the model under that name in lowercase, `qwen3.8-flash-next-fp4-local`.
It refuses requests that name any other model.

A Route B Spark has no checkout. Download the same manifest from the v0.1.0
release tag instead, then run the same check:

```sh
MODEL_SHA256_LIST="$HOME/qwen3.8-flash-next-fp4-local.sha256"
curl -fsSL https://raw.githubusercontent.com/wappy-x/vade/v0.1.0/cli/examples/qwen3.8-flash-next-fp4-local.sha256 -o "$MODEL_SHA256_LIST"
(cd "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" && sha256sum -c "$MODEL_SHA256_LIST")
```

## Route A: on this Spark

<a id="one-spark-directly-on-the-spark"></a>
If `scripts/install.sh` installed the prebuilt bundle, skip the block below
and follow [Launch from the bundle](#launch-from-the-bundle) instead. Otherwise, once the
steps above are done, run this from the checkout root in the same shell.
`vade launch` selects the local Spark and requires `VADE_CONFIG` to be unset.
If an earlier controller launch still owns an engine, stop it with that
controller's config first. Local settings go in `~/.vade/config`.

```sh
export PATH="$HOME/.cargo/bin:$PATH"
unset VADE_CONFIG
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
./cli/target/release/vade launch
```

The build creates the CLI. `launch` does the rest:

1. It finds the checkpoint under `~/models` and saves local settings.
2. It builds and installs the engine, kernels and chat program if needed.
3. It waits until the engine is ready, then opens chat.

When a checksum list is there, `launch` also does the file check in
[Check the files](#check-the-files). Later launches reuse the
installed files and the saved check. Loading the model still takes time.
When the engine is ready, it prints one line with how long the launch took,
the address, the context and output limits and the Spark count, for example
`Launch ready in 149.4 s: http://127.0.0.1:18080, context 32768 tokens, output 8192, 1 Spark.`
`vade up` prints the same line, starting with `Up ready in`, then the chat
and stop commands.

When a script runs `launch` and standard input is not a terminal, `launch`
skips chat. After the ready line, it prints
`Standard input is not a terminal, so chat was not opened. Chat: vade chat`
and exits.

Chat opens in the checkout, and the agent can edit files there. So ask it only
to say hello. Then type `/exit` to get back to your shell. The engine keeps
running.

The chat window needs an interactive terminal. A normal
`ssh you@spark-address` login is one. To open chat with a single SSH command
instead, add `-t` and name the CLI by its path. That new shell does not have
the `PATH` line below.

```sh
ssh -t you@spark-address 'cd "$HOME/vade" && ./cli/target/release/vade chat'
```

For the prebuilt bundle, use `'"$HOME/.vade/bundles/<name>/vade" chat'`
instead, with the folder name that the installer printed.
Without `-t` there is no terminal, so chat shows plain lines instead of its
window.

`cargo` and `vade` are on your `PATH` only in the shell that ran the block
above. In every new shell, such as a new SSH login, run this line first:

```sh
export PATH="$HOME/.cargo/bin:$HOME/vade/cli/target/release:$PATH"
```

Next, run these in order:

```sh
vade status
vade doctor
vade chat
```

Type `/exit` again. Then check the engine and stop it.

If you may later use this clone as a controller, first change to another
folder, such as your home folder. `vade verify` and `vade bench` write their
receipts into the current folder, and Route B's `vade install` refuses a clone
with untracked files.

```sh
cd ~
vade verify
vade down
vade status
```

After the last command, `vade status` says that the engine is stopped.
Restart with `vade up`, or with `vade launch` to start and chat. See
[first run](harness/docs/first-run.md) for the whole lifecycle.

### Check the whole route in one go

[`harness/scripts/route-a-check.sh`](harness/scripts/route-a-check.sh) runs
Route A from start to finish, with no chat window. In order, it:

1. checks the toolchains and the GPU, and builds the CLI;
2. asks `vade detect` which model `vade launch` will load;
3. runs `vade launch` and sends one chat request through the API;
4. runs `vade verify` against that same engine, then `vade down`.

Run it from the checkout root with the GPU idle. Stop any running engine with
`vade down` first:

```sh
sh harness/scripts/route-a-check.sh
```

Each step prints PASS or FAIL with its elapsed seconds, and the last line gives
the overall result. Step logs and the verify receipt go to a new private
folder under `$TMPDIR` (or `/tmp`), so the checkout stays clean.

The check uses a new private Vade home of its own, `~/.vade-route-a.XXXXXX`,
instead of `~/.vade`. It never reads or stops your own installation, and no
other `vade` command finds its engine. Because that home starts empty, the
launch is a first setup and builds the engine and kernels again, however often
you have launched before. At the end, and also when a step fails or the script
is interrupted, it tries to stop its engine and reports FAIL if that stop
fails. After a PASS, it removes the private home. After a FAIL, it keeps the
private home for inspection and prints its path.

Set `VADE_MODELS` to pick the checkpoint, as for `vade launch`. The model step
fails when `vade` would load a checkpoint outside it. `VADE_DRY=1` prints the
commands without running anything, on any computer.

### Launch from the bundle

After the model check in step 3, run this instead of Route A's source-build
block. First run the `export` line that `scripts/install.sh` printed. It puts
the bundle folder, `~/.vade/bundles/<name>`, on your `PATH`. Then run these
commands:

```sh
unset VADE_CONFIG
vade launch
```

Then follow Route A from "Next, run these in order". With the bundle folder on
your `PATH`, `vade status`, `vade verify` and `vade down` work by name, in any
folder. In a new shell, run the `export` line first.

### What `vade verify` checks

Drafting is how the engine guesses tokens ahead so answers arrive sooner. It
must not change the answer, and `verify` checks that. It repeats requests with
drafting off and on, compares the token IDs, and writes `vade-verify.json` in
the current directory.

A pass prints `IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the
reply's token count. That pass covers this set of requests. It is not the full
cross-Spark exactness check in
[architecture](docs/architecture.md#exactness-and-model-quality). That check
needs the same token IDs, with the same settings, on every run and on every
Spark.

### Where the API listens

The default API is `http://127.0.0.1:18080/v1`. If that port is taken, first
setup picks another free loopback port and saves it as `port=` in
`~/.vade/config`. Later launches reuse it. `vade status` prints the base
address, such as `http://127.0.0.1:18080`. The API is under `/v1` at that
address.
Without `--url` or `VADE_API_URL`, local `vade verify` reads the running
engine's port from `~/.vade/state/run.conf`. That record exists only while the
engine runs.

The model thinks before it answers unless the request turns that off. The
reasoning counts against the same `max_tokens` as the answer. To get the
answer alone, send `"enable_thinking": false` or `"reasoning_effort": "none"`.
`vade chat`, `vade verify` and `vade bench` already turn thinking off.

### Where Vade keeps its files

A local install keeps its files here:

| Path | What it holds |
| --- | --- |
| `~/.vade/config` | Local settings |
| `~/.vade/stage` | The source copy Vade builds from, the built engine, GPU kernels and chat program |
| `~/.vade/state` | Records of the engine Vade started (so `vade down` stops the right process), and logs |
| `~/.vade/projects` | Chat history, never in the folder you chat from (`VADE_CHAT_IN_PROJECT=1` keeps it in a git-ignored `./.vade`) |

Use `vade logs` to read engine diagnostics. `VADE_HOME` picks a different
private root. Keep it the same through launch, chat, verify and shutdown.

### Context size on one Spark

Context is the prompt plus the reply, counted in tokens. On the Spark it starts
at **32768** tokens, and output at **min(context, 8192)**. These are the same
defaults as a controller's `vade up` in Route B below.

The engine checks whether that fits in memory. If it refuses the default or
saved context for lack of memory, `vade launch` and `vade up` try smaller
standard sizes in order: 32768, 16384, 8192, 4096. They skip sizes at or above
the refused context and save the first size that starts. Vade prints one line
that tells you:

- the size it settled on
- the sizes that were refused
- any lowered output limit
- the `--context` flag that retries your original size

If no size starts, Vade does not change your saved config.

If you set the output yourself, the cap is **16384**, and the output must fit
in the context. If the engine refuses a `--context` that you set yourself, you
get an error with the engine's message. Vade never lowers it silently.

There are two other defaults. Drafting is on, and the engine chooses how far
ahead it drafts. The prefix cache is on: the engine reuses the start of an
earlier prompt.

Input and output share the context, so the real limit for each depends on the
other. The engine's `/health` endpoint shows the limits that apply. `--chats`
works only from a controller. On the Spark itself, Vade refuses it.

### Chat and the coding agent on Linux

Chat uses the agent when context is at least 8192 tokens. Smaller local
contexts get direct text chat.

As [noted above](#supported-systems), the agent's shell tool
(`workspace_shell`) and `--check` command execution need the macOS sandbox. On
Linux they are unavailable, and the launcher turns off automatic workspace
checks. Chat and the agent's other tools still work, so reading files and
applying approved patches work as usual.

On Linux the agent is not offered the shell tool at all. Its instructions say
it cannot build, test or run anything here, so after an edit it re-reads the
changed lines and tells you which build or test command to run yourself.
`/status` in chat shows `no command tool here`, and `--child-shell` is refused
with `--child-shell needs the macOS sandbox`.

Leaving chat keeps the engine running until `vade down`.

## Route B: from a controller

<a id="5-start-the-two-spark-server"></a>
<a id="5-start-and-chat"></a>
A controller is another Linux or macOS computer that drives the Spark over
SSH. Before you start:

- The controller has the prerequisites from step 1.
- It has its own clone of the same source, made with step 2's clone command.
  The controller skips the kernel build.
- Each Spark has the verified checkpoint from step 3.
- The controller can SSH into each Spark without typing a password.

From the controller's checkout root, build the CLI, then let `vade setup`
write the config and start the Spark:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
vade setup
```

Setup asks for the Sparks (what you type after `ssh`, in cable order), the
checkpoint (for 2, 4 or 6 Sparks) and the model folder on each Spark. Each question shows its default
in brackets. For a script or an agent, give the answers as flags:

```sh
vade setup --dry-run --yes --hosts you@spark-a-address
vade setup --yes --hosts you@spark-a-address
```

The dry run prints the config that setup would write and the steps it would
run, and changes nothing. Setup writes the config at the default path below,
with this checkout as the source and private folders under `~/.vade`. It
never replaces a config. A second run uses the file that is there. Then it
runs `vade install`, `vade up` and `vade doctor`, and stops at the first step
that fails. Fix the cause, then run `vade setup` again. When the config
already has an install, the second run skips `vade install`, so `vade up`
can attach to an engine that runs. After you pull new code, run
`vade update`.

To write the config by hand instead, start from the example:

```sh
mkdir -m 700 -p .local
[ -e .local/fp4-local.conf ] || cp cli/examples/fp4-local.conf .local/fp4-local.conf
export VADE_CONFIG="$PWD/.local/fp4-local.conf"
```

`VADE_CONFIG` points the CLI at that file. Without it, a controller reads its
default config: `~/.config/vade/config` on Linux,
`~/Library/Application Support/vade/config` on macOS. `XDG_CONFIG_HOME` or
`VADE_CONFIG` overrides it.

### Edit the config

If you copied the example, edit that private file before you go on. A config
that `vade setup` wrote is ready to use. Edit it only to change the setup.

**Local folders.**

- Set the local `engine_source`, `harness_source` and `cli_source` to this
  checkout's component folders.
- Pick private local `stage_dir` and `state_dir` paths. Put `state_dir` under
  your home, as the example does (`~/.vade-state`). It holds the record that
  `vade status` and `vade down` use to find a running engine. macOS clears its
  temporary folders, so a `state_dir` there could lose that record.
- On macOS, `/tmp` is a symlink and is refused. Only a throwaway `stage_dir`
  may live under the system temporary directory.
- These folders must be private to you. Let Vade create them, or make them
  yourself with `mkdir -m 700`. A folder made with a plain `mkdir` is usually
  open to other users, and `vade install` refuses it.

**Model.** Keep the supplied model, model repository and pinned
`model_revision`.

**The Spark.**

- In `[machine spark-a]`, set `ssh` and the absolute `model_path`.
- `ssh` is whatever you would type after `ssh`: `you@spark-a-address` (a host
  name or IPv4 address, with or without `you@`) or an existing `~/.ssh/config`
  alias.
- Choose dedicated remote source, stage and state directories owned by that
  user.

**Ports.** Keep the default `local_port` and `serve_port` of **18080** unless
something already uses them. The first is on this computer. The second is on
the Spark that serves. If the controller is one of the Sparks, both would land
on the same machine, so set `local_port` to another free port, such as 18181.

### Install and start

`vade setup` runs these steps for you. To run them by hand, preview each step
with `--dry-run`, then run it:

```sh
vade install --dry-run
vade install
vade up --dry-run
vade up
vade status
vade doctor
curl --fail-with-body --silent --show-error http://127.0.0.1:18080/v1/models
vade chat
```

`install` stages the tracked source, builds the engine and helper on the
Spark, and builds the harness and CLI locally. It refuses a checkout with
local changes or untracked files, so keep the controller's clone clean (see
[first-run problems](#first-run-problems)).

`up` picks FP4-Local by default (`--fp4-local` is optional). It starts the
engine, opens an SSH tunnel, and records both so that `vade down` can stop
them.

The curl command assumes you kept the example's port. Its JSON reply lists the
served model ID. Keep `VADE_CONFIG` pointing at this same file.

On macOS, the agent's sandboxed shell sees only system and Homebrew tools. So
it cannot find a Rust that you installed with rustup in your home folder. You
see `sandbox-exec: execvp() of 'cargo' failed`. `vade chat` has no option for
this. Instead, run `vade chat --dry-run` to print the command it would run.
Then run that command yourself, and add
`--shell-toolchain ~/.rustup/toolchains/<toolchain>` at the end. That folder
contains `bin/cargo`.

After `/exit` from chat:

```sh
vade verify
vade down
```

`vade verify` leaves `vade-verify.json` in the current folder. Move it out of
the checkout before your next `vade install`, which refuses untracked files.

### Defaults on a controller

A controller asks for the same defaults as Route A: context **32768** and
output **min(context, 8192)**. That holds however many Sparks you use, and
whether or not you pass `--chats`. Drafting is on, and the engine picks how far
ahead it drafts. The prefix cache is on by default. Three or more Sparks run
without it.

A controller never lowers the context for you. If a Spark lacks the memory for
32768, the engine refuses it, and you choose a smaller `--context`. Only
Route A lowers it automatically, and only when the engine says that there is
not enough memory.

If you want `vade chat`, stay at 8192 or above. From a controller, `vade chat`
always opens the coding agent, which needs at least 8192 tokens. With a
smaller context, it stops with an error and does not fall back to plain chat.

The engine decides how many chats can run at once from context and memory. It
reports that number when it starts, and the controller's `vade status` shows it
as `Concurrent chats`. If the report is missing, status says the count was not
reported. `--chats` does not change it on FP4-Local. The `--chats` row in the
[flag table](#settings-you-may-change) below has the details.

For a larger context, start with `vade up --context 262144`. For longer
replies, add `--output 16384`. The `--context` flag sets the context, not the
chat count. Check `Concurrent chats` in `vade status` for the admitted count.
The controller starts at 32768 to leave more room for concurrent chats.
Measured speeds at 32768 are on the
[benchmarks page](docs/benchmarks.md).

## Two and more Sparks

<a id="fp4-layer-chain"></a>
Several Sparks can work together in two ways:

| Use | When | Start it with |
| --- | --- | --- |
| A chain: one engine spread across the Sparks | You want long prompts read faster, or the model is too big for one Spark | `vade up --sparks N` or `--hosts` |
| [Replicas](#replicas): a separate copy on each Spark | The model fits on one Spark and you want many chats | `vade up --replicas N` |

Both use Route B. The setup below is for the chain. Replicas need less of it.

### Describe each Spark in the config

Add a `[machine name]` section for each Spark. The
[example configuration](cli/examples/fp4-local.conf) has a commented-out pair.

For each link between two adjacent Sparks, put a `fabric.<peer>` line in both
machine sections. Its value is that machine's **own address** on the link
toward the peer. Those addresses must already be set up and reachable.

Every rank (each Spark that runs part of the engine) needs the same expanded
absolute paths for the checkpoint and for the staged GPU kernel files (PTX).
The same `~/` text is not enough when home folders differ between accounts.

### Check SSH from the controller

The controller reaches every other rank over SSH. When the controller is one
of the Sparks, Vade runs that Spark's commands on it as local processes, with
no SSH to itself, so that Spark needs no key for itself. Vade finds that
Spark by its `ssh =` entry: `localhost`, a loopback address, one of its own
addresses, its host name, or an alias that `ssh -G` resolves to one of these.
Only one machine in the config can name the controller. When that machine is
rank 0, keep `local_port` equal to `serve_port` (the default), because its
engine then serves the endpoint itself. Plain `ssh = you@spark-a-address` and
`ssh = you@spark-b-address` work for every rank with no `~/.ssh/config` alias.

Before `vade install`, check each other rank from the controller exactly the way
Vade connects, with no prompt:

```sh
ssh -o BatchMode=yes you@spark-a-address true
ssh -o BatchMode=yes you@spark-b-address true
```

`Host key verification failed` or `Permission denied` means one of two things.
Either that machine does not trust the controller's key yet, or the controller
does not know that host key yet. Fix it with your usual SSH key setup. Vade
never edits SSH settings.

You need an alias only for settings that a plain destination cannot carry,
such as a jump host, a non-default port or a specific key. For that case,
write a private `~/.ssh/config` block like this one. Then write
`ssh = spark-b` in the Vade config:

```text
Host spark-b
    HostName spark-b-address
    User you
    ProxyJump you@spark-a-address
    IdentityFile ~/.ssh/your-key
```

### Install and start the chain

`vade setup` can write the config for a chain. Give the Sparks in cable order
and, with `--link`, each Spark's own address on each cable:

```sh
vade setup --hosts spark-a,spark-b \
  --link spark-a:spark-b=192.0.2.1 --link spark-b:spark-a=192.0.2.2
```

Without the `--link` flags, setup asks for each address. It then installs and
starts the chain with `vade up --hosts` in that order. The steps below do the
same by hand.

Reserve these ports on the chosen Sparks:

- HTTP port **18080**
- transport ports **29960** (`tp_port`) and **29961** (`tp_port + 1`)

Keep `serve_port` different from both transport ports. Then, from the
controller, after stopping any earlier run:

```sh
vade install
vade up --sparks 2
```

`vade install` builds from source on each Spark, unless `prebuilt_dir` points
at a bundle from the controller checkout's own commit (see
[bundle details](cli/docs/prebuilt.md)). A bundle from another commit does
not qualify. After `vade up --sparks 2`, `vade verify` reports `1 machine`,
because the two ranks form one engine, not two separate engines.

There are two ways to choose the Sparks. Use one or the other, not both:

- `--sparks N` picks the first N configured machines.
- `vade up --hosts spark-a,spark-b,spark-c` picks named machines in cable
  order.

With FP4-Local, two Sparks run the tensor split, the only two-Spark path. Both
Sparks work on every token, and each one works on half of the experts and of
most projections.

With more Sparks over RDMA, Vade runs the tensor split in parts when two
things are true. Every split width of the model cuts into whole parts for that
many Sparks, and the mailbox has room for that many ranks. For this
checkpoint, that is four Sparks, with no prefix cache.

Three, five or more Sparks, and any three or more over TCP, use the layer
chain. The layer chain spreads the layers across all the Sparks that you
chose. FP8 runs tensor parallel on two and on four Sparks instead. See
[FP8 on 2, 4 or 6 Sparks](#fp8-on-2-4-or-6-sparks).

List both end addresses for every cable between neighboring Sparks. You do not
need a cable that closes the chain into a ring.

If nobody has tested your cabling layout, the checks that Vade runs while it
builds the chain do not prove exactness. Exactness means the same token IDs,
with the same settings, on every Spark.

### RDMA or TCP

Linked Sparks use RDMA by default. RDMA needs the RDMA (verbs) device and its
user library to be set up already. It also needs a locked-memory (memlock) limit of
`unlimited`, both soft and hard (`ulimit -Sl` and `ulimit -Hl`), in the SSH
session that Vade uses.

For TCP instead, set this variable **before install**. Keep it set for `up`
and `doctor`:

```sh
export SEQPAR_RDMA=0
```

TCP uses the configured, reachable fabric addresses and needs neither RDMA nor
a fabric cable (a QSFP link). A single local Spark needs no transport between
Sparks at all. The two-Spark tensor split needs RDMA, so `vade up` refuses
TCP on two FP4-Local Sparks. TCP serves FP4-Local chains of three or more
Sparks, and FP8, which keeps its ring on two Sparks.

<a id="replicas"></a>
### Serve many chats with replicas

When the model fits on one Spark, several Sparks serve many chats better as
independent copies than as one chain. `vade up --replicas 2` starts a separate
one-Spark server on each of the first two machines (or the machines named in
`--hosts`). It also starts a local gateway on `local_port` that sends each
request to the replica with the fewest requests in flight.

The gateway also tries to keep each conversation on one replica. Every turn of
a chat has the same key: its `prompt_cache_key`, else a session header, else
the model, instructions, system prompt, tools and messages through the first
user message.

A later turn is anything after the first user message: another user turn, a
reply or a tool result. A later turn goes to the replica that served the
previous turn while that replica has fewer than 8 requests in flight or is
among the least busy. A first turn uses that replica only while it is among
the least busy, so identical prompts sent at the same time divide evenly.
Otherwise the least busy replica wins. The prefix cache is on by default, but
we have not confirmed that this routing gives cache hits.

A single chat runs at one-Spark speed. The total throughput grows with the
number of replicas. Every replica is the same one-Spark engine, so a request
gets the same token IDs whichever replica answers.

Replicas need no `fabric.*` entries or transport ports. Their tunnels use the
local ports right after `local_port`. Each replica starts with the controller
defaults (context 32768, output `min(context, 8192)`).

To verify the replicas, run `vade verify --url <replica address>` once for
each one. With the default `local_port` of 18080, the first replica's tunnel
is `http://127.0.0.1:18081` and the second's is `http://127.0.0.1:18082`. A
plain `vade verify` goes through the gateway, so it checks whichever replica
the gateway picks.

`vade status` reports the gateway. `vade down` stops the gateway, the tunnels
and every replica.

## Settings you may change

Stop the engine before you change launch settings. Then run `vade up` again
with the flags you want. Supported local overrides are saved in
`~/.vade/config`. On a controller, pass the flags explicitly each time you
launch.

<a id="defaults-by-route"></a>
These defaults apply when no saved override, launch flag or environment
setting replaces them. The environment settings are `MTP` and the
prefix-cache variable (`VADE_FLASHNEXT_PREFIX_CACHE`). Context covers both
the input and the generated output. `/health` reports the limits that apply
to a request. Drafting is multi-token prediction (MTP).

- **On this Spark, with `vade launch` or `vade up`.** The context is 32768
  tokens. Vade lowers it through 16384, 8192 and 4096 only if the engine
  refuses it for memory. The output is `min(context, 8192)` tokens. Drafting
  is on, and the engine selects the depth. The prefix cache is on.
- **From a controller, with `vade up` on any number of Sparks.** These
  defaults are the same with or without `--chats`. The context is 32768
  tokens, and the output is `min(context, 8192)` tokens. Drafting is on, and
  the engine selects the depth. On one or two FP4-Local Sparks and on FP8,
  the prefix cache is on. It is off on three or more FP4-Local Sparks:
  the layer chain, or the split in parts on four Sparks over RDMA.
- **From a controller, with `vade up --replicas N`.** Each replica has a
  context of 32768 tokens and an output of `min(context, 8192)` tokens.
  Drafting is on, and the engine selects the depth. The prefix cache is on.

`--context 262144` requests a larger context on either route. For longer
replies, add `--output 16384`. The engine still sets the chat count from
memory. An explicit context must pass the engine's memory check. If the
engine refuses it, Vade reports the refusal and does not lower the context.

| Flag | What it does |
| --- | --- |
| `--context` | How many tokens the prompt and reply can use together. On the Spark, the value must fit the checkpoint's limit and pass the engine's memory check. If the engine refuses a value you set, you get an error. Vade does not lower it. The defaults are in the list above. |
| `--output` | The most tokens a reply can use. It cannot be larger than the context or the FP4-Local limit of 16384. |
| `--mtp` | Asks for a specific drafting depth. `0` turns drafting off. It overrides the `MTP` variable. A local launch saves the selected drafting depth for later starts. To restore engine-selected drafting, stop the engine, remove the `mtp=` line from `${VADE_HOME:-$HOME/.vade}/config`, unset `MTP`, and restart without `--mtp`. A controller uses the engine default when neither the flag nor `MTP` is set. |
| `--chats` | It works only from a controller. On the Spark itself, Vade refuses it. Vade records it and passes it on, but FP4-Local sets the chat count from context and memory. So `--chats` changes neither the context nor how many chats run at once. With FP8, `--chats N` reserves room for N chats. |
| `--memory-bytes` | Sets a memory budget. The engine's memory check and the memory actually free still apply. |
| `--cache-bytes 0` | Turns the prefix cache off for a local install. Any local size above zero is refused. |

### The prefix cache

The prefix cache reuses a cached prompt prefix across requests. It is on by
default on one Spark, on two FP4-Local Sparks (the tensor split) and on FP8.
FP4-Local on three or more Sparks (the layer chain, or the split in parts on
four Sparks over RDMA) runs without it.

The cache keeps prompts only for requests that send `cache:true` or a
`prompt_cache_key`. Other requests run as before. A cached answer has the same
token IDs as an uncached one. The cache changes which prompt tokens the engine
reads again, not how it computes them.

At startup, the cache takes a memory room of its own from the chat memory
(the session pool). The engine plans the chat slots so that the pool holds one
full-context chat and at least 512 MiB beside it. Near the memory limit, this
can give fewer chat slots. On one Spark at a context of 262,144, a start that
could fund 16 slots with 38,449,244 bytes left takes 8 slots.

The room is half of the pool bytes that no full-context chat can use, at least
512 MiB and at most 4 GiB, and never more than those bytes. So the room never
lowers the number of full-context chats in that plan.

On two Sparks, each engine rank keeps a copy of every kept prompt. So there
the room is at least the memory of one full-context chat, up to 4 GiB. At a
context of 32,768, a 32K-token prompt fits. Its copy was about 2.1 GB in a
two-Spark cache test on 2026-10-29 (lab report `own-cache-screen-x2-r1`, not
in the repository). This can lower the number of full-context chats by one.

Live chats never use the room, and memory pressure does not empty it. The
cache also uses memory that live chats leave free, and it gives that memory
back first. A kept prompt can hold memory beyond the room: a prompt kept in
place on one Spark holds a whole chat's memory. That memory yields to host
memory pressure 30 seconds after the last live chat, when the engine gives
spare memory back to the host. Before then, the engine would keep that memory
anyway, so the next turn of a conversation can still use the prompt. The
engine log line `flashnext ready` shows `prefix_cache_room_bytes`.

| Setting | What it does |
| --- | --- |
| `VADE_FLASHNEXT_PREFIX_CACHE=0` | Turns the cache off. `1` turns it on. FP4-Local on three or more Sparks (the layer chain, or the split in parts on four Sparks over RDMA) refuses `1`. |
| `VADE_FLASHNEXT_CACHE_ROOM=<bytes>` | Sets the room. The slot plan keeps it beside one full-context chat, which can give fewer chat slots. `0` gives no room and the slot plan of a server without the cache. A room too large to keep beside one full-context chat at two or more slots gives one chat slot. A room that leaves less than one full-context chat is lowered to leave one. |
| `VADE_FLASHNEXT_ADOPT=0` | Keeps every prompt as a copy on one Spark. `1` keeps long prompts in place on two Sparks too. We have not measured that mode on two Sparks. |
| `--cache-bytes 0` | Turns the cache off for a local install. |

On one Spark, the cache keeps a long prompt in place. When a chat ends, its
prompt memory stays in the cache, and the next turn of that conversation
builds on it. The cache can use the whole chat memory for this, and live
chats take it back first. A short prompt, whose memory is less than a
quarter of one chat's reservation, is kept as a copy. On two Sparks, every
kept prompt is a copy.

Vade copies a prompt while the prompt's chat still holds its memory. The
room and the memory that live chats leave free must hold that copy. At a context of 262,144 on
one Spark, one full-context chat reserves 16,017,732,932 bytes (the
`flashnext ready` log line of each start at that context in lab report
`own-cache-screen-1s-b-r1`, not in the repository), and the slot plan
keeps only the room (at least 512 MiB) beside it. So with copies only
(`VADE_FLASHNEXT_ADOPT=0`), the cache does not keep a 200K-token prompt. In a
test on 2026-10-29, its copy needed 12,274,478,925 bytes (lab report
`own-warm-200k-r1`, not in the repository). In place, which is the default,
the cache keeps it.

On two Sparks, both engine ranks must start with the same cache value, or
the engine refuses to start. To turn the cache off for **local Route A**,
restart from the same local environment:

```sh
vade down
VADE_FLASHNEXT_PREFIX_CACHE=0 vade up
```

Vade does not save the environment setting. Set it each time you start the
engine.
A saved `--cache-bytes 0` stays in `${VADE_HOME:-$HOME/.vade}/config` until
you remove the `cache_bytes=0` line.

Route B has no launcher flag for the FP4-Local prefix cache. A setting must
already be in the noninteractive SSH environment on every selected Spark. A
setting only in the controller shell has no effect. The engine makes the
final decision ([source](engine/crates/serve/src/prefix_switch.rs)).

`GET /health` reports the cache in one object, `prefix_cache`: hits, misses,
reused tokens, kept snapshots, entries evicted for memory (`evicted`: room for
a new snapshot, memory pressure or the cache limit), entries removed after
their idle time (`expired`: a checkpoint at its expiry, or an idle prompt when
the cache must give memory back), entries, retained bytes, the room, and the
snapshots skipped for each reason (`no_room`, `too_large`, `fork_failed`,
`insert_refused`). On two Sparks, a prompt that the second engine rank could
not capture is not counted there. The log shows `prefix_cache_worker_refused`.
The engine log prints the same counts on a
`prefix_cache_status` line after each lookup and snapshot. After an eviction,
an expiry or another change to the entries or retained bytes, it prints the
line at the next scheduler pass.

### Use a different checkpoint folder

Before the first launch, set `VADE_MODELS` to use a checkpoint outside
`~/models`. A valid saved checkpoint takes precedence. To replace it, run
`vade down` with the current `VADE_HOME`. Then set `VADE_MODELS` to the new
checkpoint folder and choose an unused `VADE_HOME` before you run
`vade launch`. Use that new `VADE_HOME` for later commands. This route
supports only the documented FP4-Local checkpoint.

## Client gateway

To put Vade's gateway between your own client and the engine, build the
harness binaries from the checkout and run:

```sh
cargo +1.99.0 build --release --offline --locked --manifest-path harness/Cargo.toml --bins
harness/target/release/vade-gateway --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

If `vade status` shows a different engine endpoint, use that one. Point your
client at `http://127.0.0.1:8787/v1`. Ctrl-C stops the gateway.

When the engine is full (16 open connections), it turns a new request away
with HTTP 503 before it does any work. `vade-gateway` passes that refusal
straight to your client. The harness's built-in gateway waits instead. Start
it in place of `vade-gateway`:

```sh
harness/target/release/vade-harness gateway --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

With one upstream, it holds only that refusal (`connection capacity reached`).
It sends the same bytes again after a short wait, for up to 30 s. So a client
that opens its next round of chats while the last one is still closing sees
only a slightly later first token. Any other 503 can come after work started,
so it reaches the client once. This gateway also passes `GET /health` to the
engine, so `vade bench --url http://127.0.0.1:8787/v1` and health checks work
through it. See the [harness documentation](harness/README.md) for the
supported API and tool behavior.

The gateway guards against web pages driving it through your browser:

- It answers 421 to any Host other than `127.0.0.1`, `localhost` or `[::1]`
  (bare or with its port).
- It answers 403 to any request with an `Origin` header.

Command-line clients that use the loopback address and omit `Origin` pass
these checks. To use a browser client, add
`--allow-origin https://app.example`. It takes the exact origin, and you can
repeat it. `--allow-host NAME` allows another Host name. We recommend that you
also add `--token TOKEN` or `VADE_GATEWAY_TOKEN`. Then every request must send
`Authorization: Bearer TOKEN`.

<a id="6-run-the-timing-benchmark"></a>
## Run the benchmark

Start the benchmark when the engine is running and has finished its
background check of the checkpoint files. At that point, `/health` shows the
checkpoint `hash_state` as `ready`.

```sh
vade bench
```

Keep the same `VADE_HOME` / `VADE_CONFIG` you used to start the engine. The
client uses `VADE_API_URL` if set. Otherwise, it uses the endpoint recorded by
the local launch or the controller's configured port. It reads the model from
`/health` and writes
`./vade-bench.json` and a `./vade-bench.txt` table into the current folder. On a
controller, run it from outside the checkout, or move both files out before
your next `vade install`, which refuses untracked files. If health does not
name the model, it falls back to
`qwen3.8-flash-next-fp4-local`.

By default, it uses a bundled offline writing prompt and asks for the server's
drafting policy. For FP4-Local, that means drafting on. When many requests run
at once, the server can answer some of them without drafting. `vade bench` then
ends with a `Drafting:` line. For each concurrency, that line names how many
replies were plain. Those rows measure plain speed. This is why a higher
concurrency can show a lower rate than the one before it.

Each receipt (the JSON result file) records the draft mode you asked for, the
server's build SHA and the drafting (MTP) depth that the server reports.
`draft_observations` keeps what each response reported. A reported depth of
zero alone does not prove that drafting was off.

To measure plain decoding on the same running engine:

```sh
vade bench --draft plain --output vade-bench-plain.json
```

This sends `"draft":"plain"` with every request. A run is marked as drafting
off only when you ask for it this way. `--draft strict` asks for strict
drafting. Neither needs a server restart.

To change the endpoint, model, receipt file or prompt, use `--url URL`,
`--model ID`, `--output PATH` or `--prompts FILE`. In a prompt file, each
non-empty UTF-8 line is one prompt.

Every request uses greedy decoding with the cache and thinking off. Each
request keeps generating past the end-of-answer token, so every reply has the
same length. `--max-tokens` sets that length. The client measures requests in
groups (cohorts). A group with a short or incomplete reply gets no rate.

This is a quick check of your own setup, not a RigMark card. Its recipe
differs from the published cards, so you cannot compare its numbers with them.
[Reproduce a measurement in detail](docs/benchmarks.md#reproduce-a-measurement-in-detail)
lists the differences and shows how to run a card.

Read [benchmark recipes and results](docs/benchmarks.md) for workloads, receipt
fields and measurement limits. A quick test or a short greeting does not
measure real speed. If you share a result, keep the exact source revision,
checkpoint, settings and raw output with it so others can reproduce it.

## First-run problems

The messages below are excerpts from the CLI and the engine. Whatever you see,
leave the files under `~/.vade/state` in place. They tell `vade down` what to
stop. Run `vade logs` and `vade doctor` before you retry.

| Message | What to do |
| --- | --- |
| `Vade is not set up yet` | Nothing is set up for this user: no local setup in `~/.vade` and no controller config. On a Spark, run `vade launch` (or the installer above). To drive Sparks from another computer, follow [Route B](#route-b-from-a-controller). |
| `This computer is not a Spark, and it has no controller config yet.` | You ran `vade install` on another computer before any setup. Run `vade setup` there to drive Sparks over SSH, or run `vade install` on the Spark itself. |
| `Standard input is not a terminal, so setup cannot ask its questions.` | `vade setup` ran from a script. Add `--yes`, plus `--hosts` and `--link` where setup names them. |
| `vade install failed, so setup stopped.` (or `vade doctor`, `vade up`) | Read the error that step printed just above, fix it, then run `vade setup` again. It reuses the config and the work already done. |
| `This command runs only on a DGX Spark (Linux aarch64)` | You ran a local command such as `vade launch` on another computer. Run it on the Spark, add `--dry-run` to preview it, or use [Route B](#route-b-from-a-controller) to drive the Spark from this computer. |
| `engine source has local changes. Commit or preserve them before install/update` | A controller's `vade install` stages only committed source, and an untracked file counts as a local change. The same check covers `harness` and `CLI` source. Nothing was overwritten. Use a clean clone for the controller. `git status` shows what is untracked. |
| `The engine refused --context N for memory` | `vade launch` or `vade up` on the Spark prints this when the engine refuses a `--context` you set yourself. The engine's own memory message follows. Choose a smaller `--context`, or leave it out next time. Vade then tries the standard sizes (32768, 16384, 8192, 4096) in turn until the engine accepts one. Run `vade doctor` to inspect memory. A context you set yourself is never lowered silently. |
| `Requested size does not fit.` / `Not enough memory` | Read the quantities in the message. A token-limit error asks for a shorter prompt or reply. A memory error gives the required and available memory. The engine's memory ceiling is MemAvailable minus a reserve of 10 % or 2 GiB, whichever is larger. Page cache already counts as free, so dropping caches does not help. Stop other programs that hold memory, or lower the context. FP4-Local chooses its chat count from memory. `--chats` does not lower it. A local launch or install also refuses to start when another engine holds the GPU memory. Stop that engine with its own `vade down` or the launcher that started it, then retry. |
| `needs an NVIDIA GB10 GPU` or `must support CUDA 13.0 or newer` (from `vade install` or a controller's `vade doctor`, `has an NVIDIA driver with CUDA ...`) | Vade runs only on a DGX Spark's GB10, with an NVIDIA driver that supports CUDA 13.0 or newer. `nvidia-smi` prints the supported version as `CUDA Version` in its header. Driver updates are a job for the Spark's administrator. Vade does not change the driver. |
| `Cannot read the NVIDIA GPU and driver; ... nvidia-smi said: ...` | The text after `nvidia-smi said:` is the driver's own error. Run `nvidia-smi` to see it in full, and ask the administrator to fix the driver. |
| `Prepare: no checksum list for this checkpoint` | Vade found no checksum list beside the model folder, so it cannot check the files against the published hashes. The install goes on. To check the files, follow [check the files](#check-the-files), or download the model with `vade pull`, which writes the list. |
| `The model is missing or incomplete on at least one Spark` | A controller's `vade install` staged the build, but a Spark has no complete checkpoint. Copy the checkpoint to the path the message above it names, check it, then run `vade install` again. |
| `Cannot reach ... over SSH, or its login shell failed.` | A controller's `vade up` or `vade doctor` could not run a command on that Spark. Run the `ssh ... true` command it prints to see the SSH error. Check that the Spark is on and that SSH logs in with no password prompt. |
| `This machine's GPU is in use.` | On a controller, `vade up` puts the machine name in front of it. Stop the workload with the launcher that started it, run `vade doctor`, then retry. Never kill processes by name. |
| `Install cache preparation failed; the previous activation was kept.` | A controller's `vade install` prints this when a Spark's GPU is busy or its files fail a check. The build you had before is still the active one. The message starts with the machine name and the reason (for example `This machine's GPU is in use.`), and the line above it gives the path of a saved log. Fix the reason, or read the log, then run `vade install` again. |
| `Port ... is in use; close the application using it and retry vade launch.` (from `vade up`, it ends `retry vade up.`) | Find the program that uses the port and stop it through its own controls. A first local setup can pick another port, but later launches reuse the saved one. On a controller, choose free `local_port` and `serve_port` values in the private config. |
| `Model loading timed out` | Run `vade logs`, then `vade doctor`. Read the saved startup log, and retry only once you have fixed the cause. |
| `staged hashes differ from the manifest. Nothing started; run vade install.` | A controller's `vade up` prints this, with the machine name in front. Run `vade install` to restage the matching engine and PTX. A local `vade launch` or `vade up` rebuilds a changed staged build by itself. For stale PTX during a source build, run `cargo +1.99.0 kernels` again from `engine/` before you build again. |
| HTTP 503 with `{"error":{"type":"overloaded","message":"connection capacity reached"}}` | The engine accepts 16 connections at once. A client that opens a 16th chat while an earlier one is still closing, or more than 16 at a time, gets this reply at the start. The server stays ready, and nothing ran. Retry after a second. The engine sends no `Retry-After`, and it accepts the retry once a slot is free. `vade-harness gateway` ([client gateway](#client-gateway)) does this for you. With one upstream, it waits up to 30 s for a slot. It also adds `Retry-After: 1` to any 429 or 503 that it passes on without one. `vade-gateway` passes the refusal straight through. Otherwise lower the client's concurrency. |
| `generation stopped before completion` | It can appear on the first 8- or 16-request `vade bench` after a controller start, when the engine log reports an allocation that did not fit. Run `vade logs`, then retry. |
| `Transport: ... did not reply to the pre-start check; nothing was started or recorded.` | Vade gives each SSH connection 10 seconds to connect and the check 90 seconds to finish, so a brief network or SSH hiccup can cause this. Nothing started, so run `vade up` again. If it keeps happening, test the machine with `ssh -o BatchMode=yes` and read the SSH error in the retained logs. |
| `Directory ... is shared. Run chmod 700 ...` or `Directory ... on ... is open to other users.` | A `stage_dir` or `state_dir` is open to other users, on the controller or on a Spark. Run the `chmod 700` command the message prints, then run `vade install` again. Next time, create these folders with `mkdir -m 700` or let Vade create them. |
| `Auto-compaction deferred: safe compaction cannot fit a useful reply reservation; history retained. Will retry on the next turn; /compact or /new may be needed.` | It can appear in a long chat when the context is 8192, because the output limit then equals the whole context and there is no room to shorten the history safely. Your history is kept and the reply still arrives. Type `/new` to start a fresh chat, which does not show this message. |
| `Build GPU kernels failed; check the retained build log and the existing pinned Rust toolchains.` | Read the saved build log. Make sure you finished installing the pinned toolchains (step 1) and fetched the `build-std` files online (`sh scripts/install.sh` does this), which the offline kernel build needs, then retry. |
| `Vade hit an internal error at ...` | This is a bug in Vade, not in your setup. Run the same command again with `RUST_BACKTRACE=1` in front of it and report the output, with the command you ran, to the Vade maintainers. |

If shutdown cannot confirm which process it started, follow
[CLI recovery](cli/README.md#recovery). Keep the state. Do not delete it.

<a id="7-stop-the-server"></a>
## Stop the engine

Leave chat, keep the same `VADE_HOME` / `VADE_CONFIG`, and run `vade down`.
It stops the engine that this installation started. On Route B, it also stops
the SSH tunnel. Run `vade status` to confirm that the engine is down.

## FP8 on 2, 4 or 6 Sparks

The FP8 checkpoint is a supported route. It runs on 2, 4 or 6 linked Sparks,
from a controller, with its own config (`cli/examples/vade.conf`). Start it
with `vade up --fp8` or `vade up --fp8 --sparks 4`. The older flag
`--fp8-legacy` still works. Follow
[Install FP8 on 2, 4 or 6 Sparks](docs/install-fp8.md). Do not mix FP8 and
FP4-Local weights, model IDs or defaults.

## Legacy models

The separate Qwen3-8B catalog uses a different checkpoint and launches
differently. Its source recipe is in the
[legacy one-Spark quickstart](docs/quickstart-one-spark.md) and the
[CLI reference](cli/README.md). Do not mix its weights, model IDs or defaults
with the Flash-Next routes.
