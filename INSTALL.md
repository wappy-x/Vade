# Install Vade

This page gets Vade **v0.1.0** running and serving
**nvidia/Qwen3.8-Flash-Next-FP4-Local** as text-only chat. You start with one
Spark and no other computer.

If your checkpoint is already under `~/models`, use the
[quick start](#quick-start-with-an-existing-model). Otherwise, do steps 1 to 3:
prerequisites, the code and the model. Then pick a route:

- [Route A](#route-a-on-this-spark) runs everything on the Spark itself. Start
  here.
- [Route B](#route-b-from-a-controller) adds a controller: a second Linux or
  macOS computer that drives the Spark over SSH.
- [Linked Sparks](#two-and-more-sparks) build on Route B.

New terms are explained where they first appear, and the
[glossary](docs/GLOSSARY.md) has the rest.

## Quick start with an existing model

You need an idle DGX Spark with Git, `cc`, curl, a working NVIDIA driver and
the complete checkpoint under `~/models`. (For another location, first set
`VADE_MODELS` to the checkpoint folder.) During the private preview the clone
also needs access to the repository; the private-preview note in
[Step 2](#step-2-get-the-code) gives the two ways in.

Paste this block as one shell command:

```sh
git clone --branch main https://github.com/wappy-x/vade.git "$HOME/vade" &&
  sh "$HOME/vade/scripts/install.sh" --launch
```

If you already cloned Vade, open its folder and run these two commands
instead:

```sh
sh scripts/install.sh
./cli/target/release/vade launch
```

The installer runs only on Linux `aarch64`. It:

1. finds Rust in `~/.cargo/bin` and installs any missing pinned toolchains and
   GPU components;
2. builds the CLI and runs preflight checks;
3. fills the compiler cache and runs local `vade install`, which also compiles
   the GPU code into Vade's private startup cache, so the first `up` doesn't
   have to;
4. with `--launch`, starts the engine and opens chat in the folder you started
   from. Without that flag, it stops after install.

It doesn't download or edit models, change shell profiles or run sudo. Every
failed stage prints its cause and a next step. Read the script first if you
want to see what it runs.

The first run needs internet access for missing Rust and `build-std` files.
Once those are cached, `CARGO_NET_OFFLINE=true sh scripts/install.sh` works
offline from the checkout and reuses the tools it finds without asking rustup
to update them. We haven't measured how long toolchain and model downloads
take.

**After launch.** Chat can edit files in the folder it opens in, so ask it
only to say hello, then type `/exit`; the engine keeps running. Run later
commands by their full path, `"$HOME/vade/cli/target/release/vade"`, or add
that folder to `PATH`. Then `up` starts the engine without chat, `chat`
connects again and `down` stops this installation's engine. Run from a script
without a terminal, `launch` prints readiness and exits, and the engine keeps
running.

**If something goes wrong:**

- **`~/vade` already exists:** use the existing-checkout commands above, or
  clone somewhere else.
- **The clone fails:** check that this connection reaches GitHub and that you
  have access to the private repository (see [Step 2](#step-2-get-the-code)).
- **`VADE_CONFIG` or `VADE_BUNDLE` is set:** the script tells you which one to
  unset for this local source route.
- **This Spark already has a controller config:** use the manual
  [Route A steps](#route-a-on-this-spark). Without saved local settings,
  the installer's `vade install` can select that controller config.

Keep the same `VADE_HOME` (default `~/.vade`) for install, launch and every
later command.

## Before you start

You need a DGX Spark (NVIDIA GB10) running Linux `aarch64`. `nvidia-smi` must
work, and the GPU must be idle.

We tested on the **580.x driver series**. Our published benchmarks used it, and
so did the install replay (our scripted run-through of this page), which ran
**580.178.04**. That is the version we tested, not a minimum.

One Spark needs no fabric cable (the direct high-speed cable between Sparks)
and no RDMA setup (the fast memory-to-memory transport that linked Sparks use).

### Disk space

The model needs **102,262,310,289 bytes** (102.26 GB / 95.24 GiB). That is
the complete checkpoint (the model's published files, including its
configuration and tokenizer files) at one pinned revision. The
[committed SHA256 manifest](cli/examples/qwen3.8-flash-next-fp4-local.sha256)
lists every file, and we checked its sizes against the
[publisher's file list](https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300?recursive=true)
for that revision.

Beyond the model, leave room for Rust, the source, the build output and
download caches. Before it starts, the launcher checks for free space:

- **8 GiB free where Vade installs**
- **1 GiB free where the model lives**
- **9 GiB free with the model already there**, if both are on the same disk

These are minimum checks, not the most disk Vade will use.

For a rough idea, one of our source installs used **0.72 GiB** for the
checkout, the build output and a fresh `VADE_HOME` (Vade's private folder,
`~/.vade` by default). The Rust compilers and the model were already
installed, so that figure leaves out Rust and the download caches, and it isn't
the most an install can use at once. We haven't measured disk use with a fresh
toolchain and its downloads.

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

Windows can't run the engine or build the `vade` CLI, but it can still drive a
Spark. There are two ways.

**Route A over SSH.** This is the one to use. Windows 10 and 11 include an
OpenSSH client, so open PowerShell or Windows Terminal, run
`ssh you@spark-address`, and follow [Route A](#route-a-on-this-spark) in that
session. Everything runs on the Spark, and Windows is only the terminal.

To reach the API from a program on Windows, forward the port when you connect:

```powershell
ssh -L 18080:127.0.0.1:18080 you@spark-address
```

While that session stays open, `http://127.0.0.1:18080/v1` on Windows reaches
the engine on the Spark. If `vade status` shows a different port, use that one
on both sides.

**A controller inside WSL 2.** WSL 2 runs a real Linux, usually `x86_64`, and
Route B supports Linux `x86_64` controllers. So far we have only type-checked
this path: `cargo check --all-targets --target x86_64-unknown-linux-gnu`
passes for the CLI and the harness. Nobody has built, linked or run a
controller inside WSL 2 yet. If you try it, follow
[Route B](#route-b-from-a-controller) from the WSL shell, with your SSH keys
inside WSL.

**Native Windows** is not supported. The CLI and the harness use the Unix
parts of Rust's standard library (file permissions, process groups, file
descriptors), and so do two engine crates the CLI links (the model reader and
the benchmark client). None of them compile for `x86_64-pc-windows-msvc`.
Starting `ssh` and `rsync` from a native Windows build has never been tried.

## Step 1: prerequisites

There are two ways to get the Vade programs:

- **Build from source.** Route B needs this for the controller's CLI and chat
  program.
- **Use the prebuilt launcher bundle** from the v0.1.0 release (see
  [Optional release bundle](#optional-release-bundle) in step 2). The bundle is
  for running Vade on the Spark itself. Route B can also use its engine files
  through `prebuilt_dir`, as described below.

Source builds need both pinned Rust toolchains. The model download helper
needs host Rust 1.99.0. Bundle users with an existing verified checkpoint can
skip Rust setup.

<a id="1-prepare-the-sparks"></a>
On the Spark, install any missing Ubuntu packages (this needs `sudo`):

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

You should see `aarch64`, GB10 and no running compute application. Each of
the 9 names in the `command -v` line must print a path; nothing printed means
that tool is missing. Install any missing one with `apt-get` (`ssh` is in
`openssh-client`). If
something else is using the GPU, stop it through its own launcher first. Leave
driver and system changes to the Spark's administrator; Vade does not make
them.

<a id="2-install-the-pinned-rust-compilers"></a>
<a id="install-rust-source-builds-only"></a>
### Install Rust

The engine is written in Rust and has no third-party Rust dependencies.

If `~/.cargo/bin/rustup` already exists, skip to the toolchains below. (It may
be installed even when a new shell can't find it, because the installer below
leaves your shell startup files alone.) Otherwise, download the installer and
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
rustup toolchain install 1.99.0 --profile minimal
```

Source builds also need the pinned GPU toolchain:

```sh
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

### Extra tools for a controller or linked Sparks

A controller needs the host toolchain, a native linker, Git, OpenSSH, rsync,
curl and `shasum`. macOS includes `shasum`; on Ubuntu it comes with `perl`.

For linked Sparks, each machine must already have `ping`. On Ubuntu it is in
the `iputils-ping` package. Networking is a separate job for the
administrator; see [network setup](docs/network-setup.md).

## Step 2: get the code

<a id="3-clone-and-build"></a>

> [!IMPORTANT]
> **Private preview.** The repository is private for now. Until it is public,
> the HTTPS clone below fails for anyone without access (Git asks for a
> password or reports `Repository not found`), and so do the release downloads
> on this page. Ask the repository owner for access, then clone one of these
> two ways instead:
>
> - **With a GitHub account that has access.** Install the
>   [GitHub CLI](https://cli.github.com/), sign in with `gh auth login`, then
>   run `cd ~ && gh repo clone wappy-x/vade -- --branch main && cd ~/vade`.
> - **With a read-only deploy key.** On the Spark, make a key with
>   `ssh-keygen -t ed25519 -f ~/.ssh/vade-deploy -N ''` and send the owner
>   `~/.ssh/vade-deploy.pub`. Once they have added it to the repository, clone
>   over SSH:
>
>   ```sh
>   cd ~ && GIT_SSH_COMMAND='ssh -i ~/.ssh/vade-deploy -o IdentitiesOnly=yes' git clone --branch main git@github.com:wappy-x/vade.git && cd ~/vade
>   ```
>
>   The first connection asks you to confirm GitHub's host key. Compare it
>   with [GitHub's published fingerprints](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints).
>   A later `git pull` needs the same `GIT_SSH_COMMAND`.
>
> The rest of the page is the same. The [release bundle](#optional-release-bundle)
> and the [Route B manifest](#check-the-files) have their own private-preview
> notes.

On the Spark, clone the repository into your home folder and stay at its root
in this shell. Later steps assume the checkout is `~/vade`:

```sh
cd ~ && git clone --branch main https://github.com/wappy-x/vade.git && cd ~/vade
```

This gives you the latest `main`: v0.1.0 plus the changes listed under
Unreleased in the [changelog](CHANGELOG.md). For the v0.1.0 release exactly,
clone with `--branch v0.1.0` instead.

Once the repository is public, this clone needs no account or token. Do not
put credentials in the clone URL. We recommend cloning, but a downloaded source
archive (GitHub's Download ZIP or `git archive`) also works and records its
commit as the build id.

### Run one kernel build (source builds only)

If you are building from source, run one kernel build now, while you are
online and after installing the pinned toolchains above. Bundle users skip this
build:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
(cd engine && CARGO_NET_OFFLINE=false cargo +1.99.0 kernels)
```

Expect compiler warnings from the kernel crate. The build worked if it ends
with `Finished` and exit status 0. This step needs no GPU.

This build is here for its downloads. The first time it runs, the kernel
compiler downloads its `build-std` dependencies into Cargo's cache. Later,
`vade launch` builds the kernels again, offline, in its own private copy of the
source. It does not reuse this build's output, only those downloads. Without
them, the first launch on a fresh machine fails at the kernel build.

### Optional release bundle

The [release workflow](.github/workflows/release.yml) builds a Linux aarch64
launcher bundle and `SHA256SUMS`. Both are on the
[v0.1.0 release page](https://github.com/wappy-x/vade/releases/tag/v0.1.0);
building from source, as above, is the alternative.

Download them into a new, empty folder of their own, outside the checkout. The
command below uses `~/vade-launch`. The checksum file gives you the exact
bundle filename, which includes the revision. If a failed try left
`~/vade-launch`, run `rm -r ~/vade-launch` before you retry, because the
`mkdir` below stops the block when the folder already exists:

```sh
mkdir "$HOME/vade-launch" &&
(cd "$HOME/vade-launch" && \
  curl --fail --location --output SHA256SUMS https://github.com/wappy-x/vade/releases/download/v0.1.0/SHA256SUMS && \
  read -r BUNDLE_SHA256 BUNDLE < SHA256SUMS && \
  curl --fail --location --output "$BUNDLE" "https://github.com/wappy-x/vade/releases/download/v0.1.0/$BUNDLE" && \
  sha256sum -c SHA256SUMS && tar --zstd -xf "$BUNDLE")
```

> [!IMPORTANT]
> **Private preview.** While the repository is private, the two `curl`
> downloads above answer 404. Download with the GitHub CLI instead, signed in
> with an account that has access. It fetches both release files into the new
> folder:
>
> ```sh
> mkdir "$HOME/vade-launch" &&
> (cd "$HOME/vade-launch" && \
>   gh release download v0.1.0 -R wappy-x/vade && \
>   read -r BUNDLE_SHA256 BUNDLE < SHA256SUMS && \
>   sha256sum -c SHA256SUMS && tar --zstd -xf "$BUNDLE")
> ```
>
> If the Spark has no `gh`, run `gh release download v0.1.0 -R wappy-x/vade`
> on a computer that has it, copy both files into `~/vade-launch` on the Spark
> (with `scp`, for example), and finish there with the `read`, `sha256sum` and
> `tar` lines.

Keeping the bundle outside the checkout leaves the clone clean, so you can
later use the same clone as a controller. Take both files from that release
page: a checksum catches changed bytes, but it does not prove who published
them. Extraction needs `tar` and `zstd`.

The bundle replaces the Rust builds, not the model download. Keep the clone,
because step 3 uses its fetch helper and the committed model manifest.
[Launch from the bundle](#launch-from-the-bundle) in Route A shows how to start
it.

The launcher reads `launcher-bundle.conf`, the file next to `./vade` that
records the hashes of the engine, the GPU kernels, the chat program and the
launcher itself. A controller (Route B) can use the same unpacked bundle: set
`prebuilt_dir` in its config to that folder, and `vade install` copies the
bundle's engine, GPU kernels and helper to each Spark instead of building them,
as long as the bundle was built from the controller checkout's commit. The
controller still builds its own CLI and chat program. See
[bundle details](cli/docs/prebuilt.md) and [first run](harness/docs/first-run.md).

## Step 3: get the model

<a id="3-fetch-and-verify-the-qualified-checkpoint"></a>
<a id="4-put-the-official-weights-on-both-sparks"></a>
<a id="4-put-the-official-weights-on-each-spark"></a>
Use publisher revision
[`b9ce18a6887e18c089fdea8509a33e63429a3300`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300).
Keep the checkpoint complete, including the index, tokenizer and configuration,
and use the files exactly as published: don't convert or re-quantize them.

The files at this revision download **without a Hugging Face token or
click-through access approval**. The publisher's
[NVIDIA Open Model Agreement terms](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/blob/b9ce18a6887e18c089fdea8509a33e63429a3300/README.md)
still apply.

### Download

The download prints nothing until it finishes, and that can take hours (see
[download rates on rented VMs](docs/benchmarks.md#rented-cloud-gb10-vms)). Over
SSH, a dropped connection stops it, so run it under `nohup` as below (or inside
`tmux`, if you have it); its output goes to `~/vade-fetch.log`. If the
complete checkpoint is already at that path, skip the fetch and go straight to
the manifest check below.

From the checkout root on the Spark, run this fetch exactly as written:

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
rerun. A restart releases the lock too. The `.fetch/lock` file stays in place;
do not remove it. Its presence does not mean that a fetch is running.
Metadata requests have a 120-second deadline. A transfer fails if its rate
stays below 1 byte per second for 120 seconds. Large active downloads can
continue for hours.

Partial downloads can use space in addition to the model files. Leave
headroom beyond the 102.26 GB checkpoint. A complete
snapshot has the byte total given above; copying only the tensor shards is not
enough. Vade's `install`, `up` and `launch` do not download these weights.

### Check the files

On Route A, the first `vade launch`, `vade up` or `vade install` checks every
file for you while it compiles the GPU code, if it finds a checksum list: the
fetch helper's `Qwen3.8-Flash-Next-FP4-Local.sha256` beside the checkpoint
folder, or a `manifest.sha256` inside it. A file that differs stops the launch
with its name. Nothing is written into the checkpoint; the result is kept in
Vade's private state, so later starts skip the work.

If you got the checkpoint another way, or you use Route B, check the files
against the committed manifest once yourself. This reads the whole checkpoint
and takes over a minute. From the checkout root:

```sh
MODEL_SHA256_LIST="$PWD/cli/examples/qwen3.8-flash-next-fp4-local.sha256"
(cd "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" && sha256sum -c "$MODEL_SHA256_LIST")
```

Every file must pass. On macOS, use `shasum -a 256 -c` instead. Keep the
checkpoint folder name `Qwen3.8-Flash-Next-FP4-Local`: the engine serves the
model under that name in lowercase, `qwen3.8-flash-next-fp4-local`, and refuses
requests that name any other model.

A Route B Spark has no checkout. Download the same manifest from the v0.1.0
release tag instead, then run the same check:

```sh
MODEL_SHA256_LIST="$HOME/qwen3.8-flash-next-fp4-local.sha256"
curl -fsSL https://raw.githubusercontent.com/wappy-x/vade/v0.1.0/cli/examples/qwen3.8-flash-next-fp4-local.sha256 -o "$MODEL_SHA256_LIST"
(cd "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" && sha256sum -c "$MODEL_SHA256_LIST")
```

> [!IMPORTANT]
> **Private preview.** While the repository is private, that `curl` answers
> 404. Copy the manifest from the controller's checkout instead. From its root:
>
> ```sh
> scp cli/examples/qwen3.8-flash-next-fp4-local.sha256 you@spark-a-address:
> ```
>
> That puts it in the Spark's home folder, where the check above expects it.
> Then run the `sha256sum -c` line on the Spark.

## Route A: on this Spark

<a id="one-spark-directly-on-the-spark"></a>
If you use the release bundle, skip the block below and follow
[Launch from the bundle](#launch-from-the-bundle) instead. Otherwise, once the
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

When a checksum list is present, `launch` also performs the file check
described in [Check the files](#check-the-files). Later launches reuse the
installed files and the saved check. Loading the model still takes time.
When the engine is ready, it prints `Launch ready in ...` with how long the
launch took.

Run from a script, where standard input is not a terminal, `launch` skips
chat: after the ready line it prints
`Standard input is not a terminal, so chat was not opened. Chat: vade chat`
and exits.

Chat opens in the checkout, and the agent can edit files there, so ask it only
to say hello. Then type `/exit` to get back to your shell. The engine keeps
running.

The chat window needs an interactive terminal. A normal
`ssh you@spark-address` login is one. To open chat with a single SSH command
instead, add `-t`, and name the CLI by its path: that new shell does not have
the `PATH` line below.

```sh
ssh -t you@spark-address 'cd "$HOME/vade" && ./cli/target/release/vade chat'
```

For the release bundle, use `'cd "$HOME/vade-launch" && ./vade chat'` instead.
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

Type `/exit` again, then check and shut down. If you may later use this clone
as a controller, first change to another folder, such as your home folder:
`vade verify` and `vade bench` write their receipts into the current folder,
and Route B's `vade install` refuses a clone with untracked files.

```sh
cd ~
vade verify
vade down
vade status
```

After the last command, status should say the engine is stopped. Restart with
`vade up`, or with `vade launch` to start and chat. See
[first run](harness/docs/first-run.md) for the whole lifecycle.

### Check the whole route in one go

[`harness/scripts/route-a-check.sh`](harness/scripts/route-a-check.sh) runs
Route A from start to finish, with no chat window. In order, it:

1. checks the toolchains and the GPU, and builds the CLI;
2. asks `vade detect` which model `vade launch` will load;
3. runs `vade launch` and sends one chat request through the API;
4. runs `vade verify` against that same engine, then `vade down`.

Run it from the checkout root with the GPU idle (stop any running engine with
`vade down` first):

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
fails. After a PASS it removes the private home; after a FAIL it keeps it for
inspection and prints its path.

Set `VADE_MODELS` to pick the checkpoint, as for `vade launch`; the model step
fails when `vade` would load one outside it. `VADE_DRY=1` prints the commands
without running anything, on any computer.

### Launch from the bundle

After the model check in step 3, run this instead of Route A's source-build
block:

```sh
cd ~/vade-launch && unset VADE_CONFIG
./vade launch
```

Use `./vade` for the lifecycle commands that follow, or add that folder to your
`PATH`. Status hints repeat the command the way you typed it, such as
`Chat: ./vade chat`. A few messages still name the bare `vade`, such as the
port error's `retry vade launch`; from the bundle folder, type `./vade` for
those too.

### What `vade verify` checks

Drafting is how the engine guesses tokens ahead so answers arrive sooner, and
it must not change the answer. `verify` checks that. It repeats requests with
drafting OFF and ON, compares the token IDs, and writes `vade-verify.json` in
the current directory.

A pass prints `IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the
reply's token count. That pass covers this set of requests. It is not the full
cross-Spark exactness check (the same answer, token for token, on every run and
every Spark) described in
[architecture](docs/architecture.md#exactness-and-model-quality).

### Where the API listens

The default API is `http://127.0.0.1:18080/v1`. If that port is taken, first
setup picks another free loopback port and saves it as `port=` in
`~/.vade/config`; later launches reuse it. `vade status` prints the base
address, such as `http://127.0.0.1:18080`, and the API lives under its `/v1`.
Without `--url` or `VADE_API_URL`, local `vade verify` reads the running
engine's port from `~/.vade/state/run.conf`. That record exists only while the
engine runs.

An API chat request makes the model think unless it says otherwise, and the
reasoning counts against the same `max_tokens` as the answer. To get the
answer alone, send `"enable_thinking": false` (or `"reasoning_effort": "none"`).
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
telling you:

- the size it settled on
- the sizes that were refused
- any lowered output limit
- the `--context` flag that retries your original size

If no size starts, your saved config is left unchanged.

If you set output yourself, it is capped at **16384** and must fit in context.
If the engine refuses a `--context` you set yourself, you get an error with the
engine's message. It is never lowered silently.

Two other defaults: drafting is on, and the engine chooses how far ahead it
drafts; the prefix cache (reusing the start of an earlier prompt) is off.

Input and output share the context, so the real limit for each depends on the
other. The engine's `/health` endpoint shows the limits that apply. (`--chats`
works only from a controller; on the Spark itself it is refused.)

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

From the controller's checkout root, build the CLI and make a private config
file:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
mkdir -m 700 -p .local
[ -e .local/fp4-local.conf ] || cp cli/examples/fp4-local.conf .local/fp4-local.conf
export VADE_CONFIG="$PWD/.local/fp4-local.conf"
```

`VADE_CONFIG` points the CLI at that file. Without it, a controller reads its
default config: `~/.config/vade/config` on Linux,
`~/Library/Application Support/vade/config` on macOS. `XDG_CONFIG_HOME` or
`VADE_CONFIG` overrides it.

### Edit the config

Edit that private file before you go on.

**Local folders.**

- Set the local `engine_source`, `harness_source` and `cli_source` to this
  checkout's component folders.
- Pick private local `stage_dir` and `state_dir` paths. Put `state_dir` under
  your home, as the example does (`~/.vade-state`). It holds the record
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
something already uses them. The first is on this computer; the second is on
the Spark that serves. If the controller is one of the Sparks, both would land
on the same machine, so set `local_port` to another free port, such as 18181.

### Install and start

Preview each step with `--dry-run`, then run it:

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

On macOS, the agent's sandboxed shell sees only system and Homebrew tools, so
it can't find a Rust you installed with rustup in your home folder. You'll see
`sandbox-exec: execvp() of 'cargo' failed`. `vade chat` has no option for
this. Instead, run `vade chat --dry-run` to print the command it would run.
Then run that command yourself, adding
`--shell-toolchain ~/.rustup/toolchains/<toolchain>` at the end. That is the
folder that contains `bin/cargo`.

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
whether or not you pass `--chats`. Drafting is on (the engine picks how far
ahead), and the prefix cache is off by default.

A controller never lowers the context for you. If a Spark lacks the memory for
32768, the engine refuses it and you choose a smaller `--context`. Only Route A
lowers it automatically, and only when the engine says there isn't enough
memory. Stay at 8192 or above if you want `vade chat`: from a controller it
always opens the coding agent, which needs at least 8192 tokens, and with a
smaller context it stops with an error instead of falling back to plain chat.

The engine decides how many chats can run at once from context and memory. It
reports that number when it starts, and the controller's `vade status` shows it
as `Concurrent chats`. If the report is missing, status says the count was not
reported. `--chats` does not change it on FP4-Local; the `--chats` row in the
[flag table](#settings-you-may-change) below has the details.

For a larger context, start with `vade up --context 262144`
(add `--output 16384` for longer replies). This sets the context, not the chat
count. Check `Concurrent chats` in `vade status` for the admitted count.
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

Both use Route B. The setup below is for the chain; replicas need less of it.

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

The controller reaches **every** rank over SSH. That includes its own machine
when the controller is one of the Sparks: Vade has no separate path for a
local rank. Plain `ssh = you@spark-a-address` and `ssh = you@spark-b-address`
work for every rank with no `~/.ssh/config` alias.

Before `vade install`, check each rank from the controller exactly the way Vade
connects, with no prompt:

```sh
ssh -o BatchMode=yes you@spark-a-address true
ssh -o BatchMode=yes you@spark-b-address true
```

`Host key verification failed` or `Permission denied` means one of two things.
Either that machine does not trust the controller's key yet, or the controller
does not know that host key yet. Fix it with your usual SSH key setup. On a
Spark that is also the controller, that includes its own address. To record a
Spark's own host key on that Spark, run this once there:

```sh
ssh -o StrictHostKeyChecking=accept-new you@spark-a-address true
```

If that Spark then answers `Permission denied (publickey)`, it doesn't trust
its own key yet. Add the key there (run `ssh-keygen -t ed25519` first if
`~/.ssh/id_ed25519.pub` doesn't exist):

```sh
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
```

Running the controller on a separate computer avoids SSH to itself. Vade never
edits SSH settings.

You only need an alias for settings a plain destination cannot carry, such as a
jump host, a non-default port or a specific key. For that case, a private
`~/.ssh/config` block looks like this, and you then write `ssh = spark-b`:

```text
Host spark-b
    HostName spark-b-address
    User you
    ProxyJump you@spark-a-address
    IdentityFile ~/.ssh/your-key
```

### Install and start the chain

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
at a matching [release bundle](#optional-release-bundle). After `vade up --sparks 2`,
`vade verify` reports `1 machine`, because the two ranks form one engine, not
two separate engines.

There are two ways to choose the Sparks. Use one or the other, not both:

- `--sparks N` picks the first N configured machines.
- `vade up --hosts spark-a,spark-b,spark-c` picks named machines in cable
  order.

Two Sparks run the default two-stage layer pipeline, with a range of the
model's layers on each. With three or more Sparks, Vade uses the layer chain,
which spreads the layers across all the Sparks you chose.

List both end addresses for every cable between neighboring Sparks. You don't
need a cable that closes the chain into a ring.

One caution: if nobody has tested your cabling layout, the checks Vade runs
while building the chain don't prove exactness (the same tokens on every
Spark).

### RDMA or TCP

Linked Sparks use RDMA by default. That needs the RDMA (verbs) device and its
user library already set up, and a locked-memory (memlock) limit of
`unlimited`, both soft and hard (`ulimit -Sl` and `ulimit -Hl`), in the SSH
session Vade uses.
For TCP instead, set this **before install** and keep it set for `up` and
`doctor`:

```sh
export SEQPAR_RDMA=0
```

TCP uses the configured, reachable fabric addresses and needs neither RDMA nor
a fabric cable (a QSFP link). A single local Spark needs no transport between
Sparks at all.

<a id="replicas"></a>
### Serve many chats with replicas

When the model fits on one Spark, several Sparks serve many chats better as
independent copies than as one chain. `vade up --replicas 2` starts a separate
one-Spark server on each of the first two machines (or the machines named in
`--hosts`). It also starts a local gateway on `local_port` that sends each
request to the replica with the fewest requests in flight.

The gateway also tries to keep each conversation on one replica. Suppose a
request's messages, up to and including the last user message, start with the
same 2048 bytes as an earlier request. Then the gateway sends it to the replica
that handled the latest such request, as long as that replica is one of the
least busy. Otherwise the least busy replica wins. The prefix cache is off by
default, so we haven't confirmed that this routing gives cache hits.

A single chat runs at one-Spark speed; total throughput grows with the number
of replicas. Every replica is the same one-Spark engine, so a request gets the
same token IDs whichever replica answers.

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
These are the defaults when you pass no flags and haven't set `MTP` or the
prefix-cache variable (`VADE_FLASHNEXT_PREFIX_CACHE`) in your shell. Context
covers both the input and the generated output; `/health` reports the limits
that actually apply to a request.

| How you start | Context (tokens) | Output (tokens) | Drafting (multi-token prediction, MTP) | Prefix cache |
| --- | --- | --- | --- | --- |
| On this Spark: `vade launch` / `vade up` | 32768; lowered through 16384, 8192, 4096 only if the engine refuses it for memory | `min(context, 8192)` | ON, engine-selected | OFF |
| From a controller: `vade up` on any number of Sparks, with or without `--chats` | 32768 | `min(context, 8192)` | ON, engine-selected | OFF |
| From a controller: `vade up --replicas N`, each replica | 32768 | `min(context, 8192)` | ON, engine-selected | OFF |

`--context 262144` requests a larger context on either route
(add `--output 16384` for longer replies). The engine still sets the chat
count from memory. An explicit context must pass the engine's memory check;
a refusal is reported and the context is not lowered.

| Flag | What it does |
| --- | --- |
| `--context` | How many tokens the prompt and reply can use together. On the Spark, the value must fit the checkpoint's limit and pass the engine's memory check. If the engine refuses a value you set, you get an error; Vade does not lower it. Defaults are in the table above. |
| `--output` | The most tokens a reply can use. It can't be larger than the context or the FP4-Local limit of 16384. |
| `--mtp` | Asks for a specific drafting depth. `0` turns drafting off. It overrides the `MTP` variable. Without either, the engine chooses. |
| `--chats` | Works only from a controller; on the Spark itself it is refused. Vade records it and passes it on, but FP4-Local sets the chat count from context and memory, so it changes neither the context nor how many chats run at once. (With legacy FP8, `--chats N` reserves room for N chats.) |
| `--memory-bytes` | Sets a memory budget. The engine's memory check and the memory actually free still apply. |
| `--cache-bytes 0` | Turns the prefix cache off for a local install. Any local size above zero is refused. |

### Turn on the prefix cache

The prefix cache (reusing a cached prompt prefix across requests) is OFF by
default and supported only on one Spark. To turn it on for **local Route A**,
restart from the same local environment:

```sh
vade down
VADE_FLASHNEXT_PREFIX_CACHE=1 vade up
```

If you saved `--cache-bytes 0` earlier, it wins over this variable. Setting
the variable on a controller does not reach the Sparks it drives. The engine
makes the final decision ([source](engine/crates/serve/src/prefix_switch.rs)).
Sending `cache:true` in a request does not turn the cache on by itself.

## Client gateway

To put Vade's gateway between your own client and the engine, build the
harness binaries from the checkout and run:

```sh
cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml --bins
harness/target/release/vade-gateway --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

If `vade status` shows a different engine endpoint, use that one. Point your
client at `http://127.0.0.1:8787/v1`. Ctrl-C stops the gateway.

When the engine is full (16 open connections), it turns a new request away
with HTTP 503 before doing any work. `vade-gateway` passes that refusal
straight to your client. The harness's built-in gateway waits it out instead;
start it in place of `vade-gateway`:

```sh
harness/target/release/vade-harness gateway --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

With one upstream it holds only that refusal (`connection capacity reached`)
and sends the same bytes again after a short wait, for up to 30 s, so a client
that opens its next round of chats while the last one is still closing just
sees a slightly later first token. Any other 503 may come after work started,
so it reaches the client once. This gateway also passes `GET /health` to the
engine, so `vade bench --url http://127.0.0.1:8787/v1` and health checks work
through it. See the [harness documentation](harness/README.md) for the
supported API and tool behavior.

The gateway guards against web pages driving it through your browser:

- It answers 421 to any Host other than `127.0.0.1`, `localhost` or `[::1]`
  (bare or with its port).
- It answers 403 to any request with an `Origin` header.

Command-line clients that use the loopback address and omit `Origin` pass
these checks. To use a browser client, add `--allow-origin https://app.example` (exact origin,
repeatable; `--allow-host NAME` allows another Host name). Ideally add
`--token TOKEN` or `VADE_GATEWAY_TOKEN` too, which then requires
`Authorization: Bearer TOKEN` on every request.

<a id="6-run-the-timing-benchmark"></a>
## Run the benchmark

Start the benchmark once the engine is running and has finished checking the
checkpoint files in the background. At that point `/health` shows the
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

By default it uses a bundled offline writing prompt and asks for the server's
drafting policy, which for FP4-Local means drafting on. When many requests run
at once, the server may answer some of them without drafting. `vade bench` then
ends with a `Drafting:` line that names, for each concurrency, how many replies
were plain; those rows measure plain speed, which is why a higher concurrency
can show a lower rate than the one before it.

Each receipt (the JSON result file) records the draft mode you asked for, the
server's build SHA and the drafting (MTP) depth the server reports. What each
response reported is stored separately in `draft_observations`. A reported
depth of zero alone doesn't prove drafting was off.

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

Every request uses greedy sampling with the cache and thinking off. It keeps
generating past the end-of-answer token so every reply is the same length;
`--max-tokens` sets that length. Requests are measured in groups (cohorts), and
a group with a short or incomplete reply gets no rate.

This is a quick check of your own setup, not a RigMark card, and its recipe
differs from the published cards, so its numbers are not comparable with them.
[Reproduce a measurement in detail](docs/benchmarks.md#reproduce-a-measurement-in-detail)
lists the differences and shows how to run a card.

Read [benchmark recipes and results](docs/benchmarks.md) for workloads, receipt
fields and measurement limits. A quick test or a short greeting doesn't
measure real speed. If you share a result, keep the exact source revision,
checkpoint, settings and raw output with it so others can reproduce it.

## First-run problems

The messages below are excerpts from the CLI and the engine. Whatever you see,
leave the files under `~/.vade/state` in place (they tell `vade down` what to
stop), and run `vade logs` and `vade doctor` before you retry.

| Message | What to do |
| --- | --- |
| `engine source has local changes. Commit or preserve them before install/update` | A controller's `vade install` stages only committed source, and an untracked file counts as a local change. The same check covers `harness` and `CLI` source. Nothing was overwritten. Use a clean clone for the controller; `git status` shows what is untracked. |
| `The engine refused --context N for memory` | `vade launch` or `vade up` on the Spark prints this when the engine refuses a `--context` you set yourself; the engine's own memory message follows. Choose a smaller `--context`, or leave it out next time: Vade then tries the standard sizes (32768, 16384, 8192, 4096) in turn until the engine accepts one. Run `vade doctor` to inspect memory. A context you set yourself is never lowered silently. |
| `Requested size does not fit.` / `Not enough memory` | Read the quantities in the message. A token-limit error asks for a shorter prompt or reply. A memory error gives the required and available memory. The engine's memory ceiling is MemAvailable minus a reserve of 10 % or 2 GiB, whichever is larger. Page cache already counts as free, so dropping caches won't help. Stop other programs that hold memory, or lower the context. FP4-Local chooses its chat count from memory; `--chats` does not lower it. A local launch or install also refuses to start when another engine holds the GPU memory. Stop that engine with its own `vade down` or the launcher that started it, then retry. |
| `This machine's GPU is in use.` | On a controller, `vade up` puts the machine name in front of it. Stop the workload with the launcher that started it, run `vade doctor`, then retry. Never kill processes by name. |
| `Install cache preparation failed; the previous activation was kept.` | A controller's `vade install` prints this when a Spark's GPU is busy or its files fail a check. The build you had before is still the active one. The message starts with the machine name and the reason (for example `This machine's GPU is in use.`), and the line above it gives the path of a saved log. Fix the reason, or read the log, then run `vade install` again. |
| `Port ... is in use; close the application using it and retry vade launch.` (from `vade up`, it ends `retry vade up.`) | Find the program using the port and stop it through its own controls. A first local setup can pick another port, but later launches reuse the saved one. On a controller, choose free `local_port` and `serve_port` values in the private config. |
| `Model loading timed out` | Run `vade logs`, then `vade doctor`. Read the saved startup log, and retry only once you have fixed the cause. |
| `staged hashes differ from the manifest. Nothing started; run vade install.` | A controller's `vade up` prints this, with the machine name in front. Run `vade install` to restage the matching engine and PTX. A local `vade launch` or `vade up` rebuilds a changed staged build by itself. For stale PTX during a source build, run `cargo +1.99.0 kernels` again from `engine/` before you build again. |
| HTTP 503 with `{"error":{"type":"overloaded","message":"connection capacity reached"}}` | The engine accepts 16 connections at once. A client that opens a 16th chat while an earlier one is still closing, or more than 16 at a time, gets this reply at the start; the server stays ready and nothing ran. Retry after a second: the engine sends no `Retry-After`, and the retry is accepted once a slot is free. `vade-harness gateway` ([client gateway](#client-gateway)) does this for you: with one upstream it waits up to 30 s for a slot, and it adds `Retry-After: 1` to any 429 or 503 it passes on without one. `vade-gateway` passes the refusal straight through. Otherwise lower the client's concurrency. |
| `generation stopped before completion` | It can show up on the first 8- or 16-request `vade bench` after a controller start, when the engine log reports an allocation that did not fit. Run `vade logs`, then retry. |
| `Transport: ... did not reply to the pre-start check; nothing was started or recorded.` | Vade gives each SSH connection a short time limit, so a brief network or SSH hiccup can cause this. Nothing started, so run `vade up` again. If it keeps happening, test the machine with `ssh -o BatchMode=yes` and read the SSH error in the retained logs. |
| `Directory ... is shared. Run chmod 700 ...` or `Directory ... on ... is open to other users.` | A `stage_dir` or `state_dir` is open to other users, on the controller or on a Spark. Run the `chmod 700` command the message prints, then run `vade install` again. Next time, create these folders with `mkdir -m 700` or let Vade create them. |
| `Auto-compaction deferred: safe compaction cannot fit a useful reply reservation; history retained. Will retry on the next turn; /compact or /new may be needed.` | It can appear in a long chat when the context is 8192, because the output limit then equals the whole context and there is no room to shorten the history safely. Your history is kept and the reply still arrives. Type `/new` to start a fresh chat, which won't show this message. |
| `Build GPU kernels failed; check the retained build log and the existing pinned Rust toolchains.` | Read the saved build log. Make sure you finished installing the pinned toolchains (step 1) and ran the online kernel build (step 2), which downloads what the offline launch build needs, then retry. |

If shutdown cannot confirm which process it started, follow
[CLI recovery](cli/README.md#recovery). Keep the state rather than deleting it.

<a id="7-stop-the-server"></a>
## Stop the engine

Leave chat, keep the same `VADE_HOME` / `VADE_CONFIG`, and run `vade down`.
It stops the engine this installation started and, on Route B, its SSH tunnel.
Run `vade status` to confirm it is down.

## Legacy models

The older Flash-Next FP8 profile and the separate Qwen3-8B catalog use
different checkpoints and launch differently. Start with
[legacy installation](docs/install-legacy-fp8.md) and the
[CLI reference](cli/README.md). The Qwen3-8B source recipe is in the
[legacy one-Spark quickstart](docs/quickstart-one-spark.md). Do not mix those
weights, model IDs or defaults with this FP4-Local route.
