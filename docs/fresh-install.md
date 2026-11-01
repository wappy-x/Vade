# FP8 fresh install: from zero to a first answer

**Installing Vade for the first time? Start with [INSTALL.md](../INSTALL.md).**
The current model, [FP4-Local](GLOSSARY.md#dg), runs on a single Spark: you
start it on the Spark itself with `vade launch`, and you need no second
computer. [First run](../harness/docs/first-run.md) explains that
launcher. Driving Sparks from another computer, the
[controller](GLOSSARY.md#ac) route, comes after that.

This page is a longer checklist for the FP8 model. It needs **two** DGX
Sparks, yours or rented: the FP8 checkpoint does not fit on one. It takes them
to a working Vade server that you drive with the `vade` command from a Mac
(or a Linux computer, as step 0 shows), which acts as the controller. FP4-Local is
the default, so every FP8 start needs the explicit `--fp8` flag. The shorter
guide, with the measured speeds, is
[Install FP8 on 2, 4 or 6 Sparks](install-fp8.md).

> **Heads-up:** nobody has yet run this route end to end from a blank
> machine. If a step doesn't match what you see, please open an issue.

<br>

## The route at a glance

| # | Step | Where |
| --- | --- | --- |
| 0 | [Check what you have](#0-check-what-you-have) | anywhere |
| 1 | [Renting? Ask for these](#1-renting-ask-for-these) | provider |
| 2 | [Network and OS](#2-network-and-os) | each Spark |
| 3 | [SSH from your Mac](#3-ssh-from-your-mac) | Mac |
| 4 | [Rust compilers](#4-rust-compilers) | Mac + each Spark |
| 5 | [The model weights](#5-the-model-weights) | each Spark |
| 6 | [Get Vade and configure it](#6-get-vade-and-configure-it) | Mac |
| 7 | [Install, start, check](#7-install-start-check) | Mac |
| 8 | [First answer, chat, stop](#8-first-answer-chat-stop) | Mac |


<br>

---

## 0. Check what you have

- [ ] **Two DGX Sparks** with NVIDIA's DGX OS, idle GPUs (`nvidia-smi` shows no compute processes)
- [ ] **One 200 Gb/s QSFP cable** between them ([what to buy](network-setup.md#1-what-to-buy))
- [ ] **Administrator (sudo) access** on both, for the one-time OS setup
- [ ] **The same home folder on both Sparks** (`echo $HOME` prints the same path), so `~/models/...` means the same place on both. If they differ, use absolute paths in the config instead of `~/`.
- [ ] **The standard 128 GB of unified memory on each Spark** (NVIDIA's specification), with no other GPU jobs
- [ ] **Plenty of free disk on each Spark** for the checkpoint (see its size on the model page) plus room for the build
- [ ] **The basic programs and download tools**: `cc`, `curl`, `git`, `rsync`, `ping`, `sha256sum`, Rust 1.99.0. Have them ready before step 5.
- [ ] **A Mac** with Git and network access to both Sparks
- [ ] **Network access to GitHub**

**No Mac?** A Linux computer works as the controller for install, start, stop
and chat. The coding assistant's shell tool and automatic checks need the
macOS sandbox, so they're off on Linux.

<br>

---

## 1. Renting? Ask for these

Cloud and hosting providers rent Sparks as single machines or as cabled pairs.
**A cabled pair is fastest.** A rented pair usually comes as two Sparks joined
by one direct cable, and that's all Vade needs. Two Sparks on an ordinary
network also work, more slowly, in [TCP mode](#no-qsfp-link-use-tcp-mode).

Three terms come up in the questions below:

- **RDMA**: direct memory-to-memory transfers between Sparks.
- **RoCE switch**: a network switch that carries RDMA.
- **PAM**: the login system that applies limits such as locked memory
  (`memlock`). RDMA needs that limit set to unlimited.

Ask the provider before you pay:

| Question | Why it matters |
| --- | --- |
| "Are the two Sparks joined by a **200 Gb/s QSFP cable** (or a RoCE switch)?" | Then Vade uses fast **RDMA** on that link. Without it, use [TCP mode](#no-qsfp-link-use-tcp-mode). |
| "Can I have **sudo**, or will you run the OS setup for me?" | Fabric addresses, RDMA permissions and memlock need root once. |
| "Is SSH ordinary **OpenSSH with PAM**?" | For RDMA, the SSH alias Vade uses must give **unlimited locked memory**. Some managed SSH gateways skip that. (TCP mode does not need it.) |
| "Which **kernel** is installed?" | `7.0.0-1019-nvidia` needs NVIDIA's `kho=off` fix for multi-Spark RDMA. |
| "Are the **QSFP interfaces unconfigured**, or what addresses did you assign?" | You either set them up yourself or reuse theirs. |

**What is different on rented machines**

- You will not see the cable. Trust `ibdev2netdev`, `ethtool` and the
  [link checks](network-setup.md#7-check-every-link), not a diagram.
- Your Mac reaches the Sparks over the internet or a VPN. Make SSH aliases for
  them (step 3). `vade up --fp8` opens its own SSH tunnel. No ports
  need to be public.
- Download the weights **on the Sparks**: data-center internet is usually
  faster than home internet.
- **Ordinary Ethernet only, no QSFP link?** Use [TCP mode](#no-qsfp-link-use-tcp-mode).
- **Only one Spark?** This guide's Flash-Next FP8 checkpoint does not fit on one.
  Use the current FP4-Local [on-Spark install](../INSTALL.md#route-a-on-this-spark)
  and [first-run guide](../harness/docs/first-run.md).
- **Different home folders on the two rentals?** Both Sparks must see the exact
  same absolute model and build paths. Write absolute paths (for example
  `/data/vade/models/...`) in the config instead of `~/`.
- **Ping blocked?** With RDMA, `vade install` and `vade doctor` check each link
  with ICMP ping. Ask the provider to allow ping between the two Sparks. TCP
  mode does not use ping.
- **No Mac?** See the note under [step 0](#0-check-what-you-have).

### No QSFP link? Use TCP mode

Vade uses RDMA by default. On Sparks joined only by ordinary Ethernet (common
with rentals), switch to **TCP mode** on the Mac before `vade install`,
`vade up --fp8` and `vade doctor`:

```sh
export SEQPAR_RDMA=0
```

`0` means TCP and `1` means RDMA is required. Any other value is refused.
Unset it to go back to the RDMA default.

| | RDMA (default) | TCP mode (`SEQPAR_RDMA=0`) |
| --- | --- | --- |
| Link | QSFP cable or RoCE switch | Any network the two Sparks share |
| `fabric.<peer>` in the config | Own address on the QSFP cable | Own ordinary address that the other Spark can reach |
| Unlimited locked memory, RDMA setup | Required | Not needed |
| Open between the Sparks | TCP 29960, 29961, RoCE UDP 4791, ping | TCP 29960, 29961 |
| `vade up --fp8` says | `… on 2 Sparks over RDMA.` | `… on 2 Sparks over TCP.` |
| Rank logs show | `rdma=true` | `rdma=false` |
| Speed | See [dated receipts](benchmarks.md) | No current measurement published here |

The setting is **not saved** in the config. Export it again in every new
terminal before `vade install`, `vade up --fp8` (including restarts) and
`vade doctor`.

<br>

---

## 2. Network and OS

Follow **[Hardware and network setup](network-setup.md)** until every link check
passes. In short:

1. Cable port 0 of spark-1 to port 1 of spark-2.
2. Give each cable link (a [rail](GLOSSARY.md#pr)) its own `/24` subnet with MTU 9000 (example: `192.0.2.1` ↔ `192.0.2.2`, `198.51.100.1` ↔ `198.51.100.2`).
3. Clone Vade on **each Spark** (`git clone --branch main https://github.com/wappy-x/vade.git ~/vade`, then `cd ~/vade`) and run the OS-setup scripts from that repository root, in the order given in [Spark system setup](spark-setup.md). Together they install the RDMA packages, configure the network, set unlimited locked memory (`memlock`) for RDMA, apply NVIDIA's `kho=off` fix if your kernel is `7.0.0-1019-nvidia`, and create SSH keys between the Sparks.
4. Check: jumbo ping, `ib_write_bw` in both directions, `ulimit -l` = `unlimited`.

**Using [TCP mode](#no-qsfp-link-use-tcp-mode)?** Skip the QSFP, RDMA and memlock
steps. All you need is for the two Sparks to reach each other on TCP ports
29960 and 29961, plus SSH from spark-1 to spark-2 to copy the weights in step 5.

**Install the base packages too.** The setup scripts add only the RDMA and
network packages. Building Vade and downloading the model need these as well.
An administrator runs this once on each Spark. Refresh the package lists first,
even on a fresh VM:

```sh
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl git openssh-client \
  rsync iputils-ping
```

Expected: exit 0 from both commands. The model helper uses Rust 1.99.0 and
curl. Install the pinned Rust toolchains in step 4 before the model download.
Do not continue until these prerequisites and the link checks pass.

<br>

---

## 3. SSH from your Mac

`vade` drives the Sparks through **SSH aliases** you define on the Mac. Add
something like this to `~/.ssh/config` (the names and addresses are examples):

```text
# spark-1: its LAN or VPN address
Host spark-1
    HostName 192.0.2.11
    User your-user
    IdentityFile ~/.ssh/id_ed25519

Host spark-2
    HostName 192.0.2.12
    User your-user
    IdentityFile ~/.ssh/id_ed25519
```

`vade` runs SSH in batch mode, so there are **no password prompts**. Log in
once by hand to accept each host key, and load your key into `ssh-agent` if it
has a passphrase. Then check that each alias works without prompts and gives
unlimited locked memory:

```sh
ssh -o BatchMode=yes spark-1 'ulimit -Sl; ulimit -Hl'
ssh -o BatchMode=yes spark-2 'ulimit -Sl; ulimit -Hl'
```

```text
unlimited
unlimited
```

Anything else (often `8192`) means the RDMA setup step has not taken effect for
that login path. Fix it before you go on. In TCP mode the locked-memory limit
does not matter, but the logins must still work without prompts.

<br>

---

## 4. Rust compilers

Vade pins its compilers. Do not swap in other versions.

| Machine | Needs |
| --- | --- |
| **Mac** | Rust **1.99.0** (selected automatically by `rust-toolchain.toml`) |
| **Each Spark** | Rust **1.99.0** and **nightly-2026-10-01** with the `nvptx64-nvidia-cuda` target (for GPU kernels) |

If `rustup` is missing, install it from [rustup.rs](https://rustup.rs). Then:

```sh
# Mac
rustup toolchain install 1.99.0 --profile minimal

# each Spark
rustup toolchain install 1.99.0 --profile minimal
rustup toolchain install nightly-2026-10-01 --profile minimal \
  --target nvptx64-nvidia-cuda \
  --component rust-src --component llvm-tools --component llvm-bitcode-linker
```

Each Spark also needs `cc`, `rsync`, `curl`, `ping`, `sha256sum` and
`nvidia-smi`. The Mac needs `rsync`, `curl` and `ssh`. Vade does **not** need
the CUDA toolkit: it builds its own GPU kernels.

For full details, including a safe way to read the installer first, see
[INSTALL.md compiler prerequisites](../INSTALL.md#step-1-prerequisites).

<br>

---

## 5. The model weights

**Model:** the official [Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8/tree/236dfdf285828023ca3bcd3f37366c58a3469b13),
revision `236dfdf285828023ca3bcd3f37366c58a3469b13`. No conversion, no
re-quantizing. Read its license on the model page first.

The complete snapshot includes the weights, configuration, tokenizer and
supporting files. Make sure your free disk space covers that snapshot plus the
build output. How long the download takes depends on your connection and on
Hugging Face.

> This page fetches the checkpoint with the repository's helper,
> `ops/fetch-model.sh`. It uses curl for the download. Rust checks every file
> against the pinned publisher metadata. `vade pull
> Qwen/Qwen3.8-Flash-Next-FP8@236dfdf285828023ca3bcd3f37366c58a3469b13` fetches
> the files Vade serves from into the same folder. It writes a
> `<folder>.sha256` list in the same form, which the `sha256sum -c` step on
> spark-2 below reads. No FP8 start checks that list. `vade pull` skips files
> such as `.gitattributes`, so the two folders are not identical. See
> [the differences](pull-models.md#difference-from-the-fetch-helper).

**Download and verify on spark-1**, from the repository root you cloned in step 2.
Keep the folder name exactly `Qwen3.8-Flash-Next-FP8`: its lowercase name becomes
the model ID (`qwen3.8-flash-next-fp8`). The pinned Rust install in step 4 must
pass first, and no other download may be writing into this folder.

```sh
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13 "$HOME/models/Qwen/Qwen3.8-Flash-Next-FP8"
```

Expected final line: `verified <file-count> files`.

If the helper fails, stop, do what its one-line reason says and rerun the same
command. Finished downloads are reused. Continue only after it succeeds.

**What the [helper](../ops/fetch-model.sh) does.** Below, `MODEL_DIR` is the
folder you passed to it.

- It keeps partial downloads in a private sibling folder, `MODEL_DIR.fetch`.
- It checks file sizes, LFS SHA-256 hashes and Git blob hashes from the pinned
  publisher metadata. It rejects extra files and links.
- It checks every file against the official checksums and writes
  `MODEL_DIR.sha256`, which you use on spark-2 below.

**Copy to spark-2** over the QSFP link, which is usually faster than
downloading again (in TCP mode, use spark-2's ordinary address). Then verify
the copy. The key `~/.ssh/vade-cluster` below is the Spark-to-Spark key you
created in the SSH step of
[Spark system setup](spark-setup.md#6-establish-ssh-between-the-sparks).

```sh
# on spark-1; 192.0.2.2 is spark-2's example fabric address
# (TCP mode: use any key that lets spark-1 log in to spark-2)
K=~/.ssh/vade-cluster
ssh -i $K 192.0.2.2 'mkdir -p models/Qwen'
rsync -a --info=progress2 --exclude '/.cache/huggingface/' -e "ssh -i $K" \
  ~/models/Qwen/Qwen3.8-Flash-Next-FP8/ 192.0.2.2:models/Qwen/Qwen3.8-Flash-Next-FP8/
scp -i $K ~/models/Qwen/Qwen3.8-Flash-Next-FP8.sha256 192.0.2.2:models/Qwen/

# on spark-2
cd ~/models/Qwen/Qwen3.8-Flash-Next-FP8
sha256sum -c ../Qwen3.8-Flash-Next-FP8.sha256   # every file must report OK
```

<br>

---

## 6. Get Vade and configure it

**On the Mac.** `vade install` copies the source from this Mac clone to the
Sparks and builds there. (The Spark clones from step 2 are only for the
OS-setup scripts.)

The repository is <https://github.com/wappy-x/Vade>. Clone it and build the
CLI:

```sh
git clone https://github.com/wappy-x/vade ~/vade
cd ~/vade
cargo +1.99.0 build --release --offline --locked --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
mkdir -m 700 -p .local
[ -e .local/vade.conf ] || cp cli/examples/vade.conf .local/vade.conf
export VADE_CONFIG="$PWD/.local/vade.conf"
```

The two `export` lines last only for this terminal. In a new terminal, run them
again from the checkout (or add them to `~/.zshrc`), or `vade` will say the
config is missing.

`.local/` is ignored by Git, so your addresses never get committed.

**Edit `.local/vade.conf`.** The `[global]` part applies to the whole setup,
and each `[machine ...]` part describes one Spark. In each machine part,
`fabric.<peer>` is that Spark's own address on its cable to the peer (in TCP
mode, its ordinary address, as the [TCP table](#no-qsfp-link-use-tcp-mode) shows).
Here is a filled-in example for two Sparks (the addresses are examples):

```ini
[global]
# Local paths are resolved beside this file (.local/).
engine_source = ../engine
harness_source = ../harness
cli_source = ../cli
stage_dir = bin
# SSH runs from state_dir with a relative control-socket name, so a deep
# checkout path is supported. Keep this directory private.
state_dir = state
build_manifest = state/build.conf
model = qwen3.8-flash-next-fp8
model_repo = Qwen/Qwen3.8-Flash-Next-FP8
model_revision = 236dfdf285828023ca3bcd3f37366c58a3469b13
# Mac end of the SSH tunnel, and rank 0's loopback HTTP port.
local_port = 18080
serve_port = 18080
# Spark-to-Spark transport; tp_port + 1 (29961) is also used.
tp_port = 29960
health_timeout_secs = 360

[machine spark-1]
# An alias from ~/.ssh/config.
ssh = spark-1
# spark-1's OWN address on the cable to spark-2.
fabric.spark-2 = 192.0.2.1
model_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8
engine_source = ~/vade-source
cli_source = ~/vade-cli-source
stage_dir = ~/vade-stage
state_dir = ~/.local/state/vade

[machine spark-2]
ssh = spark-2
# spark-2's OWN address on the cable to spark-1.
fabric.spark-1 = 192.0.2.2
model_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8
engine_source = ~/vade-source
cli_source = ~/vade-cli-source
stage_dir = ~/vade-stage
state_dir = ~/.local/state/vade
```

The engine runs one process on each Spark, called a
[rank](GLOSSARY.md#pr). The first machine listed becomes **rank 0**, the one
that answers HTTP. More detail is in the
[CLI guide](../cli/README.md#multi-spark-source-setup).

<br>

---

## 7. Install, start, check

Run these on the Mac, in the checkout, with `VADE_CONFIG` set.

**Preview, then install.** Install checks both Sparks, copies the source,
builds the engine on them, builds the harness on the Mac and saves hashes of
everything it built.

```sh
vade install --dry-run
vade install
```

Near the end you should see:

```text
Installed build <sha>. Models were already present.
Harness: available. Engine check needs vade up, then vade doctor.
```

`<sha>` is the full 40-character Git revision of the engine source that was
built. The second line is normal: nothing is running yet. For this FP8 setup, start the server
with `vade up --fp8`, shown below.

Install refuses a checkout with uncommitted changes or untracked files. Your
`.local/` folder is ignored by Git, so it does not count.

**Start the server.**

```sh
vade up --fp8
```

```text
Starting qwen3.8-flash-next-fp8 on 2 Sparks over RDMA. Waiting for the model to load.
qwen3.8-flash-next-fp8 ready on 2 Sparks at http://localhost:18080: context 131072 tokens, drafting depth engine default.
  Chat: vade chat   Stop: vade down
```

Loading the weights takes minutes. `vade up --fp8` waits up to
`health_timeout_secs` (360 s by default). If it stops with "Requested size
does not fit.", see the note at the end of step 8.

**Check everything.**

```sh
vade status
vade doctor
```

`vade doctor` prints one row per check and ends with PASS. Among its rows:

```text
ok    controller tools      rustc, cargo, rsync, curl and ssh found
ok    spark-1 ssh           reached spark-1
ok    spark-1 nvidia-smi    NVIDIA GB10, <driver>
ok    spark-1 free memory   <available> GiB available of <total> GiB (GPU and CPU share this memory)
ok    spark-1 model files   <model path> (config, tokenizer, weights; vade install checks every SHA256)
ok    fabric                every fabric link answered ping
ok    engine health         ranks alive, tunnel up, /health ready, harness turns passed
PASS  <n> ok, 0 failed, <n> skipped in <time>
```

Run `vade doctor` while the server is up to check the engine too. Without a
server the `engine health` row says `skip  stopped; start: vade up`, and the
machine checks still decide PASS or FAIL.

In TCP mode the fabric row says `TCP; engine startup checks the addresses`.

Right after `vade up --fp8`, look for `rdma=true` (or `rdma=false` in
TCP mode) in `vade logs`. Check soon: `vade logs` shows only the last 64 KiB of
each rank's log, so the startup lines scroll away later.

<br>

---

## 8. First answer, chat, stop

**Health and model** (through the tunnel that `vade up --fp8` opened):

```sh
curl -s http://localhost:18080/health
curl -s http://localhost:18080/v1/models
```

Health shows `"status":"ready"` and `"model":"qwen3.8-flash-next-fp8"`.
Vade hashes the checkpoint in the background after it starts. In the same
health response, `vade.checkpoint.hash_state` reads `"pending"` until that
finishes and `"ready"` after. Wait for `"ready"` before you time anything.

**A first answer:**

```sh
curl -s http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"enable_thinking":false}'
```

You get JSON back, with the answer in `choices[0].message.content` and token
counts in `usage`.

**Send the greedy fields, as above.** On this FP8 server, any
sampling field you leave out falls back to the checkpoint's own sampled
settings (top_k 20, top_p 0.95, from its `generation_config.json`). That holds
even when you send no sampling fields at all, so the answer can change from run
to run. FP4-Local is different: a request that names no sampling fields at all
gets [greedy decoding](GLOSSARY.md#dg), the single most likely token at each
step.

**Chat with the coding assistant** in the current folder:

```sh
vade chat
```

**Stop everything** when you are done:

```sh
vade down
```

```text
Vade is stopped. The engines and the tunnel that vade started are closed.
```

> **If it doesn't fit.** Does `vade up --fp8` stop with "Requested
> size does not fit."? Lower `--context` or `--chats` and run it again. If it
> still fails, something else is holding memory: check `free -g` and
> `nvidia-smi` and stop what you find. Dropping the page cache won't help, because the
> free-memory figure Vade reads (MemAvailable) already counts it as free.

Something stuck? Run `vade status`, then `vade logs`, then `vade down`. Never
kill processes by name. See [CLI recovery](../cli/README.md#recovery) and the
[network troubleshooting table](network-setup.md#9-troubleshooting).

<br>

**Back:** [Hardware and network](network-setup.md) · **Next:** [README](../README.md)
