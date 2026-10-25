# Legacy FP8 fresh install: from zero to a first answer

**For a new install, start with [INSTALL.md](../INSTALL.md).** The current
FP4-Local model runs directly on one Spark with `vade launch`, no second
computer needed, and [First run](../harness/docs/first-run.md) explains that
local launcher. Driving Sparks from another computer (the controller route)
comes after that.

This page keeps the older FP8 checklist. It takes two DGX Sparks, yours or
rented, to a working Vade server that you run with the `vade` command. It uses
the controller route, and FP8 needs the explicit `--fp8-legacy` flag. The
historical FP8 evidence is in [Legacy FP8 installation](install-legacy-fp8.md).

An earlier test install covered the user-level steps on machines whose OS
and network were already set up and whose weights were already verified. Its
result file was outdated and has been [removed](../bench/receipts/README.md),
and that run does not prove that the current CLI route works from a blank
machine.

<br>

## The route at a glance

| # | Step | Where |
| --- | --- | --- |
| 0 | [Check what you have](#0-check-what-you-have) | — |
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
- [ ] **The standard 128 GB of unified memory on each Spark**, with no other GPU jobs
- [ ] **Enough free disk for the complete checkpoint, build output and download state**; we publish no measured minimum
- [ ] **The basic programs and download tools**: `cc`, `curl`, `git`, `rsync`, `ping`, `sha256sum`, `python3`, **`python3-venv`**. Have them ready before step 5.
- [ ] **A Mac** with Git and network access to both Sparks
- [ ] **Network access to GitHub**, where the Vade repository is public

<br>

---

## 1. Renting? Ask for these

Cloud and hosting providers rent Sparks as single machines or as cabled pairs.
**A cabled pair is fastest.** A rented pair usually comes as two Sparks and
one direct cable, with no ring (no cable that closes the Sparks into a loop),
and that is all Vade needs. Two Sparks on an ordinary network also work, more
slowly, in [TCP mode](#no-qsfp-link-use-tcp-mode).

Terms used below:

- **RDMA**: direct memory-to-memory transfers between Sparks.
- **RoCE switch**: a network switch that carries RDMA.
- **PAM**: the login system that applies limits such as locked memory
  (`memlock`); RDMA needs that limit set to unlimited.

Ask the provider before you pay:

| Question | Why it matters |
| --- | --- |
| "Are the two Sparks joined by a **200 Gb/s QSFP cable** (or a RoCE switch)?" | Then Vade uses fast **RDMA** on that link. Without it, use [TCP mode](#no-qsfp-link-use-tcp-mode). |
| "Can I have **sudo**, or will you run the OS setup for me?" | Fabric addresses, RDMA permissions and memlock need root once. |
| "Is SSH ordinary **OpenSSH with PAM**?" | For RDMA, the SSH alias Vade uses must give **unlimited locked memory**. Some managed SSH gateways skip that. (TCP mode does not need it.) |
| "Which **kernel** is installed?" | `7.0.0-1019-nvidia` needs NVIDIA's `kho=off` fix for multi-Spark RDMA. |
| "Are the **QSFP interfaces unconfigured**, or what addresses did you assign?" | You either set them up yourself or reuse theirs. |

**What changes on rented machines**

- You will not see the cable. Trust `ibdev2netdev`, `ethtool` and the
  [link checks](network-setup.md#7-check-every-link), not a diagram.
- Your Mac reaches the Sparks over the internet or a VPN. Make SSH aliases for
  them (step 3). `vade up --fp8-legacy` opens its own SSH tunnel; no ports
  need to be public.
- Download the weights **on the Sparks**: data-center internet is usually
  faster than home internet.
- **Ordinary Ethernet only, no QSFP link?** Use [TCP mode](#no-qsfp-link-use-tcp-mode).
- **Only one Spark?** This guide's Flash-Next FP8 checkpoint does not fit on one.
  Use the current FP4-Local [on-Spark install](../INSTALL.md#route-a-on-this-spark)
  and [first-run guide](../harness/docs/first-run.md).
- **Different home folders on the two rentals?** Both ranks must get the exact
  same absolute model and build paths. Write absolute paths (for example
  `/data/vade/models/...`) in the config instead of `~/`.
- **Ping blocked?** With RDMA, `vade install` and `vade doctor` check each link
  with ICMP ping; ask the provider to allow ping between the two Sparks. TCP
  mode does not use ping.
- **No Mac?** A Linux controller can run the SSH lifecycle and use chat, but
  the agent's shell tool and automatic checks require the macOS sandbox. The
  current one-Spark route needs neither a controller nor an SSH tunnel.

### No QSFP link? Use TCP mode

Vade uses RDMA by default. On Sparks joined only by ordinary Ethernet (common
with rentals), switch to **TCP mode** on the Mac before `vade install`,
`vade up --fp8-legacy` and `vade doctor`:

```sh
export SEQPAR_RDMA=0
```

| | RDMA (default) | TCP mode (`SEQPAR_RDMA=0`) |
| --- | --- | --- |
| Link | QSFP cable or RoCE switch | Any network the two Sparks share |
| `fabric.<peer>` in the config | Own address on the QSFP cable | Own ordinary address that the other Spark can reach |
| Unlimited locked memory, RDMA setup | Required | Not needed |
| Open between the Sparks | TCP 29960, 29961, RoCE UDP 4791, ping | TCP 29960, 29961 |
| `vade up --fp8-legacy` says | `… on 2 machines using RDMA.` | `… on 2 machines using TCP.` |
| Rank logs show | `rdma=true` | `rdma=false` |
| Speed | See [dated receipts](benchmarks.md) | No current measurement published here |

The setting is **not saved** in the config. Export it again in every new
terminal before `vade install`, `vade up --fp8-legacy` (including restarts) and
`vade doctor`.

<br>

---

## 2. Network and OS

Follow **[Hardware and network setup](network-setup.md)** until every link check
passes. In short:

1. Cable port 0 of spark-1 to port 1 of spark-2.
2. Give each cable link (rail) its own `/24` subnet with MTU 9000 (example: `10.77.0.1` ↔ `10.77.0.2`, `10.77.1.1` ↔ `10.77.1.2`).
3. Clone Vade on **each Spark** (the OS-setup scripts run from the repository root), then run the setup scripts: packages, network, RDMA access and unlimited locked memory (`memlock`), the kernel note, and SSH keys between the Sparks.
4. Check: jumbo ping, `ib_write_bw` in both directions, `ulimit -l` = `unlimited`.

Before the user-level install, an administrator installs any missing Ubuntu
packages. **Refresh the package lists before installing**, even on a fresh VM:

```sh
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential ca-certificates curl git openssh-client \
  rsync iputils-ping python3 python3-venv
```

Expected: exit 0 from both commands. Without the `python3-venv` package,
`python3 -m venv` cannot supply ensurepip. If “no installation candidate”
remains after refreshing lists, stop and have the administrator check package
sources; on Ubuntu 24.04 the dependency is `python3.12-venv`.

Before any model download, check that Python can create a virtual environment:

```sh
VADE_VENV_CHECK=$(mktemp -d)
python3 -m venv "$VADE_VENV_CHECK/check"
"$VADE_VENV_CHECK/check/bin/python" -m pip --version
```

Expected: a pip version and exit 0, normally in under a minute (an estimate).
Do not continue until these prerequisites and the link checks pass.

**Using [TCP mode](#no-qsfp-link-use-tcp-mode)?** Skip the QSFP, RDMA and memlock
steps. All you need is for the two Sparks to reach each other on TCP ports
29960 and 29961, plus SSH from spark-1 to spark-2 to copy the weights in step 5.

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
that login path. Fix it before you go on.

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

The complete snapshot includes weights, configuration, tokenizer and
supporting files. Check your free disk space against that snapshot plus the
build output. How long the download takes depends on your connection and the
service; we give no estimate here.

> `vade` does **not** download Flash-Next. You fetch it once with the
> repository's helper, `ops/fetch-model.sh`. The helper runs Hugging Face's own
> `hf` tool for the download and then checks every file.

**Download and verify on spark-1**, from the repository root you cloned in step 2.
Keep the folder name exactly `Qwen3.8-Flash-Next-FP8`: its lowercase name becomes
the model ID (`qwen3.8-flash-next-fp8`). The venv/pip check in step 2 must pass
first, and no other download may be writing into this folder.

```sh
sh ops/fetch-model.sh Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13 "$HOME/models/Qwen/Qwen3.8-Flash-Next-FP8"
```

Expected final line: `verified <file-count> files`.

The [helper](../ops/fetch-model.sh) installs its own copy of the `hf` tool
(`huggingface_hub` 2.2.0) in a private Python environment (venv) at
`MODEL_DIR.fetch/venv`, where
`MODEL_DIR` is the folder you passed. It checks the official checksums and
looks for missing or extra files, then writes and checks the trusted
`MODEL_DIR.sha256` manifest next to the model folder. While it verifies, the
helper moves the download's own metadata folder, `.cache/huggingface`, aside
so it is not counted as an extra file, and always puts it back, even on
failure or Ctrl-C. If `HF_TOKEN` is already set, the helper uses it without printing
it.

If the helper fails, stop, do what its one-line reason says and rerun the same
command; finished downloads are reused. Continue only after it succeeds. The
earlier test install mentioned at the top of this page used weights that were
already verified, so it does not show this helper fetching a full snapshot.

**Copy to spark-2** over the QSFP link (usually faster than downloading again;
in TCP mode use spark-2's ordinary address), then verify the copy. The key
`~/.ssh/vade-cluster` in the comment below is the peer key from step 6 of
[Spark system setup](spark-setup.md).

```sh
# on spark-1; 10.77.0.2 is spark-2's example fabric address
# (TCP mode: use any key that lets spark-1 log in to spark-2)
K=~/.ssh/vade-cluster
ssh -i $K 10.77.0.2 'mkdir -p models/Qwen'
rsync -a --info=progress2 --exclude '/.cache/huggingface/' -e "ssh -i $K" \
  ~/models/Qwen/Qwen3.8-Flash-Next-FP8/ 10.77.0.2:models/Qwen/Qwen3.8-Flash-Next-FP8/
scp -i $K ~/models/Qwen/Qwen3.8-Flash-Next-FP8.sha256 10.77.0.2:models/

# on spark-2
cd ~/models/Qwen/Qwen3.8-Flash-Next-FP8
sha256sum -c ~/models/Qwen3.8-Flash-Next-FP8.sha256   # every file must report OK
```

For the current FP4-Local checkpoint, follow
[INSTALL.md model acquisition](../INSTALL.md#step-3-get-the-model) instead.

<br>

---

## 6. Get Vade and configure it

**On the Mac.** `vade install` copies the source from this Mac clone to the
Sparks and builds there. (The Spark clones from step 2 are only for the
OS-setup scripts.)

The public repository is <https://github.com/wappy-x/vade>. Clone it and build the CLI:

```sh
git clone https://github.com/wappy-x/vade ~/vade
cd ~/vade
cargo build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
mkdir -m 700 -p .local
cp -n cli/examples/vade.conf .local/vade.conf
export VADE_CONFIG="$PWD/.local/vade.conf"
```

The two `export` lines last only for this terminal. In a new terminal, run them
again from the checkout (or add them to `~/.zshrc`), or `vade` will say the
config is missing.

`.local/` is ignored by Git, so your addresses never get committed.

**Edit `.local/vade.conf`.** Here is a filled-in example for two Sparks (the addresses are examples):

```ini
[global]
# Local paths are resolved beside this file (.local/).
engine_source = ../engine
harness_source = ../harness
cli_source = ../cli
stage_dir = bin
# Keep state_dir short: the SSH control socket lives here. macOS allows 103
# characters per socket path and OpenSSH adds 17. From a deep folder, use
# ~/.vade-state.
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
fabric.spark-2 = 10.77.0.1
model_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8
engine_source = ~/vade-source
cli_source = ~/vade-cli-source
stage_dir = ~/vade-stage
state_dir = ~/.local/state/vade

[machine spark-2]
ssh = spark-2
# spark-2's OWN address on the cable to spark-1.
fabric.spark-1 = 10.77.0.2
model_path = ~/models/Qwen/Qwen3.8-Flash-Next-FP8
engine_source = ~/vade-source
cli_source = ~/vade-cli-source
stage_dir = ~/vade-stage
state_dir = ~/.local/state/vade
```

Each Spark in the engine is a rank, and the first machine listed becomes
**rank 0**, the one that answers HTTP. More detail is in the
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

`<sha>` is the full 40-character Git revision that was built. The second line
is normal: nothing is running yet. For this legacy setup, start the server
with `vade up --fp8-legacy`, shown below.

Install refuses a checkout with uncommitted changes or untracked files. Your
`.local/` folder is ignored by Git, so it does not count.

**Start the server.**

```sh
vade up --fp8-legacy
```

```text
Starting qwen3.8-flash-next-fp8 on 2 machines using RDMA. Waiting for the model to be ready.
qwen3.8-flash-next-fp8: ready on 2 machines. MTP 2, context 131072.
Chat here: vade chat
```

Loading the weights takes minutes. `vade up --fp8-legacy` waits up to
`health_timeout_secs` (360 s by default).

**`vade up --fp8-legacy` stops with "Requested size does not fit."?** Read
`Asked`, `Available`, `Gap` and `Next`: they show the reservation per rank, the
budget left after reserves, the shortfall in GiB and bytes, and the supported
next steps. Lower `--context` or `--chats`, then run `vade up --fp8-legacy`
again.

After copying or downloading files, you may also see `Reclaimable cache
estimate:` in GiB and bytes, followed by `this is not a guarantee that CUDA can
allocate it.` On a Spark, Linux file cache shares memory with the GPU. Vade
already asks Linux to drop the model's own file cache; the estimate does not
promise that the context or chat count you asked for will fit.

**Check everything.**

```sh
vade status
vade doctor
```

`vade doctor` should print, among other lines:

```text
Controller tools: ready.
spark-1: Rust, driver, and build disk ready; RDMA (prerequisites checked) transport selected; model files found (shards checked after build).
spark-2: Rust, driver, and build disk ready; RDMA (prerequisites checked) transport selected; model files found (shards checked after build).
Fabric: 2 directed link checks passed. Transport ports are checked by engine startup.
Harness doctor: passed.
```

Without a server, `vade doctor` checks the machines and exits with code 1.
Start the server with `vade up --fp8-legacy` first, then run `vade doctor`
again.

In TCP mode the doctor lines say `transport: TCP`; with one Spark they say
`transport: local (one Spark; no fabric)`.

Right after `vade up --fp8-legacy`, look for `rdma=true` (or `rdma=false` in
TCP mode) in `vade logs`. It shows only the last 64 KiB of each rank's log, so
startup lines scroll away later.

<br>

---

## 8. First answer, chat, stop

**Health and model** (through the tunnel that `vade up --fp8-legacy` opened):

```sh
curl -s http://localhost:18080/health
curl -s http://localhost:18080/v1/models
```

Health shows `"status":"ready"` and `"model":"qwen3.8-flash-next-fp8"`.
Wait for `"hash_state":"ready"` (Vade finishes hashing the checkpoint in the
background) before you time anything.

**A first answer:**

```sh
curl -s http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next-fp8","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":64,"temperature":0,"top_p":1,"top_k":0,"repetition_penalty":1,"presence_penalty":0,"frequency_penalty":0,"enable_thinking":false}'
```

You get JSON with the answer in `choices[0].message.content` and token counts
in `usage`. Send every sampling field, as above. On this legacy FP8 server,
any field you leave out falls back to the checkpoint's own sampled settings
(top_k 20, top_p 0.95, from its `generation_config.json`), even when you send
none at all, so the answer can change from run to run. FP4-Local is different:
a request that names no sampling fields at all gets greedy decoding.

**Chat with the coding assistant** in the current folder:

```sh
vade chat
```

**Stop everything** when you are done:

```sh
vade down
```

```text
Vade: stopped. Its owned ranks and tunnel are closed.
```

Something stuck? Run `vade status`, then `vade logs`, then `vade down`. Never
kill processes by name. See [CLI recovery](../cli/README.md#recovery) and the
[network troubleshooting table](network-setup.md#9-troubleshooting).

<br>

**Back:** [Hardware and network](network-setup.md) · **Next:** [README](../README.md)
