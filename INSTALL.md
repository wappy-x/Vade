# Install Vade

This is the install entry point for **v0.1.0**, serving
**nvidia/Qwen3.8-Flash-Next-FP4-Local** as text-only chat. Start with one Spark
and no other computer. [Route B](#route-b-from-a-controller) adds a Linux or
macOS controller; [linked Sparks](#two-and-more-sparks) come after that.

Repository: <https://github.com/wappy-x/vade>.

## Before you start

Use a DGX Spark / NVIDIA GB10 running Linux `aarch64`, with a working
`nvidia-smi` and an idle GPU. The published benchmark runs and the install
replay used the **580.x driver series** (replay: **580.178.04**); this is a
tested version, not a claimed minimum. One Spark needs no fabric cable or RDMA.

The complete pinned checkpoint is **102,262,310,289 bytes** (102.26 GB /
95.24 GiB), including configuration and tokenizer files. The
[committed SHA256 manifest](cli/examples/qwen3.8-flash-next-fp4-local.sha256)
identifies that snapshot; its sizes were checked against the immutable
[publisher inventory](https://huggingface.co/api/models/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300?recursive=true).
Allow additional space for Rust, source/build output and download caches.
A source-route replay retained **0.72 GiB** across its checkout, build output
and fresh `VADE_HOME`, with compilers and weights already present. This excludes
the Rust installations and download caches; it is not a peak disk measurement.
The launcher requires **8 GiB free on the install volume** plus **1 GiB of
model-volume working room**; when they share a volume it requires **9 GiB
free after the model is present**. Those are admission requirements, not a
measured peak installation footprint. Fresh toolchain/download disk usage
remains unmeasured.

## Supported systems

| Role | Supported | Notes |
| --- | --- | --- |
| Engine | DGX Spark / GB10, Linux `aarch64` | No other GPU or OS runs the engine. |
| Route A shell | Any SSH client: Windows, Linux or macOS | CLI, engine and chat all run on the Spark. |
| Route B controller | Linux `aarch64` or `x86_64`, macOS | Needs Rust, `ssh`, `rsync`, `curl` and `shasum`. Default config: `~/.config/vade/config` on Linux, `~/Library/Application Support/vade/config` on macOS; `XDG_CONFIG_HOME` or `VADE_CONFIG` overrides it. Tested from macOS; Linux controller untested. |
| Windows controller | Not supported | The CLI does not build for Windows. Use Route A over SSH, or WSL 2 (untested). |

The agent's `workspace_shell` tool and workspace checks need the macOS sandbox.
On Linux, which includes every Route A session, the tool answers
`workspace_shell is unavailable on this platform: command execution requires the
macOS sandbox` and checks are disabled; reading, searching, line edits and
patches still work.

## Step 1: prerequisites

<a id="1-prepare-the-sparks"></a>
On the Spark, have its administrator install missing Ubuntu prerequisites:

```sh
sudo apt-get update && sudo apt-get install -y build-essential git curl rsync python3 python3-venv zstd
```

The model helper needs Python with venv/pip; the engine itself has no third-party
Rust dependencies. Check the existing driver and tools:

```sh
uname -m
nvidia-smi --query-gpu=name,driver_version,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
command -v cc git ssh rsync curl python3 sha256sum find sort nvidia-smi
```

Expect `aarch64`, GB10 and no active compute application. Stop an existing
workload through its own launcher before continuing. Do not change drivers or
system configuration through Vade.

<a id="2-install-the-pinned-rust-compilers"></a>
For a source build, if `rustup` is absent, download and inspect its installer
(bundle users can skip the Rust setup):

```sh
curl --proto '=https' --tlsv1.2 -sSf -o /tmp/vade-rustup-init.sh https://sh.rustup.rs
less /tmp/vade-rustup-init.sh
```

Press `q` after reading, then install without changing shell startup files:

```sh
sh /tmp/vade-rustup-init.sh -y --default-toolchain none --no-modify-path
```

For the source route, install the pinned host and GPU toolchains on each Spark
that builds source. A supplied launcher bundle does not need Rust:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
rustup toolchain install 1.99.0 --profile minimal
rustup toolchain install nightly-2026-10-01 --profile minimal --component rust-src,llvm-tools,llvm-bitcode-linker --target nvptx64-nvidia-cuda
```

A controller needs the host toolchain, a native linker, Git, OpenSSH, rsync,
curl and `shasum` (included with macOS; on Ubuntu it ships with `perl`). For linked Sparks, `ping` must
already be available on each machine; Ubuntu supplies it through
`iputils-ping`. Have the administrator prepare connectivity separately using
[network setup](docs/network-setup.md).

## Step 2: get the code

<a id="3-clone-and-build"></a>
On the Spark, clone the release repository and keep this shell at its root:

```sh
git clone --branch main https://github.com/wappy-x/vade.git
cd vade
```

Until publication, use an account with repository access or an existing Git
clone of the release source. Do not put credentials in the clone URL. Cloning
is the recommended path; a downloaded source archive (GitHub's Download ZIP or
`git archive`) also works and records its commit as the build id.

For the clone-and-build route, prepare the first kernel build while online,
after installing the pinned components above; bundle users skip this build:

```sh
(cd engine && CARGO_NET_OFFLINE=false cargo +1.99.0 kernels)
```

The kernel crate prints compiler warnings; the preparation succeeded when it
ends with `Finished` and exit status 0 (49-54 s on the release replay Spark with
a warm compiler cache). The kernel compiler's initial `build-std` dependencies may need downloading.
Local `vade launch` subsequently builds its private source snapshot offline;
preparing this cache avoids a first-run offline build failure. This preparation
needs no GPU. The measured install replay already had the compilers and their
cache; it was not a blank-toolchain replay.

### Optional release bundle

The [release workflow](.github/workflows/release.yml) produces a Linux aarch64
launcher bundle and `SHA256SUMS`. Public v0.1.0 assets are not assumed to exist
during private preview. Once that release is published, download into a new,
empty directory; the checksum file supplies the exact revision-bearing filename:

```sh
mkdir vade-launch &&
(cd vade-launch && \
  curl --fail --location --output SHA256SUMS https://github.com/wappy-x/vade/releases/download/v0.1.0/SHA256SUMS && \
  read -r BUNDLE_SHA256 BUNDLE < SHA256SUMS && \
  curl --fail --location --output "$BUNDLE" "https://github.com/wappy-x/vade/releases/download/v0.1.0/$BUNDLE" && \
  sha256sum -c SHA256SUMS && tar --zstd -xf "$BUNDLE")
```

Obtain both files from the trusted release; a checksum detects changed bytes,
not publisher identity. Extraction needs `tar` and `zstd`. This bundle replaces
the Rust builds, not the model download. Keep the clone for the fetch helper
and committed model manifest in step 3. After that model check, enter
`vade-launch`, run `unset VADE_CONFIG`, then `./vade launch` instead of
Route A's source-build block,
and use `./vade` for the following lifecycle commands. The launcher finds the
manifest beside its executable. See [bundle details](cli/docs/prebuilt.md) and
[first run](harness/docs/first-run.md). During preview, use the clone-and-build
route below or a trusted bundle supplied directly by the maintainer.

## Step 3: get the model

<a id="3-fetch-and-verify-the-qualified-checkpoint"></a>
<a id="4-put-the-official-weights-on-both-sparks"></a>
<a id="4-put-the-official-weights-on-each-spark"></a>
Use publisher revision
[`b9ce18a6887e18c089fdea8509a33e63429a3300`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/tree/b9ce18a6887e18c089fdea8509a33e63429a3300).
Preserve the published mixed precision and the complete checkpoint, including
index, tokenizer and configuration. The files at this revision download
**without a Hugging Face token or click-through access approval**. The publisher's
[NVIDIA Open Model Agreement terms](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local/blob/b9ce18a6887e18c089fdea8509a33e63429a3300/README.md)
still apply.

If the complete checkpoint is already at that path, skip the fetch and run the
manifest check below. Otherwise, from the Spark checkout root, run this literal one-line fetch:

```sh
sh ops/fetch-model.sh nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local"
```

The helper creates a private downloader environment, verifies the snapshot
against the pinned publisher revision, and writes a sibling `.sha256` manifest.
It prints nothing until it finishes, which can take hours (see
[download rates on rented VMs](docs/benchmarks.md#rented-cloud-gb10-vms)); a
successful run ends with `verified 33 files`. Over SSH, run it inside `tmux` or
under `nohup`. If it is interrupted, it prints
`Interrupted; run the same command again.`; do that.
Its download cache and Python environment use additional disk space. A complete
snapshot has the byte total stated above; copying only tensor shards is not
sufficient. Vade's `install`, `up` and `launch` do not download these weights.

Check the committed manifest, including when using an existing checkpoint or
a copy on another Spark. Run from the checkout root:

```sh
MODEL_SHA256_LIST="$PWD/cli/examples/qwen3.8-flash-next-fp4-local.sha256"
(cd "$HOME/models/nvidia/Qwen3.8-Flash-Next-FP4-Local" && sha256sum -c "$MODEL_SHA256_LIST")
```

Every file must pass; the replay Spark took 71 s. On macOS use `shasum -a 256 -c` instead.
A Route B Spark has no checkout: copy the committed manifest file to it and set
`MODEL_SHA256_LIST` to that copy's absolute path before the same check. Keep the checkpoint
basename `Qwen3.8-Flash-Next-FP4-Local`; it supplies the served model ID.

## Route A: on this Spark

<a id="one-spark-directly-on-the-spark"></a>
After steps above, from the checkout root in the same shell:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
unset VADE_CONFIG
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
vade launch
```

`vade launch` discovers the checkpoint under `~/models`, saves local settings,
builds the engine and chat program, waits for readiness, and opens chat. With a
fresh `VADE_HOME` the release replay reached ready in 181 s: 82 s building and
98 s loading the model.
Ask it to say hello, then type `/exit` to return to your shell. The engine stays
running. Continue in order:

```sh
vade status
vade doctor
vade chat
```

Type `/exit` again, then verify and shut down:

```sh
vade verify
vade down
vade status
```

`verify` compares repeated token IDs with drafting OFF and ON and writes
`vade-verify.json` in the current directory; it took 146 s on the replay Spark.
A pass prints `IDENTICAL: 20/20 runs, 1 machine, N tokens`, where N is the
reply's token count. It covers that request set; it is not the full cross-Spark
exactness check described in [architecture](docs/architecture.md#exactness-and-model-quality). Final status should say the
engine is stopped. Restart with `vade up`, or `vade launch` to start and chat.
See [first run](harness/docs/first-run.md) for the complete lifecycle.

The default API is `http://127.0.0.1:18080/v1`. Initial setup chooses another
free loopback port if that port is occupied. `vade status` prints the base
address, such as `http://127.0.0.1:18080`; the API is under its `/v1`. Chat
requests think by default, and the reasoning counts against the same
`max_tokens` as the answer; send `"enable_thinking": false` (or
`"reasoning_effort": "none"`) for the answer alone. Bare `vade verify` discovers the saved port from
`~/.vade/state/run.conf`. Local settings live in `~/.vade/config`, artifacts
under `~/.vade/stage`, ownership/logs under `~/.vade/state`, and chat history
under `~/.vade/projects` (never in the folder you chat from; `VADE_CHAT_IN_PROJECT=1`
keeps it in a git-ignored `./.vade`); use `vade logs` to read engine diagnostics.
`VADE_HOME` selects a different private root: keep it unchanged through launch,
chat, verify and shutdown.

Local default context is **32768** and output defaults to **min(context, 8192)**,
the same as a controller's `vade up`. The engine's own memory admission decides:
only if it refuses the default or saved context for memory does `vade launch`
retry 16384, 8192 and then 4096. The first one that starts is saved, and one line
names it, the refused sizes, any lowered output, and the `--context` to retry
the original; if none starts, the saved config is unchanged.
Explicit output is limited to **16384** and must fit context; an explicit
`--context` the engine refuses is an error with the engine's message, never
lowered silently. Drafting is ON and engine-selected; retained prefix cache is OFF.
The engine reports effective input/output limits through `/health`, because
input and output share context. Local `--chats` is unsupported.

Chat uses the agent when context is at least 8192 tokens; smaller local
contexts use direct text chat. On Linux, the agent's shell tool and `--check` command execution are unavailable:
they require the macOS sandbox. The launcher disables automatic workspace
checks on Linux. Chat and the other supported agent tools remain
available: reads and approved patches apply. If the model still asks to run a
command such as `cargo test`, the row shows `● Run: ...` with no result line, the
call returns `unsupported_platform`, and the model reports that it could not run
it. Exiting chat leaves the engine running until `vade down`.

## Route B: from a controller

<a id="5-start-the-two-spark-server"></a>
<a id="5-start-and-chat"></a>
Use another Linux or macOS computer with the controller prerequisites from
step 1. Clone the same source there using step 2's clone command (the controller
does not need the GPU kernel warm-up). Each Spark needs the verified checkpoint
from step 3 and SSH from the controller that works without an interactive password.
From the controller checkout root:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
cargo +1.99.0 build --release --offline --manifest-path cli/Cargo.toml
export PATH="$PWD/cli/target/release:$PATH"
mkdir -m 700 -p .local
cp -n cli/examples/fp4-local.conf .local/fp4-local.conf
export VADE_CONFIG="$PWD/.local/fp4-local.conf"
```

Edit that private file before proceeding:

- Set local `engine_source`, `harness_source` and `cli_source` to this checkout's
  component directories; choose private local `stage_dir` and `state_dir` paths.
  Put `state_dir` under your home, as the example does (`~/.vade-state`): it
  holds the ownership record that `vade status` and `vade down` need while a
  Spark engine runs, and macOS purges its temporary directories. On macOS,
  `/tmp` is a symlink and is refused; only a throwaway `stage_dir` may live
  under the system temporary directory.
- Keep the supplied model, model repository and pinned `model_revision`.
- In `[machine spark-a]`, set `ssh` and the absolute `model_path`. `ssh` is
  what you would type after `ssh`: `you@spark-a-address` (a host name or IPv4
  address, with or without `you@`) or an existing `~/.ssh/config` alias.
  Select dedicated remote source, stage and state directories owned by that user.
- Keep the default `local_port` and `serve_port` of **18080** unless occupied.
  The former is on this computer; the latter is on the serving Spark.

Then preview and start:

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

The curl command assumes the example's unchanged port; its JSON reply lists the
served model ID. On macOS, the agent's
sandboxed shell sees system and Homebrew tools only, so a Rust installed with
rustup under your home is not found (`sandbox-exec: execvp() of 'cargo' failed`).
`vade chat` has no option for this. Print its harness command with
`vade chat --dry-run`, then run that command yourself with
`--shell-toolchain ~/.rustup/toolchains/<toolchain>` appended (the directory
containing `bin/cargo`). `install` stages tracked
source and builds the remote engine/helper and the local harness/CLI. `up`
selects FP4-Local by default (`--fp4-local` is optional), starts the engine
and opens an SSH tunnel, recording both so that `vade down` can stop them. Keep `VADE_CONFIG` set to this same file.
After `/exit` from chat:

```sh
vade verify
vade down
```

The controller asks for the same defaults as Route A on any Spark count:
context **32768** and output **min(context, 8192)**; `--chats` does not change
them. Both routes start at 32768; only Route A lowers context, and only when the
engine refuses it for memory. 32768 is not a guarantee of memory capacity.
Drafting is engine-selected and ON; prefix
retention is OFF by default. The engine admits concurrent chats from context
and memory, and `vade status` prints the count it admitted. `--chats` is
recorded and forwarded, but FP4-Local admits chats from context and memory: it
changes neither context nor active rows (legacy FP8 `--chats` funds N chats).
For the single-chat
long-context preset, start with `vade up --context 262144` (add
`--output 16384` for longer replies). In the release replay on one Spark, the
earlier 262144-token controller preset served one chat at a time: `vade bench`
measured about 41 tok/s at every concurrency from 1 to 16 while first-token
time grew. At context 32768 the engine admitted 8 chats and the same Spark served 8 at
once, about 192 tok/s at both 8 and 16 requests; the controller now starts at
32768 for that reason. Known issue: in that replay the first `vade bench` after
start failed every 8- and 16-request run with `generation stopped before
completion` (the engine log reported an allocation that did not fit); two later
runs were clean. If you see it, run `vade logs`, then retry.

## Two and more Sparks

<a id="fp4-layer-chain"></a>
Use Route B and add a `[machine name]` section for each Spark. The
[example configuration](cli/examples/fp4-local.conf) includes a commented pair.
For each adjacent link, put `fabric.<peer>` in both machine sections: the value
is that section's machine's **own address** on the link toward the peer.
Addresses must already be configured and reachable. All ranks need identical
expanded absolute checkpoint and staged PTX paths; matching `~/` text is not
sufficient when account home paths differ.

The controller reaches **every** rank over SSH, including its own machine when
the controller is one of the Sparks: Vade has no separate local-rank path.
Plain `ssh = you@spark-a-address` and `ssh = you@spark-b-address` work for every
rank with no `~/.ssh/config` alias. Before `vade install`, check each rank from
the controller exactly as Vade connects, with no prompt:

```sh
ssh -o BatchMode=yes you@spark-a-address true
ssh -o BatchMode=yes you@spark-b-address true
```

`Host key verification failed` or `Permission denied` means that machine does not
yet trust the controller's key or the controller does not yet know that host
key; fix it with your usual SSH key setup (on a Spark that is also the
controller, that includes its own address). Running the controller on a separate
computer avoids SSH to itself. Vade never edits SSH settings. An alias is needed
only for settings a destination cannot carry, such as a jump host, a non-default
port or a specific key. A private `~/.ssh/config` block for that case, with the
alias then used as `ssh = spark-b`:

```text
Host spark-b
    HostName spark-b-address
    User you
    ProxyJump you@spark-a-address
    IdentityFile ~/.ssh/your-key
```

Reserve HTTP port **18080** and transport ports **29960** (`tp_port`) and
**29961** (`tp_port + 1`) on the selected Sparks. Keep `serve_port` distinct
from both transport ports. From the controller, after stopping an earlier run:

```sh
vade install
vade up --sparks 2
```

On the release replay, two RDMA-linked Sparks took 159 s for `vade install`
and 24 s for `vade up --sparks 2`. `vade verify` then reports `1 machine`:
the two ranks form one engine, not two independent engines.
`--sparks N` selects the first N configured machines. Alternatively,
`vade up --hosts spark-a,spark-b,spark-c` selects named machines in cable order.
Choose one selector. Selecting more than two Sparks opts into the layer chain;
run `vade install` from current source first so the staged build advertises
chain support. Each adjacent cable needs both endpoint addresses; no closing
cable is required. Construction checks do not establish device exactness for
an untested topology.

Linked Sparks use RDMA by default, requiring the existing verbs device/library
and suitable memlock limits. For TCP, set this **before install** and retain it
for `up` and `doctor`:

```sh
export SEQPAR_RDMA=0
```

TCP uses the configured reachable fabric addresses and needs no RDMA or QSFP
link. One local Spark needs no inter-Spark transport.

<a id="replicas"></a>
**Replicas.** When the model fits on one Spark, several Sparks serve many chats
better as independent copies than as one chain: `vade up --replicas 2` starts a
separate one-Spark server on each of the first two machines (or the machines
named in `--hosts`) and a local gateway on `local_port` that sends each request
to the replica with the fewest requests in flight. Requests whose messages
share their first 2048 bytes through the last user message prefer the replica
that served the previous such request while it is among the least busy;
otherwise load wins. Prefix retention is OFF by default, so cache hits from
this routing are unverified. A single chat runs at one-Spark speed; aggregate throughput grows with
the number of replicas. Every replica is the same one-Spark engine, so a request
gets the same token IDs whichever replica answers. Replicas need no `fabric.*`
entries or transport ports; their tunnels use the local ports right after
`local_port`. Each replica starts the controller defaults (context 32768,
output `min(context, 8192)`). Use the chain (`--sparks N` or `--hosts`) when the model is too
large for one Spark. `vade status` reports the gateway, and `vade down` stops the
gateway, the tunnels and every replica.

## Settings you may change

Stop the engine before changing launch settings, then repeat `vade up`
with the desired flags. Local supported overrides persist in `~/.vade/config`;
keep controller flags explicit for a new launch.

<a id="defaults-by-route"></a>
These are the initial defaults with no explicit overrides or inherited `MTP` or
cache settings. Context includes the input and generated output; `/health`
reports the effective request limits.

| Route | Context tokens | Output tokens | Drafting (MTP) | Prefix retention |
| --- | --- | --- | --- | --- |
| On this Spark: `vade launch` / `vade up` | 32768; lowered through 16384, 8192, 4096 only if the engine refuses it for memory | `min(context, 8192)` | ON, engine-selected | OFF |
| Controller: `vade up`, any Spark count, with or without `--chats` | 32768 | `min(context, 8192)` | ON, engine-selected | OFF |
| Controller: `vade up --replicas N`, each replica | 32768 | `min(context, 8192)` | ON, engine-selected | OFF |

`--context 262144` selects the single-chat long-context preset on either route
(add `--output 16384` for longer replies); on the Spark it is still subject to
the engine's memory admission, and a refusal is reported, not lowered.

| Setting | Meaning |
| --- | --- |
| `--context` | Total input plus output budget; a local value must fit the checkpoint limit and the engine's memory admission, and an explicit value the engine refuses is an error, not lowered. See the route defaults above. |
| `--output` | Output ceiling, no larger than context or the FP4-Local limit of 16384. |
| `--mtp` | Explicit drafting request; zero disables drafting. `--mtp` overrides `MTP`; otherwise the engine chooses. |
| `--chats` | Controller only: recorded and forwarded, but FP4-Local admits chats from context and memory; it changes neither context nor active rows (legacy FP8 `--chats` funds N chats). Refused locally; the engine chooses concurrency from context and memory. |
| `--memory-bytes` | Explicit memory budget, still subject to admission and actual free memory. |
| `--cache-bytes 0` | Locally disables prefix retention. Positive local cache sizes are refused. |

Prefix retention (reusing a cached prompt prefix across requests) is OFF by
default and supported only on one Spark. To opt in
for **local Route A**, restart from the same local environment:

```sh
vade down
VADE_FLASHNEXT_PREFIX_CACHE=1 vade up
```

A saved local `--cache-bytes 0` overrides this opt-in. Merely setting the variable
in a controller shell does not forward it to a remote worker. The
[prefix switch](engine/crates/serve/src/prefix_switch.rs) is the engine authority;
a request's `cache:true` alone does not enable retention.

## Client gateway

To put Vade's gateway between your own client and the engine, build the
harness binaries from the checkout and run:

```sh
cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml --bins
harness/target/release/vade-gateway --upstream http://127.0.0.1:18080/v1 --bind 127.0.0.1:8787
```

Use the engine endpoint from `vade status` if it differs. The client gateway
base is `http://127.0.0.1:8787/v1`; Ctrl-C stops the gateway. See
[harness documentation](harness/README.md) for supported API and tool behavior.
The gateway answers 421 to any Host other than `127.0.0.1`, `localhost` or
`[::1]` (bare or with its port) and 403 to any request carrying an `Origin` header, so a
web page cannot drive it through your browser; command-line clients send neither
and pass unchanged. To use a browser client deliberately, add
`--allow-origin https://app.example` (exact origin, repeatable; `--allow-host NAME`
admits another Host name) and preferably `--token TOKEN` or `VADE_GATEWAY_TOKEN`,
which then requires `Authorization: Bearer TOKEN` on every request.

<a id="6-run-the-timing-benchmark"></a>
## Run the benchmark

With the engine running and checkpoint hashing ready:

```sh
vade bench
```

Keep the same `VADE_HOME` / `VADE_CONFIG` used to start the engine. The client
uses the endpoint recorded by the local launch or the controller's configured port,
reads the model from `/health`, and writes
`./vade-bench.json`. If health omits the model, the fallback is
`qwen3.8-flash-next-fp4-local`. The default uses a bundled offline writing prompt
and requests the server's drafting policy, including FP4-Local's drafting-on
default. Receipts record each requested draft mode, the server build SHA and
server-reported MTP depth; response-reported modes and depths are retained
separately in `draft_observations`. A reported depth of zero alone does not
establish that drafting was off.

To measure plain decoding on the same running engine:

```sh
vade bench --draft plain --output vade-bench-plain.json
```

This sends `"draft":"plain"` in every request; only that explicit policy is
labelled drafting off. `--draft strict` requests strict drafting. Neither
override needs a server restart. Use `--url URL`, `--model ID`, `--output PATH`
or `--prompts FILE` to override the endpoint, model, receipt or bundled prompt.
Each nonempty UTF-8 line in the prompt file is one retained prompt. Requests
use greedy sampling, cache off and thinking off, and ignore EOS for a fixed
output budget. Short or incomplete responses make the cohort ineligible for
a rate; `--max-tokens` changes that budget.

Read [benchmark recipes and results](docs/benchmarks.md) for workloads, receipt
fields and measurement limits. A host test or short greeting is not a throughput
claim. Retain the exact source revision, checkpoint, settings and raw results.

## First-run problems

These diagnostic excerpts come from the CLI. Preserve ownership records and
use `vade logs` / `vade doctor` before retrying.

| Diagnostic | Next step |
| --- | --- |
| `The engine refused --context N for memory` | The engine's admission message follows. Choose a smaller `--context`, or omit it so the engine's admission selects one, then run `vade doctor`. Explicit requests are not silently reduced. |
| `Requested size does not fit.` / `Not enough memory` | The message names MemAvailable, the required bytes and the ceiling (MemAvailable less 10%, at least 2 GiB). Page cache already counts as free, so do not drop caches: stop other processes holding memory or lower the context or chats. A local launch or install also stops here while another engine, including another Vade installation, holds the GPU memory; stop that engine with its own `vade down` or original launcher, then retry. |
| `This machine's GPU is in use.` | Controller `vade up` prefixes it with the machine name. Stop the workload with its original launcher; run `vade doctor` and retry. Never kill processes by name. |
| `Install cache preparation failed; the previous activation was kept.` | Controller `vade install` prints this when a Spark's GPU is busy or its files fail checks. The machine name and the helper's own reason come first (for example `This machine's GPU is in use.`); address that reason, or read the retained log whose path is printed just above it, then rerun `vade install`. |
| `Port ... is in use; close the application using it and retry vade launch.` (`vade up` names itself) | Find the program using the port and stop it through its own controls. Fresh local setup can choose another port; saved launches reuse the saved port. On a controller, choose free `local_port` and `serve_port` values in the private config. |
| `Model loading timed out` | Run `vade logs`, then `vade doctor`; inspect the retained startup log and retry only after addressing the cause. |
| `staged hashes differ from the manifest. Nothing started; run vade install.` | Controller `vade up`, prefixed with the machine name: restage matching engine/PTX with `vade install`. Local `vade launch` / `vade up` rebuild a changed staged build automatically. For stale PTX during a source build, rerun `cargo +1.99.0 kernels` from `engine/` before building again. |
| `generation stopped before completion` | Seen intermittently in the release replay on the first 8- and 16-request `vade bench` after a controller start; the engine log reported an allocation that did not fit. Run `vade logs`, then retry; later runs were clean. |
| `Auto-compaction deferred: safe compaction cannot fit a useful reply reservation; history retained.` | Can appear in a long chat at an 8192 context, where output equals context, when no safe rewrite fits; a fresh chat no longer prints it. History is kept and the reply still arrives; `/new` starts a fresh chat. |
| `Build GPU kernels failed; check the retained build log and the existing pinned Rust toolchains.` | Read the retained build log; complete the pinned toolchain installation in step 1 and the online kernel-cache preparation in step 2 before retrying the offline local install. |

If shutdown cannot establish process identity, follow
[CLI recovery](cli/README.md#recovery); keep the state instead of deleting it.

<a id="7-stop-the-server"></a>
## Stop the engine

Exit chat, keep the same `VADE_HOME` / `VADE_CONFIG`, and run `vade down`.
This stops the engine this installation started and, on Route B, its SSH tunnel. Use `vade status`
to confirm shutdown.

## Legacy models

The older Flash-Next FP8 profile and the separate Qwen3-8B catalog have different
checkpoints and launch behavior. Start with [legacy installation](docs/install-legacy-fp8.md)
and the [CLI reference](cli/README.md); the Qwen3-8B source recipe is in the
[legacy one-Spark quickstart](docs/quickstart-one-spark.md). Do not mix those
weights, model IDs or defaults with this FP4-Local route.
