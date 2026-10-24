# vade

Run nvidia/Qwen3.8-Flash-Next-FP4-Local on one DGX Spark with `vade launch`, chat, and stop your own engine. Rust 2024, standard library only, one binary. Configured controller deployments use `vade up` for one or more Sparks; `vade run qwen3-8b` is the separate legacy model workflow. `vade test` runs the source build, Flash exactness and saved-input evaluation checks; see the [gate guide](docs/test.md) and [evaluation guide](evals.md).

For the current FP4-Local model on 1–24 ranks (one Spark per rank), use the
[FP4-Local launcher guide](docs/fp4-local.md) and `vade up`.
FP4-Local is the default profile; `--fp4-local` remains accepted. MTP selection
stays with the engine unless you override it. Legacy FP8 needs `--fp8-legacy`.

On the Spark, with no `VADE_CONFIG`, run `vade launch` (or `vade up`) to discover
the checkpoint under `~/models`, install, and serve it locally. Local defaults
are context 32768 and output up to 8192, as from a controller, and lower only
when the model requires it or the engine's memory admission refuses the
default or saved context: then startup retries 16384, 8192 and 4096, saves the
first that starts and prints it. `--context` and `--output` can raise these limits, with output capped
at 16384 tokens and the configured context; an explicit `--context` the engine
refuses is an error with the engine's memory message. Local state lives in `~/.vade`, or
`VADE_HOME` when set. `vade verify` and `vade chat` use the saved launcher port,
normally 18080; `vade down` stops the owned engine. See the
[on-Spark first-run guide](../harness/docs/first-run.md).

Chat history never lands in your project. `vade chat` and `vade launch` (and
`vade run qwen3-8b` chat) keep agent session journals and typed-input recall in
private folders under `~/.vade/projects/NAME-HASH/` (or `$VADE_HOME/projects/`),
one per project folder, keyed by a hash of its absolute path; each has a
`workspace` file naming that folder. So git in the folder where you run `vade chat`
never sees a conversation. To keep the old in-project location, set
`VADE_CHAT_IN_PROJECT=1`: state then goes to `./.vade/`, which gets a `.gitignore`
with `*` (only `.vade/check` stays visible to git). The harness flags are
`--vade-home DIR` and `--state-in-project`. Chats saved in `./.vade` by older
versions stay there; resume them with `VADE_CHAT_IN_PROJECT=1`.
Local `vade doctor` checks the driver, resources, model, installation and engine
health. Local `vade update` restages the current bundle or rebuilds the current
source and explains the restart command.
Local source installs work from a Git clone (recommended) or from a downloaded
source archive (GitHub's Download ZIP or `git archive`): the archive records its
commit as the build id in `.vade-build-id`, and an archive without that id, or
edited after download, builds as `unknown` with a notice. A source archive is
not a launcher bundle; to stage supplied programs instead, set `VADE_BUNDLE` to
an existing local launcher bundle, then run `vade install`.
Local startup forwards `VADE_FLASHNEXT_PREFIX_CACHE=1` to opt into prefix
retention; unset or `0` keeps retention off. Set it before starting the engine.
Local `--cache-bytes 0` is saved and forces that switch off, including when an
opt-in is inherited. Positive `--cache-bytes` values and local `--chats` are
refused before saving: locally the FP4 service chooses its session count from
memory, and the local launcher exposes no explicit slot or positive
retention-cap control.

Before checking machines, the launcher prints the selected model, machine names,
context and output limits, requested drafting/concurrency policy, transport and
local endpoint. Use `vade up --dry-run` to review this plan and the
commands without connecting or writing launch state.

`vade --version` (also `vade version` or `vade -V`) prints the CLI package version
without loading your configuration or connecting to a machine. Include it when
reporting an installation or launcher problem.

Launcher errors identify the command or option to fix. Diagnostics, command
previews and displayed server logs escape terminal controls and direction
overrides; ordinary Unicode and log line breaks remain readable. This affects
display only: saved logs, ownership records, command arguments and wire requests
keep their original bytes. Invalid commands and options fail before config or
network work.

`vade status --json` emits one `vade.status.v1` JSON object for scripts. It reports
saved phase, model/build, settings, endpoint, per-rank process identity, observed
health and errors. Ownership tokens and credentials are never status fields.
Checks share a 15-second deadline; an unreachable rank cannot extend every later
check by another timeout. Exit 0 means stopped or ready; exit 1 means an error or
a recorded engine that needs attention. A stopped engine has `ready:false`.

Add `--dry-run` to read saved state without SSH, HTTP or worker probes. A recorded
running engine then has `ready:null`, and unchecked observations are `null`, not
healthy. A valid offline snapshot exits 0. Missing config or corrupt records
still produce JSON with `phase:"unknown"`, `ready:null`, errors and exit 1. Invalid
command syntax keeps the normal stderr diagnostic. Status never starts, stops or
repairs a server, and it preserves ownership records for recovery.

The gate commands work without `VADE_CONFIG`, `VADE_HOME` or authentication
files. They accept the existing gate options and explicit inputs. They do not
connect to configured machines, install toolchains or own server startup.
Use explicit assigned inputs and keep receipts outside the repository.

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

The ABI command compiles twelve Unix flag assertions without linking or running
target code; the selected compiler must have that target's Rust libraries.

For the legacy Qwen3-8B catalog, start with the [Qwen3-8B section of the one-Spark quickstart](../docs/quickstart-one-spark.md#qwen3-8b-bf16-ubuntu-to-a-first-answer), including its optional CLI step. The current FP4-Local route is [INSTALL.md](../INSTALL.md).
On the Spark, with `vade` already installed:

```sh
vade run qwen3-8b
vade chat
vade status
vade down
```

`run` checks Linux/aarch64, one GB10 and its driver, compute jobs, free GPU/unified RAM, free disk, and Linux pidfd support before acquiring anything. It needs curl and sha256sum. On GB10, `nvidia-smi` may report GPU memory as N/A; the check uses available unified RAM and reports that number. A new model needs 25 GiB free disk and an idle GPU with 24 GiB free memory. Existing verified weights need 1 GiB disk headroom. Vade changes no OS, driver, network or sudo configuration.

Source builds additionally need git, cc, tar, Rust 1.99.0 and nightly-2026-10-01 with the NVPTX target and rust-src, llvm-tools and llvm-bitcode-linker components. Missing prerequisites are reported before downloading. Vade does not install toolchains. It fetches the Vade monorepo selected by [`VADE_REPO_URL`](src/repository.rs), selects `engine/` and `harness/`, saves each full commit before building, and keeps those commits for retries. Git access must already work. Clean local monorepo roots or component directories can be supplied without building inside them:

```sh
vade run qwen3-8b --engine-source /path/to/engine --harness-source /path/to/harness
```

Use an existing official BF16 checkpoint without weight downloads:

```sh
vade run qwen3-8b --weights-source /path/to/Qwen3-8B
```

The supplied directory is read-only. Every pinned payload must be present and match its SHA-256; missing or altered files fail without downloading replacements. The path is saved in the control home. A directory named `Qwen3-8B` is read in place and needs 1 GiB of disk headroom. Other names are imported into a verified canonical `Qwen3-8B` directory inside the control home, because the engine uses that name as its API model ID; this requires 25 GiB initially. Subsequent runs rehash every payload in the directory the engine uses.

For `run qwen3-8b`, `VADE_HOME` selects a dedicated private directory, defaulting to `XDG_DATA_HOME/vade` or the user's `.local/share/vade` (the FP4-Local launcher uses `~/.vade`). Source snapshots, generations, revision-specific models, logs and ownership records stay there. Re-run the same command to resume a download, build or doctor check. A running owned server is reused after its process identity, listener, artifacts and model bytes pass verification. Uncertain startup identities retain their records and require `vade logs`, `vade down`, or the Recovery steps below. Incomplete doctor/answer checks leave the server recorded for retry or shutdown.

Local source builds save private ownership and execution records under `build-jobs` before starting. A supervisor keeps a session anchor alive through compiler exit, checks its boot and birth identity before stopping the whole group, and proves the group stopped before marking the job complete. Failure, timeout, Ctrl-C and loss of the calling CLI trigger cleanup. Retry and `vade down` recover recorded jobs through the supervisor; incomplete identities or a lost supervisor retain evidence and refuse signals. Keep those job directories for inspection.

Ctrl-C during readiness stops a server started by that invocation and releases the operation lock after cleanup. A server already running before the invocation remains recorded. A missing saved `--engine-source` or `--harness-source` checkout fails without fetching, even when its build snapshot remains. Restore the original checkout, or deliberately select a fresh `VADE_HOME` without the local source options to choose network acquisition into managed storage.

The catalog pins official Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`. All five BF16 shards, the index and six tokenizer/configuration files have retained SHA-256 digests in `src/catalog.rs`. Downloads use revision-specific `.part` files and curl range resume; every payload is hashed before publication and on every reuse. Old `.ok` files prove nothing. Extra model payloads and altered published weights are refused. No draft model or numerical conversion is acquired; the server uses explicit plain drafting and the engine's default arithmetic.

Readiness requires the exact recorded process to own the loopback listener and return ready health for qwen3-8b. The command streams a short greedy first answer, reports acquisition/build, download, verification, load, first-content and answer times, then runs the harness doctor with a 120-second probe budget. Missing GPU byte telemetry is reported as unknown; reported device bytes exclude untracked driver storage. `vade chat` opens basic harness text chat on Linux. The final output includes the harness command and a Claude Code environment command; printing that command does not establish live Claude Code chat or tool compatibility.

A locally supplied, trusted release artifact avoids all Rust/build prerequisites:

```sh
vade run qwen3-8b --release /path/to/release
```

The directory must contain regular `spark-serve`, `vade-harness`, `kernels.ptx` files and `release.conf` with `version=1`, `target=linux-aarch64-gb10`, full `engine_revision` and `harness_revision` Git SHAs, and `engine_sha256`, `harness_sha256`, `ptx_sha256` SHA-256 digests. Binaries need executable permissions. Vade verifies the supplied files, verifies the staged copies, publishes an immutable generation, and atomically changes one generation-ID record. The supplied manifest is the trust input; no release URL or signature service exists in this round, and no download installer is advertised.

Lifecycle commands accept `--dry-run`. `vade run qwen3-8b --dry-run` previews checks and the flow without files, commands or downloads. `--port` changes the local port (default 8080). Keep the original port when resuming an owned server. Use `VADE_CONFIG` explicitly to select the separate controller commands below; keep it set to the same file for controller `up`, `chat`, `status`, `doctor`, `logs` and `down`, even if a local installation exists. Unset it for the local lifecycle. Generated local defaults select one installation consistently: owner defaults use bare `up`/`launch`, while Qwen3-8B setup uses `run qwen3-8b`. Conflicting starts are refused before launch. Local `up` and `launch` accept `--fp4-local`, `--context` and `--output`; stop the owned engine before changing its limits. Use a fresh `VADE_HOME` when keeping both local installations.

## Verify exactness

With a drafting-capable engine already running, `vade verify` checks the same
bundled prompt 20 times with drafting ON and OFF. Every response must match
the first response's token IDs byte for byte, including across machines when
you select a second engine. Each run includes both drafting modes, so the
default makes 40 generation requests per engine. It uses explicit greedy
sampling, disables the prompt cache, and checks 128 generated tokens by
default. It continues through end-of-answer tokens to check that fixed length.
The bundled `short`, `2k` and `8k` prompts need no downloads; the longer names
describe approximate prompt lengths, which depend on the model's tokenizer.

```sh
vade verify
vade verify -n 50 --prompt 2k
vade verify --url http://127.0.0.1:18080 --prompt 8k --max-tokens 256
```

The command prints one plain result line, exits 0 for `IDENTICAL` or 1 for a
difference or an incomplete check, and writes `vade-verify.json` in the current
directory. A difference identifies the run, token position, two IDs and their
text. A pass applies to the prompt, model, output length and engines recorded
in the receipt; it does not prove every possible prompt. `vade verify --help`
lists the endpoint and receipt options. The check uses the existing engine
connection and never starts a server, changes drafting defaults, or downloads
models or prompts.

To compare a second independently running engine, add `verify_url` to its
existing machine section in your `VADE_CONFIG`. Use the HTTP endpoint already
reachable from this computer, such as an existing local tunnel:

```ini
[machine spark-b]
ssh = spark-b-ssh-alias
verify_url = http://127.0.0.1:18081
```

Keep the rest of that machine's configuration. Then run `vade verify --machine
spark-b`; the primary engine and `spark-b` both run every ON/OFF pair. Merely
adding a rank to one distributed engine is not a second independent engine
comparison.

## Measure a running server

`vade bench` measures the wait before generated output and the total answer
tokens per second seen by the client. After `vade launch` or `vade up`, run it
with no flags in another terminal. Keep the same `VADE_HOME` and `VADE_CONFIG`
as the running installation. The client discovers its recorded endpoint, as
`vade verify` does, reads the model ID from `/health`, and writes
`./vade-bench.json` in the current directory.

```sh
vade bench
vade bench --draft plain --output vade-bench-plain.json
vade bench --self-test
vade bench --help
```

The default measures the server as served, including the FP4-Local drafting-on
default. Receipts record the draft mode requested for every request (server
default, plain, or strict), the build SHA from `/health`, and its reported MTP
depth verbatim. The depth is labelled "server-reported (known to read 0 on the
FP4-Local path until the health fix lands)"; that label text predates the
v0.1.0 fix, and the FP4-Local server now reports its served depth. Zero alone
does not establish that drafting was off. `draft_observations` retains response-reported modes and
depths separately. `--draft plain` sends `"draft":"plain"` in every request;
only receipts where every request asked for plain label drafting off.
`--draft strict` requests strict drafting. Neither override requires a server
restart.

For a server outside the selected installation, supply `--url URL`. Override
the health model with `--model ID`; if health omits it, the fallback is
`qwen3.8-flash-next-fp4-local`. Use `--output PATH` to keep separate receipts.
The default uses a bundled offline writing prompt; `--prompts FILE` replaces
it with retained prose, code and reasoning prompts, one per nonempty UTF-8
line. Every request sends explicit greedy sampling, cache off and thinking
off, and ignores EOS to generate the fixed output budget. Completion counts
come from server usage, never stream chunks.
An early stop or failed chat makes the whole cohort ineligible for a rate.
Cold 8k probes, alternating server arms, observer commands and frozen request
hashes use the [timing options](../docs/benchmarks.md#run-vade-bench). HTTP is
supported; for a remote server use an existing local SSH tunnel. TLS is outside
the standard-library-only client.

The older Rust development workloads are available through
`vade bench --suite`; see [the suite guide](../engine/tools/bench/README.md).
There is no separate `vade-bench` executable. Its crate is linked into `vade`.
A self-test checks local streams and arithmetic, not GPU performance.

## Multi-Spark source setup

Choose the checkpoint profile deliberately:

| Profile | Config | Start |
| --- | --- | --- |
| Current FP4-Local, 1–24 Sparks | [fp4-local.conf](examples/fp4-local.conf) | `vade up` (add `--sparks N` or ordered `--hosts` for a chain, or `--replicas N` for independent one-Spark copies) |
| Legacy FP8, 2/4/6-machine interface | [vade.conf](examples/vade.conf) | `vade up --fp8-legacy` |

Start with [INSTALL.md](../INSTALL.md). With a controller config, FP4-Local
asks for the same defaults as `vade launch` on the Spark, on any Spark count and
with or without `--chats`: context 32768 and output `min(context, 8192)`. Both
start at 32768; only `vade launch` lowers context, and only when the engine
refuses it for memory. The
engine admits concurrent chats from context and memory; `vade status` prints
the admitted count. An explicit `--context` wins, through 262144
(`--context 262144` is the single-chat long-context preset); explicit
`--output` reaches 16384. MTP stays engine-selected unless explicitly
overridden. A request
past the configured context is refused, and that refusal names `--context`.
Explicit
3–24 Spark chains require a current staged engine; older staged builds refuse
these counts before any remote command. The one-Spark path has a live
launcher/direct/gateway/cancel smoke. Larger-chain launch construction and
recovery, and two-Spark TCP selection, have host proofs; live qualification
for those launcher paths remains required.

Build the command on the controller (Linux or macOS), from the monorepo root:

```sh
cargo build --release --offline --manifest-path cli/Cargo.toml
./cli/target/release/vade --help
```

Copy the matching example above, replace its placeholders, and set `VADE_CONFIG` to that file. The default is `~/Library/Application Support/vade/config` on macOS or `~/.config/vade/config` on Linux; `XDG_CONFIG_HOME` overrides either. Relative local paths resolve beside the config. Machine paths start with `/` or `~/`; `ssh` takes an existing alias, a host name or IPv4 address, optionally as `user@host`.

Preview the setup first:

```sh
export VADE_CONFIG=/path/to/vade.conf
./cli/target/release/vade install --dry-run
```

On a machine window, `vade install` checks controller tools and every configured machine, uses the optional exact-source prebuilt bundle described below, or copies tracked sources and builds the engine and helper there. It copies a verified optional host-tool pair or builds the harness and CLI on the controller; see [prebuilt installation](docs/prebuilt.md) for the host artifacts and preparation behavior. It stages fresh build generations, saves their private manifests and SHA256 hashes, then atomically changes a generation-ID activation record. Old generations remain available for rollback; commands bind the manifest and harness to one resolved generation. Install prints the immutable generation CLI path; put that generation directory on your PATH, or continue using this bootstrap CLI, which resolves the activation record once per command. Use dedicated private directories for staging, state, and the manifest; existing shared directories are refused with their permissions unchanged. Machine builds reuse `build-cache/engine` under `engine_source` and `build-cache/cli` under `cli_source`. Published builds stay available for running engines; install never prunes them.

For prebuilt installation, add just one optional key under `[global]`:

```ini
prebuilt_dir = /path/to/prebuilt-bundles
```

A bundle is a directory named `<engine-sha40>-<cli-sha40>` inside that root,
containing `flash-serve`, `flash.ptx`, `vade` (the remote helper), and `manifest.conf`:

```ini
format=vade-prebuilt-v1
target=aarch64-unknown-linux-gnu
engine_sha=<engine-sha40>
cli_sha=<cli-sha40>
flash-serve.sha256=<sha256-of-flash-serve>
flash.ptx.sha256=<sha256-of-flash.ptx>
vade.sha256=<sha256-of-helper>
```

Supply artifacts built from those exact source revisions. The directory and
manifest are your trust input; take them from a trusted prebuild or verified
release. Vade checks every SHA256 locally and again after transfer, and checks
that binaries are AArch64 ELF64. An absent, incompatible or corrupt bundle falls
back to a fresh source build after checking the source toolchain. Rejected bytes
are never executed, and a failed fallback leaves the active generation unchanged.
With a valid bundle, machines need the runtime tools above and 1 GiB of free
install space, with no Rust/NVPTX compiler requirement.
The helper is verified before execution; shard inventory checks then overlap the
engine/PTX copy (or engine compilation). A repeat install rechecks the remote
hashes and reuses unchanged staged artifacts. No release service or daemon is added.

The model presence check caches a stamp in the machine state directory keyed by
a digest of the recursive file inventory (path, inode, size and modification/change
times). An unchanged inventory skips index parsing and shard checks; changed or
missing files force a check. This is not a substitute for the trusted full-file
checkpoint checksums below. A matching stamp is also used during startup.

`vade up` without flags preserves the recorded launch options and reuses a healthy
engine and tunnel only when the current configuration and activated build still
match. Explicit launch requests and `vade chat` check the same configuration and
build before reusing the server. After changing settings or installing a new
build, run `vade down` before relaunching. An older session that lacks the details
needed to verify reuse also needs `vade down` and a fresh `vade up`.
With an unchanged configuration and build, exit chat and reopen it without
loading weights again.
`vade down` releases the engine. Each PTX hash gets a persistent private CUDA JIT
cache under the machine state directory; it survives down/up and new installs.
A new PTX hash pays JIT unless installation has already prepared its cache or a
matching trusted compiled image is available; see [prebuilt installation](docs/prebuilt.md).
Required memory preparation stays before ready; no startup or arithmetic work is
silently omitted.

Each machine's paired source snapshot contains sibling `engine/` and `cli/`
directories under `engine_source/revisions/<snapshot>/`. The CLI builds against
that snapshot's engine benchmark library. Remote `cli_source` supplies the CLI
build-cache location; it does not hold a separate unpaired CLI source snapshot.

Remote source, cache and staging roots must belong to the SSH account and exclude
all group/other permissions. Their ancestors must belong to that account or root,
allow no group/other writes and contain no symlinks; shared temporary directories
are refused. Reused caches also refuse symlinks, foreign-owned entries and shared
writes. Install preserves existing permissions and creates each source/build
generation exclusively. An incomplete generation remains unreferenced; retry
uses a fresh generation instead of reusing its contents.

Source installation checks Rust, the CUDA driver, at least 10 GiB of build disk, and both directions of every configured fabric link. RDMA fabric checks use ICMP ping; TCP mode skips them. Engine startup verifies the actual transport and health. Machines need Linux with pidfd support, Rust with the engine's pinned toolchain and NVPTX build prerequisites, OpenSSH, rsync, curl, sha256sum, ping, and nvidia-smi. The controller needs Rust, OpenSSH, rsync, curl, and shasum. Offline builds require the source toolchains already installed.

Flash-Next weights stay under your control. Missing or incomplete weights report
the required checkpoint, revision and destination. Follow the
[verified-copy and trusted-manifest procedure](../INSTALL.md#4-put-the-official-weights-on-both-sparks),
then rerun `vade install`. This route has no Flash-Next downloader. Sharded
checkpoints must have every nonempty shard named in their index.

| Command | What happens |
| --- | --- |
| `vade` | Short status and four choices: start, chat, stop, doctor. |
| `vade up` (also `--fp4-local`) | Start FP4-Local on one configured Spark; add `--sparks N` for 2–24 or ordered `--hosts` for a chain. Larger chains need a current staged build and their own live qualification. |
| `vade up --replicas N` | FP4-Local only, N = 2–16: one independent one-Spark server on each of the first N configured machines (or the N `--hosts`), each behind its own tunnel on the N local ports after `local_port`, with a local gateway on `local_port`. Needs no `fabric.*` entries; not combinable with `--sparks` or `--fp8-legacy`. `vade down` stops the gateway, the tunnels and every replica. See [Replicas](../INSTALL.md#replicas). |
| `vade up --fp8-legacy` | Legacy FP8: start on the first two configured machines. |
| `vade up --fp8-legacy --sparks 4` | Legacy FP8: construct a four-machine path in config order; separate runtime qualification applies. |
| `vade up --hosts machine-a,machine-b` | Select connected machines in the given order. |
| `vade status` | Machines, model, source build, MTP, context, output limit, concurrent chats the engine admitted, memory/cache bytes per rank, ranks, tunnel, health, and speed when health reports it. |
| `vade down` | Stop only recorded ranks and the owned SSH control connection. |
| `vade logs` | Recent redacted engine logs, including saved logs after shutdown. |
| `vade chat` | `vade-harness agent` with the owned endpoint, current folder, and recorded context. For local launches, contexts below 8192 tokens use direct text chat, and Linux agent mode disables automatic workspace checks with a notice. The harness negotiates each reply's output reservation. |
| `vade doctor` | Machine checks plus the harness doctor and its two small cache test turns. |
| `vade install` | Check, build, stage, and run doctor. Without a running engine, verify doctor availability and explain the remaining engine check. |
| `vade update` | Update clean checkouts with `git pull --ff-only`, rebuild, restage, and offer a restart. A checkout with no remote stays local. |

Every lifecycle command accepts `--dry-run` in any position. It prints every remote command, including worker operations and the safe stdin fields, without running commands, writing files, opening connections, or printing secrets. Until a manifest exists, dry-run startup labels its revision/hash placeholders. Help works without config.

FP4-Local supports 1–24 ranks (one Spark per rank). Chains of 3–24 require a current staged build that advertises chain support: run `vade install` from current source first; older staged builds are refused. Every larger topology still needs its own live GPU exactness qualification. One FP4 rank needs no fabric; two ranks link to each other, and larger FP4 chains use an open path in the selected host order. The legacy FP8 profile separately supports 2, 4, or 6 machines (two in a ring, four or six on a path). Each `fabric.<peer>` value is that machine's own address on its link to the peer. Define both ends of each adjacent link. Each rank gets a separate listen/next address pair; data uses `tp_port + 1`. The serve port must differ from both transport ports.

For four Sparks, use [the re-cabling and preparation guide](../docs/four-sparks.md)
and [the separate world-four golden procedure](docs/test.md#world-four-goldens).
The default four-rank path follows three consecutive links even on a physical
four-cable ring; its closing cable is unused. Host checks do not qualify this
runtime path. Startup rejects selected ranks with different engine or PTX
hashes before launching any process.

All selected ranks must use identical expanded absolute PTX and model arguments.
The staged PTX path is `machine.stage_dir/builds/<generation>/flash.ptx`, so select
matching remote staging roots as well as matching `model_path` values. Identical
`~/` strings can expand differently under different remote homes. `vade up`
expands and compares both paths on every selected machine before launching any
rank and rejects a mismatch; matching artifact hashes are insufficient. Dry-run
previews do not perform this remote check.

For the legacy FP8 profile, defaults are strict MTP 2 for a cold standalone chat, context 131072, output `min(65536, context)`. `--context N` accepts 4096..262144; `--output N` accepts 1024..262144 and may equal the configured context. Each reply is capped at the smaller of the service output limit and context minus the tokenized prompt length, including when the request omits its output limit. Use `--context 32768 --output 8192` to compare earlier timing receipts. Flash arithmetic always uses BF16 hi/lo tensor cores; legacy `--fast-prompts` is accepted for saved-command compatibility and does not change arithmetic. Manifests record whether the engine supports `FLASH_SERVE_OUTPUT_TOKENS`; older builds accept only their fixed `min(context,8192)` limit.

In the legacy FP8 profile, the model proposes up to two tokens ahead and checks each against ordinary decoding before keeping it. Overlapping chats stay plain through completion, even after peers finish. Cached continuations without retained draft history stay plain. `MTP=0 vade up --fp8-legacy` or `vade up --fp8-legacy --mtp 0` turns drafting off; `--mtp 0..3` overrides the `MTP` environment setting. The chosen depth is saved and passed explicitly to every rank, including zero, and a restart keeps the saved depth. FP4-Local omits the depth argument when neither the command nor controller MTP supplies one, preserving the engine default through restart. Inspect each response's `model_draft_used` to see whether drafting actually ran. Automatic memory sizing includes MTP weights and working memory on every rank.

Chat passes the recorded context to the harness. Each reply reserves the smaller
of the advertised output limit and remaining context minus 256 safety tokens,
using exact admission counts when available. Answers cut off at their output
limit continue automatically within the original request's turn and active-work
limits, up to 32 consecutive continuations. A manual `/continue` starts a fresh
work budget. See the [harness generation options](../harness/README.md) for direct
launches with `--max-output-tokens` or `--no-auto-continue`.

The integrated output extension from `824ab96f` negotiates each turn's reservation.
Its
[dated compatibility note](../harness/docs/engine-contract.md#output-extension-provenance-824ab96f)
preserves the source comparison; live qualification of these limits is pending.

For the legacy FP8 profile, the engine sizes its session pool from context and the lowest safe capacity across ranks after shared workspaces and safety funding. Startup must fund at least one full-context session and exits with a plain memory message if that reservation cannot fit. `--chats 1..32` funds that many concurrent full-context requests, caps live slots and persists through status and restart; without it, the pool uses all safe negotiated capacity with elastic request slots. Default retained prefix-cache capacity is the larger of half the pool and one full-context request bound. Idle cache yields its capacity to live work. Explicit `FLASH_SERVE_KV_BYTES` overrides retention exactly, including `0` to disable retention; `SEQPAR_MEMORY_BYTES` provides an explicit memory ceiling. Startup forwards those overrides when present, otherwise clears inherited remote values so the engine chooses its defaults. It also sets profiling off, creates a fresh shared run ID and control secret, and clears the launcher's experimental environment overrides.

`--memory-bytes N` sets a per-rank memory ceiling and overrides
`SEQPAR_MEMORY_BYTES`. In legacy FP8, `--cache-bytes N` sets retained cache
capacity through `FLASH_SERVE_KV_BYTES`, including `0` to disable retention.
Local FP4-Local supports only `--cache-bytes 0`, translated to its own prefix
cache switch; a positive capacity is refused. Both flags are optional.
FP4-Local retained prefix caching remains off by engine default; sending
`cache:true` does not enable it.
The harness still sends negotiated cache hints. For legacy FP8, unset memory
and retention follow automatic sizing. The engine still checks that the model,
shared workspaces and full-context live reservations fit before becoming ready.

Ownership is kept in private local and per-machine records. All artifact hashes are checked before launch, and binary/PTX hashes are checked again by the helper. Startup checks unlimited soft and hard memlock in the actual SSH helper session before launching RDMA ranks. Use an administrator-prepared SSH alias; Vade does not change system limits. Readiness, status, chat and doctor require rank zero's recorded process to own the loopback HTTP listener as well as ready model health. The controller hashes the staged copies and verifies the generation harness before every chat or doctor invocation. Failed command logs remain in a private temporary directory for inspection. Timed-out command jobs stop their own isolated process group before reaping its leader. Run, up and down catch Ctrl-C and TERM; an interrupted up cleans up its recorded ranks and tunnel through commands that remain available after cancellation. Incomplete cleanup preserves ownership records for `vade down` to retry. Linux command children also request a parent-death signal. A hard CLI kill during a compiler tree still needs live validation. GPU use, existing records, identity changes, or a reboot cause a refusal with a next step. Stop checks every reachable rank before signalling and uses Linux process handles, not a guessed PID. An identity refusal blocks signals; a transport failure lets other verified owners stop while unresolved records remain. The controller tunnel uses a unique private OpenSSH control socket opened relative to its private state directory, so long absolute paths work. A dead socket is removed after checking that nothing listens; a listening socket that fails SSH verification is left in place. Lost launch replies can be recovered from the machine's saved run file. Uncertain startup files require operator inspection.

Machine names, fabric addresses, and personal paths belong in local config, outside the repository. The retired `queue_dir` and `hold_path` keys are refused as unknown keys; remove them from an older config before using the current CLI. Old ownership records still load their recorded settings, and missing byte overrides use automatic sizing.

`vade chat` uses the harness's normal approval policy and removes inherited cloud-provider credentials from its invocation. `vade doctor` needs a running owned engine for inference checks. Speed is shown only when supplied by health telemetry; status does not generate a benchmark request. Update preserves a running engine until restart is accepted; noninteractive updates print the restart commands. A degraded old engine cannot prevent a completed update from offering that restart.

The native coding harness condenses older closed tool rounds within a task,
keeps exact goals and recent edit/check receipts, and favors targeted code reads.
After edits, use an existing component check and repair from its diagnostics.
Named offline host-check profiles require an explicit `--shell-toolchain` on
the native harness launch; this lifecycle CLI does not select that distribution.
See the [native tool contract](../harness/docs/agent-tools.md) and
[context controls](../harness/docs/context-and-projects.md). The 60-round and
20-minute active-work limits remain in force; host checks do not qualify GPUs.

## Recovery

Try `vade status`, `vade logs`, then `vade down` first. A controller crash or dead SSH master needs only `vade down`: it reports `controller tunnel already closed.` and continues stopping the owned ranks. The launch helper journals its exact future engine identity before exec. A rebooted machine or an incomplete launch request still needs the following manual steps, during that machine's window. The local FP4 launcher records its engine in `VADE_HOME/state/run.conf`; the Qwen3-8B catalog uses `VADE_HOME/run.conf` and `VADE_HOME/worker/`. The same boot/start/executable/argv checks apply.

1. Stop concurrent launch/install/cleanup commands. Keep a private backup of the controller's `state_dir/session.conf` and the affected machine's `state_dir/active.conf` and `<token>/` directory. Use the machine paths and full token from the saved session, even if config has since changed. The directory contains `journal.conf` (phase and process identity), `env.conf` (private environment), and `server.log`. Never print `env.conf` or put it in a report. Leave each permanent `.lock` file in place.
2. Inspect every affected machine before changing files. Compare `journal.conf`'s `boot_id` with `/proc/sys/kernel/random/boot_id`. A confirmed reboot means that machine's old engine cannot survive, but its saved PID may now belong to someone else. On the same boot, verify PID, `/proc/<pid>/stat` start time, `/proc/<pid>/exe`, and the entire NUL-separated `/proc/<pid>/cmdline` against the saved `pid`, `start_ticks`, `exe_hex`, and `cmdline_hex` (decode the last two from hexadecimal). If identity is missing or uncertain, leave the files in place and establish which process the original launch started. A process name, port, or GPU entry alone is insufficient. Never signal a reused PID or another job's process.
3. Once the affected engine is confirmed gone, check `nvidia-smi` for compute processes and zero utilization. If the GPU is busy, leave its files alone until its current owner finishes. Move only this token's saved directory out of `state_dir` into the private backup; remove `active.conf` only when its token exactly matches. Keep other token directories, model files, source/build directories, and `.lock` untouched. Retry `vade down` from the controller: absent ranks are skipped and the remaining ranks still require their exact identities before any stop.
4. Let successful `vade down` remove the controller's `session.conf`. If manual controller cleanup is necessary, first confirm every owned rank is gone and its machine files are cleared, and that the token-named tunnel has closed. Then move aside `session.conf`, remove only its dead `tunnel-<token-prefix>.sock`. Do not delete a listening socket that SSH cannot verify. Start again with `vade up` only after these checks.

## Local validation

Follow the house rules in [AGENTS.md](../AGENTS.md). From `engine/`, with the
pinned Rust toolchains installed and rustup's bin directory on PATH, set
RUSTFLAGS to `-D warnings`. For cached, offline builds set CARGO_NET_OFFLINE to
`true` and RUSTUP_AUTO_INSTALL to `0`. Build kernels first on every OS, including
macOS; stale PTX fails host checks. This block is identical to AGENTS.md:

```sh
cargo +1.99.0 kernels
cargo +1.99.0 check --offline --all-targets --manifest-path ../cli/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path ../harness/Cargo.toml
cargo +1.99.0 check --offline --all-targets --manifest-path tools/bench/Cargo.toml
cargo +1.99.0 test --offline --manifest-path ../cli/Cargo.toml --test ratchet -- --nocapture
python3 ../tools/check-doc-links.py
```

Preserve the existing dense source formatting by hand; do not run cargo fmt or
clippy's automatic fixes. CI does not gate on formatting or clippy. AGENTS.md
also lists the full workspace tests and Linux cross-target checks.

Saved long-context and recurrent-state evaluation uses
`vade test longctx prepare|run|compare` and `vade test gdn-state`.
The [evaluation guide](evals.md) documents the options, outputs and frozen
formats. These commands work on saved local inputs without lifecycle config.

Tests use fake SSH and local temporary files. Live installation, CUDA, pidfd
signalling, RDMA topology, health, tunnel recovery, harness cache checks and
restart need an authorized Spark session. Host tests make no live performance
or GPU exactness claim.

Use `--api-key-file PATH` (for example, `vade chat --api-key-file PATH`) or
`VADE_API_KEY_FILE` for authenticated chat and doctor requests. The flag overrides
the environment; the file contains the server's trimmed, single-line key of at
least 16 bytes using visible ASCII `0x21-0x7E`. Prefer 32 random bytes encoded
as hexadecimal and an SSH tunnel or TLS termination for remote HTTP. The CLI
selects the path without opening the file, then validates it once at entry to
`run`, `chat`, and `doctor`, before any side effects. Open `/health` requests from
`up`, `status`, and the menu never read the key. Help, local file operations, and
dry-run commands work even if the file is missing or unreadable. The harness
sends Bearer credentials and shows HTTP 401 plainly. Local `vade run` also supplies
the file to `spark-serve`. This option does not transfer keys to remote ranks;
configure the server's key file where the server runs. Keys stay out of command
arguments, previews and saved CLI state.
