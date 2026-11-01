# Two-Spark Flash server

Connect a Mac, Vade's native harness or an existing application to the
Qwen3.8-Flash-Next FP8 server started by [INSTALL.md](../../INSTALL.md).

**Start both ranks and pass INSTALL's smoke request first.** Rank zero accepts
HTTP on its loopback port `18080`; rank one has no HTTP listener. Rust
`vade up` owns both rank processes and the Mac tunnel while clients use them.

The model ID is `qwen3.8-flash-next-fp8`, the product context default is 131,072
tokens and the output cap defaults to 65,536 tokens, bounded by context.
Explicit output limits may equal the configured context. Each reply is capped
at the smaller of that output limit and context minus the tokenized prompt
length, including when the request omits `max_tokens` or asks for more than fits.
INSTALL explicitly leaves MTP off.
The automatic pool and retained cache fund a full configured-context session;
startup refuses with a plain memory message if the pool cannot fund one.
Caching pools also fund bounded publication metadata for each explicit chat,
so a single-chat pool can retain a near-full-context continuation. Explicit
`FLASH_SERVE_KV_BYTES` retention limits remain exact, including zero.
Explicit context settings override the default. The retained manual installation
receipt used 32,768 tokens. Native engine checks at `9bcb8170` passed startup
with the 131,072-token context default and a near-full cached continuation,
using an 8,192-token output cap. See the
[October 12 results](../../docs/history/benchmarks-2026-10.md#october-12-product-defaults).
Those checks do not qualify the current 65,536-token output default or a fresh
`vade install` / `vade up` / `vade down` installation.
The separate [one-Spark server guide](server.md) describes
`spark-serve`, Qwen3-8B and port `8080`.

Current `flash-serve` and `vade up` defaults use strict MTP depth 2 for a cold
standalone chat. Overlapping chats stay plain through completion, even after
peers finish. Cached continuations without retained draft history also stay plain.
MTP guesses a few answer tokens and asks the main model to check them. Strict
drafting preserves its token choices, including fixed-seed sampled requests.
Pass `--mtp 0` to `flash-serve` on both ranks or to `vade up` to keep plain
decoding; `vade up` also accepts explicit `MTP=0`. A request
with `"draft":"plain"` skips drafting for that answer. Automatic memory sizing
includes MTP weights and working memory on both ranks. The current default and
memory changes still need their own two-Spark startup and timing check; the
earlier installation canary does not qualify them.

## Test-only kernel A/B measurements

`VADE_AB_PTX=<X.ptx>` loads a second kernel arm alongside the usual positional
A PTX, sharing the model once. Requests accept `"arm":"A"` or `"arm":"X"`
(default A); any `arm` field is rejected when the flag is absent. Test mode
serializes requests and disables request cache reuse. See the
[one-start ABBA recipe](ab-serve-switch.md) for timing and proof instructions.

## Connect your Mac

**Use the owned tunnel opened by `vade up`.** A tunnel carries requests from
its configured local Mac port to rank zero's local HTTP listener. `localhost`
on the Mac otherwise refers to the Mac, which has no engine running.

Complete the [Rust lifecycle setup](../../INSTALL.md#5-start-the-two-spark-server)
with your private `VADE_CONFIG`, then inspect the owned session from the Mac:

```sh
vade status
vade logs
curl --fail --silent --show-error http://localhost:18080/health
curl --fail --silent --show-error http://localhost:18080/v1/models
```

Use the configured `local_port` if it differs from 18080. Health must say
`ready`, and the model list must contain `qwen3.8-flash-next-fp8`. A refused
connection means the tunnel or rank-zero server is unavailable; inspect
`vade status` and `vade logs` before starting a client.

The CLI verifies its recorded process owns the listener and tracks the tunnel
with a private SSH control socket. Use `vade down` for cleanup and the
[CLI recovery steps](../../cli/README.md#recovery) for uncertain ownership.
HTTP remains loopback-bound. The Rust launch/tunnel recipe and live
Mac-to-Spark harness session still need owner-scheduled Spark validation;
the dated installation canary used a manual rank launch.

## Use an application

**Use the forwarded port and the installed Flash model.** Run the application
on the Mac while the owned tunnel is running. On rank zero, use its configured
`serve_port` if that differs from the Mac's `local_port`.

| Setting | Value |
| --- | --- |
| HTTP origin | `http://localhost:18080` |
| OpenAI API base | `http://localhost:18080/v1` |
| Chat route | `http://localhost:18080/v1/chat/completions` |
| Model ID | `qwen3.8-flash-next-fp8` |

Send a first application request from the Mac:

```sh
curl --fail --silent --show-error http://localhost:18080/v1/chat/completions \
  -H 'Content-Type: application/json' -d @- <<'JSON'
{
  "model": "qwen3.8-flash-next-fp8",
  "messages": [{"role": "user", "content": "Say hello in one sentence."}],
  "max_tokens": 64,
  "temperature": 0, "top_p": 1, "top_k": 0,
  "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0,
  "cache": false, "draft": "plain", "enable_thinking": false
}
JSON
```

This example explicitly uses plain decoding. To use the server's default for
one chat, omit `"draft":"plain"` from the request. Check the completed answer,
usage and `vade` echo for effective sampling, thinking and draft settings.
`mtp_depth` is the configured ceiling; `model_draft_used` says whether model
drafting ran in this answer. Multiple active chats use plain decoding even
when the configured depth is 2. Omitted sampling fields inherit
server/checkpoint defaults. For streaming, add `"stream": true` to the request and `--no-buffer`
to curl. This is a connection check; timing and qualification follow
[Benchmarks](../../docs/benchmarks.md).

`GET /v1/capabilities` advertises model limits, API routes and features.
Its `output_limit` and `/health`'s `caps.output_limit` report the maximum output
for a nonempty prompt; a longer prompt leaves less room for the reply.
The generation deadline defaults to one hour (3,600 seconds), also the maximum.
`FLASH_SERVE_GENERATION_TIMEOUT_MS` accepts 1 to 3,600,000 in a directly launched
Flash server; `vade up` uses the default. Queue wait has a separate bound of the
same length; admission starts a fresh generation deadline. Expiry is fatal:
non-streamed requests return HTTP 504 with `generation_deadline` and
`x-should-retry: false`; streams end with an error event (`response.failed` for
Responses), preserving deltas already sent. Cancellation similarly returns 409
with `client_cancelled` and the retry header, or a terminal stream error.
The deadline and HTTP response byte limits can stop a reply below its token cap.
See [GPU wait limits](server.md#gpu-wait-limits) for `VADE_GPU_WAIT_TIMEOUT_SECS`.
`POST /v1/admission` checks whether a complete request fits the context; it
reserves no queue slot or GPU memory. Chat Completions and Responses share the
server, but their request formats differ. See the
[engine/harness contract](../../harness/docs/engine-contract.md) and
[HTTP route implementation](../crates/serve/src/http.rs) for the exact scope.

INSTALL's loopback server requires no API credential. If an application insists
on a key for that keyless setup, use a dummy value. If you separately configured
server authentication, use its matching credential; see
[API authentication](server.md#api-authentication). HTTP has no native TLS;
the SSH tunnel protects traffic between the Mac and Spark.

## Run the native coding agent

**Build and run the harness on the Mac, where its workspace tools execute.**
The model and weights stay on the Sparks. Native shell commands require macOS;
workspace and approval limits are in the
[native agent guide](../../harness/README.md#native-coding-agent).
First [verify the owned Mac tunnel](#connect-your-mac). With `VADE_CONFIG`
selected, change into your workspace and run:

```sh
vade chat
```

The CLI uses the staged harness, owned endpoint and recorded context. The harness
reserves the smaller of the advertised output limit and remaining context minus
256 safety tokens for each reply. The direct harness build below is useful for
harness development or explicit client configuration.

### Build the Mac harness

Use a Mac checkout of the same Vade revision as the installed Sparks. If you
need a checkout ([clone and build](../../INSTALL.md#3-clone-and-build)), run
the following on the Mac, from a parent directory of your choice.
Enter the commit printed by `git rev-parse HEAD` in the rank-zero checkout:

```sh
bash
git clone --branch main https://github.com/wappy-x/Vade.git vade
cd vade
read -r -p 'Vade commit running on the Sparks: ' VADE_SPARK_COMMIT
git checkout --detach "$VADE_SPARK_COMMIT"
```

For this client, build only the harness; the Spark engine, weights and GPU
kernel steps remain on the Sparks.

If `rustup` is missing, follow the installer in
[pinned Rust setup](../../INSTALL.md#2-install-the-pinned-rust-compilers).
The Mac harness needs only host Rust `1.99.0`. From the **Mac monorepo root**:

```sh
bash
. "$HOME/.cargo/env"
rustup toolchain install 1.99.0 --profile minimal
unset RUSTUP_TOOLCHAIN CARGO_TARGET_DIR RUSTC RUSTDOC RUSTC_WRAPPER \
  RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_TARGET
cargo +1.99.0 build --release --offline --manifest-path harness/Cargo.toml
harness/target/release/vade-harness --help
```

Cargo must finish successfully and help must print. The binary is
`harness/target/release/vade-harness`; these commands do not install it on PATH.

### Choose a workspace and run

Keep the owned engine and tunnel running. In that same Mac checkout terminal, enter an existing
directory containing the files you want the assistant to work on. A relative
path resolves from the monorepo root.

```sh
read -r -p 'Workspace directory on this Mac: ' VADE_WORKSPACE
harness/target/release/vade-harness agent \
  --endpoint http://localhost:18080 --model qwen3.8-flash-next-fp8 \
  --workspace "$VADE_WORKSPACE" --approval ask \
  --temperature 0 --thinking off --max-output-tokens 1024
```

Type a task and press Enter. Reads and approved edits/commands act in the chosen
Mac workspace; model-requested tool results go back to the Spark server.
With `--approval ask`, edits and commands outside the read-only allowlist require
approval. Allowlisted commands run without asking: `pwd`, `true`, `ls` (optional
`-1` and workspace-relative paths), `cat` (one or more workspace-relative paths),
`git status` (optional `--short`), `git diff` (optional `--stat`), and `git log`
(no options, `--oneline`, or `-n N` with N from 1 to 100). Bare names resolve to
fixed system executables; their matching absolute paths are also accepted.
Other arguments and explicit shell scripts require approval. See the
[exact command forms and path restrictions](../../harness/docs/agent-tools.md#commands-and-approvals).
Sessions create private journals under the workspace's `.vade/sessions`; keep
them out of Git.

This command explicitly sets temperature, thinking and the output cap. Other
sampling fields use upstream defaults; it is an interactive usage example,
not an all-fields-explicit timing or determinism recipe.

Answers cut off at the output limit continue automatically, up to 32 consecutive
continuations within the original request's turn and active-work limits. Add
`--no-auto-continue` to wait for `/continue`, which starts a fresh work budget.
Omit `--max-output-tokens` to use the adaptive reservation described above;
explicit reservations must fit the remaining context. A length-truncated tool
call is discarded without execution.

## Connect an existing agent

**[Verify the owned Mac tunnel](#connect-your-mac) before configuring a client.** Build the
harness with the [Mac build step](#build-the-mac-harness) if you want its offline
configuration generator. From the Mac monorepo root, for example:

```sh
harness/target/release/vade-harness setup pi \
  --endpoint http://localhost:18080 --model qwen3.8-flash-next-fp8 \
  --target direct
```

This prints a configuration; it sends no request, installs no client and starts
no service. Apply the printed instructions to your separately installed client.
The [client setup guide](../../harness/docs/passthrough-matrix.md) covers dedicated
homes, launch commands, pinned clients and compatibility limits.

If you select a gateway configuration instead, start the gateway in another
Mac checkout terminal and point the client at `http://localhost:8787/v1`:

```sh
harness/target/release/vade-harness gateway --upstream http://localhost:18080
```

The tunnel stays open for either route. The gateway relays APIs without
translation; clients requiring Responses need upstream Responses support.

## Finish the session

**Close clients, then stop the owned session with `vade down`.** Exit the
native UI with Ctrl-D at an empty prompt, and stop a foreground gateway with
Ctrl-C in its own terminal. From the Mac with the same `VADE_CONFIG`:

```sh
vade down
vade status
```

Confirm both ranks exited and both GPUs are empty in your existing Spark
sessions. Preserve unresolved records and follow
[CLI recovery](../../cli/README.md#recovery) if shutdown cannot verify ownership.
See [INSTALL.md](../../INSTALL.md#6-run-the-timing-benchmark) for the timing
recipe and its qualification limits.

**Where next:** [Harness tools](../../harness/docs/agent-tools.md) ·
[Client setup](../../harness/docs/setup-doctor.md) ·
[Benchmarks](../../docs/benchmarks.md)
