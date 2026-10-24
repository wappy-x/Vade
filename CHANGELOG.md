# Changelog

## v0.1.0 - 2026-10-24

The first public release brings together the Vade engine, CLI and harness.

### Engine and serving

- A std-only Rust inference engine for NVIDIA DGX Spark, with a text-only Qwen
  Flash-Next FP4-Local release path for one Spark by default and configured
  multi-Spark deployments through the exact layer pipeline.
- Drafting is on by default, and requests without sampling fields are decoded
  greedily in every world size while checkpoint sampling defaults are preserved.
- The two-Spark drafting fix: both pipeline stages now agree on the cohort
  drafting policy, guarded cohort drafting is admitted on the two-stage
  pipeline, and worker decode pools are reclaimed.
- Pipeline startup refuses ranks whose artifacts, numerical settings or prompt
  options differ, and validates pipeline switches before choosing a topology.
- The server reports the effective drafting mode and depth it actually ran, so
  `/health` and receipts tell the truth about drafting instead of echoing the
  request.
- Release builds load verified precompiled kernels, and checkpoint hashing uses
  a safe streaming SHA-256 that does not fill the file cache.
- Startup memory admission counts reclaimable page cache as free (Linux
  `MemAvailable`, capped by CUDA total) instead of CUDA free memory, which
  inside containers leaves out the host page cache; memory refusals name
  MemAvailable, the required bytes and the ceiling. Session pools hold a
  512 MiB transient reserve back, so 8 and 16 concurrent chats at the
  default context no longer stop generation a few megabytes short.
- The Flash-Next prefix cache ships default-off until it ties the uncached path.

### Model and installation

- A pinned, verified FP4-Local publisher revision with a committed SHA256
  manifest of the complete snapshot, and a one-line fetch helper that verifies
  the download against that revision.
- A replay-tested local install on one Spark: the launcher reads the Spark's
  identity and resources, finds existing models, generates local defaults, and
  runs install, up and chat in one terminal.
- A prebuilt Linux aarch64 launcher bundle, with a release workflow that builds
  it on version tags, checks every manifest hash and attaches the bundle and its
  `SHA256SUMS` to the tag release. The launcher falls back to a source build when
  a bundle is rejected.
- Entry guides rewritten for first contact: README, INSTALL and the first-run
  guide lead with the one-Spark launcher, distinguish the source route from the
  bundle route, and state the supported models explicitly.
- A two-Spark route in INSTALL alongside the one-Spark default.
- A supported-systems table in INSTALL: the engine runs on DGX Spark / GB10
  Linux `aarch64`; a Route B controller can be Linux `aarch64` or `x86_64` or
  macOS; a Windows computer is supported as the SSH client for Route A, not as
  a controller.
- Source installs also work from a downloaded source archive without Git
  metadata: the launcher records the archive's stamped commit as the build id,
  or builds as `unknown` when the archive carries none.

### CLI, chat and gateway

- Controller commands for install, launch, up, status, logs, verify and
  shutdown, with bounded JSON status, specific and actionable errors, and help
  that stays usable in pipes.
- Controller `vade up` and `vade launch` on the Spark start the same FP4-Local
  defaults (context 32768, output 8192); `vade launch` lowers a default or saved
  context through 16384, 8192 and 4096 only when the engine refuses that context
  for memory (saved once a start succeeds), and an
  explicit `--context` the engine refuses is an error. A
  controller bench measures concurrent chats instead of one queued chat;
  `--context 262144` keeps the single-chat long-context preset, and
  `vade status` prints the chats the engine admitted.
- `vade up --replicas N` (2..16, FP4-Local) starts an independent one-Spark
  server on each of N Sparks behind one local `vade-gateway`, which sends each
  request to the least-busy replica; `vade-harness vade-gateway` runs the same
  gateway by hand with repeated `--upstream`.
- A machine's `ssh` entry may be a `user@host` destination as well as an alias,
  host name or IPv4 address.
- `vade verify` for repeated greedy token exactness checks against a running
  server, with owned endpoint discovery.
- Chat history is kept under the Vade home (`~/.vade/projects`), never in the
  folder you chat from; `VADE_CHAT_IN_PROJECT=1` keeps it in a git-ignored
  `./.vade` instead.
- Chat and coding-agent workflows with steering of active chats, retained
  interrupted replies, session management and an OpenAI-compatible HTTP gateway
  with credential handling and recorded replay.
- Both gateways refuse browser-driven requests by default: a Host other than
  loopback with the listener port is refused, any request carrying an `Origin`
  header is refused, and a browser client can be admitted deliberately with exact
  allow-lists and an optional bearer token.

### Benchmarks

- `vade bench` runs with zero flags: it finds the recorded endpoint, reads the
  model from `/health`, uses a bundled offline prompt and measures the server's
  own drafting default.
- Drafting-aware receipts that separate the requested drafting mode from the
  server-reported mode and depth, and withhold rates when the two disagree.
- A benchmark page with methods, results and limitations, linking every quoted
  source folder to its published, sanitized receipt.
- A results chart in the README and on the benchmark page, drawn by a
  standard-library script from the committed receipts so that every label is a
  value a receipt prints.

### Project and community

- Agent rules in AGENTS.md and the human contribution process in
  CONTRIBUTING.md, with a package map, architecture notes and decisions.
- Security policy, Code of Conduct, issue and pull request templates, NOTICE
  and the bundled third-party licence texts.
- A fix for a flaky stream-cancellation test that now synchronizes cancellation
  and verifies process reaping, and recovery of the shared process-test lock
  after poisoning.

See [benchmark methods and results](docs/benchmarks.md) for measurements,
recipes and their limitations. See [NOTICE](NOTICE) for bundled third-party
material and [LICENSE](LICENSE) for Vade's Apache-2.0 licence.
