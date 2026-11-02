# Documentation

This page is the one map of Vade's public pages. It lists every page with a one-line summary. People and coding agents can read it from top to bottom.

Run these three commands on a DGX Spark to install Vade and start the engine:

```sh
curl -fsSL https://vade.lol/install.sh | sh
vade pull nvidia/Qwen3.8-Flash-Next-FP4-Local@b9ce18a6887e18c089fdea8509a33e63429a3300
vade up
```

- **One Spark.** The three commands are all you need. [INSTALL.md](INSTALL.md) has the full steps, the source build and every setting. You need no cable between Sparks and no network setup.
- **Two or more Sparks.** Do the INSTALL steps 1 to 3 on every Spark. Cable the Sparks as [network setup](network-setup.md) shows. Then drive them from a controller, as [Two and more Sparks](INSTALL.md#two-and-more-sparks) shows.
- **The FP8 checkpoint.** FP8 needs 2, 4 or 6 Sparks. Follow the [FP8 install](install-fp8.md).

A controller is a computer that drives one or more Sparks over SSH with the `vade` CLI and a config file. It is usually a second Linux or macOS computer. A Spark can also be its own controller.

## Start here

- **[README](../README.md)**: what Vade is, the quick start and the benchmark table.
- **[Why Vade](why-vade.md)**: why Vade exists and what "exact" means.
- **[Agent quickstart](agents/quickstart.md)**: the setup steps written for coding agents.
- **[Glossary](GLOSSARY.md)**: every Vade term in plain words.

## Install

### Main routes

- **[Install](INSTALL.md)**: every install route and setting, and recovery.
- **[Download models](pull-models.md)**: `vade pull` and `vade models`.
- **[Context size](INSTALL.md#context-size-on-one-spark)**: the default context window, and what Vade does when memory is short.
- **[First chat on one Spark](../harness/docs/first-run.md)**: what `vade launch` does and the settings it saves.
- **[One-Spark quickstart](quickstart-one-spark.md)**: FP4-Local on one Spark from a controller.
- **[Launch FP4-Local](../cli/docs/fp4-local.md)**: FP4-Local from a controller, on one Spark or on a chain.
- **[Prebuilt installation](../cli/docs/prebuilt.md)**: the three prebuilt packages and how to check them.

### Linked Sparks

- **[Network setup](network-setup.md)**: parts, cables, addresses and link checks.
- **[Spark setup](spark-setup.md)**: the OS and network preparation for each linked Spark, with the scripts in [scripts/spark-setup](../scripts/spark-setup/).
- **[FP8 install](install-fp8.md)**: FP8 on linked Sparks.
- **[Fresh install](fresh-install.md)**: a longer checklist for FP8 on two Sparks.
- **[Four Sparks](four-sparks.md)**: how to recable two FP8 pairs into one group of four.
- **[Eight Sparks](eight-sparks.md)**: eight Sparks cabled as one ring.

### Servers and upkeep

- **[FP4-Local on Vade's own server](../engine/docs/next-http.md)**: the engine server started by hand.
- **[Two-Spark Flash server](../engine/docs/flash-server.md)**: how to connect a Mac or an app to the FP8 server.
- **[One-Spark server and API](../engine/docs/server.md)**: the older one-Spark server route.
- **[Multiple Sparks](../engine/docs/multi-spark.md)**: the older BF16 `tp-serve` path.
- **[Run, update, remove](../engine/docs/upkeep.md)**: how to keep Vade running, update it or remove it.
- **[Engine install](../engine/docs/install.md)**: a pointer from the engine docs to INSTALL.
- **[Ready-made engine files](../engine/contrib/README.md)**: files for running the engine on a Spark.
- **[DGX Spark platform](../engine/docs/platform.md)**: the machine that Vade is built for.
- **[Models](../engine/docs/models.md)**: the official models that have a serving path.

## Use

### Command line

- **[CLI reference](../cli/README.md)**: every `vade` command.
- **[How `vade` talks to you](../cli/docs/ux.md)**: the contract for everything `vade` prints.
- **[What `vade` looks like](../cli/docs/transcript.md)**: real output, with no Spark needed.
- **[`vade test`](../cli/docs/test.md)**: the build and exactness checks.
- **[Evaluation commands](../cli/evals.md)**: the long-context evaluation commands.

### API and clients

- **[OpenAI-compatible API](openai-api.md)**: endpoints and request fields.
- **[Sampling](sampling.md)**: the defaults, seeds and what is exact.
- **[Examples](../examples/README.md)**: short scripts against a running engine.
- **[Client setup](../harness/docs/passthrough-matrix.md)**: Claude Code, Codex, Hermes, OpenCode and Pi through the gateway.
- **[Client setup and doctor](../harness/docs/setup-doctor.md)**: two commands that connect another client to the engine.
- **[Standalone gateway](../harness/docs/invisible-gateway.md)**: `vade-gateway` between your client and the engine.
- **[Cache-aware gateway routing](../harness/docs/gateway-routing.md)**: how the gateway spreads requests over several endpoints.
- **[Engine input checkpoints](../harness/docs/engine-checkpoints.md)**: optional saved prompt states.

### Coding agent

- **[Harness](../harness/README.md)**: the coding agent and its tools.
- **[Harness docs](../harness/docs/README.md)**: the index of the harness guides.
- **[Shell safety](shell-safety.md)**: how the agent asks before it runs a command.
- **[Saved chat sessions](sessions.md)**: list, resume, export and delete your `vade chat` sessions.
- **[Reading files](reading-files.md)**: how `vade read` and the agent turn documents into Markdown.
- **[MCP tools in chat](mcp.md)**: how to add the MCP servers you already use to `vade chat`.
- **[Context and compaction](context-and-compaction.md)**: how a chat continues past the context window.
- **[Agent tools](../harness/docs/agent-tools.md)**: the agent's tools and their limits.
- **[Context and projects](../harness/docs/context-and-projects.md)**: project rules, context limits and branches.
- **[Project instructions](../harness/docs/project-instructions.md)**: the files that go into the system prompt.
- **[Project memory](../harness/docs/memory.md)**: short notes that the agent keeps for later chats.
- **[Slash commands](../harness/docs/slash-commands.md)**: the built-in commands and your own templates.
- **[Chat hooks](../harness/docs/hooks.md)**: shell commands that run around agent turns.
- **[Child agents](../harness/docs/child-agents.md)**: helpers that the agent can start for side tasks.
- **[Shell checker](../harness/docs/shell-checker.md)**: how the checker reads a `workspace_shell` command.
- **[Codex and Hermes modes](../harness/docs/bridge-contract.md)**: how the harness hands a turn to an external agent.
- **[Engine contract](../harness/docs/engine-contract.md)**: how the harness talks to the engine.
- **[Cache defaults](../harness/docs/cache-defaults.md)**: the harness cache defaults and their evidence.
- **[Harness examples](../harness/examples/README.md)**: small offline programs, one feature each.

## Evidence

### Results and benchmarks

- **[Results](results.md)**: every current benchmark cell, with its build, settings and receipt.
- **[Benchmarks](benchmarks.md)**: the current numbers, and how to measure your own Spark.
- **[Receipts](../bench/receipts/README.md)**: the public records of each run.
- **[Draft notes for the next release](release-notes/0.2.0.md)**: notes for v0.2.0, which is not released yet.
- **[Agent task suite](harness-suite.md)**: 22 tasks for the coding agent and their grades.
- **[Agent suite fixtures](../harness/tests/agent-suite/README.md)**: the task files of that suite.
- **[Engine validation](../engine/docs/validation.md)**: how the engine is checked, on any computer and on a Spark.
- **[Engine benchmark records](../engine/docs/benchmarks.md)**: older benchmark records with their full data.
- **[Quality cards](../harness/docs/quality-card.md)**: `vade-harness quality-card` and the engine card.
- **[Agent workload](../harness/docs/workload.md)**: the load test for a chat-completions endpoint.
- **[Harness overhead](../harness/docs/performance.md)**: what the harness costs and how the stable prefix helps.
- **[Gateway measurements](../harness/docs/gateway-measurements.md)**: older Codex runs with and without the gateway.
- **[Codex gateway card](../harness/docs/harness-card.md)**: an older measurement record.
- **[Historical scale measurements](../engine/crates/tp/MEASUREMENTS.md)**: how an earlier build scaled from one Spark to three.
- **[Owner attach benchmark](../cli/benches/README.md)**: an optional CLI benchmark.
- **[Measurement commands](measurement-commands.md)**: the first-party measurement and evaluation commands.
- **[Long-context evaluation](longctx-eval.md)**: frozen inputs and saved captures.
- **[Link checks](../engine/docs/link-checks.md)**: the check of outside links in the public pages.

### Credits and licenses

- **[Credits](CREDITS.md)**: everyone Vade builds on, on one page.
- **[Provenance](../engine/docs/provenance.md)**: the origin of Vade's code and reused data files.
- **[Reference data provenance](../engine/tools/truth/README.md)**: how the saved official reference data was made.
- **[Prefill oracle](../engine/tools/oracles/README.md)**: a host test that checks how prompt rows map onto the GPU grid.
- **[License texts](../engine/licenses/README.md)**: full license texts for third-party material.
- **[License](../LICENSE)**: the Apache-2.0 license.
- **[Notice](../NOTICE)**: the licenses of the bundled files.
- **[Documentation assets](assets/README.md)**: the README banner and the benchmark figure.
- **[Engine assets](../engine/docs/assets/README.md)**: the engine banner and logo drawings.
- **[Harness assets](../harness/docs/assets/README.md)**: the harness banner.

## Design

### Contribute

- **[Contributing](../.github/CONTRIBUTING.md)**: how to check a change and send a pull request.
- **[Agent rules](../AGENTS.md)**: the rules for a coding agent that changes Vade.
- **[Code of conduct](../.github/CODE_OF_CONDUCT.md)**: how we treat each other, and how to report a problem.
- **[Security](../.github/SECURITY.md)**: how to report a vulnerability privately.
- **[Local CI](ci.md)**: the CI commands to run on your own computer.
- **[Repository tools](../tools/README.md)**: the `vade-tools` crate.

### Project design

- **[Architecture](architecture.md)**: how a request flows, and the exactness rule.
- **[Design decisions](decisions/README.md)**: the public rules for v0.1.0.
- **[Install and flows](design/install-and-flows.md)**: the steps from install to a running engine.
- **[Adding a model](adding-a-model.md)**: what a new checkpoint needs.
- **[Model plan](model-plan.md)**: how Vade reads a checkpoint's layout without its weights.
- **[Splitter](splitter.md)**: how Vade plans the division of the model's tensors.
- **[Typed hybrid state receipts](typed-state-qualification.md)**: how tests compare a chat's saved state byte for byte.
- **[Prompt weight reuse](prefill-weight-reuse.md)**: how prompt rows reuse router and gate weights.
- **[Flash-Next prompt projections](flashnext-prefill-mma.md)**: the GPU kernels that read prompts.

### Engine pages

- **[Engine](../engine/README.md)**: an overview of the engine.
- **[Engine docs](../engine/docs/README.md)**: the index of the engine pages.
- **[Engine architecture](../engine/docs/architecture.md)**: how the engine reads model files and answers requests.
- **[Design principles](../engine/docs/concepts.md)**: the four ground rules of the engine.
- **[Module map](../engine/docs/module-map.md)**: which file to open for a given job.
- **[Unsafe code](../engine/docs/unsafe.md)**: where the engine uses `unsafe`, and why.
- **[Speculative decoding](../engine/docs/speculative-decoding.md)**: how drafting works.
- **[Flash-Next arithmetic](../engine/docs/flash-fp8-math.md)**: the FP8 arithmetic contract.
- **[Flash-Next rows](../engine/docs/flash-next-rows.md)**: how one code path serves a prompt or up to sixteen chats.
- **[Flash-Next execution options](../engine/docs/next-fast.md)**: the configuration that the engine reads at start.
- **[Flash-Next native MXFP8 decode](../engine/docs/flashnext-native.md)**: the native FP8 projection route.
- **[Dense projection schedule](../engine/docs/flashnext-dense-latency.md)**: how one GPU kernel step covers up to sixteen rows.
- **[HC up lookahead](../engine/docs/flashnext-hc-up.md)**: how the HC up kernel loads its inputs ahead of each multiply.
- **[Routed FP8 prompt tiles](../engine/docs/routed-prefill.md)**: the routed expert path for prompts.
- **[NVFP4 on GB10](../engine/docs/fp4-kernels.md)**: how the FP4 kernels build.
- **[Exact prompt recurrence](../engine/docs/gdn-scan.md)**: how GDN prompt rows combine.
- **[Typed hybrid serving memory](../engine/docs/hybrid-memory.md)**: how admission funds memory.
- **[Typed decode graphs](../engine/docs/typed-decode-graphs.md)**: graph capture for the typed decode path.
- **[FP32 QSA indexer](../engine/docs/qsa-indexer.md)**: the QSA indexer contract.
- **[C8 cohort draft selection](../engine/docs/c8-cohort-experiment.md)**: the eight-chat draft controller.
- **[One-start kernel measurements](../engine/docs/ab-serve-switch.md)**: a test-only way to compare two kernels.
- **[Export](../engine/docs/export.md)**: the exact-file snapshot tool.
- **[Build and exactness gates](../engine/scripts/gates/README.md)**: the gates that a change must pass.
- **[Export allowlist](../engine/scripts/README.md)**: the source export allowlist and gate references.
- **[GB10 memory ledger](../engine/crates/model/MEMORY.md)**: how memory was accounted for on one build.
- **[Serving budgets](../engine/crates/serve/BUDGETS.md)**: how `spark-serve` sets the memory of a request.
- **[Shared multi-Spark hooks](../engine/crates/tp/MULTI-SPARK.md)**: notes for a planned merge of Spark paths.

### Engine packages

- **[Package map](../engine/crates/README.md)**: every engine package.
- **[engine-base](../engine/crates/base/README.md)**: portable metadata utilities.
- **[engine-fabric](../engine/crates/fabric/README.md)**: shared communication interfaces for linked Sparks.
- **[engine-hash](../engine/crates/hash/README.md)**: SHA-256 for checkpoint verification.
- **[engine-model](../engine/crates/model/README.md)**: reads and checks a model's files.
- **[engine-tokenizer](../engine/crates/tokenizer/README.md)**: turns text into token IDs and back.
- **[spark-engine](../engine/crates/engine/README.md)**: runs language models on the GPU.
- **[spark-gpu-probe](../engine/crates/gpu/README.md)**: the GPU layer for the GB10.
- **[spark-build-kernels](../engine/crates/gpu/build-kernels/README.md)**: the tool behind `cargo kernels`.
- **[spark-rust-kernels](../engine/crates/gpu/kernels/README.md)**: the GPU kernels, written in Rust.
- **[spark-seqpar](../engine/crates/seqpar/README.md)**: splits one long prompt across processes.
- **[spark-serve](../engine/crates/serve/README.md)**: Vade's chat server.
- **[spark-tp](../engine/crates/tp/README.md)**: one dense model across linked Sparks.
- **[spark-tp-kernels](../engine/crates/tp/kernels/README.md)**: an optional GPU kernel for split models.
- **[Engine tools](../engine/tools/README.md)**: programs that support the engine.
- **[vade-bench](../engine/tools/bench/README.md)**: the library behind `vade bench`.
- **[vade-export](../engine/tools/export/README.md)**: the engine's exporter.

### Source folders

Each of these pages lists the Rust files of one folder.

- **[spark-engine source](../engine/crates/engine/src/README.md)**: the source of `spark-engine`.
- **[spark-engine programs](../engine/crates/engine/src/bin/README.md)**: its programs, one per file.
- **[flash-bench](../engine/crates/engine/src/bin/flash-bench/README.md)**: the modules of `flash-bench`.
- **[spark-engine checks](../engine/crates/engine/src/checks/README.md)**: the modules of `checks`.
- **[spark-engine flash](../engine/crates/engine/src/flash/README.md)**: the modules of `flash`.
- **[spark-engine session](../engine/crates/engine/src/flash/session/README.md)**: the modules of `session`.
- **[spark-engine gemma4](../engine/crates/engine/src/gemma4/README.md)**: the modules of `gemma4`.
- **[spark-gpu-probe source](../engine/crates/gpu/src/README.md)**: the source of `spark-gpu-probe`.
- **[spark-gpu-probe programs](../engine/crates/gpu/src/bin/README.md)**: its programs, one per file.
- **[spark-build-kernels source](../engine/crates/gpu/build-kernels/src/README.md)**: the source of `spark-build-kernels`.
- **[spark-rust-kernels source](../engine/crates/gpu/kernels/src/README.md)**: the source of `spark-rust-kernels`.
- **[engine-model source](../engine/crates/model/src/README.md)**: the source of `engine-model`.
- **[engine-model programs](../engine/crates/model/src/bin/README.md)**: its programs, one per file.
- **[engine-model flash](../engine/crates/model/src/flash/README.md)**: the modules of `flash`.
- **[engine-model plan](../engine/crates/model/src/plan/README.md)**: the modules of `plan`.
- **[engine-model split](../engine/crates/model/src/split/README.md)**: the modules of `split`.
- **[engine-model tests](../engine/crates/model/tests/README.md)**: the tests of `engine-model`.
- **[Model metadata examples](../engine/crates/model/examples/README.md)**: how to inspect a checkpoint with no weights.
- **[spark-seqpar source](../engine/crates/seqpar/src/README.md)**: the source of `spark-seqpar`.
- **[spark-seqpar programs](../engine/crates/seqpar/src/bin/README.md)**: its programs, one per file.
- **[spark-serve source](../engine/crates/serve/src/README.md)**: the source of `spark-serve`.
- **[spark-serve programs](../engine/crates/serve/src/bin/README.md)**: its programs, one per file.
- **[spark-serve modules](../engine/crates/serve/src/bin/spark-serve/README.md)**: the modules of the `spark-serve` program.
- **[spark-serve chat output](../engine/crates/serve/src/chat_output/README.md)**: the modules of `chat_output`.
- **[spark-serve tests](../engine/crates/serve/tests/README.md)**: the tests of `spark-serve`.
- **[engine-tokenizer source](../engine/crates/tokenizer/src/README.md)**: the source of `engine-tokenizer`.
- **[Tokenizer chat](../engine/crates/tokenizer/src/chat/README.md)**: the modules of `chat`.
- **[Tokenizer Gemma 4](../engine/crates/tokenizer/src/chat/gemma4/README.md)**: the modules of `gemma4`.
- **[Tokenizer examples](../engine/crates/tokenizer/examples/README.md)**: the tokenizer examples.
- **[Tokenizer templates](../engine/crates/tokenizer/templates/README.md)**: the chat templates, per model family.
- **[Tokenizer tools](../engine/crates/tokenizer/tools/README.md)**: the tokenizer tools.
- **[spark-tp source](../engine/crates/tp/src/README.md)**: the source of `spark-tp`.
- **[spark-tp programs](../engine/crates/tp/src/bin/README.md)**: its programs, one per file.
- **[tp-launch](../engine/crates/tp/src/bin/tp-launch/README.md)**: the modules of `tp-launch`.
- **[spark-tp flash](../engine/crates/tp/src/flash/README.md)**: the modules of `flash`.
- **[spark-tp-kernels source](../engine/crates/tp/kernels/src/README.md)**: the source of `spark-tp-kernels`.
- **[vade-bench source](../engine/tools/bench/src/README.md)**: the source of `vade-bench`.
- **[vade-export source](../engine/tools/export/src/README.md)**: the source of `vade-export`.
- **[Harness source](../harness/src/README.md)**: the source of `vade-harness`.
- **[Harness app server](../harness/src/app_server/README.md)**: the modules of `app_server`.
- **[Harness child agents](../harness/src/child_agents/README.md)**: the modules of `child_agents`.
- **[Harness terminal program](../harness/src/cli/README.md)**: the interactive terminal program.
- **[Harness gateway](../harness/src/gateway/README.md)**: the modules of `gateway`.
- **[Harness MCP](../harness/src/mcp/README.md)**: the modules of `mcp`.
- **[Harness setup](../harness/src/setup/README.md)**: the modules of `setup`.
- **[Harness shell](../harness/src/shell/README.md)**: the modules of `shell`.
- **[Harness terminal](../harness/src/terminal/README.md)**: the modules of `terminal`.
- **[Harness tools](../harness/src/tools/README.md)**: the modules of `tools`.
- **[Harness transport](../harness/src/transport/README.md)**: the modules of `transport`.
- **[Harness tests](../harness/tests/README.md)**: integration tests against local mock servers.
- **[Harness goldens](../harness/tests/goldens/README.md)**: the expected terminal screens.
- **[Harness test helpers](../harness/tests/support/README.md)**: shared helpers for the integration tests.

## History

- **[Changelog](../CHANGELOG.md)**: every change that you notice, newest first.
- **[Older evidence](history/README.md)**: older evidence that we kept.
- **[Benchmarks of October 2026](history/benchmarks-2026-10.md)**: the older benchmark tables, as they were measured.
- **[Serve Qwen3.6-35B on one Spark](history/serve-35b-one-spark.md)**: a dated route that v0.1.0 does not support.
- **[SGLang corrections](history/sglang/README.md)**: corrections to the older SGLang records.
- **[Earlier optimization research](history/sglang/earlier-optimization-research.md)**: lessons from the September audits.
- **[SGLang exactness decision](history/sglang/exactness-decision-2026-10-01.md)**: a dated decision, not a Vade qualification.
- **[Engine history](../engine/docs/history/README.md)**: older engine records and research pages.
- **[Flash-Next FP8 text architecture](../engine/docs/history/flash-next-architecture.md)**: an older architecture page.
- **[FP32-cache measurements](../engine/docs/history/fp32-cache-measurements.md)**: older integration measurements.
- **[Integration record](../engine/docs/history/integration-2026-10-02.md)**: the integration record of 2026-10-02.
- **[Long-context results](../engine/docs/history/longctx-performance.md)**: older long-context results.
- **[Harness reference material](../harness/docs/research/README.md)**: older reference material for the harness.
