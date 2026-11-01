# Design decisions

These are the public rules for v0.1.0, recorded on 2026-10-24. Implementation
details and validation commands live in [AGENTS.md](../../AGENTS.md).

- **Standard library only.** Rust uses zero third-party dependencies, including registry/git dependencies and paths outside this checkout. Keep the dependency ratchet at zero; see [the baseline](../../cli/ratchet-baseline.txt).
- **No new unsafe.** Do not introduce new `unsafe` sites. Existing FFI/kernel boundaries are not permission to add more. The one exception, from 2026-11-01, is in [the contributor rules](../../CONTRIBUTING.md).
- **Model-generic execution.** Derive behavior from validated config, tensor metadata and template semantics; no model-specific tuning or model-name dispatch. See [adding a checkpoint](../adding-a-model.md).
- **Greedy default and exactness.** With FP4-Local, unspecified sampling means greedy. With FP8, it takes the checkpoint's sampled settings; see [FP8 defaults](../install-fp8.md#defaults-and-drafting). Exactness means identical token IDs across runs and Sparks with the same setup, including alone/batched/cached/chunked and drafting ON/OFF. Request defaults are implemented in [protocol.rs](../../engine/crates/serve/src/protocol.rs#L202); the merge bar is in [AGENTS.md](../../AGENTS.md).
- **Release model and topology.** The release target is `nvidia/Qwen3.8-Flash-Next-FP4-Local`, one NVIDIA DGX Spark / GB10 by default, with drafting ON by default; 1..N Sparks is the project scope. Additional topologies need their own qualification. See [installation](../../INSTALL.md) and [FP4-Local](../../cli/docs/fp4-local.md).
- **Text-only scope.** Text input/output is supported; the vision tower is out of scope. Template handling is text-only in [chat.rs](../../engine/crates/tokenizer/src/chat.rs#L1).
- **Bundle or source build.** A supplied Linux ARM64 launcher bundle provides the engine, PTX and harness with manifest hashes. Otherwise the launcher builds them from a local source checkout with pinned toolchains. These are the two paths in [launch/install.rs](../../cli/src/launch/install.rs#L72). Builds of main come from dl.vade.lol. GitHub Releases holds only the old v0.1.0 bundle and its `SHA256SUMS` ([release page](https://github.com/wappy-x/Vade/releases/tag/v0.1.0)).

Historical performance claims remain attached to their
[receipts](../../bench/receipts/README.md).
