# Module map

**Which file to open for a given job, one level below the
[package map](../crates/README.md).** This page is for people reading or
changing engine code. Each folder's README lists every file in it, using the
file's own summary comment. This page picks out the modules you need first
and shows how they connect.

## Follow one request

**The current product server is `flash-serve` with the FP4-Local
checkpoint.** A chat request passes through these modules in order:

| Step | Module | What it does there |
| --- | --- | --- |
| 1. Start | [`tp/src/bin/flash-serve.rs`](../crates/tp/src/bin/flash-serve.rs), [`tp/src/flash/settings.rs`](../crates/tp/src/flash/settings.rs) | Reads startup settings, loads the checkpoint, sizes memory and picks the one-Spark or multi-Spark service. |
| 2. Service | [`tp/src/flash/next_service.rs`](../crates/tp/src/flash/next_service.rs) with [`next_batched_service.rs`](../crates/tp/src/flash/next_batched_service.rs) (one Spark, and the layer pipeline), [`tp/src/flash/next_rank_service.rs`](../crates/tp/src/flash/next_rank_service.rs) (rank-parallel, two or four Sparks) | Connects the engine to the shared HTTP front end, chat renderer, sampling, stops and usage. |
| 3. HTTP and API | [`serve/src/http.rs`](../crates/serve/src/http.rs), [`serve/src/auth.rs`](../crates/serve/src/auth.rs), [`serve/src/protocol.rs`](../crates/serve/src/protocol.rs), [`serve/src/responses.rs`](../crates/serve/src/responses.rs), [`serve/src/anthropic.rs`](../crates/serve/src/anthropic.rs) | Parses the request and checks the API key. The Chat Completions, Responses and Anthropic Messages shapes all become one chat. |
| 4. Scheduling | [`serve/src/scheduler.rs`](../crates/serve/src/scheduler.rs), [`serve/src/budget.rs`](../crates/serve/src/budget.rs), [`serve/src/cache.rs`](../crates/serve/src/cache.rs) | One CUDA thread runs model steps between socket work. Requests are admitted against a token and byte budget, and cached prompt beginnings are reused. |
| 5. Tokens | [`serve/src/tokenizer.rs`](../crates/serve/src/tokenizer.rs) → [`engine-tokenizer`](../crates/tokenizer/README.md) | Applies the checkpoint's chat template and turns text into token IDs and back. |
| 6. Engine adapter | [`serve/src/flashnext.rs`](../crates/serve/src/flashnext.rs) | Connects the scheduler's session and engine interfaces to Flash-Next. |
| 7. Model steps | [`engine/src/flash/next_batched.rs`](../crates/engine/src/flash/next_batched.rs), [`next_rows.rs`](../crates/engine/src/flash/next_rows.rs), [`next_execution.rs`](../crates/engine/src/flash/next_execution.rs), [`next_pool.rs`](../crates/engine/src/flash/next_pool.rs) | Admits rows in a fixed router order and runs a layer pass for a prompt tile or for independent chat rows. Each padded decode width replays its own CUDA graph. |
| 8. Drafting | [`engine/src/flash/next_mtp.rs`](../crates/engine/src/flash/next_mtp.rs), [`next_lookup.rs`](../crates/engine/src/flash/next_lookup.rs), [`draft_stack.rs`](../crates/engine/src/flash/draft_stack.rs), [`engine/src/draft.rs`](../crates/engine/src/draft.rs) | Proposes tokens from the extra model head and from earlier text, and verifies them in one batched pass. A proposal never decides an output token. |
| 9. Launch | [`gpu/src/backend.rs`](../crates/gpu/src/backend.rs) (`launch_flash*`), [`gpu/src/driver.rs`](../crates/gpu/src/driver.rs) | Looks up the kernel in the loaded module and queues it on the ordered stream through the CUDA driver. |
| 10. Kernels | [`gpu/kernels/src/`](../crates/gpu/kernels/src/README.md) | Rust compiled to PTX: projections, FP4 experts, attention, GatedDeltaNet and the output head. |

With two or more Sparks, steps 7 to 10 run on every rank. Rank zero sends
commands, and the ranks exchange activations through
[`tp/src/transport.rs`](../crates/tp/src/transport.rs) and
[`tp/src/rdma.rs`](../crates/tp/src/rdma.rs). The layer pipeline is in
[`engine/src/flash/next_pp.rs`](../crates/engine/src/flash/next_pp.rs).

## Follow one GLM request

**Main also holds a second engine, for `glm5_next` text checkpoints on two
Sparks.** It is a diagnostic service. It is not qualified, and it starts only
with `--diagnostic-unqualified-text` and a loopback HTTP address. It uses the
same HTTP front end, scheduler and GPU backend as Flash-Next (steps 3, 4 and 9
above). It does not use `engine/src/flash/`. Most of its files do not have the
model's name in their path, so use this table to find them:

| Step | Module | What it does there |
| --- | --- | --- |
| 1. Start | [`tp/src/bin/flash-serve.rs`](../crates/tp/src/bin/flash-serve.rs) (`Cartridge::resident_text`), [`tp/src/bin/flash-serve/text.rs`](../crates/tp/src/bin/flash-serve/text.rs) | Finds a `glm5_next` config and starts the text service in place of Flash-Next. Reads the settings, connects the two ranks, admits memory and calls the shared scheduler. |
| 2. Checkpoint data | [`model/src/plan/glm.rs`](../crates/model/src/plan/glm.rs), [`glm_index.rs`](../crates/model/src/plan/glm_index.rs), [`description.rs`](../crates/model/src/plan/description.rs), [`admission.rs`](../crates/model/src/plan/admission.rs) | Turns the config into operation data, gives each tensor name a role, and admits the shard headers. |
| 3. Chat text | [`serve/src/tokenizer/glm.rs`](../crates/serve/src/tokenizer/glm.rs), [`serve/src/chat_output/glm.rs`](../crates/serve/src/chat_output/glm.rs) | Renders the prompt from the pinned checkpoint template, and parses the output grammar of that template. |
| 4. Engine adapter | [`tp/src/row_split.rs`](../crates/tp/src/row_split.rs), [`tp/src/bounded_decoder/factory.rs`](../crates/tp/src/bounded_decoder/factory.rs) | Connects the scheduler's engine and session interfaces to the decoder. The split is by complete output rows, and each row keeps the decoder's fixed reduction. |
| 5. Model steps | [`tp/src/bounded_decoder.rs`](../crates/tp/src/bounded_decoder.rs), [`bounded_decoder/rows.rs`](../crates/tp/src/bounded_decoder/rows.rs), [`bounded_decoder/device.rs`](../crates/tp/src/bounded_decoder/device.rs), [`tp/src/causal_layer.rs`](../crates/tp/src/causal_layer.rs), [`tp/src/expert_path.rs`](../crates/tp/src/expert_path.rs), [`tp/src/head_path.rs`](../crates/tp/src/head_path.rs) | Runs the layer recipe for one decode token, or for a group of prompt rows one layer at a time. `device.rs` is the GPU side. The other three files are the causal KDA and dense layer, the routed and shared experts, and the output head. |
| 6. Weights | [`tp/src/resident_weights.rs`](../crates/tp/src/resident_weights.rs) | Holds each rank's matrices on the device. Complete feature rows stay local, and finished output rows gather in rank order. |
| 7. Operations and launch | [`gpu/src/prompt_ops.rs`](../crates/gpu/src/prompt_ops.rs), [`causal_ops.rs`](../crates/gpu/src/causal_ops.rs), [`latent_ops.rs`](../crates/gpu/src/latent_ops.rs), [`expert_ops.rs`](../crates/gpu/src/expert_ops.rs), [`resident_launch.rs`](../crates/gpu/src/resident_launch.rs), [`resident_program.rs`](../crates/gpu/src/resident_program.rs) | The operation shapes and math that the host and the kernels share, the indirect launch format, and the checked ordered graph pieces. |
| 8. Exchange | [`gpu/src/tp_mailbox.rs`](../crates/gpu/src/tp_mailbox.rs), [`tp_mailbox_host.rs`](../crates/gpu/src/tp_mailbox_host.rs), [`tp/src/flash/next_tp_proxy.rs`](../crates/tp/src/flash/next_tp_proxy.rs) | The mailbox that kernels fill and drain inside the stream, and the wire thread that owns its host endpoint on each rank. |
| 9. mHC on CPU cores | [`tp/src/host_conductor.rs`](../crates/tp/src/host_conductor.rs), [`gpu/src/mhc_host.rs`](../crates/gpu/src/mhc_host.rs), [`mhc_host_service.rs`](../crates/gpu/src/mhc_host_service.rs) | Runs the decode mHC boundary on CPU cores, with the same FP32 operations in the same order as the kernels. |

The host references for its operations are in `engine-model`:
[`kda.rs`](../crates/model/src/kda.rs), [`mla.rs`](../crates/model/src/mla.rs),
[`dsa.rs`](../crates/model/src/dsa.rs), [`mhc.rs`](../crates/model/src/mhc.rs)
and [`csf.rs`](../crates/model/src/csf.rs). The device check program is
[`tp/src/bin/resident-decode-check.rs`](../crates/tp/src/bin/resident-decode-check.rs).

## Package by package

### `spark-engine` ([README](../crates/engine/README.md))

| Module | Owns |
| --- | --- |
| [`src/flash/`](../crates/engine/src/flash/README.md) | The Flash-Next engine: weights, sessions, layer blocks, attention (`qsa.rs`), GatedDeltaNet (`gdn*.rs`), experts (`moe.rs`), the output head (`head.rs`) and the `next_*` FP4-Local row path. |
| [`src/flash/session.rs`](../crates/engine/src/flash/session.rs) | Batch-one text sessions with resident or layer-streamed weights and caches. |
| [`src/hybrid.rs`](../crates/engine/src/hybrid.rs) and `src/hybrid/` | The one-Spark dense hybrid models. |
| [`src/qwen.rs`](../crates/engine/src/qwen.rs), [`src/layer.rs`](../crates/engine/src/layer.rs) | Dense and expert Qwen3 execution, and the layer arithmetic it shares with tensor-parallel ranks. |
| [`src/sample.rs`](../crates/engine/src/sample.rs) | Sampling keyed by seed and absolute position, so a replay uses no random state. |
| [`src/context.rs`](../crates/engine/src/context.rs) | Checked admission for each session's key/value storage. |

Many `flash/` files are also exposed at the crate root under a
`flash_next_*` or `flash_*` name, such as `flash_next_rows` and
`flash_draft_stack`. Those are the same files, included through `#[path]`.

### `spark-gpu-probe` ([README](../crates/gpu/README.md))

| Module | Owns |
| --- | --- |
| [`src/driver.rs`](../crates/gpu/src/driver.rs) | The hand-declared CUDA Driver API (not the CUDA Runtime). |
| [`src/backend.rs`](../crates/gpu/src/backend.rs) and `src/backend/` | Device memory, the ordered stream, uploads and downloads, graph capture and replay, and every launch. |
| [`src/batch.rs`](../crates/gpu/src/batch.rs), [`src/fp8.rs`](../crates/gpu/src/fp8.rs), [`src/fp4.rs`](../crates/gpu/src/fp4.rs), [`src/moe.rs`](../crates/gpu/src/moe.rs) | Typed launch wrappers for projections and experts by number format. |
| [`kernel_params.rs`](../crates/gpu/kernel_params.rs), [`src/next_params.rs`](../crates/gpu/src/next_params.rs), [`src/gdn_params.rs`](../crates/gpu/src/gdn_params.rs) | Parameter blocks shared by host code and kernels. Host and device read the same layout. |
| [`src/load.rs`](../crates/gpu/src/load.rs), [`src/teardown.rs`](../crates/gpu/src/teardown.rs) | Batched shard loading, and bounded cleanup when the driver stops answering. |
| [`kernels/`](../crates/gpu/kernels/README.md), [`build-kernels/`](../crates/gpu/build-kernels/README.md) | The NVPTX kernel crate, and the build step that compiles, copies and stamps its PTX. |

### `spark-serve` ([README](../crates/serve/README.md))

| Module | Owns |
| --- | --- |
| [`src/scheduler.rs`](../crates/serve/src/scheduler.rs) | One CUDA thread, bounded nonblocking sockets, priority between model steps. |
| [`src/budget.rs`](../crates/serve/src/budget.rs), [`src/cache.rs`](../crates/serve/src/cache.rs) | Admission by prompt plus output limit; prompt-prefix snapshots. |
| [`src/http.rs`](../crates/serve/src/http.rs), [`src/protocol.rs`](../crates/serve/src/protocol.rs), [`src/chat_output.rs`](../crates/serve/src/chat_output.rs) | HTTP/1.1, the request and response JSON, and incremental chat-markup decoding. |
| [`src/flashnext.rs`](../crates/serve/src/flashnext.rs), [`src/qwen.rs`](../crates/serve/src/qwen.rs), [`src/hybrid.rs`](../crates/serve/src/hybrid.rs), [`src/gemma4.rs`](../crates/serve/src/gemma4.rs) | One adapter per engine family. |
| [`src/bin/spark-serve/`](../crates/serve/src/bin/spark-serve/README.md) | The one-Spark server program and its detach logic. |

### Multi-Spark: `spark-tp`, `spark-seqpar`, `engine-fabric`

| Module | Owns |
| --- | --- |
| [`fabric/src/exchange.rs`](../crates/fabric/src/exchange.rs), [`fabric/src/profile.rs`](../crates/fabric/src/profile.rs) | The collective interface every transport implements, and bounded collective profiling. |
| [`fabric/src/verbs.rs`](../crates/fabric/src/verbs.rs), [`fabric/src/discovery.rs`](../crates/fabric/src/discovery.rs), [`fabric/src/layout.rs`](../crates/fabric/src/layout.rs) | The `libibverbs` interface, run-time link discovery, and ring or path link layout. |
| [`tp/src/transport.rs`](../crates/tp/src/transport.rs), [`tp/src/rdma.rs`](../crates/tp/src/rdma.rs) | Tensor-parallel exchange: RDMA SEND/RECV and one-sided writes into pinned exchange slots. |
| [`tp/src/flash/`](../crates/tp/src/flash/README.md), [`tp/src/next_gpu.rs`](../crates/tp/src/next_gpu.rs) | Flash rank services and the rank-local FP4-Local adapter. |
| [`seqpar/src/transport.rs`](../crates/seqpar/src/transport.rs) with [`transport/connect.rs`](../crates/seqpar/src/transport/connect.rs) and [`transport/gather.rs`](../crates/seqpar/src/transport/gather.rs), [`seqpar/src/rdma.rs`](../crates/seqpar/src/rdma.rs) | Ring all-gather and fixed-rank-order reduction, neighbour rendezvous, and the mailbox relay. |
| [`seqpar/src/control.rs`](../crates/seqpar/src/control.rs), [`seqpar/src/partition.rs`](../crates/seqpar/src/partition.rs) | Challenge-response admission for peers, and contiguous prompt partitions. |

### Checkpoints, tokens and shared helpers

| Module | Owns |
| --- | --- |
| [`model/src/safetensors.rs`](../crates/model/src/safetensors.rs), [`model/src/plan/`](../crates/model/src/plan/README.md), [`model/src/flash/`](../crates/model/src/flash/README.md) | Header-only checkpoint inspection, the typed model plan, and the Flash-Next checkpoint contract. |
| [`model/src/split/`](../crates/model/src/split/README.md), [`model/src/placement.rs`](../crates/model/src/placement.rs) | Which rank owns which rows and tensors. |
| [`tokenizer/src/lib.rs`](../crates/tokenizer/src/lib.rs), [`tokenizer/src/chat.rs`](../crates/tokenizer/src/chat.rs) | Byte-level BPE with exact limits, and the chat templates. |
| [`hash/src/lib.rs`](../crates/hash/src/lib.rs) | Streaming SHA-256 for checkpoint verification. |
| [`base/src/json.rs`](../crates/base/src/json.rs), [`base/src/admission.rs`](../crates/base/src/admission.rs) | The bounded JSON reader, and the one text every memory refusal uses. |

## Where the tests are

Unit tests sit next to the code, in `#[cfg(test)]` modules and `*_tests.rs`
files. Integration tests that use only a package's public interface are in
each package's `tests/` folder. Tests that need a GPU are marked
`#[ignore]` or compiled only on Linux, so a host run without a Spark doesn't
run them, and passing it isn't a device pass.
[Validation](validation.md) explains which checks need a Spark.

## Related pages

- [Unsafe code](unsafe.md): the modules above that use `unsafe`, and why.
- [Architecture](architecture.md): the same structure, explained from the
  outside.

**Back:** [Documentation index](README.md) · [Engine packages](../crates/README.md)
