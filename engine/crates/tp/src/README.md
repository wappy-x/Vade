# engine/crates/tp/src

**The Rust source of the `spark-tp` crate.**

| File | What it does (from its own summary comment) |
| --- | --- |
| [`lib.rs`](lib.rs) | Head/output-row-sharded tensor parallelism for any number of DGX Sparks. |
| [`layer_driver.rs`](layer_driver.rs) | Checked deferred layer arguments and gathers; execution remains refused. See [the boundary](../LAYER-DRIVER.md). |
| [`session_state.rs`](session_state.rs) | Resident session byte owners, budgets, saved banks, and completion. See [session storage](../SESSION-STATE.md). |
| [`model.rs`](model.rs) | Row-sharded Qwen using the single-Spark backend and arithmetic kernels. |
| [`next.rs`](next.rs) | Host-tested mixed Next session skeleton with ordered rank gathers and per-session kernel state. |
| [`rdma.rs`](rdma.rs) | RC SEND/RECV and one-sided writes into CUDA's pinned exchange slots. |
| [`runtime.rs`](runtime.rs) | No summary comment yet; read the file. |
| [`service.rs`](service.rs) | Rank-zero session commands, shared by service and speculative callers. |
| [`hybrid_service.rs`](hybrid_service.rs) | Root-ordered typed sessions and worker replay; conservative worst-rank request/cache funding. |
| [`service_commands.rs`](service_commands.rs) | Host-side dense/hybrid command validation and gathered output verdicts. |
| [`transport.rs`](transport.rs) | No summary comment yet; read the file. |
| [`bin/`](bin/README.md) | Folder: see its README. |
| [`flash/`](flash/README.md) | Folder: see its README. |

**Why a folder:** Cargo looks for a crate's code in `src/`.

`next::NextModel` binds a rank-sharded mixed checkpoint. Its session skeleton
accepts a `RankKernels` adapter and either existing TP or seqpar transport.
Every token follows the same front/attention/MoE/head schedule; gathered FP32
partials fold by ascending rank, and MoE packets compare selected IDs/weights.
`SharedTransport` supports controller-ordered interleaving of session state.
The host test uses real synthetic FP4/CSF and MXFP8 bytes over TCP at worlds
1/2/4 (including a four-rank path). `next_gpu` streams validated rank weights into one arena and uses the reviewed
FAST projection/router/routed/HC kernels. `next-tp` is the native rank service and
full-array gate capture (EP at W1/2/4; experimental EP2xTP2 at W4). Its case file
is `NAME PROMPT_CSV EXPECTED_CSV`. It measures warm gather latency and records
routes, full stage arrays and vocabulary-sharded logits. `VADE_FLASHNEXT_FP4_EP2=1` selects whole-expert partitioning at W2/W4,
with one rank-ordered MoE reduction per layer; `next-http` shares JSON/SSE
serving at both worlds. The MXFP8 loader excludes F8 PLE scales using the
checkpoint storage, residency and logical-weight contract. GPU qualification
remains setup-specific.

**How it fits:** see the [`spark-tp` README](../README.md) for what the crate does and how it connects to the rest.

**Back:** [Up one folder](../README.md) · **Docs:** [index](../../../../docs/README.md)
# Plain FP32 operand

[`ordered_projection.rs`](ordered_projection.rs) binds retained plain FP32 file rows to the generic native backend.
The [contract](../ORDERED-PROJECTION.md) keeps runtime profiles and complete layer execution refused.
