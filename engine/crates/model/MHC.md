# Fixed-order Sinkhorn stream blocks

The [host reference](src/mhc.rs) and [device primitives](../gpu/kernels/src/mhc.rs)
implement logical FP32 stream operations. The
[byte kit](../../tools/bench/src/kernel_mhc.rs) checks the device contract.
These files do not enable model execution.

## Source and config

The source is Transformers
[`modeling_glm5_next.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py).
The saved copy has SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
It came from an unpinned main branch. The line references below refer to that
saved copy. They do not establish a pinned upstream revision.

The [saved config](tests/fixtures/glm/config.json) has SHA256
`dbf6aaf0d3651a5a297e6672bed57979c20528e2aa7475b80df5ba63d5402ba8`.
Its text fields are `mhc=true`, `hc_mult=4`, `hidden_size=4096`,
`hc_sinkhorn_iters=20`, `hc_eps=1e-6`, and `rms_norm_eps=1e-5`.
The source config lines are 2992-2997, 3159, and 3226.
The host test `saved_description_binds_operation_data_and_keeps_admission_closed`
checks these fields against the typed loader description.

`Config::from_description` binds the existing `SinkhornDescription` block kind.
Its dimensions and both epsilon fields come from operation data. It reads no
model name. `Config::from_json` also accepts direct operation fields with no
model name. The deferred plan still refuses executable admission.
The iteration limit follows the loader's `MAX_WIDTH` work limit in
[`shapes.rs`](src/shapes.rs). A zero iteration count is refused.
The source always runs the first column pass, even when its count is zero.
Vade requires a positive count so that the field names the number of column
passes. It does not silently treat zero as one. This is an admission limit;
it does not change the math for an accepted config.

## Math table

Let `N=hc_mult`, `D=hidden_size`, `W=N*D`, and `P=N*(N+2)`.
A token's raw residual is `x[N,D]`. The coefficient matrix uses source rows
and destination columns. The following order applies at both block sites.

| Step | Operation | Saved source lines |
| --- | --- | --- |
| Expand | Copy each embedding into every stream. | 1494 |
| Flatten | View the raw residual as `x[W]`. Cast it to FP32. | 299-301 |
| Unweighted RMS norm | `z=x/sqrt(sum(x*x)/W+rms_norm_eps)`. Sum across all streams and features. There is no norm weight. | 209-215, 276, 299-301 |
| Projection | Apply `fn[P,W]` to `z`. Split into pre `[N]`, post `[N]`, and comb `[N,N]`. Each group uses its own scalar scale and base entries. | 279-284, 303-310 |
| Pre | `sigmoid(projected_pre*scale[0]+base_pre)+hc_eps`. | 313 |
| Post | `2*sigmoid(projected_post*scale[1]+base_post)`. No epsilon is added here. | 314 |
| Initial comb | Apply stable row softmax to `projected_comb*scale[2]+base_comb`. Add `hc_eps` to each probability after softmax. | 315 |
| First column pass | Divide each entry by its column sum plus `hc_eps`. This is the first iteration. | 321 |
| Remaining passes | Run exactly `hc_sinkhorn_iters-1` passes. Each pass normalizes rows, then columns. Add `hc_eps` to each divisor. No convergence exit applies. | 322-324 |
| Collapse | `u[d]=sum_src(pre[src]*x[src,d])`. Use raw residuals. Cast the collapsed result to the activation dtype. | 328-329 |
| Block | Apply the block input RMS norm, then attention or FFN. The block output is `b[D]`. | 1313-1335, 1340-1344 |
| Residual mix | `y[dst,d]=post[dst]*b[d]+sum_src(comb[src,dst]*x[src,d])`. Cast post and comb to the activation dtype before this operation. Thus the matrix application uses `comb^T`. | 1336-1338, 1345-1347 |
| Next block | Recompute coefficients from the new raw streams. Attention and FFN have separate learned weights. | 1297-1298, 1340-1347 |
| Mean head | Sum raw final streams and divide by `N`. Apply the final RMS norm after this mean. There are no learned head gates. | 332-336, 1510 |

Every reduction in Vade's FP32 contract visits its fixed tensor dimension in
ascending order. Multiplication and addition are separate FP32 operations.
That order is a Vade contract. The source does not specify a device-library
reduction tree. Fixed iterations do not establish a universal convergence
bound. The row and column error test names its fixture and bound.

## Primitive reuse

The device preparation reuses `qsa_common::root`, `qsa_common::dot`, and
`qsa_common::exponential` from the [existing FP32 contract](../gpu/kernels/src/qsa_common.rs).
The square root rounds to FP32. The dot visits columns in ascending order.
The exponential uses fixed polynomial operations on nonpositive arguments.
Softmax subtracts its row maximum. Sigmoid uses `e=exp(-abs(x))`, then
`1/(1+e)` for nonnegative inputs or `e/(1+e)` for negative inputs.
This stable form avoids overflow. Its rounding can differ from the source
host expression. The byte kit compares it with the independent host source
reference under the printed numerical bound.

The current HC kernels normalize streams separately with learned weights.
Their mix uses feature gates and a stream mean. Their residual add has no
comb matrix. These operations differ from the table. The current dense MMA
projections round inputs to BF16. They cannot supply this logical FP32
projection. Existing QSA softmax launches also have different extents and
no Sinkhorn divisor epsilon. No existing kernel is changed to approximate
these missing operations.

`mhc_prepare` assigns each token to one thread. It uses caller-owned normalized
and coefficient buffers. It computes one flattened norm, the projection,
gates, and all Sinkhorn passes. `mhc_collapse`, `mhc_combine`, and `mhc_head`
assign each output element to one thread. No floating-point atomic, shared
reduction, or row-count dispatch applies. Inputs and outputs must be disjoint.
Preparation is a simple correctness primitive. No speed claim is made.

Each caller must call [`Shape::validate`](../gpu/src/mhc_params.rs) before
allocation or launch. Kernel entries do not call it. The caller must also
check buffer extents, finite input values and finite intermediate results.
No production caller or tensor binding is added.
The source's declared `hc_attn_fn/base/scale` and `hc_ffn_fn/base/scale`
checkpoint names still need physical header and dtype checks.

## Host and device checks

The host implementation reuses the reviewed first-party reference from
commit `7c44dbef2e9e480601637c2ac8fbac9253dfac4a`. It adds the typed-description
adapter, mean head, an iteration bound, and tests. The existing low-rank HC
reference and kernels stay unchanged.

The byte kit uses synthetic weights. It checks normalized values, all
coefficients, collapsed inputs, mixed residuals, and mean heads. It checks
every cut between rows, reverse row order, repeated scratch reuse, extra
launch threads, input preservation, and buffer guards. Its shapes come from
[`mhc_reference.rs`](../../tools/bench/src/mhc_reference.rs). The saved-config
shape comes from the config fixture. Other shapes check that the math is not
fixed to that checkpoint. The independent source check uses the host
reference and the printed bound `0.000003*max(1,abs(expected))` per value.
This numerical check does not replace the byte comparisons.

From `engine/`, on an authorized CUDA host:

```sh
cargo +1.99.0 kernels
cargo +1.99.0 run --release --offline --locked \
  --manifest-path tools/bench/Cargo.toml --bin kernel-bench -- \
  --mhc-bytes target/nvptx64-nvidia-cuda/release/spark_layer_kernels.ptx
```

Record source revision, PTX hash, GPU, driver, command, exit status, and full
output. Compare all printed result hashes across fresh processes and Sparks.
The repeat rule is in [AGENTS.md](../../../AGENTS.md).

## Open gates

All reference inputs and outputs are logical FP32. The source's BF16
coefficient casts, collapse cast, block multiplication, and matrix application
still need a separate rounding contract and device proof. The primitive kit
does not establish PyTorch bits, token IDs, model support, graph replay, TP
ownership, or performance. It does not admit non-finite data to production.

Before production binding, validate physical tensors, dtype conversions,
stream expansion, final norm, capture replay, and storage lifetimes. Run the
full device token-ID and quality gates. Test supported Spark counts, batches,
cache states, prompt chunks, and drafting settings. Host checks and kernel
compilation are not device passes.
