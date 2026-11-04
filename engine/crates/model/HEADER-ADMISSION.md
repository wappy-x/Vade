# Deferred text header admission

[`plan::admission`](src/plan/admission.rs) checks stored text weights against
the operation description and storage recipes. It does not select kernels.
It does not change the current Qwen loader or permit deferred execution.

`TextDescription::required_weights` derives the inventory from the logical
layer program. It adds the embedding, final norm, head, and stored draft
parameters. Appended draft data must have unambiguous latent and expert
geometry in the description. This check does not define a draft forward pass.

`admit_headers` checks a supplied header map. It requires every text weight
and scale component. It rejects extra text names. It checks dtypes, physical
axes, scalar ranks, component sizes, and byte ranges with checked arithmetic.
Vision headers contribute to the stored byte count. They do not supply text
upload views.

`open_headers` also reconciles the index with each shard header. It checks
shard paths, index membership, missing tensors, and the declared total size.
It retains each open file descriptor. Upload reads use these descriptors.
Checkpoint files must stay immutable while the admission object is live.
Header checks do not establish a content hash.

## Stored layouts

The recorded public layouts are in
[`public-header-samples.json`](tests/fixtures/glm/public-header-samples.json).
The source record is in the [fixture README](tests/fixtures/glm/README.md).
These layouts supply the physical axes. The config supplies the dimensions.

| Storage | Weight bytes | Scale bytes |
| --- | --- | --- |
| Plain | Stored BF16 or FP32, with the admitted dtype | None |
| MXFP8 | Stored E4M3 matrix | Stored E8M0 byte matrix |
| FP8 block weight-only | Stored E4M3 matrix | Stored FP32 block grid with its broadcast axes |
| Direct NVFP4 | Stored packed nibbles | Stored E4M3 scale matrix and FP32 global scale |
| CSF32 NVFP4 | Stored packed nibbles | Host expansion of the stored CSF32 components to E4M3 bytes |

FP32 coefficients retain their required dtype. Convolution headers retain
the channel axis. The logical matrix view removes only the singleton axis.
NVFP4 global and activation scales retain the stored scalar rank. Legacy
CSF remains refused by this admission API.

`WeightView::upload` returns host byte planes for a later device upload.
It checks payload sizes, scalar values, scale formats, and CSF32 prefixes
before it returns data. CSF32 expansion uses the existing host decoder.
It does not change scale bits or weight nibbles.

`UploadWeight::matrix_view` describes a bounded row-strided matrix slice.
It keeps the complete original scale plane, its block geometry, and the
global matrix origin. A slice does not rebase scale blocks. Packed slices
must contain complete bytes. Consumers must use the origin, row stride,
and scale geometry. A consumer must not treat a strided view as contiguous.

## Host checks

[`header_admission.rs`](tests/header_admission.rs) uses a complete bounded
inventory. It tests each required component, dtype, and physical rank. It
also tests malformed byte extents, scalar ranks, scale payloads, matrix
origins, index reconciliation, and retained file reads. The admission unit
test checks recorded public header groups against the operation contract.
Neither test reads model weight payloads.

Run a header-only inspection of an existing checkpoint from `engine/`:

```sh
cargo +1.99.0 run --offline --locked -p engine-model --example text-headers -- \
  /path/to/checkpoint
```

A header pass does not prove device capacity, kernel ABI support, session
state, transport, arithmetic, or token exactness. `ConfigPlan` still refuses
conversion of the deferred description to an executable `ModelPlan`.
