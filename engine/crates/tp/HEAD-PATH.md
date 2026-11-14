# Checkpoint output head

`head_path::Head::new` reads the admitted text description and tensor roles.
It checks the final norm, head, and embedding shapes and storage.
The configuration supplies the hidden width, vocabulary, stream count, and
norm epsilon. The head supports stored MXFP8 or plain BF16 weights.
It does not select an execution path from the checkpoint name.

`Head::reference` accepts one final position as `[streams, hidden]`.
`Head::reference_hidden` accepts its completed `[hidden]` mean row.
Both use finite BF16 values in FP32 lanes. The stream mean uses ascending
stream order and rounds once to BF16. The final norm uses the existing fixed
feature tree. It retains the stored norm parameter dtype.

The MXFP8 head rounds source activations to BF16. It applies dynamic
block-32 MXFP8 activation quantization. It uses the existing fixed projection
kernel and its RN-even BF16 epilogue. The plain BF16 head widens weights
without loss. It uses ascending feature products and FP32 adds, then rounds
once to BF16. The scalar reference uses the same arithmetic for each path.

`Head::upload` creates a `ResidentHead`. Its `step` method accepts the final
residual. Its `logits` method accepts a completed mean row. Both return a
`HeadPass`. Each call completes before return. The pass owns these buffers:

| Field | Layout |
| --- | --- |
| `mean` | `[hidden]`, BF16 values in FP32 lanes |
| `normalized` | `[hidden]`, stored norm output precision in FP32 lanes |
| `accumulator` | `[vocab]`, FP32, ascending token ID |
| `logits` | `[vocab]`, BF16 values in FP32 lanes |
| `next_embedding` | `[streams, hidden]`, BF16 values in FP32 lanes |
| `token` | Host `u32`, greedy ID with lowest-ID ties |

`HeadReference` has the same fields. Its buffers are `Vec<f32>`.
The device uses the served argmax kernels. A nonfinite logit fails selection.
The next embedding reads the chosen original BF16 row. It uses the stored
row loader and batch embedding kernel proved by the embedding path.
The caller commits decoder state after head success.

The `checkpoint-prefix-check head-path` kit accepts a saved final residual.
It compares mean, norm, accumulators, logits, token, and next embedding bytes.
The suite starts fresh device processes and compares their saved outputs.
The station compares the suites from the two Sparks of the pair.
A layer-zero output is a stand-in for the final residual. This isolated check
does not prove complete decoder token IDs or distributed transport.

No new kernel or unsafe block is added. The existing stream mean wrapper
moves to `Backend::text_mean`. Its safety contract and launch stay intact.
The old `ResidentTextEdges::mean` API calls this shared wrapper.
The carried causal wrapper initializes output before the device forms a
mutable slice. This is the correction from `2ada7f62d`.
The causal allowlist entries record the lab ruling from the expert report.
An independent safety review and the device gate remain merge requirements.
