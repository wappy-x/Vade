# Text request program

`text_program::TextEngine` binds the checked text description to the shared
request service. A factory supplies the execution owners. This module does
not supply a checkpoint factory. The public configuration remains deferred.

The engine validates the embedding, final norm, stored head and head scales
through `TextDescription::text_edges`. It requires the complete tokenizer
EOS set to match the model configuration. It retains those stop IDs for the
shared generator. It does not select execution from a model name.

Each token follows this order:

1. Expand the embedding into residual streams.
2. Run each decoder in description order.
3. Compute the parameter-free stream mean.
4. Apply the final norm.
5. Project through the stored head.
6. Validate complete finite logits, then commit state.

Any operator error calls `abort` and retires that session. An error in abort
also remains visible. Invalid token IDs and exhausted context fail before
execution. The factory must bind one immutable checkpoint and rank. It must
admit all allocations before work starts. The program must retain operands
through compute and transport completion. Its Drop must drain unfinished
work. A process abort cannot provide recoverable cleanup.

Rollback and snapshot forks are refused. The module does not enable model
drafting. It uses the existing service traits and generator. No active
`flash-serve` startup route calls it yet. The mocked tests check order,
validation and failure handling. They do not run decoder math.

## Resident edges

`text_checkpoint::CheckpointTextEdges` binds the file-backed header admission
to these edges. The admission retains its original operation description.
The binder uses that description and the same open tensor files.
It computes aligned arena spans from the original component dtypes and shapes.
It copies complete tensors and the original scale planes through the existing
bounded uploader. It does not create a complete host copy of the weights.
It checks reserved MXFP8 scale bytes before device allocation.

The file-backed host tests use synthetic shards. They compare every copied
byte, alignment padding, and stored norm dtype. They also replace paths after
admission and check failed reads. These tests do not read checkpoint weights.
`TextEngine::from_checkpoint` uses the admitted description and text edges.
Its factory must still supply complete decoder, state, and transport owners.
The checkpoint files must stay immutable. Retained files do not prove hashes.

The GPU `ResidentTextEdges` owner retains checked views into an immutable
weight arena. `embed` repeats the requested ID across the configured stream
count. The existing embedding kernel reads the original BF16 row.

`mean` uses the existing ascending-stream `mhc_head` reduction. It then
rounds to BF16 in FP32 lanes with the existing elementwise kernel. This is
candidate arithmetic. It is not an upstream reduction equality claim.

`norm` uses the existing fixed feature partials and butterfly reduction.
It rounds the normalized value to BF16 before weight multiplication. Norm
parameters widen losslessly from their original stored type. A stored BF16
weight gives a BF16 product. A stored FP32 weight gives an FP32 product.
The new elementwise FP32 multiply does not change existing operation arms.
This is an explicit candidate load policy. It does not claim that an
unrecorded upstream runtime retained the same loaded weight dtype.

`project_fp32` consumes a complete normalized row. It calls the existing
single-row MXFP8 projection with the original E4M3 weights and UE8M0 scales.
It does not quantize the input or cast the result. The original scale origin
stays at the start of each complete stored row. It does not create a BF16
copy of the head. The host greedy selector rejects nonfinite logits and
keeps the lowest ID on equal maxima.

The owner checks dimensions, aligned extents, disjoint stored spans and
device context. New output allocations do not alias inputs. Calls reject
graph capture and wait for completion before returning. The new unsafe
launch sites each have a local safety contract for review.

The synthetic `text-edge-check` executable checks embedding bytes, the mean,
candidate norm, full projection logits and greedy IDs. A separate projection
checks FP32 input residual bits. Its host oracle follows the proposed fixed
reductions. It does not use a checkpoint, a decoder, transport or recurrent
state. The station must retain failed outputs.

## Source and open gates

The saved Transformers model has SHA256
`9eb13f2de8e481b8edd4d7911d17a47649ce4ce5463f0b67f1e487772a71ca87`.
It matches Transformers revision `f90e93c4f83974f634860e246cdfd2c68666d605`.
Its `Glm5NextTextModel.forward` expands embeddings, visits decoder layers,
then calls the parameter-free head and final norm. Its causal LM forward
calls `lm_head`. These are operation-order sources, not device receipts.

The final norm header retains its original stored dtype. The source audits
do not qualify the loaded dtype, reduction or projection activation policy.
The candidate norm is not an upstream reference. A real factory still needs
qualified final norm and projection boundaries, the complete decoder driver,
physical state layout, and transport completion. It must pass checkpoint-backed
logit and token-ID checks before public admission. The synthetic edge kit
does not close G6.

## GLM output parser

The parser uses the grammar in the pinned
[template](templates/glm.jinja). Template selection occurs while loading
artifacts. The tokenizer reports whether the rendered prompt ends in an
open thought block. This determines the initial parser channel, including
requests whose generic `enable_thinking` flag differs from template behavior.

Outside reasoning, a complete tool call has a declared name and paired
argument key/value tags. Schema string fields become JSON strings. Other
declared field types require valid JSON. Duplicate keys, unknown tool names,
malformed calls and truncated calls stay visible text. Malformed complete
calls disable further tool recognition. No tool call event is emitted before
the complete object passes these checks. Byte and call limits still apply.
The parser never invokes a tool.

The source-backed tests cover byte partitions, Unicode, reasoning, tools,
stops and saved tokenizer artifacts. The [published vLLM fixture comparison](tests/fixtures/glm-upstream/README.md)
adds independent expected prompt bytes for the typed renderer's text cases.
It excludes inputs outside that API. It does not execute an upstream runtime.
The saved artifacts still lack an upstream tokenizer run with checkpoint IDs.
That external parity gate remains open. This work does not close G7 in full.
