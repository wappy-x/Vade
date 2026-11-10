# Causal decoder reference

`decoder-reference` connects every logical decoder layer to the text head.
It reads a local checkpoint through the reviewed
[M2 reader](M2-REFERENCE.md). It runs on the CPU.
It does not supply the resident device decoder or a service Factory.
The [runtime type map](RUNTIME-TYPES.md) lists weight producers, activation casts, and ordered gathers.
The stack rejects unqualified runtime weight profiles before reads or execution.

Run from `engine/`:

```sh
cargo run --release --offline --locked -p engine-model --example decoder-reference -- \
  generate MODEL NEW_OUT IDS_CSV
cargo run --release --offline --locked -p engine-model --example decoder-reference -- \
  verify NEW_OUT
cargo run --release --offline --locked -p engine-model --example decoder-reference -- \
  compare EXPECTED ACTUAL
cargo run --release --offline --locked -p engine-model --example decoder-reference -- \
  self-test NEW_TEST_OUT
```

Use a prebuilt executable for a station run. The tool does not fetch weights.
The ID list is an argument. The token limit comes from `generate`.
It is a reference tool limit. It does not select model arithmetic.
No tokenizer or chat template runs here.
Supply the exact prompt IDs from the separately checked text stack.

## Scalar stack

The profile remains `logical-f32-v1`. Each stored embedding row is widened
to FP32. Each residual stream receives that row. The first layer receives
these streams. Each later layer receives the preceding layer's output.
Layers carry their own KDA state and history or latent and indexer rows.
The indexer reference recomputes complete pools from its retained rows.
It preserves the incomplete tail through those rows.

After the last layer, the tool takes the mean of the residual streams.
It adds streams in ascending stream order, then divides by the stream count.
It applies scalar RMS normalization with the decoded stored gain.
It projects the complete vocabulary with the decoded head.
A tied head uses the embedding table. Dots visit features in ascending order.
Finite maximum ties select the lowest token ID.

The text operations follow the final mean, norm, and head order in the
[layer source record](LAYER-PLAN.md#source-record). Stored parameter dtype
does not establish the effective dtype of a checkpoint runtime.
The reference inserts no BF16 activation casts or activation quantizers.
Its exponentials and square roots keep the existing scalar layer profile.
Cross-OS byte equality is unqualified. Keep a mismatch as a failed comparison.
Do not increase the error limits to accept it.

`Decoder::append` stages a complete chunk. It commits causal state only after
all layers, final normalization, head, and token selection succeed.
An observer can save each complete layer result before that commit.
An observer error also leaves the lane unchanged.
External observer files can remain after an error. The tool does not seal
an incomplete result directory. Use a new directory for a retry.
Saved boundaries own independent copies. A lane rejects another lane's save.

`WeightSource` supplies decoded logical weights on demand.
The constructor checks every logical shape without reading every payload.
The layer source loads selected expert matrices only.
The checkpoint adapter reads only the requested embedding row.
The head is decoded as a complete matrix within the reader's existing cap.
This tool has no memory-use or speed qualification for a full checkpoint.

## Receipts and limits

`PROFILE.txt` identifies the source stamp, config, index, prompt IDs, and stack
scope. `WEIGHTS.tsv` identifies each consumed byte range.
`TENSORS.tsv` identifies every saved intermediate, causal output, and text
output. The files use the existing typed safetensors receipt format.
`TOKENS.tsv` records the supplied ID and next selected ID at each context.
`SHA256SUMS` seals the complete output directory.

The comparator requires the same input and weight identity.
It requires exact tensor bytes and token selections.
The error limits stay those in [M2](M2-REFERENCE.md#error-and-byte-rules).
Signed zero and integer IDs remain part of the byte rule.
The self-test creates a synthetic mixed-layer checkpoint.
It checks reader, stack, head, receipt, and repeat integration.
It cannot establish a checkpoint result or a device pass.

The [stack tests](tests/decoder_reference.rs) compare the stack with direct
layer composition. They check chunk boundaries, independent lanes, saved
state, tied heads, finite ties, and failure rollback.
The example test exercises file reads, stored BF16 parameters, an MXFP8 head,
receipt verification, and rejection of changed inputs or token records.

## Work still open

M2 needs the complete resident layer with qualified checkpoint casts,
physical state, and ordered rank gathers. M3 needs that decoder's full-stack
logits against a stated reference profile. This scalar tool cannot qualify
a nonzero device output band.

The prior resident operands and retained text stack still need a complete
device operator loop and service Factory. They must keep the normal tensor
split and the source-backed activation boundaries.
The first-token client needs that service before it can prove GLM token IDs.
Fresh process repeats, cross-rank results, and all exactness cases remain
required before merge. This host reference supplies no device pass.
