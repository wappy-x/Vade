# Bounded causal decoder probe

`checkpoint-prefix-check tokens` runs the complete checkpoint recipe through
an executable text Factory. It does not change public cartridge admission.
It derives each layer kind, shape, parameter type, and storage recipe from
the admitted checkpoint.

The probe validates the prepared request against the checkpoint chat tokenizer.
It requires greedy decoding, neutral penalties, plain execution, and no cache.
The request must set `ignore_eos`. The probe does not stop at EOS.
The request selects 8, 32, or 64 output tokens. The version 2 receipt records
the output limit, all generated IDs, and their decoded text.
It also records full logits for every generated ID and hashes prompt logits.
The comparator recomputes greedy argmax from each saved full row.

```sh
checkpoint-prefix-check tokens suite MODEL INPUT_IDS REQUEST_JSON NEW_OUT
checkpoint-prefix-check tokens compare-suite LEFT RIGHT
```

The single-device probe can refuse checkpoints that exceed one device's
weight budget. The resident pair probe shares weights across two ranks:

```sh
resident-decode-check RANK 2 LISTEN NEXT pair MODEL INPUT_IDS REQUEST_JSON NEW_OUT REPEATS
```

Its first four arguments follow the existing rank handshake. Both ranks use
the same request and repeat count. Compare saved runs and check the requested
output count separately when comparing prefixes of different lengths.
Public service admission remains closed.

The suite starts fresh device processes. The comparison checks exact IDs,
text, prompt-logit hashes, generated full logits, and source metadata.
It does not replace the host tests or the complete device exactness gate.

The Factory owns a retained checkpoint and backend. A session owns committed
causal state. Each token builds unpublished next state for every layer.
The final mean, norm, and head must succeed before commit. Failure drains the
stream and drops unpublished state. Session drop also drains the stream.
Cache forks and drafting remain unavailable.

The causal and latent operation definitions supply host and device arithmetic.
Norm, mHC, and MXFP8 use paired host and device implementations. The bounded
host stack preserves the proved dense KDA recipe. It also supplies pooled
latent attention, corrected routing, selected experts, and shared output.
The separate `logical-f32-v1` reference retains its original meaning.

MXFP8 uses dynamic BF16-source activation groups and the fixed projection tree.
Plain projections use an ascending FP32 feature fold. Ordinary outputs round
to BF16. Router logits stay FP32. Block FP8 shared projections retain BF16
activations and the checkpoint block scales.

The calibrated NVFP4 input reference uses complete activation groups and the
checkpoint's positive input scale. It rounds block scales to E4M3 and input
codes to E2M1 with nearest-even ties. The projection combines global scales
once. It combines both block scales before the fixed ascending FP32 fold.
The completed expert output rounds to BF16. Expert contributions fold in
ascending expert ID. The shared output is added once and last.

The device Factory now loads the matrix owners before prompt execution. The
[resident pair diagnostic](RESIDENT-DECODE.md) splits complete matrix rows,
keeps embeddings and selected-expert operands on the device, and uses the
device MXFP8 quantizer. The single-rank Factory refuses a checkpoint that does
not fit. The host reference retains its original arithmetic. Causal history
and recurrent or latent state remain on the device.

Set an external deadline for the full suite. Each fresh process reads the
checkpoint and executes the prompt and generated sequence. A deadline for
one process may not cover the full suite. Preserve completed process receipts
if the external deadline stops the suite. A partial suite has no suite
completion marker. The `tokens compare` command can check completed process
receipts without that marker.

The older `tokens` receipt describes complete rows on each Spark. Use the
resident pair diagnostic for row-split transport evidence. Long context, cache reuse, saved-state restore,
drafting, backend parity, quality, and public service admission remain open.
A passing bounded result does not authorize a release or merge.
