# Checkpoint expert path

This probe checks routed NVFP4 and shared FP8 block experts.
It reads the first MoE layer from the checkpoint description.
It accepts routing from its caller. It does not implement attention.

`ExpertPath::new(checkpoint, layer, ids)` binds the selected expert union.
`ExpertPath::reference(input, ids, weights)` returns every scalar output plane.
`ExpertPath::upload(backend)` retains the device weights and scale bytes.
`ResidentExpertPath::apply(input, ids, weights, observe)` returns the device output.
The observer receives each complete output plane before the call returns.
All input and output features use FP32 storage lanes.

The NVFP4 quantizer first rounds the source to BF16.
Each block contains 16 source values, as declared by NVFP4 `group_size`.
The block scale uses `amax / input_scale / 6` and nearest-even E4M3 rounding.
Each source value uses `(source / input_scale) / block_scale` and nearest-even E2M1 rounding.
The checkpoint supplies `input_scale`. The probe does not recalibrate it.
The quantizer preserves signed zero and saturates finite encodings.
A zero block scale gives signed zero codes.

The projection visits columns in ascending order with FP32 accumulation.
It folds `(weight_block * activation_block) * (weight_global * activation_global)` once per block.
It multiplies each code product by that fold before each FP32 addition.
It rounds the completed projection to BF16.
Shared experts use stored FP8 weights, FP32 block scales, and BF16 source values.
Their projection rounds the complete FP32 sum to BF16.
SwiGLU rounds SiLU to BF16 before the up multiply and BF16 middle output.
The final fold visits ascending expert IDs and rounds each contribution and addition.
It adds shared output last.
These rules follow the orchestrator contract and the scalar cast recipe.
The logical FP32 decoder reference retains its separate meaning.

Run the probe with a saved `layer-step` output directory:

```text
checkpoint-prefix-check expert-path suite MODEL SAVED_LAYER0 NEW_OUT
checkpoint-prefix-check expert-path compare-suite LEFT RIGHT
```

The bounded suite takes 8 rows from saved layer-zero `ffn_output` planes.
The `TOKENS` constant in the probe fixes this test extent.
The saved output manifest checks each source row hash.
The probe uses the existing host router with checkpoint gate parameters.
It records selected IDs, weights, logits, parameter hashes, and input hashes.
The suite starts 3 fresh device processes, as specified in its loop.
It compares all device planes with the scalar reference and the first run.
Quantizer receipts include actual packed E2M1 bytes and E4M3 scale bytes.
A second station compares both suites with `compare-suite`.
The probe uses complete rows on each Spark. It does not test distributed transport.
It does not prove complete layer output, token IDs, or speed.
