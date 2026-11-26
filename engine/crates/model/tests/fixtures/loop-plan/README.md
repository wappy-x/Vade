# Looped text plan fixture

This is synthetic test data. It contains no checkpoint payload.
The dimensions in `config.json` are small test values. They are not the
dimensions of a published checkpoint. The tensor names in
`hf_quant_config.json` are a short made-up list.

The fixture uses the field names of the public
[Inkling 2 Mini config and sidecar](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a).
The field meanings follow `configuration_inkling2_loop.py` and
`modeling_inkling2_loop.py` in public Transformers, release v5.24.0, commit
`eeee11099135592f1d03c2871fbd3d8132e5a4d7`. No source file is copied here.

`config.json` has 6 layers in blocks of 2. Layers 0, 2 and 4 have window
attention. Layers 2 and 3 are the loop range and run 3 times. So one token
has 10 layer visits. The window layers and the full layers have different
head counts, so a test can tell which geometry a layer received.

`hf_quant_config.json` has 5 groups for the main file: two plain groups, one
4-bit group for layers 0 to 3, one 8-bit group for layers 4 and 5, and one
plain group with a 32-bit forward type. It has 1 group for the draft file.

[`loop_plan.rs`](../../loop_plan.rs) checks the description, the visit order,
the storage map, the scale tensor shapes and each refusal. One ignored test
takes a real model folder as a test argument. It reads headers only. These
tests do not establish execution support or GPU exactness.
