# Tensor index fixture for the looped text family

This is synthetic test data. It contains no checkpoint payload.
The sizes in `config.json` are small test values. They are not the sizes of a published checkpoint.

| File | What it holds |
| --- | --- |
| `config.json` | A made-up config with the field names of `model_type` `inkling2_loop`. |
| `hf_quant_config.json` | The storage sidecar: five groups for the main file and one for the draft file. |
| `header.json` | The safetensors header of the main file: 292 names (286 text, 6 `model.visual.*`). |
| `draft-header.json` | The safetensors header of the draft file `mtp.safetensors`: 39 names. |

The field and tensor names come from the public
[Inkling 2 Mini files](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/b85c3cfa615fc413cc9af3379f491b963f3b586a)
(`config.json`, `hf_quant_config.json` and the two safetensors headers).

## Why the sizes differ from each other

In the published checkpoint many sizes are equal: the hidden size is twice the expert width, the expert
count equals the head size, and a window layer has the same head sizes and relative extent as a full layer.
A shape rule that takes the wrong one of two equal sizes passes on that checkpoint.
In this fixture every size is different, so such a rule fails here:

- Hidden size 96, expert width 64, 7 experts, 320 stored vocabulary rows.
- Window layers (0, 2, 4, 6): 4 query heads, 2 key and value heads, head size 14, window 36.
- Full layers (1, 3, 5, 7): 6 query heads, 3 key and value heads, head size 22, relative extent 52.
- Relative width 10, 5 convolution taps, loop layers 2 to 5 with gate rank 9.
- Layers 0 to 5 hold 4-bit expert stacks. Layers 6 and 7 hold 8-bit expert stacks.
- Draft model: 2 layers, width 176, rank 14, 4 states read from the main model.

## How the headers were made

A script that is not in this repository holds the shape rules a second time, written from the same sources as
[`inkling_index.rs`](../../../src/plan/inkling_index.rs). With the published config and sidecar, the script
gives the published headers: 822 of 822 text names and 54 of 54 draft names, with equal types and shapes.
With the made-up config, it gave these two header files. So the headers are not output of the code that the
tests check.

[`inkling_index.rs`](../../inkling_index.rs) builds a checkpoint folder from these files. The tensor files have a
header and a hole in place of the payload. The tests check roles, types, shapes, counts and each refusal.
They do not establish execution support or GPU exactness.
