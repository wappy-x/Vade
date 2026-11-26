# Typed hybrid state receipts

The typed hybrid path loads per-expert block-FP8 checkpoints on one Spark. To
test it, a session can copy its saved state out as bytes, so that two runs can
be compared byte for byte. We call such a copy a state receipt. This note says
what each receipt holds, when it is refused and what it proves.

## What the receipts hold

`Session::target_state_bytes()` holds the model's own state at the last
committed token. That is the position, the FP32 recurrent state of each
linear-attention layer, its packed BF16 convolution history and the visible
attention keys and values. It leaves out drafting state and scratch memory.

`Session::committed_state_bytes()` starts with the target receipt and adds the
rest of what a later turn would reuse:

- the prompt tokens kept for replay;
- the drafting head's state: its cursor, its last hidden row and its visible
  keys and values;
- the saved recurrent and convolution checkpoints;
- the saved hidden row the drafting head replays from.

A flag byte before each optional part says whether it is present. Every size
comes from the model's configuration.

## When a receipt is refused

The target receipt is refused while a prompt or draft is still pending, or
after an error has invalidated the session. The complete receipt is also
refused for an external draft model, for a drafting head that has not reached
the committed position, and when the replay hidden row is missing.

## How to use them

Taking a receipt synchronizes the GPU, reads device memory and allocates
temporary copies. Receipts are for tests only and never run while serving a
request. They can be large, so keep them on the Spark that made them.

Compare target receipts between a run with drafting and one without. Compare
complete receipts only when both runs keep the same drafting and replay state
and split their arithmetic the same way. Full-vocabulary logits need a
separate comparison.

A scalar reference for the linear-attention layers lives in
`engine/tools/bench/src/gdn_reference.rs`. It uses the host's math functions,
so it does not show that the GPU's approximations match.

## What they do not prove

The typed hybrid path runs on one Spark only. Both public load functions refuse
any other world size. Internal tests can call the shared loader core directly
with a checked rank range, so a test can exercise the same loader without
opening a multi-Spark serving path.

Host tests of the reference and of state copies stop short of real hardware.
They do not prove memory registration, placement or capacity, full-model
exactness on the GPU, or that the server is ready.
