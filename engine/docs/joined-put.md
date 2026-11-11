# Joined put

One launch sends one exchange message of the two-Spark resident decoder and
prepares its gathered output. It replaces up to five launches a message: the
BF16 round of each local part, the join of the parts, the fresh zero fill of
the output, the scatter copy of the own rows, and the finite put.

## What it writes

- The send slot gets the message: the local parts in order. A part that still
  owes its BF16 round is rounded first, with `bf16::rounded`, the function the
  round kernel uses.
- Every word of the gathered output is written. The own rows hold the same
  message words at their final places. Every other word is zero, so a failed
  take still leaves initialized words, as the fresh fill did.
- The length word and then the sequence word are published last.

No arithmetic changes. The round is the same integer function on the same
words. All other work is word moves. A nonfinite message word (also one that
rounds to infinity) sets the sticky mailbox error and nothing is published,
as `tp_mailbox_put_finite` does.

## Shape

`tp_put_join::table` builds a checked row table: slot rows first, then the own
rows, then zero rows for the rest of the output. `tp_put_join::encode` writes
addresses and, for each row, the count of units before it. A unit is four
words when both ends are 16-byte aligned and the count divides, else one word
(`tp_put_join::quad`, shared by host and kernel).

Block 0 (1,024 threads) moves the slot rows and publishes, exactly as the
one-block put does: one block owns the slot, and its barrier orders every slot
store before thread 0 publishes. Up to 32 more blocks move the output rows.
They take no part in publication; later launches of the same stream read the
output after this launch ends. Each thread finds its row with a binary search.

`tp_put_join::admitted` is a shape rule only: at most 32 parts, 64 jobs,
131,072 message words and 262,144 output words. Every message of a decode step
is inside it. A wider message (prompt rows) keeps the head launches, which
write the same words.

## Where it is used

`gather_device` in `crates/tp/src/resident_weights.rs` is the one call site.
`apply`, the packed batch, the routed projection and the projection stage pass
their unrounded split parts with a flag, so the round runs inside the joined
put. A whole (not split) output and a path that does not reach a device gather
round at once, as before.

The recorder kind is `PutJoin` (`resident_launch_put_join`), so the launch is
part of a graph piece in a persistent decode step.

## Proof tools

- Host tests `tp_put_join::tests`: one writer for every slot and output word
  and the head value, at every message shape and launch shape.
- `put-join-check` (GPU, no model): byte equality against the head launches
  with the slot and the output poisoned first, eager and recorded; nonfinite
  refusal; device time a message for both forms.
- `resident-decode-check exchange-stage`: stage gathers of 1 to 32 parts on
  both ranks, eager and recorded.
