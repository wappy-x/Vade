# GLM prompt rows

The bounded GLM decoder reads a prompt in large groups, up to 1024 rows of one
chat. All rows finish one layer before the next layer starts, so one weight read
serves many rows. [`Decoder::prefill_rows`](../crates/tp/src/bounded_decoder/rows.rs)
is that path. It keeps every row's scalar arithmetic: the same folds, the same
rounding boundaries and the same order of additions.

## What runs together

- Projections take token rows `[tokens,cols]` through one weight pass. NVFP4 and
  FP8 keep one accumulator per (weight row, token). MXFP8 uses the existing M
  kernels, which keep one element set, FMA chain and shuffle tree at every M.
- Experts: rows choosing one expert share its pass. One launch covers all
  selected experts of a layer, with one gather for each of the middle and down
  outputs. Gate and up meet in SwiGLU on each rank, before that gather.
- mHC, norms, the router and the expert fold run each row in the same launch.

## What stays in token order

- KDA state: one thread owns each value column and carries it through the tokens.
- Convolution history: each token reads the rows before it.
- Latent attention: each token selects from its own causal prefix.

A pool depends only on its own rows, so the last token's pools hold each earlier
token's pools as a prefix. A latent row's expansion depends only on its bits, so
it is computed once and later tokens read it. These are pure functions, and the
results have the same bits.

## Decode table

`expert_group` reads weight bytes through a table of decoded E2M1 and E4M3 values.
This has no decode branch, so lanes with different weight rows do not diverge.
The table holds the same bits as `e2m1` and `e4m3`, and a test checks each byte.

## Limits

Rows past `EXPANDED_ROWS` use the older `prefill`, which expands only selected
rows. Both paths give the same bits. The head runs only on the last row of a
served prompt. The checker can run it on every row to compare prompt logits.

## Proof

`resident-decode-check` saves all generated logits and a hash of every prompt
row's logits. `compare` checks them against a run of the scalar path.
`resident-decode-check prompt-rows` checks each row kernel against the host code.
