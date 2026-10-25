# Reusing router and gate weights across prompt rows

This note covers two prompt-time changes on the typed hybrid path. The first
loads the small router and shared-expert gate weights once per block and reuses
them for many prompt tokens. The second sends only the last prompt row through
the vocabulary head. Neither changes the order of any arithmetic, so the token
IDs stay the same.

## The router and gate kernel

The matrices in question are the per-layer and drafting-layer router and the
shared-expert gate, all stored as BF16. The prompt kernel takes them when all
of the following hold:

- the prompt has more than 8 tokens;
- the matrix has at most 512 output columns;
- its input width K is at most 4096 and divisible by 64.

Each launch covers up to 256 tokens, and a longer prompt takes several
launches. Each block has 256 threads and covers eight output columns. It copies
its eight-column weight slice into 16 × K bytes of shared memory once, then
reuses it for every token in the launch. Other BF16 matrices, and prompts of 8
tokens or fewer, keep the existing kernel. The routed-expert GEMMs, scatter and
combine are unchanged.

## Why the token IDs stay the same

Each warp handles four groups of eight tokens. Its rows 0–7 carry the high BF16
part of each input and rows 8–15 the low part, the same split that decode uses.
K is cut into four fixed ranges. Within each range, the warp runs K16
`mma.sync` instructions in ascending order and adds the high and low parts at
the end. It then merges the ranges as `(P0 + P1) + (P2 + P3)`.

The range boundaries and the operand words fed to each MMA do not depend on
the number of tokens. The weight fragments come from the same packed or
row-major layout as decode, and incomplete output tiles are masked. Running
the four ranges inside one warp, instead of across four warps, changes no
operation for any output.

## Only the last row goes through the head

A prompt only needs logits for its last position. Prompt prefill therefore
sends only its final hidden row through the final norm and the vocabulary
head. The norm keeps the same reduction order and thread count as before.

Paths that need every row still get every row. These include verifying drafted
tokens, ordinary batched decode and `Model::head_rows`. The drafting head still
sees every prompt hidden row and keeps its own one-row head. A future API that
returns log-probabilities for every prompt position must use the all-row path.

## What the tests show

Host tests check which kernel and launch shape each of 1, 7, 8, 9, 255 and
256 rows gets. A separate host test compares FP32 results of the four-range
order bit for bit against a host model. They prove the schedule, not the
GPU's internal rounding inside an MMA. Before a build counts as qualified, it
needs a full-logit byte comparison against the previous build on a Spark and a
cold 8K before-and-after timing.
