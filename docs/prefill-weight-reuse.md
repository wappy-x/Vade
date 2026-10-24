# Canonical router/gate prompt weight reuse

The typed hybrid BF16 binding gives layer and draft `Part::Whole` matrices
(router and scalar shared-expert gate) a bounded prompt tile. For 9–256 tokens,
N <= 512 and K <= 4096 divisible by 64, it launches one 256-thread CTA per
eight output columns. Each CTA stages its eight-column BF16 weight slice once
in 16*K bytes of shared memory. Other BF16 roles and smaller cohorts retain
the original M8 kernel. Routed-expert GEMMs, scatter and combine are unchanged.

Each warp owns four independent groups of eight tokens. Its A rows 0–7 carry
the high BF16 terms and rows 8–15 carry the low terms, exactly as in M8 decode.
For each of the four fixed K ranges, it executes ascending K16 `mma.sync`
instructions, adds H+L only at the end of that range, then merges
`(P0 + P1) + (P2 + P3)`. Range boundaries and MMA operand words do not depend
on the number of tokens. Staging reads the same packed or row-major B
fragments as decode and masks incomplete output tiles. Scheduling the ranges
within a warp instead of across four warps changes no per-output operation.

Prompt prefill projects only its final hidden row through the final norm and
vocabulary head. The norm keeps order 2 / 128 threads. `Session::prefill`
already returns only the final position. `Decode::verify_batch`, ordinary
batched decode, and `Model::head_rows` retain all-row output. MTP still observes
every prompt hidden row and retains its own one-row draft head. The server's
`BatchOutput::All` uses verification/step; `Last` uses prefill. Any future
prompt logprob API requiring all positions must use the all-row entry.

Portable random tests compare operand layouts and four-range host-model
FP32 result bits at 1, 7, 8, 9, 255 and 256 rows. They are scheduling proofs,
not an emulation of hardware MMA internal rounding. Native full-logit bytes
against the base and a cold 8K A/B are required before qualification.
