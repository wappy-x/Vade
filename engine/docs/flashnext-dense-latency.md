# Flash-Next dense projection schedule

M<=16 uses the persistent N8 schedule. One m16n8k16 MMA covers all sixteen rows of the
packed tile, so each output keeps the canonical K16 hi/lo chains and (p0+p1)+(p2+p3)
merge. There is no environment selector. M17..63 keep the canonical M16/N32 kernel.

192 CTAs stride over N8 tasks. Four warp-private K16 stages overlap operand
copies while retaining ascending K16 hi/lo chains and the canonical four ranges.
N<=640 uses independent range tasks and a stream-ordered second pass evaluating
`(p0+p1)+(p2+p3)`. Partial storage follows packed inputs inside the existing
reserved buffer. Larger N uses four warps per CTA; dynamic shared memory is
5120 bytes per warp. Weights remain single-copy.

Host tests cover operand identity, pipeline ownership, persistent task coverage
and partial bounds. Prefill uses the separately qualified tiled kernel whenever
its shape is eligible. Ineligible narrow/irregular shapes retain edge handling.
The integration GPU handoff compares full rows against the pinned base, C1/C8,
chunks 512/2048 and MTP/plain, with two repeats. No Mac timing is a GPU claim.
