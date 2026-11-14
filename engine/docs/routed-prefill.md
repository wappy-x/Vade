# Routed FP8 prompt tiles

The typed hybrid forward calls `fp8_routed_ffn`, independently of the legacy
BF16 `hybrid_moe::Workspace`. Its expert projections keep four ascending K32
E4M3 MMAs in each K128 block, apply that block's activation/weight scales, then
fold K128 blocks in ascending order. The prompt tile changes no accumulator
reset, split count, reduction tree, BF16 boundary, or intermediate quantizer.

At 4,096 or more input rows, each M128 routing task uses a 512-thread M128 tile.
The typed prompt schedule currently passes 256-token cohorts, averaging eight
rows per expert for top-8 routing over 256 experts. Those cohorts use M16/N128
with 128 threads, avoiding computation over 112 empty rows per expert.
Down covers N128. Gate/up share its activation tile: two warp columns read
64 gate channels and two read the corresponding 64 up channels, with separate
weight scales. Both still write the original BF16-rounded FP32 middle buffer.
Cohorts below 256 rows retain M16/N64. Dynamic shared storage is 66,560,
36,992 or 20,608 bytes; the register cap funds 512-thread launches within 65,536 registers.

Integer histogram positions may assign physical expert rows in any order.
`output_row` retains the ascending expert-ID destination for each token, so
physical placement changes neither an output element's K order nor the finish
fold. This avoids scanning every earlier pair for each assignment. Atomics
allocate integer slots only; floating-point accumulation remains ordered.

Down writes unweighted FP32 directly to those destinations. The routing clear
initializes rejected rows after its warp's validation, and the shared row is
still written separately. Rank reduction and ordered finish consume the same
FP32 values. The old down staging allocation remains reserved to avoid changing
workspace admission. Moving weighting before rank reduction or
compressing partials to BF16 would require a separate arithmetic review.

The separate Flash grouped path uses four M32 prompt planes per M128 task;
small M16 expert tails use its first two planes. Decode retains eight M16 planes.
Its live-row ownership and K128 arithmetic are unchanged.

The legacy BF16 workspace's decode selection and four-range balanced K reduction
remain coupled. Simply selecting its linear-K wide tile changes bytes; that is
not an arithmetic-preserving optimization. Host ownership and operation traces
support this implementation; full GPU answer-byte and timing qualification remains required.
