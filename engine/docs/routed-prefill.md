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

## Gathered gate rows

A prompt group gives every token's row to each of its picked experts. The gate and
up step used to copy the rows into expert order first (`PermuteRows`: eight copies
of each row on the model, 134 MB for 1,024 tokens) and `nv_quantize_pack` read the
copy. `nv_quantize_gather` reads each row where it lies, through the table that
copy would read (`Compute::rows_experts_gathered`, `Backend::expert_group_gathered`).
The copy moves words, and the operand bytes of a row depend on that row and its
input scale only, so every operand byte is equal and the product launches do not
change. Two operand rows may read one input row; none writes it. A table word at
or beyond the source rows traps on the device, as it does in the copy.

A group that does not take the gathered operands (not dense, not on the
tensor-core image, gate and up at different input scales, or an older kernel
image without the entry) copies its rows as before. The served log shows the
path: `nv_quantize_gather=` in `SERVED PROMPT COUNTS` and one `GATHERED ROWS`
line at first use. Device proof: `kernel-bench --reading-probe OLD NEW
quantize-gather`.

Both entries call one body (`blocks` in `nv_quantize_pack.rs`, never inlined), so
the image holds one instance of the quantizer's expressions. A second instance
changed the compiled code of six other kernels of the image (MEASURED
2026-11-15: `causal_fast`, `causal_operation`, `expert_operation`,
`mhc_boundary_rms`, `fp4_prefill_tc_down`, `resident_launch_causal_group`).
With one body, 638 of the base image's 640 kernels keep their code; the two
others are `nv_quantize_pack` itself and one moved compare in
`resident_launch_causal_chain`.

### Under the lookahead
A prompt group that leaves the link step of its gate-and-up exchange open (LOOKAHEAD, `glm-prompt-rows.md`) opens it
on the gathered rows too: `Compute::rows_experts_open_gathered`, then `Weights::apply_experts_open` with the order
table. The row pass asks in this order: open on gathered rows, gathered rows, open on copied rows, copied rows. Each
form answers "no" before any launch, so a later form starts from the same state.

The index table is dropped before the stream mark of the open exchange, so the close frees it with the buffers that
mark counts. The source rows stay in the caller's values until the layer ends. No copy of the rows is held while the
exchange is open: the copied form held its 134 MB copy through the close. Host proof: `tests/group_fold.rs`
(`a_prompt_group_opens_its_gate_and_up_exchange_on_rows_read_through_the_table`). Served proof: `prompt_link_ahead=`
and `nv_quantize_gather=` both rise by one a routed layer and group.
