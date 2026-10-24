# Flash-Next prompt projections

For at least 64 prompt rows, native blocked64 dense projections use a fixed
64x128x64 BF16 tensor-core tile. A group sharing one FP32 input packs its BF16
high/residual planes once into the existing reserved input arena. Each eligible
projection then has one compute launch per prompt chunk. Matrices with fewer
than 64 output columns, irregular output widths, or K not divisible by 64 retain
the existing fused edge path. Decode dispatch is unchanged.

Eight warps each own 32x32 outputs. Two 32,768-byte shared buffers overlap
`cp.async` operand loading with K64 computation; the launch reserves 65,536
shared bytes. Resident weights retain their blocked64 layout and single copy.
The packed input arena is reused on the ordered stream, with no new persistent
allocation. Partial M64 and N128 tiles zero-fill invalid operands and suppress
invalid output stores. Host dispatch requires N and K divisible by 64.
Large inputs retain the previous wide kernel's cache grouping: 16, 8 or 4
adjacent M64 tiles share a pass over N for K up to 4096, up to 8192 or larger.
Only block ownership changes; neither K order nor operand addresses change.

Each live output executes ascending `mma.sync.m16n8k16` BF16 operations within
four K ranges, whose boundaries are `floor((K/16)*part/4)`. High and residual
accumulators remain separate until each range ends. The final merge remains
`(p0+p1)+(p2+p3)`, with `pi=high_i+residual_i`. Range boundaries may cross a
K64 stage, including K=320; they never depend on M or chunk size. FP32 output
storage and operand rounding match the previous prompt path. There is no
split-K reduction or new arithmetic mode.

Fusion changes profiler attribution: the previous prompt kernel included input
conversion inside every N tile, whereas this kernel reads packed operands.
Compare MMA plus input-pack seconds and cold prompt TTFT as well as MMA seconds.
A host scalar-FMA oracle establishes operand mapping and operation ordering;
it does not reproduce the GPU's internal K16 reduction. Native full-row
comparison and workload measurements are needed to qualify a build. The
superseded wide prompt kernel has been deleted; only the narrow/irregular
edge handler remains alongside the tiled kernel.
