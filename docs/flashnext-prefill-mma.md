# Flash-Next prompt projections

When Flash-Next reads a prompt, it runs many dense BF16 projections: one
matrix multiply per weight matrix, over every prompt row at once. This note
describes the tiled kernel that runs them, which matrices it takes, and why
its results match the older prompt path bit for bit. Decode uses separate
kernels.

## Which projections use it

The tiled path needs a prompt of at least 64 rows. A weight matrix takes it
when it has at least 64 output columns, a multiple of 64, and its input width K
is also a multiple of 64.

All projections that read the same FP32 input share one packing step. It
converts the input once into separate high and low BF16 planes, inside the
input scratch the engine already reserves. Each eligible matrix then gets one
launch per prompt chunk. When a prompt is too large for that scratch, it runs in
row groups that are multiples of 128 rows, so each row keeps its kernel and its
K order.

A matrix with fewer than 64 output columns, such as a small gate, rides along
in a larger matrix's launch: in the unused columns of its last tile when they
fit, otherwise in extra tiles at the end. Its outputs keep their own K order,
so their bytes match a launch of their own. Anything left over, including
every matrix when K is not a multiple of 64, goes through the older edge
kernel.

## The tile

Each block of 256 threads computes 64 prompt rows by 128 outputs, stepping
through K 64 at a time. Its eight warps each own 32 rows by 32 outputs. Three
32 KiB shared-memory stages let `cp.async` load the next operands while the
current ones are multiplied. The weights stay in their resident blocked
layout, with a single copy. Edge tiles zero-fill missing operands and skip
stores for missing outputs.

For large inputs, adjacent blocks along the prompt share a pass over the
outputs so the input stays in cache: groups of 8 blocks when K is at most 8192,
and 4 when it is larger. This changes only which block runs which tile. It
changes neither the K order nor any operand address.

By default the engine runs a variant of this kernel that computes full-tile
addresses as 32-bit element offsets. For prompts of 8192 to 32768 rows, a
second variant also works out each thread's source addresses once, ahead of
the loop. `VADE_PROMPT_GEMM_U32=0` selects the original entry, and
`VADE_PROMPT_COLD8K_R2=0` turns off the second variant. All of them keep the
same arithmetic order.

## Why the bits do not change

Each output runs ascending `mma.sync.m16n8k16` BF16 operations within four K
ranges. With `kt = K / 16`, range `i` covers K16 tiles `kt*i/4` up to
`kt*(i+1)/4`, rounded down. The high and low accumulators stay separate until
a range ends. The final merge is `(p0 + p1) + (p2 + p3)`, where each `pi` is
that range's high plus low.

A range boundary can fall inside a K64 stage, as it does at K = 320. The
boundaries depend only on K, never on the number of rows or the chunk size.
FP32 output and operand rounding match the previous prompt path. There is no
split-K reduction and no new arithmetic mode.

## Measuring and qualifying it

Packing the input once moves time between kernels in a profile: the old
kernel converted its input inside every output tile, and this one reads packed
operands. When you compare builds, add the packing time to the multiply time,
and compare the time to first token on a cold prompt as well.

A host check cannot reproduce the GPU's internal K16 reduction, so
qualifying a build needs a full-row comparison and timed runs on a Spark.
