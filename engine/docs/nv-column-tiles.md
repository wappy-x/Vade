# NVFP4 column tiles

Routed NVFP4 expert matrices can sit on the device as column tiles. The
values, the products, their order and every rounding stay as in the row-major
form. Only the byte order changes. The decode kernel for this layout does not
read the columns whose quantized activation is zero.

## Layout

A tile is 128 consecutive rows of one matrix.

- Codes: column `j` of tile `t` is the 64 bytes at `(t*cols + j)*64`. Row `2k`
  of the tile is the low nibble of byte `k`. Row `2k+1` is the high nibble.
- Scales: 16-column block `b` of tile `t` is the 128 bytes at
  `(t*cols/16 + b)*128`, one byte a row.

The byte counts equal the row-major form. A shape with `block_rows == 128`
(`expert_ops::NV_TILE_ROWS`) names this layout. `expert_ops::nv_code_at` and
`expert_ops::nv_scale_at` give the byte of one code or scale in either layout.

## Storage rule

The rule reads the shape only (`expert_ops::nv_tile_layout_admits`): the rows
on this rank are a multiple of 128, and the columns are a multiple of 64 and at
most 4096. A matrix outside the rule, or with an invalid scale code, keeps the
row-major form and its kernels.

The loader moves an admitted matrix to column tiles on the device
(`resident_weights.rs`, `Matrix::upload`). It uploads the row-major planes to
two scratch vectors that every such matrix of the load uses again. The kernel
`nv_tile_repack` then writes the column tiles into the final vectors. The
scratch is the size of one matrix and is freed when the load ends, so the
resident set is the same as for row-major storage. A host pass over the planes
cost 14 seconds a rank on GLM-5.3-Flash; that is why the move is on the device.

`nv_tile_repack` only moves bytes. A block takes one 128-row tile and 64
columns: it stages 32 code bytes and 4 scale bytes of each row in shared
memory, and each thread stores whole output words. The staged rows have
offsets that keep every load of a warp in 32 different shared-memory banks.
The host functions `nv_tile_codes_in_place` and `nv_tile_scales_in_place` in
`expert_ops.rs` define the bytes; tests and device checks compare against them.

## Why a zero column can stay unread

Expert inputs are quantized to NVFP4. A value below about 1/24 of its
16-value group maximum gets code zero. Each output row is one chain from +0.0
of `(value * activation) * fold`. For a zero activation the product is +0 or
-0, and such a term cannot change the sum bits. `nv_cols` does not copy a
zero column from device memory. Its staged bytes are then stale, and the
kernel still forms the product: every code decodes to a finite value, so the
product is +0 or -0 for a finite fold, and the same NaN as the row-major
kernels for a non-finite fold.

## Kernels

| Work | Row-major | Column tiles |
| --- | --- | --- |
| Decode batch and device route | `nv_tile` | `nv_cols` |
| Dense prompt group | `nv_rows_tile` | `nv_rows_cols` |
| Prompt group of few tokens a part | `nv_rows_small` | `nv_cols`, one job a part a token |
| Any other launch | `expert_operation`, `expert_group` | the same bodies; they find bytes by layout |

`nv_cols`: one block of 128 threads owns one tile of one job. A thread owns
one row. The activations of the job are staged once in shared memory. A round
is four 16-column blocks. The live columns and the scales arrive by
asynchronous copies, three rounds deep. The arithmetic of a block is written
stage by stage: loads, decodes, products, then the 16 chain adds.

`nv_cols` has no L2 prefetch, on purpose. A prefetch pulls a whole 128-byte
line. A column of a tile is 64 bytes, so a prefetch of a live column also
brings its neighbour over the bus. When that neighbour is a zero column, this
is the traffic the skip was made to save. Measured on a GB10: with a prefetch
of live columns the kernel lost 10 microseconds a launch at the real zero
shares; a zero column saved time only when its line-mate was zero too. The
copies bring only the bytes they name. Keep this rule for any kernel that
leaves out units smaller than 128 bytes.

The host never launches a row-major kernel on a column-tile job
(`expert_ops::batch_launch_geometry`, `nv_stream::projection_admitted`,
`device_route::admitted`, the group choice in `backend/expert_ops.rs`). The
recorded replay uses the existing NvTile kind: the launch shape names the
layout.

## Gates

- Host: `expert_ops` tests `column_tiles_keep_every_nvfp4_output_bit_and_the_byte_counts`,
  `column_tile_jobs_never_get_a_row_major_kernel` and
  `device_repack_moves_every_byte_to_the_place_of_the_host_repack` (a host
  model of one `nv_tile_repack` block).
- One device: `resident-decode-check nv-group-rows OUT` (section
  `NV_COLUMN_PASS`: every matrix of the check goes through the device repack
  and is compared with the host repack), `resident-decode-check device-route OUT`
  and `kernel-bench --nv-cols IMAGE`.
- Real checkpoint: `resident-decode-check load-hash ROOT RANK PARTS OUT` writes
  the hashes of all device buffers; two builds that must hold the same bytes
  give the same `DEVICE-BUFFER-SHA256`.

[Back to the documentation index](README.md)
