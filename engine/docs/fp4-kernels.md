# NVFP4 on GB10

`cargo kernels --fp4` selects feature `fp4` and Rust flags
`-C target-cpu=sm_121a -C target-feature=+ptx88 -C overflow-checks=off` on
`nightly-2026-10-01`, with `-Z build-std=core --release --target nvptx64-nvidia-cuda`.
The emitted header is `.version 8.8`, `.target sm_121a`. The explicit `+ptx88`
pins the FP4 instruction version; the pinned compiler also emits PTX 8.8 for
ordinary `cargo kernels`, which retains its original `sm_121` flags and
excludes the native FP4 MMA entries.

FP4 builds use their own Cargo target directory, `target/fp4`, and publish
`spark_rust_kernels_fp4.ptx` / `spark_layer_kernels_fp4.ptx` in
`target/fp4/nvptx64-nvidia-cuda/release/`. They never overwrite or restamp the
production files in `target/nvptx64-nvidia-cuda/release/`.

Host builds default to the production `sm_121` PTX, even after an FP4 build.
Select the FP4 artifacts explicitly for both the GPU layer and server:

```sh
cargo kernels --fp4
SPARK_PTX_VARIANT=fp4 cargo build --release --offline --workspace --exclude spark-rust-kernels
```

Unset `SPARK_PTX_VARIANT` (or set it to `production`) for ordinary host builds.
Cargo tracks changes to this variable. Each `.ptx.sha256` stamp records and
hashes the feature set, target CPU, PTX version and Rust flags along with the
sources, ABI and PTX bytes. Host builds reject a different variant's stamp or
PTX header, including through `SPARK_LAYER_PTX`. Missing production PTX never
falls back to FP4; an explicit FP4 selection requires its artifacts even on a
non-Linux host. Rebuild both variants after shared kernel or ABI changes.
Selecting FP4 artifacts does not enable an inference mode.

CUDA 13.0.88 ptxas and driver 580.178.04 accept and execute the Rust PTX on GB10.
Changing only the target to `sm_121` makes ptxas reject block scaling.
Disassembly contains `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` (30 registers,
no stack, spills or shared memory).

## Native tile

```text
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
  {d0,d1,d2,d3}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1,c2,c3},
  scale_a, {0,0}, scale_b, {0,0};
```

`fp4::mma` consumes registers. `fp4_mma_tile` loads row-major A[16,64] and
transposed B[8,64], low nibble first, with A[16,4]/B[8,4] E4M3 scales per 16 K.
C/D are row-major FP32[16,8]. Launch one block of 32 threads using `Fp4MmaTile`;
all six pointers are 4-byte aligned. The operation is `(A*SA)(B*SB)^T+C`.

For lane l, g=l/4 and t=l%4. Each operand word contains eight K values:

| Registers | Row in stored operand | First K / output column |
| --- | --- | --- |
| A0, A1, A2, A3 | g, g+8, g, g+8 | 8t, 8t, 8t+32, 8t+32 |
| B0, B1 | g, g | 8t, 8t+32 |
| C/D registers 0,1 | g | columns 2t, 2t+1 |
| C/D registers 2,3 | g+8 | columns 2t, 2t+1 |

Each b32 scale operand packs four consecutive K16 scales, low byte first.
Selectors are `{byte_id,thread_id}`, unsigned 16-bit operands (immediate zero
works). With `{0,0}`, quad lanes 0/1 supply A rows g/g+8; quad lane 0 supplies B
column g. Byte selectors must be zero for `4X`. Finite nonnegative E4M3 codes
are 0..126.

Other documented forms: dense `kind::f8f6f4` m16n8k32 without scales;
`kind::mxf8f6f4.block_scale.scale_vec::1X` m16n8k32 with UE8M0 scales. Their
E2M1 operands occupy bits 5:2 of each byte, not packed nibbles. MXFP4 uses
m16n8k64, `2X`, UE8M0, with kind `mxf4` or `mxf4nvf4`. Sparse ordered-metadata
forms use m16n8k128 for packed FP4, or m16n8k64 for f8f6f4, requiring compressed
A and sparsity metadata. These alternatives were not executed; use the tested
dense `mxf4nvf4`/`4X`/UE4M3 form for this checkpoint.

## Dequantization and reference

`fp4::dequant_pair` promotes two E2M1 nibbles with register bit operations,
reuses Vade's E4M3 decoder, multiplies by block scale then FP32 global scale,
and rounds once to BF16 ties-to-even. `fp4_dequant_bf16` exposes the conversion
(14 registers, no stack/spills/shared memory). Projections should call the
helper at their BF16 MMA loads and retain only packed weights in device memory.

The independent CPU reference uses E2M1 magnitudes `[0,.5,1,1.5,2,3,4,6]` and
sign bit 3. For E4M3 exponent e and mantissa m it uses `m/512` if e=0, otherwise
`(8+m)*2^(e-10)`. The dot starts at C and adds `(decode(A)*SA)*(decode(B)*SB)`
in ascending K. BF16 conversion adds `0x7fff+((bits>>16)&1)` then shifts 16.

The probe passed 16,384 bitwise FP32 comparisons (64 single-K basis tiles,
64 dense tiles, distinct row/block scales, both signs and nonzero C). Its
bounded dyadic inputs make every product and partial sum exact in FP32.
Scales were chosen from 0, 0.5, 0.75, 1, 1.25, 1.5, 2 and 2.5, with
C[i]=(i-64)/8. Products are multiples of 1/64 and absolute partial sums are
at most 14,408: fewer than 24 significant bits, regardless of summation order.
This proves layout/scaling and execution. NVIDIA leaves internal MMA order
unspecified; arbitrary-input scalar FP32 equivalence is not established.
Dequantization passed 8,128 bitwise BF16 comparisons: 16 codes, 127 finite
nonnegative scales, globals 1, 0.1, 2^-20 and 12345.25, including signed zero
and E4M3 subnormals. The temporary Rust runner/reference was outside the tree.

The projection and model integration work streams should use the native
primitive and an ascending K64 instruction chain independent of batch width.
Specify global FP32 scaling and BF16 output
rounding separately: global scales are not UE4M3 metadata. Native FP4 and
BF16-dequantized MMA are different arithmetic paths; do not silently alternate.
Full checkpoint, cold/cached and batch parity remain integration work.

## Sources (ideas and instruction contracts; no code copied)

- NVIDIA: [block scaling](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-block-scaling),
  [fragments](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16864),
  [mma](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma).
  Instruction support began in PTX 8.7; sm_121a was added in 8.8.
- TensorFold `python-0.6` at `ed78d6fc204d89d90b045bf033d6551e7714f3a1`:
  [build](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/src/tensorfold/cuda/build.py),
  [checkpoint](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/src/tensorfold/cuda/nvfp4/checkpoint.py),
  [MMA](https://github.com/ashhart/TensorFold/blob/ed78d6fc204d89d90b045bf033d6551e7714f3a1/src/tensorfold/cuda/nvfp4/mma4.cuh).
  Its native path sets `arch_specific=True`, producing
  `-gencode=arch=compute_121a,code=sm_121a`. Current main has a different tree.

## Flash inference

The `vade up` CLI route supports FP8 and has no NVFP4 option. The native
engine also serves the official mixed NVFP4 checkpoint on main. See
[Flash-Next NVFP4 serving](next-http.md) for the one-Spark route and the
separate multi-Spark preview qualification limits.

`FLASH_NVFP4=1` (or `flash-serve --nvfp4`) enables the official mixed NVFP4
checkpoint with W4A16 routed experts for both decoding and prompts. Ordinary
`cargo kernels` includes these BF16 tensor-core kernels; the isolated `--fp4`
PTX variant is needed only for the native W4A4 primitives above. The inference
path keeps packed weights in one arena, promotes operands in registers, and
does not implement the checkpoint's W4A4 activation quantizer. Main expert
descriptors occupy 40,960 bytes per layer. Empty rank-local experts are zero
descriptors; their output partials are zero. MTP experts remain FP8; all dense,
router, attention, shared-expert and PLE paths retain their existing dtypes.

The loader requires its SHA-256 manifest, supplied by `FLASH_NVFP4_MANIFEST`
or `manifest.sha256` in the model directory. It verifies every payload once
per process before loading. `flash-bench` can compare full logit dumps at
identical prefixes using `FLASH_TEACHER_FORCING=1`: each fixture's expected
token becomes the next input, while the runner still reports actual top-1.
