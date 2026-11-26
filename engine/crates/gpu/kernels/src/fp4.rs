//! NVFP4 primitives for GB10, opt-in through `cargo kernels --fp4`.
//! E2M1 nibbles, positive E4M3 scales per 16 K, FP32 accumulators.
//! Fragment/scale mapping follows NVIDIA PTX ISA 8.8, sections 9-7-14-3/5.11.
use core::arch::{asm, nvptx::_thread_idx_x};
use crate::canonical::Acc;

/// Every lane participates. A has four packed words, B two; C/D use the
/// existing m16n8 accumulator layout. Selectors {0,0}: quad lanes 0/1 supply
/// A scales for rows group/group+8; lane 0 supplies B scales for column group.
/// Each scale word packs the four consecutive K16 scales, low byte first.
#[inline(always)]
pub(crate) unsafe fn mma(a: [u32; 4], b: [u32; 2], sa: u32, sb: u32, c: Acc) -> Acc {
    let (d0, d1, d2, d3): (f32, f32, f32, f32);
    asm!(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 \
         {{{d0},{d1},{d2},{d3}}}, {{{a0},{a1},{a2},{a3}}}, {{{b0},{b1}}}, \
         {{{c0},{c1},{c2},{c3}}}, {sa}, {{0,0}}, {sb}, {{0,0}};",
        d0=out(reg32)d0, d1=out(reg32)d1, d2=out(reg32)d2, d3=out(reg32)d3,
        a0=in(reg32)a[0], a1=in(reg32)a[1], a2=in(reg32)a[2], a3=in(reg32)a[3],
        b0=in(reg32)b[0], b1=in(reg32)b[1],
        c0=in(reg32)c.a, c1=in(reg32)c.b, c2=in(reg32)c.c, c3=in(reg32)c.d,
        sa=in(reg32)sa, sb=in(reg32)sb,
    );
    Acc { a: d0, b: d1, c: d2, d: d3 }
}

/// D = (A * scale_A) (B * scale_B)^T + C. No global scales are applied here.
/// # Safety
/// Exactly one block of 32 threads. A/B contain 512/256 bytes, scale_A/B
/// 64/32 bytes, all aligned to 4. C/D contain 128 FP32 values in row order;
/// D may alias C but no other input. E4M3 scales are nonnegative and finite.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_mma_tile(p: crate::kernel_params::Fp4MmaTile) {
    let lane = _thread_idx_x() as usize;
    let (row, col) = (lane / 4, lane % 4);
    let a = p.a as *const u32;
    let b = p.b as *const u32;
    let sa = *((p.scale_a as *const u32).add(row + (col % 2) * 8));
    let sb = *((p.scale_b as *const u32).add(row));
    let c = p.c as *const f32;
    let out = row * 8 + col * 2;
    let d = mma(
        [*a.add(row * 8 + col), *a.add((row + 8) * 8 + col),
         *a.add(row * 8 + col + 4), *a.add((row + 8) * 8 + col + 4)],
        [*b.add(row * 8 + col), *b.add(row * 8 + col + 4)], sa, sb,
        Acc { a: *c.add(out), b: *c.add(out + 1), c: *c.add(out + 64), d: *c.add(out + 65) },
    );
    let y = p.d as *mut f32;
    *y.add(out) = d.a;
    *y.add(out + 1) = d.b;
    *y.add(out + 64) = d.c;
    *y.add(out + 65) = d.d;
}

/// Exact E2M1 promotion, including signed zero. Integer bit operations keep
/// the conversion in registers; no lookup table or expanded weight storage.
#[inline(always)]
fn decode(code: u8) -> f32 {
    let magnitude = (code & 7) as u32;
    let bits = if magnitude < 2 { magnitude * 0x3f00 } else { 0x3f00 + (magnitude << 6) };
    f32::from_bits((bits | ((code as u32 & 8) << 12)) << 16)
}

/// Decode two adjacent nibbles directly into one BF16 MMA register. Apply the
/// block scale then the global scale in FP32, round once to BF16 (ties even).
#[inline(always)]
pub(crate) unsafe fn dequant_pair(raw: u8, scale: u8, global: f32) -> u32 {
    let s = crate::fp8_bf16::decode(scale);
    crate::gemm::bf16_pair((decode(raw) * s) * global, (decode(raw >> 4) * s) * global)
}

/// Expose the register conversion for small tiles and operand inspection.
/// In a projection, call dequant_pair at the MMA load instead of expanding W.
/// # Safety
/// elements is a multiple of 16. Input has elements/2 bytes, scales elements/16
/// nonnegative finite E4M3 bytes, output elements BF16 values aligned to 4.
/// Buffers do not overlap. Launch a 1D grid/block covering elements/2 threads.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn fp4_dequant_bf16(p: crate::kernel_params::Fp4Dequant) {
    let pair = crate::index();
    if pair >= p.elements as usize / 2 { return; }
    let raw = *((p.input as *const u8).add(pair));
    let scale = *((p.scales as *const u8).add(pair / 8));
    *((p.output as *mut u32).add(pair)) = dequant_pair(raw, scale, p.global_scale);
}
