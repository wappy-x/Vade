//! Standalone zero-spill prompt HC-down `on_read` GEMM (`norm != 0`, `K=10240`, `rows=320`,
//! `rows2=4`): replaces the 176 B stack / 532 B spill generic `flash_bf16_mma_prefill_tiled_hc`
//! on Cold 8K while preserving the exact canonical 4-range K16 hi/lo graph, stream-norm rounding,
//! `(p0+p1)+(p2+p3)` merge, `hc_sigmoid`, and packed output planes bit-for-bit.
//! Kept in its own compilation unit so `flash_prefill_mma.rs` stays completely untouched.
#![cfg(target_arch = "nvptx64")]

use crate::{
    canonical::{self, ZERO},
    kernel_params::Bf16Rows,
    ptx,
};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _block_idx_y, _syncthreads, _thread_idx_x},
};

const PLANE: usize = 8192;
const STAGE: usize = 32768;
const ROWS_A: usize = STAGE;
const ROWS_W: usize = STAGE + 16384;
const ROWS_STAGE: usize = ROWS_W + 256; // 49,408 B; 2 * ROWS_STAGE = 98,816 B

macro_rules! unroll {
    (2, $i:ident, $body:block) => {{
        #[allow(non_upper_case_globals)]
        const $i: usize = 0;
        $body
    }
    {
        #[allow(non_upper_case_globals)]
        const $i: usize = 1;
        $body
    }};
    (4, $i:ident, $body:block) => {
        unroll!(2, $i, $body);
        {
            #[allow(non_upper_case_globals)]
            const $i: usize = 2;
            $body
        }
        {
            #[allow(non_upper_case_globals)]
            const $i: usize = 3;
            $body
        }
    };
}

macro_rules! cp16 {
    ($dst:expr, $src:expr, $valid:expr) => {{
        let (dst, src, valid): (u32, *const u32, bool) = ($dst, $src, $valid);
        asm!(
            "cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
            d = in(reg32) dst,
            s = in(reg64) src,
            n = in(reg32) if valid { 16u32 } else { 0u32 },
        );
    }};
}

macro_rules! cp4 {
    ($dst:expr, $src:expr, $valid:expr) => {{
        let (dst, src, valid): (u32, *const u32, bool) = ($dst, $src, $valid);
        asm!(
            "cp.async.ca.shared.global [{d}], [{s}], 4, {n};",
            d = in(reg32) dst,
            s = in(reg64) src,
            n = in(reg32) if valid { 4u32 } else { 0u32 },
        );
    }};
}

/// Zero-spill prompt HC-down `on_read` GEMM for `K = 10240`, `rows = 320`, `rows2 = 4`.
/// # Safety
/// Launch grid `ceil(M/64) x 3`, block `256`, dynamic shared `FLASH_BF16_PREFILL_ROWS_SHARED`
/// (98,816 B). Requires `p.rows == 320`, `p.rows2 == 4`, `p.cols == 10240`, `scales` covering
/// `M x 4` F32, `norm` covering `10240` F32, `p.input` covering `M x 10240` F32, packed BF16
/// `p.weight` `[320][10240]` and `p.weight2` `[4][10240]`, and disjoint `p.output`/`p.output2`.
// SAFETY: Hc::mix_chunk (blocks.rs) validates on_read && K == 10240 && rows == 320 && rows2 == 4
// and launches with exact grid ceil(M/64) x 3, 256 threads, and FLASH_BF16_PREFILL_ROWS_SHARED.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_tiled_hc_onread(
    p: Bf16Rows,
    scales: u64,
    norm: u64,
) {
    let tid = _thread_idx_x() as u32;
    let (mut mi, mut ni) = (_block_idx_x(), _block_idx_y());
    let tokens = p.tokens;
    let mt_count = tokens.div_ceil(64);
    let nt_count = 3u32;
    if (tokens as u64) * 10240 * 4 > 16 * 1024 * 1024 {
        let group = 4u32;
        let linear = ni * mt_count + mi;
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows;
        ni = within / rows;
    }
    let mt = mi * 4;
    let nt = ni * 16;
    let lane = tid % 32;
    let warp = tid / 32;
    let wm = (warp / 4) * 2;
    let wn = (warp % 4) * 4;

    macro_rules! issue_stage {
        ($sh:expr, $k:expr) => {{
            let sh: u32 = $sh;
            let k: u32 = $k;
            let in_ptr = p.input as *const u32;
            unroll!(4, step, {
                let i = tid + (step * 256) as u32;
                let r = i / 16;
                let c = i % 16;
                let token = mt * 16 + r;
                let valid = token < tokens;
                let at = if valid { token * 10240 + k * 16 + c * 4 } else { 0 };
                let dst = sh + ROWS_A as u32 + (r * 16 + (c ^ (r & 7))) * 16;
                cp16!(dst, in_ptr.add(at as usize), valid);
            });
            if tid < 64 {
                cp4!(sh + ROWS_W as u32 + tid * 4, (norm as *const u32).add((k * 16 + tid) as usize), true);
            }
            let w1 = p.weight as *const u32;
            let w2 = p.weight2 as *const u32;
            unroll!(4, step, {
                let i = tid + (step * 256) as u32;
                let g = nt + i / 64;
                let c = i % 64;
                let dst = sh + (2 * PLANE) as u32 + i * 16;
                if g < 40 {
                    let tile = (g / 8 * 160 + k / 4) * 32 + (g % 8) * 4;
                    let at = tile * 64 + c * 4;
                    cp16!(dst, w1.add(at as usize), true);
                } else if g == 40 && (c % 16) < 8 {
                    let z = c / 16;
                    let part = c % 8;
                    let at = (k + z) * 32 + part * 4;
                    cp16!(dst, w2.add(at as usize), true);
                } else {
                    cp16!(dst, w1, false);
                }
            });
            ptx::commit();
        }};
    }

    macro_rules! convert_stage {
        ($sh:expr, $k:expr) => {{
            let sh: u32 = $sh;
            let k: u32 = $k;
            let stream = k / 160;
            let a32 = sh + ROWS_A as u32;
            let w32 = sh + ROWS_W as u32;
            let sc_ptr = scales as *const f32;
            unroll!(2, step, {
                let i = tid + (step * 256) as u32;
                let m = i / 128;
                let kk = (i % 128) / 32;
                let g = (i % 32) / 4;
                let q = i % 4;
                let r0 = m * 16 + g;
                let tok0 = mt * 16 + r0;
                let local = kk * 16 + q * 2;
                let w0 = ptx::load2(w32 + local * 4);
                let w1 = ptx::load2(w32 + (local + 8) * 4);
                let (w0a, w0b) = (1.0f32 + f32::from_bits(w0[0]), 1.0f32 + f32::from_bits(w0[1]));
                let (w1a, w1b) = (1.0f32 + f32::from_bits(w1[0]), 1.0f32 + f32::from_bits(w1[1]));
                let (hi0, lo0, hi2, lo2) = if tok0 < tokens {
                    let s0 = *sc_ptr.add((tok0 * 4 + stream) as usize);
                    let p0 = ptx::load2(a32 + (r0 * 16 + ((local / 4) ^ (r0 & 7))) * 16 + (local % 4) * 4);
                    let p1 = ptx::load2(a32 + (r0 * 16 + (((local + 8) / 4) ^ (r0 & 7))) * 16 + ((local + 8) % 4) * 4);
                    let (h0, l0) = crate::gemm::split2_pair(
                        f32::from_bits(p0[0]) * s0 * w0a,
                        f32::from_bits(p0[1]) * s0 * w0b,
                    );
                    let (h2, l2) = crate::gemm::split2_pair(
                        f32::from_bits(p1[0]) * s0 * w1a,
                        f32::from_bits(p1[1]) * s0 * w1b,
                    );
                    (h0, l0, h2, l2)
                } else {
                    (0u32, 0u32, 0u32, 0u32)
                };
                let r1 = r0 + 8;
                let tok1 = tok0 + 8;
                let (hi1, lo1, hi3, lo3) = if tok1 < tokens {
                    let s1 = *sc_ptr.add((tok1 * 4 + stream) as usize);
                    let p0 = ptx::load2(a32 + (r1 * 16 + ((local / 4) ^ (r1 & 7))) * 16 + (local % 4) * 4);
                    let p1 = ptx::load2(a32 + (r1 * 16 + (((local + 8) / 4) ^ (r1 & 7))) * 16 + ((local + 8) % 4) * 4);
                    let (h1, l1) = crate::gemm::split2_pair(
                        f32::from_bits(p0[0]) * s1 * w0a,
                        f32::from_bits(p0[1]) * s1 * w0b,
                    );
                    let (h3, l3) = crate::gemm::split2_pair(
                        f32::from_bits(p1[0]) * s1 * w1a,
                        f32::from_bits(p1[1]) * s1 * w1b,
                    );
                    (h1, l1, h3, l3)
                } else {
                    (0u32, 0u32, 0u32, 0u32)
                };
                let at = sh + i * 16;
                asm!(
                    "st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg32) at,
                    a = in(reg32) hi0, b = in(reg32) hi1, c = in(reg32) hi2, d = in(reg32) hi3,
                );
                asm!(
                    "st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg32) at + PLANE as u32,
                    a = in(reg32) lo0, b = in(reg32) lo1, c = in(reg32) lo2, d = in(reg32) lo3,
                );
            });
        }};
    }

    issue_stage!(0u32, 0u32);
    issue_stage!(ROWS_STAGE as u32, 4u32);

    let (mut left, mut right) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
    let mut current = 0u32;
    let mut part = 0u32;
    while part < 4 {
        let (mut high, mut low) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
        let mut t = part * 160;
        let t_end = (part + 1) * 160;
        while t < t_end {
            if t % 4 == 0 {
                let s = t / 4;
                if s == 0 {
                    asm!("cp.async.wait_group 1;");
                } else {
                    ptx::wait();
                }
                _syncthreads();
                if s > 0 && t + 4 < 640 {
                    issue_stage!(((s + 1) % 2) * ROWS_STAGE as u32, t + 4);
                }
                current = (s % 2) * ROWS_STAGE as u32;
                convert_stage!(current, t);
                _syncthreads();
            }
            let a = current + wm * 2048 + (t % 4) * 512 + lane * 16;
            let b = current + (2 * PLANE) as u32 + wn * 1024 + (t % 4) * 256 + lane * 8;
            let hi = [ptx::load4(a), ptx::load4(a + 2048)];
            let lo = [ptx::load4(a + PLANE as u32), ptx::load4(a + PLANE as u32 + 2048)];
            unroll!(4, j, {
                let w = ptx::load2(b + (j * 1024) as u32);
                unroll!(2, i, {
                    high[i][j] = canonical::madd(hi[i], w, high[i][j]);
                    low[i][j] = canonical::madd(lo[i], w, low[i][j]);
                });
            });
            t += 1;
        }
        unroll!(2, i, {
            unroll!(4, j, {
                let v = canonical::add(high[i][j], low[i][j]);
                match part {
                    0 => left[i][j] = v,
                    1 => left[i][j] = canonical::add(left[i][j], v),
                    2 => right[i][j] = v,
                    _ => right[i][j] = canonical::add(right[i][j], v),
                }
            });
        });
        part += 1;
    }

    let m = tokens as usize;
    let plane_out = m.div_ceil(16) * 20 * 128;
    let out1 = p.output as *mut u32;
    let out2 = p.output2 as *mut f32;
    let stride2 = p.stride2 as usize;
    let act = |x: f32| {
        let value = x * 0.25;
        value * crate::flash_next::hc_sigmoid(value)
    };
    unroll!(2, i, {
        unroll!(4, j, {
            let v = canonical::add(left[i][j], right[i][j]);
            let tok = (mt + wm + i as u32) as usize * 16 + lane as usize / 4;
            let col = (nt + wn + j as u32) as usize * 8 + (lane as usize % 4) * 2;
            if col < 320 {
                if tok < m {
                    let word = ((tok / 16 * 20 + col / 16) * 32 + tok % 8 * 4 + col % 8 / 2) * 4
                        + tok % 16 / 8
                        + col % 16 / 8 * 2;
                    let (hi, lo) = crate::gemm::split2_pair(act(v.a), act(v.b));
                    *out1.add(word) = hi;
                    *out1.add(plane_out + word) = lo;
                }
                if tok + 8 < m {
                    let t8 = tok + 8;
                    let word = ((t8 / 16 * 20 + col / 16) * 32 + t8 % 8 * 4 + col % 8 / 2) * 4
                        + t8 % 16 / 8
                        + col % 16 / 8 * 2;
                    let (hi, lo) = crate::gemm::split2_pair(act(v.c), act(v.d));
                    *out1.add(word) = hi;
                    *out1.add(plane_out + word) = lo;
                }
            } else {
                let c2 = col - 320;
                if tok < m {
                    if c2 < 4 { *out2.add(tok * stride2 + c2) = v.a; }
                    if c2 + 1 < 4 { *out2.add(tok * stride2 + c2 + 1) = v.b; }
                }
                if tok + 8 < m {
                    if c2 < 4 { *out2.add((tok + 8) * stride2 + c2) = v.c; }
                    if c2 + 1 < 4 { *out2.add((tok + 8) * stride2 + c2 + 1) = v.d; }
                }
            }
        });
    });
}
