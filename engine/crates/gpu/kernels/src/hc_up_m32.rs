//! Standalone M=17..32 LBF12 HC-up latency kernel: reads each LBF12 K32 weight slot once per CTA
//! across both 16-token tiles, mixes all four streams in stream order, and optionally emits both
//! the next projections' dense hi/lo planes (`planes != 0`) and RN BF16 `mixed16` (`p.output2 != 0`).
//! Kept in its own module so flash_bf16.rs codegen stays untouched.
#![cfg(target_arch = "nvptx64")]

use crate::{
    canonical::{self, Acc, ZERO},
    kernel_params::{flash_bf16_element, Bf16Rows},
    ptx,
};
use core::arch::nvptx::{_block_idx_x, _syncthreads, _thread_idx_x};

#[repr(C, align(16))]
struct Hc32Fragment([u32; 4]);

macro_rules! hc32_decode {
    ($slot:expr, $base:expr) => {{
        let w = ($slot) as *const u32;
        let base: u32 = $base;
        let lo = *w as u64 | (*w.add(1) as u64) << 32;
        let hi = *w.add(2) as u64;
        let f = |j: u32| -> u32 {
            (if j < 5 { lo >> (12 * j) } else if j == 5 { lo >> 60 | hi << 4 } else { hi >> (12 * j - 64) }) as u32 & 0xFFF
        };
        let v = |x: u32| -> u32 {
            let off = (x >> 7) & 15;
            (x >> 11) << 15 | if off == 15 { 0 } else { (base + off) << 7 | (x & 0x7F) }
        };
        [v(f(0)) | v(f(1)) << 16, v(f(2)) | v(f(3)) << 16, v(f(4)) | v(f(5)) << 16, v(f(6)) | v(f(7)) << 16]
    }};
}

macro_rules! hc32_store_planes {
    ($planes:expr, $rows:expr, $t:expr, $f:expr, $a:expr, $b:expr) => {{
        let (planes, rows, t, f): (u64, usize, usize, usize) = ($planes, $rows, $t, $f);
        let kt = 2560usize / 16;
        let plane = rows.div_ceil(16) * kt * 128;
        let word = ((t / 16 * kt + f / 16) * 32 + t % 8 * 4 + f % 8 / 2) * 4 + t % 16 / 8 + f % 16 / 8 * 2;
        let (hi, lo) = crate::gemm::split2_pair($a, $b);
        let p = planes as *mut u32;
        *p.add(word) = hi;
        *p.add(plane + word) = lo;
        if t % 16 < 8 && t + 8 >= rows {
            *p.add(word + 1) = 0;
            *p.add(plane + word + 1) = 0;
        }
    }};
}

/// M=17..32 LBF12 HC-up: reads the 10 K32 weight slots once per CTA and evaluates both M16 tiles
/// sequentially with the exact canonical 5-tile K16 hi/lo chains and (p0+p1)+(p2+p3) merge.
/// # Safety
/// 17 <= M <= 32, K = 320, H = 2560, grid (320, 1), block 128; twin = flash_bf16_lbf12_twin of
/// the up weight; planes = 0 or the two dense planes of M x 2560; p.output2 = 0 or M x 2560 BF16.
// SAFETY: Hc::mix_chunk (blocks.rs) checks 17 <= tokens <= 32, lbf12_of(self.up), dense planes and
// optional mixed16 destination, launching grid (320, 1) x 128 threads.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_latency32_lbf12(
    p: Bf16Rows,
    normalized: u64,
    twin: u64,
    planes: u64,
) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (h, k) = (2560usize, 320usize);
    let feature = _block_idx_x() as usize * 8;
    let row = warp * h + feature;
    let (pairs, slots) = (k / 32, 4 * h / 8 * (k / 32));
    let slot0 = row / 8 * pairs;
    let data = twin as *const u8;
    let flags = *(data.add(slots * 392 + row / 8 * 4) as *const u32);
    let load = |j: usize| -> [u32; 4] {
        if flags >> j & 1 != 0 {
            hc32_decode!(
                data.add((slot0 + j) * 384 + lane * 12),
                *data.add(slots * 384 + (slot0 + j) * 8 + lane / 4) as u32
            )
        } else {
            let element = |t: usize| (p.weight as *const u16).add(flash_bf16_element(row + lane / 4, t * 16 + lane % 4 * 2, k, 4 * h));
            let (a, b): (u64, u64);
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v = out(reg64) a, p = in(reg64) element(2 * j));
            core::arch::asm!("ld.global.u64 {v}, [{p}];", v = out(reg64) b, p = in(reg64) element(2 * j + 1));
            [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32]
        }
    };
    let v = [load(0), load(1), load(2), load(3), load(4), load(5), load(6), load(7), load(8), load(9)];
    let frag = |t: usize| -> [u32; 2] {
        let w = v[t / 2];
        if t % 2 == 0 { [w[0], w[1]] } else { [w[2], w[3]] }
    };
    let x = p.input as *const u32;
    let sh = ptx::shared::<2048>();
    let rows = p.tokens as usize;
    let f = feature + lane % 4 * 2;
    let sig = |val: f32| crate::flash_next::hc_sigmoid(val);

    let mut mt = 0usize;
    while mt < 2 {
        let (mut left, mut right) = (ZERO, ZERO);
        let mut part = 0usize;
        while part < 4 {
            let (mut hi, mut lo) = (ZERO, ZERO);
            let mut i = 0usize;
            while i < 5 {
                let kt = part * 5 + i;
                let at = ((mt * 20 + kt) * 32 + lane) * 4;
                let a = (*(x.add(at).cast::<Hc32Fragment>())).0;
                let low = (*(x.add(at + p.input_plane_words as usize).cast::<Hc32Fragment>())).0;
                let b = frag(kt);
                hi = canonical::madd(a, b, hi);
                lo = canonical::madd(low, b, lo);
                i += 1;
            }
            let z = canonical::add(hi, lo);
            match part {
                0 => left = z,
                1 => left = canonical::add(left, z),
                2 => right = z,
                _ => right = canonical::add(right, z),
            }
            part += 1;
        }
        let gate = canonical::add(left, right);
        if warp > 0 {
            let at = sh + (((warp - 1) * 32 + lane) * 16) as u32;
            ptx::store(at, gate.a); ptx::store(at + 4, gate.b);
            ptx::store(at + 8, gate.c); ptx::store(at + 12, gate.d);
        }
        _syncthreads();
        if warp == 0 {
            let t = mt * 16 + lane / 4;
            let mut mixed = ZERO;
            let mut stream = 0usize;
            while stream < 4 {
                let g = if stream == 0 {
                    gate
                } else {
                    let at = sh + (((stream - 1) * 32 + lane) * 16) as u32;
                    Acc { a: ptx::load(at), b: ptx::load(at + 4), c: ptx::load(at + 8), d: ptx::load(at + 12) }
                };
                if t < rows {
                    let src = (normalized as *const f32).add(t * 4 * h + stream * h + f);
                    mixed.a = crate::next_math::hc_stream(mixed.a, *src, g.a, sig);
                    mixed.b = crate::next_math::hc_stream(mixed.b, *src.add(1), g.b, sig);
                }
                if t + 8 < rows {
                    let src = (normalized as *const f32).add((t + 8) * 4 * h + stream * h + f);
                    mixed.c = crate::next_math::hc_stream(mixed.c, *src, g.c, sig);
                    mixed.d = crate::next_math::hc_stream(mixed.d, *src.add(1), g.d, sig);
                }
                stream += 1;
            }
            if t < rows {
                let (a, b) = (mixed.a * 0.25, mixed.b * 0.25);
                *(p.output as *mut f32).add(t * h + f) = a;
                *(p.output as *mut f32).add(t * h + f + 1) = b;
                if planes != 0 { hc32_store_planes!(planes, rows, t, f, a, b); }
                if p.output2 != 0 { *(p.output2 as *mut u32).add(t * (h / 2) + f / 2) = crate::gemm::bf16_pair(a, b); }
            }
            if t + 8 < rows {
                let (a, b) = (mixed.c * 0.25, mixed.d * 0.25);
                *(p.output as *mut f32).add((t + 8) * h + f) = a;
                *(p.output as *mut f32).add((t + 8) * h + f + 1) = b;
                if planes != 0 { hc32_store_planes!(planes, rows, t + 8, f, a, b); }
                if p.output2 != 0 { *(p.output2 as *mut u32).add((t + 8) * (h / 2) + f / 2) = crate::gemm::bf16_pair(a, b); }
            }
        }
        _syncthreads();
        mt += 1;
    }
}
