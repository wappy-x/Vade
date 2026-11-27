//! Flash split 128x64x32 tile; dense wide tiles retain their original operands.
//! Three async stages retain the packed operands and canonical K16 chains.
use super::{Bf16Rows, ZERO, TERMS, canonical, add, madd, write, ptx};
use core::arch::nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x};
use crate::kernel_params::FLASH_BF16_WIDE_STAGE;

macro_rules! unroll {
    (2, $i:ident, $body:block) => {
        { #[allow(non_upper_case_globals)] const $i: usize = 0; $body } { #[allow(non_upper_case_globals)] const $i: usize = 1; $body }
    };
    (4, $i:ident, $body:block) => {
        unroll!(2, $i, $body);
        { #[allow(non_upper_case_globals)] const $i: usize = 2; $body } { #[allow(non_upper_case_globals)] const $i: usize = 3; $body }
    };
}

// One K32 stage: 8192 B per A plane, 4096 B of weights. Every
// 16-byte copy preserves the original lane-major fragment words.
macro_rules! stage {
    ($p:expr, $sh:expr, $mt:expr, $nt:expr, $k:expr, $tid:expr) => {{
        let (p, sh, mt, nt, k, tid): (Bf16Rows, u32, usize, usize, usize, usize) = ($p, $sh, $mt, $nt, $k, $tid);
        let kt = p.cols as usize / 16;
        let x = (p.input as *const u32).add((mt + tid / 64) * kt * 128 + k * 128 + tid % 64 * 4);
        let w = (p.weight as *const u32).add(canonical::weight_tile(nt + tid / 32, k, kt) * 64 + tid % 32 * 4);
        let dst = sh + (tid * 16) as u32;
        for plane in 0..TERMS {
            ptx::copy16(dst + (plane * 8192) as u32, x.add(plane * p.input_plane_words as usize));
            ptx::copy16(dst + (plane * 8192 + 4096) as u32,
                x.add(plane * p.input_plane_words as usize + kt * 512));
        }
        ptx::copy16(dst + (TERMS * 8192) as u32, w);
        ptx::commit();
    }};
}

/// Block256, grid(T/128,(N0+N1)/64), shared61440 at two planes. A four-rank K-split
/// shard runs its own two-range copy; four ranges run the code without ranges.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma128(p: Bf16Rows) {
    let (cols, ranges) = crate::kernel_params::bf16_cols_ranges(p.cols);
    if ranges == 2 { wide::<2>(Bf16Rows { cols, ..p }); } else { wide::<0>(p); }
}

#[inline(always)]
unsafe fn wide<const RANGES: u32>(mut p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 2 * 2, warp % 2 * 4);
    let mt_count = p.tokens as usize / 128;
    let nt_count = (p.rows + p.rows2) as usize / 64;
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    if p.tokens as usize * p.cols as usize * 2 * TERMS > 16 * 1024 * 1024 {
        let linear = ni * mt_count + mi;
        let group = if p.cols <= 4096 { 8 } else if p.cols <= 8192 { 4 } else { 2 };
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows; ni = within / rows;
    }
    let (mt, mut nt) = (mi * 8, ni * 8);
    if nt * 8 >= p.rows as usize {
        nt -= p.rows as usize / 8;
        p.weight = p.weight2; p.output = p.output2; p.rows = p.rows2; p.stride = p.stride2;
    }
    p.rows2 = 0;
    let kt = p.cols as usize / 16;
    let (mut left, mut right) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
    let mut sums = [[[ZERO; 4]; 2]; TERMS];
    let mut range = 0;
    stage!(p, 0, mt, nt, 0, tid);
    stage!(p, FLASH_BF16_WIDE_STAGE as u32, mt, nt, 2, tid);
    ptx::wait_one(); ptx::sync();
    let (mut k, mut current, mut next) = (0, 0, 2 * FLASH_BF16_WIDE_STAGE);
    while k < kt {
        let ahead = k + 4;
        if ahead < kt { stage!(p, next as u32, mt, nt, ahead, tid); }
        unroll!(2, z, {
            let base_a = (current + wm * 1024 + z * 512 + lane * 16) as u32;
            let base_b = (current + TERMS * 8192 + wn * 512 + z * 256 + lane * 8) as u32;
            let a = core::array::from_fn::<_, TERMS, _>(|plane| {
                let at = base_a + (plane * 8192) as u32;
                [ptx::load4(at), ptx::load4(at + 1024)]
            });
            unroll!(4, j, {
                let b = ptx::load2(base_b + (j * 512) as u32);
                unroll!(2, i, {
                    for plane in 0..TERMS { sums[plane][i][j] = madd(a[plane][i], b, sums[plane][i][j]); }
                });
            });
            // Five K16 instructions per K=320 range may cross a K32 stage.
            if k + z + 1 == canonical::end(kt, range, RANGES) {
                unroll!(2, i, { unroll!(4, j, {
                    let mut value = sums[0][i][j];
                    for plane in 1..TERMS { value = add(value, sums[plane][i][j]); }
                    match range {
                        0 => left[i][j] = value, 1 => left[i][j] = add(left[i][j], value),
                        2 => right[i][j] = value, _ => right[i][j] = add(right[i][j], value),
                    }
                    for plane in 0..TERMS { sums[plane][i][j] = ZERO; }
                }); });
                // Parts outside the shard add -0.0, so skipping them is exact.
                range = canonical::next(range, RANGES);
            }
        });
        k += 2;
        current += FLASH_BF16_WIDE_STAGE;
        if current == 3 * FLASH_BF16_WIDE_STAGE { current = 0; }
        next += FLASH_BF16_WIDE_STAGE;
        if next == 3 * FLASH_BF16_WIDE_STAGE { next = 0; }
        if ahead < kt { ptx::wait_one(); } else { ptx::wait(); }
        // Finish every reader before reusing its retired stage.
        ptx::sync();
    }
    unroll!(2, i, { unroll!(4, j, {
        write(p, add(left[i][j], right[i][j]), (mt + wm + i) * 16 + lane / 4,
            (nt + wn + j) * 8 + lane % 4 * 2);
    }); });
}
