//! Next-only fused FP32 input staging; canonical K16 arithmetic from the qualified dense fold.
use super::*;
#[path = "flash_prefill_mma.rs"]
mod tiled;
#[inline(always)]
unsafe fn stage_f32(p: Bf16Rows, dst: u32, plane_bytes: usize, token: usize, col: usize) {
    let load = |t: usize, c: usize| {
        if t < p.tokens as usize && c < p.cols as usize {
            pair((p.input as *const f32).add(t * p.cols as usize + c))
        } else {
            (0, 0)
        }
    };
    let a = [
        load(token, col),
        load(token + 8, col),
        load(token, col + 8),
        load(token + 8, col + 8),
    ];
    for (j, &(hi, lo)) in a.iter().enumerate() {
        ptx::store(dst + (j * 4) as u32, f32::from_bits(hi));
        ptx::store(dst + (plane_bytes + j * 4) as u32, f32::from_bits(lo));
    }
}
#[inline(always)]
unsafe fn stage(p: Bf16Rows, sh: u32, row: usize, token: usize, col: usize, tid: usize) {
    let k = p.cols as usize;
    let threads = 256;
    let x_offset = 64 * 128;
    for step in 0..2 {
        let i = tid + step * threads;
        let (group, tile, lane) = (i / 32, i % 32 / 8, i % 8 * 2);
        let (r, c) = (group * 4 + lane / 4, tile * 16 + lane % 4 * 2);
        let dst = sh + (i * 16) as u32;
        if row + r < (p.rows + p.rows2) as usize && col + c < k {
            let (w, _, local, _) = target(p, row + r);
            if packed(p, row + r) {
                ptx::copy16(
                    dst,
                    w.add(flash_bf16_element(
                        local,
                        col + c,
                        k,
                        matrix_rows(p, row + r),
                    ))
                    .cast(),
                );
            } else {
                let a = weight_fragment(p, row + r, col + c);
                let b = weight_fragment(p, row + r, col + c + 2);
                ptx::store(dst, f32::from_bits(a[0]));
                ptx::store(dst + 4, f32::from_bits(a[1]));
                ptx::store(dst + 8, f32::from_bits(b[0]));
                ptx::store(dst + 12, f32::from_bits(b[1]));
            }
        } else {
            for j in 0..4 {
                ptx::store(dst + j * 4, 0.0);
            }
        }
    }
    for step in 0..2 {
        let i = tid + step * threads;
        let (t, c) = (token / 16 + i / 128, col / 16 + i % 128 / 32);
        stage_f32(
            p,
            sh + (x_offset + i * 16) as u32,
            64 * 128,
            t * 16 + i % 32 / 4,
            c * 16 + i % 4 * 2,
        );
    }
    ptx::commit();
}
#[inline(always)]
unsafe fn project(p: Bf16Rows, block: usize) {
    let tid = _thread_idx_x() as usize;
    let (warp, lane) = (tid / 32, tid % 32);
    let (m, n) = (64, 64);
    let (wm, wn) = (warp / 4 * 32, warp % 4 * 16);
    let x_offset = n * 128;
    let stage_bytes = x_offset + TERMS * m * 128;
    let tiles = (p.tokens as usize).div_ceil(m);
    let (row, token) = (block / tiles * n, block % tiles * m);
    let sh = 0;
    // Uneven rank shards keep row-major weights and zero the final K tile.
    let kt = (p.cols as usize).div_ceil(16);
    let (mut left, mut right) = (TILE_ZERO, TILE_ZERO);
    let (mut sum, mut low, mut part) = (TILE_ZERO, TILE_ZERO, 0);
    stage(p, sh, row, token, 0, tid);
    ptx::wait();
    ptx::sync();
    let mut tile = 0;
    while tile < kt {
        let buffer = tile / 4 % 2;
        let current = sh + (buffer * stage_bytes) as u32;
        if tile + 4 < kt {
            stage(
                p,
                sh + ((1 - buffer) * stage_bytes) as u32,
                row,
                token,
                (tile + 4) * 16,
                tid,
            );
        }
        for z in 0..4 {
            if tile + z < kt {
                let a = current + (x_offset + (wm / 16 * 4 + z) * 512 + lane * 16) as u32;
                let b = current + (((wn / 4 + lane / 16) * 4 + z) * 16 + lane % 16) as u32 * 8;
                let (a0, a0_lo, b0) = (
                    a_fragment(a),
                    a_fragment(a + (m * 128) as u32),
                    b_fragment(b),
                );
                sum.a = madd(a0, b0, sum.a);
                low.a = madd(a0_lo, b0, low.a);
                {
                    let (a1, a1_lo, b1) = (
                        a_fragment(a + 2048),
                        a_fragment(a + 2048 + (m * 128) as u32),
                        b_fragment(b + 1024),
                    );
                    sum.b = madd(a0, b1, sum.b);
                    low.b = madd(a0_lo, b1, low.b);
                    sum.c = madd(a1, b0, sum.c);
                    low.c = madd(a1_lo, b0, low.c);
                    sum.d = madd(a1, b1, sum.d);
                    low.d = madd(a1_lo, b1, low.d);
                }
                if tile + z + 1 == canonical::range(kt, part).1 {
                    let partial = plus(sum, low);
                    match part {
                        0 => left = partial,
                        1 => left = plus(left, partial),
                        2 => right = partial,
                        _ => right = plus(right, partial),
                    }
                    sum = TILE_ZERO;
                    low = TILE_ZERO;
                    part += 1;
                }
            }
        }
        ptx::wait();
        ptx::sync();
        tile += 4;
    }
    let result = plus(left, right);
    let (t, r) = (token + wm + lane / 4, row + wn + lane % 4 * 2);
    write(p, result.a, t, r);
    {
        write(p, result.b, t, r + 8);
        write(p, result.c, t + 16, r);
        write(p, result.d, t + 16, r + 8);
    }
}
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_pairs(p: crate::kernel_params::Bf16Pairs) {
    let first = crate::kernel_params::flash_bf16_prefill_blocks(p.pairs[0]);
    let block = _block_idx_x() as usize;
    let (p, block) = if block < first {
        (p.pairs[0], block)
    } else {
        (p.pairs[1], block - first)
    };
    if block >= crate::kernel_params::flash_bf16_prefill_blocks(p) {
        return;
    }
    project(p, block);
}
