//! Packed Flash-Next 64x128x64 GEMM. Three cp.async stages, no split-K.
//! Conversion runs once per input group; every output keeps the old K16 graph.
use super::{add, canonical, madd, ptx, write, Acc, Bf16Rows, ZERO};
use core::arch::{
    asm,
    nvptx::{_block_idx_x, _block_idx_y, _thread_idx_x},
};
const PLANE: usize = 8192;
const STAGE: usize = 32768;
/// K16 tile of each stage at which the three-stage ring refills (1..=3).
const REFILL: usize = 1;
/// Dynamic shared bytes of flash_bf16_mma_prefill_tiled (three stages); hosts read it.
#[no_mangle]
pub static FLASH_BF16_MMA_PREFILL_TILED_SHARED: u32 = crate::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32;

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

// Zero-fill edge fragments without constructing an out-of-allocation pointer.
#[inline(always)]
unsafe fn copy(dst: u32, src: *const u32, valid: bool) {
    asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
        d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if valid {16u32} else {0u32});
}
#[inline(always)]
unsafe fn copy4(dst: u32, src: *const u32, valid: bool) {
    asm!("cp.async.ca.shared.global [{d}], [{s}], 4, {n};",
        d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if valid {4u32} else {0u32});
}
/// Stage mbarriers after the three stages: full[b] at BARS + 8b, empty[b] at BARS + 24 + 8b.
const BARS: usize = 3 * STAGE;
#[inline(always)]
unsafe fn bar_init(bar: u32, count: u32) {
    asm!("mbarrier.init.shared::cta.b64 [{b}], {c};", b = in(reg32) bar, c = in(reg32) count);
}
/// Arrive on `bar` once every cp.async this thread issued so far has landed.
#[inline(always)]
unsafe fn bar_copies(bar: u32) {
    asm!("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [{b}];", b = in(reg32) bar);
}
#[inline(always)]
unsafe fn bar_arrive(bar: u32) {
    asm!("{{ .reg .b64 state; mbarrier.arrive.shared::cta.b64 state, [{b}]; }}", b = in(reg32) bar);
}
#[inline(always)]
unsafe fn bar_try(bar: u32, parity: u32) -> bool {
    let done: u32;
    asm!("{{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [{b}], {ph}; selp.u32 {d}, 1, 0, p; }}",
        b = in(reg32) bar, ph = in(reg32) parity, d = out(reg32) done);
    done != 0
}
#[inline(always)]
unsafe fn bar_wait(bar: u32, parity: u32) {
    // Bounded in time: a phase still open 2 s after the first failed poll traps (clean
    // launch failure, no GPU wedge); a poll count is no bound because try_wait may suspend.
    // The low timer word wraps every 4.29 s, so the wrapping difference measures 2 s.
    if !bar_try(bar, parity) {
        let start: u32;
        asm!("mov.u32 {t}, %globaltimer_lo;", t = out(reg32) start);
        while !bar_try(bar, parity) {
            let now: u32;
            asm!("mov.u32 {t}, %globaltimer_lo;", t = out(reg32) now);
            if now.wrapping_sub(start) > 2_000_000_000 {
                asm!("trap;");
            }
        }
    }
}
/// ROWS: the caller copies A as FP32 rows (tile_hc), so only B is staged here.
#[inline(always)]
unsafe fn stage<const TAIL: bool, const ROWS: bool>(p: Bf16Rows, sh: u32, mt: usize, nt: usize, k: usize, tid: usize) {
    let kt = p.cols as usize / 16;
    if !ROWS {
        unroll!(2, step, {
            let i = tid + step * 256;
            let row = mt + i / 128;
            let valid = row * 16 < p.tokens as usize;
            let at = ((row * kt + k + i % 128 / 32) * 32 + i % 32) * 4;
            unroll!(2, plane, {
                let x = (p.input as *const u32).add(if valid {
                    at + plane * p.input_plane_words as usize
                } else {
                    0
                });
                copy(sh + (plane * PLANE + i * 16) as u32, x, valid);
            });
        });
    }
    unroll!(4, step, {
        let i = tid + step * 256;
        let row = nt + i / 64;
        let dst = sh + (2 * PLANE + i * 16) as u32;
        // Only the last N tiles reach past the first matrix (TAIL CTAs).
        if !TAIL || row * 8 < p.rows as usize {
            let at = canonical::weight_tile(row, k, kt) * 64 + i % 64 * 4;
            copy(dst, (p.weight as *const u32).add(at), true);
        } else if p.rows2 % 4 == 0 {
            // Optional second matrix (rows % 8 == 0 before it; under 64 rows): its packed
            // four-row K16 tiles (32 words) are the low and high halves of an 8-row tile.
            let (z, half, part) = (i % 64 / 16, i % 16 / 8, i % 8);
            let quad = (row * 8 - p.rows as usize) / 4 + half;
            let valid = quad * 4 < p.rows2 as usize;
            let at = if valid { (quad * kt + k + z) * 32 + part * 4 } else { 0 };
            copy(dst, (p.weight2 as *const u32).add(at), valid);
        } else {
            // Row-major second matrix (MoE shared gate, one row): each fragment lane holds
            // the (c, c+1) and (c+8, c+9) words of its row, copied four bytes at a time.
            let z = i % 64 / 16;
            unroll!(2, h, {
                let lane = i % 16 * 2 + h;
                let local = row * 8 - p.rows as usize + lane / 4;
                let valid = local < p.rows2 as usize;
                let at = if valid { (local * p.cols as usize + (k + z) * 16) / 2 + lane % 4 } else { 0 };
                unroll!(2, w, {
                    copy4(dst + (h * 8 + w * 4) as u32, (p.weight2 as *const u32).add(at + if valid { w * 4 } else { 0 }), valid);
                });
            });
        }
    });
    ptx::commit();
}

/// One thread's copy plan for a tile: every source word index is affine in the K16 offset,
/// so a stage adds k times a fixed step to bases computed once per tile.
#[derive(Clone, Copy)]
struct Plan {
    /// A chunk sources (hi plane) at k = 0, or None past the M tail.
    a: [Option<usize>; 2],
    /// B chunk sources at k = 0 with their per-K16 step in words: 512 (blocked first matrix),
    /// 32 (packed four-row second matrix) or 8 (row-major second matrix, four-byte copies).
    b: [Option<(usize, usize)>; 4],
}
#[inline(always)]
fn plan(p: Bf16Rows, mt: usize, groups: impl Fn(usize) -> usize, tid: usize) -> Plan {
    let kt = p.cols as usize / 16;
    let mut a = [None; 2];
    unroll!(2, step, {
        let i = tid + step * 256;
        let row = mt + i / 128;
        if row * 16 < p.tokens as usize {
            a[step] = Some(((row * kt + i % 128 / 32) * 32 + i % 32) * 4);
        }
    });
    let mut b = [None; 4];
    unroll!(4, step, {
        let i = tid + step * 256;
        // B rows of local 8-row group g are global 8-row group groups(g).
        let (row, c) = (groups(i / 64), i % 64);
        b[step] = if row * 8 < p.rows as usize {
            Some((canonical::weight_tile(row, 0, kt) * 64 + c * 4, 512))
        } else if p.rows2 % 4 == 0 {
            // Optional second matrix (rows % 8 == 0 before it; under 64 rows): its packed
            // four-row K16 tiles (32 words) are the low and high halves of an 8-row tile.
            let quad = (row * 8 - p.rows as usize) / 4 + c % 16 / 8;
            if quad * 4 < p.rows2 as usize { Some(((quad * kt + c / 16) * 32 + c % 8 * 4, 32)) } else { None }
        } else {
            // Row-major second matrix (MoE shared gate): fragment lanes 2(c%16), 2(c%16)+1 of
            // one row hold the (c, c+1) and (c+8, c+9) words, copied four bytes at a time.
            let local = row * 8 - p.rows as usize + c % 16 / 2;
            if local < p.rows2 as usize { Some(((local * p.cols as usize + c / 16 * 16) / 2 + c % 2 * 2, 8)) } else { None }
        };
    });
    Plan { a, b }
}
/// ROWS: as in `stage`, B only.
#[inline(always)]
unsafe fn stage_plan<const ROWS: bool>(p: Bf16Rows, plan: &Plan, sh: u32, k: usize, tid: usize) {
    let (input, plane) = (p.input as *const u32, p.input_plane_words as usize);
    if !ROWS {
        unroll!(2, step, {
            let i = tid + step * 256;
            let at = plan.a[step].map_or(0, |a| a + k * 128);
            unroll!(2, half, {
                copy(sh + (half * PLANE + i * 16) as u32, input.add(at + half * plane * usize::from(plan.a[step].is_some())), plan.a[step].is_some());
            });
        });
    }
    unroll!(4, step, {
        let i = tid + step * 256;
        let dst = sh + (2 * PLANE + i * 16) as u32;
        match plan.b[step] {
            Some((at, 512)) => copy(dst, (p.weight as *const u32).add(at + k * 512), true),
            Some((at, 32)) => copy(dst, (p.weight2 as *const u32).add(at + k * 32), true),
            Some((at, _)) => {
                let w = (p.weight2 as *const u32).add(at + k * 8);
                unroll!(2, h, {
                    unroll!(2, x, {
                        copy4(dst + (h * 8 + x * 4) as u32, w.add(h + x * 4), true);
                    });
                });
            }
            None => copy(dst, p.weight as *const u32, false),
        }
    });
    ptx::commit();
}

/// One HC up stage, inline addressing: the 64-token A tile (both planes) and the tile's 16
/// weight 8-row groups (group g = stream g / 4, features 32ni + 8(g % 4)).
#[inline(always)]
unsafe fn stage_hc_up(p: Bf16Rows, sh: u32, mt: usize, ni: usize, k: usize, tid: usize) {
    let (h, kt) = (2560usize, p.cols as usize / 16);
    let copy = |dst: u32, src: *const u32, valid: bool| {
        asm!("cp.async.cg.shared.global [{d}], [{s}], 16, {n};",
            d = in(reg32) dst, s = in(reg64) src, n = in(reg32) if valid { 16u32 } else { 0u32 });
    };
    unroll!(2, step, {
        let i = tid + step * 256;
        let row = mt + i / 128;
        let valid = row * 16 < p.tokens as usize;
        let at = ((row * kt + k + i % 128 / 32) * 32 + i % 32) * 4;
        unroll!(2, plane, {
            let x = (p.input as *const u32).add(if valid { at + plane * p.input_plane_words as usize } else { 0 });
            copy(sh + (plane * PLANE + i * 16) as u32, x, valid);
        });
    });
    unroll!(4, step, {
        let i = tid + step * 256;
        let g = i / 64;
        let at = canonical::weight_tile(g / 4 * (h / 8) + ni * 4 + g % 4, k, kt) * 64 + i % 64 * 4;
        copy(sh + (2 * PLANE + i * 16) as u32, (p.weight as *const u32).add(at), true);
    });
    ptx::commit();
}

/// One projection, block256, grid ceil(M/64),ceil((N+N2)/128), dynamic shared98304.
/// Requires resident blocked64 B, K%64=0, N%64=0; masks M and N tails. An optional
/// second packed matrix of N2 < 64 rows (N2 % 4 == 0) continues at row N; each output
/// keeps its own K16 graph, so its bytes equal its solo launch. A row-major second
/// matrix (N2 % 4 != 0, the layout flash_bf16_packed rejects) is staged four bytes at a time.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_tiled(p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let mt_count = (p.tokens as usize).div_ceil(64);
    let nt_count = ((p.rows + p.rows2) as usize).div_ceil(128);
    if p.tokens as usize * p.cols as usize * 4 > 16 * 1024 * 1024 {
        // Retain the previous wide tile's bounded input reuse window in L2.
        let group = if p.cols <= 4096 {
            8
        } else if p.cols <= 8192 {
            8
        } else {
            4
        };
        let linear = ni * mt_count + mi;
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows;
        ni = within / rows;
    }
    let (mt, nt) = (mi * 4, ni * 16);
    if (nt + 16) * 8 > p.rows as usize {
        tile::<true>(p, mt, nt, tid);
    } else {
        tile::<false>(p, mt, nt, tid);
    }
}
#[inline(always)]
unsafe fn tile<const TAIL: bool>(p: Bf16Rows, mt: usize, nt: usize, tid: usize) {
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    // Tail tiles stage from a copy plan computed once (second-matrix sources); the
    // others keep the inline first-matrix addressing.
    let tail = if TAIL { Some(plan(p, mt, |g| nt + g, tid)) } else { None };
    let issue = |sh: u32, k: usize| match &tail {
        Some(plan) => stage_plan::<false>(p, plan, sh, k, tid),
        None => stage::<false, false>(p, sh, mt, nt, k, tid),
    };
    // A tail tile with one live N8 tile (the MoE shared gate's row): only its warps' first
    // fragments compute, without a per-fragment branch in the K16 loop.
    let out = if TAIL && (p.rows + p.rows2) as usize <= nt * 8 + 8 {
        product_j::<3, 1>(p, tid, wn == 0, issue)
    } else {
        product::<3>(p, tid, issue)
    };
    unroll!(2, i, {
        unroll!(4, j, {
            write_pair(
                p,
                out[i][j],
                (mt + wm + i) * 16 + lane / 4,
                (nt + wn + j) * 8 + lane % 4 * 2,
            );
        });
    });
}
/// write() with each token's two adjacent outputs (rows row, row + 1; row is even) as one 8-byte
/// store when both rows are the first matrix's and the address is 8-byte aligned: a lane quad
/// fills a whole 32-byte sector of the token's row. Same values; otherwise write().
#[inline(always)]
unsafe fn write_pair(p: Bf16Rows, value: Acc, token: usize, row: usize) {
    let y = p.output as *mut f32;
    if row + 1 < p.rows as usize && p.stride % 2 == 0 && (y as usize) % 8 == 0 {
        let stride = p.stride as usize;
        if token < p.tokens as usize {
            asm!("st.global.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) y.add(token * stride + row), a = in(reg32) value.a, b = in(reg32) value.b);
        }
        if token + 8 < p.tokens as usize {
            asm!("st.global.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) y.add((token + 8) * stride + row), a = in(reg32) value.c, b = in(reg32) value.d);
        }
    } else {
        write(p, value, token, row);
    }
}
/// Prompt HC down: flash_bf16_mma_prefill_tiled whose first-matrix outputs (the 320 low
/// features) leave as flash_next_hc_activate's SiLU(x/4), written only as the HC up operand's
/// dense hi/lo planes at p.output (K = rows, flash_bf16_input's layout; rows past M are left
/// unwritten, so M % 16 == 0); the second matrix (HC injection) stays FP32 rows.
/// norm != 0: p.input is the FP32 residual [M][cols] and A is formed in shared memory, not read
/// from the norm's packed planes: `scales` = the norm's scales [M][4] (flash_next_hc_add_norm
/// with scales != 0), `norm` = its F32 weights [cols]. Each A word is the norm's: residual *
/// scale * (1 + weight) in that order, then split2_pair, so every output keeps its K16 graph.
/// Dynamic shared FLASH_BF16_PREFILL_ROWS_SHARED (it covers both forms).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_tiled_hc(p: Bf16Rows, scales: u64, norm: u64) {
    let tid = _thread_idx_x() as usize;
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let mt_count = (p.tokens as usize).div_ceil(64);
    let nt_count = ((p.rows + p.rows2) as usize).div_ceil(128);
    if p.tokens as usize * p.cols as usize * 4 > 16 * 1024 * 1024 {
        // Retain the previous wide tile's bounded input reuse window in L2.
        let group = if p.cols <= 4096 {
            8
        } else if p.cols <= 8192 {
            8
        } else {
            4
        };
        let linear = ni * mt_count + mi;
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows;
        ni = within / rows;
    }
    let (mt, nt) = (mi * 4, ni * 16);
    let tail = (nt + 16) * 8 > p.rows as usize;
    if norm != 0 {
        if tail {
            tile_hc::<true, true>(p, scales, norm, mt, nt, tid);
        } else {
            tile_hc::<false, true>(p, scales, norm, mt, nt, tid);
        }
    } else if tail {
        tile_hc::<true, false>(p, scales, norm, mt, nt, tid);
    } else {
        tile_hc::<false, false>(p, scales, norm, mt, nt, tid);
    }
}
/// One ROWS stage of tile_hc: the A planes and B as in `stage`, then the FP32 A tile
/// [64 tokens][64 K] (16-byte chunks XOR-swizzled by token % 8), then the stage's 64 norm weights.
const ROWS_A: usize = STAGE;
const ROWS_W: usize = STAGE + 16384;
const ROWS_STAGE: usize = ROWS_W + 256;
const _: () = assert!(2 * ROWS_STAGE == crate::kernel_params::FLASH_BF16_PREFILL_ROWS_SHARED);
#[inline(always)]
unsafe fn tile_hc<const TAIL: bool, const ROWS: bool>(p: Bf16Rows, scales: u64, norm: u64, mt: usize, nt: usize, tid: usize) {
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    // Tail tiles stage from a copy plan computed once (second-matrix sources); the
    // others keep the inline first-matrix addressing.
    let tail = if TAIL { Some(plan(p, mt, |g| nt + g, tid)) } else { None };
    let issue = |sh: u32, k: usize| {
        if ROWS {
            // Token mt * 16 + r, 16-byte chunk c of the stage's 64 K at chunk c ^ (r % 8) of row
            // r (zero-filled past the tokens); then the 64 norm weights of those K, four bytes
            // each (the weights need no 16-byte alignment).
            unroll!(4, step, {
                let i = tid + step * 256;
                let (r, c) = (i / 16, i % 16);
                let token = mt * 16 + r;
                let valid = token < p.tokens as usize;
                let at = if valid { token * p.cols as usize + k * 16 + c * 4 } else { 0 };
                copy(sh + (ROWS_A + (r * 16 + (c ^ (r % 8))) * 16) as u32, (p.input as *const u32).add(at), valid);
            });
            if tid < 64 {
                copy4(sh + (ROWS_W + tid * 4) as u32, (norm as *const u32).add(k * 16 + tid), true);
            }
        }
        match &tail {
            Some(plan) => stage_plan::<ROWS>(p, plan, sh, k, tid),
            None => stage::<false, ROWS>(p, sh, mt, nt, k, tid),
        }
    };
    // ROWS: the stage's A planes from its FP32 tile, word for word as flash_next_hc_add_norm
    // packs them: chunk i = (M16 tile i / 128, K16 tile i % 128 / 32, lane i % 32) holds words
    // (t; c, c+1), (t + 8; c, c+1), (t; c+8, c+9), (t + 8; c+8, c+9) of both planes.
    let convert = |sh: u32, k: usize| {
        let group = p.cols as usize / 4;
        let (a32, w32) = (sh + ROWS_A as u32, sh + ROWS_W as u32);
        unroll!(2, step, {
            let i = tid + step * 256;
            let (m, kk, g, q) = (i / 128, i % 128 / 32, i % 32 / 4, i % 4);
            let r = m * 16 + g;
            let token = mt * 16 + r;
            let (mut hi, mut lo) = ([0u32; 4], [0u32; 4]);
            // M % 16 == 0: row t + 8 is live with row t.
            if token < p.tokens as usize {
                let (local, c) = (kk * 16 + q * 2, (k + kk) * 16 + q * 2);
                let stream = c / group;
                let (w0, w1) = (ptx::load2(w32 + (local * 4) as u32), ptx::load2(w32 + ((local + 8) * 4) as u32));
                let weight = [w0[0], w0[1], w1[0], w1[1]].map(f32::from_bits);
                let mut a = [[0.0f32; 4]; 2];
                unroll!(2, half, {
                    let row = r + half * 8;
                    let scale = *(scales as *const f32).add((token + half * 8) * 4 + stream);
                    let at = |x: usize| a32 + ((row * 16 + ((x / 4) ^ (row % 8))) * 16 + x % 4 * 4) as u32;
                    let v = [ptx::load2(at(local)), ptx::load2(at(local + 8))];
                    let v = [f32::from_bits(v[0][0]), f32::from_bits(v[0][1]), f32::from_bits(v[1][0]), f32::from_bits(v[1][1])];
                    let mut j = 0;
                    while j < 4 {
                        a[half][j] = v[j] * scale * (1.0 + weight[j]);
                        j += 1;
                    }
                });
                (hi[0], lo[0]) = crate::gemm::split2_pair(a[0][0], a[0][1]);
                (hi[1], lo[1]) = crate::gemm::split2_pair(a[1][0], a[1][1]);
                (hi[2], lo[2]) = crate::gemm::split2_pair(a[0][2], a[0][3]);
                (hi[3], lo[3]) = crate::gemm::split2_pair(a[1][2], a[1][3]);
            }
            let at = sh + (i * 16) as u32;
            asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) at,
                a = in(reg32) hi[0], b = in(reg32) hi[1], c = in(reg32) hi[2], d = in(reg32) hi[3]);
            asm!("st.shared.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};", p = in(reg32) at + PLANE as u32,
                a = in(reg32) lo[0], b = in(reg32) lo[1], c = in(reg32) lo[2], d = in(reg32) lo[3]);
        });
    };
    let out = if ROWS {
        // product_hc's two stages, both issued first (an empty group when K has one stage).
        issue(0, 0);
        if p.cols as usize / 16 > 4 {
            issue(ROWS_STAGE as u32, 4);
        } else {
            ptx::commit();
        }
        product_hc::<true>(p, tid, issue, convert)
    } else {
        product::<3>(p, tid, issue)
    };
    unroll!(2, i, {
        unroll!(4, j, {
            let (token, col) = ((mt + wm + i) * 16 + lane / 4, (nt + wn + j) * 8 + lane % 4 * 2);
            // rows % 8 == 0: an N8 fragment is all first matrix or all second.
            if col < p.rows as usize {
                hc_planes(p, out[i][j], token, col);
            } else {
                write(p, out[i][j], token, col);
            }
        });
    });
}
/// A fragment's two row pairs as flash_next_hc_activate leaves them ((x / 4) * sigmoid(x / 4)),
/// split2 into the dense A-fragment words of the next GEMM's operand (K = p.rows, M = tokens).
#[inline(always)]
unsafe fn hc_planes(p: Bf16Rows, v: Acc, token: usize, col: usize) {
    let act = |x: f32| {
        let value = x * 0.25;
        value * crate::flash_next::hc_sigmoid(value)
    };
    let kt = p.rows as usize / 16;
    let plane = (p.tokens as usize).div_ceil(16) * kt * 128;
    let put = |t: usize, a: f32, b: f32| {
        if t < p.tokens as usize {
            let word = ((t / 16 * kt + col / 16) * 32 + t % 8 * 4 + col % 8 / 2) * 4 + t % 16 / 8 + col % 16 / 8 * 2;
            let (hi, lo) = crate::gemm::split2_pair(act(a), act(b));
            *(p.output as *mut u32).add(word) = hi;
            *(p.output as *mut u32).add(plane + word) = lo;
        }
    };
    put(token, v.a, v.b);
    put(token + 8, v.c, v.d);
}
/// The tile's canonical outputs (p0+p1)+(p2+p3) for this thread's 2 x 4 MMA fragments: warp
/// (wm, wn) = (warp / 4 * 2, warp % 4 * 4) holds M16 tiles wm..wm+2 and N8 tiles wn..wn+4.
/// `issue(sh, k)` stages K16 tiles k..k+4 into the buffer at sh.
#[inline(always)]
unsafe fn product<const STAGES: usize>(p: Bf16Rows, tid: usize, issue: impl Fn(u32, usize)) -> [[Acc; 4]; 2] {
    product_j::<STAGES, 4>(p, tid, true, issue)
}
/// product computing only N8 tiles wn..wn+J of this warp, and none unless `active` (warp
/// uniform; J = 4 and active is product). The others stay ZERO; the stage copies and barrier
/// arrivals are every thread's as in product, and each computed fragment keeps its K16 graph.
#[inline(always)]
unsafe fn product_j<const STAGES: usize, const J: usize>(p: Bf16Rows, tid: usize, active: bool, issue: impl Fn(u32, usize)) -> [[Acc; 4]; 2] {
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    let kt = p.cols as usize / 16;
    let (mut buffer, mut current) = (0, 0);
    if STAGES == 3 {
        // Stage s lives in buffer s % 3: full[s % 3] completes its phase (s / 3) & 1 when
        // all 256 threads' copies of it landed; empty[s % 3] when all 256 finished reading it.
        if tid == 0 {
            unroll!(2, e, {
                bar_init((BARS + e * 24) as u32, 256);
                bar_init((BARS + e * 24 + 8) as u32, 256);
                bar_init((BARS + e * 24 + 16) as u32, 256);
            });
        }
        ptx::sync();
    }
    issue(0, 0);
    if STAGES == 3 {
        bar_copies(BARS as u32);
    }
    if STAGES == 3 && 4 < kt {
        issue(STAGE as u32, 4);
        bar_copies((BARS + 8) as u32);
    }
    // One K16 chain pair (hi, lo) over K16 tiles t0..t1; the pipeline advances one stage at
    // every fourth tile. Only high/low are carried in the loop; the part merge stays outside.
    macro_rules! chain {
        ($t0:expr, $t1:expr) => {{
            let (mut high, mut low) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
            let mut t = $t0;
            while t < $t1 {
                if STAGES == 3 {
                    let s = t / 4;
                    if t % 4 == 0 {
                        // This thread's reads of stage s - 1 are done; wait for stage s's data.
                        if s > 0 {
                            bar_arrive((BARS + 24 + (s - 1) % 3 * 8) as u32);
                        }
                        bar_wait((BARS + s % 3 * 8) as u32, (s / 3 % 2) as u32);
                        current = s % 3 * STAGE;
                    } else if t % 4 == REFILL && t + 8 - REFILL < kt {
                        // Stage s + 2 refills buffer (s + 2) % 3 once every thread read stage s - 1
                        // from it (a tile into stage s, so the slowest warp rarely holds it up).
                        let b = (s + 2) % 3;
                        if s > 0 {
                            bar_wait((BARS + 24 + b * 8) as u32, ((s - 1) / 3 % 2) as u32);
                        }
                        issue((b * STAGE) as u32, t + 8 - REFILL);
                        bar_copies((BARS + b * 8) as u32);
                    }
                } else if t % 4 == 0 {
                    ptx::wait();
                    // Every reader of the buffer refilled below retired in the previous stage.
                    ptx::sync();
                    if t + 4 < kt {
                        issue(((1 - buffer) * STAGE) as u32, t + 4);
                    }
                    current = buffer * STAGE;
                    buffer = 1 - buffer;
                }
                if J == 4 || active {
                    let a = current + wm * 2048 + t % 4 * 512 + lane * 16;
                    let b = current + 2 * PLANE + wn * 1024 + t % 4 * 256 + lane * 8;
                    let hi = [ptx::load4(a as u32), ptx::load4((a + 2048) as u32)];
                    let lo = [ptx::load4((a + PLANE) as u32), ptx::load4((a + PLANE + 2048) as u32)];
                    unroll!(4, j, {
                        if j < J {
                            let w = ptx::load2((b + j * 1024) as u32);
                            unroll!(2, i, {
                                high[i][j] = madd(hi[i], w, high[i][j]);
                                low[i][j] = madd(lo[i], w, low[i][j]);
                            });
                        }
                    });
                }
                t += 1;
            }
            let mut value = [[ZERO; 4]; 2];
            unroll!(2, i, {
                unroll!(4, j, {
                    value[i][j] = add(high[i][j], low[i][j]);
                });
            });
            value
        }};
    }
    // The four canonical parts are fixed by K alone. Spell them out so the hot
    // chains carry no run-time part dispatch; the K16 order and merge tree stay
    // canonical::range(kt, part), (p0+p1)+(p2+p3), including short/tail rows.
    let mut left = chain!(0, kt / 4);
    let p1 = chain!(kt / 4, kt * 2 / 4);
    unroll!(2, i, {
        unroll!(4, j, { left[i][j] = add(left[i][j], p1[i][j]); });
    });
    let mut right = chain!(kt * 2 / 4, kt * 3 / 4);
    let p3 = chain!(kt * 3 / 4, kt);
    unroll!(2, i, {
        unroll!(4, j, { right[i][j] = add(right[i][j], p3[i][j]); });
    });
    let mut out = [[ZERO; 4]; 2];
    unroll!(2, i, {
        unroll!(4, j, {
            out[i][j] = add(left[i][j], right[i][j]);
        });
    });
    out
}

/// Prompt HC up (option A at M >= 64): the up projection [M][4H] of the packed K=320 operand
/// and the stream-ordered mix in one tile. Slice (mi, ni) owns tokens 64mi.. and features
/// 32ni.. of all four streams: its 8-row group g reads weight rows (g/4)H + 32ni + 8(g%4), so
/// warp column wn/4 is stream s. Every gate keeps flash_next_hc_up_latency's K16 graph
/// (five-tile ranges, hi/lo chains, (p0+p1)+(p2+p3)); the gates meet in shared memory and
/// each output mixes streams 0..3 in order with hc_stream, then x0.25, as latency4 does.
/// Two pipeline stages; the mix reads the normalized streams from global.
/// planes != 0: the epilogue also writes the outputs' dense hi/lo planes (K = H, M rows; the
/// layout flash_bf16_input gives them), so the next projections skip their pack.
/// residual != 0: `normalized` holds the norm's scales [M][4] (flash_next_hc_add_norm with
/// scales != 0) and each stream value is formed here from the residual row as the norm forms it,
/// residual * scale * (1 + weight[4H]) in that order, so the mix sees the same bits.
/// Block256, grid ceil(M/64) x H/32/FLASH_HC_UP_SLICES, dynamic shared98304; H = 2560, K = 320,
/// 4H/K weights. Each CTA runs FLASH_HC_UP_SLICES consecutive feature slices, the next slice's
/// first two stages loading behind the current slice's mix (gates in the third 32 KiB).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_tiled(p: Bf16Rows, normalized: u64, planes: u64, residual: u64, weight: u64) {
    let (h, slices) = (2560usize, crate::kernel_params::FLASH_HC_UP_SLICES);
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, mt, nj) = (warp / 4 * 2, _block_idx_x() as usize * 4, _block_idx_y() as usize);
    // CTA column nj owns feature slices nj*slices..; a full-width grid's extra columns idle.
    if (nj + 1) * slices > h / 32 || p.cols as usize / 16 <= 4 {
        return;
    }
    // The streams and the outputs pass through once (L2 evicts them first), so the re-read A/B
    // operands stay.
    let first: u64;
    asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) first);
    // One slice per CTA runs the tiled GEMM's three-stage ring (the gates then reuse stage 0);
    // more slices keep two stages and load the next slice's first two behind the mix.
    let ring = slices == 1;
    let gates_at = if ring { 0 } else { 2 * STAGE };
    let source = if residual != 0 { residual } else { normalized };
    if !ring {
        stage_hc_up(p, 0, mt, nj * slices, 0, tid);
        stage_hc_up(p, STAGE as u32, mt, nj * slices, 4, tid);
    }
    let sig = |v: f32| crate::flash_next::hc_sigmoid(v);
    let kt = h / 16;
    let plane = (p.tokens as usize).div_ceil(16) * kt * 128;
    let mut r = 0;
    while r < slices {
        // Opaque per slice, so the copy addresses are not hoisted out of the slice loop (and
        // held across every GEMM).
        let (mt, ni): (u64, u64);
        asm!("mov.b64 {m}, {a}; mov.b64 {n}, {b};", m = out(reg64) mt, n = out(reg64) ni,
            a = in(reg64) (_block_idx_x() as usize * 4) as u64, b = in(reg64) (nj * slices + r) as u64);
        let (mt, ni) = (mt as usize, ni as usize);
        let issue = |sh: u32, k: usize| stage_hc_up(p, sh, mt, ni, k, tid);
        let gates = if ring { product::<3>(p, tid, issue) } else { product_hc::<false>(p, tid, issue, |_, _| {}) };
        // Every reader of the stage buffers retires before the next slice refills them.
        ptx::sync();
        unroll!(2, i, {
            unroll!(4, j, {
                let (t, f) = ((wm + i) * 16 + lane / 4, j * 8 + lane % 4 * 2);
                // Gate row t's feature pair (f, f + 1) at word f ^ 8(t % 4): a half warp's four
                // rows land on distinct banks.
                let at = |t: usize| (gates_at + ((warp % 4 * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32;
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at(t), a = in(reg32) gates[i][j].a, b = in(reg32) gates[i][j].b);
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at(t + 8), a = in(reg32) gates[i][j].c, b = in(reg32) gates[i][j].d);
            });
        });
        // The next slice's first two stages load behind this slice's mix.
        if r + 1 < slices {
            stage_hc_up(p, 0, mt, ni + 1, 0, tid);
            stage_hc_up(p, STAGE as u32, mt, ni + 1, 4, tid);
        }
        ptx::sync();
        // Two adjacent features per thread and q; the streams load straight from global. With planes != 0 each split2 pair also lands in the next
        // projections' dense A-fragment word (K = H, rows = M).
        // All sixteen stream loads are in flight before the first mix (the GEMM's registers are free).
        let mut xs = [[0f32; 8]; 4];
        // residual != 0: the thread's eight weights (its feature pair is the same for every q) and
        // its rows' scales, issued before the stream loads.
        let (mut ws, mut ss) = ([0f32; 8], [[0f32; 4]; 4]);
        if residual != 0 {
            unroll!(4, stream, {
                asm!("ld.global.nc.v2.f32 {{{a}, {b}}}, [{s}];", a = out(reg32) ws[stream * 2], b = out(reg32) ws[stream * 2 + 1],
                    s = in(reg64) (weight as *const f32).add(stream * h + ni * 32 + tid % 16 * 2));
            });
            unroll!(4, q, {
                let token = mt * 16 + (tid + q * 256) / 16;
                if token < p.tokens as usize {
                    asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{s}];", a = out(reg32) ss[q][0], b = out(reg32) ss[q][1],
                        c = out(reg32) ss[q][2], d = out(reg32) ss[q][3], s = in(reg64) (normalized as *const f32).add(token * 4));
                }
            });
        }
        unroll!(4, q, {
            let e = tid + q * 256;
            let (t, f) = (e / 16, e % 16 * 2);
            let (token, feature) = (mt * 16 + t, ni * 32 + f);
            if token < p.tokens as usize {
                unroll!(4, stream, {
                    let at = (source as *const f32).add(token * 4 * h + stream * h + feature);
                    asm!("ld.global.nc.L2::cache_hint.v2.f32 {{{a}, {b}}}, [{s}], {p};",
                        a = out(reg32) xs[q][stream * 2], b = out(reg32) xs[q][stream * 2 + 1], s = in(reg64) at, p = in(reg64) first);
                });
            }
        });
        unroll!(4, q, {
            let e = tid + q * 256;
            let (t, f) = (e / 16, e % 16 * 2);
            let (token, feature) = (mt * 16 + t, ni * 32 + f);
            if token < p.tokens as usize {
                let mut x = xs[q];
                if residual != 0 {
                    unroll!(4, stream, {
                        unroll!(2, half, {
                            x[stream * 2 + half] = x[stream * 2 + half] * ss[q][stream] * (1.0 + ws[stream * 2 + half]);
                        });
                    });
                }
                let mut g = [[0u32; 2]; 4];
                unroll!(4, stream, {
                    g[stream] = ptx::load2((gates_at + ((stream * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32);
                });
                let mix = |half: usize| {
                    let mut mixed = 0.0;
                    for stream in 0..4 {
                        mixed = crate::next_math::hc_stream(mixed, x[stream * 2 + half], f32::from_bits(g[stream][half]), sig);
                    }
                    mixed * 0.25
                };
                let (a, b) = (mix(0), mix(1));
                asm!("st.global.cs.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) (p.output as *mut f32).add(token * h + feature), a = in(reg32) a, b = in(reg32) b);
                if planes != 0 {
                    let word = ((token / 16 * kt + feature / 16) * 32 + token % 8 * 4 + feature % 8 / 2) * 4
                        + token % 16 / 8 + feature % 16 / 8 * 2;
                    let (hi, lo) = crate::gemm::split2_pair(a, b);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(word), v = in(reg32) hi);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(plane + word), v = in(reg32) lo);
                }
            }
        });
        r += 1;
    }
}
/// product::<2> for prompt HC up with stages 0 and 1 already issued (stage s in buffer s % 2):
/// stage 0 computes while stage 1 may still land; from stage 1 on, the refill of stage s + 1
/// follows the barrier that retires stage s - 1's readers, as in product.
/// ROWS (tile_hc on the residual rows): stages of ROWS_STAGE bytes; after each stage's barrier
/// `convert(sh, t)` forms the stage's A planes and a second barrier publishes them.
#[inline(always)]
unsafe fn product_hc<const ROWS: bool>(p: Bf16Rows, tid: usize, issue: impl Fn(u32, usize), convert: impl Fn(u32, usize)) -> [[Acc; 4]; 2] {
    let size = if ROWS { ROWS_STAGE } else { STAGE };
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    let kt = p.cols as usize / 16;
    let mut current = 0;
    macro_rules! chain {
        ($t0:expr, $t1:expr) => {{
            let (mut high, mut low) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
            let mut t = $t0;
            while t < $t1 {
                if t % 4 == 0 {
                    let s = t / 4;
                    if s == 0 {
                        asm!("cp.async.wait_group 1;");
                    } else {
                        ptx::wait();
                    }
                    ptx::sync();
                    if s > 0 && t + 4 < kt {
                        issue(((s + 1) % 2 * size) as u32, t + 4);
                    }
                    current = s % 2 * size;
                    if ROWS {
                        convert(current as u32, t);
                        ptx::sync();
                    }
                }
                let a = current + wm * 2048 + t % 4 * 512 + lane * 16;
                let b = current + 2 * PLANE + wn * 1024 + t % 4 * 256 + lane * 8;
                let hi = [ptx::load4(a as u32), ptx::load4((a + 2048) as u32)];
                let lo = [ptx::load4((a + PLANE) as u32), ptx::load4((a + PLANE + 2048) as u32)];
                unroll!(4, j, {
                    let w = ptx::load2((b + j * 1024) as u32);
                    unroll!(2, i, {
                        high[i][j] = madd(hi[i], w, high[i][j]);
                        low[i][j] = madd(lo[i], w, low[i][j]);
                    });
                });
                t += 1;
            }
            let mut value = [[ZERO; 4]; 2];
            unroll!(2, i, {
                unroll!(4, j, {
                    value[i][j] = add(high[i][j], low[i][j]);
                });
            });
            value
        }};
    }
    let (mut left, mut right) = ([[ZERO; 4]; 2], [[ZERO; 4]; 2]);
    // canonical::range: K16 parts [kt*p/4, kt*(p+1)/4), merged (p0+p1)+(p2+p3).
    let mut part = 0;
    while part < 4 {
        let v = chain!(kt * part / 4, kt * (part + 1) / 4);
        unroll!(2, i, {
            unroll!(4, j, {
                match part {
                    0 => left[i][j] = v[i][j],
                    1 => left[i][j] = add(left[i][j], v[i][j]),
                    2 => right[i][j] = v[i][j],
                    _ => right[i][j] = add(right[i][j], v[i][j]),
                }
            });
        });
        part += 1;
    }
    let mut out = [[ZERO; 4]; 2];
    unroll!(2, i, {
        unroll!(4, j, {
            out[i][j] = add(left[i][j], right[i][j]);
        });
    });
    out
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_tiled_u32(p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let mt_count = (p.tokens as usize).div_ceil(64);
    let nt_count = ((p.rows + p.rows2) as usize).div_ceil(128);
    if p.tokens as usize * p.cols as usize * 4 > 16 * 1024 * 1024 {
        // Retain the previous wide tile's bounded input reuse window in L2.
        let group = if p.cols <= 4096 {
            8
        } else if p.cols <= 8192 {
            8
        } else {
            4
        };
        let linear = ni * mt_count + mi;
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows;
        ni = within / rows;
    }
    let (mt, nt) = (mi * 4, ni * 16);
    // Tail CTAs always retain the original second-matrix bounds handling.
    if (nt + 16) * 8 > p.rows as usize {
        tile::<true>(p, mt, nt, tid);
    } else if p.tokens as u64 * p.cols as u64 > u32::MAX as u64
        || p.rows as u64 * p.cols as u64 > u32::MAX as u64
    {
        tile::<false>(p, mt, nt, tid);
    } else {
        tile_u32(p, mt, nt, tid);
    }
}

/// Same full-tile addresses calculated in 32-bit element offsets. The entry's
/// full tensor span is validated before dispatch; only pointer formation widens.
#[inline(always)]
unsafe fn stage_u32(p: Bf16Rows, sh: u32, mt: u32, nt: u32, k: u32, tid: u32) {
    let kt=p.cols/16;
    unroll!(2, step, {
        let i=tid+(step*256) as u32;
        let row=mt+i/128;
        let valid=row*16<p.tokens;
        let at=((row*kt+k+i%128/32)*32+i%32)*4;
        unroll!(2, plane, {
            let off=if valid { at as usize+plane*p.input_plane_words as usize } else {0};
            copy(sh+(plane*PLANE) as u32+i*16,(p.input as *const u32).add(off),valid);
        });
    });
    unroll!(4, step, {
        let i=tid+(step*256) as u32;
        let row=nt+i/64;
        let tile=if kt%4==0 {(row/8*(kt/4)+k/4)*32+row%8*4+k%4} else {row*kt+k};
        let at=tile*64+i%64*4;
        copy(sh+(2*PLANE) as u32+i*16,(p.weight as *const u32).add(at as usize),true);
    });
    ptx::commit();
}
#[inline(always)]
unsafe fn tile_u32(p:Bf16Rows,mt:usize,nt:usize,tid:usize) {
    let (lane,warp)=(tid%32,tid/32);
    let (wm,wn)=(warp/4*2,warp%4*4);
    let out=product::<3>(p,tid,|sh,k|stage_u32(p,sh,mt as u32,nt as u32,k as u32,tid as u32));
    unroll!(2,i,{unroll!(4,j,{
        write_pair(p,out[i][j],(mt+wm+i)*16+lane/4,(nt+wn+j)*8+lane%4*2);
    });});
}

/// Prompt HC-up epilogue also emits the row-major BF16 expert operand.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_next_hc_up_tiled_a16(p: Bf16Rows, normalized: u64, planes: u64, residual: u64, weight: u64) {
    let (h, slices) = (2560usize, crate::kernel_params::FLASH_HC_UP_SLICES);
    let tid = _thread_idx_x() as usize;
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, mt, nj) = (warp / 4 * 2, _block_idx_x() as usize * 4, _block_idx_y() as usize);
    // CTA column nj owns feature slices nj*slices..; a full-width grid's extra columns idle.
    if (nj + 1) * slices > h / 32 || p.cols as usize / 16 <= 4 {
        return;
    }
    // The streams and the outputs pass through once (L2 evicts them first), so the re-read A/B
    // operands stay.
    let first: u64;
    asm!("createpolicy.fractional.L2::evict_first.b64 {p}, 1.0;", p = out(reg64) first);
    // One slice per CTA runs the tiled GEMM's three-stage ring (the gates then reuse stage 0);
    // more slices keep two stages and load the next slice's first two behind the mix.
    let ring = slices == 1;
    let gates_at = if ring { 0 } else { 2 * STAGE };
    let source = if residual != 0 { residual } else { normalized };
    if !ring {
        stage_hc_up(p, 0, mt, nj * slices, 0, tid);
        stage_hc_up(p, STAGE as u32, mt, nj * slices, 4, tid);
    }
    let sig = |v: f32| crate::flash_next::hc_sigmoid(v);
    let kt = h / 16;
    let plane = (p.tokens as usize).div_ceil(16) * kt * 128;
    let mut r = 0;
    while r < slices {
        // Opaque per slice, so the copy addresses are not hoisted out of the slice loop (and
        // held across every GEMM).
        let (mt, ni): (u64, u64);
        asm!("mov.b64 {m}, {a}; mov.b64 {n}, {b};", m = out(reg64) mt, n = out(reg64) ni,
            a = in(reg64) (_block_idx_x() as usize * 4) as u64, b = in(reg64) (nj * slices + r) as u64);
        let (mt, ni) = (mt as usize, ni as usize);
        let issue = |sh: u32, k: usize| stage_hc_up(p, sh, mt, ni, k, tid);
        let gates = if ring { product::<3>(p, tid, issue) } else { product_hc::<false>(p, tid, issue, |_, _| {}) };
        // Every reader of the stage buffers retires before the next slice refills them.
        ptx::sync();
        unroll!(2, i, {
            unroll!(4, j, {
                let (t, f) = ((wm + i) * 16 + lane / 4, j * 8 + lane % 4 * 2);
                // Gate row t's feature pair (f, f + 1) at word f ^ 8(t % 4): a half warp's four
                // rows land on distinct banks.
                let at = |t: usize| (gates_at + ((warp % 4 * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32;
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at(t), a = in(reg32) gates[i][j].a, b = in(reg32) gates[i][j].b);
                asm!("st.shared.v2.f32 [{s}], {{{a}, {b}}};", s = in(reg32) at(t + 8), a = in(reg32) gates[i][j].c, b = in(reg32) gates[i][j].d);
            });
        });
        // The next slice's first two stages load behind this slice's mix.
        if r + 1 < slices {
            stage_hc_up(p, 0, mt, ni + 1, 0, tid);
            stage_hc_up(p, STAGE as u32, mt, ni + 1, 4, tid);
        }
        ptx::sync();
        // Two adjacent features per thread and q; the streams load straight from global. With planes != 0 each split2 pair also lands in the next
        // projections' dense A-fragment word (K = H, rows = M).
        // All sixteen stream loads are in flight before the first mix (the GEMM's registers are free).
        let mut xs = [[0f32; 8]; 4];
        // residual != 0: the thread's eight weights (its feature pair is the same for every q) and
        // its rows' scales, issued before the stream loads.
        let (mut ws, mut ss) = ([0f32; 8], [[0f32; 4]; 4]);
        if residual != 0 {
            unroll!(4, stream, {
                asm!("ld.global.nc.v2.f32 {{{a}, {b}}}, [{s}];", a = out(reg32) ws[stream * 2], b = out(reg32) ws[stream * 2 + 1],
                    s = in(reg64) (weight as *const f32).add(stream * h + ni * 32 + tid % 16 * 2));
            });
            unroll!(4, q, {
                let token = mt * 16 + (tid + q * 256) / 16;
                if token < p.tokens as usize {
                    asm!("ld.global.nc.v4.f32 {{{a}, {b}, {c}, {d}}}, [{s}];", a = out(reg32) ss[q][0], b = out(reg32) ss[q][1],
                        c = out(reg32) ss[q][2], d = out(reg32) ss[q][3], s = in(reg64) (normalized as *const f32).add(token * 4));
                }
            });
        }
        unroll!(4, q, {
            let e = tid + q * 256;
            let (t, f) = (e / 16, e % 16 * 2);
            let (token, feature) = (mt * 16 + t, ni * 32 + f);
            if token < p.tokens as usize {
                unroll!(4, stream, {
                    let at = (source as *const f32).add(token * 4 * h + stream * h + feature);
                    asm!("ld.global.nc.L2::cache_hint.v2.f32 {{{a}, {b}}}, [{s}], {p};",
                        a = out(reg32) xs[q][stream * 2], b = out(reg32) xs[q][stream * 2 + 1], s = in(reg64) at, p = in(reg64) first);
                });
            }
        });
        unroll!(4, q, {
            let e = tid + q * 256;
            let (t, f) = (e / 16, e % 16 * 2);
            let (token, feature) = (mt * 16 + t, ni * 32 + f);
            if token < p.tokens as usize {
                let mut x = xs[q];
                if residual != 0 {
                    unroll!(4, stream, {
                        unroll!(2, half, {
                            x[stream * 2 + half] = x[stream * 2 + half] * ss[q][stream] * (1.0 + ws[stream * 2 + half]);
                        });
                    });
                }
                let mut g = [[0u32; 2]; 4];
                unroll!(4, stream, {
                    g[stream] = ptx::load2((gates_at + ((stream * 64 + t) * 32 + (f ^ t % 4 * 8)) * 4) as u32);
                });
                let mix = |half: usize| {
                    let mut mixed = 0.0;
                    for stream in 0..4 {
                        mixed = crate::next_math::hc_stream(mixed, x[stream * 2 + half], f32::from_bits(g[stream][half]), sig);
                    }
                    mixed * 0.25
                };
                let (a, b) = (mix(0), mix(1));
                asm!("st.global.cs.v2.f32 [{y}], {{{a}, {b}}};", y = in(reg64) (p.output as *mut f32).add(token * h + feature), a = in(reg32) a, b = in(reg32) b);
                if planes != 0 {
                    let word = ((token / 16 * kt + feature / 16) * 32 + token % 8 * 4 + feature % 8 / 2) * 4
                        + token % 16 / 8 + feature % 16 / 8 * 2;
                    let (hi, lo) = crate::gemm::split2_pair(a, b);
                    // Same RN BF16 pair as fp4_prefill_tc_a16; independent destination.
                    // This separate entry requires output2=[tokens][2560] BF16 and planes!=0.
                    asm!("st.global.b32 [{y}], {v};", y = in(reg64) (p.output2 as *mut u32).add((token * h + feature) / 2), v = in(reg32) hi);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(word), v = in(reg32) hi);
                    asm!("st.global.cs.b32 [{y}], {v};", y = in(reg64) (planes as *mut u32).add(plane + word), v = in(reg32) lo);
                }
            }
        });
        r += 1;
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_bf16_mma_prefill_tiled_plan(p: Bf16Rows) {
    let tid = _thread_idx_x() as usize;
    let (mut mi, mut ni) = (_block_idx_x() as usize, _block_idx_y() as usize);
    let mt_count = (p.tokens as usize).div_ceil(64);
    let nt_count = ((p.rows + p.rows2) as usize).div_ceil(128);
    if p.tokens as usize * p.cols as usize * 4 > 16 * 1024 * 1024 {
        // Retain the previous wide tile's bounded input reuse window in L2.
        let group = if p.cols <= 4096 {
            8
        } else if p.cols <= 8192 {
            8
        } else {
            4
        };
        let linear = ni * mt_count + mi;
        let first = linear / (nt_count * group) * group;
        let rows = group.min(mt_count - first);
        let within = linear - first * nt_count;
        mi = first + within % rows;
        ni = within / rows;
    }
    let (mt, nt) = (mi * 4, ni * 16);
    // Tail CTAs always retain the original second-matrix bounds handling.
    if (nt + 16) * 8 > p.rows as usize {
        tile::<true>(p, mt, nt, tid);
    } else if p.tokens as u64 * p.cols as u64 > u32::MAX as u64
        || p.rows as u64 * p.cols as u64 > u32::MAX as u64
    {
        tile::<false>(p, mt, nt, tid);
    } else {
        if p.cols % 64 == 0 { tile_plan(p, mt, nt, tid); } else { tile_u32(p, mt, nt, tid); }
    }
}

/// Full-tile global sources are affine in the stage's K16 index (always a multiple
/// of four). Compute the K-independent offsets once; retain the original ring.
#[inline(always)]
unsafe fn tile_plan(p: Bf16Rows, mt: usize, nt: usize, tid: usize) {
    let (mt, nt, tid32, kt) = (mt as u32, nt as u32, tid as u32, p.cols / 16);
    let a0 = ((mt + tid32 / 128) * kt * 32 + tid32 % 128) * 4;
    let a = [a0, a0 + kt * 256];
    let valid = [(mt + tid32 / 128) * 16 < p.tokens,
        (mt + 2 + tid32 / 128) * 16 < p.tokens];
    let b0 = ((nt + tid32 / 64) / 8 * (kt / 4) * 32
        + (nt + tid32 / 64) % 8 * 4) * 64 + tid32 % 64 * 4;
    let b = [b0, b0 + 1024, b0 + kt * 512, b0 + kt * 512 + 1024];
    let out = product::<3>(p, tid, |sh, k| {
        unroll!(2, step, {
            let i = tid32 + (step * 256) as u32;
            unroll!(2, plane, {
                let at = if valid[step] { (a[step] + k as u32 * 128) as usize
                    + plane * p.input_plane_words as usize } else { 0 };
                copy(sh + (plane * PLANE) as u32 + i * 16,
                    (p.input as *const u32).add(at), valid[step]);
            });
        });
        unroll!(4, step, {
            let i = tid32 + (step * 256) as u32;
            copy(sh + (2 * PLANE) as u32 + i * 16,
                (p.weight as *const u32).add((b[step] + k as u32 * 512) as usize), true);
        });
        ptx::commit();
    });
    let (lane, warp) = (tid % 32, tid / 32);
    let (wm, wn) = (warp / 4 * 2, warp % 4 * 4);
    unroll!(2, i, { unroll!(4, j, {
        write_pair(p, out[i][j], (mt as usize + wm + i) * 16 + lane / 4,
            (nt as usize + wn + j) * 8 + lane % 4 * 2);
    }); });
}
