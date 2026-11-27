//! Zero-stack fast Inkling mix kernels (`FAST_MIX_BLEND = 107`, `FAST_ROW_DOTS = 108`).
//!
//! - `mix_blend`: each block (256 threads) per channel part of a `(row, query)` computes the `kept` softmax weights
//!   in shared memory (exact `fixed_math::exp` in rising order, bit-for-bit identical to `Op::MixBlend`),
//!   then blends all `dim` channels with coalesced reads/writes and optional direct `[asked][rows][dim]`
//!   transpose output to eliminate post-mix `PermuteRows` swaps.
//! - `row_dots`: 1 block (128 threads = 4 warps) per `(row, e)` computes all `asked` dot products with
//!   `joined[e, row]` in fixed 32-lane warp-butterfly order (100% deterministic, zero atomics).

use crate::block_ops::{LANES, NORM_CHUNK, root};
use crate::block_stream::CONV_TOKEN_TAPS;
use crate::fast_ops::{
    CONV_KV_CLIP, CONV_KV_DIM, CONV_KV_NORM, KV_FINISH_WORD, KV_NEXT_RING_WORD,
    KV_PREFILL_TILE_WORD, ROUTE_FOLD_CONV_TAPS_MAX, ROW_DOTS_THREADS, STREAM_OP_SHORT_CONV_WIN,
    STREAM_OPS_THREADS, conv_kv_blocks, conv_kv_tiled_blocks_y, stream_ops_admitted,
};
use crate::fast_ops::{
    KV_FINISH_HEAD_DIM, KV_FINISH_TAG, MIX_DOTS, MIX_KEPT_MAX, MIX_LIST_NORM_HALF, MIX_ONE,
    MIX_ONE_DIM_MAX, MIX_ONE_EPSILON_AT, MIX_ONE_FUSED_Q, MIX_ONE_FUSED_Q_MAX, MIX_ONE_KEPT_MAX,
    MIX_ONE_NORMED_AT, MIX_ONE_POST_EPS_AT, MIX_ONE_POST_HALF_OUT, MIX_ONE_POST_NORM,
    MIX_ONE_POST_OUT_AT, MIX_ONE_POST_WEIGHT_AT, MIX_ONE_POST_WEIGHTS_PER_QUERY,
    MIX_ONE_SKIP_MIXED, kv_finish_admitted, mix_blend_admitted, mix_blend_parts,
    mix_one_admitted, mix_one_parts, mix_one_post_norm_admitted, row_dots_admitted,
};
use crate::fast_ops::{MIX_BLEND_THREADS, NORM_CHUNKS_MAX};
use crate::fixed_math::{exp, rn};
use crate::tc_ptx::{load_u32, shfl_xor_f32, shfl_xor_u32, store_u32};

const WEIGHT_SMEM_BYTES: usize = MIX_KEPT_MAX * 4; // 256 bytes
/// Warps of a list mix block, and the most norm chunks of one warp in the list mix with the gate norm (`MIX_LIST_NORM`).
const LIST_NORM_WARPS: usize = MIX_BLEND_THREADS / 32;
const LIST_NORM_CHUNKS: usize = NORM_CHUNKS_MAX / LIST_NORM_WARPS;
const _: () = assert!(LIST_NORM_CHUNKS * LIST_NORM_WARPS == NORM_CHUNKS_MAX);

/// Replay one recorded command of kind `FAST_MIX_BLEND`.
/// SAFETY (caller contract, `resident_launch_fast_mix_blend`): `table[index]` is a valid `FAST_MIX_BLEND`
/// command recorded by `Backend::fast_mix_blend_into`, `Backend::fast_mix_list`, or `Backend::fast_mix_rows` after operand checks.
#[inline(always)]
pub(crate) unsafe fn replay_mix_blend(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host verified via `fast_ops::check_mix_blend` that `scores` holds `rows * asked * kept` FP32 words,
    //   `joined` holds `kept * rows * dim` FP32 words, `output` holds `rows * asked * dim` FP32 words and is
    //   disjoint from inputs, and all pointers are word-aligned.
    // - A null joined pointer selects the prefill command. The payload then holds `kept` checked
    //   pointers to immutable `rows * dim`-word streams, disjoint from the private output.
    // - With `MIX_DOTS` in word 8 (prefill only, `Backend::fast_mix_list`), word 0 is the checked `asked * dim`-word
    //   queries in place of the scores. Warp `w` writes dot `e` (`e = w, w + warps, ..` below `kept`) to shared word `e`
    //   (lane 0 only) before the first barrier; thread 0 reads them after it, as it reads the scores otherwise.
    // - Device re-checks `mix_blend_admitted`, grid bounds (`block < rows * asked`, the grid is `rows * asked * parts`
    //   blocks), and alignment. Each output word has one writer: the block of its (row, query) and its part.
    // - Shared memory holds `kept <= MIX_KEPT_MAX` FP32 weights strictly inside `WEIGHT_SMEM_BYTES` (256 B);
    //   all 256 threads participate in `bar.sync 0`.
    unsafe {
        // SAFETY: prefetch hints on `table[index]` and the commands after it in the stage buffer (`warm_command!`).
        crate::fast_elem::warm_command!(table, index);
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_MIX_BLEND { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let stage = crate::ptx::shared::<{ MIX_ONE_DIM_MAX * 4 }>();
        let sums = crate::ptx::shared::<{ (3 * NORM_CHUNKS_MAX + 1) * 4 }>();
        // Fused mix (`MIX_ONE` in word 8, optional transpose in bit 0, optional post-blend RoundedNorm in bit 4,
        // skip un-normed blend write in bit 5, per-query post-norm weights in bit 6, packed BF16 post-out in bit 7):
        // grid is `(asked * parts, rows, 1)`. Each block `(q * parts + part, row)` computes the plain norm of
        // `stream[row]`, the `kept` deterministic 32-lane warp-butterfly dot products for `(row, q)` (matching
        // `MIX_DOTS`/`FAST_ROW_DOTS` across M=1/M=2..16/prefill), register softmax weights, 1-chunk/warp channel
        // blend, and optional register-resident `RoundedNorm` in 2 barriers.
        // SAFETY: check_mix_rows_normed_form verifies queries (asked * dim words), each source (rows * dim), optional
        // post-norm weight (`dim` or `asked * dim`), and disjoint storage for optional normed/output/post_out.
        // Rows start on 16-byte boundaries. Every barrier is convergent `_syncthreads` reached by all threads;
        // every shuffle runs across full 32-lane warps (uniform chunk/dim bounds). Each output word has one writer.
        if *w.add(8) & MIX_ONE != 0 {
            macro_rules! mul_rn {($a:expr, $b:expr)=>{{let p: f32; core::arch::asm!("mul.rn.f32 {p}, {a}, {b};", p = out(reg32) p, a = in(reg32) $a, b = in(reg32) $b, options(nomem, nostack, pure)); p}}}
            let flags = *w.add(8);
            let rows = *w.add(3) as usize;
            let (asked, kept, dim) = (*w.add(4) as usize, *w.add(5) as usize, *w.add(6) as usize);
            let threads = core::arch::nvptx::_block_dim_x() as usize;
            let parts = mix_one_parts(rows, asked, dim);
            let post_norm = (flags & MIX_ONE_POST_NORM) != 0;
            let skip_mixed = post_norm && (flags & MIX_ONE_SKIP_MIXED) != 0;
            let fused_q = (flags & MIX_ONE_FUSED_Q) != 0;
            let expect_gx = if fused_q { 1 } else { asked * parts };
            if !mix_one_admitted(rows, asked, kept, dim)
                || core::arch::nvptx::_grid_dim_x() as usize != expect_gx
                || core::arch::nvptx::_grid_dim_y() as usize != rows
                || core::arch::nvptx::_block_idx_y() as usize >= rows
                || (fused_q && (!post_norm || asked < 2 || asked > MIX_ONE_FUSED_Q_MAX || parts != 1))
                || (post_norm && (parts != 1 || !mix_one_post_norm_admitted(dim, threads)))
                || (!post_norm && threads != MIX_BLEND_THREADS)
            {
                core::arch::nvptx::trap();
            }
            let bx = core::arch::nvptx::_block_idx_x() as usize;
            let row = core::arch::nvptx::_block_idx_y() as usize;
            let q = if fused_q { 0 } else { bx / parts };
            let row_bytes_off = row * dim;
            let stream = (*w.add(9 + kept - 1) as *const f32).add(row_bytes_off);
            if ((*w as usize) | (stream as usize)) & 15 != 0
                || ((*w.add(2) as usize) | (*w.add(MIX_ONE_NORMED_AT) as usize)
                    | (*w.add(MIX_ONE_POST_OUT_AT) as usize) | (*w.add(MIX_ONE_POST_WEIGHT_AT) as usize)) & 3 != 0
                || (!skip_mixed && *w.add(2) == 0)
                || (post_norm && *w.add(MIX_ONE_POST_OUT_AT) == 0)
            {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let (warp, lane) = (tid >> 5, tid & 31);
            // A read-only global word (the query rows, list rows and norm weights do not change while the launch runs).
            macro_rules! word {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
            if threads / 32 < kept { core::arch::nvptx::trap(); }
            const NORM_DOT_AT: u32 = (MIX_ONE_FUSED_Q_MAX * MIX_ONE_KEPT_MAX * 4) as u32;
            let dots = crate::ptx::shared::<{ (MIX_ONE_FUSED_Q_MAX * MIX_ONE_KEPT_MAX + 1) * 4 }>();
            let warps = threads >> 5;
            let chunks = dim / NORM_CHUNK;
            let b_chunk = warp * NORM_CHUNK + lane;
            let has_chunk = post_norm && warp < chunks;
            let queries = *w as *const f32;
            let qrow = queries.add(q * dim);
            let post_w = if post_norm && !fused_q { *w.add(MIX_ONE_POST_WEIGHT_AT) as *const f32 } else { core::ptr::null() };
            let pw_ptr = if post_w.is_null() {
                core::ptr::null()
            } else if (flags & MIX_ONE_POST_WEIGHTS_PER_QUERY) != 0 {
                post_w.add(q * dim)
            } else {
                post_w
            };
            // Release downstream dependent panel/narrow products immediately (all warps at t=0) so their read-only
            // static weight prefetches overlap 100% of `MIX_ONE` without waiting for `Warps 1..kept` dot loops.
            core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
            let (mut pw0, mut pw1, mut pw2, mut pw3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            macro_rules! load_pw {() => {
                if !fused_q && has_chunk && !pw_ptr.is_null() {
                    pw0 = word!(pw_ptr.add(b_chunk));
                    pw1 = word!(pw_ptr.add(b_chunk + LANES));
                    pw2 = word!(pw_ptr.add(b_chunk + 2 * LANES));
                    pw3 = word!(pw_ptr.add(b_chunk + 3 * LANES));
                }
            }}
            macro_rules! copy_stage {($base_warp:expr) => {{
                let copy_threads = (warps - ($base_warp)) << 5;
                let mut i = tid - (($base_warp) << 5);
                let nvec = dim >> 2;
                while i < nvec {
                    let (v0, v1, v2, v3): (u32, u32, u32, u32);
                    core::arch::asm!(
                        "ld.global.v4.u32 {{{a},{b},{c},{d}}}, [{p}];",
                        a = out(reg32) v0, b = out(reg32) v1, c = out(reg32) v2, d = out(reg32) v3,
                        p = in(reg64) stream.add(4 * i),
                        options(readonly, nostack),
                    );
                    core::arch::asm!(
                        "st.shared.v4.u32 [{s}], {{{a},{b},{c},{d}}};",
                        s = in(reg32) stage + (16 * i) as u32,
                        a = in(reg32) v0, b = in(reg32) v1, c = in(reg32) v2, d = in(reg32) v3,
                        options(nostack),
                    );
                    i += copy_threads;
                }
            }}}
            macro_rules! reduce_dot {($s:expr, $slot:expr) => {{
                let mut r: f32 = $s;
                r += shfl_xor_f32(r, 16);
                r += shfl_xor_f32(r, 8);
                r += shfl_xor_f32(r, 4);
                r += shfl_xor_f32(r, 2);
                r += shfl_xor_f32(r, 1);
                if lane == 0 { store_u32(dots + (4 * ($slot)) as u32, r.to_bits()); }
            }}}
            if warp == 0 {
                if !fused_q {
                    let mut off = lane * 32;
                    while off < dim {
                        core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) qrow.add(off) as u64);
                        off += 32 * 32;
                    }
                    load_pw!();
                }
                core::arch::asm!("griddepcontrol.wait;", options(nostack));
                let mut sq = 0.0f32;
                let mut j = lane;
                while j + 7 * LANES < dim {
                    let v0 = *stream.add(j);
                    let v1 = *stream.add(j + LANES);
                    let v2 = *stream.add(j + 2 * LANES);
                    let v3 = *stream.add(j + 3 * LANES);
                    let v4 = *stream.add(j + 4 * LANES);
                    let v5 = *stream.add(j + 5 * LANES);
                    let v6 = *stream.add(j + 6 * LANES);
                    let v7 = *stream.add(j + 7 * LANES);
                    sq += v0 * v0;
                    sq += v1 * v1;
                    sq += v2 * v2;
                    sq += v3 * v3;
                    sq += v4 * v4;
                    sq += v5 * v5;
                    sq += v6 * v6;
                    sq += v7 * v7;
                    j += 8 * LANES;
                }
                while j < dim { let v = *stream.add(j); sq += v * v; j += LANES; }
                sq += shfl_xor_f32(sq, 16);
                sq += shfl_xor_f32(sq, 8);
                sq += shfl_xor_f32(sq, 4);
                sq += shfl_xor_f32(sq, 2);
                sq += shfl_xor_f32(sq, 1);
                let eps = f32::from_bits(*w.add(MIX_ONE_EPSILON_AT) as u32);
                let norm = 1.0f32 / root(sq / dim as f32 + eps);
                if lane == 0 { store_u32(dots + NORM_DOT_AT, norm.to_bits()); }
                if fused_q && asked == 3 {
                    let (mut s0, mut s1, mut s2) = (0.0f32, 0.0f32, 0.0f32);
                    j = lane;
                    while j < dim {
                        let x = mul_rn!(*stream.add(j), norm);
                        s0 += x * word!(queries.add(j));
                        s1 += x * word!(queries.add(dim + j));
                        s2 += x * word!(queries.add(2 * dim + j));
                        j += LANES;
                    }
                    reduce_dot!(s0, kept - 1);
                    reduce_dot!(s1, MIX_ONE_KEPT_MAX + kept - 1);
                    reduce_dot!(s2, 2 * MIX_ONE_KEPT_MAX + kept - 1);
                } else if fused_q {
                    let mut qi = 0usize;
                    while qi < asked {
                        core::arch::asm!("/* {0} */", inout(reg64) qi, options(nomem, nostack));
                        let qp = queries.add(qi * dim);
                        let mut sum = 0.0f32;
                        j = lane;
                        while j < dim {
                            sum += mul_rn!(*stream.add(j), norm) * word!(qp.add(j));
                            j += LANES;
                        }
                        reduce_dot!(sum, qi * MIX_ONE_KEPT_MAX + kept - 1);
                        qi += 1;
                    }
                } else {
                    let mut sum = 0.0f32;
                    j = lane;
                    while j + 3 * LANES < dim {
                        let q0 = word!(qrow.add(j));
                        let q1 = word!(qrow.add(j + LANES));
                        let q2 = word!(qrow.add(j + 2 * LANES));
                        let q3 = word!(qrow.add(j + 3 * LANES));
                        let x0 = mul_rn!(*stream.add(j), norm);
                        let x1 = mul_rn!(*stream.add(j + LANES), norm);
                        let x2 = mul_rn!(*stream.add(j + 2 * LANES), norm);
                        let x3 = mul_rn!(*stream.add(j + 3 * LANES), norm);
                        sum += x0 * q0;
                        sum += x1 * q1;
                        sum += x2 * q2;
                        sum += x3 * q3;
                        j += 4 * LANES;
                    }
                    while j < dim {
                        let x = mul_rn!(*stream.add(j), norm);
                        sum += x * word!(qrow.add(j));
                        j += LANES;
                    }
                    reduce_dot!(sum, kept - 1);
                }
            } else if warp < kept {
                let lrow = (*w.add(9 + warp - 1) as *const f32).add(row_bytes_off);
                let mut off = lane * 32;
                while off < dim {
                    core::arch::asm!("prefetch.global.L2 [{a}];", a = in(reg64) lrow.add(off) as u64);
                    off += 32 * 32;
                }
                if fused_q && asked == 3 {
                    let (mut s0, mut s1, mut s2) = (0.0f32, 0.0f32, 0.0f32);
                    let mut j = lane;
                    while j < dim {
                        let x = word!(lrow.add(j));
                        s0 += x * word!(queries.add(j));
                        s1 += x * word!(queries.add(dim + j));
                        s2 += x * word!(queries.add(2 * dim + j));
                        j += LANES;
                    }
                    reduce_dot!(s0, warp - 1);
                    reduce_dot!(s1, MIX_ONE_KEPT_MAX + warp - 1);
                    reduce_dot!(s2, 2 * MIX_ONE_KEPT_MAX + warp - 1);
                } else if fused_q {
                    let mut qi = 0usize;
                    while qi < asked {
                        core::arch::asm!("/* {0} */", inout(reg64) qi, options(nomem, nostack));
                        let qp = queries.add(qi * dim);
                        let mut sum = 0.0f32;
                        let mut j = lane;
                        while j < dim {
                            sum += word!(lrow.add(j)) * word!(qp.add(j));
                            j += LANES;
                        }
                        reduce_dot!(sum, qi * MIX_ONE_KEPT_MAX + warp - 1);
                        qi += 1;
                    }
                } else {
                    let mut sum = 0.0f32;
                    let mut j = lane;
                    while j + 3 * LANES < dim {
                        let q0 = word!(qrow.add(j));
                        let x0 = word!(lrow.add(j));
                        let q1 = word!(qrow.add(j + LANES));
                        let x1 = word!(lrow.add(j + LANES));
                        let q2 = word!(qrow.add(j + 2 * LANES));
                        let x2 = word!(lrow.add(j + 2 * LANES));
                        let q3 = word!(qrow.add(j + 3 * LANES));
                        let x3 = word!(lrow.add(j + 3 * LANES));
                        sum += x0 * q0;
                        sum += x1 * q1;
                        sum += x2 * q2;
                        sum += x3 * q3;
                        j += 4 * LANES;
                    }
                    while j < dim {
                        sum += word!(lrow.add(j)) * word!(qrow.add(j));
                        j += LANES;
                    }
                    reduce_dot!(sum, warp - 1);
                    load_pw!();
                }
                core::arch::asm!("griddepcontrol.wait;", options(nostack));
                if warps <= kept {
                    copy_stage!(1usize);
                }
            } else {
                load_pw!();
                core::arch::asm!("griddepcontrol.wait;", options(nostack));
                copy_stage!(kept);
            }
            // Barrier #1: all `dots[0..q_count][0..kept]`, `norm`, and raw `stage` are ready.
            core::arch::nvptx::_syncthreads();
            macro_rules! softmax_reg {($dot_base:expr) => {{
                let scale = f32::from_bits(*w.add(7) as u32);
                let val = if lane < kept { f32::from_bits(load_u32(dots + (4 * (($dot_base) + lane)) as u32)) * scale } else { f32::NEG_INFINITY };
                let mut largest = val;
                largest = largest.max(shfl_xor_f32(largest, 4));
                largest = largest.max(shfl_xor_f32(largest, 2));
                largest = largest.max(shfl_xor_f32(largest, 1));
                let e_exp = if lane < kept { exp(val - largest) } else { 0.0f32 };
                let mut total_exp = 0.0f32;
                let mut k = 0u32;
                while k < kept as u32 {
                    let ek_bits: u32;
                    core::arch::asm!(
                        "shfl.sync.idx.b32 {d}, {v}, {s}, 31, 0xffffffff;",
                        d = out(reg32) ek_bits, v = in(reg32) e_exp.to_bits(), s = in(reg32) k,
                        options(nomem, nostack),
                    );
                    total_exp += f32::from_bits(ek_bits);
                    k += 1;
                }
                if lane < kept { (e_exp / total_exp).to_bits() } else { 0u32 }
            }}}
            macro_rules! get_wt {($bits:expr, $idx:expr)=>{{
                let wb: u32;
                core::arch::asm!(
                    "shfl.sync.idx.b32 {d}, {v}, {s}, 31, 0xffffffff;",
                    d = out(reg32) wb, v = in(reg32) $bits, s = in(reg32) ($idx as u32),
                    options(nomem, nostack),
                );
                f32::from_bits(wb)
            }}}
            if post_norm && fused_q {
                if asked == 3 {
                    // Single-pass 3-query fused blend: reads `list[0..kept-1]` and `stage` ONCE for all 3 queries,
                    // parks query-1 in the now-dead `stage` row to keep `<= 56` regs (2 blk/SM at 576t), and
                    // synchronizes all 3 RMSNorm reductions in ONE Barrier #2.
                    let wb0 = softmax_reg!(0usize);
                    let wb1 = softmax_reg!(MIX_ONE_KEPT_MAX);
                    let wb2 = softmax_reg!(2 * MIX_ONE_KEPT_MAX);
                    let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut c0, mut c1, mut c2, mut c3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let d0 = *w.add(6) as usize;
                    if warp < d0 / NORM_CHUNK {
                        let bc = warp * NORM_CHUNK + lane;
                        let k0 = *w.add(5) as usize;
                        let r_off = (core::arch::nvptx::_block_idx_y() as usize) * d0;
                        let (mut b0, mut b1, mut b2, mut b3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                        let mut e = 0usize;
                        while e + 1 < k0 {
                            core::arch::asm!("/* {0} */", inout(reg64) e, options(nomem, nostack));
                            let ptr = (*w.add(9 + e) as *const f32).add(r_off);
                            let x0 = word!(ptr.add(bc));
                            let x1 = word!(ptr.add(bc + LANES));
                            let x2 = word!(ptr.add(bc + 2 * LANES));
                            let x3 = word!(ptr.add(bc + 3 * LANES));
                            let w0 = get_wt!(wb0, e);
                            a0 += w0 * x0; a1 += w0 * x1; a2 += w0 * x2; a3 += w0 * x3;
                            let w1 = get_wt!(wb1, e);
                            b0 += w1 * x0; b1 += w1 * x1; b2 += w1 * x2; b3 += w1 * x3;
                            let w2 = get_wt!(wb2, e);
                            c0 += w2 * x0; c1 += w2 * x1; c2 += w2 * x2; c3 += w2 * x3;
                            e += 1;
                        }
                        let norm = f32::from_bits(load_u32(dots + NORM_DOT_AT));
                        let s0 = mul_rn!(f32::from_bits(load_u32(stage + (4 * bc) as u32)), norm);
                        let s1 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (bc + LANES)) as u32)), norm);
                        let s2 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (bc + 2 * LANES)) as u32)), norm);
                        let s3 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (bc + 3 * LANES)) as u32)), norm);
                        let normed = *w.add(MIX_ONE_NORMED_AT) as *mut f32;
                        if !normed.is_null() {
                            let np = normed.add(r_off);
                            *np.add(bc) = s0; *np.add(bc + LANES) = s1;
                            *np.add(bc + 2 * LANES) = s2; *np.add(bc + 3 * LANES) = s3;
                        }
                        let w0 = get_wt!(wb0, k0 - 1);
                        a0 += w0 * s0; a1 += w0 * s1; a2 += w0 * s2; a3 += w0 * s3;
                        let w1 = get_wt!(wb1, k0 - 1);
                        b0 += w1 * s0; b1 += w1 * s1; b2 += w1 * s2; b3 += w1 * s3;
                        let w2 = get_wt!(wb2, k0 - 1);
                        c0 += w2 * s0; c1 += w2 * s1; c2 += w2 * s2; c3 += w2 * s3;
                        // Park query-1 into this warp's private slice of `stage` (dead after reading `s0..s3`).
                        store_u32(stage + (4 * bc) as u32, b0.to_bits());
                        store_u32(stage + (4 * (bc + LANES)) as u32, b1.to_bits());
                        store_u32(stage + (4 * (bc + 2 * LANES)) as u32, b2.to_bits());
                        store_u32(stage + (4 * (bc + 3 * LANES)) as u32, b3.to_bits());
                        let fl = *w.add(8);
                        if (fl & MIX_ONE_SKIP_MIXED) == 0 {
                            let out_base = *w.add(2) as *mut f32;
                            let r_tot = *w.add(3) as usize;
                            let r_idx = core::arch::nvptx::_block_idx_y() as usize;
                            let tr = (fl & 1) != 0;
                            macro_rules! put_mix3 {($qi:expr, $x0:expr, $x1:expr, $x2:expr, $x3:expr) => {{
                                let orow = if tr { ($qi) * r_tot + r_idx } else { r_idx * 3 + ($qi) };
                                let op = out_base.add(orow * d0);
                                *op.add(bc) = $x0; *op.add(bc + LANES) = $x1;
                                *op.add(bc + 2 * LANES) = $x2; *op.add(bc + 3 * LANES) = $x3;
                            }}}
                            put_mix3!(0usize, a0, a1, a2, a3);
                            put_mix3!(1usize, b0, b1, b2, b3);
                            put_mix3!(2usize, c0, c1, c2, c3);
                        }
                        macro_rules! put_sq3 {($qi:expr, $x0:expr, $x1:expr, $x2:expr, $x3:expr) => {{
                            let mut sq = 0.0f32;
                            sq += ($x0) * ($x0); sq += ($x1) * ($x1); sq += ($x2) * ($x2); sq += ($x3) * ($x3);
                            sq += shfl_xor_f32(sq, 16); sq += shfl_xor_f32(sq, 8); sq += shfl_xor_f32(sq, 4);
                            sq += shfl_xor_f32(sq, 2); sq += shfl_xor_f32(sq, 1);
                            if lane == 0 { store_u32(sums + ((($qi) * NORM_CHUNKS_MAX + warp) * 4) as u32, sq.to_bits()); }
                        }}}
                        put_sq3!(0usize, a0, a1, a2, a3);
                        put_sq3!(1usize, b0, b1, b2, b3);
                        put_sq3!(2usize, c0, c1, c2, c3);
                    }
                    // Single Barrier #2 for all 3 queries.
                    core::arch::nvptx::_syncthreads();
                    let d_len = *w.add(6) as usize;
                    let chks = d_len / NORM_CHUNK;
                    if warp < chks {
                        let bc = warp * NORM_CHUNK + lane;
                        let mut qi = 0usize;
                        while qi < 3 {
                            core::arch::asm!("/* {0} */", inout(reg64) qi, options(nomem, nostack));
                            let (v0, v1, v2, v3) = if qi == 0 {
                                (a0, a1, a2, a3)
                            } else if qi == 1 {
                                (
                                    f32::from_bits(load_u32(stage + (4 * bc) as u32)),
                                    f32::from_bits(load_u32(stage + (4 * (bc + LANES)) as u32)),
                                    f32::from_bits(load_u32(stage + (4 * (bc + 2 * LANES)) as u32)),
                                    f32::from_bits(load_u32(stage + (4 * (bc + 3 * LANES)) as u32)),
                                )
                            } else {
                                (c0, c1, c2, c3)
                            };
                            let mut total = if lane < chks { f32::from_bits(load_u32(sums + ((qi * NORM_CHUNKS_MAX + lane) * 4) as u32)) } else { 0.0f32 };
                            total += shfl_xor_f32(total, 16); total += shfl_xor_f32(total, 8); total += shfl_xor_f32(total, 4);
                            total += shfl_xor_f32(total, 2); total += shfl_xor_f32(total, 1);
                            let post_eps = f32::from_bits(*w.add(MIX_ONE_POST_EPS_AT) as u32);
                            let nscale = 1.0f32 / root(total / d_len as f32 + post_eps);
                            let fl2 = *w.add(8);
                            let post_half = (fl2 & MIX_ONE_POST_HALF_OUT) != 0;
                            let r_tot = *w.add(3) as usize;
                            let r_idx = core::arch::nvptx::_block_idx_y() as usize;
                            let orow = if (fl2 & 1) != 0 { qi * r_tot + r_idx } else { r_idx * 3 + qi };
                            let post_out = *w.add(MIX_ONE_POST_OUT_AT) as *mut f32;
                            let pptr = post_out.add(if post_half { orow * (d_len / 2) } else { orow * d_len });
                            let pw_base = *w.add(MIX_ONE_POST_WEIGHT_AT) as *const f32;
                            let wp = if pw_base.is_null() { core::ptr::null() }
                                else if (fl2 & MIX_ONE_POST_WEIGHTS_PER_QUERY) != 0 { pw_base.add(qi * d_len) }
                                else { pw_base };
                            macro_rules! put_p {($col:expr, $val:expr) => {{
                                let mut y: f32 = ($val) * nscale;
                                if !wp.is_null() { y *= word!(wp.add($col)); }
                                let y = rn(y);
                                if post_half {
                                    let h = y.to_bits() >> 16;
                                    let pair = h | (shfl_xor_u32(h, 1) << 16);
                                    if (lane & 1) == 0 { *pptr.cast::<u32>().add(($col) / 2) = pair; }
                                } else { *pptr.add($col) = y; }
                            }}}
                            put_p!(bc, v0); put_p!(bc + LANES, v1); put_p!(bc + 2 * LANES, v2); put_p!(bc + 3 * LANES, v3);
                            qi += 1;
                        }
                    }
                    return;
                }
                // Generic `fused_q` fallback for `asked == 2` or `asked == 4`.
                if has_chunk {
                    let norm = f32::from_bits(load_u32(dots + NORM_DOT_AT));
                    let s0 = mul_rn!(f32::from_bits(load_u32(stage + (4 * b_chunk) as u32)), norm);
                    let s1 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + LANES)) as u32)), norm);
                    let s2 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + 2 * LANES)) as u32)), norm);
                    let s3 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + 3 * LANES)) as u32)), norm);
                    store_u32(stage + (4 * b_chunk) as u32, s0.to_bits());
                    store_u32(stage + (4 * (b_chunk + LANES)) as u32, s1.to_bits());
                    store_u32(stage + (4 * (b_chunk + 2 * LANES)) as u32, s2.to_bits());
                    store_u32(stage + (4 * (b_chunk + 3 * LANES)) as u32, s3.to_bits());
                    let np_raw = *w.add(MIX_ONE_NORMED_AT) as *mut f32;
                    if !np_raw.is_null() {
                        let np = np_raw.add(row_bytes_off);
                        *np.add(b_chunk) = s0; *np.add(b_chunk + LANES) = s1;
                        *np.add(b_chunk + 2 * LANES) = s2; *np.add(b_chunk + 3 * LANES) = s3;
                    }
                }
                let mut cur_q = 0usize;
                while cur_q < asked {
                    core::arch::asm!("/* {0} */", inout(reg64) cur_q, options(nomem, nostack));
                    let fl = *w.add(8);
                    let (k_cnt, d_len) = (*w.add(5) as usize, *w.add(6) as usize);
                    let r_idx = core::arch::nvptx::_block_idx_y() as usize;
                    let r_off = r_idx * d_len;
                    let my_w_bits = softmax_reg!(cur_q * MIX_ONE_KEPT_MAX);
                    let (mut v0, mut v1, mut v2, mut v3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    if warp < d_len / NORM_CHUNK {
                        let bc = warp * NORM_CHUNK + lane;
                        let mut e = 0usize;
                        while e + 1 < k_cnt {
                            core::arch::asm!("/* {0} */", inout(reg64) e, options(nomem, nostack));
                            let wt = get_wt!(my_w_bits, e);
                            let ptr = (*w.add(9 + e) as *const f32).add(r_off);
                            v0 += wt * word!(ptr.add(bc)); v1 += wt * word!(ptr.add(bc + LANES));
                            v2 += wt * word!(ptr.add(bc + 2 * LANES)); v3 += wt * word!(ptr.add(bc + 3 * LANES));
                            e += 1;
                        }
                        let wt_last = get_wt!(my_w_bits, k_cnt - 1);
                        v0 += wt_last * f32::from_bits(load_u32(stage + (4 * bc) as u32));
                        v1 += wt_last * f32::from_bits(load_u32(stage + (4 * (bc + LANES)) as u32));
                        v2 += wt_last * f32::from_bits(load_u32(stage + (4 * (bc + 2 * LANES)) as u32));
                        v3 += wt_last * f32::from_bits(load_u32(stage + (4 * (bc + 3 * LANES)) as u32));
                        if (fl & MIX_ONE_SKIP_MIXED) == 0 {
                            let (r_tot, a_tot) = (*w.add(3) as usize, *w.add(4) as usize);
                            let out_row = if (fl & 1) != 0 { cur_q * r_tot + r_idx } else { r_idx * a_tot + cur_q };
                            let out_ptr = (*w.add(2) as *mut f32).add(out_row * d_len);
                            *out_ptr.add(bc) = v0; *out_ptr.add(bc + LANES) = v1;
                            *out_ptr.add(bc + 2 * LANES) = v2; *out_ptr.add(bc + 3 * LANES) = v3;
                        }
                        let mut sq = 0.0f32;
                        sq += v0 * v0; sq += v1 * v1; sq += v2 * v2; sq += v3 * v3;
                        sq += shfl_xor_f32(sq, 16); sq += shfl_xor_f32(sq, 8); sq += shfl_xor_f32(sq, 4);
                        sq += shfl_xor_f32(sq, 2); sq += shfl_xor_f32(sq, 1);
                        if lane == 0 { store_u32(sums + (warp * 4) as u32, sq.to_bits()); }
                    }
                    core::arch::nvptx::_syncthreads();
                    let d_len2 = *w.add(6) as usize;
                    let chks = d_len2 / NORM_CHUNK;
                    if warp < chks {
                        let bc = warp * NORM_CHUNK + lane;
                        let mut total = if lane < chks { f32::from_bits(load_u32(sums + (lane * 4) as u32)) } else { 0.0f32 };
                        total += shfl_xor_f32(total, 16); total += shfl_xor_f32(total, 8); total += shfl_xor_f32(total, 4);
                        total += shfl_xor_f32(total, 2); total += shfl_xor_f32(total, 1);
                        let post_eps = f32::from_bits(*w.add(MIX_ONE_POST_EPS_AT) as u32);
                        let nscale = 1.0f32 / root(total / d_len2 as f32 + post_eps);
                        let fl2 = *w.add(8);
                        let post_half = (fl2 & MIX_ONE_POST_HALF_OUT) != 0;
                        let (r_tot, a_tot) = (*w.add(3) as usize, *w.add(4) as usize);
                        let r_idx2 = core::arch::nvptx::_block_idx_y() as usize;
                        let out_row = if (fl2 & 1) != 0 { cur_q * r_tot + r_idx2 } else { r_idx2 * a_tot + cur_q };
                        let post_out = *w.add(MIX_ONE_POST_OUT_AT) as *mut f32;
                        let post_ptr = post_out.add(if post_half { out_row * (d_len2 / 2) } else { out_row * d_len2 });
                        let pw_base = *w.add(MIX_ONE_POST_WEIGHT_AT) as *const f32;
                        let wp = if pw_base.is_null() { core::ptr::null() }
                            else if (fl2 & MIX_ONE_POST_WEIGHTS_PER_QUERY) != 0 { pw_base.add(cur_q * d_len2) }
                            else { pw_base };
                        macro_rules! put_post {($col:expr, $val:expr) => {{
                            let mut y: f32 = $val * nscale;
                            if !wp.is_null() { y *= word!(wp.add($col)); }
                            let y = rn(y);
                            if post_half {
                                let h = y.to_bits() >> 16;
                                let pair = h | (shfl_xor_u32(h, 1) << 16);
                                if (lane & 1) == 0 { *post_ptr.cast::<u32>().add(($col) / 2) = pair; }
                            } else { *post_ptr.add($col) = y; }
                        }}}
                        put_post!(bc, v0); put_post!(bc + LANES, v1); put_post!(bc + 2 * LANES, v2); put_post!(bc + 3 * LANES, v3);
                    }
                    cur_q += 1;
                    if cur_q < asked { core::arch::nvptx::_syncthreads(); }
                }
                return;
            }
            let norm = f32::from_bits(load_u32(dots + NORM_DOT_AT));
            let normed = *w.add(MIX_ONE_NORMED_AT) as *mut f32;
            let write_normed = bx == 0 && !normed.is_null();
            let transpose = (flags & 1) != 0;
            let my_w_bits = softmax_reg!(0usize);
            let out_row = if transpose { q * rows + row } else { row * asked + q };
            if post_norm {
                let post_eps = f32::from_bits(*w.add(MIX_ONE_POST_EPS_AT) as u32);
                let out_ptr = if skip_mixed { core::ptr::null_mut() } else { (*w.add(2) as *mut f32).add(out_row * dim) };
                let post_out = *w.add(MIX_ONE_POST_OUT_AT) as *mut f32;
                let (mut v0, mut v1, mut v2, mut v3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                if has_chunk {
                    let mut e = 0usize;
                    while e + 1 < kept {
                        core::arch::asm!("/* {0} */", inout(reg64) e, options(nomem, nostack));
                        let wt = get_wt!(my_w_bits, e);
                        let ptr = (*w.add(9 + e) as *const f32).add(row_bytes_off);
                        v0 += wt * word!(ptr.add(b_chunk));
                        v1 += wt * word!(ptr.add(b_chunk + LANES));
                        v2 += wt * word!(ptr.add(b_chunk + 2 * LANES));
                        v3 += wt * word!(ptr.add(b_chunk + 3 * LANES));
                        e += 1;
                    }
                    let wt_last = get_wt!(my_w_bits, kept - 1);
                    let s0 = mul_rn!(f32::from_bits(load_u32(stage + (4 * b_chunk) as u32)), norm);
                    let s1 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + LANES)) as u32)), norm);
                    let s2 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + 2 * LANES)) as u32)), norm);
                    let s3 = mul_rn!(f32::from_bits(load_u32(stage + (4 * (b_chunk + 3 * LANES)) as u32)), norm);
                    if write_normed {
                        *normed.add(row_bytes_off + b_chunk) = s0;
                        *normed.add(row_bytes_off + b_chunk + LANES) = s1;
                        *normed.add(row_bytes_off + b_chunk + 2 * LANES) = s2;
                        *normed.add(row_bytes_off + b_chunk + 3 * LANES) = s3;
                    }
                    v0 += wt_last * s0;
                    v1 += wt_last * s1;
                    v2 += wt_last * s2;
                    v3 += wt_last * s3;
                    if !skip_mixed {
                        *out_ptr.add(b_chunk) = v0;
                        *out_ptr.add(b_chunk + LANES) = v1;
                        *out_ptr.add(b_chunk + 2 * LANES) = v2;
                        *out_ptr.add(b_chunk + 3 * LANES) = v3;
                    }
                    let mut sq = 0.0f32;
                    sq += v0 * v0;
                    sq += v1 * v1;
                    sq += v2 * v2;
                    sq += v3 * v3;
                    sq += shfl_xor_f32(sq, 16);
                    sq += shfl_xor_f32(sq, 8);
                    sq += shfl_xor_f32(sq, 4);
                    sq += shfl_xor_f32(sq, 2);
                    sq += shfl_xor_f32(sq, 1);
                    if lane == 0 { store_u32(sums + (warp * 4) as u32, sq.to_bits()); }
                }
                core::arch::nvptx::_syncthreads();
                if has_chunk {
                    let mut total = if lane < chunks { f32::from_bits(load_u32(sums + (lane * 4) as u32)) } else { 0.0f32 };
                    total += shfl_xor_f32(total, 16);
                    total += shfl_xor_f32(total, 8);
                    total += shfl_xor_f32(total, 4);
                    total += shfl_xor_f32(total, 2);
                    total += shfl_xor_f32(total, 1);
                    let nscale = 1.0f32 / root(total / dim as f32 + post_eps);
                    let post_half = (flags & MIX_ONE_POST_HALF_OUT) != 0;
                    let post_ptr = post_out.add(if post_half { out_row * (dim / 2) } else { out_row * dim });
                    macro_rules! put_post {
                        ($col:expr, $val:expr) => {{
                            let y: f32 = $val;
                            if post_half {
                                let h = y.to_bits() >> 16;
                                let pair = h | (shfl_xor_u32(h, 1) << 16);
                                if (lane & 1) == 0 { *post_ptr.cast::<u32>().add(($col) / 2) = pair; }
                            } else {
                                *post_ptr.add($col) = y;
                            }
                        }};
                    }
                    if pw_ptr.is_null() {
                        put_post!(b_chunk, rn(v0 * nscale));
                        put_post!(b_chunk + LANES, rn(v1 * nscale));
                        put_post!(b_chunk + 2 * LANES, rn(v2 * nscale));
                        put_post!(b_chunk + 3 * LANES, rn(v3 * nscale));
                    } else {
                        put_post!(b_chunk, rn((v0 * nscale) * pw0));
                        put_post!(b_chunk + LANES, rn((v1 * nscale) * pw1));
                        put_post!(b_chunk + 2 * LANES, rn((v2 * nscale) * pw2));
                        put_post!(b_chunk + 3 * LANES, rn((v3 * nscale) * pw3));
                    }
                }
                return;
            }
            let out_ptr = (*w.add(2) as *mut f32).add(out_row * dim);
            let wt_last = get_wt!(my_w_bits, kept - 1);
            let mut ch = (bx % parts) * threads + tid;
            while ch < dim {
                let mut sum = 0.0f32;
                let mut i = 0usize;
                while i + 1 < kept {
                    let wt = get_wt!(my_w_bits, i);
                    sum += wt * word!((*w.add(9 + i) as *const f32).add(row_bytes_off + ch));
                    i += 1;
                }
                let sn = mul_rn!(f32::from_bits(load_u32(stage + (4 * ch) as u32)), norm);
                if write_normed { *normed.add(row_bytes_off + ch) = sn; }
                sum += wt_last * sn;
                *out_ptr.add(ch) = sum;
                ch += parts * threads;
            }
            return;
        }
        core::arch::asm!("griddepcontrol.wait;", options(nostack));
        core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
        let scores = *w as *const f32;
        let joined = *w.add(1) as *const f32;
        let output = *w.add(2) as *mut f32;
        let rows = *w.add(3) as usize;
        let asked = *w.add(4) as usize;
        let kept = *w.add(5) as usize;
        let dim = *w.add(6) as usize;
        let scale = f32::from_bits(*w.add(7) as u32);
        let transpose = *w.add(8) & 1 != 0;
        let fused = *w.add(8) & MIX_DOTS != 0;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        // Block b is part b % parts of the (row, query) b / parts (`fast_ops::mix_blend_parts`).
        let parts = if joined.is_null() { 1 } else { mix_blend_parts(dim) };
        let (block, part) = (core::arch::nvptx::_block_idx_x() as usize / parts, core::arch::nvptx::_block_idx_x() as usize % parts);
        if !mix_blend_admitted(rows, asked, kept, dim, threads) || block >= rows * asked || core::arch::nvptx::_grid_dim_x() as usize != rows * asked * parts { core::arch::nvptx::trap(); }
        if ((scores as usize) | (joined as usize) | (output as usize)) & 3 != 0 || (fused && (!joined.is_null() || threads % 32 != 0)) {
            core::arch::nvptx::trap();
        }

        let row = block / asked;
        let q = block % asked;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let smem = crate::ptx::shared::<WEIGHT_SMEM_BYTES>();

        if fused {
            // The dots of query row `q` with the `kept` streams of `row`: the chain of `replay_row_dots` (lane `l` adds
            // `stream[j] * query[j]` for j = l, l + 32, .. in rising order from +0.0, then the butterfly 16, 8, 4, 2, 1),
            // one warp a stream. The loop bounds are the same for every lane of a warp, so all 32 lanes meet each shuffle.
            let queries = scores.add(q * dim);
            let (warp, lane, warps) = (tid >> 5, tid & 31, threads >> 5);
            let mut e = warp;
            while e < kept {
                let v_ptr = (*w.add(9 + e) as *const f32).add(row * dim);
                let mut sum = 0.0f32;
                let mut j = lane;
                while j < dim {
                    sum += *v_ptr.add(j) * *queries.add(j);
                    j += 32;
                }
                sum += shfl_xor_f32(sum, 16);
                sum += shfl_xor_f32(sum, 8);
                sum += shfl_xor_f32(sum, 4);
                sum += shfl_xor_f32(sum, 2);
                sum += shfl_xor_f32(sum, 1);
                if lane == 0 { store_u32(smem + (e * 4) as u32, sum.to_bits()); }
                e += warps;
            }
            core::arch::nvptx::_syncthreads();
        }
        if tid == 0 {
            // The score of stream `i`: the dot in shared word `i` (`MIX_DOTS`; word `i` is read before it is written
            // below) or the score word of the dots launch.
            let s_ptr = scores.add((row * asked + q) * kept);
            let score = |i: usize| if fused { f32::from_bits(load_u32(smem + (i * 4) as u32)) } else { *s_ptr.add(i) };
            let mut largest = f32::NEG_INFINITY;
            let mut i = 0usize;
            while i < kept {
                let v = score(i) * scale;
                largest = largest.max(v);
                i += 1;
            }
            let mut total = 0.0f32;
            i = 0;
            while i < kept {
                let e = exp(score(i) * scale - largest);
                store_u32(smem + (i * 4) as u32, e.to_bits());
                total += e;
                i += 1;
            }
            i = 0;
            while i < kept {
                let e = f32::from_bits(load_u32(smem + (i * 4) as u32));
                store_u32(smem + (i * 4) as u32, (e / total).to_bits());
                i += 1;
            }
        }
        core::arch::nvptx::_syncthreads();

        // The list mix with the gate norm of each blended row (`fast_ops::MIX_LIST_NORM`, recorded by
        // `Backend::fast_mix_list_normed`; prefill messages): the blend words of the list form below, and the words of
        // `fast_norm`'s rounded wide form with no weight on them. Warp `w` blends the norm chunks `w, w + 8, ..` of the
        // row: lane `l` the places `chunk * 128 + l + 32 * g` (g < 4), each `sum += weight * word` in list order from
        // +0.0. A chunk's sum is `fast_norm`'s (`+=` of the four squares from +0.0, then the butterfly 16, 8, 4, 2, 1),
        // warp 0 joins the chunk sums as `fast_norm` does (lane `c` holds chunk `c`, 0 past the last), and every place
        // is written twice: the blend word, and `rn(word * scale)` with `scale = 1 / root(total / dim + eps)`.
        // SAFETY:
        // - The host (`Backend::fast_mix_list_normed`) admitted the list form (null joined pointer), the transposed output,
        //   `dim` a multiple of 128 with `dim / 128 <= NORM_CHUNKS_MAX` and 256 threads (`mix_list_norm_admitted`,
        //   re-checked here), and made the normed output as a new private vector of `rows * asked * dim` words, so it
        //   meets no input and not the output. The weights are the softmax above (of the block's dots with `MIX_DOTS`,
        //   of the dots launch's scores otherwise). Every read is the list form's: list row `e < kept` at
        //   `row * dim + j`, `j < dim`.
        // - Lane `l` of warp `w` writes the places `(w + 8 * i) * 128 + l + 32 * g` (< dim) of its row of each output:
        //   one writer a word; it reads back only output words it wrote itself, after writing them. Chunk sum `c` is
        //   written by lane 0 of the one warp that owns chunk `c` before the
        //   barrier and read by warp 0 after it; the scale slot is written by lane 0 of warp 0 before the next barrier
        //   and read after it. Every barrier is the convergent `_syncthreads` reached by all threads (no return before
        //   it), and each shuffle runs with the whole warp (the chunk test is the same for the lanes of a warp).
        if *w.add(8) & crate::fast_ops::MIX_LIST_NORM != 0 {
            let normed = *w.add(crate::fast_ops::MIX_LIST_NORMED_AT) as *mut f32;
            let eps = f32::from_bits(*w.add(crate::fast_ops::MIX_LIST_EPSILON_AT) as u32);
            if !joined.is_null() || !transpose || !crate::fast_ops::mix_list_norm_admitted(dim, threads)
                || ((normed as usize) | (output as usize)) & 3 != 0 { core::arch::nvptx::trap(); }
            let (warp, lane) = (tid / 32, tid % 32);
            let chunks = dim / NORM_CHUNK;
            let base_j = row * dim;
            let at = (q * rows + row) * dim;
            if dim <= MIX_ONE_DIM_MAX {
                // Stage the blended row in `stage` (10,240 B shared memory already allocated for `MIX_ONE`) so
                // `MIX_LIST_NORM` does zero global readback while keeping register count <= 64 (4 blocks/SM).
                macro_rules! word {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
                let mut i = 0usize;
                while i < LIST_NORM_CHUNKS {
                    let ch = warp + LIST_NORM_WARPS * i;
                    if ch < chunks {
                        let mut sum = 0.0f32;
                        let mut pair = 0usize;
                        while pair < 2 {
                            let j = ch * NORM_CHUNK + lane + 2 * LANES * pair;
                            let (mut x0, mut x1) = (0.0f32, 0.0f32);
                            let mut e = 0usize;
                            while e < kept {
                                core::arch::asm!("/* {0} */", inout(reg64) e, options(nomem, nostack));
                                let wt = f32::from_bits(load_u32(smem + (4 * e) as u32));
                                let src = (*w.add(9 + e) as *const f32).add(base_j + j);
                                let (l0, l1) = (word!(src), word!(src.add(LANES)));
                                x0 += wt * l0;
                                x1 += wt * l1;
                                e += 1;
                            }
                            let out = output.add(at + j);
                            *out = x0;
                            *out.add(LANES) = x1;
                            store_u32(stage + (4 * j) as u32, x0.to_bits());
                            store_u32(stage + (4 * (j + LANES)) as u32, x1.to_bits());
                            sum += x0 * x0;
                            sum += x1 * x1;
                            pair += 1;
                        }
                        sum += shfl_xor_f32(sum, 16);
                        sum += shfl_xor_f32(sum, 8);
                        sum += shfl_xor_f32(sum, 4);
                        sum += shfl_xor_f32(sum, 2);
                        sum += shfl_xor_f32(sum, 1);
                        if lane == 0 { store_u32(sums + (ch * 4) as u32, sum.to_bits()); }
                    }
                    i += 1;
                }
                core::arch::nvptx::_syncthreads();
                if warp == 0 {
                    let mut total = if lane < chunks { f32::from_bits(load_u32(sums + (lane * 4) as u32)) } else { 0.0f32 };
                    total += shfl_xor_f32(total, 16);
                    total += shfl_xor_f32(total, 8);
                    total += shfl_xor_f32(total, 4);
                    total += shfl_xor_f32(total, 2);
                    total += shfl_xor_f32(total, 1);
                    if lane == 0 { store_u32(sums + (NORM_CHUNKS_MAX * 4) as u32, (1.0f32 / root(total / dim as f32 + eps)).to_bits()); }
                }
                core::arch::nvptx::_syncthreads();
                let scale = f32::from_bits(load_u32(sums + (NORM_CHUNKS_MAX * 4) as u32));
                let normed_half = (*w.add(8) & MIX_LIST_NORM_HALF) != 0;
                let dst_base = normed.add(if normed_half { at / 2 } else { at });
                i = 0;
                while i < LIST_NORM_CHUNKS {
                    let ch = warp + LIST_NORM_WARPS * i;
                    if ch < chunks {
                        let j = ch * NORM_CHUNK + lane;
                        let x0 = f32::from_bits(load_u32(stage + (4 * j) as u32));
                        let x1 = f32::from_bits(load_u32(stage + (4 * (j + LANES)) as u32));
                        let x2 = f32::from_bits(load_u32(stage + (4 * (j + 2 * LANES)) as u32));
                        let x3 = f32::from_bits(load_u32(stage + (4 * (j + 3 * LANES)) as u32));
                        let (y0, y1, y2, y3) = (rn(x0 * scale), rn(x1 * scale), rn(x2 * scale), rn(x3 * scale));
                        if normed_half {
                            let (h0, h1, h2, h3) = (y0.to_bits() >> 16, y1.to_bits() >> 16, y2.to_bits() >> 16, y3.to_bits() >> 16);
                            let p0 = h0 | (shfl_xor_u32(h0, 1) << 16);
                            let p1 = h1 | (shfl_xor_u32(h1, 1) << 16);
                            let p2 = h2 | (shfl_xor_u32(h2, 1) << 16);
                            let p3 = h3 | (shfl_xor_u32(h3, 1) << 16);
                            if (lane & 1) == 0 {
                                let dst_u32 = dst_base.cast::<u32>();
                                *dst_u32.add(j / 2) = p0;
                                *dst_u32.add((j + LANES) / 2) = p1;
                                *dst_u32.add((j + 2 * LANES) / 2) = p2;
                                *dst_u32.add((j + 3 * LANES) / 2) = p3;
                            }
                        } else {
                            let dst = dst_base.add(j);
                            *dst = y0;
                            *dst.add(LANES) = y1;
                            *dst.add(2 * LANES) = y2;
                            *dst.add(3 * LANES) = y3;
                        }
                    }
                    i += 1;
                }
                return;
            }
            let mut i = 0usize;
            while i < LIST_NORM_CHUNKS {
                let ch = warp + LIST_NORM_WARPS * i;
                if ch < chunks {
                    let mut sum = 0.0f32;
                    let mut pair = 0usize;
                    while pair < 2 {
                        let j = ch * NORM_CHUNK + lane + 2 * LANES * pair;
                        let (mut x0, mut x1) = (0.0f32, 0.0f32);
                        let mut e = 0usize;
                        while e < kept {
                            core::arch::asm!("/* {0} */", inout(reg64) e, options(nomem, nostack));
                            let wt = f32::from_bits(load_u32(smem + (4 * e) as u32));
                            let src = (*w.add(9 + e) as *const f32).add(base_j + j);
                            let (l0, l1) = (*src, *src.add(LANES));
                            x0 += wt * l0;
                            x1 += wt * l1;
                            e += 1;
                        }
                        let out = output.add(at + j);
                        *out = x0;
                        *out.add(LANES) = x1;
                        sum += x0 * x0;
                        sum += x1 * x1;
                        pair += 1;
                    }
                    sum += shfl_xor_f32(sum, 16);
                    sum += shfl_xor_f32(sum, 8);
                    sum += shfl_xor_f32(sum, 4);
                    sum += shfl_xor_f32(sum, 2);
                    sum += shfl_xor_f32(sum, 1);
                    if lane == 0 { store_u32(sums + (ch * 4) as u32, sum.to_bits()); }
                }
                i += 1;
            }
            core::arch::nvptx::_syncthreads();
            if warp == 0 {
                let mut total = if lane < chunks { f32::from_bits(load_u32(sums + (lane * 4) as u32)) } else { 0.0f32 };
                total += shfl_xor_f32(total, 16);
                total += shfl_xor_f32(total, 8);
                total += shfl_xor_f32(total, 4);
                total += shfl_xor_f32(total, 2);
                total += shfl_xor_f32(total, 1);
                if lane == 0 { store_u32(sums + (NORM_CHUNKS_MAX * 4) as u32, (1.0f32 / root(total / dim as f32 + eps)).to_bits()); }
            }
            core::arch::nvptx::_syncthreads();
            let scale = f32::from_bits(load_u32(sums + (NORM_CHUNKS_MAX * 4) as u32));
            i = 0;
            while i < LIST_NORM_CHUNKS {
                let ch = warp + LIST_NORM_WARPS * i;
                if ch < chunks {
                    let j = at + ch * NORM_CHUNK + lane;
                    let (src, dst) = (output.add(j), normed.add(j));
                    let x0 = core::ptr::read_volatile(src);
                    let x1 = core::ptr::read_volatile(src.add(LANES));
                    let x2 = core::ptr::read_volatile(src.add(2 * LANES));
                    let x3 = core::ptr::read_volatile(src.add(3 * LANES));
                    *dst = rn(x0 * scale);
                    *dst.add(LANES) = rn(x1 * scale);
                    *dst.add(2 * LANES) = rn(x2 * scale);
                    *dst.add(3 * LANES) = rn(x3 * scale);
                }
                i += 1;
            }
            return;
        }

        let out_row = if transpose { q * rows + row } else { row * asked + q };
        let out_ptr = output.add(out_row * dim);
        let row_stride = rows * dim;
        let base_j = row * dim;
        let mut ch = part * threads + tid;
        while ch < dim {
            let mut sum = 0.0f32;
            let mut i = 0usize;
            while i < kept {
                core::arch::asm!("/* {0} */", inout(reg64) i, options(nomem, nostack));
                let wt = f32::from_bits(load_u32(smem + (i * 4) as u32));
                let stream = if joined.is_null() { *w.add(9 + i) as *const f32 } else { joined.add(i * row_stride) };
                sum += wt * *stream.add(base_j + ch);
                i += 1;
            }
            *out_ptr.add(ch) = sum;
            ch += parts * threads;
        }
    }
}

/// Replay one recorded command of kind `FAST_ROW_DOTS`.
/// SAFETY (caller contract, `resident_launch_fast_row_dots`): `table[index]` is a valid `FAST_ROW_DOTS`
/// command recorded by `Backend::fast_row_dots_into`, `Backend::fast_mix_list`, `Backend::fast_mix_rows` or
/// `Backend::fast_short_conv_kv_into` after operand checks.
#[inline(always)]
pub(crate) unsafe fn replay_row_dots(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host verified via `fast_ops::check_row_dots` that `joined` holds `kept * rows * k` FP32 words,
    //   `queries` holds `asked * k` FP32 words, `output` holds `rows * asked * kept` FP32 words and is
    //   disjoint from inputs, and all pointers are word-aligned.
    // - A null joined pointer selects `kept` checked pointers in the command payload, one per
    //   immutable `rows * k`-word stream. All streams are disjoint from the private output.
    // - Device re-checks `row_dots_admitted`, grid bounds (`block < rows * kept`), and alignment.
    // - All 32 lanes of each warp participate in every `shfl.sync.bfly.b32`; each `(row, q, e)` has one writer.
    // - A KV finish has the checked read spans of the prefill window convolution with separate ring and no addend.
    //   Every warp owns one head in a row of the packed plane, disjoint from all its inputs and every other head.
    //   Even lanes write disjoint pairs. No incomplete warp reaches a shuffle.
    unsafe {
        // SAFETY: prefetch hints on `table[index]` and the commands after it in the stage buffer (`warm_command!`).
        crate::fast_elem::warm_command!(table, index);
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_ROW_DOTS { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        // Multi-row paired KV finish (`*w == KV_FINISH_TAG && *w.add(1) == 0`): one warp per `(side, row, kv_head)`
        // computes `ShortConv` (from `ring + inputs`, optionally writing `every`), `HeadClip` (key) or `RoundedNorm`
        // (value) for `head_dim == 128`, and register-level `PackBf16` via `shfl_xor(v, 1)`. Zero shared memory and
        // zero block barriers. Host verified all extents, alignments, and non-overlap via `check_kv_finish_pair`.
        if *w == KV_FINISH_TAG && *w.add(1) == 0 {
            core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
            let rows = *w.add(2) as usize;
            let kv_heads = *w.add(3) as usize;
            let taps = *w.add(4) as usize;
            let first_slot = *w.add(5) as usize;
            let threads = core::arch::nvptx::_block_dim_x() as usize;
            let total_warps = 2 * rows * kv_heads;
            let warps_per_block = threads / 32;
            if !kv_finish_admitted(rows, kv_heads, taps, threads) || first_slot >= taps
                || core::arch::nvptx::_grid_dim_x() as usize != total_warps.div_ceil(warps_per_block)
            {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let (warp, lane) = (tid >> 5, tid & 31);
            let w_idx = (core::arch::nvptx::_block_idx_x() as usize) * warps_per_block + warp;
            if w_idx >= total_warps { return; }
            let is_val = w_idx >= rows * kv_heads;
            let idx = if is_val { w_idx - rows * kv_heads } else { w_idx };
            let row = idx / kv_heads;
            let head = idx % kv_heads;
            let base = if is_val { 14 } else { 8 };
            let inputs = *w.add(base) as *const f32;
            // SAFETY: the host checks each inline chat ring against all output spans and
            // validates its slot. Word 20 selects one triple (key, value, slot) per row.
            // When `chats == 0 && rows == 1` and word 21 (`step_ptr`) is non-zero, the recorder upgraded a
            // checked single-row `conv_norm::pair` to read `own = *step_ptr as usize`, update `ring[own]` in place,
            // and also store unpacked outputs at word 22/23 (0 for none).
            let chats = *w.add(20) as usize;
            if chats != 0 && (chats != rows || chats > 16) { core::arch::nvptx::trap(); }
            let step_ptr = if chats == 0 && rows == 1 { *w.add(21) as *const f32 } else { core::ptr::null() };
            let unpacked_out = if !step_ptr.is_null() {
                *w.add(if is_val { 23 } else { 22 }) as *mut f32
            } else {
                core::ptr::null_mut()
            };
            let (ring, own, preceding) = if chats == 0 {
                let own = if !step_ptr.is_null() {
                    if (step_ptr as usize) & 3 != 0 { core::arch::nvptx::trap(); }
                    let sf = *step_ptr;
                    let s = sf as usize;
                    if s >= taps || s as f32 != sf { core::arch::nvptx::trap(); }
                    s
                } else {
                    (first_slot + row) % taps
                };
                (*w.add(base + 1) as *const f32, own, if !step_ptr.is_null() { 0 } else { row })
            } else {
                let first = *w.add(23 + 3 * row) as usize;
                if first >= taps { core::arch::nvptx::trap(); }
                (*w.add(21 + 3 * row + is_val as usize) as *const f32, first, 0)
            };
            let fixed = *w.add(base + 2) as *const f32;
            let made = *w.add(base + 3) as *const f32;
            let packed_out = *w.add(base + 4) as *mut f32;
            let every_out = *w.add(base + 5) as *mut f32;
            if inputs.is_null() || ring.is_null() || fixed.is_null() || packed_out.is_null()
                || ((inputs as usize) | (ring as usize) | (fixed as usize) | (made as usize)
                    | (packed_out as usize) | (every_out as usize) | (unpacked_out as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let shift = if is_val { 32 } else { 0 };
            let param = f32::from_bits((*w.add(6) >> shift) as u32);
            let conv_scale = f32::from_bits((*w.add(7) >> shift) as u32);
            let width = kv_heads * KV_FINISH_HEAD_DIM;
            let tok_ptr = if made.is_null() { made } else { made.add((row * kv_heads + head) * taps) };
            let (y0, y1, y2, y3) = if taps == 4 && preceding == 0 && every_out.is_null() && (fixed as usize) & 15 == 0 {
                let (tk0, tk1, tk2, tk3) = if tok_ptr.is_null() {
                    (0.0f32, 0.0f32, 0.0f32, 0.0f32)
                } else {
                    (rn(*tok_ptr), rn(*tok_ptr.add(1)), rn(*tok_ptr.add(2)), rn(*tok_ptr.add(3)))
                };
                let r1_ptr = ring.add(((own + 3) & 3) * width);
                let r2_ptr = ring.add(((own + 2) & 3) * width);
                let r3_ptr = ring.add(((own + 1) & 3) * width);
                let own_ptr = (ring as *mut f32).add(own * width);
                let in_ptr = inputs.add(row * width);
                macro_rules! conv4 {($k:literal)=>{{
                    let ch = head * KV_FINISH_HEAD_DIM + $k * LANES + lane;
                    let ft = crate::tc_ptx::ld_global_nc_v4(fixed.add(ch * 4));
                    let x3 = *r3_ptr.add(ch);
                    let x2 = *r2_ptr.add(ch);
                    let x1 = *r1_ptr.add(ch);
                    let cur_raw = *in_ptr.add(ch);
                    if !step_ptr.is_null() { *own_ptr.add(ch) = cur_raw; }
                    let mut sum = 0.0f32;
                    sum += (if tok_ptr.is_null() { ft[3] } else { ft[3] + tk3 }) * rn(x3);
                    sum += (if tok_ptr.is_null() { ft[2] } else { ft[2] + tk2 }) * rn(x2);
                    sum += (if tok_ptr.is_null() { ft[1] } else { ft[1] + tk1 }) * rn(x1);
                    let cur = rn(cur_raw);
                    sum += (if tok_ptr.is_null() { ft[0] } else { ft[0] + tk0 }) * cur;
                    rn((sum + cur) * conv_scale)
                }};}
                (conv4!(0), conv4!(1), conv4!(2), conv4!(3))
            } else {
                macro_rules! conv_ch {($k:literal)=>{{
                    let ch = head * KV_FINISH_HEAD_DIM + $k * LANES + lane;
                    let mut sum = 0.0f32;
                    let mut lag = taps;
                    while lag > 0 {
                        lag -= 1;
                        let mut tap = *fixed.add(ch * taps + lag);
                        if !tok_ptr.is_null() {
                            tap += rn(*tok_ptr.add(lag));
                        }
                        let read = if lag <= preceding {
                            *inputs.add((row - lag) * width + ch)
                        } else {
                            *ring.add(((own + taps - lag) % taps) * width + ch)
                        };
                        if !every_out.is_null() {
                            *every_out.add((row * taps + (own + taps - lag) % taps) * width + ch) = read;
                        }
                        sum += tap * rn(read);
                    }
                    let cur_raw = *inputs.add(row * width + ch);
                    if !step_ptr.is_null() {
                        *(ring as *mut f32).add(own * width + ch) = cur_raw;
                    }
                    let cur = rn(cur_raw);
                    rn((sum + cur) * conv_scale)
                }};}
                (conv_ch!(0), conv_ch!(1), conv_ch!(2), conv_ch!(3))
            };
            let mut sq = 0.0f32;
            sq += y0 * y0;
            sq += y1 * y1;
            sq += y2 * y2;
            sq += y3 * y3;
            sq += shfl_xor_f32(sq, 16);
            sq += shfl_xor_f32(sq, 8);
            sq += shfl_xor_f32(sq, 4);
            sq += shfl_xor_f32(sq, 2);
            sq += shfl_xor_f32(sq, 1);
            let scale = if is_val {
                1.0f32 / root(sq / 128.0f32 + param)
            } else {
                let bits: u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",
                    got = out(reg32) bits, v = in(reg32) sq.to_bits());
                crate::block_attention::clip_factor(root(f32::from_bits(bits)), param)
            };
            let out_base = packed_out.add(row * (width / 2) + head * 64);
            macro_rules! pack_ch {($k:literal, $yk:expr)=>{{
                let val = rn($yk * scale);
                if !unpacked_out.is_null() {
                    *unpacked_out.add(head * KV_FINISH_HEAD_DIM + $k * LANES + lane) = val;
                }
                let hi = shfl_xor_f32(val, 1);
                if (lane & 1) == 0 {
                    let lb = val.to_bits();
                    let hb = hi.to_bits();
                    let lo16 = lb.wrapping_add(0x7fff + ((lb >> 16) & 1)) >> 16;
                    let hi16 = hb.wrapping_add(0x7fff + ((hb >> 16) & 1)) >> 16;
                    *out_base.add($k * 16 + (lane >> 1)) = f32::from_bits(lo16 | (hi16 << 16));
                }
            }};}
            pack_ch!(0, y0);
            pack_ch!(1, y1);
            pack_ch!(2, y2);
            pack_ch!(3, y3);
            return;
        }
        let finish = *w.add(KV_FINISH_WORD);
        if finish != 0 {
            let (in0, in1, weight, ring, output) = (
                *w.add(1) as *const f32, *w.add(2) as *const f32, *w.add(4) as *const f32,
                *w.add(12) as *const f32, *w.add(5) as *mut f32,
            );
            let (rows, dim, taps, heads, form) = (*w.add(6) as usize, *w.add(7) as usize,
                *w.add(8) as usize, *w.add(9) as usize, *w.add(10) as u32);
            let bound = f32::from_bits(*w.add(11) as u32);
            let threads = core::arch::nvptx::_block_dim_x() as usize;
            let first = *w.add(13) as usize;
            let param = f32::from_bits(*w.add(17) as u32);
            let plane_first = *w.add(18) as usize;
            let plane_rows = *w.add(19) as usize;
            let next_ring = *w.add(KV_NEXT_RING_WORD) as *mut f32;
            let kv_tile = *w.add(KV_PREFILL_TILE_WORD) as usize;
            if threads != ROW_DOTS_THREADS || !stream_ops_admitted(STREAM_OP_SHORT_CONV_WIN, rows, dim, taps, STREAM_OPS_THREADS)
                || ring.is_null() || first >= taps || heads == 0 || dim != heads * CONV_KV_DIM
                || !matches!(finish, CONV_KV_CLIP | CONV_KV_NORM)
                || form & !CONV_TOKEN_TAPS != 0 || (form & CONV_TOKEN_TAPS != 0) == in1.is_null()
                || plane_rows < rows || plane_first >= plane_rows
                || ((in0 as usize) | (in1 as usize) | (ring as usize) | (weight as usize) | (output as usize) | (next_ring as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            if kv_tile != 0 {
                if !matches!(kv_tile, 8 | 16 | 32)
                    || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || rows < taps
                    || core::arch::nvptx::_grid_dim_x() as usize != heads
                    || core::arch::nvptx::_grid_dim_y() as usize != conv_kv_tiled_blocks_y(rows, kv_tile)
                {
                    core::arch::nvptx::trap();
                }
                let head = core::arch::nvptx::_block_idx_x() as usize;
                let tile_y = core::arch::nvptx::_block_idx_y() as usize;
                let tid = core::arch::nvptx::_thread_idx_x() as usize;
                let (warp, lane) = (tid >> 5, tid & 31);
                let r_start = (tile_y * 4 + warp) * kv_tile;
                if r_start >= rows { return; }
                let r_end = (r_start + kv_tile).min(rows);
                let ch = head * CONV_KV_DIM + lane;
                macro_rules! ld_ev {($p:expr, $ev:expr)=>{{
                    let v: f32;
                    core::arch::asm!("ld.global.nc.L2::cache_hint.f32 {v}, [{p}], {q};",
                        v = out(reg32) v, p = in(reg64) $p, q = in(reg64) $ev, options(readonly, nostack));
                    v
                }}}
                macro_rules! st_ev_u32 {($p:expr, $v:expr, $ev:expr)=>{{
                    core::arch::asm!("st.global.L2::cache_hint.u32 [{p}], {v}, {q};",
                        p = in(reg64) $p, v = in(reg32) $v, q = in(reg64) $ev, options(nostack));
                }}}
                // Pack 4 `rn` BF16 values into 2 `u32` regs (low 16 bits of each `rn` f32 are 0 -> 100% bit-exact).
                macro_rules! pack2 {($a:expr, $b:expr)=>{ ($a.to_bits() >> 16) | ($b.to_bits() & 0xffff_0000) }}
                macro_rules! unlo {($p:expr)=>{ f32::from_bits(($p) << 16) }}
                macro_rules! unhi {($p:expr)=>{ f32::from_bits(($p) & 0xffff_0000) }}
                macro_rules! load4_pk {($base:expr, $ev:expr)=>{{
                    let p = $base;
                    let (a, b, c, d) = (rn(ld_ev!(p, $ev)), rn(ld_ev!(p.add(32), $ev)), rn(ld_ev!(p.add(64), $ev)), rn(ld_ev!(p.add(96), $ev)));
                    [pack2!(a, b), pack2!(c, d)]
                }}}
                let (mut p1, mut p2, mut p3) = {
                    let ev0 = crate::tc_ptx::l2_evict_first();
                    if r_start == 0 {
                        let rptr = *w.add(12) as *const f32;
                        let mut slot = if first == 0 { taps - 1 } else { first - 1 };
                        let v1 = load4_pk!(rptr.add(slot * dim + ch), ev0);
                        let v2 = if taps >= 3 {
                            slot = if slot == 0 { taps - 1 } else { slot - 1 };
                            load4_pk!(rptr.add(slot * dim + ch), ev0)
                        } else { [0u32; 2] };
                        let v3 = if taps >= 4 {
                            slot = if slot == 0 { taps - 1 } else { slot - 1 };
                            load4_pk!(rptr.add(slot * dim + ch), ev0)
                        } else { [0u32; 2] };
                        (v1, v2, v3)
                    } else {
                        (
                            load4_pk!(in0.add((r_start - 1) * dim + ch), ev0),
                            if taps >= 3 { load4_pk!(in0.add((r_start - 2) * dim + ch), ev0) } else { [0u32; 2] },
                            if taps >= 4 { load4_pk!(in0.add((r_start - 3) * dim + ch), ev0) } else { [0u32; 2] },
                        )
                    }
                };
                let mut row = r_start;
                while row < r_end {
                    let cur = {
                        let ev = crate::tc_ptx::l2_evict_first();
                        load4_pk!(in0.add(row * dim + ch), ev)
                    };
                    let in1_p = *w.add(2) as *const f32;
                    let has_tok = !in1_p.is_null();
                    let (tk01, tk23) = if has_tok {
                        let tok_ptr = in1_p.add((row * heads + head) * taps);
                        if taps == 4 && (tok_ptr as usize) & 15 == 0 {
                            let tv = crate::tc_ptx::ld_global_nc_v4(tok_ptr);
                            (pack2!(rn(tv[0]), rn(tv[1])), pack2!(rn(tv[2]), rn(tv[3])))
                        } else {
                            let t0 = rn(f32::from_bits(crate::tc_ptx::ld_global_nc_u32(tok_ptr.cast())));
                            let t1 = rn(f32::from_bits(crate::tc_ptx::ld_global_nc_u32(tok_ptr.add(1).cast())));
                            let t2 = if taps >= 3 { rn(f32::from_bits(crate::tc_ptx::ld_global_nc_u32(tok_ptr.add(2).cast()))) } else { 0.0f32 };
                            let t3 = if taps >= 4 { rn(f32::from_bits(crate::tc_ptx::ld_global_nc_u32(tok_ptr.add(3).cast()))) } else { 0.0f32 };
                            (pack2!(t0, t1), pack2!(t2, t3))
                        }
                    } else {
                        (0u32, 0u32)
                    };
                    let w_ptr = *w.add(4) as *const f32;
                    let fast_w4 = taps == 4 && (w_ptr as usize) & 15 == 0;
                    let bound_r = f32::from_bits(*w.add(11) as u32);
                    macro_rules! conv_val {($ck:expr, $c0:expr, $c1:expr, $c2:expr, $c3:expr)=>{{
                        let ck = $ck;
                        let (f0, f1, f2, f3) = if fast_w4 {
                            let ft = crate::tc_ptx::ld_global_nc_v4(w_ptr.add(ck * 4));
                            (ft[0], ft[1], ft[2], ft[3])
                        } else {
                            (
                                f32::from_bits(crate::tc_ptx::ld_global_nc_u32(w_ptr.add(ck * taps).cast())),
                                f32::from_bits(crate::tc_ptx::ld_global_nc_u32(w_ptr.add(ck * taps + 1).cast())),
                                if taps >= 3 { f32::from_bits(crate::tc_ptx::ld_global_nc_u32(w_ptr.add(ck * taps + 2).cast())) } else { 0.0f32 },
                                if taps >= 4 { f32::from_bits(crate::tc_ptx::ld_global_nc_u32(w_ptr.add(ck * taps + 3).cast())) } else { 0.0f32 },
                            )
                        };
                        let mut s = 0.0f32;
                        if taps >= 4 { s += (if has_tok { f3 + unhi!(tk23) } else { f3 }) * $c3; }
                        if taps >= 3 { s += (if has_tok { f2 + unlo!(tk23) } else { f2 }) * $c2; }
                        s += (if has_tok { f1 + unhi!(tk01) } else { f1 }) * $c1;
                        s += (if has_tok { f0 + unlo!(tk01) } else { f0 }) * $c0;
                        rn((s + $c0) * bound_r)
                    }}}
                    let v0 = conv_val!(ch, unlo!(cur[0]), unlo!(p1[0]), unlo!(p2[0]), unlo!(p3[0]));
                    let v1 = conv_val!(ch + 32, unhi!(cur[0]), unhi!(p1[0]), unhi!(p2[0]), unhi!(p3[0]));
                    let v2 = conv_val!(ch + 64, unlo!(cur[1]), unlo!(p1[1]), unlo!(p2[1]), unlo!(p3[1]));
                    let v3 = conv_val!(ch + 96, unhi!(cur[1]), unhi!(p1[1]), unhi!(p2[1]), unhi!(p3[1]));
                    p3 = p2;
                    p2 = p1;
                    p1 = cur;
                    let mut sum = 0.0f32;
                    sum += v0 * v0;
                    sum += v1 * v1;
                    sum += v2 * v2;
                    sum += v3 * v3;
                    sum += shfl_xor_f32(sum, 16);
                    sum += shfl_xor_f32(sum, 8);
                    sum += shfl_xor_f32(sum, 4);
                    sum += shfl_xor_f32(sum, 2);
                    sum += shfl_xor_f32(sum, 1);
                    let fin_r = *w.add(KV_FINISH_WORD);
                    let par_r = f32::from_bits(*w.add(17) as u32);
                    let scale = if fin_r == CONV_KV_CLIP {
                        let bits: u32;
                        core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",
                            got = out(reg32) bits, v = in(reg32) sum.to_bits());
                        crate::block_attention::clip_factor(crate::block_ops::root(f32::from_bits(bits)), par_r)
                    } else {
                        1.0 / crate::block_ops::root(sum / CONV_KV_DIM as f32 + par_r)
                    };
                    let pf = *w.add(18) as usize;
                    let pr = *w.add(19) as usize;
                    let slot = if row < pr - pf { pf + row } else { row - (pr - pf) };
                    let plane = (*w.add(5) as *mut u32).add(slot * (dim / 2) + ch / 2);
                    let ev_st = crate::tc_ptx::l2_evict_first();
                    let store_pair = |part: usize, value: f32| {
                        let bits = rn(value * scale).to_bits();
                        let peer = shfl_xor_u32(bits, 1);
                        if lane % 2 == 0 { st_ev_u32!(plane.add(part * 16), (bits >> 16) | (peer & 0xffff_0000), ev_st); }
                    };
                    store_pair(0, v0);
                    store_pair(1, v1);
                    store_pair(2, v2);
                    store_pair(3, v3);
                    row += 1;
                }
                let rows_r = *w.add(6) as usize;
                let nr = *w.add(KV_NEXT_RING_WORD) as *mut f32;
                if r_end == rows_r && !nr.is_null() {
                    let first_r = *w.add(13) as usize;
                    let mut slot = ((first_r as u32 + rows_r as u32 - 1) % (taps as u32)) as usize;
                    let mut back = 0usize;
                    while back < taps {
                        let src = in0.add((rows_r - 1 - back) * dim + ch);
                        let dst = nr.add(slot * dim + ch);
                        *dst = *src;
                        *dst.add(32) = *src.add(32);
                        *dst.add(64) = *src.add(64);
                        *dst.add(96) = *src.add(96);
                        slot = if slot == 0 { taps - 1 } else { slot - 1 };
                        back += 1;
                    }
                }
                return;
            }
            if core::arch::nvptx::_grid_dim_x() as usize != conv_kv_blocks(rows, heads) || !next_ring.is_null() {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let unit = core::arch::nvptx::_block_idx_x() as usize * (threads / 32) + tid / 32;
            if unit >= rows * heads { return; }
            let (row, head, lane) = (unit / heads, unit % heads, tid % 32);
            let ch = head * CONV_KV_DIM + lane;
            let [mut s0, mut s1, mut s2, mut s3] = [0.0f32; 4];
            let mut current = [0.0f32; 4];
            let mut lag = taps;
            while lag > 0 {
                lag -= 1;
                let source = if lag <= row {
                    in0.add((row - lag) * dim)
                } else {
                    let slot = ((first + taps + row - lag) as u32 % taps as u32) as usize;
                    ring.add(slot * dim)
                };
                let token = if in1.is_null() { None } else { Some(rn(*in1.add(unit * taps + lag))) };
                let tap = |channel| {
                    let fixed = *weight.add(channel * taps + lag);
                    token.map_or(fixed, |v| fixed + v)
                };
                current = [rn(*source.add(ch)), rn(*source.add(ch + 32)), rn(*source.add(ch + 64)), rn(*source.add(ch + 96))];
                s0 += tap(ch) * current[0];
                s1 += tap(ch + 32) * current[1];
                s2 += tap(ch + 64) * current[2];
                s3 += tap(ch + 96) * current[3];
            }
            // Keep the four ascending channel terms in registers through the FastNorm warp reduction.
            let values = [rn((s0 + current[0]) * bound), rn((s1 + current[1]) * bound),
                rn((s2 + current[2]) * bound), rn((s3 + current[3]) * bound)];
            let mut sum = 0.0f32;
            sum += values[0] * values[0];
            sum += values[1] * values[1];
            sum += values[2] * values[2];
            sum += values[3] * values[3];
            sum += shfl_xor_f32(sum, 16);
            sum += shfl_xor_f32(sum, 8);
            sum += shfl_xor_f32(sum, 4);
            sum += shfl_xor_f32(sum, 2);
            sum += shfl_xor_f32(sum, 1);
            let scale = if finish == CONV_KV_CLIP {
                let bits: u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, 0, 31, 0xffffffff;",
                    got = out(reg32) bits, v = in(reg32) sum.to_bits());
                crate::block_attention::clip_factor(crate::block_ops::root(f32::from_bits(bits)), param)
            } else {
                1.0 / crate::block_ops::root(sum / CONV_KV_DIM as f32 + param)
            };
            let to_end = plane_rows - plane_first;
            let slot = if row < to_end { plane_first + row } else { row - to_end };
            let plane = output.cast::<u32>().add(slot * (dim / 2) + ch / 2);
            let store_pair = |part: usize, value: f32| {
                let bits = rn(value * scale).to_bits();
                let peer = shfl_xor_u32(bits, 1);
                if lane % 2 == 0 { *plane.add(part * 16) = (bits >> 16) | (peer & 0xffff_0000); }
            };
            store_pair(0, values[0]);
            store_pair(1, values[1]);
            store_pair(2, values[2]);
            store_pair(3, values[3]);
            return;
        }

        let joined = *w as *const f32;
        let queries = *w.add(1) as *const f32;
        let output = *w.add(2) as *mut f32;
        let rows = *w.add(3) as usize;
        let asked = *w.add(4) as usize;
        let kept = *w.add(5) as usize;
        let k = *w.add(6) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        if !row_dots_admitted(rows, asked, kept, k, threads) || block >= rows * kept { core::arch::nvptx::trap(); }
        if ((joined as usize) | (queries as usize) | (output as usize)) & 3 != 0 { core::arch::nvptx::trap(); }

        let row = block / kept;
        let e = block % kept;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid >> 5, tid & 31);
        let warps = threads >> 5;
        let stream = if joined.is_null() { *w.add(7 + e) as *const f32 } else { joined.add(e * rows * k) };
        let v_ptr = stream.add(row * k);

        let mut q = warp;
        while q < asked {
            let q_ptr = queries.add(q * k);
            let mut sum = 0.0f32;
            let mut j = lane;
            while j < k {
                sum += *v_ptr.add(j) * *q_ptr.add(j);
                j += 32;
            }
            sum += shfl_xor_f32(sum, 16);
            sum += shfl_xor_f32(sum, 8);
            sum += shfl_xor_f32(sum, 4);
            sum += shfl_xor_f32(sum, 2);
            sum += shfl_xor_f32(sum, 1);
            if lane == 0 {
                *output.add((row * asked + q) * kept + e) = sum;
            }
            q += warps;
        }
    }
}
