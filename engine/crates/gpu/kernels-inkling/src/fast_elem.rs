//! Zero-stack bit-exact Inkling elementwise and routing kernels:
//! - `FAST_CAPPED_GATE` (110): 1 block (256 threads) per pick, zero divisions.
//! - `FAST_ROUTER_PICK` (111): 1 warp per token (4 tokens/block), computes `sigmoid` once per expert.
//! - `FAST_STREAM_OPS` (112): `ShortConv` (window form), `RouteFold`, and `GateBlend`, 1 block per row.
//!
//! Every kernel matches its `block_experts`/`block_stream` reference bit-for-bit with zero stack frame.

use crate::block_ops::{LANES, NORM_CHUNK, root};
use crate::block_stream::{CONV_ADDEND, CONV_ROUND_SUM, CONV_TOKEN_TAPS};
use crate::fast_ops::{
    CAPPED_GATE_TABLE_WORDS, FOLD_FORM_WORD, FOLD_HALF, GATE_BLEND_NORM_TABLE_AT, NORM_CHUNKS_MAX,
    ROUTE_FOLD_CONV_FORM_WORD, ROUTE_FOLD_CONV_TAPS_MAX, ROUTE_FOLD_CONV_WEIGHTS_MAX,
    ROUTER_PICK_COMPACT_WORD, ROUTER_PICKS_MAX, SHORT_CONV_MULTI_TAG, SHORT_CONV_PREFILL_TILE,
    STREAM_OP_GATE_BLEND_NORM, STREAM_OP_RING_CONV_ONE, STREAM_OP_ROUTE_FOLD,
    STREAM_OP_ROUTE_FOLD_CONV, STREAM_OP_ROUTE_FOLD_CONV_PREFILL, STREAM_OP_SHORT_CONV_WIN,
    capped_gate_admitted, router_pick_admitted, router_pick_compact_admitted,
    short_conv_multi_decode, stream_ops_admitted, stream_ops_parts,
};
use crate::fixed_math::{rn, sigmoid, tanh};
use crate::tc_ptx::{ld_global_nc_u32, ld_global_nc_v4, load_u32, shfl_xor_f32, shfl_xor_u32, store_u32};

const FOLD_SMEM_BYTES: usize = ROUTE_FOLD_CONV_WEIGHTS_MAX * 4; // 512 bytes
const ROUTER_COMPACT_SMEM_BYTES: usize = 1152; // 128*8 expert pick bitmasks + 8*4*4 LPT warp bucket counts
const _: () = assert!((NORM_CHUNKS_MAX + 1) * 4 <= FOLD_SMEM_BYTES);

/// Replay one recorded command of kind `FAST_CAPPED_GATE`.
/// SAFETY (caller contract, `resident_launch_fast_capped_gate`): `table[index]` is a valid command recorded by
/// `Backend::fast_capped_gate_into` after `fast_ops::check_capped_gate`, or the coefficient-table build recorded by
/// `Backend::record_capped_gate_tables` after cap and private-storage validation.
#[inline(always)]
pub(crate) unsafe fn replay_capped_gate(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - In modes 0 and 1, the host verified via `fast_ops::check_capped_gate` that `input` holds
    //   `picks * 2 * dim` FP32 words and `output` holds `picks * dim` FP32 words, disjoint and word-aligned.
    // - In mode 2, input and output are not read or written and may alias the coefficient address. The host checked
    //   positive caps, finite reciprocals, and a private, contiguous, writable coefficient allocation of exactly
    //   `2 * CAPPED_GATE_TABLE_WORDS` FP32 words. The resident queue retains that allocation and records this build
    //   before every consumer. The exact grid gives one writer to each word in both tables.
    // - Device re-checks `capped_gate_admitted`, grid bounds (`pick < picks`), and alignment.
    // - Prefill vector accesses require aligned bases and a width divisible by four, so each
    //   16-byte load or store stays inside its checked row.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_CAPPED_GATE { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let input = *w as *const f32;
        let output = *w.add(1) as *mut f32;
        let picks = *w.add(2) as usize;
        let dim = *w.add(3) as usize;
        let gate_cap = f32::from_bits(*w.add(4) as u32);
        let up_cap = f32::from_bits(*w.add(5) as u32);
        let inv_gate = f32::from_bits(*w.add(6) as u32);
        let inv_up = f32::from_bits(*w.add(7) as u32);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let pick = core::arch::nvptx::_block_idx_x() as usize;
        let mode = *w.add(8);
        let coefficients = *w.add(9) as *mut f32;
        if !capped_gate_admitted(picks, dim, threads)
            || ((input as usize) | (output as usize) | (coefficients as usize)) & 3 != 0
        {
            core::arch::nvptx::trap();
        }
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let gate_coeff = |gate| (tanh(gate * inv_gate) * gate_cap) * sigmoid(gate);
        let up_coeff = |up| tanh(up * inv_up) * up_cap;
        if mode == 2 {
            let j = pick * threads + tid;
            if coefficients.is_null() || j >= CAPPED_GATE_TABLE_WORDS { core::arch::nvptx::trap(); }
            let x = f32::from_bits((j as u32) << 16);
            *coefficients.add(j) = gate_coeff(x);
            *coefficients.add(CAPPED_GATE_TABLE_WORDS + j) = up_coeff(x);
            return;
        }
        if pick >= picks { core::arch::nvptx::trap(); }

        let g_ptr = input.add(pick * 2 * dim);
        let u_ptr = g_ptr.add(dim);
        let o_ptr = output.add(pick * dim);
        let capped = |gate, up| {
            let gate = rn(gate);
            let up = rn(up);
            if coefficients.is_null() {
                rn(gate_coeff(gate) * up_coeff(up))
            } else {
                let g = (gate.to_bits() >> 16) as usize;
                let u = (up.to_bits() >> 16) as usize;
                rn(*coefficients.add(g) * *coefficients.add(CAPPED_GATE_TABLE_WORDS + u))
            }
        };
        if mode == 1 && dim % 4 == 0 && ((input as usize) | (output as usize)) & 15 == 0 {
            let mut j = tid * 4;
            while j < dim {
                let g = ld_global_nc_v4(g_ptr.add(j));
                let u = ld_global_nc_v4(u_ptr.add(j));
                let a = capped(g[0], u[0]); let b = capped(g[1], u[1]);
                let c = capped(g[2], u[2]); let d = capped(g[3], u[3]);
                core::arch::asm!("st.global.v4.b32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg64) o_ptr.add(j), a = in(reg32) a, b = in(reg32) b,
                    c = in(reg32) c, d = in(reg32) d);
                j += threads * 4;
            }
            return;
        }
        let mut j = tid;
        while j < dim {
            *o_ptr.add(j) = capped(*g_ptr.add(j), *u_ptr.add(j));
            j += threads;
        }
    }
}

/// Replay one recorded command of kind `FAST_ROUTER_PICK`.
/// SAFETY (caller contract, `resident_launch_fast_router_pick`): `table[index]` is a valid command
/// recorded by `Backend::fast_router_pick_into` after `fast_ops::check_router_pick`.
#[inline(always)]
pub(crate) unsafe fn replay_router_pick(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host verified via `fast_ops::check_router_pick` (or `check_router_pick_compact` when word 8 is non-zero)
    //   that `logits` holds `rows * experts`, `bias` holds `experts`, `output` holds `rows * 2 * picks`, and optional
    //   16-byte aligned `compact` holds `rows * picks * 4` u32 words, all disjoint from inputs and word-aligned.
    // - Device re-checks `router_pick_admitted` / `router_pick_compact_admitted` and alignment.
    // - All 32 lanes of each active warp participate in every `shfl.sync.bfly.b32` and `ballot`; in compact mode
    //   all 256 threads of the single block participate in every `_syncthreads`, and each compacted entry has one writer.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_ROUTER_PICK { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let logits = *w as *const f32;
        let bias = *w.add(1) as *const f32;
        let output = *w.add(2) as *mut f32;
        let rows = *w.add(3) as usize;
        let experts = *w.add(4) as usize;
        let picks = *w.add(5) as usize;
        let normalize = *w.add(6) != 0;
        let bound = f32::from_bits(*w.add(7) as u32);
        let compact = *w.add(ROUTER_PICK_COMPACT_WORD) as *mut u32;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let do_compact = !compact.is_null();
        if do_compact {
            if !router_pick_compact_admitted(rows, experts, picks, threads)
                || block != 0 || core::arch::nvptx::_grid_dim_x() != 1
                || ((compact as usize) & 15) != 0
            {
                core::arch::nvptx::trap();
            }
        } else if !router_pick_admitted(rows, experts, picks, threads) {
            core::arch::nvptx::trap();
        }
        if ((logits as usize) | (bias as usize) | (output as usize)) & 3 != 0 { core::arch::nvptx::trap(); }

        let smem = crate::ptx::shared::<ROUTER_COMPACT_SMEM_BYTES>();
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        if do_compact {
            store_u32(smem + (tid * 4) as u32, 0);
            core::arch::nvptx::_syncthreads();
        }
        let (warp, lane) = (tid >> 5, tid & 31);
        let row = block * (threads >> 5) + warp;
        if row < rows {
            let l_ptr = logits.add(row * experts);
            let o_ptr = output.add(row * 2 * picks);

            // Each lane holds up to 4 experts (`lane + 32 * k` for k in 0..4) in registers (0 stack), computing sigmoid ONCE.
            let s0 = if lane < experts { sigmoid(*l_ptr.add(lane)) } else { 0.0f32 };
            let s1 = if lane + 32 < experts { sigmoid(*l_ptr.add(lane + 32)) } else { 0.0f32 };
            let s2 = if lane + 64 < experts { sigmoid(*l_ptr.add(lane + 64)) } else { 0.0f32 };
            let s3 = if lane + 96 < experts { sigmoid(*l_ptr.add(lane + 96)) } else { 0.0f32 };
            let c0 = if lane < experts { s0 + *bias.add(lane) } else { f32::NEG_INFINITY };
            let c1 = if lane + 32 < experts { s1 + *bias.add(lane + 32) } else { f32::NEG_INFINITY };
            let c2 = if lane + 64 < experts { s2 + *bias.add(lane + 64) } else { f32::NEG_INFINITY };
            let c3 = if lane + 96 < experts { s3 + *bias.add(lane + 96) } else { f32::NEG_INFINITY };
            let mut alive_mask = u32::from(lane < experts)
                | (u32::from(lane + 32 < experts) << 1)
                | (u32::from(lane + 64 < experts) << 2)
                | (u32::from(lane + 96 < experts) << 3);

            let mut sum = 0.0f32;
            let mut my_id = 0u32;
            let mut my_s = 0.0f32;
            let mut slot = 0usize;
            while slot < picks {
                let mut best_c = f32::NEG_INFINITY;
                let mut best_id = 0u32;
                if (alive_mask & 1) != 0 && c0 > best_c { best_c = c0; best_id = lane as u32; }
                if (alive_mask & 2) != 0 && c1 > best_c { best_c = c1; best_id = (lane + 32) as u32; }
                if (alive_mask & 4) != 0 && c2 > best_c { best_c = c2; best_id = (lane + 64) as u32; }
                if (alive_mask & 8) != 0 && c3 > best_c { best_c = c3; best_id = (lane + 96) as u32; }
                // Fixed-order warp butterfly argmax: strict `>`, lower expert ID wins on ties.
                let mut mask = 16u32;
                while mask > 0 {
                    let oc = shfl_xor_f32(best_c, mask);
                    let oid = shfl_xor_u32(best_id, mask);
                    if oc > best_c || (oc == best_c && oc > f32::NEG_INFINITY && oid < best_id) {
                        best_c = oc;
                        best_id = oid;
                    }
                    mask >>= 1;
                }
                // Fetch the precomputed sigmoid score from the winning lane via warp shuffle (zero recomputation).
                let owner_k = (best_id as usize) >> 5;
                let owner_lane = (best_id & 31) as u32;
                let owned_s = match owner_k { 0 => s0, 1 => s1, 2 => s2, _ => s3 };
                let sc_bits: u32;
                core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, {src}, 31, 0xffffffff;",
                    got = out(reg32) sc_bits, v = in(reg32) owned_s.to_bits(), src = in(reg32) owner_lane);
                let sc = f32::from_bits(sc_bits);
                // Mark picked expert inactive in its owning lane.
                if (owner_lane as usize) == lane && owner_k < 4 {
                    alive_mask &= !(1u32 << owner_k);
                }
                if lane == slot {
                    my_id = best_id;
                    my_s = sc;
                }
                sum += sc;
                slot += 1;
            }
            if lane < picks {
                let denom = sum + 1.0e-20f32;
                let v = if normalize { my_s / denom } else { my_s };
                *o_ptr.add(lane) = my_id as f32;
                *o_ptr.add(picks + lane) = v * bound;
                if do_compact {
                    let j = row * picks + lane;
                    let mask_addr = smem + (((my_id as usize) * 2 + (j >> 5)) * 4) as u32;
                    let bit = 1u32 << (j & 31);
                    let _old: u32;
                    core::arch::asm!(
                        "atom.shared.or.b32 {old}, [{p}], {v};",
                        old = out(reg32) _old, p = in(reg32) mask_addr, v = in(reg32) bit,
                        options(nostack)
                    );
                }
            }
        }
        if !do_compact { return; }
        core::arch::nvptx::_syncthreads();
        // Compact unique active experts in descending multiplicity (LPT) order across 4 warps (128 experts).
        let total_picks = rows * picks;
        let mut indices = 0u64;
        let mut cnt = 0usize;
        let mut my_rank = 0u32;
        if tid < 128 {
            if tid < experts {
                let lo_m = load_u32(smem + (tid * 8) as u32);
                let hi_m = load_u32(smem + (tid * 8 + 4) as u32);
                cnt = (lo_m.count_ones() + hi_m.count_ones()) as usize;
                if cnt > 8 { core::arch::nvptx::trap(); }
                if cnt > 0 {
                    let mut pos = 0usize;
                    let mut m = lo_m;
                    while m != 0 {
                        let j = m.trailing_zeros() as u64;
                        indices |= j << (8 * pos);
                        pos += 1;
                        m &= m - 1;
                    }
                    let mut m = hi_m;
                    while m != 0 {
                        let j = 32 + m.trailing_zeros() as u64;
                        indices |= j << (8 * pos);
                        pos += 1;
                        m &= m - 1;
                    }
                }
            }
            let mut b = 1usize;
            while b <= rows {
                let m = crate::ptx::ballot(cnt == b);
                if lane == 0 {
                    store_u32(smem + 1024 + (((b - 1) * 4 + warp) * 4) as u32, m.count_ones());
                }
                if cnt == b {
                    my_rank = (m & ((1u32 << lane) - 1)).count_ones();
                }
                b += 1;
            }
        }
        core::arch::nvptx::_syncthreads();
        if tid < 32 {
            let b_idx = lane >> 2;
            let w_i = lane & 3;
            let b = rows.saturating_sub(b_idx);
            let val = if b_idx < rows {
                load_u32(smem + 1024 + (((b - 1) * 4 + w_i) * 4) as u32) * (b as u32)
            } else {
                0
            };
            let mut incl = val;
            let mut d = 1u32;
            while d <= 16 {
                let up: u32;
                core::arch::asm!(
                    "shfl.sync.up.b32 {got}, {v}, {d}, 0, 0xffffffff;",
                    got = out(reg32) up, v = in(reg32) incl, d = in(reg32) d
                );
                if lane >= d as usize { incl += up; }
                d <<= 1;
            }
            if b_idx < rows {
                store_u32(smem + 1024 + (((b - 1) * 4 + w_i) * 4) as u32, incl - val);
            }
            if lane == 31 && incl as usize != total_picks {
                core::arch::nvptx::trap();
            }
        }
        core::arch::nvptx::_syncthreads();
        if tid < 128 && cnt > 0 {
            let base = load_u32(smem + 1024 + (((cnt - 1) * 4 + warp) * 4) as u32) + my_rank * (cnt as u32);
            let lo = indices as u32;
            let hi = (indices >> 32) as u32;
            let mut ord = 0u32;
            while (ord as usize) < cnt {
                let meta = (cnt as u32) | (ord << 16);
                let dst = compact.add(((base + ord) as usize) * 4);
                core::arch::asm!(
                    "st.global.v4.u32 [{p}], {{{a}, {b}, {c}, {d}}};",
                    p = in(reg64) dst,
                    a = in(reg32) lo, b = in(reg32) hi, c = in(reg32) tid as u32, d = in(reg32) meta,
                    options(nostack)
                );
                ord += 1;
            }
        }
    }
}

/// Replay one recorded command of kind `FAST_STREAM_OPS` (`ShortConv` window, `RouteFold`, `GateBlend`).
/// SAFETY (caller contract, `resident_launch_fast_stream_ops`): `table[index]` is a valid command
/// recorded by `Backend` after `fast_ops::check_short_conv_win`, `check_route_fold`, or `check_gate_blend`.
#[inline(always)]
pub(crate) unsafe fn replay_stream_ops(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host checked every operand extent, word alignment, and non-overlap via `fast_ops::check_*`.
    // - Device re-checks `stream_ops_admitted`, grid bounds (`row < rows`), and pointer alignments.
    // - Prefill ShortConv may hold a separate `taps * width`-word ring and `rows * width`-word input.
    //   The first slot is below taps. The lag selects a valid row in exactly one of those buffers.
    // - A window ShortConv of a step that keeps each row may hold `every` (payload word 14, 0 for none): `rows * taps
    //   * width` private words, disjoint from every operand (host check), and `first` (word 15) < taps, the ring slot
    //   of row 0. The thread of (row, ch) writes word (row * taps + slot) * width + ch for each slot, once.
    // - For `RouteFold`, `smem` holds `picks <= ROUTER_PICKS_MAX` FP32 words inside `FOLD_SMEM_BYTES` (128 B)
    //   and all 256 threads participate in `bar.sync 0`. Each `(row, ch)` has one writer: the block of part
    //   `ch / threads % parts` of its row (the grid is checked to be `rows * parts` blocks).
    // - A `RouteFold` with `FOLD_HALF` in its form word (`FOLD_FORM_WORD`) reads `in1` as `rows * picks * dim` BF16
    //   values (host checked `check_route_fold_strided` with `half`: the span is that many 2-byte values, word
    //   aligned): the value
    //   `(row * picks + p) * dim + ch` with `row < rows`, `p < picks`, `ch < dim` lies inside it.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FAST_STREAM_OPS { core::arch::nvptx::trap(); }
        // Allow a dependent decode panel product (`resident_launch::dependent_launch`) to overlap its panel prefetches
        // before its `griddepcontrol.wait`. This launch is never itself dependent, so all prior work has ended.
        core::arch::asm!("griddepcontrol.launch_dependents;", options(nostack));
        let w = c.payload.as_ptr().cast::<u64>();
        let subop = *w;
        let rows = *w.add(6) as usize;
        let dim = *w.add(7) as usize;
        let aux = *w.add(8) as usize;
        let bound = f32::from_bits(*w.add(11) as u32);
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        // Block b is part b % parts of row b / parts (`fast_ops::stream_ops_parts`): its channels are
        // part * threads + tid, then every parts * threads on. The grid is exactly rows * parts blocks.
        let smem = crate::ptx::shared::<FOLD_SMEM_BYTES>();
        if subop == STREAM_OP_GATE_BLEND_NORM {
            // Fused `GateBlend` + optional weighted `RoundedNorm` for 1..=3 gates (`grid = (rows, count, 1)`).
            // Launches `warps = threads / 32` (`<= 9` warps = `<= 288` threads -> 3 blocks/SM on SM121a), with
            // each warp owning `<= 3` 128-element chunks (`<= 12` floats/lane in scalar registers, zero stack frame).
            // SAFETY: host checked via `fast_ops::check_gate_blend_norm` that all `count` gate tuples have valid
            // extents, word alignment, and mutually disjoint outputs. `smem` (512 B) holds `chunks + 1 <= 33` words;
            // all `threads` participate in every `_syncthreads` and full 32-lane warps in every `shfl_xor`.
            let count = aux;
            let row = core::arch::nvptx::_block_idx_x() as usize;
            let gate_i = core::arch::nvptx::_block_idx_y() as usize;
            if !stream_ops_admitted(subop, rows, dim, count, threads)
                || core::arch::nvptx::_grid_dim_x() as usize != rows
                || core::arch::nvptx::_grid_dim_y() as usize != count
                || row >= rows || gate_i >= count
            {
                core::arch::nvptx::trap();
            }
            let base = GATE_BLEND_NORM_TABLE_AT + gate_i * 5;
            let gates_p = *w.add(base) as *const f32;
            let kept_p = *w.add(base + 1) as *const f32;
            let msg_p = *w.add(base + 2) as *const f32;
            let wt_p = *w.add(base + 3) as *const f32;
            let out_p = *w.add(base + 4) as *mut f32;
            if gates_p.is_null() || kept_p.is_null() || msg_p.is_null() || out_p.is_null()
                || ((gates_p as usize) | (kept_p as usize) | (msg_p as usize) | (wt_p as usize) | (out_p as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let (warp, lane) = (tid >> 5, tid & 31);
            let warps = threads >> 5;
            let chunks = dim / NORM_CHUNK;
            let f_ptr = gates_p.add(row * 2 * dim);
            let g_ptr = f_ptr.add(dim);
            let k_ptr = kept_p.add(row * dim);
            let m_ptr = msg_p.add(row * dim);
            let o_ptr = out_p.add(row * dim);
            macro_rules! word {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
            macro_rules! blend_col {($c:expr) => {{
                let f = word!(f_ptr.add($c));
                let g = word!(g_ptr.add($c));
                let k = word!(k_ptr.add($c));
                let m = word!(m_ptr.add($c));
                (1.0f32 - tanh(f)) * k + tanh(g) * m
            }};}
            let (ch0, ch1, ch2) = (warp, warp + warps, warp + 2 * warps);
            let (has0, has1, has2) = (ch0 < chunks, ch1 < chunks, ch2 < chunks);
            let (b0, b1, b2) = (ch0 * NORM_CHUNK + lane, ch1 * NORM_CHUNK + lane, ch2 * NORM_CHUNK + lane);
            macro_rules! run_chunk {($has:ident, $ch:ident, $b:ident) => {{
                if $has {
                    let v0 = blend_col!($b);
                    let v1 = blend_col!($b + LANES);
                    let v2 = blend_col!($b + 2 * LANES);
                    let v3 = blend_col!($b + 3 * LANES);
                    let mut sq = 0.0f32;
                    sq += v0 * v0; sq += v1 * v1; sq += v2 * v2; sq += v3 * v3;
                    sq += shfl_xor_f32(sq, 16);
                    sq += shfl_xor_f32(sq, 8);
                    sq += shfl_xor_f32(sq, 4);
                    sq += shfl_xor_f32(sq, 2);
                    sq += shfl_xor_f32(sq, 1);
                    if lane == 0 { store_u32(smem + ($ch * 4) as u32, sq.to_bits()); }
                    (v0, v1, v2, v3)
                } else {
                    (0.0f32, 0.0f32, 0.0f32, 0.0f32)
                }
            }};}
            let (v00, v01, v02, v03) = run_chunk!(has0, ch0, b0);
            let (v10, v11, v12, v13) = run_chunk!(has1, ch1, b1);
            let (v20, v21, v22, v23) = run_chunk!(has2, ch2, b2);
            core::arch::nvptx::_syncthreads();
            if warp == 0 {
                let mut total = if lane < chunks { f32::from_bits(load_u32(smem + (lane * 4) as u32)) } else { 0.0f32 };
                total += shfl_xor_f32(total, 16);
                total += shfl_xor_f32(total, 8);
                total += shfl_xor_f32(total, 4);
                total += shfl_xor_f32(total, 2);
                total += shfl_xor_f32(total, 1);
                if lane == 0 { store_u32(smem + (NORM_CHUNKS_MAX * 4) as u32, (1.0f32 / root(total / dim as f32 + bound)).to_bits()); }
            }
            core::arch::nvptx::_syncthreads();
            let scale = f32::from_bits(load_u32(smem + (NORM_CHUNKS_MAX * 4) as u32));
            macro_rules! store_chunk {($has:ident, $b:ident, $a:ident, $bb:ident, $c:ident, $d:ident) => {{
                if $has {
                    if wt_p.is_null() {
                        *o_ptr.add($b) = rn($a * scale);
                        *o_ptr.add($b + LANES) = rn($bb * scale);
                        *o_ptr.add($b + 2 * LANES) = rn($c * scale);
                        *o_ptr.add($b + 3 * LANES) = rn($d * scale);
                    } else {
                        let (w0, w1, w2, w3) = (word!(wt_p.add($b)), word!(wt_p.add($b + LANES)), word!(wt_p.add($b + 2 * LANES)), word!(wt_p.add($b + 3 * LANES)));
                        *o_ptr.add($b) = rn(($a * scale) * w0);
                        *o_ptr.add($b + LANES) = rn(($bb * scale) * w1);
                        *o_ptr.add($b + 2 * LANES) = rn(($c * scale) * w2);
                        *o_ptr.add($b + 3 * LANES) = rn(($d * scale) * w3);
                    }
                }
            }};}
            store_chunk!(has0, b0, v00, v01, v02, v03);
            store_chunk!(has1, b1, v10, v11, v12, v13);
            store_chunk!(has2, b2, v20, v21, v22, v23);
            return;
        }
        let in0 = *w.add(1) as *const f32;
        let in1 = *w.add(2) as *const f32;
        let in2 = *w.add(3) as *const f32;
        let weight = *w.add(4) as *const f32;
        let output = *w.add(5) as *mut f32;
        if subop == STREAM_OP_RING_CONV_ONE {
            // M=1 fused ring-write (+ optional RouteFold) + 1-head ShortConv + addend + round on a full-SM
            // 72x32 grid with zero stack frame, vectorized taps, and zero RAW reload of the written ring row.
            // SAFETY: host verified via `block_check::operands` + `resident_conv_ring::{ring, fold}` that
            // every buffer has checked extents, word alignment, and non-overlap. Each channel `ch` has one writer.
            let taps = aux;
            let conv_form = *w.add(9) as u32;
            let picks = *w.add(10) as usize;
            let ring_mut = in0 as *mut f32;
            let add_ptr = in1;
            let step_ptr = in2;
            let source = *w.add(12) as *mut f32;
            let ring_row = *w.add(13) as *mut f32;
            let weights_p = *w.add(14) as *const f32;
            let downs_p = *w.add(15) as *const f32;
            let parts = dim.div_ceil(threads);
            if !stream_ops_admitted(subop, rows, dim, taps, threads)
                || core::arch::nvptx::_grid_dim_x() as usize != parts
                || picks > ROUTER_PICKS_MAX
                || ring_mut.is_null() || step_ptr.is_null() || source.is_null() || ring_row.is_null()
                || (picks != 0 && (weights_p.is_null() || downs_p.is_null()))
                || ((ring_mut as usize) | (add_ptr as usize) | (step_ptr as usize) | (weight as usize)
                    | (output as usize) | (source as usize) | (ring_row as usize)
                    | (weights_p as usize) | (downs_p as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let row_bytes = dim * 4;
            let off_bytes = (ring_row as usize).wrapping_sub(ring_mut as usize);
            let written_slot = off_bytes / row_bytes;
            if written_slot >= taps || written_slot * row_bytes != off_bytes {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            if picks > 0 {
                if tid < picks {
                    store_u32(smem + (tid * 4) as u32, (*weights_p.add(tid)).to_bits());
                }
                core::arch::nvptx::_syncthreads();
            }
            let ch = (core::arch::nvptx::_block_idx_x() as usize) * threads + tid;
            if ch < dim {
                let cur = if picks == 0 {
                    f32::from_bits(ld_global_nc_u32(source.add(ch).cast()))
                } else {
                    let d_base = downs_p.add(ch).cast::<u32>();
                    let mut fsum = 0.0f32;
                    let mut p = 0usize;
                    while p < picks {
                        let wt = f32::from_bits(load_u32(smem + (p * 4) as u32));
                        let d = f32::from_bits(ld_global_nc_u32(d_base.add(p * dim)));
                        fsum += rn(d) * wt;
                        p += 1;
                    }
                    let v = rn(fsum);
                    *source.add(ch) = v;
                    v
                };
                *ring_row.add(ch) = cur;
                let (t0, t1, t2, t3) = if taps == 4 && (weight as usize) & 15 == 0 {
                    let v = ld_global_nc_v4(weight.add(ch * 4));
                    (v[0], v[1], v[2], v[3])
                } else {
                    (
                        *weight.add(ch * taps),
                        if 1 < taps { *weight.add(ch * taps + 1) } else { 0.0f32 },
                        if 2 < taps { *weight.add(ch * taps + 2) } else { 0.0f32 },
                        if 3 < taps { *weight.add(ch * taps + 3) } else { 0.0f32 },
                    )
                };
                let r0 = if written_slot == 0 { cur } else { *ring_mut.add(ch) };
                let r1 = if written_slot == 1 { cur } else if 1 < taps { *ring_mut.add(dim + ch) } else { 0.0f32 };
                let r2 = if written_slot == 2 { cur } else if 2 < taps { *ring_mut.add(2 * dim + ch) } else { 0.0f32 };
                let r3 = if written_slot == 3 { cur } else if 3 < taps { *ring_mut.add(3 * dim + ch) } else { 0.0f32 };
                let step_f = f32::from_bits(ld_global_nc_u32(step_ptr.cast()));
                let slot = step_f as usize;
                if slot >= taps || slot as f32 != step_f {
                    *output.add(ch) = f32::NAN;
                } else {
                    let has_add = (conv_form & CONV_ADDEND) != 0;
                    let round_sum = (conv_form & CONV_ROUND_SUM) != 0;
                    let addend = if has_add { f32::from_bits(ld_global_nc_u32(add_ptr.add(ch).cast())) } else { 0.0f32 };
                    let mut csum = 0.0f32;
                    let mut lag = taps;
                    while lag > 0 {
                        lag -= 1;
                        let tap = match lag { 0 => t0, 1 => t1, 2 => t2, _ => t3 };
                        let read = match (slot + taps - lag) % taps { 0 => r0, 1 => r1, 2 => r2, _ => r3 };
                        csum += tap * rn(read);
                    }
                    let cur_slot = match slot { 0 => r0, 1 => r1, 2 => r2, _ => r3 };
                    let y = rn((csum + rn(cur_slot)) * bound);
                    *output.add(ch) = if !has_add { y } else if !round_sum { addend + y } else { rn(addend + y) };
                }
            }
            return;
        }
        if subop == STREAM_OP_ROUTE_FOLD_CONV {
            let ring = *w.add(12) as *const f32;
            let every_ptr = *w.add(14) as *mut f32;
            let picks = aux;
            let w_stride = *w.add(9) as usize;
            let w_off = *w.add(10) as usize;
            let taps = *w.add(13) as usize;
            let half = (*w.add(ROUTE_FOLD_CONV_FORM_WORD) & FOLD_HALF) != 0;
            // SAFETY: the host validates each chat ring, slot and disjoint output before
            // recording its address. Independent blocks read only their own chat's ring.
            let chats = *w.add(16) as usize;
            let separate = chats != 0;
            let chat = core::arch::nvptx::_block_idx_y() as usize;
            if separate && (chats != rows || chat >= rows || rows > 16) { core::arch::nvptx::trap(); }
            let (ring, first_slot, start, end) = if separate {
                (*w.add(17 + chat * 2) as *const f32, *w.add(18 + chat * 2) as usize, chat, chat + 1)
            } else { (ring, *w.add(15) as usize, 0, rows) };
            let parts = dim.div_ceil(threads);
            if !stream_ops_admitted(subop, rows, dim, picks, threads)
                || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || first_slot >= taps
                || w_off + picks > w_stride
                || (half && (dim & 1) != 0)
                || core::arch::nvptx::_grid_dim_x() as usize != parts
                || ring.is_null()
                || ((in0 as usize) | (in1 as usize) | (in2 as usize) | (weight as usize)
                    | (output as usize) | (ring as usize) | (every_ptr as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let tid = core::arch::nvptx::_thread_idx_x() as usize;
            let total_w = end * picks;
            let mut wi = start * picks + tid;
            while wi < total_w {
                let r = wi / picks;
                let p = wi % picks;
                store_u32(smem + (wi * 4) as u32, (*in0.add(r * w_stride + w_off + p)).to_bits());
                wi += threads;
            }
            core::arch::nvptx::_syncthreads();
            let ch = (core::arch::nvptx::_block_idx_x() as usize) * threads + tid;
            if ch < dim {
                // Hoist fixed taps into scalar registers ONCE across all rows (coalesced 16B load when taps==4).
                let (t0, t1, t2, t3) = if taps == 4 && (weight as usize) & 15 == 0 {
                    let v = ld_global_nc_v4(weight.add(ch * 4));
                    (v[0], v[1], v[2], v[3])
                } else {
                    (
                        *weight.add(ch * taps),
                        if 1 < taps { *weight.add(ch * taps + 1) } else { 0.0f32 },
                        if 2 < taps { *weight.add(ch * taps + 2) } else { 0.0f32 },
                        if 3 < taps { *weight.add(ch * taps + 3) } else { 0.0f32 },
                    )
                };
                let mut r0 = *ring.add(ch);
                let mut r1 = if 1 < taps { *ring.add(dim + ch) } else { 0.0f32 };
                let mut r2 = if 2 < taps { *ring.add(2 * dim + ch) } else { 0.0f32 };
                let mut r3 = if 3 < taps { *ring.add(3 * dim + ch) } else { 0.0f32 };
                let mut row = start;
                while row < end {
                    let res = f32::from_bits(ld_global_nc_u32(in2.add(row * dim + ch).cast()));
                    let mut fsum = 0.0f32;
                    let w_base = (row * picks * 4) as u32;
                    if half {
                        let h_base = in1.cast::<u16>().add(row * picks * dim + ch);
                        let mut p = 0usize;
                        while p < picks {
                            let wt = f32::from_bits(load_u32(smem + w_base + (p * 4) as u32));
                            fsum += f32::from_bits(u32::from(*h_base.add(p * dim)) << 16) * wt;
                            p += 1;
                        }
                    } else {
                        let d_base = in1.add(row * picks * dim + ch).cast::<u32>();
                        let mut p = 0usize;
                        while p < picks {
                            let wt = f32::from_bits(load_u32(smem + w_base + (p * 4) as u32));
                            let d = f32::from_bits(ld_global_nc_u32(d_base.add(p * dim)));
                            fsum += rn(d) * wt;
                            p += 1;
                        }
                    }
                    let folded = rn(fsum);
                    let own = (first_slot + row - start) % taps;
                    match own { 0 => r0 = folded, 1 => r1 = folded, 2 => r2 = folded, _ => r3 = folded }
                    if !every_ptr.is_null() {
                        *every_ptr.add((row * taps) * dim + ch) = r0;
                        if 1 < taps { *every_ptr.add((row * taps + 1) * dim + ch) = r1; }
                        if 2 < taps { *every_ptr.add((row * taps + 2) * dim + ch) = r2; }
                        if 3 < taps { *every_ptr.add((row * taps + 3) * dim + ch) = r3; }
                    }
                    let mut csum = 0.0f32;
                    let mut lag = taps;
                    while lag > 0 {
                        lag -= 1;
                        let tap = match lag { 0 => t0, 1 => t1, 2 => t2, _ => t3 };
                        let read = match (own + taps - lag) % taps { 0 => r0, 1 => r1, 2 => r2, _ => r3 };
                        csum += tap * rn(read);
                    }
                    let y = rn((csum + rn(folded)) * bound);
                    *output.add(row * dim + ch) = rn(res + y);
                    row += 1;
                }
            }
            return;
        }
        let heads = *w.add(9) as usize;
        let form = *w.add(10) as u32;
        if subop == STREAM_OP_ROUTE_FOLD_CONV_PREFILL {
            // Prefill fused `RouteFold (FOLD_HALF)` + `AfterExperts ShortConv` + `next_ring` over
            // `SHORT_CONV_PREFILL_TILE` (64)-row channel tiles (`grid = (dim.div_ceil(256), rows.div_ceil(64), 1)`).
            // Eliminates the `[rows, dim]` intermediate `folded` buffer (37.7 MB write + 4x read) and
            // `prefill_ring_after` gather launch while keeping all 4 fixed taps and 4-row ring in registers.
            // SAFETY: host checked via `fast_ops::check_route_fold_conv_prefill` that all extents, alignments,
            // and disjointness hold (`rows >= taps`, `taps in 2..=4`, `dim` even). Each `(row, ch)` of `output`
            // and each `(slot, ch)` of `next_ring` (written only by the last row tile `r_end == rows`) has one writer.
            let picks = aux;
            let taps = heads;
            let first_slot = form as usize;
            let p_ring = *w.add(12) as *const f32;
            let next_ring = *w.add(13) as *mut f32;
            let parts = dim.div_ceil(threads);
            let row_tiles = rows.div_ceil(SHORT_CONV_PREFILL_TILE);
            if !stream_ops_admitted(subop, rows, dim, picks, threads)
                || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || rows < taps || first_slot >= taps
                || core::arch::nvptx::_grid_dim_x() as usize != parts
                || core::arch::nvptx::_grid_dim_y() as usize != row_tiles
                || p_ring.is_null() || next_ring.is_null()
                || ((in0 as usize) | (in1 as usize) | (in2 as usize) | (weight as usize)
                    | (output as usize) | (p_ring as usize) | (next_ring as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let ch = (core::arch::nvptx::_block_idx_x() as usize) * threads + (core::arch::nvptx::_thread_idx_x() as usize);
            if ch >= dim { return; }
            let tile = core::arch::nvptx::_block_idx_y() as usize;
            let r0 = tile * SHORT_CONV_PREFILL_TILE;
            if r0 >= rows { return; }
            let r_end = (r0 + SHORT_CONV_PREFILL_TILE).min(rows);
            macro_rules! ldf {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
            macro_rules! ldh {($p:expr)=>{{let v: u16; core::arch::asm!("ld.global.nc.u16 {v}, [{p}];", v = out(reg16) v, p = in(reg64) $p, options(readonly, nostack)); f32::from_bits(u32::from(v) << 16)}}}
            let h_base = in1.cast::<u16>();
            let (mut p1, mut p2, mut p3, mut p4) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            // `r0` is a multiple of 64 >= `taps - 1`, so either `r0 == 0` (all history from `p_ring`, zero division)
            // or `r0 >= 64` (all history from rows `r0 - (taps - 1) .. r0`).
            if r0 == 0 {
                let mut slot = if first_slot == 0 { taps - 1 } else { first_slot - 1 };
                p1 = ldf!(p_ring.add(slot * dim + ch));
                if taps >= 3 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    p2 = ldf!(p_ring.add(slot * dim + ch));
                }
                if taps >= 4 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    p3 = ldf!(p_ring.add(slot * dim + ch));
                }
            } else {
                let mut r = r0 + 1 - taps;
                while r < r0 {
                    let wp = in0.add(r * picks);
                    let hp = h_base.add(r * picks * dim + ch);
                    let mut fsum = 0.0f32;
                    let mut p = 0usize;
                    while p < picks {
                        fsum += ldh!(hp.add(p * dim)) * ldf!(wp.add(p));
                        p += 1;
                    }
                    p3 = p2;
                    p2 = p1;
                    p1 = rn(fsum);
                    r += 1;
                }
            }
            let t0 = ldf!(weight.add(ch * taps));
            let t1 = ldf!(weight.add(ch * taps + 1));
            let t2 = if taps >= 3 { ldf!(weight.add(ch * taps + 2)) } else { 0.0f32 };
            let t3 = if taps >= 4 { ldf!(weight.add(ch * taps + 3)) } else { 0.0f32 };
            let mut r = r0;
            while r < r_end {
                let wp = in0.add(r * picks);
                let hp = h_base.add(r * picks * dim + ch);
                let mut fsum = 0.0f32;
                let mut p = 0usize;
                while p < picks {
                    fsum += ldh!(hp.add(p * dim)) * ldf!(wp.add(p));
                    p += 1;
                }
                let folded = rn(fsum);
                let mut csum = 0.0f32;
                if taps >= 4 { csum += t3 * rn(p3); }
                if taps >= 3 { csum += t2 * rn(p2); }
                csum += t1 * rn(p1);
                csum += t0 * rn(folded);
                let y = rn((csum + rn(folded)) * bound);
                *output.add(r * dim + ch) = rn(ldf!(in2.add(r * dim + ch)) + y);
                p4 = p3;
                p3 = p2;
                p2 = p1;
                p1 = folded;
                r += 1;
            }
            if r_end == rows {
                let mut slot = ((first_slot as u32 + rows as u32 - 1) % (taps as u32)) as usize;
                *next_ring.add(slot * dim + ch) = p1;
                slot = if slot == 0 { taps - 1 } else { slot - 1 };
                *next_ring.add(slot * dim + ch) = p2;
                if taps >= 3 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    *next_ring.add(slot * dim + ch) = p3;
                }
                if taps >= 4 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    *next_ring.add(slot * dim + ch) = p4;
                }
            }
            return;
        }
        let ring = if subop == STREAM_OP_SHORT_CONV_WIN { *w.add(12) as *const f32 } else { core::ptr::null() };
        let every_ptr = if subop == STREAM_OP_SHORT_CONV_WIN { *w.add(14) as *mut f32 } else { core::ptr::null_mut() };
        // 2D channel-tiled prefill `ShortConv` (`heads == 1`, no token taps, `taps in 2..=4`, `ring` present,
        // `every` null, word 15 == `SHORT_CONV_PREFILL_TILE`): reads each input row once instead of `taps` times
        // and hoists fixed taps into registers once per 64-row tile.
        if subop == STREAM_OP_SHORT_CONV_WIN && !ring.is_null() && every_ptr.is_null()
            && *w.add(15) as usize == SHORT_CONV_PREFILL_TILE
        {
            let taps = aux;
            let width = dim;
            let first_slot = *w.add(13) as usize;
            let parts = width.div_ceil(threads);
            let row_tiles = rows.div_ceil(SHORT_CONV_PREFILL_TILE);
            if !stream_ops_admitted(subop, rows, width, taps, threads)
                || heads != 1 || (form & CONV_TOKEN_TAPS) != 0
                || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || first_slot >= taps
                || core::arch::nvptx::_grid_dim_x() as usize != parts
                || core::arch::nvptx::_grid_dim_y() as usize != row_tiles
                || ((in0 as usize) | (in2 as usize) | (weight as usize) | (output as usize) | (ring as usize)) & 3 != 0
            {
                core::arch::nvptx::trap();
            }
            let ch = (core::arch::nvptx::_block_idx_x() as usize) * threads + (core::arch::nvptx::_thread_idx_x() as usize);
            if ch >= width { return; }
            let tile = core::arch::nvptx::_block_idx_y() as usize;
            let r0 = tile * SHORT_CONV_PREFILL_TILE;
            if r0 >= rows { return; }
            let r_end = (r0 + SHORT_CONV_PREFILL_TILE).min(rows);
            macro_rules! ldf {($p:expr)=>{{let v: f32; core::arch::asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v, p = in(reg64) $p, options(readonly, nostack)); v}}}
            let (mut p1, mut p2, mut p3) = if r0 == 0 {
                let mut slot = if first_slot == 0 { taps - 1 } else { first_slot - 1 };
                let v1 = ldf!(ring.add(slot * width + ch));
                let v2 = if taps >= 3 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    ldf!(ring.add(slot * width + ch))
                } else { 0.0f32 };
                let v3 = if taps >= 4 {
                    slot = if slot == 0 { taps - 1 } else { slot - 1 };
                    ldf!(ring.add(slot * width + ch))
                } else { 0.0f32 };
                (v1, v2, v3)
            } else {
                (
                    ldf!(in0.add((r0 - 1) * width + ch)),
                    if taps >= 3 { ldf!(in0.add((r0 - 2) * width + ch)) } else { 0.0f32 },
                    if taps >= 4 { ldf!(in0.add((r0 - 3) * width + ch)) } else { 0.0f32 },
                )
            };
            let t0 = ldf!(weight.add(ch * taps));
            let t1 = ldf!(weight.add(ch * taps + 1));
            let t2 = if taps >= 3 { ldf!(weight.add(ch * taps + 2)) } else { 0.0f32 };
            let t3 = if taps >= 4 { ldf!(weight.add(ch * taps + 3)) } else { 0.0f32 };
            let has_add = (form & CONV_ADDEND) != 0;
            let round_sum = (form & CONV_ROUND_SUM) != 0;
            let mut row = r0;
            while row < r_end {
                let cur = ldf!(in0.add(row * width + ch));
                let mut csum = 0.0f32;
                if taps >= 4 { csum += t3 * rn(p3); }
                if taps >= 3 { csum += t2 * rn(p2); }
                csum += t1 * rn(p1);
                csum += t0 * rn(cur);
                let y = rn((csum + rn(cur)) * bound);
                *output.add(row * width + ch) = if !has_add { y } else if !round_sum { ldf!(in2.add(row * width + ch)) + y } else { rn(ldf!(in2.add(row * width + ch)) + y) };
                p3 = p2;
                p2 = p1;
                p1 = cur;
                row += 1;
            }
            return;
        }
        let parts = if !ring.is_null() && every_ptr.is_null() { 1 } else { stream_ops_parts(dim) };
        let block = core::arch::nvptx::_block_idx_x() as usize;
        let (row, part) = (block / parts, block % parts);
        if !stream_ops_admitted(subop, rows, dim, aux, threads) || row >= rows || core::arch::nvptx::_grid_dim_x() as usize != rows * parts { core::arch::nvptx::trap(); }
        if ((in0 as usize) | (in1 as usize) | (in2 as usize) | (weight as usize) | (output as usize)) & 3 != 0 {
            core::arch::nvptx::trap();
        }
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (first, step) = (part * threads + tid, parts * threads);

        if subop == STREAM_OP_SHORT_CONV_WIN {
            // Window ShortConv: `in0` = window[(rows + taps - 1) * width] (or inputs[rows * width] with `ring`),
            // `in1` = optional token taps, `in2` = optional addend, `weight` = fixed taps [width * taps].
            // Multi-chat kept-ring mode (`slot_word & SHORT_CONV_MULTI_TAG != 0`): `ring` holds `chats * taps * width`
            // contiguous words verified by `check_short_conv_form`; `short_conv_multi_decode` re-validates `start_mask`,
            // per-chat `first_slot < taps`, and `chat < chats` before indexing `ring.add(chat * taps * width)`.
            let taps = aux;
            let width = dim;
            if heads == 0 || !width.is_multiple_of(heads) { core::arch::nvptx::trap(); }
            let slot_word = *w.add(13) as usize;
            let multi = (slot_word & SHORT_CONV_MULTI_TAG) != 0;
            let every = every_ptr;
            let (chat_ring, first_slot, local_row) = if !ring.is_null() && multi {
                let Some((start_mask, slots, _chats)) = short_conv_multi_decode(rows, taps, slot_word) else {
                    core::arch::nvptx::trap();
                };
                if (ring as usize) & 3 != 0 || every.is_null() || *w.add(15) as usize != slot_word {
                    core::arch::nvptx::trap();
                }
                let prefix = u32::from(start_mask) & ((2u32 << row) - 1);
                let chat = (prefix.count_ones() as usize) - 1;
                let chat_start = 31 - (prefix.leading_zeros() as usize);
                (ring.add(chat * taps * width), ((slots >> (2 * chat)) & 3) as usize, row - chat_start)
            } else {
                if !ring.is_null() && (slot_word >= taps || (ring as usize) & 3 != 0) { core::arch::nvptx::trap(); }
                (ring, slot_word, row)
            };
            let source = |lag| {
                if chat_ring.is_null() {
                    in0.add((row + taps - 1 - lag) * width)
                } else if lag <= local_row {
                    in0.add((row - lag) * width)
                } else {
                    chat_ring.add(((first_slot + taps + local_row - lag) % taps) * width)
                }
            };
            let head_dim = width / heads;
            let has_token = (form & CONV_TOKEN_TAPS) != 0;
            let has_add = (form & CONV_ADDEND) != 0;
            let round_sum = (form & CONV_ROUND_SUM) != 0;
            // The ring after row `row` of a step that keeps each row (`every`): slot (own - lag) mod taps holds the
            // window row the lag reads, own = (first_slot + local_row) mod taps (the rule of state::many::conv_rings).
            let own = if every.is_null() { 0 } else if multi {
                if (every as usize) & 3 != 0 { core::arch::nvptx::trap(); }
                (first_slot + local_row) % taps
            } else {
                let first = *w.add(15) as usize;
                if first >= taps || (every as usize) & 3 != 0 { core::arch::nvptx::trap(); }
                (first + row) % taps
            };
            let o_ptr = output.add(row * width);
            let add_ptr = if has_add { in2.add(row * width) } else { core::ptr::null() };
            let tok_ptr = if has_token { in1.add(row * heads * taps) } else { core::ptr::null() };
            let mut ch = first;
            while ch < width {
                let head = if heads == 1 { 0 } else { ch / head_dim };
                let mut sum = 0.0f32;
                let mut lag = taps;
                while lag > 0 {
                    lag -= 1;
                    let mut tap = *weight.add(ch * taps + lag);
                    if has_token {
                        tap += rn(*tok_ptr.add(head * taps + lag));
                    }
                    let read = *source(lag).add(ch);
                    if !every.is_null() { *every.add((row * taps + (own + taps - lag) % taps) * width + ch) = read; }
                    sum += tap * rn(read);
                }
                let cur = rn(*source(0).add(ch));
                let y = rn((sum + cur) * bound);
                *o_ptr.add(ch) = if !has_add { y } else if !round_sum { *add_ptr.add(ch) + y } else { rn(*add_ptr.add(ch) + y) };
                ch += step;
            }
        } else if subop == STREAM_OP_ROUTE_FOLD {
            // RouteFold: `in0` = weights[rows * w_stride], `in1` = downs[rows * picks * dim].
            let picks = aux;
            let w_stride = if *w.add(9) == 0 { picks } else { *w.add(9) as usize };
            let w_off = *w.add(10) as usize;
            if tid < picks {
                store_u32(smem + (tid * 4) as u32, (*in0.add(row * w_stride + w_off + tid)).to_bits());
            }
            core::arch::nvptx::_syncthreads();
            let o_ptr = output.add(row * dim);
            if *w.add(FOLD_FORM_WORD) & FOLD_HALF != 0 {
                // `FOLD_HALF`: `in1` holds `rows * picks * dim` BF16 values (host checked, `dim` even), and the value
                // `(row * picks + p) * dim + ch` is the down word `rn` would give: its bits are the upper half of FP32.
                // Its own loop, so the FP32 loop below keeps its registers.
                let h_base = in1.cast::<u16>().add(row * picks * dim);
                let mut ch = first;
                while ch < dim {
                    let mut sum = 0.0f32;
                    let mut p = 0usize;
                    while p < picks {
                        let wt = f32::from_bits(load_u32(smem + (p * 4) as u32));
                        sum += f32::from_bits(u32::from(*h_base.add(p * dim + ch)) << 16) * wt;
                        p += 1;
                    }
                    *o_ptr.add(ch) = rn(sum);
                    ch += step;
                }
                return;
            }
            let d_base = in1.add(row * picks * dim);
            let mut ch = first;
            while ch < dim {
                let mut sum = 0.0f32;
                let mut p = 0usize;
                while p < picks {
                    let wt = f32::from_bits(load_u32(smem + (p * 4) as u32));
                    sum += rn(*d_base.add(p * dim + ch)) * wt;
                    p += 1;
                }
                *o_ptr.add(ch) = rn(sum);
                ch += step;
            }
        } else {
            // GateBlend: `in0` = gates[rows * 2 * dim], `in1` = kept[rows * dim], `in2` = msg[rows * dim].
            let f_ptr = in0.add(row * 2 * dim);
            let g_ptr = f_ptr.add(dim);
            let k_ptr = in1.add(row * dim);
            let m_ptr = in2.add(row * dim);
            let o_ptr = output.add(row * dim);
            let mut ch = first;
            while ch < dim {
                *o_ptr.add(ch) = (1.0f32 - tanh(*f_ptr.add(ch))) * *k_ptr.add(ch) + tanh(*g_ptr.add(ch)) * *m_ptr.add(ch);
                ch += step;
            }
        }
    }
}
