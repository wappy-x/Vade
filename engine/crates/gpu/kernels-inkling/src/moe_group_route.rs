//! Deterministic device-side MoE router grouping (`resident_launch_moe_group_route`, kind `MOE_GROUP_ROUTE = 104`).
//!
//! Eliminates the per-visit GPU-to-CPU router readback and CPU grouping in prefill (`experts.rs`).
//! Given `route: [rows][each]` FP32 (first `picks` words = expert IDs as whole floats, next `picks` words =
//! router weights), validates every expert ID (`< experts`, trapping on NaN/invalid), extracts contiguous
//! `weights_out: [all]` FP32 for `RouteFold`, and groups all `all = rows * picks` picks by expert in
//! strictly ascending `pick = row * picks + slot` order with ZERO atomics.
//!
//! Outputs:
//! - `weights_out`: `[all]` FP32
//! - `in_rows_out`: `[all]` u32 (`in_rows_out[pos] = pick / picks`, input token row for `GateUp`)
//! - `out_slots_out`: `[all]` u32 (`out_slots_out[pos] = pick`, output slot for `Down` -> `RouteFold`)
//! - `tile_meta_out`: `[1 + 3 * max_m_tiles]` u32 (`meta[0] = total_m_tiles`, followed by `(expert, first_pos, m_count)`)
//!
//! Launch: 1 block of 256 threads. Static shared memory = 77,824 bytes.

use crate::tc_ptx::{load_u32, match_any_u32, store_u16, store_u32};
use crate::tc_rows::{MOE_ALL_MAX, MOE_EXPERTS_MAX, MOE_ROUTE_THREADS, TC_BM, moe_route_admitted};

const SLICES: usize = 8;
const IDS_BYTES: usize = MOE_ALL_MAX * 2; // 65,536 bytes (u16 per pick)
const HIST_BYTES: usize = SLICES * MOE_EXPERTS_MAX * 2; // 8,192 bytes (u16 per slice x expert)
const OFF_BYTES: usize = MOE_EXPERTS_MAX * 4; // 2,048 bytes (u32 per expert)
const TILE_OFF_BYTES: usize = MOE_EXPERTS_MAX * 4; // 2,048 bytes (u32 per expert)
const BLOCK_BYTES: usize = IDS_BYTES + HIST_BYTES + OFF_BYTES + TILE_OFF_BYTES; // 77,824 bytes
const _: () = assert!(BLOCK_BYTES == 77_824 && MOE_ROUTE_THREADS == 256 && SLICES == 8);
/// The lanes of a warp; warp `s` takes slice `s` (`MOE_ROUTE_THREADS / LANES == SLICES`).
const LANES: usize = 32;
const _: () = assert!(MOE_ROUTE_THREADS / LANES == SLICES);
/// The expert word of a lane past the end of its slice: no expert has it (`MOE_EXPERTS_MAX` < 2^16).
const NO_EXPERT: u32 = u32::MAX;

/// Replay one recorded command of kind `MOE_GROUP_ROUTE`.
/// SAFETY (caller contract, `resident_launch_moe_group_route`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `MOE_GROUP_ROUTE` recorded by `Backend` after `tc_rows::check_moe_group_route`.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    // - Host checked via `tc_rows::check_moe_group_route` that `route` holds `rows * each` FP32 words,
    //   `weights_out`, `in_rows_out`, `out_slots_out` each hold `all = rows * picks` words, `tile_meta_out`
    //   holds `1 + 3 * max_m_tiles` words, all 4-byte aligned and pairwise non-overlapping.
    // - Device re-checks `moe_route_admitted`, grid == 1, alignment, and every expert ID `< experts` (trapping
    //   on NaN, negative, fractional, or out-of-range IDs).
    // - Every shared-memory access lies within `BLOCK_BYTES`:
    //   `ids[0..all]` (u16) in `0..IDS_BYTES`, `hist[0..8][0..experts]` (u16) in `IDS_BYTES..+HIST_BYTES`,
    //   `expert_offset[0..experts]` (u32) and `expert_tile_offset[0..experts]` (u32) in the remaining 4,096 bytes.
    // - All 256 threads participate in every `bar.sync 0`, and all 32 lanes of a warp in every `match.any.sync` and
    //   `bar.warp.sync` (the loops of passes 2 and 4 run the same count of 32-pick groups on every lane of a warp).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::MOE_GROUP_ROUTE { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let route = *words as *const f32;
        let weights_out = *words.add(1) as *mut f32;
        let in_rows_out = *words.add(2) as *mut u32;
        let out_slots_out = *words.add(3) as *mut u32;
        let tile_meta_out = *words.add(4) as *mut u32;
        let rows = *words.add(5) as usize;
        let each = *words.add(6) as usize;
        let picks = *words.add(7) as usize;
        let experts = *words.add(8) as usize;
        let max_m_tiles = *words.add(9) as usize;

        let threads = core::arch::nvptx::_block_dim_x() as usize;
        if core::arch::nvptx::_block_idx_x() != 0 || core::arch::nvptx::_grid_dim_x() != 1 { core::arch::nvptx::trap(); }
        if !moe_route_admitted(rows, each, picks, experts, max_m_tiles, threads) { core::arch::nvptx::trap(); }
        if ((route as usize) | (weights_out as usize) | (in_rows_out as usize) | (out_slots_out as usize) | (tile_meta_out as usize)) & 3 != 0 {
            core::arch::nvptx::trap();
        }

        let all = rows * picks;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let smem = crate::ptx::shared::<BLOCK_BYTES>();
        let ids_base = smem;
        let hist_base = smem + IDS_BYTES as u32;
        let off_base = hist_base + HIST_BYTES as u32;
        let toff_base = off_base + OFF_BYTES as u32;

        // Pass 1: Zero `hist` (only `SLICES * experts` u16s) and extract+validate `route` -> `ids_smem` & `weights_out`.
        let hist_u32 = (SLICES * experts).div_ceil(2);
        let mut i = tid;
        while i < hist_u32 {
            store_u32(hist_base + (i * 4) as u32, 0);
            i += MOE_ROUTE_THREADS;
        }
        let mut p = tid;
        while p < all {
            let row = p / picks;
            let slot = p % picks;
            let id_f = *route.add(row * each + slot);
            if !(id_f >= 0.0 && id_f < experts as f32) || (id_f as u32) as f32 != id_f || id_f as u32 >= experts as u32 {
                core::arch::nvptx::trap();
            }
            store_u16(ids_base + (p * 2) as u32, id_f as u16);
            *weights_out.add(p) = *route.add(row * each + picks + slot);
            p += MOE_ROUTE_THREADS;
        }
        core::arch::nvptx::_syncthreads();

        // Pass 2: 8 slices count picks per expert in parallel (zero atomics, fixed partition), warp `s` slice `s`, 32
        // picks a group: the lanes of one expert in a group find each other (`match.any.sync`), and the highest of them
        // adds their count. A `bar.warp.sync` orders each group's updates before the next group's.
        let (warp, lane) = (tid / LANES, tid % LANES);
        let below = (1u32 << lane) - 1;
        let slice = |s: usize| ((all * s) / SLICES, (all * (s + 1)) / SLICES);
        {
            let (start, end) = slice(warp);
            let row_hist = hist_base + (warp * experts * 2) as u32;
            let mut k0 = start;
            while k0 < end {
                let k = k0 + lane;
                let e = if k < end { crate::ptx::load_u16(ids_base + (k * 2) as u32) } else { NO_EXPERT };
                let same = match_any_u32(e);
                if e != NO_EXPERT && (same >> lane) == 1 {
                    let addr = row_hist + e * 2;
                    store_u16(addr, (crate::ptx::load_u16(addr) + same.count_ones()) as u16);
                }
                crate::ptx::warp_sync();
                k0 += LANES;
            }
        }
        core::arch::nvptx::_syncthreads();

        // Pass 3a: Per-expert exclusive prefix sum across the 8 slices; store total count in `off_base[e]`.
        let mut e = tid;
        while e < experts {
            let mut sum = 0u32;
            let mut s = 0usize;
            while s < SLICES {
                let addr = hist_base + ((s * experts + e) * 2) as u32;
                let cnt = crate::ptx::load_u16(addr);
                store_u16(addr, sum as u16);
                sum += cnt;
                s += 1;
            }
            store_u32(off_base + (e * 4) as u32, sum);
            e += MOE_ROUTE_THREADS;
        }
        core::arch::nvptx::_syncthreads();

        // Pass 3b: Exclusive prefix sum across experts for pick offsets and M-tile offsets (thread 0, <= 512 iters).
        if tid == 0 {
            let mut off = 0u32;
            let mut toff = 0u32;
            let mut ex = 0usize;
            while ex < experts {
                let cnt = load_u32(off_base + (ex * 4) as u32);
                store_u32(off_base + (ex * 4) as u32, off);
                store_u32(toff_base + (ex * 4) as u32, toff);
                off += cnt;
                toff += cnt.div_ceil(TC_BM as u32);
                ex += 1;
            }
            if off != all as u32 || toff > max_m_tiles as u32 { core::arch::nvptx::trap(); }
            *tile_meta_out = toff;
        }
        core::arch::nvptx::_syncthreads();

        // Pass 3c: Add `expert_offset[e]` into every `hist[s][e]` and emit `tile_meta_out` for expert `e`.
        let mut e = tid;
        while e < experts {
            let base = load_u32(off_base + (e * 4) as u32);
            let next = if e + 1 < experts { load_u32(off_base + ((e + 1) * 4) as u32) } else { all as u32 };
            let cnt = next - base;
            let tbase = load_u32(toff_base + (e * 4) as u32);
            let ntiles = cnt.div_ceil(TC_BM as u32);
            let mut t = 0u32;
            while t < ntiles {
                let rem = cnt - t * TC_BM as u32;
                let m_cnt = if rem > TC_BM as u32 { TC_BM as u32 } else { rem };
                let out = tile_meta_out.add(1 + 3 * (tbase + t) as usize);
                *out = e as u32;
                *out.add(1) = base + t * TC_BM as u32;
                *out.add(2) = m_cnt;
                t += 1;
            }
            let mut s = 0usize;
            while s < SLICES {
                let addr = hist_base + ((s * experts + e) * 2) as u32;
                let local = crate::ptx::load_u16(addr);
                store_u16(addr, (base + local) as u16);
                s += 1;
            }
            e += MOE_ROUTE_THREADS;
        }
        core::arch::nvptx::_syncthreads();

        // Pass 4: 8 slices scatter `in_rows_out` and `out_slots_out` in strictly ascending pick order per expert, warp
        // `s` slice `s`, 32 picks a group: a lane's place is its expert's next place plus the count of the lanes below
        // it with the same expert, and the highest of those lanes moves the next place past all of them.
        {
            let (start, end) = slice(warp);
            let row_hist = hist_base + (warp * experts * 2) as u32;
            let mut k0 = start;
            while k0 < end {
                let k = k0 + lane;
                let e = if k < end { crate::ptx::load_u16(ids_base + (k * 2) as u32) } else { NO_EXPERT };
                let same = match_any_u32(e);
                let addr = row_hist + e.wrapping_mul(2);
                let pos = if e != NO_EXPERT { crate::ptx::load_u16(addr) + (same & below).count_ones() } else { 0 };
                crate::ptx::warp_sync();
                if e != NO_EXPERT {
                    if (same >> lane) == 1 { store_u16(addr, (pos + 1) as u16); }
                    *in_rows_out.add(pos as usize) = (k / picks) as u32;
                    *out_slots_out.add(pos as usize) = k as u32;
                }
                crate::ptx::warp_sync();
                k0 += LANES;
            }
        }
    }
}
