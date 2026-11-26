//! SM121a BF16 tensor-core fused causal/sliding-window attention (`resident_launch_flash_window`, kind `FLASH_WINDOW_ATTN = 101`).
//!
//! Replaces `RelBias + WindowScores + SinkSoftmax + WindowValues` for contiguous multi-row chat spans with
//! `head_dim == 128` and `rel_dim == 16`, eliminating `[rows, heads, view]` score/weight global buffers.
//!
//! Tile geometry: `BQ = 64`, `BKV = 32`, `HEAD_DIM = 128`, 128 threads (4 warps, 16 query rows x 128 cols/warp).
//! Shared memory (`44,288` bytes static, 2 blocks/SM on SM121a):
//! - `Q`: `64 x 128` BF16 (16,384 B, XOR-swizzled, loaded once in prologue)
//! - `K`: `32 x 128` BF16 (8,192 B, XOR-swizzled, `cp.async` overlapped with `P * V`)
//! - `V`: `32 x 128` BF16 (8,192 B, XOR-swizzled, `cp.async` overlapped with `Q * K^T` + softmax)
//! - `M`: `64 x 16` FP32 rounded to BF16 (5,120 B, rows 80 bytes apart, relative state loaded once in prologue)
//! - `L`: `64` u32 (`live_r` per query row, 256 B)
//! - `T`: `16 x 96` FP32 (6,144 B): the relative table words of the 95 distances of one key tile, `cp.async` with `K`
//!
//! Determinism: tiles start at absolute key positions divisible by 32 and run in ascending order.
//! The feature index `d` rises from 0 to 127 in `Q*K^T`, with
//! fixed warp-butterfly (`xor 1`, `xor 2`) reductions, `fixed_math::{exp, sigmoid, rn}`, zero FP atomics.

use crate::fixed_math::{exp, rn, sigmoid};
use crate::block_attention::{BLOCK, CLIP_EPSILON, SOFTMAX_GROUP, live_of, softmax_add, softmax_best, softmax_exp, softmax_pick, softmax_total, softmax_word};
use crate::block_ops::{APART, join_parts};
use crate::flash_window_abi::{
    FW_BKV, FW_BQ, FW_CLIP_IN, FW_CLIP_WORD, FW_EXACT_ONE, FW_HALF_OUT, FW_HEAD_DIM, FW_MODE_WORD, FW_ONE_SPLIT,
    FW_ONE_VIEW_MAX, FW_REL_DIM, FW_SPLIT_WORD, FW_THREADS, FW_WRITTEN_KEYS, FW_WRITTEN_VALUES,
    flash_window_admitted, flash_window_bq_pos, flash_window_gpack, flash_window_one_admitted,
};
use crate::tc_ptx::{
    cp_async_wait_zero, cvt_bf16x2, ld_global_nc_v4, ld_global_nc_v4_u32, ldmatrix_x4, ldmatrix_x4_trans, load_u32,
    mma_m16n8k16_bf16, shfl_xor_f32, shfl_xor_u32, st_shared_v4, st_shared_zero16, store_u16, store_u32,
};

/// Maximum query rows staged in one CTA (`3` GQA heads x `32` positions = `96`).
const MAX_Q_ROWS: usize = 96;
const K_BYTES: usize = FW_BKV * FW_HEAD_DIM * 2; // 8,192
const V_BYTES: usize = FW_BKV * FW_HEAD_DIM * 2; // 8,192
/// BF16 `u16` pitch of one row of `M`: 16 values + 2 pad = 18 u16 = 36 bytes (9 u32 words, bank-conflict-free).
const M_PITCH_U16: usize = FW_REL_DIM + 2;
/// Static shared memory covers `gpack = 3` (`24,576 Q + 8,192 K + 8,192 V + 3,456 M + 128 L + 4,096 T = 48,640 B`),
/// `gpack = 2` (`39,296 B`), `gpack = 1` (`41,472 B`), and `FW_EXACT_ONE` (`42,112 B`): 2 blocks/SM on SM121a.
const SMEM_BYTES: usize = MAX_Q_ROWS * FW_HEAD_DIM * 2 + K_BYTES + V_BYTES + MAX_Q_ROWS * M_PITCH_U16 * 2 + 32 * 4 + FW_REL_DIM * 64 * 4;
const _: () = assert!(FW_BQ == 64 && FW_BKV == 32 && FW_HEAD_DIM == 128 && FW_REL_DIM == 16 && FW_THREADS == 128);
const _: () = assert!(SMEM_BYTES == 48_640 && 64 * FW_HEAD_DIM * 2 + K_BYTES + V_BYTES + 64 * M_PITCH_U16 * 2 + 64 * 4 + FW_REL_DIM * 96 * 4 <= SMEM_BYTES);

/// Zero-bank-conflict XOR-swizzled byte offset within a 256-byte-stride BF16 tile (`row`, 16B `chunk` in 0..16).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 256 + ((chunk ^ (row & 7)) * 16)) as u32
}

#[inline(always)]
fn fmax(a: f32, b: f32) -> f32 {
    if b > a { b } else { a }
}

/// Replay one recorded command of kind `FLASH_WINDOW_ATTN`.
/// SAFETY (caller contract, `resident_launch_flash_window`): `table` and `index` point to an initialized
/// 16-byte aligned `Command` of kind `FLASH_WINDOW_ATTN` recorded by `Backend::flash_window_attn_into` after
/// `flash_window::check_flash_window` verified every extent, alignment, and non-overlap.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY:
    // - `table[index]` is one initialized, read-only, 16-byte aligned Command recorded in the resident queue.
    //   Wrong kind traps before reading payload words.
    // - Host verified via `check_flash_window` that `query`/`gate`/`output` hold `rows * heads * 128` FP32,
    //   `keys`/`values` hold `view * kv_heads * 64` u32 packed BF16, `moved` holds `rows * heads * 16` FP32,
    //   `rel_table` holds `16 * extent` FP32, `sink` holds `rows * heads` FP32, `steps` holds `rows * 2` FP32,
    //   and `output` is disjoint from all inputs.
    // - Device re-checks `flash_window_admitted`, contiguous chat step invariants (`1 <= live <= view`,
    //   `slot < view`, `slot_r == (slot_0 + r) % view`, `arc_len <= view`), and traps on any malformed step.
    // - Shared memory stays strictly inside the block's static `SMEM_BYTES` (44,288 bytes). `T` word (j, i) is written
    //   only by the copy or the zero store of thread `(j * T_PITCH + i) % FW_THREADS`, and only between the barrier that
    //   ends every read of the tile before (after the V wait) and the barrier after the K wait; a copy reads relative
    //   table word `j * extent + d` only for `0 <= d < extent` (the host checked `16 * extent` words).
    // - All 128 threads participate in every `bar.sync 0` and every warp-synchronous `ldmatrix`/`mma`/`shfl`.
    // - With `FW_CLIP_IN` in word `FW_CLIP_WORD` (prefill), query and moved are the products before their clips. Words 17
    //   and 18 are 0 or the scales of `heads * 128` and `heads * 16` words, word 19 is 0 or one position factor word a
    //   row (`rows` words), words 20 and 21 the bounds (`Backend::flash_window_attn_into` checked them). The block reads
    //   only the rows `q0 .. q0 + m_cnt` of its head; warp w writes the Q and M rows w, w + 4, .. (each word once).
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::FLASH_WINDOW_ATTN { core::arch::nvptx::trap(); }
        let w = c.payload.as_ptr().cast::<u64>();
        let query = *w as *const f32;
        let keys = *w.add(1) as *const u32;
        let values = *w.add(2) as *const u32;
        let moved = *w.add(3) as *const f32;
        let rel_table = *w.add(4) as *const f32;
        let sink = *w.add(5) as *const f32;
        let gate = *w.add(6) as *const f32;
        let steps = *w.add(7) as *const f32;
        let output = *w.add(8) as *mut f32;
        let rows = *w.add(9) as usize;
        let heads = *w.add(10) as usize;
        let kv_heads = *w.add(11) as usize;
        let view = *w.add(12) as usize;
        let extent = *w.add(13) as usize;
        let scale = f32::from_bits(*w.add(14) as u32);
        let first_key_phase = *w.add(15) as usize;
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let block = core::arch::nvptx::_block_idx_x() as usize;

        // The one shared array of the entry: both forms take their words from it (a second `shared` call would declare
        // a second array).
        let smem = crate::ptx::shared::<SMEM_BYTES>();
        if *w.add(FW_MODE_WORD) == FW_EXACT_ONE {
            // The exact form, one query row or a few. Blocks of a head (grid == rows * heads * FW_ONE_SPLIT) make the
            // attention of one query head of one row: the words of arms RelBias, WindowScores, SinkSoftmax and
            // WindowValues (block_attention::unit), each with the products, the sums in the order and the roundings of
            // the arm, in shared memory between them. A row reads its own query, relative state, sink, gate, step pair
            // and position word, and every row reads the same key and value view: a row's words are those of the
            // one-row form with its own operands, as the arms give a row of a launch for rows.
            // Barriers are core::arch::nvptx::_syncthreads (a convergent call for the compiler): the asm barrier of
            // crate::ptx::sync was copied into both sides of a branch on the thread index, and the block hung.
            // SAFETY (this form):
            // - The host (`Backend::flash_window_one_into`) checked with `rows` rows: query and gate hold rows * heads *
            //   128 FP32, keys and values view * kv_heads * 64 packed words (16-byte aligned), moved rows * heads * 16,
            //   rel_table 16 * extent, sink rows * heads, steps rows * 2, output rows * heads * 128, disjoint from every
            //   input; the device re-checks the shape, the alignment of the planes, the 128 threads and the grid
            //   (flash_window_one_admitted). Query and moved are the products before their clips; words 17 and 18 are 0
            //   or scales of heads * 128 and heads * 16 words (one set for every row), word 19 is 0 or one position
            //   factor word a row (checked by the host likewise). Row r = block / (heads * FW_ONE_SPLIT) < rows reads
            //   and writes only its own part of each of these.
            // - The live count n and the slot come from `block_attention::live_of` (n <= view, slot < view), so every
            //   key or value row (slot - dist) mod view is a row of the view for dist < n; a bad step word gives NaN
            //   words, as the arms do.
            // - Shared words used: E [0, 4 * FW_ONE_VIEW_MAX) (n <= FW_ONE_VIEW_MAX == 6144 words, checked after live_of),
            //   Q [24576, 25088), M [25088, 25152), R [25152, 25728), T [25728, 42112) (the clip lanes, the key stages of
            //   the 4 warps, the bests, then the value sums): all below SMEM_BYTES (44,288). Each word is written by
            //   one thread between barriers that every thread of the block reaches (all loop bounds depend only on n,
            //   the block's head and constants). Key rows (row (slot - dist) mod view of key head kv, 16-byte aligned: the
            //   plane is, and a row is 256 bytes) are staged through shared words; value words are read into registers.
            // - Only the thread of sum group 0 writes, output word head * 128 + each of its channels, through the raw
            //   pointer.
            let split = *w.add(FW_SPLIT_WORD) as usize;
            let (wk_addr, wv_addr) = (*w.add(FW_WRITTEN_KEYS), *w.add(FW_WRITTEN_VALUES));
            let plane_count = *w.add(crate::flash_window_abi::FW_PLANE_COUNT) as usize;
            if !flash_window_one_admitted(rows, heads, kv_heads, view, extent, keys as usize, values as usize, threads,
                core::arch::nvptx::_grid_dim_x() as usize, split) || FW_HEAD_DIM != FW_THREADS
                || (wk_addr == 0) != (wv_addr == 0)
                || (wk_addr != 0 && (plane_count != 0 || ((wk_addr | wv_addr) & 15) != 0)) {
                core::arch::nvptx::trap();
            }
            // Block b: row b / (heads * split) (split, payload word FW_SPLIT_WORD: the blocks of a query head, 1, 2 or
            // FW_ONE_SPLIT); in the row, block h of the row makes query head h / split and its part h mod split: the
            // FW_ONE_SPLIT / split quarters of the channels from quarter part * FW_ONE_SPLIT / split, one after the
            // other (each block makes the scores and the weights of its head itself). Thread t: channel quarter * 32 +
            // t mod 32 and sum group t / 32 of the values of each of its quarters.
            let (row, block) = (block / (heads * split), block % (heads * split));
            // SAFETY: a per-row table contains exactly two host-checked addresses for each
            // admitted row. Each plane has the same checked extent and cannot meet the output.
            let (keys, values) = if plane_count == 0 { (keys, values) } else {
                if plane_count != rows || row >= rows { core::arch::nvptx::trap(); }
                let pair = w.add(crate::flash_window_abi::FW_PLANES_AT + 2 * row);
                (*pair as *const u32, *pair.add(1) as *const u32)
            };
            let (query, moved, sink, gate, steps, output) = (query.add(row * heads * FW_HEAD_DIM), moved.add(row * heads * FW_REL_DIM),
                sink.add(row * heads), gate.add(row * heads * FW_HEAD_DIM), steps.add(2 * row), output.add(row * heads * FW_HEAD_DIM));
            let (head, part, tid) = (block / split, block % split, core::arch::nvptx::_thread_idx_x() as usize);
            let quarters = FW_ONE_SPLIT / split;
            let channel_of = |q: usize| (part * quarters + q) * (FW_HEAD_DIM / FW_ONE_SPLIT) + tid % 32;
            let Some((n, slot)) = live_of(core::slice::from_raw_parts(steps, 2), view) else {
                if tid < 32 { let mut q = 0; while q < quarters { output.add(head * FW_HEAD_DIM + channel_of(q)).write(f32::NAN); q += 1; } }
                return;
            };
            if n > FW_ONE_VIEW_MAX { core::arch::nvptx::trap(); }
            const E: u32 = 0;
            const Q: u32 = (FW_ONE_VIEW_MAX * 4) as u32;
            const M: u32 = Q + (FW_HEAD_DIM * 4) as u32;
            const R: u32 = M + (FW_REL_DIM * 4) as u32;
            const T: u32 = R + 64 + (FW_THREADS * 4) as u32;
            const _: () = assert!(FW_REL_DIM == 16 && T as usize + 4096 * 4 <= SMEM_BYTES && T % 16 == 0 && FW_HEAD_DIM == 32 * FW_ONE_SPLIT && FW_THREADS == 128);
            let at = |dist: usize| smem + E + (4 * dist) as u32;
            let kv = head / (heads / kv_heads);
            let row_of = |dist: usize| if dist <= slot { slot - dist } else { slot + view - dist };
            // HeadClip of query head `head` and of its relative state (block_attention::clipped): each value read as
            // BF16, times its scale when one is given (payload words 17 and 18, 0 for none); the squares of the head
            // in the 32-lane tree (lane = place mod 32, a chain in rising place from +0.0, joined as block_ops::join:
            // warp 0 for the query, warp 1 for the relative state); the clip factor; one round; with a position
            // factor (word 19, 0 for none) the product with it and one more round. Bounds: words 20 and 21.
            let (query_scale, moved_scale, position) = (*w.add(17) as *const f32, *w.add(18) as *const f32, *w.add(19) as *const f32);
            let position = if position.is_null() { position } else { position.add(row) };
            let (query_bound, moved_bound) = (f32::from_bits(*w.add(20) as u32), f32::from_bits(*w.add(21) as u32));
            let scaled = |x: f32, scale: *const f32, at: usize| if scale.is_null() { rn(x) } else { rn(x) * *scale.add(at) };
            let query_value = scaled(*query.add(head * FW_HEAD_DIM + tid), query_scale, head * FW_HEAD_DIM + tid);
            let moved_value = if tid < FW_REL_DIM { scaled(*moved.add(head * FW_REL_DIM + tid), moved_scale, head * FW_REL_DIM + tid) } else { 0.0 };
            crate::ptx::store(smem + Q + (4 * tid) as u32, query_value);
            if tid < FW_REL_DIM { crate::ptx::store(smem + M + (4 * tid) as u32, moved_value); }
            core::arch::nvptx::_syncthreads();
            // The 32 lane chains of each head go to shared words (query lanes at T, relative lanes at T + 128) and are
            // joined there as block_ops::join does: at distance 16, 8, 4, 2, 1 lane l below the distance adds lane
            // l + distance, one level between two barriers (no shuffle: every lane of a warp would have to meet the
            // same shuffle instruction).
            if tid < 64 {
                let lane = tid % 32;
                let mut squares = 0.0f32;
                if tid < 32 {
                    let mut j = lane;
                    while j < FW_HEAD_DIM { let v = crate::ptx::load(smem + Q + (4 * j) as u32); squares += v * v; j += 32; }
                } else if lane < FW_REL_DIM {
                    let v = crate::ptx::load(smem + M + (4 * lane) as u32); squares += v * v;
                }
                crate::ptx::store(smem + T + (4 * tid) as u32, squares);
            }
            core::arch::nvptx::_syncthreads();
            let mut distance = 16usize;
            while distance > 0 {
                if tid < 64 && tid % 32 < distance {
                    let at_lane = smem + T + (4 * tid) as u32;
                    crate::ptx::store(at_lane, crate::ptx::load(at_lane) + crate::ptx::load(at_lane + (4 * distance) as u32));
                }
                core::arch::nvptx::_syncthreads();
                distance >>= 1;
            }
            let clip = |value: f32, squares: f32, bound: f32| {
                // block_attention::clip_factor: the reciprocal of the length, then the product with the threshold.
                let reciprocal = 1.0 / (crate::block_ops::root(squares) + CLIP_EPSILON);
                let factor = bound * reciprocal;
                let clipped = rn(value * if factor > 1.0 { 1.0 } else { factor });
                if position.is_null() { clipped } else { rn(clipped * *position) }
            };
            crate::ptx::store(smem + Q + (4 * tid) as u32, rn(clip(query_value, crate::ptx::load(smem + T), query_bound)));
            if tid < FW_REL_DIM { crate::ptx::store(smem + M + (4 * tid) as u32, rn(clip(moved_value, crate::ptx::load(smem + T + 128), moved_bound))); }
            let sink_word = rn(*sink.add(head)); // softmax_sink: the sink of the head read as BF16
            core::arch::nvptx::_syncthreads();
            // WindowScores with the RelBias word of each distance: thread t takes distance first + t of each group of
            // 128. Its key row (64 packed words) is read into registers first, 16 loads of 4 words, then the chain
            // runs over the words in rising place: query value 2i times the low half of word i, then 2i + 1 times
            // the high half.
            // Rounds of 128 distances, thread t takes distance round + t. Warp w stages the key rows of its 32
            // distances in its 4 KB of T, half a row (32 words) at a time: lane l reads chunk l mod 8 of row
            // 4i + l / 8 (8 lanes a row, so a warp read covers 4 rows), and chunk c of row j lies at j * 128 +
            // (c ^ (j mod 8)) * 16. Then each lane runs the chain of its own row over the words in rising place. A
            // distance at or past n reads the row of n - 1 and makes no score (with n == 0 no round runs). The loop bounds
            // are the same for the whole block, so every thread reaches every barrier.
            let (warp, lane) = (tid / 32, tid % 32);
            let stage = smem + T + (4096 * warp) as u32;
            let written_k = wk_addr as *const u32;
            let key_row_plain = |dist: usize| keys.add((row_of(if dist < n { dist } else { n.saturating_sub(1) }) * kv_heads + kv) * 64);
            let mut round = 0usize;
            while round < n {
                let dist = round + tid;
                let mut sum = 0.0f32;
                let use_written = round == 0 && !written_k.is_null();
                let mut half = 0usize;
                while half < 2 {
                    let got = if use_written {
                        let key_row = |d0: usize| {
                            let d = if d0 < n { d0 } else { n.saturating_sub(1) };
                            if d <= row { written_k.add(((row - d) * kv_heads + kv) * 64) } else { keys.add((row_of(d) * kv_heads + kv) * 64) }
                        };
                        let piece = |i: usize| ld_global_nc_v4_u32(key_row(32 * warp + 4 * i + lane / 8).add(32 * half + 4 * (lane % 8)));
                        [piece(0), piece(1), piece(2), piece(3), piece(4), piece(5), piece(6), piece(7)]
                    } else {
                        let piece = |i: usize| ld_global_nc_v4_u32(key_row_plain(round + 32 * warp + 4 * i + lane / 8).add(32 * half + 4 * (lane % 8)));
                        [piece(0), piece(1), piece(2), piece(3), piece(4), piece(5), piece(6), piece(7)]
                    };
                    let put = |i: usize, v: [u32; 4]| { let j = 4 * i + lane / 8; st_shared_v4(stage + (j * 128 + ((lane % 8) ^ (j % 8)) * 16) as u32, v); };
                    put(0, got[0]); put(1, got[1]); put(2, got[2]); put(3, got[3]); put(4, got[4]); put(5, got[5]); put(6, got[6]); put(7, got[7]);
                    // Each warp owns a disjoint 4 KB slice (`stage = smem + T + 4096 * warp`), so warp sync suffices.
                    core::arch::asm!("bar.warp.sync 0xffffffff;", options(nostack));
                    macro_rules! chunk { ($c:expr) => {{
                        let key = crate::ptx::load4(stage + (lane * 128 + (($c ^ (lane % 8)) * 16)) as u32);
                        let qa = crate::ptx::load4(smem + Q + (32 * (8 * half + $c)) as u32);
                        let qb = crate::ptx::load4(smem + Q + (32 * (8 * half + $c) + 16) as u32);
                        sum += f32::from_bits(qa[0]) * f32::from_bits(key[0] << 16);
                        sum += f32::from_bits(qa[1]) * f32::from_bits(key[0] & 0xffff_0000);
                        sum += f32::from_bits(qa[2]) * f32::from_bits(key[1] << 16);
                        sum += f32::from_bits(qa[3]) * f32::from_bits(key[1] & 0xffff_0000);
                        sum += f32::from_bits(qb[0]) * f32::from_bits(key[2] << 16);
                        sum += f32::from_bits(qb[1]) * f32::from_bits(key[2] & 0xffff_0000);
                        sum += f32::from_bits(qb[2]) * f32::from_bits(key[3] << 16);
                        sum += f32::from_bits(qb[3]) * f32::from_bits(key[3] & 0xffff_0000);
                    }}; }
                    chunk!(0); chunk!(1); chunk!(2); chunk!(3); chunk!(4); chunk!(5); chunk!(6); chunk!(7);
                    core::arch::asm!("bar.warp.sync 0xffffffff;", options(nostack));
                    half += 1;
                }
                if dist < n {
                    // The RelBias word: the 16 table words of the distance are read first (as fast_arms::rel_word
                    // does), then the chain in rising place.
                    let bias = if dist < extent {
                        let tap = |i: usize| *rel_table.add(i * extent + dist);
                        let taps = [tap(0), tap(1), tap(2), tap(3), tap(4), tap(5), tap(6), tap(7), tap(8), tap(9), tap(10), tap(11), tap(12), tap(13), tap(14), tap(15)];
                        let m = |i: usize| crate::ptx::load(smem + M + (4 * i) as u32);
                        let mut b = 0.0f32;
                        b += m(0) * taps[0]; b += m(1) * taps[1]; b += m(2) * taps[2]; b += m(3) * taps[3];
                        b += m(4) * taps[4]; b += m(5) * taps[5]; b += m(6) * taps[6]; b += m(7) * taps[7];
                        b += m(8) * taps[8]; b += m(9) * taps[9]; b += m(10) * taps[10]; b += m(11) * taps[11];
                        b += m(12) * taps[12]; b += m(13) * taps[13]; b += m(14) * taps[14]; b += m(15) * taps[15];
                        rn(b)
                    } else { 0.0 };
                    crate::ptx::store(at(dist), rn(rn(rn(sum) * scale) + bias));
                }
                round += FW_THREADS;
            }
            core::arch::nvptx::_syncthreads();
            // SinkSoftmax, the staged form of the arms entry: the best of each thread, of each warp, the top; the exps
            // in place of the scores; the block parts and their chain; the weights in place of the exps.
            let score = |dist: usize| crate::ptx::load(at(dist));
            let (value, best) = softmax_best(n, tid, FW_THREADS, &score);
            // The best of every thread goes to shared words (T, two a thread) and a tree keeps, of two bests, the larger
            // value or on equal values the earlier distance: the winner is the first of the largest scores, as the
            // pick of `softmax_pick` over the bests in any order gives it; then the pick against the sink.
            crate::ptx::store(smem + T + (8 * tid) as u32, value);
            crate::ptx::store(smem + T + (8 * tid + 4) as u32, f32::from_bits(best));
            core::arch::nvptx::_syncthreads();
            let mut distance = FW_THREADS / 2;
            while distance > 0 {
                if tid < distance {
                    let (mine, other) = (smem + T + (8 * tid) as u32, smem + T + (8 * (tid + distance)) as u32);
                    let (v, d) = (crate::ptx::load(mine), crate::ptx::load(mine + 4).to_bits());
                    let (w2, e2) = (crate::ptx::load(other), crate::ptx::load(other + 4).to_bits());
                    if w2 > v || w2 == v && e2 < d {
                        crate::ptx::store(mine, w2);
                        crate::ptx::store(mine + 4, f32::from_bits(e2));
                    }
                }
                core::arch::nvptx::_syncthreads();
                distance >>= 1;
            }
            if tid == 0 {
                let winner = (crate::ptx::load(smem + T), crate::ptx::load(smem + T + 4).to_bits());
                crate::ptx::store(smem + R + 32, softmax_pick(sink_word, core::iter::once(winner)));
            }
            core::arch::nvptx::_syncthreads();
            let top = crate::ptx::load(smem + R + 32);
            let mut j = tid;
            while j < n { crate::ptx::store(at(j), softmax_exp(crate::ptx::load(at(j)), top)); j += FW_THREADS; }
            core::arch::nvptx::_syncthreads();
            let parts_at = smem + R + 64;
            let (blocks, mut chain, mut group) = (n.div_ceil(BLOCK), 0.0f32, 0usize);
            while group < blocks {
                let (q, part) = (group + tid / APART, tid % APART);
                if q < blocks {
                    let begin = q * BLOCK;
                    let end = if n - begin > BLOCK { begin + BLOCK } else { n };
                    // softmax_part_sum (block_ops::lane_part) written out here, the same four chains and the same
                    // join: as a call it takes the terms by address, and this entry keeps no stack.
                    let (mut a, mut b, mut c, mut d) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut base = begin + part;
                    while base + 3 * APART < end {
                        a += score(base); b += score(base + APART); c += score(base + 2 * APART); d += score(base + 3 * APART);
                        base += 32;
                    }
                    if base < end { a += score(base); }
                    if base + APART < end { b += score(base + APART); }
                    if base + 2 * APART < end { c += score(base + 2 * APART); }
                    crate::ptx::store(parts_at + (4 * tid) as u32, (a + c) + (b + d));
                }
                core::arch::nvptx::_syncthreads();
                let mut k = 0usize;
                while k < SOFTMAX_GROUP && group + k < blocks {
                    let part = |p: usize| crate::ptx::load(parts_at + (4 * (k * APART + p)) as u32);
                    chain = softmax_add(chain, [part(0), part(1), part(2), part(3), part(4), part(5), part(6), part(7)]);
                    k += 1;
                }
                core::arch::nvptx::_syncthreads();
                group += SOFTMAX_GROUP;
            }
            let total = softmax_total(chain, sink_word, top);
            let mut j = tid;
            while j < n { crate::ptx::store(at(j), softmax_word(j, n, crate::ptx::load(at(j)), total)); j += FW_THREADS; }
            core::arch::nvptx::_syncthreads();
            // WindowValues of each channel of the block, quarter after quarter: in blocks of `BLOCK` distances (as
            // `block_attention::distance_tree` does), the terms weight * value in the lanes of `lane_part` (distance
            // blk + 8a + p + 32m goes to part p, sum a, in rising m). Thread t keeps the eight sums of a = t / 32;
            // the 8 value words of a group of 32 distances are read first. All loop bounds depend only on `n` and
            // `quarters`, so every thread reaches every barrier.
            let mut q = 0usize;
            while q < quarters {
            let channel = channel_of(q);
            let out = output.add(head * FW_HEAD_DIM + channel);
            let (place, low, a) = (kv * 64 + channel / 2, channel % 2 == 0, tid / 32);
            let stride = kv_heads * 64;
            // The value word of distance d (a distance at or past n reads the row of n - 1, a row of the view; its
            // term is not added; with n == 0 no value row is read). The 8 words of the next group are read before the terms of the current one.
            // Because `row < FW_ONE_ROWS_MAX == 16`, only the initial group (`blk == 0 && first = 8 * a < 16`) can have `d <= row`;
            // every subsequent iteration (`first + 32 >= 32 > row`) reads unconditionally from `values`.
            let value_word = |d: usize| crate::tc_ptx::ld_global_nc_u32(values.add(row_of(if d < n { d } else { n.saturating_sub(1) }) * stride + place));
            let read_group = |first: usize| [value_word(first), value_word(first + 1), value_word(first + 2), value_word(first + 3),
                value_word(first + 4), value_word(first + 5), value_word(first + 6), value_word(first + 7)];
            let written_v = wv_addr as *const u32;
            let lane = tid % 32;
            let mut total = 0.0f32;
            let mut blk = 0usize;
            while blk < n {
                let end = if n - blk > BLOCK { blk + BLOCK } else { n };
                let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
                let mut first = blk + 8 * a;
                let mut words = if blk == 0 && !written_v.is_null() && first <= row {
                    let vw = |d0: usize| {
                        let d = if d0 < end { d0 } else { end.saturating_sub(1) };
                        if d <= row { crate::tc_ptx::ld_global_nc_u32(written_v.add((row - d) * stride + place)) }
                        else { crate::tc_ptx::ld_global_nc_u32(values.add(row_of(d) * stride + place)) }
                    };
                    [vw(first), vw(first + 1), vw(first + 2), vw(first + 3), vw(first + 4), vw(first + 5), vw(first + 6), vw(first + 7)]
                } else { read_group(first) };
                while first < end {
                    let next = read_group(first + 32);
                    macro_rules! term { ($sum:ident, $p:expr) => {
                        if first + $p < end {
                            let word = words[$p];
                            $sum += crate::ptx::load(at(first + $p)) * f32::from_bits(if low { word << 16 } else { word & 0xffff_0000 });
                        }
                    }; }
                    term!(p0, 0); term!(p1, 1); term!(p2, 2); term!(p3, 3); term!(p4, 4); term!(p5, 5); term!(p6, 6); term!(p7, 7);
                    words = next;
                    first += 32;
                }
                // Sum a of part p of the channel at T + 4 * (32 * (8 * a + p) + tid mod 32); the thread of a = 0 joins them:
                // each part (a0 + a2) + (a1 + a3) as lane_part, the parts as join_parts, added to `total` per `BLOCK`.
                let keep = |p: usize, v: f32| crate::ptx::store(smem + T + (4 * (32 * (8 * a + p) + lane)) as u32, v);
                keep(0, p0); keep(1, p1); keep(2, p2); keep(3, p3); keep(4, p4); keep(5, p5); keep(6, p6); keep(7, p7);
                core::arch::nvptx::_syncthreads();
                if a == 0 {
                    let sum = |a: usize, p: usize| crate::ptx::load(smem + T + (4 * (32 * (8 * a + p) + lane)) as u32);
                    let part = |p: usize| (sum(0, p) + sum(2, p)) + (sum(1, p) + sum(3, p));
                    let parts = [part(0), part(1), part(2), part(3), part(4), part(5), part(6), part(7)];
                    total += join_parts(parts);
                }
                // The sums of the next block/quarter go to the same shared words: every read of these ends before it.
                core::arch::nvptx::_syncthreads();
                blk += BLOCK;
            }
            if a == 0 {
                out.write(rn(rn(total) * sigmoid(rn(*gate.add(head * FW_HEAD_DIM + channel)))));
            }
            q += 1;
            }
            return;
        }
        if !flash_window_admitted(rows, heads, kv_heads, view, extent, query as usize, keys as usize, values as usize, moved as usize, gate as usize, threads) {
            core::arch::nvptx::trap();
        }
        let gpack = flash_window_gpack(heads, kv_heads);
        let bq_pos = flash_window_bq_pos(gpack);
        let group = heads / kv_heads;
        let sub_groups = group / gpack;
        let q_tiles = rows.div_ceil(bq_pos);
        // LPT ordering: launch deepest Q tiles (largest q_tile) first so tail waves hold short causal arcs.
        let sub_g = block % sub_groups;
        let tmp = block / sub_groups;
        let q_tile_rev = tmp % q_tiles;
        let kv_head = tmp / q_tiles;
        if kv_head >= kv_heads { return; }
        let q_tile = q_tiles - 1 - q_tile_rev;
        let head_base = kv_head * group + sub_g * gpack;
        let q0 = q_tile * bq_pos;
        let m_cnt = if rows - q0 > bq_pos { bq_pos } else { rows - q0 };

        // Validate contiguous chat step invariant for this CTA's rows and compute key arc.
        let parse_count = |v: f32, max_v: usize| -> usize {
            let n = v as usize;
            if n as f32 != v || n > max_v { core::arch::nvptx::trap(); }
            n
        };
        let live0 = parse_count(*steps.add(2 * q0), view);
        let slot0 = parse_count(*steps.add(2 * q0 + 1), view - 1);
        if live0 == 0 { core::arch::nvptx::trap(); }
        let arc_start = if live0 - 1 <= slot0 { slot0 - (live0 - 1) } else { slot0 + view - (live0 - 1) };
        let arc_len = (live0 - 1) + m_cnt;
        if arc_len > view { core::arch::nvptx::trap(); }
        // Indices in the tile loop include padding back to the previous absolute key boundary.
        // The physical ring slot alone cannot give this phase when the ring wraps.
        let pad = (first_key_phase % FW_BKV + q0 + FW_BKV - (live0 - 1) % FW_BKV) % FW_BKV;
        let arc_end = pad + arc_len;

        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let (warp, lane) = (tid / 32, tid % 32);
        let (g, t) = (lane / 4, lane % 4);
        let q_rows = gpack * bq_pos;
        let t_pitch = if bq_pos == FW_BQ { 96usize } else { 64usize };
        let warps_per_head = bq_pos / 16;
        let h_local = warp / warps_per_head;
        let w_pos = warp % warps_per_head;
        let head = head_base + h_local;
        let q_row_base = h_local * bq_pos;

        let q_base = smem;
        let k_base = q_base + (q_rows * FW_HEAD_DIM * 2) as u32;
        let m_base = k_base + (K_BYTES + V_BYTES) as u32;
        let l_base = m_base + (q_rows * M_PITCH_U16 * 2) as u32;
        let t_smem = l_base + (bq_pos * 4) as u32;

        // Validate all m_cnt step rows in parallel and store `live_r` into shared memory.
        if tid < bq_pos {
            if tid < m_cnt {
                let lr = parse_count(*steps.add(2 * (q0 + tid)), view);
                let sr = parse_count(*steps.add(2 * (q0 + tid) + 1), view - 1);
                let expect_s = { let s = slot0 + tid; if s >= view { s - view } else { s } };
                if lr == 0 || lr > live0 + tid || sr != expect_s { core::arch::nvptx::trap(); }
                store_u32(l_base + (tid * 4) as u32, lr as u32);
            } else {
                store_u32(l_base + (tid * 4) as u32, 0);
            }
        }

        if (*w.add(FW_CLIP_WORD) & FW_CLIP_IN) != 0 {
            // HeadClip of each warp's 16 query rows (`4` passes x `4` rows/pass, 8 lanes/row): exact match with
            // `fast_norm` mode 2. Each warp clips its own `(head, w_pos*16 .. +16)` slice directly into `Q` and `M` (u16).
            let (query_scale, moved_scale, position) = (*w.add(17) as *const f32, *w.add(18) as *const f32, *w.add(19) as *const f32);
            let (query_bound, moved_bound) = (f32::from_bits(*w.add(20) as u32), f32::from_bits(*w.add(21) as u32));
            let (grp, l) = (lane / 8, lane % 8);
            macro_rules! group_zero {
                ($sum:expr) => {{
                    let mut sum: f32 = $sum;
                    sum += shfl_xor_f32(sum, 4);
                    sum += shfl_xor_f32(sum, 2);
                    sum += shfl_xor_f32(sum, 1);
                    let bits: u32;
                    core::arch::asm!("shfl.sync.idx.b32 {got}, {v}, {src}, 31, 0xffffffff;", got = out(reg32) bits, v = in(reg32) sum.to_bits(), src = in(reg32) (lane & 24) as u32);
                    f32::from_bits(bits)
                }};
            }
            let mut pass = 0usize;
            while pass < 4 {
                let r = w_pos * 16 + pass * 4 + grp;
                let qr = q_row_base + r;
                let live = r < m_cnt;
                let factor = if position.is_null() || !live { 0.0f32 } else { *position.add(q0 + r) };
                let finish = |v: f32, sc: f32| { let clipped = rn(v * sc); if position.is_null() { clipped } else { rn(clipped * factor) } };
                let x = query.add(((q0 + r) * heads + head) * FW_HEAD_DIM);
                let value = |j: usize| { let v = rn(*x.add(j)); if query_scale.is_null() { v } else { v * *query_scale.add(head * FW_HEAD_DIM + j) } };
                let v: [f32; 16] = if live {
                    [value(l), value(l + 8), value(l + 16), value(l + 24), value(l + 32), value(l + 40), value(l + 48), value(l + 56),
                     value(l + 64), value(l + 72), value(l + 80), value(l + 88), value(l + 96), value(l + 104), value(l + 112), value(l + 120)]
                } else { [0.0f32; 16] };
                macro_rules! chain {
                    ($k:literal) => {{
                        let mut sum = 0.0f32;
                        sum += v[$k] * v[$k];
                        sum += v[$k + 4] * v[$k + 4];
                        sum += v[$k + 8] * v[$k + 8];
                        sum += v[$k + 12] * v[$k + 12];
                        sum
                    }};
                }
                let (low, high) = (chain!(0) + chain!(2), chain!(1) + chain!(3));
                let qscale = crate::block_attention::clip_factor(crate::block_ops::root(group_zero!(low + high)), query_bound);
                macro_rules! put {
                    ($($m:literal)*) => {$(
                        let word = if live { finish(v[$m], qscale).to_bits() } else { 0 };
                        store_u16(q_base + swiz128(qr, $m) + (2 * l) as u32, (word >> 16) as u16);
                    )*};
                }
                put!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
                let mrow = moved.add(((q0 + r) * heads + head) * FW_REL_DIM);
                let mvalue = |j: usize| { let v = rn(*mrow.add(j)); if moved_scale.is_null() { v } else { v * *moved_scale.add(head * FW_REL_DIM + j) } };
                let (mv0, mv1) = if live { (mvalue(l), mvalue(l + 8)) } else { (0.0f32, 0.0f32) };
                let (mut s0, mut s1) = (0.0f32, 0.0f32);
                s0 += mv0 * mv0;
                s1 += mv1 * mv1;
                let mscale = crate::block_attention::clip_factor(crate::block_ops::root(group_zero!((s0 + 0.0f32) + (s1 + 0.0f32))), moved_bound);
                let (word0, word1) = if live { (rn(finish(mv0, mscale)).to_bits() >> 16, rn(finish(mv1, mscale)).to_bits() >> 16) } else { (0, 0) };
                store_u16(m_base + ((qr * M_PITCH_U16 + l) * 2) as u32, word0 as u16);
                store_u16(m_base + ((qr * M_PITCH_U16 + l + 8) * 2) as u32, word1 as u16);
                pass += 1;
            }
        } else {
            // Load Q[q0..q0+m_cnt, head_base..+gpack, 0..128] into Q_smem as packed BF16.
            let mut idx = tid;
            let total_q = q_rows * 16;
            while idx < total_q {
                let qr = idx >> 4;
                let ch = idx & 15;
                let hl = qr / bq_pos;
                let r = qr % bq_pos;
                let dst = q_base + swiz128(qr, ch);
                if r < m_cnt {
                    let p = query.add(((q0 + r) * heads + head_base + hl) * FW_HEAD_DIM + ch * 8);
                    let v0 = ld_global_nc_v4(p);
                    let v1 = ld_global_nc_v4(p.add(4));
                    st_shared_v4(dst, [
                        cvt_bf16x2(v0[0], v0[1]),
                        cvt_bf16x2(v0[2], v0[3]),
                        cvt_bf16x2(v1[0], v1[1]),
                        cvt_bf16x2(v1[2], v1[3]),
                    ]);
                } else {
                    st_shared_zero16(dst);
                }
                idx += threads;
            }
            // Load moved[q0..q0+m_cnt, head_base..+gpack, 0..16] into M_smem as BF16 u16.
            let mut midx = tid;
            let total_m = q_rows * FW_REL_DIM;
            while midx < total_m {
                let qr = midx / FW_REL_DIM;
                let j = midx % FW_REL_DIM;
                let hl = qr / bq_pos;
                let r = qr % bq_pos;
                let bits = if r < m_cnt {
                    rn(*moved.add(((q0 + r) * heads + head_base + hl) * FW_REL_DIM + j)).to_bits() >> 16
                } else { 0 };
                store_u16(m_base + ((qr * M_PITCH_U16 + j) * 2) as u32, bits as u16);
                midx += threads;
            }
        }

        let mut t_base = 0u32;
        // Narrow loop-live scalars to u32 and derive constants on the fly so the entry fits 168 regs with 0 spills (2 blk/SM).
        let (view32, extent32, pad32, arc_start32, arc_end32, live0_32, m_cnt32, t_pitch32, threads32) = (
            view as u32, extent as u32, pad as u32, arc_start as u32, arc_end as u32,
            live0 as u32, m_cnt as u32, t_pitch as u32, threads as u32,
        );
        let q0_32 = q0 as u32;
        let head32 = head as u32;
        let tid32 = tid as u32;
        let kv_row_stride = (kv_heads * 64) as u32;
        let keys_h = keys.add(kv_head * 64);
        let values_h = values.add(kv_head * 64);
        macro_rules! issue_kv32 {
            ($dst_base:expr, $plane:expr, $tb:expr) => {{
                let tb: u32 = $tb;
                let mut idx = tid32;
                if tb >= pad32 && tb + FW_BKV as u32 <= arc_end32 {
                    let p0 = { let x = arc_start32 + tb - pad32; if x >= view32 { x - view32 } else { x } };
                    if p0 + FW_BKV as u32 <= view32 {
                        while idx < (FW_BKV * 16) as u32 {
                            let r = idx >> 4;
                            let ch = idx & 15;
                            crate::ptx::copy16($dst_base + swiz128(r as usize, ch as usize), $plane.add(((p0 + r) * kv_row_stride + ch * 4) as usize));
                            idx += threads32;
                        }
                    } else {
                        while idx < (FW_BKV * 16) as u32 {
                            let r = idx >> 4;
                            let ch = idx & 15;
                            let mut phys = p0 + r;
                            if phys >= view32 { phys -= view32; }
                            crate::ptx::copy16($dst_base + swiz128(r as usize, ch as usize), $plane.add((phys * kv_row_stride + ch * 4) as usize));
                            idx += threads32;
                        }
                    }
                } else {
                    while idx < (FW_BKV * 16) as u32 {
                        let r = idx >> 4;
                        let ch = idx & 15;
                        let dst = $dst_base + swiz128(r as usize, ch as usize);
                        let col = tb + r;
                        if col >= pad32 && col < arc_end32 {
                            let mut phys = arc_start32 + col - pad32;
                            if phys >= view32 { phys -= view32; }
                            crate::ptx::copy16(dst, $plane.add((phys * kv_row_stride + ch * 4) as usize));
                        } else {
                            st_shared_zero16(dst);
                        }
                        idx += threads32;
                    }
                }
            }};
        }
        macro_rules! issue_rel32 {
            ($tb:expr) => {{
                let tb: u32 = $tb;
                if tb + FW_BKV as u32 + extent32 > pad32 + live0_32 {
                    let rel_p = *w.add(4) as *const f32;
                    let first = (pad32 + live0_32 - 1) as i32 - (tb + FW_BKV as u32 - 1) as i32;
                    let max_i = (if t_pitch32 == 64 { 32u32 } else { 64u32 }) + FW_BKV as u32 - 1;
                    let total = FW_REL_DIM as u32 * t_pitch32;
                    let mut k = tid32;
                    while k < total {
                        let (j, i) = if t_pitch32 == 64 { (k >> 6, k & 63) } else { (k / 96, k % 96) };
                        let d = first + i as i32;
                        let dst = t_smem + (4 * k);
                        if d >= 0 && (d as u32) < extent32 && i < max_i {
                            crate::ptx::copy4(dst, rel_p.add((j * extent32 + d as u32) as usize).cast::<u32>());
                        } else {
                            store_u32(dst, 0);
                        }
                        k += threads32;
                    }
                }
            }};
        }

        // Prologue: issue K[0] and the table words of tile 0, wait and sync so Q, M, L, T and K[0] are ready.
        issue_kv32!(k_base, keys_h, 0);
        issue_rel32!(0);
        crate::ptx::commit();
        cp_async_wait_zero();
        core::arch::nvptx::_syncthreads();

        // Compute exact `max_min_col = max_{r < m_cnt} (pad + live0 + r - live_r)` in 1 warp butterfly so
        // the Fast Interior path is sound for any admitted `steps` vector.
        let lane32 = lane as u32;
        let my_mc0 = if lane32 < m_cnt32 { pad32 + live0_32 + lane32 - load_u32(l_base + lane32 * 4) } else { 0 };
        let my_mc1 = if t_pitch32 == 96 && lane32 + 32 < m_cnt32 { pad32 + live0_32 + lane32 + 32 - load_u32(l_base + (lane32 + 32) * 4) } else { 0 };
        let mut max_mc = if my_mc1 > my_mc0 { my_mc1 } else { my_mc0 };
        let mut sh = 16u32;
        while sh > 0 {
            let o = shfl_xor_u32(max_mc, sh);
            if o > max_mc { max_mc = o; }
            sh >>= 1;
        }
        let max_min_col = max_mc;

        // Warp's two query row indices per thread in m16n8k16 output (`r0 = w_pos*16 + g`, `r1 = r0 + 8`).
        let r0 = (w_pos * 16 + g) as u32;
        let t32 = t as u32;
        let live_r0 = load_u32(l_base + r0 * 4);
        let live_r1 = load_u32(l_base + (r0 + 8) * 4);
        let qpos_r0 = pad32 + (live0_32 - 1) + r0;

        // Online softmax state initialized with the exact attention sink.
        let mut m_r0 = if r0 < m_cnt32 { rn(*sink.add((q0 + r0 as usize) * heads + head)) } else { 0.0f32 };
        let mut m_r1 = if r0 + 8 < m_cnt32 { rn(*sink.add((q0 + r0 as usize + 8) * heads + head)) } else { 0.0f32 };
        let mut l_r0 = 1.0f32;
        let mut l_r1 = 1.0f32;

        // 16 output MMA accumulators (`16 x 128` per warp) held strictly in registers (zero stack frame).
        let (mut o0, mut o1, mut o2, mut o3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o4, mut o5, mut o6, mut o7) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o8, mut o9, mut o10, mut o11) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
        let (mut o12, mut o13, mut o14, mut o15) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);

        let q_ld_row = (q_row_base + w_pos * 16 + (lane & 15)) as u32;
        let q_row_smem = q_base + q_ld_row * 256;
        let q_swz = ((q_ld_row & 7) ^ (lane >> 4) as u32) * 16;
        let k_ld_row0 = (((lane >> 4) * 8) + (lane & 7)) as u32;
        let k_row_smem = k_base + k_ld_row0 * 256;
        let k_swz = ((k_ld_row0 & 7) ^ ((lane >> 3) & 1) as u32) * 16;
        let v_ld_r0 = (lane & 15) as u32;
        let v_row_smem = k_base + K_BYTES as u32 + v_ld_r0 * 256;
        let v_swz = ((v_ld_r0 & 7) ^ (lane >> 4) as u32) * 16;
        let m_row0 = m_base + ((q_row_base + r0 as usize) * M_PITCH_U16 * 2) as u32;
        let t_rel_base = t_smem + 4 * (r0 + FW_BKV as u32 - 1 - t32 * 2);

        while t_base < arc_end32 {
            // Overlap V[t_base] async load with Q * K^T + relative bias + online softmax!
            issue_kv32!(k_base + K_BYTES as u32, values_h, t_base);
            crate::ptx::commit();

            // 1. S = Q * K^T (`16 x 32` per warp = 4 m16n8k16 tiles) in ascending d = 0, 16, ..., 112.
            // Keep base addresses and swizzle masks opaque to LICM so LLVM does not hoist 24 Q/K addresses across `t_base`.
            let (mut q_cur, mut k_cur, mut qs, mut ks_m) = (q_row_smem, k_row_smem, q_swz, k_swz);
            core::arch::asm!("/* {0} {1} {2} {3} */", inout(reg32) q_cur, inout(reg32) k_cur, inout(reg32) qs, inout(reg32) ks_m, options(nomem, nostack));
            let (mut s0, mut s1, mut s2, mut s3) = ([0.0f32; 4], [0.0f32; 4], [0.0f32; 4], [0.0f32; 4]);
            let mut ks = 0u32;
            while ks < 8 {
                let qa = ldmatrix_x4(q_cur + ((ks * 32) ^ qs));
                let k_off = (ks * 32) ^ ks_m;
                let kb0 = ldmatrix_x4(k_cur + k_off);
                s0 = mma_m16n8k16_bf16(qa, [kb0[0], kb0[1]], s0);
                s1 = mma_m16n8k16_bf16(qa, [kb0[2], kb0[3]], s1);
                let kb1 = ldmatrix_x4(k_cur + 4096 + k_off);
                s2 = mma_m16n8k16_bf16(qa, [kb1[0], kb1[1]], s2);
                s3 = mma_m16n8k16_bf16(qa, [kb1[2], kb1[3]], s3);
                ks += 1;
            }

            // 2. Fast Interior path (all 32 keys valid for all rows, zero relative bias, zero mask branches)
            // vs boundary/relative-bias path. Uniform across the CTA.
            let need_rel = t_base + FW_BKV as u32 + extent32 > pad32 + live0_32;
            let bq_pos_cur = if t_pitch32 == 64 { 32u32 } else { 64u32 };
            let interior = !need_rel && m_cnt32 == bq_pos_cur && t_base >= max_min_col && t_base >= pad32 && t_base + FW_BKV as u32 <= pad32 + live0_32;
            if interior {
                macro_rules! sc_fast {
                    ($s:ident) => {
                        $s[0] = rn(rn($s[0]) * scale); $s[1] = rn(rn($s[1]) * scale);
                        $s[2] = rn(rn($s[2]) * scale); $s[3] = rn(rn($s[3]) * scale);
                    };
                }
                sc_fast!(s0); sc_fast!(s1); sc_fast!(s2); sc_fast!(s3);
            } else {
                macro_rules! finalize_s {
                    ($s:ident, $col_off:expr, $b:expr) => {{
                        let c0 = t_base + $col_off + t32 * 2;
                        let c1 = c0 + 1;
                        let [b00, b01, b10, b11]: [f32; 4] = $b;
                        macro_rules! eval_sc {
                            ($val:expr, $row_local:expr, $qpos:expr, $live_r:expr, $col:expr, $bsum:expr) => {{
                                let val: f32 = $val;
                                let col: u32 = $col;
                                let qpos: u32 = $qpos;
                                if $row_local >= m_cnt32 || col >= arc_end32 || col > qpos {
                                    f32::NEG_INFINITY
                                } else {
                                    let dist = qpos - col;
                                    if dist >= $live_r {
                                        f32::NEG_INFINITY
                                    } else {
                                        let scaled = rn(rn(val) * scale);
                                        if need_rel && dist < extent32 { rn(scaled + rn($bsum)) } else { scaled }
                                    }
                                }
                            }};
                        }
                        $s[0] = eval_sc!($s[0], r0, qpos_r0, live_r0, c0, b00);
                        $s[1] = eval_sc!($s[1], r0, qpos_r0, live_r0, c1, b01);
                        $s[2] = eval_sc!($s[2], r0 + 8, qpos_r0 + 8, live_r1, c0, b10);
                        $s[3] = eval_sc!($s[3], r0 + 8, qpos_r0 + 8, live_r1, c1, b11);
                    }};
                }
                if need_rel {
                    // Two-pass relative bias (fragments 0,1 then 2,3): keeps live bias accumulators at 8 floats
                    // instead of 16 while reading M as packed u32 BF16 pairs in exact rising j = 0..16 order.
                    let m_row1 = m_row0 + (8 * M_PITCH_U16 * 2) as u32;
                    let t_row_stride = 4 * t_pitch32;
                    let (mut b00, mut b01, mut b02, mut b03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut b10, mut b11, mut b12, mut b13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let mut j = 0u32;
                    let mut at = t_rel_base;
                    while j < FW_REL_DIM as u32 {
                        let mp0 = load_u32(m_row0 + 2 * j);
                        let mp1 = load_u32(m_row1 + 2 * j);
                        let (a0, a1) = (f32::from_bits(mp0 << 16), f32::from_bits(mp0 & 0xffff_0000));
                        let (m0, m1) = (f32::from_bits(mp1 << 16), f32::from_bits(mp1 & 0xffff_0000));
                        let p0 = f32::from_bits(load_u32(at + 32)); let q0 = f32::from_bits(load_u32(at + 28));
                        let p1 = f32::from_bits(load_u32(at));      let q1 = f32::from_bits(load_u32(at - 4));
                        let p2 = f32::from_bits(load_u32(at - 32)); let q2 = f32::from_bits(load_u32(at - 36));
                        b00 += a0 * p1; b01 += a0 * q1; b02 += m0 * p0; b03 += m0 * q0;
                        b10 += a0 * p2; b11 += a0 * q2; b12 += m0 * p1; b13 += m0 * q1;
                        at += t_row_stride;
                        let p0 = f32::from_bits(load_u32(at + 32)); let q0 = f32::from_bits(load_u32(at + 28));
                        let p1 = f32::from_bits(load_u32(at));      let q1 = f32::from_bits(load_u32(at - 4));
                        let p2 = f32::from_bits(load_u32(at - 32)); let q2 = f32::from_bits(load_u32(at - 36));
                        b00 += a1 * p1; b01 += a1 * q1; b02 += m1 * p0; b03 += m1 * q0;
                        b10 += a1 * p2; b11 += a1 * q2; b12 += m1 * p1; b13 += m1 * q1;
                        at += t_row_stride;
                        j += 2;
                    }
                    finalize_s!(s0, 0, [b00, b01, b02, b03]);
                    finalize_s!(s1, 8, [b10, b11, b12, b13]);
                    let (mut b20, mut b21, mut b22, mut b23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    let (mut b30, mut b31, mut b32, mut b33) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                    j = 0;
                    at = t_rel_base;
                    while j < FW_REL_DIM as u32 {
                        let mp0 = load_u32(m_row0 + 2 * j);
                        let mp1 = load_u32(m_row1 + 2 * j);
                        let (a0, a1) = (f32::from_bits(mp0 << 16), f32::from_bits(mp0 & 0xffff_0000));
                        let (m0, m1) = (f32::from_bits(mp1 << 16), f32::from_bits(mp1 & 0xffff_0000));
                        let p2 = f32::from_bits(load_u32(at - 32)); let q2 = f32::from_bits(load_u32(at - 36));
                        let p3 = f32::from_bits(load_u32(at - 64)); let q3 = f32::from_bits(load_u32(at - 68));
                        let p4 = f32::from_bits(load_u32(at - 96)); let q4 = f32::from_bits(load_u32(at - 100));
                        b20 += a0 * p3; b21 += a0 * q3; b22 += m0 * p2; b23 += m0 * q2;
                        b30 += a0 * p4; b31 += a0 * q4; b32 += m0 * p3; b33 += m0 * q3;
                        at += t_row_stride;
                        let p2 = f32::from_bits(load_u32(at - 32)); let q2 = f32::from_bits(load_u32(at - 36));
                        let p3 = f32::from_bits(load_u32(at - 64)); let q3 = f32::from_bits(load_u32(at - 68));
                        let p4 = f32::from_bits(load_u32(at - 96)); let q4 = f32::from_bits(load_u32(at - 100));
                        b20 += a1 * p3; b21 += a1 * q3; b22 += m1 * p2; b23 += m1 * q2;
                        b30 += a1 * p4; b31 += a1 * q4; b32 += m1 * p3; b33 += m1 * q3;
                        at += t_row_stride;
                        j += 2;
                    }
                    finalize_s!(s2, 16, [b20, b21, b22, b23]);
                    finalize_s!(s3, 24, [b30, b31, b32, b33]);
                } else {
                    finalize_s!(s0, 0, [0.0; 4]);
                    finalize_s!(s1, 8, [0.0; 4]);
                    finalize_s!(s2, 16, [0.0; 4]);
                    finalize_s!(s3, 24, [0.0; 4]);
                }
            }

            // 3. Online softmax: shared max/alpha reduction, uniform fast/masked exp.
            let mut mt0 = fmax(fmax(fmax(s0[0], s0[1]), fmax(s1[0], s1[1])), fmax(fmax(s2[0], s2[1]), fmax(s3[0], s3[1])));
            mt0 = fmax(mt0, shfl_xor_f32(mt0, 1));
            mt0 = fmax(mt0, shfl_xor_f32(mt0, 2));
            let mut mt1 = fmax(fmax(fmax(s0[2], s0[3]), fmax(s1[2], s1[3])), fmax(fmax(s2[2], s2[3]), fmax(s3[2], s3[3])));
            mt1 = fmax(mt1, shfl_xor_f32(mt1, 1));
            mt1 = fmax(mt1, shfl_xor_f32(mt1, 2));
            let m_new0 = fmax(m_r0, mt0);
            let m_new1 = fmax(m_r1, mt1);
            let alpha0 = exp(m_r0 - m_new0);
            let alpha1 = exp(m_r1 - m_new1);
            m_r0 = m_new0;
            m_r1 = m_new1;

            macro_rules! to_p {
                ($v:expr, $mn:expr) => {{
                    let v: f32 = $v;
                    if !interior && v == f32::NEG_INFINITY { 0.0f32 } else { exp(v - $mn) }
                }};
            }
            let p00 = to_p!(s0[0], m_new0); let p01 = to_p!(s0[1], m_new0);
            let p02 = to_p!(s0[2], m_new1); let p03 = to_p!(s0[3], m_new1);
            let p10 = to_p!(s1[0], m_new0); let p11 = to_p!(s1[1], m_new0);
            let p12 = to_p!(s1[2], m_new1); let p13 = to_p!(s1[3], m_new1);
            let pa0 = [cvt_bf16x2(p00, p01), cvt_bf16x2(p02, p03), cvt_bf16x2(p10, p11), cvt_bf16x2(p12, p13)];
            let sum0_a = (p00 + p01) + (p10 + p11);
            let sum1_a = (p02 + p03) + (p12 + p13);
            let p20 = to_p!(s2[0], m_new0); let p21 = to_p!(s2[1], m_new0);
            let p22 = to_p!(s2[2], m_new1); let p23 = to_p!(s2[3], m_new1);
            let p30 = to_p!(s3[0], m_new0); let p31 = to_p!(s3[1], m_new0);
            let p32 = to_p!(s3[2], m_new1); let p33 = to_p!(s3[3], m_new1);
            let pa1 = [cvt_bf16x2(p20, p21), cvt_bf16x2(p22, p23), cvt_bf16x2(p30, p31), cvt_bf16x2(p32, p33)];
            let mut sum0 = sum0_a + ((p20 + p21) + (p30 + p31));
            sum0 += shfl_xor_f32(sum0, 1);
            sum0 += shfl_xor_f32(sum0, 2);
            let mut sum1 = sum1_a + ((p22 + p23) + (p32 + p33));
            sum1 += shfl_xor_f32(sum1, 1);
            sum1 += shfl_xor_f32(sum1, 2);

            l_r0 = l_r0 * alpha0 + sum0;
            l_r1 = l_r1 * alpha1 + sum1;

            // Rescale O accumulators by alpha only when the running max rose (bit-identical: finite x * 1.0 == x).
            if alpha0 != 1.0f32 {
                macro_rules! scale_o0 { ($($o:ident)*) => { $($o[0] *= alpha0; $o[1] *= alpha0;)* }; }
                scale_o0!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15);
            }
            if alpha1 != 1.0f32 {
                macro_rules! scale_o1 { ($($o:ident)*) => { $($o[2] *= alpha1; $o[3] *= alpha1;)* }; }
                scale_o1!(o0 o1 o2 o3 o4 o5 o6 o7 o8 o9 o10 o11 o12 o13 o14 o15);
            }

            // Wait for V[t_base] to arrive, then issue K[t_base + 32] overlapped with O += P * V!
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            let next_t = t_base + FW_BKV as u32;
            if next_t < arc_end32 {
                issue_kv32!(k_base, keys_h, next_t);
                issue_rel32!(next_t);
                crate::ptx::commit();
            }

            // 4. O += P * V (`16 x 128 x 32`), K-substep 0 (`0..16`) then K-substep 1 (`16..32`), unrolled.
            // Keep V base and swizzle mask opaque to LICM so LLVM does not hoist 16 V addresses across `t_base`.
            let (mut v_cur, mut vs) = (v_row_smem, v_swz);
            core::arch::asm!("/* {0} {1} */", inout(reg32) v_cur, inout(reg32) vs, options(nomem, nostack));
            macro_rules! pv_pair {
                ($pair:expr, $oa:ident, $ob:ident) => {
                    let v_off = ($pair * 32) ^ vs;
                    let vb0 = ldmatrix_x4_trans(v_cur + v_off);
                    $oa = mma_m16n8k16_bf16(pa0, [vb0[0], vb0[1]], $oa);
                    $ob = mma_m16n8k16_bf16(pa0, [vb0[2], vb0[3]], $ob);
                    let vb1 = ldmatrix_x4_trans(v_cur + 4096 + v_off);
                    $oa = mma_m16n8k16_bf16(pa1, [vb1[0], vb1[1]], $oa);
                    $ob = mma_m16n8k16_bf16(pa1, [vb1[2], vb1[3]], $ob);
                };
            }
            pv_pair!(0, o0, o1);
            pv_pair!(1, o2, o3);
            pv_pair!(2, o4, o5);
            pv_pair!(3, o6, o7);
            pv_pair!(4, o8, o9);
            pv_pair!(5, o10, o11);
            pv_pair!(6, o12, o13);
            pv_pair!(7, o14, o15);

            if next_t < arc_end32 {
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
            }
            t_base = next_t;
        }

        // Epilogue: reload epilogue-only pointers/counts via read_volatile so they hold zero registers in the loop.
        let gate_e = w.add(6).read_volatile() as *const f32;
        let output_e = w.add(8).read_volatile() as *mut f32;
        let heads_e = w.add(10).read_volatile() as usize;
        let half_out = (w.add(FW_CLIP_WORD).read_volatile() & FW_HALF_OUT) != 0;
        let output_u32 = output_e.cast::<u32>();
        let r0u = r0 as usize;
        let r1u = r0u + 8;
        let tu = t32 as usize;
        let m_cntu = m_cnt32 as usize;
        let q0u = q0_32 as usize;
        let headu = head32 as usize;
        let inv0 = 1.0f32 / l_r0;
        let inv1 = 1.0f32 / l_r1;
        macro_rules! store_o {
            ($ni:expr, $o:ident) => {{
                let d0 = $ni * 8 + tu * 2;
                let d1 = d0 + 1;
                if r0u < m_cntu {
                    let off = ((q0u + r0u) * heads_e + headu) * FW_HEAD_DIM;
                    let v0 = rn(rn($o[0] * inv0) * sigmoid(rn(*gate_e.add(off + d0))));
                    let v1 = rn(rn($o[1] * inv0) * sigmoid(rn(*gate_e.add(off + d1))));
                    if half_out {
                        *output_u32.add((off + d0) / 2) = cvt_bf16x2(v0, v1);
                    } else {
                        *output_e.add(off + d0) = v0;
                        *output_e.add(off + d1) = v1;
                    }
                }
                if r1u < m_cntu {
                    let off = ((q0u + r1u) * heads_e + headu) * FW_HEAD_DIM;
                    let v0 = rn(rn($o[2] * inv1) * sigmoid(rn(*gate_e.add(off + d0))));
                    let v1 = rn(rn($o[3] * inv1) * sigmoid(rn(*gate_e.add(off + d1))));
                    if half_out {
                        *output_u32.add((off + d0) / 2) = cvt_bf16x2(v0, v1);
                    } else {
                        *output_e.add(off + d0) = v0;
                        *output_e.add(off + d1) = v1;
                    }
                }
            }};
        }
        store_o!(0, o0); store_o!(1, o1); store_o!(2, o2); store_o!(3, o3);
        store_o!(4, o4); store_o!(5, o5); store_o!(6, o6); store_o!(7, o7);
        store_o!(8, o8); store_o!(9, o9); store_o!(10, o10); store_o!(11, o11);
        store_o!(12, o12); store_o!(13, o13); store_o!(14, o14); store_o!(15, o15);
    }
}
