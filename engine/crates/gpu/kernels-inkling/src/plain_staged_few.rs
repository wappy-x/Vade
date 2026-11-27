//! The small-batch BF16 product (`few_product`): the entry `resident_launch_plain_staged_few` of the Inkling image
//! (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED_FEW`. Its three unsafe sites stay counted:
//! `unsafe fn replay`, its block, `unsafe fn warp`.
//!
//! SM121a BF16 tensor-core decode kernel (`M = 1..=16`): computes `Y = X * W^T`
//! via `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `X` rounded once via `cvt.rn.bf16x2.f32`.
//! Both forms accumulate in ascending K order:
//! - Narrow (same shape rule as `PLAIN_STAGED_NARROW`): one warp computes four weight rows, with activations as A.
//! - Wide (64 weight rows/block, 16 rows/warp): Weights-as-A `m16n8k16` (`A = W [16, 16]`,
//!   `B = X [8, 16]` or `2 x [8, 16]`), a `FEW_STAGES`-stage `cp.async` ring of XOR-swizzled SMEM (`30,720` B =
//!   3 blocks/SM) sharing BF16 `X` across all 4 warps (one `ld.global.nc.v4` per thread for `M <= 8`, overlapped
//!   with MMAs, L2 prefetch on W `WIDE_PREFETCH` stages ahead, one `bar.sync 0` per 64 cols).
//!   Matches `PLAIN_STAGED` and `PLAIN_TC_ROWS` bit-for-bit.
//! The stage counts and prefetch distances change only when data arrives, never the operands or the order of an MMA.
use core::arch::asm;
use crate::bf16_panels::{self, DATA_WORDS, FEW_PANEL_WORD, RAW};
use crate::few_product::{
    FEW_BLEND_WORD, FEW_BLOCK_BYTES, FEW_GATE_BLEND, FEW_GROUP_PAIR, FEW_GROUP_WORD, FEW_INPUT_AT, FEW_MAX,
    FEW_NARROW_THREADS, FEW_PANEL_RING, FEW_PANEL_RING_DEPTH, FEW_ROUND_OUT, FEW_SPLIT_MAX, FEW_SPLIT_OUT, FEW_STAGES, FEW_STAGE_BYTES, FEW_THREADS,
};
use crate::fixed_math::rn;
use crate::stream_pairs::gate_tanh;
use crate::tc_ptx::{cvt_bf16x2, ld_global_nc_v2_u32, ld_global_nc_v4, ldmatrix_x4, load_u32, mma_m16n8k16_bf16, st_shared_zero16, store_u32};

/// Columns of a stage (4 x k16 substeps).
const BK: usize = crate::staged_product::STAGE;
/// Narrow stage layout: one stage holds four 144-byte weight rows.
const NARROW_ROW_BYTES: usize = BK * 2 + 16;
const NARROW_STAGE_BYTES: usize = 4 * NARROW_ROW_BYTES;
const NS_NARROW: usize = 16;
/// Narrow stages a loop pass covers (the 256 columns of `staged_narrow::STAGE_COLS`), and passes in the ring.
const NARROW_UNROLL: usize = 4;
const NARROW_GROUPS: usize = NS_NARROW / NARROW_UNROLL;
/// Stages ahead of the current wide stage that the L2 prefetch of the weights reaches.
const WIDE_PREFETCH: usize = FEW_STAGES + 2;
const _: () = assert!(BK == 64 && FEW_MAX == 16 && FEW_NARROW_THREADS == 32 && FEW_THREADS == 128);
const _: () = assert!(FEW_INPUT_AT == 8_192 && FEW_STAGE_BYTES == 10_240 && FEW_STAGES == 3 && FEW_BLOCK_BYTES == 30_720);
const _: () = assert!(NARROW_STAGE_BYTES == 576 && NS_NARROW * NARROW_STAGE_BYTES <= FEW_BLOCK_BYTES);
const _: () = assert!(NARROW_UNROLL == 4 && NARROW_UNROLL * BK == crate::staged_narrow::STAGE_COLS && NS_NARROW % NARROW_UNROLL == 0 && NARROW_GROUPS >= 2);

/// Zero-bank-conflict XOR-swizzled byte offset within a 128-byte-stride BF16 tile (`row` in 0..16, `chunk` in 0..8).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 128 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// The replay of one recorded command of this kind: `table[index]` holds seven words, the addresses of the weights
/// (rows x cols BF16 patterns), of the input (inputs x cols FP32 values) and of the output, then rows, cols, inputs
/// and the output stride. A launch that the rule `few_product::few_admitted` does not take stops the block before
/// any read.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged_few`): `table` and `index` are the arguments
/// the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::project_staged_few_into, after
    // few_product::few_check passed for its seven words. That check holds: the weights to exactly rows x cols BF16
    // values, the input to exactly inputs x cols words, the output to exactly (inputs - 1) x stride + rows words that
    // meet no operand, stride at least rows, 1 to 16 inputs, the weights and the input to 16-byte addresses, cols to
    // whole stages of 64 inside the envelope. All three live through completion. The rule few_product::few_admitted is
    // asked here again before the first read (shape, inputs, matching geometry, alignment, stride >= rows).
    // `block_first < rows` is uniform across the block. All four wide warps participate in every `bar.sync 0`.
    // For paired inputs the host checks two live arrays of inputs * (cols / 2) words, with both halves
    // ending on a 64-column stage. For split output it checks disjoint writable segments covering all rows.
    // Each output element `(t, r)` with `t < inputs` and `r < rows` has one writer at `t * stride + r`.
    // A non-null panel pointer covers all requested tiles including a partial final row band.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED_FEW { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (weights0, input0, output0) = (*words as *const u16, *words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, cols, inputs, stride) = (*words.add(3) as usize, *words.add(4) as usize, *words.add(5) as usize, *words.add(6) as usize);
        let input2_0 = *words.add(7) as *const f32;
        let flags = *words.add(8);
        let round_out = (flags & FEW_ROUND_OUT) != 0;
        let grouped = (flags & FEW_GROUP_PAIR) != 0;
        let split_count = if (flags & FEW_SPLIT_OUT) != 0 { *words.add(9) as usize } else { 0 };
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        let group = core::arch::nvptx::_block_idx_y() as usize;
        // SAFETY: dynamic shared memory is checked before the first access on every path.
        let smem = 0u32;
        let shared_bytes: u32;
        asm!("mov.u32 {bytes}, %dynamic_smem_size;", bytes = out(reg32) shared_bytes);
        if (flags & FEW_PANEL_RING) != 0 {
            let (panels_at, meta_at) = (*words.add(FEW_PANEL_WORD) as *const u32, *words.add(FEW_PANEL_WORD + 2) as *const u32);
            let part_tiles = *words.add(7) as usize;
            let tiles = cols / 16;
            const DEPTH: usize = FEW_PANEL_RING_DEPTH;
            const _: () = assert!(DEPTH == 8);
            let warps = threads / 32;
            if group != 0 || (threads != 32 && threads != 64 && threads != 128) || stride < rows
                || !crate::few_product::few_panel_ring_admitted(rows, cols, inputs)
                || (weights0 as usize | input0 as usize | panels_at as usize | meta_at as usize) & 15 != 0
                || panels_at.is_null() || meta_at.is_null()
                || (flags & !(FEW_PANEL_RING | FEW_SPLIT_OUT)) != 0
                || (split_count == 0 && ((flags & FEW_SPLIT_OUT) != 0 || output0.is_null())) || split_count > FEW_SPLIT_MAX
                || part_tiles == 0 || tiles % part_tiles != 0 || (part_tiles != tiles && part_tiles % DEPTH != 0)
                || crate::few_product::one_panel_shared_warps(inputs, cols, part_tiles, warps) > shared_bytes as usize
                || core::arch::nvptx::_grid_dim_x() as usize != rows.div_ceil(16 * warps)
            {
                core::arch::nvptx::trap();
            }
            let bgrp = core::arch::nvptx::_block_idx_x() as usize;
            let bands = rows.div_ceil(16);
            let first_band = bgrp * warps;
            let group_bands = (bands - first_band).min(warps);
            let row_bytes = crate::few_product::one_panel_row_bytes(inputs, part_tiles);
            let headers = smem + (inputs * row_bytes) as u32;
            for i in (tid..group_bands * tiles / 4).step_by(threads) {
                crate::ptx::copy16(headers + (i * 16) as u32, meta_at.add(first_band * tiles + i * 4));
            }
            crate::ptx::commit();
            crate::ptx::wait();
            core::arch::nvptx::_syncthreads();
            let band = first_band + tid / 32;
            let own = band.min(bands - 1);
            let part = own - first_band;
            let lane = tid % 32;
            let (g, s) = (lane / 4, lane % 4);
            let band_tiles = panels_at.add(own * tiles * DATA_WORDS);
            let row_top = (own * 16 + g).min(rows - 1);
            let row_bottom = (own * 16 + g + 8).min(rows - 1);
            let raw_top = weights0.add(row_top * cols + 2 * s).cast::<u32>();
            let raw_bottom = weights0.add(row_bottom * cols + 2 * s).cast::<u32>();
            let header = headers + (part * tiles * 4) as u32;
            let mut next_base = load_u32(header);
            macro_rules! fetch {
                ($k:expr) => {{
                    let k: usize = $k;
                    let raw = next_base == RAW;
                    let tile = band_tiles.add(k * DATA_WORDS + lane);
                    let a0 = if raw { raw_top.add(k * 8) } else { tile };
                    let a1 = if raw { raw_bottom.add(k * 8) } else { tile.add(32) };
                    let a2 = if raw { raw_top.add(k * 8 + 4) } else { tile.add(64) };
                    let (w0, w1, w2, w3): (u32, u32, u32, u32);
                    asm!("ld.global.nc.u32 {w0}, [{a0}]; ld.global.nc.u32 {w1}, [{a1}]; ld.global.nc.u32 {w2}, [{a2}];",
                         "{{ .reg .pred q; setp.eq.u32 q, {base}, 0xffffffff; mov.u32 {w3}, 0; @q ld.global.nc.u32 {w3}, [{a3}]; }}",
                         a0 = in(reg64) a0, a1 = in(reg64) a1, a2 = in(reg64) a2, a3 = in(reg64) raw_bottom.add(k * 8 + 4),
                         base = in(reg32) next_base, w0 = out(reg32) w0, w1 = out(reg32) w1, w2 = out(reg32) w2, w3 = out(reg32) w3,
                         options(readonly, nostack));
                    let ahead = if k + 1 < tiles { k + 1 } else { k };
                    next_base = load_u32(header + (ahead * 4) as u32);
                    [w0, w1, w2, w3]
                }};
            }
            macro_rules! bytes {
                ($lo:expr, $hi:expr, $mask:literal) => {{
                    let word: u32;
                    asm!("prmt.b32 {word}, {lo}, {hi}, {mask};", word = out(reg32) word,
                         lo = in(reg32) $lo, hi = in(reg32) $hi, mask = const $mask, options(pure, nomem, nostack));
                    word
                }};
            }
            let mut c0 = [0.0f32; 4];
            let mut c1 = [0.0f32; 4];
            let x_lane = smem + (g * row_bytes + s * 8) as u32;
            let x_slab1_delta = (((8 + g).min(inputs - 1) - g) * row_bytes) as u32;
            macro_rules! consume {
                ($slot:expr, $k:expr, $x:expr) => {{
                    let k: usize = $k;
                    let [w0, w1, w2, w3]: [u32; 4] = $slot;
                    let base = load_u32(header + (k * 4) as u32);
                    let high0 = bf16_panels::high_bytes(w2, base);
                    let high1 = bf16_panels::high_bytes(w2 >> 4, base);
                    let decoded = [bytes!(w0, high0, 0x5140), bytes!(w0, high0, 0x7362),
                                   bytes!(w1, high1, 0x5140), bytes!(w1, high1, 0x7362)];
                    let raw = base == RAW;
                    let a = [if raw { w0 } else { decoded[0] }, if raw { w1 } else { decoded[1] },
                             if raw { w2 } else { decoded[2] }, if raw { w3 } else { decoded[3] }];
                    let x0 = $x;
                    c0 = mma_m16n8k16_bf16(a, crate::ptx::load2(x0), c0);
                    c1 = mma_m16n8k16_bf16(a, crate::ptx::load2(x0 + x_slab1_delta), c1);
                }};
            }
            macro_rules! load4 {
                ($p:expr) => {{
                    let (a, b, c, d): (u32, u32, u32, u32);
                    asm!("ld.global.v4.u32 {{{a}, {b}, {c}, {d}}}, [{p}];", p = in(reg64) $p,
                         a = out(reg32) a, b = out(reg32) b, c = out(reg32) c, d = out(reg32) d, options(readonly, nostack));
                    [a, b, c, d]
                }};
            }
            macro_rules! put4 {
                ($row:expr, $i:expr, $v:expr) => {{
                    let (i, v): (u32, [u32; 4]) = ($i, $v);
                    let at = $row + (i / 4 * 32 + (i & 1) * 16 + (i & 2) * 2);
                    store_u32(at, cvt_bf16x2(f32::from_bits(v[0]), f32::from_bits(v[1])));
                    store_u32(at + 8, cvt_bf16x2(f32::from_bits(v[2]), f32::from_bits(v[3])));
                }};
            }
            macro_rules! word {
                ($i:expr) => { core::ptr::read_volatile(words.add($i)) };
            }
            macro_rules! stage_input {
                ($first:expr) => {{
                    let first = $first as u32;
                    let (input, cols, inputs) = (word!(1) as *const f32, word!(4) as u32, word!(5) as u32);
                    let part_tiles = word!(7) as u32;
                    let row_bytes = crate::few_product::one_panel_row_bytes(inputs as usize, part_tiles as usize) as u32;
                    let items = part_tiles * 4;
                    let th = core::arch::nvptx::_block_dim_x() as u32;
                    let mut t = 0u32;
                    while t < inputs {
                        let source = input.add((t * cols + first * 16) as usize);
                        let row = smem + t * row_bytes;
                        let mut i = core::arch::nvptx::_thread_idx_x() as u32;
                        while i < items {
                            let next = i + th;
                            let more = next < items;
                            let v = load4!(source.add((4 * i) as usize));
                            let w = load4!(source.add((4 * if more { next } else { i }) as usize));
                            put4!(row, i, v);
                            if more { put4!(row, next, w); }
                            i += 2 * th;
                        }
                        t += 1;
                    }
                }};
            }
            let last = tiles - 1;
            let mut ring = [[0u32; 4]; DEPTH];
            macro_rules! prime {
                ($($j:literal)*) => {$( ring[$j] = fetch!(if $j < last { $j } else { last }); )*};
            }
            prime!(0 1 2 3 4 5 6 7);
            asm!("griddepcontrol.wait;", options(nostack));
            stage_input!(0);
            core::arch::nvptx::_syncthreads();
            if band >= bands && part_tiles == tiles { return; }
            let mut x_turn = x_lane;
            let mut part_end = part_tiles;
            let mut k = 0;
            while k + DEPTH <= tiles {
                if k == part_end {
                    core::arch::nvptx::_syncthreads();
                    stage_input!(k);
                    core::arch::nvptx::_syncthreads();
                    let part_tiles = word!(7) as usize;
                    part_end = k + part_tiles;
                    x_turn -= (part_tiles * 32) as u32;
                }
                macro_rules! stage {
                    ($($j:literal)*) => {$(
                        consume!(ring[$j], k + $j, x_turn + ($j * 32) as u32);
                        ring[$j] = fetch!(if k + $j + DEPTH < last { k + $j + DEPTH } else { last });
                    )*};
                }
                stage!(0 1 2 3 4 5 6 7);
                k += DEPTH;
                x_turn += (DEPTH * 32) as u32;
            }
            let (output, rows, inputs, stride) = (word!(2) as *mut f32, word!(3) as usize, word!(5) as usize, word!(6) as usize);
            let split_count = if word!(8) & FEW_SPLIT_OUT != 0 { word!(9) as usize } else { 0 };
            macro_rules! tail {
                ($($j:literal)*) => {$( if k + $j < tiles { consume!(ring[$j], k + $j, x_turn + ($j * 32) as u32); } )*};
            }
            tail!(0 1 2 3 4 5 6);
            let (row, t0, t2) = (band * 16 + g, 2 * s, 8 + 2 * s);
            macro_rules! put {
                ($t:expr, $r:expr, $v:expr) => {{
                    let (t, r, v): (usize, usize, f32) = ($t, $r, $v);
                    if t < inputs && r < rows {
                        if split_count == 0 {
                            *output.add(t * stride + r) = v;
                        } else {
                            let (mut first, mut si) = (0usize, 0usize);
                            while si < split_count {
                                let width = *words.add(10 + 2 * si + 1) as usize;
                                if r < first + width {
                                    *(*words.add(10 + 2 * si) as *mut f32).add(t * width + (r - first)) = v;
                                    break;
                                }
                                first += width;
                                si += 1;
                            }
                        }
                    }
                }};
            }
            put!(t0, row, c0[0]);
            put!(t0 + 1, row, c0[1]);
            put!(t0, row + 8, c0[2]);
            put!(t0 + 1, row + 8, c0[3]);
            put!(t2, row, c1[0]);
            put!(t2 + 1, row, c1[1]);
            put!(t2, row + 8, c1[2]);
            put!(t2 + 1, row + 8, c1[3]);
            return;
        }
        if FEW_BLOCK_BYTES > shared_bytes as usize { core::arch::nvptx::trap(); }
        let narrow = crate::staged_narrow::shape_admitted(stride, cols);
        let (weights, input, input2, output) = match group {
            0 => (weights0, input0, input2_0, output0),
            1 | 2 if grouped && narrow && (flags & !(FEW_ROUND_OUT | FEW_GROUP_PAIR)) == 0 => {
                let base = FEW_GROUP_WORD + (group - 1) * 3;
                (*words.add(base) as *const u16, *words.add(base + 1) as *const f32, *words.add(base + 2) as *const f32, output0.add(group * inputs * stride))
            }
            _ => core::arch::nvptx::trap(),
        };
        if (grouped && (!narrow || (flags & !(FEW_ROUND_OUT | FEW_GROUP_PAIR)) != 0))
            || !crate::few_product::few_admitted(rows, cols, inputs, stride, weights as usize, input as usize, threads)
            || (!input2.is_null() && ((input2 as usize) & 15 != 0 || cols % (2 * BK) != 0))
            || split_count > FEW_SPLIT_MAX
        {
            core::arch::nvptx::trap();
        }
        let panels = if grouped { core::ptr::null() } else { *words.add(FEW_PANEL_WORD) as *const u32 };
        let metadata = if grouped { core::ptr::null() } else { *words.add(FEW_PANEL_WORD + 2) as *const u32 };
        let panel_row = if grouped { 0 } else { *words.add(FEW_PANEL_WORD + 1) as usize };
        if flags & FEW_GATE_BLEND != 0 {
            // The attached gate blend (`few_product::FEW_GATE_BLEND`, 2026-11-25, W-BOSS-SC-DENSE-VERIFY), the 2-to-16-row
            // form of the one-row entry's: block b takes the 32 channels from 32 b. Warps 0 and 1 compute forget rows
            // 32 b .. 32 b + 32 and warps 2 and 3 the gain rows dim + 32 b .. dim + 32 b + 32, with the chains of the
            // product's own blocks (`warp` on the matrix from row `shift`, the same input; with panels, the panel row
            // moves by the same whole bands, since dim and so shift are multiples of 32). Every product word is still
            // written, at its own address. The forget and gain words of the block also go to shared memory (input row t,
            // local row j at word 64 t + j); after the barrier the 32 blended words of each input row are written with
            // the GateBlend arithmetic of `fast_elem::replay_stream_ops`, word for word. The host joined the commands
            // only after `Backend::attach_gate_blend_rows` checked the shapes, that the blend reads this product's
            // output, and that the kept, message and blended rows (inputs x dim words each) meet no operand they must not.
            let dim = rows / 2;
            if !crate::few_product::few_gate_blend_admitted(rows, cols, inputs) || narrow || stride != rows
                || !input2.is_null() || flags != FEW_GATE_BLEND || threads != FEW_THREADS
                || core::arch::nvptx::_grid_dim_x() as usize != dim / 32 { core::arch::nvptx::trap(); }
            let kept = *words.add(FEW_BLEND_WORD) as *const f32;
            let message = *words.add(FEW_BLEND_WORD + 1) as *const f32;
            let blended = *words.add(FEW_BLEND_WORD + 2) as *mut f32;
            let first = core::arch::nvptx::_block_idx_x() as usize * 32;
            let shift = if tid >= 64 { dim - 32 } else { 0 };
            let (c0, c1) = if inputs <= 8 {
                warp::<false, false>(weights.add(shift * cols), input, input2, cols, inputs, first, rows - shift, smem, panels, metadata, panel_row + shift)
            } else {
                warp::<false, true>(weights.add(shift * cols), input, input2, cols, inputs, first, rows - shift, smem, panels, metadata, panel_row + shift)
            };
            crate::ptx::wait();
            core::arch::nvptx::_syncthreads();
            let lane = tid & 31;
            let (g, s) = (lane >> 2, lane & 3);
            // Local row j (0..32 forget, 32..64 gain) of the block; its matrix row is shift + first + j.
            let j = tid / 32 * 16 + g;
            let (t0, t1) = (2 * s, 2 * s + 1);
            macro_rules! keep {
                ($j:expr, $t:expr, $v:expr) => {{
                    let (j, t, v): (usize, usize, f32) = ($j, $t, $v);
                    if t < inputs {
                        *output.add(t * stride + shift + first + j) = v;
                        store_u32(smem + ((t * 64 + j) * 4) as u32, v.to_bits());
                    }
                }};
            }
            keep!(j, t0, c0[0]);
            keep!(j, t1, c0[1]);
            keep!(j + 8, t0, c0[2]);
            keep!(j + 8, t1, c0[3]);
            if inputs > 8 {
                let (t2, t3) = (8 + 2 * s, 8 + 2 * s + 1);
                keep!(j, t2, c1[0]);
                keep!(j, t3, c1[1]);
                keep!(j + 8, t2, c1[2]);
                keep!(j + 8, t3, c1[3]);
            }
            core::arch::nvptx::_syncthreads();
            let mut i = tid;
            while i < inputs * 32 {
                let (t, d) = (i / 32, i % 32);
                let forget = f32::from_bits(load_u32(smem + ((t * 64 + d) * 4) as u32));
                let gain = f32::from_bits(load_u32(smem + ((t * 64 + 32 + d) * 4) as u32));
                let at = t * dim + first + d;
                *blended.add(at) = (1.0f32 - gate_tanh(forget)) * *kept.add(at) + gate_tanh(gain) * *message.add(at);
                i += FEW_THREADS;
            }
            return;
        }
        let block_first = core::arch::nvptx::_block_idx_x() as usize * if narrow { 4 } else { 64 };
        if block_first < rows {
            let warp_id = tid / 32;
            let lane = tid & 31;
            let (g, s) = (lane >> 2, lane & 3);
            let write_out = |t: usize, r: usize, v: f32| {
                let v = if round_out { rn(v) } else { v };
                if split_count == 0 {
                    *output.add(t * stride + r) = v;
                } else {
                    let mut off = 0usize;
                    let mut si = 0usize;
                    while si < split_count {
                        let sw = *words.add(10 + 2 * si + 1) as usize;
                        if r < off + sw {
                            let sp = *words.add(10 + 2 * si) as *mut f32;
                            *sp.add(t * sw + (r - off)) = v;
                            break;
                        }
                        off += sw;
                        si += 1;
                    }
                }
            };
            if narrow {
                let (c0, _) = if inputs <= 8 {
                    warp::<true, false>(weights, input, input2, cols, inputs, block_first, rows, smem, panels, metadata, panel_row)
                } else {
                    warp::<true, true>(weights, input, input2, cols, inputs, block_first, rows, smem, panels, metadata, panel_row)
                };
                if s < 2 {
                    let (r0, r1) = (block_first + 2 * s, block_first + 2 * s + 1);
                    if g < inputs {
                        if r0 < rows { write_out(g, r0, c0[0]); }
                        if r1 < rows { write_out(g, r1, c0[1]); }
                    }
                    if inputs > 8 && g + 8 < inputs {
                        if r0 < rows { write_out(g + 8, r0, c0[2]); }
                        if r1 < rows { write_out(g + 8, r1, c0[3]); }
                    }
                }
            } else {
                let (c0, c1) = if inputs <= 8 {
                    warp::<false, false>(weights, input, input2, cols, inputs, block_first, rows, smem, panels, metadata, panel_row)
                } else {
                    warp::<false, true>(weights, input, input2, cols, inputs, block_first, rows, smem, panels, metadata, panel_row)
                };
                let first = block_first + warp_id * 16;
                let (r0, r1) = (first + g, first + g + 8);
                let (t0, t1) = (2 * s, 2 * s + 1);
                if r0 < rows {
                    if t0 < inputs { write_out(t0, r0, c0[0]); }
                    if t1 < inputs { write_out(t1, r0, c0[1]); }
                }
                if r1 < rows {
                    if t0 < inputs { write_out(t0, r1, c0[2]); }
                    if t1 < inputs { write_out(t1, r1, c0[3]); }
                }
                if inputs > 8 {
                    let (t2, t3) = (8 + 2 * s, 8 + 2 * s + 1);
                    if r0 < rows {
                        if t2 < inputs { write_out(t2, r0, c1[0]); }
                        if t3 < inputs { write_out(t3, r0, c1[1]); }
                    }
                    if r1 < rows {
                        if t2 < inputs { write_out(t2, r1, c1[2]); }
                        if t3 < inputs { write_out(t3, r1, c1[3]); }
                    }
                }
            }
        }
    }
}

/// The `m16n8k16` BF16 CTA body:
/// - `NARROW == true`: one warp owns four weight rows and accumulates in ascending K order.
/// - `NARROW == false`: 4 warps own 64 weight rows sharing a `FEW_STAGES`-stage ring of XOR-swizzled BF16 `X` in SMEM.
#[inline(always)]
unsafe fn warp<const NARROW: bool, const TWO: bool>(
    weights: *const u16, input: *const f32, input2: *const f32, cols: usize, inputs: usize, block_first: usize, rows: usize, smem: u32, panels: *const u32, metadata: *const u32, panel_row: usize,
) -> ([f32; 4], [f32; 4]) {
    // SAFETY: Private to this file; called only from `replay` with all threads of the block (`block_first < rows`
    // uniform across the CTA), `few_admitted` true (`cols` whole stages of 64 up to `FEW_COLS_MAX = 16,384`,
    // `rows * cols` inside the envelope, `1 <= inputs <= 16`, 16-byte aligned weights and input), and `smem` the
    // block's static `FEW_BLOCK_BYTES = 30,720` shared buffer. Every thread reaches every `bar.sync 0` and every lane
    // reaches every warp collective. All global reads stay strictly in-bounds and aligned (16B weight chunks for
    // `gr < rows`; 8B/16B input chunks guarded by `< inputs`). Out-of-bounds weight and input rows zero-fill their
    // shared slots. No global memory is written.
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid & 31;
    let (g, s) = (lane >> 2, lane & 3);
    let chunks = cols / BK;

    let half = if input2.is_null() { cols } else { cols / 2 };

    let compressed = !panels.is_null();
    macro_rules! interleave {($low:expr, $high:expr, $selector:expr) => {{
        let word: u32;
        asm!("prmt.b32 {word}, {low}, {high}, {select};",
            word = out(reg32) word, low = in(reg32) $low, high = in(reg32) $high, select = in(reg32) $selector,
            options(pure, nomem, nostack));
        word
    }};}
    macro_rules! decode_tile {($base:expr, $low0:expr, $low1:expr, $codes:expr) => {{
        let (base, low0, low1, codes) = ($base, $low0, $low1, $codes);
        let high0 = bf16_panels::high_bytes(codes, base);
        let high1 = bf16_panels::high_bytes(codes >> 4, base);
        [interleave!(low0, high0, 0x5140u32), interleave!(low0, high0, 0x7362u32),
         interleave!(low1, high1, 0x5140u32), interleave!(low1, high1, 0x7362u32)]
    }};}

    // Compressed wide BF16 panels (`M = 1..=16`): direct `P_DEPTH`-tile register load ring
    // with `.L2::evict_first` + `.L2::256B` and part-staged BF16 `X` in the existing 30,720 B SMEM.
    // Reuses each decoded 16x16 BF16 weight tile across both 8-row MMA N-tiles (`TWO`) in one DRAM pass
    // with ascending-K `m16n8k16` accumulation (bit-identical to `plain_staged_one` and raw `plain_staged_few`).
    if compressed && !NARROW {
        const P_DEPTH: usize = 12;
        let tiles = cols / 16;
        let meta_bytes = tiles * 16; // 4 warps * tiles * 4 bytes = cols bytes
        let x_budget = FEW_BLOCK_BYTES - meta_bytes;
        let max_part = ((x_budget / inputs).saturating_sub(32)) / 32;
        let part_tiles = if inputs * (tiles * 32 + 32) <= x_budget {
            tiles
        } else {
            let p = (max_part / P_DEPTH) * P_DEPTH;
            if p >= P_DEPTH { p } else { P_DEPTH }
        };
        let row_bytes = part_tiles * 32 + 32;
        let headers = smem + (inputs * row_bytes) as u32;
        let first = block_first + warp_id * 16;
        let max_band = (rows + panel_row).div_ceil(16) - 1;
        let own_band = ((first + panel_row) / 16).min(max_band);
        // Each warp copies its own 16-row band's metadata words into SMEM once; warp-local barrier suffices.
        let warp_header = headers + (warp_id * tiles * 4) as u32;
        let mut mi = lane;
        while mi < tiles / 4 {
            crate::ptx::copy16(warp_header + (mi * 16) as u32, metadata.add(own_band * tiles + mi * 4).cast());
            mi += 32;
        }
        crate::ptx::commit();
        crate::ptx::wait();
        asm!("bar.warp.sync 0xffffffff;");

        let policy = crate::tc_ptx::l2_evict_first();
        let band_tiles = panels.add(own_band * tiles * DATA_WORDS);
        let row_top = (first + g).min(rows - 1);
        let row_bottom = (first + g + 8).min(rows - 1);
        let raw_top = weights.add(row_top * cols + 2 * s).cast::<u32>();
        let raw_bottom = weights.add(row_bottom * cols + 2 * s).cast::<u32>();
        let mut next_base = load_u32(warp_header);

        macro_rules! fetch_p {($k:expr) => {{
            let k: usize = $k;
            let raw = next_base == RAW;
            let tile_p = band_tiles.add(k * DATA_WORDS + lane);
            let a0 = if raw { raw_top.add(k * 8) } else { tile_p };
            let a1 = if raw { raw_bottom.add(k * 8) } else { tile_p.add(32) };
            let a2 = if raw { raw_top.add(k * 8 + 4) } else { tile_p.add(64) };
            let (w0, w1, w2, w3): (u32, u32, u32, u32);
            asm!(
                "ld.global.nc.L2::cache_hint.L2::256B.u32 {w0}, [{a0}], {pol};",
                "ld.global.nc.L2::cache_hint.L2::256B.u32 {w1}, [{a1}], {pol};",
                "ld.global.nc.L2::cache_hint.L2::256B.u32 {w2}, [{a2}], {pol};",
                "{{ .reg .pred q; setp.eq.u32 q, {base}, 0xffffffff; mov.u32 {w3}, 0; @q ld.global.nc.L2::cache_hint.u32 {w3}, [{a3}], {pol}; }}",
                a0 = in(reg64) a0, a1 = in(reg64) a1, a2 = in(reg64) a2, a3 = in(reg64) raw_bottom.add(k * 8 + 4),
                base = in(reg32) next_base, pol = in(reg64) policy,
                w0 = out(reg32) w0, w1 = out(reg32) w1, w2 = out(reg32) w2, w3 = out(reg32) w3,
                options(readonly, nostack)
            );
            let ahead = if k + 1 < tiles { k + 1 } else { k };
            next_base = load_u32(warp_header + (ahead * 4) as u32);
            [w0, w1, w2, w3]
        }};}

        macro_rules! stage_p_x {($first_t:expr, $count_t:expr) => {{
            let first_t = $first_t as usize;
            let count_t = $count_t as usize;
            let items = count_t * 4;
            let mut t = 0usize;
            while t < inputs {
                let row_dst = smem + (t * row_bytes) as u32;
                let mut i = tid;
                while i < items {
                    let kb = first_t * 16 + i * 4;
                    let (src, off) = if kb < half { (input, kb) } else { (input2, kb - half) };
                    let v = ld_global_nc_v4(src.add(t * half + off));
                    let at = row_dst + ((i / 4) * 32 + (i & 1) * 16 + (i & 2) * 2) as u32;
                    store_u32(at, cvt_bf16x2(v[0], v[1]));
                    store_u32(at + 8, cvt_bf16x2(v[2], v[3]));
                    i += FEW_THREADS;
                }
                t += 1;
            }
        }};}

        let last_t = tiles - 1;
        let mut ring = [[0u32; 4]; P_DEPTH];
        macro_rules! prime_p {
            ($($j:literal)*) => {$( ring[$j] = fetch_p!(if $j < last_t { $j } else { last_t }); )*};
        }
        prime_p!(0 1 2 3 4 5 6 7 8 9 10 11);

        let init_len = part_tiles.min(tiles);
        stage_p_x!(0, init_len);
        core::arch::nvptx::_syncthreads();

        let x_lane0 = smem + (g.min(inputs - 1) * row_bytes + s * 8) as u32;
        let x_lane1 = if TWO { smem + ((8 + g).min(inputs - 1) * row_bytes + s * 8) as u32 } else { x_lane0 };
        let mut acc0 = [0.0f32; 4];
        let mut acc1 = [0.0f32; 4];

        macro_rules! consume_p {($slot:expr, $k:expr, $xoff:expr) => {{
            let [w0, w1, w2, w3]: [u32; 4] = $slot;
            let base = load_u32(warp_header + (($k) * 4) as u32);
            let decoded = decode_tile!(base, w0, w1, w2);
            let raw = base == RAW;
            let a = [
                if raw { w0 } else { decoded[0] },
                if raw { w1 } else { decoded[1] },
                if raw { w2 } else { decoded[2] },
                if raw { w3 } else { decoded[3] },
            ];
            let xoff: u32 = $xoff;
            acc0 = mma_m16n8k16_bf16(a, crate::ptx::load2(x_lane0 + xoff), acc0);
            if TWO {
                acc1 = mma_m16n8k16_bf16(a, crate::ptx::load2(x_lane1 + xoff), acc1);
            }
        }};}

        let mut x_off = 0u32;
        let mut part_end = init_len;
        let mut k = 0usize;
        while k + P_DEPTH <= tiles {
            if k == part_end {
                core::arch::nvptx::_syncthreads();
                let next_len = part_tiles.min(tiles - k);
                stage_p_x!(k, next_len);
                core::arch::nvptx::_syncthreads();
                part_end = k + next_len;
                x_off = 0;
            }
            macro_rules! step_p {
                ($($j:literal)*) => {$(
                    consume_p!(ring[$j], k + $j, x_off + ($j * 32) as u32);
                    let nxt = k + $j + P_DEPTH;
                    ring[$j] = fetch_p!(if nxt < last_t { nxt } else { last_t });
                )*};
            }
            step_p!(0 1 2 3 4 5 6 7 8 9 10 11);
            k += P_DEPTH;
            x_off += (P_DEPTH * 32) as u32;
        }
        if k < tiles && k == part_end {
            core::arch::nvptx::_syncthreads();
            stage_p_x!(k, tiles - k);
            core::arch::nvptx::_syncthreads();
            x_off = 0;
        }
        macro_rules! tail_p {
            ($($j:literal)*) => {$( if k + $j < tiles { consume_p!(ring[$j], k + $j, x_off + ($j * 32) as u32); } )*};
        }
        tail_p!(0 1 2 3 4 5 6 7 8 9 10);
        return (acc0, acc1);
    }

    if NARROW {
        let in_valid0 = g < inputs;
        let in_valid1 = TWO && (8 + g < inputs);
        let r = lane >> 3;
        let w_chunk = lane & 7;
        let gr = block_first + r;
        let w_dst_off = (r * NARROW_ROW_BYTES + w_chunk * 16) as u32;
        let b_row_off = (g * NARROW_ROW_BYTES + 4 * s) as u32;
        let b_valid = g < 4;
        let w_smem = smem;

        macro_rules! issue_n {($chunk:expr,$stage:expr)=>{{
            let dst = $stage + w_dst_off;
            if gr < rows {
                crate::ptx::copy16(dst, weights.add(gr * cols + $chunk * BK + w_chunk * 8).cast());
            } else {
                st_shared_zero16(dst);
            }
        }};}

        // One loop pass covers NARROW_UNROLL stages (one whole narrow unit of 256 columns): one wait, one pair of
        // warp barriers and one prefetch per pass. The MMAs still run stage by stage in ascending K.
        macro_rules! issue_group {($step:expr)=>{{
            let mut u = 0usize;
            while u < NARROW_UNROLL {
                let chunk = $step * NARROW_UNROLL + u;
                issue_n!(chunk, w_smem + ((chunk % NS_NARROW) * NARROW_STAGE_BYTES) as u32);
                u += 1;
            }
        }};}
        let steps = chunks / NARROW_UNROLL;
        let mut st = 0usize;
        while st + 1 < NARROW_GROUPS {
            if st < steps { issue_group!(st); }
            crate::ptx::commit();
            st += 1;
        }

        // The FP32 input words of stage `$chunk` (zeros for an absent input row), loaded when the warp consumes each stage.
        macro_rules! load_x {($valid:expr, $row_idx:expr, $chunk:expr)=>{{
            if $valid && $chunk < chunks {
                let kb = $chunk * BK;
                let (src, off) = if kb < half { (input, kb) } else { (input2, kb - half) };
                let base = ($row_idx) * half + off + 2 * s;
                [ld_global_nc_v2_u32(src.add(base).cast()), ld_global_nc_v2_u32(src.add(base + 8).cast()),
                 ld_global_nc_v2_u32(src.add(base + 16).cast()), ld_global_nc_v2_u32(src.add(base + 24).cast()),
                 ld_global_nc_v2_u32(src.add(base + 32).cast()), ld_global_nc_v2_u32(src.add(base + 40).cast()),
                 ld_global_nc_v2_u32(src.add(base + 48).cast()), ld_global_nc_v2_u32(src.add(base + 56).cast())]
            } else {
                [[0u32; 2]; 8]
            }
        }};}
        let mut acc = [0.0f32; 4];
        macro_rules! kstep_n {($stage:expr, $x0:expr, $x1:expr, $ks:literal)=>{{
            let (b0, b1) = if b_valid {
                (load_u32($stage + ($ks * 32) as u32), load_u32($stage + ($ks * 32 + 16) as u32))
            } else {
                (0u32, 0u32)
            };
            let (a1, a3) = if TWO { ($x1[$ks][0], $x1[$ks][1]) } else { (0u32, 0u32) };
            acc = mma_m16n8k16_bf16([$x0[$ks][0], a1, $x0[$ks][1], a3], [b0, b1], acc);
        }};}

        // The FP32 input of pass `$pass` into L1 (rows 0..8, plus 8..16 when `TWO` is true):
        // `load_x!` reads from L1 in place of waiting on L2 at every stage.
        macro_rules! prefetch_x {($pass:expr)=>{{
            let pass: usize = $pass;
            if pass < steps {
                let mut j = 0usize;
                while j < 2 {
                    let line = lane + 32 * j;
                    let (u, r, h) = (line / 16, line % 16 / 2, line % 2);
                    let kb = (pass * NARROW_UNROLL + u) * BK;
                    let (src, off) = if kb < half { (input, kb) } else { (input2, kb - half) };
                    if r < inputs {
                        asm!("prefetch.global.L1 [{a}];", a = in(reg64) src.add(r * half + off + h * 32) as u64);
                    }
                    if TWO && r + 8 < inputs {
                        asm!("prefetch.global.L1 [{a}];", a = in(reg64) src.add((r + 8) * half + off + h * 32) as u64);
                    }
                    j += 1;
                }
            }
        }};}
        prefetch_x!(0);
        prefetch_x!(1);
        let mut step = 0usize;
        while step < steps {
            let ahead = step + NARROW_GROUPS - 1;
            if ahead < steps { issue_group!(ahead); }
            crate::ptx::commit();
            // Lanes 0..15: row lane % 4, stage lane / 4 of the pass after the copies in flight.
            let pc_chunk = (step + NARROW_GROUPS) * NARROW_UNROLL + (lane >> 2);
            if lane < 16 && pc_chunk < chunks && block_first + (lane & 3) < rows {
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add((block_first + (lane & 3)) * cols + pc_chunk * BK) as u64);
            }
            prefetch_x!(step + 2);
            asm!("cp.async.wait_group {n};", n = const (NARROW_GROUPS - 1));
            asm!("bar.warp.sync 0xffffffff;");
            macro_rules! unit {($u:literal)=>{{
                let chunk = step * NARROW_UNROLL + $u;
                let stage = w_smem + ((chunk % NS_NARROW) * NARROW_STAGE_BYTES) as u32 + b_row_off;
                let xu0 = load_x!(in_valid0, g, chunk);
                let xu1 = if TWO { load_x!(in_valid1, 8 + g, chunk) } else { [[0u32; 2]; 8] };
                macro_rules! pair {($xu:expr, $i:expr)=>{{
                    [cvt_bf16x2(f32::from_bits($xu[$i][0]), f32::from_bits($xu[$i][1])), cvt_bf16x2(f32::from_bits($xu[$i + 1][0]), f32::from_bits($xu[$i + 1][1]))]
                }};}
                let x0 = [pair!(xu0, 0), pair!(xu0, 2), pair!(xu0, 4), pair!(xu0, 6)];
                let x1 = if TWO { [pair!(xu1, 0), pair!(xu1, 2), pair!(xu1, 4), pair!(xu1, 6)] } else { [[0u32; 2]; 4] };
                kstep_n!(stage, x0, x1, 0);
                kstep_n!(stage, x0, x1, 1);
                kstep_n!(stage, x0, x1, 2);
                kstep_n!(stage, x0, x1, 3);
            }};}
            unit!(0);
            unit!(1);
            unit!(2);
            unit!(3);
            asm!("bar.warp.sync 0xffffffff;");
            step += 1;
        }

        return (acc, [0.0f32; 4]);
    }

    // Wide uncompressed: 4 warps (128 threads) owning 64 weight rows, sharing a FEW_STAGES-stage ring of XOR-swizzled BF16 X in SMEM
    // (30,720 B = 3 blocks/SM; each thread loads 1 x v4 per 8-token tile, overlapped with MMAs).
    let first_row = block_first + warp_id * 16;
    let r_base = lane >> 3;
    let w_chunk = lane & 7;
    let off0 = swiz128(lane & 15, lane >> 4);
    let x_off0 = swiz128(g, 0) + (s * 4) as u32;

    let tx_row = tid >> 4;
    let tx_col4 = tid & 15;
    let tx_dst = swiz128(tx_row, tx_col4 >> 1) + ((tx_col4 & 1) * 8) as u32;
    let tx_valid0 = tx_row < inputs;
    let tx_valid1 = TWO && (tx_row + 8 < inputs);

    macro_rules! issue_w {($chunk:expr,$w_stage:expr)=>{{
        let col = $chunk * BK + w_chunk * 8;
        let mut i = 0usize;
        while i < 4 {
            let r = i * 4 + r_base;
            let dst = $w_stage + swiz128(r, w_chunk);
            let gr = first_row + r;
            if gr < rows {
                crate::ptx::copy16(dst, weights.add(gr * cols + col).cast());
            } else {
                st_shared_zero16(dst);
            }
            i += 1;
        }
    }};}

    macro_rules! fetch_x {($valid:expr, $row:expr, $k_base:expr)=>{{
        if $valid {
            let kb = $k_base;
            let (src, k_off) = if kb < half { (input, kb) } else { (input2, kb - half) };
            let v = ld_global_nc_v4(src.add($row * half + k_off + tx_col4 * 4));
            [cvt_bf16x2(v[0], v[1]), cvt_bf16x2(v[2], v[3])]
        } else {
            [0u32; 2]
        }
    }};}

    macro_rules! store_x {($x_stage:expr, $off:expr, $packed:expr)=>{{
        store_u32($x_stage + $off, $packed[0]);
        store_u32($x_stage + $off + 4, $packed[1]);
    }};}

    let w_warp_off = (warp_id * 2_048) as u32;
    macro_rules! stage_at {($chunk:expr)=>{ smem + ((($chunk) % FEW_STAGES) * FEW_STAGE_BYTES) as u32 };}
    macro_rules! prefetch_w {($chunk:expr)=>{{
        if lane < 16 && $chunk < chunks && first_row + lane < rows {
            asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add((first_row + lane) * cols + $chunk * BK) as u64);
        }
    }};}
    // Stages 0 .. FEW_STAGES - 2 in flight (one commit group each, empty past the last stage), then the L2 prefetch of
    // the stages up to WIDE_PREFETCH - 1.
    let mut st = 0usize;
    while st + 1 < FEW_STAGES {
        if st < chunks { issue_w!(st, stage_at!(st) + w_warp_off); }
        crate::ptx::commit();
        st += 1;
    }
    while st < WIDE_PREFETCH {
        prefetch_w!(st);
        st += 1;
    }
    let x0_init = fetch_x!(tx_valid0, tx_row, 0);
    store_x!(smem + FEW_INPUT_AT as u32, tx_dst, x0_init);
    if TWO {
        let x1_init = fetch_x!(tx_valid1, tx_row + 8, 0);
        store_x!(smem + FEW_INPUT_AT as u32, tx_dst + 1_024, x1_init);
    }
    asm!("cp.async.wait_group {n};", n = const (FEW_STAGES - 2));
    core::arch::nvptx::_syncthreads();

    let mut c0 = [0.0f32; 4];
    let mut c1 = [0.0f32; 4];
    let mut c = 0usize;
    while c < chunks {
        // Stage c + FEW_STAGES - 1 reuses the slot of stage c - 1, which every warp left before the last barrier.
        let ahead = c + FEW_STAGES - 1;
        if ahead < chunks { issue_w!(ahead, stage_at!(ahead) + w_warp_off); }
        crate::ptx::commit();
        prefetch_w!(c + WIDE_PREFETCH);
        let has_next = c + 1 < chunks;
        let x0_nxt = if has_next { fetch_x!(tx_valid0, tx_row, (c + 1) * BK) } else { [0u32; 2] };
        let x1_nxt = if has_next && TWO { fetch_x!(tx_valid1, tx_row + 8, (c + 1) * BK) } else { [0u32; 2] };

        let cur_stage = stage_at!(c);
        let cur_w = cur_stage + w_warp_off;
        let cur_x = cur_stage + FEW_INPUT_AT as u32;
        macro_rules! kstep_w {($xmask:expr)=>{{
            let a = ldmatrix_x4(cur_w + (off0 ^ $xmask));
            let b0 = [load_u32(cur_x + (x_off0 ^ $xmask)), load_u32(cur_x + (x_off0 ^ ($xmask ^ 16u32)))];
            c0 = mma_m16n8k16_bf16(a, b0, c0);
            if TWO {
                let b1 = [load_u32(cur_x + 1_024 + (x_off0 ^ $xmask)), load_u32(cur_x + 1_024 + (x_off0 ^ ($xmask ^ 16u32)))];
                c1 = mma_m16n8k16_bf16(a, b1, c1);
            }
        }};}
        kstep_w!(0u32);
        kstep_w!(32u32);
        kstep_w!(64u32);
        kstep_w!(96u32);

        if has_next {
            // The input slot of stage c + 1 was last read in stage c + 1 - FEW_STAGES, before the last barrier.
            let nxt_x = stage_at!(c + 1) + FEW_INPUT_AT as u32;
            store_x!(nxt_x, tx_dst, x0_nxt);
            if TWO { store_x!(nxt_x, tx_dst + 1_024, x1_nxt); }
            // All but the newest FEW_STAGES - 2 groups are complete: stage c + 1 has arrived.
            asm!("cp.async.wait_group {n};", n = const (FEW_STAGES - 2));
            core::arch::nvptx::_syncthreads();
        }
        c += 1;
    }
    (c0, c1)
}
