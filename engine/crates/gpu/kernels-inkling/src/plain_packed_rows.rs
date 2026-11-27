//! The packed rows tile (`packed_product`, THE PACKED ROWS TILE) and the small-batch routed expert product
//! (`M = 2..=16` verify and multi-chat decode): the entry `resident_launch_plain_packed_rows` of the Inkling image
//! (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_PACKED_ROWS`.
//!
//! Word 0 of the recorded command distinguishes the two forms:
//! - Word 0 == 0: the grouped rows tile (`Backend::project_packed_rows_grouped_into`, 256 threads, 45,056 dynamic
//!   shared bytes), reading each run's packed unit and running the exact ascending FP32 chains of `plain_staged_rows`.
//! - Word 0 != 0: the strided multi-row picked MoE product (`Backend::project_packed_picked_strided_into`, 128 threads,
//!   dynamic shared bytes), grouping up to 16 picks of one expert into one pass (`acc0` + `acc1`), with overlapped
//!   8 KB SMEM weight LUT + coalesced `v4` BF16 `X` staging (up to 17 KB bank-free `X` budget), `DEPTH=4`
//!   `.L2::evict_first` continuous streaming panel ring, and 3-stage `wait_group 1` single-pass `1..=16` MXFP8.
use crate::packed_product::{
    BLOCK_ROWS, COLS, NARROW_TABLE, PAIRS, PICKED_THREADS, PICKS_MAX, ROWS_TABLE_BYTES, STAGE,
    THREADS, WIDE, admitted, code_stage, layout, output_row, power, round_bf16, rows_admitted,
    table_copies,
};
use crate::stream_pairs::gate_tanh;
use crate::tc_ptx::{
    cp_async_wait_one, cp_async_wait_zero, cvt_bf16x2, l2_evict_first, ld_global_nc_v4, load_u32,
    mma_m16n8k16_bf16, store_u32,
};
use core::arch::{asm, nvptx};

/// Weight rows and launch rows of a grouped-rows block.
const BR: usize = crate::staged_product::ROWS_TILE;
const BT: usize = crate::staged_product::ROWS_TILE;
/// Columns of a grouped-rows chunk, and shared words of one row of a chunk.
const KC: usize = 32;
const SK: usize = 36;
/// Shared layout of the grouped-rows mode (passed as dynamic shared bytes).
const STAGE_WORDS: usize = (BR + BT) * SK;
const TABLE_AT: usize = 2 * STAGE_WORDS * 4;
const BLOCK_BYTES: usize = TABLE_AT + ROWS_TABLE_BYTES;
const _: () = assert!(crate::staged_product::ROWS_THREADS == BR / 4 * (BT / 4) && crate::staged_product::STAGE % KC == 0);
const _: () = assert!(TABLE_AT == 36_864 && BLOCK_BYTES == 45_056 && TABLE_AT % 16 == 0);
const _: () = assert!(BR * 8 % crate::staged_product::ROWS_THREADS == 0 && BT * 8 == 2 * crate::staged_product::ROWS_THREADS);

/// Picked multi-row mode constants: 4 warps x 16 rows = 64 rows/block, up to 16 picks/group.
const PICKED_ROWS: u32 = 64;
const PANEL_META_BYTES: u32 = 128;
const PANEL_OVERHEAD_BYTES: u32 = NARROW_TABLE as u32 + PANEL_META_BYTES;
/// Three async stages for MXFP8 (`3 * 6,400 = 19,200 B`, 5 blk/SM on SM121a).
const WIDE_STAGES: u32 = 3;

#[inline(always)]
fn picked_input_offset16(row: u32, column: u32) -> u32 {
    (row * STAGE as u32 + (column ^ ((row & 7) * 8))) * 2
}

#[inline(always)]
fn picked_code_offset(row: u32, column: u32, width: u32) -> u32 {
    let base = row * width;
    base + (column ^ ((base / 128 * 16) & (width - 16)))
}

/// The replay of one recorded command of kind `PLAIN_PACKED_ROWS`.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_packed_rows`): `table` and `index` are the
/// arguments the recorded replay passes to the entry; the block inside names what the host checks for both modes.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. Dynamic shared memory starts at 0 and its size `%dynamic_smem_size` is validated before any shared
    // access in either mode.
    // Mode A (word 0 != 0): recorded only in `Backend::project_packed_picked_strided_into` after
    // `packed_product::check_picked_strided` passed. `stack` holds `experts` contiguous 16-byte aligned units of
    // `unit_bytes = layout(rows, cols, wide).bytes`; `ids` holds the checked strided route words (4-byte aligned);
    // `input` is 16-byte aligned with `(picks / repeat) * cols` words (`cols` when `repeat == 0`); `output` holds
    // `rows * picks` writable words disjoint from `stack`, `ids`, and `input`. Every route ID is checked against
    // `0..experts` as an exact whole number before indexing `stack`; `picks`, `blockIdx.y`, `admitted`, alignment,
    // and dynamic shared bytes are re-checked before any read or write. Picks sharing an expert are partitioned into
    // groups of 1..=16 (`idx_lo`, `idx_hi`); only each group's leader (`(prev_count & 15) == 0`) runs all contiguous
    // 64-row bands (`block = bx`), so every `(pick, row)` output word has at most one writer. For fused `CappedGate`, `pairs` is checked, `caps`
    // points to 4 initialized words in the command, and the butterfly mask includes all active `s` columns across all
    // 8 `g` rows so paired lanes converge.
    // Mode B (word 0 == 0): recorded only in `Backend::project_packed_rows_grouped_into` after
    // `packed_product::check_rows_grouped` passed, with `dynamic_smem_size >= BLOCK_BYTES` (45,056 B); all bounds,
    // alignments, table indices, and `rows_admitted` are checked before any read or write, and all 256 threads reach
    // every barrier.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_PACKED_ROWS { nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let smem = 0u32;
        let shared_bytes: u32;
        asm!("mov.u32 {b}, %dynamic_smem_size;", b = out(reg32) shared_bytes, options(pure, nomem, nostack));

        if *words != 0 {
            let stack = *words as *const u8;
            let (input, output) = (*words.add(1) as *const f32, *words.add(2) as *mut f32);
            let (rows, word, picks) = (*words.add(3) as usize, *words.add(4) as u32, *words.add(5) as usize);
            let (cols, wide, pairs) = ((word & COLS) as usize, word & WIDE != 0, word & PAIRS != 0);
            let input_words = *words.add(6) as usize;
            let (ids, experts, unit_bytes) = (*words.add(7) as *const f32, *words.add(8) as u32, *words.add(9) as usize);
            let threads = nvptx::_block_dim_x() as usize;
            let e = nvptx::_block_idx_y() as usize;
            if threads != PICKED_THREADS || picks == 0 || picks > PICKS_MAX || e >= picks
                || ids.is_null() || (ids as usize & 3) != 0 || experts == 0
                || !admitted(rows, cols, wide, pairs, stack as usize, input as usize, THREADS)
            {
                nvptx::trap();
            }
            let caps = if *words.add(10) == 0 {
                core::ptr::null()
            } else {
                if !pairs { nvptx::trap(); }
                words.add(10)
            };
            let repeat = { let r = *words.add(14) as u32; if r == 0 { 1 } else { r } };
            let ppr = *words.add(15) as u32;
            let id_stride = *words.add(16) as u32;
            let id_at = |j: u32| -> f32 {
                if ppr == 0 { *ids.add(j as usize) } else { *ids.add(((j / ppr) * id_stride + (j % ppr)) as usize) }
            };
            let e_u32 = e as u32;
            let picks_u32 = picks as u32;
            let is_down_pdl = caps.is_null() && !pairs && repeat == 1 && input_words != 0;
            if !is_down_pdl {
                asm!("griddepcontrol.wait;", options(nostack));
            }
            let id = id_at(e_u32);
            if !(id >= 0.0 && id < experts as f32) || (id as u32) as f32 != id || id as u32 >= experts {
                nvptx::trap();
            }

            let tid = nvptx::_thread_idx_x() as u32;
            let lane = tid & 31;
            let rows_u32 = rows as u32;
            let cols_u32 = cols as u32;
            let stride_u32 = input_words as u32;
            let gx = nvptx::_grid_dim_x() as u32;
            if gx * PICKED_ROWS < rows_u32 { nvptx::trap(); }
            let block = nvptx::_block_idx_x() as u32;
            let first = block * PICKED_ROWS;
            if first >= rows_u32 { return; }

            // Fast leader test: count earlier picks `< e_u32` sharing expert `id`. Only the 0th, 16th, 32nd, ...
            // pick of each expert (`(prev_count & 15) == 0`) runs all contiguous 64-row bands `bx = 0..row_tiles`
            // together (maximizing LPDDR5X sequential row-buffer hits and L2 wtable/X sharing); duplicates exit fast.
            let mut prev_count = 0u32;
            let mut fp = 0u32;
            while fp < e_u32 {
                let at = fp + lane;
                let m = crate::ptx::ballot(at < e_u32 && id_at(at) == id);
                prev_count += m.count_ones();
                fp += 32;
            }
            if (prev_count & 15) != 0 { return; }

            let (mut idx_lo, mut idx_hi, mut count) = (0u64, 0u64, 0u32);
            let mut first_p = e_u32 & !31u32;
            while first_p < picks_u32 && count < 16 {
                let at = first_p + lane;
                let mut m = crate::ptx::ballot(at >= e_u32 && at < picks_u32 && id_at(at) == id);
                while m != 0 && count < 16 {
                    let j = first_p + m.trailing_zeros();
                    m &= m - 1;
                    if count < 8 {
                        idx_lo |= (j as u64) << (8 * count);
                    } else {
                        idx_hi |= (j as u64) << (8 * (count - 8));
                    }
                    count += 1;
                }
                first_p += 32;
            }

            let unit = stack.add(id as usize * unit_bytes);
            let parts = layout(rows, cols, wide);
            let codes = unit;
            let scales = unit.add(parts.scales);
            let wtable = unit.add(parts.table);
            let (g, s) = (lane >> 2, lane & 3);
            let warp = tid >> 5;

            let panel = !wide && rows % 64 == 0 && {
                let t32 = wtable.cast::<u32>();
                *t32.add(0x7f * 8) == crate::packed_product::NVFP4_PANEL_MAGIC_LO
                    && *t32.add(0x7f * 8 + 1) == crate::packed_product::NVFP4_PANEL_MAGIC_HI
            };

            if panel {
                const DEPTH: u32 = 4;
                if (shared_bytes & 15) != 0 || shared_bytes < PANEL_OVERHEAD_BYTES + 16 * (DEPTH * 128 + 32) {
                    nvptx::trap();
                }
                let x_budget = shared_bytes - PANEL_OVERHEAD_BYTES;
                let table_addr = smem + x_budget;
                let meta = table_addr + NARROW_TABLE as u32;
                let mut ti = tid;
                while ti < table_copies(false) as u32 {
                    crate::ptx::copy16(table_addr + ti * 16, wtable.add((ti * 16) as usize).cast());
                    ti += PICKED_THREADS as u32;
                }
                crate::ptx::commit();

                let chunks = cols_u32 / 64;
                let max_slab = ((x_budget / count) - 32) / 128;
                let one_shot_x = chunks <= max_slab;
                let slab_chunks = if one_shot_x { chunks } else { (max_slab / DEPTH) * DEPTH };
                let row_stride = slab_chunks * 128 + 32;
                if tid == 0 {
                    let put64 = |off: u32, v: u64| {
                        store_u32(meta + off, v as u32);
                        store_u32(meta + off + 4, (v >> 32) as u32);
                    };
                    put64(0, output as u64);
                    put64(8, caps as u64);
                    put64(16, idx_lo);
                    put64(24, idx_hi);
                    put64(32, input as u64);
                    store_u32(meta + 40, stride_u32);
                    store_u32(meta + 44, repeat);
                    store_u32(meta + 48, row_stride);
                    store_u32(meta + 52, slab_chunks);
                    store_u32(meta + 56, rows_u32);
                }

                let codes_at = codes.cast::<u32>();
                let scales_at = scales.cast::<u32>();
                let policy = l2_evict_first();
                let stripes = (first >> 6) * chunks;
                let last_c = chunks - 1;
                macro_rules! fetch_panel {
                    ($chunk:expr) => {{
                        let stripe = (stripes + $chunk) * 4 + warp;
                        let addr = codes_at.add((stripe * 128 + lane * 4) as usize);
                        let (a, b, c_w, d): (u32, u32, u32, u32);
                        asm!(
                            "ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}}, [{addr}], {pol};",
                            a = out(reg32) a, b = out(reg32) b, c = out(reg32) c_w, d = out(reg32) d,
                            addr = in(reg64) addr, pol = in(reg64) policy, options(readonly, nostack)
                        );
                        let saddr = scales_at.add((stripe * 16 + g * 2) as usize);
                        let (s0, s1): (u32, u32);
                        asm!(
                            "ld.global.nc.L2::cache_hint.L2::128B.v2.u32 {{{s0},{s1}}}, [{saddr}], {pol};",
                            s0 = out(reg32) s0, s1 = out(reg32) s1,
                            saddr = in(reg64) saddr, pol = in(reg64) policy, options(readonly, nostack)
                        );
                        ([a, b, c_w, d], [s0, s1])
                    }};
                }
                macro_rules! load_meta64 {
                    ($off:expr) => {{
                        (load_u32(meta + $off) as u64) | ((load_u32(meta + $off + 4) as u64) << 32)
                    }};
                }
                macro_rules! load_x_slab {
                    ($in_ptr:expr, $in_stride:expr, $in_repeat:expr, $ilo:expr, $ihi:expr, $r_stride:expr, $slab_start:expr, $slab_len:expr) => {{
                        let quads_in_slab = ($slab_len) * 16;
                        let mut g_idx = 0u32;
                        while g_idx < count {
                            let pk = if g_idx < 8 {
                                (($ilo >> (8 * g_idx)) & 255) as u32
                            } else {
                                (($ihi >> (8 * (g_idx - 8))) & 255) as u32
                            };
                            let xb = (pk / $in_repeat) * $in_stride + ($slab_start) * 64;
                            let g_smem = smem + g_idx * ($r_stride);
                            let mut quad = tid;
                            while quad < quads_in_slab {
                                let c_i = quad >> 4;
                                let q = quad & 15;
                                let v = ld_global_nc_v4($in_ptr.add((xb + c_i * 64 + q * 4) as usize));
                                let dst0 = g_smem + c_i * 128 + ((q >> 2) * 32) + ((q & 1) * 16) + ((q & 2) * 2);
                                store_u32(dst0, cvt_bf16x2(v[0], v[1]));
                                store_u32(dst0 + 8, cvt_bf16x2(v[2], v[3]));
                                quad += PICKED_THREADS as u32;
                            }
                            g_idx += 1;
                        }
                    }};
                }

                // Overlap the initial DEPTH=4 DRAM panel loads with wtable cp.async + initial X slab staging + barrier.
                let mut ring = [
                    fetch_panel!(0u32),
                    fetch_panel!(if 1u32 < last_c { 1u32 } else { last_c }),
                    fetch_panel!(if 2u32 < last_c { 2u32 } else { last_c }),
                    fetch_panel!(if 3u32 < last_c { 3u32 } else { last_c }),
                ];
                if is_down_pdl {
                    asm!("griddepcontrol.wait;", options(nostack));
                }
                load_x_slab!(input, stride_u32, repeat, idx_lo, idx_hi, row_stride, 0u32, slab_chunks);
                cp_async_wait_zero();
                nvptx::_syncthreads();

                let g_slot0 = if g < count { g } else { 0 };
                let x_base0 = smem + g_slot0 * row_stride + s * 8;
                let two_tiles = count > 8;
                let x_base1 = if two_tiles {
                    smem + (if 8 + g < count { 8 + g } else { 0 }) * row_stride + s * 8
                } else {
                    x_base0
                };

                macro_rules! decode_k16 {
                    ($packed:expr, $sc_half:expr) => {{
                        let packed: u32 = $packed;
                        let sc_half: u32 = $sc_half;
                        let top = table_addr + ((sc_half & 255) << 5);
                        let bot = table_addr + (((sc_half >> 8) & 255) << 5);
                        let dec = |line: u32, byte: u32| {
                            crate::ptx::load_u16(line + ((byte & 15) << 1))
                                | (crate::ptx::load_u16(line + ((byte >> 3) & 30)) << 16)
                        };
                        [dec(top, packed), dec(bot, packed >> 8), dec(top, packed >> 16), dec(bot, packed >> 24)]
                    }};
                }
                macro_rules! reload_x {
                    ($start_c:expr, $len_c:expr) => {{
                        nvptx::_syncthreads();
                        let in_ptr = load_meta64!(32) as *const f32;
                        let in_stride = load_u32(meta + 40);
                        let in_repeat = load_u32(meta + 44);
                        let ilo = load_meta64!(16);
                        let ihi = load_meta64!(24);
                        let r_stride = load_u32(meta + 48);
                        load_x_slab!(in_ptr, in_stride, in_repeat, ilo, ihi, r_stride, $start_c, $len_c);
                        nvptx::_syncthreads();
                    }};
                }

                let mut acc0 = [0.0f32; 4];
                let mut acc1 = [0.0f32; 4];

                macro_rules! consume_slot {
                    ($is_two:literal, $slot:expr, $off:expr) => {{
                        let (cw, sw): ([u32; 4], [u32; 2]) = $slot;
                        let off: u32 = $off;
                        let x0 = x_base0 + off;
                        let w0 = decode_k16!(cw[0], sw[0]);
                        acc0 = mma_m16n8k16_bf16(w0, crate::ptx::load2(x0), acc0);
                        if $is_two { acc1 = mma_m16n8k16_bf16(w0, crate::ptx::load2(x_base1 + off), acc1); }
                        let w1 = decode_k16!(cw[1], sw[0] >> 16);
                        acc0 = mma_m16n8k16_bf16(w1, crate::ptx::load2(x0 + 32), acc0);
                        if $is_two { acc1 = mma_m16n8k16_bf16(w1, crate::ptx::load2(x_base1 + off + 32), acc1); }
                        let w2 = decode_k16!(cw[2], sw[1]);
                        acc0 = mma_m16n8k16_bf16(w2, crate::ptx::load2(x0 + 64), acc0);
                        if $is_two { acc1 = mma_m16n8k16_bf16(w2, crate::ptx::load2(x_base1 + off + 64), acc1); }
                        let w3 = decode_k16!(cw[3], sw[1] >> 16);
                        acc0 = mma_m16n8k16_bf16(w3, crate::ptx::load2(x0 + 96), acc0);
                        if $is_two { acc1 = mma_m16n8k16_bf16(w3, crate::ptx::load2(x_base1 + off + 96), acc1); }
                    }};
                }
                macro_rules! step_one {
                    ($is_two:literal, $j:literal, $k:expr, $x_off:expr) => {{
                        consume_slot!($is_two, ring[$j], $x_off + ($j as u32) * 128);
                        let nxt = $k + ($j as u32) + DEPTH;
                        ring[$j] = fetch_panel!(if nxt < last_c { nxt } else { last_c });
                    }};
                }
                macro_rules! run_tile {
                    ($is_two:literal) => {{
                        let mut k = 0u32;
                        let mut x_off = 0u32;
                        let mut part_end = load_u32(meta + 52);
                        while k + DEPTH <= chunks {
                            if !one_shot_x && k == part_end {
                                let sc = load_u32(meta + 52);
                                let next_len = (chunks - k).min(sc);
                                reload_x!(k, next_len);
                                part_end = k + next_len;
                                x_off = 0;
                            }
                            if k + DEPTH >= chunks {
                                asm!("griddepcontrol.launch_dependents;", options(nostack));
                            }
                            step_one!($is_two, 0, k, x_off);
                            step_one!($is_two, 1, k, x_off);
                            step_one!($is_two, 2, k, x_off);
                            step_one!($is_two, 3, k, x_off);
                            k += DEPTH;
                            x_off += DEPTH * 128;
                        }
                        if k < chunks {
                            if !one_shot_x && k == part_end {
                                reload_x!(k, chunks - k);
                                x_off = 0;
                            }
                            asm!("griddepcontrol.launch_dependents;", options(nostack));
                            consume_slot!($is_two, ring[0], x_off);
                            if k + 1 < chunks { consume_slot!($is_two, ring[1], x_off + 128); }
                            if k + 2 < chunks { consume_slot!($is_two, ring[2], x_off + 256); }
                        }
                    }};
                }

                if !two_tiles {
                    run_tile!(false);
                } else {
                    run_tile!(true);
                }

                let out_ptr = load_meta64!(0) as *mut f32;
                let caps_ptr = load_meta64!(8) as *const u64;
                let rows_m = load_u32(meta + 56);
                let top = first + warp * 16 + g;
                macro_rules! store_tile {
                    ($acc:expr, $idx_word:expr, $tile_count:expr) => {{
                        let tc: u32 = $tile_count;
                        let iw: u64 = $idx_word;
                        let mut col_i = 0u32;
                        while col_i < 2 {
                            let local_slot = 2 * s + col_i;
                            if local_slot < tc {
                                let pk = ((iw >> (8 * local_slot)) & 255) as u32;
                                let (v0, v1) = if col_i == 0 { ($acc[0], $acc[2]) } else { ($acc[1], $acc[3]) };
                                if caps_ptr.is_null() {
                                    let row_off = |row: u32| if pairs { row / 2 + (row & 1) * (rows_m / 2) } else { row };
                                    let out = out_ptr.add((pk * rows_m) as usize);
                                    *out.add(row_off(top) as usize) = v0;
                                    *out.add(row_off(top + 8) as usize) = v1;
                                } else {
                                    use crate::fixed_math::{rn, sigmoid};
                                    let half = (g & 1) as usize;
                                    let cap = f32::from_bits(*caps_ptr.add(half) as u32);
                                    let inv = f32::from_bits(*caps_ptr.add(2 + half) as u32);
                                    let columns = (tc + 1 - col_i) / 2;
                                    let mask = 0x1111_1111u32 * ((1u32 << columns) - 1);
                                    let out = out_ptr.add((pk * (rows_m / 2)) as usize);
                                    let write = |row: u32, value: f32| {
                                        let value = rn(value);
                                        let capped = gate_tanh(value * inv) * cap;
                                        let other: u32;
                                        asm!(
                                            "shfl.sync.bfly.b32 {out}, {v}, 4, 31, {mask};",
                                            out = out(reg32) other, v = in(reg32) capped.to_bits(), mask = in(reg32) mask
                                        );
                                        if half == 0 {
                                            *out.add((row / 2) as usize) = rn((capped * sigmoid(value)) * f32::from_bits(other));
                                        }
                                    };
                                    write(top, v0);
                                    write(top + 8, v1);
                                }
                            }
                            col_i += 1;
                        }
                    }};
                }
                store_tile!(acc0, load_meta64!(16), count.min(8));
                if two_tiles {
                    store_tile!(acc1, load_meta64!(24), count - 8);
                }
                return;
            }

            // MXFP8 (`wide == true`, 3-stage wait_group 1) and unpacked NVFP4 (`!wide && !panel`, 2-stage) path,
            // single-pass `count = 1..=16`, with cold epilogue state in SMEM to keep register count <= 128 (4 blk/SM).
            let code_bytes = code_stage(wide) as u32;
            let scale_at = PICKED_ROWS * code_bytes;
            let input_at = scale_at + PICKED_ROWS * 4;
            let stage_bytes = input_at + 16 * (STAGE as u32) * 2;
            let nstages = if wide { WIDE_STAGES } else { 2 };
            let meta_np = smem + nstages * stage_bytes;
            if shared_bytes < nstages * stage_bytes + 64 { nvptx::trap(); }

            if tid == 0 {
                let put64 = |off: u32, v: u64| {
                    store_u32(meta_np + off, v as u32);
                    store_u32(meta_np + off + 4, (v >> 32) as u32);
                };
                put64(0, output as u64);
                put64(8, caps as u64);
                put64(16, idx_lo);
                put64(24, idx_hi);
            }

            let native = if wide {
                let pair = (tid | ((128 + tid) << 8)) as u16;
                let (lo, hi): (f32, f32);
                asm!(
                    "{{ .reg .b32 h; .reg .b16 a, b; cvt.rn.f16x2.e4m3x2 h, {pair}; mov.b32 {{a, b}}, h; cvt.f32.f16 {lo}, a; cvt.f32.f16 {hi}, b; }}",
                    pair = in(reg16) pair, lo = out(reg32) lo, hi = out(reg32) hi, options(pure, nomem, nostack)
                );
                let t32 = wtable.cast::<u32>();
                let ok = *t32.add(tid as usize) == (if tid == 127 { 0 } else { lo.to_bits() })
                    && *t32.add(tid as usize + 128) == (if tid == 127 { 0 } else { hi.to_bits() });
                let result: u32;
                asm!(
                    "{{ .reg .pred p, q; setp.ne.u32 p, {ok}, 0; bar.red.and.pred q, 0, p; selp.u32 {result}, 1, 0, q; }}",
                    ok = in(reg32) u32::from(ok), result = out(reg32) result
                );
                result != 0
            } else {
                false
            };

            let chunks = cols_u32 / STAGE as u32;
            let warp_row = warp * 16;
            let (copy_row, copy_part) = (tid * 16 / code_bytes, tid * 16 % code_bytes);
            let pick_of = |slot: u32| -> u32 {
                if slot < 8 { ((idx_lo >> (8 * slot)) & 255) as u32 } else { ((idx_hi >> (8 * (slot - 8))) & 255) as u32 }
            };
            let two_tiles = count > 8;
            let (xr0, xc0) = (tid >> 4, (tid & 15) * 4);
            let x_live0 = xr0 < count;
            let x_base_in0 = if x_live0 { (pick_of(xr0) / repeat) * stride_u32 + xc0 } else { 0 };
            let x_dst_off0 = input_at + picked_input_offset16(xr0, xc0);
            let xr1 = xr0 + 8;
            let x_live1 = xr1 < count;
            let x_base_in1 = if x_live1 { (pick_of(xr1) / repeat) * stride_u32 + xc0 } else { 0 };
            let x_dst_off1 = input_at + picked_input_offset16(xr1, xc0);
            let copy_dst0 = picked_code_offset(copy_row, copy_part, code_bytes);
            let copy_dst1 = if wide { picked_code_offset(copy_row + BLOCK_ROWS as u32, copy_part, code_bytes) } else { 0 };

            let (r0, r1) = (warp_row + g, warp_row + g + 8);
            let live0 = first + r0 < rows_u32;
            let live1 = first + r1 < rows_u32;
            let code_base = first * chunks * code_bytes;
            let scale_stages = if wide { chunks >> 1 } else { chunks };
            let scale_base = first * scale_stages * 4;

            let issue_stage = |ch: u32, slot_idx: u32| {
                    let dst = smem + slot_idx * stage_bytes;
                    let group_rows = BLOCK_ROWS as u32;
                    let row_start = ((copy_row / group_rows * chunks + ch) * group_rows + copy_row % group_rows) * code_bytes;
                    let code_off = code_base + row_start + copy_part;
                    if first + copy_row < rows_u32 {
                        let d = dst + copy_dst0;
                        let s_p = codes.add(code_off as usize);
                        asm!("cp.async.cg.shared.global.L2::128B [{d}], [{s}], 16;", d = in(reg32) d, s = in(reg64) s_p);
                    }
                    if wide && first + copy_row + group_rows < rows_u32 {
                        let d = dst + copy_dst1;
                        let s_p = codes.add((code_off + group_rows * chunks * code_bytes) as usize);
                        asm!("cp.async.cg.shared.global.L2::128B [{d}], [{s}], 16;", d = in(reg32) d, s = in(reg64) s_p);
                    }
                    if tid < PICKED_ROWS && first + tid < rows_u32 {
                        let scale_chunk = if wide { ch >> 1 } else { ch };
                        let scale_off = scale_base + ((tid / group_rows * scale_stages + scale_chunk) * group_rows + tid % group_rows) * 4;
                        crate::ptx::copy4(dst + scale_at + tid * 4, scales.add(scale_off as usize).cast());
                    }
                    crate::ptx::commit();
                    if x_live0 {
                        let v = ld_global_nc_v4(input.add((x_base_in0 + ch * STAGE as u32) as usize));
                        store_u32(dst + x_dst_off0, cvt_bf16x2(v[0], v[1]));
                        store_u32(dst + x_dst_off0 + 4, cvt_bf16x2(v[2], v[3]));
                    }
                    if x_live1 {
                        let v = ld_global_nc_v4(input.add((x_base_in1 + ch * STAGE as u32) as usize));
                        store_u32(dst + x_dst_off1, cvt_bf16x2(v[0], v[1]));
                        store_u32(dst + x_dst_off1 + 4, cvt_bf16x2(v[2], v[3]));
                    }
                };

                let mut acc0 = [0.0f32; 4];
                let mut acc1 = [0.0f32; 4];

                if is_down_pdl {
                    asm!("griddepcontrol.wait;", options(nostack));
                }
                if wide {
                    issue_stage(0, 0);
                    if chunks > 1 {
                        issue_stage(1, 1);
                        cp_async_wait_one();
                    } else {
                        cp_async_wait_zero();
                    }
                    nvptx::_syncthreads();
                    let mut ch = 0u32;
                    let mut slot = 0u32;
                    while ch < chunks {
                        if ch + 2 < chunks {
                            let nxt_slot = if slot == 0 { 2 } else { slot - 1 };
                            issue_stage(ch + 2, nxt_slot);
                        } else if ch + 1 >= chunks {
                            asm!("griddepcontrol.launch_dependents;", options(nostack));
                        }
                        let cur = smem + slot * stage_bytes;
                        let (sc0, sc1) = (
                            if live0 { load_u32(cur + scale_at + r0 * 4) } else { 0 },
                            if live1 { load_u32(cur + scale_at + r1 * 4) } else { 0 },
                        );
                        let sc_shift = (ch & 1) * 16;
                        let x_sm = cur + input_at;
                        let mut k = 0u32;
                        while k < 4 {
                            let sh_k = sc_shift + ((k >> 1) << 3);
                            let unpack_mx = |row: u32, live: bool, scale: u32| -> (u32, u32) {
                                if !live { return (0, 0); }
                                let code_p = cur + picked_code_offset(row, 16 * k + 2 * s, code_bytes);
                                let c0 = crate::ptx::load_u16(code_p);
                                let c1 = crate::ptx::load_u16(code_p + 8);
                                if native && (1..143).contains(&scale) {
                                    let shifted = ((scale + 112) << 7) * 0x10001;
                                    let unpack_fast = |bits: u32| -> u32 {
                                        let clean = (if bits & 127 == 127 { 0 } else { bits & 255 })
                                            | (if bits & 0x7f00 == 0x7f00 { 0 } else { bits & 0xff00 });
                                        let h: u32;
                                        asm!("cvt.rn.f16x2.e4m3x2 {h}, {pair};", h = out(reg32) h, pair = in(reg16) clean as u16, options(pure, nomem, nostack));
                                        let vals = (h & 0x8000_8000) | ((h >> 3) & 0x0fff_0fff);
                                        let out_w: u32;
                                        asm!("mul.rn.bf16x2 {out_w}, {vals}, {sh};", out_w = out(reg32) out_w, vals = in(reg32) vals, sh = in(reg32) shifted, options(pure, nomem, nostack));
                                        out_w
                                    };
                                    (unpack_fast(c0), unpack_fast(c1))
                                } else {
                                    let mult = power(scale);
                                    let tf = wtable.cast::<f32>();
                                    let unpack_slow = |bits: u32| -> u32 {
                                        let (v0, v1): (f32, f32);
                                        asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v0, p = in(reg64) tf.add((bits & 255) as usize), options(readonly, nostack));
                                        asm!("ld.global.nc.f32 {v}, [{p}];", v = out(reg32) v1, p = in(reg64) tf.add((bits >> 8) as usize), options(readonly, nostack));
                                        cvt_bf16x2(v0 * mult, v1 * mult)
                                    };
                                    (unpack_slow(c0), unpack_slow(c1))
                                }
                            };
                            let (a0, a2) = unpack_mx(r0, live0, (sc0 >> sh_k) & 255);
                            let (a1, a3) = unpack_mx(r1, live1, (sc1 >> sh_k) & 255);
                            let b0 = if g < count {
                                [load_u32(x_sm + picked_input_offset16(g, k * 16 + 2 * s)), load_u32(x_sm + picked_input_offset16(g, k * 16 + 2 * s + 8))]
                            } else { [0, 0] };
                            acc0 = mma_m16n8k16_bf16([a0, a1, a2, a3], b0, acc0);
                            if two_tiles {
                                let b1 = if 8 + g < count {
                                    [load_u32(x_sm + picked_input_offset16(8 + g, k * 16 + 2 * s)), load_u32(x_sm + picked_input_offset16(8 + g, k * 16 + 2 * s + 8))]
                                } else { [0, 0] };
                                acc1 = mma_m16n8k16_bf16([a0, a1, a2, a3], b1, acc1);
                            }
                            k += 1;
                        }
                        if ch + 2 < chunks {
                            cp_async_wait_one();
                            nvptx::_syncthreads();
                        } else if ch + 1 < chunks {
                            cp_async_wait_zero();
                            nvptx::_syncthreads();
                        }
                        slot = if slot == 2 { 0 } else { slot + 1 };
                        ch += 1;
                    }
                } else {
                    issue_stage(0, 0);
                    cp_async_wait_zero();
                    nvptx::_syncthreads();
                    let mut ch = 0u32;
                    while ch < chunks {
                        if ch + 1 < chunks {
                            issue_stage(ch + 1, (ch + 1) & 1);
                        } else {
                            asm!("griddepcontrol.launch_dependents;", options(nostack));
                        }
                        let cur = smem + (ch & 1) * stage_bytes;
                        let (sc0, sc1) = (
                            if live0 { load_u32(cur + scale_at + r0 * 4) } else { 0 },
                            if live1 { load_u32(cur + scale_at + r1 * 4) } else { 0 },
                        );
                        let x_sm = cur + input_at;
                        let mut k = 0u32;
                        while k < 4 {
                            let unpack_n4 = |row: u32, live: bool, sc_w: u32| -> (u32, u32) {
                                if !live { return (0, 0); }
                                let sc = (sc_w >> (8 * k)) & 255;
                                let line = wtable.cast::<u16>().add((sc * 16) as usize);
                                let col_b = k * 8;
                                let p_lo = cur + picked_code_offset(row, col_b, code_bytes);
                                let even = load_u32(p_lo);
                                let odd = load_u32(p_lo + 4);
                                let dec_l2 = |byte: u32| -> u32 {
                                    let (lo_h, hi_h): (u16, u16);
                                    asm!("ld.global.nc.u16 {v}, [{p}];", v = out(reg16) lo_h, p = in(reg64) line.add((byte & 15) as usize), options(readonly, nostack));
                                    asm!("ld.global.nc.u16 {v}, [{p}];", v = out(reg16) hi_h, p = in(reg64) line.add(((byte >> 4) & 15) as usize), options(readonly, nostack));
                                    (lo_h as u32) | ((hi_h as u32) << 16)
                                };
                                (dec_l2((even >> (8 * s)) & 255), dec_l2((odd >> (8 * s)) & 255))
                            };
                            let (a0, a2) = unpack_n4(r0, live0, sc0);
                            let (a1, a3) = unpack_n4(r1, live1, sc1);
                            let b0 = if g < count {
                                [load_u32(x_sm + picked_input_offset16(g, k * 16 + 2 * s)), load_u32(x_sm + picked_input_offset16(g, k * 16 + 2 * s + 8))]
                            } else { [0, 0] };
                            acc0 = mma_m16n8k16_bf16([a0, a1, a2, a3], b0, acc0);
                            if two_tiles {
                                let b1 = if 8 + g < count {
                                    [load_u32(x_sm + picked_input_offset16(8 + g, k * 16 + 2 * s)), load_u32(x_sm + picked_input_offset16(8 + g, k * 16 + 2 * s + 8))]
                                } else { [0, 0] };
                                acc1 = mma_m16n8k16_bf16([a0, a1, a2, a3], b1, acc1);
                            }
                            k += 1;
                        }
                        if ch + 1 < chunks {
                            cp_async_wait_zero();
                            nvptx::_syncthreads();
                        }
                        ch += 1;
                    }
                }

                let load_np64 = |off: u32| -> u64 { (load_u32(meta_np + off) as u64) | ((load_u32(meta_np + off + 4) as u64) << 32) };
                let out_ptr = load_np64(0) as *mut f32;
                let caps_ptr = load_np64(8) as *const u64;
                let (gr0, gr1) = (first + r0, first + r1);
                macro_rules! store_tile_gen {
                    ($acc:expr, $idx_word:expr, $tile_count:expr) => {{
                        let tc: u32 = $tile_count;
                        let iw: u64 = $idx_word;
                        let mut col_i = 0u32;
                        while col_i < 2 {
                            let local_slot = 2 * s + col_i;
                            if local_slot < tc {
                                let pk = ((iw >> (8 * local_slot)) & 255) as u32;
                                let (v0, v1) = if col_i == 0 { ($acc[0], $acc[2]) } else { ($acc[1], $acc[3]) };
                                if caps_ptr.is_null() {
                                    let row_off = |row: u32| if pairs { row / 2 + (row & 1) * (rows_u32 / 2) } else { row };
                                    let out = out_ptr.add((pk * rows_u32) as usize);
                                    if live0 { *out.add(row_off(gr0) as usize) = v0; }
                                    if live1 { *out.add(row_off(gr1) as usize) = v1; }
                                } else {
                                    use crate::fixed_math::{rn, sigmoid};
                                    let half = (g & 1) as usize;
                                    let cap = f32::from_bits(*caps_ptr.add(half) as u32);
                                    let inv = f32::from_bits(*caps_ptr.add(2 + half) as u32);
                                    let columns = (tc + 1 - col_i) / 2;
                                    let mask = 0x1111_1111u32 * ((1u32 << columns) - 1);
                                    let out = out_ptr.add((pk * (rows_u32 / 2)) as usize);
                                    let store_gate = |row: u32, value: f32| {
                                        let value = rn(value);
                                        let capped = gate_tanh(value * inv) * cap;
                                        let other: u32;
                                        asm!(
                                            "shfl.sync.bfly.b32 {out}, {v}, 4, 31, {mask};",
                                            out = out(reg32) other, v = in(reg32) capped.to_bits(), mask = in(reg32) mask
                                        );
                                        if half == 0 && row < rows_u32 {
                                            *out.add((row / 2) as usize) = rn((capped * sigmoid(value)) * f32::from_bits(other));
                                        }
                                    };
                                    store_gate(gr0, v0);
                                    store_gate(gr1, v1);
                                }
                            }
                            col_i += 1;
                        }
                    }};
                }
                store_tile_gen!(acc0, load_np64(16), count.min(8));
                if two_tiles {
                    store_tile_gen!(acc1, load_np64(24), count - 8);
                }
            return;
        }

        // Mode B (word 0 == 0): grouped rows tile (256 threads, 45,056 dynamic shared bytes).
        if (shared_bytes as usize) < BLOCK_BYTES { nvptx::trap(); }
        let (input, output) = (*words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, word, launch_rows) = (*words.add(3) as usize, *words.add(4) as u32, *words.add(5) as usize);
        let (k, wide, pairs) = ((word & COLS) as usize, word & WIDE != 0, word & PAIRS != 0);
        let stride = *words.add(6) as usize;
        if stride < rows { nvptx::trap(); }
        let (in_table, in_count) = (*words.add(7) as *const u32, *words.add(8) as usize);
        let (out_table, out_count) = (*words.add(9) as *const u32, *words.add(10) as usize);
        if (in_table as usize | out_table as usize) & 3 != 0 { nvptx::trap(); }
        let place = |table: *const u32, count: usize, t: usize| -> usize {
            let row = if table.is_null() { t } else { table.add(t).read() as usize };
            if row >= count { nvptx::trap(); }
            row
        };
        let (runs_table, runs) = (*words.add(11) as *const u32, *words.add(12) as usize);
        let (blocks_table, blocks) = (*words.add(13) as *const u32, *words.add(14) as usize);
        let block = nvptx::_block_idx_x() as usize;
        if runs_table.is_null() || (runs_table as usize & 15) != 0 || (blocks_table as usize & 7) != 0 || block >= blocks { nvptx::trap(); }
        let entry = blocks_table.add(2 * block);
        let (run, tiles) = (entry.read() as usize, entry.add(1).read() as usize);
        if run >= runs { nvptx::trap(); }
        let at = runs_table.add(4 * run);
        let unit = (at.read() as usize | (at.add(1).read() as usize) << 32) as *const u8;
        let (base, m) = (at.add(2).read() as usize, at.add(3).read() as usize);
        if base + m > launch_rows { nvptx::trap(); }
        let (tile_t, tile_r) = (tiles & 0xffff, tiles >> 16);
        let threads = nvptx::_block_dim_x() as usize;
        if !rows_admitted(rows, k, wide, pairs, m, unit as usize, input as usize, threads) { nvptx::trap(); }
        if tile_r * BR >= rows { return; }
        let tid = nvptx::_thread_idx_x() as usize;
        let (ty, tx) = (tid / 16, tid % 16);
        let parts = layout(rows, k, wide);
        let (codes, scales) = (unit, unit.add(parts.scales));
        let t32 = unit.add(parts.table).cast::<u32>();
        let panel = !wide && rows % 64 == 0 && k % 64 == 0
            && t32.add(0x7f * 8).read() == crate::packed_product::NVFP4_PANEL_MAGIC_LO
            && t32.add(0x7f * 8 + 1).read() == crate::packed_product::NVFP4_PANEL_MAGIC_HI;
        let reg_nvfp4 = !wide
            && t32.add(0xff * 8).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_LO
            && t32.add(0xff * 8 + 1).read() == crate::packed_product::NVFP4_FACTOR_MAGIC_HI;
        let factor = if reg_nvfp4 { f32::from_bits(t32.add(0xff * 8 + 2).read()) } else { 0.0f32 };
        // Copy weight table into shared memory only when not using bit-exact SM121a register NVFP4 decode.
        if !reg_nvfp4 {
            let mut t = tid;
            while t < table_copies(wide) { crate::ptx::copy16(smem + (TABLE_AT + t * 16) as u32, unit.add(parts.table + t * 16).cast()); t += 256; }
            crate::ptx::commit();
            crate::ptx::wait();
            nvptx::_syncthreads();
        }
        let table_at = smem + TABLE_AT as u32;
        let mode: u32 = if panel { 2 } else if wide { 1 } else { 0 };
        let first_row = tile_t * BT + tid / 8;
        let in_low = if first_row < m { input.add(place(in_table, in_count, base + first_row) * k) } else { input };
        let in_high = if first_row + 32 < m { input.add(place(in_table, in_count, base + first_row + 32) * k) } else { input };
        let chunks = (k / KC) as u32;
        let rows_u32 = rows as u32;
        let k_u32 = k as u32;
        let m_u32 = m as u32;
        let tile_r_base = (tile_r * BR) as u32;
        let tile_t_base = (tile_t * BT) as u32;
        let tid_u32 = tid as u32;
        let issue = |c: u32| {
            let stage = smem + (c & 1) * (STAGE_WORDS * 4) as u32;
            let first = c * KC as u32;
            let mut item = tid_u32;
            while item < ((BR + BT) * 8) as u32 {
                let which = item >> 9;
                let row = (item >> 3) & 63;
                let quad = item & 7;
                let column = first + (quad << 2);
                let dst = stage + ((which * BR as u32 + row) * (SK * 4) as u32 + (quad << 4));
                if which == 0 {
                    let r = tile_r_base + row;
                    if r < rows_u32 {
                        let mut e = 0u32;
                        if mode == 1 {
                            let (code, scale) = crate::packed_product::rows_quad(r as usize, column as usize, k_u32 as usize, true);
                            let four = codes.add(code).cast::<u32>().read();
                            let power = power(*scales.add(scale) as u32);
                            while e < 4 {
                                let value = crate::ptx::load(table_at + ((four >> (8 * e)) & 255) * 4);
                                crate::ptx::store(dst + 4 * e, round_bf16(value * power));
                                e += 1;
                            }
                        } else {
                            let (four, sc) = if mode == 2 {
                                let ch = (((r >> 6) * (k_u32 >> 6) + (column >> 6)) << 2) | ((r >> 4) & 3);
                                let within = (column >> 4) & 3;
                                let pair = r & 7;
                                let bot = (r >> 3) & 1;
                                let base = (ch << 9) + (((pair << 2) | ((column >> 1) & 3)) << 4) + (within << 2) + (((column >> 3) & 1) << 1) + bot;
                                let b0 = *codes.add(base as usize) as u32;
                                let b1 = *codes.add(base as usize + 16) as u32;
                                let s = *scales.add(((ch << 6) + (pair << 3) + (within << 1) + bot) as usize) as u32;
                                (b0 | (b1 << 8), s)
                            } else {
                                let (code, scale) = crate::packed_product::rows_quad(r as usize, column as usize, k_u32 as usize, false);
                                (codes.add(code).cast::<u16>().read() as u32, *scales.add(scale) as u32)
                            };
                            if reg_nvfp4 {
                                let (w0, w1): (u32, u32);
                                core::arch::asm!(
                                    "{{",
                                    ".reg .b16 %hs, %ht, %hb, %h0, %h1;",
                                    ".reg .b8  %b0, %b1, %b2, %b3;",
                                    ".reg .b32 %sc, %st, %p0, %p1;",
                                    ".reg .f32 %f0, %f1;",
                                    "mov.b32 {{%b0, %b1, %b2, %b3}}, {four};",
                                    "cvt.u16.u32 %hs, {sc};",
                                    "cvt.rn.f16x2.e4m3x2 %sc, %hs;",
                                    "mov.b32 {{%ht, %hb}}, %sc;",
                                    "mov.b32 %st, {{%ht, %ht}};",
                                    "cvt.rn.f16x2.e2m1x2 %p0, %b0;",
                                    "cvt.rn.f16x2.e2m1x2 %p1, %b1;",
                                    "mul.rn.f16x2 %p0, %p0, %st;",
                                    "mul.rn.f16x2 %p1, %p1, %st;",
                                    "mov.b32 {{%h0, %h1}}, %p0;",
                                    "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                    "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                    "cvt.rn.bf16x2.f32 {w0}, %f1, %f0;",
                                    "mov.b32 {{%h0, %h1}}, %p1;",
                                    "cvt.f32.f16 %f0, %h0; cvt.f32.f16 %f1, %h1;",
                                    "mul.rn.f32 %f0, %f0, {fac}; mul.rn.f32 %f1, %f1, {fac};",
                                    "cvt.rn.bf16x2.f32 {w1}, %f1, %f0;",
                                    "}}",
                                    four = in(reg32) four,
                                    sc = in(reg32) sc,
                                    fac = in(reg32) factor,
                                    w0 = out(reg32) w0,
                                    w1 = out(reg32) w1,
                                    options(pure, nomem, nostack)
                                );
                                crate::ptx::store(dst, f32::from_bits(w0 << 16));
                                crate::ptx::store(dst + 4, f32::from_bits(w0 & 0xffff_0000));
                                crate::ptx::store(dst + 8, f32::from_bits(w1 << 16));
                                crate::ptx::store(dst + 12, f32::from_bits(w1 & 0xffff_0000));
                            } else {
                                let line = table_at + sc * 32;
                                while e < 4 {
                                    crate::ptx::store(dst + 4 * e, f32::from_bits(crate::ptx::load_u16(line + ((four >> (4 * e)) & 15) * 2) << 16));
                                    e += 1;
                                }
                            }
                        }
                    }
                } else {
                    let t = tile_t_base + row;
                    if t < m_u32 {
                        let src = if item < (BR * 8 + 256) as u32 { in_low } else { in_high };
                        crate::ptx::copy16(dst, src.add(column as usize).cast());
                    }
                }
                item += 256;
            }
        };
        let mut acc = [[0.0f32; 4]; 4];
        issue(0);
        crate::ptx::commit();
        let mut c = 0u32;
        while c < chunks {
            if c + 1 < chunks { issue(c + 1); crate::ptx::commit(); crate::ptx::wait_one(); } else { crate::ptx::wait(); }
            nvptx::_syncthreads();
            let stage = smem + (c & 1) * (STAGE_WORDS * 4) as u32;
            let mut q = 0u32;
            while q < (KC / 4) as u32 {
                let mut w = [[0u32; 4]; 4];
                let mut x = [[0u32; 4]; 4];
                for i in 0..4u32 { w[i as usize] = crate::ptx::load4(stage + (((ty as u32 * 4 + i) * SK as u32 + q * 4) * 4)); }
                for j in 0..4u32 { x[j as usize] = crate::ptx::load4(stage + (((BR as u32 + tx as u32 + 16 * j) * SK as u32 + q * 4) * 4)); }
                for e in 0..4 {
                    for i in 0..4 { for j in 0..4 { acc[i][j] += f32::from_bits(x[j][e]) * f32::from_bits(w[i][e]); } }
                }
                q += 1;
            }
            nvptx::_syncthreads();
            c += 1;
        }
        let mut target = [0usize; 4];
        for (j, slot) in target.iter_mut().enumerate() {
            let token = tile_t * BT + tx + 16 * j;
            if token < m { *slot = place(out_table, out_count, base + token); }
        }
        for i in 0..4 {
            let row = tile_r * BR + ty * 4 + i;
            if row < rows {
                let at = output_row(row, rows, pairs);
                for j in 0..4 {
                    let token = tile_t * BT + tx + 16 * j;
                    if token < m { *output.add(target[j] * stride + at) = acc[i][j]; }
                }
            }
        }
    }
}
