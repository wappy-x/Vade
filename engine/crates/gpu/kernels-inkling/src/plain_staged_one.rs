//! The one-row and small-batch panel BF16 product (`few_product`): the entry `resident_launch_plain_staged_one` of
//! the Inkling image (crate `kernels-inkling`, Ruling R6h), kind `inkling_launch::PLAIN_STAGED_ONE`. Its three unsafe
//! sites stay counted: `unsafe fn replay`, its block, `unsafe fn warp`.
//!
//! SM121a BF16 tensor-core decode/verify kernel (`M = 1..=8`): computes `Y = X * W^T`
//! via `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `X` rounded once via `cvt.rn.bf16x2.f32`.
//! Forms:
//! - Vade-BF11-BP Panels (word 7 in `1..=2`, `inputs` in `1..=8`): lossless bit-parallel 3-bit exponent panels
//!   (`bf16_panels`), streaming 64-column chunks (`DEPTH = 2`) via `L2::evict_first` + `L2::256B`
//!   (`2 x LDG.E.128` `sm0, sm1` + `LDG.E.64` `e3_lo` + `LDG.E.32` `e3_hi`), bit-parallel 3-bit unpack
//!   (`lop3`/`prmt`) with single atomic `asm!` escape prefetch, paired `LDS.128` BF16 `X`, and either 1-stream
//!   independent warps (`mode = 1`, e.g. LM head) or 4-warp cooperative ascending-K (`mode = 2`, Warps 1..3 decode
//!   interleaved chunks `4*step + 1..3` into double-buffered 12 KB SMEM while Warp 0 executes all MMAs in strictly
//!   ascending K order, zero Split-K reduction, 100% bit-exact, up to 3 fused independent panel items `Q+K+V`).
//!   `read_volatile()` epilogue keeps register pressure `<= 82` regs (`0` stack, `0` spills, `6 blk/SM` = `24 warps/SM`).
//! - Attached GateBlend (word 7 > 4, `cols == 64`, `inputs == 1`): 2-warp raw `cp.async` blocks (`dim / 16 x 64`,
//!   `4,352 B` dynamic SMEM so PDL stays off and occupancy reaches `12+ blk/SM`).
//! - Narrow (same shape rule as `PLAIN_STAGED_NARROW`): one warp computes four weight rows, with activations as A.
//! - Wide uncompressed: 16-row or 64-row streaming MMA paths.
use core::arch::asm;
use crate::bf16_panels::{
    BAND_META_WORDS, CHUNK_E3_HI_OFF, CHUNK_E3_LO_OFF, CHUNK_SM1_OFF, CHUNK_WORDS, ESC_STEP_WORDS,
    HEADER_WORDS, RAW_EXP_CHUNK_WORDS,
};
use crate::few_product::{
    FEW_INPUT_AT, FEW_MAX, FEW_NARROW_THREADS, FEW_SPLIT_MAX, FEW_SPLIT_OUT, FEW_STAGE_BYTES, FEW_THREADS,
    ONE_BLOCK_BYTES, ONE_FLAGS_WORD, ONE_OFFS_WORD, ONE_PAIR_WORD, ONE_PANEL_INPUTS, ONE_PANEL_WORD, ONE_PART_WORD,
    ONE_SPLIT_WORD, ONE_STAGES, ONE_TRIPLE_WORD,
};
use crate::tc_ptx::{
    cp_async_wait_zero, cvt_bf16x2, ld_global_nc_u32, ld_global_nc_v2_u32, ld_global_nc_v4, ld_global_nc_v4_u32,
    ldmatrix_x4, load_u32, mma_m16n8k16_bf16, st_shared_zero16, store_u32,
};

/// Columns of a stage (4 x k16 substeps).
const BK: usize = crate::staged_product::STAGE;
/// Narrow stage layout: four stages of four 144-byte weight rows.
const NARROW_ROW_BYTES: usize = BK * 2 + 16;
const NARROW_STAGE_BYTES: usize = 4 * NARROW_ROW_BYTES;
const NS_NARROW: usize = 4;
const _: () = assert!(BK == 64 && FEW_MAX == 16 && FEW_NARROW_THREADS == 32 && FEW_THREADS == 128);
const _: () = assert!(FEW_INPUT_AT == 8_192 && FEW_STAGE_BYTES == 10_240 && ONE_STAGES == 2 && ONE_BLOCK_BYTES == 20_480);
const _: () = assert!(NARROW_STAGE_BYTES == 576 && NS_NARROW * NARROW_STAGE_BYTES <= ONE_BLOCK_BYTES);

/// Zero-bank-conflict XOR-swizzled byte offset within a 128-byte-stride BF16 tile (`row` in 0..16, `chunk` in 0..8).
#[inline(always)]
const fn swiz128(row: usize, chunk: usize) -> u32 {
    (row * 128 + ((chunk ^ (row & 7)) * 16)) as u32
}

/// The replay of one recorded command of this kind: `table[index]` holds seven words, the addresses of the weights
/// (rows x cols BF16 patterns), of the input (inputs x cols FP32 values) and of the output, then rows, cols, inputs
/// and the output stride. A launch that the rule `few_product::few_admitted` does not take stops the block before
/// any read.
/// SAFETY (contract of the caller, the entry `resident_launch_plain_staged_one`): `table` and `index` are the arguments
/// the recorded replay passes to the entry; the block inside names what the host checks.
#[inline(always)]
pub(crate) unsafe fn replay(table: *const crate::inkling_launch::Command, index: u64) {
    // SAFETY: `table[index]` is one initialized Command in a live, private, 16-byte aligned device vector (the
    // recorded table of the queue); no replay writes it, and a command of another kind traps before its words are
    // read. The host records a command of this kind only in Backend::few_into or Backend::few_split_into after host
    // checks passed for all operands, extents, alignments, and non-overlap. For the panel form (word 7 in 1..=4) the
    // host checked that word 10 is the base of a registered Vade-BF11-BP `bf16_panels` encoding of the matrix that
    // word 0 views, word 11 is the starting 16-row band, word 12 is `part_chunks` dividing `chunks / split_k`, and
    // `split_k == slices_enc` (stored on `Panel`). Dynamic shared memory fits `one_panel_shared_slices(inputs,
    // part_chunks, split_k)` (checked here before any shared read/write). When `ONE_PAIR_WORD + 5` (`bands_a`) is
    // non-zero, `resident_launch::attach_panel_pair` checked both commands' operands and independence. All warps
    // participate in every `_syncthreads()`. Each output element `(t, r)` with `t < inputs` and `r < rows` has one
    // writer.
    unsafe {
        let c = &*table.add(index as usize);
        if c.kind != crate::inkling_launch::PLAIN_STAGED_ONE { core::arch::nvptx::trap(); }
        let words = c.payload.as_ptr().cast::<u64>();
        let (weights, input, output) = (*words as *const u16, *words.add(1) as *const f32, *words.add(2) as *mut f32);
        let (rows, cols, inputs, stride) = (*words.add(3) as usize, *words.add(4) as usize, *words.add(5) as usize, *words.add(6) as usize);
        let mode = *words.add(7);
        let is_panel = (1..=4).contains(&mode);
        if inputs != 1 && !is_panel { core::arch::nvptx::trap(); }
        let threads = core::arch::nvptx::_block_dim_x() as usize;
        let tid = core::arch::nvptx::_thread_idx_x() as usize;
        // SAFETY: this entry uses only dynamic shared memory. Check the recorded
        // allocation before the first access, including both uncompressed paths.
        let smem = 0u32;
        let shared_bytes: u32;
        asm!("mov.u32 {bytes}, %dynamic_smem_size;", bytes = out(reg32) shared_bytes);
        if is_panel {
            let coop = mode >= 2;
            let coop8_dbl = mode == 4;
            let coop8_sgl = mode == 3;
            let slices_want = if mode >= 3 { 8usize } else if mode == 2 { 4usize } else { 1usize };
            let flags = *words.add(ONE_FLAGS_WORD);
            let split_count = if flags & FEW_SPLIT_OUT != 0 { *words.add(ONE_FLAGS_WORD + 1) as usize } else { 0 };
            let part_chunks = *words.add(ONE_PART_WORD) as usize;
            let block_idx = core::arch::nvptx::_block_idx_x() as usize;
            let bands_a = *words.add(ONE_PAIR_WORD + 5) as usize;
            let bands_ab = *words.add(ONE_TRIPLE_WORD + 5) as usize;
            let use_c = bands_ab != 0 && block_idx >= bands_ab;
            let use_b = !use_c && bands_a != 0 && block_idx >= bands_a;
            let (rows, panels_base, first_band, bidx, packed_offs) = if use_c {
                (
                    *words.add(ONE_TRIPLE_WORD + 1) as usize,
                    *words.add(ONE_TRIPLE_WORD + 3) as *const u32,
                    *words.add(ONE_TRIPLE_WORD + 4) as usize,
                    block_idx - bands_ab,
                    *words.add(ONE_TRIPLE_WORD + 6),
                )
            } else if use_b {
                (
                    *words.add(ONE_PAIR_WORD + 1) as usize,
                    *words.add(ONE_PAIR_WORD + 3) as *const u32,
                    *words.add(ONE_PAIR_WORD + 4) as usize,
                    block_idx - bands_a,
                    *words.add(ONE_PAIR_WORD + 7),
                )
            } else {
                (
                    rows,
                    *words.add(ONE_PANEL_WORD) as *const u32,
                    *words.add(ONE_PANEL_WORD + 1) as usize,
                    block_idx,
                    *words.add(ONE_OFFS_WORD),
                )
            };
            let input_check = if use_c {
                *words.add(ONE_TRIPLE_WORD + 2) as usize
            } else if use_b {
                *words.add(ONE_PAIR_WORD + 2) as usize
            } else {
                input as usize
            };
            if !matches!(threads, 32 | 64 | 96 | 128 | 256) {
                core::arch::nvptx::trap();
            }
            let warps_per_block = threads / 32;
            let bands = rows.div_ceil(16);
            let chunks = cols / 64;
            let expected_blocks = if bands_a != 0 {
                *words.add(ONE_PAIR_WORD + 6) as usize
            } else if coop {
                bands
            } else {
                rows.div_ceil(16 * warps_per_block)
            };
            if inputs == 0 || inputs > ONE_PANEL_INPUTS
                || (bands_a == 0 && stride < rows)
                || !crate::few_product::one_panel_rows_admitted(rows, cols, inputs)
                || (weights as usize | input_check | panels_base as usize) & 15 != 0
                || panels_base.is_null()
                || flags & !FEW_SPLIT_OUT != 0
                || (split_count == 0 && (flags & FEW_SPLIT_OUT != 0 || (bands_a == 0 && output.is_null())))
                || split_count > FEW_SPLIT_MAX
                || (mode == 2 && threads != 128)
                || ((coop8_sgl || coop8_dbl) && threads != 256)
                || (coop8_dbl && inputs > 4)
                || (coop8_sgl && inputs <= 4)
                || (!coop && threads > 128)
                || (bands_ab != 0 && (bands_a == 0 || bands_ab <= bands_a))
                || (bands_a != 0 && (split_count != 0 || !(mode == 2 || threads == 32) || bidx >= bands))
                || !chunks.is_multiple_of(slices_want)
                || part_chunks == 0 || part_chunks > chunks
                || (coop && !part_chunks.is_multiple_of(slices_want))
                || crate::few_product::one_panel_shared_slices(inputs, part_chunks, slices_want) > shared_bytes as usize
                || core::arch::nvptx::_grid_dim_x() as usize != expected_blocks
            {
                core::arch::nvptx::trap();
            }
            let warp_id = tid / 32;
            let lane = (tid & 31) as u32;
            let local_band = if coop { bidx } else { bidx * warps_per_block + warp_id };
            let own_local = local_band.min(bands - 1);
            let abs_band = first_band + own_local;

            // Use host-validated `(chunks_off, esc_off)` from Command payload to eliminate cold DRAM header read;
            // fire initial weight chunk fetches `r0, r1` in parallel with per-band metadata at cycle 0.
            let (meta_off, chunks_off, esc_off) = if packed_offs != 0 {
                (HEADER_WORDS, (packed_offs & 0xFFFF_FFFF) as usize, (packed_offs >> 32) as usize)
            } else {
                let hdr = ld_global_nc_v4_u32(panels_base);
                if (hdr[3] >> 16) as usize != slices_want {
                    core::arch::nvptx::trap();
                }
                (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize)
            };

            let chunks_u32 = chunks as u32;
            let part_chunks_u32 = part_chunks as u32;
            let row_bytes = crate::few_product::one_panel_row_bytes_chunks(inputs, part_chunks) as u32;
            let mut x_cur = smem + ((lane >> 2).min((inputs as u32) - 1) * row_bytes + (lane & 3) * 16);
            let w_smem_base = smem + (inputs as u32) * row_bytes;
            let chunk_band = panels_base.add(chunks_off + abs_band * chunks * CHUNK_WORDS);
            let w_u32 = warp_id as u32;
            let s_u32 = slices_want as u32;

            // Prime DEPTH = 2 sequential 64-col chunks per warp at cycle 0 before waiting on `meta0`.
            const DEPTH: u32 = 2;
            const _: () = assert!(DEPTH == 2);
            macro_rules! fetch_chunk {
                ($ch:expr) => {{
                    let ch_idx = ($ch).min(chunks_u32 - 1) as usize;
                    let base = chunk_band.add(ch_idx * CHUNK_WORDS);
                    let l = lane as usize;
                    let (a0, a1, a2, a3): (u32, u32, u32, u32);
                    let (b0, b1, b2, b3): (u32, u32, u32, u32);
                    let (e0, e1, eh): (u32, u32, u32);
                    asm!("{{ .reg .b64 h; createpolicy.fractional.L2::evict_first.b64 h, 1.0; ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}}, [{p}], h; }}",
                         a = out(reg32) a0, b = out(reg32) a1, c = out(reg32) a2, d = out(reg32) a3,
                         p = in(reg64) base.add(l * 4), options(readonly, nostack));
                    asm!("{{ .reg .b64 h; createpolicy.fractional.L2::evict_first.b64 h, 1.0; ld.global.nc.L2::cache_hint.L2::256B.v4.u32 {{{a},{b},{c},{d}}}, [{p}], h; }}",
                         a = out(reg32) b0, b = out(reg32) b1, c = out(reg32) b2, d = out(reg32) b3,
                         p = in(reg64) base.add(CHUNK_SM1_OFF + l * 4), options(readonly, nostack));
                    asm!("{{ .reg .b64 h; createpolicy.fractional.L2::evict_first.b64 h, 1.0; ld.global.nc.L2::cache_hint.L2::128B.v2.u32 {{{a},{b}}}, [{p}], h; }}",
                         a = out(reg32) e0, b = out(reg32) e1,
                         p = in(reg64) base.add(CHUNK_E3_LO_OFF + l * 2), options(readonly, nostack));
                    asm!("{{ .reg .b64 h; createpolicy.fractional.L2::evict_first.b64 h, 1.0; ld.global.nc.L2::cache_hint.L2::128B.u32 {a}, [{p}], h; }}",
                         a = out(reg32) eh,
                         p = in(reg64) base.add(CHUNK_E3_HI_OFF + l), options(readonly, nostack));
                    ([a0, a1, a2, a3], [b0, b1, b2, b3], [e0, e1], eh)
                }};
            }

            // Fire initial weight chunk fetches `r0, r1` and per-band metadata in parallel at cycle 0
            // before any dependent escape-pool lookup (`esc_cur`, `esc_nxt`).
            let mut r0 = fetch_chunk!(if coop { w_u32 } else { 0u32 });
            let mut r1 = fetch_chunk!(if coop { w_u32 + s_u32 } else { 1u32 });
            let meta_ptr = panels_base.add(meta_off + abs_band * BAND_META_WORDS);
            let meta0 = ld_global_nc_v4_u32(meta_ptr);
            let q_stream = if coop { warp_id } else { 0 };
            let meta_rel_w = if q_stream >= 3 {
                ld_global_nc_u32(meta_ptr.add(4 + ((q_stream - 3) >> 1)))
            } else {
                0u32
            };

            // Stage `part_chunks` chunks of BF16 `X` starting at global chunk `start_ch` packed for 16-byte `load4`
            // per 2-tile pair: `row_dst + kp * 64 + s * 16` holds `[b0_k0, b1_k0, b0_k1, b1_k1]`.
            macro_rules! stage_x_body {
                ($start_ch:expr) => {{
                    let sx_ch: u32 = $start_ch;
                    let sx_ba = words.add(ONE_PAIR_WORD + 5).read_volatile() as u32;
                    let sx_bab = words.add(ONE_TRIPLE_WORD + 5).read_volatile() as u32;
                    let (sx_bidx, sx_tid, sx_step): (u32, u32, u32);
                    asm!("mov.u32 {b}, %ctaid.x; mov.u32 {t}, %tid.x; mov.u32 {n}, %ntid.x;",
                         b = out(reg32) sx_bidx, t = out(reg32) sx_tid, n = out(reg32) sx_step,
                         options(nomem, nostack));
                    let sx_input = if sx_bab != 0 && sx_bidx >= sx_bab {
                        words.add(ONE_TRIPLE_WORD + 2).read_volatile() as *const f32
                    } else if sx_ba != 0 && sx_bidx >= sx_ba {
                        words.add(ONE_PAIR_WORD + 2).read_volatile() as *const f32
                    } else {
                        words.add(1).read_volatile() as *const f32
                    };
                    let sx_cols = words.add(4).read_volatile() as u32;
                    let sx_inputs = words.add(5).read_volatile() as u32;
                    let sx_pc = words.add(ONE_PART_WORD).read_volatile() as u32;
                    let sx_rb = sx_pc * 128 + if sx_inputs > 1 { 64 } else { 0 };
                    let sx_take = sx_pc.min((sx_cols >> 6) - sx_ch);
                    let sx_items = sx_take * 16;
                    let mut t = 0u32;
                    while t < sx_inputs {
                        let src_row = sx_input.add((t * sx_cols + sx_ch * 64) as usize);
                        let dst_row = smem + t * sx_rb;
                        let mut vec4 = sx_tid;
                        while vec4 < sx_items {
                            let v = ld_global_nc_v4(src_row.add((vec4 * 4) as usize));
                            let p0 = cvt_bf16x2(v[0], v[1]);
                            let p1 = cvt_bf16x2(v[2], v[3]);
                            let dst = dst_row
                                + (vec4 >> 3) * 64
                                + ((vec4 & 1) << 5)
                                + (((vec4 >> 2) & 1) << 3)
                                + (((vec4 >> 1) & 1) << 2);
                            store_u32(dst, p0);
                            store_u32(dst + 16, p1);
                            asm!("", options(nomem, nostack));
                            vec4 += sx_step;
                        }
                        t += 1;
                    }
                    (sx_pc, sx_step)
                }};
            }
            macro_rules! stage_x {
                ($start_ch:expr) => {{
                    let (sx_pc, sx_step) = stage_x_body!($start_ch);
                    if sx_step == 32 {
                        asm!("bar.warp.sync 0xffffffff;");
                    } else {
                        core::arch::nvptx::_syncthreads();
                    }
                    sx_pc
                }};
            }

            if coop8_dbl {
                // Hide `meta0`/`r0`/`r1` DRAM latency inside initial L2-hot `X` staging before reading `esc_ptr`.
                asm!("griddepcontrol.wait;", options(nostack));
                let _ = stage_x_body!(0u32);
            }

            let (lut0, lut1, esc_meta) = (meta0[0], meta0[1], meta0[2]);
            let is_raw_exp = (esc_meta & 1) != 0;
            let start_rel = if q_stream == 0 {
                0usize
            } else if q_stream == 1 {
                (meta0[3] & 0xFFFF) as usize
            } else if q_stream == 2 {
                (meta0[3] >> 16) as usize
            } else {
                ((meta_rel_w >> (((q_stream - 3) & 1) * 16)) & 0xFFFF) as usize
            };
            let mut esc_ptr = panels_base.add(esc_off + (((esc_meta >> 1) as usize) + start_rel) * ESC_STEP_WORDS + lane as usize);
            // Prime 2 x 4-byte escape words (8 escapes = ~400 weights ahead) before `griddepcontrol.wait`.
            let mut esc_cur = ld_global_nc_u32(esc_ptr);
            esc_ptr = esc_ptr.add(ESC_STEP_WORDS);
            let mut esc_nxt = ld_global_nc_u32(esc_ptr);
            esc_ptr = esc_ptr.add(ESC_STEP_WORDS);
            let mut esc_rem = 4u32;

            macro_rules! lop3 {
                ($a:expr, $b:expr, $c:expr, $imm:literal) => {{
                    let d: u32;
                    asm!("lop3.b32 {d}, {a}, {b}, {c}, {imm};",
                         d = out(reg32) d, a = in(reg32) $a, b = in(reg32) $b, c = in(reg32) $c,
                         imm = const $imm, options(pure, nomem, nostack));
                    d
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
            macro_rules! unpack_bf16 {
                ($sm4:expr, $er4:expr) => {{
                    let (sm4, er4): (u32, u32) = ($sm4, $er4);
                    let lo4 = lop3!(er4, sm4, 0x8080_8080u32, 0xE4);
                    let hi4 = lop3!(sm4, er4, 0x8080_8080u32, 0xE4);
                    (bytes!(lo4, hi4, 0x5140), bytes!(lo4, hi4, 0x7362))
                }};
            }
            macro_rules! patch_half {
                ($er4:ident, $esc:expr) => {{
                    let mut m: u32 = $esc;
                    while m != 0 {
                        let shift = (m.trailing_zeros() - 3) << 1;
                        let b = esc_cur & 0xFF;
                        $er4 = ($er4 & !(0xFFu32 << shift)) | (b << shift);
                        esc_cur >>= 8;
                        esc_rem -= 1;
                        let refill = esc_rem == 0;
                        if refill { esc_rem = 4; }
                        let rf = u32::from(refill);
                        asm!(
                            "{{ .reg .pred q; setp.ne.u32 q, {f}, 0; @q mov.u32 {cur}, {nxt}; @q ld.global.nc.u32 {nxt}, [{p}]; }}",
                            f = in(reg32) rf, cur = inout(reg32) esc_cur, nxt = inout(reg32) esc_nxt, p = in(reg64) esc_ptr,
                            options(readonly, nostack)
                        );
                        if refill { esc_ptr = esc_ptr.add(ESC_STEP_WORDS); }
                        m &= m - 1;
                    }
                }};
            }
            macro_rules! decode_tile {
                ($sm_lo:expr, $sm_hi:expr, $lo0:expr, $hi0:expr, $lo1:expr, $hi1:expr) => {{
                    let sel0 = lop3!(($lo0) & 0x3333u32, $hi0, 0x4444u32, 0xF8);
                    let sel1 = lop3!(($lo1) & 0x3333u32, $hi1, 0x4444u32, 0xF8);
                    let mut er0: u32;
                    let mut er1: u32;
                    asm!("prmt.b32 {d}, {a}, {b}, {s};",
                         d = out(reg32) er0, a = in(reg32) lut0, b = in(reg32) lut1, s = in(reg32) sel0,
                         options(pure, nomem, nostack));
                    asm!("prmt.b32 {d}, {a}, {b}, {s};",
                         d = out(reg32) er1, a = in(reg32) lut0, b = in(reg32) lut1, s = in(reg32) sel1,
                         options(pure, nomem, nostack));
                    let esc0 = sel0.wrapping_add(0x1111u32) & 0x8888u32;
                    let esc1 = sel1.wrapping_add(0x1111u32) & 0x8888u32;
                    if (esc0 | esc1) != 0 {
                        if esc0 != 0 { patch_half!(er0, esc0); }
                        if esc1 != 0 { patch_half!(er1, esc1); }
                    }
                    let (a0, a1) = unpack_bf16!($sm_lo, er0);
                    let (a2, a3) = unpack_bf16!($sm_hi, er1);
                    [a0, a1, a2, a3]
                }};
            }
            macro_rules! store_v4_shared {
                ($addr:expr, $v:expr) => {{
                    let v: [u32; 4] = $v;
                    asm!("st.shared.v4.u32 [{p}], {{{a},{b},{c},{d}}};",
                         p = in(reg32) $addr, a = in(reg32) v[0], b = in(reg32) v[1], c = in(reg32) v[2], d = in(reg32) v[3],
                         options(nostack));
                }};
            }
            macro_rules! decode_chunk_to_smem {
                ($slot:expr, $dst_base:expr) => {{
                    let (sm0, sm1, e3_lo, w2) = $slot;
                    let (w0, w1) = (e3_lo[0], e3_lo[1]);
                    let dst: u32 = $dst_base + lane * 16;
                    let a = decode_tile!(sm0[0], sm0[1], w0, w2, w0 >> 2, w2 >> 1);
                    store_v4_shared!(dst, a);
                    let b = decode_tile!(sm0[2], sm0[3], w0 >> 16, w2 << 2, w0 >> 18, w2 << 1);
                    store_v4_shared!(dst + 512, b);
                    let c = decode_tile!(sm1[0], sm1[1], w1, w2 >> 16, w1 >> 2, w2 >> 17);
                    store_v4_shared!(dst + 1024, c);
                    let d = decode_tile!(sm1[2], sm1[3], w1 >> 16, w2 >> 14, w1 >> 18, w2 >> 15);
                    store_v4_shared!(dst + 1536, d);
                }};
            }

            if coop8_dbl {
                // `X` is already staged above; Warps 1..7 decode slot 0 and sync both `X` and slot 0 in ONE barrier.
                if !is_raw_exp && warp_id > 0 {
                    decode_chunk_to_smem!(r0, w_smem_base + (w_u32 - 1) * 2048);
                    if w_u32 + 2 * s_u32 < chunks_u32 {
                        r0 = fetch_chunk!(w_u32 + 2 * s_u32);
                    }
                }
                core::arch::nvptx::_syncthreads();
            } else {
                // In PDL cooperative modes (`mode == 2`), Warps 1..slices decode slot 0 BEFORE `griddepcontrol.wait`.
                if coop && !is_raw_exp && warp_id > 0 {
                    decode_chunk_to_smem!(r0, w_smem_base + (w_u32 - 1) * 2048);
                    if w_u32 + 2 * s_u32 < chunks_u32 {
                        r0 = fetch_chunk!(w_u32 + 2 * s_u32);
                    }
                }
                asm!("griddepcontrol.wait;", options(nostack));
                let _ = stage_x!(0u32);
            }

            let mut ch = 0u32;
            let mut part_end = part_chunks_u32.min(chunks_u32);
            let mut acc = [0.0f32; 4];

            macro_rules! consume_chunk {
                ($slot:expr, $x_addr:expr) => {{
                    let (sm0, sm1, e3_lo, w2) = $slot;
                    let (w0, w1) = (e3_lo[0], e3_lo[1]);
                    let x_a: u32 = $x_addr;
                    let bx0 = crate::ptx::load4(x_a);
                    let a = decode_tile!(sm0[0], sm0[1], w0, w2, w0 >> 2, w2 >> 1);
                    acc = mma_m16n8k16_bf16(a, [bx0[0], bx0[1]], acc);
                    let b = decode_tile!(sm0[2], sm0[3], w0 >> 16, w2 << 2, w0 >> 18, w2 << 1);
                    acc = mma_m16n8k16_bf16(b, [bx0[2], bx0[3]], acc);
                    let bx1 = crate::ptx::load4(x_a + 64);
                    let c = decode_tile!(sm1[0], sm1[1], w1, w2 >> 16, w1 >> 2, w2 >> 17);
                    acc = mma_m16n8k16_bf16(c, [bx1[0], bx1[1]], acc);
                    let d = decode_tile!(sm1[2], sm1[3], w1 >> 16, w2 >> 14, w1 >> 18, w2 >> 15);
                    acc = mma_m16n8k16_bf16(d, [bx1[2], bx1[3]], acc);
                }};
            }
            macro_rules! consume_smem_chunk {
                ($src_base:expr, $x_addr:expr) => {{
                    let sb: u32 = $src_base + lane * 16;
                    let x_a: u32 = $x_addr;
                    let bx0 = crate::ptx::load4(x_a);
                    let a = crate::ptx::load4(sb);
                    acc = mma_m16n8k16_bf16(a, [bx0[0], bx0[1]], acc);
                    let b = crate::ptx::load4(sb + 512);
                    acc = mma_m16n8k16_bf16(b, [bx0[2], bx0[3]], acc);
                    let bx1 = crate::ptx::load4(x_a + 64);
                    let c = crate::ptx::load4(sb + 1024);
                    acc = mma_m16n8k16_bf16(c, [bx1[0], bx1[1]], acc);
                    let d = crate::ptx::load4(sb + 1536);
                    acc = mma_m16n8k16_bf16(d, [bx1[2], bx1[3]], acc);
                }};
            }

            // Unified part loop: every warp in the CTA converges on `_syncthreads()` and `stage_x!(ch)`.
            loop {
                if is_raw_exp {
                    if !coop || warp_id == 0 {
                        let rx_mode = words.add(7).read_volatile() as u32;
                        let rx_ba = words.add(ONE_PAIR_WORD + 5).read_volatile() as usize;
                        let rx_bab = words.add(ONE_TRIPLE_WORD + 5).read_volatile() as usize;
                        let rx_bidx = core::arch::nvptx::_block_idx_x() as usize;
                        let rx_use_c = rx_bab != 0 && rx_bidx >= rx_bab;
                        let rx_use_b = !rx_use_c && rx_ba != 0 && rx_bidx >= rx_ba;
                        let (rx_rows, rx_pbase, rx_fb, rx_bi) = if rx_use_c {
                            (
                                words.add(ONE_TRIPLE_WORD + 1).read_volatile() as usize,
                                words.add(ONE_TRIPLE_WORD + 3).read_volatile() as *const u32,
                                words.add(ONE_TRIPLE_WORD + 4).read_volatile() as usize,
                                rx_bidx - rx_bab,
                            )
                        } else if rx_use_b {
                            (
                                words.add(ONE_PAIR_WORD + 1).read_volatile() as usize,
                                words.add(ONE_PAIR_WORD + 3).read_volatile() as *const u32,
                                words.add(ONE_PAIR_WORD + 4).read_volatile() as usize,
                                rx_bidx - rx_ba,
                            )
                        } else {
                            (
                                words.add(3).read_volatile() as usize,
                                words.add(ONE_PANEL_WORD).read_volatile() as *const u32,
                                words.add(ONE_PANEL_WORD + 1).read_volatile() as usize,
                                rx_bidx,
                            )
                        };
                        let rx_w = (core::arch::nvptx::_thread_idx_x() as usize) / 32;
                        let rx_wpb = (core::arch::nvptx::_block_dim_x() as usize) / 32;
                        let rx_lb = if rx_mode >= 2 { rx_bi } else { rx_bi * rx_wpb + rx_w };
                        let rx_abs = rx_fb + rx_lb.min(rx_rows.div_ceil(16) - 1);
                        let rx_h = ld_global_nc_v4_u32(rx_pbase);
                        let rx_em = ld_global_nc_u32(rx_pbase.add(rx_h[0] as usize + rx_abs * BAND_META_WORDS + 2));
                        let esc_raw_base = rx_pbase.add(rx_h[2] as usize + ((rx_em >> 1) as usize) * ESC_STEP_WORDS);
                        let l = lane as usize;
                        while ch < part_end {
                            let base = chunk_band.add((ch as usize) * CHUNK_WORDS);
                            let rbase = esc_raw_base.add((ch as usize) * RAW_EXP_CHUNK_WORDS);
                            let sm0 = ld_global_nc_v4_u32(base.add(l * 4));
                            let sm1 = ld_global_nc_v4_u32(base.add(CHUNK_SM1_OFF + l * 4));
                            let er0 = ld_global_nc_v4_u32(rbase.add(l * 4));
                            let er1 = ld_global_nc_v4_u32(rbase.add(128 + l * 4));
                            let bx0 = crate::ptx::load4(x_cur);
                            let bx1 = crate::ptx::load4(x_cur + 64);
                            let (a0, a1) = unpack_bf16!(sm0[0], er0[0]);
                            let (a2, a3) = unpack_bf16!(sm0[1], er0[1]);
                            acc = mma_m16n8k16_bf16([a0, a1, a2, a3], [bx0[0], bx0[1]], acc);
                            let (b0, b1) = unpack_bf16!(sm0[2], er0[2]);
                            let (b2, b3) = unpack_bf16!(sm0[3], er0[3]);
                            acc = mma_m16n8k16_bf16([b0, b1, b2, b3], [bx0[2], bx0[3]], acc);
                            let (c0, c1) = unpack_bf16!(sm1[0], er1[0]);
                            let (c2, c3) = unpack_bf16!(sm1[1], er1[1]);
                            acc = mma_m16n8k16_bf16([c0, c1, c2, c3], [bx1[0], bx1[1]], acc);
                            let (d0, d1) = unpack_bf16!(sm1[2], er1[2]);
                            let (d2, d3) = unpack_bf16!(sm1[3], er1[3]);
                            acc = mma_m16n8k16_bf16([d0, d1, d2, d3], [bx1[2], bx1[3]], acc);
                            ch += 1;
                            x_cur += 128;
                        }
                    } else {
                        ch = part_end;
                    }
                } else if coop8_dbl {
                    // 8-warp DOUBLE-BUFFERED cooperative ascending-K (`mode == 4`, `28,672 B` W_SMEM, 3 blk/SM =
                    // 24 warps/SM for `M = 1..=4`): Warps 1..7 decode `nxt_slot` while Warp 0 runs 32 HMMAs on
                    // `cur_slot` (1 barrier per 8 chunks, zero serialization, zero tail over-fetch, 100% bit-exact!).
                    macro_rules! coop8_step {
                        ($cur_slot:literal, $nxt_slot:literal, $w0_reg:ident, $p_reg:ident) => {{
                            if ch + 8 >= chunks_u32 {
                                asm!("griddepcontrol.launch_dependents;", options(nostack));
                            }
                            if warp_id == 0 {
                                consume_chunk!($w0_reg, x_cur);
                                if ch + 16 < chunks_u32 {
                                    $w0_reg = fetch_chunk!(ch + 16);
                                }
                                let sb = w_smem_base + ($cur_slot * 14_336) as u32;
                                let mut p = 0u32;
                                while p < 7 {
                                    consume_smem_chunk!(sb + p * 2048, x_cur + (p + 1) * 128);
                                    p += 1;
                                }
                                x_cur += 1024;
                            } else if ch + 8 < chunks_u32 {
                                let db = w_smem_base + ($nxt_slot * 14_336) as u32 + (w_u32 - 1) * 2048;
                                decode_chunk_to_smem!($p_reg, db);
                                if ch + 24 + w_u32 < chunks_u32 {
                                    $p_reg = fetch_chunk!(ch + 24 + w_u32);
                                }
                            }
                            if ch + 8 < chunks_u32 {
                                core::arch::nvptx::_syncthreads();
                            }
                            ch += 8;
                        }};
                    }
                    if (ch & 8) != 0 && ch < part_end {
                        coop8_step!(1, 0, r1, r0);
                    }
                    while ch + 16 <= part_end {
                        coop8_step!(0, 1, r0, r1);
                        coop8_step!(1, 0, r1, r0);
                    }
                    if ch < part_end {
                        coop8_step!(0, 1, r0, r1);
                    }
                } else if coop8_sgl {
                    // 8-warp single-buffered cooperative ascending-K (`mode == 3`, `14,336 B` W_SMEM, 3 blk/SM for M=5..8).
                    while ch < part_end {
                        if ch + 8 >= chunks_u32 {
                            asm!("griddepcontrol.launch_dependents;", options(nostack));
                        }
                        if warp_id == 0 {
                            consume_chunk!(r0, x_cur);
                            r0 = r1;
                            if ch + 16 < chunks_u32 {
                                r1 = fetch_chunk!(ch + 16);
                            }
                            let mut p = 0u32;
                            while p < 7 {
                                consume_smem_chunk!(w_smem_base + p * 2048, x_cur + (p + 1) * 128);
                                p += 1;
                            }
                            x_cur += 1024;
                        }
                        core::arch::nvptx::_syncthreads();
                        if warp_id > 0 && ch + 8 < chunks_u32 {
                            decode_chunk_to_smem!(r1, w_smem_base + (w_u32 - 1) * 2048);
                            r1 = r0;
                            if ch + 24 + w_u32 < chunks_u32 {
                                r0 = fetch_chunk!(ch + 24 + w_u32);
                            }
                        }
                        ch += 8;
                        if ch < chunks_u32 {
                            core::arch::nvptx::_syncthreads();
                        }
                    }
                } else if coop {
                    // 4-warp cooperative ascending-K: Warps 1..3 decode chunks `ch+4+w` into `W_SMEM[nxt_slot]`
                    // while Warp 0 runs all 4 chunks `ch+0..4` in strictly ascending K order (100% bit-exact!).
                    macro_rules! coop_step {
                        ($cur_slot:literal, $nxt_slot:literal, $w0_reg:ident, $p_reg:ident) => {{
                            if warp_id == 0 {
                                consume_chunk!($w0_reg, x_cur);
                                if ch + 8 < chunks_u32 {
                                    $w0_reg = fetch_chunk!(ch + 8);
                                }
                                let sb = w_smem_base + ($cur_slot * 6144) as u32;
                                consume_smem_chunk!(sb, x_cur + 128);
                                consume_smem_chunk!(sb + 2048, x_cur + 256);
                                consume_smem_chunk!(sb + 4096, x_cur + 384);
                                x_cur += 512;
                            } else if ch + 4 < chunks_u32 {
                                let db = w_smem_base + ($nxt_slot * 6144) as u32 + (w_u32 - 1) * 2048;
                                decode_chunk_to_smem!($p_reg, db);
                                if ch + 12 + w_u32 < chunks_u32 {
                                    $p_reg = fetch_chunk!(ch + 12 + w_u32);
                                }
                            }
                            if ch + 4 < chunks_u32 {
                                core::arch::nvptx::_syncthreads();
                            }
                            ch += 4;
                        }};
                    }
                    if (ch & 4) != 0 && ch < part_end {
                        coop_step!(1, 0, r1, r0);
                    }
                    while ch + 8 <= part_end {
                        coop_step!(0, 1, r0, r1);
                        coop_step!(1, 0, r1, r0);
                    }
                    if ch < part_end {
                        coop_step!(0, 1, r0, r1);
                    }
                } else {
                    while ch + DEPTH < part_end {
                        consume_chunk!(r0, x_cur);
                        r0 = fetch_chunk!(ch + DEPTH);
                        consume_chunk!(r1, x_cur + 128);
                        r1 = fetch_chunk!(ch + 1 + DEPTH);
                        ch += DEPTH;
                        x_cur += 256;
                    }
                    if ch + 1 < part_end {
                        consume_chunk!(r0, x_cur);
                        consume_chunk!(r1, x_cur + 128);
                        ch += 2;
                        x_cur += 256;
                    } else if ch < part_end {
                        consume_chunk!(r0, x_cur);
                        ch += 1;
                        x_cur += 128;
                    }
                }
                if ch >= chunks_u32 { break; }
                if !coop {
                    let ntid_u: u32;
                    asm!("mov.u32 {n}, %ntid.x;", n = out(reg32) ntid_u, options(nomem, nostack));
                    if ntid_u == 32 {
                        asm!("bar.warp.sync 0xffffffff;");
                    } else {
                        core::arch::nvptx::_syncthreads();
                    }
                } else if is_raw_exp {
                    core::arch::nvptx::_syncthreads();
                }
                let pc = stage_x!(ch);
                part_end = (ch + pc).min(chunks_u32);
                if !coop || warp_id == 0 {
                    x_cur -= pc * 128;
                }
                if !coop && !is_raw_exp {
                    r0 = fetch_chunk!(ch);
                    r1 = fetch_chunk!(ch + 1);
                }
            }
            let _ = (esc_ptr, esc_cur, esc_nxt, esc_rem);

            // Reload epilogue metadata from `words` via read_volatile and non-pure SReg moves so prologue
            // `tid`/`block_idx`/`threads` are never kept live across the hot loop.
            let (ep_tid_u, ep_bidx_u, ep_ntid_u): (u32, u32, u32);
            asm!("mov.u32 {t}, %tid.x; mov.u32 {b}, %ctaid.x; mov.u32 {n}, %ntid.x;",
                 t = out(reg32) ep_tid_u, b = out(reg32) ep_bidx_u, n = out(reg32) ep_ntid_u,
                 options(nomem, nostack));
            let ep_tid = ep_tid_u as usize;
            let ep_warp = ep_tid / 32;
            let ep_lane = ep_tid & 31;
            let (ep_g, ep_s) = (ep_lane >> 2, ep_lane & 3);
            let ep_mode = words.add(7).read_volatile();
            let ep_coop = ep_mode >= 2;
            if ep_coop && ep_warp != 0 { return; }
            let ep_inputs = words.add(5).read_volatile() as usize;
            let ep_bidx = ep_bidx_u as usize;
            let ep_bands_a = words.add(ONE_PAIR_WORD + 5).read_volatile() as usize;
            let ep_bands_ab = words.add(ONE_TRIPLE_WORD + 5).read_volatile() as usize;
            let ep_use_c = ep_bands_ab != 0 && ep_bidx >= ep_bands_ab;
            let ep_use_b = !ep_use_c && ep_bands_a != 0 && ep_bidx >= ep_bands_a;
            let (ep_out, ep_rows, ep_stride, ep_band) = if ep_use_c {
                let cr = words.add(ONE_TRIPLE_WORD + 1).read_volatile() as usize;
                (words.add(ONE_TRIPLE_WORD).read_volatile() as *mut f32, cr, cr, ep_bidx - ep_bands_ab)
            } else if ep_use_b {
                let br = words.add(ONE_PAIR_WORD + 1).read_volatile() as usize;
                (words.add(ONE_PAIR_WORD).read_volatile() as *mut f32, br, br, ep_bidx - ep_bands_a)
            } else {
                let w_per_b = (ep_ntid_u as usize) / 32;
                let lb = if ep_coop { ep_bidx } else { ep_bidx * w_per_b + ep_warp };
                (
                    words.add(2).read_volatile() as *mut f32,
                    words.add(3).read_volatile() as usize,
                    words.add(6).read_volatile() as usize,
                    lb,
                )
            };
            let ep_split_count = if !ep_use_b && !ep_use_c && (words.add(ONE_FLAGS_WORD).read_volatile() & FEW_SPLIT_OUT != 0) {
                words.add(ONE_FLAGS_WORD + 1).read_volatile() as usize
            } else {
                0usize
            };

            macro_rules! put {
                ($t:expr, $r:expr, $v:expr) => {{
                    let (t, r, v): (usize, usize, f32) = ($t, $r, $v);
                    if t < ep_inputs && r < ep_rows {
                        if ep_split_count == 0 {
                            *ep_out.add(t * ep_stride + r) = v;
                        } else {
                            let (mut first, mut si) = (0usize, 0usize);
                            while si < ep_split_count {
                                let width = words.add(ONE_SPLIT_WORD + 2 * si + 1).read_volatile() as usize;
                                if r < first + width {
                                    let sp = words.add(ONE_SPLIT_WORD + 2 * si).read_volatile() as *mut f32;
                                    *sp.add(t * width + (r - first)) = v;
                                    break;
                                }
                                first += width;
                                si += 1;
                            }
                        }
                    }
                }};
            }

            let (row, t) = (ep_band * 16 + ep_g, 2 * ep_s);
            put!(t, row, acc[0]);
            put!(t + 1, row, acc[1]);
            put!(t, row + 8, acc[2]);
            put!(t + 1, row + 8, acc[3]);
            return;
        }
        if mode == 5 {
            // Fast 2-warp single-row `cols == 64` BF16 projection (`GateForget+GateGain` 4608x64, 1..=3 batched
            // matrices across `grid.y`, 64 threads, 4,224 B dynamic SMEM = 12+ blk/SM, PDL off).
            if 4_224 > shared_bytes as usize { core::arch::nvptx::trap(); }
            let count = *words.add(8) as usize;
            let gy = core::arch::nvptx::_block_idx_y() as usize;
            if inputs != 1 || cols != 64 || rows < 32 || rows % 32 != 0 || stride != rows || threads != 64
                || count == 0 || count > 3 || gy >= count
                || core::arch::nvptx::_grid_dim_x() as usize != rows / 32
                || core::arch::nvptx::_grid_dim_y() as usize != count
            {
                core::arch::nvptx::trap();
            }
            let (gw, gx, go) = if gy == 0 {
                (weights, input, output)
            } else {
                let base = 10 + (gy - 1) * 3;
                (*words.add(base) as *const u16, *words.add(base + 1) as *const f32, *words.add(base + 2) as *mut f32)
            };
            if (gw as usize | gx as usize) & 15 != 0 || (go as usize) & 3 != 0 || go.is_null() {
                core::arch::nvptx::trap();
            }
            let warp_id = tid / 32;
            let lane = tid & 31;
            let first_row = core::arch::nvptx::_block_idx_x() as usize * 32 + warp_id * 16;
            let w_warp = smem + (warp_id * 2_048) as u32;
            let r_base = lane >> 3;
            let w_chunk = lane & 7;
            let mut i = 0usize;
            while i < 4 {
                let r = i * 4 + r_base;
                let dst = w_warp + swiz128(r, w_chunk);
                crate::ptx::copy16_stream(dst, gw.add((first_row + r) * 64 + w_chunk * 8).cast());
                i += 1;
            }
            crate::ptx::commit();
            asm!("griddepcontrol.wait;", options(nostack));
            if tid < 16 {
                let v = ld_global_nc_v4(gx.add(tid * 4));
                let x_dst = smem + 4_096 + (tid * 8) as u32;
                store_u32(x_dst, cvt_bf16x2(v[0], v[1]));
                store_u32(x_dst + 4, cvt_bf16x2(v[2], v[3]));
            }
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            asm!("griddepcontrol.launch_dependents;", options(nostack));
            let off0 = swiz128(lane & 15, lane >> 4);
            let s = lane & 3;
            let mut dots = [0.0f32; 4];
            macro_rules! rkstep {
                ($k:literal, $xmask:literal) => {{
                    let a = ldmatrix_x4(w_warp + (off0 ^ $xmask));
                    let x = smem + 4_096 + ($k * 32 + s * 4) as u32;
                    dots = mma_m16n8k16_bf16(a, [load_u32(x), load_u32(x + 16)], dots);
                }};
            }
            rkstep!(0, 0u32);
            rkstep!(1, 32u32);
            rkstep!(2, 64u32);
            rkstep!(3, 96u32);
            if s == 0 {
                let r = lane >> 2;
                *go.add(first_row + r) = dots[0];
                *go.add(first_row + r + 8) = dots[2];
            }
            return;
        }
        if mode != 0 {
            if 4_352 > shared_bytes as usize { core::arch::nvptx::trap(); }
            let dim = rows / 2;
            if inputs != 1 || cols != 64 || dim < 16 || dim % 16 != 0 || rows != 2 * dim
                || stride != rows || threads != 64
                || (weights as usize | input as usize) & 15 != 0
                || core::arch::nvptx::_grid_dim_x() as usize != dim / 16
            {
                core::arch::nvptx::trap();
            }
            // SAFETY: attach_gate_blend joins checked adjacent commands (`cols == 64`). Two warps/block (`144 x 64` =
            // 3.00 exact waves on 48 SMs) with 4,352 B dynamic SMEM (4,096 B weights + 128 B BF16 X + 128 B disjoint
            // dot scratch at +4,224 B, 12+ blocks/SM, PDL off): warp 0 reads 16 forget rows, warp 1 reads 16 gain rows,
            // single-stage 16B `cp.async` + 4 `ldmatrix_x4`/`mma` + fused `gate_tanh`.
            let first = core::arch::nvptx::_block_idx_x() as usize * 16;
            let warp_id = tid / 32;
            let lane = tid & 31;
            let first_row = first + if warp_id == 1 { dim } else { 0 };
            let w_warp = smem + (warp_id * 2_048) as u32;
            let r_base = lane >> 3;
            let w_chunk = lane & 7;
            let mut i = 0usize;
            while i < 4 {
                let r = i * 4 + r_base;
                let dst = w_warp + swiz128(r, w_chunk);
                crate::ptx::copy16_stream(dst, weights.add((first_row + r) * 64 + w_chunk * 8).cast());
                i += 1;
            }
            crate::ptx::commit();
            asm!("griddepcontrol.wait;", options(nostack));
            if tid < 16 {
                let v = ld_global_nc_v4(input.add(tid * 4));
                let x_dst = smem + 4_096 + (tid * 8) as u32;
                store_u32(x_dst, cvt_bf16x2(v[0], v[1]));
                store_u32(x_dst + 4, cvt_bf16x2(v[2], v[3]));
            }
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
            let off0 = swiz128(lane & 15, lane >> 4);
            let s = lane & 3;
            let mut dots = [0.0f32; 4];
            macro_rules! gkstep {
                ($k:literal, $xmask:literal) => {{
                    let a = ldmatrix_x4(w_warp + (off0 ^ $xmask));
                    let x = smem + 4_096 + ($k * 32 + s * 4) as u32;
                    dots = mma_m16n8k16_bf16(a, [load_u32(x), load_u32(x + 16)], dots);
                }};
            }
            gkstep!(0, 0u32);
            gkstep!(1, 32u32);
            gkstep!(2, 64u32);
            gkstep!(3, 96u32);
            if s == 0 {
                let r = lane >> 2;
                *output.add(first_row + r) = dots[0];
                *output.add(first_row + r + 8) = dots[2];
                let slot = warp_id * 16 + r;
                store_u32(smem + 4_224 + (slot * 4) as u32, dots[0].to_bits());
                store_u32(smem + 4_224 + ((slot + 8) * 4) as u32, dots[2].to_bits());
            }
            core::arch::nvptx::_syncthreads();
            if tid < 16 {
                let ch = first + tid;
                let forget = crate::stream_pairs::gate_tanh(f32::from_bits(load_u32(smem + 4_224 + (4 * tid) as u32)));
                let gain = crate::stream_pairs::gate_tanh(f32::from_bits(load_u32(smem + 4_224 + (4 * (tid + 16)) as u32)));
                let kept = *(*words.add(7) as *const f32).add(ch);
                let msg = *(*words.add(8) as *const f32).add(ch);
                *(*words.add(9) as *mut f32).add(ch) = (1.0f32 - forget) * kept + gain * msg;
            }
            return;
        }
        if ONE_BLOCK_BYTES > shared_bytes as usize { core::arch::nvptx::trap(); }
        if inputs != 1 || !crate::few_product::few_admitted(rows, cols, inputs, stride, weights as usize, input as usize, threads) { core::arch::nvptx::trap(); }
        let narrow = crate::staged_narrow::shape_admitted(stride, cols);
        // One block streams a 16-row MMA tile. In a partial tile, lanes
        // above the matrix read its final row and suppress their stores.
        if !narrow && cols % 256 == 0 {
            const WEIGHT_BYTES: u32 = 16 * 256 * 2;
            const STAGE_BYTES: u32 = WEIGHT_BYTES + 256 * 2;
            const _: () = assert!(2 * STAGE_BYTES as usize <= ONE_BLOCK_BYTES);
            let first = core::arch::nvptx::_block_idx_x() as usize * 16;
            if first >= rows { return; }
            let lane = tid % 32;
            let mut acc = [0.0; 4];
            let mut stage = 0;
            for chunk in 0..=cols / 256 {
                if chunk < cols / 256 {
                    for group in 0..4 {
                        let row = group * 4 + tid / 32;
                        let col = lane * 8;
                        let dst = stage + (row * 512 + ((lane ^ (row & 7)) * 16)) as u32;
                        crate::ptx::copy16(smem + dst, weights.add((first + row).min(rows - 1) * cols + chunk * 256 + col).cast());
                    }
                    let pair = ld_global_nc_v2_u32(input.add(chunk * 256 + tid * 2).cast());
                    store_u32(smem + stage + WEIGHT_BYTES + (tid * 4) as u32,
                              cvt_bf16x2(f32::from_bits(pair[0]), f32::from_bits(pair[1])));
                    crate::ptx::commit();
                }
                stage ^= STAGE_BYTES;
                if chunk > 0 && tid < 32 {
                    let row = lane % 16;
                    let part = lane / 16;
                    let tile = smem + stage + (row * 512) as u32;
                    for k in 0..16 {
                        let a = ldmatrix_x4(tile + (((2 * k + part) ^ (row & 7)) * 16) as u32);
                        let x = smem + stage + WEIGHT_BYTES + (32 * k + (lane % 4) * 4) as u32;
                        acc = mma_m16n8k16_bf16(a, [load_u32(x), load_u32(x + 16)], acc);
                    }
                }
                cp_async_wait_zero();
                core::arch::nvptx::_syncthreads();
            }
            if tid < 32 && lane % 4 == 0 {
                let row = first + lane / 4;
                if row < rows { *output.add(row) = acc[0]; }
                if row + 8 < rows { *output.add(row + 8) = acc[2]; }
            }
            return;
        }
        let block_first = core::arch::nvptx::_block_idx_x() as usize * if narrow { 4 } else { 64 };
        if block_first < rows {
            let warp_id = tid / 32;
            let lane = tid & 31;
            let (g, s) = (lane >> 2, lane & 3);
            if narrow {
                const _: () = assert!(crate::plain_staged_narrow::BLOCK_BYTES <= ONE_BLOCK_BYTES);
                let c0 = crate::plain_staged_narrow::warp(weights, input, core::ptr::null(), cols, block_first, rows, smem);
                if s < 2 && g == 0 {
                    let (r0, r1) = (block_first + 2 * s, block_first + 2 * s + 1);
                    if r0 < rows { *output.add(r0) = c0[0]; }
                    if r1 < rows { *output.add(r1) = c0[1]; }
                }
            } else {
                let first = block_first + warp_id * 16;
                let (r0, r1) = (first + g, first + g + 8);
                let c0 = warp(weights, input, cols, block_first, rows, smem, 0, cols / BK);
                if s == 0 {
                    if r0 < rows { *output.add(r0) = c0[0]; }
                    if r1 < rows { *output.add(r1) = c0[2]; }
                }
            }
        }
    }
}

/// The `m16n8k16` BF16 single-row wide CTA body (non-256-multiple `cols`, 4 warps own 64 weight rows sharing
/// double-buffered XOR-swizzled BF16 `X` in SMEM) for one Split-K chunk range `[ch_start, ch_end)`.
#[inline(always)]
unsafe fn warp(
    weights: *const u16, input: *const f32, cols: usize, block_first: usize, rows: usize, smem: u32, ch_start: usize, ch_end: usize,
) -> [f32; 4] {
    // SAFETY: Private to this file; called only from `replay` with all threads of the block (`block_first < rows`
    // uniform across the CTA), `few_admitted` true (`cols` whole stages of 64 up to `FEW_COLS_MAX = 16,384`,
    // `rows * cols` inside the envelope, `inputs == 1`, 16-byte aligned weights and input), and `smem` the
    // block's `ONE_BLOCK_BYTES = 20,480` shared buffer. Every thread reaches every `bar.sync 0` and every lane
    // reaches every warp collective. All global reads stay strictly in-bounds and aligned (16B weight chunks for
    // `gr < rows`; 16B input chunks for `tx_row == 0`). Out-of-bounds weight and input rows zero-fill their
    // shared slots. No global memory is written.
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    let warp_id = tid / 32;
    let lane = tid & 31;
    let (g, s) = (lane >> 2, lane & 3);
    let local_chunks = ch_end - ch_start;

    let first_row = block_first + warp_id * 16;
    let r_base = lane >> 3;
    let w_chunk = lane & 7;
    let off0 = swiz128(lane & 15, lane >> 4);
    let x_off0 = swiz128(g, 0) + (s * 4) as u32;

    let tx_row = tid >> 4;
    let tx_col4 = tid & 15;
    let tx_dst = swiz128(tx_row, tx_col4 >> 1) + ((tx_col4 & 1) * 8) as u32;
    let tx_valid0 = tx_row == 0;

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

    macro_rules! fetch_x {($valid:expr, $k_base:expr)=>{{
        if $valid {
            let v = ld_global_nc_v4(input.add($k_base + tx_col4 * 4));
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
    issue_w!(ch_start, smem + w_warp_off);
    crate::ptx::commit();
    let x0_init = fetch_x!(tx_valid0, ch_start * BK);
    store_x!(smem + FEW_INPUT_AT as u32, tx_dst, x0_init);
    cp_async_wait_zero();
    core::arch::nvptx::_syncthreads();

    let mut c0 = [0.0f32; 4];
    let mut c = 0usize;
    while c < local_chunks {
        let gc = ch_start + c;
        let cur = c & 1;
        let nxt = cur ^ 1;
        let has_next = c + 1 < local_chunks;
        let nxt_stage = smem + (nxt * FEW_STAGE_BYTES) as u32;
        if has_next {
            issue_w!(gc + 1, nxt_stage + w_warp_off);
            crate::ptx::commit();
            let pc = (gc + 2) * BK;
            if lane < 16 && c + 2 < local_chunks && first_row + lane < rows {
                asm!("prefetch.global.L2 [{a}];", a = in(reg64) weights.add((first_row + lane) * cols + pc) as u64);
            }
        }
        let x0_nxt = if has_next { fetch_x!(tx_valid0, (gc + 1) * BK) } else { [0u32; 2] };

        let cur_stage = smem + (cur * FEW_STAGE_BYTES) as u32;
        let cur_w = cur_stage + w_warp_off;
        let cur_x = cur_stage + FEW_INPUT_AT as u32;
        macro_rules! kstep_w {($xmask:expr)=>{{
            let a = ldmatrix_x4(cur_w + (off0 ^ $xmask));
            let b0 = [load_u32(cur_x + (x_off0 ^ $xmask)), load_u32(cur_x + (x_off0 ^ ($xmask ^ 16u32)))];
            c0 = mma_m16n8k16_bf16(a, b0, c0);
        }};}
        kstep_w!(0u32);
        kstep_w!(32u32);
        kstep_w!(64u32);
        kstep_w!(96u32);

        if has_next {
            store_x!(nxt_stage + FEW_INPUT_AT as u32, tx_dst, x0_nxt);
            cp_async_wait_zero();
            core::arch::nvptx::_syncthreads();
        }
        c += 1;
    }
    c0
}
