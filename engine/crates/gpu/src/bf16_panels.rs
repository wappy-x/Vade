//! Lossless Vade-BF11-BP (Bit-Parallel) 16 x 16 BF16 weight panels in `m16n8k16` order.
//!
//! Each BF16 weight `w = [sign:1][exp:8][mant:7]` is bijectively split into:
//! - `sm = (sign << 7) | mant` (8 bits): packed for 16-byte `ld.global.nc.v4.u32` loads.
//! - `er = exp.rotate_right(1) = [exp0][exp7..exp1]` (8 bits): encoded with a per-band top-7
//!   table (`lut0`, `lut1`) as a fixed-width 3-bit code (`0..=6` = table lookup, `7` = escape)
//!   with zero serial dependency across weights:
//!   - Each 64-column chunk (4 tiles = 32 weights/lane) is stored sequentially in 128-byte-aligned
//!     blocks of `CHUNK_WORDS = 352` words (`1,408` B = 11 cache lines):
//!     - `sm0[128]` (512 B): tiles `0..2`, `[lane * 4 + 0..4]` (`LDG.E.128`)
//!     - `sm1[128]` (512 B): tiles `2..4`, `[lane * 4 + 0..4]` (`LDG.E.128`)
//!     - `e3_lo[64]` (256 B): 2-bit low codes (`c & 3`) of all 32 weights at 4-bit `prmt` stride (`LDG.E.64`)
//!     - `e3_hi[32]` (128 B): 1-bit high bits (`c >> 2`) of all 32 weights at 4-bit `prmt` stride (`LDG.E.32`)
//!   - Rare escapes (`c == 7`, ~1.5-2.3% of real LLM weights) are stored per band in a coalesced
//!     32-byte-strided `[step][lane]` byte pool (`esc_off`), read only when `(sel + 0x1111) & 0x8888 != 0`.
//!   - Adversarial synthetic bands with `total_steps >= 6 * chunks` fall back to uncompressed `er`
//!     (`is_raw_exp = 1`, `RAW_EXP_CHUNK_WORDS = 256` words/chunk in the escape pool), keeping total
//!     size `<= storage_words(rows, cols)` for every possible BF16 matrix.
//!
//! Reconstruction from `(sm4, er4)` is 100% lossless for all 65,536 BF16 bit patterns:
//! `lo4 = (er4 & 0x8080_8080) | (sm4 & 0x7F7F_7F7F)` (`lop3.b32 0xE4`)
//! `hi4 = (sm4 & 0x8080_8080) | (er4 & 0x7F7F_7F7F)` (`lop3.b32 0xE4`)
//! followed by `prmt.b32` (`0x5140`, `0x7362`).
pub const TILE: usize = 16;
/// Legacy 12-bpw constants retained for untouched `plain_staged_few.rs`.
pub const DATA_WORDS: usize = 96;
pub const RAW: u32 = u32::MAX;
/// Columns per sequential chunk (4 tiles = 32 weights/lane).
pub const CHUNK_COLS: usize = 64;
/// Global header words: `[meta_off, chunks_off, esc_off, chunks_per_band | (slices_enc << 16)]`.
pub const HEADER_WORDS: usize = 4;
/// Per-band metadata words (32 B = `2 x LDG.E.128`):
/// `[lut0, lut1, (esc_step_q0 << 1) | is_raw_exp, rel1 | (rel2 << 16), rel3 | (rel4 << 16), rel5 | (rel6 << 16), rel7, 0]`,
/// where `rel_s` (`s = 1..8`) is the `u16` escape-step offset of slice `s` relative to `esc_step_q0`.
pub const BAND_META_WORDS: usize = 8;
/// Number of interleaved escape streams per band:
/// - `4` for decode-sized matrices (`rows <= 10_000` and `chunks` a multiple of 4), enabling 4-warp
///   cooperative parallel chunk decode with Warp 0 running all MMAs in strictly ascending K order
///   (100% bit-exact with `slices == 1` and `main`).
/// - `1` for huge matrices (`rows > 10_000`, e.g. LM Head, where 4 bands/block already launches >3,000 CTAs)
///   or shapes whose chunk count is not a multiple of 4.
#[inline(always)]
pub const fn slice_count(rows: usize, chunks: usize) -> usize {
    if rows <= 10_000 && chunks >= 8 && chunks.is_multiple_of(4) { 4 } else { 1 }
}
/// Words of one 64-column chunk: `sm0(128) + sm1(128) + e3_lo(64) + e3_hi(32) = 352` words (1,408 B).
pub const CHUNK_WORDS: usize = 352;
/// Word offsets inside one chunk.
pub const CHUNK_SM1_OFF: usize = 128;
pub const CHUNK_E3_LO_OFF: usize = 256;
pub const CHUNK_E3_HI_OFF: usize = 320;
/// Words of one 128-byte escape step (4 escape bytes packed per lane x 32 lanes = 1 cache line).
pub const ESC_STEP_WORDS: usize = 32;
/// Trailing guard steps at the end of the escape pool so double-buffered prefetch is always in bounds.
pub const ESC_GUARD_STEPS: usize = 2;
/// Words of one uncompressed `er` chunk in the escape pool (`is_raw_exp = 1`: 2 x 128 words = 8 steps).
pub const RAW_EXP_CHUNK_WORDS: usize = 256;
/// Payload word in `PLAIN_STAGED_FEW` holding the legacy panel base pointer (left 0 under Vade-BF11-BP).
pub const FEW_PANEL_WORD: usize = 26;

/// Legacy 12-bpw helper retained for untouched `plain_staged_few.rs`.
#[inline(always)]
pub fn high_bytes(codes: u32, base: u32) -> u32 {
    ((codes & 0x0707_0707) + base) | ((codes & 0x0808_0808) << 4)
}

/// Align `words` up to a multiple of `align` (power of two).
#[inline(always)]
pub const fn align_up(words: usize, align: usize) -> usize {
    (words + align - 1) & !(align - 1)
}

/// Word offset of the sequential chunk plane for a matrix with `bands` 16-row bands.
#[inline(always)]
pub const fn chunks_offset_words(bands: usize) -> usize {
    align_up(HEADER_WORDS + bands * BAND_META_WORDS, 32)
}

/// Upper-bound storage words (for admission budgeting before inspecting weight values).
#[cfg(not(target_arch = "nvptx64"))]
pub fn storage_words(rows: usize, cols: usize) -> Option<usize> {
    (rows > 0 && cols.is_multiple_of(CHUNK_COLS) && (CHUNK_COLS..=16_384).contains(&cols)).then_some(())?;
    let bands = rows.div_ceil(TILE);
    let chunks = cols / CHUNK_COLS;
    let chunks_off = chunks_offset_words(bands);
    let chunk_plane = bands.checked_mul(chunks)?.checked_mul(CHUNK_WORDS)?;
    let esc_off = align_up(chunks_off.checked_add(chunk_plane)?, ESC_STEP_WORDS);
    // Worst case: every band falls back to raw_exp (256 words/chunk) + trailing guard steps.
    let max_esc = bands
        .checked_mul(chunks)?
        .checked_mul(RAW_EXP_CHUNK_WORDS)?
        .checked_add(ESC_GUARD_STEPS * ESC_STEP_WORDS)?;
    esc_off.checked_add(max_esc)
}

/// Encode a row-major BF16 matrix (`rows x cols` little-endian `u16` bytes) into Vade-BF11-BP.
/// Returns `(encoded_bytes, total_tiles, raw_exp_bands)`.
#[cfg(not(target_arch = "nvptx64"))]
pub fn encode_with_stats(rows: usize, cols: usize, bytes: &[u8]) -> Result<(Vec<u8>, usize, usize), &'static str> {
    let max_words = storage_words(rows, cols).ok_or("BF16 panels: unsupported shape")?;
    if rows.checked_mul(cols).and_then(|n| n.checked_mul(2)) != Some(bytes.len()) {
        return Err("BF16 panels: source extent differs");
    }
    let bands = rows.div_ceil(TILE);
    let chunks = cols / CHUNK_COLS;
    let workers = std::thread::available_parallelism().map_or(1, usize::from).min(16);
    let bands_per_worker = bands.div_ceil(workers).max(1);

    struct BandOut {
        lut0: u32,
        lut1: u32,
        is_raw: bool,
        esc_q_rel: [u16; 8],
        chunks: Vec<u32>,
        esc: Vec<u32>,
    }

    let mut band_results: Vec<BandOut> = Vec::with_capacity(bands);
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for start_band in (0..bands).step_by(bands_per_worker) {
            let end_band = (start_band + bands_per_worker).min(bands);
            handles.push(scope.spawn(move || {
                let mut out = Vec::with_capacity(end_band - start_band);
                for b in start_band..end_band {
                    let (lut0, lut1, is_raw, esc_q_rel, cwords, ewords) = encode_band(b, rows, cols, bytes, chunks);
                    out.push(BandOut { lut0, lut1, is_raw, esc_q_rel, chunks: cwords, esc: ewords });
                }
                out
            }));
        }
        for h in handles {
            band_results.extend(h.join().unwrap());
        }
    });

    let chunks_off = chunks_offset_words(bands);
    let chunk_band_words = chunks * CHUNK_WORDS;
    let esc_off = align_up(chunks_off + bands * chunk_band_words, ESC_STEP_WORDS);
    let total_esc_words: usize = band_results.iter().map(|b| b.esc.len()).sum();
    let total_words = esc_off + total_esc_words + ESC_GUARD_STEPS * ESC_STEP_WORDS;
    debug_assert!(total_words <= max_words);

    let mut out = vec![0u8; total_words * 4];
    let write_u32 = |buf: &mut [u8], word_idx: usize, val: u32| {
        buf[word_idx * 4..word_idx * 4 + 4].copy_from_slice(&val.to_le_bytes());
    };
    write_u32(&mut out, 0, HEADER_WORDS as u32);
    write_u32(&mut out, 1, u32::try_from(chunks_off).map_err(|_| "BF16 panels: chunks offset overflow")?);
    write_u32(&mut out, 2, u32::try_from(esc_off).map_err(|_| "BF16 panels: esc offset overflow")?);
    let slices_enc = slice_count(rows, chunks);
    write_u32(&mut out, 3, u32::try_from(chunks | (slices_enc << 16)).map_err(|_| "BF16 panels: chunk count overflow")?);

    let mut esc_step_cursor = 0usize;
    let mut raw_bands = 0usize;
    for (b, band) in band_results.into_iter().enumerate() {
        if band.is_raw {
            raw_bands += 1;
        }
        let meta_at = HEADER_WORDS + b * BAND_META_WORDS;
        let q0 = esc_step_cursor + usize::from(band.esc_q_rel[0]);
        let esc_meta = u32::try_from((q0 << 1) | usize::from(band.is_raw))
            .map_err(|_| "BF16 panels: esc offset overflow")?;
        let r = band.esc_q_rel;
        write_u32(&mut out, meta_at, band.lut0);
        write_u32(&mut out, meta_at + 1, band.lut1);
        write_u32(&mut out, meta_at + 2, esc_meta);
        write_u32(&mut out, meta_at + 3, u32::from(r[1]) | (u32::from(r[2]) << 16));
        write_u32(&mut out, meta_at + 4, u32::from(r[3]) | (u32::from(r[4]) << 16));
        write_u32(&mut out, meta_at + 5, u32::from(r[5]) | (u32::from(r[6]) << 16));
        write_u32(&mut out, meta_at + 6, u32::from(r[7]));
        write_u32(&mut out, meta_at + 7, 0);

        let c_dst = (chunks_off + b * chunk_band_words) * 4;
        for (i, &w) in band.chunks.iter().enumerate() {
            out[c_dst + i * 4..c_dst + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        let e_dst = (esc_off + esc_step_cursor * ESC_STEP_WORDS) * 4;
        for (i, &w) in band.esc.iter().enumerate() {
            out[e_dst + i * 4..e_dst + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        debug_assert!(band.esc.len().is_multiple_of(ESC_STEP_WORDS));
        esc_step_cursor += band.esc.len() / ESC_STEP_WORDS;
    }

    Ok((out, bands * (cols / TILE), raw_bands))
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn encode(rows: usize, cols: usize, bytes: &[u8]) -> Result<Vec<u8>, &'static str> {
    Ok(encode_with_stats(rows, cols, bytes)?.0)
}

#[cfg(not(target_arch = "nvptx64"))]
fn encode_band(
    band: usize, rows: usize, cols: usize, bytes: &[u8], chunks: usize,
) -> (u32, u32, bool, [u16; 8], Vec<u32>, Vec<u32>) {
    let base_row = band * TILE;
    let end_row = (base_row + TILE).min(rows);

    // Per-band histogram of rotated exponents `er = [exp0][exp7..exp1]`.
    let mut hist = [0u32; 256];
    for r in base_row..end_row {
        let row_slice = &bytes[r * cols * 2..(r + 1) * cols * 2];
        for &[lo, hi] in row_slice.as_chunks::<2>().0 {
            let er = (lo & 0x80) | (hi & 0x7F);
            hist[usize::from(er)] += 1;
        }
    }
    let mut order: [usize; 256] = core::array::from_fn(|i| i);
    order.sort_by(|&a, &b| hist[b].cmp(&hist[a]).then_with(|| a.cmp(&b)));
    let top7: [u8; 7] = core::array::from_fn(|i| order[i] as u8);
    let pad_er = top7[0];
    let lut0 = u32::from_le_bytes([top7[0], top7[1], top7[2], top7[3]]);
    let lut1 = u32::from_le_bytes([top7[4], top7[5], top7[6], 0]);
    let mut er_slot = [7u8; 256];
    for (s, &er) in top7.iter().enumerate() {
        er_slot[usize::from(er)] = s as u8;
    }

    let read_pair = |r: usize, c: usize| -> (u8, u8) {
        if r >= rows {
            return (0, pad_er);
        }
        let at = (r * cols + c) * 2;
        let lo = bytes[at];
        let hi = bytes[at + 1];
        ((hi & 0x80) | (lo & 0x7F), (lo & 0x80) | (hi & 0x7F))
    };

    let q_count = slice_count(rows, chunks);
    let mut chunk_words = vec![0u32; chunks * CHUNK_WORDS];
    let mut lane_esc_q: [[Vec<u8>; 32]; 8] = core::array::from_fn(|_| core::array::from_fn(|_| Vec::new()));
    let mut raw_er: Vec<[u32; 8]> = Vec::with_capacity(chunks * 32);

    // High-bit offsets inside `w2` for each half `h in 0..8` at 4-bit stride.
    const HI_BASE: [u32; 8] = [2, 3, 0, 1, 18, 19, 16, 17];
    const LO_BASE: [u32; 4] = [0, 2, 16, 18];

    for ch in 0..chunks {
        let q = if q_count > 1 { ch % q_count } else { 0 };
        let ch_base = ch * CHUNK_WORDS;
        for lane in 0..32 {
            let r = base_row + lane / 4;
            let c_lane = 2 * (lane % 4);
            let mut sm0 = [0u32; 4];
            let mut sm1 = [0u32; 4];
            let mut er_all = [0u32; 8];
            let mut w0 = 0u32;
            let mut w1 = 0u32;
            let mut w2 = 0u32;

            for t in 0..4 {
                let col0 = (ch * 4 + t) * TILE + c_lane;
                for half in 0..2 {
                    let h = t * 2 + half;
                    let cs = col0 + 8 * half;
                    let (sm_0, er_0) = read_pair(r, cs);
                    let (sm_1, er_1) = read_pair(r, cs + 1);
                    let (sm_2, er_2) = read_pair(r + 8, cs);
                    let (sm_3, er_3) = read_pair(r + 8, cs + 1);
                    let sm4 = u32::from_le_bytes([sm_0, sm_1, sm_2, sm_3]);
                    let er4 = u32::from_le_bytes([er_0, er_1, er_2, er_3]);
                    if h < 4 {
                        sm0[h] = sm4;
                    } else {
                        sm1[h - 4] = sm4;
                    }
                    er_all[h] = er4;

                    let lo_sh_base = LO_BASE[h & 3];
                    let hi_sh_base = HI_BASE[h];
                    for (j, er) in [er_0, er_1, er_2, er_3].into_iter().enumerate() {
                        let code = u32::from(er_slot[usize::from(er)]);
                        let lo2 = code & 3;
                        let hi1 = (code >> 2) & 1;
                        let j4 = (j as u32) * 4;
                        if h < 4 {
                            w0 |= lo2 << (lo_sh_base + j4);
                        } else {
                            w1 |= lo2 << (lo_sh_base + j4);
                        }
                        w2 |= hi1 << (hi_sh_base + j4);
                        if code == 7 {
                            lane_esc_q[q][lane].push(er);
                        }
                    }
                }
            }

            let sm0_at = ch_base + lane * 4;
            chunk_words[sm0_at..sm0_at + 4].copy_from_slice(&sm0);
            let sm1_at = ch_base + CHUNK_SM1_OFF + lane * 4;
            chunk_words[sm1_at..sm1_at + 4].copy_from_slice(&sm1);
            let lo_at = ch_base + CHUNK_E3_LO_OFF + lane * 2;
            chunk_words[lo_at] = w0;
            chunk_words[lo_at + 1] = w1;
            chunk_words[ch_base + CHUNK_E3_HI_OFF + lane] = w2;
            raw_er.push(er_all);
        }
    }

    let total_steps: usize = (0..q_count)
        .map(|q| lane_esc_q[q].iter().map(Vec::len).max().unwrap_or(0).div_ceil(4))
        .sum();
    // Fall back to raw_exp only on adversarial matrices where packed escapes approach raw_exp or exceed u16 relative offset.
    if total_steps >= chunks * 6 || total_steps > u16::MAX as usize {
        let mut esc_words = vec![0u32; chunks * RAW_EXP_CHUNK_WORDS];
        for ch in 0..chunks {
            let base = ch * RAW_EXP_CHUNK_WORDS;
            for lane in 0..32 {
                let er = raw_er[ch * 32 + lane];
                esc_words[base + lane * 4..base + lane * 4 + 4].copy_from_slice(&er[0..4]);
                esc_words[base + 128 + lane * 4..base + 128 + lane * 4 + 4].copy_from_slice(&er[4..8]);
            }
        }
        return (lut0, lut1, true, [0; 8], chunk_words, esc_words);
    }

    // Store coalesced [word_step][lane] 4-byte-packed escapes per slice so 1/4/6/8 warps can decode independently.
    let mut esc_words = Vec::with_capacity(total_steps * ESC_STEP_WORDS);
    let mut esc_q_rel = [0u16; 8];
    for q in 0..q_count {
        esc_q_rel[q] = (esc_words.len() / ESC_STEP_WORDS) as u16;
        let max_esc = lane_esc_q[q].iter().map(Vec::len).max().unwrap_or(0);
        let steps = max_esc.div_ceil(4);
        let base_len = esc_words.len();
        esc_words.resize(base_len + steps * ESC_STEP_WORDS, 0);
        for (lane, escs) in lane_esc_q[q].iter().enumerate() {
            for (step, group) in escs.chunks(4).enumerate() {
                let mut b = [0u8; 4];
                b[..group.len()].copy_from_slice(group);
                esc_words[base_len + step * ESC_STEP_WORDS + lane] = u32::from_le_bytes(b);
            }
        }
    }
    (lut0, lut1, false, esc_q_rel, chunk_words, esc_words)
}

#[cfg(test)]
fn decode_bf16_pair(sm4: u32, er4: u32) -> [u32; 2] {
    let lo4 = (er4 & 0x8080_8080) | (sm4 & 0x7F7F_7F7F);
    let hi4 = (sm4 & 0x8080_8080) | (er4 & 0x7F7F_7F7F);
    let interleave = |lo: u32, hi: u32| {
        (lo & 255) | ((lo & 0xff00) << 8) | ((hi & 255) << 8) | ((hi & 0xff00) << 16)
    };
    [interleave(lo4, hi4), interleave(lo4 >> 16, hi4 >> 16)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn verify_matrix(rows: usize, cols: usize, original: &[u8]) -> (usize, usize) {
        let (packed, total_tiles, raw_bands) = encode_with_stats(rows, cols, original).unwrap();
        assert!(packed.len() <= storage_words(rows, cols).unwrap() * 4);
        assert_eq!(packed.len() % 4, 0);
        let words: Vec<u32> = packed.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let meta_off = words[0] as usize;
        let chunks_off = words[1] as usize;
        let esc_off = words[2] as usize;
        let chunks = (words[3] & 0xFFFF) as usize;
        let slices_enc = (words[3] >> 16) as usize;
        assert_eq!(meta_off, HEADER_WORDS);
        assert_eq!(chunks, cols / CHUNK_COLS);
        assert_eq!(slices_enc, slice_count(rows, chunks));
        let bands = rows.div_ceil(TILE);
        assert_eq!(total_tiles, bands * (cols / TILE));

        let prmt_lut = |lut0: u32, lut1: u32, sel: u32| -> u32 {
            let bytes_lut = [
                lut0 as u8, (lut0 >> 8) as u8, (lut0 >> 16) as u8, (lut0 >> 24) as u8,
                lut1 as u8, (lut1 >> 8) as u8, (lut1 >> 16) as u8, (lut1 >> 24) as u8,
            ];
            let mut out = [0u8; 4];
            for j in 0..4 {
                out[j] = bytes_lut[((sel >> (j * 4)) & 7) as usize];
            }
            u32::from_le_bytes(out)
        };

        for b in 0..bands {
            let m = meta_off + b * BAND_META_WORDS;
            let (lut0, lut1, esc_meta) = (words[m], words[m + 1], words[m + 2]);
            let eq0 = (esc_meta >> 1) as usize;
            let slice_eq = |s: usize| -> usize {
                let rel = if s == 0 {
                    0usize
                } else if s == 1 {
                    (words[m + 3] & 0xFFFF) as usize
                } else if s == 2 {
                    (words[m + 3] >> 16) as usize
                } else {
                    let w = words[m + 4 + ((s - 3) >> 1)];
                    ((w >> (((s - 3) & 1) * 16)) & 0xFFFF) as usize
                };
                eq0 + rel
            };
            let is_raw = (esc_meta & 1) != 0;
            let band_chunks = chunks_off + b * chunks * CHUNK_WORDS;
            for lane in 0..32 {
                let mut esc_idx = [0usize; 8];
                let mut esc_cur = [0u32; 8];
                let mut esc_nxt = [0u32; 8];
                let mut esc_rem = [4u32; 8];
                for q in 0..slices_enc {
                    let mut idx = esc_off + slice_eq(q) * ESC_STEP_WORDS + lane;
                    esc_cur[q] = words[idx];
                    idx += ESC_STEP_WORDS;
                    esc_nxt[q] = words[idx];
                    idx += ESC_STEP_WORDS;
                    esc_idx[q] = idx;
                }
                for ch in 0..chunks {
                    let q = if slices_enc > 1 { ch % slices_enc } else { 0 };
                    let cbase = band_chunks + ch * CHUNK_WORDS;
                    let sm0 = &words[cbase + lane * 4..cbase + lane * 4 + 4];
                    let sm1 = &words[cbase + CHUNK_SM1_OFF + lane * 4..cbase + CHUNK_SM1_OFF + lane * 4 + 4];
                    let w0 = words[cbase + CHUNK_E3_LO_OFF + lane * 2];
                    let w1 = words[cbase + CHUNK_E3_LO_OFF + lane * 2 + 1];
                    let w2 = words[cbase + CHUNK_E3_HI_OFF + lane];

                    let hi_halves = [
                        w2 & 0x4444,
                        (w2 >> 1) & 0x4444,
                        (w2 << 2) & 0x4444,
                        (w2 << 1) & 0x4444,
                        (w2 >> 16) & 0x4444,
                        (w2 >> 17) & 0x4444,
                        (w2 >> 14) & 0x4444,
                        (w2 >> 15) & 0x4444,
                    ];
                    let lo_halves = [
                        w0 & 0x3333,
                        (w0 >> 2) & 0x3333,
                        (w0 >> 16) & 0x3333,
                        (w0 >> 18) & 0x3333,
                        w1 & 0x3333,
                        (w1 >> 2) & 0x3333,
                        (w1 >> 16) & 0x3333,
                        (w1 >> 18) & 0x3333,
                    ];

                    for h in 0..8 {
                        let sm4 = if h < 4 { sm0[h] } else { sm1[h - 4] };
                        let er4 = if is_raw {
                            let rbase = esc_off + eq0 * ESC_STEP_WORDS + ch * RAW_EXP_CHUNK_WORDS;
                            if h < 4 {
                                words[rbase + lane * 4 + h]
                            } else {
                                words[rbase + 128 + lane * 4 + (h - 4)]
                            }
                        } else {
                            let sel = lo_halves[h] | hi_halves[h];
                            let mut er = prmt_lut(lut0, lut1, sel);
                            let esc = sel.wrapping_add(0x1111) & 0x8888;
                            if esc != 0 {
                                let mut mask = esc;
                                let mut sh = 0u32;
                                while mask != 0 {
                                    if (mask & 8) != 0 {
                                        let byte = esc_cur[q] & 0xFF;
                                        esc_cur[q] >>= 8;
                                        esc_rem[q] -= 1;
                                        if esc_rem[q] == 0 {
                                            esc_cur[q] = esc_nxt[q];
                                            esc_nxt[q] = words[esc_idx[q]];
                                            esc_idx[q] += ESC_STEP_WORDS;
                                            esc_rem[q] = 4;
                                        }
                                        er = (er & !(0xFFu32 << sh)) | (byte << sh);
                                    }
                                    mask >>= 4;
                                    sh += 8;
                                }
                            }
                            er
                        };

                        let t = h / 2;
                        let half = h & 1;
                        let k = ch * 4 + t;
                        let vals = decode_bf16_pair(sm4, er4);
                        for (upper, val) in vals.into_iter().enumerate() {
                            let row = b * TILE + lane / 4 + upper * 8;
                            if row >= rows {
                                continue;
                            }
                            for j in 0..2 {
                                let col = k * TILE + 2 * (lane % 4) + half * 8 + j;
                                let actual = (val >> (16 * j)) as u16;
                                let at = (row * cols + col) * 2;
                                let expected = u16::from_le_bytes(original[at..at + 2].try_into().unwrap());
                                assert_eq!(actual, expected, "row {row}, col {col}, is_raw={is_raw}");
                            }
                        }
                    }
                }
            }
        }
        (packed.len(), raw_bands)
    }

    #[test]
    fn round_trips_every_bf16_pattern_and_outlier_tiles() {
        // 1. All 65,536 BF16 bit patterns (including NaN, Inf, subnormals, +/-0) -> exercises raw_exp fallback too.
        let (rows, cols) = (257, 256);
        let mut original = vec![0u8; rows * cols * 2];
        for r in 0..256 {
            for c in 0..256 {
                let bits = ((r << 8) | c) as u16;
                let at = (r * cols + c) * 2;
                original[at..at + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        original[(256 * cols + 5) * 2 + 1] = 0x7f;
        let (_len, raw_bands) = verify_matrix(rows, cols, &original);
        assert!(raw_bands > 0, "expected raw_exp fallback on uniform 256-exponent test");

        // 2. Decode-sized matrices with chunks % 4 == 0 use 4 interleaved escape streams (strict ascending-K);
        // huge matrices (LM Head) and non-multiple-of-4 chunk counts use 1 stream.
        assert_eq!(slice_count(129, 36), 4);
        assert_eq!(slice_count(1024, 36), 4);
        assert_eq!(slice_count(1056, 36), 4);
        assert_eq!(slice_count(6552, 36), 4);
        assert_eq!(slice_count(8632, 36), 4);
        assert_eq!(slice_count(2304, 48), 4);
        assert_eq!(slice_count(2304, 64), 4);
        assert_eq!(slice_count(33, 1), 1);
        assert_eq!(slice_count(200_058, 36), 1);

        let common = [121u16, 121, 121, 121, 120, 120, 120, 119, 119, 122, 122, 118, 117, 116];
        for (rows, cols) in [(33usize, 64usize), (145, 256), (33, 2304), (513, 2304), (1600, 2304), (513, 3072)] {
            let mut llm = vec![0u8; rows * cols * 2];
            for r in 0..rows {
                for c in 0..cols {
                    let h = r * 257 + c * 131;
                    let exp = if h % 67 == 0 { (h / 67 % 256) as u16 } else { common[h % common.len()] };
                    let sign = ((r + c) & 1) as u16;
                    let mant = ((r * 7 + c * 11) & 0x7F) as u16;
                    let bits = (sign << 15) | (exp << 7) | mant;
                    let at = (r * cols + c) * 2;
                    llm[at..at + 2].copy_from_slice(&bits.to_le_bytes());
                }
            }
            let (len, raw_bands) = verify_matrix(rows, cols, &llm);
            assert_eq!(raw_bands, 0, "realistic exponent distribution should compress all bands ({rows}x{cols})");
            let padded_rows = rows.div_ceil(TILE) * TILE;
            let bpw = (len as f64 * 8.0) / (padded_rows * cols) as f64;
            if cols >= 256 {
                assert!(bpw < 11.7, "expected < 11.7 bits/weight for {rows}x{cols}, got {bpw:.3} ({len} bytes)");
            }
        }
    }
}
