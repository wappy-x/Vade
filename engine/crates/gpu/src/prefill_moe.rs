//! Bit-exact W4A16/W8A16 BF16 tensor-core prefill MoE ABI, admission guards, host checks, and unit tests
//! (`PREFILL_PACKED_TC`, kind 121 in `kernels-inkling`).
//!
//! Pure layout, admission, and host checks shared by path with `kernels-inkling`, keeping shared Qwen/GLM PTX
//! byte-for-byte untouched and avoiding any edits to decode-owned packed MoE kernels.
//!
//! Sub-operations selected by payload word `PREFILL_OP_WORD` (18):
//! - `PREFILL_OP_GEMM = 0`: `PAIR_M = 128` x `BN = 128` grouped MoE GEMM using exact SM121a BF16
//!   `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` in strictly ascending `k = 0, 16, ..., K - 16` order
//!   from `+0.0`, with register-only exact NVFP4/MXFP8 weight unpacking (zero 8 KB SMEM LUT), exact
//!   `FAST_CAPPED_GATE` LUT gating, and bank-free dead-SMEM coalesced `st.global.v4.b32` BF16 epilogues.
//!   Produces bit-for-bit identical results to `PLAIN_PACKED_TC_ROWS`.
//! - `PREFILL_OP_PACK = 1`: vectorized FP32 -> packed BF16 activation packer (`cvt.rn.bf16x2.f32`, bit-exact
//!   with `TC_MODE_PACK`, using `L2::evict_first` on the one-use FP32 read so BF16 rows stay hot in L2).
//! - `PREFILL_OP_FOLD = 2`: vectorized BF16 `RouteFold` (`sum_p rn(down[p]) * weight[p]`, bit-exact with
//!   `fast_route_fold_half_into`).

#[cfg(not(target_arch = "nvptx64"))]
use crate::fast_ops::CAPPED_GATE_TABLE_WORDS;
use crate::packed_product::{HALF_IN, layout};
use crate::tc_rows::{
    MOE_ALL_MAX, MOE_EXPERTS_MAX, MOE_PICKS_MAX, MOE_ROWS_MAX, TC_BM, TC_COLS_MAX, TC_ROWS_MAX,
    TC_THREADS, moe_max_tiles, tc_packed_moe_shape_admitted,
};

/// Payload word index holding the sub-operation selector (`PREFILL_OP_*`).
pub const PREFILL_OP_WORD: usize = 18;
/// Grouped bit-exact BF16 tensor-core prefill MoE GEMM (`PAIR_M = 128`, `BN = 128`).
pub const PREFILL_OP_GEMM: u64 = 0;
/// Vectorized FP32 -> packed BF16 activation packer (bit-exact with `TC_MODE_PACK`).
pub const PREFILL_OP_PACK: u64 = 1;
/// Vectorized BF16 `RouteFold` (bit-exact with `fast_route_fold_half_into`).
pub const PREFILL_OP_FOLD: u64 = 2;

/// GEMM output formats encoded in bits `32..64` of payload word 4.
pub const PREFILL_OUT_FP32: u64 = 0;
/// Exact LUT `CappedGate` -> packed BF16 activation output (`rows / 2` BF16, two per u32) for `GateUp`.
pub const PREFILL_OUT_BF16_GATED: u64 = 1;
/// Ungated packed BF16 output (`rn(out)` two per u32 word) for `Down` -> `RouteFold`.
pub const PREFILL_OUT_BF16: u64 = 2;

/// Input rows per CTA (`2 * TC_BM = 128`) and stored output rows per CTA (`128`).
pub const PREFILL_PAIR_M: usize = 2 * TC_BM;
pub const PREFILL_BN: usize = 128;
pub const PREFILL_THREADS: usize = TC_THREADS;
/// Neighboring route tiles grouped per N-tile to share expert weight lines in L2.
pub const PREFILL_M_GROUP: usize = 8;

/// Input form marker for `PREFILL_OP_GEMM` payload word 17 (always packed BF16 rows, matching `PLAIN_PACKED_TC_ROWS`).
pub const PREFILL_HALF_IN: u64 = HALF_IN;

/// Device-side admission guard for `PREFILL_OP_PACK` (FP32 -> packed BF16, 8 floats per lane).
#[inline(always)]
pub const fn pack_admitted(elements: usize, input: usize, output: usize, threads: usize) -> bool {
    threads == PREFILL_THREADS
        && elements >= 8
        && elements <= MOE_ALL_MAX * TC_COLS_MAX
        && elements % 8 == 0
        && ((input | output) & 15) == 0
}

/// Grid blocks for `PREFILL_OP_PACK` (each lane processes 8 contiguous FP32 values -> 4 packed BF16 words).
#[inline(always)]
pub const fn pack_blocks(elements: usize) -> u32 {
    (elements / 8).div_ceil(PREFILL_THREADS) as u32
}

/// Device-side admission guard for `PREFILL_OP_FOLD` (vectorized BF16 `RouteFold`).
#[inline(always)]
pub const fn fold_admitted(
    rows: usize, picks: usize, dim: usize, weights: usize, downs: usize, output: usize, threads: usize,
) -> bool {
    threads == PREFILL_THREADS
        && rows >= 1 && rows <= MOE_ROWS_MAX
        && picks >= 1 && picks <= MOE_PICKS_MAX
        && rows * picks <= MOE_ALL_MAX
        && dim >= 8 && dim <= TC_ROWS_MAX && dim % 8 == 0
        && (weights & 3) == 0
        && ((downs | output) & 15) == 0
}

/// Grid blocks for `PREFILL_OP_FOLD` (one block per token row, 256 threads processing 8 BF16 values per lane).
#[inline(always)]
pub const fn fold_blocks(rows: usize) -> u32 {
    rows as u32
}

/// Grid blocks for `PREFILL_OP_GEMM`: `N`-tiles times `max_m_tiles`, grouped by `PREFILL_M_GROUP`.
#[inline(always)]
pub const fn gemm_blocks(rows: usize, max_m_tiles: usize) -> u32 {
    (rows.div_ceil(PREFILL_BN) * max_m_tiles) as u32
}

/// Device-side admission guard for `PREFILL_OP_GEMM`.
#[inline(always)]
#[allow(clippy::too_many_arguments, reason = "shape, format, MoE counts, unit size, pointers and threads")]
pub const fn gemm_admitted(
    rows: usize, cols: usize, wide: bool, pairs: bool, format: u64, gated: bool,
    all: usize, stride: usize, experts: usize, max_m_tiles: usize, unit_bytes: usize,
    stack: usize, input: usize, output: usize, coefficients: usize, threads: usize,
) -> bool {
    let width = if gated { rows / 2 } else { rows };
    let format_ok = match format {
        PREFILL_OUT_FP32 => stride >= width && (output & 3) == 0,
        PREFILL_OUT_BF16_GATED => gated && pairs && width % 2 == 0 && stride == width && (output & 3) == 0,
        PREFILL_OUT_BF16 => !gated && !pairs && rows % 2 == 0 && stride == width && stride % 2 == 0 && (output & 3) == 0,
        _ => false,
    };
    let coeff_ok = if gated { coefficients != 0 && (coefficients & 3) == 0 } else { coefficients == 0 };
    tc_packed_moe_shape_admitted(rows, cols, wide, pairs)
        && (!gated || pairs)
        && format_ok
        && coeff_ok
        && all >= 1 && all <= MOE_ALL_MAX
        && experts >= 1 && experts <= MOE_EXPERTS_MAX
        && max_m_tiles == moe_max_tiles(all, experts)
        && unit_bytes == layout(rows, cols, wide).bytes
        && threads == PREFILL_THREADS
        && ((stack | input) & 15) == 0
}

/// Exact BF16 bytes of `rows x cols` packed BF16 rows (two BF16 values per u32 word).
#[inline(always)]
pub const fn bf16_row_bytes(rows: usize, cols: usize) -> Option<usize> {
    if rows == 0 || cols == 0 || cols % 2 != 0 { return None; }
    let Some(elements) = rows.checked_mul(cols) else { return None };
    elements.checked_mul(2)
}

#[cfg(not(target_arch = "nvptx64"))]
pub use host::*;

#[cfg(not(target_arch = "nvptx64"))]
mod host {
    use super::*;
    use crate::tc_rows::{Span, moe_tile_meta_words};

    fn ends(s: Span) -> bool { s.address != 0 && s.address.checked_add(s.bytes).is_some() }
    fn meets(a: Span, b: Span) -> bool {
        a.address < b.address.saturating_add(b.bytes) && b.address < a.address.saturating_add(a.bytes)
    }

    /// Host check for `PREFILL_OP_PACK`.
    pub fn check_pack(elements: usize, input: Span, output: Span) -> Result<(), &'static str> {
        if !pack_admitted(elements, input.address as usize, output.address as usize, PREFILL_THREADS) {
            return Err("prefill moe pack: shape or alignment outside envelope");
        }
        let in_bytes = (elements as u64).checked_mul(4).ok_or("prefill moe pack: input byte overflow")?;
        let out_bytes = (elements as u64).checked_mul(2).ok_or("prefill moe pack: output byte overflow")?;
        if input.bytes != in_bytes { return Err("prefill moe pack: input extent mismatch"); }
        if output.bytes != out_bytes { return Err("prefill moe pack: output extent mismatch"); }
        if !ends(input) || !ends(output) { return Err("prefill moe pack: null or wrapping operand"); }
        if meets(input, output) { return Err("prefill moe pack: output overlaps input"); }
        Ok(())
    }

    /// Host check for `PREFILL_OP_FOLD`.
    pub fn check_fold(
        rows: usize, picks: usize, dim: usize, weights: Span, downs: Span, output: Span,
    ) -> Result<(), &'static str> {
        if !fold_admitted(rows, picks, dim, weights.address as usize, downs.address as usize, output.address as usize, PREFILL_THREADS) {
            return Err("prefill moe fold: shape or alignment outside envelope");
        }
        let all = (rows as u64).checked_mul(picks as u64).ok_or("prefill moe fold: pick overflow")?;
        let w_bytes = all.checked_mul(4).ok_or("prefill moe fold: weight byte overflow")?;
        let d_bytes = all.checked_mul(dim as u64).and_then(|n| n.checked_mul(2)).ok_or("prefill moe fold: down byte overflow")?;
        let o_bytes = (rows as u64).checked_mul(dim as u64).and_then(|n| n.checked_mul(4)).ok_or("prefill moe fold: output byte overflow")?;
        if weights.bytes != w_bytes || downs.bytes != d_bytes || output.bytes != o_bytes {
            return Err("prefill moe fold: operand extent mismatch");
        }
        if !ends(weights) || !ends(downs) || !ends(output) { return Err("prefill moe fold: null or wrapping operand"); }
        if meets(output, weights) || meets(output, downs) { return Err("prefill moe fold: output overlaps input"); }
        Ok(())
    }

    /// Host check for `PREFILL_OP_GEMM`.
    #[allow(clippy::too_many_arguments, reason = "shape, format, route tables, optional gate tables and spans")]
    pub fn check_gemm(
        rows: usize, cols: usize, wide: bool, pairs: bool, format: u64,
        all: usize, experts: usize, stack: Span, tile_meta: Span,
        input: Span, in_count: usize, in_table: Option<Span>,
        output: Span, out_count: usize, out_table: Option<Span>,
        coefficients: Option<Span>,
    ) -> Result<(), &'static str> {
        let gated = coefficients.is_some();
        let width = if gated { rows / 2 } else { rows };
        let max_tiles = moe_max_tiles(all, experts);
        let unit_bytes = layout(rows, cols, wide).bytes;
        let coeff_addr = coefficients.map_or(0, |s| s.address as usize);
        if !gemm_admitted(
            rows, cols, wide, pairs, format, gated, all, width, experts, max_tiles, unit_bytes,
            stack.address as usize, input.address as usize, output.address as usize, coeff_addr, PREFILL_THREADS,
        ) {
            return Err("prefill moe gemm: shape, format, or alignment outside envelope");
        }
        if in_count == 0 || out_count == 0 { return Err("prefill moe gemm: empty count"); }
        if in_table.is_none() && in_count != all { return Err("prefill moe gemm: direct input differs from picks"); }
        if out_table.is_none() && out_count != all { return Err("prefill moe gemm: direct output differs from picks"); }
        let stack_bytes = (experts as u64).checked_mul(unit_bytes as u64).ok_or("prefill moe gemm: stack overflow")?;
        let meta_bytes = (moe_tile_meta_words(all, experts) as u64).checked_mul(4).ok_or("prefill moe gemm: meta overflow")?;
        let in_bytes = bf16_row_bytes(in_count, cols).ok_or("prefill moe gemm: input overflow")? as u64;
        let out_bytes = match format {
            PREFILL_OUT_FP32 => (out_count as u64).checked_mul(width as u64).and_then(|n| n.checked_mul(4)),
            PREFILL_OUT_BF16_GATED | PREFILL_OUT_BF16 => bf16_row_bytes(out_count, width).map(|n| n as u64),
            _ => None,
        }.ok_or("prefill moe gemm: output byte overflow")?;
        if stack.bytes != stack_bytes || tile_meta.bytes != meta_bytes || input.bytes != in_bytes || output.bytes != out_bytes {
            return Err("prefill moe gemm: operand extent mismatch");
        }
        if !ends(stack) || !ends(tile_meta) || !ends(input) || !ends(output) {
            return Err("prefill moe gemm: null or wrapping operand");
        }
        if (tile_meta.address & 3) != 0 { return Err("prefill moe gemm: tile_meta misaligned"); }
        if meets(output, stack) || meets(output, tile_meta) || meets(output, input) {
            return Err("prefill moe gemm: output overlaps input");
        }
        if let Some(c) = coefficients {
            let need_bytes = (2 * CAPPED_GATE_TABLE_WORDS) as u64 * 4;
            if c.bytes != need_bytes || !ends(c) || (c.address & 3) != 0 || meets(output, c) {
                return Err("prefill moe gemm: invalid or overlapping coefficient tables");
            }
        }
        let tbl_bytes = (all as u64).checked_mul(4).ok_or("prefill moe gemm: table overflow")?;
        for tbl in [in_table, out_table].into_iter().flatten() {
            if tbl.bytes != tbl_bytes || !ends(tbl) || (tbl.address & 3) != 0 || meets(output, tbl) {
                return Err("prefill moe gemm: invalid or overlapping index table");
            }
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn admission_and_check_guards_enforce_shapes_alignment_and_overlap() {
            let span = |addr: u64, bytes: u64| Span { address: addr, bytes };
            // Pack checks
            assert!(check_pack(64 * 2304, span(0x1000, 64 * 2304 * 4), span(0x20_0000, 64 * 2304 * 2)).is_ok());
            assert!(check_pack(64 * 2304, span(0x1004, 64 * 2304 * 4), span(0x20_0000, 64 * 2304 * 2)).is_err());
            assert!(check_pack(64 * 2304, span(0x1000, 64 * 2304 * 4), span(0x1000, 64 * 2304 * 2)).is_err());
            assert!(check_pack(12, span(0x1000, 48), span(0x20_0000, 24)).is_err());

            // Fold checks
            assert!(check_fold(16, 8, 2304, span(0x1000, 128 * 4), span(0x2000, 128 * 2304 * 2), span(0x20_0000, 16 * 2304 * 4)).is_ok());
            assert!(check_fold(16, 8, 2300, span(0x1000, 128 * 4), span(0x2000, 128 * 2300 * 2), span(0x20_0000, 16 * 2300 * 4)).is_err());
            assert!(check_fold(16, 8, 2304, span(0x1000, 128 * 4), span(0x2004, 128 * 2304 * 2), span(0x20_0000, 16 * 2304 * 4)).is_err());

            // GEMM checks: gated BF16 GateUp, ungated BF16 Down, and MXFP8 gated BF16 GateUp.
            let unit_nv = layout(2304, 2304, false).bytes as u64;
            let meta = moe_tile_meta_words(512, 16) as u64 * 4;
            let in_bf16 = bf16_row_bytes(64, 2304).unwrap() as u64;
            let out_bf16 = bf16_row_bytes(512, 1152).unwrap() as u64;
            let coeff = span(0x0500_0000, (2 * CAPPED_GATE_TABLE_WORDS) as u64 * 4);
            assert!(check_gemm(
                2304, 2304, false, true, PREFILL_OUT_BF16_GATED, 512, 16,
                span(0x1000_0000, 16 * unit_nv), span(0x0100_0000, meta),
                span(0x0200_0000, in_bf16), 64, Some(span(0x0300_0000, 512 * 4)),
                span(0x0400_0000, out_bf16), 512, None, Some(coeff),
            ).is_ok());
            // Reject gated BF16 without coefficient tables or ungated BF16 with pairs/coefficients.
            assert!(check_gemm(
                2304, 2304, false, true, PREFILL_OUT_BF16_GATED, 512, 16,
                span(0x1000_0000, 16 * unit_nv), span(0x0100_0000, meta),
                span(0x0200_0000, in_bf16), 64, Some(span(0x0300_0000, 512 * 4)),
                span(0x0400_0000, out_bf16), 512, None, None,
            ).is_err());
            let unit_dn = layout(2304, 1152, false).bytes as u64;
            assert!(check_gemm(
                2304, 1152, false, false, PREFILL_OUT_BF16, 512, 16,
                span(0x1000_0000, 16 * unit_dn), span(0x0100_0000, meta),
                span(0x0200_0000, bf16_row_bytes(512, 1152).unwrap() as u64), 512, None,
                span(0x0400_0000, bf16_row_bytes(512, 2304).unwrap() as u64), 512, Some(span(0x0300_0000, 512 * 4)), None,
            ).is_ok());
            let unit_mx = layout(2304, 2304, true).bytes as u64;
            assert!(check_gemm(
                2304, 2304, true, true, PREFILL_OUT_BF16_GATED, 512, 16,
                span(0x1000_0000, 16 * unit_mx), span(0x0100_0000, meta),
                span(0x0200_0000, in_bf16), 64, Some(span(0x0300_0000, 512 * 4)),
                span(0x0400_0000, out_bf16), 512, None, Some(coeff),
            ).is_ok());
        }

        #[test]
        fn register_unpack_identity_matches_nvfp4_and_mxfp8_tables_bit_for_bit() {
            // Verify the exact (V1 * 2^112, V3_half * 2^112) * pow2 * 2^-112 identity across all 127 x 16
            // NVFP4 entries for non-dyadic factors (repeating binary denominator 2688.0), and all 254 x 254
            // MXFP8 pairs!
            let e2m1 = |n: u8| -> f32 {
                let m = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(n & 7) as usize];
                if (n & 8) != 0 { -m } else { m }
            };
            let e4m3 = |b: u8| -> f32 {
                let e = (b >> 3) & 15;
                let m = b & 7;
                let v = if e == 0 { m as f32 / 512.0 } else { f32::from_bits(((e as u32 + 120) << 23) | ((m as u32) << 20)) };
                if (b & 128) != 0 { -v } else { v }
            };
            let rn16 = |x: f32| -> u16 { (crate::fixed_math::rn(x).to_bits() >> 16) as u16 };
            let from16 = |u: u16| -> f32 { f32::from_bits(u32::from(u) << 16) };
            let up112 = from16(0x7780);
            for factor in [0.699_218_75f32 / 2688.0, 0.300_781_25 / 2688.0, 1.75 / 2688.0, 0.000_123_456, 2.345] {
                for sc in 0u8..127 {
                    let s = e4m3(sc);
                    let v1_up = rn16(from16(rn16(1.0 * s * factor)) * up112);
                    let v3h_up = rn16(from16(rn16(1.5 * s * factor)) * up112);
                    for code in 0u8..16 {
                        let want = rn16((e2m1(code) * s) * factor);
                        let is3 = matches!(code & 7, 3 | 5 | 7);
                        let p_bits: u16 = match code & 7 {
                            0 => 0x0000, 1 => 0x0700, 2 | 3 => 0x0780, 4 | 5 => 0x0800, _ => 0x0880,
                        } | if (code & 8) != 0 { 0x8000 } else { 0 };
                        let base = if is3 { v3h_up } else { v1_up };
                        let got = rn16(from16(base) * from16(p_bits));
                        assert_eq!(got, want, "nvfp4 factor={factor} sc={sc:#x} code={code:#x}: got={got:#x} want={want:#x}");
                    }
                }
            }
            // MXFP8: all finite E4M3 codes x all scales 0..=142 (plus 2-step 143..=246 where finite).
            for sc in 0u32..=245 {
                let p_ref = crate::packed_product::power(sc);
                for code in 0u8..=255 {
                    if (code & 0x7f) == 0x7f { continue; }
                    let val = e4m3(code);
                    let prod = val * p_ref;
                    if !prod.is_finite() { continue; }
                    let want = rn16(prod);
                    let shifted = rn16(val * from16(0x0780)); // val * 2^-112
                    let got = if sc <= 142 {
                        let sc_up = ((sc + 112) << 7) as u16;
                        rn16(from16(shifted) * from16(sc_up))
                    } else {
                        let mid = rn16(from16(shifted) * up112);
                        rn16(from16(mid) * from16((sc << 7) as u16))
                    };
                    assert_eq!(got, want, "mxfp8 sc={sc} code={code:#x}: got={got:#x} want={want:#x}");
                }
            }
        }
    }
}
