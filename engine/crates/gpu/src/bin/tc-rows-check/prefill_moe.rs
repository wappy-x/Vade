//! Device bit-exactness, guard-word, bounds-refusal, and memory-safety check for `PREFILL_PACKED_TC` (kind 121):
//! - `PREFILL_OP_PACK` vs `TC_MODE_PACK` (exact `cvt.rn.bf16x2.f32`) bit-for-bit (`0` diffs).
//! - `PREFILL_OP_FOLD` vs `fast_route_fold_half_into` bit-for-bit (`0` diffs).
//! - `PREFILL_OP_GEMM` across `mode == 2` (tagged NVFP4 panels), `mode == 0` (shipped NVFP4), and
//!   `mode == 1` (shipped MXFP8) vs `PLAIN_PACKED_TC_ROWS`:
//!   - Standalone FP32 `GateUp`, gated BF16 `GateUp`, ungated BF16 `Down`, and full end-to-end
//!     `GateUp -> BF16 -> Down -> BF16 -> RouteFold`: **0.0 max abs diff, 0 bit diffs** on every single word!
use spark_gpu_probe::{
    decode_panels,
    driver::{Backend, DeviceVector},
    fixed_math::rn,
    packed_product, prefill_moe,
    resident_launch::Kind,
};

const GUARD: usize = 64;
const POISON: u32 = 0x7fc0_dead;

fn input(k: usize) -> f32 {
    (((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0) * 1.5
}

fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> {
    b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared()
}

fn upload_bytes<'d>(b: &Backend<'d>, bytes: &[u8]) -> Result<DeviceVector<'d>, String> {
    let mut v = b.vector(bytes.len() / 4)?;
    b.write_vector_bytes(&mut v, bytes)?;
    v.into_resident_shared()
}

fn check_guards(b: &Backend<'_>, whole: &DeviceVector<'_>, body_words: usize, label: &str) -> Result<Vec<f32>, String> {
    let back = b.read_vector_values(whole)?;
    if back.len() != body_words + 2 * GUARD {
        return Err(format!("{label}: bad readback len {} != {}", back.len(), body_words + 2 * GUARD));
    }
    let bad = back[..GUARD].iter().chain(&back[GUARD + body_words..]).filter(|w| w.to_bits() != POISON).count();
    if bad > 0 {
        return Err(format!("{label}: {bad} guard words overwritten"));
    }
    Ok(back[GUARD..GUARD + body_words].to_vec())
}

fn check_bit_exact(got: &[f32], want: &[f32], label: &str) -> Result<(), String> {
    if got.len() != want.len() {
        return Err(format!("{label}: len {} != {}", got.len(), want.len()));
    }
    if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
        return Err(format!(
            "{label} word {i}: got {:#x} ({}) want {:#x} ({})",
            got[i].to_bits(), got[i], want[i].to_bits(), want[i]
        ));
    }
    Ok(())
}

fn decode_e2m1(nibble: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = MAG[(nibble & 7) as usize];
    if (nibble & 8) != 0 { -v } else { v }
}

fn decode_e4m3fn(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = (byte >> 3) & 0xf;
    let mant = byte & 7;
    if exp == 0xf && mant == 7 { return f32::NAN; }
    let mag = if exp == 0 { (mant as f32) / 512.0 }
    else { f32::from_bits(((u32::from(exp) + 120) << 23) | (u32::from(mant) << 20)) };
    sign * mag
}

/// Build synthetic NVFP4 (`wide = false`) or MXFP8 (`wide = true`) expert stack using true non-dyadic
/// `amax_bf16 / 2688.0` factors (including subnormal scales `0..7` and normal scales `8..126`) matching `loop_packed_unit`.
fn make_exact_stack(
    experts: usize, rows: usize, cols: usize, wide: bool, tagged_panel: bool, seed: usize,
) -> Vec<u8> {
    let parts = packed_product::layout(rows, cols, wide);
    let mut stack = vec![0u8; experts * parts.bytes];
    for e in 0..experts {
        let unit = &mut stack[e * parts.bytes..(e + 1) * parts.bytes];
        // Non-dyadic FP32 factor (repeating binary expansion) to prove zero double-rounding vs main's 8 KB LUT.
        let amax = packed_product::round_bf16(0.3f32 + (e as f32) * 0.13);
        let factor = amax / 2688.0f32;
        if wide {
            for code in 0..256usize {
                let v = if (code & 0x7f) == 0x7f { 0.0 } else { decode_e4m3fn(code as u8) };
                unit[parts.table + code * 4..parts.table + code * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        } else {
            for sc in 0..256usize {
                let sv = decode_e4m3fn(sc as u8);
                for nib in 0..16usize {
                    let w = if sv.is_nan() { 0.0 } else { packed_product::round_bf16((decode_e2m1(nib as u8) * sv) * factor) };
                    let u16_bits = (w.to_bits() >> 16) as u16;
                    let idx = sc * 16 + nib;
                    unit[parts.table + idx * 2..parts.table + idx * 2 + 2].copy_from_slice(&u16_bits.to_le_bytes());
                }
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                let stage = c / packed_product::STAGE;
                let lane = c % packed_product::STAGE;
                let c_off = packed_product::code_copy(r, stage, 0, cols, wide);
                let s_off = packed_product::scale_copy(r, stage, cols, wide);
                if wide {
                    let mut code_b = ((r * 31 + c * 17 + e * 13 + seed) & 0x7f) as u8;
                    if code_b == 0x7f { code_b = 0x38; }
                    if ((r + c + e) & 1) != 0 { code_b |= 0x80; }
                    // Exercise subnormal scale 0, typical scales 118..134, and high scale 145 (> 142 two-step path).
                    let sc_pool = [0u8, 118, 121, 125, 127, 130, 134, 145];
                    let sc_b = sc_pool[(r * 7 + (c / 32) * 5 + e * 3) % sc_pool.len()];
                    unit[c_off + lane] = code_b;
                    unit[parts.scales + s_off + (c / 32) % 4] = sc_b;
                } else {
                    let nib = ((r * 29 + c * 19 + e * 11 + seed) & 15) as u8;
                    // Exercise scale 0, subnormal E4M3 scales 1..7, and full normal range 8..126.
                    let sc_b = ((r * 13 + (c / 16) * 7 + e * 5 + seed) % 127) as u8;
                    if c % 2 == 0 {
                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0xf0) | nib;
                    } else {
                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0x0f) | (nib << 4);
                    }
                    unit[parts.scales + s_off + lane / 16] = sc_b;
                }
            }
        }
        if tagged_panel && !wide {
            let packed = decode_panels::pack_expert_tagged(unit, rows, cols);
            unit.copy_from_slice(&packed);
        }
    }
    stack
}

pub fn run(b: &Backend<'_>) -> Result<usize, String> {
    let mut total_words = 0usize;

    // 1. PREFILL_OP_PACK bit-exact parity against exact CPU BF16 rounding (matching TC_MODE_PACK).
    for &(rows, cols) in &[(1usize, 64usize), (17, 256), (65, 128), (9, 2304)] {
        let x_vals: Vec<f32> = (0..rows * cols).map(|i| input(i + rows * 13 + cols)).collect();
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let p_dev = b.prefill_pack_bf16(rows, cols, &x_dev)?;
        let got = b.read_vector_values(&p_dev)?;
        let want: Vec<f32> = x_vals.as_chunks::<2>().0.iter().map(|pair| {
            let lo = rn(pair[0]).to_bits() >> 16;
            let hi = rn(pair[1]).to_bits() & 0xffff_0000;
            f32::from_bits(lo | hi)
        }).collect();
        check_bit_exact(&got, &want, &format!("prefill pack {rows}x{cols}"))?;
        total_words += got.len();
    }
    println!("PREFILL MOE PACK BIT-EXACT PASS");

    // 2. PREFILL_OP_FOLD bit-exact parity against fast_route_fold_half_into + guard words.
    for &(rows, picks, dim) in &[(1usize, 8usize, 256usize), (17, 8, 2304), (65, 4, 128)] {
        let all = rows * picks;
        let w_vals: Vec<f32> = (0..all).map(|i| ((i % 19) as f32) * 0.0625 + 0.05).collect();
        let d_vals: Vec<f32> = (0..all * dim / 2).map(|i| {
            let a = packed_product::round_bf16(input(2 * i + 7));
            let c = packed_product::round_bf16(input(2 * i + 8));
            f32::from_bits((a.to_bits() >> 16) | (c.to_bits() & 0xffff_0000))
        }).collect();
        let w_dev = b.upload_vector(&w_vals)?.into_resident_shared()?;
        let d_dev = b.upload_vector(&d_vals)?.into_resident_shared()?;
        let out_ref = poisoned(b, rows * dim)?;
        let out_new = poisoned(b, rows * dim)?;
        b.fast_route_fold_half_into(rows, picks, dim, &w_dev, &d_dev, &mut out_ref.resident_range(GUARD..GUARD + rows * dim)?)?;
        b.prefill_route_fold_half_into(rows, picks, dim, &w_dev, &d_dev, &mut out_new.resident_range(GUARD..GUARD + rows * dim)?)?;
        let want = check_guards(b, &out_ref, rows * dim, "fold ref")?;
        let got = check_guards(b, &out_new, rows * dim, "fold new")?;
        check_bit_exact(&got, &want, &format!("prefill fold {rows}x{picks}x{dim}"))?;
        total_words += want.len();
    }
    println!("PREFILL MOE FOLD BIT-EXACT PASS");

    // 3. Bit-exact GEMM parity (`0` bit diffs vs PLAIN_PACKED_TC_ROWS!) across:
    //    - mode == 2 (tagged NVFP4 panels, full PAIR_M=128 + partial tail + small M)
    //    - mode == 0 (shipped NVFP4)
    //    - mode == 1 (shipped MXFP8, including scale 0 and scale > 142)
    for &(rows, picks, experts, cols, inner, wide, tagged_panel, label) in &[
        (140usize, 8usize, 8usize, 256usize, 128usize, false, true, "nvfp4-panel-pair128"),
        (19, 8, 8, 128, 128, false, true, "nvfp4-panel-small"),
        (135, 8, 8, 128, 128, false, false, "nvfp4-shipped-pair128"),
        (135, 8, 8, 256, 128, true, false, "mxfp8-shipped-pair128"),
        (33, 8, 8, 256, 192, false, true, "nvfp4-panel-non128n"),
    ] {
        let all = rows * picks;
        let each = 2 * picks;
        let mut route_host = vec![0.0f32; rows * each];
        for r in 0..rows {
            for s in 0..picks {
                // Skew expert 0 to get > 128 picks so PAIR_M = 128 runs multiple full + partial pairs.
                let e = if s == 0 { 0 } else { (r * 3 + s * 2) % experts };
                let wt = ((r * 7 + s * 5) % 17) as f32 * 0.03125 + 0.0625;
                route_host[r * each + s] = e as f32;
                route_host[r * each + picks + s] = wt;
            }
        }
        let route_dev = b.upload_vector(&route_host)?.into_resident_shared()?;
        let w_out = poisoned(b, all)?;
        let plan = b.moe_group_route_into(rows, each, picks, experts, &route_dev, &mut w_out.resident_range(GUARD..GUARD + all)?)?;
        let weights_dev = w_out.resident_range(GUARD..GUARD + all)?;

        let g_bytes = make_exact_stack(experts, 2 * inner, cols, wide, tagged_panel, 101);
        let d_bytes = make_exact_stack(experts, cols, inner, wide, tagged_panel, 202);
        let g_dev = upload_bytes(b, &g_bytes)?;
        let d_dev = upload_bytes(b, &d_bytes)?;
        let x_vals: Vec<f32> = (0..rows * cols).map(|i| input(i + 53)).collect();
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;

        // 3a. Standalone FP32 GateUp GEMM: 0 bit diffs vs project_tc_packed_moe_into!
        let gu_ref = poisoned(b, all * 2 * inner)?;
        let gu_new = poisoned(b, all * 2 * inner)?;
        b.project_tc_packed_moe_into(
            2 * inner, cols, wide, true, &g_dev, &plan, &x_dev,
            &mut gu_ref.resident_range(GUARD..GUARD + all * 2 * inner)?, false,
        )?;
        b.project_prefill_packed_moe_into(
            2 * inner, cols, wide, true, prefill_moe::PREFILL_OUT_FP32,
            &g_dev, &plan, &x_dev, false,
            &mut gu_new.resident_range(GUARD..GUARD + all * 2 * inner)?, false, None,
        )?;
        let want_gu = check_guards(b, &gu_ref, all * 2 * inner, &format!("{label}-gu-ref"))?;
        let got_gu = check_guards(b, &gu_new, all * 2 * inner, &format!("{label}-gu-new"))?;
        check_bit_exact(&got_gu, &want_gu, &format!("{label} fp32-gu"))?;
        total_words += got_gu.len();

        // 3b. Gated BF16 GateUp: 0 bit diffs vs project_tc_packed_gate_half_into!
        let caps = [4.0f32, 25.0, 0.25, 0.04];
        let act_ref = poisoned(b, all * inner / 2)?;
        let act_new = poisoned(b, all * inner / 2)?;
        b.project_tc_packed_gate_half_into(
            2 * inner, cols, wide, &g_dev, &plan, &x_dev, caps,
            &mut act_ref.resident_range(GUARD..GUARD + all * inner / 2)?,
        )?;
        b.project_prefill_packed_moe_into(
            2 * inner, cols, wide, true, prefill_moe::PREFILL_OUT_BF16_GATED,
            &g_dev, &plan, &x_dev, false,
            &mut act_new.resident_range(GUARD..GUARD + all * inner / 2)?, false, Some(caps),
        )?;
        let want_act = check_guards(b, &act_ref, all * inner / 2, &format!("{label}-act-ref"))?;
        let got_act = check_guards(b, &act_new, all * inner / 2, &format!("{label}-act-new"))?;
        check_bit_exact(&got_act, &want_act, &format!("{label} gated-bf16"))?;
        total_words += got_act.len();

        // 3c. Full end-to-end GateUp -> BF16 -> Down -> BF16 -> RouteFold: 0 bit diffs vs main!
        let dn_ref = poisoned(b, all * cols / 2)?;
        let full_ref = poisoned(b, rows * cols)?;
        let full_new = poisoned(b, rows * cols)?;
        b.project_tc_packed_down_half_from_half_into(
            cols, inner, wide, &d_dev, &plan,
            &act_ref.resident_range(GUARD..GUARD + all * inner / 2)?,
            &mut dn_ref.resident_range(GUARD..GUARD + all * cols / 2)?,
        )?;
        b.fast_route_fold_half_into(
            rows, picks, cols, &weights_dev,
            &dn_ref.resident_range(GUARD..GUARD + all * cols / 2)?,
            &mut full_ref.resident_range(GUARD..GUARD + rows * cols)?,
        )?;
        b.project_prefill_packed_experts_into(
            2 * inner, cols, wide, &g_dev,
            cols, inner, wide, &d_dev,
            &plan, &x_dev, caps, &weights_dev, picks,
            &mut full_new.resident_range(GUARD..GUARD + rows * cols)?,
        )?;
        let want_full = check_guards(b, &full_ref, rows * cols, &format!("{label}-full-ref"))?;
        let got_full = check_guards(b, &full_new, rows * cols, &format!("{label}-full-new"))?;
        check_bit_exact(&got_full, &want_full, &format!("{label} full-e2e"))?;
        total_words += got_full.len();
        println!("PREFILL MOE {label} BIT-EXACT PASS max_abs_diff=0.0 (0 bit diffs)");
    }

    // 4. Bounds/alignment refusal check: nothing recorded on failure.
    let before = b.stats().recorded_kinds[Kind::PrefillPackedTc as usize];
    let bad_x = b.upload_vector(&vec![0.0f32; 8 * 64 + 8])?.into_resident_shared()?;
    assert!(b.prefill_pack_bf16(8, 64, &bad_x.resident_range(1..1 + 8 * 64)?).is_err());
    assert!(b.prefill_pack_bf16(1, 12, &bad_x.resident_range(0..12)?).is_err());
    assert_eq!(b.stats().recorded_kinds[Kind::PrefillPackedTc as usize], before);
    println!("PREFILL MOE REFUSALS PASS");

    Ok(total_words)
}
