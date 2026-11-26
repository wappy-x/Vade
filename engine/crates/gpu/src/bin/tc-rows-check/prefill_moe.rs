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

    // 5. PREFILL_FLASH_WINDOW (kind 122) bit-exactness vs FLASH_WINDOW_ATTN across:
    //    - Full attention & sliding window (W=1024, wrapped ring & unwrapped)
    //    - 3:1 GQA (`24/8`), 2:1 GQA (`8/4`), 1:1 MHA (`4/4`)
    //    - Non-multiple-of-64 rows (`65`, `130`), non-zero `first_position` (`17`, `4096`), `FW_CLIP_IN`, `half_out=true/false`.
    for &(rows, heads, kv_heads, view, extent, window, first_pos, half_out, label) in &[
        (65usize, 24usize, 8usize, 2048usize, 1024usize, 0usize, 17usize, true, "full-gqa3-r65-half"),
        (130, 24, 8, 1200, 1024, 1024, 1024, true, "win-gqa3-r130-half"),
        (128, 24, 8, 1200, 1024, 1024, 4096, false, "win-gqa3-r128-f32"),
        (64, 8, 4, 256, 64, 0, 0, true, "full-gqa2-r64-half"),
        (65, 4, 4, 256, 64, 0, 5, false, "full-gqa1-r65-f32"),
    ] {
        let (hd, rd) = (128usize, 16usize);
        let q_v: Vec<f32> = (0..rows * heads * hd).map(|i| input(i + rows * 7 + first_pos)).collect();
        let m_v: Vec<f32> = (0..rows * heads * rd).map(|i| input(i + rows * 11 + 3) * 0.5).collect();
        let g_v: Vec<f32> = (0..rows * heads * hd).map(|i| input(i + rows * 13 + 9)).collect();
        let s_v: Vec<f32> = (0..rows * heads).map(|i| input(i + 31) * 0.3).collect();
        let k_v: Vec<f32> = (0..view * kv_heads * (hd / 2)).map(|i| {
            let a = rn(input(2 * i + 101));
            let c = rn(input(2 * i + 102));
            f32::from_bits((a.to_bits() >> 16) | (c.to_bits() & 0xffff_0000))
        }).collect();
        let v_v: Vec<f32> = (0..view * kv_heads * (hd / 2)).map(|i| {
            let a = rn(input(2 * i + 301));
            let c = rn(input(2 * i + 302));
            f32::from_bits((a.to_bits() >> 16) | (c.to_bits() & 0xffff_0000))
        }).collect();
        let rt_v: Vec<f32> = (0..rd * extent).map(|i| input(i + 77) * 0.25).collect();
        let qs_v: Vec<f32> = (0..heads * hd).map(|i| 0.8 + (i % 7) as f32 * 0.05).collect();
        let ms_v: Vec<f32> = (0..heads * rd).map(|i| 0.9 + (i % 5) as f32 * 0.04).collect();
        let pos_v: Vec<f32> = (0..rows).map(|r| 1.0 + (r as f32) * 0.001).collect();
        let mut st_v = Vec::with_capacity(2 * rows);
        for r in 0..rows {
            let abs = first_pos + r;
            let live = if window == 0 { (abs + 1).min(view) } else { (abs + 1).min(window) };
            let slot = abs % view;
            st_v.push(live as f32);
            st_v.push(slot as f32);
        }
        let q_d = b.upload_vector(&q_v)?.into_resident_shared()?;
        let m_d = b.upload_vector(&m_v)?.into_resident_shared()?;
        let g_d = b.upload_vector(&g_v)?.into_resident_shared()?;
        let s_d = b.upload_vector(&s_v)?.into_resident_shared()?;
        let k_d = b.upload_vector(&k_v)?.into_resident_shared()?;
        let v_d = b.upload_vector(&v_v)?.into_resident_shared()?;
        let rt_d = b.upload_vector(&rt_v)?.into_resident_shared()?;
        let qs_d = b.upload_vector(&qs_v)?.into_resident_shared()?;
        let ms_d = b.upload_vector(&ms_v)?.into_resident_shared()?;
        let pos_d = b.upload_vector(&pos_v)?.into_resident_shared()?;
        let st_d = b.upload_vector(&st_v)?.into_resident_shared()?;
        let clips = || spark_gpu_probe::driver::OneClips {
            query_scale: Some(&qs_d), moved_scale: Some(&ms_d), position: Some(&pos_d),
            query_bound: 22.627_417, moved_bound: 8.0, written_keys: None, written_values: None,
        };
        let scale = (1.0f64 / (hd as f64).sqrt()) as f32;
        let out_words = if half_out { rows * heads * hd / 2 } else { rows * heads * hd };
        let ref_buf = poisoned(b, out_words)?;
        let new_buf = poisoned(b, out_words)?;
        if half_out {
            b.flash_window_attn_half_into(
                rows, heads, kv_heads, hd, rd, view, extent, scale, first_pos as u64,
                &q_d, &k_d, &v_d, &m_d, &rt_d, &s_d, &g_d, &st_d, Some(clips()),
                &mut ref_buf.resident_range(GUARD..GUARD + out_words)?,
            )?;
        } else {
            b.flash_window_attn_into(
                rows, heads, kv_heads, hd, rd, view, extent, scale, first_pos as u64,
                &q_d, &k_d, &v_d, &m_d, &rt_d, &s_d, &g_d, &st_d, Some(clips()),
                &mut ref_buf.resident_range(GUARD..GUARD + out_words)?,
            )?;
        }
        b.prefill_flash_window_into(
            rows, heads, kv_heads, hd, rd, view, extent, scale, first_pos as u64, half_out,
            &q_d, &k_d, &v_d, &m_d, &rt_d, &s_d, &g_d, &st_d, Some(clips()),
            &mut new_buf.resident_range(GUARD..GUARD + out_words)?,
        )?;
        let want = check_guards(b, &ref_buf, out_words, &format!("pfw-ref-{label}"))?;
        let got = check_guards(b, &new_buf, out_words, &format!("pfw-new-{label}"))?;
        check_bit_exact(&got, &want, &format!("prefill-flash-window {label}"))?;
        total_words += got.len();
        println!("PREFILL FLASH WINDOW {label} BIT-EXACT PASS (0 bit diffs)");
    }

    // -------------------------------------------------------------------------
    // 5. PREFILL_GATE_FUSE (kind 123): bit-exact vs MIX_LIST_NORM_HALF + TC_MODE_NARROW_HALF + TC_MODE_GATE_BLEND_NORM.
    // -------------------------------------------------------------------------
    let gate_cases: &[(&str, usize, usize, usize, usize, usize, bool, bool)] = &[
        ("ink-3g-k2-same-half", 33, 2304, 64, 3, 2, true, true),
        ("ink-3g-k2-each-half", 19, 2304, 64, 3, 2, false, true),
        ("ink-1g-k2-half", 35, 2304, 64, 1, 2, true, true),
        ("ink-3g-k3-each-f32", 17, 1280, 64, 3, 3, false, false),
        ("ink-2g-k4-dim2432-half", 18, 2432, 32, 2, 4, false, true),
    ];
    for &(label, rows, dim, k, count, kept, same_kept, half_out) in gate_cases {
        let asked = count + 5;
        let scale = 1.0f32 / (dim as f32).sqrt();
        let eps = 1.0e-5f32;
        let q_v: Vec<f32> = (0..asked * dim).map(|i| input(i + 11) * 0.2).collect();
        let q_d = b.upload_vector(&q_v)?.into_resident_shared()?;
        let mut src_ds = Vec::with_capacity(kept);
        for e in 0..kept {
            let sv: Vec<f32> = (0..rows * dim).map(|i| input(i + 53 * (e + 1)) * 0.5).collect();
            src_ds.push(b.upload_vector(&sv)?.into_resident_shared()?);
        }
        let src_refs: Vec<&DeviceVector<'_>> = src_ds.iter().collect();
        // Reference path: MIX_LIST_NORM_HALF -> TC_MODE_NARROW_HALF -> TC_MODE_GATE_BLEND_NORM
        let (ref_msgs, ref_msg_nh) = b.fast_mix_list_normed_half(rows, dim, scale, eps, &q_d, &src_refs)?;
        let ref_msgs = ref_msgs.into_resident_shared()?;
        let ref_msg_nh = ref_msg_nh.into_resident_shared()?;
        // New path: PG_OP_MSG_META
        let (mw_all, mn_all) = b.prefill_msg_meta(rows, dim, scale, eps, &q_d, &src_refs)?;
        let mw_all = mw_all.into_resident_shared()?;
        let mn_all = mn_all.into_resident_shared()?;

        let mut wd_ds = Vec::with_capacity(count);
        let mut wo_ds = Vec::with_capacity(count);
        let mut kv_ds = Vec::with_capacity(count);
        let mut kh_ds = Vec::with_capacity(count);
        let mut nw_ds = Vec::with_capacity(count);
        let mut ref_inners = Vec::with_capacity(count);
        let mut ref_msg_slices = Vec::with_capacity(count);
        let mut mw_slices = Vec::with_capacity(count);
        let mut mn_slices = Vec::with_capacity(count);
        let kv0_v: Vec<f32> = (0..rows * dim).map(|i| input(i + 201) * 0.4).collect();
        let kv0_d = b.upload_vector(&kv0_v)?.into_resident_shared()?;
        let kh0_v: Vec<f32> = (0..rows * (dim / 2)).map(|i| {
            let a = (input(2 * i + 201) * 0.4).to_bits() >> 16;
            let c = (input(2 * i + 202) * 0.4).to_bits() >> 16;
            f32::from_bits(a | (c << 16))
        }).collect();
        let kh0_d = b.upload_vector(&kh0_v)?.into_resident_shared()?;
        for gi in 0..count {
            let q_idx = gi + 2;
            let wd_v: Vec<f32> = (0..k * dim).map(|i| {
                let a = (input(i * 2 + 301 * (gi + 1)) * 0.15).to_bits() >> 16;
                let c = (input(i * 2 + 302 * (gi + 1)) * 0.15).to_bits() >> 16;
                f32::from_bits(a | (c << 16))
            }).collect();
            let wo_v: Vec<f32> = (0..dim * k).map(|i| {
                let a = (input(i * 2 + 501 * (gi + 1)) * 0.15).to_bits() >> 16;
                let c = (input(i * 2 + 502 * (gi + 1)) * 0.15).to_bits() >> 16;
                f32::from_bits(a | (c << 16))
            }).collect();
            let nw_v: Vec<f32> = (0..dim).map(|i| 0.75 + ((i + gi) % 13) as f32 * 0.03).collect();
            wd_ds.push(b.upload_vector(&wd_v)?.into_resident_shared()?);
            wo_ds.push(b.upload_vector(&wo_v)?.into_resident_shared()?);
            nw_ds.push(b.upload_vector(&nw_v)?.into_resident_shared()?);
            if same_kept {
                kv_ds.push(kv0_d.resident_range(0..rows * dim)?);
                kh_ds.push(kh0_d.resident_range(0..rows * (dim / 2))?);
            } else {
                let kv_v: Vec<f32> = (0..rows * dim).map(|i| input(i + 201 * (gi + 1)) * 0.4).collect();
                let kh_v: Vec<f32> = (0..rows * (dim / 2)).map(|i| {
                    let a = (input(2 * i + 201 * (gi + 1)) * 0.4).to_bits() >> 16;
                    let c = (input(2 * i + 202 * (gi + 1)) * 0.4).to_bits() >> 16;
                    f32::from_bits(a | (c << 16))
                }).collect();
                kv_ds.push(b.upload_vector(&kv_v)?.into_resident_shared()?);
                kh_ds.push(b.upload_vector(&kh_v)?.into_resident_shared()?);
            }
            let m_slice = ref_msgs.resident_range(q_idx * rows * dim..(q_idx + 1) * rows * dim)?;
            let n_slice = ref_msg_nh.resident_range(q_idx * rows * (dim / 2)..(q_idx + 1) * rows * (dim / 2))?;
            let mut inner_buf = b.vector(rows * k)?;
            b.project_tc_rows_pair_half_in_into(k, 2 * dim, rows, &kh_ds[gi], &n_slice, &wd_ds[gi], &mut inner_buf)?;
            ref_inners.push(inner_buf);
            ref_msg_slices.push(m_slice);
            mw_slices.push(mw_all.resident_range(q_idx * rows * kept..(q_idx + 1) * rows * kept)?);
            mn_slices.push(mn_all.resident_range(q_idx * rows..(q_idx + 1) * rows)?);
        }
        let ref_entries: Vec<_> = (0..count)
            .map(|gi| (&wo_ds[gi], &ref_inners[gi], &kv_ds[gi], &ref_msg_slices[gi], Some(&nw_ds[gi])))
            .collect();
        let ref_outs = b.project_tc_gate_blend_norm(rows, dim, k, eps, half_out, &ref_entries)?;
        let new_entries: Vec<_> = (0..count)
            .map(|gi| (&wd_ds[gi], &wo_ds[gi], &kv_ds[gi], &kh_ds[gi], &mw_slices[gi], &mn_slices[gi], Some(&nw_ds[gi])))
            .collect();
        let new_outs = b.prefill_gate_fuse(rows, dim, k, eps, half_out, &src_refs, &new_entries)?;
        for gi in 0..count {
            let want = b.read_vector_values(&ref_outs[gi])?;
            let got = b.read_vector_values(&new_outs[gi])?;
            check_bit_exact(&got, &want, &format!("prefill-gate-fuse {label} g={gi}"))?;
            total_words += got.len();
        }
        println!("PREFILL GATE FUSE {label} BIT-EXACT PASS (0 bit diffs)");
    }

    Ok(total_words)
}
