//! Comprehensive bit-exact + guard-word device verification for Wave 1 `PLAIN_TC_ROWS` paths:
//! - Narrow 1-warp direct-FP32 (single, dual-input pair, 1..=3 grouped pair, fused `rn` rounding, tail K % 256 != 0)
//! - Narrow 4-warp prefill direct-FP32 (`M > 256`, single, pair, fused `rn` rounding)
//! - Short-K (`cols == 64`, `GateOuter`) direct-FP32 for both small M (`<= 32`) and prefill (`> 32`)
//! - Small-M wide pipelined (`1w 16x16x6`, `1w 32x16x6`, `2w 32x32x4`, `4w 32x64x4`) + split outputs
//! - Medium-M `2w 32x32x4` (`M = 33..=256`, `128 < rows <= 2560`, `cols > 64`)
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    tc_rows,
};

const GUARD: usize = 64;
const POISON: u32 = 0x7fc0_dead;

fn weight(i: usize) -> u16 {
    let n = (i as u32).wrapping_mul(2_654_435_761) >> 16;
    match i % 11 {
        0 => 0x0000,
        1 => 0x8000,
        2 => 0x0001 | (n as u16 & 0x007f),
        3 => 0x4100 | (n as u16 & 0x007f),
        _ => (0x3800 + (n % 0x0800)) as u16 | ((n as u16 & 1) << 15),
    }
}

fn input_val(k: usize) -> f32 {
    ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0
}

fn bf16_round_f32(x: f32) -> f32 {
    let bits = spark_gpu_probe::tc_twin_bf16::bf16(x);
    f32::from_bits(u32::from(bits) << 16)
}

fn matrix_with_outliers<'d>(b: &Backend<'d>, values: &[u16], rows: usize, cols: usize, register_panels: bool) -> Result<DeviceVector<'d>, String> {
    let mut v = b.vector(values.len() / 2)?;
    let bytes: Vec<u8> = values.iter().flat_map(|w| w.to_le_bytes()).collect();
    b.write_vector_bytes(&mut v, &bytes)?;
    let v = v.into_resident_shared()?;
    if register_panels {
        b.register_bf16_panels(&v, rows, cols, &bytes)?;
    }
    Ok(v)
}

fn panel_weight(r: usize, c: usize, seed: usize) -> u16 {
    // Tiles where `(r / 16 + c / 16) % 7 == 0` get an intentional wide-exponent outlier (`RAW` fallback);
    // all other tiles stay within 8 consecutive high-byte magnitudes so `bf16_panels` compresses them!
    let tr = r / 16;
    let tc = c / 16;
    let n = ((r * 131 + c * 17 + seed) as u32).wrapping_mul(2_654_435_761) >> 16;
    if (tr + tc).is_multiple_of(7) && r % 16 == 3 && c % 16 == 5 {
        return 0x4580 | (n as u16 & 0x007f);
    }
    if (tr + tc).is_multiple_of(7) && r % 16 == 11 && c % 16 == 9 {
        return 0x3500 | (n as u16 & 0x007f);
    }
    let base_mag = (0x38 + ((tr * 5 + tc * 3 + seed) % 6)) as u16;
    let delta = (n % 8) as u16;
    let sign = ((n >> 3) & 1) as u16;
    (sign << 15) | ((base_mag + delta) << 8) | (n as u16 & 0x00ff)
}

fn pack_bf16_words(x: &[f32]) -> Vec<u32> {
    x.as_chunks::<2>().0.iter()
        .map(|p| u32::from(spark_gpu_probe::tc_twin_bf16::bf16(p[0])) | (u32::from(spark_gpu_probe::tc_twin_bf16::bf16(p[1])) << 16))
        .collect()
}

fn upload_u32<'d>(b: &Backend<'d>, words: &[u32]) -> Result<DeviceVector<'d>, String> {
    let mut v = b.vector(words.len())?;
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    b.write_vector_bytes(&mut v, &bytes)?;
    v.into_resident_shared()
}

fn matrix<'d>(b: &Backend<'d>, values: &[u16]) -> Result<DeviceVector<'d>, String> {
    matrix_with_outliers(b, values, 0, 0, false)
}

fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> {
    b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared()
}

fn tc_ref(values: &[u16], x: &[f32], rows: usize, cols: usize, inputs: usize, round_out: bool) -> Vec<f32> {
    let mut out = Vec::with_capacity(inputs * rows);
    for m in 0..inputs {
        let xm = &x[m * cols..(m + 1) * cols];
        for n in 0..rows {
            let v = tc_rows::tc_word(&values[n * cols..(n + 1) * cols], xm);
            out.push(if round_out { bf16_round_f32(v) } else { v });
        }
    }
    out
}

fn check_buf(b: &Backend<'_>, whole: &DeviceVector<'_>, want: &[f32], label: &str) -> Result<usize, String> {
    let back = b.read_vector_values(whole)?;
    if back.len() != want.len() + 2 * GUARD {
        return Err(format!("{label}: bad readback len"));
    }
    let diff = back[GUARD..GUARD + want.len()].iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let guard = back[..GUARD].iter().chain(&back[GUARD + want.len()..]).filter(|w| w.to_bits() != POISON).count();
    if diff > 0 || guard > 0 {
        if let Some((idx, (g, w))) = back[GUARD..GUARD + want.len()].iter().zip(want).enumerate().find(|(_, (a, b))| a.to_bits() != b.to_bits()) {
            return Err(format!("{label}: diff={diff} guard={guard}; first at {idx}: got={g} ({:#010x}) want={w} ({:#010x})", g.to_bits(), w.to_bits()));
        }
        return Err(format!("{label}: diff={diff} guard={guard}"));
    }
    Ok(want.len())
}

pub struct PanelFixtures<'d> {
    dense: Vec<(usize, usize, usize, Vec<u16>, DeviceVector<'d>)>,
    split_w_vals: Vec<u16>,
    split_w_dev: DeviceVector<'d>,
}

const PANEL_DENSE_SHAPES: &[(usize, usize, usize)] = &[
    (1024usize, 256usize, 16usize),
    (1024, 256, 32),
    (2304, 256, 16),
    (2304, 256, 32),
    (2304, 512, 32),
    (2304, 512, 70),
    (2304, 3072, 32),
    (6552, 512, 32),
    (2305, 192, 29),
    (2600, 192, 13),
    (6552, 192, 32),
    (8600, 192, 16),
    (1024, 192, 125),
    (2304, 192, 70),
    (16_400, 128, 16),
    (16_400, 128, 32),
    // PLAIN_TC_PANEL64 (`33..=128` inputs, `rows > 2560`): 1 and 8 slices, one and two 64-input halves, 4 and 8
    // warps, partial last band (2600 = 162 x 16 + 8), partial last n8 tile (33, 65, 100).
    (6552, 512, 33),
    (6552, 512, 64),
    (6552, 192, 65),
    (8600, 192, 128),
    (2600, 192, 64),
    (16_400, 128, 100),
    (16_400, 128, 64),
    // PLAIN_TC_BIG M-strip serpentine (`k <= 2304` height 16 / `k > 2304` height 8), partial bands & tail M=388.
    (2304, 192, 520),
    (645, 192, 388),
    (520, 192, 650),
    (1300, 128, 257),
];

/// Register all lossless `bf16_panels` matrices while the resident stream is idle (before `enable_resident_launches`).
pub fn prepare_panels<'d>(b: &Backend<'d>) -> Result<PanelFixtures<'d>, String> {
    let mut dense = Vec::with_capacity(PANEL_DENSE_SHAPES.len());
    for &(rows, cols, inputs) in PANEL_DENSE_SHAPES {
        let w_vals: Vec<u16> = (0..rows).flat_map(|r| (0..cols).map(move |c| panel_weight(r, c, rows + cols + inputs))).collect();
        let w_dev = matrix_with_outliers(b, &w_vals, rows, cols, true)?;
        dense.push((rows, cols, inputs, w_vals, w_dev));
    }
    let (rows, cols) = (3072usize, 256usize);
    let split_w_vals: Vec<u16> = (0..rows).flat_map(|r| (0..cols).map(move |c| panel_weight(r, c, 911))).collect();
    let split_w_dev = matrix_with_outliers(b, &split_w_vals, rows, cols, true)?;
    Ok(PanelFixtures { dense, split_w_vals, split_w_dev })
}

pub fn run(b: &Backend<'_>, panels: &PanelFixtures<'_>) -> Result<usize, String> {
    let mut total_words = 0usize;

    // 1. Narrow 1-warp (`rows <= 128, inputs <= 256`) & 4-warp prefill (`inputs > 256`): single, pair, rounded.
    for &(rows, cols, inputs, round_out) in &[
        (1usize, 64usize, 1usize, false),
        (5, 192, 17, false),
        (64, 512, 32, true),
        (64, 320, 31, true),
        (127, 256, 32, false),
        (128, 320, 125, false),
        (64, 320, 300, true),
        (128, 256, 321, false),
        (63, 192, 273, true),
    ] {
        let half = cols / 2;
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 13 + cols + inputs)).collect();
        let x1_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + rows * 5 + 11)).collect();
        let x2_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + rows * 7 + 97)).collect();
        let mut x_cat = Vec::with_capacity(inputs * cols);
        for t in 0..inputs {
            x_cat.extend_from_slice(&x1_vals[t * half..(t + 1) * half]);
            x_cat.extend_from_slice(&x2_vals[t * half..(t + 1) * half]);
        }
        let w_dev = matrix(b, &w_vals)?;
        let x1_dev = b.upload_vector(&x1_vals)?.into_resident_shared()?;
        let x2_dev = b.upload_vector(&x2_vals)?.into_resident_shared()?;
        let whole = poisoned(b, inputs * rows)?;
        b.project_tc_rows_pair_form_into(
            rows, cols, inputs, round_out, &x1_dev, &x2_dev, &w_dev,
            &mut whole.resident_range(GUARD..GUARD + inputs * rows)?,
        )?;
        let want = tc_ref(&w_vals, &x_cat, rows, cols, inputs, round_out);
        total_words += check_buf(b, &whole, &want, &format!("narrow_pair {rows}x{cols}x{inputs} rnd={round_out}"))?;

        if !round_out {
            let x_dev = b.upload_vector(&x_cat)?.into_resident_shared()?;
            let whole_s = poisoned(b, inputs * rows)?;
            b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole_s.resident_range(GUARD..GUARD + inputs * rows)?)?;
            total_words += check_buf(b, &whole_s, &want, &format!("narrow_single {rows}x{cols}x{inputs}"))?;
        }
    }

    // 2. Grouped 3-way dual-input narrow (`GateDown` Q/K/V at C32 + edge counts).
    for &(rows, cols, inputs, groups, round_out) in &[
        (64usize, 512usize, 32usize, 3usize, true),
        (64, 320, 29, 3, true),
        (63, 192, 17, 2, false),
        (128, 256, 32, 3, true),
    ] {
        let half = cols / 2;
        let mut w_devs = Vec::new();
        let mut x1_devs = Vec::new();
        let mut x2_devs = Vec::new();
        let mut want = Vec::with_capacity(groups * inputs * rows);
        for g in 0..groups {
            let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + g * 997 + rows * 3)).collect();
            let x1_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + g * 503 + 17)).collect();
            let x2_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + g * 701 + 41)).collect();
            let mut x_cat = Vec::with_capacity(inputs * cols);
            for t in 0..inputs {
                x_cat.extend_from_slice(&x1_vals[t * half..(t + 1) * half]);
                x_cat.extend_from_slice(&x2_vals[t * half..(t + 1) * half]);
            }
            want.extend(tc_ref(&w_vals, &x_cat, rows, cols, inputs, round_out));
            w_devs.push(matrix(b, &w_vals)?);
            x1_devs.push(b.upload_vector(&x1_vals)?.into_resident_shared()?);
            x2_devs.push(b.upload_vector(&x2_vals)?.into_resident_shared()?);
        }
        let items: Vec<(&DeviceVector<'_>, &DeviceVector<'_>, &DeviceVector<'_>)> =
            (0..groups).map(|g| (&w_devs[g], &x1_devs[g], &x2_devs[g])).collect();
        let whole = poisoned(b, groups * inputs * rows)?;
        b.project_tc_rows_group_pair_into(
            rows, cols, inputs, round_out, &items,
            &mut whole.resident_range(GUARD..GUARD + groups * inputs * rows)?,
        )?;
        total_words += check_buf(b, &whole, &want, &format!("narrow_group g={groups} {rows}x{cols}x{inputs} rnd={round_out}"))?;
    }

    // 3. Short-K (`cols == 64`, `GateOuter`) for both small M (`<= 32`) and prefill (`> 32`).
    for &(rows, inputs) in &[(130usize, 17usize), (257, 32), (1024, 32), (257, 65), (513, 130)] {
        let cols = 64usize;
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 19 + inputs)).collect();
        let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + rows * 11)).collect();
        let w_dev = matrix(b, &w_vals)?;
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let whole = poisoned(b, inputs * rows)?;
        b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
        let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs, false);
        total_words += check_buf(b, &whole, &want, &format!("short_k {rows}x64x{inputs}"))?;
    }

    // 4. Small-M wide + 4w Coop-W (`M=17..=32` & `M=33..=128`) & Med-M `2w 32x32` (`M=129..=256`).
    for &(rows, cols, inputs) in &[
        (1024usize, 256usize, 16usize),
        (1024, 256, 32),
        (1056, 192, 29),
        (2304, 256, 16),
        (2304, 256, 32),
        (2305, 192, 17),
        (2600, 192, 13),
        (2600, 192, 32),
        (1024, 192, 125),
        (2304, 192, 70),
        (3072, 192, 125),
        (1024, 192, 180),
        (16_400, 128, 16),
        (16_400, 128, 32),
    ] {
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 3 + cols)).collect();
        let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + rows + inputs)).collect();
        let w_dev = matrix(b, &w_vals)?;
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let whole = poisoned(b, inputs * rows)?;
        b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
        let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs, false);
        total_words += check_buf(b, &whole, &want, &format!("wide_tc {rows}x{cols}x{inputs}"))?;
    }

    // 5. Overlapped Direct-FP32 `64x128x32` prefill (`512..=1280`: K/V & K+KTap) + `PLAIN_TC_BIG` handoff (`2304`).
    for &(rows, cols, inputs) in &[(513usize, 128usize, 513usize), (1056, 192, 577), (1280, 128, 520), (2304, 128, 520)] {
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 17 + cols)).collect();
        let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + rows * 23 + inputs)).collect();
        let w_dev = matrix(b, &w_vals)?;
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let whole = poisoned(b, inputs * rows)?;
        b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
        let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs, false);
        total_words += check_buf(b, &whole, &want, &format!("big_fp32 {rows}x{cols}x{inputs}"))?;
    }
    // Split output check on `64x128x32` (`K+KTap` shape 1056 = 1024 + 32) and Coop-W `M=125` (`3072 = 1024 + 2048`).
    for &(parts, cols, inputs) in &[([1024usize, 32usize], 128usize, 520usize), ([1024, 2048], 128, 125)] {
        let rows: usize = parts.iter().sum();
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 29 + inputs)).collect();
        let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + rows * 31 + inputs)).collect();
        let w_dev = matrix(b, &w_vals)?;
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let out0 = poisoned(b, inputs * parts[0])?;
        let out1 = poisoned(b, inputs * parts[1])?;
        let mut r0 = out0.resident_range(GUARD..GUARD + inputs * parts[0])?;
        let mut r1 = out1.resident_range(GUARD..GUARD + inputs * parts[1])?;
        b.project_tc_rows_split_into(rows, cols, inputs, &x_dev, &w_dev, &mut [(&mut r0, parts[0]), (&mut r1, parts[1])])?;
        let full = tc_ref(&w_vals, &x_vals, rows, cols, inputs, false);
        let mut want0 = Vec::with_capacity(inputs * parts[0]);
        let mut want1 = Vec::with_capacity(inputs * parts[1]);
        for m in 0..inputs {
            want0.extend_from_slice(&full[m * rows..m * rows + parts[0]]);
            want1.extend_from_slice(&full[m * rows + parts[0]..(m + 1) * rows]);
        }
        total_words += check_buf(b, &out0, &want0, &format!("split_0 {rows}x{cols}x{inputs}"))?;
        total_words += check_buf(b, &out1, &want1, &format!("split_1 {rows}x{cols}x{inputs}"))?;
    }

    // 6. Narrow `TC_FLAG_HALF_IN` (M <= 256) and `TC_MODE_NARROW_HALF` (`BM = 64`, M >= 257: single + dual-input BF16).
    for &(rows, cols, inputs) in &[
        (64usize, 512usize, 32usize),
        (64, 512, 125),
        (63, 192, 17),
        (128, 256, 125),
        (64, 512, 388),
        (128, 256, 388),
        (63, 192, 257),
        (127, 256, 529),
    ] {
        let half = cols / 2;
        let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 7 + cols + inputs)).collect();
        let x1_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + rows * 13 + 3)).collect();
        let x2_vals: Vec<f32> = (0..inputs * half).map(|k| input_val(k + rows * 17 + 71)).collect();
        let mut x_cat = Vec::with_capacity(inputs * cols);
        for t in 0..inputs {
            x_cat.extend_from_slice(&x1_vals[t * half..(t + 1) * half]);
            x_cat.extend_from_slice(&x2_vals[t * half..(t + 1) * half]);
        }
        let w_dev = matrix(b, &w_vals)?;
        let x_half = upload_u32(b, &pack_bf16_words(&x_cat))?;
        let x1_half = upload_u32(b, &pack_bf16_words(&x1_vals))?;
        let x2_half = upload_u32(b, &pack_bf16_words(&x2_vals))?;
        let want = tc_ref(&w_vals, &x_cat, rows, cols, inputs, false);
        let whole_s = poisoned(b, inputs * rows)?;
        b.project_tc_rows_half_in_into(rows, cols, inputs, &x_half, &w_dev, &mut whole_s.resident_range(GUARD..GUARD + inputs * rows)?)?;
        total_words += check_buf(b, &whole_s, &want, &format!("narrow_half_single {rows}x{cols}x{inputs}"))?;
        let whole_p = poisoned(b, inputs * rows)?;
        b.project_tc_rows_pair_half_in_into(rows, cols, inputs, &x1_half, &x2_half, &w_dev, &mut whole_p.resident_range(GUARD..GUARD + inputs * rows)?)?;
        total_words += check_buf(b, &whole_p, &want, &format!("narrow_half_pair {rows}x{cols}x{inputs}"))?;
    }

    // 7. Lossless `bf16_panels` in `PLAIN_TC_ROWS` (Path 4 1w/2w/4w, Path 4b 4w Coop-W M=17..=128) + `PLAIN_TC_BIG`,
    //    with deliberate `RAW` outlier tiles, partial tail bands (`rows % 16 != 0`), split outputs, dual-input pairs,
    //    and 16-row-aligned + unaligned sub-views.
    for (rows, cols, inputs, w_vals, w_dev) in &panels.dense {
        let (rows, cols, inputs) = (*rows, *cols, *inputs);
        let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + rows * 5 + inputs)).collect();
        let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
        let whole = poisoned(b, inputs * rows)?;
        b.project_tc_rows_into(rows, cols, inputs, &x_dev, w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
        let want = tc_ref(w_vals, &x_vals, rows, cols, inputs, false);
        total_words += check_buf(b, &whole, &want, &format!("panel_dense {rows}x{cols}x{inputs}"))?;
    }
    // Registered-panel split outputs (QKV/Coop-W/Big), dual-input pair, and 16-row-aligned / unaligned sub-views.
    {
        let (rows, cols) = (3072usize, 256usize);
        let w_vals = &panels.split_w_vals;
        let w_dev = &panels.split_w_dev;
        for &inputs in &[16usize, 32, 125, 520] {
            let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + inputs * 37)).collect();
            let x_half = upload_u32(b, &pack_bf16_words(&x_vals))?;
            let (p0, p1) = (1024usize, 2048usize);
            let out0 = poisoned(b, inputs * p0)?;
            let out1 = poisoned(b, inputs * p1)?;
            let mut r0 = out0.resident_range(GUARD..GUARD + inputs * p0)?;
            let mut r1 = out1.resident_range(GUARD..GUARD + inputs * p1)?;
            b.project_tc_rows_split_half_in_into(rows, cols, inputs, &x_half, w_dev, &mut [(&mut r0, p0), (&mut r1, p1)])?;
            let full = tc_ref(w_vals, &x_vals, rows, cols, inputs, false);
            let mut want0 = Vec::with_capacity(inputs * p0);
            let mut want1 = Vec::with_capacity(inputs * p1);
            for m in 0..inputs {
                want0.extend_from_slice(&full[m * rows..m * rows + p0]);
                want1.extend_from_slice(&full[m * rows + p0..(m + 1) * rows]);
            }
            total_words += check_buf(b, &out0, &want0, &format!("panel_split0 {rows}x{cols}x{inputs}"))?;
            total_words += check_buf(b, &out1, &want1, &format!("panel_split1 {rows}x{cols}x{inputs}"))?;
        }
        // Dual-input pair with panels.
        for &inputs in &[16usize, 32, 64, 100] {
            let half = cols / 2;
            let x1: Vec<f32> = (0..inputs * half).map(|k| input_val(k + 101)).collect();
            let x2: Vec<f32> = (0..inputs * half).map(|k| input_val(k + 202)).collect();
            let mut x_cat = Vec::with_capacity(inputs * cols);
            for t in 0..inputs { x_cat.extend_from_slice(&x1[t * half..(t + 1) * half]); x_cat.extend_from_slice(&x2[t * half..(t + 1) * half]); }
            let x1_d = b.upload_vector(&x1)?.into_resident_shared()?;
            let x2_d = b.upload_vector(&x2)?.into_resident_shared()?;
            let whole = poisoned(b, inputs * rows)?;
            b.project_tc_rows_pair_into(rows, cols, inputs, &x1_d, &x2_d, w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
            let want = tc_ref(w_vals, &x_cat, rows, cols, inputs, false);
            total_words += check_buf(b, &whole, &want, &format!("panel_pair {rows}x{cols}x{inputs}"))?;
        }
        // 16-row-aligned sub-view (hits `panel_at`) + non-16-aligned sub-view (clean fallback).
        for &(row_off, sub_rows, inputs) in &[(32usize, 1056usize, 32usize), (32, 1056, 16), (8, 1024, 32), (32, 2592, 64), (16, 2608, 128)] {
            let sub_w = w_dev.resident_range(row_off * cols / 2..(row_off + sub_rows) * cols / 2)?;
            let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input_val(k + row_off * 19 + inputs)).collect();
            let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
            let whole = poisoned(b, inputs * sub_rows)?;
            b.project_tc_rows_into(sub_rows, cols, inputs, &x_dev, &sub_w, &mut whole.resident_range(GUARD..GUARD + inputs * sub_rows)?)?;
            let want = tc_ref(&w_vals[row_off * cols..(row_off + sub_rows) * cols], &x_vals, sub_rows, cols, inputs, false);
            total_words += check_buf(b, &whole, &want, &format!("panel_sub off={row_off} {sub_rows}x{cols}x{inputs}"))?;
        }
    }

    println!("DENSE TC WAVE1+2 BIT-EXACT PASS words={total_words}");
    Ok(total_words)
}
