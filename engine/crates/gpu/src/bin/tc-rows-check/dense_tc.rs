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

fn matrix<'d>(b: &Backend<'d>, values: &[u16]) -> Result<DeviceVector<'d>, String> {
    let mut v = b.vector(values.len() / 2)?;
    let bytes: Vec<u8> = values.iter().flat_map(|w| w.to_le_bytes()).collect();
    b.write_vector_bytes(&mut v, &bytes)?;
    v.into_resident_shared()
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

pub fn run(b: &Backend<'_>) -> Result<usize, String> {
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

    println!("DENSE TC WAVE1+2 BIT-EXACT PASS words={total_words}");
    Ok(total_words)
}
