//! Device correctness, guard-word, bounds-refusal, and memory-safety check for `PLAIN_TC_ROWS` (kind 103)
//! and `MOE_GROUP_ROUTE` (kind 104) in `kernels-inkling`.
//!
//! Compares every output word BIT-FOR-BIT against `tc_rows::tc_word` (exact CPU integer model of SM121a
//! `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` in strictly ascending K order), with poison
//! guard words on both sides of every output buffer. Designed to run cleanly under
//! `compute-sanitizer --tool memcheck tc-rows-check`.
#[cfg(target_os = "linux")]
mod split_window;
#[cfg(target_os = "linux")]
mod fast_ops;
#[cfg(target_os = "linux")]
mod dense_tc;
#[cfg(target_os = "linux")]
mod prefill_moe;
#[cfg(target_os = "linux")]
mod check {
    use spark_gpu_probe::{driver::{Backend, DeviceVector}, resident_launch::Kind, tc_rows};

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
    pub(super) fn input(k: usize) -> f32 { ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0 }
    pub(super) fn matrix<'d>(b: &Backend<'d>, values: &[u16]) -> Result<DeviceVector<'d>, String> {
        let mut v = b.vector(values.len() / 2)?;
        let bytes: Vec<u8> = values.iter().flat_map(|w| w.to_le_bytes()).collect();
        b.write_vector_bytes(&mut v, &bytes)?;
        v.into_resident_shared()
    }
    fn poisoned<'d>(b: &Backend<'d>, words: usize) -> Result<DeviceVector<'d>, String> {
        b.upload_vector(&vec![f32::from_bits(POISON); words + 2 * GUARD])?.into_resident_shared()
    }
    fn tc_ref(values: &[u16], x: &[f32], rows: usize, cols: usize, inputs: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(inputs * rows);
        for m in 0..inputs {
            let xm = &x[m * cols..(m + 1) * cols];
            for n in 0..rows {
                out.push(tc_rows::tc_word(&values[n * cols..(n + 1) * cols], xm));
            }
        }
        out
    }
    fn check_buf(b: &Backend<'_>, whole: &DeviceVector<'_>, want: &[f32], label: &str) -> Result<usize, String> {
        let back = b.read_vector_values(whole)?;
        if back.len() != want.len() + 2 * GUARD { return Err(format!("{label}: bad readback len")); }
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

    pub fn run() -> Result<(), String> {
        spark_gpu_probe::driver::with_backend(0, spark_gpu_probe::layer_ptx()?, usize::MAX, |b| {
            let panel_fixtures = super::dense_tc::prepare_panels(b)?;
            let queue = b.resident_queue()?;
            b.enable_resident_launches()?;
            let mut total_words = 0usize;

            // 1. DENSE shapes (edge tiles: 1 row, non-multiple of 128, non-multiple of 64, model-like dims).
            for &(rows, cols, inputs) in &[
                (1usize, 64usize, 1usize),
                (24, 128, 7),
                (64, 192, 64),
                (129, 128, 65),
                (256, 256, 130),
                (384, 128, 16),
                (130, 128, 17),
                (257, 192, 24),
                (384, 128, 32),
                (20_058, 64, 32),
            ] {
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 7 + inputs)).collect();
                let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + rows * 3)).collect();
                let w_dev = matrix(b, &w_vals)?;
                let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                total_words += check_buf(b, &whole, &want, &format!("dense {rows}x{cols}x{inputs}"))?;
            }
            println!("TC ROWS DENSE PASS words={total_words}");

            // 2. STRIDED (parts writing columns into one output).
            {
                let (rows, cols, inputs, part) = (384usize, 128usize, 65usize, 256usize);
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 99)).collect();
                let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + 55)).collect();
                let w_dev = matrix(b, &w_vals)?;
                let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                for (first, cnt) in [(0usize, part), (part, rows - part)] {
                    let w_part = w_dev.resident_range(first * cols / 2..(first + cnt) * cols / 2)?;
                    b.project_tc_rows_strided_into(cnt, cols, inputs, rows, &x_dev, &w_part,
                        &mut whole.resident_range(GUARD + first..GUARD + first + (inputs - 1) * rows + cnt)?)?;
                }
                let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                total_words += check_buf(b, &whole, &want, "strided 384x128x65")?;
            }
            println!("TC ROWS STRIDED PASS");

            // 3. FUSED SPLIT OUTPUTS (6-way attention layout with Sink=24 non-multiple of 128, small 32x128 & 64x64!).
            for &inputs in &[24usize, 32, 70] {
                let segs = [128usize, 32, 24, 128, 64, 64];
                let rows: usize = segs.iter().sum();
                let cols = 128usize;
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 313 + inputs)).collect();
                let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + 17 + inputs)).collect();
                let w_dev = matrix(b, &w_vals)?;
                let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                let wholes: Vec<DeviceVector<'_>> = segs.iter().map(|&s| poisoned(b, inputs * s)).collect::<Result<_, _>>()?;
                let mut views: Vec<DeviceVector<'_>> = wholes.iter().zip(&segs).map(|(w, &s)| w.resident_range(GUARD..GUARD + inputs * s)).collect::<Result<_, _>>()?;
                {
                    let mut refs: Vec<(&mut DeviceVector<'_>, usize)> = views.iter_mut().zip(segs.iter().copied()).collect();
                    b.project_tc_rows_split_into(rows, cols, inputs, &x_dev, &w_dev, &mut refs)?;
                }
                let full_want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                let mut col_off = 0usize;
                for (si, (&seg_w, whole)) in segs.iter().zip(&wholes).enumerate() {
                    let mut want = Vec::with_capacity(inputs * seg_w);
                    for m in 0..inputs {
                        want.extend_from_slice(&full_want[m * rows + col_off..m * rows + col_off + seg_w]);
                    }
                    total_words += check_buf(b, whole, &want, &format!("split seg {si} (w={seg_w}, in={inputs})"))?;
                    col_off += seg_w;
                }
            }
            println!("TC ROWS SPLIT PASS");

            // 3a. BIG-TILE DENSE (PLAIN_TC_BIG, `tc_rows::tc_big_wanted`): partial row and column tiles, 2 to 18 steps.
            {
                let before = total_words;
                for &(rows, cols, inputs) in &[(520usize, 192usize, 777usize), (640, 64, 513), (768, 576, 512), (1032, 128, 1100)] {
                    assert!(tc_rows::tc_big_wanted(rows, inputs));
                    let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 5 + inputs)).collect();
                    let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + rows * 11)).collect();
                    let w_dev = matrix(b, &w_vals)?;
                    let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                    let whole = poisoned(b, inputs * rows)?;
                    b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                    let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                    total_words += check_buf(b, &whole, &want, &format!("big dense {rows}x{cols}x{inputs}"))?;
                }
                // Strided parts: the first (640 rows) takes the big tile, the second (384 rows) the 64 x 64 tile.
                let (rows, cols, inputs, part) = (1024usize, 128usize, 530usize, 640usize);
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 41)).collect();
                let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + 73)).collect();
                let w_dev = matrix(b, &w_vals)?;
                let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                for (first, cnt) in [(0usize, part), (part, rows - part)] {
                    let w_part = w_dev.resident_range(first * cols / 2..(first + cnt) * cols / 2)?;
                    b.project_tc_rows_strided_into(cnt, cols, inputs, rows, &x_dev, &w_part,
                        &mut whole.resident_range(GUARD + first..GUARD + first + (inputs - 1) * rows + cnt)?)?;
                }
                let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                total_words += check_buf(b, &whole, &want, "big strided 1024x128x530")?;
                // Split outputs with a 24-wide segment and segment edges inside column tiles.
                let segs = [256usize, 24, 128, 104, 64];
                let rows: usize = segs.iter().sum();
                let (cols, inputs) = (128usize, 600usize);
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 517)).collect();
                let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + 23)).collect();
                let w_dev = matrix(b, &w_vals)?;
                let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                let wholes: Vec<DeviceVector<'_>> = segs.iter().map(|&s| poisoned(b, inputs * s)).collect::<Result<_, _>>()?;
                let mut views: Vec<DeviceVector<'_>> = wholes.iter().zip(&segs).map(|(w, &s)| w.resident_range(GUARD..GUARD + inputs * s)).collect::<Result<_, _>>()?;
                {
                    let mut refs: Vec<(&mut DeviceVector<'_>, usize)> = views.iter_mut().zip(segs.iter().copied()).collect();
                    b.project_tc_rows_split_into(rows, cols, inputs, &x_dev, &w_dev, &mut refs)?;
                }
                let full_want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                let mut col_off = 0usize;
                for (si, (&seg_w, whole)) in segs.iter().zip(&wholes).enumerate() {
                    let mut want = Vec::with_capacity(inputs * seg_w);
                    for m in 0..inputs {
                        want.extend_from_slice(&full_want[m * rows + col_off..m * rows + col_off + seg_w]);
                    }
                    total_words += check_buf(b, whole, &want, &format!("big split seg {si} (w={seg_w})"))?;
                    col_off += seg_w;
                }
                println!("TC BIG PASS words={}", total_words - before);
            }

            // 3b'. NARROW FORM (PLAIN_TC_BIG, TC_MODE_NARROW): FP32 input rounded as it is read, no pack. One input
            // (router shape) and the GateDown pair; idle warps (24 and 64 rows), partial row blocks, 2 to 16 steps.
            {
                let before = total_words;
                for &(rows, cols, inputs) in &[(128usize, 256usize, 600usize), (64, 192, 517), (24, 64, 530), (100, 512, 513)] {
                    assert!(tc_rows::tc_narrow_wanted(rows, inputs));
                    let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + rows * 13 + inputs)).collect();
                    let x_vals: Vec<f32> = (0..inputs * cols).map(|k| input(k + rows * 19)).collect();
                    let w_dev = matrix(b, &w_vals)?;
                    let x_dev = b.upload_vector(&x_vals)?.into_resident_shared()?;
                    let whole = poisoned(b, inputs * rows)?;
                    b.project_tc_rows_into(rows, cols, inputs, &x_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                    let want = tc_ref(&w_vals, &x_vals, rows, cols, inputs);
                    total_words += check_buf(b, &whole, &want, &format!("narrow {rows}x{cols}x{inputs}"))?;
                }
                for &(rows, cols, inputs) in &[(64usize, 512usize, 700usize), (100, 256, 513)] {
                    let half = cols / 2;
                    let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 5 * rows + inputs)).collect();
                    let x1_vals: Vec<f32> = (0..inputs * half).map(|k| input(k + 61)).collect();
                    let x2_vals: Vec<f32> = (0..inputs * half).map(|k| input(k + 3 * inputs + 7)).collect();
                    let mut x_cat = Vec::with_capacity(inputs * cols);
                    for r in 0..inputs {
                        x_cat.extend_from_slice(&x1_vals[r * half..(r + 1) * half]);
                        x_cat.extend_from_slice(&x2_vals[r * half..(r + 1) * half]);
                    }
                    let w_dev = matrix(b, &w_vals)?;
                    let x1_dev = b.upload_vector(&x1_vals)?.into_resident_shared()?;
                    let x2_dev = b.upload_vector(&x2_vals)?.into_resident_shared()?;
                    let whole = poisoned(b, inputs * rows)?;
                    b.project_tc_rows_pair_into(rows, cols, inputs, &x1_dev, &x2_dev, &w_dev, &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                    let want = tc_ref(&w_vals, &x_cat, rows, cols, inputs);
                    total_words += check_buf(b, &whole, &want, &format!("narrow pair {rows}x{cols}x{inputs}"))?;
                }
                println!("TC NARROW PASS words={}", total_words - before);
            }

            // 3b. DUAL-INPUT PLAIN_TC_ROWS PAIR (GateDown with two [inputs, cols/2] inputs, small 32x128 & 64x64).
            for &inputs in &[17usize, 32, 70] {
                let (rows, cols) = (128usize, 256usize);
                let half = cols / 2;
                let w_vals: Vec<u16> = (0..rows * cols).map(|i| weight(i + 409 + inputs)).collect();
                let x1_vals: Vec<f32> = (0..inputs * half).map(|k| input(k + 29 + inputs)).collect();
                let x2_vals: Vec<f32> = (0..inputs * half).map(|k| input(k + 91 + inputs)).collect();
                let mut x_cat = Vec::with_capacity(inputs * cols);
                for r in 0..inputs {
                    x_cat.extend_from_slice(&x1_vals[r * half..(r + 1) * half]);
                    x_cat.extend_from_slice(&x2_vals[r * half..(r + 1) * half]);
                }
                let w_dev = matrix(b, &w_vals)?;
                let x1_dev = b.upload_vector(&x1_vals)?.into_resident_shared()?;
                let x2_dev = b.upload_vector(&x2_vals)?.into_resident_shared()?;
                let whole = poisoned(b, inputs * rows)?;
                b.project_tc_rows_pair_into(rows, cols, inputs, &x1_dev, &x2_dev, &w_dev,
                    &mut whole.resident_range(GUARD..GUARD + inputs * rows)?)?;
                let want = tc_ref(&w_vals, &x_cat, rows, cols, inputs);
                total_words += check_buf(b, &whole, &want, &format!("tc rows pair 128x256x{inputs}"))?;
            }
            println!("TC ROWS PAIR PASS");

            // 3c. MOE_GROUP_ROUTE tables against the CPU order: picks grouped by expert in ascending pick order, route
            // tiles of TC_BM picks. Slices that end inside a 32-pick group (4093 rows) and the model shape (128 experts).
            for &(rows, picks, experts) in &[(4093usize, 8usize, 128usize), (130, 8, 16), (37, 4, 9)] {
                let (all, each) = (rows * picks, 2 * picks);
                let mut route_host = vec![0.0f32; rows * each];
                for r in 0..rows {
                    let first = (r.wrapping_mul(2_654_435_761) >> 7) % experts;
                    for s in 0..picks {
                        route_host[r * each + s] = ((first + s * 17) % experts) as f32;
                        route_host[r * each + picks + s] = (s + 1) as f32 * 0.125;
                    }
                }
                let mut counts = vec![0usize; experts];
                for k in 0..all { counts[route_host[(k / picks) * each + k % picks] as usize] += 1; }
                let mut next = vec![0usize; experts];
                let mut meta = vec![0u32];
                let mut at = 0usize;
                for e in 0..experts {
                    next[e] = at;
                    let mut t = 0;
                    while t < counts[e] { meta.extend([e as u32, (at + t) as u32, (counts[e] - t).min(tc_rows::TC_BM) as u32]); t += tc_rows::TC_BM; }
                    at += counts[e];
                }
                meta[0] = ((meta.len() - 1) / 3) as u32;
                let (mut in_rows, mut out_slots) = (vec![0u32; all], vec![0u32; all]);
                for k in 0..all {
                    let e = route_host[(k / picks) * each + k % picks] as usize;
                    in_rows[next[e]] = (k / picks) as u32;
                    out_slots[next[e]] = k as u32;
                    next[e] += 1;
                }
                let route_dev = b.upload_vector(&route_host)?.into_resident_shared()?;
                let w_out = poisoned(b, all)?;
                let plan = b.moe_group_route_into(rows, each, picks, experts, &route_dev, &mut w_out.resident_range(GUARD..GUARD + all)?)?;
                let [d_in, d_out, d_meta] = plan.route_tables();
                let words = |v: &DeviceVector<'_>| -> Result<Vec<u32>, String> { Ok(b.read_vector_values(v)?.iter().map(|w| w.to_bits()).collect()) };
                let (got_in, got_out, got_meta) = (words(d_in)?, words(d_out)?, words(d_meta)?);
                if got_in[..all] != in_rows[..] || got_out[..all] != out_slots[..] || got_meta.len() < meta.len() || got_meta[..meta.len()] != meta[..] {
                    return Err(format!("moe route tables {rows}x{picks}x{experts}: differ from the CPU order"));
                }
                total_words += 2 * all + meta.len();
            }
            println!("TC MOE ROUTE TABLES PASS");

            // 4. DEVICE-ROUTED MOE (MOE_GROUP_ROUTE + GateUp + Down, zero CPU sync!).
            {
                let (rows, picks, experts, cols, inner) = (130usize, 8usize, 16usize, 128usize, 128usize);
                let all = rows * picks;
                let each = 2 * picks;
                // Deterministic route: each row picks 8 distinct experts with known weights.
                let mut route_host = vec![0.0f32; rows * each];
                let mut want_weights = vec![0.0f32; all];
                for r in 0..rows {
                    for s in 0..picks {
                        let e = (r * 3 + s * 2) % experts;
                        let w = ((r * 7 + s * 5) % 23) as f32 * 0.0625 + 0.125;
                        route_host[r * each + s] = e as f32;
                        route_host[r * each + picks + s] = w;
                        want_weights[r * picks + s] = w;
                    }
                }
                let route_dev = b.upload_vector(&route_host)?.into_resident_shared()?;
                let w_out = poisoned(b, all)?;
                let plan = b.moe_group_route_into(rows, each, picks, experts, &route_dev,
                    &mut w_out.resident_range(GUARD..GUARD + all)?)?;
                check_buf(b, &w_out, &want_weights, "moe route weights")?;

                // GateUp stack: [experts, inner, cols] BF16, input z: [rows, cols] FP32.
                let stack_vals: Vec<u16> = (0..experts * inner * cols).map(|i| weight(i + 777)).collect();
                let stack_dev = matrix(b, &stack_vals)?;
                let z_vals: Vec<f32> = (0..rows * cols).map(|k| input(k + 41)).collect();
                let z_dev = b.upload_vector(&z_vals)?.into_resident_shared()?;
                let halves = poisoned(b, all * inner)?;
                b.project_tc_moe_into(inner, cols, &stack_dev, &plan, &z_dev,
                    &mut halves.resident_range(GUARD..GUARD + all * inner)?, false)?;

                // Down stack: [experts, cols, inner] BF16, input halves contiguous -> output scattered via out_slots!
                let down_vals: Vec<u16> = (0..experts * cols * inner).map(|i| weight(i + 1337)).collect();
                let down_dev = matrix(b, &down_vals)?;
                let downs = poisoned(b, all * cols)?;
                b.project_tc_moe_into(cols, inner, &down_dev, &plan,
                    &halves.resident_range(GUARD..GUARD + all * inner)?,
                    &mut downs.resident_range(GUARD..GUARD + all * cols)?, true)?;

                // Reference: for every pick p = r * picks + s, compute Down[e] * (GateUp[e] * z[r]).
                let mut want_downs = vec![0.0f32; all * cols];
                for r in 0..rows {
                    let xr = &z_vals[r * cols..(r + 1) * cols];
                    for s in 0..picks {
                        let e = (r * 3 + s * 2) % experts;
                        let gu = &stack_vals[e * inner * cols..(e + 1) * inner * cols];
                        let h: Vec<f32> = (0..inner).map(|n| tc_rows::tc_word(&gu[n * cols..(n + 1) * cols], xr)).collect();
                        let dn = &down_vals[e * cols * inner..(e + 1) * cols * inner];
                        let p = r * picks + s;
                        for n in 0..cols {
                            want_downs[p * cols + n] = tc_rows::tc_word(&dn[n * inner..(n + 1) * inner], &h);
                        }
                    }
                }
                total_words += check_buf(b, &downs, &want_downs, "moe device-routed GateUp+Down")?;
            }
            println!("TC ROWS MOE PASS");

            // 4b. DEVICE-ROUTED PACKED MOE (PLAIN_PACKED_TC_ROWS, kind 106): NVFP4 (wide=false, pairs=true) + MXFP8 (wide=true, pairs=false).
            {
                use spark_gpu_probe::packed_product;
                let (rows, picks, experts, cols, inner) = (130usize, 8usize, 16usize, 128usize, 130usize);
                let all = rows * picks;
                let each = 2 * picks;
                let mut route_host = vec![0.0f32; rows * each];
                for r in 0..rows {
                    for s in 0..picks {
                        let e = (r * 5 + s * 3) % experts;
                        let w = ((r * 11 + s * 7) % 19) as f32 * 0.0625 + 0.125;
                        route_host[r * each + s] = e as f32;
                        route_host[r * each + picks + s] = w;
                    }
                }
                let route_dev = b.upload_vector(&route_host)?.into_resident_shared()?;
                let w_out = poisoned(b, all)?;
                let plan = b.moe_group_route_into(rows, each, picks, experts, &route_dev,
                    &mut w_out.resident_range(GUARD..GUARD + all)?)?;

                // Build tiled packed NVFP4 stack (GateUp: inner x cols, pairs=true) & MXFP8 stack (Down: cols x cols, pairs=false)
                // alongside their exact widened BF16 twins to prove 100% bit-for-bit parity.
                let make_packed_stack = |n_rows: usize, n_cols: usize, wide: bool, pairs: bool, seed: usize| -> (Vec<u8>, Vec<Vec<u16>>) {
                    let parts = packed_product::layout(n_rows, n_cols, wide);
                    let mut stack_bytes = vec![0u8; experts * parts.bytes];
                    let mut widened = Vec::with_capacity(experts);
                    for e in 0..experts {
                        let unit = &mut stack_bytes[e * parts.bytes..(e + 1) * parts.bytes];
                        // Build deterministic table for this expert.
                        let mut nv_table = [0u16; 256 * 16];
                        let mut mx_table = [0.0f32; 256];
                        if wide {
                            for code in 0..256usize {
                                let v = ((code as i32 - 128) as f32) * 0.03125 * (1.0 + (e as f32) * 0.01);
                                mx_table[code] = v;
                                unit[parts.table + code * 4..parts.table + code * 4 + 4].copy_from_slice(&v.to_le_bytes());
                            }
                        } else {
                            for sc in 0..256usize {
                                for nib in 0..16usize {
                                    let idx = sc * 16 + nib;
                                    let val = weight(idx + seed + e * 97);
                                    nv_table[idx] = val;
                                    unit[parts.table + idx * 2..parts.table + idx * 2 + 2].copy_from_slice(&val.to_le_bytes());
                                }
                            }
                        }
                        let mut wide_e = vec![0u16; n_rows * n_cols];
                        for r in 0..n_rows {
                            let out_r = packed_product::output_row(r, n_rows, pairs);
                            for c in 0..n_cols {
                                let stage = c / packed_product::STAGE;
                                let lane = c % packed_product::STAGE;
                                let c_off = packed_product::code_copy(r, stage, 0, n_cols, wide);
                                let s_off = packed_product::scale_copy(r, stage, n_cols, wide);
                                if wide {
                                    let code_b = ((r * 31 + c * 17 + e * 13 + seed) & 255) as u8;
                                    let sc_b = (120 + ((r * 7 + (c / 32) * 5 + e * 3) % 15)) as u8;
                                    unit[c_off + lane] = code_b;
                                    unit[parts.scales + s_off + (c / 32) % 4] = sc_b;
                                    let w_f32 = packed_product::round_bf16(mx_table[code_b as usize] * packed_product::power(sc_b as u32));
                                    wide_e[out_r * n_cols + c] = (w_f32.to_bits() >> 16) as u16;
                                } else {
                                    let nib = ((r * 29 + c * 19 + e * 11 + seed) & 15) as u8;
                                    let sc_b = ((r * 13 + (c / 16) * 7 + e * 5 + seed) & 255) as u8;
                                    if c % 2 == 0 {
                                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0xf0) | nib;
                                    } else {
                                        unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0x0f) | (nib << 4);
                                    }
                                    unit[parts.scales + s_off + lane / 16] = sc_b;
                                    wide_e[out_r * n_cols + c] = nv_table[sc_b as usize * 16 + nib as usize];
                                }
                            }
                        }
                        widened.push(wide_e);
                    }
                    (stack_bytes, widened)
                };
                let upload_bytes = |bytes: &[u8]| -> Result<DeviceVector<'_>, String> {
                    let mut v = b.vector(bytes.len() / 4)?;
                    b.write_vector_bytes(&mut v, bytes)?;
                    v.into_resident_shared()
                };

                // NVFP4 GateUp (`inner = 130`, non-multiple of 128, even for `pairs = true`, `cols = 128`).
                let (nv_bytes, nv_wide) = make_packed_stack(inner, cols, false, true, 101);
                let nv_dev = upload_bytes(&nv_bytes)?;
                let z_vals: Vec<f32> = (0..rows * cols).map(|k| input(k + 83)).collect();
                let z_dev = b.upload_vector(&z_vals)?.into_resident_shared()?;
                let nv_out = poisoned(b, all * inner)?;
                b.project_tc_packed_moe_into(inner, cols, false, true, &nv_dev, &plan, &z_dev,
                    &mut nv_out.resident_range(GUARD..GUARD + all * inner)?, false)?;
                let mut grouped: Vec<_> = (0..all).collect();
                grouped.sort_by_key(|&p| ((p / picks * 5 + p % picks * 3) % experts, p));
                let mut want_nv = vec![0.0f32; all * inner];
                for (pos, &p) in grouped.iter().enumerate() {
                    let (r, s) = (p / picks, p % picks);
                    let e = (r * 5 + s * 3) % experts;
                    let xr = &z_vals[r * cols..(r + 1) * cols];
                    for n in 0..inner {
                        want_nv[pos * inner + n] = tc_rows::tc_word(&nv_wide[e][n * cols..(n + 1) * cols], xr);
                    }
                }
                total_words += check_buf(b, &nv_out, &want_nv, "packed tc moe NVFP4 pairs")?;

                // Gated NVFP4 GateUp: the words of the product above followed by `CappedGate` (exact reference).
                {
                    use spark_gpu_probe::causal_ops::{self, Op, Shape};
                    let half = inner / 2;
                    for (gcap, ucap) in [(4.0f32, 25.0f32), (0.75, 3.0)] {
                        let caps = [gcap, ucap, 1.0 / gcap, 1.0 / ucap];
                        let gated_out = poisoned(b, all * half)?;
                        b.project_tc_packed_gate_into(inner, cols, false, &nv_dev, &plan, &z_dev, caps,
                            &mut gated_out.resident_range(GUARD..GUARD + all * half)?)?;
                        let shape = Shape { op: Op::CappedGate, heads: 0, dim: half as u32, taps: 0, rows: all as u32, cols: 0, epsilon: gcap, bound: ucap };
                        let want = causal_ops::reference(shape, [&want_nv, &[], &[], &[]], &caps[2..])?;
                        total_words += check_buf(b, &gated_out, &want, &format!("packed tc moe NVFP4 gated caps={gcap},{ucap}"))?;
                    }
                }

                // MXFP8 Down (`133 x 256`, non-multiple of 128 rows, 256 cols = 2 stages of 128, `pairs = false`).
                let (dn_rows, dn_cols) = (133usize, 256usize);
                let (mx_bytes, mx_wide) = make_packed_stack(dn_rows, dn_cols, true, false, 202);
                let mx_dev = upload_bytes(&mx_bytes)?;
                let a_vals: Vec<f32> = (0..all * dn_cols).map(|k| input(k + 191)).collect();
                let a_dev = b.upload_vector(&a_vals)?.into_resident_shared()?;
                let mx_out = poisoned(b, all * dn_rows)?;
                b.project_tc_packed_moe_into(dn_rows, dn_cols, true, false, &mx_dev, &plan, &a_dev,
                    &mut mx_out.resident_range(GUARD..GUARD + all * dn_rows)?, true)?;
                let mut want_mx = vec![0.0f32; all * dn_rows];
                for (pos, &p) in grouped.iter().enumerate() {
                    let r = p / picks;
                    let s = p % picks;
                    let e = (r * 5 + s * 3) % experts;
                    let xp = &a_vals[pos * dn_cols..(pos + 1) * dn_cols];
                    for n in 0..dn_rows {
                        want_mx[p * dn_rows + n] = tc_rows::tc_word(&mx_wide[e][n * dn_cols..(n + 1) * dn_cols], xp);
                    }
                }
                total_words += check_buf(b, &mx_out, &want_mx, "packed tc moe MXFP8")?;

                // Half Down (`HALF_OUT`, even rows `134`): each BF16 value is `rn` of the FP32 word, two a word, and the
                // half fold on them gives the words of the FP32 fold on the FP32 downs.
                {
                    let rn = |x: f32| f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) << 16);
                    let hd_rows = 134usize;
                    let (hd_bytes, hd_wide) = make_packed_stack(hd_rows, dn_cols, true, false, 303);
                    let hd_dev = upload_bytes(&hd_bytes)?;
                    let full = poisoned(b, all * hd_rows)?;
                    b.project_tc_packed_moe_into(hd_rows, dn_cols, true, false, &hd_dev, &plan, &a_dev,
                        &mut full.resident_range(GUARD..GUARD + all * hd_rows)?, true)?;
                    let mut want_full = vec![0.0f32; all * hd_rows];
                    for (pos, &p) in grouped.iter().enumerate() {
                        let e = (p / picks * 5 + p % picks * 3) % experts;
                        let xp = &a_vals[pos * dn_cols..(pos + 1) * dn_cols];
                        for n in 0..hd_rows {
                            want_full[p * hd_rows + n] = tc_rows::tc_word(&hd_wide[e][n * dn_cols..(n + 1) * dn_cols], xp);
                        }
                    }
                    total_words += check_buf(b, &full, &want_full, "packed tc moe MXFP8 even rows")?;
                    let half_out = poisoned(b, all * hd_rows / 2)?;
                    b.project_tc_packed_down_half_into(hd_rows, dn_cols, true, &hd_dev, &plan, &a_dev,
                        &mut half_out.resident_range(GUARD..GUARD + all * hd_rows / 2)?)?;
                    let want_half: Vec<f32> = want_full.as_chunks::<2>().0.iter()
                        .map(|p| f32::from_bits((rn(p[0]).to_bits() >> 16) | (rn(p[1]).to_bits() & 0xffff_0000))).collect();
                    total_words += check_buf(b, &half_out, &want_half, "packed tc moe MXFP8 half down")?;
                    let weights = w_out.resident_range(GUARD..GUARD + all)?;
                    let fold_full = poisoned(b, rows * hd_rows)?;
                    b.fast_route_fold_into(rows, picks, hd_rows, &weights, &full.resident_range(GUARD..GUARD + all * hd_rows)?,
                        &mut fold_full.resident_range(GUARD..GUARD + rows * hd_rows)?)?;
                    let fold_want = b.read_vector_values(&fold_full)?[GUARD..GUARD + rows * hd_rows].to_vec();
                    let fold_half = poisoned(b, rows * hd_rows)?;
                    b.fast_route_fold_half_into(rows, picks, hd_rows, &weights, &half_out.resident_range(GUARD..GUARD + all * hd_rows / 2)?,
                        &mut fold_half.resident_range(GUARD..GUARD + rows * hd_rows)?)?;
                    total_words += check_buf(b, &fold_half, &fold_want, "route fold of the half down")?;
                    println!("TC ROWS PACKED MOE HALF DOWN FOLD BIT-EXACT PASS rows={rows} picks={picks} width={hd_rows}");
                }
                // In-place tagged NVFP4 MMA-lane panel stack (`rows = 192`, `cols = 256`, `pairs = true`, ungated + gated)
                // bit-for-bit identical to the widened BF16 reference and guard words untouched.
                {
                    let (p_rows, p_cols) = (192usize, 256usize);
                    let parts = packed_product::layout(p_rows, p_cols, false);
                    let mut raw_nv = vec![0u8; experts * parts.bytes];
                    let mut wide_nv = Vec::with_capacity(experts);
                    for e in 0..experts {
                        let unit = &mut raw_nv[e * parts.bytes..(e + 1) * parts.bytes];
                        let mut nv_table = [0u16; 256 * 16];
                        for sc in 0..256usize {
                            for nib in 0..16usize {
                                let idx = sc * 16 + nib;
                                let val = weight(idx + 404 + e * 97);
                                nv_table[idx] = val;
                                unit[parts.table + idx * 2..parts.table + idx * 2 + 2].copy_from_slice(&val.to_le_bytes());
                            }
                        }
                        let mut wide_e = vec![0u16; p_rows * p_cols];
                        for r in 0..p_rows {
                            let out_r = packed_product::output_row(r, p_rows, true);
                            for c in 0..p_cols {
                                let stage = c / packed_product::STAGE;
                                let lane = c % packed_product::STAGE;
                                let c_off = packed_product::code_copy(r, stage, 0, p_cols, false);
                                let s_off = packed_product::scale_copy(r, stage, p_cols, false);
                                let nib = ((r * 29 + c * 19 + e * 11 + 404) & 15) as u8;
                                let mut sc_b = ((r * 13 + (c / 16) * 7 + e * 5 + 404) & 255) as u8;
                                if (sc_b & 0x7f) == 0x7f { sc_b ^= 1; }
                                if c % 2 == 0 {
                                    unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0xf0) | nib;
                                } else {
                                    unit[c_off + lane / 2] = (unit[c_off + lane / 2] & 0x0f) | (nib << 4);
                                }
                                unit[parts.scales + s_off + lane / 16] = sc_b;
                                wide_e[out_r * p_cols + c] = nv_table[sc_b as usize * 16 + nib as usize];
                            }
                        }
                        wide_nv.push(wide_e);
                    }
                    let mut panel_nv = Vec::with_capacity(raw_nv.len());
                    for unit in raw_nv.chunks_exact(parts.bytes) {
                        panel_nv.extend(spark_gpu_probe::decode_panels::pack_expert_tagged(unit, p_rows, p_cols));
                    }
                    let panel_dev = upload_bytes(&panel_nv)?;
                    let zp_vals: Vec<f32> = (0..rows * p_cols).map(|k| input(k + 271)).collect();
                    let zp_dev = b.upload_vector(&zp_vals)?.into_resident_shared()?;
                    let p_out = poisoned(b, all * p_rows)?;
                    b.project_tc_packed_moe_into(p_rows, p_cols, false, true, &panel_dev, &plan, &zp_dev,
                        &mut p_out.resident_range(GUARD..GUARD + all * p_rows)?, false)?;
                    let mut want_p = vec![0.0f32; all * p_rows];
                    for (pos, &p) in grouped.iter().enumerate() {
                        let (r, s) = (p / picks, p % picks);
                        let e = (r * 5 + s * 3) % experts;
                        let xr = &zp_vals[r * p_cols..(r + 1) * p_cols];
                        for n in 0..p_rows {
                            want_p[pos * p_rows + n] = tc_rows::tc_word(&wide_nv[e][n * p_cols..(n + 1) * p_cols], xr);
                        }
                    }
                    total_words += check_buf(b, &p_out, &want_p, "packed tc moe NVFP4 in-place panel")?;
                    let caps = [4.0f32, 25.0, 0.25, 0.04];
                    let pg_out = poisoned(b, all * (p_rows / 2))?;
                    b.project_tc_packed_gate_into(p_rows, p_cols, false, &panel_dev, &plan, &zp_dev, caps,
                        &mut pg_out.resident_range(GUARD..GUARD + all * (p_rows / 2))?)?;
                    let shape = spark_gpu_probe::causal_ops::Shape {
                        op: spark_gpu_probe::causal_ops::Op::CappedGate,
                        heads: 0, dim: (p_rows / 2) as u32, taps: 0, rows: all as u32, cols: 0, epsilon: caps[0], bound: caps[1],
                    };
                    let want_pg = spark_gpu_probe::causal_ops::reference(shape, [&want_p, &[], &[], &[]], &caps[2..])?;
                    total_words += check_buf(b, &pg_out, &want_pg, "packed tc moe NVFP4 in-place panel gated")?;
                    println!("TC ROWS PACKED MOE IN-PLACE PANEL PASS rows={p_rows} cols={p_cols}");
                }
            }
            println!("TC ROWS PACKED MOE PASS");

            // 4c. ZERO-STACK FAST BLOCK-OP KERNELS (kinds 107..=112) BIT-EXACTNESS & GUARD WORDS.
            {
                use spark_gpu_probe::{block_rows, causal_ops::{self, Op, Shape}};
                // FAST_MIX_BLEND (107): normal & transposed, bit-exact vs block_rows::reference.
                {
                    let (rows, asked, kept, dim, scale) = (17usize, 5usize, 6usize, 260usize, 0.020833334f32);
                    let scores: Vec<f32> = (0..rows * asked * kept).map(|i| input(i + 11) * 3.0).collect();
                    let joined: Vec<f32> = (0..kept * rows * dim).map(|i| input(i + 23)).collect();
                    let s_dev = b.upload_vector(&scores)?;
                    let j_dev = b.upload_vector(&joined)?;
                    let out_norm = poisoned(b, rows * asked * dim)?;
                    let out_tr = poisoned(b, rows * asked * dim)?;
                    b.fast_mix_blend_into(rows, asked, kept, dim, scale, false, &s_dev, &j_dev,
                        &mut out_norm.resident_range(GUARD..GUARD + rows * asked * dim)?)?;
                    b.fast_mix_blend_into(rows, asked, kept, dim, scale, true, &s_dev, &j_dev,
                        &mut out_tr.resident_range(GUARD..GUARD + rows * asked * dim)?)?;
                    let shape = Shape { op: Op::MixBlend, heads: asked as u32, dim: dim as u32, taps: 0, rows: 0, cols: kept as u32, epsilon: 0.0, bound: scale };
                    let want = block_rows::reference(shape, rows, block_rows::LIST, [&scores, &joined, &[], &[]], &[])?;
                    total_words += check_buf(b, &out_norm, &want, "fast_mix_blend normal")?;
                    let mut want_tr = vec![0.0f32; rows * asked * dim];
                    for r in 0..rows {
                        for q in 0..asked {
                            want_tr[(q * rows + r) * dim..(q * rows + r + 1) * dim]
                                .copy_from_slice(&want[(r * asked + q) * dim..(r * asked + q + 1) * dim]);
                        }
                    }
                    total_words += check_buf(b, &out_tr, &want_tr, "fast_mix_blend transposed")?;
                }
                // FAST_ROW_DOTS (108): deterministic + matches row=1 vs row=N bit-for-bit + close to serial ref.
                // Asked 3 and 1 take the one-launch prefill mix (`fast_ops::MIX_DOTS`), asked 7 the dots launch.
                for &asked in &[7usize, 3, 1] {
                    let (rows, kept, k) = (9usize, 5usize, 260usize);
                    let joined: Vec<f32> = (0..kept * rows * k).map(|i| input(i + 37)).collect();
                    let queries: Vec<f32> = (0..asked * k).map(|i| input(i + 43)).collect();
                    let j_dev = b.upload_vector(&joined)?;
                    let q_dev = b.upload_vector(&queries)?;
                    let out_all = poisoned(b, rows * asked * kept)?;
                    b.fast_row_dots_into(rows, asked, kept, k, &j_dev, &q_dev,
                        &mut out_all.resident_range(GUARD..GUARD + rows * asked * kept)?)?;
                    // Compare row r=3 run alone (rows=1) against row r=3 in batch (rows=9): MUST BE BIT-EXACT!
                    let r_solo = 3usize;
                    let mut j_solo = Vec::with_capacity(kept * k);
                    for e in 0..kept {
                        j_solo.extend_from_slice(&joined[(e * rows + r_solo) * k..(e * rows + r_solo + 1) * k]);
                    }
                    let j_solo_dev = b.upload_vector(&j_solo)?;
                    let out_solo = poisoned(b, asked * kept)?;
                    b.fast_row_dots_into(1, asked, kept, k, &j_solo_dev, &q_dev,
                        &mut out_solo.resident_range(GUARD..GUARD + asked * kept)?)?;
                    let back_all = b.read_vector_values(&out_all)?;
                    let want_solo = &back_all[GUARD + r_solo * asked * kept..GUARD + (r_solo + 1) * asked * kept];
                    total_words += check_buf(b, &out_solo, want_solo, "fast_row_dots solo==batch bit-exact")?;
                    // Also check numerical accuracy vs CPU serial reference (<= 1e-4).
                    let shape = Shape { op: Op::RowDots, heads: kept as u32, dim: 0, taps: 0, rows: asked as u32, cols: k as u32, epsilon: 0.0, bound: 0.0 };
                    let ref_dots = block_rows::reference(shape, rows, block_rows::LIST, [&joined, &[], &[], &[]], &queries)?;
                    let max_err = back_all[GUARD..GUARD + rows * asked * kept].iter().zip(&ref_dots).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    if max_err > 1e-4 { return Err(format!("fast_row_dots max_err={max_err}")); }
                    total_words += rows * asked * kept;
                    let streams = joined.chunks_exact(rows * k).map(|x| b.upload_vector(x)).collect::<Result<Vec<_>, _>>()?;
                    let list = streams.iter().collect::<Vec<_>>();
                    let scores = out_all.resident_range(GUARD..GUARD + rows * asked * kept)?;
                    for transpose in [false, true] {
                        let (words, scale) = (rows * asked * k, 0.02);
                        let mut contiguous = b.vector(words)?;
                        b.fast_mix_blend_into(rows, asked, kept, k, scale, transpose, &scores, &j_dev, &mut contiguous)?;
                        let separate = b.fast_mix_list(rows, k, scale, transpose, &q_dev, &list)?;
                        let actual = b.read_vector_values(&separate)?;
                        let expected = b.read_vector_values(&contiguous)?;
                        if !actual.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()) {
                            return Err("prefill mix stream list differs from joined mix".into());
                        }
                        total_words += words;
                    }
                }
                // FAST_MIX_BLEND list form with the gate norm (MIX_LIST_NORM): one launch against the transposed list mix
                // and the prefill rounded norm of its rows with no weight: both outputs BIT-EXACT.
                for &(rows, asked, kept, dim) in &[(9usize, 48usize, 3usize, 2304usize), (33, 1, 5, 2304), (17, 5, 4, 1024), (3, 2, 1, 128), (5, 7, 8, 4096), (2, 3, 2, 2432)] {
                    let (scale, eps) = (1.0 / (dim as f32).sqrt(), 1e-6f32);
                    let streams: Vec<Vec<f32>> = (0..kept).map(|e| (0..rows * dim).map(|i| input(i + 1301 + 211 * e) * (1.0 + 2.0 * e as f32)).collect()).collect();
                    let queries: Vec<f32> = (0..asked * dim).map(|i| input(i + 1409) * 0.5).collect();
                    let qd = b.upload_vector(&queries)?;
                    let sd = streams.iter().map(|r| b.upload_vector(r)).collect::<Result<Vec<_>, _>>()?;
                    let list = sd.iter().collect::<Vec<_>>();
                    let (mixed, normed) = b.fast_mix_list_normed(rows, dim, scale, eps, &qd, &list)?;
                    let mixed_sep = b.fast_mix_list(rows, dim, scale, true, &qd, &list)?;
                    for prefill in [true, false] {
                        let mut normed_sep = b.vector(rows * asked * dim)?;
                        b.fast_norm_into(rows * asked, dim, true, eps, prefill, &mixed_sep, None, &mut normed_sep)?;
                        for (label, got, want) in [("mixed", &mixed, &mixed_sep), ("normed", &normed, &normed_sep)] {
                            let (got, want) = (b.read_vector_values(got)?, b.read_vector_values(want)?);
                            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                                return Err(format!("mix list norm rows={rows} asked={asked} kept={kept} dim={dim} prefill={prefill} {label} word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
                            }
                            total_words += want.len();
                        }
                    }
                    println!("FAST MIX LIST NORM BIT-EXACT PASS rows={rows} asked={asked} kept={kept} dim={dim}");
                }
                // FAST_MIX_BLEND one-row form (MIX_ONE): one launch against separate PlainNorm + FAST_ROW_DOTS + FAST_MIX_BLEND
                // (32-lane warp-butterfly dot order, bit-identical to Prefill MIX_DOTS and across M=1/M=2..16).
                for &(asked, kept, dim) in &[(1usize, 2usize, 2304usize), (3, 5, 2304), (1, 8, 2304), (2, 7, 1024), (1, 4, 2336), (1, 3, 2560)] {
                    let (scale, eps) = (1.0 / (dim as f32).sqrt(), 1e-6f32);
                    let x: Vec<f32> = (0..dim).map(|i| input(i + 701) * 6.0).collect();
                    let stack: Vec<Vec<f32>> = (0..kept - 1).map(|e| (0..dim).map(|i| input(i + 811 + 97 * e) * (1.0 + e as f32)).collect()).collect();
                    let queries: Vec<f32> = (0..asked * dim).map(|i| input(i + 907) * 0.5).collect();
                    let (xd, qd) = (b.upload_vector(&x)?, b.upload_vector(&queries)?);
                    let sd = stack.iter().map(|r| b.upload_vector(r)).collect::<Result<Vec<_>, _>>()?;
                    let (normed, mixed) = b.fast_mix_one(asked, dim, scale, eps, &qd, &sd.iter().collect::<Vec<_>>(), &xd)?;
                    let mut normed_sep = b.vector(dim)?;
                    b.fast_norm_into(1, dim, false, eps, false, &xd, None, &mut normed_sep)?;
                    let mut joined = stack.concat();joined.extend_from_slice(&b.read_vector_values(&normed_sep)?);
                    let jd = b.upload_vector(&joined)?;
                    let mut dd = b.vector(asked * kept)?;
                    b.fast_row_dots_into(1, asked, kept, dim, &jd, &qd, &mut dd)?;
                    let mut mixed_sep = b.vector(asked * dim)?;
                    b.fast_mix_blend_into(1, asked, kept, dim, scale, false, &dd, &jd, &mut mixed_sep)?;
                    for (label, got, want) in [("normed", &normed, &normed_sep), ("mixed", &mixed, &mixed_sep)] {
                        let (got, want) = (b.read_vector_values(got)?, b.read_vector_values(want)?);
                        if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                            return Err(format!("mix one asked={asked} kept={kept} dim={dim} {label} word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
                        }
                        total_words += want.len();
                    }
                    println!("MIX ONE asked={asked} kept={kept} dim={dim} BIT-EXACT PASS words={}", dim + asked * dim);
                }
                // FAST_NORM (109): PlainNorm (2304), RoundedNorm (128 unweighted & weighted, 2304 unweighted & weighted) BIT-EXACT!
                for &(rounded, rows, dim, has_w) in &[
                    (false, 13usize, 2304usize, false),
                    (true, 19, 128, false),
                    (true, 11, 128, true),
                    (true, 9, 2304, false),
                    (true, 7, 2304, true),
                ] {
                    let x_vals: Vec<f32> = (0..rows * dim).map(|i| input(i + dim + rows * 13) * 4.0).collect();
                    let w_vals: Vec<f32> = if has_w { (0..dim).map(|i| input(i + 503) + 1.5).collect() } else { Vec::new() };
                    let x_dev = b.upload_vector(&x_vals)?;
                    let w_dev = if has_w { Some(b.upload_vector(&w_vals)?) } else { None };
                    let out = poisoned(b, rows * dim)?;
                    b.fast_norm_into(rows, dim, rounded, 1e-6, true, &x_dev, w_dev.as_ref(),
                        &mut out.resident_range(GUARD..GUARD + rows * dim)?)?;
                    let shape = Shape {
                        op: if rounded { Op::RoundedNorm } else { Op::PlainNorm },
                        heads: rows as u32, dim: dim as u32, taps: u32::from(has_w), rows: 0, cols: 0, epsilon: 1e-6, bound: 0.0,
                    };
                    let want = causal_ops::reference(shape, [&x_vals, &[], &[], &[]], &w_vals)?;
                    total_words += check_buf(b, &out, &want, &format!("fast_norm r={rounded} d={dim} w={has_w}"))?;
                }
                for (dim, weighted, positioned) in [(16, true, true), (128, true, false), (128, false, false)] {
                    let (rows, heads) = (5usize, 3usize);
                    let words = rows * heads * dim;
                    let x: Vec<f32> = (0..words).map(|i| input(i + 601) * 8.0).collect();
                    let weight: Vec<f32> = (0..if weighted { heads * dim } else { 0 }).map(|i| input(i + 613) + 1.5).collect();
                    let factors: Vec<f32> = (0..if positioned { rows } else { 0 }).map(|i| input(i + 627) + 1.0).collect();
                    let xv = b.upload_vector(&x)?;
                    let wv = weighted.then(|| b.upload_vector(&weight)).transpose()?;
                    let fv = positioned.then(|| b.upload_vector(&factors)).transpose()?;
                    let out = poisoned(b, words)?;
                    let shape = Shape { op: Op::HeadClip, heads: heads as u32, dim: dim as u32, taps: u32::from(weighted),
                        rows: 0, cols: u32::from(positioned), epsilon: 0.0, bound: 2.0 };
                    b.fast_head_clip_into(shape, rows, &xv, wv.as_ref(), fv.as_ref(), &mut out.resident_range(GUARD..GUARD + words)?)?;
                    let want = block_rows::reference(shape, rows, 0, [&x, &factors, &[], &[]], &weight)?;
                    total_words += check_buf(b, &out, &want, "prefill head clip")?;
                }
                // FAST_CAPPED_GATE (110): BIT-EXACT vs causal_ops::reference!
                for picks in [23usize, 457] {
                    let dim = 1152usize;
                    let (gcap, ucap, inv_g, inv_u) = (4.0f32, 25.0f32, 0.25f32, 0.04f32);
                    let x_vals: Vec<f32> = (0..picks * 2 * dim).map(|i| input(i + 601) * 8.0).collect();
                    let x_dev = b.upload_vector(&x_vals)?;
                    let out = poisoned(b, picks * dim)?;
                    b.fast_capped_gate_into(picks, dim, gcap, ucap, inv_g, inv_u, true, &x_dev,
                        &mut out.resident_range(GUARD..GUARD + picks * dim)?)?;
                    let shape = Shape { op: Op::CappedGate, heads: 0, dim: dim as u32, taps: 0, rows: picks as u32, cols: 0, epsilon: gcap, bound: ucap };
                    let want = causal_ops::reference(shape, [&x_vals, &[], &[], &[]], &[inv_g, inv_u])?;
                    total_words += check_buf(b, &out, &want, "fast_capped_gate")?;
                }
                // FAST_ROUTER_PICK (111): BIT-EXACT including ties & normalize=true/false!
                for &norm in &[true, false] {
                    let (rows, experts, picks, bound) = (17usize, 128usize, 8usize, 8.0f32);
                    let mut logits: Vec<f32> = (0..rows * experts).map(|i| input(i + 701) * 5.0).collect();
                    // Inject exact ties on row 2 to test tie-breaking!
                    for e in 10..20 { logits[2 * experts + e] = 0.5; }
                    let mut bias: Vec<f32> = (0..experts).map(|i| input(i + 809) * 0.5).collect();
                    bias[10..20].fill(0.25);
                    let l_dev = b.upload_vector(&logits)?;
                    let b_dev = b.upload_vector(&bias)?;
                    let out = poisoned(b, rows * 2 * picks)?;
                    b.fast_router_pick_into(rows, experts, picks, norm, bound, &l_dev, &b_dev,
                        &mut out.resident_range(GUARD..GUARD + rows * 2 * picks)?)?;
                    let shape = Shape { op: Op::RouterPick, heads: 1, dim: 1, taps: u32::from(norm), rows: experts as u32, cols: picks as u32, epsilon: 0.0, bound };
                    let want = block_rows::reference(shape, rows, 0, [&logits, &[], &[], &[]], &bias)?;
                    total_words += check_buf(b, &out, &want, &format!("fast_router_pick norm={norm}"))?;
                }
                // FAST_STREAM_OPS (112): ShortConv window (forms 0, 1, 2, 6, 7), RouteFold, GateBlend BIT-EXACT!
                {
                    let (rows, heads, dim, taps, bound) = (15usize, 8usize, 128usize, 4usize, 0.2f32);
                    let width = heads * dim;
                    let win: Vec<f32> = (0..(rows + taps - 1) * width).map(|i| input(i + 907) * 2.0).collect();
                    let fixed: Vec<f32> = (0..width * taps).map(|i| input(i + 919)).collect();
                    let tok: Vec<f32> = (0..rows * heads * taps).map(|i| input(i + 929)).collect();
                    let add: Vec<f32> = (0..rows * width).map(|i| input(i + 937)).collect();
                    let step = [(taps - 1) as f32];
                    let win_dev = b.upload_vector(&win)?;
                    let new_inputs = b.upload_vector(&win[(taps - 1) * width..])?;
                    let first = 2;
                    let mut old_ring = vec![0.0; taps * width];
                    for lag in 1..taps {
                        let slot = (first + taps - lag) % taps;
                        old_ring[slot * width..(slot + 1) * width]
                            .copy_from_slice(&win[(taps - 1 - lag) * width..(taps - lag) * width]);
                    }
                    let old_ring = b.upload_vector(&old_ring)?;
                    let fix_dev = b.upload_vector(&fixed)?;
                    let tok_dev = b.upload_vector(&tok)?;
                    let add_dev = b.upload_vector(&add)?;
                    for &form in &[0u32, 1, 2, 6, 7] {
                        let out = poisoned(b, rows * width)?;
                        b.fast_short_conv_win_into(rows, heads, dim, taps, form, bound, &win_dev,
                            (form & 1 != 0).then_some(&tok_dev), (form & 2 != 0).then_some(&add_dev), &fix_dev, None,
                            &mut out.resident_range(GUARD..GUARD + rows * width)?, None)?;
                        let shape = Shape { op: Op::ShortConv, heads: heads as u32, dim: dim as u32, taps: taps as u32, rows: 0, cols: form, epsilon: 0.0, bound };
                        let want = block_rows::reference(shape, rows, block_rows::WINDOW | 8,
                            [&win, if form & 1 != 0 { &tok } else { &[] }, if form & 2 != 0 { &add } else { &[] }, &step], &fixed)?;
                        total_words += check_buf(b, &out, &want, &format!("fast_short_conv form={form}"))?;
                        // The same launch with the ring after each row (a step that keeps each row): the words above, and
                        // slot s of row r's ring = window row r + taps - 1 - ((own - s) mod taps), own = (first + r) mod taps.
                        for keep_first in [0usize, 3] {
                            let (kept_out, kept) = (poisoned(b, rows * width)?, poisoned(b, rows * taps * width)?);
                            b.fast_short_conv_win_into(rows, heads, dim, taps, form, bound, &win_dev,
                                (form & 1 != 0).then_some(&tok_dev), (form & 2 != 0).then_some(&add_dev), &fix_dev, None,
                                &mut kept_out.resident_range(GUARD..GUARD + rows * width)?,
                                Some((&mut kept.resident_range(GUARD..GUARD + rows * taps * width)?, keep_first)))?;
                            total_words += check_buf(b, &kept_out, &want, &format!("kept short conv form={form} first={keep_first}"))?;
                            let rings: Vec<f32> = (0..rows).flat_map(|r| { let own = (keep_first + r) % taps;
                                (0..taps).flat_map(move |slot| { let at = r + taps - 1 - (own + taps - slot) % taps; at * width..(at + 1) * width }) })
                                .map(|i| win[i]).collect();
                            total_words += check_buf(b, &kept, &rings, &format!("kept short conv rings form={form} first={keep_first}"))?;
                        }
                        let split = poisoned(b, rows * width)?;
                        b.fast_short_conv_win_into(rows, heads, dim, taps, form, bound, &new_inputs,
                            (form & 1 != 0).then_some(&tok_dev), (form & 2 != 0).then_some(&add_dev), &fix_dev,
                            Some((&old_ring, first)), &mut split.resident_range(GUARD..GUARD + rows * width)?, None)?;
                        total_words += check_buf(b, &split, &want, &format!("prefill short conv form={form}"))?;
                    }
                    for form in [0, 1] {
                        let shape = Shape { op: Op::ShortConv, heads: heads as u32, dim: dim as u32, taps: taps as u32,
                            rows: 0, cols: form, epsilon: 0.0, bound };
                        let conv = block_rows::reference(shape, rows, block_rows::WINDOW | 8,
                            [&win, if form == 0 { &[] } else { &tok }, &[], &step], &fixed)?;
                        for clip in [true, false] {
                            let param = if clip { 2.0 } else { 1e-6 };
                            let finish = Shape {
                                op: if clip { Op::HeadClip } else { Op::RoundedNorm },
                                heads: if clip { heads } else { rows * heads } as u32,
                                dim: dim as u32, taps: 0, rows: 0, cols: 0,
                                bound: if clip { param } else { 0.0 },
                                epsilon: if clip { 0.0 } else { param },
                            };
                            let finished = if clip {
                                block_rows::reference(finish, rows, 0, [&conv, &[], &[], &[]], &[])?
                            } else {
                                causal_ops::reference(finish, [&conv, &[], &[], &[]], &[])?
                            };
                            let plane_rows = rows + 2;
                            let plane_words = plane_rows * width / 2;
                            let out = poisoned(b, plane_words)?;
                            let mut expected = b.read_vector_values(&out)?[GUARD..GUARD + plane_words].to_vec();
                            let plane_first = plane_rows - 3;
                            for (row, values) in finished.chunks_exact(width).enumerate() {
                                let offset = (plane_first + row) % plane_rows * (width / 2);
                                for (i, [lo, hi]) in values.as_chunks::<2>().0.iter().enumerate() {
                                    expected[offset + i] = f32::from_bits((lo.to_bits() >> 16) | (hi.to_bits() & 0xffff_0000));
                                }
                            }
                            let kv = spark_gpu_probe::fast_ops::KvWrite { first: plane_first, plane_rows, clip, param };
                            b.fast_short_conv_kv_into(shape, rows, &new_inputs, (form == 1).then_some(&tok_dev), &fix_dev,
                                (&old_ring, first), &mut out.resident_range(GUARD..GUARD + plane_words)?, kv)?;
                            total_words += check_buf(b, &out, &expected, &format!("prefill KV finish clip={clip} form={form}"))?;
                            println!("FAST OPS PREFILL KV BIT-EXACT PASS clip={clip} form={form} plane_words={plane_words}");
                        }
                    }
                    // RouteFold
                    let (rf_rows, picks, rf_dim) = (13usize, 8usize, 2304usize);
                    let wts: Vec<f32> = (0..rf_rows * picks).map(|i| input(i + 947)).collect();
                    let dns: Vec<f32> = (0..rf_rows * picks * rf_dim).map(|i| input(i + 953) * 2.0).collect();
                    let w_dev = b.upload_vector(&wts)?;
                    let d_dev = b.upload_vector(&dns)?;
                    let out = poisoned(b, rf_rows * rf_dim)?;
                    b.fast_route_fold_into(rf_rows, picks, rf_dim, &w_dev, &d_dev,
                        &mut out.resident_range(GUARD..GUARD + rf_rows * rf_dim)?)?;
                    let shape = Shape { op: Op::RouteFold, heads: 0, dim: rf_dim as u32, taps: 0, rows: picks as u32, cols: 0, epsilon: 0.0, bound: 0.0 };
                    let want = block_rows::reference(shape, rf_rows, 0, [&wts, &dns, &[], &[]], &[])?;
                    total_words += check_buf(b, &out, &want, "fast_route_fold")?;
                    // RouteFold on half downs (`FOLD_HALF`): the downs as BF16 `rn` values, two a word, give the same words.
                    {
                        let rn = |x: f32| f32::from_bits((x.to_bits().wrapping_add(0x7fff + ((x.to_bits() >> 16) & 1)) >> 16) << 16);
                        let packed: Vec<f32> = dns.as_chunks::<2>().0.iter()
                            .map(|p| f32::from_bits((rn(p[0]).to_bits() >> 16) | (rn(p[1]).to_bits() & 0xffff_0000))).collect();
                        let h_dev = b.upload_vector(&packed)?;
                        let out = poisoned(b, rf_rows * rf_dim)?;
                        b.fast_route_fold_half_into(rf_rows, picks, rf_dim, &w_dev, &h_dev,
                            &mut out.resident_range(GUARD..GUARD + rf_rows * rf_dim)?)?;
                        total_words += check_buf(b, &out, &want, "fast_route_fold half")?;
                        println!("FAST OPS ROUTE FOLD HALF BIT-EXACT PASS rows={rf_rows} picks={picks} width={rf_dim}");
                    }
                    // GateBlend
                    let (gb_rows, gb_dim) = (11usize, 2304usize);
                    let gates: Vec<f32> = (0..gb_rows * 2 * gb_dim).map(|i| input(i + 967) * 3.0).collect();
                    let kept: Vec<f32> = (0..gb_rows * gb_dim).map(|i| input(i + 971)).collect();
                    let msg: Vec<f32> = (0..gb_rows * gb_dim).map(|i| input(i + 977)).collect();
                    let g_dev = b.upload_vector(&gates)?;
                    let k_dev = b.upload_vector(&kept)?;
                    let m_dev = b.upload_vector(&msg)?;
                    let out = poisoned(b, gb_rows * gb_dim)?;
                    b.fast_gate_blend_into(gb_rows, gb_dim, &g_dev, &k_dev, &m_dev,
                        &mut out.resident_range(GUARD..GUARD + gb_rows * gb_dim)?)?;
                    let shape = Shape { op: Op::GateBlend, heads: 0, dim: gb_dim as u32, taps: 0, rows: 0, cols: 0, epsilon: 0.0, bound: 0.0 };
                    let want = block_rows::reference(shape, gb_rows, 0, [&gates, &kept, &msg, &[]], &[])?;
                    total_words += check_buf(b, &out, &want, "fast_gate_blend")?;
                }
                println!("FAST OPS (107..=112) BIT-EXACT PASS");
            }

            // 5. FLASH_WINDOW_ATTN: full causal + sliding-window ring wrap-around + GQA 24/8 + rel_bias + sink + gate!
            {
                use spark_gpu_probe::flash_window::{self, FW_HEAD_DIM, FW_REL_DIM};
                let (heads, kv_heads, extent, scale) = (24usize, 8usize, 64usize, 0.088_388_35f32);
                let cases: &[(usize, usize, usize, usize, &str)] = &[
                    (17, 64, 0, 64, "causal_early"),
                    (73, 160, 40, 160, "causal_multi_tile"),
                    (71, 134, 95, 64, "sliding_ring_wrap"),
                ];
                let pack_kv = |vals: &[u16]| -> Vec<f32> {
                    vals.as_chunks::<2>().0.iter().map(|p| f32::from_bits(u32::from(p[0]) | (u32::from(p[1]) << 16))).collect()
                };
                for &(rows, view, first_pos, window, label) in cases {
                    let q_vals: Vec<f32> = (0..rows * heads * FW_HEAD_DIM).map(|i| input(i + 17) * 0.5).collect();
                    let k_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 31)).collect();
                    let v_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 59)).collect();
                    let k_u32: Vec<u32> = k_u16.as_chunks::<2>().0.iter().map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16)).collect();
                    let v_u32: Vec<u32> = v_u16.as_chunks::<2>().0.iter().map(|p| u32::from(p[0]) | (u32::from(p[1]) << 16)).collect();
                    let m_vals: Vec<f32> = (0..rows * heads * FW_REL_DIM).map(|i| input(i + 73) * 0.25).collect();
                    let t_vals: Vec<f32> = (0..FW_REL_DIM * extent).map(|i| input(i + 97) * 0.25).collect();
                    let s_vals: Vec<f32> = (0..rows * heads).map(|i| input(i + 113) * 0.5).collect();
                    let g_vals: Vec<f32> = (0..rows * heads * FW_HEAD_DIM).map(|i| input(i + 131) * 0.5).collect();
                    let mut st_vals = Vec::with_capacity(rows * 2);
                    for r in 0..rows {
                        let pos = first_pos + r;
                        st_vals.push((pos + 1).min(window) as f32);
                        st_vals.push((pos % view) as f32);
                    }
                    let q_dev = b.upload_vector(&q_vals)?.into_resident_shared()?;
                    let k_dev = b.upload_vector(&pack_kv(&k_u16))?;
                    let v_dev = b.upload_vector(&pack_kv(&v_u16))?;
                    let m_dev = b.upload_vector(&m_vals)?.into_resident_shared()?;
                    let t_dev = b.upload_vector(&t_vals)?;
                    let s_dev = b.upload_vector(&s_vals)?.into_resident_shared()?;
                    let g_dev = b.upload_vector(&g_vals)?.into_resident_shared()?;
                    let st_dev = b.upload_vector(&st_vals)?.into_resident_shared()?;
                    let out1 = poisoned(b, rows * heads * FW_HEAD_DIM)?;
                    let out2 = poisoned(b, rows * heads * FW_HEAD_DIM)?;
                    b.flash_window_attn_into(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent, scale, first_pos as u64,
                        &q_dev, &k_dev, &v_dev, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev, None,
                        &mut out1.resident_range(GUARD..GUARD + rows * heads * FW_HEAD_DIM)?)?;
                    b.flash_window_attn_into(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent, scale, first_pos as u64,
                        &q_dev, &k_dev, &v_dev, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev, None,
                        &mut out2.resident_range(GUARD..GUARD + rows * heads * FW_HEAD_DIM)?)?;
                    b.synchronize()?;
                    let got1 = b.read_vector_values(&out1)?;
                    let got2 = b.read_vector_values(&out2)?;
                    let n = rows * heads * FW_HEAD_DIM;
                    for (side, slice) in [("before", &got1[..GUARD]), ("after", &got1[GUARD + n..])] {
                        if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                            return Err(format!("flash_window {label}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                        }
                    }
                    // 100% bit-exact determinism between repeated GPU launches.
                    for i in 0..n {
                        if got1[GUARD + i].to_bits() != got2[GUARD + i].to_bits() {
                            return Err(format!("flash_window {label}: non-deterministic at {i}"));
                        }
                    }
                    // Changing the prompt cut changes both the query's position in a CTA and the CTA's key arc.
                    for cuts in [&[1, 13, 37][..], &[7, 32, 65][..]] {
                        let cut_out = poisoned(b, n)?;
                        let mut first = 0;
                        for end in cuts.iter().copied().filter(|&cut| cut < rows).chain([rows]) {
                            let at = |width| first * width..end * width;
                            let channels = heads * FW_HEAD_DIM;
                            b.flash_window_attn_into(end - first, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM,
                                view, extent, scale, (first_pos + first) as u64,
                                &q_dev.resident_range(at(channels))?, &k_dev, &v_dev,
                                &m_dev.resident_range(at(heads * FW_REL_DIM))?, &t_dev,
                                &s_dev.resident_range(at(heads))?, &g_dev.resident_range(at(channels))?,
                                &st_dev.resident_range(at(2))?, None,
                                &mut cut_out.resident_range(GUARD + first * channels..GUARD + end * channels)?)?;
                            first = end;
                        }
                        let cut_words = b.read_vector_values(&cut_out)?;
                        if let Some(i) = cut_words.iter().zip(&got1).position(|(a, b)| a.to_bits() != b.to_bits()) {
                            return Err(format!("flash_window {label}: prompt cuts {cuts:?} differ at word {i}"));
                        }
                    }
                    println!("FLASH WINDOW {label} CHUNK CUT PASS words={n} cut_patterns=2");
                    // Check numerical accuracy vs CPU reference (BF16 rounding tolerance <= 2e-2).
                    let want = flash_window::flash_window_reference(
                        rows, heads, kv_heads, view, extent, scale,
                        &q_vals, &k_u32, &v_u32, &m_vals, &t_vals, &s_vals, &g_vals, &st_vals,
                    );
                    let mut max_abs = 0.0f32;
                    let mut max_rel = 0.0f32;
                    for i in 0..n {
                        let a = got1[GUARD + i];
                        let w = want[i];
                        if !a.is_finite() { return Err(format!("flash_window {label}: non-finite at {i}")); }
                        let diff = (a - w).abs();
                        if diff > max_abs { max_abs = diff; }
                        let tol = 0.01f32.max(w.abs() * 0.02);
                        let rel = diff / w.abs().max(1.0);
                        if rel > max_rel { max_rel = rel; }
                        if diff > tol {
                            return Err(format!("flash_window {label}: at {i} got={a} want={w} diff={diff} tol={tol}"));
                        }
                    }
                    println!("FLASH WINDOW {label} PASS words={n} max_abs={max_abs:.6} max_rel={max_rel:.6} det_exact=true");
                    total_words += n;
                    // The clips inside the launch (`FW_CLIP_IN`): bit for bit the prefill clip launches, then this form.
                    {
                        use spark_gpu_probe::{block_ops::{Op, Shape}, driver::OneClips};
                        let positioned = label != "causal_early";
                        let (query_bound, moved_bound) = (22.627_417f32, 8.0f32);
                        let raw_q: Vec<f32> = q_vals.iter().map(|v| v * 6.0).collect();
                        let raw_m: Vec<f32> = m_vals.iter().map(|v| v * 16.0).collect();
                        let qs: Vec<f32> = (0..heads * FW_HEAD_DIM).map(|i| 1.0 + input(i + 151) * 0.25).collect();
                        let ms: Vec<f32> = (0..heads * FW_REL_DIM).map(|i| 1.0 + input(i + 157) * 0.25).collect();
                        let pf: Vec<f32> = (0..rows).map(|i| 1.25 + 0.125 * (i % 7) as f32).collect();
                        let shared = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
                        let (rq, rm, qsd, msd, pfd) = (shared(&raw_q)?, shared(&raw_m)?, shared(&qs)?, shared(&ms)?, shared(&pf)?);
                        let clip = |dim: usize, bound: f32| Shape { op: Op::HeadClip, heads: heads as u32, dim: dim as u32, taps: 1, rows: 0,
                            cols: u32::from(positioned), epsilon: 0.0, bound };
                        let mut cq = b.vector(rows * heads * FW_HEAD_DIM)?;
                        b.fast_head_clip_into(clip(FW_HEAD_DIM, query_bound), rows, &rq, Some(&qsd), positioned.then_some(&pfd), &mut cq)?;
                        let mut cm = b.vector(rows * heads * FW_REL_DIM)?;
                        b.fast_head_clip_into(clip(FW_REL_DIM, moved_bound), rows, &rm, Some(&msd), positioned.then_some(&pfd), &mut cm)?;
                        let (want, got) = (poisoned(b, n)?, poisoned(b, n)?);
                        b.flash_window_attn_into(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent, scale, first_pos as u64,
                            &cq, &k_dev, &v_dev, &cm, &t_dev, &s_dev, &g_dev, &st_dev, None, &mut want.resident_range(GUARD..GUARD + n)?)?;
                        let clips = OneClips { query_scale: Some(&qsd), moved_scale: Some(&msd), position: positioned.then_some(&pfd), query_bound, moved_bound, written_keys: None, written_values: None };
                        b.flash_window_attn_into(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent, scale, first_pos as u64,
                            &rq, &k_dev, &v_dev, &rm, &t_dev, &s_dev, &g_dev, &st_dev, Some(clips), &mut got.resident_range(GUARD..GUARD + n)?)?;
                        b.synchronize()?;
                        let (want, got) = (b.read_vector_values(&want)?, b.read_vector_values(&got)?);
                        if let Some(i) = got.iter().zip(&want).position(|(a, w)| a.to_bits() != w.to_bits()) {
                            return Err(format!("flash_window {label}: clips inside differ at word {i}: {:#x} vs {:#x}", got[i].to_bits(), want[i].to_bits()));
                        }
                        if want[GUARD..GUARD + n].iter().any(|v| !v.is_finite()) { return Err(format!("flash_window {label}: clipped output not finite")); }
                        println!("FLASH WINDOW {label} CLIP IN BIT-EXACT PASS words={n} positioned={positioned}");
                        total_words += n;
                    }
                }
            }

            // 5b. FLASH_WINDOW_ATTN exact form against the block arms on the host (HeadClip of the query and of the
            // relative state, RelBias, WindowScores, SinkSoftmax, WindowValues), bit for bit: one row, an empty
            // window, and the rows of check passes (each row with its own live count, slot, query, sink, gate and
            // position word; every row against the one-row arms with its own operands).
            {
                use spark_gpu_probe::{block_rows, causal_ops::{Op, Shape}, driver::OneClips, flash_window::{FW_HEAD_DIM, FW_REL_DIM}};
                let (heads, kv_heads, extent, scale, query_bound, moved_bound) = (24usize, 8usize, 1024usize, 0.088_388_35f32, 8.0f32, 8.0f32);
                let arm = |op: Op, heads: usize, dim: usize, taps: usize, rows: usize, cols: usize, bound: f32| Shape {
                    op, heads: heads as u32, dim: dim as u32, taps: taps as u32, rows: rows as u32, cols: cols as u32, epsilon: 0.0, bound };
                // (view, live and slot of each row, position factor, label). A check pass of `r` rows over a view of
                // `prefix + r` rows: row i sees prefix + i + 1 keys and sits in slot prefix + i.
                let check = |prefix: usize, r: usize| (0..r).map(|i| (prefix + i + 1, prefix + i)).collect::<Vec<_>>();
                let cases: Vec<(usize, Vec<(usize, usize)>, bool, &str)> = vec![
                    (1024, vec![(1, 0)], true, "one_live"), (1024, vec![(23, 22)], true, "early"), (1024, vec![(300, 299)], false, "straight"),
                    (1024, vec![(1024, 77)], true, "ring_full"), (40, vec![(17, 16)], true, "short_view"), (513, vec![(513, 512)], false, "odd_view"),
                    (64, vec![(0, 5)], true, "empty"), (40, check(36, 4), true, "check_4"), (130, check(122, 8), false, "check_8"),
                    (513, check(511, 2), true, "check_2_long"), (1024, vec![(1024, 1022), (1024, 1023), (1024, 0)], true, "rows_ring_wrap"),
                    (2048, vec![(1536, 1535)], true, "global_1536_one"), (2048, check(1532, 4), true, "global_1536_check_4"),
                    (128, check(96, 8), true, "chat_planes"), (200, check(184, 16), true, "check_16"), (96, vec![(90, 89), (0, 3), (91, 90)], false, "rows_with_empty"),
                ];
                for (view, row_steps, factor, label) in &cases {
                    let (view, rows, factor) = (*view, row_steps.len(), *factor);
                    let q: Vec<f32> = (0..rows * heads * FW_HEAD_DIM).map(|i| input(i + 7) * 3.0).collect();
                    let q_scale: Vec<f32> = (0..heads * FW_HEAD_DIM).map(|i| 1.0 + input(i + 11) * 0.25).collect();
                    let m: Vec<f32> = (0..rows * heads * FW_REL_DIM).map(|i| input(i + 13) * 4.0).collect();
                    let m_scale: Vec<f32> = (0..heads * FW_REL_DIM).map(|i| 1.0 + input(i + 19) * 0.25).collect();
                    let position: Vec<f32> = (0..rows).map(|i| 1.25 + 0.125 * i as f32).collect();
                    let k_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 31)).collect();
                    let v_u16: Vec<u16> = (0..view * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 59)).collect();
                    let pack = |vals: &[u16]| -> Vec<f32> {
                        vals.as_chunks::<2>().0.iter().map(|[lo, hi]| {
                            f32::from_bits(u32::from(*lo) | (u32::from(*hi) << 16))
                        }).collect()
                    };
                    let (keys, values) = (pack(&k_u16), pack(&v_u16));
                    let table: Vec<f32> = (0..FW_REL_DIM * extent).map(|i| input(i + 97) * 0.25).collect();
                    let sink: Vec<f32> = (0..rows * heads).map(|i| input(i + 113) * 2.0).collect();
                    let gate: Vec<f32> = (0..rows * heads * FW_HEAD_DIM).map(|i| input(i + 131) * 2.0).collect();
                    let steps: Vec<f32> = row_steps.iter().flat_map(|&(live, slot)| [live as f32, slot as f32]).collect();
                    // One row with its own operands through the separate arms.
                    let reference = |r: usize| -> Result<Vec<f32>, String> {
                        let (hd, hr) = (heads * FW_HEAD_DIM, heads * FW_REL_DIM);
                        let (q, m, sink, gate, steps) = (&q[r * hd..(r + 1) * hd], &m[r * hr..(r + 1) * hr], &sink[r * heads..(r + 1) * heads], &gate[r * hd..(r + 1) * hd], &steps[2 * r..2 * r + 2]);
                        let (keys,values)=if *label=="chat_planes" && r%2==1 {(&values,&keys)} else {(&keys,&values)};
                        let factor_words: &[f32] = if factor { &position[r..r + 1] } else { &[] };
                        let qc = block_rows::reference(arm(Op::HeadClip, heads, FW_HEAD_DIM, 1, 0, usize::from(factor), query_bound), 1, 0, [q, factor_words, &[], &[]], &q_scale)?;
                        let mc = block_rows::reference(arm(Op::HeadClip, heads, FW_REL_DIM, 1, 0, usize::from(factor), moved_bound), 1, 0, [m, factor_words, &[], &[]], &m_scale)?;
                        let bias = block_rows::reference(arm(Op::RelBias, heads, FW_REL_DIM, 0, 0, extent, 0.0), 1, 0, [&mc, &[], &[], &[]], &table)?;
                        let scores = block_rows::reference(arm(Op::WindowScores, heads, FW_HEAD_DIM, heads / kv_heads, view, extent, scale), 1, 0, [&qc, keys, &bias, steps], &[])?;
                        let weights = block_rows::reference(arm(Op::SinkSoftmax, heads, 0, 0, view, 0, 0.0), 1, 0, [&scores, sink, &steps[..1], &[]], &[])?;
                        block_rows::reference(arm(Op::WindowValues, heads, FW_HEAD_DIM, heads / kv_heads, view, 0, 0.0), 1, 0, [&weights, values, gate, steps], &[])
                    };
                    let want = (0..rows).map(reference).collect::<Result<Vec<_>, _>>()?.concat();
                    let shared = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
                    let (q_dev, qs_dev, m_dev, ms_dev, p_dev) = (shared(&q)?, shared(&q_scale)?, shared(&m)?, shared(&m_scale)?, shared(&position)?);
                    let (k_dev, v_dev, t_dev, s_dev, g_dev, st_dev) = (shared(&keys)?, shared(&values)?, shared(&table)?, shared(&sink)?, shared(&gate)?, shared(&steps)?);
                    let n = rows * heads * FW_HEAD_DIM;
                    let out = poisoned(b, n)?;
                    let clips = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: factor.then_some(&p_dev), query_bound, moved_bound, written_keys: None, written_values: None };
                    let planes=if *label=="chat_planes" {(0..rows).map(|r|if r%2==0 {(&k_dev,&v_dev)} else {(&v_dev,&k_dev)}).collect()} else {Vec::new()};
                    b.flash_window_planes_into(rows, heads, kv_heads, view, extent, scale, &q_dev, &k_dev, &v_dev, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev,
                        clips, &planes, &mut out.resident_range(GUARD..GUARD + n)?)?;
                    b.synchronize()?;
                    let got = b.read_vector_values(&out)?;
                    for (side, slice) in [("before", &got[..GUARD]), ("after", &got[GUARD + n..])] {
                        if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                            return Err(format!("flash_window_one {label}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                        }
                    }
                    if let Some(i) = (0..n).find(|&i| got[GUARD + i].to_bits() != want[i].to_bits()) {
                        return Err(format!("flash_window_one {label}: word {i} got {:#x} want {:#x}", got[GUARD + i].to_bits(), want[i].to_bits()));
                    }
                    println!("FLASH WINDOW ONE {label} rows={rows} BIT-EXACT PASS words={n}");
                    total_words += n;
                    // Zero-copy provisional KV: move the `rows` contiguous written slots into `written_keys`/`written_values`
                    // and poison those slots in the prior plane (when live < view so earlier rows never read later rows'
                    // overwritten slots), or test a true wrapped ring with distinct prior-ring keys at wrapped slots.
                    if *label != "chat_planes" && rows > 1 && row_steps.windows(2).all(|w| w[1].0 == w[0].0 + 1 && w[1].1 == (w[0].1 + 1) % view) {
                        let kv_words = kv_heads * (FW_HEAD_DIM / 2);
                        let (mut old_k, mut old_v) = (keys.clone(), values.clone());
                        let (mut wk, mut wv) = (Vec::with_capacity(rows * kv_words), Vec::with_capacity(rows * kv_words));
                        for &(_, slot) in row_steps {
                            let s = slot * kv_words;
                            wk.extend_from_slice(&keys[s..s + kv_words]);
                            wv.extend_from_slice(&values[s..s + kv_words]);
                            for x in &mut old_k[s..s + kv_words] { *x = f32::from_bits(POISON); }
                            for x in &mut old_v[s..s + kv_words] { *x = f32::from_bits(POISON); }
                        }
                        let (ok_dev, ov_dev, wk_dev, wv_dev) = (shared(&old_k)?, shared(&old_v)?, shared(&wk)?, shared(&wv)?);
                        let out_w = poisoned(b, n)?;
                        let clips_w = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: factor.then_some(&p_dev),
                            query_bound, moved_bound, written_keys: Some(&wk_dev), written_values: Some(&wv_dev) };
                        b.flash_window_one_into(rows, heads, kv_heads, view, extent, scale, &q_dev, &ok_dev, &ov_dev, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev,
                            clips_w, &mut out_w.resident_range(GUARD..GUARD + n)?)?;
                        b.synchronize()?;
                        let got_w = b.read_vector_values(&out_w)?;
                        for (side, slice) in [("before", &got_w[..GUARD]), ("after", &got_w[GUARD + n..])] {
                            if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                                return Err(format!("flash_window_one_written {label}: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                            }
                        }
                        if let Some(i) = (0..n).find(|&i| got_w[GUARD + i].to_bits() != want[i].to_bits()) {
                            return Err(format!("flash_window_one_written {label}: word {i} got {:#x} want {:#x}", got_w[GUARD + i].to_bits(), want[i].to_bits()));
                        }
                        total_words += n;
                        println!("FLASH WINDOW ONE WRITTEN {label} rows={rows} BIT-EXACT PASS words={n}");
                    } else if *label == "rows_ring_wrap" {
                        // True wrapped sliding-window verify: `keys`/`values` hold the unmodified prior ring (so row 0
                        // still reads the old wrapped slots 1023 and 0), while `wk`/`wv` hold the `rows` new provisional rows.
                        let kv_words = kv_heads * (FW_HEAD_DIM / 2);
                        let wk_u16: Vec<u16> = (0..rows * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 777)).collect();
                        let wv_u16: Vec<u16> = (0..rows * kv_heads * FW_HEAD_DIM).map(|i| weight(i + 999)).collect();
                        let (wk, wv) = (pack(&wk_u16), pack(&wv_u16));
                        let mut want_wrap = Vec::with_capacity(n);
                        for r in 0..rows {
                            let mut rk = keys.clone();
                            let mut rv = values.clone();
                            for w_i in 0..=r {
                                let dst = row_steps[w_i].1 * kv_words;
                                rk[dst..dst + kv_words].copy_from_slice(&wk[w_i * kv_words..(w_i + 1) * kv_words]);
                                rv[dst..dst + kv_words].copy_from_slice(&wv[w_i * kv_words..(w_i + 1) * kv_words]);
                            }
                            let (hd, hr) = (heads * FW_HEAD_DIM, heads * FW_REL_DIM);
                            let (qr, mr, sr, gr, st) = (&q[r * hd..(r + 1) * hd], &m[r * hr..(r + 1) * hr], &sink[r * heads..(r + 1) * heads], &gate[r * hd..(r + 1) * hd], &steps[2 * r..2 * r + 2]);
                            let fw: &[f32] = if factor { &position[r..r + 1] } else { &[] };
                            let qc = block_rows::reference(arm(Op::HeadClip, heads, FW_HEAD_DIM, 1, 0, usize::from(factor), query_bound), 1, 0, [qr, fw, &[], &[]], &q_scale)?;
                            let mc = block_rows::reference(arm(Op::HeadClip, heads, FW_REL_DIM, 1, 0, usize::from(factor), moved_bound), 1, 0, [mr, fw, &[], &[]], &m_scale)?;
                            let bias = block_rows::reference(arm(Op::RelBias, heads, FW_REL_DIM, 0, 0, extent, 0.0), 1, 0, [&mc, &[], &[], &[]], &table)?;
                            let scores = block_rows::reference(arm(Op::WindowScores, heads, FW_HEAD_DIM, heads / kv_heads, view, extent, scale), 1, 0, [&qc, &rk, &bias, st], &[])?;
                            let weights = block_rows::reference(arm(Op::SinkSoftmax, heads, 0, 0, view, 0, 0.0), 1, 0, [&scores, sr, &st[..1], &[]], &[])?;
                            want_wrap.extend(block_rows::reference(arm(Op::WindowValues, heads, FW_HEAD_DIM, heads / kv_heads, view, 0, 0.0), 1, 0, [&weights, &rv, gr, st], &[])?);
                        }
                        let (wk_dev, wv_dev) = (shared(&wk)?, shared(&wv)?);
                        let out_w = poisoned(b, n)?;
                        let clips_w = OneClips { query_scale: Some(&qs_dev), moved_scale: Some(&ms_dev), position: factor.then_some(&p_dev),
                            query_bound, moved_bound, written_keys: Some(&wk_dev), written_values: Some(&wv_dev) };
                        b.flash_window_one_into(rows, heads, kv_heads, view, extent, scale, &q_dev, &k_dev, &v_dev, &m_dev, &t_dev, &s_dev, &g_dev, &st_dev,
                            clips_w, &mut out_w.resident_range(GUARD..GUARD + n)?)?;
                        b.synchronize()?;
                        let got_w = b.read_vector_values(&out_w)?;
                        for (side, slice) in [("before", &got_w[..GUARD]), ("after", &got_w[GUARD + n..])] {
                            if let Some((i, &v)) = slice.iter().enumerate().find(|(_, v)| v.to_bits() != POISON) {
                                return Err(format!("flash_window_one_written rows_ring_wrap: guard {side}[{i}] overwritten: {:#x}", v.to_bits()));
                            }
                        }
                        if let Some(i) = (0..n).find(|&i| got_w[GUARD + i].to_bits() != want_wrap[i].to_bits()) {
                            return Err(format!("flash_window_one_written rows_ring_wrap: word {i} got {:#x} want {:#x}", got_w[GUARD + i].to_bits(), want_wrap[i].to_bits()));
                        }
                        total_words += n;
                        println!("FLASH WINDOW ONE WRITTEN rows_ring_wrap rows={rows} BIT-EXACT PASS words={n}");
                    }
                }
            }

            // 5b2. FLASH_WINDOW_SPLIT (GQA-packed Split-KV decode/verify): accuracy, M=2/4/8==M=1, ring delta, C8==C1.
            total_words += super::split_window::run(b)?;

            // 5c. The key or value row in one arms launch (conv_finish_into) against the separate arms on the host: the
            // ring fill, ShortConv, HeadClip or RoundedNorm, PackBf16 and the plane fill, bit for bit; every other word
            // of the ring and of the plane stays as it was. Two calls of one shape and one step word in a row are the
            // key and the value of a token: the recorder joins them into one launch (conv_norm::pair).
            {
                use spark_gpu_probe::{block_rows, block_stream::CONV_TOKEN_TAPS, causal_ops::{Op, Shape}};
                let (heads, dim, taps, rows) = (8usize, 128usize, 4usize, 6usize);
                let width = heads * dim;
                type Case = (Op, bool, usize, usize, f32, &'static str);
                let groups: &[&[Case]] = &[
                    &[(Op::HeadClip, false, 0, 0, 8.0, "clip_slot0")],
                    &[(Op::HeadClip, false, 2, 3, 8.0, "pair_clip_slot2"), (Op::RoundedNorm, false, 2, 1, 1.0e-6, "pair_norm_slot2")],
                    &[(Op::HeadClip, true, 3, 5, 0.5, "pair_clip_made_slot3"), (Op::RoundedNorm, true, 3, 4, 1.0e-6, "pair_norm_made_slot3")],
                ];
                let shared = |v: &[f32]| b.upload_vector(v).and_then(|d| d.into_resident_shared());
                for (g, group) in groups.iter().enumerate() {
                    let slot = group[0].2;
                    let step = [slot as f32];
                    let step_dev = shared(&step)?;
                    let mut runs = Vec::new();
                    for (k, &(op, made, _, row, param, label)) in group.iter().enumerate() {
                        let seed = 1000 * (2 * g + k);
                        let ring: Vec<f32> = (0..width * taps).map(|i| input(i + seed + 211) * 3.0).collect();
                        let product: Vec<f32> = (0..width).map(|i| input(i + seed + 223) * 3.0).collect();
                        let fixed: Vec<f32> = (0..width * taps).map(|i| input(i + seed + 227) * 0.5).collect();
                        let made_taps: Vec<f32> = (0..heads * taps).map(|i| input(i + seed + 229) * 0.5).collect();
                        let plane: Vec<f32> = (0..rows * width / 2).map(|i| f32::from_bits(0x3f80_0000 ^ ((i + seed) as u32).wrapping_mul(2_654_435_761))).collect();
                        let conv = Shape { op: Op::ShortConv, heads: heads as u32, dim: dim as u32, taps: taps as u32, rows: 0,
                            cols: if made { CONV_TOKEN_TAPS } else { 0 }, epsilon: 0.0, bound: if made { 1.0 / 9.0 } else { 0.2 } };
                        let mut want_ring = ring.clone();
                        want_ring[slot * width..(slot + 1) * width].copy_from_slice(&product);
                        let made_words: &[f32] = if made { &made_taps } else { &[] };
                        let y = block_rows::reference(conv, 1, 0, [&want_ring, made_words, &[], &step], &fixed)?;
                        let finish = Shape { op, heads: heads as u32, dim: dim as u32, taps: 0, rows: 0, cols: 0,
                            epsilon: if op == Op::RoundedNorm { param } else { 0.0 }, bound: if op == Op::HeadClip { param } else { 0.0 } };
                        let want = block_rows::reference(finish, 1, 0, [&y, &[], &[], &[]], &[])?;
                        let bf16 = |x: f32| { let bits = x.to_bits(); bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16 };
                        let mut want_plane = plane.clone();
                        for i in 0..width / 2 { want_plane[row * width / 2 + i] = f32::from_bits(bf16(want[2 * i]) | (bf16(want[2 * i + 1]) << 16)); }
                        let (product_dev, fixed_dev, made_dev) = (shared(&product)?, shared(&fixed)?, shared(&made_taps)?);
                        let (ring_dev, plane_dev, out_dev) = (b.upload_vector(&ring)?, b.upload_vector(&plane)?, b.vector(width)?);
                        runs.push((op, made, row, param, label, conv, product_dev, fixed_dev, made_dev, ring_dev, plane_dev, out_dev, want_ring, want, want_plane));
                    }
                    for run in &mut runs {
                        let (op, made, row, param, _, conv, product_dev, fixed_dev, made_dev, ring_dev, plane_dev, out_dev, ..) = run;
                        b.conv_finish_into(*conv, *op, *param, ring_dev, made.then_some(&*made_dev), &step_dev, fixed_dev, product_dev,
                            plane_dev, *row * width / 2, out_dev)?;
                    }
                    b.synchronize()?;
                    for (.., label, _, _, _, _, ring_dev, plane_dev, out_dev, want_ring, want, want_plane) in runs {
                        for (what, got, want) in [("ring", b.read_vector_values(&ring_dev)?, want_ring), ("output", b.read_vector_values(&out_dev)?, want),
                            ("plane", b.read_vector_values(&plane_dev)?, want_plane)] {
                            if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                                return Err(format!("conv_finish {label}: {what} word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
                            }
                        }
                        println!("CONV FINISH {label} BIT-EXACT PASS words={}", width * taps + width + rows * width / 2);
                        total_words += width;
                    }
                }
            }

            // 5d. Fused multi-row mix (`fast_mix_rows` + `fast_mix_rows_normed`, `rows = 1..=16`) and fused
            // `fast_gate_blend_norm` (1 and 3 gates) compared bit-for-bit against separate launches.
            total_words += super::fast_ops::run(b)?;

            // 5e. Few-row dual-input (`project_staged_few_pair_into`, with/without BF16 output rounding) and
            // split-output (`project_staged_few_split_into`) compared bit-for-bit against `project_staged_few_into`.
            {
                for &(rows, cols, inputs, round_out) in &[(64usize, 4608usize, 4usize, true), (64, 4608, 8, false), (128, 256, 16, true), (320, 256, 4, true)] {
                    let half = cols / 2;
                    let w_host: Vec<u16> = (0..rows * cols).map(|i| (((i * 37 + 17) % 241) as u16) << 6 | 0x3c00).collect();
                    let x1_host: Vec<f32> = (0..inputs * half).map(|i| input(i + rows + 11)).collect();
                    let x2_host: Vec<f32> = (0..inputs * half).map(|i| input(i + cols + 29)).collect();
                    let mut xcat = Vec::with_capacity(inputs * cols);
                    for t in 0..inputs {
                        xcat.extend_from_slice(&x1_host[t * half..(t + 1) * half]);
                        xcat.extend_from_slice(&x2_host[t * half..(t + 1) * half]);
                    }
                    let w_dev = matrix(b, &w_host)?;
                    let x1_dev = b.upload_vector(&x1_host)?.into_resident_shared()?;
                    let x2_dev = b.upload_vector(&x2_host)?.into_resident_shared()?;
                    let xcat_dev = b.upload_vector(&xcat)?.into_resident_shared()?;
                    let (mut got_dev, mut ref_dev) = (b.vector(inputs * rows)?, b.vector(inputs * rows)?);
                    b.project_staged_few_pair_into(rows, cols, inputs, round_out, &x1_dev, &x2_dev, &w_dev, &mut got_dev)?;
                    b.project_staged_few_into(rows, cols, inputs, rows, &xcat_dev, &w_dev, &mut ref_dev)?;
                    b.synchronize()?;
                    let (got, mut want) = (b.read_vector_values(&got_dev)?, b.read_vector_values(&ref_dev)?);
                    if round_out {
                        for v in &mut want {
                            let bits = v.to_bits();
                            *v = f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000);
                        }
                    }
                    if let Some(i) = (0..want.len()).find(|&i| got[i].to_bits() != want[i].to_bits()) {
                        return Err(format!("few_pair r={rows} c={cols} m={inputs} rnd={round_out}: word {i} got {:#x} want {:#x}", got[i].to_bits(), want[i].to_bits()));
                    }
                    println!("FEW PAIR r={rows} c={cols} m={inputs} rnd={round_out} BIT-EXACT PASS words={}", want.len());
                    total_words += want.len();
                }
                for &(inputs, parts) in &[(4usize, &[3072usize, 256, 24, 3072, 1024, 1024][..]), (8, &[1024, 32][..])] {
                    let cols = 256usize;
                    let rows: usize = parts.iter().sum();
                    let w_host: Vec<u16> = (0..rows * cols).map(|i| (((i * 29 + 7) % 239) as u16) << 6 | 0x3c00).collect();
                    let x_host: Vec<f32> = (0..inputs * cols).map(|i| input(i + rows + 71)).collect();
                    let w_dev = matrix(b, &w_host)?;
                    let x_dev = b.upload_vector(&x_host)?.into_resident_shared()?;
                    let mut ref_dev = b.vector(inputs * rows)?;
                    b.project_staged_few_into(rows, cols, inputs, rows, &x_dev, &w_dev, &mut ref_dev)?;
                    let mut outs: Vec<_> = parts.iter().map(|&p| b.vector(inputs * p)).collect::<Result<_, _>>()?;
                    {
                        let mut refs: Vec<_> = outs.iter_mut().zip(parts.iter().copied()).collect();
                        b.project_staged_few_split_into(rows, cols, inputs, &x_dev, &w_dev, &mut refs)?;
                    }
                    b.synchronize()?;
                    let want = b.read_vector_values(&ref_dev)?;
                    let mut off = 0usize;
                    for (out, &p) in outs.iter().zip(parts) {
                        let got = b.read_vector_values(out)?;
                        for t in 0..inputs {
                            for j in 0..p {
                                if got[t * p + j].to_bits() != want[t * rows + off + j].to_bits() {
                                    return Err(format!("few_split m={inputs} part={p} t={t} j={j}: got {:#x} want {:#x}", got[t * p + j].to_bits(), want[t * rows + off + j].to_bits()));
                                }
                            }
                        }
                        off += p;
                    }
                    println!("FEW SPLIT m={inputs} parts={parts:?} BIT-EXACT PASS words={}", want.len());
                    total_words += want.len();
                }
            }

            total_words += super::dense_tc::run(b, &panel_fixtures)?;

            // 6. REFUSALS: verify bad shapes/alignments/overlaps record nothing.
            let tc_before = b.stats().recorded_kinds[Kind::PlainTcRows as usize];
            let mr_before = b.stats().recorded_kinds[Kind::MoeGroupRoute as usize];
            let w_dev = matrix(b, &vec![0u16; 128 * 64])?;
            let x_dev = b.upload_vector(&vec![0.0f32; 8 * 64 + 8])?.into_resident_shared()?;
            let mut y_dev = b.vector(8 * 128)?;
            assert!(b.project_tc_rows_into(128, 64, 8, &x_dev.resident_range(1..1 + 8 * 64)?, &w_dev, &mut y_dev).is_err());
            assert!(b.project_tc_rows_into(128, 64, 8, &x_dev.resident_range(0..7 * 64)?, &w_dev, &mut y_dev).is_err());
            assert!(b.project_tc_rows_strided_into(128, 64, 8, 127, &x_dev.resident_range(0..8 * 64)?, &w_dev, &mut y_dev).is_err());
            assert_eq!(b.stats().recorded_kinds[Kind::PlainTcRows as usize], tc_before);
            assert_eq!(b.stats().recorded_kinds[Kind::MoeGroupRoute as usize], mr_before);
            total_words += super::prefill_moe::run(b)?;
            drop(queue);
            println!("TC ROWS CHECK PASS total_words={total_words} guard_words_changed=0");
            Ok(())
        })
    }
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = check::run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("tc-rows-check requires Linux CUDA"); std::process::exit(1); }
}
