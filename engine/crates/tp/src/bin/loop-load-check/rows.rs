//! The weight products for many input rows on the device (mode `--product-rows`): the check and the time table.
//!
//! The contract is `Weights::product_rows`: `x` is `[inputs][columns]`, the result is `[inputs][matrix rows]`, and
//! each word is the word of `product` on its input row. For each picked matrix and each count of input rows:
//!
//! - the driver's call (`product_rows` on the loaded weights) against one `product` a row: every word;
//! - one launch for all rows (`prompt_ops::Op::ProjectRowsBf16`, then `Op::Transpose`) against the same words;
//! - one launch of the row product for each input row (`Backend::project_bf16_chain_into`) against the same words;
//! - the rows tile, an entry of its own (`Backend::project_staged_rows_into`), for every count of input rows that it
//!   admits, into an output with guard words on both sides: every word against the same words, no guard word changed;
//! - the time of the three, and for scale the time of `Op::ProjectRows` on made-up 32-bit weights of the same
//!   shape (the current family's fast form for rows; this model's weights are not held in that form).
//!
//! The driver's call takes the rows tile where it is ahead (`staged_product::rows_tile_ahead`), else one of the two
//! other forms by cost (`staged_product::rows_one_launch`); the table shows every form for every shape, so the rules
//! can be read against it. The driver's calls and the rows tile run in a scope of work (recorded launches).
//!
//! Last, the 32-bit rows tile on made-up matrices in shapes of the current family, to compare two builds.
//!
//! First, the expert runs of a step (`Weights::product_runs`, index rows): runs of 2, 9, 1, 24, 64 and 1,030 launch
//! rows over six experts of the first layer and of the last (a checkpoint may store the two in two forms). Gate-and-up: each launch row reads a scattered row of a smaller input
//! (rows read many times). Down: the launch rows in order, the result rows a permutation. Every result row against
//! `product` of the matrix of its run with its input row, every word; the launches of the rows tile are counted.
//! The driver's call reads the packed units (the packed rows tile, kind `PlainPackedRows`) where the load holds them;
//! then the same runs also go through the BF16 rows tile on the widened matrices (`resident_product_runs`), and
//! every word of the two must be the same.
use spark_gpu_probe::{driver::{Backend, DeviceVector}, prompt_ops::{Op, Shape}};
use spark_tp::{Result, bounded_decoder::{Compute, Device}, loop_decoder::{mix::{Weight, Weights}, operands::weights::{LoadedWeights, ResidentProduct}}};
use std::time::Instant;

/// Most multiply-adds of one product for many rows in this check (a bound on the time of the slow form).
const WORK: usize = 1 << 33;

/// One launch for `inputs` rows, then the turn to `[inputs][rows]`.
fn one_launch<'a>(backend: &Backend<'a>, op: Op, rows: usize, columns: usize, inputs: usize, x: &DeviceVector<'_>, matrix: &DeviceVector<'_>) -> Result<DeviceVector<'a>> {
    let mut shape = Shape::new(op, inputs).ok_or("no shape for the rows product")?;
    (shape.rows, shape.cols) = (rows as u32, columns as u32);
    let by_matrix_row = backend.prompt_operation(shape, [Some(x), None, None, None, None], Some(matrix))?;
    let mut turn = Shape::new(Op::Transpose, 1).ok_or("no shape for the turn")?;
    (turn.rows, turn.cols) = (rows as u32, inputs as u32);
    backend.prompt_operation(turn, [Some(&by_matrix_row), None, None, None, None], None)
}
/// One launch of the row product for each input row, into one output `[inputs][rows]`. A matrix above the envelope
/// of the warp tile is not run here (the driver's call runs it in parts).
fn a_launch_a_row<'a>(backend: &Backend<'a>, rows: usize, columns: usize, inputs: usize, x: &DeviceVector<'_>, matrix: &DeviceVector<'_>) -> Result<DeviceVector<'a>> {
    let whole = backend.vector(inputs * rows)?.into_resident_shared()?;
    for input in 0..inputs {
        let mut output = whole.resident_range(input * rows..(input + 1) * rows)?;
        backend.project_bf16_chain_into(rows, columns, &x.resident_range(input * columns..(input + 1) * columns)?, matrix, &mut output)?;
    }
    Ok(whole)
}
fn unequal(a: &[f32], b: &[f32]) -> usize { a.iter().zip(b).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + a.len().abs_diff(b.len()) }

/// The check of the expert runs (see the head of the file). It gives the words, the words different and the launches
/// of the rows tile.
fn runs<'d>(backend: &Backend<'d>, compute: &Device<'_, 'd>, weights: &LoadedWeights<'_, DeviceVector<'d>>, layers: usize) -> Result<(usize, usize, u128)> {
    use spark_gpu_probe::resident_launch::Kind;
    let tiles = || backend.stats().recorded_kinds[Kind::PlainStagedRows as usize] + backend.stats().recorded_kinds[Kind::PlainPackedRows as usize];
    let packed_kind = Kind::PlainPackedRows as usize;
    let counts = [2usize, 9, 1, 24, 64, 1030];
    let n = counts.iter().sum::<usize>();
    let (mut words, mut different, mut launches) = (0usize, 0usize, 0u128);
    let last = layers.saturating_sub(1);
    for (layer, down) in [(0, false), (0, true), (last, false), (last, true)] {
        let weight = |expert: usize| if down { Weight::Down(layer, expert) } else { Weight::GateUp(layer, expert) };
        let (_, _, _, columns) = weights.matrix(weight(0))?.ok_or("the expert matrix is not on the device")?;
        let runs = counts.iter().enumerate().map(|(expert, &count)| (weight(expert), count)).collect::<Vec<_>>();
        // Gate-and-up: 37 input rows, launch row i reads row (13 i + 5) mod 37. Down: n input rows in launch order,
        // result row j is launch row (7 j + 3) mod n (7 and n share no factor).
        let sources = if down { n } else { 37 };
        let input = (!down).then(|| (0..n).map(|i| (13 * i + 5) % sources).collect::<Vec<_>>());
        let output = down.then(|| (0..n).map(|j| (7 * j + 3) % n).collect::<Vec<_>>());
        if let Some(rows) = &output { let mut sorted = rows.clone(); sorted.sort_unstable(); if sorted != (0..n).collect::<Vec<_>>() { return Err("the check's output table is no permutation".into()); } }
        let host = (0..sources * columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>();
        let x = compute.upload(&host)?;
        let (recorded, packed_before) = (tiles(), backend.stats().recorded_kinds[packed_kind]);
        let got = compute.work(|| compute.read(&weights.product_runs(compute, &runs, &x, columns, input.as_deref(), output.as_deref())?))?;
        launches += tiles() - recorded;
        let packed = backend.stats().recorded_kinds[packed_kind] - packed_before;
        // The same runs on the widened matrices (the BF16 rows tile), where the load holds them: every word the same.
        let widened = runs.iter().map(|&(weight, count)| weights.matrix(weight).ok().flatten().map(|(value, range, rows, _)| (value, range, rows, count))).collect::<Option<Vec<_>>>();
        if let Some(matrices) = widened {
            let bf16 = compute.work(|| match ResidentProduct::resident_product_runs(compute, &matrices, &x, columns, input.as_deref(), output.as_deref())? {
                Some(value) => compute.read(&value).map(Some),
                None => Ok(None),
            })?;
            if let Some(bf16) = bf16 {
                words += bf16.len();
                different += unequal(&got, &bf16);
                eprintln!("LOOP LOAD RUNS PACKED-VS-BF16 layer={layer} matrix={} launch_rows={n} packed_launches={packed} words={} different={}", if down { "down" } else { "gate-and-up" }, bf16.len(), unequal(&got, &bf16));
            }
        }
        // The run of each launch row, then the words that result row j must hold.
        let run_of = counts.iter().enumerate().flat_map(|(expert, &count)| std::iter::repeat_n(expert, count)).collect::<Vec<_>>();
        let mut want = Vec::new();
        for j in 0..n {
            let launch = output.as_ref().map_or(j, |rows| rows[j]);
            let row = input.as_ref().map_or(launch, |rows| rows[launch]);
            want.extend(compute.work(|| compute.read(&weights.product(compute, weight(run_of[launch]), &compute.range(&x, row * columns, columns)?)?))?);
        }
        words += want.len();
        different += unequal(&got, &want);
        eprintln!("LOOP LOAD RUNS layer={layer} matrix={} launch_rows={n} input_rows={sources} words={} different={}", if down { "down" } else { "gate-and-up" }, want.len(), unequal(&got, &want));
    }
    Ok((words, different, launches))
}

pub fn run<'d>(backend: &Backend<'d>, weights: &LoadedWeights<'_, DeviceVector<'d>>, picked: &[(String, Weight)], layers: usize) -> Result<()> {
    let compute = Device::new(backend);
    let (run_words, run_different, run_launches) = runs(backend, &compute, weights, layers)?;
    eprintln!("LOOP LOAD RUNS words={run_words} different={run_different} tile_launches={run_launches}");
    if run_different > 0 { return Err(format!("the expert runs: {run_different} of {run_words} words differ from one product a row")); }
    if run_launches == 0 { return Err("the expert runs recorded no launch of the rows tile".into()); }
    let (mut words, mut different, mut first) = (0usize, 0usize, None);
    // The rows tile, the entry of its own: launches of the driver's call, and the check with guard words.
    let tile_kind = spark_gpu_probe::resident_launch::Kind::PlainStagedRows as usize;
    let (mut tile_launches, mut tile_words, mut tile_different, mut guard_words, mut guard_changed) = (0u128, 0usize, 0usize, 0usize, 0usize);
    const GUARD: usize = 64;
    let poison = f32::from_bits(0x7fc0_dead);
    for (label, weight) in picked {
        let (value, range, rows, columns) = weights.matrix(*weight)?.ok_or_else(|| format!("{weight:?} is not on the device"))?;
        let matrix = value.resident_range(range)?;
        for inputs in [1usize, 4, 8, 16, 64, 256, 1024] {
            let work = rows * columns * inputs;
            if work > WORK { continue; }
            // Input rows with 24 bits in each value and both signs, another row for each input.
            let host = (0..inputs * columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>();
            let x = compute.upload(&host)?;
            let direct = backend.upload_vector(&host)?.into_resident_shared()?;
            let mut want = Vec::with_capacity(inputs * rows);
            // The driver's calls run in a scope of work, as the driver runs them: launches are recorded there, so the
            // entries of their own take the shapes they admit.
            for input in 0..inputs { want.extend(compute.work(|| compute.read(&weights.product(&compute, *weight, &compute.range(&x, input * columns, columns)?)?))?); }
            let recorded = backend.stats().recorded_kinds[tile_kind];
            let through = unequal(&compute.work(|| compute.read(&weights.product_rows(&compute, *weight, &x, inputs)?))?, &want);
            tile_launches += backend.stats().recorded_kinds[tile_kind] - recorded;
            // The rows tile for every count of input rows that it admits (the driver's call takes it from 8), into an
            // output with guard words on both sides.
            let tile = spark_gpu_probe::staged_product::rows_shape_admitted(rows, columns, inputs);
            if tile {
                let whole = backend.upload_vector(&vec![poison; inputs * rows + 2 * GUARD])?.into_resident_shared()?;
                let ran = compute.work(|| {
                    if !backend.staged_rows_ready(rows, columns, inputs, &direct, &matrix) { return Ok(false); }
                    let mut out = whole.resident_range(GUARD..GUARD + inputs * rows)?;
                    backend.project_staged_rows_into(rows, columns, inputs, &direct, &matrix, &mut out)?;
                    Ok(true)
                })?;
                if !ran { return Err(format!("{label} {rows} x {columns}, {inputs} input rows: the rows tile did not run in a scope of work")); }
                let back = backend.read_vector_values(&whole)?;
                if back.len() != inputs * rows + 2 * GUARD { return Err(format!("{label}: {} words read back for {inputs} x {rows} words and the guards", back.len())); }
                guard_words += 2 * GUARD;
                guard_changed += back[..GUARD].iter().chain(&back[GUARD + inputs * rows..]).filter(|word| word.to_bits() != poison.to_bits()).count();
                tile_words += want.len();
                tile_different += unequal(&back[GUARD..GUARD + inputs * rows], &want);
            }
            let launch = unequal(&backend.read_vector_values(&one_launch(backend, Op::ProjectRowsBf16, rows, columns, inputs, &direct, &matrix)?)?, &want);
            let in_envelope = spark_gpu_probe::staged_product::part_rows(rows, columns) >= rows;
            let by_row = if in_envelope { unequal(&backend.read_vector_values(&a_launch_a_row(backend, rows, columns, inputs, &direct, &matrix)?)?, &want) } else { 0 };
            words += (2 + usize::from(in_envelope)) * want.len(); different += through + launch + by_row;
            if through + launch + by_row > 0 && first.is_none() { first = Some(format!("{label} {rows} x {columns}, {inputs} input rows: {through} words of the driver's call, {launch} words of the one launch, {by_row} words of a launch a row")); }
            // The times: each form some times back to back, then one wait for the device.
            let reps = (WORK / work).clamp(2, 40);
            // `scope`: the form runs in a scope of work (recorded launches), as the driver runs it.
            let time = |form: &str, scope: bool, run: &dyn Fn() -> Result<()>| -> Result<()> {
                let timed = || -> Result<f64> {
                    run()?;
                    backend.synchronize()?;
                    let began = Instant::now();
                    for _ in 0..reps { run()?; }
                    backend.synchronize()?;
                    Ok(began.elapsed().as_secs_f64() / reps as f64)
                };
                let seconds = if scope { compute.work(timed)? } else { timed()? };
                eprintln!("LOOP LOAD ROWS TIME shape={rows}x{columns} weight=\"{label}\" inputs={inputs} form={form} one_launch_rule={} reps={reps} microseconds={:.1} microseconds_a_row={:.2} multiply_adds_a_second={:.3e}",
                    spark_gpu_probe::staged_product::rows_one_launch(rows, columns, inputs), seconds * 1e6, seconds * 1e6 / inputs as f64, work as f64 / seconds);
                Ok(())
            };
            time("driver-call", true, &|| weights.product_rows(&compute, *weight, &x, inputs).map(drop))?;
            if tile {
                time("staged-rows-tile", true, &|| { let mut out = backend.vector(inputs * rows)?; backend.project_staged_rows_into(rows, columns, inputs, &direct, &matrix, &mut out) })?;
            }
            time("one-launch-bf16", false, &|| one_launch(backend, Op::ProjectRowsBf16, rows, columns, inputs, &direct, &matrix).map(drop))?;
            if in_envelope { time("a-launch-a-row", false, &|| a_launch_a_row(backend, rows, columns, inputs, &direct, &matrix).map(drop))?; }
            // For scale: the same shape with made-up 32-bit weights (4 bytes a weight; at most 256 MB here).
            if rows * columns <= 1 << 26 {
                let made = backend.upload_vector(&(0..rows * columns).map(|i| ((i * 37) % 512) as f32 / 512.0 - 0.5).collect::<Vec<_>>())?;
                time("one-launch-f32-made-up", false, &|| one_launch(backend, Op::ProjectRows, rows, columns, inputs, &direct, &made).map(drop))?;
            }
        }
    }
    // The 32-bit rows tile of the current family on made-up matrices, some launches back to back.
    for (rows, columns, inputs) in [(16384usize, 512usize, 256usize), (16384, 512, 1024), (4096, 1536, 64), (2304, 2304, 256), (1024, 4096, 1024)] {
        let x = backend.upload_vector(&(0..inputs * columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>())?;
        let made = backend.upload_vector(&(0..rows * columns).map(|i| ((i * 37) % 512) as f32 / 512.0 - 0.5).collect::<Vec<_>>())?;
        let reps = (WORK / (rows * columns * inputs)).clamp(2, 40);
        let mut best = f64::MAX;
        for _ in 0..3 {
            drop(one_launch(backend, Op::ProjectRows, rows, columns, inputs, &x, &made)?);
            backend.synchronize()?;
            let began = Instant::now();
            for _ in 0..reps { drop(one_launch(backend, Op::ProjectRows, rows, columns, inputs, &x, &made)?); }
            backend.synchronize()?;
            best = best.min(began.elapsed().as_secs_f64() / reps as f64);
        }
        eprintln!("LOOP LOAD ROWS TILE32 shape={rows}x{columns} inputs={inputs} tile={} reps={reps} best_of_3_microseconds={:.1} multiply_adds_a_second={:.3e}",
            spark_gpu_probe::plain_tile::admitted(rows, inputs, columns, x.device_address(), made.device_address()), best * 1e6, (rows * columns * inputs) as f64 / best);
    }
    eprintln!("LOOP LOAD ROWS STAGED tile_launches_of_the_driver_call={tile_launches} tile_words={tile_words} tile_words_different={tile_different} guard_words={guard_words} guard_words_changed={guard_changed}");
    if tile_different > 0 || guard_changed > 0 { return Err(format!("the rows tile: {tile_different} of {tile_words} words differ from one product a row, {guard_changed} of {guard_words} guard words changed")); }
    if tile_launches == 0 { return Err("the driver's call recorded no launch of the rows tile".into()); }
    eprintln!("LOOP LOAD ROWS PRODUCTS matrices={} words={words} different={different}", picked.len());
    match first { Some(first) => Err(format!("{different} of {words} words of a product for many rows differ from one product a row; first: {first}")), None => Ok(()) }
}
