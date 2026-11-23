//! Load check of the looped text family on one device: the weights of the load plan and the state of one chat.
//! It prints the host ledger, loads, compares the device bytes with the ledger, and reads every unit back.
//! It runs no model arithmetic and serves nothing.
//!
//! `loop-load-check MODEL CONTEXT OUT [--budget BYTES] [--workers N] [--plan-only] [--products] [--product-times] [--product-rows]`
//! - `MODEL`: the checkpoint folder. `CONTEXT`: tokens of one chat. `OUT`: a folder for `units.tsv`.
//! - `--budget BYTES`: a lower ceiling than the memory rule gives. `--plan-only`: stop after the host ledger; no device.
//! - `--products`: after the load, read every weight name of the driver. A weight that is read through a product
//!   runs one product on the device, and each output row is compared with the host chain on the device's own bytes.
//! - `--product-times`: after the load, time the launches of one product of each matrix shape of the model, and
//!   of made-up matrices in listed step shapes of the BF16 row product, for scale.
//! - `--product-rows`: after the load, the products for many input rows (child file `loop-load-check/rows.rs`): the
//!   driver's call and one launch for all rows against one product a row, every word, and the time of each form.
//!
//! Lines on standard error: `STARTUP` for each step, `LOOP LOAD PLAN`, `LOOP LOAD LEDGER`, `LOOP LOAD MEMORY`,
//! `LOOP LOAD DEVICE`, the line with ` ready `, `LOOP LOAD VERIFY`, `LOOP LOAD PRODUCTS`, `LOOP LOAD ROWS PRODUCTS` and `LOOP LOAD PASS`. A refusal or a difference
//! is one `FAIL` line and exit code 1.
#![forbid(unsafe_code)]
use engine_model::plan::{loop_load::{self, LoadForm, LoopCheckpoint}, loop_state::{LoopLedger, LoopStatePlan}};
use spark_tp::{Result, startup};
use std::{path::PathBuf, time::Instant};

/// The rule of the grid of a launch whose extent follows the context: a straight plane holds whole buckets.
const BUCKET: u64 = spark_gpu_probe::causal_ops::BUCKET_ROWS as u64;
#[cfg(target_os = "linux")]
#[path = "loop-load-check/rows.rs"]
mod rows;

const USAGE: &str = "loop-load-check MODEL CONTEXT OUT [--budget BYTES] [--workers N] [--plan-only] [--products] [--product-times] [--product-rows]";

#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // Only the device half, which is Linux code, reads every field.
struct Settings { model: PathBuf, context: u64, out: PathBuf, budget: Option<u64>, workers: usize, plan_only: bool, products: bool, product_times: bool, product_rows: bool }
fn settings(args: &[String]) -> Result<Settings> {
    let [model, context, out, options @ ..] = args else { return Err(format!("usage: {USAGE}")); };
    let number = |name: &str, text: &str| text.parse::<u64>().ok().filter(|&n| n > 0).ok_or_else(|| format!("{name} must be a positive whole number, got {text}"));
    let mut s = Settings { model: model.into(), context: number("CONTEXT", context)?, out: out.into(), budget: None,
        workers: std::thread::available_parallelism().map_or(4, |n| n.get()).min(16), plan_only: false, products: false, product_times: false, product_rows: false };
    let mut options = options.iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--plan-only" => s.plan_only = true,
            "--products" => s.products = true,
            "--product-times" => s.product_times = true,
            "--product-rows" => s.product_rows = true,
            "--budget" => s.budget = Some(number("--budget", options.next().ok_or("--budget needs a byte count")?)?),
            "--workers" => s.workers = number("--workers", options.next().ok_or("--workers needs a count")?)?.min(64) as usize,
            other => return Err(format!("unknown option {other}; usage: {USAGE}")),
        }
    }
    Ok(s)
}
fn plan_lines(checkpoint: &LoopCheckpoint, state: &LoopStatePlan, ledger: &LoopLedger) {
    let (b, tensors) = (checkpoint.load.bytes, &checkpoint.load.tensors);
    let count = |pick: fn(&LoadForm) -> bool| tensors.values().filter(|t| pick(&t.form)).count();
    eprintln!("LOOP LOAD PLAN tensors={} stored={} head={} expert_stacks={} stored_bytes={} head_bytes={} head_unused_bytes={} expert_bytes={} expert_source_bytes={} file_read_bytes={} image_tower_not_loaded_bytes={}",
        tensors.len(), count(|f| *f == LoadForm::Stored), count(|f| *f == LoadForm::LeadingRows), count(|f| matches!(f, LoadForm::WidenedExperts { .. })),
        b.stored, b.head, b.head_unused, b.experts, b.expert_sources, b.file, b.unused);
    eprintln!("LOOP LOAD LEDGER context={} plane_rows={} visits={} window_visits={} full_visits={} weights_bytes={} state_fixed_bytes={} state_plane_bytes={} step_bytes={} chat_bytes={} ledger_bytes={}",
        ledger.context, state.plane_rows, state.visits.len(), state.window_visits(), state.full_visits(), ledger.weights, ledger.state_fixed, ledger.state_planes, ledger.step, ledger.chat(), ledger.total);
}
fn run(args: &[String], started: Instant) -> Result<()> {
    let s = settings(args)?;
    startup::begin(0, started, "header_admission");
    let checkpoint = loop_load::read_folder(&s.model).map_err(|e| format!("This engine cannot load this model: {e}."))?;
    startup::step("ledger");
    let d = &checkpoint.description;
    let state = LoopStatePlan::new(d, s.context, BUCKET).map_err(|e| e.to_string())?;
    let ledger = LoopLedger::new(&checkpoint.load.bytes, &state).map_err(|e| e.to_string())?;
    plan_lines(&checkpoint, &state, &ledger);
    if s.plan_only { startup::done(); eprintln!("LOOP LOAD PLAN ONLY no device was opened"); return Ok(()); }
    device(&s, &checkpoint, &state, &ledger, started)
}
#[cfg(not(target_os = "linux"))]
fn device(_: &Settings, _: &LoopCheckpoint, _: &LoopStatePlan, _: &LoopLedger, _: Instant) -> Result<()> { Err("the device half of this check needs Linux and a GPU; use --plan-only for the host ledger".into()) }
#[cfg(target_os = "linux")]
fn device(s: &Settings, checkpoint: &LoopCheckpoint, state: &LoopStatePlan, ledger: &LoopLedger, started: Instant) -> Result<()> {
    use engine_model::plan::loop_state::context_limit;
    use spark_tp::loop_decoder::operands::{device::DeviceStore, load};
    startup::step("memory");
    let (_, total) = spark_gpu_probe::driver::device_memory_info(0)?;
    let free = spark_gpu_probe::driver::unified_available_bytes(total)?;
    // The memory rule of the text service: the free bytes less the larger of one tenth and 2 GiB.
    let mut budget = free.saturating_sub((free / 10).max(2 << 30)) as u64;
    if let Some(ceiling) = s.budget { budget = budget.min(ceiling); }
    let limit = context_limit(&checkpoint.description, &checkpoint.load.bytes, BUCKET, budget).map_err(|e| e.to_string())?;
    eprintln!("LOOP LOAD MEMORY device_total_bytes={total} available_bytes={free} budget_bytes={budget} context_limit={limit}");
    startup::step("memory_admission");
    ledger.admit(budget, limit).map_err(|refusal| format!("text service memory refusal: {refusal}"))?;
    let ptx = spark_gpu_probe::layer_ptx()?;
    let budget = usize::try_from(budget).map_err(|_| "budget exceeds the address space")?;
    startup::step("gpu_context_and_kernel_image");
    spark_gpu_probe::driver::with_backend(0, ptx, budget, |backend| {
        let before = backend.allocated_bytes();
        startup::step("weights");
        let mut store = DeviceStore::new(backend);
        let loaded = load::load(checkpoint, &s.model, &mut store, s.workers)?;
        backend.synchronize()?;
        let weights = backend.allocated_bytes() - before;
        startup::step("state");
        // One vector for each key store, each value store and each convolution ring of each visit, then the values of one
        // token step. A new chat starts from zeros.
        let words = |bytes: u64| usize::try_from(bytes / 4).map_err(|_| "state plane exceeds the address space".to_owned());
        let mut planes = Vec::new();
        for visit in &state.visits {
            for bytes in [visit.keys, visit.values].into_iter().chain(visit.conv.iter().map(|ring| ring.bytes)) {
                let mut plane = backend.vector(words(bytes)?)?;
                backend.zero_vector(&mut plane)?;
                planes.push(plane);
            }
        }
        let mut step = backend.vector(words(state.step.bytes)?)?;
        backend.zero_vector(&mut step)?;
        backend.synchronize()?;
        let device = backend.allocated_bytes() - before;
        let equal = device as u64 == ledger.total && weights as u64 == ledger.weights;
        eprintln!("LOOP LOAD DEVICE weights_bytes={weights} state_bytes={} device_bytes={device} ledger_bytes={} equal={equal} tensors={} units={} state_vectors={} allocated_before_bytes={before} pinned_staging_bytes={} accounted_bytes={} allocate_seconds={:.3} fill_seconds={:.3}",
            device - weights, ledger.total, loaded.report.tensors, loaded.report.units, planes.len() + 1, backend.pinned_staging_bytes(), backend.accounted_bytes(), loaded.report.allocate_seconds, loaded.report.fill_seconds);
        if !equal { return Err(format!("device bytes differ from the host ledger: device {device}, ledger {}; weights {weights}, ledger {}", ledger.total, ledger.weights)); }
        // Startup ends here: the weights are on the device and the state of one chat is allocated and clear.
        startup::done();
        eprintln!("loop_load_check ready world=1 rank=0 context={} resident_bytes={weights} session_bytes={} device_bytes={device} ledger_bytes={} seconds={:.3}",
            ledger.context, device - weights, ledger.total, started.elapsed().as_secs_f64());
        let verified = load::verify(&loaded, &mut store, s.workers)?;
        let fingerprint = loaded.fingerprint();
        std::fs::create_dir_all(&s.out).and_then(|_| std::fs::write(s.out.join("units.tsv"), load::unit_lines(&loaded.units))).map_err(|e| format!("{}: {e}", s.out.display()))?;
        eprintln!("LOOP LOAD VERIFY units={} bytes={} different={} seconds={:.3} fingerprint={fingerprint}", verified.units, verified.bytes, verified.different.len(), verified.seconds);
        if !verified.different.is_empty() {
            return Err(format!("{} of {} units differ on the device from the bytes of the files; first: {}", verified.different.len(), verified.units, verified.different[0]));
        }
        if s.products {
            use engine_model::{flash::nvfp4::bf16_to_f32, shapes::Ffn};
            use spark_tp::{bounded_decoder::{Compute, Device}, loop_decoder::{mix::{Weight, Weights}, operands::{load::Store, weights::LoadedWeights}}};
            let began = Instant::now();
            let d = &checkpoint.description;
            for visit in &d.visits { loaded.operands.visit(d, visit)?; }
            let weights = LoadedWeights::new(checkpoint, &load::Files::open(checkpoint, &s.model)?, &loaded.operands)?;
            let compute = Device::new(backend);
            let mut names = weights.names().map(|(weight, _)| weight).collect::<Vec<_>>();
            // Three experts of each layer: the first, the second and the last.
            for (i, layer) in d.layers.iter().enumerate() {
                if let Ffn::Experts(e) = layer.ffn { for expert in [0, 1, e.count.saturating_sub(1)] { names.extend([Weight::GateUp(i, expert as usize), Weight::Down(i, expert as usize)]); } }
            }
            let (mut read, mut device_products, mut host_products, mut rows_compared, mut different, mut first) = (0usize, 0usize, 0usize, 0usize, 0usize, None);
            let (mut poisoned_rows, mut poisoned_different) = (0usize, 0usize);
            // The staged product, the entry of its own: launches of the driver's call, and the check with guard words.
            let staged_kind = spark_gpu_probe::resident_launch::Kind::PlainStaged as usize;
            let (mut staged_launches, mut staged_rows, mut staged_different, mut guard_words, mut guard_changed) = (0u128, 0usize, 0usize, 0usize, 0usize);
            const GUARD: usize = 64;
            let poison = f32::from_bits(0x7fc0_dead);
            for weight in names {
                let Some((rows, columns)) = weights.product_shape(weight)? else {
                    if weights.host_words(weight)?.is_empty() { return Err(format!("{weight:?} has no words")); }
                    read += 1;
                    continue;
                };
                // An input row with 24 bits in each value and both signs.
                let x = (0..columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>();
                // The matrix as the device holds it: read back for a matrix that stays there, the host words for the others.
                let matrix = match weights.matrix(weight)? {
                    Some((value, words, _, _)) => {
                        let mut bytes = vec![0u8; words.len() * 4];
                        store.read(value, words.start, &mut bytes)?;
                        device_products += 1;
                        bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes(*b))).collect::<Vec<_>>()
                    }
                    None => { host_products += 1; weights.host_words(weight)?.to_vec() }
                };
                // The driver's call, in a scope of work as the driver runs it: launches are recorded there, so a matrix
                // of whole stages takes the staged product.
                let recorded = backend.stats().recorded_kinds[staged_kind];
                let got = compute.work(|| compute.read(&weights.product(&compute, weight, &compute.upload(&x)?)?))?;
                staged_launches += backend.stats().recorded_kinds[staged_kind] - recorded;
                // A matrix on the device once more, into an output that holds a pattern that is not a number: a row
                // that the kernel does not write is seen.
                if let Some((value, words, _, _)) = weights.matrix(weight)? {
                    let mut out = backend.upload_vector(&vec![f32::from_bits(0x7fc0_dead); rows])?;
                    backend.project_bf16_chain_into(rows, columns, &backend.upload_vector(&x)?, &value.resident_range(words.clone())?, &mut out)?;
                    let again = backend.read_vector_values(&out)?;
                    poisoned_rows += rows;
                    poisoned_different += again.iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + again.len().abs_diff(got.len());
                    // The staged product into an output with guard words on both sides (the first part of a matrix
                    // above the envelope): the rows are the rows of the driver's call and no guard word changes.
                    let part = spark_gpu_probe::staged_product::part_rows(rows, columns).min(rows);
                    if spark_gpu_probe::staged_product::shape_admitted(part, columns) {
                        let whole = backend.upload_vector(&vec![poison; part + 2 * GUARD])?.into_resident_shared()?;
                        let input = backend.upload_vector(&x)?;
                        let matrix = value.resident_range(words.start..words.start + part * columns / 2)?;
                        let ran = compute.work(|| {
                            if !backend.staged_product_ready(part, columns, &input, &matrix) { return Ok(false); }
                            let mut out = whole.resident_range(GUARD..GUARD + part)?;
                            backend.project_staged_into(part, columns, &input, &matrix, &mut out)?;
                            Ok(true)
                        })?;
                        if !ran { return Err(format!("{weight:?} {part} x {columns}: the staged product did not run in a scope of work")); }
                        let back = backend.read_vector_values(&whole)?;
                        if back.len() != part + 2 * GUARD { return Err(format!("{weight:?}: {} words read back for {part} rows and the guards", back.len())); }
                        guard_words += 2 * GUARD;
                        guard_changed += back[..GUARD].iter().chain(&back[GUARD + part..]).filter(|word| word.to_bits() != poison.to_bits()).count();
                        staged_rows += part;
                        staged_different += back[GUARD..GUARD + part].iter().zip(&got).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                    }
                }
                if matrix.len() != rows * columns || got.len() != rows { return Err(format!("{weight:?}: {} matrix words and {} output rows for {rows} x {columns}", matrix.len(), got.len())); }
                // The host chain: rising columns from +0.0, each product rounded before its add.
                let unequal = matrix.chunks(columns).zip(&got).filter(|(row, out)| row.iter().zip(&x).fold(0.0f32, |sum, (w, x)| sum + w * x).to_bits() != out.to_bits()).count();
                rows_compared += rows; different += unequal;
                if unequal > 0 && first.is_none() { first = Some(format!("{weight:?} {rows} x {columns}: {unequal} rows")); }
            }
            eprintln!("LOOP LOAD PRODUCTS STAGED launches_of_the_driver_call={staged_launches} guard_rows={staged_rows} guard_rows_different={staged_different} guard_words={guard_words} guard_words_changed={guard_changed}");
            eprintln!("LOOP LOAD PRODUCTS visits={} read_as_words={read} device_products={device_products} host_matrix_products={host_products} rows={rows_compared} different={different} poisoned_rows={poisoned_rows} poisoned_different={poisoned_different} host_words_bytes={} seconds={:.3}",
                d.visits.len(), weights.host_bytes(), began.elapsed().as_secs_f64());
            if let Some(first) = first { return Err(format!("{different} of {rows_compared} product rows differ from the host chain; first: {first}")); }
            if poisoned_different > 0 { return Err(format!("{poisoned_different} of {poisoned_rows} rows of a launch into a poisoned output differ from the first launch")); }
            if staged_different > 0 || guard_changed > 0 { return Err(format!("the staged product: {staged_different} of {staged_rows} rows differ from the driver's call, {guard_changed} of {guard_words} guard words changed")); }
            if staged_launches == 0 { return Err("the driver's call recorded no launch of the staged product".into()); }
        }
        if s.product_times {
            use spark_gpu_probe::driver::DeviceVector;
            use spark_tp::loop_decoder::{mix::Weight, operands::{load::Store, weights::LoadedWeights}};
            let weights = LoadedWeights::new(checkpoint, &load::Files::open(checkpoint, &s.model)?, &loaded.operands)?;
            // `entry`: "listed" = the entry of the listed step shapes; "any-shape" = the same kernel for a matrix of any
            // shape; "staged" = the staged product, recorded in a scope of work as the driver runs it.
            let scope = spark_tp::bounded_decoder::Device::new(backend);
            let time = |label: &str, rows: usize, columns: usize, weight: &DeviceVector<'_>, entry: &str| -> Result<()> {
                use spark_tp::bounded_decoder::Compute;
                let x = backend.upload_vector(&(0..columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>())?;
                let mut out = backend.vector(rows)?;
                let bytes = rows * columns * 2;
                let threads = spark_gpu_probe::plain_admission::bf16_geometry(rows).1 as usize;
                let tile = spark_gpu_probe::plain_admission::bf16_admitted(rows, columns, weight.device_address() as usize, x.device_address() as usize, threads);
                if entry == "staged" && !spark_gpu_probe::staged_product::shape_admitted(rows, columns) { return Ok(()); }
                // About 4 GB of weight bytes for each shape, in 8 to 400 launches.
                let launches = ((1usize << 32) / bytes).clamp(8, 400);
                let mut run = |count: usize| -> Result<()> {
                    for _ in 0..count {
                        match entry {
                            "listed" => backend.project_bf16_into(rows, columns, &x, weight, &mut out)?,
                            "staged" => backend.project_staged_into(rows, columns, &x, weight, &mut out)?,
                            _ => backend.project_bf16_chain_into(rows, columns, &x, weight, &mut out)?,
                        }
                    }
                    backend.synchronize()
                };
                let mut timed = || -> Result<f64> {
                    run(4)?;
                    let began = Instant::now();
                    run(launches)?;
                    Ok(began.elapsed().as_secs_f64())
                };
                let seconds = if entry == "staged" { scope.work(&mut timed)? } else { timed()? };
                eprintln!("LOOP LOAD PRODUCT TIME shape={rows}x{columns} weight=\"{label}\" entry={} form={} bytes={bytes} launches={launches} microseconds_a_launch={:.1} gb_a_second={:.1} nanoseconds_a_column={:.2}",
                    entry, if entry == "staged" { "staged-tile" } else if tile { "warp-tile" } else { "scalar-chain" }, seconds * 1e6 / launches as f64, bytes as f64 * launches as f64 / seconds / 1e9, seconds * 1e9 / launches as f64 / columns as f64);
                Ok(())
            };
            // The experts of the first and the last layer, then one weight for each other matrix shape that stays on the device.
            let last = checkpoint.description.layers.len().saturating_sub(1);
            let mut picked = vec![("expert gate-and-up, first layer".to_owned(), Weight::GateUp(0, 0)), ("expert down, first layer".to_owned(), Weight::Down(0, 0)),
                ("expert gate-and-up, last layer".to_owned(), Weight::GateUp(last, 0)), ("expert down, last layer".to_owned(), Weight::Down(last, 0))];
            let mut seen = std::collections::BTreeSet::new();
            for (weight, _) in weights.names() {
                if let Some((_, _, rows, columns)) = weights.matrix(weight)? && seen.insert((rows, columns)) { picked.push((format!("{weight:?}"), weight)); }
            }
            for (label, weight) in picked {
                let (value, words, rows, columns) = weights.matrix(weight)?.ok_or_else(|| format!("{weight:?} is not on the device"))?;
                time(&label, rows, columns, &value.resident_range(words.clone())?, "staged")?;
                // The kernel of the listed shapes on the same matrix (one thread a row here): few launches, it is slow.
                time(&label, rows, columns, &value.resident_range(words)?, "any-shape")?;
                // A matrix above the envelope of the warp tile: the driver runs it in parts. Time that form too.
                let part = spark_gpu_probe::staged_product::part_rows(rows, columns);
                if part < rows {
                    use spark_tp::{bounded_decoder::{Compute, Device}, loop_decoder::mix::Weights};
                    let compute = Device::new(backend);
                    let x = compute.upload(&(0..columns).map(|k| ((k as u32).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>())?;
                    let (launches, bytes) = (8usize, rows * columns * 2);
                    // In a scope of work, as the driver runs it: the parts take the staged product there.
                    let seconds = compute.work(|| {
                        drop(weights.product(&compute, weight, &x)?);
                        backend.synchronize()?;
                        let began = Instant::now();
                        for _ in 0..launches { drop(weights.product(&compute, weight, &x)?); }
                        backend.synchronize()?;
                        Ok(began.elapsed().as_secs_f64())
                    })?;
                    eprintln!("LOOP LOAD PRODUCT TIME shape={rows}x{columns} weight=\"{label}\" entry=driver form=parts-of-{part}-rows bytes={bytes} launches={launches} microseconds_a_launch={:.1} gb_a_second={:.1} nanoseconds_a_column={:.2}",
                        seconds * 1e6 / launches as f64, bytes as f64 * launches as f64 / seconds / 1e9, seconds * 1e9 / launches as f64 / columns as f64);
                }
            }
            // For scale: made-up matrices in listed step shapes of the row product, through the entry of the listed shapes.
            for (rows, columns) in [(4096usize, 1536usize), (1024, 8192), (32768, 512), (128, 4096)] {
                let mut made = store.create(rows * columns / 2)?;
                let bytes = (0..rows * columns).flat_map(|i| (0x3c00u16 + ((i * 37) % 0x200) as u16 + ((i as u16 & 1) << 15)).to_le_bytes()).collect::<Vec<_>>();
                for (part, chunk) in bytes.chunks(8 << 20).enumerate() { store.write(&mut made, part * (2 << 20), chunk)?; }
                time("made-up values", rows, columns, &made, "listed")?;
                time("made-up values", rows, columns, &made, "staged")?;
            }
        }
        if s.product_rows {
            use spark_tp::loop_decoder::{mix::Weight, operands::weights::LoadedWeights};
            let weights = LoadedWeights::new(checkpoint, &load::Files::open(checkpoint, &s.model)?, &loaded.operands)?;
            // The experts of the first layer, then one weight for each other matrix shape that stays on the device.
            let mut picked = vec![("expert gate-and-up, first layer".to_owned(), Weight::GateUp(0, 0)), ("expert down, first layer".to_owned(), Weight::Down(0, 0))];
            let mut seen = std::collections::BTreeSet::new();
            for (_, weight) in &picked { if let Some((_, _, rows, columns)) = weights.matrix(*weight)? { seen.insert((rows, columns)); } }
            for (weight, _) in weights.names() {
                if let Some((_, _, rows, columns)) = weights.matrix(weight)? && seen.insert((rows, columns)) { picked.push((format!("{weight:?}"), weight)); }
            }
            rows::run(backend, &weights, &picked)?;
        }
        eprintln!("LOOP LOAD PASS context={} device_bytes={device} ledger_bytes={} units={} fingerprint={fingerprint}", ledger.context, ledger.total, verified.units);
        Ok(())
    })
}
fn main() {
    let started = Instant::now();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Err(error) = run(&args, started) {
        startup::failed(&error);
        eprintln!("FAIL {error}");
        std::process::exit(1);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn args(text: &str) -> Vec<String> { text.split_whitespace().map(str::to_owned).collect() }
    #[test]
    fn settings_take_three_places_and_named_options() {
        let s = settings(&args("model 4096 out --budget 1000 --workers 3 --plan-only --products")).unwrap();
        assert_eq!((s.model, s.context, s.out, s.budget, s.workers, s.plan_only, s.products), (PathBuf::from("model"), 4096, PathBuf::from("out"), Some(1000), 3, true, true));
        let s = settings(&args("model 1 out")).unwrap();
        assert!(s.budget.is_none() && !s.plan_only && !s.products && (1..=16).contains(&s.workers));
        for bad in ["model 4096", "model 0 out", "model many out", "model 1 out --budget", "model 1 out --budget 0", "model 1 out --fast"] {
            assert!(settings(&args(bad)).is_err(), "{bad}");
        }
    }
}
