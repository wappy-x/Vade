//! Check of the packed expert product of the looped text family on one device.
//!
//! `loop-pack-check MODEL [--experts N] [--workers N] [--verify] [--times] [--plan-only] [--sweep-only] [--bad-id VALUE]`
//! - `MODEL`: the checkpoint folder. The whole load plan is loaded: every expert stack in both forms, the widened
//!   one and the packed one.
//! - `--experts N`: experts of each layer that the product check reads (the first, the second, the last and others
//!   between them; 3 when not given).
//! - `--verify`: read every unit of the load back and compare its hash.
//! - `--times`: time the packed product and the BF16 product on 64 different experts of one 4-bit layer and of one
//!   8-bit layer, each expert launched one time after the other, as a token does.
//!   Then `PACK CHECK TALL`: the stored rows of up to 8 experts as ONE matrix in one launch, against the same
//!   experts launched one by one (what a launch for all picked experts of a visit could gain).
//! - `--plan-only`: stop after the byte sums of the load plan; no device.
//! - `--sweep-only`: the sweep of made-up matrices alone, with no weights on the device. This is the run for a
//!   memory checker (`compute-sanitizer --tool memcheck`), which is slow.
//! - `--bad-id VALUE`: one launch of kind 96 whose expert number is VALUE (for example 3, -1, 0.5 or NaN with a stack
//!   of 3): the device must stop it. The run passes when the launch ends in an error. One start a value.
//!
//! What it checks after the load (`PACK CHECK DEVICE`: the device bytes equal the load plan), in this order:
//! 1. `PACK CHECK SWEEP`: made-up matrices of small shapes, both forms, both row orders, block scales over their
//!    whole range. Each launch writes into an output with guard words on both sides, and each output row is
//!    compared with the host chain on the matrix widened by the load plan's widen step.
//!    Then `PACK CHECK REFUSALS`: bad calls of the backend (an operand one word short or long, no rows, columns
//!    that are not whole stages, an operand that is not aligned, an output inside an operand, storage in segments,
//!    launches not recorded): each must be an error and must record no launch; a good call with the same operands
//!    is the control. And `PACK CHECK FALLBACKS`: the three calls for which the driver quietly reads the widened
//!    matrix.
//! 2. `PACK CHECK PRODUCTS`: the real experts through the driver's own call (`LoadedWeights::product` inside
//!    `Compute::work`). Each output row is compared with the BF16 product of the device on the widened stack and
//!    with the host chain on the widened matrix as the device holds it. A product that did not launch the packed
//!    kernel is a failure.
//! 3. `PACK CHECK GUARDS`: the same real experts once more through the backend call, into an output that holds a
//!    pattern that is not a number, with guard words on both sides.
//! 4. `PACK CHECK GROUPS`: 8 real experts of each layer and projection in ONE grouped launch through the driver's
//!    call (`Weights::products`), against one launch an expert, and again through the backend with guard words. The
//!    sweep holds groups of made-up matrices too (`PACK CHECK GROUP SWEEP`) and bad grouped calls. Both also run the
//!    launch that reads its expert numbers on the device (kind 96, `project_packed_picked_into`) on a stack of the
//!    same matrices, against one launch an expert, with guard words.
//!
//! A refusal or a difference is one `FAIL` line and exit code 1. `PACK CHECK PASS` ends a run with no difference.
#![forbid(unsafe_code)]
use engine_model::plan::loop_load::{self, LoopCheckpoint};
use spark_tp::{Result, startup};
use std::{path::PathBuf, time::Instant};

const USAGE: &str = "loop-pack-check MODEL [--experts N] [--workers N] [--verify] [--times] [--plan-only] [--sweep-only] [--bad-id VALUE]";
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // Only the device half, which is Linux code, reads every field.
struct Settings { model: PathBuf, experts: usize, workers: usize, verify: bool, times: bool, plan_only: bool, sweep_only: bool, bad_id: Option<f32> }
fn settings(args: &[String]) -> Result<Settings> {
    let [model, options @ ..] = args else { return Err(format!("usage: {USAGE}")); };
    let number = |name: &str, text: &str| text.parse::<usize>().ok().filter(|&n| n > 0).ok_or_else(|| format!("{name} must be a positive whole number, got {text}"));
    let mut s = Settings { model: model.into(), experts: 3, workers: std::thread::available_parallelism().map_or(4, |n| n.get()).min(16), verify: false, times: false, plan_only: false, sweep_only: false, bad_id: None };
    let mut options = options.iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--verify" => s.verify = true,
            "--times" => s.times = true,
            "--plan-only" => s.plan_only = true,
            "--sweep-only" => s.sweep_only = true,
            // One launch of kind 96 whose expert number is this word; it must stop the device. No weights are loaded.
            "--bad-id" => { s.bad_id = Some(options.next().ok_or("--bad-id needs a value")?.parse::<f32>().map_err(|e| format!("--bad-id: {e}"))?); s.sweep_only = true; }
            "--experts" => s.experts = number("--experts", options.next().ok_or("--experts needs a count")?)?,
            "--workers" => s.workers = number("--workers", options.next().ok_or("--workers needs a count")?)?.min(64),
            other => return Err(format!("unknown option {other}; usage: {USAGE}")),
        }
    }
    Ok(s)
}
/// `count` different experts of a stack of `total`: the first, the second, the last, and others spread between.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))] // Only the device half, which is Linux code, and the tests call it.
fn sample(total: u64, count: usize) -> Vec<u64> {
    let mut picked = std::collections::BTreeSet::new();
    for expert in [0, 1, total.saturating_sub(1)] { if expert < total && picked.len() < count { picked.insert(expert); } }
    let mut next = 0u64;
    while picked.len() < count.min(total as usize) { next = (next + 37) % total; picked.insert(next); }
    picked.into_iter().collect()
}
fn run(args: &[String], started: Instant) -> Result<()> {
    let s = settings(args)?;
    startup::begin(0, started, "header_admission");
    let checkpoint = loop_load::read_folder(&s.model).map_err(|e| format!("This engine cannot load this model: {e}."))?;
    let (b, tensors) = (checkpoint.load.bytes, &checkpoint.load.tensors);
    let (mut four_bit, mut eight_bit, mut units) = (0, 0, 0u64);
    for t in tensors.values() {
        let Some(unit) = t.unit() else { continue; };
        if unit.codes == loop_load::ExpertCodes::Nvfp4 { four_bit += 1; } else { eight_bit += 1; }
        units += t.stack().map_or(0, |[count, ..]| count);
    }
    eprintln!("PACK CHECK PLAN tensors={} widened_stacks={} packed_stacks={} packed_4bit_stacks={four_bit} packed_8bit_stacks={eight_bit} packed_units={units} stored_bytes={} head_bytes={} widened_expert_bytes={} packed_expert_bytes={} device_weight_bytes={} file_read_bytes={}",
        tensors.len(), b.expert_stacks, b.packed_stacks, b.stored, b.head, b.experts, b.packed, b.device, b.file);
    if s.plan_only { startup::done(); eprintln!("PACK CHECK PLAN ONLY no device was opened"); return Ok(()); }
    device(&s, &checkpoint, started)
}
#[cfg(not(target_os = "linux"))]
fn device(_: &Settings, _: &LoopCheckpoint, _: Instant) -> Result<()> { Err("the device half of this check needs Linux and a GPU; use --plan-only for the byte sums".into()) }
#[cfg(target_os = "linux")]
fn device(s: &Settings, checkpoint: &LoopCheckpoint, started: Instant) -> Result<()> {
    use engine_model::{flash::nvfp4::bf16_to_f32, plan::{loop_load::ExpertCodes, loop_packed_unit::UnitShape, loop_widen, storage_groups::{GATE_UP_ROWS, NVFP4_NIBBLES}}, shapes::{Ffn, Proj}};
    use spark_gpu_probe::{driver::{Backend, DeviceVector}, packed_product, resident_launch::Kind};
    use spark_tp::{bounded_decoder::{Compute, Device}, loop_decoder::{mix::{Weight, Weights}, operands::{device::DeviceStore, load::{self, Store}, weights::{LoadedWeights, ResidentProduct}}}};

    /// Guard words on each side of an output, and the pattern of a guard and of a word that no launch wrote.
    const GUARD: usize = 64;
    const MARK: u32 = 0x7fc0_dead;
    startup::step("memory");
    let (_, total) = spark_gpu_probe::driver::device_memory_info(0)?;
    let free = spark_gpu_probe::driver::unified_available_bytes(total)?;
    // The memory rule of the text service: the free bytes less the larger of one tenth and 2 GiB.
    let budget = free.saturating_sub((free / 10).max(2 << 30));
    eprintln!("PACK CHECK MEMORY device_total_bytes={total} available_bytes={free} budget_bytes={budget} device_weight_bytes={}", checkpoint.load.bytes.device);
    if !s.sweep_only && checkpoint.load.bytes.device > budget as u64 { return Err(format!("the weights in both expert forms need {} bytes and the budget is {budget}", checkpoint.load.bytes.device)); }
    let ptx = spark_gpu_probe::layer_ptx()?;
    startup::step("gpu_context_and_kernel_image");
    spark_gpu_probe::driver::with_backend(0, ptx, budget, |backend| {
        let input = |columns: usize, salt: u32| (0..columns as u32).map(|k| ((k + salt).wrapping_mul(2_654_435_761) >> 8) as f32 / 8_388_608.0 - 1.0).collect::<Vec<_>>();
        let made = |count: usize, salt: usize| (0..count).map(|i| ((i + salt).wrapping_mul(2_654_435_761) >> 9) as u8).collect::<Vec<_>>();
        let packed_launches = |backend: &Backend<'_>| backend.stats().recorded_kinds[Kind::PlainPacked as usize];
        // One packed launch through the backend into an output with guard words on both sides. Returns the rows and
        // the number of guard words that changed.
        let guarded = |backend: &Backend<'_>, unit: UnitShape, matrix: &DeviceVector<'_>, x: &DeviceVector<'_>| -> Result<(Vec<f32>, usize)> {
            let whole = backend.upload_vector(&vec![f32::from_bits(MARK); unit.rows + 2 * GUARD])?.into_resident_shared()?;
            let mut output = whole.resident_range(GUARD..GUARD + unit.rows)?;
            backend.project_packed_into(unit.rows, unit.columns, unit.codes == ExpertCodes::Mxfp8, unit.order.is_some(), x, matrix, &mut output)?;
            backend.flush_resident_launches()?;
            backend.synchronize()?;
            let words = backend.read_vector_values(&whole)?;
            let broken = words[..GUARD].iter().chain(&words[GUARD + unit.rows..]).filter(|w| w.to_bits() != MARK).count();
            Ok((words[GUARD..GUARD + unit.rows].to_vec(), broken))
        };
        // One grouped launch into an output with guard words on both sides. Returns the rows and the guard words that changed.
        let guarded_group = |backend: &Backend<'_>, unit: UnitShape, matrices: &[&DeviceVector<'_>], x: &DeviceVector<'_>, shared: bool| -> Result<(Vec<f32>, usize)> {
            let words = unit.rows * matrices.len();
            let whole = backend.upload_vector(&vec![f32::from_bits(MARK); words + 2 * GUARD])?.into_resident_shared()?;
            let mut output = whole.resident_range(GUARD..GUARD + words)?;
            backend.project_packed_group_into(unit.rows, unit.columns, unit.codes == ExpertCodes::Mxfp8, unit.order.is_some(), x, shared, matrices, &mut output)?;
            backend.flush_resident_launches()?;
            backend.synchronize()?;
            let all = backend.read_vector_values(&whole)?;
            let broken = all[..GUARD].iter().chain(&all[GUARD + words..]).filter(|w| w.to_bits() != MARK).count();
            Ok((all[GUARD..GUARD + words].to_vec(), broken))
        };
        let group_launches = |backend: &Backend<'_>| backend.stats().recorded_kinds[Kind::PlainPackedGroup as usize];
        // The launch whose experts the route picked on the device (kind 96): the same marked output with guard words.
        #[allow(clippy::too_many_arguments)]
        let guarded_picked = |backend: &Backend<'_>, unit: UnitShape, stack: &DeviceVector<'_>, experts: usize, ids: &DeviceVector<'_>, x: &DeviceVector<'_>, shared: bool| -> Result<(Vec<f32>, usize)> {
            let words = unit.rows * ids.len();
            let whole = backend.upload_vector(&vec![f32::from_bits(MARK); words + 2 * GUARD])?.into_resident_shared()?;
            let mut output = whole.resident_range(GUARD..GUARD + words)?;
            backend.project_packed_picked_into(unit.rows, unit.columns, unit.codes == ExpertCodes::Mxfp8, unit.order.is_some(), x, shared, stack, experts, ids, &mut output)?;
            backend.flush_resident_launches()?;
            backend.synchronize()?;
            let all = backend.read_vector_values(&whole)?;
            let broken = all[..GUARD].iter().chain(&all[GUARD + words..]).filter(|w| w.to_bits() != MARK).count();
            Ok((all[GUARD..GUARD + words].to_vec(), broken))
        };
        let picked_launches = |backend: &Backend<'_>| backend.stats().recorded_kinds[Kind::PlainPackedPicked as usize];
        let mut store = DeviceStore::new(backend);

        // The sweep of made-up matrices: step 1 of the check, and the whole run with `--sweep-only`. Returns its launches.
        let sweep = |store: &mut DeviceStore<'_, '_>| -> Result<usize> {
            startup::step("sweep");
            let began = Instant::now();
            let (mut launches, mut rows_compared, mut different, mut unwritten, mut broken_guards, mut first) = (0usize, 0usize, 0usize, 0usize, 0usize, None);
            {
                let queue = backend.resident_queue()?;
                backend.enable_resident_launches()?;
                backend.resident_row_pass(1);
                let factor = loop_widen::expert_factor(0.7, 2688.0).map_err(|e| e.to_string())?;
                for (codes, column_list) in [(ExpertCodes::Nvfp4, [64usize, 128, 192, 448, 1152, 8192]), (ExpertCodes::Mxfp8, [128, 256, 384, 640, 1152, 8192])] {
                    for columns in column_list {
                        for rows in [1usize, 2, 31, 32, 33, 64, 70, 160] {
                            for order in [None, Some(GATE_UP_ROWS)] {
                                if order.is_some() && rows % 2 != 0 { continue; }
                                let unit = UnitShape { codes, rows, columns, order };
                                let salt = rows * 7 + columns;
                                // Every code. The 4-bit form: every finite scale byte of both signs. The 8-bit form: scale bytes from
                                // 2^-127 (weights below the normal range of BF16) to 2^73, so that no sum leaves the range of a 32-bit value.
                                let (stored, scales) = if codes == ExpertCodes::Nvfp4 {
                                    (made(rows * columns / 2, salt), made(rows * columns / 16, salt + 1).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>())
                                } else {
                                    (made(rows * columns, salt).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>(), made(rows * columns / 32, salt + 1).into_iter().map(|b| b % 201).collect::<Vec<_>>())
                                };
                                let mut bytes = vec![0u8; unit.layout().map_err(|e| e.to_string())?.bytes];
                                unit.fill(&stored, &scales, (codes == ExpertCodes::Nvfp4).then_some(factor), &mut bytes).map_err(|e| e.to_string())?;
                                let mut widened = vec![0u8; rows * columns * 2];
                                if codes == ExpertCodes::Nvfp4 { loop_widen::widen_nvfp4(&stored, &scales, factor, rows, columns, order, NVFP4_NIBBLES, &mut widened) } else { loop_widen::widen_mxfp8(&stored, &scales, rows, columns, order, &mut widened) }.map_err(|e| e.to_string())?;
                                let mut matrix = store.create(bytes.len() / 4)?;
                                store.write(&mut matrix, 0, &bytes)?;
                                for salt in [3u32, 4] {
                                    let x = input(columns, salt);
                                    let (got, broken) = guarded(backend, unit, &matrix, &backend.upload_vector(&x)?)?;
                                    let unequal = widened.chunks(columns * 2).zip(&got).filter(|(row, out)| row.as_chunks::<2>().0.iter().zip(&x).fold(0.0f32, |sum, (w, x)| sum + bf16_to_f32(u16::from_le_bytes(*w)) * x).to_bits() != out.to_bits()).count();
                                    launches += 1; rows_compared += rows; different += unequal; broken_guards += broken;
                                    unwritten += got.iter().filter(|w| w.to_bits() == MARK).count();
                                    if (unequal > 0 || broken > 0) && first.is_none() { first = Some(format!("{codes:?} {rows} x {columns} {order:?}: {unequal} rows differ, {broken} guard words changed")); }
                                }
                            }
                        }
                    }
                }

                // Groups of made-up matrices: three matrices of one shape in one launch, one input for all and one
                // input a matrix, against one launch a matrix; guard words around the whole output.
                let (mut group_rows, mut picked_rows, mut picked_groups) = (0usize, 0usize, 0usize);
                for (codes, columns) in [(ExpertCodes::Nvfp4, 192usize), (ExpertCodes::Nvfp4, 1152), (ExpertCodes::Mxfp8, 256)] {
                    for rows in [2usize, 33, 64] {
                        for order in [None, Some(GATE_UP_ROWS)] {
                            if order.is_some() && rows % 2 != 0 { continue; }
                            let unit = UnitShape { codes, rows, columns, order };
                            let (mut matrices, mut unit_bytes) = (Vec::new(), Vec::new());
                            for m in 0..3usize {
                                let salt = 1000 * m + rows + columns;
                                let (stored, scales) = if codes == ExpertCodes::Nvfp4 {
                                    (made(rows * columns / 2, salt), made(rows * columns / 16, salt + 1).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>())
                                } else {
                                    (made(rows * columns, salt).into_iter().map(|b| if b & 0x7f == 0x7f { b ^ 1 } else { b }).collect::<Vec<_>>(), made(rows * columns / 32, salt + 1).into_iter().map(|b| b % 201).collect::<Vec<_>>())
                                };
                                let mut bytes = vec![0u8; unit.layout().map_err(|e| e.to_string())?.bytes];
                                // Each matrix its own factor, so that each its own weight table.
                                let own = loop_widen::expert_factor(0.5 + 0.25 * m as f32, 2688.0).map_err(|e| e.to_string())?;
                                unit.fill(&stored, &scales, (codes == ExpertCodes::Nvfp4).then_some(own), &mut bytes).map_err(|e| e.to_string())?;
                                let mut matrix = store.create(bytes.len() / 4)?;
                                store.write(&mut matrix, 0, &bytes)?;
                                matrices.push(matrix);
                                unit_bytes.push(bytes);
                            }
                            let refs = matrices.iter().collect::<Vec<_>>();
                            // The same three matrices as one stack, for the launch that reads its expert numbers on the device.
                            let each = unit_bytes[0].len() / 4;
                            let mut stack = store.create(3 * each)?;
                            for (m, bytes) in unit_bytes.drain(..).enumerate() { store.write(&mut stack, m * each, &bytes)?; }
                            for shared in [true, false] {
                                let xs = (0..if shared { 1 } else { 3 }).flat_map(|m| input(columns, 7 + m as u32)).collect::<Vec<_>>();
                                let x_all = backend.upload_vector(&xs)?.into_resident_shared()?;
                                let launched = group_launches(backend);
                                let (got, broken) = guarded_group(backend, unit, &refs, &x_all, shared)?;
                                if group_launches(backend) != launched + 1 { return Err("a group of made-up matrices did not record one grouped launch".into()); }
                                let mut one = Vec::new();
                                for (m, matrix) in refs.iter().enumerate() {
                                    let x = if shared { x_all.resident_range(0..columns)? } else { x_all.resident_range(m * columns..(m + 1) * columns)? };
                                    one.extend(guarded(backend, unit, matrix, &x)?.0);
                                }
                                let unequal = got.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + got.len().abs_diff(one.len());
                                launches += 1; rows_compared += got.len(); different += unequal; broken_guards += broken; group_rows += got.len();
                                unwritten += got.iter().filter(|w| w.to_bits() == MARK).count();
                                if (unequal > 0 || broken > 0) && first.is_none() { first = Some(format!("group {codes:?} {rows} x {columns} {order:?} shared {shared}: {unequal} rows differ from one launch a matrix, {broken} guard words changed")); }
                                // Kind 96 on the stack of the three matrices, expert numbers on the device (review REVIDS: one
                                // pick, 32 picks, a number picked twice), and on a stack of one expert (matrix 0 alone).
                                let sets: [(Vec<usize>, usize); 5] = [(vec![2, 0, 1], 3), (vec![1], 3), (vec![2, 2, 0], 3), ((0..32).map(|e| (e * 7 + 1) % 3).collect(), 3), (vec![0], 1)];
                                for (order, experts) in sets {
                                    let picks = order.len();
                                    let xs = (0..if shared { 1 } else { picks }).flat_map(|m| input(columns, 7 + m as u32)).collect::<Vec<_>>();
                                    let x_set = backend.upload_vector(&xs)?.into_resident_shared()?;
                                    let ids = backend.upload_vector(&order.iter().map(|&m| m as f32).collect::<Vec<_>>())?.into_resident_shared()?;
                                    let one_stack = if experts == 1 { Some(stack.resident_range(0..each)?) } else { None };
                                    let launched = picked_launches(backend);
                                    let (got, broken) = guarded_picked(backend, unit, one_stack.as_ref().unwrap_or(&stack), experts, &ids, &x_set, shared)?;
                                    if picked_launches(backend) != launched + 1 { return Err("a made-up stack did not record one picked launch".into()); }
                                    let mut one = Vec::new();
                                    for (e, &m) in order.iter().enumerate() {
                                        let x = if shared { x_set.resident_range(0..columns)? } else { x_set.resident_range(e * columns..(e + 1) * columns)? };
                                        one.extend(guarded(backend, unit, refs[m], &x)?.0);
                                    }
                                    let unequal = got.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + got.len().abs_diff(one.len());
                                    launches += 1; rows_compared += got.len(); different += unequal; broken_guards += broken; picked_rows += got.len(); picked_groups += 1;
                                    unwritten += got.iter().filter(|w| w.to_bits() == MARK).count();
                                    if (unequal > 0 || broken > 0) && first.is_none() { first = Some(format!("picked {codes:?} {rows} x {columns} {order:?} of {experts} shared {shared}: {unequal} rows differ from one launch a matrix, {broken} guard words changed")); }
                                }
                            }
                        }
                    }
                }
                eprintln!("PACK CHECK GROUP SWEEP groups=30 rows={group_rows} matrices_a_group=3 picked_groups={picked_groups} picked_rows={picked_rows} pick_sets=2,0,1|1|2,2,0|32_picks|one_expert_stack");
                // Bad grouped calls: an error and no launch each.
                {
                    let (rows, columns) = (34usize, 192usize);
                    let words = packed_product::layout(rows, columns, false).bytes / 4;
                    let unit_all = backend.upload_vector(&vec![0.0f32; words * 2 + 8])?.into_resident_shared()?;
                    let x_all = backend.upload_vector(&vec![0.5f32; columns * 2])?.into_resident_shared()?;
                    let out_all = backend.upload_vector(&vec![f32::from_bits(MARK); rows * 2 + 8])?.into_resident_shared()?;
                    let (a, b) = (unit_all.resident_range(0..words)?, unit_all.resident_range(words..2 * words)?);
                    let before = group_launches(backend);
                    let mut bad = vec![
                        ("no matrices", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, true, &[], &mut out_all.resident_range(0..rows)?)),
                        ("one input for two matrices that take an input each", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, false, &[&a, &b], &mut out_all.resident_range(0..2 * rows)?)),
                        ("an output one matrix short", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, true, &[&a, &b], &mut out_all.resident_range(0..rows)?)),
                        ("a unit one word short", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, true, &[&a, &unit_all.resident_range(words..2 * words - 1)?], &mut out_all.resident_range(0..2 * rows)?)),
                        ("the output over the second unit", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, true, &[&a, &b], &mut unit_all.resident_range(words + 4..words + 4 + 2 * rows)?)),
                    ];
                    let was = backend.pause_resident_launches(true)?;
                    bad.push(("launches are not recorded", backend.project_packed_group_into(rows, columns, false, false, &x_all.resident_range(0..columns)?, true, &[&a, &b], &mut out_all.resident_range(0..2 * rows)?)));
                    backend.pause_resident_launches(was)?;
                    for (name, result) in &bad { match result { Err(e) => eprintln!("PACK CHECK GROUP REFUSAL case=\"{name}\" error=\"{e}\""), Ok(()) => eprintln!("PACK CHECK GROUP REFUSAL case=\"{name}\" NOT REFUSED") } }
                    let (taken, recorded) = (bad.iter().filter(|(_, r)| r.is_ok()).count(), group_launches(backend) - before);
                    eprintln!("PACK CHECK GROUP REFUSALS cases={} refused={} launches_recorded={recorded}", bad.len(), bad.len() - taken);
                    if taken > 0 || recorded > 0 { return Err(format!("the grouped call took {taken} of {} bad calls and recorded {recorded} launches", bad.len())); }
                }

                // The refusals of the backend call. Each bad call must be an error and must record no launch.
                let (rows, columns) = (34usize, 192usize);
                let words = packed_product::layout(rows, columns, false).bytes / 4;
                let unit_all = backend.upload_vector(&vec![0.0f32; words + 8])?.into_resident_shared()?;
                let x_all = backend.upload_vector(&vec![0.5f32; columns + 8])?.into_resident_shared()?;
                let out_all = backend.upload_vector(&vec![f32::from_bits(MARK); rows + 8])?.into_resident_shared()?;
                let (unit_ok, x_ok) = (unit_all.resident_range(0..words)?, x_all.resident_range(0..columns)?);
                let recorded_before = packed_launches(backend);
                let mut cases: Vec<(&str, std::result::Result<(), String>)> = Vec::new();
                let mut not_made: Vec<String> = Vec::new();
                {
                    let mut call = |name: &'static str, rows: usize, columns: usize, wide: bool, pairs: bool, x: &DeviceVector<'_>, unit: &DeviceVector<'_>, out: &mut DeviceVector<'_>| cases.push((name, backend.project_packed_into(rows, columns, wide, pairs, x, unit, out)));
                    let out = || out_all.resident_range(0..rows);
                    call("a unit one word short", rows, columns, false, false, &x_ok, &unit_all.resident_range(0..words - 1)?, &mut out()?);
                    call("a unit one word long", rows, columns, false, false, &x_ok, &unit_all.resident_range(0..words + 1)?, &mut out()?);
                    call("an input one word short", rows, columns, false, false, &x_all.resident_range(0..columns - 1)?, &unit_ok, &mut out()?);
                    call("an input one word long", rows, columns, false, false, &x_all.resident_range(0..columns + 1)?, &unit_ok, &mut out()?);
                    call("an output one word short", rows, columns, false, false, &x_ok, &unit_ok, &mut out_all.resident_range(0..rows - 1)?);
                    call("an output one word long", rows, columns, false, false, &x_ok, &unit_ok, &mut out_all.resident_range(0..rows + 1)?);
                    call("no rows", 0, columns, false, false, &x_ok, &unit_ok, &mut out()?);
                    call("no columns", rows, 0, false, false, &x_ok, &unit_ok, &mut out()?);
                    call("columns that are not whole stages", rows, 96, false, false, &x_ok, &unit_ok, &mut out()?);
                    call("8-bit codes with columns that are not whole double stages", rows, columns, true, false, &x_ok, &unit_ok, &mut out()?);
                    call("more than 8192 columns", rows, 8256, false, false, &x_ok, &unit_ok, &mut out()?);
                    call("a matrix above the envelope of the tile", (1 << 25) / columns + 1, columns, false, false, &x_ok, &unit_ok, &mut out()?);
                    call("mixed rows with an odd row count", rows - 1, columns, false, true, &x_ok, &unit_ok, &mut out()?);
                    call("a unit that is not 16-byte aligned", rows, columns, false, false, &x_ok, &unit_all.resident_range(1..words + 1)?, &mut out()?);
                    call("an input that is not 16-byte aligned", rows, columns, false, false, &x_all.resident_range(1..columns + 1)?, &unit_ok, &mut out()?);
                    call("an output inside the input", rows, columns, false, false, &x_ok, &unit_ok, &mut x_all.resident_range(8..8 + rows)?);
                    call("an output inside the unit", rows, columns, false, false, &x_ok, &unit_ok, &mut unit_all.resident_range(16..16 + rows)?);
                    // A buffer of another kind: storage in segments. Not made in the run for the memory checker.
                    if !s.sweep_only {
                        match backend.segmented_vector(1 << 22, words) { Ok(v) => call("a unit in segments", rows, columns, false, false, &x_ok, &v, &mut out()?), Err(e) => not_made.push(format!("a unit in segments: {e}")) }
                        match backend.segmented_vector(1 << 22, columns) { Ok(v) => call("an input in segments", rows, columns, false, false, &v, &unit_ok, &mut out()?), Err(e) => not_made.push(format!("an input in segments: {e}")) }
                        match backend.segmented_vector(1 << 22, rows) { Ok(mut v) => call("an output in segments", rows, columns, false, false, &x_ok, &unit_ok, &mut v), Err(e) => not_made.push(format!("an output in segments: {e}")) }
                    }
                    let was = backend.pause_resident_launches(true)?;
                    call("launches are not recorded", rows, columns, false, false, &x_ok, &unit_ok, &mut out()?);
                    backend.pause_resident_launches(was)?;
                }
                let recorded = packed_launches(backend) - recorded_before;
                for (name, result) in &cases {
                    match result { Err(e) => eprintln!("PACK CHECK REFUSAL case=\"{name}\" error=\"{e}\""), Ok(()) => eprintln!("PACK CHECK REFUSAL case=\"{name}\" NOT REFUSED") }
                }
                for note in &not_made { eprintln!("PACK CHECK REFUSAL NOT RUN {note}"); }
                let accepted = cases.iter().filter(|(_, result)| result.is_ok()).count();
                if accepted > 0 || recorded > 0 { return Err(format!("the backend call took {accepted} of {} bad calls and recorded {recorded} launches", cases.len())); }
                // The control: the same operands in a good call are taken. A unit of zeros gives +0.0 in every row.
                let mut good = out_all.resident_range(0..rows)?;
                backend.project_packed_into(rows, columns, false, false, &x_ok, &unit_ok, &mut good)?;
                backend.flush_resident_launches()?;
                backend.synchronize()?;
                let control = backend.read_vector_values(&out_all)?;
                let control_wrong = control[..rows].iter().filter(|w| w.to_bits() != 0).count() + control[rows..].iter().filter(|w| w.to_bits() != MARK).count();
                eprintln!("PACK CHECK REFUSALS cases={} refused={} launches_recorded={recorded} not_run={} control_rows={rows} control_words_wrong={control_wrong}", cases.len(), cases.len() - accepted, not_made.len());
                if control_wrong > 0 || packed_launches(backend) != recorded_before + 1 { return Err(format!("the control of the refusals: {control_wrong} words are wrong after a good call with the same operands")); }
                drop(queue);
            }
            // The three quiet answers of the driver's call (`Device::packed_product` gives None and the caller reads the
            // widened matrix), and one call that is taken.
            {
                let (rows, columns) = (34usize, 192usize);
                let unit = UnitShape { codes: ExpertCodes::Nvfp4, rows, columns, order: None };
                let words = packed_product::layout(rows, columns, false).bytes / 4;
                let unit_all = backend.upload_vector(&vec![0.0f32; words + 8])?.into_resident_shared()?;
                let compute = Device::new(backend);
                let x = compute.upload(&vec![0.5f32; columns])?;
                let launched = packed_launches(backend);
                // 1. Launches are not recorded: no work scope is open.
                let idle = ResidentProduct::packed_product(&compute, &unit_all, 0..words, unit, &x)?.is_none();
                let (odd, unaligned, taken) = compute.work(|| {
                    // 2. A shape that the tile does not take. 3. A unit that is not 16-byte aligned. Then a good call.
                    let odd = ResidentProduct::packed_product(&compute, &unit_all, 0..words, UnitShape { columns: 96, ..unit }, &x)?.is_none();
                    let unaligned = ResidentProduct::packed_product(&compute, &unit_all, 1..words + 1, unit, &x)?.is_none();
                    let taken = match ResidentProduct::packed_product(&compute, &unit_all, 0..words, unit, &x)? { Some(out) => Some(compute.read(&out)?), None => None };
                    Ok((odd, unaligned, taken))
                })?;
                let taken = taken.is_some_and(|out| out.len() == rows && out.iter().all(|w| w.to_bits() == 0));
                let packed = packed_launches(backend) - launched;
                eprintln!("PACK CHECK FALLBACKS not_recorded_gives_none={idle} shape_not_taken_gives_none={odd} unaligned_unit_gives_none={unaligned} good_call_taken={taken} packed_launches={packed}");
                if !(idle && odd && unaligned && taken) || packed != 1 { return Err("the driver's packed call: a quiet answer was not None, or the good call was not taken one time".into()); }
            }
            eprintln!("PACK CHECK SWEEP launches={launches} rows={rows_compared} different={different} unwritten={unwritten} guard_words_changed={broken_guards} guard_words_a_side={GUARD} seconds={:.3}", began.elapsed().as_secs_f64());
            if let Some(first) = first { return Err(format!("the sweep of made-up matrices: {different} of {rows_compared} rows differ from the chain on the widened matrix, {broken_guards} guard words changed; first: {first}")); }
            if unwritten > 0 { return Err(format!("the sweep of made-up matrices: {unwritten} output rows were not written")); }
            Ok(launches)
        };
        // `--bad-id`: a stack of 3 units of a small 4-bit shape and ONE pick whose number is the given word (not a
        // whole number below 3). The device must trap before the tile: the launch ends in an error. Its output cannot be
        // read after the trap (the context is gone); the trap comes before the first write of the tile in the body.
        if let Some(value) = s.bad_id {
            startup::done();
            let unit = UnitShape { codes: ExpertCodes::Nvfp4, rows: 64, columns: 192, order: None };
            let each = unit.layout().map_err(|e| e.to_string())?.bytes / 4;
            let stack = backend.upload_vector(&vec![0.0f32; 3 * each])?.into_resident_shared()?;
            let x = backend.upload_vector(&input(192, 7))?.into_resident_shared()?;
            let ids = backend.upload_vector(&[value])?.into_resident_shared()?;
            let queue = backend.resident_queue()?;
            backend.enable_resident_launches()?;
            let launched = picked_launches(backend);
            let result = guarded_picked(backend, unit, &stack, 3, &ids, &x, true);
            let recorded = picked_launches(backend) - launched;
            drop(queue);
            return match result {
                Err(e) if recorded == 1 => { eprintln!("PACK CHECK BAD ID value={value} launches_recorded=1 stopped=true error=\"{e}\""); Ok(()) }
                Err(e) => Err(format!("bad id {value}: refused before the launch ({recorded} recorded), not by the device: {e}")),
                Ok((_, broken)) => Err(format!("bad id {value}: the launch ran to its end ({broken} guard words changed)")),
            };
        }
        if s.sweep_only {
            startup::done();
            let launches = sweep(&mut store)?;
            eprintln!("PACK CHECK SWEEP ONLY PASS launches={launches} different=0 guard_words_changed=0 no weights were loaded seconds={:.3}", started.elapsed().as_secs_f64());
            return Ok(());
        }


        // The load: every tensor of the plan, the expert stacks in both forms.
        let before = backend.allocated_bytes();
        startup::step("weights");
        let loaded = load::load(checkpoint, &s.model, &mut store, s.workers)?;
        backend.synchronize()?;
        let weights_bytes = backend.allocated_bytes() - before;
        let fingerprint = loaded.fingerprint();
        let equal = weights_bytes as u64 == checkpoint.load.bytes.device;
        eprintln!("PACK CHECK DEVICE weights_bytes={weights_bytes} plan_bytes={} equal={equal} tensors={} units={} allocate_seconds={:.3} fill_seconds={:.3} fingerprint={fingerprint}",
            checkpoint.load.bytes.device, loaded.report.tensors, loaded.report.units, loaded.report.allocate_seconds, loaded.report.fill_seconds);
        if !equal { return Err(format!("device bytes of the weights differ from the load plan: device {weights_bytes}, plan {}", checkpoint.load.bytes.device)); }
        startup::done();
        if s.verify {
            let verified = load::verify(&loaded, &mut store, s.workers)?;
            eprintln!("PACK CHECK VERIFY units={} bytes={} different={} seconds={:.3}", verified.units, verified.bytes, verified.different.len(), verified.seconds);
            if !verified.different.is_empty() { return Err(format!("{} of {} units differ on the device from the bytes of the files; first: {}", verified.different.len(), verified.units, verified.different[0])); }
        }

        // 1. The sweep of made-up matrices.
        let launches = sweep(&mut store)?;

        // 2. The real experts through the driver's own call.
        let began = Instant::now();
        let d = &checkpoint.description;
        let weights = LoadedWeights::new(checkpoint, &load::Files::open(checkpoint, &s.model)?, &loaded.operands)?;
        let compute = Device::new(backend);
        let mut names = Vec::new();
        for (i, layer) in d.layers.iter().enumerate() {
            if let Ffn::Experts(e) = layer.ffn { for expert in sample(e.count, s.experts) { names.extend([(i, expert, Weight::GateUp(i, expert as usize), Proj::GateUp), (i, expert, Weight::Down(i, expert as usize), Proj::Down)]); } }
        }
        let (mut products, mut rows_compared, mut device_different, mut host_different, mut first) = (0usize, 0usize, 0usize, 0usize, None);
        let (mut by_form, mut kept) = ([0usize; 2], Vec::new());
        for &(layer, expert, weight, proj) in &names {
            let (value, words, rows, columns) = weights.matrix(weight)?.ok_or_else(|| format!("{weight:?} is not on the device"))?;
            let packed = loaded.operands.packed_expert(layer as u32, proj, expert)?.ok_or_else(|| format!("{weight:?} has no packed stack"))?;
            let x = input(columns, 0);
            let launched = packed_launches(backend);
            // The driver's call: the seam picks the packed stack inside a work scope.
            let got = compute.work(|| compute.read(&weights.product(&compute, weight, &compute.upload(&x)?)?))?;
            if packed_launches(backend) != launched + 1 { return Err(format!("{weight:?} ({rows} x {columns}): the driver's product did not launch the packed kernel one time")); }
            // The BF16 product of the device on the widened stack of the same expert.
            let mut out = backend.upload_vector(&vec![f32::from_bits(MARK); rows])?;
            backend.project_bf16_chain_into(rows, columns, &backend.upload_vector(&x)?, &value.resident_range(words.clone())?, &mut out)?;
            let widened = backend.read_vector_values(&out)?;
            // The host chain on the widened matrix as the device holds it.
            let mut bytes = vec![0u8; words.len() * 4];
            store.read(value, words.start, &mut bytes)?;
            if got.len() != rows || widened.len() != rows || bytes.len() != rows * columns * 2 { return Err(format!("{weight:?}: {} packed rows, {} widened rows and {} matrix bytes for {rows} x {columns}", got.len(), widened.len(), bytes.len())); }
            let on_device = got.iter().zip(&widened).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let on_host = bytes.chunks(columns * 2).zip(&got).filter(|(row, out)| row.as_chunks::<2>().0.iter().zip(&x).fold(0.0f32, |sum, (w, x)| sum + bf16_to_f32(u16::from_le_bytes(*w)) * x).to_bits() != out.to_bits()).count();
            products += 1; rows_compared += rows; device_different += on_device; host_different += on_host;
            by_form[usize::from(packed.unit.codes == ExpertCodes::Mxfp8)] += 1;
            if (on_device > 0 || on_host > 0) && first.is_none() { first = Some(format!("{weight:?} {rows} x {columns}: {on_device} rows differ from the BF16 product of the device, {on_host} from the host chain")); }
            kept.push((weight, packed.unit, got));
        }
        eprintln!("PACK CHECK PRODUCTS layers={} experts_a_layer={} products={products} four_bit_products={} eight_bit_products={} rows={rows_compared} different_from_device_bf16={device_different} different_from_host_chain={host_different} seconds={:.3}",
            d.layers.len(), s.experts, by_form[0], by_form[1], began.elapsed().as_secs_f64());
        if let Some(first) = first { return Err(format!("{device_different} of {rows_compared} rows differ from the BF16 product of the device and {host_different} from the host chain; first: {first}")); }

        // 3. The same experts through the backend call, into a marked output with guard words.
        let began = Instant::now();
        let (mut guard_rows, mut guard_different, mut unwritten, mut broken_guards) = (0usize, 0usize, 0usize, 0usize);
        {
            let queue = backend.resident_queue()?;
            backend.enable_resident_launches()?;
            backend.resident_row_pass(1);
            for (&(layer, expert, _, proj), (weight, unit, got)) in names.iter().zip(&kept) {
                let packed = loaded.operands.packed_expert(layer as u32, proj, expert)?.ok_or_else(|| format!("{weight:?} has no packed stack"))?;
                let x = backend.upload_vector(&input(unit.columns, 0))?;
                let (again, broken) = guarded(backend, *unit, &packed.value.resident_range(packed.words)?, &x)?;
                guard_rows += unit.rows; broken_guards += broken;
                unwritten += again.iter().filter(|w| w.to_bits() == MARK).count();
                guard_different += again.iter().zip(got).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + again.len().abs_diff(got.len());
            }
            drop(queue);
        }
        eprintln!("PACK CHECK GUARDS launches={} rows={guard_rows} different_from_first_launch={guard_different} unwritten={unwritten} guard_words_changed={broken_guards} guard_words_a_side={GUARD} seconds={:.3}", kept.len(), began.elapsed().as_secs_f64());
        if guard_different > 0 || unwritten > 0 || broken_guards > 0 { return Err(format!("guarded launches: {guard_different} of {guard_rows} rows differ from the first launch, {unwritten} rows were not written, {broken_guards} guard words changed")); }

        // 4. The real experts as groups: 8 experts of each layer and projection in one launch through the driver's
        // call (`Weights::products` inside `Compute::work`), against one launch an expert; then the same group through
        // the backend into a marked output with guard words.
        let began = Instant::now();
        let (mut groups, mut group_rows, mut group_different, mut group_broken, mut group_unwritten, mut picked_different) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for (i, layer) in d.layers.iter().enumerate() {
            let Ffn::Experts(e) = layer.ffn else { continue; };
            let ids = sample(e.count, 8).into_iter().rev().collect::<Vec<_>>();
            for proj in [Proj::GateUp, Proj::Down] {
                let picks = ids.iter().map(|&x| if proj == Proj::GateUp { Weight::GateUp(i, x as usize) } else { Weight::Down(i, x as usize) }).collect::<Vec<_>>();
                let unit = loaded.operands.packed_expert(i as u32, proj, ids[0])?.ok_or("a layer has no packed stack")?.unit;
                let shared = proj == Proj::GateUp;
                let xs = (0..if shared { 1 } else { ids.len() }).flat_map(|m| input(unit.columns, 11 + m as u32)).collect::<Vec<_>>();
                let launched = group_launches(backend);
                let (all, one) = compute.work(|| {
                    let x = compute.upload(&xs)?;
                    let all = weights.products(&compute, &picks, &x, shared)?.ok_or("the driver's grouped call gave no group")?;
                    let all = compute.read(&all)?;
                    let mut one = Vec::new();
                    for (m, &w) in picks.iter().enumerate() {
                        let x = if shared { compute.upload(&xs)? } else { compute.upload(&xs[m * unit.columns..(m + 1) * unit.columns])? };
                        one.extend(compute.read(&weights.product(&compute, w, &x)?)?);
                    }
                    Ok((all, one))
                })?;
                if group_launches(backend) != launched + 1 { return Err(format!("layer {i} {proj:?}: the driver's grouped call did not record one grouped launch")); }
                group_different += all.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + all.len().abs_diff(one.len());
                // The same group through the backend, with guard words.
                let queue = backend.resident_queue()?;
                backend.enable_resident_launches()?;
                backend.resident_row_pass(1);
                let matrices = ids.iter().map(|&x| { let p = loaded.operands.packed_expert(i as u32, proj, x)?.ok_or("a packed stack lost an expert")?; p.value.resident_range(p.words).map_err(|e| e.to_string()) }).collect::<Result<Vec<_>>>()?;
                let x = backend.upload_vector(&xs)?.into_resident_shared()?;
                let (again, broken) = guarded_group(backend, unit, &matrices.iter().collect::<Vec<_>>(), &x, shared)?;
                group_different += again.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
                group_broken += broken; group_unwritten += again.iter().filter(|w| w.to_bits() == MARK).count();
                // Kind 96: the whole stack of the layer and projection, the same 8 experts by numbers on the device.
                let stack = loaded.operands.packed_expert(i as u32, proj, 0)?.ok_or("a layer has no packed stack")?.value;
                let numbers = backend.upload_vector(&ids.iter().map(|&x| x as f32).collect::<Vec<_>>())?.into_resident_shared()?;
                let launched = picked_launches(backend);
                let (picked, broken) = guarded_picked(backend, unit, stack, e.count as usize, &numbers, &x, shared)?;
                if picked_launches(backend) != launched + 1 { return Err(format!("layer {i} {proj:?}: the picked call did not record one launch")); }
                drop(queue);
                picked_different += picked.iter().zip(&one).filter(|(a, b)| a.to_bits() != b.to_bits()).count() + picked.len().abs_diff(one.len());
                group_broken += broken; group_unwritten += picked.iter().filter(|w| w.to_bits() == MARK).count();
                groups += 1; group_rows += all.len();
            }
        }
        eprintln!("PACK CHECK GROUPS groups={groups} experts_a_group=8 rows={group_rows} different_from_one_launch_an_expert={group_different} picked_on_the_device_different={picked_different} unwritten={group_unwritten} guard_words_changed={group_broken} seconds={:.3}", began.elapsed().as_secs_f64());
        if picked_different > 0 { return Err(format!("picked launches on the real experts: {picked_different} rows differ from one launch an expert")); }
        if group_different > 0 || group_broken > 0 || group_unwritten > 0 { return Err(format!("grouped launches on the real experts: {group_different} rows differ from one launch an expert, {group_broken} guard words changed, {group_unwritten} rows not written")); }

        // The times: 64 different experts, each launched one time, a wait after each 8, three passes.
        if s.times {
            let queue = backend.resident_queue()?;
            backend.enable_resident_launches()?;
            backend.resident_row_pass(1);
            let mut seen = [false; 2];
            for (i, layer) in d.layers.iter().enumerate() {
                let Ffn::Experts(e) = layer.ffn else { continue; };
                let Some(first) = loaded.operands.packed_expert(i as u32, Proj::GateUp, 0)? else { continue; };
                let form = usize::from(first.unit.codes == ExpertCodes::Mxfp8);
                if std::mem::replace(&mut seen[form], true) { continue; }
                let experts = (e.count as usize).min(64) / 8 * 8;
                if experts == 0 { continue; }
                for (proj, name) in [(Proj::GateUp, "gate-and-up"), (Proj::Down, "down")] {
                    let unit = loaded.operands.packed_expert(i as u32, proj, 0)?.ok_or("a layer has one packed stack and not the other")?.unit;
                    let x = backend.upload_vector(&input(unit.columns, 0))?;
                    let mut outs = (0..8).map(|_| backend.vector(unit.rows)).collect::<std::result::Result<Vec<_>, _>>()?;
                    let (wide, pairs) = (unit.codes == ExpertCodes::Mxfp8, unit.order.is_some());
                    for packed_form in [true, false, true, false] {
                        let mut groups = Vec::new();
                        for _pass in 0..3 {
                            for group in 0..experts / 8 {
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                let began = Instant::now();
                                for (slot, out) in outs.iter_mut().enumerate() {
                                    let expert = (group * 8 + slot) as u64;
                                    if packed_form {
                                        let p = loaded.operands.packed_expert(i as u32, proj, expert)?.ok_or("a packed stack lost an expert")?;
                                        backend.project_packed_into(unit.rows, unit.columns, wide, pairs, &x, &p.value.resident_range(p.words)?, out)?;
                                    } else {
                                        let m = loaded.operands.expert(i as u32, proj, expert)?;
                                        backend.project_bf16_chain_into(unit.rows, unit.columns, &x, &m.value.resident_range(m.words)?, out)?;
                                    }
                                }
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                groups.push(began.elapsed().as_secs_f64() * 1e6 / 8.0);
                            }
                        }
                        groups.sort_by(f64::total_cmp);
                        let bytes = if packed_form { packed_product::layout(unit.rows, unit.columns, wide).bytes - packed_product::table_bytes(wide) } else { unit.rows * unit.columns * 2 };
                        eprintln!("PACK CHECK TIME layer={i} codes={} proj=\"{name}\" shape={}x{} form={} experts={experts} groups={} microseconds_a_launch_middle={:.1} least={:.1} most={:.1} bytes_a_launch={bytes}",
                            if wide { "8-bit" } else { "4-bit" }, unit.rows, unit.columns, if packed_form { "packed" } else { "bf16" }, groups.len(), groups[groups.len() / 2], groups[0], groups[groups.len() - 1]);
                    }
                    // One tall launch against the same experts launched one by one. A tall unit holds the stored rows of
                    // `k` experts as one matrix, so its launch has the blocks and the bytes of a launch that would take
                    // the `k` experts of a visit together. Each group reads experts that the group before did not read.
                    let k = (1..=8usize).rev().find(|k| packed_product::shape_admitted(k * unit.rows, unit.columns, wide, pairs)).unwrap_or(1);
                    let (single, tall) = (packed_product::layout(unit.rows, unit.columns, wide), packed_product::layout(k * unit.rows, unit.columns, wide));
                    let (code_bytes, scale_bytes) = (unit.rows * packed_product::code_stride(unit.columns, wide), unit.rows * packed_product::scale_stride(unit.columns, wide));
                    backend.flush_resident_launches()?;
                    backend.synchronize()?;
                    let mut talls = Vec::new();
                    for set in 0..experts / 8 {
                        let mut bytes = vec![0u8; tall.bytes];
                        for slot in 0..k {
                            let p = loaded.operands.packed_expert(i as u32, proj, (set * 8 + slot) as u64)?.ok_or("a packed stack lost an expert")?;
                            let mut one = vec![0u8; single.bytes];
                            store.read(p.value, p.words.start, &mut one)?;
                            bytes[slot * code_bytes..(slot + 1) * code_bytes].copy_from_slice(&one[..code_bytes]);
                            bytes[tall.scales + slot * scale_bytes..tall.scales + (slot + 1) * scale_bytes].copy_from_slice(&one[single.scales..single.scales + scale_bytes]);
                            // One weight table a unit: the table of the first expert (a stack has one factor).
                            if slot == 0 { bytes[tall.table..].copy_from_slice(&one[single.table..]); }
                        }
                        let mut matrix = store.create(bytes.len() / 4)?;
                        store.write(&mut matrix, 0, &bytes)?;
                        talls.push(matrix);
                    }
                    let mut out_tall = backend.vector(k * unit.rows)?;
                    for tall_form in [false, true, false, true] {
                        let mut groups = Vec::new();
                        for _pass in 0..3 {
                            for (set, matrix) in talls.iter().enumerate() {
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                let began = Instant::now();
                                if tall_form {
                                    backend.project_packed_into(k * unit.rows, unit.columns, wide, pairs, &x, matrix, &mut out_tall)?;
                                } else {
                                    for (slot, out) in outs.iter_mut().take(k).enumerate() {
                                        let p = loaded.operands.packed_expert(i as u32, proj, (set * 8 + slot) as u64)?.ok_or("a packed stack lost an expert")?;
                                        backend.project_packed_into(unit.rows, unit.columns, wide, pairs, &x, &p.value.resident_range(p.words)?, out)?;
                                    }
                                }
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                groups.push(began.elapsed().as_secs_f64() * 1e6);
                            }
                        }
                        groups.sort_by(f64::total_cmp);
                        eprintln!("PACK CHECK TALL layer={i} codes={} proj=\"{name}\" shape={}x{} experts_a_group={k} form={} groups={} microseconds_a_group_middle={:.1} least={:.1} most={:.1} bytes_a_group={}",
                            if wide { "8-bit" } else { "4-bit" }, unit.rows, unit.columns, if tall_form { "one-launch" } else { "launch-an-expert" }, groups.len(), groups[groups.len() / 2], groups[0], groups[groups.len() - 1],
                            k * (single.bytes - packed_product::table_bytes(wide)));
                    }
                    // The rows of the last tall launch against the rows of the last launches one by one (the last group).
                    // Only the first expert of the group: the tall unit has its weight table, and in the 4-bit form each
                    // expert's table holds its own scale, so the rows of the other experts are not its products.
                    let (all, half) = (backend.read_vector_values(&out_tall)?, unit.rows / 2);
                    let mut tall_different = 0usize;
                    for (slot, out) in outs.iter().take(1).enumerate() {
                        let one = backend.read_vector_values(out)?;
                        for (row, word) in one.iter().enumerate() {
                            // Gate rows come first in an output of mixed rows: the gate rows of all experts, then the up rows.
                            let at = if !pairs { slot * unit.rows + row } else if row < half { slot * half + row } else { k * half + slot * half + (row - half) };
                            tall_different += usize::from(all[at].to_bits() != word.to_bits());
                        }
                    }
                    eprintln!("PACK CHECK TALL ROWS layer={i} proj=\"{name}\" experts_a_group={k} rows_of_the_first_expert={} different_from_launch_an_expert={tall_different}", unit.rows);
                    if tall_different > 0 { return Err(format!("layer {i} {name}: {tall_different} rows of the tall launch differ from the launches one by one")); }
                    // The grouped launch of 8 experts against the same 8 launched one by one, as a visit takes them.
                    let shared = proj == Proj::GateUp;
                    let xg = backend.upload_vector(&(0..if shared { 1 } else { 8 }).flat_map(|m| input(unit.columns, 21 + m as u32)).collect::<Vec<_>>())?.into_resident_shared()?;
                    let mut out_group = backend.vector(8 * unit.rows)?;
                    for group_form in [false, true, false, true] {
                        let mut times = Vec::new();
                        for _pass in 0..3 {
                            for set in 0..experts / 8 {
                                let ms = (0..8).map(|slot| { let p = loaded.operands.packed_expert(i as u32, proj, (set * 8 + slot) as u64)?.ok_or("a packed stack lost an expert")?; p.value.resident_range(p.words).map_err(|e| e.to_string()) }).collect::<Result<Vec<_>>>()?;
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                let began = Instant::now();
                                if group_form {
                                    backend.project_packed_group_into(unit.rows, unit.columns, wide, pairs, &xg, shared, &ms.iter().collect::<Vec<_>>(), &mut out_group)?;
                                } else {
                                    for (slot, (m, out)) in ms.iter().zip(outs.iter_mut()).enumerate() {
                                        let x = if shared { xg.resident_range(0..unit.columns)? } else { xg.resident_range(slot * unit.columns..(slot + 1) * unit.columns)? };
                                        backend.project_packed_into(unit.rows, unit.columns, wide, pairs, &x, m, out)?;
                                    }
                                }
                                backend.flush_resident_launches()?;
                                backend.synchronize()?;
                                times.push(began.elapsed().as_secs_f64() * 1e6);
                            }
                        }
                        times.sort_by(f64::total_cmp);
                        eprintln!("PACK CHECK GROUP TIME layer={i} codes={} proj=\"{name}\" shape={}x{} experts_a_group=8 form={} groups={} microseconds_a_group_middle={:.1} least={:.1} most={:.1}",
                            if wide { "8-bit" } else { "4-bit" }, unit.rows, unit.columns, if group_form { "one-grouped-launch" } else { "launch-an-expert" }, times.len(), times[times.len() / 2], times[0], times[times.len() - 1]);
                    }
                }
            }
            drop(queue);
        }
        eprintln!("PACK CHECK PASS products={products} rows={rows_compared} different=0 sweep_launches={launches} guard_words_changed=0 device_weight_bytes={weights_bytes} fingerprint={fingerprint} seconds={:.3}", started.elapsed().as_secs_f64());
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
    fn settings_take_the_model_and_named_options() {
        let s = settings(&args("model --experts 5 --workers 3 --verify --times --plan-only")).unwrap();
        assert_eq!((s.model, s.experts, s.workers, s.verify, s.times, s.plan_only), (PathBuf::from("model"), 5, 3, true, true, true));
        let s = settings(&args("model")).unwrap();
        assert!(s.experts == 3 && !s.verify && !s.times && !s.plan_only && !s.sweep_only && (1..=16).contains(&s.workers));
        for bad in ["", "model --experts", "model --experts 0", "model --workers many", "model --fast"] { assert!(settings(&args(bad)).is_err(), "{bad}"); }
    }
    #[test]
    fn a_sample_holds_the_first_the_second_and_the_last_expert() {
        assert_eq!(sample(128, 3), [0, 1, 127]);
        assert_eq!(sample(7, 3), [0, 1, 6]);
        assert_eq!(sample(2, 3), [0, 1]);
        assert_eq!(sample(1, 3), [0]);
        let more = sample(128, 9);
        assert!(more.len() == 9 && more.starts_with(&[0, 1]) && more.ends_with(&[127]) && more.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(sample(4, 9), [0, 1, 2, 3]);
    }
}
