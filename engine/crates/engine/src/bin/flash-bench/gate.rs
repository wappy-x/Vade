//! The determinism matrix uses the engine's real prefill, batch and MTP paths.
use super::Result;
#[path = "../../checks/native_gate.rs"]
pub mod native_gate;
use native_gate::GatePlan;
#[cfg(target_os = "linux")]
use super::{Fixture, SharedRing};
#[cfg(target_os = "linux")]
use spark_engine::flash::{head::VOCABULARY, session::{DecodeBatch, Model, Output, Session}};
#[cfg(target_os = "linux")]
use spark_gpu_probe::driver::{Backend, HostRegistration};
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::{path::{Path, PathBuf}, rc::Rc, time::Instant};

/// Verification depth of the row-launch phase: its widest launch holds the
/// 16-row decode width.
#[cfg(target_os = "linux")]
const ROWS_DEPTH: usize = DecodeBatch::MAX_ROWS - 1;

const INJECTION_USAGE: &str = "VADE_TEST_ROUTING_REJECT=1:3:16: full three-fixture world-2 or world-4 gate, including the 16-row poison cell (vade test determinism --require-poison).\nVADE_TEST_ROUTING_REJECT=N:3:8[:MASK]: standalone one-fixture eight-slot matrix; N=1/2/3 selects eager/capture/replay, MASK=3 (default) compares seven survivors to fresh all-good-8, MASK=1/2 must fail with ROUTING_RANK_MISMATCH on both ranks.\nThe matrix requires spark-engine/flash-routing-per-row, MTP 0, world 2 and profiling off. VADE_TEST_ROUTING_GREEDY=1 selects token bytes for the matrix; default is full FP32 logits.";

enum Injection { Full, Matrix { verdict: usize, mask: u8 } }

fn injection() -> Result<Option<Injection>> {
    let Some(spec) = std::env::var("VADE_TEST_ROUTING_REJECT").ok().filter(|v| v != "0") else { return Ok(None); };
    let hook = if spark_gpu_probe::TEST_ROUTING_REJECT_WIDTH { "1:3:16" } else { "1:3" };
    if spec == hook { return Ok(Some(Injection::Full)); }
    let fields = spec.split(':').collect::<Vec<_>>();
    if (3..=4).contains(&fields.len()) && fields[1] == "3" && fields[2] == "8" {
        let verdict = fields[0].parse::<usize>().ok().filter(|n| (1..=3).contains(n));
        let mask = fields.get(3).copied().unwrap_or("3");
        if let Some(verdict) = verdict.filter(|_| matches!(mask, "1" | "2" | "3")) {
            if !cfg!(feature = "flash-routing-per-row") {
                return Err("Flash eight-slot matrix requires spark-engine/flash-routing-per-row".into());
            }
            return Ok(Some(Injection::Matrix { verdict, mask: mask.parse().unwrap() }));
        }
    }
    Err(format!("Flash routing injection usage:\n{INJECTION_USAGE}"))
}

/// Resolve the native gate hook without CUDA, weights, fixtures or transport.
pub fn dry_run() -> Result<()> {
    let args = std::env::args().skip(2).collect::<Vec<_>>();
    if args.first().is_some_and(|s| s == "--model") {
        if args.len() != 6 { return Err("usage: --gate-dry-run --model CHECKPOINT WORLD RANK FIXTURES GROUPS".into()); }
        let fixtures = args[4].split(',').map(super::Fixture::read).collect::<Result<Vec<_>>>()?;
        let groups = args[5].split(',').map(|s| s.parse::<usize>().map_err(|e| e.to_string())).collect::<Result<Vec<_>>>()?;
        let groups = groups.try_into().map_err(|_| "gate needs exactly two groups")?;
        let (_, receipt) = planned(std::path::Path::new(&args[1]), args[2].parse::<usize>().map_err(|e| e.to_string())?,
            args[3].parse::<usize>().map_err(|e| e.to_string())?, &fixtures, groups)?;
        print!("{receipt}");
        return Ok(());
    }
    println!("usage: flash-bench --gate-dry-run [--help]\nNative invocation: flash-bench PTX MODEL FIXTURES ADDR0 ADDR1 RANK --mtp 0 --gate DIR --prefill-group 256 --gate-group 512\n{INJECTION_USAGE}");
    match std::env::args().nth(2).as_deref() {
        Some("--help") => return Ok(()), None => {}, _ => return Err("usage: flash-bench --gate-dry-run [--help]".into()),
    }
    if std::env::args().nth(3).is_some() { return Err("usage: flash-bench --gate-dry-run [--help]".into()); }
    match injection()? {
        Some(Injection::Matrix { verdict, mask }) => println!("DRY-RUN accepted matrix N={verdict} rows=8 rejected_row=3 mask={mask} result={}",
            if mask == 3 { "survivor-bytes-and-collectives" } else { "ROUTING_RANK_MISMATCH" }),
        _ => println!("DRY-RUN accepted full-gate rows=16 poison={}", if cfg!(feature = "flash-routing-per-row") { "required" } else { "SKIP" }),
    }
    Ok(())
}

fn planned(root: &std::path::Path, world: usize, rank: usize, fixtures: &[super::Fixture], groups: [usize; 2]) -> Result<(GatePlan, String)> {
    let plan = GatePlan::inspect(root, world, rank)?;
    let mut lengths = fixtures.iter().map(|f| f.prompt.len()).collect::<Vec<_>>(); lengths.sort_unstable();
    if lengths != [285, 2048, 8192] || groups[0] == groups[1]
        || groups.iter().any(|&g| g == 0 || g > 512 || g % 64 != 0) { return Err("invalid gate fixtures/groups".into()); }
    let mut tokens = Vec::new();
    for fixture in fixtures {
        let rows = fixture.rows.iter().map(|r| r.0).collect::<Vec<_>>();
        plan.fixture(&fixture.prompt, &rows)?;
        tokens.extend((fixture.prompt.len() as u64).to_le_bytes()); tokens.extend((rows.len() as u64).to_le_bytes());
        tokens.extend(fixture.prompt.iter().chain(&rows).flat_map(|t| t.to_le_bytes()));
    }
    let receipt = plan.receipt(&tokens, groups);
    Ok((plan, receipt))
}

#[cfg(target_os = "linux")]
pub fn validate(fixtures: &[Fixture], mtp: usize, groups: [usize; 2], world: usize) -> Result<()> {
    let matrix = matches!(injection()?, Some(Injection::Matrix { .. }));
    if matrix && (mtp != 0 || world != 2) { return Err("Flash eight-slot matrix needs MTP 0 and world 2".into()); }
    let vocabulary = if matrix { VOCABULARY } else {
        let root = std::env::args().nth(2).ok_or("missing gate model")?;
        planned(std::path::Path::new(&root), world, 0, fixtures, groups)?.0.row_bytes / 4
    };
    let mut lengths = fixtures.iter().map(|f| f.prompt.len()).collect::<Vec<_>>();
    lengths.sort_unstable();
    let valid_lengths = if matrix { lengths.len() == 1 && [285, 2048, 8192].contains(&lengths[0]) }
        else { lengths == [285, 2048, 8192] };
    if !valid_lengths || fixtures.iter().any(|f| f.rows.len() < 64
        || f.prompt.iter().chain(f.rows.iter().map(|r| &r.0)).any(|&t| t as usize >= vocabulary)) {
        return Err("Flash gate needs 285/2048/8192-token fixtures with at least 64 continuation rows".into());
    }
    if !matches!(mtp, 0 | 3 | ROWS_DEPTH) || groups[0] == groups[1]
        || groups.iter().any(|&g| g == 0 || g > 512 || g % 64 != 0) {
        return Err("Flash gate needs MTP 0, 3 or 15 and two distinct 64-token-multiple groups within 512".into());
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn file(root: &Path, prompt: usize, row: usize, position: usize, rank: usize) -> PathBuf {
    root.join(format!("prompt-{prompt:05}/session-{row:05}/position-{position:05}-rank-{rank}.f32"))
}
#[cfg(target_os = "linux")]
fn bytes(output: &Output, width: usize) -> Result<Vec<u8>> {
    if output.logits.is_none() && std::env::var("VADE_TEST_ROUTING_GREEDY").as_deref() == Ok("1") {
        return Ok(output.token.to_le_bytes().to_vec());
    }
    native_gate::row(output.logits.as_deref(), width)
}
#[derive(Default)]
struct Compared { rows: usize, bytes: usize, different: usize, first: Option<String>, reference: bool,
    #[cfg(target_os = "linux")] width: usize,
    #[cfg(target_os = "linux")] same_row: bool }
impl Compared {
    #[cfg(target_os = "linux")]
    fn row(&mut self, root: &Path, reference: &Path, fixture: &Fixture, row: usize,
        position: usize, rank: usize, output: &Output, baseline: bool) -> Result<()> {
        let actual = bytes(output, if self.width == 0 { VOCABULARY * 4 } else { self.width })?;
        let path = file(root, fixture.prompt.len(), row, position, rank);
        fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
        // New runs never accept old or partially overwritten captures.
        use std::io::Write;
        fs::OpenOptions::new().write(true).create_new(true).open(path)
            .and_then(|mut f| f.write_all(&actual)).map_err(|e| e.to_string())?;
        if baseline { self.rows += 1; self.reference = true; return Ok(()); }
        let expected = fs::read(file(reference, fixture.prompt.len(), if self.same_row { row } else { 0 }, position, rank)).map_err(|e| e.to_string())?;
        self.compare(&expected, &actual, fixture.prompt.len(), row, position)
    }
    fn compare(&mut self, expected: &[u8], actual: &[u8], prompt: usize, row: usize, position: usize) -> Result<()> {
        if expected.len() != actual.len() { return Err("Flash gate reference row size differs".into()); }
        native_gate::finite_row(expected, actual.len())?;
        native_gate::finite_row(actual, actual.len())?;
        let different = expected.iter().zip(actual).filter(|(a, b)| a != b).count();
        if different != 0 && self.first.is_none() {
            let offset = expected.iter().zip(actual).position(|(a, b)| a != b).unwrap();
            let word = offset / 4 * 4;
            self.first = Some(format!("prompt={} row={row} position={position} vocabulary={} byte={} reference_bits={:08x} actual_bits={:08x}",
                prompt, offset / 4, offset % 4,
                u32::from_le_bytes(expected[word..word+4].try_into().unwrap()),
                u32::from_le_bytes(actual[word..word+4].try_into().unwrap())));
        }
        self.rows += 1; self.bytes += actual.len(); self.different += different;
        Ok(())
    }
    fn print(&self, name: &str) {
        println!("{name} {} {} {} {}", self.rows, self.bytes, self.different,
            if self.reference { "REF" } else if self.different == 0 { "PASS" } else { "FAIL" });
        if let Some(first) = &self.first { eprintln!("first_difference case={name} {first}"); }
    }
}

/// Offline entry to the same byte comparator used by every native gate row.
pub fn compare_dump() -> Result<()> {
    let args = std::env::args().skip(2).collect::<Vec<_>>();
    if args.len() != 5 && !(args.len() == 7 && args[5] == "--model")
        && !(args.len() == 11 && args[5] == "--model" && args[7] == "--world" && args[9] == "--rank") {
        return Err("usage: flash-bench --gate-compare REFERENCE.f32 ACTUAL.f32 PROMPT ROW POSITION [--model CONFIG | --model CHECKPOINT --world N --rank R]".into());
    }
    // Preserve the legacy Flash selftest; other checkpoints supply their config.
    let width = if args.len() == 11 {
        let plan = GatePlan::inspect(std::path::Path::new(&args[6]), args[8].parse::<usize>().map_err(|e| e.to_string())?,
            args[10].parse::<usize>().map_err(|e| e.to_string())?)?;
        plan.check_dumps(std::path::Path::new(&args[0]), std::path::Path::new(&args[1]))?;
        plan.row_bytes
    } else if args.len() == 7 {
        let path = std::path::Path::new(&args[6]);
        let config = if path.is_dir() { path.join("config.json") } else { path.to_path_buf() };
        vocabulary_bytes(&engine_model::safetensors::read_bounded(&config, 1 << 20).map_err(|e| e.to_string())?)?
    } else { 248320 * 4 };
    let expected = engine_model::safetensors::read_bounded(std::path::Path::new(&args[0]), width).map_err(|e| e.to_string())?;
    let actual = engine_model::safetensors::read_bounded(std::path::Path::new(&args[1]), width).map_err(|e| e.to_string())?;
    if actual.len() != width || expected.len() != actual.len() {
        return Err("Flash dump comparison needs complete vocabulary rows".into());
    }
    let indices = args[2..5].iter().map(|s| s.parse::<usize>().map_err(|e| e.to_string())).collect::<Result<Vec<_>>>()?;
    let mut compared = Compared::default();
    compared.compare(&expected, &actual, indices[0], indices[1], indices[2])?;
    compared.print("logits-dump-compare");
    if compared.different != 0 { Err("Flash logit determinism differs".into()) } else { Ok(()) }
}

fn vocabulary_bytes(config: &[u8]) -> Result<usize> {
    use engine_model::json::{self, field};
    let value = json::parse(config, json::Limits { bytes: 1 << 20, ..Default::default() }).map_err(|e| e.to_string())?;
    let root = value.object().map_err(|e| e.to_string())?;
    let text = root.get("text_config").unwrap_or(&value).object().map_err(|e| e.to_string())?;
    let vocabulary = field(text, "vocab_size").and_then(|v| v.u64()).map_err(|e| e.to_string())?;
    usize::try_from(vocabulary).ok().filter(|&v| v != 0).and_then(|v| v.checked_mul(4))
        .filter(|&bytes| bytes <= isize::MAX as usize).ok_or_else(|| "invalid vocabulary row size".into())
}

#[cfg(test)] #[test] fn configured_vocabulary_rows() {
    for (config, bytes) in [(r#"{"vocab_size":151936}"#, 607744), (r#"{"text_config":{"vocab_size":248320}}"#, 993280)] { assert_eq!(vocabulary_bytes(config.as_bytes()).unwrap(), bytes); }
    for config in [r#"{}"#, r#"{"vocab_size":0}"#, r#"{"vocab_size":1.5}"#, r#"{"vocab_size":18446744073709551615}"#] { assert!(vocabulary_bytes(config.as_bytes()).is_err()); }
    assert!(native_gate::row(None, 8).is_err());
    for row in [vec![], vec![0.0], vec![0.0, f32::NAN], vec![f32::INFINITY, 0.0]] { assert!(native_gate::row(Some(&row), 8).is_err()); }
    let a = native_gate::row(Some(&[0.0, 1.0]), 8).unwrap(); let b = native_gate::row(Some(&[-0.0, 1.0]), 8).unwrap();
    let mut c = Compared::default(); c.compare(&a, &b, 285, 7, 63).unwrap(); assert_eq!((c.rows,c.bytes,c.different),(1,8,1));
    for bad in [vec![], vec![0;7], f32::NAN.to_le_bytes().repeat(2)] { assert!(c.compare(&bad,&bad,285,0,0).is_err()); }
}

pub fn capabilities() {
    println!("flash_gate routing_reject_per_row={} detector=cargo-feature:flash-routing-per-row routing_reject_width={}",
        usize::from(cfg!(feature = "flash-routing-per-row")), usize::from(spark_gpu_probe::TEST_ROUTING_REJECT_WIDTH));
}

#[cfg(target_os = "linux")]
fn forward_batch<'a, 'w>(b: &Backend<'a>, batch: &mut DecodeBatch<'a, 'w>,
    sessions: &mut [&mut Session<'a, 'w, SharedRing>], tokens: &[u32], active: u16) -> Result<Vec<Result<Output>>> {
    #[cfg(feature = "flash-routing-per-row")]
    { Session::forward_batch_survivors(b, batch, sessions, tokens, &vec![true; sessions.len()], active) }
    #[cfg(not(feature = "flash-routing-per-row"))]
    {
        if active != u16::MAX >> (16-sessions.len()) { return Err("Flash per-row routing rejection unavailable".into()); }
        Session::forward_batch(b, batch, sessions, tokens, &vec![true; sessions.len()])
    }
}

#[cfg(target_os = "linux")]
fn routing_result(result: &Result<Vec<Result<Output>>>, rank: usize, name: &str, position: usize) {
    match result {
        Err(error) => eprintln!("routing_result rank={rank} case={name} position={position} error={error}"),
        Ok(rows) => {
            eprintln!("routing_result rank={rank} case={name} position={position} rows={} result=OK", rows.len());
            for (row, result) in rows.iter().enumerate() {
                if let Err(error) = result {
                    eprintln!("routing_result rank={rank} case={name} position={position} row={row} error={error}");
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn routing_collectives(batch: &DecodeBatch<'_, '_>, rank: usize, name: &str, position: usize,
    rows: usize, active: u16, expected: Option<u64>) -> Result<u64> {
    let count = batch.collective_count().ok_or("missing collective counter")?;
    println!("routing_collectives rank={rank} case={name} position={position} rows={rows} active={active:04x} count={count}");
    if expected.is_some_and(|n| n != count) {
        return Err(format!("Flash routing collective count differs rank={rank} case={name} position={position} expected={} actual={count}", expected.unwrap()));
    }
    Ok(count)
}

// Restored from 76dc3603/f86b09c8/c991c592/1cd5f242. The control owns
// fresh sessions and graph history; rejected slot 3 remains physical padding.
#[cfg(all(target_os = "linux", feature = "flash-routing-per-row"))]
fn injection_matrix<'a>(b: &Backend<'a>, model: &Model<'a>, ring: SharedRing, fixture: &Fixture,
    rank: usize, verdict: usize, mask: u8, dir: &Path) -> Result<()> {
    if b.is_profiling() { return Err("safety gate needs eager/capture/replay verdict 1/2/3 and profiling off".into()); }
    let started = Instant::now();
    let weights = b.allocated_bytes();
    let mailbox = ring.lock()?.mailbox_memory().map(|(base, n)|
        unsafe { b.register_host(base, n) }.map(Rc::new)).transpose()?;
    let capacity = (fixture.prompt.len() + verdict + 64).div_ceil(512) * 512;
    let reference = dir.join("injected-8");
    let greedy = std::env::var("VADE_TEST_ROUTING_GREEDY").as_deref() == Ok("1");
    let mut collectives = vec![0; verdict + 63];
    let mut failed = false;
    println!("case rows bytes_compared bytes_different result");
    for name in ["injected-8", "all-good-8"] {
        let baseline = name == "injected-8";
        b.defer_test_routing_rejection(true);
        let mut batch = DecodeBatch::new(b, model)?;
        let mut sessions = (0..8).map(|_| model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone()))
            .collect::<Result<Vec<_>>>()?;
        sessions[0].enable_cache();
        let output = sessions[0].forward_prefill(b, &fixture.prompt, true)?;
        let mut compared = Compared { same_row: true, ..Default::default() };
        let root = dir.join(name);
        compared.row(&root, &reference, fixture, 0, 0, rank, &output, baseline)?;
        let seed = sessions[0].snapshot(b)?;
        for session in &mut sessions[1..] { session.restore(b, &seed, false)?; }
        drop(seed);
        let mut active = 0xff;
        for position in 1..verdict + 64 {
            let tokens = (0..8).map(|row| fixture.rows[(position - 1 + 5 * row) % 64].0).collect::<Vec<_>>();
            let entry = sessions.iter().map(Session::position).collect::<Vec<_>>();
            b.defer_test_routing_rejection(!baseline);
            let result = Session::forward_batch_survivors(b, &mut batch, &mut sessions.iter_mut().collect::<Vec<_>>(),
                &tokens, &[!greedy; 8], active);
            b.defer_test_routing_rejection(true);
            routing_result(&result, rank, name, position);
            let count = routing_collectives(&batch, rank, name, position, 8, active,
                (!baseline).then_some(collectives[position - 1]));
            if mask != 3 && position == verdict {
                if let Err(error) = result {
                    if error.starts_with("ROUTING_RANK_MISMATCH") && sessions.iter().zip(&entry)
                        .all(|(s, &p)| s.position() == p && !s.is_valid()) {
                        eprintln!("routing_mismatch rank={rank} case={name} position={position} mask={mask} result=ROUTING_RANK_MISMATCH PASS");
                        return Err(error);
                    }
                    return Err(format!("safety gate expected ROUTING_RANK_MISMATCH without row commits: {error}"));
                }
                return Err("safety gate expected ROUTING_RANK_MISMATCH on both ranks".into());
            }
            let outputs = result?;
            let count = count?;
            if baseline { collectives[position - 1] = count; }
            if outputs.len() != 8 { return Err("safety gate returned an incomplete eight-slot cohort".into()); }
            for (row, output) in outputs.iter().enumerate() {
                if sessions[row].position() != entry[row] + usize::from(output.is_ok())
                    || sessions[row].is_valid() != output.is_ok() { return Err("safety gate row ledger mismatch".into()); }
                if position >= verdict && row == 3 {
                    if baseline && !output.as_ref().is_err_and(|e| e.starts_with("Flash MoE routing rejected"))
                        || !baseline && output.is_err() { return Err("safety gate expected routing rejection for row 3".into()); }
                    continue;
                }
                compared.row(&root, &reference, fixture, row, position, rank, output.as_ref().map_err(Clone::clone)?, baseline)?;
            }
            if baseline && position == verdict { active &= !(1 << 3); }
        }
        if !baseline {
            sessions.clear();
            let mut fresh = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
            let output = fresh.forward_prefill(b, &fixture.prompt, true)?;
            compared.row(&dir.join("new-request"), &reference, fixture, 0, 0, rank, &output, false)?;
        }
        compared.print(&format!("logits-rank-{rank}-{name}"));
        failed |= compared.different != 0;
    }
    eprintln!("flash_gate complete rank={rank} matrix=injected-8 verdict={verdict} mask={mask} elapsed_ms={:.3} weights_bytes={weights} device_peak_bytes={}",
        started.elapsed().as_secs_f64()*1000.0, b.peak_allocated_bytes());
    b.defer_test_routing_rejection(false);
    if failed { Err("Flash logit determinism differs".into()) } else { Ok(()) }
}

#[cfg(target_os = "linux")]
#[derive(Default)]
struct MixedCalls { prefill: usize, setup_decode: usize, batch_decode: usize }

#[cfg(target_os = "linux")]
fn mixed_sessions<'a, 'w>(b: &Backend<'a>, model: &'w Model<'a>, ring: &SharedRing,
    mailbox: &Option<Rc<spark_gpu_probe::driver::HostRegistration<'a>>>, fixtures: &[&Fixture], started: Instant,
    calls: &mut MixedCalls)
    -> Result<Vec<Session<'a, 'w, SharedRing>>> {
    let mut sessions = (0..16).map(|row| {
        let capacity = fixtures[row % 3].capacity().max(512).div_ceil(512) * 512;
        model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())
    }).collect::<Result<Vec<_>>>()?;
    // Only three prefills. Copies are immutable seeds, never comparisons;
    // solo decode advances lane r by r frozen inputs before batching.
    for (index, fixture) in fixtures.iter().enumerate() {
        mixed_time(started)?;
        sessions[index].enable_cache();
        sessions[index].forward_prefill(b, &fixture.prompt, false)?;
        calls.prefill += 1;
        let seed = sessions[index].snapshot(b)?;
        for row in (index..16).step_by(3) {
            if row != index { sessions[row].restore(b, &seed, false)?; }
            for position in 0..row {
                mixed_time(started)?;
                sessions[row].forward(b, &[fixture.rows[position].0], false)?;
                calls.setup_decode += 1;
            }
        }
    }
    Ok(sessions)
}

#[cfg(target_os = "linux")]
fn mixed_time(started: Instant) -> Result<()> {
    if started.elapsed().as_secs() >= 120 { Err("Flash mixed-16 cases exceeded 120-second work cap".into()) }
    else { Ok(()) }
}

#[cfg(target_os = "linux")]
fn mtp_history_roundtrip<'a>(b: &Backend<'a>, model: &Model<'a>, ring: &SharedRing,
    fixture: &Fixture, mailbox: Option<Rc<HostRegistration<'a>>>, row_width: usize) -> Result<()> {
    let capacity = fixture.capacity().max(512).div_ceil(512) * 512;
    let inputs = fixture.rows[..9].iter().map(|r| r.0).collect::<Vec<_>>();
    let mut session = model.session_plain_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
    if !session.enable_mtp_history(b)? { return Err("MTP resize regression could not enable history".into()); }
    session.forward_prefill(b, &fixture.prompt, false)?;
    if !session.prepare_mtp_workspace(b)? { return Err("MTP resize regression needs verification workspace".into()); }
    let mut actual = Vec::new();
    let before = b.stats();
    let mut captures = 0;
    for (cycle, chunk) in inputs[..8].chunks(4).enumerate() {
        actual.extend(session.verify_mtp(b, chunk, true)?.iter().map(|o| bytes(o, row_width)).collect::<Result<Vec<_>>>()?);
        session.commit_verification(b, session.position())?;
        let stats = b.stats();
        if stats.graph_launches == before.graph_launches || (cycle == 1 && stats.graph_captures != captures) {
            return Err("MTP resize regression did not capture then replay verification graphs".into());
        }
        captures = stats.graph_captures;
    }
    if !session.disable_mtp_history(b)? { return Err("MTP resize regression could not disable history".into()); }
    actual.push(bytes(&session.forward(b, &inputs[8..], true)?, row_width)?);
    drop(session);
    let mut plain = model.session_plain_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox)?;
    plain.forward_prefill(b, &fixture.prompt, false)?;
    for (&token, row) in inputs.iter().zip(actual) {
        if bytes(&plain.forward(b, &[token], true)?, row_width)? != row { return Err("MTP resize regression changed full-row bytes".into()); }
    }
    eprintln!("flash_gate mtp_history_roundtrip verify_cycles=2 rows=9 full_row_bytes_equal=true PASS");
    Ok(())
}

#[cfg(target_os = "linux")]
pub fn run<'a>(b: &Backend<'a>, model: &Model<'a>, ring: SharedRing, fixtures: &[Fixture],
    rank: usize, mtp: usize, groups: [usize; 2], dir: &Path) -> Result<()> {
    let row_width = if !matches!(injection()?, Some(Injection::Matrix { .. })) {
        let args = std::env::args().collect::<Vec<_>>();
        let world = args.iter().position(|s| s == "--world").map(|i| args[i+1].parse::<usize>())
            .transpose().map_err(|e| e.to_string())?.unwrap_or(2);
        let (plan, receipt) = planned(Path::new(&args[2]), world, rank, fixtures, groups)?;
        plan.bind(dir, &receipt)?;
        print!("{receipt}");
        plan.row_bytes
    } else { VOCABULARY * 4 };
    if let Some(Injection::Matrix { verdict, mask }) = injection()? {
        #[cfg(feature = "flash-routing-per-row")]
        { return injection_matrix(b, model, ring, &fixtures[0], rank, verdict, mask, dir); }
        #[cfg(not(feature = "flash-routing-per-row"))]
        { let _ = (verdict, mask); return Err("Flash eight-slot matrix requires spark-engine/flash-routing-per-row".into()); }
    }
    let started = Instant::now();
    let injection = std::env::var("VADE_TEST_ROUTING_REJECT").ok().filter(|v| v != "0");
    let width_hook = spark_gpu_probe::TEST_ROUTING_REJECT_WIDTH;
    let hook = if width_hook { "1:3:16" } else { "1:3" };
    if injection.as_deref().is_some_and(|v| v != hook) {
        return Err(format!("Flash distinct-row gate needs VADE_TEST_ROUTING_REJECT={hook}"));
    }
    // Defer the env hook through references and setup; its first counted
    // verdict must be the poison cohort's chosen step, never a solo prefill.
    b.defer_test_routing_rejection(true);
    let weights = b.allocated_bytes();
    let reference = dir.join("reference");
    let mailbox = ring.lock()?.mailbox_memory().map(|(base, n)|
        unsafe { b.register_host(base, n) }.map(Rc::new)).transpose()?;
    if mtp == 3 { mtp_history_roundtrip(b, model, &ring, &fixtures[0], mailbox.clone(), row_width)?; }
    let mut batch = DecodeBatch::new(b, model)?;
    // Each independent axis compares to cold, ordinary batch-one MTP-0.
    // Homogeneous cohorts prefill one row, then copy its immutable checkpoint
    // into independent sessions before their real forward_batch decode.
    let cases = if mtp == 0 {
        vec![("reference".to_owned(), 1, None, false),
            ("batch-2".into(), 2, None, false), ("batch-8".into(), 8, None, false),
            ("batch-16".into(), 16, None, false),
            (format!("group-{}", groups[0]), 1, Some(groups[0]), false),
            (format!("group-{}", groups[1]), 1, Some(groups[1]), false),
            ("cache-hit".into(), 1, None, true)]
    } else if mtp == 3 { vec![("mtp-3".into(), 1, None, false), ("mtp-3-cache-hit".into(), 1, None, true)] }
    else { Vec::new() };
    println!("case rows bytes_compared bytes_different result");
    let mut failed = false;
    let follow_root = dir.join("follow-up-reference");
    let mut follow_reference = Compared { width: row_width, ..Default::default() };
    let mut follow_compared = Compared { width: row_width, ..Default::default() };
    for (name, count, group, cached) in cases {
        let baseline = name == "reference";
        let mut compared = Compared { width: row_width, ..Default::default() };
        let (mut proposals, mut accepted, mut verifications) = (0, 0, 0);
        let root = dir.join(&name);
        for fixture in fixtures {
            let capacity = fixture.capacity().max(512).div_ceil(512) * 512;
            let mut sessions = (0..count).map(|_| {
                let mut s = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
                if let Some(group) = group { s.set_prefill_group(group)?; }
                Ok(s)
            }).collect::<Result<Vec<_>>>()?;
            let mut seed = None;
            for (row, session) in sessions.iter_mut().enumerate() {
                if count > 1 {
                    session.enable_cache();
                    if row != 0 {
                        session.restore(b, seed.as_ref().ok_or("missing cohort prompt snapshot")?, false)?;
                        // No copied prompt-end output is counted as a comparison.
                        // Every restored row is checked from its first decode.
                        continue;
                    }
                }
                if cached {
                    session.enable_cache();
                    // The first exact restore retains a one-token tail beyond
                    // a 64-token boundary; the next prefill must merge it.
                    for (end, prefix) in [(fixture.prompt.len()/3/64*64 + 1, false),
                        (2*fixture.prompt.len()/3, true)] {
                        session.forward_prefill(b, &fixture.prompt[session.position()..end], false)?;
                        session.canonicalize(b)?;
                        let saved = session.snapshot(b)?;
                        session.release_mtp_workspace(b)?;
                        session.restore(b, &saved, prefix)?;
                    }
                }
                let output = session.forward_prefill(b, &fixture.prompt[session.position()..], true)?;
                compared.row(&root, &reference, fixture, row, 0, rank, &output, baseline)?;
                if cached {
                    let saved = session.snapshot(b)?;
                    let mut restored = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
                    restored.enable_cache();
                    restored.restore(b, &saved, false)?;
                    *session = restored;
                }
                if count > 1 { seed = Some(session.snapshot(b)?); }
            }
            drop(seed);
            let mut position = 1;
            while position < 64 {
                if mtp == 0 {
                    let tokens = vec![fixture.rows[position-1].0; count];
                    let mut refs = sessions.iter_mut().collect::<Vec<_>>();
                    let outputs = forward_batch(b, &mut batch, &mut refs, &tokens, u16::MAX >> (16-count))?;
                    for (row, output) in outputs.iter().enumerate() {
                        compared.row(&root, &reference, fixture, row, position, rank, output.as_ref().map_err(Clone::clone)?, baseline)?;
                    }
                    position += 1;
                } else {
                    let s = &mut sessions[0];
                    if s.mtp_depth() != 3 { return Err("Flash gate MTP-3 history is unavailable".into()); }
                    let depth = 3.min(63-position);
                    let mut proposed = vec![fixture.rows[position-1].0];
                    if depth != 0 { proposed.extend(s.propose_mtp(b, proposed[0], depth)?); }
                    let mut outputs = s.verify_mtp(b, &proposed, true)?;
                    let mut kept = 1;
                    // Keep only real accepted proposals whose inputs also
                    // equal the frozen teacher-forced continuation. Rejection
                    // commits the prefix; the next input uses the fixture ID.
                    while kept < proposed.len() && outputs[kept-1].token == proposed[kept]
                        && proposed[kept] == fixture.rows[position+kept-1].0 { kept += 1; }
                    proposals += proposed.len() - 1;
                    accepted += kept - 1;
                    verifications += 1;
                    s.commit_verification(b, s.position()-proposed.len()+kept)?;
                    outputs.truncate(kept);
                    for (i, output) in outputs.iter().enumerate() {
                        compared.row(&root, &reference, fixture, 0, position+i, rank, output, false)?;
                    }
                    position += kept;
                }
            }
            if cached && mtp == 0 {
                // 63 teacher-forced decode inputs, then a new turn's suffix.
                // Canonicalize decode arithmetic before restoring its exact tail.
                sessions[0].canonicalize(b)?;
                let saved = sessions[0].snapshot(b)?;
                sessions.clear();
                let mut restored = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
                restored.enable_cache();
                restored.restore(b, &saved, false)?;
                let output = restored.forward_prefill(b, &[fixture.rows[63].0], true)?;
                drop(restored);
                drop(saved);
                let mut cold = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
                let mut tokens = fixture.prompt.clone();
                tokens.extend(fixture.rows[..64].iter().map(|r| r.0));
                let expected = cold.forward_prefill(b, &tokens, true)?;
                follow_reference.row(&follow_root, &follow_root, fixture, 0, 64, rank, &expected, true)?;
                follow_compared.row(&dir.join("follow-up"), &follow_root, fixture, 0, 64, rank, &output, false)?;
            }
        }
        compared.print(&format!("logits-rank-{rank}-{name}"));
        failed |= compared.different != 0;
        if mtp != 0 {
            eprintln!("flash_gate mtp case={name} verifications={verifications} proposed_tokens={proposals} accepted_proposals={accepted}");
            if accepted == 0 {
                println!("logits-rank-{rank}-{name}-acceptance 0 0 0 FAIL");
                failed = true;
            }
        }
    }
    if mtp == 0 {
        follow_reference.print(&format!("logits-rank-{rank}-follow-up-reference"));
        follow_compared.print(&format!("logits-rank-{rank}-follow-up"));
        failed |= follow_compared.different != 0;
        let root = dir.join("batch-mixed");
        let mut compared = Compared { width: row_width, ..Default::default() };
        let mut sessions: Vec<Session<'_, '_, SharedRing>> = Vec::new();
        let mut positions = [0; 16];
        for row in 0..16 {
            let fixture = &fixtures[row % fixtures.len()];
            let capacity = fixture.capacity().max(512).div_ceil(512) * 512;
            let mut session = model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone())?;
            session.enable_cache();
            if row < fixtures.len() {
                let output = session.forward_prefill(b, &fixture.prompt, true)?;
                compared.row(&root, &reference, fixture, row, 0, rank, &output, false)?;
            } else {
                session.restore(b, &sessions[row % fixtures.len()].snapshot(b)?, false)?;
            }
            sessions.push(session);
        }
        // Cycle widths 3..16; compare each lane's independent cursor to its singleton, including tails.
        for step in 1..64 {
            let count = 3 + (step - 1) % 14;
            let tokens = (0..count).map(|row| fixtures[row % fixtures.len()].rows[positions[row]].0).collect::<Vec<_>>();
            let outputs = forward_batch(b, &mut batch, &mut sessions.iter_mut().take(count).collect::<Vec<_>>(), &tokens, u16::MAX >> (16-count))?;
            for (row, output) in outputs.iter().enumerate() {
                positions[row] += 1;
                compared.row(&root, &reference, &fixtures[row % fixtures.len()], row, positions[row], rank, output.as_ref().map_err(Clone::clone)?, false)?;
            }
        }
        compared.print(&format!("logits-rank-{rank}-batch-mixed"));
        failed |= compared.different != 0;
        drop(sessions);

        let mixed_started = Instant::now();
        // The earlier homogeneous 16 x 8192 case bounds these mixed sessions.
        let memory_cap = b.peak_allocated_bytes();
        b.set_allocation_budget(memory_cap)?;
        // Fixed fixture order makes lane identity independent of CLI ordering.
        let mut fixtures = fixtures.iter().collect::<Vec<_>>();
        fixtures.sort_by_key(|f| std::cmp::Reverse(f.prompt.len()));
        let mut mixed_collectives = [0; 48];
        for poison in [false, true] {
            let name = if poison { "batch-mixed-16-poison" } else { "batch-mixed-16" };
            if poison && !cfg!(feature = "flash-routing-per-row") {
                println!("logits-rank-{rank}-{name} 0 0 0 SKIP");
                eprintln!("flash_gate skip case={name} routing_reject_per_row=0; aggregate Flash rejection cannot isolate row 3");
                continue;
            }
            if poison && injection.is_none() {
                return Err(format!("Flash per-row gate needs VADE_TEST_ROUTING_REJECT={hook} at startup"));
            }
            let root = dir.join(name);
            let reference = if poison { dir.join("batch-mixed-16") } else { reference.clone() };
            let mut compared = Compared { same_row: poison, width: row_width, ..Default::default() };
            let case_started = Instant::now();
            let mut calls = MixedCalls::default();
            let mut sessions = mixed_sessions(b, model, &ring, &mailbox, &fixtures, mixed_started, &mut calls)?;
            let mut active = u16::MAX;
            // 48 passes keep all sixteen slots within the existing 64-row
            // references: lane r compares continuation positions r+1..r+48.
            for step in 1..=48 {
                mixed_time(mixed_started)?;
                let tokens = (0..16).map(|row| fixtures[row % 3].rows[row+step-1].0).collect::<Vec<_>>();
                if poison && step == 1 { b.defer_test_routing_rejection(false); }
                let result = forward_batch(b, &mut batch, &mut sessions.iter_mut().collect::<Vec<_>>(), &tokens, active);
                calls.batch_decode += 1;
                b.defer_test_routing_rejection(true);
                routing_result(&result, rank, name, step);
                let count = routing_collectives(&batch, rank, name, step, 16, active,
                    poison.then_some(mixed_collectives[step - 1]));
                let outputs = result?;
                let count = count?;
                if !poison { mixed_collectives[step - 1] = count; }
                if outputs.len() != 16 { return Err("Flash mixed-16 batch returned an incomplete cohort".into()); }
                for (row, output) in outputs.iter().enumerate() {
                    if poison && row == 3 {
                        if !output.as_ref().is_err_and(|e| e.starts_with("Flash MoE routing rejected")) {
                            return Err("Flash mixed-16 poison expected only row 3 to reject".into());
                        }
                        continue;
                    }
                    compared.row(&root, &reference, fixtures[row % 3], row, row+step, rank,
                        output.as_ref().map_err(Clone::clone)?, false)?;
                }
                if poison { active &= !(1 << 3); }
            }
            compared.print(&format!("logits-rank-{rank}-{name}"));
            failed |= compared.different != 0;
            let device_peak_bytes = b.peak_allocated_bytes();
            let extra_device_peak_bytes = device_peak_bytes - memory_cap;
            eprintln!("flash_gate {name} rank={rank} elapsed_ms={:.3} weights_bytes={weights} device_live_bytes={} device_peak_bytes={device_peak_bytes} device_cap_bytes={memory_cap} extra_device_peak_bytes={extra_device_peak_bytes} prefill_calls={} setup_decode_calls={} batch_decode_calls={} dump_bytes={}",
                case_started.elapsed().as_secs_f64()*1000.0, b.allocated_bytes(), calls.prefill, calls.setup_decode, calls.batch_decode, compared.bytes);
        }
        mixed_time(mixed_started)?;
        eprintln!("flash_gate mixed-16-bound elapsed_ms={:.3} time_cap_ms=120000 weights_bytes={weights} device_cap_bytes={memory_cap}",
            mixed_started.elapsed().as_secs_f64()*1000.0);
    }
    if mtp == ROWS_DEPTH { failed |= rows(b, model, &ring, fixtures, rank, dir, &reference, mailbox, row_width)?; }
    eprintln!("flash_gate complete rank={rank} mtp={mtp} elapsed_ms={:.3} weights_bytes={weights} device_peak_bytes={}",
        started.elapsed().as_secs_f64()*1000.0, b.peak_allocated_bytes());
    b.defer_test_routing_rejection(false);
    if failed { Err("Flash logit determinism differs".into()) } else { Ok(()) }
}

/// One-row versus (k+1)-row launches for k = 3 (the MTP draft length), 7 and
/// 15. Each fixture's prompt is prefilled once; per k a fresh session restores
/// it and verifies the 63 teacher-forced decode inputs in launches of k + 1
/// rows. Every row must equal the one-row MTP-0 reference at its position.
#[cfg(target_os = "linux")]
fn rows<'a>(b: &Backend<'a>, model: &Model<'a>, ring: &SharedRing, fixtures: &[Fixture], rank: usize,
    dir: &Path, reference: &Path, mailbox: Option<Rc<HostRegistration<'a>>>, row_width: usize) -> Result<bool> {
    let widths = [3, 7, ROWS_DEPTH].map(|k| k + 1);
    let mut compared = widths.map(|_| Compared { width: row_width, ..Default::default() });
    for fixture in fixtures {
        let capacity = fixture.capacity().max(512).div_ceil(512) * 512;
        let session = || model.session_with_mailbox(b, vec![ring.clone()], capacity, 64, mailbox.clone());
        let mut lead = session()?;
        lead.enable_cache();
        lead.forward_prefill(b, &fixture.prompt, false)?;
        let prompt = lead.snapshot(b)?;
        drop(lead);
        for (width, compared) in widths.iter().zip(&mut compared) {
            let mut s = session()?;
            s.restore(b, &prompt, false)?;
            if !s.prepare_mtp_workspace(b)? { return Err("Flash gate verification workspace is unavailable".into()); }
            let root = dir.join(format!("rows-{width}"));
            let mut position = 1;
            while position < 64 {
                let tokens = fixture.rows[position - 1..(position - 1 + width).min(63)].iter().map(|r| r.0).collect::<Vec<_>>();
                let outputs = s.verify_mtp(b, &tokens, true)?;
                s.commit_verification(b, s.position())?;
                for (i, output) in outputs.iter().enumerate() {
                    compared.row(&root, reference, fixture, 0, position + i, rank, output, false)?;
                }
                position += tokens.len();
            }
        }
    }
    for (width, compared) in widths.iter().zip(&compared) { compared.print(&format!("logits-rank-{rank}-rows-{width}")); }
    Ok(compared.iter().any(|c| c.different != 0))
}
