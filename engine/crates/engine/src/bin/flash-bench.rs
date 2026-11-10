//! Resident Flash fixture runner. Each repetition allocates a fresh session.
type Result<T> = std::result::Result<T, String>;
#[path = "flash-bench/server.rs"] mod server;
#[path = "flash-bench/gate.rs"] mod gate;
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/partition.rs"] mod partition;
#[cfg(target_os = "linux")] #[allow(dead_code)]
use engine_fabric::discovery;
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/control.rs"] mod control;
#[cfg(target_os = "linux")] #[allow(dead_code)]
#[path = "../../../seqpar/src/transport.rs"] mod transport;
#[cfg(target_os = "linux")] #[allow(dead_code)]
use engine_fabric::verbs;

#[cfg(target_os = "linux")]
#[derive(Clone)]
struct SharedRing(std::sync::Arc<std::sync::Mutex<transport::Ring>>);
#[cfg(target_os = "linux")]
impl SharedRing {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, transport::Ring>> {
        self.0.lock().map_err(|_| "Flash transport poisoned".into())
    }
}
#[cfg(target_os = "linux")]
impl spark_engine::flash::session::Exchange for SharedRing {
    fn profile_stats(&self) -> Option<spark_engine::flash::session::ExchangeStats> {
        let stats = self.lock().ok()?.stats;
        Some(spark_engine::flash::session::ExchangeStats { elapsed_ns: stats.elapsed_ns,
            sent: stats.sent, received: stats.received, exchanges: stats.exchanges })
    }
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<()> { self.lock()?.all_gather_bytes(values, offsets) }
    fn gather_f32(&mut self, values: &mut [f32], offsets: &[usize]) -> Result<()> { self.lock()?.all_gather_f32(values, offsets) }
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> { self.lock()?.all_reduce_f32(values, scratch) }
    fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> { self.lock().ok()?.mailbox_memory() }
    fn gather_slots_pipeline<T>(&mut self, offsets: &[Vec<usize>],
        fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.lock()?.gather_slots_pipeline(offsets, fill, receive, wait)
    }
    fn exchange_slots_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.lock()?.exchange_slots_pipeline(sizes, fill, receive, wait)
    }
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Fixture {
    path: std::path::PathBuf,
    prompt: Vec<u32>,
    rows: Vec<(u32, Option<std::path::PathBuf>)>,
}
impl Fixture {
    fn read(path: &str) -> Result<Self> {
        use engine_model::json::{self, field};
        let path = std::path::PathBuf::from(path);
        let doc = json::parse(&std::fs::read(&path).map_err(|e| e.to_string())?, json::Limits::default()).map_err(|e| e.to_string())?;
        let object = doc.object().map_err(|e| e.to_string())?;
        let token = |v: &json::Value| v.u64().map_err(|e| e.to_string())
            .and_then(|n| u32::try_from(n).map_err(|e| e.to_string()));
        let prompt = field(object, "prompt_ids").and_then(|v| v.array()).map_err(|e| e.to_string())?
            .iter().map(token).collect::<Result<Vec<_>>>()?;
        let rows = field(object, "rows").and_then(|v| v.array()).map_err(|e| e.to_string())?
            .iter().map(|row| {
                let row = row.object().map_err(|e| e.to_string())?;
                let expected = token(field(row, "token").map_err(|e| e.to_string())?)?;
                let saved = row.get("full_logits").map(|v| v.string()).transpose().map_err(|e| e.to_string())?
                    .map(|name| path.with_file_name(name));
                Ok((expected, saved))
            }).collect::<Result<Vec<_>>>()?;
        if prompt.is_empty() || rows.is_empty() || prompt.len() + rows.len() > 262144 {
            return Err("Flash fixture needs a nonempty prompt and rows within context capacity".into());
        }
        Ok(Self { path, prompt, rows })
    }
    #[cfg(target_os = "linux")]
    fn capacity(&self) -> usize { (self.prompt.len() + self.rows.len()).max(64) }
}

#[cfg(target_os = "linux")]
fn run() -> Result<()> {
    use engine_model::flash::Checkpoint;
    use spark_engine::flash::{moe::Moe, session::Model};
    use std::{ffi::CString, net::SocketAddr, path::{Path, PathBuf}, time::{Duration, Instant}};
    let start = Instant::now();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let usage = "usage: flash-bench PTX MODEL FIXTURE.json[,FIXTURE.json...] ADDR0 ADDR1 RANK [--sessions N] [--world N] [--path] [--mtp 0..15] [--prefill-group N] [--gate DIR --gate-group N]\nFor world > 2 or --path, ADDR0/ADDR1 mean this rank's LISTEN/NEXT endpoints.";
    if args.len() < 6 { return Err(usage.into()); }
    let (mut repetitions, mut world, mut path, mut mtp) = (1usize, 2usize, false, 0usize);
    let (mut gate_dir, mut prefill_group, mut gate_group) = (None, None, 512usize);
    let mut options = args[6..].iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--sessions" | "--world" | "--mtp" | "--prefill-group" | "--gate-group" => {
                let value = options.next().ok_or(usage)?.parse::<usize>().map_err(|e| e.to_string())?;
                match option.as_str() {
                    "--sessions" => repetitions = value, "--mtp" => mtp = value, "--world" => world = value,
                    "--prefill-group" => prefill_group = Some(value), _ => gate_group = value,
                }
            }
            "--gate" => gate_dir = Some(PathBuf::from(options.next().ok_or(usage)?)),
            "--path" => path = true,
            _ => return Err(usage.into()),
        }
    }
    let rank = args[5].parse::<usize>().map_err(|e| e.to_string())?;
    if repetitions == 0 || world > 128 || rank >= world { return Err("invalid Flash sessions/world/rank".into()); }
    let fixtures = args[2].split(',').map(Fixture::read).collect::<Result<Vec<_>>>()?;
    let total = repetitions.checked_mul(fixtures.len()).ok_or("Flash session count overflow")?;
    let capacity = fixtures.iter().map(Fixture::capacity).max().unwrap();
    let capacity = if gate_dir.is_some() { capacity.max(512).div_ceil(512) * 512 } else { capacity };
    let addresses = [args[3].parse::<SocketAddr>().map_err(|e| e.to_string())?, args[4].parse::<SocketAddr>().map_err(|e| e.to_string())?];
    let (listen, next) = if world == 2 && !path { (addresses[rank], addresses[1-rank]) } else { (addresses[0], addresses[1]) };
    let activation_quantization = Moe::activation_quantization()?;
    let dump = std::env::var_os("FLASH_DUMP_LOGITS").map(PathBuf::from);
    let cached = std::env::var("FLASH_CACHE_REPLAY").as_deref() == Ok("1");
    let teacher_forcing = std::env::var("FLASH_TEACHER_FORCING").as_deref() == Ok("1");
    let injection = std::env::var("VADE_TEST_ROUTING_REJECT").ok().filter(|v| v != "0");
    let injection_greedy = std::env::var("VADE_TEST_ROUTING_GREEDY").as_deref() == Ok("1");
    if (injection.is_some() && (gate_dir.is_none() || mtp != 0 || !matches!(world, 2 | 4)))
        || (injection_greedy && injection.is_none()) {
        return Err("routing injection requires the native world-2 or world-4 gate with MTP 0".into());
    }
    if gate_dir.is_none() && teacher_forcing && mtp != 0 { return Err("teacher forcing requires plain decoding".into()); }
    if gate_dir.is_some() {
        gate::validate(&fixtures, mtp, [prefill_group.unwrap_or(256), gate_group], world)?;
        if repetitions != 1 { return Err("Flash gate runs each case once".into()); }
    }
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let root = Path::new(&args[1]);
    let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
    let run_id = std::env::var("TP_RUN_ID").unwrap_or_else(|_| "flash-bench".into());
    let mut ring = transport::Ring::connect_topology(rank, world, listen, next, &run_id, path, Duration::from_secs(600))?;
    // Agree on every operation before entering the fixture loop. Paths may differ
    // across ranks; prompt IDs, row counts and logit requests must not.
    let mut schedule = vec![repetitions as u64, fixtures.len() as u64, u64::from(activation_quantization),
        u64::from(dump.is_some()), mtp as u64, u64::from(cached), u64::from(teacher_forcing),
        u64::from(gate_dir.is_some()), prefill_group.unwrap_or(0) as u64, gate_group as u64];
    schedule.push(u64::from(injection_greedy));
    schedule.push(injection.as_ref().map_or(0, String::len) as u64);
    if let Some(spec) = &injection { schedule.extend(spec.bytes().map(u64::from)); }
    for fixture in &fixtures {
        schedule.extend([fixture.prompt.len() as u64, fixture.rows.len() as u64]);
        schedule.extend(fixture.prompt.iter().map(|&token| token as u64));
        for (token, saved) in &fixture.rows { schedule.extend([*token as u64, u64::from(saved.is_some())]); }
    }
    let schedule = schedule.iter().flat_map(|n| n.to_le_bytes()).collect::<Vec<_>>();
    let mut lengths = vec![0; world * 8];
    lengths[rank*8..rank*8+8].copy_from_slice(&(schedule.len() as u64).to_le_bytes());
    ring.all_gather_bytes(&mut lengths, &(0..=world).map(|r| r*8).collect::<Vec<_>>())?;
    if lengths.as_chunks::<8>().0.iter().any(|n| *n != (schedule.len() as u64).to_le_bytes()) { return Err("Flash ranks differ in fixture schedule length".into()); }
    let mut wire = vec![0; schedule.len() * world];
    wire[rank*schedule.len()..(rank+1)*schedule.len()].copy_from_slice(&schedule);
    ring.all_gather_bytes(&mut wire, &(0..=world).map(|r| r*schedule.len()).collect::<Vec<_>>())?;
    if wire.chunks_exact(schedule.len()).any(|s| s != schedule) { return Err("Flash ranks differ in fixture schedule or precision".into()); }
    ring.enable_mailbox(8 << 20)?;
    ring.seal_registrations()?;
    println!("flash_session transport rank={rank} rdma={}", ring.uses_rdma());
    spark_gpu_probe::driver::with_backend(0, &ptx, 100 * 1024 * 1024 * 1024, |backend| {
        let model = Model::load_rank_with_mtp(backend, root, checkpoint, rank, world, capacity, 64, mtp)?;
        // Let slower loaders arrive before starting any 120-second mailbox collective.
        ring.barrier_with_timeout(Duration::from_secs(600))?;
        let precision = model.expert_precision();
        if !model.is_resident() { return Err("flash-bench requires resident weights; increase SEQPAR_MEMORY_BYTES or reduce the largest fixture".into()); }
        let weights_bytes = backend.allocated_bytes();
        backend.set_profiling(std::env::var("SPARK_PROFILE").is_ok_and(|v| v == "1"));
        let ring = SharedRing(std::sync::Arc::new(std::sync::Mutex::new(ring)));
        if let Some(dir) = &gate_dir {
            return gate::run(backend, &model, ring, &fixtures, rank, mtp,
                [prefill_group.unwrap_or(256), gate_group], dir);
        }
        let mut all_match = true;
        for repetition in 0..repetitions {
            for (fixture_index, fixture) in fixtures.iter().enumerate() {
                let index = repetition * fixtures.len() + fixture_index;
                println!("flash_bench begin session={index} repetition={repetition} fixture={fixture_index} path={}", fixture.path.display());
                let session_dump = dump.as_ref().map(|dir| dir.join(format!("session-{index:05}")));
                if let Some(dir) = &session_dump { std::fs::create_dir_all(dir).map_err(|e| e.to_string())?; }
                let session_start = Instant::now();
                let mut session = model.session(backend, vec![ring.clone()], fixture.prompt.len().max(512).div_ceil(512) * 512, 64)?;
                if let Some(group) = prefill_group { session.set_prefill_group(group)?; }
                let mut input = fixture.prompt.clone();
                if cached {
                    session.enable_cache();
                    for end in [input.len() / 3, 2 * input.len() / 3].into_iter().filter(|&n| n != 0) {
                        session.forward_prefill(backend, &input[session.position()..end], false)?;
                        session.canonicalize(backend)?;
                        let saved = session.snapshot(backend)?;
                        session.release_mtp_workspace(backend)?;
                        session.restore(backend, &saved, true)?;
                    }
                    input.drain(..session.position());
                }
                let (mut matches, mut logits_ok) = (true, true);
                let (mut prefill_ms, mut ttft_ms) = (0.0, 0.0);
                let (mut tokens, mut decode) = (Vec::new(), Vec::new());
                let (mut ready, mut cycles, mut accepted) = (std::collections::VecDeque::new(), 0usize, 0usize);
                for (position, (expected, saved)) in fixture.rows.iter().enumerate() {
                    let before = backend.stats();
                    session.reserve(backend, (session.position() + input.len() + mtp).div_ceil(512).saturating_mul(512).min(262144))?;
                    let output = if position == 0 || mtp == 0 { session.forward(backend, &input, saved.is_some() || dump.is_some())? }
                    else {
                        if ready.is_empty() {
                            let started = Instant::now();
                            let depth = mtp.min(fixture.rows.len() - position - 1);
                            let mut proposed = input.clone();
                            if depth != 0 { proposed.extend(session.propose_mtp(backend, input[0], depth)?); }
                            let mut verified = session.verify_mtp(backend, &proposed,
                                dump.is_some() || fixture.rows[position..].iter().any(|r| r.1.is_some()))?;
                            let mut kept = 1;
                            while kept < proposed.len() && verified[kept - 1].token == proposed[kept] { kept += 1; }
                            session.commit_verification(backend, session.position() - proposed.len() + kept)?;
                            verified.truncate(kept);
                            let ms = started.elapsed().as_secs_f64() * 1000.0 / kept as f64;
                            for output in &mut verified { output.elapsed_ms = ms; }
                            cycles += 1; accepted += kept;
                            ready.extend(verified);
                        }
                        ready.pop_front().ok_or("missing verified Flash output")?
                    };
                    if backend.is_profiling() {
                        println!("flash_profile session={index} position={position} wall_ms={:.3} stats={:?}", output.elapsed_ms, backend.stats().since(before));
                    }
                    if cached && position == 0 {
                        let saved = session.snapshot(backend)?;
                        session.restore(backend, &saved, false)?;
                        if session.mtp_depth() != mtp { return Err("MTP lost after cached prompt replay/restart".into()); }
                    }
                    if position == 0 { prefill_ms = output.elapsed_ms; ttft_ms = session_start.elapsed().as_secs_f64()*1000.0; }
                    else { decode.push(output.elapsed_ms); }
                    tokens.push(output.token);
                    matches &= output.token == *expected;
                    println!("flash_session position={position} token={} expected={expected} stage={} input_tokens={} elapsed_ms={:.3} load_ms={:.3} forward_ms={:.3} ranks={world} sequential_two_rank_one_spark=false", output.token, if position==0 { "prefill" } else { "generated" }, input.len(), output.elapsed_ms, output.load_ms, output.elapsed_ms-output.load_ms);
                    if let Some(dir) = &session_dump {
                        let logits = output.logits.as_ref().ok_or("missing Flash dump logits")?;
                        if logits.iter().any(|v| !v.is_finite()) { return Err("nonfinite Flash dump logits".into()); }
                        let bytes = logits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                        std::fs::write(dir.join(format!("position-{position:05}-rank-{rank}.f32")), bytes).map_err(|e| format!("write Flash logits: {e}"))?;
                    }
                    if let Some(saved) = saved {
                        let bytes = std::fs::read(saved).map_err(|e| e.to_string())?;
                        let actual = output.logits.ok_or("missing requested logit row")?;
                        if bytes.len() != actual.len()*4 || actual.iter().any(|v| !v.is_finite()) { return Err("invalid full logit row".into()); }
                        let error = actual.iter().zip(bytes.as_chunks::<4>().0.iter()).try_fold(0.0f32, |max, (a,b)| {
                            let expected = f32::from_le_bytes(*b);
                            if !expected.is_finite() { return Err("nonfinite reference logit"); }
                            Ok(max.max((a-expected).abs()))
                        })?;
                        println!("flash_logits position={position} max_abs_error={error:.9e}");
                        logits_ok &= error <= 1.33;
                    }
                    input.clear(); input.push(if teacher_forcing { *expected } else { output.token });
                }
                if mtp != 0 { println!("flash_mtp cycles={cycles} accepted_rows={accepted} mean_accept_length={:.6}", accepted as f64 / cycles.max(1) as f64); }
                session.report();
                println!("flash_session tokens_match={} logits_within_1.33={logits_ok} precision={precision} other_activations=fp32 activation_quantization={}", if matches { "yes" } else { "no" }, model.main_activation_quantization());
                let mean = decode.iter().sum::<f64>() / decode.len().max(1) as f64;
                decode.sort_by(f64::total_cmp);
                let median = if decode.is_empty() { 0.0 } else { (decode[(decode.len()-1)/2] + decode[decode.len()/2]) / 2.0 };
                println!("flash_bench session={index} fixture={fixture_index} repetition={repetition} prefill_ms={prefill_ms:.3} ttft_ms={ttft_ms:.3} decode_ms_per_token={mean:.3} decode_median_ms={median:.3} decode_n={} tokens={tokens:?} weights_bytes={weights_bytes} session_device_bytes={} session_wall_ms={:.3}", decode.len(), backend.allocated_bytes(), session_start.elapsed().as_secs_f64()*1000.0);
                all_match &= matches && logits_ok;
                // Session drop releases KV, recurrent/PLE state, scratch and graphs;
                // only the borrowed model and transport survive to the next fixture.
            }
        }
        println!("flash_bench complete sessions={total} weights_bytes={weights_bytes} wall_ms={:.3}", start.elapsed().as_secs_f64()*1000.0);
        if !all_match { return Err("Flash complete-session reference mismatch".into()); }
        Ok(())
    })
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--gate-dry-run") {
        if let Err(error) = gate::dry_run() { eprintln!("FAIL {error}"); std::process::exit(1); }
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--gate-capabilities") {
        gate::capabilities();
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--gate-compare") {
        if let Err(error) = gate::compare_dump() { eprintln!("FAIL {error}"); std::process::exit(1); }
        return;
    }
    if std::env::args().nth(1).as_deref() == Some("--gate-server") {
        if let Err(error) = server::run() { eprintln!("FAIL {error}"); std::process::exit(1); }
        return;
    }
    #[cfg(target_os = "linux")]
    if let Err(error) = run() { eprintln!("FAIL {error}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("flash-bench requires Linux CUDA"); std::process::exit(1); }
}
