//! Reproducible plain/STRICT/FAST comparison, using the public engine APIs.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_engine::{draft::{Config, CostCurve, Mode, Speculator}, sample::{Sampler, Sampling}};
    use std::{ffi::CString, path::{Path, PathBuf}, time::Instant};
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=11).contains(&args.len()) {
        return Err("usage: draft-generate PTX MODEL PROMPT_IDS_FILE_OR_DIR MAX_NEW [greedy|sampled|both] [SEED=42] [MAX_ROWS=16] [adaptive|fixed] [lookup] [strict|fast|both] [bench|check]".into());
    }
    let max_new = args[3].parse::<usize>().map_err(|e|e.to_string())?;
    let sampling = args.get(4).map(String::as_str).unwrap_or("both");
    let seed = args.get(5).map_or(Ok(42), |s|s.parse::<u64>()).map_err(|e|e.to_string())?;
    let max_rows = args.get(6).map_or(Ok(16), |s|s.parse::<usize>()).map_err(|e|e.to_string())?;
    let policy = args.get(7).map(String::as_str).unwrap_or("adaptive");
    let strategy = args.get(8).map(String::as_str).unwrap_or("lookup");
    let modes = args.get(9).map(String::as_str).unwrap_or("strict");
    let check = args.get(10).map(String::as_str).unwrap_or("bench");
    if max_new == 0 || !(1..=16).contains(&max_rows) || !["greedy","sampled","both"].contains(&sampling)
        || !["adaptive","fixed"].contains(&policy) || strategy != "lookup"
        || !["strict","fast","both"].contains(&modes) || !["bench","check"].contains(&check) {
        return Err("invalid benchmark options".into());
    }
    let path = Path::new(&args[2]);
    let mut paths: Vec<PathBuf> = if path.is_dir() {
        std::fs::read_dir(path).map_err(|e|e.to_string())?
            .map(|p|p.map(|p|p.path())).collect::<Result<Vec<_>,_>>().map_err(|e|e.to_string())?
            .into_iter().filter(|p|p.extension().is_some_and(|s|s=="ids")).collect()
    } else { vec![path.to_path_buf()] };
    paths.sort();
    let prompts = paths.iter().map(|p| {
        let ids = std::fs::read_to_string(p).map_err(|e|e.to_string())?.split_whitespace()
            .map(|s|s.parse::<u32>().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
        if ids.is_empty() { return Err("empty prompt".into()); }
        Ok((p.file_stem().ok_or("missing filename")?.to_string_lossy().into_owned(), ids))
    }).collect::<Result<Vec<_>,String>>()?;
    let capacity = prompts.iter().map(|p|p.1.len()).max().ok_or("no prompts")?
        .checked_add(max_new.max(16)).ok_or("capacity overflow")?;
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
    let bf16 = std::env::var("SPARK_KV_BF16").as_deref() == Ok("1");
    let guard_error = std::env::var("DRAFT_GUARD_ERROR").ok().map(|s|s.parse::<f64>()).transpose().map_err(|e|e.to_string())?
        .unwrap_or(if bf16 {0.2} else {0.01});
    spark_gpu_probe::driver::with_backend_mode(0, &ptx, 100*1024*1024*1024, spark_engine::configured_matvec_mode()?, |backend| {
        let load = Instant::now();
        let mut model = spark_engine::qwen::Model::load_with_kv_budget(backend, Path::new(&args[1]), capacity, 12*1024*1024*1024)?;
        let mut setup_peak = model.prepare_batch_weights()?;
        let model_bytes = backend.allocated_bytes();
        let weights = model.config().expected_shapes().values().map(|s|s.iter().product::<u64>()*2).sum::<u64>();
        println!("MODEL load_s={:.3} capacity={capacity} weights_payload_bytes={weights} model_device_bytes={model_bytes} setup_peak_device_bytes={setup_peak} pinned_upload_bytes={} kv_bf16={bf16} guard_error={guard_error}", load.elapsed().as_secs_f64(),backend.pinned_upload_bytes());
        let mut costs = [CostCurve::gb10(Mode::Strict),CostCurve::gb10(Mode::Fast)];
        let supplied_costs = std::env::var("DRAFT_COSTS").ok().map(|value| {
            let parsed = value.split(',').map(|v|v.parse::<f64>().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
            Ok::<_,String>(CostCurve(parsed.try_into().map_err(|_|"DRAFT_COSTS needs five comma-separated milliseconds for 1/2/4/8/16 rows")?))
        }).transpose()?;
        // Calibrate once for this loaded model, before timing any requests.
        // All prompts and samplers reuse the curve; useful cycles refine it.
        if check == "bench" || supplied_costs.is_some() {
            for (index, mode) in [(0, Mode::Strict), (1, Mode::Fast)] {
                let name = if mode == Mode::Strict { "strict" } else { "fast" };
                if modes != "both" && modes != name { continue; }
                let calibration = Instant::now();
                costs[index] = if let Some(curve) = supplied_costs { curve } else {
                    let mut session = model.session();
                    session.configure_kv_bf16(bf16)?;
                    session.prefill(&prompts[0].1)?;
                    let curve = CostCurve::measure(&mut session, &[*prompts[0].1.last().unwrap(); 16], mode)?;
                    setup_peak = setup_peak.max(session.device_peak_bytes());
                    curve
                };
                println!("CURVE mode={name} startup_ms={:.3} costs_ms={:?} charged_to_decode=false peak_device_bytes={setup_peak}", calibration.elapsed().as_secs_f64()*1000.0, costs[index].0);
            }
        }
        let mut all_match = true;
        for (name,prompt) in &prompts {
            let mut qualified = false;
            for sampling_mode in ["greedy","sampled"] {
                if sampling != "both" && sampling != sampling_mode { continue; }
                let options = Sampling { seed, ..if sampling_mode == "greedy" { Sampling::greedy() } else { Sampling::qwen(false) } };
                let mut reference = Vec::new();
                let mut reference_prefill = Vec::<f32>::new();
                let logit_audit = std::env::var_os("DRAFT_LOGIT_AUDIT").is_some() && sampling_mode=="sampled";
                let mut gold_rows = Vec::new();
                for mode in ["plain","strict","fast"] {
                    if mode != "plain" && modes != "both" && modes != mode { continue; }
                    let mut session = model.session();
                    session.configure_kv_bf16(bf16)?;
                    let prefill_start = Instant::now();
                    let logits = session.prefill(prompt)?;
                    let prefill_ms = prefill_start.elapsed().as_secs_f64()*1000.0;
                    if mode == "plain" { reference_prefill = logits.clone(); }
                    else {
                        let different = logits.iter().zip(&reference_prefill).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
                        if different > 0 {
                            let error = logits.iter().zip(&reference_prefill).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
                            return Err(format!("prefill is nondeterministic case={name} mode={mode} bit_differences={different} max_abs={error}"));
                        }
                    }
                    let mut scratch_peak = session.device_peak_bytes().saturating_sub(model_bytes+session.cache_bytes());
                    let start = Instant::now();
                    let (tokens, stats, draft_host_bytes, ttft_ms) = if mode == "plain" {
                        let mut sampler = Sampler::new(options)?;
                        let mut history = prompt.clone();
                        let mut next = sampler.sample(&logits,&history,prompt.len(),history.len(), |_|true)?;
                        let ttft_ms = prefill_ms + start.elapsed().as_secs_f64()*1000.0;
                        let mut stats = spark_engine::draft::Stats::default();
                        for _ in 0..max_new {
                            let token = next;
                            history.push(token);
                            next = if sampling_mode == "greedy" { session.step_greedy(token)? }
                                else {
                                    let row = session.step(token)?;
                                    let next = sampler.sample(&row, &history, prompt.len(), history.len(), |_|true)?;
                                    if logit_audit { gold_rows.push(row); }
                                    next
                                };
                            scratch_peak = scratch_peak.max(backend.allocated_bytes().saturating_sub(model_bytes+session.cache_bytes()));
                            stats.cycles += 1; stats.rows += 1;
                            if token as u64 == model.config().eos_token { break; }
                        }
                        let tokens = history[prompt.len()..].to_vec();
                        stats.max_rows = 1;
                        (tokens,stats,history.capacity()*4,ttft_ms)
                    } else {
                        let selected_mode = if mode == "strict" { Mode::Strict } else { Mode::Fast };
                        let curve = supplied_costs.unwrap_or(costs[usize::from(mode=="fast")]);
                        let config = Config { max_new, max_rows, mode: selected_mode,
                            adaptive: policy == "adaptive", automatic: false, costs: curve, guard_error };
                        let mut spec = Speculator::new(prompt, &logits, config, options)?;
                        let mut first_cycle_ms = 0.0;
                        let trace = std::env::var_os("DRAFT_TRACE").is_some();
                        let preempt = std::env::var_os("DRAFT_PREEMPT").is_some();
                        let mut turn = 0;
                        while !spec.is_done() {
                            let (prefix, rows_before) = (session.checkpoint(),spec.stats.rows);
                            let cap = if preempt { [16,0,1,4,8][turn%5].min(max_rows) } else { max_rows };
                            turn += 1;
                            if logit_audit && mode=="strict" {
                                spec.step(&mut Gold {session:&mut session, prompt:prompt.len(), tokens:&reference, rows:&gold_rows}, cap, |t|t as u64 == model.config().eos_token)?;
                            } else {
                                spec.step(&mut session, cap, |t|t as u64 == model.config().eos_token)?;
                            }
                            if spec.stats.cycles == 1 { first_cycle_ms = start.elapsed().as_secs_f64()*1000.0; }
                            if trace { println!("CYCLE mode={mode} sampling={sampling_mode} prefix={prefix} rows={} committed={}",spec.stats.rows-rows_before,session.checkpoint()-prefix); }
                            scratch_peak = scratch_peak.max(backend.allocated_bytes().saturating_sub(model_bytes+session.cache_bytes()));
                        }
                        let ttft_ms = prefill_ms + first_cycle_ms;
                        let bytes = spec.memory_bytes();
                        (spec.tokens().to_vec(),spec.stats,bytes,ttft_ms)
                    };
                    let seconds = start.elapsed().as_secs_f64();
                    if session.checkpoint() != prompt.len()+tokens.len() { return Err("uncommitted output token".into()); }
                    let matches = if mode == "plain" { reference = tokens.clone(); true } else { tokens == reference };
                    all_match &= matches;
                    let mismatch = tokens.iter().zip(&reference).position(|(a,b)|a!=b).unwrap_or(tokens.len().min(reference.len()));
                    let count = tokens.len();
                    let cycles = stats.cycles.max(1) as f64;
                    let peak = session.device_peak_bytes();
                    let process_peak = backend.peak_allocated_bytes();
                    let kv = session.cache_bytes();
                    let retained = backend.allocated_bytes();
                    println!("RESULT case={name} sampling={sampling_mode} mode={mode} seed={seed} strategy={strategy} policy={policy} max_rows={max_rows} prompt={} generated={count} matches_plain={matches} first_difference={mismatch} decode_s={seconds:.6} tok_s={:.3} ms_token={:.3} ttft_ms={ttft_ms:.3} cycles={} target_rows={} proposed={} accepted={} committed_per_cycle={:.3} guarded={} guard_fires={} guard_rate={:.6} rollbacks={} max_batch={} weights_payload_bytes={weights} model_device_bytes={model_bytes} kv_bytes={kv} retained_device_bytes={retained} forward_peak_device_bytes={peak} process_lifetime_peak_device_bytes={process_peak} peak_including_setup_bytes={} retained_scratch_bytes={} scratch_peak_bytes={scratch_peak} logits_upper_bytes={} draft_host_bytes={draft_host_bytes} sampler_host_upper_bytes={} host_rss_hwm_bytes={}",
                        prompt.len(), count as f64/seconds, seconds*1000.0/count as f64, stats.cycles,stats.rows,
                        stats.proposed,stats.accepted,count as f64/cycles,stats.guarded,stats.guard_fires,
                        stats.guard_fires as f64/stats.guarded.max(1) as f64,stats.rollbacks,stats.max_rows,setup_peak.max(peak),
                        retained.saturating_sub(model_bytes+kv),(stats.max_rows.max(if mode=="plain" {1} else {16})*2*logits.len()+logits.capacity()+reference_prefill.capacity())*4,
                        if sampling_mode=="greedy" {0} else {model.config().vocab_size*16},host_high_water());
                    if mode != "plain" {
                        println!("DRAFTER case={name} sampling={sampling_mode} mode={mode} draft_ms={:.3} recycle_ms={:.3} target_ms={:.3} costs_ms={:?} source_cycles={:?} source_accepted={:?} depth_trials={:?} depth_hits={:?} draft_host_bytes={draft_host_bytes}",
                            stats.draft_ms, stats.recycle_ms, stats.target_ms, stats.costs_ms,
                            stats.source_cycles, stats.source_accepted, stats.depth_trials, stats.depth_hits);
                    }
                    if let Ok(directory) = std::env::var("DRAFT_SAVE") {
                        std::fs::create_dir_all(&directory).map_err(|e|e.to_string())?;
                        let file = Path::new(&directory).join(format!("{name}-{sampling_mode}-{mode}.ids"));
                        std::fs::write(file,tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")).map_err(|e|e.to_string())?;
                    }
                    drop(session);
                    if mode == "plain" && check == "check" && !qualified {
                        costs = qualify(&model, name, prompt, &reference, bf16)?;
                        qualified = true;
                    }
                    if mode == "strict" && !matches { return Err(format!("STRICT token mismatch {name}/{sampling_mode} at {mismatch}")); }
                }
            }
        }
        println!("COMPLETE all_tokens_match_plain={all_match} remaining_device_bytes={}",backend.allocated_bytes());
        if !all_match { return Err("FAST differs from plain; see first_difference (not an exact result)".into()); }
        Ok(())
    })
}

#[cfg(target_os = "linux")]
struct Gold<'s,'m,'a,'b> {
    session: &'s mut spark_engine::qwen::Session<'m,'a,'b>,
    prompt: usize, tokens: &'s [u32], rows: &'s [Vec<f32>],
}
#[cfg(target_os = "linux")]
impl Gold<'_, '_, '_, '_> {
    fn check(&self, position: usize, row: &[f32]) -> Result<(),String> {
        let Some(gold) = self.rows.get(position-self.prompt) else { return Ok(()); };
        let different = row.iter().zip(gold).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
        if different > 0 {
            let error = row.iter().zip(gold).map(|(a,b)|(a-b).abs()).fold(0.0f32,f32::max);
            return Err(format!("GOLD position={position} bit_differences={different} max_abs={error}"));
        }
        Ok(())
    }
}
#[cfg(target_os = "linux")]
impl spark_engine::draft::Target for Gold<'_, '_, '_, '_> {
    fn checkpoint(&self) -> usize { self.session.checkpoint() }
    fn truncate(&mut self, p:usize) -> Result<(),String> { self.session.truncate(p) }
    fn step(&mut self, t:u32) -> Result<Vec<f32>,String> {
        let p = self.session.checkpoint(); let row = self.session.step(t)?;
        if self.tokens.get(p-self.prompt)==Some(&t) { self.check(p,&row)?; }
        Ok(row)
    }
    fn forward_batch(&mut self, t:&[u32]) -> Result<Vec<Vec<f32>>,String> { self.session.forward_batch(t) }
    fn verify_batch(&mut self, tokens:&[u32]) -> Result<Vec<Vec<f32>>,String> {
        let p = self.session.checkpoint(); let rows = self.session.verify_batch(tokens)?;
        for (i,&t) in tokens.iter().enumerate() {
            if self.tokens.get(p+i-self.prompt)!=Some(&t) { break; }
            self.check(p+i,&rows[i]).map_err(|e|format!("{e} batch_prefix={p} batch_rows={} row={i}",tokens.len()))?;
        }
        Ok(rows)
    }
}

#[cfg(target_os = "linux")]
fn qualify(model: &spark_engine::qwen::Model<'_, '_>, name: &str, prompt: &[u32], continuation: &[u32], selected_bf16: bool) -> Result<[spark_engine::draft::CostCurve;2],String> {
    use std::time::Instant;
    use spark_engine::draft::{CostCurve, Mode};
    let mut costs = [CostCurve::gb10(Mode::Strict),CostCurve::gb10(Mode::Fast)];
    let tokens = &continuation[..16.min(continuation.len())];
    for bf16 in [false,true] {
        let mut session = model.session();
        session.configure_kv_bf16(bf16)?;
        session.prefill(prompt)?;
        let reference = tokens.iter().map(|&t|session.step(t)).collect::<Result<Vec<_>,_>>()?;
        for n in 1..=tokens.len() {
            for exact in [true,false] {
                let mut times = Vec::new();
                let mut rows = Vec::new();
                for _ in 0..3 {
                    session.truncate(prompt.len())?;
                    let start = Instant::now();
                    rows = if exact {session.verify_batch(&tokens[..n])?} else {session.forward_batch(&tokens[..n])?};
                    times.push(start.elapsed().as_secs_f64()*1000.0);
                }
                times.sort_by(f64::total_cmp);
                let ms = times[1];
                if bf16 == selected_bf16 {
                    if let Some(slot) = [1,2,4,8,16].iter().position(|&v| v == n) { costs[usize::from(!exact)].0[slot] = ms; }
                }
                let mut different = 0;
                let mut error = 0.0f32;
                let mut top1 = 0;
                for (a,b) in rows.iter().zip(&reference) {
                    for (&a,&b) in a.iter().zip(b) { different += usize::from(a.to_bits()!=b.to_bits()); error = error.max((a-b).abs()); }
                    top1 += usize::from(spark_engine::greedy_token(a)? == spark_engine::greedy_token(b)?);
                }
                println!("CHECK case={name} kv_bf16={bf16} exact={exact} rows={n} ms={ms:.3} bit_differences={different} max_abs={error:.8} top1_matches={top1} peak_device_bytes={}",session.device_peak_bytes());
                if exact && different!=0 { return Err(format!("verify_batch differs from {n} serial steps ({different} logits), BF16={bf16}")); }
            }
        }
    }
    // Short continuations cannot measure every knot; let real cycles calibrate.
    for curve in &mut costs { if curve.0.contains(&0.0) { *curve = CostCurve([0.0; 5]); } }
    println!("CURVE case={name} strict_ms={:?} fast_ms={:?}",costs[0].0,costs[1].0);
    Ok(costs)
}

#[cfg(target_os = "linux")]
fn host_high_water() -> usize {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|s|s.lines().find(|l|l.starts_with("VmHWM:"))
        .and_then(|l|l.split_whitespace().nth(1)?.parse::<usize>().ok())).unwrap_or(0)*1024
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("requires Linux CUDA driver"); std::process::exit(1); }
}
