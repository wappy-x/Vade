// Actual-checkpoint batch parity, wall timings, and allocation ledger.
#[cfg(target_os = "linux")]
#[path = "../checks/budget.rs"]
mod budget;
#[cfg(target_os = "linux")]
#[path = "../checks/cache.rs"]
mod cache;
#[cfg(target_os = "linux")]
#[path = "../checks/prefill.rs"]
mod prefill;
#[cfg(any(target_os = "linux", test))]
fn exact(a: &[f32], b: &[f32], phase: &str, row: usize) -> Result<(), String> {
    if a.is_empty() || a.len() != b.len() { return Err(format!("{phase} row={row} invalid logit widths {} vs {}", a.len(), b.len())); }
    for (index, (a, b)) in a.iter().zip(b).enumerate() {
        if !a.is_finite() || !b.is_finite() || a.to_bits() != b.to_bits() {
            return Err(format!("{phase} row={row} index={index} reference_bits={:08x} actual_bits={:08x} reference={a:?} actual={b:?}", a.to_bits(), b.to_bits()));
        }
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use std::{ffi::CString, path::Path, time::Instant};
    use spark_engine::{greedy_token, qwen::Model};
    use spark_gpu_probe::driver::{with_backend_mode, KvFormat, MatvecMode};
    let mut args: Vec<_> = std::env::args().skip(1).collect();
    let gpu_budget = budget::take(&mut args)?;
    let mut verify_prefixes = None;
    let mut verify_widths = None;
    for (flag, limit, values) in [("--verify-prefixes", 38912, &mut verify_prefixes), ("--verify-widths", 64, &mut verify_widths)] {
        if let Some(i) = args.iter().position(|s| s == flag) {
            if i + 1 == args.len() { return Err(format!("{flag} requires comma-separated integers")); }
            let parsed = args.remove(i + 1).split(',').map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            if parsed.is_empty() || parsed.iter().any(|&n| n > limit || (flag == "--verify-widths" && n == 0)) {
                return Err(format!("invalid {flag}"));
            }
            args.remove(i); *values = Some(parsed);
        }
    }
    let fast_prefill = args.iter().any(|s| s == "--fast-prefill"); args.retain(|s| s != "--fast-prefill");
    let tokens_file = if let Some(i) = args.iter().position(|s| s == "--tokens-file") {
        if i+1 == args.len() { return Err("--tokens-file requires a file of whitespace-separated token IDs".into()); }
        let path = args.remove(i+1); args.remove(i); Some(path)
    } else { None };
    if args.len() < 2 { return Err("usage: batch-bench PTX MODEL [MAX_PROMPT=2048] [DECODE=32] [bench|check|verify|profile|profile-bf16] [LOGITS_FILE] [fp32|bf16|i16] [--fast-prefill] [--tokens-file FILE] [--gpu-budget-mib N], or PTX MODEL cache|prefill [--gpu-budget-mib N]".into()); }
    match args.get(2).map(String::as_str) {
        Some("cache") => return cache::run(&args[..2], gpu_budget),
        Some("prefill") => return prefill::run(&args[..2], gpu_budget),
        _ => {}
    }
    let max = args.get(2).map_or(Ok(2048), |s| s.parse::<usize>()).map_err(|e| e.to_string())?;
    let decode = args.get(3).map_or(Ok(32), |s| s.parse::<usize>()).map_err(|e| e.to_string())?;
    let profile_bf16 = args.get(4).is_some_and(|s| s == "profile-bf16");
    let profile = profile_bf16 || args.get(4).is_some_and(|s| s == "profile");
    let parse_format = |name: &str| match name {
        "fp32" => Some(KvFormat::Fp32), "bf16" => Some(KvFormat::Bf16),
        "i16" => Some(KvFormat::ScaledI16), _ => None,
    };
    // Keep the existing saved-logits slot; a format alone may occupy it.
    let (logits_path, format_name) = match (args.get(5), args.get(6)) {
        (path, Some(format)) => (path, Some(format)),
        (Some(format), None) if parse_format(format).is_some() => (None, Some(format)),
        (path, None) => (path, None),
    };
    let formats = if let Some(name) = format_name {
        vec![parse_format(name).ok_or("KV format must be fp32, bf16, or i16")?]
    } else if profile {
        vec![if profile_bf16 { KvFormat::Bf16 } else { KvFormat::Fp32 }]
    } else { vec![KvFormat::Fp32, KvFormat::Bf16] };
    let verify_only = args.get(4).is_some_and(|s| s == "verify");
    if (!verify_only && (verify_prefixes.is_some() || verify_widths.is_some())) || verify_prefixes.as_ref().is_some_and(|v| v.iter().any(|&p| p > max)) {
        return Err("verify selection requires verify mode and prefixes <= MAX_PROMPT".into());
    }
    let check = !profile && !verify_only && args.get(4).is_none_or(|s| s != "bench");
    if max < 16 || max > 38912 || (max > 2048 && !profile && !verify_only) { return Err("MAX_PROMPT must be 16..=38912; above 2048 use profile or verify".into()); }
    let baseline = if profile {
        let path = logits_path.ok_or("profile requires an existing baseline LOGITS_FILE")?;
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read profile baseline {path}: {e}; baseline must already exist"))?;
        if bytes.is_empty() || bytes.len()%4 != 0 { return Err("invalid baseline logit bytes".into()); }
        Some(bytes.chunks_exact(4).map(|x| f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>())
    } else { None };
    let needed = max + if verify_only {64} else {0};
    let prompt: Vec<u32> = if let Some(path) = tokens_file {
        let tokens = std::fs::read_to_string(&path).map_err(|e|format!("{path}: {e}"))?.split_whitespace()
            .map(|s|s.parse::<u32>().map_err(|e|e.to_string())).collect::<Result<Vec<_>,_>>()?;
        if tokens.len()<needed { return Err(format!("tokens file has {} IDs; need {needed}",tokens.len())); }
        tokens
    } else { (1..=needed as u32).collect() };
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let compare = |a: &[f32], b: &[f32]| -> Result<(bool, f32), String> {
        if a.len() != b.len() { return Err("logit widths differ".into()); }
        if a.iter().chain(b).any(|x|!x.is_finite()) { return Err("non-finite comparison logits".into()); }
        let error = a.iter().zip(b).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
        Ok((greedy_token(a)? == greedy_token(b)?, error))
    };
    if engine_model::cartridge::Kind::detect(Path::new(&args[1])).map_err(|e| e.to_string())? == engine_model::cartridge::Kind::Qwen3_5 {
        return with_backend_mode(0, &ptx, gpu_budget.unwrap_or(100*1024*1024*1024), MatvecMode::Warp32, |b| {
            let model = spark_engine::hybrid::Model::load(b, Path::new(&args[1]), max+decode+64, 1)?;
            let mut reference = model.plain_session()?;
            let cold = reference.prefill(&prompt[..max])?;
            for split in verify_prefixes.unwrap_or_default() {
                let mut s = model.plain_session()?;
                if split > 0 && split < max { s.prefill_advance(&prompt[..split])?; }
                let out = if split < max { s.prefill(&prompt[split..max])? } else { s.prefill(&prompt[..max])? };
                if cold.iter().zip(out).any(|(a,b)| a.to_bits()!=b.to_bits()) { return Err(format!("hybrid split {split} differs")); }
            }
            let mut sliced = model.plain_session()?;
            let mut yields = 0;
            let sliced_logits = loop {
                if let Some(logits) = sliced.prefill_slice(&prompt[..max], true, Some(std::time::Duration::ZERO))? { break logits; }
                if sliced.can_fork() { return Err("hybrid exposed an unfinished prefill".into()); }
                yields += 1;
            };
            if cold.len() != sliced_logits.len() || cold.iter().zip(&sliced_logits).any(|(a,b)| a.to_bits()!=b.to_bits()) { return Err("hybrid sliced prefill differs".into()); }
            let token = greedy_token(&cold)?;
            let gold = reference.step(token)?;
            if gold.iter().zip(sliced.step(token)?).any(|(a,b)| a.to_bits()!=b.to_bits()) { return Err("hybrid sliced decode differs".into()); }
            println!("HYBRID prefill_yields={yields} bit_identical_logits=true peak_device_bytes={}", b.peak_allocated_bytes());
            drop(sliced);
            let start = Instant::now();
            for width in verify_widths.unwrap_or_else(|| (1..=64).collect()) {
                let mut sessions = (0..width).map(|_| model.plain_session()).collect::<Result<Vec<_>, _>>()?;
                for s in &mut sessions { s.prefill_advance(&prompt[..max])?; }
                let mut batch = spark_engine::hybrid::decode::Decode::with_rows(&model, width)?;
                let mut rows = sessions.iter_mut().map(|session| spark_engine::hybrid::decode::Row { session, token, logits: true }).collect::<Vec<_>>();
                let output = batch.step(&mut rows)?;
                if !batch.rejected().is_empty() || output.iter().any(|row| row.iter().zip(&gold).any(|(a,b)| a.to_bits()!=b.to_bits())) { return Err(format!("hybrid width {width} differs")); }
                println!("HYBRID width={width} bit_identical_logits=true elapsed_ms={:.3} peak_device_bytes={}", start.elapsed().as_secs_f64()*1000.0, b.peak_allocated_bytes());
            }
            Ok(())
        });
    }
    if engine_model::cartridge::Cartridge::inspect(Path::new(&args[1])).map_err(|e| e.to_string())?.gemma4.is_some() {
        return with_backend_mode(0, &ptx, gpu_budget.unwrap_or(100*1024*1024*1024), MatvecMode::Warp32, |b| {
            let model = spark_engine::gemma4::Model::load(b, Path::new(&args[1]), max+decode+64)?;
            let mut reference = model.session();
            let mut logits = reference.prefill(&prompt[..max], true)?;
            let mut gold = Vec::new(); let mut inputs = Vec::new();
            for _ in 0..decode {
                let token = greedy_token(&logits)?; inputs.push(token);
                logits = reference.prefill(&[token], true)?; gold.push(logits.clone());
            }
            let mut sessions = (0..16).map(|_| model.session()).collect::<Vec<_>>();
            for session in &mut sessions { session.prefill(&prompt[..max], false)?; }
            let start = Instant::now();
            for (step, &token) in inputs.iter().enumerate() {
                let mut rows = sessions.iter_mut().map(|s| (s, token)).collect::<Vec<_>>();
                let output = model.step_batch(&mut rows)?;
                if output.iter().any(|row| row.iter().zip(&gold[step]).any(|(a,b)| a.to_bits()!=b.to_bits())) {
                    return Err(format!("Gemma batched logits differ at decode step {step}"));
                }
            }
            println!("GEMMA parity=16/16 bit_identical_logits=true steps={decode} elapsed_ms={:.3} peak_device_bytes={}",
                start.elapsed().as_secs_f64()*1000.0, b.peak_allocated_bytes()); Ok(())
        });
    }
    with_backend_mode(0, &ptx, gpu_budget.unwrap_or(100 * 1024 * 1024 * 1024), MatvecMode::Warp32, |b| {
        b.use_vec8(false)?;
        let mut model = Model::load_with_kv_budget(b, Path::new(&args[1]), max + decode + 64, 12 * 1024 * 1024 * 1024)?;
        let model_bytes = b.allocated_bytes();
        let kv_elements = model.config().layers * model.config().kv_heads * model.config().head_dim * 2;
        println!("MEMORY model_device_bytes={model_bytes} fp32_kv_bytes_per_token={} bf16_kv_bytes_per_token={} fp32_tokens_per_gib={:.3} bf16_tokens_per_gib={:.3}",
            kv_elements*4, kv_elements*2, 1073741824.0/(kv_elements as f64*4.0), 1073741824.0/(kv_elements as f64*2.0));
        let prompt = &prompt[..needed];
        let lengths: Vec<usize> = if verify_only {vec![]} else if profile { vec![max] } else { [16,64,512,1024,2000,2048].into_iter().filter(|&n| n <= max).collect() };
        let mut fp32_ok = true;
        let mut bf16_ok = true;
        let mut i16_ok = true;
        let mut prepared = false;
        for &format in &formats {
            let (vec8, bf16, packed) = (false, format == KvFormat::Bf16, true);
            if packed && !prepared {
                let start = Instant::now();
                let peak = model.prepare_batch_weights()?;
                prepared = true;
                println!("PACK_WEIGHTS ms={:.3} retained_device_bytes={} conversion_peak_bytes={peak}", start.elapsed().as_secs_f64()*1000.0, b.allocated_bytes());
            }
            b.use_vec8(vec8)?;
            for &n in &lengths {
                // Same KV format, prompt phase and chunk geometry. All-head
                // references fit the public 512-row API; longer prompts replay
                // the identical prefill geometry, never scalar decode.
                let reference = if check {
                    let mut reference = model.session();
                    reference.configure_kv_format(format)?;
                    reference.configure_fast_prefill(fast_prefill)?;
                    let all = n <= 512 && !fast_prefill;
                    let row = if all { reference.forward_batch(&prompt[..n])?.pop().ok_or("missing prompt reference")? }
                        else { reference.prefill(&prompt[..n])? };
                    let mut next = greedy_token(&row)?;
                    let mut expected = Vec::new();
                    if n == max { for _ in 0..decode { next = reference.step_greedy(next)?; expected.push(next); } }
                    println!("REFERENCE phase=prompt tokens={n} kv_format={format:?} heads={} continuation_from=prompt", if all { "all" } else { "last-repeat" });
                    Some((row, expected))
                } else { None };
                let mut session = model.session();
                session.configure_kv_format(format)?;
                session.configure_fast_prefill(fast_prefill)?;
                let before = b.stats();
                b.set_profiling(profile);
                let start = Instant::now();
                let logits = session.prefill(&prompt[..n])?;
                let ms = start.elapsed().as_secs_f64()*1000.0;
                if format == KvFormat::ScaledI16 && logits.iter().any(|x| !x.is_finite()) { return Err("non-finite I16 prefill logits".into()); }
                b.set_profiling(false);
                if profile { println!("PREFILL_PROFILE {:?}", b.stats().since(before)); }
                if let Some(reference) = &baseline {
                    let (same, error) = compare(&logits, reference)?;
                    let bit_differences = logits.iter().zip(reference).filter(|(a,b)| a.to_bits() != b.to_bits()).count();
                    println!("COMPARE diagnostic_only=true exact_checked=false top1_same={same} max_abs={error:.8} bit_differences={bit_differences} peak_device_bytes={}",session.device_peak_bytes());
                }
                if let Some((row, _)) = &reference { exact(row, &logits, "prompt prefill", n - 1)?; }
                let (same, error) = (true, 0.0);
                if format == KvFormat::ScaledI16 { i16_ok &= same && error < 0.01; }
                else if bf16 { bf16_ok &= same; } else { fp32_ok &= same && error < 0.01; }
                println!("PREFILL tokens={n} vec8={vec8} bf16={bf16} kv_format={format:?} packed={packed} ms={ms:.3} top1={} same={same} max_abs={error:.8} model_bytes={model_bytes} kv_bytes={} retained_device_bytes={} peak_device_bytes={} peak_scratch_bytes={}",
                    greedy_token(&logits)?, session.cache_bytes(), b.allocated_bytes(), session.device_peak_bytes(), session.device_peak_bytes()-model_bytes-session.cache_bytes());
                if n == max && decode > 0 {
                    let mut next = greedy_token(&logits)?;
                    let start = Instant::now();
                    let mut matches = 0; let mut generated = Vec::with_capacity(decode);
                    for i in 0..decode { generated.push(next); next = session.step_greedy(next)?; if reference.as_ref().is_none_or(|(_, expected)| expected[i] == next) { matches += 1; } }
                    println!("GENERATED fast_prefill={fast_prefill} context={n} ids={generated:?}");
                    let ms = start.elapsed().as_secs_f64()*1000.0/decode as f64;
                    if format == KvFormat::ScaledI16 { i16_ok &= matches == decode; }
                    else if bf16 { bf16_ok &= matches == decode; } else { fp32_ok &= matches == decode; }
                    println!("DECODE vec8={vec8} bf16={bf16} kv_format={format:?} packed={packed} context={n} ms={ms:.3} tok_s={:.3} matches={matches}/{decode} retained_device_bytes={} peak_device_bytes={}", 1000.0/ms, b.allocated_bytes(), session.device_peak_bytes());
                    if profile {
                        let before = b.stats(); b.set_profiling(true);
                        session.step_greedy(next)?; b.set_profiling(false);
                        println!("DECODE_PROFILE {:?}", b.stats().since(before));
                    }
                }
            }
            if profile { continue; }
            let prefixes = if verify_only { verify_prefixes.clone().unwrap_or_else(|| [0,32,63,64,2000,2040,4088,4095,16376,16383].into_iter().filter(|&p| p<=max).collect::<Vec<_>>()) }
                else { vec![if max >= 80 {64} else {0}] };
            for prefix in prefixes {
            let widths = if verify_only { verify_widths.clone().unwrap_or_else(|| (1..=64).collect::<Vec<_>>()) } else { vec![1,4,8,16] };
            for t in widths {
                let mut session = model.session();
                session.configure_kv_format(format)?;
                session.configure_fast_prefill(fast_prefill)?;
                if prefix > 0 { session.prefill(&prompt[..prefix])?; }
                let reference = if verify_only || check {
                    let mut branch = session.fork()?;
                    Some(if verify_only { prompt[prefix..prefix+t].iter().map(|&id| branch.step(id)).collect::<Result<Vec<_>,_>>()? }
                        else { branch.forward_batch(&prompt[prefix..prefix+t])? })
                } else { None };
                let mut times = Vec::new();
                let error = 0.0f32;
                let same = true;
                for repeat in 0..4 {
                    session.truncate(prefix)?;
                    let profile_verify = repeat == 1 && std::env::var("SPARK_PROFILE_VERIFY").as_deref() == Ok("1");
                    let before = b.stats();
                    b.set_profiling(profile_verify);
                    let start = Instant::now();
                    let rows = if verify_only {session.verify_batch(&prompt[prefix..prefix+t])?} else {session.forward_batch(&prompt[prefix..prefix+t])?};
                    if let Some(reference) = reference.as_ref().filter(|_|verify_only) {
                        let mismatches = rows.iter().flatten().zip(reference.iter().flatten()).filter(|(a,b)| a.to_bits()!=b.to_bits()).count();
                        if mismatches != 0 { return Err(format!("verify prefix={prefix} t={t} bf16={bf16} kv_format={format:?} repeat={repeat} differs from canonical decode in {mismatches} elements")); }
                    }
                    let ms = start.elapsed().as_secs_f64()*1000.0;
                    b.set_profiling(false);
                    if profile_verify {
                        let stats = b.stats().since(before);
                        println!("VERIFY_PROFILE t={t} packed={packed} bf16={bf16} kv_format={format:?} wall_ms={ms:.3} download_ns={} wait_ns={} kernels={:?}", stats.output_download_ns, stats.stream_wait_ns, stats.device_kernel_ns);
                    }
                    if repeat > 0 { times.push(ms); }
                    if check {
                        let reference = reference.as_ref().unwrap();
                        if rows.len() != reference.len() { return Err("prompt batch row count differs".into()); }
                        for (i, (expected, row)) in reference.iter().zip(&rows).enumerate() { exact(expected, row, "prompt batch", prefix + i)?; }
                    }
                }
                times.sort_by(f64::total_cmp);
                if format == KvFormat::ScaledI16 { i16_ok &= same && error < 0.01; }
                else if bf16 { bf16_ok &= same; } else { fp32_ok &= same && error < 0.01; }
                println!("BATCH exact={verify_only} t={t} prefix={prefix} vec8={vec8} bf16={bf16} kv_format={format:?} packed={packed} median_ms={:.3} same={same} max_abs={error:.8} retained_device_bytes={} peak_device_bytes={}", times[1], b.allocated_bytes(), session.device_peak_bytes());
            }
            }
            if check && packed && !bf16 && max >= 511 {
                let mut session = model.session();
                session.configure_kv_format(format)?;
                session.configure_fast_prefill(fast_prefill)?;
                let mut start = 0;
                for end in [33,162,511] {
                    let expected = session.fork()?.forward_batch(&prompt[start..end])?.pop().ok_or("missing append reference")?;
                    let row = session.prefill(&prompt[start..end])?;
                    exact(&expected, &row, "prompt append", end - 1)?;
                    let (same,error) = (true, 0.0);
                    if format == KvFormat::ScaledI16 { i16_ok &= same && error < 0.01; }
                    else { fp32_ok &= same && error < 0.01; }
                    println!("APPEND kv_format={format:?} base={start} t={} same={same} max_abs={error:.8}",end-start);
                    start=end;
                }
            }
        }
        let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        for line in status.lines().filter(|line| line.starts_with("VmHWM:") || line.starts_with("VmRSS:")) { println!("HOST {line}"); }
        println!("RESULT checked={check} exact_checked={verify_only} prompt_exact_checked={check} profile_diagnostic_only={profile} fp32_ok={fp32_ok} bf16_top1_ok={bf16_ok} i16_ok={i16_ok} remaining_device_bytes={}", b.allocated_bytes());
        if !i16_ok { return Err("Scaled I16 continuation differs".into()); }
        if !fp32_ok { return Err("FP32 continuation differs".into()); }
        if !bf16_ok { return Err("BF16 continuation differs".into()); }
        Ok(())
    })
}
#[cfg(test)] #[test] fn prompt_comparison_requires_all_finite_bits() {
    assert!(exact(&[1.0, 0.0], &[1.0, 0.0], "prompt", 3).is_ok()); for b in [vec![1.0, -0.0], vec![1.0, f32::from_bits(1)], vec![1.0], vec![]] { assert!(exact(&[1.0, 0.0], &b, "prompt", 3).is_err()); }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] { assert!(exact(&[value], &[value], "prompt", 3).is_err()); } assert!(exact(&[0.0], &[-0.0], "prompt", 3).unwrap_err().contains("row=3 index=0 reference_bits=00000000 actual_bits=80000000")); }
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("requires Linux CUDA driver"); std::process::exit(1); }
}
