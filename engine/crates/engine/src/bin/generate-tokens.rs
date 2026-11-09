//! Fixed-ID autoregressive runner. Text/tokenizer semantics belong to frontend.
#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use std::{ffi::CString, io::Write, path::Path};
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    let ignore_eos = if let Some(at) = args.iter().position(|s| s == "--ignore-eos") { args.remove(at); true } else { false };
    let mtp = if let Some(at) = args.iter().position(|s| s == "--mtp") { args.remove(at); true } else { false };
    let plain = if let Some(at) = args.iter().position(|s| s == "--plain") { args.remove(at); true } else { false };
    let depth = if let Some(at) = args.iter().position(|s| s == "--draft-depth") {
        args.remove(at); args.get(at).ok_or("missing draft depth")?.parse::<usize>().map_err(|e| e.to_string()).map(|n| { args.remove(at); n })?
    } else { 0 };
    let draft_model = if let Some(at) = args.iter().position(|s| s == "--draft-model") {
        args.remove(at); Some(std::path::PathBuf::from(args.get(at).ok_or("missing draft model")?.clone())).map(|p| { args.remove(at); p })
    } else { None };
    let depth = if let Some(at) = args.iter().position(|s| s == "--draft-k") {
        args.remove(at); let n = args.get(at).ok_or("missing draft K")?.parse::<usize>().map_err(|e| e.to_string())?;
        args.remove(at); if !(1..=8).contains(&n) { return Err("draft K must be in 1..8".into()); } n
    } else { depth };
    let usage = "usage: generate-tokens PTX CHECKPOINT_ROOT MAX_TOKENS MAX_NEW_TOKENS [--ignore-eos] [--plain|--mtp] [--draft-depth 0..4 (hybrid MTP default 3)] [--draft-model CHECKPOINT] (PROMPT_ID... | --tokens-file FILE)";
    if plain && (mtp || draft_model.is_some()) { return Err("--plain cannot select a model draft".into()); }
    if args.len() < 5 { return Err(usage.into()); }
    let capacity = args[2].parse::<usize>().map_err(|e| e.to_string())?;
    let max_new = args[3].parse::<usize>().map_err(|e| e.to_string())?;
    let prompt = if args[4] == "--tokens-file" {
        if args.len() != 6 { return Err(format!("--tokens-file requires exactly one file and no positional token IDs; {usage}")); }
        std::fs::read_to_string(&args[5]).map_err(|e| format!("tokens file {}: {e}", args[5]))?
            .lines().enumerate().map(|(line, id)| id.trim().parse::<u32>()
                .map_err(|e| format!("tokens file {} line {}: invalid token ID: {e}", args[5], line + 1)))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        if args[4..].iter().any(|arg| arg == "--tokens-file") { return Err(format!("cannot mix positional token IDs and --tokens-file; {usage}")); }
        args[4..].iter().map(|id| id.parse::<u32>().map_err(|e| format!("invalid prompt token ID {id:?}: {e}"))).collect::<Result<Vec<_>, _>>()?
    };
    if prompt.is_empty() { return Err("prompt must contain at least one token ID".into()); }
    if max_new == 0 || prompt.len().checked_add(max_new).ok_or("token count overflow")? > capacity { return Err(spark_gpu_probe::admission::token_refusal(prompt.len() as u128 + max_new as u128, capacity, &format!("input={}; OUTPUT={max_new}; output must be positive", prompt.len()), "MAX_TOKENS", "prompt length or MAX_NEW_TOKENS")); }
    let root = Path::new(&args[1]);
    if engine_model::cartridge::Cartridge::inspect(root).map_err(|e| e.to_string())?.gemma4.is_some() {
        let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        return spark_gpu_probe::driver::with_backend(0, &ptx, 100*1024*1024*1024, |b| {
            let mut model = spark_engine::gemma4::Model::load(b, root, capacity)?;
            if let Some(path) = &draft_model { model.load_assistant(path)?; }
            let mut session = model.session();
            b.set_profiling(std::env::var("SPARK_PROFILE").as_deref() == Ok("1"));
            let start = std::time::Instant::now();
            let mut logits = session.prefill(&prompt, true)?;
            if b.is_profiling() { eprintln!("prefill_device_profile {:?}", b.stats().device_kernel_ns); }
            b.set_profiling(false);
            let prefill = start.elapsed();
            let decode = std::time::Instant::now();
            let mut generated = 0;
            if draft_model.is_some() {
                let first = spark_engine::greedy_token(&logits)?;
                writeln!(std::io::stdout().lock(), "{first}").map_err(|e| e.to_string())?;
                generated = 1;
                let mut skip = 1;
                let mut spec = spark_engine::draft::Speculator::new(&prompt, &logits,
                    spark_engine::draft::Config::strict(max_new), spark_engine::sample::Sampling::greedy())?;
                let eos = |t| !ignore_eos && model.eos_token_ids().contains(&t);
                while generated < max_new && !(!ignore_eos && model.eos_token_ids().contains(&first)) && !spec.is_done() {
                    let plan = spec.prepare_with_target(&mut session, 16, eos, |_| true)?;
                    let rows = session.verify_batch(&plan.input)?;
                    for &token in spec.finish_exact(&mut session, plan, &rows, eos, |_| true)?.iter().skip(skip) {
                        writeln!(std::io::stdout().lock(), "{token}").map_err(|e| e.to_string())?;
                        generated += 1;
                    }
                    skip = 0;
                }
                eprintln!("assistant cycles={} target_rows={} proposed={} accepted={} proposal_ms={:.3} target_ms={:.3} session_bytes={} peak_device_bytes={}",
                    spec.stats.cycles, spec.stats.rows, spec.stats.proposed, spec.stats.accepted, spec.stats.model_proposal_ms,
                    spec.stats.target_ms, session.bytes(), b.peak_allocated_bytes());
            } else { for _ in 0..max_new {
                let token = spark_engine::greedy_token(&logits)?;
                writeln!(std::io::stdout().lock(), "{token}").map_err(|e| e.to_string())?; generated += 1;
                if !ignore_eos && model.eos_token_ids().contains(&token) { break; }
                if generated < max_new { logits = session.prefill(&[token], true)?; }
            }
            }
            let decode_seconds = decode.elapsed().as_secs_f64();
            eprintln!("prompt_tokens={} generated={generated} ttft_ms={:.3} elapsed_ms={:.3} decode_steps={} decode_ms_per_token={:.3} decode_tokens_per_sec={:.3} session_bytes={} peak_device_bytes={}",
                prompt.len(), prefill.as_secs_f64()*1000.0, start.elapsed().as_secs_f64()*1000.0,
                generated.saturating_sub(1), decode_seconds*1000.0 / generated.saturating_sub(1).max(1) as f64,
                generated.saturating_sub(1) as f64 / decode_seconds, session.bytes(), b.peak_allocated_bytes());
            Ok(())
        });
    }
    if engine_model::cartridge::Kind::detect(Path::new(&args[1])).map_err(|e| e.to_string())?
        == engine_model::cartridge::Kind::Qwen3_5 {
        let c = engine_model::qwen3_5::Config::open(&Path::new(&args[1]).join("config.json")).map_err(|e| e.to_string())?;
        let mtp = mtp || (!plain && draft_model.is_none() && max_new >= 32 && c.hidden == 5120
            && !c.fp8 && c.layers.len() == 64 && c.mtp_layers == 1 && !c.tied_embeddings);
        let depth = if mtp && draft_model.is_none() && !std::env::args().any(|s| s == "--draft-depth" || s == "--draft-k") { 3 } else { depth };
        return generate_hybrid(&args[0], Path::new(&args[1]), capacity, max_new, &prompt, ignore_eos, mtp, depth, draft_model.as_deref());
    }
    if mtp || draft_model.is_some() { return Err("hybrid draft options require a qwen3_5 checkpoint".into()); }
    let config = engine_model::qwen3::Checkpoint::inspect(Path::new(&args[1])).map_err(|e| e.to_string())?.config;
    let kv_budget = [config.layers, config.kv_heads, config.head_dim, capacity as u64, 2, 4]
        .into_iter().try_fold(1u64, |bytes, n| bytes.checked_mul(n).ok_or("KV byte count overflow"))?;
    let kv_budget = usize::try_from(kv_budget).map_err(|_| "KV byte count exceeds host address space")?;
    let ptx = CString::new(std::fs::read(&args[0]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let profile = std::env::var("SPARK_PROFILE").as_deref() == Ok("1");
    spark_gpu_probe::driver::with_backend_mode(0, &ptx, 100 * 1024 * 1024 * 1024, spark_engine::configured_matvec_mode()?, |backend| {
        let mut model = spark_engine::qwen::Model::load_with_kv_budget(backend, Path::new(&args[1]), capacity, kv_budget)?;
        let setup_peak = model.prepare_batch_weights()?;
        let model_bytes = backend.allocated_bytes();
        if prompt.iter().any(|&t| t as u64 >= model.config().vocab_size) { return Err("prompt token outside vocabulary".into()); }
        eprintln!("policy=greedy_lowest_id prompt_tokens={} max_new={} capacity={} eos_token={}", prompt.len(), max_new, capacity, model.config().eos_token);
        let mut session = model.session();
        let start = std::time::Instant::now();
        let logits = session.prefill(&prompt)?;
        let prefill = start.elapsed();
        let mut next = spark_engine::greedy_token(&logits)?;
        let mut score = logits[next as usize];
        let ttft = start.elapsed();
        let decode = std::time::Instant::now();
        let mut output = std::io::stdout().lock();
        let mut generated = 0;
        let mut stop = "length";
        for i in 0..max_new {
            writeln!(output, "generated_index={i} token={next} logit={score}").map_err(|e| e.to_string())?;
            output.flush().map_err(|e| e.to_string())?;
            generated += 1;
            if !ignore_eos && model.config().is_eos(next) {
                stop = "eos";
                break;
            }
            if i + 1 < max_new {
                backend.set_profiling(profile && i == 0);
                (next, score) = session.step_greedy_with_score(next)?;
                if profile && i == 0 {
                    let stats = backend.stats();
                    eprintln!("device_profile position={} queued_calls={} kernel_ns_matvec_rms_qk_attention_softmax_values_swiglu_residual_embed_argmax_inputpack_kvpack_copy={:?}", session.len() - 1, stats.queued_calls, stats.device_kernel_ns);
                }
            }
        }
        let seconds = decode.elapsed().as_secs_f64();
        writeln!(output, "stop={stop} generated={generated}").map_err(|e| e.to_string())?;
        eprintln!("timing prompt_tokens={} prefill_s={:.6} ttft_ms={:.3} decode_steps={} decode_ms_per_token={:.3} decode_tokens_per_sec={:.3}",
            prompt.len(), prefill.as_secs_f64(), ttft.as_secs_f64() * 1000.0, generated - 1,
            seconds * 1000.0 / (generated - 1).max(1) as f64, (generated - 1) as f64 / seconds);
        eprintln!("memory model_device_bytes={model_bytes} kv_bytes={} retained_device_bytes={} peak_device_bytes={}",session.cache_bytes(),backend.allocated_bytes(),setup_peak.max(session.device_peak_bytes()));
        Ok(())
    })
}
#[cfg(target_os = "linux")]
fn generate_hybrid(ptx: &str, root: &std::path::Path, capacity: usize, max_new: usize, prompt: &[u32], ignore_eos: bool, mtp: bool, depth: usize, draft_root: Option<&std::path::Path>) -> Result<(), String> {
    use std::{ffi::CString, io::Write, time::Instant};
    use spark_engine::draft::Target;
    if depth > if draft_root.is_some() { 8 } else { 4 } { return Err("draft depth exceeds model draft capacity".into()); }
    if let Some(draft) = draft_root { spark_engine::hybrid::mtp::validate_tokenizer(root, draft)?; }
    let ptx = CString::new(std::fs::read(ptx).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    spark_gpu_probe::driver::with_backend_mode(0, &ptx, 100 * 1024 * 1024 * 1024, spark_engine::configured_matvec_mode()?, |b| {
        let load = Instant::now();
        let model = spark_engine::hybrid::Model::load_with_mtp(b, root, capacity, 1, mtp)?;
        let draft_model = draft_root.map(|root| spark_engine::hybrid::Model::load(b, root, capacity, 1)).transpose()?;
        eprintln!("cartridge=qwen3_5 world_size=1 load_ms={:.3} text_weights_bytes={} host_weight_mirror_bytes=0 kv_precision=BF16 recurrent_precision=FP32", load.elapsed().as_secs_f64() * 1000.0, model.weight_bytes());
        let profile = std::env::var("SPARK_PROFILE").is_ok_and(|v| v != "0");
        let mut session = model.session()?;
        if let Some(draft) = &draft_model { session = session.with_draft_model(draft, depth)?; }
        else { session.set_draft_depth(depth)?; }
        b.set_profiling(profile);
        let before = b.stats();
        let start = Instant::now();
        let mut logits = session.prefill(prompt)?;
        let prefill = start.elapsed();
        b.set_profiling(false);
        if profile {
            let stats = b.stats().since(before);
            eprintln!("hybrid_prefill_kernel_ns={:?}", stats.device_kernel_ns);
            eprintln!("prefill_phase_ms dense={:.3} attention={:.3} gdn={:.3} norm={:.3} rope={:.3} gate={:.3} residual={:.3} pack={:.3} device_total={:.3}",
                stats.device_kernel_ns[0] as f64 / 1e6, stats.device_kernel_ns[3] as f64 / 1e6, (stats.device_kernel_ns[18] + stats.device_kernel_ns[19]) as f64 / 1e6,
                stats.device_kernel_ns[1] as f64 / 1e6, stats.device_kernel_ns[2] as f64 / 1e6, stats.device_kernel_ns[6] as f64 / 1e6,
                stats.device_kernel_ns[7] as f64 / 1e6, stats.device_kernel_ns[10] as f64 / 1e6,
                stats.device_kernel_ns[..14].iter().chain(stats.device_kernel_ns[18..20].iter()).sum::<u128>() as f64 / 1e6);
            eprintln!("prefill_detail_ms gdn_prepare={:.3} gdn_scan={:.3} mlp_gate_up={:.3} mlp_down={:.3}",
                stats.device_kernel_ns[18] as f64 / 1e6, stats.device_kernel_ns[19] as f64 / 1e6,
                stats.device_kernel_ns[20] as f64 / 1e6, stats.device_kernel_ns[17] as f64 / 1e6);
            eprintln!("gdn_prepare_ms normalize={:.3} products={:.3} inverse={:.3} solve={:.3}",
                stats.device_kernel_ns[23] as f64 / 1e6, stats.device_kernel_ns[24] as f64 / 1e6,
                stats.device_kernel_ns[25] as f64 / 1e6, stats.device_kernel_ns[26] as f64 / 1e6);
        }
        let mut next = spark_engine::greedy_token(&logits)?;
        let ttft = start.elapsed();
        let decode = Instant::now();
        let before_decode = b.stats();
        let mut output = std::io::stdout().lock();
        let mut generated = 0;
        let mut stop = "length";
        let (mut proposed, mut accepted, mut cycles) = (0, 0, 0);
        let mut history = prompt.to_vec();
        while generated < max_new {
            let i = generated;
            writeln!(output, "generated_index={i} token={next} logit={}", logits[next as usize]).map_err(|e| e.to_string())?;
            output.flush().map_err(|e| e.to_string())?;
            generated += 1;
            if !ignore_eos && next as u64 == model.config().eos_token { stop = "eos"; break; }
            if i + 1 < max_new {
                if mtp || draft_model.is_some() {
                    let entry = session.len();
                    let mut guesses = session.propose_model(&history, next, (if depth == 0 { 4 } else { depth }).min(max_new - generated - 1))?.ok_or("missing model draft")?.tokens;
                    if !ignore_eos { if let Some(at) = guesses.iter().position(|&t| t as u64 == model.config().eos_token) { guesses.truncate(at + 1); } }
                    let mut input = vec![next]; input.extend_from_slice(&guesses);
                    let rows = session.verify_mtp(&input)?;
                    proposed += guesses.len(); cycles += 1;
                    let mut kept = 1;
                    for (j, &draft) in guesses.iter().enumerate() {
                        if spark_engine::greedy_token(&rows[j])? != draft { break; }
                        accepted += 1;
                        kept += 1;
                        writeln!(output, "generated_index={} token={draft} logit={}", generated, rows[j][draft as usize]).map_err(|e| e.to_string())?;
                        generated += 1;
                        if !ignore_eos && draft as u64 == model.config().eos_token { stop = "eos"; break; }
                    }
                    session.truncate(entry + kept)?;
                    history.extend_from_slice(&input[..kept]);
                    if stop == "eos" { break; }
                    logits = rows[kept - 1].clone(); next = spark_engine::greedy_token(&logits)?;
                    continue;
                }
                b.set_profiling(profile && i == 0);
                logits = session.step(next)?; next = spark_engine::greedy_token(&logits)?;
                if profile && i == 0 {
                    let stats = b.stats().since(before_decode);
                    eprintln!("hybrid_profile kernel_ns={:?} gdn_ns={} gdn_launches={}", stats.device_kernel_ns, stats.device_kernel_ns[18], stats.device_launches[18]);
                }
            }
        }
        let seconds = decode.elapsed().as_secs_f64();
        writeln!(output, "stop={stop} generated={generated}").map_err(|e| e.to_string())?;
        eprintln!("timing prompt_tokens={} prefill_s={:.6} ttft_ms={:.3} decode_steps={} decode_ms_per_token={:.3} decode_tokens_per_sec={:.3}",
            prompt.len(), prefill.as_secs_f64(), ttft.as_secs_f64() * 1000.0, generated - 1,
            seconds * 1000.0 / (generated - 1).max(1) as f64, (generated - 1) as f64 / seconds);
        let stats = b.stats();
        eprintln!("decode_graph captures={} replays={} host_launch_ns={}", stats.graph_captures, stats.graph_launches, stats.host_graph_launch_ns);
        eprintln!("draft_depth_cycles={:?}", session.draft_depth_cycles());
        eprintln!("mtp enabled={mtp} external={} depth={depth} proposed={proposed} accepted={accepted} acceptance={:.6} cycles={cycles} mtp_weights_bytes={} mtp_kv_bytes={}", draft_model.is_some(),
            accepted as f64 / proposed.max(1) as f64,
            if mtp { model.config().mtp_weight_bytes().map_err(|e| e.to_string())? } else { 0 }, session.mtp_cache_bytes());
        let host_hwm = std::fs::read_to_string("/proc/self/status").ok().and_then(|s|
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<usize>().ok()).map(|v| v * 1024)).unwrap_or(0);
        eprintln!("memory text_weights_bytes={} kv_state_bytes={} retained_device_bytes={} peak_device_bytes={} pinned_host_bytes={} scratch_bytes={} host_rss_hwm_bytes={host_hwm}",
            model.weight_bytes() + draft_model.as_ref().map_or(0, |m| m.weight_bytes()), session.cache_bytes(), b.allocated_bytes(), b.peak_allocated_bytes(), b.pinned_host_bytes(),
            b.allocated_bytes().saturating_sub(model.weight_bytes() + draft_model.as_ref().map_or(0, |m| m.weight_bytes()) + session.cache_bytes()));
        Ok(())
    })
}
fn main() {
    #[cfg(target_os = "linux")]
    if let Err(e) = run() { eprintln!("FAIL {e}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("GPU runner requires Linux CUDA driver"); std::process::exit(1); }
}
