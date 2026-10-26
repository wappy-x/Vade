//! Flash-Next rank service: authenticated HTTP on rank zero, command workers elsewhere.
#[cfg(target_os = "linux")]
#[path = "../flash/next_service.rs"]
mod next_service;
#[cfg(target_os = "linux")]
#[path = "../flash/service.rs"]
mod service;
#[cfg(any(target_os = "linux", test))]
#[cfg_attr(all(test, not(target_os = "linux")), allow(dead_code))] // Host tests use the command ledger; startup needs Linux.
#[path = "../flash/settings.rs"]
mod settings;
#[cfg(target_os = "linux")]
#[path = "../flash/next_rank_service.rs"]
mod next_rank_service;

#[cfg(target_os = "linux")]
static STOP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
#[cfg(target_os = "linux")]
extern "C" fn stop(_: i32) { STOP.store(true, std::sync::atomic::Ordering::Relaxed); spark_gpu_probe::teardown::arm(); }

#[cfg(target_os = "linux")]
fn advise_checkpoint_cache(root: &std::path::Path, checkpoint: &engine_model::flash::Checkpoint) -> (usize, u64, usize) {
    use std::{fs::File, io, os::fd::AsRawFd, path::PathBuf};
    unsafe extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
    const POSIX_FADV_DONTNEED: i32 = 4;
    let metadata = ["config.json", "model.safetensors.index.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "hf_quant_config.json",
        "manifest.sha256", "manifest.sha256.sizes"];
    let paths = checkpoint.shards.keys().map(|name| root.join(name))
        .chain(metadata.into_iter().map(|name| root.join(name)))
        .chain(std::env::var_os("FLASH_NVFP4_MANIFEST").map(PathBuf::from))
        .collect::<std::collections::BTreeSet<_>>();
    let (mut files, mut bytes, mut errors) = (0, 0u64, 0);
    for path in paths {
        let result = (|| -> io::Result<u64> {
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            // Whole-file advice releases clean cache pages; the checkpoint is read-only.
            let error = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
            if error != 0 { return Err(io::Error::from_raw_os_error(error)); }
            Ok(len)
        })();
        match result {
            Ok(len) => { files += 1; bytes = bytes.saturating_add(len); }
            Err(error) if error.kind() == io::ErrorKind::NotFound => (), // Optional metadata.
            Err(_) => errors += 1,
        }
    }
    (files, bytes, errors)
}

/// A 0/1 environment switch, None when unset; any other value is an error.
#[cfg(target_os = "linux")]
fn switch(name: &str) -> Result<Option<bool>, String> {
    match std::env::var(name) {
        Ok(v) if v == "1" => Ok(Some(true)),
        Ok(v) if v == "0" => Ok(Some(false)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        _ => Err(format!("{name} must be 0 or 1")),
    }
}

#[cfg(target_os = "linux")]
fn prepare_model_for_install(root: &std::path::Path) -> Result<(), String> {
    use engine_model::flash::{Checkpoint, WeightFormat, manifest};
    let started = std::time::Instant::now();
    let checkpoint = Checkpoint::inspect(root).map_err(|e| e.to_string())?;
    let (hashed_bytes, reused_files) = if checkpoint.config.format == WeightFormat::Nvfp4 {
        // Verification does not execute lower-precision weights, so no inference
        // opt-in or process-wide model options are needed here.
        let verification = manifest::verify_cached(&checkpoint, root, &root.join("manifest.sha256"))
            .map_err(|e| e.to_string())?;
        (verification.hashed_bytes, verification.reused_files)
    } else {
        let plan = manifest::ReceiptPlan::open(root).map_err(|e| e.to_string())?;
        let receipt = match plan.cached().map_err(|e| e.to_string())? {
            Some(receipt) => receipt,
            None => plan.finish().map_err(|e| e.to_string())?,
        };
        (receipt.hashed_bytes, receipt.reused_files)
    };
    println!("model_verify_ms={:.3}\nmodel_hashed_bytes={hashed_bytes}\nmodel_reused_files={reused_files}",
        started.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}

#[cfg(target_os = "linux")]
fn run() -> Result<(), String> {
    use spark_engine::flash::session::{DecodeBatch, Exchange, Model};
    use spark_serve::Engine as _;
    use std::{ffi::CString, net::{SocketAddr, TcpListener}, path::PathBuf, time::{Duration, Instant}};
    spark_gpu_probe::teardown::install()?;
    let started = Instant::now();
    let server_start = std::time::SystemTime::now();
    let raw_args = std::env::args().skip(1).collect::<Vec<_>>();
    if let Some((path, model)) = settings::warm_ptx_arg(&raw_args)? {
        let ptx = CString::new(std::fs::read(path).map_err(|e| format!("PTX warm-up: {e}"))?)
            .map_err(|e| format!("PTX warm-up: {e}"))?;
        // Compilation and full-file verification are independent. Neither job
        // loads device weights, runs inference, or creates a serving listener.
        std::thread::scope(|scope| -> Result<(), String> {
            let verification = model.map(|root| scope.spawn(move || prepare_model_for_install(std::path::Path::new(root))));
            let compiled = spark_gpu_probe::driver::with_backend(0, &ptx, 64 << 20,
                |backend| backend.require_layer_kernels());
            println!("ptx_warmup_ms={:.3}", started.elapsed().as_secs_f64() * 1000.0);
            let verified = verification.map(|worker| worker.join()
                .map_err(|_| "Model verification worker stopped unexpectedly".to_string())?).transpose();
            compiled?;
            verified?;
            Ok(())
        })?;
        return Ok(());
    }
    let context_explicit = raw_args.get(6).is_some_and(|arg| arg.parse::<SocketAddr>().is_err());
    let args = settings::startup_args(raw_args);
    if args.len() < 9 {
        return Err("usage: flash-serve RANK WORLD LISTEN NEXT PTX MODEL [CONTEXT] BIND REQUESTS (0=continuous) [--allow-remote] [--api-key-file PATH] [--mtp 0..3] [--nvfp4] [--verify-full] [--fp4-local] [--indexer sparse|dense] [--tensor-map PATH]; FP4-Local defaults to context 32768, output cap 16384, up to sixteen memory-admitted chats with automatic local prompt tiles (pipeline default 2048 rows) and sparse indexing; mixed directories auto-select NextModel; world 1 strict MTP 3 plus lookup, world 2 the layer pipeline of the world-1 engine with the same MTP (VADE_W2_PP=0: the rank path; VADE_W2_TP=1 on both ranks: the FP4-Local tensor split), world 2/4 rank path plain only with --tensor-map PATH; FP8 defaults to context 131072, output cap min(65536, context), strict MTP 2 (the FP8 rank path: one chat chooses its depth each cycle up to 7; VADE_FLASH_MTP_DEPTH=1..7 sets that ceiling, VADE_MTP_ADAPT=0 or --mtp keeps the fixed depth; its drafts read an MXFP8 head copy, VADE_FP8_DRAFT_MX8=0 keeps the BF16 head); data transport uses LISTEN/NEXT ports + 1; world-1 Flash-Next defaults to C8 cohort control and healthy capture fallback (VADE_MTP_COHORT=wide, the guarded two-/four-stage pipelines' default and opt-in on world 1: 5-8 chats draft depth 3 in one 32-row verify tick, cohort control opt-in; VADE_MTP_COHORT=off restores the 16-row controller tick); all three VADE_MTP_C8/VADE_MTP_COHORT_ADAPT/VADE_POOL_GRAPH_FALLBACK=0 restore the prior automatic policy; explicit VADE_MTP_BATCH retains its table; VADE_W2_PP_C8 unset/1 enables guarded Auto8 and cohort control on the two- and four-stage pipelines, 0 restores their prior policy; explicit cohort=1 requires Auto8 and a supported route; other pipeline sizes and native ranks retain their prior defaults; invalid switch values fail at startup; VADE_C8_GUARD_ROBUST unset/1 stays on draft unless plain is 8% cheaper per token, 0 keeps the live 2% latch".into());
    }
    spark_engine::flash_prompt_options::load()?;
    spark_engine::flash_head_options::validate()?;
    spark_serve::intake_grace::load()?;
    let (mut mtp, mut nvfp4, mut verify_full) = (None, false, false);
    let mut api_key_file = None;
    let mut allow_remote = false;
    let mut fp4_local = false;
    let mut tensor_map = None;
    let mut sparse = None;
    let mut options = args[9..].iter();
    while let Some(option) = options.next() {
        match option.as_str() {
            "--mtp" if mtp.is_none() => {
                mtp = Some(settings::mtp_depth(Some(options.next().ok_or("--mtp needs a depth")?))?);
            }
            "--allow-remote" if !allow_remote => allow_remote = true,
            "--api-key-file" if api_key_file.is_none() => {
                api_key_file = Some(PathBuf::from(options.next().ok_or("--api-key-file needs a path")?));
            }
            "--nvfp4" if !nvfp4 => nvfp4 = true,
            "--verify-full" if !verify_full => verify_full = true,
            "--fp4-local" if !fp4_local => fp4_local = true,
            "--tensor-map" if tensor_map.is_none() => {
                tensor_map = Some(PathBuf::from(
                    options.next().ok_or("--tensor-map needs a path")?,
                ));
            }
            "--indexer" if sparse.is_none() => {
                sparse = Some(settings::sparse_indexer(options.next().ok_or("--indexer needs sparse or dense")?)?)
            }
            _ => return Err(format!("unknown or duplicate Flash option: {option}")),
        }
    }
    let mtp_explicit = mtp.is_some() || std::env::var_os("MTP").is_some();
    let mtp = match mtp {
        Some(depth) => depth,
        None => match std::env::var("MTP") {
            Ok(value) => settings::mtp_depth(Some(&value))?,
            Err(std::env::VarError::NotPresent) => settings::mtp_depth(None)?,
            Err(error) => return Err(error.to_string()),
        },
    };
    let api_key = spark_serve::auth::ApiKey::load(api_key_file.as_deref(), allow_remote)?;
    if nvfp4 {
        // Startup is still single-threaded; freeze the explicit CLI choice for
        // all later resident/streamed loader calls before creating any workers.
        unsafe { std::env::set_var("FLASH_NVFP4", "1"); }
    }
    if verify_full { unsafe { std::env::set_var("FLASH_VERIFY_FULL", "1"); } }
    let rank = args[0].parse::<usize>().map_err(|e| e.to_string())?;
    let phase = |name: &str, start: Instant| eprintln!("flash_startup rank={rank} phase={name} elapsed_ms={:.3}",
        start.elapsed().as_secs_f64() * 1000.0);
    let world = args[1].parse::<usize>().map_err(|e| e.to_string())?;
    if !(1..=64).contains(&world) || rank >= world { return Err("rank must be less than world in 1..=64".into()); }
    let listen = args[2].parse::<SocketAddr>().map_err(|e| e.to_string())?;
    let next = args[3].parse::<SocketAddr>().map_err(|e| e.to_string())?;
    let mut context = args[6].parse::<usize>().map_err(|e| e.to_string())?;
    if !(service::TILE..=262144).contains(&context) { return Err(spark_gpu_probe::admission::token_refusal(context as u128, 262144, &format!("CONTEXT={context}; OUTPUT not yet configured; context must be in 64..=262144"), "the service context ceiling", spark_gpu_probe::admission::FLASH)); }
    let bind = args[7].parse::<SocketAddr>().map_err(|e| e.to_string())?;
    if !bind.ip().is_loopback() && !allow_remote { return Err("remote bind requires explicit --allow-remote".into()); }
    let requests = args[8].parse::<usize>().map_err(|e| e.to_string())?;
    let root = if let Some(path) = args[5].strip_prefix("~/") {
        PathBuf::from(std::env::var_os("HOME").ok_or("missing home directory")?).join(path)
    } else { PathBuf::from(&args[5]) };
    let start = Instant::now();
    let checkpoint = engine_model::flash::Checkpoint::inspect(&root).map_err(|e| e.to_string())?;
    if nvfp4 && checkpoint.config.format != engine_model::flash::WeightFormat::Nvfp4 {
        return Err("--nvfp4 requires an NVFP4 checkpoint declared by hf_quant_config.json".into());
    }
    // World two keeps its default pipeline. Larger chains opt in with VADE_W2_PP=1;
    // world four without that opt-in keeps the tensor-map rank path. Pipeline stages run
    // world-one arithmetic; startup validates their geometry before opening any link.
    let mixed = checkpoint.config.mixed.is_some();
    let tp = settings::tensor_split(mixed, world, switch)?;
    let (pipeline, pipeline_mtp, distributed) = settings::pipeline_recipe(mixed, rank, world, switch)?;
    if pipeline { spark_engine::flash::next_batched::PpStage::from_env(rank, world)?; }
    settings::pipeline_depth(pipeline, pipeline_mtp, mtp_explicit.then_some(mtp))?;
    let next_model = settings::next_model(mixed, fp4_local, if pipeline { 0 } else { rank }, if pipeline { 1 } else { world },
        settings::startup_mtp(pipeline_mtp, mtp_explicit.then_some(mtp)))?;
    let multi = next_model && distributed;
    let pp = multi && pipeline;
    let rank_next = multi && !pp;
    if tp && !pp { return Err("VADE_W2_TP=1 needs the FP4-Local service on two connected ranks".into()); }
    let pp_mtp = pp && pipeline_mtp;
    // Parse once after route resolution, before any GPU load or peer connection.
    let experiments = spark_serve::mtp_batch::Experiments::startup_pipeline(world, pp)?;
    if pp {
        // Startup is still single-threaded and nothing has read these yet: every unset companion
        // takes its pipeline default (settings::PIPELINE_DEFAULTS), so both ranks and every reader
        // see one recipe; a set one ("0", another tile) stays as given.
        // VADE_W2_TP=1 adds its own pin (rank 1 starts at layer 2); it refuses a set split.
        let tensor = settings::tensor_pins(tp, |name| std::env::var_os(name).is_some())?;
        for (name, value) in settings::pipeline_pins(|name| std::env::var_os(name).is_some()).into_iter().chain(tensor) {
            unsafe { std::env::set_var(name, value); }
        }
    }
    let mtp = settings::serving_mtp(next_model, if pp { if pp_mtp { 1 } else { 2 } } else { world }, mtp_explicit.then_some(mtp));
    let (mtp, adaptive) = if next_model { (mtp, false) } else {
        let ceiling = match std::env::var("VADE_FLASH_MTP_DEPTH") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(format!("VADE_FLASH_MTP_DEPTH: {error}")),
        };
        let adapt = spark_serve::mtp_batch::adapt_depth()? && !spark_serve::mtp_batch::adapt_full()?;
        settings::rank_mtp(mtp, mtp_explicit, adapt, ceiling.as_deref())?
    };
    if tensor_map.is_some() && !rank_next {
        return Err(if pp {
            "--tensor-map selects the rank path: world 2 FP4-Local serves the layer pipeline unless VADE_W2_PP=0"
        } else {
            "--tensor-map requires FP4-Local world 2/4"
        }.into());
    }
    if rank_next {
        let split = std::env::var("NEXT_TP_SPLIT");
        let split = match &split {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("NEXT_TP_SPLIT must be Unicode".into()),
        };
        spark_tp::next::require_serving_split(split)?;
        let chunk = std::env::var("VADE_GDN_CHUNK");
        settings::rank_gdn_chunk(match &chunk {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(_) => return Err("VADE_GDN_CHUNK must be Unicode".into()),
        })?;
        // Startup is still single-threaded and no GDN state is sized yet.
        unsafe { std::env::set_var("VADE_GDN_CHUNK", "0"); }
        if [listen, next].iter().any(|address| matches!(address.port(), 0 | 65535)) {
            return Err("control ports must leave room for the data port".into());
        }
        if bind.port() == 0 { return Err("HTTP bind port must be nonzero".into()); }
        if tensor_map.is_none() {
            return Err("FP4-Local world 2/4 requires --tensor-map PATH".into());
        }
        if sparse == Some(false) {
            return Err("FP4-Local world 2/4 requires --indexer sparse".into());
        }
        if context_explicit && context != spark_tp::next_http::CONTEXT {
            return Err("FP4-Local world 2/4 CONTEXT must be 32768".into());
        }
    }
    if next_model && !context_explicit { context = 32768; }
    if !next_model && sparse.is_some() { return Err("--indexer requires the FP4-Local service".into()); }
    // The mixed checkpoint contract is also the explicit precision selection.
    // Startup remains single-threaded here, before the HTTP/tokenizer workers.
    if next_model { unsafe { std::env::set_var("FLASH_NVFP4", "1"); } }
    if !next_model { spark_engine::flash::require_inference(checkpoint.config.format)?; }
    phase("checkpoint_inspect", start);
    if context > checkpoint.config.max_positions as usize { return Err(spark_gpu_probe::admission::token_refusal(context as u128, checkpoint.config.max_positions as usize, &format!("CONTEXT={context}; OUTPUT not yet configured"), "the checkpoint position ceiling", spark_gpu_probe::admission::FLASH)); }
    let vocab = checkpoint.config.vocab as usize;
    let generation = settings::Generation::open(&root, checkpoint.config.eos_token as u32, vocab)?;
    let mut limits = spark_serve::Limits { api_key, context_tokens: context, input_tokens: context - 1,
        output_tokens: context.min(if next_model { 16384 } else { 65536 }), body_bytes: 4 * 1024 * 1024,
        default_sampling: Some(generation.sampling), greedy_if_unsampled: next_model,
        ..Default::default() };
    if rank_next {
        let full = std::env::var("FLASH_VERIFY_FULL");
        let full = match &full {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error.to_string()),
        };
        settings::rank_limits(&mut limits, full)?;
    }
    if let Ok(value) = std::env::var("FLASH_SERVE_OUTPUT_TOKENS") {
        limits.output_tokens = value.parse().map_err(|_| "invalid output token limit")?;
        if limits.output_tokens > context { return Err(spark_gpu_probe::admission::token_refusal(limits.output_tokens as u128, context, &format!("CONTEXT={context}; OUTPUT={}", limits.output_tokens), "configured CONTEXT", "OUTPUT / FLASH_SERVE_OUTPUT_TOKENS")); }
    }
    if next_model && limits.output_tokens > 16384 { return Err("FP4-Local OUTPUT must be at most 16384 tokens".into()); }
    limits.output_bytes = limits.output_bytes.max(limits.output_tokens.saturating_mul(256).saturating_add(4096).min(16 * 1024 * 1024));
    if let Ok(value) = std::env::var("FLASH_SERVE_GENERATION_TIMEOUT_MS") {
        limits.generation_timeout = Duration::from_millis(value.parse().map_err(|_| "invalid generation timeout")?);
    }
    limits.ab_ptx = std::env::var_os("VADE_AB_PTX").is_some();
    limits.ab_cohorts = match std::env::var("VADE_AB_COHORTS").as_deref() {
        Ok("1") if limits.ab_ptx => true,
        Ok("1") => return Err("VADE_AB_COHORTS=1 requires VADE_AB_PTX".into()),
        Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("VADE_AB_COHORTS must be 0 or 1".into()),
    };
    limits.validate()?;
    let start = Instant::now();
    let tokenizer = if rank == 0 { Some(settings::tokenizer(&root, &limits)?) } else { None };
    phase("tokenizer", start);
    let listener = if rank == 0 { Some(TcpListener::bind(bind).map_err(|e| e.to_string())?) } else { None };
    if next_model {
        if pp {
            return next_service::run_pp(rank, world, listen, next, &root, &args[4], listener, tokenizer, limits,
                generation.eos, requests, server_start, &STOP, sparse.unwrap_or(true), mtp, checkpoint, experiments);
        }
        if rank_next {
            drop(checkpoint);
            return next_rank_service::run(next_rank_service::Startup {
                root: &root,
                tensor_map: tensor_map.as_deref().ok_or("missing tensor map")?,
                ptx_path: &args[4],
                rank_args: next_rank_service::Rank {
                    rank,
                    world,
                    listen,
                    next,
                },
                listener,
                tokenizer,
                limits,
                eos: generation.eos,
                requests,
                server_start,
                shutdown: &STOP,
            });
        }
        return next_service::run(&root, &args[4], listener.ok_or("missing rank-zero listener")?,
            tokenizer.ok_or("missing rank-zero tokenizer")?, limits,
            generation.eos, requests, server_start, &STOP, sparse.unwrap_or(true), mtp, checkpoint, experiments);
    }
    let pool_request = spark_tp::flash_pool::Request::from_env()?;
    let path = spark_engine::flash::path_topology(world)?;
    let run_id = std::env::var("TP_RUN_ID").map_err(|_| "set the same fresh TP_RUN_ID on every rank")?;
    let start = Instant::now();
    let ptx = CString::new(std::fs::read(&args[4]).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let ab_ptx = spark_gpu_probe::driver::ab_ptx_from_env()?;
    phase("ptx_read", start);
    let start = Instant::now();
    let mut commands = spark_tp::transport::Ring::connect_options(rank, world, listen, next, path)?;
    phase("command_connect", start);
    let start = Instant::now();
    spark_tp::runtime::verify_artifacts_with_ab(&mut commands, &ptx, &root, ab_ptx.as_deref())?;
    phase("artifact_handshake", start);
    let data_address = |mut address: SocketAddr| -> Result<SocketAddr, String> {
        address.set_port(address.port().checked_add(1).ok_or("data port overflow")?);
        Ok(address)
    };
    let start = Instant::now();
    let mut data = if path {
        spark_seqpar::Ring::connect_path(rank, world, data_address(listen)?, data_address(next)?, &run_id)?
    } else { spark_seqpar::Ring::connect(rank, world, data_address(listen)?, data_address(next)?, &run_id)? };
    let activation = spark_engine::flash::moe::Moe::activation_quantization()?;
    let sampled = match std::env::var("FLASH_MTP_SAMPLED").as_deref() {
        Ok("1") => true, Ok("0") | Err(std::env::VarError::NotPresent) => false,
        _ => return Err("FLASH_MTP_SAMPLED must be 0 or 1".into()),
    };
    let mut precision = vec![0; world * 2];
    precision[rank * 2..rank * 2 + 2].copy_from_slice(&[u8::from(activation) | (u8::from(sampled) << 2), mtp as u8]);
    data.all_gather_bytes(&mut precision, &(0..=world).map(|r| r * 2).collect::<Vec<_>>())?;
    if precision.chunks_exact(2).any(|value| value != &precision[..2]) { return Err("Flash ranks disagree on projection precision, MTP depth or sampled verification".into()); }
    data.enable_mailbox(8 << 20)?;
    data.seal_registrations()?;
    phase("data_connect_precision_mailbox", start);
    let mailbox_bytes = data.mailbox_bytes();
    let mailbox = data.mailbox_memory().is_some();
    eprintln!("flash_serve transport rank={rank} world={world} rdma={} transport_mailbox_bytes={mailbox_bytes}", data.uses_rdma());
    let explicit_budget = match std::env::var("SEQPAR_MEMORY_BYTES") {
        Ok(value) => Some(value.parse::<usize>().map_err(|_| "invalid SEQPAR_MEMORY_BYTES")?),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.to_string()),
    };
    if explicit_budget == Some(0) { return Err("SEQPAR_MEMORY_BYTES must be positive".into()); }
    spark_gpu_probe::driver::with_backend(0, &ptx, explicit_budget.unwrap_or(usize::MAX), |backend| {
        if let Some(ptx) = &ab_ptx { backend.load_ab_ptx(ptx)?; }
        // Query after context/module setup. Reclaimable checkpoint pages fund
        // both residency and allocations through this one bounded ceiling.
        let cuda_free_before = backend.device_memory_info()?.0;
        let (files, bytes, errors) = advise_checkpoint_cache(&root, &checkpoint);
        let bootstrap_free = backend.available_memory_bytes()?;
        let bootstrap_cuda_free = backend.device_memory_info()?.0;
        eprintln!("flash_serve checkpoint_cache rank={rank} files={files} bytes_advised={bytes} errors={errors} cuda_free_before={cuda_free_before} cuda_free_after={bootstrap_cuda_free}");
        let bootstrap_safety = (bootstrap_free / 10).max(2 << 30);
        let budget = explicit_budget.unwrap_or(usize::MAX)
            .min(bootstrap_free.saturating_sub(bootstrap_safety));
        backend.set_allocation_budget(budget)?;
        let resident = Model::rank_will_be_resident(&checkpoint, rank, world, context, service::TILE, mtp,
            budget.saturating_sub(backend.allocated_bytes()))?;
        let mut residency = vec![0; world];
        residency[rank] = u8::from(resident);
        data.all_gather_bytes(&mut residency, &(0..=world).collect::<Vec<_>>())?;
        if residency.iter().any(|&value| value != residency[0]) {
            let mut budgets = vec![0; world * 8]; budgets[rank * 8..rank * 8 + 8].copy_from_slice(&(budget.saturating_sub(backend.allocated_bytes()) as u64).to_le_bytes());
            data.all_gather_bytes(&mut budgets, &(0..=world).map(|r| r * 8).collect::<Vec<_>>())?;
            let constrained = residency.iter().position(|&value| value == 0).unwrap();
            return Err(format!("{}\nRanks disagree on residency; constrained rank={constrained}; local rank {rank} inputs: CUDA free={bootstrap_cuda_free} bytes; unified available={bootstrap_free} bytes; safety={bootstrap_safety} bytes; explicit ceiling={explicit_budget:?}.", Model::rank_residency_refusal(&checkpoint, constrained, world, context, service::TILE, mtp, u64::from_le_bytes(budgets[constrained * 8..constrained * 8 + 8].try_into().unwrap()) as usize, limits.output_tokens)?));
        }
        if mtp != 0 && !resident { return Err(format!("{}\nMTP needs resident weights; rank {rank} inputs: CUDA free={bootstrap_cuda_free} bytes; unified available={bootstrap_free} bytes; safety={bootstrap_safety} bytes; explicit ceiling={explicit_budget:?}.", Model::rank_residency_refusal(&checkpoint, rank, world, context, service::TILE, mtp, budget.saturating_sub(backend.allocated_bytes()), limits.output_tokens)?)); }
        eprintln!("flash_serve bootstrap rank={rank} cuda_free_bytes={bootstrap_cuda_free} unified_available_bytes={bootstrap_free} safety_reserve_bytes={bootstrap_safety} budget_bytes={budget} explicit_budget_bytes={} resident={resident}",
            explicit_budget.unwrap_or(0));
        let start = Instant::now();
        let model = Model::load_rank_with_mtp(backend, &root, checkpoint, rank, world, context, service::TILE, mtp)?;
        phase("model_load", start);
        let start = Instant::now();
        data.barrier_with_timeout(Duration::from_secs(600))?;
        residency[rank] = u8::from(model.is_resident());
        data.all_gather_bytes(&mut residency, &(0..=world).collect::<Vec<_>>())?;
        if residency.iter().any(|&value| value != u8::from(resident)) {
            return Err("Flash loaded weight residency differs from the agreed bootstrap plan".into());
        }
        phase("model_load_barrier", start);
        let data = service::Data::new(data);
        let start = Instant::now();
        // Negotiate bytes first so every rank funds identical active and cached
        // lifetimes, including ancestors retained by an in-flight continuation.
        let execution = model.session_reservation_bytes(context, service::TILE, mailbox);
        let prompt_growth = model.prepare_prompt(backend, context, service::TILE, mailbox)?;
        let draft_reservation = model.draft_reservation_bytes(context)?;
        let allocated = backend.allocated_bytes();
        let free = backend.available_memory_bytes()?;
        let cuda_free = backend.device_memory_info()?.0;
        let shared_buffers = prompt_growth.checked_add(DecodeBatch::reservation_bytes(&model))
            .and_then(|n| n.checked_add(draft_reservation)).ok_or("Flash workspace byte count overflow")?;
        let shared_graphs = DecodeBatch::graph_reservation_bytes(&model)
            .checked_add(spark_gpu_probe::driver::DecodeGraph::MAX_RESERVATION_BYTES)
            .and_then(|n| n.checked_add(model.graph_draft_reservation_bytes()))
            .ok_or("Flash graph byte count overflow")?;
        let shared_graphs = if limits.ab_ptx { shared_graphs.checked_mul(2).ok_or("A/B graph byte count overflow")? } else { shared_graphs };
        let shared = shared_buffers.checked_add(shared_graphs).ok_or("Flash shared byte count overflow")?;
        // Bootstrap safety protected loading. Re-budget against the measured
        // remaining DRAM now; retain an explicit ceiling for the whole service.
        let (safety, local_available) = spark_tp::flash_pool::capacity_bytes(free, allocated, shared, explicit_budget);
        let mut limits_wire = vec![0; world * 56];
        for (slot, value) in limits_wire[rank * 56..(rank + 1) * 56].chunks_exact_mut(8)
            .zip([local_available, execution, model.workspace_slots(mailbox),
                pool_request.chats.unwrap_or(0), pool_request.cache_bytes.unwrap_or(0),
                usize::from(pool_request.cache_bytes.is_some()), context]) {
            slot.copy_from_slice(&(value as u64).to_le_bytes());
        }
        data.clone().gather_bytes(&mut limits_wire, &(0..=world).map(|r| r * 56).collect::<Vec<_>>())?;
        if limits_wire.chunks_exact(56).any(|v| v[24..] != limits_wire[24..56]) {
            return Err("Flash ranks disagree on context, requested chats or retained-cache override".into());
        }
        let limit = |offset| limits_wire.chunks_exact(56)
            .map(move |v| u64::from_le_bytes(v[offset..offset + 8].try_into().unwrap()) as usize);
        let available = limit(0).min().unwrap();
        // KV heads can be replicated unevenly (world3 owns 1/2/1). Every
        // worker and the root scheduler must fund the most expensive rank.
        let execution = limit(8).max().unwrap();
        let workspace_slots = limit(16).max().unwrap();
        let plan = pool_request.plan(context, available, |tokens, caching|
            model.request_reservation_bytes(tokens, service::TILE, workspace_slots, caching),
            |tokens| service::publication_metadata_bytes(tokens, vocab, limits.header_bytes))
            .map_err(|error| if error.starts_with("Requested size does not fit.") { format!("{error}\nRank {rank} inputs: weights/rank={} bytes; OUTPUT={}; CUDA free={cuda_free} bytes; unified available={free} bytes; allocated={allocated} bytes; shared buffers={shared_buffers} bytes; shared graphs={shared_graphs} bytes; safety={safety} bytes; local pool={local_available} bytes; negotiated pool={available} bytes; explicit ceiling={explicit_budget:?}.", model.resident_bytes(), limits.output_tokens) } else { error })
            // Preserve the actionable refusal before resident CUDA cleanup;
            // the teardown deadline may end a stuck driver before main runs.
            .inspect_err(|error| eprintln!("FAIL {error}"))?;
        let (pool, cache_bytes, reservation, minimum, slots) =
            (plan.pool_bytes, plan.cache_bytes, plan.reservation, plan.minimum, plan.slots);
        let full = reservation + plan.publication_bytes;
        let allocation_budget = allocated.checked_add(shared).and_then(|n| n.checked_add(pool))
            .ok_or("Flash allocation byte count overflow")?;
        backend.set_allocation_budget(allocation_budget)?;
        let graph_budget = model.graph_session_reservation_bytes().checked_mul(slots)
            .and_then(|n| n.checked_add(shared_graphs)).ok_or("Flash retained graph byte count overflow")?;
        backend.set_graph_storage_budget(graph_budget)?;
        let pool_policy = if pool_request.chats.is_none() { "largest_safe" } else { "explicit_chats" };
        eprintln!("flash_serve capacity rank={rank} cuda_free_bytes={cuda_free} unified_available_bytes={free} safety_reserve_bytes={safety} explicit_budget_bytes={} allocated_bytes={allocated} shared_buffer_reservation_bytes={shared_buffers} shared_graph_reservation_bytes={shared_graphs} local_available_bytes={local_available} transient_reserve_bytes={} available_bytes={available} pool_policy={pool_policy} session_pool_bytes={pool}",
            explicit_budget.unwrap_or(0), spark_serve::budget::TRANSIENT_RESERVE);
        eprintln!("flash_serve budget rank={rank} budget_bytes={allocation_budget} initial_budget_bytes={budget} allocated_bytes={allocated} free_after_load_bytes={free} safety_reserve_bytes={safety} prompt_growth_bytes={prompt_growth} decode_batch_reservation_bytes={} shared_draft_reservation_bytes={draft_reservation} shared_draft_graph_reservation_bytes={} shared_graph_reservation_bytes={shared_graphs} session_graph_reservation_bytes={} graph_budget_bytes={graph_budget} available_bytes={available} minimum_session_reservation_bytes={minimum} kv_budget_bytes={cache_bytes} requested_chats={} full_context_slots={} resident={}",
            DecodeBatch::reservation_bytes(&model), model.graph_draft_reservation_bytes(), model.graph_session_reservation_bytes(), pool_request.chats.unwrap_or(0), (pool / full).min(slots), model.is_resident());
        // Cache retention is elastic. The scheduler reclaims it before each
        // admission so live reservations plus unfunded cache stay within pool.
        // Admission charges each request's token bound and reclaims idle cache.
        // All active sessions reuse one fixed-width decode workspace in turn.
        eprintln!("flash_serve limits rank={rank} available_bytes={available} execution_reservation_bytes={execution} session_reservation_bytes={reservation} publication_metadata_bytes={} slots={slots} session_pool_bytes={pool} kv_budget_bytes={cache_bytes} shared_draft_reservation_bytes={draft_reservation}", plan.publication_bytes);
        phase("admission_negotiate", start);
        let start = Instant::now();
        commands.barrier()?;
        phase("ready_barrier", start);
        let start = Instant::now();
        let result = if rank != 0 { service::worker(&model, backend, commands, data, context, vocab, slots, cache_bytes != 0, started) }
        else {
            let engine = service::Engine::new(&model, backend, commands, data, context, vocab, generation.eos, slots, cache_bytes, reservation, workspace_slots, adaptive)?;
            let reservation = engine.session_reservation_bytes().ok_or("missing Flash reservation")?;
            let cache = spark_serve::cache::CacheLimits { entries: 128, retained_bytes: cache_bytes,
                tokens_per_entry: context, logits_per_entry: vocab, idle_ttl: Duration::from_secs(3600) };
            // Queue headroom follows memory-backed slots; it does not admit extra sessions.
            let schedule = spark_serve::scheduler::Config { active: slots, connections: slots.saturating_mul(2).max(64).min(slots.max(128)), prefill_chunk: 512,
                foreground_rows: slots.max(mtp + 1), background_rows: slots.max(mtp + 1), active_bytes: pool, session_budget_bytes: pool,
                draft: if mtp == 0 { spark_serve::protocol::DraftMode::Plain }
                    else if sampled { spark_serve::protocol::DraftMode::Sampled }
                    else { spark_serve::protocol::DraftMode::Strict }, mtp_depth: mtp, draft_global_bank: false, prefix_cache: cache_bytes != 0,
                sharing: spark_serve::scheduler::Sharing::default() };
            unsafe extern "C" { fn signal(number: i32, handler: extern "C" fn(i32)) -> usize; }
            // The signal handler only writes an atomic. The engine thread closes
            // every session, then sends STOP before releasing resident weights.
            unsafe { signal(2, stop); signal(15, stop); }
            let name = root.file_name().ok_or("checkpoint needs a model name")?.to_string_lossy().to_ascii_lowercase();

            // Workers are now parked on commands, outside the connection/load deadlines.
            let served = spark_serve::contract::Identity::load(&root).and_then(|identity| {
                limits.identity = Some(identity);
                limits.metadata = Some(spark_serve::metadata::Metadata::load(&root, &ptx,
                    mtp, world, if path { "path" } else { "ring" }, server_start)?);
                backend.synchronize()?;
                phase("controller_setup_identity", start);
                eprintln!("flash_serve ready startup_ms={:.3} peak_pinned_staging_bytes={} transport_mailbox_bytes={mailbox_bytes} device_bytes={} device_peak_bytes={} host_rss_bytes={} process_hwm_bytes={} graph_captures={} mtp_depth={mtp} bind={bind} model={name} world={world} context={context} model_bytes={} session_reservation_bytes={reservation} slots={slots} session_pool_bytes={pool} kv_budget_bytes={cache_bytes} decode_batch_reservation_bytes={} checkpoint_temperature={} checkpoint_top_p={} checkpoint_top_k={}",
                    started.elapsed().as_secs_f64() * 1000.0, backend.peak_pinned_staging_bytes(), backend.allocated_bytes(), backend.peak_allocated_bytes(),
                    spark_tp::runtime::host_memory().0, spark_tp::runtime::host_memory().1, backend.stats().graph_captures,
                    model.resident_bytes(), DecodeBatch::reservation_bytes(&model), generation.sampling.temperature, generation.sampling.top_p, generation.sampling.top_k);
                spark_serve::scheduler::serve_requests(listener.unwrap(), &engine, tokenizer.as_ref(),
                    &limits, &cache, &schedule, &STOP, &name, requests)
            });
            if served.is_err() { spark_gpu_probe::teardown::arm(); }
            let stopped = engine.shutdown();
            served.and(stopped)
        };
        let remaining = backend.available_memory_bytes().map(|bytes| bytes.to_string())
            .unwrap_or_else(|_| "unavailable".into());
        eprintln!("flash_serve exit rank={rank} model_bytes={} device_bytes={} device_peak_bytes={} graph_storage_bound_bytes={} graph_storage_peak_bound_bytes={} accounted_peak_bytes={} unified_available_bytes={remaining} pinned_bytes={} host_heap_peak_bytes={} process_hwm_bytes={}",
            model.resident_bytes(), backend.allocated_bytes(), backend.peak_allocated_bytes(),
            backend.graph_storage_bytes(), backend.peak_graph_storage_bytes(), backend.peak_accounted_bytes(),
            backend.pinned_host_bytes(), spark_tp::runtime::heap_peak(), spark_tp::runtime::host_memory().1);
        result
    })
}

fn main() {
    #[cfg(target_os = "linux")]
    if let Err(error) = run() { eprintln!("FAIL {error}"); std::process::exit(1); }
    #[cfg(not(target_os = "linux"))]
    { eprintln!("Flash serving requires the Linux CUDA driver"); std::process::exit(1); }
}

#[global_allocator]
static ALLOCATOR: spark_tp::runtime::TrackedAllocator = spark_tp::runtime::TrackedAllocator;
