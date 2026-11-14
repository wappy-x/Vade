#[cfg(target_os = "linux")]
#[path = "spark-serve/gemma4.rs"]
mod gemma4;
// Start the real GPU service on loopback after checkpoint/tokenizer load.
#[cfg(target_os = "linux")]
#[path = "spark-serve/lifecycle.rs"]
mod lifecycle;
#[path = "spark-serve/cartridges.rs"]
mod cartridges;
#[cfg(target_os = "linux")]
#[allow(clippy::unnecessary_unwrap)] // the borrow shape stays as written.
fn run() -> Result<(), String> {
    use spark_serve::{
        Engine, Limits,
        budget::{Admission, Sizing, parse_mib},
        cache::CacheLimits,
        qwen::{Qwen, Seqpar},
        scheduler,
    };
    use spark_gpu_probe::admission::{refusal, bytes, CUDA, SERVE};
    use std::{
        ffi::{CStr, CString},
        io::Read,
        net::{IpAddr, SocketAddr, TcpListener},
        path::{Path, PathBuf},
        sync::atomic::{AtomicBool, Ordering},
        time::Instant,
    };
    let started = Instant::now();
    let server_start = std::time::SystemTime::now();
    let mut args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "--list") {
        if args.len() != 2 { return Err("usage: spark-serve --list MODELS_ROOT".into()); }
        return cartridges::list(Path::new(&args[1]));
    }
    let daemon = args.iter().any(|arg| arg == "--daemon");
    args.retain(|arg| arg != "--daemon");
    if args.is_empty() || args[0].starts_with("--") {
        eprintln!("draft options: --draft-depth 0..4 (hybrid default 3); --mtp loads built-in hybrid draft; auto discovers sibling Qwen3-0.6B for dense models; --draft-model CHECKPOINT overrides; --draft plain disables; --draft-k 1..8 (ceiling, hybrid MTP default 3); HYBRID_MTP_COPY/GATE default 1, set 0 to disable; --draft-global-bank explicitly enables cross-session proposals");
        eprintln!("prefix reuse: --prefix-cache on|off (default on), --prefix-cache-scope global|api-key (default api-key), --report-prefix-reuse (opt-in response header)");
        eprintln!("thinking: --default-thinking on|off (default: the chat template's own default) for chat and Responses requests that set neither enable_thinking nor a reasoning effort");
        eprintln!("sharing: --allow-host AUTHORITY / --allow-origin ORIGIN extend the allow-lists; --admin-socket PATH enables model swaps on a private Unix socket; --client-priority-cap 0..254 (default 0); --owner-key-file PATH (0600, 64 hex digits) reserves owner priority");
        return Err("usage: spark-serve --list MODELS_ROOT | spark-serve [PTX] CHECKPOINT_ROOT [--daemon] [--models-root PATH] [--draft auto|plain|strict|fast] [--bind IP:PORT] [--allow-remote] [--api-key-file PATH] [--context-tokens N] [--rope-scaling yarn] [--input-tokens N] [--output-tokens N] [--device N] [--gpu-budget-mib N] [--kv-budget-mib N] [--session-budget-mib N] [--max-active N] [--prefill-chunk N] [--generation-timeout-secs N] [--output-bytes N] [--seqpar-peer CONTROL,DATA]... [--seqpar-listen IP:PORT] [--seqpar-threshold N]; PTX defaults to embedded kernels; default loopback:8080, native context limited by free GPU memory (up to 32 active), output=remaining context, YaRN off by default, automatic STRICT/plain, seeded sampling; generation deadline 1..3600 seconds, default and maximum 3600, starts at admission (queue wait has a separate bound of the same length); expiry is fatal HTTP 504 or a terminal stream error even below the output token limit; output bytes auto-size from output tokens within 1..16MiB unless explicitly set; --daemon detaches with PID/log files under XDG_STATE_HOME/vade (default ~/.local/state/vade)".into());
    }
    // Preserve the original PTX CHECKPOINT_ROOT form as an explicit override.
    let external_ptx = args.get(1).is_some_and(|arg| !arg.starts_with("--"));
    let checkpoint_arg = usize::from(external_ptx);
    let mut limits = Limits::default();
    let mut output_bytes_explicit = false;
    let mut input_tokens_explicit = false;
    let mut output_tokens_explicit = false;
    let mut thinking_explicit = false;
    let mut yarn = false;
    let mut bind: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let mut allow_remote = false;
    let mut api_key_file = None;
    let mut models_root: Option<PathBuf> = None;
    let mut device: i32 = 0;
    let mut sizing = Sizing::default();
    let mut prefill_chunk = 512;
    let mut prefix_cache = true;
    let mut sharing = scheduler::Sharing::default();
    let mut admin_socket: Option<PathBuf> = None;
    let mut foreground_rows = 8;
    let mut background_rows = None;
    let mut draft = spark_serve::protocol::DraftMode::Auto;
    let mut draft_global_bank = false;
    let mut draft_model: Option<PathBuf> = None;
    let mut draft_k: Option<usize> = None;
    let mut mtp = false;
    let mut hybrid_depth = 3;
    let mut seqpar_peers = Vec::new();
    let mut seqpar_listen = None;
    let mut seqpar_threshold = 6144;
    let mut i = checkpoint_arg + 1;
    while i < args.len() {
        if args[i] == "--draft-global-bank" {
            draft_global_bank = true;
            i += 1;
            continue;
        }
        if args[i] == "--mtp" { mtp = true; i += 1; continue; }
        if args[i] == "--allow-remote" {
            allow_remote = true;
            i += 1;
            continue;
        }
        if args[i] == "--report-prefix-reuse" {
            sharing.report_prefix_reuse = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or("missing CLI option value")?;
        match args[i].as_str() {
            "--api-key-file" => {
                if api_key_file.is_some() { return Err("duplicate --api-key-file".into()); }
                api_key_file = Some(PathBuf::from(value));
            }
            "--prefix-cache" => prefix_cache = match value.as_str() {
                "on" => true, "off" => false,
                _ => return Err("--prefix-cache must be on or off".into()),
            },
            "--default-thinking" => {
                thinking_explicit = true;
                limits.default_thinking = match value.as_str() {
                    "on" => true, "off" => false,
                    _ => return Err("--default-thinking must be on or off".into()),
                };
            },
            "--prefix-cache-scope" => sharing.prefix_cache_per_api_key = match value.as_str() {
                "global" => false, "api-key" => true,
                _ => return Err("--prefix-cache-scope must be global or api-key".into()),
            },
            "--allow-host" => limits.allowed_hosts.push(value.clone()),
            "--allow-origin" => limits.allowed_origins.push(value.clone()),
            "--admin-socket" => admin_socket = Some(PathBuf::from(value)),
            "--owner-key-file" => sharing.load_owner_key(Path::new(value))?,
            "--client-priority-cap" => sharing.client_priority_cap = value.parse()
                .map_err(|_| "--client-priority-cap must be in 0..254")?,
            "--draft-depth" => { hybrid_depth = value.parse::<usize>().map_err(|_| "invalid hybrid draft depth")?; if hybrid_depth > 4 { return Err("--draft-depth must be 0 (adaptive) or 1..=4".into()); } }
            "--draft" => draft = spark_serve::protocol::DraftMode::parse(value)?,
            "--draft-model" => draft_model = Some(PathBuf::from(value)),
            "--draft-k" => draft_k = Some(value.parse().map_err(|_| "invalid draft K")?),
            "--rope-scaling" => {
                if value != "yarn" { return Err("--rope-scaling must be yarn (official Qwen3 factor 4, 131072 positions)".into()); }
                yarn = true;
            }
            "--seqpar-peer" => {
                let (control, data) = value.split_once(',').ok_or("--seqpar-peer requires CONTROL,DATA socket addresses")?;
                seqpar_peers.push((control.parse().map_err(|_| "invalid seqpar control address")?,
                    data.parse().map_err(|_| "invalid seqpar data address")?));
            }
            "--seqpar-listen" => seqpar_listen = Some(value.parse().map_err(|_| "invalid seqpar listen address")?),
            "--seqpar-threshold" => seqpar_threshold = value.parse().map_err(|_| "invalid seqpar threshold")?,
            "--foreground-rows" => foreground_rows = value.parse().map_err(|_| "invalid foreground rows")?,
            "--background-rows" => background_rows = Some(value.parse().map_err(|_| "invalid background rows")?),
            "--bind" => bind = value.parse().map_err(|_| "invalid bind address")?,
            "--models-root" => models_root = Some(PathBuf::from(value)),
            "--context-tokens" => {
                sizing.context_tokens = Some(value.parse().map_err(|_| "invalid context token limit")?);
            }
            "--input-tokens" => {
                limits.input_tokens = value.parse().map_err(|_| "invalid input token limit")?;
                input_tokens_explicit = true;
            }
            "--output-tokens" => {
                limits.output_tokens = value.parse().map_err(|_| "invalid output token limit")?;
                output_tokens_explicit = true;
            }
            "--device" => device = value.parse().map_err(|_| "invalid device ordinal")?,
            "--gpu-budget-mib" => {
                sizing.gpu_bytes = Some(parse_mib(value, "--gpu-budget-mib")?);
            }
            "--kv-budget-mib" => sizing.kv_bytes = Some(parse_mib(value, "--kv-budget-mib")?),
            "--session-budget-mib" => sizing.session_bytes = Some(parse_mib(value, "--session-budget-mib")?),
            "--max-active" => {
                sizing.max_active = Some(value.parse().map_err(|_| "invalid active session count")?);
            }
            "--prefill-chunk" => {
                prefill_chunk = value.parse().map_err(|_| "invalid prefill chunk")?
            }
            "--generation-timeout-secs" => {
                limits.generation_timeout = std::time::Duration::from_secs(
                    value.parse().map_err(|_| "invalid generation timeout")?,
                )
            }
            "--output-bytes" => {
                limits.output_bytes = value.parse().map_err(|_| "invalid output byte limit")?;
                output_bytes_explicit = true;
            }
            _ => return Err(format!("unknown CLI option: {}", args[i])),
        }
        i += 2;
    }
    limits.api_key = spark_serve::auth::ApiKey::load(api_key_file.as_deref(), allow_remote)?
        .map(|key| key.with_owner(sharing.owner_key));
    if seqpar_threshold == 0 { return Err("--seqpar-threshold must be positive".into()); }
    if sharing.client_priority_cap == 255 { return Err("--client-priority-cap must be in 0..254".into()); }
    if allow_remote && !sharing.prefix_cache_per_api_key {
        eprintln!("WARNING: global prefix-cache scope with --allow-remote shares cached prompts across clients; use api-key scope for isolation");
    }
    if draft_k.is_some_and(|k| !(1..=8).contains(&k)) { return Err("--draft-k must be in 1..8".into()); }
    if seqpar_peers.is_empty() != seqpar_listen.is_none() {
        return Err("--seqpar-peer and --seqpar-listen must be supplied together".into());
    }
    if !(1..=512).contains(&prefill_chunk) {
        return Err("--prefill-chunk must be in 1..512".into());
    }
    if !matches!(bind.ip(), IpAddr::V4(ip) if ip.is_loopback())
        && !matches!(bind.ip(), IpAddr::V6(ip) if ip.is_loopback())
        && !allow_remote
    {
        return Err(
            "remote bind requires explicit --allow-remote"
                .into(),
        );
    }
    if device < 0 {
        return Err("invalid device or GPU budget".into());
    }
    let mut checkpoint = Path::new(&args[checkpoint_arg]).canonicalize().map_err(|e| e.to_string())?;
    let cartridge = engine_model::cartridge::Cartridge::preflight(&checkpoint).map_err(|e| e.to_string())?;
    eprintln!("cartridge architecture={} precision={} weight_bytes={} supported=yes",
        cartridge.architecture, cartridge.precision, cartridge.weight_bytes);
    // Persistent workers retain the startup checkpoint across local model swaps.
    let seqpar_checkpoint = checkpoint.clone();
    let root = models_root.unwrap_or_else(|| checkpoint.parent().unwrap_or(&checkpoint).to_owned())
        .canonicalize().map_err(|e| e.to_string())?;
    if !checkpoint.starts_with(&root) { return Err("checkpoint is outside --models-root".into()); }
    let hybrid = cartridge.kind == engine_model::cartridge::Kind::Qwen3_5;
    let hybrid_plan = if hybrid {
        let inspected = engine_model::plan::ModelPlan::inspect_parts(&checkpoint).map_err(|e| e.to_string())?;
        Some(inspected)
    } else { None };
    let (mut draft_model, draft_source) = if cartridge.gemma4.is_some() {
        if !seqpar_peers.is_empty() { return Err("Gemma text requires local generation".into()); }
        if draft == spark_serve::protocol::DraftMode::Fast { return Err("Gemma assistant supports strict verification only".into()); }
        let assistant = if draft == spark_serve::protocol::DraftMode::Plain { None } else {
            let selected = draft_model.or_else(|| {
                let mut name = checkpoint.file_name()?.to_os_string();
                name.push("-assistant");
                let path = checkpoint.with_file_name(name);
                path.is_dir().then_some(path)
            });
            selected.map(|p| p.canonicalize().map_err(|e| e.to_string())).transpose()?
        };
        if let Some(path) = &assistant {
            if !path.starts_with(&root) { return Err("assistant is outside --models-root".into()); }
            let config = spark_engine::gemma4::Assistant::open_config(path)?;
            config.validate_target(cartridge.gemma4.as_ref().unwrap()).map_err(|e| e.to_string())?;
            let vocabulary = |dir: &Path| -> Result<engine_model::json::Value, String> {
                let bytes = engine_model::safetensors::read_bounded(&dir.join("tokenizer.json"), 32*1024*1024).map_err(|e| e.to_string())?;
                let doc = engine_model::json::parse(&bytes, engine_model::json::Limits {
                    bytes: 32*1024*1024, nodes: 2_000_000, ..Default::default()
                }).map_err(|e| e.to_string())?;
                let model = engine_model::json::field(doc.object().map_err(|e| e.to_string())?, "model").map_err(|e| e.to_string())?;
                engine_model::json::field(model.object().map_err(|e| e.to_string())?, "vocab")
                    .cloned().map_err(|e| e.to_string())
            };
            if vocabulary(&checkpoint)? != vocabulary(path)? { return Err("assistant vocabulary differs from target".into()); }
        }
        let source = if assistant.is_some() {"gemma-official-assistant"} else {"gemma-plain"};
        (assistant, source)
    } else if hybrid { (draft_model, "hybrid") } else { select_draft_model(&checkpoint, &root, draft_model, draft)? };
    let draft_config = draft_model.as_ref().filter(|_| !hybrid && cartridge.gemma4.is_none()).map(|path|
        engine_model::qwen3::Config::open(&path.join("config.json")).map_err(|e| e.to_string())
    ).transpose()?;
    let override_ptx = if external_ptx {
        let mut bytes = Vec::new();
        std::fs::File::open(&args[0])
            .map_err(|e| e.to_string())?
            .take(64 * 1024 * 1024 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > 64 * 1024 * 1024 {
            return Err("PTX byte limit exceeded".into());
        }
        Some(CString::new(bytes).map_err(|e| e.to_string())?)
    } else { None };
    let ptx = match &override_ptx {
        Some(ptx) => ptx.as_c_str(),
        None => CStr::from_bytes_with_nul(include_bytes!(concat!(env!("OUT_DIR"), "/kernels.ptx")))
            .map_err(|e| e.to_string())?,
    };
    let listener = TcpListener::bind(bind).map_err(|e| e.to_string())?;
    let _daemon = if daemon {
        match lifecycle::detach()? { Some(guard) => Some(guard), None => return Ok(()) }
    } else { None };
    lifecycle::install_signals()?;
    let mut admin = scheduler::ModelAdmin::new(root, model_name(&checkpoint), admin_socket.as_deref())?;
    eprintln!("sharing client_priority_cap={} owner_key={} model_admin={}",
        sharing.client_priority_cap, sharing.owner_key.is_some(),
        if admin_socket.is_some() { "private-unix-socket" } else { "disabled" });
    let shutdown = &lifecycle::SHUTDOWN;
    limits.default_sampling = Some(spark_engine::sample::Sampling::from_checkpoint(&checkpoint)?);
    let mut tokenizer = Some(load_tokenizer(&checkpoint)?);
    // Unset requests follow the chat template's own default.
    if !thinking_explicit { limits.default_thinking = tokenizer.as_ref().unwrap().chat_options().enable_thinking; }
    if !limits.default_thinking { eprintln!("default_thinking enabled=false"); }
    limits.identity = Some(spark_serve::contract::Identity::load(&checkpoint)?);
    if hybrid && draft_model.is_none()
        && let Some(k) = draft_k {
            let ceiling = if args.iter().any(|s| s == "--draft-depth") && hybrid_depth != 0 { hybrid_depth } else { 4 };
            hybrid_depth = k.min(ceiling);
        }
    let mtp_depth = if hybrid && mtp { if hybrid_depth == 0 { 4 } else { hybrid_depth } } else { 0 };
    let world_size = seqpar_peers.len() + 1;
    let topology = if seqpar_peers.is_empty() { "local" } else { "path" };
    limits.metadata = Some(spark_serve::metadata::Metadata::prepare(&checkpoint, ptx,
        mtp_depth, world_size, topology, server_start)?);
    if shutdown.load(Ordering::Relaxed) { return Ok(()); }
    spark_gpu_probe::driver::with_backend(device, ptx, sizing.gpu_bytes.unwrap_or(0), |backend| {
        if cartridge.gemma4.is_some() {
            if !input_tokens_explicit { limits.input_tokens = usize::MAX; }
            if !output_tokens_explicit { limits.output_tokens = usize::MAX; }
            return gemma4::serve(backend, &checkpoint, &listener, tokenizer.as_mut().unwrap(), &mut limits,
                &sizing, output_bytes_explicit, sharing.clone(), prefix_cache, draft_model.as_deref(), draft, draft_k, shutdown, &mut admin);
        }
        if engine_model::cartridge::Kind::detect(&checkpoint).map_err(|e| e.to_string())?
            == engine_model::cartridge::Kind::Qwen3_5 {
            if yarn { return Err("--rope-scaling yarn requires a Qwen3 checkpoint".into()); }
            if draft_global_bank || draft == spark_serve::protocol::DraftMode::Fast {
                return Err("qwen3_5 supports one Spark with plain or canonical exact drafting".into());
            }
            let (plan,index,headers)=hybrid_plan.as_ref().ok_or("hybrid serving plan absent")?;
            let c=spark_engine::hybrid::plan::config(plan)?;
            let fixed_rows=spark_engine::hybrid::plan::uses_typed_loader(plan);
            if !seqpar_peers.is_empty() { return Err(if fixed_rows { "typed hybrid peers require tp-serve rank startup; world2 remains refused" } else { "hybrid requires local generation" }.into()); }
            if draft == spark_serve::protocol::DraftMode::Plain { draft_model = None; }
            // Sparse targets read only active experts; the sibling draft's
            // catch-up and verification cost exceeds plain decode on GB10.
            else if !mtp && draft_model.is_none() && c.experts == 0 { draft_model = discover_hybrid_draft(&checkpoint, &admin.root, &c); }
            // The sibling draft wins the reasoning and long-answer guards. Use
            // built-in MTP automatically only when that sibling is unavailable.
            let mtp = mtp || (draft == spark_serve::protocol::DraftMode::Auto && draft_model.is_none()
                && spark_engine::arms::Limits::auto_mtp(&c));
            if draft == spark_serve::protocol::DraftMode::Strict && !mtp && draft_model.is_none() {
                return Err("hybrid strict mode requires --mtp or --draft-model".into());
            }
            if fixed_rows && mtp { spark_serve::protocol::typed_mtp_admission(sizing.max_active, draft_model.is_some(), draft)?; }
            let draft_checkpoint = draft_model.as_ref().map(|root| {
                spark_engine::hybrid::mtp::validate_tokenizer(&checkpoint, root)?;
                engine_model::qwen3_5::Checkpoint::inspect(root).map_err(|e| e.to_string())
            }).transpose()?;
            let available = backend.available_memory_bytes()?;
            let usable=available.saturating_sub((available / 10).max(1024 * 1024 * 1024));
            let budget = sizing.gpu_bytes.unwrap_or(usable).min(usable);
            if sizing.max_active.is_some_and(|n| !(1..=32).contains(&n)) { return Err("--max-active must be in 1..32".into()); }
            let mtp_weights = if !mtp { 0 } else if fixed_rows {
                usize::try_from(engine_model::qwen3_5::plan::mtp_weight_bytes(plan)?).map_err(|_| "MTP weight address overflow")?
            } else { c.mtp_weight_bytes().map_err(|e| e.to_string())? as usize };
            let external_weights = draft_checkpoint.as_ref().map_or(0, |m| m.text_bytes as usize);
            let pooled_prompt=spark_engine::hybrid::memory::pool_prompt(fixed_rows,mtp || draft_checkpoint.is_some(),sizing.max_active);
            let workspace_context=sizing.context_tokens.unwrap_or(c.max_positions as usize);
            let shared_scratch = if mtp || draft_checkpoint.is_some() { 144 * 1024 * 1024 } else { 64 * 1024 * 1024 };
            let shared_scratch = if fixed_rows { shared_scratch } else { shared_scratch.max(spark_engine::hybrid::decode::Decode::reservation_bytes_for_rows(&c,
                if mtp || draft_checkpoint.is_some() { spark_engine::hybrid::decode::MAX_VERIFY_ROWS } else { spark_engine::hybrid::decode::MAX_CHAT_ROWS })) };
            let shared_scratch=if fixed_rows {
                spark_engine::hybrid::Model::fixed_decode_bytes_at(&c,if mtp || draft_checkpoint.is_some() {
                    spark_engine::hybrid::decode::MAX_VERIFY_ROWS } else { spark_engine::hybrid::decode::MAX_CHAT_ROWS },workspace_context).ok_or("shared row byte overflow")?
                    .checked_add(spark_engine::hybrid::memory::host_logits(&c,if mtp || draft_checkpoint.is_some() {
                        spark_engine::hybrid::decode::MAX_VERIFY_ROWS } else { spark_engine::hybrid::decode::MAX_CHAT_ROWS }).ok_or("shared logits overflow")?).ok_or("shared readback overflow")?
                    .checked_add(if pooled_prompt { spark_engine::hybrid::Model::fixed_prompt_bytes(&c,workspace_context).ok_or("shared prompt byte overflow")? } else { 0 }).ok_or("shared prompt overflow")?
                    .checked_add(if mtp { spark_engine::hybrid::mtp::fixed_reservation_bytes(&c,64.min(c.max_positions as usize),spark_engine::hybrid_draft::Options::startup()).ok_or("shared draft byte overflow")? } else { 0 })
                    .ok_or("shared workspace overflow")?
            } else { shared_scratch };
            let split=if fixed_rows { Some(engine_model::split::SplitPlan::new(&plan.shape,&engine_model::split::SplitRequest {ranks:world_size as u32,memory:vec![u64::MAX;world_size],reserve:vec![0;world_size],
                context:limits.context_tokens as u64,sequences:1,cache_bytes:2,state_bytes:4,draft:mtp,expert_rows:false}).map_err(|e|e.to_string())?) } else { None };
            let resident=if let Some(split)=&split { spark_engine::hybrid::tp::Binding::new(plan,split,0)?.rank_reservations(index,headers,limits.context_tokens,mtp)?[0] }
                else { usize::try_from(plan.shape.tensors.iter().filter(|t| matches!(t.role.place,engine_model::shapes::Place::Model|engine_model::shapes::Place::Layer(_))
                    || mtp && matches!(t.role.place,engine_model::shapes::Place::Draft {..})).try_fold(0u64,|n,t|
                    t.shape.iter().try_fold(t.dtype.bytes(),|a,b| a.checked_mul(*b)).and_then(|bytes| n.checked_add(bytes)))
                    .ok_or("resident byte overflow")?).map_err(|_| "resident address overflow")? };
            let resident=resident.checked_add(external_weights).ok_or("combined resident overflow")?;
            let pool=spark_serve::budget::HybridBudget::new(budget,resident,shared_scratch,sizing.session_bytes)?;
            let available_sessions=pool.sessions;
            let reservation_for = |tokens: usize, context: usize, retain: bool| -> Option<usize> {
                let target=if fixed_rows { spark_engine::hybrid::Model::fixed_request_rows(&c,tokens,context,mtp || draft_checkpoint.is_some(),
                    spark_engine::hybrid::memory::private_rows(pooled_prompt),retain)? }
                    else { spark_engine::hybrid::Model::session_bytes(&c,tokens).checked_add(if mtp || draft_checkpoint.is_some() {
                        spark_engine::hybrid::mtp::reservation_bytes(&c,tokens,spark_engine::hybrid_draft::Options::startup()) } else { 0 })? };
                target.checked_add(draft_checkpoint.as_ref().map_or(Some(0),|m|
                    spark_engine::hybrid::Model::session_bytes(&m.config,tokens).checked_add((if fixed_rows { 16usize } else { 4 }).checked_mul(m.config.state_bytes() as usize)?))?)
            };
            let ceiling = (c.max_positions as usize).min(262144)
                .min(draft_checkpoint.as_ref().map_or(usize::MAX, |m| m.config.max_positions as usize));
            let session_bytes = sizing.session_bytes.unwrap_or(available_sessions);
            let memory_error = |required, tokens| refusal(required, available_sessions.min(session_bytes), &format!("weights/rank {} bytes; shared workspace {shared_scratch} bytes; per-session {} x 1 session; CONTEXT={tokens} (position ceiling {ceiling}); OUTPUT={}", resident, bytes(reservation_for(tokens,tokens,true).unwrap_or(usize::MAX) as u128), limits.output_tokens), &format!("{CUDA}; available input {available} bytes, capped by GPU ceiling {budget} bytes, less weights and shared workspace; --session-budget-mib={session_bytes} bytes; --kv-budget-mib={:?}", sizing.kv_bytes), SERVE);
            if session_bytes > available_sessions { return Err(memory_error(session_bytes as u128, sizing.context_tokens.unwrap_or(1))); }
            let fits = |tokens: usize| pool.fits(reservation_for(tokens,tokens,true))
                && sizing.kv_bytes.is_none_or(|n| tokens * c.kv_bytes_per_token() as usize <= n);
            limits.context_tokens = if let Some(tokens) = sizing.context_tokens {
                if tokens == 0 || tokens > ceiling || !fits(tokens) { return Err(if tokens == 0 || tokens > ceiling { spark_gpu_probe::admission::token_refusal(tokens as u128, ceiling, &format!("CONTEXT={tokens}; OUTPUT={}; context must be positive", limits.output_tokens), "the checkpoint position ceiling", SERVE) } else if sizing.kv_bytes.is_some_and(|n| tokens * c.kv_bytes_per_token() as usize > n) { refusal(tokens as u128 * c.kv_bytes_per_token() as u128, sizing.kv_bytes.unwrap(), &format!("weights/rank {} bytes; KV payload/session x 1; CONTEXT={tokens}; OUTPUT={}", resident, limits.output_tokens), "--kv-budget-mib", SERVE) } else { memory_error(reservation_for(tokens,tokens,true).unwrap_or(usize::MAX) as u128, tokens) }); }
                tokens
            } else {
                let (mut low, mut high) = (1, ceiling);
                if !fits(low) { return Err(if sizing.kv_bytes.is_some_and(|n| low * c.kv_bytes_per_token() as usize > n) { refusal(low as u128 * c.kv_bytes_per_token() as u128, sizing.kv_bytes.unwrap(), &format!("weights/rank {} bytes; KV payload/session x 1; CONTEXT={low}; OUTPUT={}", resident, limits.output_tokens), "--kv-budget-mib", SERVE) } else { memory_error(reservation_for(low,low,true).unwrap_or(usize::MAX) as u128, low) }); }
                while low < high {
                    let middle = low + (high - low).div_ceil(2);
                    if fits(middle) { low = middle; } else { high = middle - 1; }
                }
                low
            };
            if !output_tokens_explicit { limits.output_tokens = limits.context_tokens.saturating_sub(1).max(1); }
            if !input_tokens_explicit { limits.input_tokens = limits.context_tokens.saturating_sub(1).max(1); }
            limits.body_bytes = limits.body_bytes.max(limits.input_tokens.saturating_mul(16)).min(4 * 1024 * 1024);
            if !output_bytes_explicit { limits.output_bytes = limits.output_tokens.saturating_mul(256).saturating_add(4096).min(16 * 1024 * 1024); }
            limits.validate()?;
            tokenizer.as_mut().unwrap().set_limits(limits.tokenizer_limits()).map_err(|e| e.to_string())?;
            let reservation = reservation_for(limits.context_tokens,limits.context_tokens,true).ok_or("request byte overflow")?;
            let active=pool.active_limit(sizing.max_active,reservation_for(1,limits.context_tokens,false).ok_or("minimum request overflow")?)?;
            backend.configure_kv_reserve(available.saturating_sub(budget).max(1024 * 1024 * 1024));
            eprintln!("context_sizing model_max_positions={} context_tokens={} memory_capped={} rope_scaling=default kv_pages=on-demand recurrent_state=on-demand",
                c.max_positions, limits.context_tokens, limits.context_tokens < c.max_positions as usize);
            let active_bytes = session_bytes;
            backend.set_allocation_budget(budget)?;
            if fixed_rows { backend.set_graph_storage_budget(0)?; }
            let model = if fixed_rows {
                spark_engine::hybrid::Model::load_generic_rank(backend,&checkpoint,limits.context_tokens,mtp,plan.clone(),index,headers,split.unwrap(),0,&[1],1,spark_engine::hybrid::tp::LocalExchange)?
            } else { spark_engine::hybrid::Model::load_with_mtp(backend, &checkpoint, limits.context_tokens, world_size, mtp)? };
            if model.model_plan()!=plan { return Err("checkpoint plan changed after admission".into()); }
            let external = draft_model.as_ref().map(|root| spark_engine::hybrid::Model::load(backend, root, limits.context_tokens, 1)).transpose()?;
            let mut engine = spark_serve::hybrid::Hybrid::new(&model, tokenizer.as_ref().unwrap().eos_token_id()).with_draft_depth(hybrid_depth)?;
            if pooled_prompt { engine=engine.with_shared_prompt()?; }
            if let Some(external) = &external { engine = engine.with_draft_model(external, if args.iter().any(|s| s == "--draft-depth") { hybrid_depth } else { draft_k.unwrap_or(4) })?; }
            let decode_scratch = engine.reserve_workspace()?;
            if decode_scratch > shared_scratch || backend.allocated_bytes() > resident.checked_add(shared_scratch).ok_or("startup bound overflow")? { return Err("hybrid shared allocation exceeded its funded bound".into()); }
            limits.metadata.as_ref().unwrap().start();
            eprintln!("hybrid_decode rows={} shared_scratch_bytes={decode_scratch}",
                if mtp || external.is_some() { spark_engine::hybrid::decode::MAX_VERIFY_ROWS } else { spark_engine::hybrid::decode::MAX_CHAT_ROWS });
            let mtp_depth = if mtp { if hybrid_depth == 0 { 4 } else { hybrid_depth } } else { 0 };
            eprintln!("mtp enabled={mtp} weights_bytes={mtp_weights} external_weights_bytes={external_weights} depth={mtp_depth} shared_embedding=true shared_head=true verification=canonical");
            eprintln!("gdn_chunk_products=FP32");
            eprintln!("ready model={} cartridge=qwen3_5 world_size=1 weight_precision={} compute_precision={} text_weights_bytes={} device_bytes={} context_tokens={} max_active={} session_reservation_bytes={} gpu_budget_bytes={} kv_precision=BF16 recurrent_precision=FP32 startup_ms={:.3}",
                admin.model, if c.fp8 { "FP8_E4M3" } else { "BF16" }, if c.fp8 { "W8A8_E4M3_FP32_ACC" } else { "BF16" }, model.weight_bytes(), backend.allocated_bytes(), limits.context_tokens, active, reservation, budget, started.elapsed().as_secs_f64() * 1000.0);
            let cache_limits = CacheLimits { entries: 128, retained_bytes: session_bytes, tokens_per_entry: limits.context_tokens,
                logits_per_entry: c.vocab as usize, idle_ttl: std::time::Duration::from_secs(300) };
            return scheduler::serve_requests(listener.try_clone().map_err(|e| e.to_string())?, &engine, tokenizer.as_ref(), &limits, &cache_limits,
                &scheduler::Config { active, connections: active.max(32), prefill_chunk, active_bytes, session_budget_bytes: session_bytes,
                    sharing: sharing.clone(), prefix_cache,
                    draft: if mtp || external.is_some() { draft } else { spark_serve::protocol::DraftMode::Plain }, mtp_depth, draft_global_bank: false, foreground_rows, background_rows: background_rows.unwrap_or(32) }, shutdown, &admin.model, 0);
        }
        if mtp { return Err("--mtp requires a qwen3_5 checkpoint".into()); }
        let mut config = cartridge.require_supported().map_err(|e| e.to_string())?.clone();
        if yarn { config.enable_yarn().map_err(|e| e.to_string())?; }
        let (free_device_bytes, _) = backend.device_memory_info()?;
        // GB10 shares RAM with Linux. Admission reads MemAvailable, which
        // counts reclaimable page cache, including checkpoint pages.
        let available_bytes = backend.available_memory_bytes()?;
        let sized = sizing.resolve(&config, available_bytes, draft_config.as_ref()).map_err(|error| format!("{error}\nStartup inputs: CUDA free={free_device_bytes} bytes; unified available={available_bytes} bytes; OUTPUT={}{}.", limits.output_tokens, if output_tokens_explicit { " (explicit)" } else { " (automatic: selected CONTEXT minus one)" }))?;
        let (kv_budget, session_budget, max_active, device_budget) =
            (sized.kv_bytes, sized.session_bytes, sized.max_active, sized.gpu_bytes);
        backend.configure_kv_reserve(available_bytes.saturating_sub(device_budget).max(1024 * 1024 * 1024));
        limits.context_tokens = sized.context_tokens;
        if !output_tokens_explicit { limits.output_tokens = limits.context_tokens.saturating_sub(1).max(1); }
        if !input_tokens_explicit {
            limits.input_tokens = limits.context_tokens.saturating_sub(1);
        }
        limits.body_bytes = limits.body_bytes.max(limits.input_tokens.saturating_mul(16))
            .min(4 * 1024 * 1024);
        if !output_bytes_explicit {
            limits.output_bytes = limits.output_bytes.max(
                limits.output_tokens.saturating_mul(256).saturating_add(4096)
            ).min(16 * 1024 * 1024);
        }
        limits.validate()?;
        let load = |path: &Path, context_tokens: usize| {
            // Bound session admission before allocating any new weights.
            let mut config = engine_model::qwen3::Config::open(&path.join("config.json"))
                .map_err(|e| e.to_string())?;
            if yarn { config.enable_yarn().map_err(|e| e.to_string())?; }
            let admission = Admission::for_config(&config, context_tokens, kv_budget, session_budget)?
                .concurrent(max_active, session_budget)?;
            backend.set_allocation_budget(device_budget)?;
            let mut model = spark_engine::qwen::Model::load_with_yarn(
                backend, path, context_tokens, kv_budget, yarn,
            )?;
            model.prepare_batch_weights()?;
            let loaded_admission = Admission::for_config(model.config(), model.max_tokens(),
                model.kv_budget_bytes(), session_budget)?.concurrent(max_active, session_budget)?;
            let engine = Qwen::new(&model, context_tokens);
            if loaded_admission != admission || model.max_cache_bytes() != admission.kv_payload_bytes
                || engine.session_reservation_bytes() != Some(admission.backend_reservation_bytes) {
                return Err("loaded model no longer matches preflight session admission".to_owned());
            }
            Ok((model, admission, device_budget))
        };
        let mut loaded = Some(load(&checkpoint, limits.context_tokens)?);
        backend.trim_host_memory();
        loop {
            if shutdown.load(Ordering::Relaxed) {
                drop(loaded.take());
                backend.clear_kv_reserve()?;
                eprintln!("shutdown model_released device_bytes={}", backend.allocated_bytes());
                return Ok(());
            }
            let swap = if let Some((model, _admission, device_budget)) = &loaded {
                // GB10: extra draft depths cost more than extra target rows.
                // Keep small dense targets shallow; amortize larger weights.
                let large = model.active_weight_bytes_per_token() >= 32 * 1024 * 1024 * 1024;
                let draft_k = draft_k.unwrap_or(if model.config().num_experts != 0 || large {3} else {2});
                let background_rows = background_rows.unwrap_or(if draft_model.is_none() {32}
                    else { (if large || model.config().num_experts != 0 {64} else {48}).min(max_active*(draft_k+1)) });
                eprintln!("draft_selection source={draft_source} model={} mode={draft:?} adaptive_cost=true k={draft_k} background_rows={background_rows}",
                    draft_model.as_ref().map(|path| model_name(path)).unwrap_or_else(|| "none".into()));
                let draft_loaded = if let Some(path) = draft_model.as_ref() {
                    validate_draft_tokenizer(&checkpoint, path)?;
                    let before = backend.allocated_bytes();
                    let weights_before = backend.stats().weight_bytes;
                    let mut draft = spark_engine::qwen::Model::load_with_kv_budget(
                        backend, path, sized.draft_context_tokens, kv_budget)?;
                    draft.prepare_batch_weights()?;
                    eprintln!("draft_model_loaded k={draft_k} weight_bytes={} device_bytes={} kv_capacity_bytes={}",
                        backend.stats().weight_bytes - weights_before, backend.allocated_bytes() - before, draft.max_cache_bytes());
                    Some(draft)
                } else { None };
                let mut engine = Qwen::new(model, limits.context_tokens);
                if let Some(draft) = &draft_loaded { engine = engine.with_draft_model(draft, draft_k)?; }
                // Unguided STRICT replays expert rows more slowly than plain decode.
                let draft = if draft == spark_serve::protocol::DraftMode::Auto
                    && model.config().num_experts != 0 && draft_loaded.is_none() {
                    spark_serve::protocol::DraftMode::Plain
                } else { draft };
                if let Some(listen) = seqpar_listen.filter(|_| checkpoint == seqpar_checkpoint) {
                    engine = engine.with_seqpar(Seqpar::new(listen, &seqpar_peers, seqpar_threshold)?);
                    eprintln!("seqpar_ready final_rank={} world={} threshold_tokens={} topology=path",
                        seqpar_peers.len(), seqpar_peers.len() + 1, seqpar_threshold);
                }
                let workspace_bytes = engine.reserve_workspace(foreground_rows.max(background_rows), max_active)?;
                let resident = backend.allocated_bytes();
                let available = session_budget.checked_sub(workspace_bytes)
                    .ok_or_else(|| refusal(workspace_bytes as u128, session_budget, &format!("weights/rank {} bytes; shared workspace {workspace_bytes} bytes; per-session {} x {max_active} slots; CONTEXT={}; OUTPUT={}", cartridge.weight_bytes, engine.session_reservation_bytes().map(|n| bytes(n as u128)).unwrap_or_else(|| "unavailable".into()), model.max_tokens(), limits.output_tokens), "--session-budget-mib", SERVE))?
                    .min(device_budget.checked_sub(resident).ok_or_else(|| refusal(resident as u128, *device_budget, &format!("weights/rank {} bytes; resident weights/workspace {resident} bytes; per-session {} x {max_active} slots; CONTEXT={}; OUTPUT={}", cartridge.weight_bytes, engine.session_reservation_bytes().map(|n| bytes(n as u128)).unwrap_or_else(|| "unavailable".into()), model.max_tokens(), limits.output_tokens), &format!("{CUDA}; startup CUDA free={free_device_bytes} bytes; unified available={available_bytes} bytes; capped by the configured GPU ceiling after loading weights and shared workspace"), SERVE))?);
                let (context, max_active, admission) = sizing.finish(model.config(), model.max_tokens(),
                    limits.context_tokens, draft_loaded.as_ref().map(|m| (m.config(), m.max_tokens())),
                    max_active, model.kv_budget_bytes(), available)?;
                limits.context_tokens = context;
                engine.set_context_limit(context);
                if !input_tokens_explicit { limits.input_tokens = context.saturating_sub(1); }
                if !output_tokens_explicit { limits.output_tokens = context.saturating_sub(1).max(1); }
                limits.validate()?;
                tokenizer.as_mut().unwrap().set_limits(limits.tokenizer_limits()).map_err(|e| e.to_string())?;
                eprintln!("context_sizing model_max_positions={} context_tokens={} memory_capped={} rope_scaling={} safety_reserve_bytes={} weights_bytes={} kv_pages=on-demand draft_context_tokens={} resident_bytes={} workspace_bytes={workspace_bytes}",
                    model.config().max_positions, context, context < model.config().max_positions as usize,
                    if model.config().yarn { "yarn-factor4-original32768" } else { "default" },
                    available_bytes.saturating_sub(*device_budget), cartridge.weight_bytes,
                    sized.draft_context_tokens, resident);
                let retained_bytes = admission.retained_bytes;
                let active_bytes = admission.active_bytes;
                if backend.allocated_bytes().checked_add(available)
                    .is_none_or(|bytes| bytes > *device_budget) {
                    return Err(refusal(backend.allocated_bytes() as u128 + available as u128, *device_budget, &format!("weights/rank {} bytes; shared workspace {workspace_bytes} bytes; per-session {} x {max_active} slots; session pool {available} bytes; CONTEXT={context}; OUTPUT={}", cartridge.weight_bytes, engine.session_reservation_bytes().map(|n| bytes(n as u128)).unwrap_or_else(|| "unavailable".into()), limits.output_tokens), &format!("{CUDA}; startup CUDA free={free_device_bytes} bytes; unified available={available_bytes} bytes; capped by the configured GPU ceiling after loading weights and shared workspace"), SERVE));
                }
                let cache_limits = CacheLimits {
                    entries: 128, retained_bytes,
                    tokens_per_entry: limits.context_tokens, logits_per_entry: engine.vocab_size(),
                    idle_ttl: std::time::Duration::from_secs(300),
                };
                let kv_token_bytes = model.max_cache_bytes() / 2 / model.max_tokens();
                eprintln!("prefix_cache enabled={prefix_cache} elastic=true idle_expiry=pressure scope={}",
                    if sharing.prefix_cache_per_api_key { "api-key" } else { "global" });
                eprintln!("ready bind={} model={} device_bytes={} host_rss_bytes={} context_tokens={} session_budget_bytes={} workspace_bytes={workspace_bytes} active_budget_bytes={active_bytes} retained_budget_bytes={retained_bytes} kv_precision=BF16 kv_bytes_per_token={kv_token_bytes} output_byte_limit={} generation_timeout_secs={} input_limit={} output_limit={} kv_budget_bytes={} gpu_budget_bytes={} max_active={max_active} startup_free_device_bytes={free_device_bytes} startup_available_bytes={available_bytes} startup_ms={:.3}",
                    listener.local_addr().map_err(|e| e.to_string())?, admin.model,
                    backend.allocated_bytes(), host_rss_bytes()?, limits.context_tokens, session_budget, limits.output_bytes, limits.generation_timeout.as_secs(),
                    limits.input_tokens, limits.output_tokens, kv_budget, device_budget, started.elapsed().as_secs_f64() * 1000.0);
                limits.metadata.as_ref().unwrap().start();
                scheduler::serve_model(&listener, &engine, tokenizer.as_ref(), &limits, &cache_limits,
                    &scheduler::Config {
                        active: max_active, connections: 32.max(max_active), prefill_chunk,
                        foreground_rows, background_rows, active_bytes,
                        session_budget_bytes: available, draft, mtp_depth: 0, draft_global_bank,
                        prefix_cache, sharing: sharing.clone(),
                    }, shutdown, &admin)?
            } else {
                scheduler::wait_for_model(&listener, &limits, &admin, shutdown)?
            };
            let Some(swap) = swap else {
                drop(loaded.take());
                backend.clear_kv_reserve()?;
                eprintln!("shutdown model_released device_bytes={}", backend.allocated_bytes());
                return Ok(());
            };
            let drain_ms = swap.started.elapsed().as_secs_f64() * 1000.0;
            let next = swap.path.clone();
            let next_name = model_name(&next);
            let mut unload_ms = 0.0;
            let mut tokenizer_ms = 0.0;
            let mut weights_ms = 0.0;
            let mut unloaded_device_bytes = 0;
            let loading_stop = AtomicBool::new(false);
            let load_start = Instant::now();
            let before = backend.stats();
            let mut next_identity = None;
            let mut next_metadata = None;
            let mut loading_limits = limits.clone();
            loading_limits.metadata = None;
            let result = std::thread::scope(|scope| {
                let loading = scope.spawn(|| scheduler::serve_loading(&listener, &loading_limits, &next_name, &loading_stop, &admin));
                let had_model = loaded.is_some();
                let mut result = (|| {
                    engine_model::cartridge::Cartridge::preflight(&next).map_err(|e| e.to_string())?;
                    // Keep the previous tokenizer and path until the replacement succeeds.
                    let start = Instant::now();
                    let next_tokenizer = if checkpoint != next || tokenizer.is_none() {
                        Some(load_tokenizer(&next)?)
                    } else { None };
                    let next_sampling = spark_engine::sample::Sampling::from_checkpoint(&next)?;
                    tokenizer_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let identity = spark_serve::contract::Identity::load(&next)?;
                    let sequence = next == seqpar_checkpoint && !seqpar_peers.is_empty();
                    let metadata = spark_serve::metadata::Metadata::prepare(&next, ptx,
                        0, if sequence { world_size } else { 1 },
                        if sequence { "path" } else { "local" }, server_start)?;
                    let unload_start = Instant::now();
                    backend.synchronize()?;
                    drop(loaded.take());
                    backend.clear_kv_reserve()?;
                    unload_ms = unload_start.elapsed().as_secs_f64() * 1000.0;
                    unloaded_device_bytes = backend.allocated_bytes();
                    if unloaded_device_bytes != 0 {
                        return Err("old model left live device allocations".to_owned());
                    }
                    let start = Instant::now();
                    let model = load(&next, limits.context_tokens)?;
                    weights_ms = start.elapsed().as_secs_f64() * 1000.0;
                    loaded = Some(model);
                    if let Some(next_tokenizer) = next_tokenizer { tokenizer = Some(next_tokenizer); }
                    checkpoint = next.clone();
                    next_identity = Some(identity);
                    next_metadata = Some(metadata);
                    Ok(next_sampling)
                })();
                if let Err(error) = &result
                    && had_model && loaded.is_none() {
                        // Partial replacement weights have dropped; restore one weight copy.
                        match load(&checkpoint, limits.context_tokens) {
                            Ok(model) => loaded = Some(model),
                            Err(recovery) => result = Err(format!("{error}; previous model recovery failed: {recovery}")),
                        }
                    }
                loading_stop.store(true, Ordering::Relaxed);
                match loading.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => eprintln!("model loading listener: {error}"),
                    Err(_) => eprintln!("model loading listener panicked"),
                }
                result
            });
            if let Some(identity) = next_identity { limits.identity = Some(identity); }
            if let Some(metadata) = next_metadata { limits.metadata = Some(metadata); }
            backend.trim_host_memory();
            let host_rss_bytes = host_rss_bytes()?;
            let load_ms = load_start.elapsed().as_secs_f64() * 1000.0 - unload_ms;
            let total_ms = swap.started.elapsed().as_secs_f64() * 1000.0;
            let stats = backend.stats().since(before);
            let loader_ms = stats.weight_read_upload_ns as f64 / 1_000_000.0;
            let allocation_wait_ms = stats.allocation_ns as f64 / 1_000_000.0;
            let body = match &result {
                Ok(sampling) => {
                    limits.default_sampling = Some(*sampling);
                    admin.model = next_name;
                    format!("{{\"status\":\"ready\",\"model\":{},\"drain_ms\":{drain_ms:.3},\"unload_ms\":{unload_ms:.3},\"load_ms\":{load_ms:.3},\"total_ms\":{total_ms:.3},\"tokenizer_ms\":{tokenizer_ms:.3},\"weights_ms\":{weights_ms:.3},\"loader_ms\":{loader_ms:.3},\"allocation_wait_ms\":{allocation_wait_ms:.3},\"weight_bytes\":{},\"unloaded_device_bytes\":{unloaded_device_bytes},\"device_bytes\":{},\"peak_device_bytes\":{},\"pinned_host_bytes\":{},\"host_rss_bytes\":{host_rss_bytes}}}",
                        spark_serve::protocol::quote(&admin.model), stats.weight_bytes, backend.allocated_bytes(),
                        backend.peak_allocated_bytes(), backend.pinned_host_bytes())
                }
                Err(error) => spark_serve::protocol::error_json("model_load_error", error),
            };
            eprintln!("model_swap {body}");
            // A disconnected admin client does not undo a completed swap.
            if let Err(error) = swap.finish(if result.is_ok() { 200 } else { 503 }, &body, &limits) {
                eprintln!("model_swap response: {error}");
            }
        }
    })
}
#[cfg(target_os = "linux")]
fn select_draft_model(target: &std::path::Path, root: &std::path::Path,
    explicit: Option<std::path::PathBuf>, mode: spark_serve::protocol::DraftMode,
) -> Result<(Option<std::path::PathBuf>, &'static str), String> {
    if mode == spark_serve::protocol::DraftMode::Plain { return Ok((None, "disabled")); }
    if let Some(path) = explicit {
        validate_draft_tokenizer(target, &path)?;
        return Ok((Some(path), "explicit"));
    }
    let config = engine_model::qwen3::Config::open(&target.join("config.json")).map_err(|e| e.to_string())?;
    if !config.qk_norm { return Ok((None, "auto-unavailable-dense")); }
    let path = target.parent().unwrap_or(root).join("Qwen3-0.6B");
    if !path.is_dir() || path == target { return Ok((None, "auto-unavailable")); }
    let path = path.canonicalize().map_err(|e| e.to_string())?;
    if !path.starts_with(root) || path == target
        || engine_model::qwen3::Config::open(&path.join("config.json")).is_err()
        || validate_draft_tokenizer(target, &path).is_err() {
        return Ok((None, "auto-incompatible"));
    }
    Ok((Some(path), "auto"))
}
#[cfg(target_os = "linux")]
fn discover_hybrid_draft(target: &std::path::Path, root: &std::path::Path,
    config: &engine_model::qwen3_5::Config) -> Option<std::path::PathBuf> {
    let path = target.parent()?.join("Qwen3.5-0.8B");
    if !path.is_dir() { return None; }
    let candidate = (|| -> Result<std::path::PathBuf, String> {
        let path = path.canonicalize().map_err(|e| e.to_string())?;
        if !path.starts_with(root) || path == target { return Err("draft is outside model root or equals target".into()); }
        let c = engine_model::qwen3_5::Config::open(&path.join("config.json")).map_err(|e| e.to_string())?;
        if c.fp8 || c.hidden >= config.hidden || c.vocab != config.vocab {
            return Err("draft must be a smaller BF16 hybrid with matching vocabulary".into());
        }
        spark_engine::hybrid::mtp::validate_tokenizer(target, &path)?;
        Ok(path)
    })();
    match candidate {
        Ok(path) => { eprintln!("draft_model_discovered cartridge=qwen3_5 sibling=Qwen3.5-0.8B"); Some(path) }
        Err(error) => { engine_model::events::note_line("draft_model_skipped", &format!("draft_model_skipped {error}")); None }
    }
}
#[cfg(target_os = "linux")]
fn validate_draft_tokenizer(target: &std::path::Path, draft: &std::path::Path) -> Result<(), String> {
    let read = |path: std::path::PathBuf| std::fs::read(path).map_err(|e| e.to_string());
    for name in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"] {
        if read(target.join(name))? != read(draft.join(name))? {
            return Err(format!("draft model {name} differs from target"));
        }
    }
    Ok(())
}
#[cfg(target_os = "linux")]
fn load_tokenizer(path: &std::path::Path) -> Result<spark_serve::tokenizer::ModelTokenizer, String> {
    // Keep long-lived vocabulary allocations out of the model loader's heap.
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let tokenizer = spark_serve::tokenizer::ModelTokenizer::from_directory(path)?;
            if engine_model::cartridge::Kind::detect(path).map_err(|e| e.to_string())? == engine_model::cartridge::Kind::Qwen3 {
                let config = engine_model::qwen3::Config::open(&path.join("config.json")).map_err(|e| e.to_string())?;
                if tokenizer.eos_token_id() as u64 != config.eos_token || !tokenizer.contains_id(config.bos_token as u32) {
                    return Err("model and tokenizer special IDs disagree".into());
                }
            }
            Ok(tokenizer)
        }).join().map_err(|_| "tokenizer loader panicked".to_owned())?
    })
}
#[cfg(target_os = "linux")]
fn model_name(path: &std::path::Path) -> String {
    path.file_name().unwrap_or_default().to_string_lossy().to_ascii_lowercase()
}
#[cfg(target_os = "linux")]
fn host_rss_bytes() -> Result<usize, String> {
    std::fs::read_to_string("/proc/self/status").map_err(|e| e.to_string())?
        .lines().find_map(|line| line.strip_prefix("VmRSS:")?.split_whitespace().next()?.parse::<usize>().ok())
        .and_then(|kb| kb.checked_mul(1024)).ok_or_else(|| "missing process RSS".into())
}
fn main() {
    #[cfg(target_os = "linux")]
    {
        if let Err(error) = run() {
            eprintln!("FAIL {error}");
            std::process::exit(1);
        }
        if lifecycle::SHUTDOWN.load(std::sync::atomic::Ordering::Relaxed) {
            eprintln!("shutdown complete exit=0");
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let args: Vec<_> = std::env::args().skip(1).collect();
        if args.first().is_some_and(|a| a == "--list") && args.len() == 2 {
            if let Err(error) = cartridges::list(std::path::Path::new(&args[1])) {
                eprintln!("FAIL {error}");
                std::process::exit(1);
            }
            return;
        }
        eprintln!(
            "spark-serve requires Linux and the CUDA driver; no generation backend is available on this platform"
        );
        std::process::exit(1);
    }
}
