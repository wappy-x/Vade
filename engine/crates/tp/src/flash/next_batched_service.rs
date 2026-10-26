//! World-one adapter; shared HTTP, chat renderer, sampling, stops and usage.
//! Bounded prompt tiles and up to sixteen resident chats through the public row adapter.
use spark_engine::flash::next_batched::{NextModel, POOL_SIZE};
use spark_serve::{cache::CacheLimits, flashnext::FlashNext, scheduler};
use std::{
    ffi::CString,
    net::TcpListener,
    path::Path,
    sync::atomic::AtomicBool,
    time::{Duration, SystemTime},
};

pub fn run(
    root: &Path,
    ptx_path: &str,
    listener: TcpListener,
    tokenizer: engine_tokenizer::Tokenizer,
    mut limits: spark_serve::Limits,
    eos: Vec<u32>,
    requests: usize,
    server_start: SystemTime,
    shutdown: &'static AtomicBool,
    options: spark_engine::flash_next_options::Options,
    mtp: usize,
    checkpoint: engine_model::flash::Checkpoint,
    mut pp: Option<(std::net::TcpStream, spark_engine::flash::next_batched::PpStage)>,
    fast: Option<Box<dyn spark_engine::flash::next_batched::PpWire>>,
    experiments: spark_serve::mtp_batch::Experiments,
    tp: Option<(std::net::TcpStream, super::tp_proxy::Wire)>,
) -> Result<(), String> {
    let ptx = CString::new(std::fs::read(ptx_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let ab_ptx = spark_gpu_probe::driver::ab_ptx_from_env()?;
    if let Some((stream, _)) = pp.as_mut() {
        spark_tp::runtime::verify_pipeline_artifacts(stream, &ptx, root, ab_ptx.as_deref(), experiments,
            spark_engine::flash::next_batched::pp_timeout_from_env()?)?;
    }
    let ceiling = std::env::var("SEQPAR_MEMORY_BYTES")
        .ok()
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| "invalid SEQPAR_MEMORY_BYTES")
        })
        .transpose()?
        .unwrap_or(usize::MAX);
    if ceiling == 0 {
        return Err("SEQPAR_MEMORY_BYTES must be positive".into());
    }
    // The serving identity and receipt (index parse, header / PTX hashes, ~2 s) are
    // only read once serving starts: prepare them beside the load.
    // Pipeline responses state the actual stage count and topology "pipeline".
    let (world, topology) = pp.as_ref().map_or((1, "local"), |(_, stage)| (stage.ranks, "pipeline"));
    std::thread::scope(|scope| {
        let receipts = scope.spawn(|| -> Result<_, String> {
            Ok((spark_serve::contract::Identity::load(root)?,
                spark_serve::metadata::Metadata::prepare(root, &ptx, 0, world, topology, server_start)?))
        });
        spark_gpu_probe::driver::with_backend(0, &ptx, ceiling, |backend| {
            if let Some(ptx) = &ab_ptx { backend.load_ab_ptx(ptx)?; }
            let advise_start = std::time::Instant::now();
            let (files, bytes, errors) = super::super::advise_checkpoint_cache(root, &checkpoint);
            let vocab = checkpoint.config.vocab as usize;
            let available = backend.available_memory_bytes()?;
            let reserve = (available / 10).max(2 << 30);
            let budget = ceiling.min(available.saturating_sub(reserve));
            backend.set_allocation_budget(budget)?;
            eprintln!(
                "flashnext startup cache_files={files} cache_bytes_advised={bytes} cache_errors={errors} budget_bytes={budget}"
            );
            let load_start = std::time::Instant::now();
            // VADE_FLASHNEXT_MTP_DEPTH=1..=15 (unset = 7 on one Spark and on the VADE_W2_PP
            // pipeline; spark_engine::flash_draft_stack): resident MTP depth. Requests keep
            // mtp_depth 0..=3; deeper drafts serve only VADE_MTP_ADAPT's default-depth requests.
            let depth = spark_engine::flash_draft_stack::resident_mtp_depth(pp.is_some())?;
            // Rank 0 projects the head on every world (VADE_W2_PP front too): it funds the bounded head.
            let model = NextModel::load_with_checkpoint_role(backend, root, Some(checkpoint), options, depth, false)?;
            // VADE_W2_TP=1: rank 0 runs every layer itself, so it keeps all 48 layers' state.
            if let Some((_, stage)) = pp.as_ref() && tp.is_none() && spark_engine::flash_next_rows::pp_stage_state()? {
                model.own_stage_state(stage.first, stage.end, &stage.intervals())?;
            }
            // VADE_FLASHNEXT_KV_RESERVE (default on; =0 off): freed session KV arenas stay in
            // the backend reserve (released on demand or under host memory pressure).
            let kv_reserve = match std::env::var("VADE_FLASHNEXT_KV_RESERVE").as_deref() {
                Ok("0") => false,
                Err(std::env::VarError::NotPresent) | Ok("1") => true,
                _ => return Err("VADE_FLASHNEXT_KV_RESERVE must be 0 or 1".into()),
            };
            if kv_reserve { backend.configure_kv_reserve(available.saturating_sub(budget).max(1 << 30)); }
            let vector_pool = spark_gpu_probe::driver::vector_pool_from_env()?;
            let prepare_start = std::time::Instant::now();
            limits.next_mtp_depth = Some(model.mtp_depth());
            let context = limits.context_tokens;
            backend.prepare_flash_dense()?;
            let tile = spark_engine::flash_next_rows::prompt_tile();
            let needed = model.session_reservation_bytes_for_chunk(context, tile)? as u128;
            if kv_reserve { backend.kv_conversation_ended(needed as usize); }
            let resident = backend.allocated_bytes();
            let resident_tiles = backend.flash_dense_bytes() + backend.flash_mixed_bytes();
            // Live sessions stop TRANSIENT_RESERVE short of the allocator ceiling.
            let (active, pool, shared) = model.session_pool_plan(
                context, 1 + usize::from(limits.ab_ptx),
                spark_serve::budget::after_transient_reserve(budget.saturating_sub(resident)), resident_tiles)?;
            // The prompt workspace is part of `shared`: allocate it before the first request.
            model.prepare_prompt_workspace(backend, context)?;
            if needed > pool as u128 {
                return Err(spark_serve::budget::refusal(
                    needed,
                    pool,
                    &format!(
                        "FP4-Local CONTEXT={context}; OUTPUT={}; up to sixteen memory-admitted row chats; resident_bytes={resident}",
                        limits.output_tokens
                    ),
                    "the world-1 session pool; reduce CONTEXT or raise SEQPAR_MEMORY_BYTES",
                    "CONTEXT / SEQPAR_MEMORY_BYTES",
                ));
            }
            model.set_session_limit(active)?;
            // VADE_VECTOR_POOL (default on; =0 off): a request's freed device vectors (GDN
            // states, rollback storage, draft and head arenas, workspaces) wait in a size-keyed
            // pool for the next request instead of cuMemFree + cuMemAlloc at every request
            // boundary (GPU idle meanwhile). After the load and the admission plan, so neither
            // counts it; capped and released as the KV reserve.
            if vector_pool {
                backend.configure_vector_pool(available.saturating_sub(budget).max(1 << 30));
                backend.kv_conversation_ended(needed as usize);
            }
            // Every worker must fund the front's admitted chats before serving begins.
            if let Some((stream, stage)) = pp {
                model.set_pp_front(stream, fast, stage, context, active, tile, shutdown)?;
                eprintln!("flashnext pp front ranks={} split={} context={context} active={active} tile={tile}", stage.ranks, stage.end);
            }
            // VADE_W2_TP=1: the swap wire's TCP link stays open while serving; the proxy thread
            // owns the wire.
            let (_tp_link, tp_wire) = match tp {
                Some((link, swap)) => { model.set_tp(backend, 0, link)?; (None::<std::net::TcpStream>, Some(swap)) }
                None => (None, None),
            };
            let (identity, metadata) = receipts.join().map_err(|_| "startup receipt worker panicked")??;
            limits.identity = Some(identity);
            limits.metadata = Some(metadata.clone());
            let engine = FlashNext::with_experiments(backend, &model, context, eos, experiments)?;
            // VADE_FLASHNEXT_PREFIX_CACHE (default off; 1 = on): exact 64-aligned prompt prefixes,
            // world 1 only (a pipeline front refuses it: prefix_switch::refuse_on_pipeline).
            // Retention is elastic: admission reclaims it before any live reservation.
            // VADE_FLASHNEXT_ADOPT: an adopted entry holds a whole session arena (16 GB at
            // CONTEXT 262144), so retention may use the pool; admission still reclaims first.
            let prefix_cache = engine.prefix_cache();
            let cache = CacheLimits {
                entries: if prefix_cache { 16 } else { 1 },
                retained_bytes: if !prefix_cache { 0 } else if engine.adopt() { pool } else { pool / 2 },
                tokens_per_entry: context,
                logits_per_entry: vocab,
                idle_ttl: Duration::from_secs(if prefix_cache { 3600 } else { 60 }),
            };
            let config = scheduler::Config {
                active,
                connections: POOL_SIZE,
                prefill_chunk: tile,
                foreground_rows: active * (model.mtp_depth() + 1),
                background_rows: active * (model.mtp_depth() + 1),
                active_bytes: pool,
                session_budget_bytes: pool,
                draft: if mtp == 0 {
                    spark_serve::protocol::DraftMode::Plain
                } else {
                    spark_serve::protocol::DraftMode::Strict
                },
                mtp_depth: mtp,
                draft_global_bank: false,
                prefix_cache,
                sharing: scheduler::Sharing::default(),
            };
            unsafe extern "C" {
                fn signal(number: i32, handler: extern "C" fn(i32)) -> usize;
            }
            unsafe {
                signal(2, super::super::stop);
                signal(15, super::super::stop);
            }
            backend.synchronize()?;
            let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1000.0;
            let now = std::time::Instant::now();
            eprintln!("flashnext startup_phases inspect_advise_ms={:.0} load_ms={:.0} prepare_ms={:.0} process_ms={:.0}",
                ms(advise_start, load_start), ms(load_start, prepare_start), ms(prepare_start, now),
                server_start.elapsed().map_or(0.0, |d| d.as_secs_f64() * 1000.0));
            metadata.start();
            let name = spark_tp::next::model_id(root)?;
            eprintln!(
                "flashnext ready world={world} model={name} context={context} concurrency=rows active={active} connections={POOL_SIZE} mtp_depth={mtp} resident_bytes={resident} session_pool_bytes={pool} session_reservation_bytes={needed} shared_pool_reservation_bytes={shared} transient_reserve_bytes={} prefix_cache={prefix_cache}",
                spark_serve::budget::TRANSIENT_RESERVE
            );
            let worker = tp_wire.zip(model.tp_mailbox_host()).map(|(wire, host)| wire.bind(host, true)).transpose()?;
            let (proxy_stop, attached) = (AtomicBool::new(false), AtomicBool::new(false));
            let served = std::thread::scope(|scope| {
                let proxy = worker.map(|worker| super::tp_proxy::spawn(scope, worker, &proxy_stop, &attached, shutdown)).transpose()?;
                if let Some(Err(error)) = proxy.as_ref().map(|p| super::tp_proxy::wait_attached(&attached, p)) {
                    eprintln!("flashnext tp: {error}");
                    return Err(error);
                }
                let served = scheduler::serve_requests(
                    listener,
                    &engine,
                    Some(&tokenizer),
                    &limits,
                    &cache,
                    &config,
                    shutdown,
                    &name,
                    requests,
                );
                proxy_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                // A waiting doorbell (VADE_W2_TP_DOORBELL=wait) may be parked: wake it to see stop.
                if let Some(p) = &proxy { p.thread().unpark(); }
                let proxied = proxy.map_or(Ok(()), |p| p.join().map_err(|_| "tensor split proxy panicked".to_string()).and_then(|r| r));
                served.and_then(|s| proxied.map(|()| s))
            });
            if served.is_err() {
                spark_gpu_probe::teardown::arm();
            }
            if served.is_ok() && model.pp_failed() {
                return Err("VADE_W2_PP: the pipeline link failed; the front stopped".into());
            }
            served
        })
    })
}

/// Pipeline worker: the world-one load/admission plan, then this stage's layers and
/// downstream relay until the upstream link closes. No HTTP/tokenizer/scheduler here.
pub fn run_back(
    root: &Path,
    ptx_path: &str,
    context: usize,
    mut stream: std::net::TcpStream,
    fast: Option<Box<dyn spark_engine::flash::next_batched::PpWire>>,
    shutdown: &'static AtomicBool,
    options: spark_engine::flash_next_options::Options,
    checkpoint: engine_model::flash::Checkpoint,
    mut next: Option<(std::net::TcpStream, Option<Box<dyn spark_engine::flash::next_batched::PpWire>>)>,
    stage: spark_engine::flash::next_batched::PpStage,
    experiments: spark_serve::mtp_batch::Experiments,
    tp: Option<(std::net::TcpStream, super::tp_proxy::Wire)>,
) -> Result<(), String> {
    let ptx = CString::new(std::fs::read(ptx_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    // Check the downstream edge first, then upstream: admission propagates toward
    // the front, which cannot finish its identity handshake after a tail mismatch.
    if let Some((stream, _)) = next.as_mut() {
        spark_tp::runtime::verify_pipeline_artifacts(stream, &ptx, root, None, experiments,
            spark_engine::flash::next_batched::pp_timeout_from_env()?)?;
    }
    // Workers have no A/B cohort scheduler; the front's fingerprint includes its A/B PTX.
    spark_tp::runtime::verify_pipeline_artifacts(&mut stream, &ptx, root, None, experiments,
        spark_engine::flash::next_batched::pp_timeout_from_env()?)?;
    let ceiling = std::env::var("SEQPAR_MEMORY_BYTES")
        .ok()
        .map(|s| s.parse::<usize>().map_err(|_| "invalid SEQPAR_MEMORY_BYTES"))
        .transpose()?
        .unwrap_or(usize::MAX);
    if ceiling == 0 {
        return Err("SEQPAR_MEMORY_BYTES must be positive".into());
    }
    spark_gpu_probe::driver::with_backend(0, &ptx, ceiling, |backend| {
        let (files, bytes, errors) = super::super::advise_checkpoint_cache(root, &checkpoint);
        let available = backend.available_memory_bytes()?;
        let reserve = (available / 10).max(2 << 30);
        let budget = ceiling.min(available.saturating_sub(reserve));
        backend.set_allocation_budget(budget)?;
        eprintln!("flashnext pp back startup cache_files={files} cache_bytes_advised={bytes} cache_errors={errors} budget_bytes={budget}");
        // The pipeline back keeps the pipeline's resident depth (unset = 7), as its front does.
        let depth = spark_engine::flash_draft_stack::resident_mtp_depth(true)?;
        let mut model = NextModel::load_with_checkpoint_role(backend, root, Some(checkpoint), options, depth, true)?;
        model.disable_prompt_frontend()?;
        if spark_engine::flash_next_rows::pp_stage_state()? {
            model.own_stage_state(stage.first, stage.end, &stage.intervals())?;
        }
        let kv_reserve = match std::env::var("VADE_FLASHNEXT_KV_RESERVE").as_deref() {
            Ok("0") => false,
            Err(std::env::VarError::NotPresent) | Ok("1") => true,
            _ => return Err("VADE_FLASHNEXT_KV_RESERVE must be 0 or 1".into()),
        };
        if kv_reserve { backend.configure_kv_reserve(available.saturating_sub(budget).max(1 << 30)); }
        backend.prepare_flash_dense()?;
        let tile = spark_engine::flash_next_rows::prompt_tile();
        let needed = model.session_reservation_bytes_for_chunk(context, tile)? as u128;
        if kv_reserve { backend.kv_conversation_ended(needed as usize); }
        let resident = backend.allocated_bytes();
        let resident_tiles = backend.flash_dense_bytes() + backend.flash_mixed_bytes();
        let (active, pool, _shared) = model.session_pool_plan(context, 1, spark_serve::budget::after_transient_reserve(budget.saturating_sub(resident)), resident_tiles)?;
        model.prepare_prompt_workspace(backend, context)?;
        if needed > pool as u128 {
            return Err(format!("pipeline back cannot fund one session: needed {needed} bytes, pool {pool} bytes"));
        }
        unsafe extern "C" {
            fn signal(number: i32, handler: extern "C" fn(i32)) -> usize;
        }
        unsafe {
            signal(2, super::super::stop);
            signal(15, super::super::stop);
        }
        backend.synchronize()?;
        eprintln!("flashnext pp back loaded context={context} active={active} resident_bytes={resident} session_pool_bytes={pool}");
        let (_link, tp_wire) = match tp {
            Some((link, swap)) => { model.set_tp(backend, 1, link)?; (None::<std::net::TcpStream>, Some(swap)) }
            None => (None, None),
        };
        let worker = tp_wire.zip(model.tp_mailbox_host()).map(|(wire, host)| wire.bind(host, false)).transpose()?;
        let (proxy_stop, attached) = (AtomicBool::new(false), AtomicBool::new(false));
        let served = std::thread::scope(|scope| {
            let proxy = worker.map(|worker| super::tp_proxy::spawn(scope, worker, &proxy_stop, &attached, shutdown)).transpose()?;
            let served = model.serve_pp_back(backend, stream, fast, next, stage, context, active, tile, shutdown);
            proxy_stop.store(true, std::sync::atomic::Ordering::Relaxed);
            if let Some(p) = &proxy { p.thread().unpark(); }
            let proxied = proxy.map_or(Ok(()), |p| p.join().map_err(|_| "tensor split proxy panicked".to_string()).and_then(|r| r));
            served.and_then(|()| proxied)
        });
        if served.is_err() {
            spark_gpu_probe::teardown::arm();
        }
        served
    })
}
