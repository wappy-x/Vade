//! Native world-2/4 backend behind the common Flash request front end.
use std::{
    net::{SocketAddr, TcpListener},
    path::Path,
    sync::atomic::AtomicBool,
    time::SystemTime,
};

pub struct Rank {
    pub rank: usize,
    pub world: usize,
    pub listen: SocketAddr,
    pub next: SocketAddr,
}

pub struct Startup<'a> {
    pub root: &'a Path,
    pub tensor_map: &'a Path,
    pub ptx_path: &'a str,
    pub rank_args: Rank,
    pub listener: Option<TcpListener>,
    pub tokenizer: Option<engine_tokenizer::Tokenizer>,
    pub limits: spark_serve::Limits,
    pub eos: Vec<u32>,
    pub requests: usize,
    pub server_start: SystemTime,
    pub shutdown: &'a AtomicBool,
}

pub fn run(startup: Startup<'_>) -> Result<(), String> {
    let Startup {
        root,
        tensor_map,
        ptx_path,
        rank_args,
        listener,
        tokenizer,
        mut limits,
        eos,
        requests,
        server_start,
        shutdown,
    } = startup;
    use spark_tp::{
        next::{NextModel, RankTransport, SharedTransport},
        next_gpu::{Arena, Gpu},
        next_http::{self, CONTEXT, SLOTS},
    };
    use std::{ffi::CString, time::Duration};
    let Rank {
        rank,
        world,
        listen,
        next,
    } = rank_args;
    let ep2 = spark_tp::next::ep2_enabled(world)?;
    let mut cp = engine_model::flash::shard::ShardCheckpoint::open(
        root, tensor_map, rank, world,
    )
    .map_err(|e| e.to_string())?;
    if ep2 {
        cp.plan = cp.plan.with_ep2().map_err(|e| e.to_string())?;
    }
    let model = NextModel::from_checkpoint(cp)?;
    if model.checkpoint.plan.shape.max_positions < CONTEXT as u64 {
        return Err("checkpoint context below 32768".into());
    }
    let vocab = model.checkpoint.plan.shape.vocab as usize;
    if limits.context_tokens != CONTEXT {
        return Err("FP4-Local world 2/4 CONTEXT must be 32768".into());
    }
    limits.next_mtp_depth = Some(0);
    let output = limits.output_tokens;
    let ptx = CString::new(std::fs::read(ptx_path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let ab_ptx = spark_gpu_probe::driver::ab_ptx_from_env()?;
    let path = spark_engine::flash::path_topology(world)?;
    let metadata = if rank == 0 {
        Some(spark_serve::metadata::Metadata::prepare(
            root,
            &ptx,
            0,
            world,
            if path { "path" } else { "ring" },
            server_start,
        )?)
    } else {
        None
    };
    if rank == 0 {
        limits.identity = Some(spark_serve::contract::Identity::load(root)?);
        limits.metadata = metadata.clone();
    }
    limits.validate()?;
    let mut commands = spark_tp::transport::Ring::connect_options(rank, world, listen, next, path)?;
    spark_tp::runtime::verify_artifacts_with_ab(&mut commands, &ptx, root, ab_ptx.as_deref())?;
    let mut agreement = vec![0; world * 16];
    agreement[rank * 16..rank * 16 + 8]
        .copy_from_slice(&((ep2 as u64) << 1).to_le_bytes());
    agreement[rank * 16 + 8..rank * 16 + 16].copy_from_slice(&(output as u64).to_le_bytes());
    commands.all_gather(&mut agreement, 16)?;
    if agreement.chunks_exact(16).any(|v| v != &agreement[..16]) {
        return Err("Next HTTP rank settings disagree".into());
    }
    let addr = |mut a: SocketAddr| -> Result<SocketAddr, String> {
        a.set_port(a.port().checked_add(1).ok_or("data port overflow")?);
        Ok(a)
    };
    let run = std::env::var("TP_RUN_ID").map_err(|_| "fresh TP_RUN_ID required")?;
    let mut data = if path {
        spark_seqpar::Ring::connect_path(rank, world, addr(listen)?, addr(next)?, &run)?
    } else {
        spark_seqpar::Ring::connect(rank, world, addr(listen)?, addr(next)?, &run)?
    };
    data.enable_mailbox(spark_tp::next::RANK_MAILBOX_SLOT)?;
    data.seal_registrations()?;
    if !data.uses_rdma() {
        return Err("Next HTTP requires RDMA".into());
    }
    let memory = data.mailbox_memory();
    let data = SharedTransport::new(data);
    let budget = std::env::var("SEQPAR_MEMORY_BYTES")
        .unwrap_or_else(|_| "107374182400".into())
        .parse()
        .map_err(|_| "memory budget")?;
    spark_gpu_probe::teardown::install()?;
    spark_gpu_probe::driver::with_backend(0, &ptx, budget, |b| {
        if let Some(ptx) = &ab_ptx { b.load_ab_ptx(ptx)?; }
        let _registration = memory
            .map(|(base, n)| unsafe { b.register_host(base, n) })
            .transpose()?;
        let arena = Arena::load(b, &model.checkpoint)?;
        eprintln!(
            "next_http rank={rank} world={world} ep2={ep2} arena_bytes={}",
            arena.bytes
        );
        // Full-capacity KV/recurrent storage plus a conservative 512MiB for
        // M1 scratch, norms, head/PLE/HC and allocation alignment per session.
        let (kv, state) = model
            .checkpoint
            .plan
            .cache_bytes(CONTEXT as u64)
            .map_err(|e| e.to_string())?;
        let reservation = (kv + state) as usize + (512 << 20);
        eprintln!(
            "next_http rank={rank} kv_bytes_per_chat={kv} recurrent_bytes_per_chat={state} scratch_bytes_per_chat={} session_reservation_bytes={reservation} slots={SLOTS} shared_rows_funded_in_arena=true",
            512 << 20
        );
        let mut available = vec![0; world * 8];
        available[rank * 8..rank * 8 + 8].copy_from_slice(
            &(b.available_memory_bytes()?
                .min(b.allocation_budget().saturating_sub(b.accounted_bytes()))
                as u64)
                .to_le_bytes(),
        );
        commands.all_gather(&mut available, 8)?;
        if available.chunks_exact(8).any(|v| {
            u64::from_le_bytes([v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]])
                < (reservation * SLOTS) as u64
        }) {
            return Err("Next HTTP cannot fund eight full-context sessions on every rank".into());
        }
        b.set_allocation_budget(b.accounted_bytes() + reservation * SLOTS)?;
        let mut runtime = next_http::Runtime::new(
            vocab,
            Box::new(|id| {
                let mut exchange = data.clone();
                let gpu = Gpu::new(
                    b,
                    &arena,
                    &model.checkpoint,
                    CONTEXT,
                    Box::new(move |v, row| exchange.bytes(v, row)),
                );
                let mut ready = vec![0; world];
                ready[rank] = u8::from(gpu.is_ok());
                data.clone().bytes(&mut ready, 1)?;
                if ready.contains(&0) {
                    return Err("Next HTTP session allocation failed on a rank".into());
                }
                Ok(Box::new(model.session(data.clone(), gpu?, id, CONTEXT)?))
            }),
        );
        if limits.ab_ptx { runtime = runtime.with_arm_selector(Box::new(|x| b.select_arm(x))); }
        if rank != 0 {
            return next_http::worker(commands, runtime);
        }
        let engine = next_http::Engine::new(commands, runtime, eos, reservation);
        let cache = spark_serve::cache::CacheLimits {
            entries: 1,
            retained_bytes: 0,
            tokens_per_entry: CONTEXT,
            logits_per_entry: vocab,
            idle_ttl: Duration::from_secs(3600),
        };
        unsafe extern "C" {
            fn signal(n: i32, h: extern "C" fn(i32)) -> usize;
        }
        unsafe {
            signal(2, super::stop);
            signal(15, super::stop);
        }
        b.synchronize()?;
        metadata
            .as_ref()
            .ok_or("missing rank-zero metadata")?
            .start();
        let schedule = next_http::schedule(reservation);
        let name = spark_tp::next::model_id(root)?;
        eprintln!("flashnext preview world={world} model={name}");
        let result = spark_serve::scheduler::serve_requests(
            listener.ok_or("missing rank-zero listener")?,
            &engine,
            tokenizer.as_ref(),
            &limits,
            &cache,
            &schedule,
            shutdown,
            &name,
            requests,
        );
        let shutdown = engine.shutdown();
        result.and(shutdown)
    })
}
