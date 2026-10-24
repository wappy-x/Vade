//! World-one rows service; the exact reference remains in flash::next for gates.
use std::{net::TcpListener, path::Path, sync::atomic::AtomicBool, time::SystemTime};
#[path = "next_batched_service.rs"]
mod batched;
#[cfg(target_os = "linux")]
#[path = "next_pp_wire.rs"]
mod wire;
pub fn run(
    root: &Path,
    ptx_path: &str,
    listener: TcpListener,
    tokenizer: engine_tokenizer::Tokenizer,
    limits: spark_serve::Limits,
    eos: Vec<u32>,
    requests: usize,
    server_start: SystemTime,
    shutdown: &'static AtomicBool,
    sparse: bool,
    mtp: usize,
    checkpoint: engine_model::flash::Checkpoint,
    experiments: spark_serve::mtp_batch::Experiments,
) -> Result<(), String> {
    let options = spark_engine::flash_next_options::Options::load()?;
    spark_tp::next::parse_ep2(Some(if options.fp4_ep { "1" } else { "0" }), 1)?;
    if !sparse {
        return Err("rows require --indexer sparse".into());
    }
    batched::run(
        root,
        ptx_path,
        listener,
        tokenizer,
        limits,
        eos,
        requests,
        server_start,
        shutdown,
        options,
        mtp,
        checkpoint,
        None,
        None,
        experiments,
    )
}

/// VADE_W2_PP: the world-one layer engine across an N-stage chain (world-two default).
/// Rank zero serves HTTP and the head; workers relay over the same neighbor cables.
#[allow(clippy::too_many_arguments)]
pub fn run_pp(
    rank: usize,
    world: usize,
    listen: std::net::SocketAddr,
    next: std::net::SocketAddr,
    root: &Path,
    ptx_path: &str,
    listener: Option<TcpListener>,
    tokenizer: Option<engine_tokenizer::Tokenizer>,
    limits: spark_serve::Limits,
    eos: Vec<u32>,
    requests: usize,
    server_start: SystemTime,
    shutdown: &'static AtomicBool,
    sparse: bool,
    mtp: usize,
    checkpoint: engine_model::flash::Checkpoint,
    experiments: spark_serve::mtp_batch::Experiments,
) -> Result<(), String> {
    let options = spark_engine::flash_next_options::Options::load()?;
    spark_tp::next::parse_ep2(Some(if options.fp4_ep { "1" } else { "0" }), 1)?;
    if !sparse {
        return Err("rows require --indexer sparse".into());
    }
    // Both ranks refuse the prefix cache before they link (prefix_switch::refuse_on_pipeline).
    spark_serve::prefix_switch::refuse_on_pipeline(spark_serve::prefix_switch::prefix_cache()?)?;
    if mtp != 0 && !spark_engine::flash::next_batched::pp_mtp_from_env()? {
        return Err("VADE_W2_PP serves plain decoding only (--mtp 0) unless VADE_W2_PP_MTP=1".into());
    }
    let stage = spark_engine::flash::next_batched::PpStage::from_env(rank, world)?;
    let limit = spark_engine::flash::next_batched::pp_timeout_from_env()?;
    // VADE_W2_PP_RDMA=1 (flash-serve pins 1 when unset): decode steps and short projected tiles ride a polled
    // RoCE wire; queued tiles stay on TCP, and synchronous requests/replies use
    // the wire only when both complete messages fit. Both ranks agree capacity.
    let rdma = match std::env::var("VADE_W2_PP_RDMA").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("0") => false,
        Ok("1") => true,
        _ => return Err("VADE_W2_PP_RDMA must be 0 or 1".into()),
    };
    let wire = |link: &mut std::net::TcpStream| -> Result<Option<Box<dyn spark_engine::flash::next_batched::PpWire>>, String> {
        #[cfg(target_os = "linux")]
        {
            let slot = spark_engine::flash::next_batched::pp_wire_capacity(
                limits.context_tokens, spark_engine::flash_next_rows::prompt_tile())?;
            Ok(wire::RdmaWire::connect(link, rdma, slot, limit)?
                .map(|w| Box::new(w) as Box<dyn spark_engine::flash::next_batched::PpWire>))
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (link, limit);
            if rdma { Err("VADE_W2_PP_RDMA needs Linux".into()) } else { Ok(None) }
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    // Each cable carries both directions. Accept upstream and connect downstream;
    // no closing ring edge exists or is required. Wire setup progresses front to back.
    let (outgoing, incoming) = spark_seqpar::control::connect_peers(
        (rank > 0).then_some(listen), stage.successor().map(|_| next), deadline)?;
    let mut incoming = incoming.map(|mut link| {
        let fast = wire(&mut link)?;
        Ok::<_, String>((link, fast))
    }).transpose()?;
    let outgoing = outgoing.map(|mut link| {
        let fast = wire(&mut link)?;
        Ok::<_, String>((link, fast))
    }).transpose()?;
    eprintln!("flashnext pp stage rank={rank} world={world} layers={}..{} upstream={} downstream={}",
        stage.first, stage.end, incoming.is_some(), outgoing.is_some());
    if rank == 0 {
        let (link, fast) = outgoing.ok_or("pipeline front has no downstream link")?;
        batched::run(root, ptx_path, listener.ok_or("missing rank-zero listener")?,
            tokenizer.ok_or("missing rank-zero tokenizer")?, limits, eos, requests, server_start,
            shutdown, options, mtp, checkpoint, Some((link, stage)), fast, experiments)
    } else {
        let (link, fast) = incoming.take().ok_or("pipeline worker has no upstream link")?;
        batched::run_back(root, ptx_path, limits.context_tokens, link, fast, shutdown, options,
            checkpoint, outgoing, stage, experiments)
    }
}
