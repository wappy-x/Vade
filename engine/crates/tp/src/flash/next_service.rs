//! World-one rows service; the exact reference remains in flash::next for gates.
use std::{net::TcpListener, path::Path, sync::atomic::AtomicBool, time::SystemTime};
#[path = "next_batched_service.rs"]
mod batched;
#[path = "next_tp_proxy.rs"]
mod tp_proxy;

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
        None,
    )
}

/// The pipeline service: the world-one layer engine across an N-stage chain (the tensor split at world two).
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
    // The split has paired prefix snapshots. The layer chain still refuses retention.
    spark_serve::prefix_switch::refuse_on_pipeline(spark_serve::prefix_switch::prefix_cache()?, spark_engine::flash_tp_switch::on())?;
    let stage = spark_engine::flash::next_batched::PpStage::for_rank(rank, world)?;
    let limit = spark_engine::flash::next_batched::PP_LINK_TIMEOUT;
    // Over RDMA (SEQPAR_RDMA unset, auto or 1; the tensor split needs it): decode steps and short
    // projected tiles ride a polled RoCE wire; queued tiles stay on TCP, and synchronous
    // requests/replies use the wire only when both complete messages fit. Both ranks agree capacity.
    let rdma = crate::settings::rdma(std::env::var("SEQPAR_RDMA"))?;
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
            if rdma { Err("the RDMA wire needs Linux (SEQPAR_RDMA=0 selects TCP)".into()) } else { Ok(None) }
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    // Each cable carries both directions. Accept upstream and connect downstream;
    // no closing ring edge exists or is required. Wire setup progresses front to back.
    let (outgoing, incoming) = spark_seqpar::control::connect_peers(
        (rank > 0).then_some(listen), stage.successor().map(|_| next), deadline)?;
    let tensor = spark_engine::flash_tp_switch::on();
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
    // The tensor split: a second cable link (opened after the pipeline's, same order on both
    // ranks) carries the per-layer column swaps as one-sided writes between the mapped
    // mailboxes (tp_proxy).
    let tp = if tensor {
        let (out, inc) = spark_seqpar::control::connect_peers((rank > 0).then_some(listen), (rank == 0).then_some(next), deadline)?;
        let link = out.or(inc).ok_or("tensor split has no peer link")?;
        #[cfg(target_os = "linux")]
        let swap = tp_proxy::Wire::Write(tp_proxy::WriteLink::new(&link)?);
        #[cfg(not(target_os = "linux"))]
        let swap: tp_proxy::Wire = return Err("the tensor split needs Linux".into());
        eprintln!("flashnext tp link rank={rank} world={world}");
        eprintln!("flashnext tp wire=write");
        Some((link, swap))
    } else {
        None
    };
    if rank == 0 {
        let (link, fast) = outgoing.ok_or("pipeline front has no downstream link")?;
        batched::run(root, ptx_path, listener.ok_or("missing rank-zero listener")?,
            tokenizer.ok_or("missing rank-zero tokenizer")?, limits, eos, requests, server_start,
            shutdown, options, mtp, checkpoint, Some((link, stage)), fast, experiments, tp)
    } else {
        let (link, fast) = incoming.take().ok_or("pipeline worker has no upstream link")?;
        batched::run_back(root, ptx_path, limits.context_tokens, link, fast, shutdown, options,
            checkpoint, outgoing, stage, experiments, tp)
    }
}
