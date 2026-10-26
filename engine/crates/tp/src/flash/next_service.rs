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
    let (mut outgoing, mut incoming) = spark_seqpar::control::connect_peers(
        (rank > 0).then_some(listen), stage.successor().map(|_| next), deadline)?;
    // Both ends of every pipeline link agree on VADE_W2_TP before anything else, so a pair
    // with the switch set on one rank only stops here instead of waiting for a second link.
    let tensor = spark_engine::flash_tp_switch::from_env()?;
    let split = tensor as u8 | (tensor && spark_engine::flash_tp_switch::proj_from_env()?) as u8 * 2;
    for link in [outgoing.as_mut(), incoming.as_mut()].into_iter().flatten() { agree_tensor_split(link, split)?; }
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
    // VADE_W2_TP=1: a second cable link (opened after the pipeline's, same order on both
    // ranks) carries the per-layer column swaps on its own RDMA wire.
    let tp = if tensor {
        let (out, inc) = spark_seqpar::control::connect_peers((rank > 0).then_some(listen), (rank == 0).then_some(next), deadline)?;
        let mut link = out.or(inc).ok_or("tensor split has no peer link")?;
        // VADE_W2_TP_WIRE=write: one-sided writes between the mapped mailboxes (tp_proxy).
        let write = tp_proxy::write_from_env()?;
        // Swaps are at most 16 rows of halves (about 100 KB): the wire's smallest slot.
        #[cfg(target_os = "linux")]
        let swap = if write {
            agree_write_wire(&mut link, limit)?;
            tp_proxy::Wire::Write(tp_proxy::WriteLink::new(&link)?)
        } else {
            tp_proxy::Wire::Proxy(Box::new(wire::RdmaWire::connect(&mut link, true, 8 << 20, limit)?.ok_or("tensor split wire refused")?))
        };
        #[cfg(not(target_os = "linux"))]
        let swap: tp_proxy::Wire = { let _ = write; return Err("VADE_W2_TP=1 needs Linux".into()) };
        eprintln!("flashnext tp link rank={rank} world={world}");
        if write { eprintln!("flashnext tp wire=write"); }
        Some((link, swap))
    } else {
        if tp_proxy::write_from_env()? { return Err("VADE_W2_TP_WIRE=write needs VADE_W2_TP=1".into()); }
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

/// VADE_W2_TP_WIRE=write: the byte RdmaWire::connect's negotiation sends is 1, so a peer on
/// today's wire fails its negotiation and this rank fails here (the link keeps the same timeouts).
#[cfg(target_os = "linux")]
fn agree_write_wire(link: &mut std::net::TcpStream, limit: std::time::Duration) -> Result<(), String> {
    use std::io::{Read, Write};
    let mut peer = [0u8];
    link.set_read_timeout(Some(limit)).and_then(|()| link.set_write_timeout(Some(limit)))
        .and_then(|()| link.write_all(&[2])).and_then(|()| link.read_exact(&mut peer)).map_err(|e| format!("VADE_W2_TP_WIRE agreement: {e}"))?;
    if peer[0] != 2 {
        return Err("VADE_W2_TP_WIRE is write here and unset on the peer rank: set it on both ranks or neither".into());
    }
    Ok(())
}

/// Send this rank's VADE_W2_TP (bit 0) and VADE_W2_TP_PROJ (bit 1) and read the peer's (one byte
/// each way, 30 s at most).
fn agree_tensor_split(link: &mut std::net::TcpStream, on: u8) -> Result<(), String> {
    use std::io::{Read, Write};
    let before = link.read_timeout().map_err(|e| e.to_string())?;
    link.set_read_timeout(Some(std::time::Duration::from_secs(30))).map_err(|e| e.to_string())?;
    let mut peer = [0u8];
    let agreed = link.write_all(&[on]).and_then(|()| link.read_exact(&mut peer)).map_err(|e| format!("VADE_W2_TP agreement: {e}"));
    link.set_read_timeout(before).map_err(|e| e.to_string())?;
    agreed?;
    if peer[0] != on {
        return Err(format!("VADE_W2_TP / VADE_W2_TP_PROJ are {}/{} here and {}/{} on the peer rank: set them on both ranks or neither",
            on & 1, on >> 1, peer[0] & 1, peer[0] >> 1));
    }
    Ok(())
}
