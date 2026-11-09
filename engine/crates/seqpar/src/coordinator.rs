//! Warm peer admission and prefill for a server retaining the final causal rank.
use crate::{Partition, Result, Ring, TransportStats, prefill::{PrefillStats, prefill}};
use crate::control::{authenticate_coordinator, authenticate_worker, control_secret, release_controls, RELEASE_TIMEOUT};
use spark_engine::qwen::{Model, Session, PREFILL_CHUNK};
use std::{
    io::{Read, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{atomic::{AtomicBool, Ordering}, mpsc, Arc, Weak},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const RECYCLE_TIMEOUT: Duration = Duration::from_secs(120);
const TIMEOUT: Duration = Duration::from_secs(120);
const ADMISSION_TIMEOUT: Duration = Duration::from_millis(1500);
const SETUP_TIMEOUT: Duration = Duration::from_secs(10);
const MAGIC: &[u8; 8] = b"SEQJOB01";

/// Ring listener and address of the next rank's listener, on their direct link.
#[derive(Clone, Copy, Debug)]
pub struct Endpoint { pub listen: SocketAddr, pub next: SocketAddr }

/// A warm worker and its data-plane position. List peers in physical order;
/// the server follows the final peer and owns the last prompt slice.
#[derive(Clone, Copy, Debug)]
pub struct Peer { pub control: SocketAddr, pub endpoint: Endpoint }

#[derive(Clone, Copy, Debug)]
pub enum Topology { Ring, Path }

pub struct Coordinator {
    pub endpoint: Endpoint,
    pub topology: Topology,
    /// Position-dependent attention cost divided by projection/FFN cost.
    /// Optional override; the default uses the selected attention kernel's cost.
    /// The total prompt weight is this coefficient times prompt length.
    pub attention_cost_per_token: Option<f64>,
    buffers: std::sync::Mutex<ExchangeBuffers>,
}

pub struct Output {
    pub logits: Vec<f32>,
    pub stats: PrefillStats,
    pub peers: Vec<PeerStats>,
    /// Includes peer reservation and prompt delivery; completion ACKs are optional.
    pub elapsed: Duration,
}

#[derive(Clone, Debug)]
pub struct PeerStats {
    pub rank: usize,
    pub elapsed: Duration,
    /// Allocation before release; the worker drops the entire request session.
    pub cache_bytes: usize,
    /// Worker's lifetime CUDA allocation peak, including model and KV.
    pub peak_device_bytes: usize,
}

type ExchangeBuffers = Arc<std::sync::Mutex<crate::transport::KvStaging>>;

enum NetworkEvent {
    Ready,
    Prefix(ExchangeBuffers, Duration, TransportStats),
    Complete(Vec<PeerStats>),
}

enum Stage { Admission, Begin, Project, Exchange, Import, FinishLayer, Complete, Done }

/// Cooperative final-rank prefill. CUDA work remains on the scheduler thread;
/// a host-only worker owns the sockets and the one reusable K/V staging pair.
/// Dropping this job disconnects its channels; it never joins a blocked peer.
pub struct Pending {
    partition: Partition,
    local_tokens: Vec<u32>,
    width: usize,
    layers: usize,
    layer: usize,
    imported: usize,
    stage: Stage,
    buffers: Option<ExchangeBuffers>,
    network: mpsc::Receiver<Result<NetworkEvent>>,
    recycle: mpsc::SyncSender<()>,
    lifetime: Arc<()>,
    cancelled: Arc<AtomicBool>,
    stats: PrefillStats,
    started: Instant,
    prefill_started: Option<Instant>,
    exchange_wait: Option<Instant>,
}

impl Pending {
    /// Remains live until both the scheduler job and its network worker exit.
    /// Admission guards can keep falling back locally while cancellation drains.
    pub fn lifetime(&self) -> Weak<()> { Arc::downgrade(&self.lifetime) }

    /// Advance through bounded GPU chunks, returning immediately when waiting
    /// for a peer. The complete gathered cache does not depend on a final ACK.
    pub fn advance(&mut self, session: &mut Session<'_, '_, '_>, budget: Duration) -> Result<Option<Output>> {
        let before = session.backend_stats();
        let mut result = self.advance_inner(session, budget);
        self.stats.add_backend(session.backend_stats().since(before));
        if let Ok(Some(output)) = &mut result { output.stats = self.stats.clone(); }
        result
    }
    fn advance_inner(&mut self, session: &mut Session<'_, '_, '_>, budget: Duration) -> Result<Option<Output>> {
        let turn = Instant::now();
        let local = self.partition.range(self.partition.world() - 1);
        loop {
            let remaining = budget.saturating_sub(turn.elapsed());
            if remaining.is_zero() { return Ok(None); }
            match self.stage {
                Stage::Admission => {
                    let Some(event) = self.poll()? else {
                        if self.started.elapsed() >= ADMISSION_TIMEOUT { return Err("coordinated prefill admission timed out".into()); }
                        return Ok(None);
                    };
                    if !matches!(event, NetworkEvent::Ready) { return Err("unexpected prefill admission response".into()); }
                    self.stage = Stage::Begin;
                }
                Stage::Begin => {
                    let phase = Instant::now();
                    self.prefill_started.get_or_insert(phase);
                    let done = session.sequence_begin_slice(&self.local_tokens, local.start, local.end,
                        self.partition.tokens(), remaining)?;
                    self.stats.begin += phase.elapsed();
                    if !done { return Ok(None); }
                    self.stage = Stage::Project;
                }
                Stage::Project => {
                    let phase = Instant::now();
                    let done = session.sequence_project_slice(self.layer, remaining)?;
                    self.stats.project += phase.elapsed();
                    if !done { return Ok(None); }
                    self.stage = Stage::Exchange;
                }
                Stage::Exchange => {
                    let Some(event) = self.poll()? else {
                        if self.exchange_wait.get_or_insert_with(Instant::now).elapsed() >= Duration::from_secs(5) {
                            return Err("coordinated prefill peer exchange timed out".into());
                        }
                        return Ok(None);
                    };
                    self.exchange_wait = None;
                    let NetworkEvent::Prefix(buffers, elapsed, transport) = event else {
                        return Err("unexpected prefill exchange response".into());
                    };
                    self.buffers = Some(buffers);
                    self.stats.exchange += elapsed;
                    self.stats.transport = transport;
                    self.imported = 0;
                    self.stage = Stage::Import;
                }
                Stage::Import => {
                    let end = (self.imported + PREFILL_CHUNK).min(local.start);
                    let buffers = self.buffers.as_ref().unwrap().lock().map_err(|_| "prefill buffers poisoned")?;
                    let (keys, values) = (&buffers.keys, &buffers.values);
                    let rows = self.imported * self.width..end * self.width;
                    let phase = Instant::now();
                    session.sequence_import(self.layer, self.imported, &keys[rows.clone()], &values[rows])?;
                    self.stats.import += phase.elapsed();
                    self.imported = end;
                    drop(buffers);
                    if end == local.start {
                        if self.layer + 1 < self.layers {
                            self.buffers = None;
                            self.recycle.try_send(()).map_err(|_| "prefill exchange worker stopped")?;
                        }
                        self.stage = Stage::FinishLayer;
                    }
                }
                Stage::FinishLayer => {
                    let phase = Instant::now();
                    let done = session.sequence_finish_layer_slice(self.layer, remaining)?;
                    self.stats.attention_ffn += phase.elapsed();
                    if !done { return Ok(None); }
                    self.layer += 1;
                    self.stage = if self.layer == self.layers { Stage::Complete } else { Stage::Project };
                }
                Stage::Complete => {
                    let phase = Instant::now();
                    let logits = session.sequence_finish()?;
                    let peers = match self.poll() {
                        Ok(Some(NetworkEvent::Complete(peers))) => peers,
                        _ => Vec::new(),
                    };
                    self.stats.head = phase.elapsed();
                    self.stats.elapsed = self.prefill_started.unwrap().elapsed();
                    self.stats.schedule_wait = self.stats.elapsed.saturating_sub(self.stats.begin + self.stats.project
                        + self.stats.import + self.stats.attention_ffn + self.stats.head);
                    self.stage = Stage::Done;
                    return Ok(Some(Output { logits, stats: self.stats.clone(), peers, elapsed: self.started.elapsed() }));
                }
                Stage::Done => return Err("coordinated prefill already completed".into()),
            }
        }
    }

    fn poll(&self) -> Result<Option<NetworkEvent>> {
        match self.network.try_recv() {
            Ok(event) => event.map(Some),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err("prefill exchange worker stopped".into()),
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) { self.cancelled.store(true, Ordering::Release); }
}

impl Coordinator {
    pub fn new(endpoint: Endpoint, topology: Topology) -> Self {
        Self { endpoint, topology, attention_cost_per_token: None,
            buffers: std::sync::Mutex::new(Arc::new(std::sync::Mutex::new(Default::default()))) }
    }

    /// Start admission without waiting for sockets or CUDA. The caller owns the
    /// session and resumes this job from its ordinary background-prefill queue.
    pub fn start(&self, session: &Session<'_, '_, '_>, tokens: &[u32], peers: &[Peer]) -> Result<Pending> {
        if peers.is_empty() { return Err("cooperative coordinated prefill requires peers".into()); }
        if !session.is_empty() || session.kv_format() != spark_gpu_probe::driver::KvFormat::Fp32 {
            return Err("coordinated prefill requires a fresh FP32 session".into());
        }
        let started = Instant::now();
        let weight = self.attention_cost_per_token.map_or_else(|| Partition::prefill_weight(tokens.len()),
            |cost| cost * tokens.len() as f64);
        let partition = Partition::prefill(tokens.len(), peers.len() + 1, weight)?;
        let local = partition.range(peers.len());
        let local_tokens = tokens[local.clone()].to_vec();
        let width = (session.config().kv_heads * session.config().head_dim) as usize;
        let layers = session.config().layers as usize;
        // The final causal rank never exports its own GPU-local K/V slice.
        let exchange_len = local.start.checked_mul(width).ok_or("K/V exchange size overflow")?;
        let host_exchange_bytes = exchange_len.checked_mul(8).ok_or("K/V exchange byte size overflow")?;
        let (events, network) = mpsc::sync_channel(1);
        let (recycle, recycled) = mpsc::sync_channel(1);
        let lifetime = Arc::new(());
        let running = lifetime.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let stopped = cancelled.clone();
        // A cancelled job may still own the previous buffers while its sockets
        // drain. Lend that storage again only after every borrower releases it.
        let buffers = {
            let pooled = self.buffers.lock().map_err(|_| "prefill buffer pool poisoned")?;
            if Arc::strong_count(&pooled) == 1 { pooled.clone() }
            else { Arc::new(std::sync::Mutex::new(Default::default())) }
        };
        let (endpoint, topology, signature) = (self.endpoint, self.topology, signature(session));
        let (prompt, peers, remote_partition) = (tokens.to_vec(), peers.to_vec(), partition.clone());
        std::thread::Builder::new().name("seqpar-exchange".into()).spawn(move || {
            let _running = running;
            let mut controls = Vec::with_capacity(peers.len());
            let result = (|| {
                let run_id = new_run_id()?;
                let deadline = started + ADMISSION_TIMEOUT;
                reserve(&peers, topology, &run_id, &remote_partition, &prompt, signature, deadline,
                    &stopped, &mut controls)?;
                if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
                let run_id = prompt_run_id(&run_id, &remote_partition, &prompt);
                drop(prompt);
                let setup = deadline.checked_duration_since(Instant::now()).ok_or("peer admission timed out")?;
                // Keep an owner on this thread until the ring deregisters, even
                // if cancellation drops the scheduler's last buffer reference.
                let mut ring = connect(peers.len(), peers.len() + 1, endpoint, topology, &run_id, setup)?;
                ring.retain_regions(true);
                if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
                send_network(&events, Ok(NetworkEvent::Ready), peers.len(), "ready")?;
                {
                    let mut storage = buffers.lock().map_err(|_| "prefill buffers poisoned")?;
                    storage.reserve(exchange_len);
                    unsafe { ring.prepare_staging(&mut storage)?; }
                }
                if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
                for layer in 0..layers {
                    if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
                    let phase = Instant::now();
                    {
                        let mut storage = buffers.lock().map_err(|_| "prefill buffers poisoned")?;
                        let storage = &mut *storage;
                        ring.exchange_prefix_kv(&mut storage.keys[..exchange_len], &mut storage.values[..exchange_len], &remote_partition, width)?;
                    }
                    send_network(&events, Ok(NetworkEvent::Prefix(buffers.clone(), phase.elapsed(), ring.stats)), peers.len(), "prefix")?;
                    if layer + 1 == layers { break; }
                    recycled.recv_timeout(RECYCLE_TIMEOUT).map_err(|e|
                        format!("seqpar peer {} layer {layer} recycle phase: {e}", peers.len()))?;
                }
                drop(ring);
                let peers = completions(&mut controls)?;
                release(&mut controls)?;
                controls.clear();
                let count = peers.len();
                send_network(&events, Ok(NetworkEvent::Complete(peers)), count, "complete")?;
                Ok::<(), String>(())
            })();
            let released = release(&mut controls);
            if let Err(error) = result.and(released) { let _ = send_network(&events, Err(error), peers.len(), "error"); }
            // The ring has closed. Keep the admission guard until every peer
            // has cleared its busy flag and closed its completion stream.
        }).map_err(|e| e.to_string())?;
        Ok(Pending { partition, local_tokens, width, layers, layer: 0, imported: 0,
            stage: Stage::Admission, buffers: None, network, recycle, lifetime, cancelled,
            stats: PrefillStats { host_exchange_bytes, ..Default::default() }, started, prefill_started: None, exchange_wait: None })
    }

    /// Prefill a fresh FP32 session using the supplied connected set of peers.
    /// Empty peers runs locally. Busy workers return an error before local GPU
    /// work starts; they are never queued behind another request. On any error,
    /// discard the local session before retrying or falling back to local prefill.
    /// All workers must have the same immutable checkpoint as the server.
    pub fn prefill(&self, session: &mut Session<'_, '_, '_>, tokens: &[u32], peers: &[Peer]) -> Result<Output> {
        if !session.is_empty() || session.kv_format() != spark_gpu_probe::driver::KvFormat::Fp32 {
            return Err("coordinated prefill requires a fresh FP32 session".into());
        }
        let start = Instant::now();
        if peers.is_empty() {
            let logits = session.prefill(tokens)?;
            let elapsed = start.elapsed();
            return Ok(Output { logits, stats: PrefillStats { elapsed, ..Default::default() },
                peers: Vec::new(), elapsed });
        }
        let world = peers.len() + 1;
        let weight = self.attention_cost_per_token.map_or_else(|| Partition::prefill_weight(tokens.len()),
            |cost| cost * tokens.len() as f64);
        let partition = Partition::prefill(tokens.len(), world, weight)?;
        let run_id = new_run_id()?;
        let mut controls = Vec::with_capacity(peers.len());
        let result = (|| {
            reserve(peers, self.topology, &run_id, &partition, tokens, signature(session),
                start + ADMISSION_TIMEOUT, &AtomicBool::new(false), &mut controls)?;
            let run_id = prompt_run_id(&run_id, &partition, tokens);
            let mut ring = connect(world - 1, world, self.endpoint, self.topology, &run_id, SETUP_TIMEOUT)?;
            let (logits, stats) = prefill(session, tokens, Some(&mut ring), &partition, true)?;
            drop(ring);
            let completed = completions(&mut controls)?;
            Ok(Output { logits, stats, peers: completed, elapsed: start.elapsed() })
        })();
        let released = release(&mut controls);
        result.and_then(|output| released.map(|()| output))
    }
}

fn send_network(events: &mpsc::SyncSender<Result<NetworkEvent>>, mut event: Result<NetworkEvent>,
    peer: usize, phase: &str) -> Result<()> {
    let deadline = Instant::now() + RECYCLE_TIMEOUT;
    loop {
        match events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(format!("seqpar peer {peer} {phase}: prefill cancelled")),
            Err(mpsc::TrySendError::Full(value)) => event = value,
        }
        if Instant::now() >= deadline { return Err(format!("seqpar peer {peer} {phase} event delivery timed out")); }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn new_run_id() -> Result<String> {
    Ok(format!("{}-{}", std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos()))
}

#[allow(clippy::too_many_arguments)]
fn reserve(peers: &[Peer], topology: Topology, run_id: &str, partition: &Partition,
    tokens: &[u32], signature: u64, deadline: Instant, stopped: &AtomicBool,
    controls: &mut Vec<TcpStream>) -> Result<()>
{
    let secret = control_secret()?;
    eprintln!("seqpar_partition tokens={} offsets={:?}", tokens.len(), partition.offsets());
    // Keep partial claims so cleanup waits for their release too.
    for (rank, peer) in peers.iter().enumerate() {
        if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
        let timeout = deadline.checked_duration_since(Instant::now()).ok_or("peer admission timed out")?;
        controls.push(TcpStream::connect_timeout(&peer.control, timeout).map_err(|e| e.to_string())?);
        let stream = controls.last_mut().unwrap();
        stream.set_nodelay(true).map_err(|e| e.to_string())?;
        if let Err(error) = authenticate_coordinator(stream, secret.as_deref(), deadline) {
            let rejected = controls.pop().unwrap();
            let _ = rejected.shutdown(Shutdown::Both);
            return Err(error);
        }
        let stream = controls.last_mut().unwrap();
        control_deadline(stream, deadline)?;
        read_reply(stream)?;
        if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
        control_deadline(stream, deadline)?;
        write_job(stream, rank, peer.endpoint, topology, run_id, partition, tokens, signature)?;
        control_deadline(stream, deadline)?;
        read_reply(stream)?;
    }
    for control in controls {
        if stopped.load(Ordering::Acquire) { return Err("prefill cancelled".into()); }
        control_deadline(control, deadline)?;
        control.write_all(b"G").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn release(controls: &mut [TcpStream]) -> Result<()> {
    release_controls(controls, Instant::now() + RELEASE_TIMEOUT)
}

fn completions(controls: &mut [TcpStream]) -> Result<Vec<PeerStats>> {
    let deadline = Instant::now() + ADMISSION_TIMEOUT;
    controls.iter_mut().enumerate().map(|(rank, control)| {
        control_deadline(control, deadline).map_err(|e| format!("seqpar peer {rank} completion: {e}"))?;
        let bytes = read_reply(control).map_err(|e| format!("seqpar peer {rank} completion: {e}"))?;
        if bytes.len() != 24 { return Err("invalid peer completion".into()); }
        Ok(PeerStats { rank,
            elapsed: Duration::from_nanos(u64::from_le_bytes(bytes[..8].try_into().unwrap())),
            cache_bytes: usize::try_from(u64::from_le_bytes(bytes[8..16].try_into().unwrap())).map_err(|_| "peer cache size overflow")?,
            peak_device_bytes: usize::try_from(u64::from_le_bytes(bytes[16..].try_into().unwrap())).map_err(|_| "peer peak size overflow")?,
        })
    }).collect()
}

fn control_deadline(stream: &TcpStream, deadline: Instant) -> Result<()> {
    let timeout = deadline.checked_duration_since(Instant::now()).ok_or("peer control timed out")?;
    stream.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())
}

/// Serve one request at a time using an already loaded full model. The accepting
/// thread answers busy immediately; all CUDA work stays on the calling thread.
/// Weights and host staging stay warm; device KV and scratch drop before the
/// completion reply, including failures. Authentication precedes reservation.
pub fn serve_peer(model: &Model<'_, '_>, listener: TcpListener) -> Result<()> {
    let secret = control_secret()?;
    if !listener.local_addr().map_err(|e| e.to_string())?.ip().is_loopback() && secret.is_none() {
        return Err("non-loopback worker control requires SEQPAR_CONTROL_SECRET".into());
    }
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let busy = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (tx, rx) = mpsc::sync_channel(1);
        let busy = &busy;
        scope.spawn(move || {
            loop {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5)); continue;
                    }
                    Err(e) => { eprintln!("seqpar worker accept: {e}"); break; }
                };
                if stream.set_nodelay(true).is_err() { continue; }
                let authenticating = Instant::now();
                if let Err(error) = authenticate_worker(&mut stream, secret.as_deref()) {
                    eprintln!("seqpar worker authentication rejected elapsed_ms={:.3}: {error}",
                        authenticating.elapsed().as_secs_f64() * 1000.0);
                    continue;
                }
                if configure(&stream).is_err() { continue; }
                if busy.swap(true, Ordering::AcqRel) {
                    let _ = write_reply(&mut stream, Err("peer busy"));
                } else if write_reply(&mut stream, Ok(&[])).is_err() {
                    busy.store(false, Ordering::Release);
                } else if tx.send(stream).is_err() { break; }
            }
        });
        let mut staging = crate::transport::KvStaging::default();
        for mut stream in rx {
            let result = run_peer(model, &mut stream, &mut staging);
            // Device state is released; host staging is retained for reuse.
            // Reopen admission before the ACK so
            // an immediate next request cannot spuriously observe this job busy.
            busy.store(false, Ordering::Release);
            let reply = result.as_ref().map(Vec::as_slice).map_err(String::as_str);
            let _ = write_reply(&mut stream, reply);
            if let Err(error) = result { eprintln!("seqpar worker job: {error}"); }
        }
        Err("peer listener stopped".into())
    })
}

fn run_peer(model: &Model<'_, '_>, stream: &mut TcpStream, staging: &mut crate::transport::KvStaging) -> Result<Vec<u8>> {
    let mut magic = [0; 8];
    stream.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != MAGIC { return Err("invalid seqpar job protocol".into()); }
    let rank = read_usize(stream)?;
    let world = read_usize(stream)?;
    let count = read_usize(stream)?;
    if world < 2 || rank >= world - 1 || count < world || count > model.max_tokens() {
        return Err("peer rank or prompt exceeds worker capacity".into());
    }
    let topology = match read_usize(stream)? { 0 => Topology::Ring, 1 => Topology::Path, _ => return Err("invalid topology".into()) };
    let endpoint = Endpoint { listen: read_text(stream)?.parse().map_err(|_| "invalid listen address")?,
        next: read_text(stream)?.parse().map_err(|_| "invalid next address")? };
    let run_id = read_text(stream)?;
    let mut session = model.session();
    if read_u64(stream)? != signature(&session) { return Err("peer model configuration differs".into()); }
    let offsets = (0..=world).map(|_| read_usize(stream)).collect::<Result<Vec<_>>>()?;
    let partition = Partition::from_offsets(offsets)?;
    if partition.tokens() != count || (0..world).any(|r| partition.range(r).is_empty())
        || partition.offsets()[1..world].iter().any(|n| n % Partition::ALIGNMENT != 0)
    {
        return Err("invalid peer partition".into());
    }
    let mut bytes = vec![0u8; count.checked_mul(4).ok_or("prompt byte count overflow")?];
    stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    let tokens = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    drop(bytes);
    if tokens.iter().any(|&t| u64::from(t) >= model.config().vocab_size) { return Err("prompt token outside vocabulary".into()); }
    session.configure_kv_bf16(false)?;
    write_reply(stream, Ok(&[]))?;
    let mut go = [0u8];
    stream.read_exact(&mut go).map_err(|e| e.to_string())?;
    if go != *b"G" { return Err("prefill cancelled".into()); }
    let run_id = prompt_run_id(&run_id, &partition, &tokens);
    let mut ring = connect(rank, world, endpoint, topology, &run_id, SETUP_TIMEOUT)?;
    std::mem::swap(&mut ring.staging, staging);
    let result = prefill(&mut session, &tokens, Some(&mut ring), &partition, true);
    std::mem::swap(&mut ring.staging, staging);
    let (_, stats) = result?;
    eprintln!("seqpar_worker rank={rank} tokens={count} elapsed_ms={:.3} kv_bytes={} peak_device_bytes={} host_exchange_bytes={}",
        stats.elapsed.as_secs_f64() * 1000.0, session.cache_bytes(), model.memory_bytes().1, stats.host_exchange_bytes);
    stats.report("worker", rank);
    let mut reply = Vec::with_capacity(24);
    put_u64(&mut reply, stats.elapsed.as_nanos().min(u64::MAX as u128) as u64);
    put_u64(&mut reply, session.cache_bytes() as u64);
    put_u64(&mut reply, model.memory_bytes().1 as u64);
    // Reset only clears the logical length. Drop is required to free peer KV.
    drop(session);
    drop(ring);
    Ok(reply)
}

fn connect(rank: usize, world: usize, endpoint: Endpoint, topology: Topology, run_id: &str, setup: Duration) -> Result<Ring> {
    let ring = Ring::connect_topology(rank, world, endpoint.listen, endpoint.next, run_id,
        matches!(topology, Topology::Path), setup.min(SETUP_TIMEOUT))?;
    eprintln!("seqpar_transport rank={rank} world={world} transport={} topology={}",
        if ring.uses_rdma() { "rdma" } else { "tcp" }, ring.topology());
    Ok(ring)
}

fn prompt_run_id(run_id: &str, partition: &Partition, tokens: &[u32]) -> String {
    let hash = partition.offsets().iter().map(|&n| n as u64).chain(tokens.iter().map(|&n| u64::from(n)))
        .fold(0xcbf29ce484222325u64, |h, n| (h ^ n).wrapping_mul(0x100000001b3));
    format!("{run_id}:{hash:016x}")
}

// Deployment supplies the same official checkpoint; this catches configuration
// mistakes, and deliberately does not reread or hash the model's weight bytes.
fn signature(session: &Session<'_, '_, '_>) -> u64 {
    let c = session.config();
    [c.layers, c.hidden_size, c.intermediate_size, c.attention_heads, c.kv_heads,
        c.head_dim, c.vocab_size, c.rms_norm_eps.to_bits(), c.rope_theta.to_bits()]
        .into_iter().fold(0xcbf29ce484222325, |h, n| (h ^ n).wrapping_mul(0x100000001b3))
}

fn configure(stream: &TcpStream) -> Result<()> {
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(TIMEOUT)).map_err(|e| e.to_string())?;
    stream.set_write_timeout(Some(TIMEOUT)).map_err(|e| e.to_string())
}

fn put_u64(bytes: &mut Vec<u8>, n: u64) { bytes.extend_from_slice(&n.to_le_bytes()); }
fn put_text(bytes: &mut Vec<u8>, text: &str) { put_u64(bytes, text.len() as u64); bytes.extend_from_slice(text.as_bytes()); }
fn read_u64(stream: &mut TcpStream) -> Result<u64> {
    let mut bytes = [0; 8];
    stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(bytes))
}
fn read_usize(stream: &mut TcpStream) -> Result<usize> { usize::try_from(read_u64(stream)?).map_err(|_| "job integer overflow".into()) }
fn read_text(stream: &mut TcpStream) -> Result<String> {
    let len = read_usize(stream)?;
    if len > 128 { return Err("job text too long".into()); }
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    String::from_utf8(bytes).map_err(|e| e.to_string())
}

#[allow(clippy::too_many_arguments)]
fn write_job(stream: &mut TcpStream, rank: usize, endpoint: Endpoint, topology: Topology,
    run_id: &str, partition: &Partition, tokens: &[u32], signature: u64) -> Result<()>
{
    let mut bytes = Vec::with_capacity(tokens.len() * 4 + 1024);
    bytes.extend_from_slice(MAGIC);
    for n in [rank, partition.world(), tokens.len(), usize::from(matches!(topology, Topology::Path))] { put_u64(&mut bytes, n as u64); }
    put_text(&mut bytes, &endpoint.listen.to_string());
    put_text(&mut bytes, &endpoint.next.to_string());
    put_text(&mut bytes, run_id);
    put_u64(&mut bytes, signature);
    for &offset in partition.offsets() { put_u64(&mut bytes, offset as u64); }
    for &token in tokens { bytes.extend_from_slice(&token.to_le_bytes()); }
    stream.write_all(&bytes).map_err(|e| e.to_string())
}

fn write_reply(stream: &mut TcpStream, result: std::result::Result<&[u8], &str>) -> Result<()> {
    let (status, bytes) = match result { Ok(bytes) => (0u8, bytes), Err(error) => (1, error.as_bytes()) };
    let bytes = &bytes[..bytes.len().min(4096)];
    stream.write_all(&[status]).map_err(|e| e.to_string())?;
    stream.write_all(&(bytes.len() as u32).to_le_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(bytes).map_err(|e| e.to_string())
}

fn read_reply(stream: &mut TcpStream) -> Result<Vec<u8>> {
    let mut header = [0u8; 5];
    stream.read_exact(&mut header).map_err(|e| e.to_string())?;
    let len = u32::from_le_bytes(header[1..].try_into().unwrap()) as usize;
    if len > 4096 { return Err("invalid peer reply size".into()); }
    let mut bytes = vec![0; len];
    stream.read_exact(&mut bytes).map_err(|e| e.to_string())?;
    if header[0] != 0 { return Err(String::from_utf8_lossy(&bytes).into_owned()); }
    Ok(bytes)
}
