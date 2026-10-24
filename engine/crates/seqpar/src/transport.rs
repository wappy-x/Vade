//! FP32 all-gather and fixed-rank-order reduction over ring neighbours.
//! TCP rendezvous and large-message verbs ABI derive from Vade's scale builder.
//! Transfers preserve bits and write directly into globally ordered host slices.
use crate::partition::Partition;
use engine_fabric::layout::PeerLayout;
use spark_engine::collective::{self, Bytes, Kind, Parts, Path, Profile};
use std::{
    io::{ErrorKind, Read, Write},
    net::{Shutdown, TcpStream},
    ops::Range,
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

#[path = "transport/connect.rs"]
mod connect;

#[path = "transport/gather.rs"]
mod gather;

#[cfg(target_os = "linux")]
#[path = "rdma.rs"]
mod rdma;
/// The RC send/receive endpoint (mailbox slot API) for other two-rank links (VADE_W2_PP_RDMA).
#[cfg(target_os = "linux")]
#[allow(unused_imports)] // used by spark-tp through the crate; unused where bins include this module by path
pub use rdma::Endpoint as RdmaEndpoint;

type Result<T> = std::result::Result<T, String>;
const TIMEOUT: Duration = Duration::from_secs(120);

// Callback intervals include enqueues and any completion wait, even on error.
// The residual public-call span is transport/peer wait and host orchestration.
fn timed<T>(enabled: bool, ns: &mut u128, work: impl FnOnce() -> Result<T>) -> Result<T> {
    if !enabled { return work(); }
    let start = Instant::now();
    let result = work();
    *ns += start.elapsed().as_nanos();
    result
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TransportStats {
    pub sent: u64,
    pub received: u64,
    pub control_sent: u64,
    pub control_received: u64,
    pub exchanges: u64,
    pub elapsed_ns: u128,
    pub send_ns: u128,
    pub recv_ns: u128,
    pub wait_ns: u128,
    pub producer_ns: u128,
    pub consumer_ns: u128,
}

impl TransportStats {
    pub fn since(self, previous: Self) -> Self {
        Self {
            sent: self.sent - previous.sent,
            received: self.received - previous.received,
            control_sent: self.control_sent - previous.control_sent,
            control_received: self.control_received - previous.control_received,
            exchanges: self.exchanges - previous.exchanges,
            elapsed_ns: self.elapsed_ns - previous.elapsed_ns,
            send_ns: self.send_ns - previous.send_ns,
            recv_ns: self.recv_ns - previous.recv_ns,
            wait_ns: self.wait_ns - previous.wait_ns,
            producer_ns: self.producer_ns - previous.producer_ns,
            consumer_ns: self.consumer_ns - previous.consumer_ns,
        }
    }
}

/// Request-independent host K/V storage. Registrations drop before their buffers.
#[cfg(target_os = "linux")]
#[derive(Default)]
pub struct KvStaging {
    regions: Vec<std::sync::Arc<rdma::Region>>,
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
}
#[cfg(target_os = "linux")]
impl KvStaging {
    pub(crate) fn reserve(&mut self, len: usize) {
        if self.keys.len() < len {
            self.regions.clear();
            self.keys.resize(len, 0.0);
            self.values.resize(len, 0.0);
        }
    }
}

pub struct Ring {
    rank: usize,
    world: usize,
    link: PeerLayout,
    right: Option<TcpStream>,
    left: Option<TcpStream>,
    poisoned: bool,
    trace_transfers: bool,
    profile_layers: bool,
    setup_deadline: Option<Instant>,
    #[cfg(target_os = "linux")]
    retain_regions: bool,
    #[cfg(target_os = "linux")]
    registrations_sealed: bool,
    pub stats: TransportStats,
    profile: Option<Box<Profile>>,
    #[cfg(target_os = "linux")]
    endpoints: [Option<rdma::Endpoint>; 2],
    #[cfg(target_os = "linux")]
    auxiliary: Option<rdma::Endpoint>,
    /// Second rail of each path link (right, left); mailbox messages of 64 KiB
    /// and more are split across both rails of a cable.
    #[cfg(target_os = "linux")]
    second: [Option<rdma::Endpoint>; 2],
    #[cfg(target_os = "linux")]
    pub(crate) staging: KvStaging,
    /// Registrations of `collective`; declared first so they drop before it.
    #[cfg(target_os = "linux")]
    collective_regions: Vec<std::sync::Arc<rdma::Region>>,
    /// Ring-owned host buffer for collectives outside retained-region phases.
    /// It is registered once per rail and reused, never per collective. Words
    /// keep FP32 views of it aligned.
    #[cfg(target_os = "linux")]
    collective: Vec<u64>,
    /// Staged collectives skip the TCP readiness byte: a send that reaches a
    /// peer before its receive is posted is retried by the RC queue pair.
    #[cfg(target_os = "linux")]
    rdma_ready_only: bool,
    /// Two receive slots per neighbour stay posted between collectives.
    #[cfg(target_os = "linux")]
    mailbox: [Option<Mailbox>; 2],
    #[cfg(target_os = "linux")]
    mailbox2: [Option<Mailbox>; 2],
}

#[cfg(any(target_os = "linux", test))]
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct Mailbox { slot: usize, offset: usize, next: usize, sends: usize, arrived: std::collections::VecDeque<(usize, usize)> }

#[cfg(any(target_os = "linux", test))]
trait SlotSender {
    fn poll(&mut self, sends: &mut usize, arrived: &mut std::collections::VecDeque<(usize, usize)>) -> Result<()>;
    unsafe fn send(&mut self, packet: *mut u8, bytes: usize) -> Result<()>;
}
#[cfg(target_os = "linux")]
impl SlotSender for rdma::Endpoint {
    fn poll(&mut self, sends: &mut usize, arrived: &mut std::collections::VecDeque<(usize, usize)>) -> Result<()> {
        self.poll_slots(sends, arrived)
    }
    unsafe fn send(&mut self, packet: *mut u8, bytes: usize) -> Result<()> { unsafe { self.post_slot_send(packet, bytes) } }
}
#[cfg(any(target_os = "linux", test))]
impl Mailbox {
    fn wait_send(&mut self, endpoint: &mut impl SlotSender, deadline: Instant) -> Result<()> {
        while self.sends != 0 {
            endpoint.poll(&mut self.sends, &mut self.arrived)?;
            if Instant::now() >= deadline { return Err("RDMA mailbox send timed out".into()); }
        }
        Ok(())
    }
    // Caller keeps this receive slot unposted until wait_send completes.
    unsafe fn relay(&mut self, endpoint: &mut impl SlotSender, packet: *mut u8, bytes: usize, deadline: Instant) -> Result<()> {
        self.wait_send(endpoint, deadline)?;
        unsafe { endpoint.send(packet, bytes)?; }
        self.sends += 1;
        Ok(())
    }
}

#[cfg(any(target_os = "linux", test))]
fn register_relay_slots<E>(endpoints: &mut [Option<E>; 2], mailboxes: &[Option<Mailbox>; 2],
    mut register: impl FnMut(&mut E, usize, usize) -> Result<()>) -> Result<()> {
    for (endpoint, source) in endpoints.iter_mut().zip(mailboxes.iter().rev()) {
        if let (Some(endpoint), Some(m)) = (endpoint, source) { register(endpoint, m.offset, 2 * m.slot)?; }
    }
    Ok(())
}

// Packet metadata includes its owner so only the exact outgoing range can relay.
#[cfg(any(target_os = "linux", test))]
type SlotPacket = (usize, *const u8, usize, usize);
#[cfg(any(target_os = "linux", test))]
fn relay_packet(k: usize, owner: usize, bytes: usize, dual: [bool; 2], arrived: &[Option<SlotPacket>; 4]) -> Option<*const u8> {
    let first = Ring::first_rail(bytes, dual[k / 2]);
    if first != Ring::first_rail(bytes, dual[1 - k / 2]) { return None; }
    let len = if k % 2 == 0 { first } else { bytes - first };
    let (_, packet, n, rank) = arrived[k ^ 2]?;
    (len != 0 && n == len && rank == owner).then_some(packet)
}
#[cfg(any(target_os = "linux", test))]
unsafe fn copy_relay(k: usize, bytes: usize, dual: [bool; 2], arrived: &[Option<SlotPacket>; 4], slot: &mut [u8]) {
    let at = if k % 2 == 0 { 0 } else { Ring::first_rail(bytes, dual[k / 2]) };
    let source = 1 - k / 2;
    let split = Ring::first_rail(bytes, dual[source]);
    for (rail, from) in [0, split].into_iter().enumerate() {
        if let Some((_, packet, n, _)) = arrived[2 * source + rail] {
            let lo = at.max(from); let hi = (at + slot.len()).min(from + n);
            if hi > lo { unsafe { parallel_copy(std::slice::from_raw_parts(packet.add(lo - from), hi - lo), &mut slot[lo - at..hi - at]); } }
        }
    }
}

#[cfg(target_os = "linux")]
impl Mailbox {
    // The registered allocation contains two receive slots and one send slot.
    // The caller keeps it alive until the endpoint is destroyed.
    unsafe fn post(&mut self, endpoint: &mut rdma::Endpoint, base: *mut u8,
        bytes: usize, deadline: Instant, fill: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<()> {
        if bytes > self.slot { return Err("collective exceeds the RDMA mailbox slot".into()); }
        if bytes == 0 { return Ok(()); }
        self.wait_send(endpoint, deadline)?;
        unsafe {
            let send = base.add(self.offset + 2 * self.slot);
            fill(std::slice::from_raw_parts_mut(send, bytes))?;
            endpoint.post_slot_send(send, bytes)?;
        }
        self.sends += 1;
        Ok(())
    }

    unsafe fn receive(&mut self, endpoint: &mut rdma::Endpoint, base: *mut u8,
        bytes: usize, deadline: Instant, receive: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
        if bytes > self.slot { return Err("collective exceeds the RDMA mailbox slot".into()); }
        if bytes == 0 { return Ok(()); }
        while self.arrived.is_empty() {
            endpoint.poll_slots(&mut self.sends, &mut self.arrived)?;
            if Instant::now() >= deadline { return Err("RDMA mailbox receive timed out".into()); }
            std::hint::spin_loop();
        }
        let (index, arrived) = self.arrived.pop_front().unwrap();
        if index != self.next || arrived != bytes { return Err("RDMA mailbox message out of order or wrong length".into()); }
        unsafe {
            let slot = base.add(self.offset + index * self.slot);
            receive(std::slice::from_raw_parts(slot, bytes))?;
            endpoint.post_slot_receive(slot, self.slot, index)?;
        }
        self.next ^= 1;
        Ok(())
    }
}

/// Host copies and sums above 1 MiB run on up to eight threads (one core
/// moves a 7 MB prompt partial in about a millisecond); each element's
/// arithmetic is unchanged.
#[cfg(any(target_os = "linux", test))]
fn parallel_chunks<A: Send, B: Sync>(a: &mut [A], b: &[B], f: impl Fn(&mut [A], &[B]) + Sync) {
    let threads = (std::mem::size_of_val(a) >> 20).clamp(1, 8);
    if threads == 1 || a.len() != b.len() { return f(a, b); }
    let chunk = a.len().div_ceil(threads);
    let mut chunks=a.chunks_mut(chunk).zip(b.chunks(chunk));
    let first=chunks.next();
    std::thread::scope(|s| {
        for (a,b) in chunks { let f=&f; s.spawn(move || f(a,b)); }
        if let Some((a,b))=first { f(a,b); }
    });
}
#[cfg(any(target_os = "linux", test))]
fn parallel_copy(source: &[u8], destination: &mut [u8]) {
    parallel_chunks(destination, source, |destination, source| destination.copy_from_slice(source));
}

impl Ring {
    pub fn rank(&self) -> usize { self.rank }
    pub fn world(&self) -> usize { self.world }
    pub fn topology(&self) -> &'static str { if self.link.is_path() { "path" } else { "ring" } }
    pub fn collective_profiling(&self) -> bool { self.profile.is_some() }
    fn collective<T>(&mut self, kind: Kind, work: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        if self.profile.is_none() { return work(self); }
        let mut path = if self.uses_rdma() { Path::Rdma } else { Path::Tcp };
        #[cfg(target_os = "linux")]
        if self.mailbox.iter().any(Option::is_some) { path = Path::Mailbox; }
        if matches!(kind, Kind::Barrier) { path = Path::Tcp; }
        let before = self.stats;
        let start = Instant::now();
        let result = work(self);
        let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let stats = self.stats;
        self.profile.as_mut().unwrap().record(kind, path, ns, Parts {
            producer_ns: (stats.producer_ns - before.producer_ns).min(u64::MAX as u128) as u64,
            consumer_ns: (stats.consumer_ns - before.consumer_ns).min(u64::MAX as u128) as u64,
        }, Bytes {
            sent: stats.sent - before.sent, received: stats.received - before.received,
            control_sent: stats.control_sent - before.control_sent,
            control_received: stats.control_received - before.control_received,
            pieces: stats.exchanges - before.exchanges,
        }, result.is_ok());
        result
    }
    pub fn profile_flash(&mut self, ns: u64) {
        if let Some(profile) = &mut self.profile { profile.record_flash(ns); }
    }
    pub fn profile_point(&mut self, kind: usize, ns: u64, ok: bool) {
        if let Some(profile) = &mut self.profile { profile.record_point(kind, ns, ok); }
    }
    pub fn profile_begin_command(&mut self) {
        if let Some(profile) = &mut self.profile { profile.begin_command(); }
    }
    pub fn profile_command(&mut self, phase: &'static str, rows: usize, path: Path, ns: u64, bytes: Bytes, ok: bool) {
        if let Some(profile) = &mut self.profile {
            if phase != "command" { profile.command_phase(phase, rows); }
            profile.record(Kind::Command, path, ns, Parts::default(), bytes, ok);
        }
    }
    pub fn profile_begin_step(&mut self, phase: &'static str, rows: usize) {
        if let Some(profile) = &mut self.profile { profile.begin_step(phase, rows); }
    }
    pub fn profile_end_step(&mut self, ok: bool, device_bytes: usize) {
        if let Some(profile) = &mut self.profile { profile.end_step(self.rank, self.world, ok, device_bytes); }
    }
    pub fn profile_begin_tick(&mut self, tick: u64) {
        if let Some(profile) = &mut self.profile { profile.begin_tick(tick); }
    }
    pub fn profile_end_tick(&mut self, device_bytes: usize) {
        if let Some(profile) = &mut self.profile { profile.end_tick(self.rank, self.world, device_bytes); }
    }
    /// Cancel peer collectives and stop DMA before their borrowed buffers leave scope.
    pub fn abort(&mut self) {
        self.poisoned = true;
        for stream in self.right.iter().chain(self.left.iter()) { let _ = stream.shutdown(Shutdown::Both); }
        #[cfg(target_os = "linux")]
        { self.endpoints = [None, None]; self.auxiliary = None; self.second = [None, None]; }
    }
    fn rounds(&self) -> usize { if self.link.is_path() { self.world - 1 } else { self.world / 2 } }
    pub fn uses_rdma(&self) -> bool {
        #[cfg(target_os = "linux")]
        { self.endpoints.iter().any(Option::is_some) }
        #[cfg(not(target_os = "linux"))]
        { false }
    }

    /// Finish transport setup before loading weights. Every rail may reuse
    /// existing regions, but a cache miss must fail before calling ibv_reg_mr.
    pub fn seal_registrations(&mut self) -> Result<()> {
        if self.poisoned { return Err("cannot seal an unusable ring".into()); }
        #[cfg(target_os = "linux")]
        {
            if self.retain_regions || (self.uses_rdma() && self.mailbox.iter().all(Option::is_none)) {
                return Err("RDMA registrations require a fixed mailbox before sealing".into());
            }
            for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) {
                endpoint.seal_registrations();
            }
            self.registrations_sealed = true;
        }
        eprintln!("registrations sealed before weights rank={} world={} rdma={}", self.rank, self.world, self.uses_rdma());
        Ok(())
    }

    /// Prefill keeps its staging allocations stable across layers. Clear the
    /// registrations on both transitions, before those allocations are freed.
    #[cfg(target_os = "linux")]
    pub fn retain_regions(&mut self, retain: bool) {
        for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) { endpoint.clear_regions(); }
        self.retain_regions = retain;
    }

    #[cfg(target_os = "linux")]
    pub(crate) unsafe fn prepare_staging(&mut self, staging: &mut KvStaging) -> Result<()> {
        for buffer in [&mut staging.keys, &mut staging.values] {
            let (base, bytes) = (buffer.as_mut_ptr().cast::<u8>(), buffer.len() * 4);
            // Both rails cover the retained capacity, so changing prompt sizes
            // cannot turn their changing split point into new registrations.
            for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) {
                unsafe { endpoint.register_cached(base, bytes, &mut staging.regions)?; }
            }
        }
        Ok(())
    }

    /// Fault in and register the existing staging pair during setup. The caller
    /// keeps both allocations fixed until retain_regions(false) or ring drop.
    #[cfg(target_os = "linux")]
    pub unsafe fn prepare_prefix_regions(&mut self, k: &mut [f32], v: &mut [f32]) -> Result<()> {
        let rails = 1 + usize::from(self.auxiliary.is_some());
        for buffer in [k, v] {
            let (base, bytes) = (buffer.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(buffer));
            let primary = bytes / rails;
            for endpoint in self.endpoints.iter_mut().flatten() {
                unsafe { endpoint.register(base, primary)?; }
            }
            if let Some(endpoint) = self.auxiliary.as_mut() {
                unsafe { endpoint.register(base.add(primary), bytes - primary)?; }
            }
        }
        Ok(())
    }

    /// TCP-only rendezvous with one deadline across all hops; no RDMA registration.
    pub fn barrier_with_timeout(&mut self, timeout: Duration) -> Result<()> {
        let previous = self.setup_deadline.replace(Instant::now() + timeout);
        let result = self.barrier();
        self.setup_deadline = previous;
        result
    }

    pub fn barrier(&mut self) -> Result<()> {
        self.collective(Kind::Barrier, |ring| ring.barrier_inner())
    }
    fn barrier_inner(&mut self) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        if self.world == 1 { return Ok(()); }
        let mut bytes = vec![0; self.world];
        bytes[self.rank] = 1;
        let result = self.gather_tcp(&mut bytes, &(0..=self.world).collect::<Vec<_>>(), false);
        if result.is_err() { self.abort(); }
        result
    }

    /// Local rank's token-major K and V slices must already occupy their global
    /// positions. Each output is tokens * width FP32 values. No reduction occurs.
    pub fn exchange_kv(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize) -> Result<()> {
        self.exchange(k, v, partition, width, false)
    }

    /// Gather only the causal prefix each rank consumes. Ring neighbours relay
    /// blocks along their shortest route; a path sends prefixes toward rank N-1.
    /// Suffix slots may remain untouched and must not be consumed by attention.
    pub fn exchange_prefix_kv(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize) -> Result<()> {
        self.exchange(k, v, partition, width, true)
    }

    /// Host prefix needed for local export/import and every relay on this rank.
    /// The final rank's local slice stays on its GPU and needs no host slots.
    pub fn prefix_tokens(&self, partition: &Partition) -> Result<usize> {
        if partition.world() != self.world { return Err("K/V partition shape mismatch".into()); }
        let local = partition.range(self.rank);
        let mut end = if self.rank + 1 < self.world { local.end } else { local.start };
        for step in 0..self.rounds() {
            for range in self.ranges(step, partition.offsets(), true) {
                if !range.is_empty() { end = end.max(range.end); }
            }
        }
        Ok(end)
    }

    fn exchange_offsets(&self, partition: &Partition, width: usize, len: usize, prefix: bool) -> Result<Vec<usize>> {
        if partition.world() != self.world || width == 0 { return Err("K/V partition shape mismatch".into()); }
        let full = partition.tokens().checked_mul(width).ok_or("K/V partition size overflow")?;
        let required = if prefix { self.prefix_tokens(partition)? * width } else { full };
        if len < required || len > full || len % width != 0 { return Err("K/V partition shape mismatch".into()); }
        // Every nonempty routed range ends within required. Clipping later
        // offsets therefore changes only slots this rank never transfers.
        Ok(partition.offsets().iter().map(|n| (n * width).min(len)).collect())
    }

    fn exchange(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize, prefix: bool) -> Result<()> {
        self.collective(if prefix { Kind::PrefixKv } else { Kind::Kv }, |ring|
            ring.exchange_inner(k, v, partition, width, prefix))
    }
    fn exchange_inner(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize, prefix: bool) -> Result<()> {
        if k.len() != v.len() { return Err("K/V partition shape mismatch".into()); }
        let offsets = self.exchange_offsets(partition, width, k.len(), prefix)?.into_iter().map(|n| n * 4).collect::<Vec<_>>();
        unsafe { self.gather_buffers(&[(k.as_mut_ptr().cast(), std::mem::size_of_val(k)),
            (v.as_mut_ptr().cast(), std::mem::size_of_val(v))], &offsets, prefix, None) }
    }

    /// Run work on the caller's CUDA context thread while the host-only ring
    /// progresses on a scoped worker. Work may consume only already-local KV.
    /// Both operations complete before the host staging borrows are released.
    pub fn exchange_kv_while<T>(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize,
        work: impl FnOnce() -> Result<T>) -> Result<T>
    {
        std::thread::scope(|scope| {
            let exchange = scope.spawn(|| self.exchange_kv(k, v, partition, width));
            let result = work();
            exchange.join().map_err(|_| "K/V exchange worker panicked".to_string())??;
            result
        })
    }

    /// As above, but transmit only the causal prefixes required by prefill.
    pub fn exchange_prefix_kv_while<T>(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize,
        work: impl FnOnce() -> Result<T>) -> Result<T>
    {
        std::thread::scope(|scope| {
            let exchange = scope.spawn(|| self.exchange_prefix_kv(k, v, partition, width));
            let result = work();
            exchange.join().map_err(|_| "K/V exchange worker panicked".to_string())??;
            result
        })
    }

    /// Receive the next layer while producing this rank's local K/V. Only the
    /// local slices reach work; transport cannot read them until work returns.
    /// Both branches finish before any staging allocation can be reused.
    pub fn exchange_prefix_kv_producing<T>(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize,
        work: impl FnOnce(&mut [f32], &mut [f32]) -> Result<T>) -> Result<T>
    {
        self.collective(Kind::PrefixKv, |ring| ring.exchange_prefix_kv_producing_inner(k, v, partition, width, work))
    }
    fn exchange_prefix_kv_producing_inner<T>(&mut self, k: &mut [f32], v: &mut [f32], partition: &Partition, width: usize,
        work: impl FnOnce(&mut [f32], &mut [f32]) -> Result<T>) -> Result<T>
    {
        if k.len() != v.len() { return Err("K/V partition shape mismatch".into()); }
        let offsets = self.exchange_offsets(partition, width, k.len(), true)?;
        let local = partition.range(self.rank);
        let (offset, count) = (local.start * width, local.len() * width);
        if offset + count > k.len() { return Err("K/V producer local slice is outside staging".into()); }
        let bytes = std::mem::size_of_val(k);
        let offsets = offsets.into_iter().map(|n| n * 4).collect::<Vec<_>>();
        let (kp, vp) = (k.as_mut_ptr() as usize, v.as_mut_ptr() as usize);
        let (ready, pending) = mpsc::channel();
        std::thread::scope(|scope| {
            let exchange = scope.spawn(move || unsafe {
                self.gather_buffers(&[(kp as *mut u8, bytes), (vp as *mut u8, bytes)], &offsets, true, Some(&pending))
            });
            // Receive ranges never include the local owner. No whole-buffer
            // reference or local send reference exists while these are mutable.
            let result = unsafe { work(std::slice::from_raw_parts_mut((kp as *mut f32).add(offset), count),
                std::slice::from_raw_parts_mut((vp as *mut f32).add(offset), count)) };
            if result.is_ok() { let _ = ready.send(()); }
            drop(ready); // Also cancel a gated transfer after a producer error.
            let transferred = exchange.join().map_err(|_| "K/V exchange worker panicked".to_string()).and_then(|r| r);
            result.and_then(|value| transferred.map(|()| value))
        })
    }

    /// offsets are FP32 element displacements, including the final buffer length.
    pub fn all_gather_f32(&mut self, buffer: &mut [f32], offsets: &[usize]) -> Result<()> {
        self.collective(Kind::GatherF32, |ring| ring.all_gather_f32_inner(buffer, offsets))
    }
    fn all_gather_f32_inner(&mut self, buffer: &mut [f32], offsets: &[usize]) -> Result<()> {
        #[cfg(target_os = "linux")]
        if self.staged() {
            if offsets.len() != self.world + 1 || offsets[0] != 0 || offsets[self.world] != buffer.len()
                || offsets.windows(2).any(|w| w[0] > w[1])
            { return Err("all-gather offsets do not partition the buffer".into()); }
            let offsets = offsets.iter().map(|n| n * 4).collect::<Vec<_>>();
            // Plain FP32 values; byte views preserve every bit.
            let bytes = unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast::<u8>(), buffer.len() * 4) };
            return self.gather_staged(bytes, &offsets);
        }
        self.gather_f32(buffer, offsets, false)
    }

    /// Gather rank-owned official-dtype rows without FP32 expansion on the wire.
    /// Offsets partition bytes, allowing empty ranks and mixed BF16/FP8 rows.
    pub fn all_gather_bytes(&mut self, buffer: &mut [u8], offsets: &[usize]) -> Result<()> {
        self.collective(Kind::GatherBytes, |ring| ring.all_gather_bytes_inner(buffer, offsets))
    }
    fn all_gather_bytes_inner(&mut self, buffer: &mut [u8], offsets: &[usize]) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        if offsets.len() != self.world + 1 || offsets.first() != Some(&0)
            || offsets.last() != Some(&buffer.len()) || offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err("all-gather offsets do not partition the byte buffer".into());
        }
        #[cfg(target_os = "linux")]
        if self.staged() { return self.gather_staged(buffer, offsets); }
        unsafe { self.gather_bytes(buffer.as_mut_ptr(), buffer.len(), offsets, false, None) }
    }

    #[cfg(target_os = "linux")]
    fn staged(&self) -> bool { self.world > 1 && self.uses_rdma() && !self.retain_regions }

    /// Route every later collective through two pre-posted receive slots of
    /// `slot` bytes per neighbour. All ranks call this at
    /// the same point, before any mailbox collective; retained-region phases and
    /// messages above `slot` are refused afterwards. TCP rings are unchanged.
    #[cfg(target_os = "linux")]
    pub fn enable_mailbox(&mut self, slot: usize) -> Result<()> {
        if !self.uses_rdma() || self.mailbox.iter().any(Option::is_some) { return Ok(()); }
        if self.registrations_sealed { return Err("FAIL RDMA mailbox registration attempted after sealing".into()); }
        if self.retain_regions || slot == 0 || slot > u32::MAX as usize { return Err("invalid RDMA mailbox".into()); }
        self.collective_regions.clear();
        let links = self.endpoints.iter().flatten().count() + self.second.iter().flatten().count();
        let bytes = slot.checked_mul(3 * links).ok_or("RDMA mailbox size overflow")?;
        self.collective = vec![0; bytes.div_ceil(8)];
        let base = self.collective.as_mut_ptr().cast::<u8>();
        if self.link.duplex() && self.auxiliary.is_some() {
            // The rails scatter into disjoint stripes of the SAME receive
            // slots. The GPU still sees one contiguous, unchanged byte range.
            let stripe = slot.div_ceil(32).div_ceil(64) * 64;
            for (rail, (endpoint, mailbox)) in [(&mut self.endpoints[0], &mut self.mailbox[0]),
                (&mut self.auxiliary, &mut self.mailbox2[0])].into_iter().enumerate() {
                let endpoint = endpoint.as_mut().ok_or("mailbox lost its RDMA rail")?;
                unsafe {
                    endpoint.register_cached(base, 3 * slot, &mut self.collective_regions)?;
                    for index in 0..2 {
                        endpoint.post_slot_stripes(base.add(index * slot), slot, stripe, rail, Some(index))?;
                    }
                }
                *mailbox = Some(Mailbox { slot, offset:0, next:0, sends:0, arrived:Default::default() });
            }
            return Ok(());
        }
        let mut offset = 0;
        for (endpoint, mailbox) in self.endpoints.iter_mut().zip(&mut self.mailbox)
            .chain(self.second.iter_mut().zip(&mut self.mailbox2)) {
            let Some(endpoint) = endpoint else { continue; };
            unsafe {
                endpoint.register_cached(base.add(offset), 3 * slot, &mut self.collective_regions)?;
                for index in 0..2 { endpoint.post_slot_receive(base.add(offset + index * slot), slot, index)?; }
            }
            *mailbox = Some(Mailbox { slot, offset, next: 0, sends: 0, arrived: Default::default() });
            offset += 3 * slot;
        }
        // Relay sends read the opposite neighbour's receive slots. Register
        // those on the outgoing HCA now, before weights and registration sealing.
        for (endpoints, mailboxes) in [(&mut self.endpoints, &self.mailbox), (&mut self.second, &self.mailbox2)] {
            register_relay_slots(endpoints, mailboxes, |endpoint, offset, bytes| unsafe {
                endpoint.register_cached(base.add(offset), bytes, &mut self.collective_regions)
            })?;
        }
        Ok(())
    }

    /// Exchange this rank's range for the peer's, through the mailbox slots.
    #[cfg(target_os = "linux")]
    fn gather_mailbox(&mut self, buffer: &mut [u8], offsets: &[usize]) -> Result<()> {
        if !self.link.duplex() { return self.gather_neighbour_mailboxes(buffer, offsets); }
        if offsets.len() != 3 || offsets[0] > offsets[1] || offsets[1] > offsets[2] || offsets[2] > buffer.len() {
            return Err("mailbox gather needs two ordered ranges inside the buffer".into());
        }
        let (own, peer) = (offsets[self.rank]..offsets[self.rank + 1], offsets[1 - self.rank]..offsets[2 - self.rank]);
        // Safety: the two ranges are disjoint parts of `buffer`; `own` is read
        // into the send slot before `receive` writes `peer`.
        let base = buffer.as_mut_ptr();
        let own_bytes = unsafe { std::slice::from_raw_parts(base.add(own.start), own.len()) };
        let peer_bytes = unsafe { std::slice::from_raw_parts_mut(base.add(peer.start), peer.len()) };
        self.exchange_mailbox(own.len(), peer.len(), |slot| { parallel_copy(own_bytes, slot); Ok(()) },
            |arrived| { parallel_copy(arrived, peer_bytes); Ok(()) })
    }

    /// Two-rank sum through the mailbox: the own values go straight to the
    /// send slot and the peer's are added from their receive slot. Every
    /// element is rank 0's value + rank 1's value, as after a gather.
    #[cfg(target_os = "linux")]
    fn reduce_mailbox(&mut self, values: &mut [f32]) -> Result<()> {
        let (own, bytes, second) = (values.as_ptr().cast::<u8>(), values.len() * 4, self.rank == 1);
        // Safety: `own` is copied into the send slot before `receive` writes `values`.
        let own = unsafe { std::slice::from_raw_parts(own, bytes) };
        self.exchange_mailbox(bytes, bytes, |slot| { parallel_copy(own, slot); Ok(()) }, |arrived| {
            let peer = unsafe { std::slice::from_raw_parts(arrived.as_ptr().cast::<f32>(), arrived.len() / 4) };
            parallel_chunks(values, peer, |values, peer| for (value, other) in values.iter_mut().zip(peer) {
                *value = if second { *other + *value } else { *value + *other };
            });
            Ok(())
        })
    }

    /// Pipeline an ordered sum down the path, then broadcast it back. Each
    /// element is still ((rank 0 + rank 1) + rank 2) ...; small pieces let
    /// neighbouring links progress together without spawning copy workers.
    #[cfg(target_os = "linux")]
    fn reduce_path_mailboxes(&mut self, values: &mut [f32]) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        const PIECE: usize = 65536;
        let profiling = self.profile.is_some();
        let start = Instant::now();
        let wall = if self.trace_transfers { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()) } else { 0 };
        let before = self.stats;
        let result = (|| -> Result<()> {
            let base = self.collective.as_mut_ptr().cast::<u8>();
            for (endpoint, mailbox) in self.endpoints.iter_mut().zip(&self.mailbox).chain(self.second.iter_mut().zip(&self.mailbox2)) {
                if let (Some(endpoint), Some(m)) = (endpoint, mailbox) {
                    if m.slot < PIECE * 4 || m.slot % 4 != 0 { return Err("path reduction needs aligned mailbox slots of at least 256 KiB".into()); }
                    unsafe { endpoint.register_cached(base.add(m.offset), 3 * m.slot, &mut self.collective_regions)?; }
                }
            }
            let deadline = start + TIMEOUT;
            // Each element is prior + own; the first rail carries the piece's head.
            let merge = |arrived: &[u8], piece: &mut [f32], broadcast: bool| {
                let prefix = unsafe { std::slice::from_raw_parts(arrived.as_ptr().cast::<f32>(), piece.len()) };
                if broadcast { piece.copy_from_slice(prefix); }
                else { for (value, prior) in piece.iter_mut().zip(prefix) { *value = *prior + *value; } }
            };
            for broadcast in [false, true] {
                let source = usize::from(!broadcast);
                let destination = 1 - source;
                for piece in values.chunks_mut(PIECE) {
                    let bytes = std::mem::size_of_val(piece);
                    let from = Self::first_rail(bytes, self.second[source].is_some());
                    let to = Self::first_rail(bytes, self.second[destination].is_some());
                    let (head, tail) = piece.split_at_mut(from / 4);
                    if let (Some(endpoint), Some(m)) = (&mut self.endpoints[source], &mut self.mailbox[source]) {
                        unsafe { m.receive(endpoint, base, from, deadline, |arrived| timed(profiling, &mut self.stats.consumer_ns, || { merge(arrived, head, broadcast); Ok(()) }))?; }
                        self.stats.received += bytes as u64;
                    }
                    if let (Some(endpoint), Some(m)) = (&mut self.second[source], &mut self.mailbox2[source]) {
                        unsafe { m.receive(endpoint, base, bytes - from, deadline, |arrived| timed(profiling, &mut self.stats.consumer_ns, || { merge(arrived, tail, broadcast); Ok(()) }))?; }
                    }
                    if let (Some(endpoint), Some(m)) = (&mut self.endpoints[destination], &mut self.mailbox[destination]) {
                        unsafe { m.post(endpoint, base, to, deadline, |slot| timed(profiling, &mut self.stats.producer_ns, || { parallel_copy(std::slice::from_raw_parts(piece.as_ptr().cast(), to), slot); Ok(()) }))?; }
                        self.stats.sent += bytes as u64;
                    }
                    if let (Some(endpoint), Some(m)) = (&mut self.second[destination], &mut self.mailbox2[destination]) {
                        unsafe { m.post(endpoint, base, bytes - to, deadline, |slot| timed(profiling, &mut self.stats.producer_ns, || { parallel_copy(std::slice::from_raw_parts(piece.as_ptr().cast::<u8>().add(to), bytes - to), slot); Ok(()) }))?; }
                    }
                }
            }
            Ok(())
        })();
        if result.is_err() { self.abort(); }
        let elapsed = start.elapsed().as_nanos();
        self.stats.elapsed_ns += elapsed;
        self.stats.exchanges += 1;
        if self.trace_transfers {
            let delta = self.stats.since(before);
            println!("seqpar_transfer rank={} exchange={} rdma=true sent_bytes={} received_bytes={} control_sent_bytes=0 control_received_bytes=0 elapsed_ns={elapsed} post_ns=0 wait_ns={elapsed} start_unix_ns={wall} mailbox=true path_reduce=true ok={}",
                self.rank, self.stats.exchanges, delta.sent, delta.received, result.is_ok());
        }
        result
    }

    /// Host mailbox allocation bytes across every neighbour and rail. Multi-rank
    /// mailboxes consume unified memory even without the pair-only GPU handoff.
    #[cfg(target_os = "linux")]
    pub fn mailbox_bytes(&self) -> usize {
        if self.mailbox.iter().any(Option::is_some) { self.collective.len() * 8 } else { 0 }
    }

    /// Stable host memory across all neighbours/rails, for GPU slot callbacks.
    #[cfg(target_os = "linux")]
    pub fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> {
        if self.poisoned || !self.mailbox.iter().any(Option::is_some) { return None; }
        Some((self.collective.as_mut_ptr().cast(), self.mailbox_bytes()))
    }

    /// Two-rank exchange without host copies: `fill` writes this rank's
    /// `own` bytes into the send slot, and `receive` consumes the peer's `peer`
    /// bytes from their receive slot, finishing before it returns (the slot is
    /// posted again afterwards). Both ranks call this at the same point with
    /// mirrored lengths; an empty side sends or receives nothing.
    #[cfg(target_os = "linux")]
    pub fn exchange_slots(&mut self, own: usize, peer: usize, fill: impl FnOnce(&mut [u8]) -> Result<()>,
        receive: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
        self.collective(Kind::Slots, |ring| ring.exchange_slots_inner(own, peer, fill, receive))
    }
    #[cfg(target_os = "linux")]
    fn exchange_slots_inner(&mut self, own: usize, peer: usize, fill: impl FnOnce(&mut [u8]) -> Result<()>,
        receive: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
        if !self.link.duplex() || self.mailbox[0].is_none() { return Err("slot exchange needs a two-rank RDMA mailbox".into()); }
        self.exchange_mailbox(own, peer, fill, receive)
    }

    /// Send `own_len` bytes that `fill` writes into the send slot and hand the
    /// peer's `peer_len` bytes to `receive` while they are still in their receive slot.
    #[cfg(target_os = "linux")]
    fn exchange_mailbox(&mut self, own_len: usize, peer_len: usize, fill: impl FnOnce(&mut [u8]) -> Result<()>,
        receive: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        let start = Instant::now();
        let wall = if self.trace_transfers { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()) } else { 0 };
        let peer = 0..peer_len;
        let (mut producer_ns, mut consumer_ns) = (0, 0);
        let result = (|| -> Result<()> {
            if self.auxiliary.is_some() {
                return self.exchange_striped_mailbox(own_len, peer_len, fill, receive);
            }
            let m = self.mailbox[0].as_mut().unwrap();
            if own_len > m.slot || peer.len() > m.slot { return Err("collective exceeds the RDMA mailbox slot".into()); }
            let (slot, base) = (m.slot, self.collective.as_mut_ptr().cast::<u8>());
            let endpoint = self.endpoints[0].as_mut().ok_or("mailbox lost its RDMA rail")?;
            unsafe { endpoint.register_cached(base, 3 * slot, &mut self.collective_regions)?; }
            let deadline = start + TIMEOUT;
            let mut posted = Instant::now();
            let mut arrived = posted;
            unsafe {
                m.post(endpoint, base, own_len, deadline, |slot| {
                    let phase = Instant::now();
                    let result = fill(slot);
                    producer_ns = phase.elapsed().as_nanos();
                    result?;
                    posted = Instant::now();
                    Ok(())
                })?;
                m.receive(endpoint, base, peer.len(), deadline, |slot| {
                    arrived = Instant::now();
                    let result = receive(slot);
                    consumer_ns = arrived.elapsed().as_nanos();
                    result?;
                    Ok(())
                })?;
            }
            if self.trace_transfers {
                while m.sends != 0 {
                    endpoint.poll_slots(&mut m.sends, &mut m.arrived)?;
                    if Instant::now() >= deadline { return Err("RDMA mailbox send timed out".into()); }
                }
                println!("seqpar_rail rank={} exchange={} rail=0 sent_bytes={} received_bytes={} completion_ns={} receive_ns={} producer_ns={} consumer_ns={} mailbox_bytes={}",
                    self.rank, self.stats.exchanges + 1, own_len, peer_len, posted.elapsed().as_nanos(),
                    arrived.saturating_duration_since(posted).as_nanos(), producer_ns, consumer_ns, self.collective.len() * 8);
            }
            Ok(())
        })();
        if result.is_err() { self.abort(); }
        let elapsed = start.elapsed().as_nanos();
        if self.profile.is_some() {
            self.stats.producer_ns += producer_ns;
            self.stats.consumer_ns += consumer_ns;
        }
        self.stats.elapsed_ns += elapsed;
        self.stats.exchanges += 1;
        self.stats.sent += own_len as u64;
        self.stats.received += peer.len() as u64;
        if self.trace_transfers {
            println!("seqpar_transfer rank={} exchange={} rdma=true sent_bytes={} received_bytes={} control_sent_bytes=0 control_received_bytes=0 elapsed_ns={elapsed} post_ns=0 wait_ns={elapsed} start_unix_ns={wall} mailbox=true ok={}",
                self.rank, self.stats.exchanges, own_len, peer.len(), result.is_ok());
        }
        result
    }

    /// Ordered pieces whose GPU producers and consumers return owned completions.
    /// The next producer is queued before the current consumer: its NIC send
    /// can run while that consumer executes. Every completion is drained on
    /// error, before aborting DMA or releasing the registered mailbox.
    #[cfg(target_os = "linux")]
    pub fn exchange_slots_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.collective(Kind::Pipeline, |ring| ring.exchange_slots_pipeline_inner(sizes, fill, receive, wait))
    }
    #[cfg(target_os = "linux")]
    fn exchange_slots_pipeline_inner<T>(&mut self, sizes: &[(usize, usize)],
        mut fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        mut receive: impl FnMut(usize, &[u8]) -> Result<T>,
        mut wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        if !self.link.duplex() || self.mailbox[0].is_none() { return Err("slot exchange needs a two-rank RDMA mailbox".into()); }
        if self.auxiliary.is_none() {
            let wait = std::cell::RefCell::new(wait);
            for (i, &(own, peer)) in sizes.iter().enumerate() {
                self.exchange_slots_inner(own, peer, |slot| (wait.borrow_mut())(fill(i, slot)?),
                    |slot| (wait.borrow_mut())(receive(i, slot)?))?;
            }
            return Ok(());
        }
        let start = Instant::now();
        let result = self.exchange_striped_pipeline(sizes, &mut fill, &mut receive, &mut wait);
        self.stats.elapsed_ns += start.elapsed().as_nanos();
        self.stats.exchanges += sizes.len() as u64;
        self.stats.sent += sizes.iter().map(|s| s.0 as u64).sum::<u64>();
        self.stats.received += sizes.iter().map(|s| s.1 as u64).sum::<u64>();
        if result.is_err() { self.abort(); }
        result
    }

    #[cfg(target_os = "linux")]
    fn exchange_striped_mailbox(&mut self, own: usize, peer: usize,
        fill: impl FnOnce(&mut [u8]) -> Result<()>, receive: impl FnOnce(&[u8]) -> Result<()>) -> Result<()> {
        let (mut fill, mut receive) = (Some(fill), Some(receive));
        self.exchange_striped_pipeline(&[(own, peer)],
            &mut |_, slot| fill.take().unwrap()(slot),
            &mut |_, slot| receive.take().unwrap()(slot), &mut |()| Ok(()))
    }

    #[cfg(target_os = "linux")]
    fn exchange_striped_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: &mut impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: &mut impl FnMut(usize, &[u8]) -> Result<T>,
        wait: &mut impl FnMut(T) -> Result<()>) -> Result<()> {
        let slot = self.mailbox[0].as_ref().ok_or("missing pair mailbox")?.slot;
        if sizes.iter().any(|&(own, peer)| own > slot || peer > slot) { return Err("collective exceeds the RDMA mailbox slot".into()); }
        let stripe = slot.div_ceil(32).div_ceil(64) * 64;
        let base = self.collective.as_mut_ptr().cast::<u8>();
        let profiling = self.profile.is_some();
        let (mut producer_total, mut consumer_total) = (0, 0);
        let mut producer = None;
        let mut consumer = None;
        let result = (|| -> Result<()> {
            let mut rails = [(&mut self.endpoints[0], &mut self.mailbox[0]),
                (&mut self.auxiliary, &mut self.mailbox2[0])].map(|(e, m)| (e.as_mut().unwrap(), m.as_mut().unwrap()));
            let deadline = Instant::now() + TIMEOUT;
            // Both send CQs own the single send slot until they complete.
            for (endpoint, m) in &mut rails {
                while m.sends != 0 {
                    endpoint.poll_slots(&mut m.sends, &mut m.arrived)?;
                    if Instant::now() >= deadline { return Err("RDMA mailbox send timed out".into()); }
                }
            }
            if let Some(&(own, _)) = sizes.first().filter(|s| s.0 != 0) {
                producer = Some(timed(profiling, &mut producer_total, || fill(0, unsafe { std::slice::from_raw_parts_mut(base.add(2 * slot), own) }))?);
            }
            for (i, &(own, peer)) in sizes.iter().enumerate() {
                let deadline = Instant::now() + TIMEOUT;
                let ready = Instant::now();
                let produced = if let Some(event) = producer.take() { wait(event) } else { Ok(()) };
                let producer_ns = ready.elapsed().as_nanos();
                if profiling { producer_total += producer_ns; }
                produced?;
                let posted = Instant::now();
                if own != 0 {
                    for (rail, (endpoint, m)) in rails.iter_mut().enumerate() {
                        unsafe { endpoint.post_slot_stripes(base.add(2 * slot), own, stripe, rail, None)?; }
                        m.sends += 1;
                    }
                }
                // Post the next send before waiting for the preceding GPU
                // consumer. Its receive slot cannot be reposted until this event.
                if let Some((index, event)) = consumer.take() {
                    timed(profiling, &mut consumer_total, || wait(event))?;
                    for (rail, (endpoint, _)) in rails.iter_mut().enumerate() {
                        unsafe { endpoint.post_slot_stripes(base.add(index * slot), slot, stripe, rail, Some(index))?; }
                    }
                }
                let mut completion_ns = [0; 2];
                let mut receive_ns = [0; 2];
                loop {
                    let mut done = true;
                    for (rail, (endpoint, m)) in rails.iter_mut().enumerate() {
                        endpoint.poll_slots(&mut m.sends, &mut m.arrived)?;
                        if own != 0 && m.sends == 0 && completion_ns[rail] == 0 { completion_ns[rail] = posted.elapsed().as_nanos(); }
                        if peer != 0 && !m.arrived.is_empty() && receive_ns[rail] == 0 { receive_ns[rail] = posted.elapsed().as_nanos(); }
                        done &= m.sends == 0 && (peer == 0 || !m.arrived.is_empty());
                    }
                    // Prefetch the next piece as soon as the send slot is free,
                    // ahead of this piece's add/finish kernels on the GPU stream.
                    if producer.is_none() && rails.iter().all(|(_, m)| m.sends == 0) {
                        if let Some(&(next, _)) = sizes.get(i + 1).filter(|s| s.0 != 0) {
                            producer = Some(timed(profiling, &mut producer_total, || fill(i + 1, unsafe { std::slice::from_raw_parts_mut(base.add(2 * slot), next) }))?);
                        }
                    }
                    if done { break; }
                    if Instant::now() >= deadline { return Err("RDMA striped mailbox timed out".into()); }
                    std::hint::spin_loop();
                }
                let rail_bytes = |bytes: usize, rail: usize| {
                    (rail * stripe..bytes).step_by(2 * stripe).map(|at| (bytes - at).min(stripe)).sum::<usize>()
                };
                let queued = Instant::now();
                if peer != 0 {
                    let index = rails[0].1.next;
                    for (rail, (_, m)) in rails.iter_mut().enumerate() {
                        let (arrived_index, arrived_bytes) = m.arrived.pop_front().unwrap();
                        if arrived_index != index || m.next != index || arrived_bytes != rail_bytes(peer, rail) {
                            return Err("RDMA mailbox message out of order or wrong length".into());
                        }
                        m.next ^= 1;
                    }
                    consumer = Some((index, timed(profiling, &mut consumer_total, || receive(i, unsafe { std::slice::from_raw_parts(base.add(index * slot), peer) }))?));
                }
                if self.trace_transfers {
                    let consumer_enqueue_ns = queued.elapsed().as_nanos();
                    for rail in 0..2 {
                        println!("seqpar_rail rank={} exchange={} rail={} sent_bytes={} received_bytes={} completion_ns={} receive_ns={} producer_ns={} consumer_enqueue_ns={} mailbox_bytes={}",
                            self.rank, self.stats.exchanges + i as u64 + 1, rail, rail_bytes(own, rail), rail_bytes(peer, rail),
                            completion_ns[rail], receive_ns[rail], producer_ns, consumer_enqueue_ns, self.collective.len() * 8);
                    }
                }
            }
            if let Some((index, event)) = consumer.take() {
                timed(profiling, &mut consumer_total, || wait(event))?;
                for (rail, (endpoint, _)) in rails.iter_mut().enumerate() {
                    unsafe { endpoint.post_slot_stripes(base.add(index * slot), slot, stripe, rail, Some(index))?; }
                }
            }
            Ok(())
        })();
        // A failed poll/length check/callback must not leave GPU DMA accessing
        // mailbox memory after the caller aborts the transport or unregisters it.
        let mut drained = Ok(());
        if let Some(event) = producer {
            drained = timed(profiling, &mut producer_total, || wait(event));
        }
        if let Some((_, event)) = consumer {
            let result = timed(profiling, &mut consumer_total, || wait(event));
            if drained.is_ok() { drained = result; }
        }
        if profiling {
            self.stats.producer_ns += producer_total;
            self.stats.consumer_ns += consumer_total;
        }
        result.and(drained)
    }

    /// Copy through the ring's registered buffer. Growth replaces the buffer
    /// and its registrations; no other registration happens per collective.
    #[cfg(target_os = "linux")]
    fn stage(&mut self, bytes: usize) -> Result<*mut u8> {
        if self.collective.len() * 8 < bytes {
            if self.registrations_sealed { return Err(format!("FAIL RDMA staging registration attempted after sealing: bytes={bytes}")); }
            self.collective_regions.clear();
            self.collective = vec![0; bytes.next_power_of_two().max(4096) / 8];
        }
        let (base, capacity) = (self.collective.as_mut_ptr().cast::<u8>(), self.collective.len() * 8);
        for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) {
            unsafe { endpoint.register_cached(base, capacity, &mut self.collective_regions)?; }
        }
        Ok(base)
    }

    #[cfg(target_os = "linux")]
    fn gather_staged(&mut self, buffer: &mut [u8], offsets: &[usize]) -> Result<()> {
        if buffer.is_empty() { return Ok(()); }
        if self.mailbox.iter().any(Option::is_some) { return self.gather_mailbox(buffer, offsets); }
        let base = self.stage(buffer.len())?;
        unsafe {
            std::ptr::copy_nonoverlapping(buffer.as_ptr(), base, buffer.len());
            self.rdma_ready_only = true;
            let result = self.gather_bytes(base, buffer.len(), offsets, false, None);
            self.rdma_ready_only = false;
            result?;
            std::ptr::copy_nonoverlapping(base, buffer.as_mut_ptr(), buffer.len());
        }
        Ok(())
    }

    /// Gather partials, then add rank 1 through rank N-1 to rank 0 in FP32.
    /// Every rank obtains the same result, independent of network arrival order.
    /// Scratch holds at least world * values.len() FP32 values and is reusable.
    pub fn all_reduce_f32(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        self.collective(Kind::ReduceF32, |ring| ring.all_reduce_f32_inner(values, scratch))
    }
    fn all_reduce_f32_inner(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        let profiling = self.profile.is_some();
        let started = Instant::now();
        let before = self.stats.elapsed_ns;
        let width = values.len();
        let count = width.checked_mul(self.world).ok_or("reduction scratch size overflow")?;
        let partials = scratch.get_mut(..count).ok_or("reduction scratch is too short")?;
        let offsets = (0..=self.world).map(|rank| rank * width).collect::<Vec<_>>();
        #[cfg(target_os = "linux")]
        if self.link.duplex() && self.mailbox[0].is_some() && width != 0 { return self.reduce_mailbox(values); }
        #[cfg(target_os = "linux")]
        if self.link.is_path() && self.world > 2 && width >= 262144 && self.mailbox.iter().any(Option::is_some) {
            return self.reduce_path_mailboxes(values);
        }
        #[cfg(target_os = "linux")]
        let partials = if self.staged() && width != 0 && self.mailbox.iter().all(Option::is_none) {
            // Gather in the registered buffer itself; the ordered sum below is unchanged.
            let base = self.stage(count * 4)?;
            let staged = unsafe { std::slice::from_raw_parts_mut(base.cast::<f32>(), count) };
            timed(profiling, &mut self.stats.producer_ns, || { staged[self.rank * width..(self.rank + 1) * width].copy_from_slice(values); Ok(()) })?;
            let bytes = offsets.iter().map(|n| n * 4).collect::<Vec<_>>();
            self.rdma_ready_only = true;
            let result = unsafe { self.gather_bytes(base, count * 4, &bytes, false, None) };
            self.rdma_ready_only = false;
            result?;
            staged
        } else {
            timed(profiling, &mut self.stats.producer_ns, || { partials[self.rank * width..(self.rank + 1) * width].copy_from_slice(values); Ok(()) })?;
            self.all_gather_f32_inner(partials, &offsets)?;
            partials
        };
        #[cfg(not(target_os = "linux"))]
        {
            timed(profiling, &mut self.stats.producer_ns, || { partials[self.rank * width..(self.rank + 1) * width].copy_from_slice(values); Ok(()) })?;
            self.all_gather_f32_inner(partials, &offsets)?;
        }
        if width == 0 { return Ok(()); }
        let sum = profiling.then(Instant::now);
        values.copy_from_slice(&partials[..width]);
        for partial in partials[width..].chunks_exact(width) {
            #[cfg(target_os = "linux")]
            parallel_chunks(values, partial, |values, partial| {
                for (value, addend) in values.iter_mut().zip(partial) { *value += *addend; }
            });
            #[cfg(not(target_os = "linux"))]
            for (value, addend) in values.iter_mut().zip(partial) { *value += *addend; }
        }
        if let Some(sum) = sum { self.stats.consumer_ns += sum.elapsed().as_nanos(); }
        // Gather already accounts for its span; include the ordered host sum
        // without counting that span twice in per-layer collective profiles.
        self.stats.elapsed_ns += started.elapsed().as_nanos().saturating_sub(self.stats.elapsed_ns - before);
        Ok(())
    }

    /// Backend waits for its CUDA stream and the device-to-host copy before the
    /// ring reads any partial; the result is uploaded on that same ordered stream.
    #[cfg(target_os = "linux")]
    pub fn all_reduce_device_f32(&mut self, backend: &spark_gpu_probe::driver::Backend<'_>,
        values: &mut spark_gpu_probe::driver::DeviceVector<'_>, scratch: &mut [f32]) -> Result<()>
    {
        backend.exchange_vector(values, |values| self.all_reduce_f32(values, scratch))
    }

    /// Local output must already occupy offsets[rank]..offsets[rank + 1] in the
    /// full device vector. This shares the synchronized boundary used by reduce.
    #[cfg(target_os = "linux")]
    pub fn all_gather_device_f32(&mut self, backend: &spark_gpu_probe::driver::Backend<'_>,
        values: &mut spark_gpu_probe::driver::DeviceVector<'_>, offsets: &[usize]) -> Result<()>
    {
        backend.exchange_vector(values, |values| self.all_gather_f32(values, offsets))
    }

    fn gather_f32(&mut self, buffer: &mut [f32], offsets: &[usize], prefix: bool) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        if offsets.len() != self.world + 1 || offsets[0] != 0 || offsets[self.world] != buffer.len()
            || offsets.windows(2).any(|w| w[0] > w[1])
        { return Err("all-gather offsets do not partition the buffer".into()); }
        if self.world == 1 || buffer.is_empty() { return Ok(()); }
        let offsets = offsets.iter().map(|n| n * 4).collect::<Vec<_>>();
        unsafe { self.gather_bytes(buffer.as_mut_ptr().cast(), std::mem::size_of_val(buffer), &offsets, prefix, None) }
    }

    // All ranges are validated by the caller. Until ready, only nonlocal receive
    // ranges may be referenced; the producer exclusively owns the local range.
    unsafe fn gather_bytes(&mut self, base: *mut u8, bytes: usize, offsets: &[usize], prefix: bool, ready: Option<&Receiver<()>>) -> Result<()> {
        unsafe { self.gather_buffers(&[(base, bytes)], offsets, prefix, ready) }
    }

    unsafe fn gather_buffers(&mut self, buffers: &[(*mut u8, usize)], offsets: &[usize], prefix: bool, ready: Option<&Receiver<()>>) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        // Posted mailbox receives would capture this path's sends.
        #[cfg(target_os = "linux")]
        if self.mailbox.iter().any(Option::is_some) { return Err("ring collective outside the enabled RDMA mailbox".into()); }
        if self.world == 1 || buffers.iter().all(|&(_, bytes)| bytes == 0) { return Ok(()); }
        let start = Instant::now();
        let wall = if self.trace_transfers { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()) } else { 0 };
        let before = self.stats;
        let tcp = |ring: &mut Self| -> Result<()> {
            for (index, &(base, _)) in buffers.iter().enumerate() {
                unsafe { ring.gather_tcp_raw(base, offsets, prefix, if index == 0 { ready } else { None })?; }
            }
            Ok(())
        };
        #[cfg(target_os = "linux")]
        let result = if self.uses_rdma() {
            // Even an unwind must stop DMA before releasing the staging borrow.
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { self.gather_rdma(buffers, offsets, prefix, ready) })) {
                Ok(result) => result,
                Err(panic) => {
                    self.abort();
                    std::panic::resume_unwind(panic)
                }
            }
        } else { tcp(self) };
        #[cfg(not(target_os = "linux"))]
        let result = tcp(self);
        if result.is_err() { self.abort(); }
        self.stats.elapsed_ns += start.elapsed().as_nanos();
        self.stats.exchanges += buffers.len() as u64;
        if self.trace_transfers {
            let delta = self.stats.since(before);
            let ns = delta.elapsed_ns.max(1) as f64;
            // Effective payload rates include registration, readiness and peer wait.
            println!("seqpar_transfer rank={} exchange={} rdma={} sent_bytes={} received_bytes={} control_sent_bytes={} control_received_bytes={} elapsed_ns={} post_ns={} wait_ns={} start_unix_ns={wall} send_gbps={:.6} recv_gbps={:.6} ok={}",
                self.rank, self.stats.exchanges, self.uses_rdma(), delta.sent, delta.received,
                delta.control_sent, delta.control_received, delta.elapsed_ns, delta.send_ns + delta.recv_ns, delta.wait_ns,
                delta.sent as f64 / ns, delta.received as f64 / ns, result.is_ok());
        }
        if self.profile_layers {
            let delta = self.stats.since(before);
            spark_engine::collective::write_profile(&format!("transfer kind=seqpar_layer rank={} sent_bytes={} received_bytes={} elapsed_s={:.9}\n",
                self.rank, delta.sent, delta.received, delta.elapsed_ns as f64 * 1e-9));
        }
        result
    }

    fn ranges(&self, step: usize, offsets: &[usize], prefix: bool) -> [Range<usize>; 4] {
        let owner = |shift: isize, clockwise: bool| {
            let rank = self.rank as isize + shift;
            if self.link.is_path() && !(0..self.world as isize).contains(&rank) { return 0..0; }
            let rank = rank.rem_euclid(self.world as isize) as usize;
            if prefix {
                // Keep a relay even if this recipient does not consume it when
                // a later hop on the same route still needs this owner's KV.
                let last = if self.link.is_path() { self.world - 1 } else if clockwise { self.world / 2 } else { (self.world - 1) / 2 };
                let needed = if self.link.is_path() { clockwise } else { (step + 1..=last).any(|hop| {
                    let recipient = if clockwise { (rank + hop) % self.world } else { (rank + self.world - hop) % self.world };
                    recipient > rank
                }) };
                if !needed { return 0..0; }
            }
            offsets[rank]..offsets[rank + 1]
        };
        let clockwise = self.right.is_some();
        let counter = (self.link.is_path() || step < (self.world - 1) / 2) && self.left.is_some();
        [if clockwise { owner(-(step as isize), true) } else { 0..0 }, if counter { owner(step as isize, false) } else { 0..0 },
            if self.left.is_some() { owner(-(step as isize) - 1, true) } else { 0..0 },
            if self.right.is_some() && (self.link.is_path() || step < (self.world - 1) / 2) { owner(step as isize + 1, false) } else { 0..0 }]
    }

    fn gather_tcp(&mut self, bytes: &mut [u8], offsets: &[usize], prefix: bool) -> Result<()> {
        unsafe { self.gather_tcp_raw(bytes.as_mut_ptr(), offsets, prefix, None) }
    }

    unsafe fn gather_tcp_raw(&mut self, base: *mut u8, offsets: &[usize], prefix: bool, ready: Option<&Receiver<()>>) -> Result<()> {
        if self.poisoned { return Err("ring is unusable after an earlier collective failure".into()); }
        for step in 0..self.rounds() {
            let [sr, sl, rl, rr] = self.ranges(step, offsets, prefix);
            // The receive owners are distinct from each other and both sends.
            // Round zero sends the immutable local owner in both directions.
            unsafe {
                let send = |r: Range<usize>| std::slice::from_raw_parts(base.add(r.start), r.len());
                let recv = |r: Range<usize>| std::slice::from_raw_parts_mut(base.add(r.start), r.len());
                let received = if step == 0 {
                    if let Some(ready) = ready { receive_until_ready(self.right.as_ref(), self.left.as_ref(), recv(rl.clone()), recv(rr.clone()), ready, &mut self.stats)? }
                    else { [0, 0] }
                } else { [0, 0] };
                pump(self.right.as_ref(), self.left.as_ref(), send(sr), send(sl),
                    recv(rl.start + received[0]..rl.end), recv(rr.start + received[1]..rr.end), &mut self.stats, self.setup_deadline)?;
            }
        }
        Ok(())
    }

    #[cfg(target_os = "linux")]
    unsafe fn gather_rdma(&mut self, buffers: &[(*mut u8, usize)], offsets: &[usize], prefix: bool, ready: Option<&Receiver<()>>) -> Result<()> {
        let rails = 1 + usize::from(self.auxiliary.is_some());
        let stripe = |range: &Range<usize>, rail: usize| {
            range.start + range.len() * rail / rails..range.start + range.len() * (rail + 1) / rails
        };
        for step in 0..self.rounds() {
            let [sr, sl, rl, rr] = self.ranges(step, offsets, prefix);
            let posting = Instant::now();
            for &(base, bytes) in buffers { unsafe {
                if self.link.duplex() {
                    self.endpoints[0].as_mut().unwrap().post(base, bytes, 0..0, stripe(&rl, 0))?;
                } else {
                    if let Some(endpoint) = &mut self.endpoints[0] { endpoint.post(base, bytes, 0..0, stripe(&rr, 0))?; }
                    if let Some(endpoint) = &mut self.endpoints[1] { endpoint.post(base, bytes, 0..0, stripe(&rl, 0))?; }
                }
                if let Some(endpoint) = &mut self.auxiliary {
                    endpoint.post(base, bytes, 0..0, stripe(if rl.is_empty() { &rr } else { &rl }, 1))?;
                }
            } }
            self.stats.recv_ns += posting.elapsed().as_nanos();
            // Exchange readiness only after both QPs have receives. This prevents
            // receiver-not-ready retries when ranks finish their GPU work unevenly.
            let wait = Instant::now();
            let (mut ready_left, mut ready_right) = ([0u8; 1], [0u8; 1]);
            let sr_ready = self.right.is_some();
            let sl_ready = self.left.is_some() && !self.link.duplex();
            let rl_ready = self.left.is_some();
            let rr_ready = self.right.is_some() && !self.link.duplex();
            let mut control = TransportStats::default();
            let ready_result = if self.rdma_ready_only { Ok(()) } else { pump(self.right.as_ref(), self.left.as_ref(), if sr_ready { &[1] } else { &[] }, if sl_ready { &[1] } else { &[] },
                if rl_ready { &mut ready_left } else { &mut [] }, if rr_ready { &mut ready_right } else { &mut [] }, &mut control, self.setup_deadline) };
            self.stats.control_sent += control.sent;
            self.stats.control_received += control.received;
            ready_result?;
            if step == 0 {
                if let Some(ready) = ready { ready.recv_timeout(TIMEOUT).map_err(|_| "K/V producer failed or timed out".to_string())?; }
            }
            self.stats.wait_ns += wait.elapsed().as_nanos();
            let posting = Instant::now();
            for &(base, bytes) in buffers { unsafe {
                if let Some(endpoint) = &mut self.endpoints[0] { endpoint.post(base, bytes, stripe(&sr, 0), 0..0)?; }
                if let Some(endpoint) = &mut self.endpoints[1] { endpoint.post(base, bytes, stripe(&sl, 0), 0..0)?; }
                if let Some(endpoint) = &mut self.auxiliary {
                    endpoint.post(base, bytes, stripe(if sr.is_empty() { &sl } else { &sr }, 1), 0..0)?;
                }
            } }
            self.stats.send_ns += posting.elapsed().as_nanos();
            let wait = Instant::now();
            for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) { endpoint.finish()?; }
            self.stats.wait_ns += wait.elapsed().as_nanos();
            self.stats.sent += ((sr.len() + sl.len()) * buffers.len()) as u64;
            self.stats.received += ((rl.len() + rr.len()) * buffers.len()) as u64;
        }
        if !self.retain_regions {
            for endpoint in self.endpoints.iter_mut().flatten().chain(self.auxiliary.iter_mut()).chain(self.second.iter_mut().flatten()) { endpoint.clear_regions(); }
        }
        Ok(())
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        if let Some(profile) = &self.profile { profile.report(self.rank, self.world); }
    }
}

// Receive the first hop while local projection/export owns the outgoing slice.
// Only after readiness may gather_tcp_raw construct references to local sends.
fn receive_until_ready<'a>(mut right: Option<&'a TcpStream>, mut left: Option<&'a TcpStream>, rl: &mut [u8], rr: &mut [u8],
    ready: &Receiver<()>, stats: &mut TransportStats) -> Result<[usize; 2]>
{
    let start = Instant::now();
    let mut received = [0usize; 2];
    let mut recv_ns = 0;
    loop {
        match ready.try_recv() {
            Ok(()) => break,
            Err(TryRecvError::Disconnected) => return Err("K/V producer failed".into()),
            Err(TryRecvError::Empty) => (),
        }
        let before = received;
        for (i, (stream, bytes)) in [(&mut left, &mut *rl), (&mut right, &mut *rr)].into_iter().enumerate() {
            if received[i] == bytes.len() { continue; }
            let at = Instant::now();
            match stream.as_mut().ok_or("receive has no neighbor")?.read(&mut bytes[received[i]..]) {
                Ok(0) => return Err("peer closed while receiving".into()),
                Ok(n) => received[i] += n,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                Err(e) => return Err(e.to_string()),
            }
            recv_ns += at.elapsed().as_nanos();
        }
        if start.elapsed() >= TIMEOUT { return Err("K/V producer timed out".into()); }
        if received == before { std::thread::yield_now(); }
    }
    stats.received += (received[0] + received[1]) as u64;
    stats.recv_ns += recv_ns;
    stats.wait_ns += start.elapsed().as_nanos().saturating_sub(recv_ns);
    Ok(received)
}

/// Progress both neighbours on one CPU, with no per-message threads or copies.
fn pump<'a>(mut right: Option<&'a TcpStream>, mut left: Option<&'a TcpStream>, sr: &[u8], sl: &[u8], rl: &mut [u8], rr: &mut [u8], stats: &mut TransportStats, setup_deadline: Option<Instant>) -> Result<()> {
    let start = Instant::now();
    let deadline = setup_deadline.unwrap_or(start + TIMEOUT);
    let (mut sent, mut received) = ([0usize; 2], [0usize; 2]);
    let (mut send_ns, mut recv_ns) = (0, 0);
    let mut idle = 0usize;
    while sent != [sr.len(), sl.len()] || received != [rl.len(), rr.len()] {
        let before = sent[0] + sent[1] + received[0] + received[1];
        for (i, (stream, bytes)) in [(&mut right, sr), (&mut left, sl)].into_iter().enumerate() {
            if sent[i] == bytes.len() { continue; }
            let at = Instant::now();
            match stream.as_mut().ok_or("send has no neighbor")?.write(&bytes[sent[i]..]) {
                Ok(0) => return Err("peer closed while sending".into()),
                Ok(n) => sent[i] += n,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                Err(e) => return Err(e.to_string()),
            }
            send_ns += at.elapsed().as_nanos();
        }
        for (i, (stream, bytes)) in [(&mut left, &mut *rl), (&mut right, &mut *rr)].into_iter().enumerate() {
            if received[i] == bytes.len() { continue; }
            let at = Instant::now();
            match stream.as_mut().ok_or("receive has no neighbor")?.read(&mut bytes[received[i]..]) {
                Ok(0) => return Err("peer closed while receiving".into()),
                Ok(n) => received[i] += n,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                Err(e) => return Err(e.to_string()),
            }
            recv_ns += at.elapsed().as_nanos();
        }
        let progress = sent[0] + sent[1] + received[0] + received[1] != before;
        idle = if progress { 0 } else { idle + 1 };
        if Instant::now() >= deadline { return Err("ring exchange timed out".into()); }
        if idle > 1024 { std::thread::yield_now(); } else { std::hint::spin_loop(); }
    }
    stats.sent += (sent[0] + sent[1]) as u64;
    stats.received += (received[0] + received[1]) as u64;
    stats.send_ns += send_ns;
    stats.recv_ns += recv_ns;
    stats.wait_ns += start.elapsed().as_nanos().saturating_sub(send_ns + recv_ns);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[derive(Default)]
    struct CpuSender { pending: Option<(*mut u8, usize)>, output: Vec<u8>, blocked: bool, fail: bool }
    impl SlotSender for CpuSender {
        fn poll(&mut self, sends: &mut usize, arrived: &mut std::collections::VecDeque<(usize, usize)>) -> Result<()> {
            if self.fail { return Err("injected poll failure".into()); }
            if !self.blocked {
                let (packet, bytes) = self.pending.take().unwrap();
                self.output.extend_from_slice(unsafe { std::slice::from_raw_parts(packet, bytes) });
                *sends -= 1;
                arrived.push_back((1, bytes));
            }
            Ok(())
        }
        unsafe fn send(&mut self, packet: *mut u8, bytes: usize) -> Result<()> {
            if self.fail { return Err("injected send failure".into()); }
            assert!(self.pending.replace((packet, bytes)).is_none());
            Ok(())
        }
    }
    fn cpu_mailbox() -> Mailbox { Mailbox { slot: 8 << 20, offset: 0, next: 0, sends: 0, arrived: Default::default() } }

    #[test]
    fn relay_registers_opposite_receive_slots_and_propagates_errors() {
        let mut mailboxes = [Some(cpu_mailbox()), Some(cpu_mailbox())];
        mailboxes[1].as_mut().unwrap().offset = 24 << 20;
        let mut endpoints = [Some(Vec::new()), Some(Vec::new())];
        register_relay_slots(&mut endpoints, &mailboxes, |e, offset, bytes| { e.push((offset, bytes)); Ok(()) }).unwrap();
        assert_eq!(endpoints, [Some(vec![(24 << 20, 16 << 20)]), Some(vec![(0, 16 << 20)])]);
        assert!(register_relay_slots(&mut endpoints, &mailboxes, |_, _, _| Err("registration failed".into())).is_err());
        mailboxes[1] = None; endpoints[1] = None;
        register_relay_slots(&mut endpoints, &mailboxes, |_, _, _| panic!("absent endpoint or source")).unwrap();
        assert_eq!(mailboxes[0].as_ref().unwrap().next, 0);
    }

    #[test]
    fn relay_matches_copy_bytes_and_falls_back_on_split_mismatch() {
        for bytes in [0, 1, 65535, 65536, 65537, 131199, 4_194_305] {
            let data = (0..bytes).map(|i| (i * 137 + i / 256) as u8).collect::<Vec<_>>();
            for dual in [[false, false], [true, true], [false, true], [true, false]] {
                for k in 0..4 {
                    let source = 1 - k / 2;
                    let split = Ring::first_rail(bytes, dual[source]);
                    let first = Ring::first_rail(bytes, dual[k / 2]);
                    let (at, len) = if k % 2 == 0 { (0, first) } else { (first, bytes - first) };
                    let mut arrived = [None; 4];
                    for (rail, (from, n)) in [(0, split), (split, bytes - split)].into_iter().enumerate() {
                        if n != 0 { arrived[2 * source + rail] = Some((rail, unsafe { data.as_ptr().add(from) }, n, 3)); }
                    }
                    let mut copied = vec![0xa5; len];
                    unsafe { copy_relay(k, bytes, dual, &arrived, &mut copied); }
                    assert_eq!(copied, data[at..at + len]);
                    let packet = relay_packet(k, 3, bytes, dual, &arrived);
                    assert_eq!(packet.is_some(), len != 0 && first == split);
                    if let Some(packet) = packet {
                        let (mut mailbox, mut endpoint) = (cpu_mailbox(), CpuSender::default());
                        unsafe { mailbox.relay(&mut endpoint, packet as *mut u8, len, Instant::now() + TIMEOUT).unwrap(); }
                        assert_eq!(endpoint.pending.unwrap().0.cast_const(), packet);
                        assert_eq!(mailbox.sends, 1); assert!(endpoint.output.is_empty());
                        mailbox.wait_send(&mut endpoint, Instant::now() + TIMEOUT).unwrap();
                        assert_eq!(endpoint.output, copied); assert_eq!(mailbox.sends, 0);
                        assert_eq!(mailbox.arrived.pop_front(), Some((1, len)));
                    }
                }
            }
        }
    }

    #[test]
    fn relay_rejects_missing_wrong_owner_or_partial_packets() {
        let data = [0x81; 8];
        let mut arrived = [None; 4];
        assert!(relay_packet(0, 2, 8, [false; 2], &arrived).is_none());
        for (n, owner) in [(8, 1), (7, 2), (9, 2)] {
            arrived[2] = Some((0, data.as_ptr(), n, owner));
            assert!(relay_packet(0, 2, 8, [false; 2], &arrived).is_none());
        }
    }

    #[test]
    fn relay_waits_for_prior_send_and_propagates_failures() {
        let mut data = [0x80, 0xff, 0x00, 0x7f];
        let (mut mailbox, mut endpoint) = (cpu_mailbox(), CpuSender::default());
        let deadline = Instant::now() + TIMEOUT;
        unsafe {
            mailbox.relay(&mut endpoint, data.as_mut_ptr(), data.len(), deadline).unwrap();
            mailbox.relay(&mut endpoint, data.as_mut_ptr(), data.len(), deadline).unwrap();
        }
        assert_eq!(endpoint.output, data); assert_eq!(mailbox.sends, 1);
        endpoint.blocked = true;
        assert!(mailbox.wait_send(&mut endpoint, Instant::now()).is_err());
        assert_eq!(mailbox.sends, 1); assert!(endpoint.pending.is_some());
        endpoint.fail = true;
        assert!(mailbox.wait_send(&mut endpoint, deadline).is_err());
        assert!(unsafe { mailbox.relay(&mut endpoint, data.as_mut_ptr(), data.len(), deadline) }.is_err());
        assert_eq!(mailbox.sends, 1);
        endpoint.fail = false; endpoint.blocked = false;
        mailbox.wait_send(&mut endpoint, deadline).unwrap();
        assert_eq!(endpoint.output, data.repeat(2)); assert_eq!(mailbox.sends, 0);
        endpoint.fail = true;
        assert!(unsafe { mailbox.relay(&mut endpoint, data.as_mut_ptr(), data.len(), deadline) }.is_err());
        assert_eq!(mailbox.sends, 0);
    }

    #[test]
    fn uneven_ring_preserves_kv_bits_for_any_rank_count() {
        for (world, path) in [1, 2, 3, 4, 5, 6, 8].into_iter().flat_map(|n| [false, true].map(move |p| (n, p))) {
            let listeners = (0..world).map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap()).collect::<Vec<_>>();
            let addresses = listeners.iter().map(|l| l.local_addr().unwrap()).collect::<Vec<_>>();
            drop(listeners);
            std::thread::scope(|scope| {
                for rank in 0..world {
                    let addresses = &addresses;
                    scope.spawn(move || {
                        let mut ring = Ring::connect_topology(rank, world, addresses[rank], addresses[(rank + 1) % world], "ring-test", path, TIMEOUT).unwrap();
                        ring.barrier().unwrap();
                        for (tokens, prefix) in [0, 1, 3, 19, 4099].into_iter().flat_map(|n| [false, true].map(move |p| (n, p))) {
                            let p = Partition::balanced(tokens, world, 2.0).unwrap();
                            let staging = if prefix { ring.prefix_tokens(&p).unwrap() } else { tokens };
                            let (mut k, mut v) = (vec![f32::NAN; staging * 3], vec![f32::NAN; staging * 3]);
                            for i in p.range(rank).start * 3..(p.range(rank).end * 3).min(k.len()) {
                                k[i] = f32::from_bits(i as u32 + 1);
                                v[i] = f32::from_bits(0x80000000 | (i as u32 + 1));
                            }
                            if prefix { ring.exchange_prefix_kv(&mut k, &mut v, &p, 3).unwrap(); }
                            else { ring.exchange_kv(&mut k, &mut v, &p, 3).unwrap(); }
                            let end = if prefix { (p.range(rank).end * 3).min(k.len()) } else { k.len() };
                            assert!(k[..end].iter().enumerate().all(|(i, x)| x.to_bits() == i as u32 + 1));
                            assert!(v[..end].iter().enumerate().all(|(i, x)| x.to_bits() == (0x80000000 | (i as u32 + 1))));
                        }
                        ring.barrier().unwrap();
                    });
                }
            });
        }
    }

    #[test]
    fn disconnected_peer_poisoned_collective_fails_without_waiting() {
        let listeners = (0..2).map(|_| TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, 0)).unwrap()).collect::<Vec<_>>();
        let addresses = listeners.iter().map(|l| l.local_addr().unwrap()).collect::<Vec<_>>();
        drop(listeners);
        std::thread::scope(|scope| {
            let addresses = &addresses;
            let peer = scope.spawn(move || {
                let mut ring = Ring::connect(1, 2, addresses[1], addresses[0], "disconnect-test").unwrap();
                ring.barrier().unwrap();
            });
            let mut ring = Ring::connect(0, 2, addresses[0], addresses[1], "disconnect-test").unwrap();
            ring.barrier().unwrap();
            peer.join().unwrap();
            let mut values = [0.0; 16];
            assert!(ring.all_gather_f32(&mut values, &[0, 8, 16]).is_err());
            assert!(ring.all_gather_f32(&mut values, &[0, 8, 16]).unwrap_err().contains("unusable"));
        });
    }
}
