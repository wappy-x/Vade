//! VADE_W2_TP=1 mailbox proxy: one thread per rank, owner of the mailbox's host endpoint,
//! moves every message the GPU publishes in its send slots over the tensor split's RDMA wire
//! and publishes every arriving message in the receive slots its kernels wait on. Messages
//! keep their order on both sides (sequence 1, 2, ... per direction). The thread does not
//! spin: while a pass runs it polls with yield_now (the core goes to any runnable thread) for
//! up to ACTIVE after the last message, then sleeps with a doubling backoff up to IDLE_MAX.
//! After any error it marks the mailbox failed (every waiting take stops with an error) and
//! tells the peer, so no rank ever computes on missing rows.
//!
//! VADE_W2_TP_WIRE=write (unset: the proxy above over RdmaWire, unchanged): no host copies and no
//! receive-side host work. The rank's thread is a doorbell: when the GPU publishes message s it
//! posts two chained RC writes on one queue pair, the send slot's published bytes straight into
//! the peer's receive slot s % SLOTS, then the staged sequence word (FLAG_OUT, host-owned) into
//! the peer's RECV_SEQ, which the peer's take polls. A slot is posted again only after the write
//! completion of the message that used it before (four messages earlier); the lockstep already
//! implies the peer placed that message, so this check only waits for its acknowledgement.
//! Inside the ACTIVE window the thread spins at most VADE_W2_TP_DOORBELL_SPIN_US (1 ms) between
//! yields of its core (VADE_W2_TP_DOORBELL_CPU pins it to one core). By default
//! (VADE_W2_TP_DOORBELL=wait) the thread blocks (park, a futex on Linux) whenever every message the
//! host queued is posted and completed; the next put launch wakes it, so its core is free when no
//! message is in flight (between passes, between requests, while rank 1 idles).
//! VADE_W2_TP_DOORBELL=spin keeps polling through those gaps as well. After
//! any error it fails its mailbox and writes 1 into the peer's ERROR word, which the peer's takes
//! poll; their timeout stays the backstop.
use spark_engine::flash::next_batched::PpWire;
use spark_gpu_probe::driver::TpMailboxHost;
#[cfg(target_os = "linux")]
use spark_gpu_probe::tp_mailbox_host::Ring;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

/// Message header: sequence (u64), then flags (FAILED: the sender's mailbox failed; no rows).
const HEAD: usize = 12;
const FAILED: u32 = 1;
/// Yield-polling after the last message: covers the GPU work between a pass's exchanges and
/// the head, acceptance and draft between two passes of a chat (a sleeping proxy delays the
/// next pass's first message by its sleep).
const ACTIVE: Duration = Duration::from_millis(20);
/// Longest sleep between polls once the link is quiet (between requests).
const IDLE_MAX: Duration = Duration::from_micros(100);

pub fn run(mut host: TpMailboxHost<'_>, mut wire: Box<dyn PpWire>, stop: &AtomicBool) -> Result<(), String> {
    let result = forward(&mut host, wire.as_mut(), stop);
    if let Err(error) = &result {
        host.fail();
        // The second send waits (bounded by the wire's timeout) for the first to complete, so
        // the FAILED notice is on the wire before this endpoint is torn down.
        let notice = [&0u64.to_le_bytes()[..], &FAILED.to_le_bytes()[..]];
        let _ = wire.send(&notice).and_then(|()| wire.send(&notice));
        eprintln!("flashnext tp proxy: {error}");
    }
    result
}

fn forward(host: &mut TpMailboxHost<'_>, wire: &mut dyn PpWire, stop: &AtomicBool) -> Result<(), String> {
    let (mut sent, mut received) = (1u64, 1u64);
    let (mut rows, mut arrived) = (Vec::new(), Vec::new());
    let (mut active, mut pause) = (Instant::now(), Duration::from_micros(1));
    while !stop.load(Ordering::Relaxed) {
        if host.sent(sent) == sent as u32 {
            if host.error() {
                return Err(format!("mailbox failed before message {sent}"));
            }
            host.read_send(sent, &mut rows)?;
            wire.send(&[&sent.to_le_bytes(), &0u32.to_le_bytes(), &rows])?;
            sent += 1;
            (active, pause) = (Instant::now(), Duration::from_micros(1));
            continue;
        }
        if wire.try_recv(&mut arrived)? {
            let (peer, flags) = parse_head(&arrived)?;
            if flags & FAILED != 0 {
                return Err("the peer's mailbox failed".into());
            }
            if peer != received {
                return Err(format!("tensor split message {peer} arrived, expected {received}"));
            }
            host.deliver(received, &arrived[HEAD..])?;
            received += 1;
            (active, pause) = (Instant::now(), Duration::from_micros(1));
            continue;
        }
        if active.elapsed() < ACTIVE {
            std::thread::yield_now();
        } else {
            std::thread::sleep(pause);
            pause = (pause * 2).min(IDLE_MAX);
        }
    }
    Ok(())
}

/// VADE_W2_TP_WIRE: unset = the proxy over RdmaWire, write = one-sided writes (both ranks agree).
pub fn write_from_env() -> Result<bool, String> {
    match std::env::var("VADE_W2_TP_WIRE").as_deref() {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok("write") => Ok(true),
        Ok(v) => Err(format!("VADE_W2_TP_WIRE must be unset or write (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_WIRE: {e}")),
    }
}

/// The tensor split's wire, before the mailbox exists.
pub enum Wire {
    Proxy(Box<dyn PpWire>),
    #[cfg(target_os = "linux")]
    Write(WriteLink),
}
/// VADE_W2_TP_WIRE=write before the mailbox exists: the split's TCP link (for the one key swap)
/// and the RDMA device, RoCE v2 GID and path MTU code behind its local address.
#[cfg(target_os = "linux")]
pub struct WriteLink { link: std::net::TcpStream, device: String, index: u8, gid: [u8; 16], mtu: i32 }
#[cfg(target_os = "linux")]
impl WriteLink {
    /// Both ranks, right after the split's TCP link is up: find the HCA port of its interface.
    pub fn new(link: &std::net::TcpStream) -> Result<Self, String> {
        let ip = link.local_addr().map_err(|e| e.to_string())?.ip();
        let interface = spark_tp::discovery::links()?.into_iter().find(|l| std::net::IpAddr::V4(l.address) == ip)
            .ok_or("tensor split link has no QSFP interface")?.interface;
        let net_mtu = std::fs::read_to_string(format!("/sys/class/net/{interface}/mtu")).map_err(|e| e.to_string())?.trim().parse::<u32>().map_err(|e| e.to_string())?;
        let mtu = (1..=5).rev().find(|&m| (128u32 << m) + 128 <= net_mtu).ok_or("interface MTU too small for RoCE")?;
        for device in std::fs::read_dir("/sys/class/infiniband").map_err(|e| e.to_string())? {
            let device = device.map_err(|e| e.to_string())?;
            let port = device.path().join("ports/1");
            for index in 0..16u8 {
                let read = |p: String| std::fs::read_to_string(port.join(p)).unwrap_or_default();
                if read(format!("gid_attrs/ndevs/{index}")).trim() == interface && read(format!("gid_attrs/types/{index}")).trim() == "RoCE v2" {
                    let gid = read(format!("gids/{index}")).trim().parse::<std::net::Ipv6Addr>().map_err(|e| e.to_string())?.octets();
                    if gid[10..12] == [255, 255] {
                        let link = link.try_clone().map_err(|e| e.to_string())?;
                        return Ok(Self { link, device: device.file_name().to_string_lossy().into_owned(), index, gid, mtu: mtu as i32 });
                    }
                }
            }
        }
        Err("no IPv4 RoCE-v2 GID on the tensor split interface".into())
    }
    /// Open the queue pair over this rank's mailbox and swap `hello` with the peer's thread (swap),
    /// for at most ATTACH_WAIT and only until `cancel` (rechecked every ATTACH_POLL).
    fn attach<'m>(&mut self, words: &'m [std::sync::atomic::AtomicU32], lead: bool, cancel: &dyn Fn() -> bool) -> Result<spark_gpu_probe::tp_rdma::TpRdma<'m>, String> {
        let mut rdma = spark_gpu_probe::tp_rdma::TpRdma::open(&self.device, self.index, self.gid, self.mtu, words)?;
        let before = self.link.read_timeout().map_err(|e| e.to_string())?;
        self.link.set_read_timeout(Some(ATTACH_POLL)).map_err(|e| e.to_string())?;
        let hello = rdma.hello();
        let swapped = swap(&mut self.link, &hello, lead, Instant::now() + ATTACH_WAIT, cancel, |peer| rdma.ready(peer));
        self.link.set_read_timeout(before).map_err(|e| e.to_string())?;
        swapped.map(|()| rdma)
    }
}
/// The key swap on the split's TCP link: write this rank's `hello`, read the peer's, bring the
/// queue pair up with it (`ready`), then the acknowledgement: rank 1 writes it only after its own
/// queue pair is up, and rank 0 (`lead`) reads it last, so rank 0 serves only once both ranks
/// are ready and no swap byte is left unread on either side. Any failure is returned before the
/// acknowledgement moves, so the other rank's swap fails too once the caller shuts the link.
/// The link's reads time out (ATTACH_POLL) so a silent peer cannot hold a stopping rank: each
/// timeout rechecks `cancel` and the `deadline`.
#[cfg(target_os = "linux")]
fn swap<L: std::io::Read + std::io::Write>(link: &mut L, hello: &[u8], lead: bool, deadline: Instant, cancel: &dyn Fn() -> bool,
    ready: impl FnOnce(&[u8; HELLO]) -> Result<(), String>) -> Result<(), String> {
    let mut peer = [0u8; HELLO];
    let io = |e: std::io::Error| format!("tensor split mailbox exchange: {e}");
    link.write_all(hello).map_err(io)?;
    read_full(link, &mut peer, deadline, cancel)?;
    ready(&peer)?;
    if lead { read_full(link, &mut [0u8; 1], deadline, cancel) } else { link.write_all(&[1]).map_err(io) }
}
/// Fill `out` from the link, through read timeouts, until `deadline` or `cancel`.
#[cfg(target_os = "linux")]
fn read_full(link: &mut impl std::io::Read, out: &mut [u8], deadline: Instant, cancel: &dyn Fn() -> bool) -> Result<(), String> {
    let mut at = 0;
    while at < out.len() {
        match link.read(&mut out[at..]) {
            Ok(0) => return Err("tensor split mailbox exchange: the peer closed the link".into()),
            Ok(n) => at += n,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut | std::io::ErrorKind::Interrupted) => {
                if cancel() { return Err("tensor split mailbox exchange: stopped before the peer attached".into()); }
                if Instant::now() >= deadline { return Err("tensor split mailbox exchange: the peer did not attach in time".into()); }
            }
            Err(e) => return Err(format!("tensor split mailbox exchange: {e}")),
        }
    }
    Ok(())
}
#[cfg(target_os = "linux")]
use spark_gpu_probe::tp_rdma::HELLO;
/// The wire thread's work, bound to this rank's mailbox.
pub enum Worker<'m> {
    Proxy(TpMailboxHost<'m>, Box<dyn PpWire>),
    #[cfg(target_os = "linux")]
    Write(TpMailboxHost<'m>, WriteLink, bool),
}
/// The longest a rank waits for its peer to attach its mailbox (both ranks load their models first).
#[cfg(target_os = "linux")]
const ATTACH_WAIT: Duration = Duration::from_secs(600);
/// How often a waiting swap rechecks its stop flags.
#[cfg(target_os = "linux")]
const ATTACH_POLL: Duration = Duration::from_millis(100);
impl Wire {
    /// Both ranks, after set_tp and before any take is queued (no peer traffic here); `lead` is
    /// rank 0, which serves only once its thread attached.
    pub fn bind(self, host: TpMailboxHost<'_>, lead: bool) -> Result<Worker<'_>, String> {
        match self {
            Wire::Proxy(wire) => { let _ = lead; Ok(Worker::Proxy(host, wire)) }
            #[cfg(target_os = "linux")]
            Wire::Write(link) => {
                let mut host = host;
                host.set_take_pause(take_pause()?);
                host.set_wide(wide()?);
                host.set_take_ctas(take_ctas()?);
                doorbell_cpu()?;
                doorbell_spin()?;
                doorbell_wait()?;
                Ok(Worker::Write(host, link, lead))
            }
        }
    }
}
impl Worker<'_> {
    /// The wire thread. The write wire first swaps mailbox keys with the peer's thread over the
    /// split's TCP link (each rank's thread starts with its serving, so neither rank's startup
    /// waits on the other's), then sets `attached`; the proxy sets it at once. Rank 0 serves only
    /// once it is set, which is after rank 1's thread read its last swap byte, so every state
    /// handoff on the link follows the swap on both ranks. A failed attach fails this mailbox and
    /// shuts the link down, so the peer's swap fails at once too and rank 0 does not serve.
    /// `failed` is the service's stop flag: a failed attach also stops this rank's service (rank 1's
    /// back has no other way to learn it before its next request). Either flag set during the swap
    /// ends it (within ATTACH_POLL), so stopping never waits for a silent peer.
    pub fn run(self, stop: &AtomicBool, attached: &AtomicBool, failed: &AtomicBool) -> Result<(), String> {
        match self {
            Worker::Proxy(host, wire) => { let _ = failed; attached.store(true, Ordering::Release); run(host, wire, stop) }
            #[cfg(target_os = "linux")]
            Worker::Write(mut host, mut link, lead) => {
                let words = host.words();
                let cancel = || stop.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed);
                let rdma = match doorbell_cpu().and_then(|cpu| cpu.map_or(Ok(()), spark_tp::runtime::pin_cpu)).and_then(|()| link.attach(words, lead, &cancel)) {
                    Ok(rdma) => rdma,
                    Err(error) => {
                        host.fail();
                        let _ = link.link.shutdown(std::net::Shutdown::Both);
                        failed.store(true, Ordering::Release);
                        eprintln!("flashnext tp doorbell: {error}");
                        return Err(error);
                    }
                };
                attached.store(true, Ordering::Release);
                run_write(host, rdma, stop)
            }
        }
    }
}
/// Start the wire thread (named vade-tp-wire, so its core time can be read per thread).
pub fn spawn<'scope, 'env, 'm: 'scope>(scope: &'scope std::thread::Scope<'scope, 'env>, worker: Worker<'m>, stop: &'scope AtomicBool, attached: &'scope AtomicBool,
    failed: &'scope AtomicBool) -> Result<std::thread::ScopedJoinHandle<'scope, Result<(), String>>, String> {
    std::thread::Builder::new().name("vade-tp-wire".into()).spawn_scoped(scope, move || worker.run(stop, attached, failed)).map_err(|e| format!("tensor split wire thread: {e}"))
}
/// Rank 0, after spawning the wire thread: wait until it attached; if it ended first (its attach
/// failed), that is a startup error and rank 0 does not serve.
pub fn wait_attached(attached: &AtomicBool, worker: &std::thread::ScopedJoinHandle<'_, Result<(), String>>) -> Result<(), String> {
    while !attached.load(Ordering::Acquire) {
        if worker.is_finished() { return Err("the tensor split wire failed to attach".into()); }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}
/// VADE_W2_TP_TAKE_PAUSE_NS (write wire only, unset = 128): the takes' longest poll pause; the
/// one-sided wire lands the sequence word with no host hop, so a short pause sees it sooner.
#[cfg(target_os = "linux")]
fn take_pause() -> Result<u16, String> {
    match std::env::var("VADE_W2_TP_TAKE_PAUSE_NS") {
        Err(std::env::VarError::NotPresent) => Ok(128),
        Ok(v) => v.parse().ok().filter(|&ns| ns > 0).ok_or_else(|| format!("VADE_W2_TP_TAKE_PAUSE_NS must be 1..=65535 (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_TAKE_PAUSE_NS: {e}")),
    }
}
/// VADE_W2_TP_WIDE (write wire only, unset = 1): four-word put and take copies (0: one word).
#[cfg(target_os = "linux")]
fn wide() -> Result<bool, String> {
    match std::env::var("VADE_W2_TP_WIDE").as_deref() {
        Ok("0") => Ok(false),
        Err(std::env::VarError::NotPresent) | Ok("1") => Ok(true),
        _ => Err("VADE_W2_TP_WIDE must be 0 or 1".into()),
    }
}
/// VADE_W2_TP_TAKE_CTAS (write wire only, unset = 6): the most CTAs a take launches. Six
/// four-word CTAs measured best over 10-120 KB on a GB10 pair (tp-gather-bench, 2026-10-26).
#[cfg(target_os = "linux")]
fn take_ctas() -> Result<u32, String> {
    match std::env::var("VADE_W2_TP_TAKE_CTAS") {
        Err(std::env::VarError::NotPresent) => Ok(6),
        Ok(v) => v.parse().ok().filter(|n| (1..=96).contains(n)).ok_or_else(|| format!("VADE_W2_TP_TAKE_CTAS must be 1..=96 (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_TAKE_CTAS: {e}")),
    }
}
/// VADE_W2_TP_DOORBELL_SPIN_US (write wire only, unset = 1000, 0..=1000): the longest the doorbell
/// spins between two yields of its core while a pass runs (20 cost about 2 % of one-chat code
/// speed against 1000 on a GB10 pair, 2026-10-26).
#[cfg(target_os = "linux")]
fn doorbell_spin() -> Result<Duration, String> {
    match std::env::var("VADE_W2_TP_DOORBELL_SPIN_US") {
        Err(std::env::VarError::NotPresent) => Ok(Duration::from_micros(1000)),
        Ok(v) => v.parse().ok().filter(|&us| us <= 1000).map(Duration::from_micros).ok_or_else(|| format!("VADE_W2_TP_DOORBELL_SPIN_US must be 0..=1000 (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_DOORBELL_SPIN_US: {e}")),
    }
}
/// VADE_W2_TP_DOORBELL (write wire only): wait (unset) parks the doorbell while nothing is in
/// flight (TpMailboxHost::ring, Ring::rest), woken by the next put launch or, at shutdown, by the
/// service; spin polls through those gaps too. The two tied on one-chat speed on a GB10 pair, and
/// wait used 75 % of a core per rank during a chat against 92 % (2026-10-26).
#[cfg(target_os = "linux")]
fn doorbell_wait() -> Result<bool, String> {
    match std::env::var("VADE_W2_TP_DOORBELL").as_deref() {
        Ok("spin") => Ok(false),
        Err(std::env::VarError::NotPresent) | Ok("wait") => Ok(true),
        Ok(v) => Err(format!("VADE_W2_TP_DOORBELL must be unset, spin or wait (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_DOORBELL: {e}")),
    }
}
/// A parked doorbell rechecks its stop flag and mailbox at least this often (the service also
/// unparks it after setting stop).
#[cfg(target_os = "linux")]
const REST: Duration = Duration::from_millis(50);
/// VADE_W2_TP_DOORBELL_CPU: the core the write doorbell runs on (unset: not pinned).
#[cfg(target_os = "linux")]
fn doorbell_cpu() -> Result<Option<usize>, String> {
    match std::env::var("VADE_W2_TP_DOORBELL_CPU") {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(v) => v.parse().map(Some).map_err(|_| format!("VADE_W2_TP_DOORBELL_CPU must be a CPU index (got {v:?})")),
        Err(e) => Err(format!("VADE_W2_TP_DOORBELL_CPU: {e}")),
    }
}

/// The write doorbell (TpMailboxHost::ring); after an error the peer's ERROR word is written too.
#[cfg(target_os = "linux")]
fn run_write(mut host: TpMailboxHost<'_>, mut rdma: spark_gpu_probe::tp_rdma::TpRdma<'_>, stop: &AtomicBool) -> Result<(), String> {
    let timing = doorbell_spin().and_then(|spin| Ok(Ring { active: ACTIVE, spin, idle_max: IDLE_MAX, rest: doorbell_wait()?.then_some(REST), ..Ring::default() }));
    let result = timing.and_then(|timing| host.ring(&mut rdma, stop, timing));
    if let Err(error) = &result {
        host.fail_peer(&mut rdma);
        eprintln!("flashnext tp doorbell: {error}");
    }
    if let (Some((parks, ns)), true) = (host.bell().map(|b| b.stats()), doorbell_wait() == Ok(true)) {
        eprintln!("flashnext tp doorbell waits={parks} parked_ms={:.0}", ns as f64 / 1e6);
    }
    result
}

fn parse_head(message: &[u8]) -> Result<(u64, u32), String> {
    if message.len() < HEAD {
        return Err("tensor split message header".into());
    }
    Ok((u64::from_le_bytes(message[..8].try_into().unwrap()), u32::from_le_bytes(message[8..12].try_into().unwrap())))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn message_head_carries_sequence_and_flags() {
        let message = [&(u32::MAX as u64 + 5).to_le_bytes()[..], &FAILED.to_le_bytes()[..], &[9, 9][..]].concat();
        assert_eq!(parse_head(&message), Ok((u32::MAX as u64 + 5, FAILED)));
        assert!(parse_head(&message[..11]).is_err());
    }
    struct Wire(std::sync::mpsc::Sender<Vec<u8>>, std::sync::mpsc::Receiver<Vec<u8>>);
    impl PpWire for Wire {
        fn capacity(&self) -> usize { 1 << 20 }
        fn send(&mut self, parts: &[&[u8]]) -> Result<(), String> { self.0.send(parts.concat()).map_err(|e| e.to_string()) }
        fn try_recv(&mut self, out: &mut Vec<u8>) -> Result<bool, String> {
            match self.1.try_recv() {
                Ok(m) => { *out = m; Ok(true) }
                Err(std::sync::mpsc::TryRecvError::Empty) => Ok(false),
                Err(e) => Err(e.to_string()),
            }
        }
    }
    /// The proxy forwards a published send slot (header + exactly its published bytes),
    /// publishes an arriving message in its receive slot, and on the peer's FAILED notice
    /// fails the mailbox, notifies the peer and returns an error; with stop set it returns.
    #[test]
    fn proxy_forwards_delivers_fails_and_stops() {
        use spark_gpu_probe::tp_mailbox_host::{HEADER, RECV_SEQ, SEND_LEN, SEND_SEQ, TP_MAILBOX_SLOTS};
        use std::sync::{atomic::AtomicU32, mpsc::channel};
        let slot_words = 4;
        let memory: Vec<AtomicU32> = (0..HEADER + 2 * TP_MAILBOX_SLOTS * slot_words).map(|_| AtomicU32::new(0)).collect();
        let ((to_proxy, proxy_in), (proxy_out, from_proxy)) = (channel(), channel());
        let stop = AtomicBool::new(false);
        // Any failed check below stops the worker before the scope joins it (no hung test).
        struct Stop<'s>(&'s AtomicBool);
        impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
        std::thread::scope(|scope| {
            let _stop = Stop(&stop);
            let worker = scope.spawn(|| run(TpMailboxHost::new(&memory, slot_words), Box::new(Wire(proxy_out, proxy_in)), &stop));
            memory[HEADER + slot_words].store(5, Ordering::Relaxed);
            memory[SEND_LEN + 1].store(4, Ordering::Relaxed);
            memory[SEND_SEQ + 1].store(1, Ordering::Release);
            let sent = from_proxy.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(parse_head(&sent), Ok((1, 0)));
            assert_eq!(&sent[HEAD..], &5u32.to_le_bytes());
            to_proxy.send([&1u64.to_le_bytes()[..], &0u32.to_le_bytes(), &9u32.to_le_bytes()].concat()).unwrap();
            let deadline = Instant::now() + Duration::from_secs(5);
            while memory[RECV_SEQ + 1].load(Ordering::Acquire) != 1 { assert!(Instant::now() < deadline); std::thread::yield_now(); }
            assert_eq!(memory[HEADER + (TP_MAILBOX_SLOTS + 1) * slot_words].load(Ordering::Relaxed), 9);
            to_proxy.send([&0u64.to_le_bytes()[..], &FAILED.to_le_bytes()].concat()).unwrap();
            assert!(worker.join().unwrap().is_err());
            for _ in 0..2 { assert_eq!(parse_head(&from_proxy.recv_timeout(Duration::from_secs(5)).unwrap()), Ok((0, FAILED))); }
        });
        assert!(TpMailboxHost::new(&memory, slot_words).error());
        let (_keep, idle) = channel();
        assert!(run(TpMailboxHost::new(&memory, slot_words), Box::new(Wire(channel().0, idle)), &stop).is_ok());
    }
    /// Rank 0 serves only after its wire thread attached: a thread that ends first (its attach
    /// failed) is a startup error, found before any request, and an attached thread lets it serve.
    #[test]
    fn rank_zero_serves_only_after_the_wire_attached() {
        let (stop, attached) = (AtomicBool::new(false), AtomicBool::new(false));
        // A failed assertion below stops the looping child before the scope joins it (no hung test).
        struct Stop<'s>(&'s AtomicBool);
        impl Drop for Stop<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Relaxed); } }
        std::thread::scope(|scope| {
            let _stop = Stop(&stop);
            let failed = scope.spawn(|| -> Result<(), String> { Err("attach failed".into()) });
            assert_eq!(wait_attached(&attached, &failed), Err("the tensor split wire failed to attach".into()));
            let up = scope.spawn(|| -> Result<(), String> {
                attached.store(true, Ordering::Release);
                while !stop.load(Ordering::Relaxed) { std::thread::yield_now(); }
                Ok(())
            });
            assert_eq!(wait_attached(&attached, &up), Ok(()));
            stop.store(true, Ordering::Relaxed);
        });
    }
    /// The key swap over a socket pair: both ranks up -> both Ok and rank 0 read the
    /// acknowledgement; rank 1's queue pair fails to come up -> rank 1 returns the error without
    /// acknowledging, shuts its end, and rank 0's swap fails instead of serving.
    #[test]
    fn rank_zero_learns_of_a_back_side_readiness_failure() {
        use std::os::unix::net::UnixStream;
        for back_ok in [true, false] {
            let (mut front, mut back) = UnixStream::pair().unwrap();
            front.set_read_timeout(Some(ATTACH_POLL)).unwrap();
            back.set_read_timeout(Some(ATTACH_POLL)).unwrap();
            let (front_hello, back_hello) = ([1u8; HELLO], [2u8; HELLO]);
            let (lead, rank1) = std::thread::scope(|scope| {
                let rank1 = scope.spawn(move || {
                    let result = swap(&mut back, &back_hello, false, Instant::now() + Duration::from_secs(10), &|| false, |peer| {
                        assert_eq!(peer, &[1u8; HELLO]);
                        if back_ok { Ok(()) } else { Err("QP RTR: failed".into()) }
                    });
                    if result.is_err() { back.shutdown(std::net::Shutdown::Both).unwrap(); }
                    result
                });
                let lead = swap(&mut front, &front_hello, true, Instant::now() + Duration::from_secs(10), &|| false, |peer| { assert_eq!(peer, &[2u8; HELLO]); Ok(()) });
                (lead, rank1.join().unwrap())
            });
            if back_ok {
                assert_eq!((lead, rank1), (Ok(()), Ok(())));
            } else {
                assert_eq!(rank1, Err("QP RTR: failed".into()));
                assert!(lead.unwrap_err().starts_with("tensor split mailbox exchange"), "rank 0 must not serve");
            }
        }
    }
    /// A silent peer (connected, never sends its hello): a stop during the swap ends it within a
    /// few polls on either rank, and without a stop the swap ends at its deadline, never later.
    #[test]
    fn a_stop_ends_the_key_swap_with_a_silent_peer() {
        use std::os::unix::net::UnixStream;
        for lead in [true, false] {
            let (mut rank, _silent) = UnixStream::pair().unwrap();
            rank.set_read_timeout(Some(ATTACH_POLL)).unwrap();
            let stop = AtomicBool::new(false);
            let (result, waited) = std::thread::scope(|scope| {
                let swapping = scope.spawn(|| {
                    let at = Instant::now();
                    let r = swap(&mut rank, &[1u8; HELLO], lead, Instant::now() + Duration::from_secs(600), &|| stop.load(Ordering::Relaxed), |_| Ok(()));
                    (r, at.elapsed())
                });
                std::thread::sleep(Duration::from_millis(300));
                stop.store(true, Ordering::Relaxed);
                swapping.join().unwrap()
            });
            assert_eq!(result, Err("tensor split mailbox exchange: stopped before the peer attached".into()));
            assert!(waited < Duration::from_secs(5), "stop took {waited:?}");
        }
        let (mut rank, _silent) = UnixStream::pair().unwrap();
        rank.set_read_timeout(Some(ATTACH_POLL)).unwrap();
        let at = Instant::now();
        let result = swap(&mut rank, &[1u8; HELLO], false, Instant::now() + Duration::from_millis(250), &|| false, |_| Ok(()));
        assert_eq!(result, Err("tensor split mailbox exchange: the peer did not attach in time".into()));
        assert!(at.elapsed() < Duration::from_secs(5));
    }
}
