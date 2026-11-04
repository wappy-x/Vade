//! The tensor split's mailbox wire: one thread per rank, owner of the mailbox's host endpoint,
//! with no host copies and no receive-side host work. The rank's thread is a doorbell: when the
//! GPU publishes message s it posts two chained RC writes on one queue pair, the send slot's
//! published bytes straight into the peer's receive slot s % SLOTS, then the staged sequence
//! word (FLAG_OUT, host-owned) into the peer's RECV_SEQ, which the peer's take polls. Messages
//! keep their order on both sides (sequence 1, 2, ... per direction). A slot is posted again
//! only after the write completion of the message that used it before (four messages earlier);
//! the lockstep already implies the peer placed that message, so this check only waits for its
//! acknowledgement. Inside the ACTIVE window after the last message the thread spins at most
//! DOORBELL_SPIN (1 ms) between yields of its core, then sleeps with a doubling backoff up to
//! IDLE_MAX. The thread blocks (park, a futex on Linux) whenever every message the host queued
//! is posted and completed; the next put launch wakes it, so its core is free when no message
//! is in flight (between passes, between requests, while rank 1 idles). After
//! any error it fails its mailbox and writes 1 into the peer's ERROR word, which the peer's takes
//! poll; their timeout stays the backstop, so no rank ever computes on missing rows.
//!
//! On more than two ranks (a path) the thread is a relay (spark_gpu_probe::tp_route::Relay): one
//! queue pair per neighbor, each rank's own messages to both sides and the other ranks' messages
//! on from its receive slots, with no host copy. It polls with the same backoff and does not park.
use spark_gpu_probe::driver::TpMailboxHost;
use spark_gpu_probe::tp_route::Channels;
#[cfg(target_os = "linux")]
use spark_gpu_probe::tp_route::Links;
#[cfg(target_os = "linux")]
use spark_gpu_probe::tp_mailbox_host::{Ring, TpDoorbell};
#[cfg(target_os = "linux")]
use spark_gpu_probe::tp_route::Side;
#[cfg(target_os = "linux")]
use std::time::Instant;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

/// Polling after the last message: covers the GPU work between a pass's exchanges and the head,
/// acceptance and draft between two passes of a chat (a sleeping doorbell delays the next pass's
/// first message by its sleep).
#[cfg(target_os = "linux")]
const ACTIVE: Duration = Duration::from_millis(20);
/// Longest sleep between polls once the link is quiet (between requests).
#[cfg(target_os = "linux")]
const IDLE_MAX: Duration = Duration::from_micros(100);

/// The tensor split's wire before the mailbox exists.
pub enum Wire {
    #[cfg(target_os = "linux")]
    Write(WriteLink),
    /// More than two ranks: the links to the left and right neighbors (None at an end of the
    /// path), the path and this rank.
    #[cfg(target_os = "linux")]
    Relay([Option<WriteLink>; 2], Links, usize),
}
/// The write wire before the mailbox exists: the split's TCP link (for the one key swap)
/// and the RDMA device, RoCE v2 GID and path MTU code behind its local address.
#[cfg(target_os = "linux")]
pub struct WriteLink { link: std::net::TcpStream, device: String, index: u8, gid: [u8; 16], mtu: i32 }
#[cfg(target_os = "linux")]
impl WriteLink {
    /// Both ranks, right after the split's TCP link is up: find the HCA port of its interface.
    pub fn new(link: &std::net::TcpStream) -> Result<Self, String> {
        let ip = link.local_addr().map_err(|e| e.to_string())?.ip();
        let interface = engine_fabric::discovery::links()?.into_iter().find(|l| std::net::IpAddr::V4(l.address) == ip)
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
    /// Off Linux the split never starts (next_service refuses it); this keeps the lifetime.
    #[cfg(not(target_os = "linux"))]
    Unsupported(std::convert::Infallible, std::marker::PhantomData<TpMailboxHost<'m>>),
    #[cfg(target_os = "linux")]
    Write(TpMailboxHost<'m>, WriteLink, bool),
    #[cfg(target_os = "linux")]
    Relay(TpMailboxHost<'m>, [Option<WriteLink>; 2], Links, Channels, usize),
}
/// The longest a rank waits for its peer to attach its mailbox (both ranks load their models first).
#[cfg(target_os = "linux")]
const ATTACH_WAIT: Duration = Duration::from_secs(600);
/// How often a waiting swap rechecks its stop flags.
#[cfg(target_os = "linux")]
const ATTACH_POLL: Duration = Duration::from_millis(100);
impl Wire {
    /// Every rank, after set_tp and before any take is queued (no peer traffic here); `lead` is
    /// rank 0, which serves only once its thread attached; `layout` is the mailbox's.
    pub fn bind(self, host: TpMailboxHost<'_>, lead: bool, layout: Channels) -> Result<Worker<'_>, String> {
        #[cfg(not(target_os = "linux"))]
        { let _ = (host, lead, layout); match self {} }
        #[cfg(target_os = "linux")]
        {
            let mut host = host;
            host.set_take_pause(TAKE_PAUSE_NS);
            host.set_wide(true);
            host.set_take_ctas(TAKE_CTAS);
            match self {
                Wire::Write(link) => Ok(Worker::Write(host, link, lead)),
                Wire::Relay(links, route, rank) => Ok(Worker::Relay(host, links, route, layout, rank)),
            }
        }
    }
}
impl Worker<'_> {
    /// The wire thread. The write wire first swaps mailbox keys with the peer's thread over the
    /// split's TCP link (each rank's thread starts with its serving, so neither rank's startup
    /// waits on the other's), then sets `attached`. Rank 0 serves only
    /// once it is set, which is after rank 1's thread read its last swap byte, so every state
    /// handoff on the link follows the swap on both ranks. A failed attach fails this mailbox and
    /// shuts the link down, so the peer's swap fails at once too and rank 0 does not serve.
    /// `failed` is the service's stop flag: a failed attach also stops this rank's service (rank 1's
    /// back has no other way to learn it before its next request). Either flag set during the swap
    /// ends it (within ATTACH_POLL), so stopping never waits for a silent peer.
    pub fn run(self, stop: &AtomicBool, attached: &AtomicBool, failed: &AtomicBool) -> Result<(), String> {
        match self {
            #[cfg(not(target_os = "linux"))]
            Worker::Unsupported(never, _) => { let _ = (stop, attached, failed); match never {} }
            #[cfg(target_os = "linux")]
            Worker::Write(mut host, mut link, lead) => {
                let words = host.words();
                let cancel = || stop.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed);
                let rdma = match link.attach(words, lead, &cancel) {
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
            #[cfg(target_os = "linux")]
            Worker::Relay(mut host, mut links, route, layout, rank) => {
                let words = host.words();
                let cancel = || stop.load(Ordering::Relaxed) || failed.load(Ordering::Relaxed);
                // The lower rank of each link leads its swap. Links attach in their peers' rank
                // order from the highest down, so a rank attaches its lower neighbor only after its
                // higher one: rank 0's last attach ends after every link of the path is up, and
                // rank 0 serves only then (wait_attached).
                let mut wires: [Option<spark_gpu_probe::tp_rdma::TpRdma<'_>>; 2] = [None, None];
                let order = attach_order(&route, rank, [links[0].is_some(), links[1].is_some()]);
                let relay = (|| {
                    for &i in &order {
                        let peer = route.neighbor(rank, [Side::Left, Side::Right][i]).ok_or("tensor split relay: link without a neighbor")?;
                        let link = links[i].as_mut().ok_or("tensor split relay: link missing")?;
                        wires[i] = Some(link.attach(words, rank < peer, &cancel)?);
                    }
                    spark_gpu_probe::tp_route::Relay::new(words, route, layout, rank)
                })();
                // A failure fails this mailbox, writes 1 into the ERROR word of each neighbor already
                // attached (its takes stop at once) and shuts every link down.
                let mut relay = match relay {
                    Ok(relay) => relay,
                    Err(error) => {
                        host.fail();
                        for wire in wires.iter_mut().flatten() { host.fail_peer(wire); }
                        links.iter_mut().flatten().for_each(|l| { let _ = l.link.shutdown(std::net::Shutdown::Both); });
                        failed.store(true, Ordering::Release);
                        eprintln!("flashnext tp relay: {error}");
                        return Err(error);
                    }
                };
                attached.store(true, Ordering::Release);
                let timing = Ring { active: ACTIVE, spin: DOORBELL_SPIN, idle_max: IDLE_MAX, ..Ring::default() };
                let [left, right] = &mut wires;
                let result = relay.run([left.as_mut().map(|w| w as &mut dyn TpDoorbell), right.as_mut().map(|w| w as &mut dyn TpDoorbell)], stop, timing);
                if let Err(error) = &result { eprintln!("flashnext tp relay: {error}"); }
                result
            }
        }
    }
}
/// The relay's links (0 left, 1 right; `present` marks those it has) in attach order: the higher
/// neighbor first. A rank's swap with its lower neighbor then waits until every link above it is
/// up, so rank 0's swaps end last, on a path and on a ring.
#[cfg(any(target_os = "linux", test))]
fn attach_order(route: &spark_gpu_probe::tp_route::Links, rank: usize, present: [bool; 2]) -> Vec<usize> {
    use spark_gpu_probe::tp_route::Side;
    let mut order: Vec<usize> = (0..2).filter(|&i| present[i]).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(route.neighbor(rank, [Side::Left, Side::Right][i])));
    order
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
/// The takes' longest poll pause (ns); the one-sided wire lands the sequence word with no host
/// hop, so a short pause sees it sooner.
#[cfg(target_os = "linux")]
const TAKE_PAUSE_NS: u16 = 128;
/// The most CTAs a take launches. Six four-word CTAs measured best over 10-120 KB on a GB10 pair
/// (tp-gather-bench, 2026-10-26).
#[cfg(target_os = "linux")]
const TAKE_CTAS: u32 = 6;
/// The longest the doorbell spins between two yields of its core while a pass runs (20 us cost
/// about 2 % of one-chat code speed against 1000 us on a GB10 pair, 2026-10-26).
#[cfg(target_os = "linux")]
const DOORBELL_SPIN: Duration = Duration::from_micros(1000);
/// While nothing is in flight the doorbell parks (TpMailboxHost::ring, Ring::rest), woken by the
/// next put launch or, at shutdown, by the service. Parking tied polling on one-chat speed on a
/// GB10 pair and used 75 % of a core per rank during a chat against 92 % (2026-10-26). A parked
/// doorbell rechecks its stop flag and mailbox at least this often (the service also unparks it
/// after setting stop).
#[cfg(target_os = "linux")]
const REST: Duration = Duration::from_millis(50);

/// The write doorbell (TpMailboxHost::ring); after an error the peer's ERROR word is written too.
#[cfg(target_os = "linux")]
fn run_write(mut host: TpMailboxHost<'_>, mut rdma: spark_gpu_probe::tp_rdma::TpRdma<'_>, stop: &AtomicBool) -> Result<(), String> {
    let timing = Ring { active: ACTIVE, spin: DOORBELL_SPIN, idle_max: IDLE_MAX, rest: Some(REST), ..Ring::default() };
    let result = host.ring(&mut rdma, stop, timing);
    if let Err(error) = &result {
        host.fail_peer(&mut rdma);
        eprintln!("flashnext tp doorbell: {error}");
    }
    if let Some((parks, ns)) = host.bell().map(|b| b.stats()) {
        eprintln!("flashnext tp doorbell waits={parks} parked_ms={:.0}", ns as f64 / 1e6);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
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
    /// Every rank swaps its links in attach_order, one swap at a time, and a swap runs once both of
    /// its ranks reach it. On paths and rings of 2 to 8 ranks every link comes up (no rank waits
    /// in a cycle), and rank 0's last swap is the last of all: rank 0 serves only when the whole
    /// path is attached.
    #[test]
    fn rank_zero_attaches_last_on_every_path_and_ring() {
        use spark_gpu_probe::tp_route::{Links, Side};
        for (count, ring) in (2..=8).flat_map(|n| [(n, false), (n, true)]) {
            let route = Links::new(count, ring).unwrap();
            // Each rank's peers in its attach order (a two-rank ring links its pair once).
            let peers: Vec<Vec<usize>> = (0..count).map(|r| {
                let sides = [Side::Left, Side::Right].map(|side| route.neighbor(r, side));
                let present = [sides[0].is_some(), sides[1].is_some() && sides[1] != sides[0]];
                attach_order(&route, r, present).into_iter().filter_map(|i| sides[i]).collect()
            }).collect();
            let (mut next, mut done) = (vec![0usize; count], Vec::new());
            while let Some((a, b)) = (0..count).find_map(|a| {
                let b = *peers[a].get(next[a])?;
                (peers[b].get(next[b]) == Some(&a)).then_some((a, b))
            }) {
                next[a] += 1;
                next[b] += 1;
                done.push((a.min(b), a.max(b)));
            }
            let links = peers.iter().map(Vec::len).sum::<usize>() / 2;
            assert_eq!(done.len(), links, "{count} ranks, ring {ring}: a swap never ran");
            let last_zero = done.iter().rposition(|&(a, _)| a == 0).unwrap();
            assert_eq!(last_zero, links - 1, "{count} ranks, ring {ring}: rank 0 attached before {:?}", &done[last_zero + 1..]);
        }
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
