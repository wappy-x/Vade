use std::{hint::spin_loop, io::{ErrorKind, Read, Write}, net::{SocketAddr, TcpStream}, time::{Duration, Instant}};
use crate::{Result, rows};
use engine_fabric::layout::PeerLayout;

pub const COMMAND_TIMEOUT: Duration = Duration::from_secs(120);
/// How long a rank waits at a barrier for the other rank, on the control stream and on the mailbox alike.
pub const BARRIER_TIMEOUT: Duration = Duration::from_secs(180);
/// The barrier that ends a served step is one word on the pair's mailbox unless VADE_TP_STEP_BARRIER=tcp is set, on
/// BOTH ranks, which keeps it on the control stream (the form before 2026-11-11). Read once a process.
#[cfg(target_os="linux")]
fn step_barrier_on_mailbox() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VADE_TP_STEP_BARRIER").is_none_or(|form| form != "tcp");
        eprintln!("TP STEP BARRIER form={}", ["tcp", "mailbox"][usize::from(on)]);
        on
    })
}
const TIMEOUT: Duration = Duration::from_secs(30);

/// Fewest ranks whose commands may ride the RDMA command relay (Ring::open_command_relay).
/// Smaller worlds, the pair included, keep main's command path.
pub const COMMAND_RELAY_RANKS: usize = 4;
/// The relay opens only on a path of COMMAND_RELAY_RANKS or more ranks (the configured world).
pub fn command_relay_world(world: usize, path: bool) -> bool { world >= COMMAND_RELAY_RANKS && path }

/// Route of one command part. The pair test runs first and alone, as on main: a pair asks
/// only has_mailbox, never the relay; any other layout asks only the relay.
#[cfg(any(target_os="linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommandRoute { PairMailbox, Relay, Tcp }
#[cfg(any(target_os="linux", test))]
#[inline]
fn command_route(pair: bool, mailbox: impl FnOnce() -> bool, relay: impl FnOnce() -> bool) -> CommandRoute {
    if pair { if mailbox() { CommandRoute::PairMailbox } else { CommandRoute::Tcp } }
    else if relay() { CommandRoute::Relay } else { CommandRoute::Tcp }
}

fn command_read(mut stream: &TcpStream, mut bytes: &mut [u8], mut deadline: Option<Instant>, timeout: Duration) -> Result<()> {
    while !bytes.is_empty() {
        let remaining = deadline.map(|d| d.checked_duration_since(Instant::now()).filter(|d| !d.is_zero())
            .ok_or("command receive deadline expired")).transpose()?;
        stream.set_read_timeout(remaining).map_err(|e| e.to_string())?;
        match stream.read(bytes) {
            Ok(0) => return Err("peer closed during command receive".into()),
            Ok(n) => { bytes = &mut bytes[n..]; deadline.get_or_insert_with(|| Instant::now() + timeout); },
            Err(e) if e.kind() == ErrorKind::Interrupted => (),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
fn command_write(mut stream: &TcpStream, mut bytes: &[u8], deadline: Instant) -> Result<()> {
    while !bytes.is_empty() {
        let remaining = deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero())
            .ok_or("command send deadline expired")?;
        stream.set_write_timeout(Some(remaining)).map_err(|e| e.to_string())?;
        match stream.write(bytes) {
            Ok(0) => return Err("peer closed during command send".into()),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == ErrorKind::Interrupted => (),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
/// Decode commands follow the previous collective within a millisecond. Poll
/// the nonblocking socket that long before parking: a parked reader pays a
/// wakeup on every hop of the command chain.
const COMMAND_SPIN: Duration = Duration::from_millis(20);
fn command_receive(mut stream: &TcpStream, bytes: &mut [u8], deadline: Option<Instant>) -> Result<()> {
    let (spin, mut at) = (Instant::now() + COMMAND_SPIN, 0);
    while at < bytes.len() && Instant::now() < spin {
        match stream.read(&mut bytes[at..]) {
            Ok(0) => return Err("peer closed during command receive".into()),
            Ok(n) => at += n,
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => spin_loop(),
            Err(e) => return Err(e.to_string()),
        }
    }
    if at == bytes.len() { return Ok(()); }
    // A started command keeps the payload deadline; an idle one parks unbounded.
    let deadline = if at == 0 { deadline } else { Some(deadline.unwrap_or_else(|| Instant::now() + COMMAND_TIMEOUT)) };
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    command_read(stream, &mut bytes[at..], deadline, COMMAND_TIMEOUT)
}
fn command_send(mut stream: &TcpStream, bytes: &[u8]) -> Result<()> {
    let mut at = 0;
    while at < bytes.len() {
        match stream.write(&bytes[at..]) {
            Ok(0) => return Err("peer closed during command send".into()),
            Ok(n) => at += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => (),
            Err(e) if e.kind() == ErrorKind::WouldBlock => break,
            Err(e) => return Err(e.to_string()),
        }
    }
    if at == bytes.len() { return Ok(()); }
    stream.set_nonblocking(false).map_err(|e| e.to_string())?;
    command_write(stream, &bytes[at..], Instant::now() + COMMAND_TIMEOUT)
}

pub fn configure(stream: &TcpStream) -> Result<()> {
    #[cfg(target_os = "linux")]
    if let Ok(value) = std::env::var("TP_BUSY_POLL_US") {
        use std::os::fd::AsRawFd;
        unsafe extern "C" { fn setsockopt(fd: i32, level: i32, option: i32, value: *const i32, len: u32) -> i32; }
        let value = value.parse::<i32>().map_err(|e| e.to_string())?;
        if unsafe { setsockopt(stream.as_raw_fd(), 1, 46, &value, 4) } != 0 {
            return Err(format!("SO_BUSY_POLL: {}", std::io::Error::last_os_error()));
        }
    }
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    stream.set_nonblocking(true).map_err(|e| e.to_string())
}

/// Pump both directions on one CPU without per-message threads or allocations.
pub fn exchange(tx: &TcpStream, rx: &TcpStream, send: &[u8], recv: &mut [u8]) -> Result<()> {
    exchange_timeout(tx, rx, send, recv, TIMEOUT)
}
fn exchange_timeout(mut tx: &TcpStream, mut rx: &TcpStream, send: &[u8], recv: &mut [u8], timeout: Duration) -> Result<()> {
    let (mut sent, mut received, mut idle) = (0, 0, 0u32);
    let mut slow = false;
    let deadline = Instant::now() + timeout;
    while sent < send.len() || received < recv.len() {
        let before = sent + received;
        if sent < send.len() {
            match tx.write(&send[sent..]) {
                Ok(0) => return Err("peer closed while sending".into()),
                Ok(n) => sent += n,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                Err(e) => return Err(e.to_string()),
            }
        }
        if received < recv.len() {
            match rx.read(&mut recv[received..]) {
                Ok(0) => return Err("peer closed while receiving".into()),
                Ok(n) => received += n,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                Err(e) => return Err(e.to_string()),
            }
        }
        idle = idle.wrapping_add(1);
        if idle & 4095 == 0 {
            let now = Instant::now();
            if now > deadline { return Err("TCP exchange timed out".into()); }
            slow = deadline - now < timeout - Duration::from_millis(1);
        }
        if sent + received != before { slow = false; }
        if slow { std::thread::sleep(Duration::from_micros(100)); }
        spin_loop();
    }
    Ok(())
}

/// Batch handoff byte floor (`transfer_from_last_batch`), unchanged from `main`.
pub const DUAL_RAIL_FLOOR_BYTES: usize = 64 << 10;

/// Per-owner byte floor before striping an un-prepared two-rank pair gather across
/// a second rail: keeps short prompt groups (15 tokens: up to 960 KiB an owner)
/// on one doorbell and one CQ while leaving 3+ rank rings/paths unchanged.
pub const PAIR_GATHER_FLOOR_BYTES: usize = 1 << 20;

#[cfg(any(target_os="linux", test))]
const AUX_RDMA_TIMEOUT: Duration = Duration::from_secs(2);
const AUX_CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
#[cfg(test)]
type RdmaSetup = fn(&mut Ring, Duration) -> Result<()>;
#[cfg(test)]
thread_local! { static AUX_RDMA_SETUP: std::cell::Cell<Option<RdmaSetup>> = const { std::cell::Cell::new(None) }; }
thread_local! { static NEXT_CONNECT_WAIT: std::cell::Cell<Option<Duration>> = const { std::cell::Cell::new(None) }; }

#[inline]
fn active_rdma_rails(parts: usize, is_pair: bool, prepared: bool, write_pair: bool, has_aux: bool, range: &impl Fn(usize) -> std::ops::Range<usize>) -> usize {
    if !has_aux || write_pair { return 1; }
    if !is_pair || prepared || (0..parts).all(|part| range(part).len() >= PAIR_GATHER_FLOOR_BYTES) { 2 } else { 1 }
}

#[inline]
fn rail_stripe(full: std::ops::Range<usize>, rail: usize, rails: usize) -> std::ops::Range<usize> {
    let unit = if full.len().is_multiple_of(4) { 4 } else { 1 };
    let stripe = rows(full.len() / unit, rail, rails);
    full.start + stripe.start * unit..full.start + stripe.end * unit
}

fn exchange_pair_rails(rails: &mut [(&TcpStream, &TcpStream, &[u8], &mut [u8])]) -> Result<()> {
    let (mut sent, mut received, mut idle, mut slow) = ([0usize; 2], [0usize; 2], 0u32, false);
    let deadline = Instant::now() + TIMEOUT;
    while rails.iter().enumerate().any(|(i, (_, _, s, r))| sent[i] < s.len() || received[i] < r.len()) {
        let before = sent[0] + sent[1] + received[0] + received[1];
        for (i, (tx, rx, send, recv)) in rails.iter_mut().enumerate() {
            if sent[i] < send.len() {
                match tx.write(&send[sent[i]..]) {
                    Ok(0) => return Err(format!("rail {i} peer closed while sending")),
                    Ok(n) => sent[i] += n,
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                    Err(e) => return Err(format!("rail {i} send: {e}")),
                }
            }
            if received[i] < recv.len() {
                match rx.read(&mut recv[received[i]..]) {
                    Ok(0) => return Err(format!("rail {i} peer closed while receiving")),
                    Ok(n) => received[i] += n,
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                    Err(e) => return Err(format!("rail {i} receive: {e}")),
                }
            }
        }
        idle = idle.wrapping_add(1);
        if idle & 4095 == 0 {
            let now = Instant::now();
            if now > deadline { return Err("TCP exchange timed out".into()); }
            slow = deadline - now < TIMEOUT - Duration::from_millis(1);
        }
        if sent[0] + sent[1] + received[0] + received[1] != before { slow = false; }
        if slow { std::thread::sleep(Duration::from_micros(100)); }
        spin_loop();
    }
    Ok(())
}

pub struct Ring {
    rank: usize, world: usize,
    tx: Option<TcpStream>, rx: Option<TcpStream>,
    link: PeerLayout,
    packed: Vec<f32>, unpacked: Vec<f32>,
    pub exchanges: u64, pub elapsed: Duration,
    pub reduction_bytes: u64, pub reduction_elapsed: Duration,
    #[cfg(target_os="linux")]
    rdma:[Option<crate::rdma::Endpoint>;2],
    #[cfg(target_os="linux")]
    mailbox:[Option<crate::rdma::Mailbox>;2],
    // Command relay on a path of four or more ranks: [to rank + 1, from rank - 1]. Only
    // command_phase reads it; no collective uses these mailboxes.
    #[cfg(target_os="linux")]
    relay:[Option<crate::rdma::Mailbox>;2],
    #[cfg(target_os="linux")]
    peer_receive:Option<(usize,usize,usize,usize)>,
    aux:Option<Box<Ring>>,
}
impl Ring {
    /// Connect one rank. The second rail opens only when `TP_AUX_LISTEN` and
    /// `TP_AUX_NEXT` are both configured; without them the wire protocol is
    /// unchanged, and any failure to bring up the second rail falls back to one
    /// rail with one clear log line.
    pub fn connect(rank: usize, world: usize, listen: SocketAddr, next: SocketAddr) -> Result<Self> {
        let path = match std::env::var("TP_TOPOLOGY").as_deref() {
            Ok("path") => true, Ok("ring") | Err(_) => false, _ => return Err("TP_TOPOLOGY must be ring or path".into()),
        };
        let aux = match (std::env::var("TP_AUX_LISTEN").ok(), std::env::var("TP_AUX_NEXT").ok()) {
            (Some(listen), Some(next)) => match (listen.parse(), next.parse()) {
                (Ok(l), Ok(n)) => Some((l, n)),
                _ => {
                    eprintln!("tp_aux rank={rank} world={world} fallback=single_rail reason=invalid auxiliary socket address");
                    None
                }
            },
            (None, None) => None,
            _ => {
                eprintln!("tp_aux rank={rank} world={world} fallback=single_rail reason=both auxiliary endpoints must be supplied");
                None
            }
        };
        let rdma_allowed = std::env::var("TP_RDMA").as_deref() != Ok("0");
        let mut ring = Self::connect_options(rank, world, listen, next, path)?;
        if world > 1 {
            #[cfg(target_os="linux")]
            let rdma = rdma_allowed && ring.tx.as_ref().or(ring.rx.as_ref()).is_some_and(|s| s.local_addr().is_ok_and(|a| !a.ip().is_loopback()));
            #[cfg(not(target_os="linux"))]
            let rdma = rdma_allowed && false;
            #[cfg(target_os="linux")]
            if rdma { ring.initialise_rdma(true, TIMEOUT)?; }
            if let Some((aux_listen, aux_next)) = aux {
                ring.aux = Self::open_configured_aux(rank, world, aux_listen, aux_next, path, rdma);
            }
            #[cfg(target_os="linux")]
            if rdma {
                eprintln!("tp_links rank={rank} world={world} topology={} rails={}",
                    if path { "path" } else { "ring" }, 1 + usize::from(ring.aux.is_some()));
            }
        }
        Ok(ring)
    }
    /// Connect primary and optional configured second-rail sockets without
    /// reading environment variables.
    pub fn connect_options_with_aux(slot: usize, members: usize, listen: SocketAddr, next: SocketAddr, aux: Option<(SocketAddr, SocketAddr)>, path: bool) -> Result<Self> {
        let mut ring = Self::connect_options(slot, members, listen, next, path)?;
        if let Some((aux_listen, aux_next)) = aux.filter(|_| ring.tx.is_some() || ring.rx.is_some()) {
            ring.aux = Self::open_configured_aux(slot, members, aux_listen, aux_next, path, false);
        }
        Ok(ring)
    }
    fn open_configured_aux(slot: usize, members: usize, listen: SocketAddr, next: SocketAddr, path: bool, rdma: bool) -> Option<Box<Self>> {
        match Self::try_open_configured_aux(slot, members, listen, next, path, rdma) {
            Ok(aux) => Some(aux),
            Err(err) => {
                eprintln!("tp_aux rank={slot} world={members} fallback=single_rail reason={err}");
                None
            }
        }
    }
    fn try_open_configured_aux(slot: usize, members: usize, listen: SocketAddr, next: SocketAddr, path: bool, rdma: bool) -> Result<Box<Self>> {
        NEXT_CONNECT_WAIT.with(|w| w.set(Some(AUX_CONNECT_TIMEOUT)));
        let connected = Self::connect_options(slot, members, listen, next, path);
        NEXT_CONNECT_WAIT.with(|w| w.set(None));
        let mut aux = connected?;
        #[cfg(any(target_os="linux", test))]
        let local = if rdma {
            aux.initialise_rdma(false, AUX_RDMA_TIMEOUT)?;
            let ok = aux.gather_batch(&[1.0f32], members, 1, 1).and_then(|all| {
                if all.iter().all(|&v| v == 1.0) { Ok(()) } else { Err("auxiliary RDMA wire probe mismatch".into()) }
            });
            #[cfg(target_os="linux")]
            for e in aux.rdma.iter_mut().flatten() { e.set_timeout(TIMEOUT); }
            ok
        } else {
            Ok(())
        };
        #[cfg(not(any(target_os="linux", test)))]
        let local: Result<()> = { let _ = rdma; Ok(()) };
        #[cfg(target_os="linux")]
        let saved = std::mem::take(&mut aux.rdma);
        let mut failed = [f32::from(u8::from(local.is_err()))];
        let check = aux.reduce_small(&mut failed);
        #[cfg(target_os="linux")]
        { aux.rdma = saved; }
        aux.release_scratch();
        aux.exchanges = 0;
        aux.elapsed = Duration::ZERO;
        aux.reduction_bytes = 0;
        aux.reduction_elapsed = Duration::ZERO;
        check?;
        if failed[0] == 0.0 { Ok(Box::new(aux)) }
        else { Err(local.err().unwrap_or_else(|| "peer rank failed auxiliary rail setup".into())) }
    }
    #[cfg(any(target_os="linux", test))]
    #[allow(clippy::possible_missing_else)] // dense one-line guards.
    fn initialise_rdma(&mut self,mailbox:bool,timeout:Duration)->Result<()> {
        #[cfg(test)]
        if let Some(setup)=AUX_RDMA_SETUP.with(|hook|hook.get()) {return setup(self,timeout);}
        #[cfg(not(target_os="linux"))]
        let _=(mailbox,timeout);
        #[cfg(target_os="linux")]
        {
            let ring=self;
            if ring.link.is_pair() {
                let socket=ring.tx.as_ref().or(ring.rx.as_ref()).unwrap();let mut endpoint=crate::rdma::Endpoint::open(socket,timeout)?;
                let mut peer=[0;24];exchange_timeout(socket,socket,&endpoint.hello(),&mut peer,timeout)?;endpoint.ready(&peer)?;ring.rdma[0]=Some(endpoint);
                if mailbox&&std::env::var("TP_RDMA_SMALL").as_deref()!=Ok("0"){ring.mailbox[0]=Some(crate::rdma::Mailbox::open(socket)?);}
            } else {
                let mut right=ring.tx.as_ref().map(|socket|crate::rdma::Endpoint::open(socket,timeout)).transpose()?;
                let mut left=ring.rx.as_ref().map(|socket|crate::rdma::Endpoint::open(socket,timeout)).transpose()?;
                let sr=right.as_ref().map(|e|e.hello());let sl=left.as_ref().map(|e|e.hello());let mut rr=[0;24];let mut rl=[0;24];
                exchange_duplex_timeout(ring.tx.as_ref(),ring.rx.as_ref(),sr.as_ref().map_or(&[],|v|v.as_slice()),sl.as_ref().map_or(&[],|v|v.as_slice()),if left.is_some(){&mut rl}else{&mut []},if right.is_some(){&mut rr}else{&mut []},timeout)?;
                if let Some(e)=&mut right{e.ready(&rr)?;}if let Some(e)=&mut left{e.ready(&rl)?;}ring.rdma=[right,left];
                if mailbox&&std::env::var("TP_RDMA_SMALL").as_deref()!=Ok("0"){
                    let mut right=ring.tx.as_ref().map(crate::rdma::Mailbox::prepare).transpose()?;
                    let mut left=ring.rx.as_ref().map(crate::rdma::Mailbox::prepare).transpose()?;
                    let sr=right.as_ref().map(|m|m.hello());let sl=left.as_ref().map(|m|m.hello());
                    exchange_duplex(ring.tx.as_ref(),ring.rx.as_ref(),sr.as_ref().map_or(&[],|v|v.as_slice()),sl.as_ref().map_or(&[],|v|v.as_slice()),if left.is_some(){&mut rl}else{&mut []},if right.is_some(){&mut rr}else{&mut []})?;
                    if let Some(m)=&mut right{m.ready(&rr)?;}if let Some(m)=&mut left{m.ready(&rl)?;}ring.mailbox=[right,left];
                }
            }
        }
        Ok(())
    }
    /// RDMA mailboxes that carry small commands rank to rank on a path of
    /// COMMAND_RELAY_RANKS or more ranks (smaller worlds and the pair keep main's
    /// command path). Every rank calls this at the same point, after connect_options;
    /// the bytes of every command stay the same, and no collective reads these mailboxes.
    #[cfg(target_os="linux")]
    #[allow(clippy::possible_missing_else)] // dense one-line guards.
    pub fn open_command_relay(&mut self)->Result<()> {
        if !command_relay_world(self.world,self.link.is_path()) { return Err("the command relay needs a path of four or more ranks".into()); }
        let mut right=self.tx.as_ref().map(crate::rdma::Mailbox::prepare).transpose()?;
        let mut left=self.rx.as_ref().map(crate::rdma::Mailbox::prepare).transpose()?;
        let sr=right.as_ref().map(|m|m.hello());let sl=left.as_ref().map(|m|m.hello());let mut rr=[0;24];let mut rl=[0;24];
        exchange_duplex(self.tx.as_ref(),self.rx.as_ref(),sr.as_ref().map_or(&[],|v|v.as_slice()),sl.as_ref().map_or(&[],|v|v.as_slice()),if left.is_some(){&mut rl}else{&mut []},if right.is_some(){&mut rr}else{&mut []})?;
        if let Some(m)=&mut right{m.ready(&rr)?;}if let Some(m)=&mut left{m.ready(&rl)?;}
        self.relay=[right,left];
        Ok(())
    }
    pub fn connect_options(rank: usize, world: usize, listen: SocketAddr, next: SocketAddr, path: bool) -> Result<Self> {
        crate::runtime::pin()?;
        if world == 0 || world > 64 || rank >= world { return Err("rank must be less than world in 1..=64".into()); }
        let mut ring = Self { rank, world, tx: None, rx: None, link: PeerLayout::new(world, path), packed: Vec::new(), unpacked: Vec::new(), exchanges: 0, elapsed: Duration::ZERO, reduction_bytes:0,reduction_elapsed:Duration::ZERO, #[cfg(target_os="linux")]rdma:[None,None],#[cfg(target_os="linux")]mailbox:[None,None],#[cfg(target_os="linux")]relay:[None,None],#[cfg(target_os="linux")]peer_receive:None,aux:None };
        if world == 1 { return Ok(ring); }
        let deadline = Instant::now() + NEXT_CONNECT_WAIT.with(|w| w.take()).unwrap_or(TIMEOUT);
        let run_hash = crate::runtime::run_hash();
        let mut hello = [0u8; 32];
        hello[..8].copy_from_slice(b"VADETP\0\x04");
        hello[8..16].copy_from_slice(&(rank as u64).to_le_bytes());
        hello[16..24].copy_from_slice(&(world as u64).to_le_bytes());
        hello[24..].copy_from_slice(&run_hash.to_le_bytes());
        let validate = |message: &[u8; 32], expected: usize| -> Result<()> {
            if message[..8] != hello[..8] ||
                u64::from_le_bytes(message[8..16].try_into().unwrap()) != expected as u64 ||
                u64::from_le_bytes(message[16..24].try_into().unwrap()) != world as u64 ||
                u64::from_le_bytes(message[24..].try_into().unwrap()) != run_hash {
                return Err("TP protocol/rank/world/run-arguments handshake mismatch".into());
            }
            Ok(())
        };
        let (tx, rx) = spark_seqpar::control::connect_peers(
            (!path || rank > 0).then_some(listen), (!path || rank + 1 < world).then_some(next), deadline)?;
        if let Some(mut tx) = tx.as_ref() {
            tx.set_nodelay(true).map_err(|e| e.to_string())?;
            tx.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
            tx.write_all(&hello).map_err(|e| e.to_string())?;
        }
        if let Some(mut rx) = rx.as_ref() {
            let mut peer = [0u8; 32];
            rx.set_nonblocking(false).map_err(|e| e.to_string())?;
            rx.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
            rx.set_write_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
            rx.read_exact(&mut peer).map_err(|e| e.to_string())?;
            let check = validate(&peer, (rank + world - 1) % world);
            if check.is_err() {
                if &peer[..6] == b"VADETP" { let _ = rx.write_all(&hello); }
                check?;
            }
            rx.write_all(&hello).map_err(|e| e.to_string())?;
        }
        if let Some(mut tx) = tx.as_ref() {
            let mut peer = [0u8; 32];
            tx.set_read_timeout(Some(Duration::from_secs(5))).map_err(|e| e.to_string())?;
            tx.read_exact(&mut peer).map_err(|e| e.to_string())?;
            validate(&peer, (rank + 1) % world)?;
        }
        let (mut tx, mut rx) = (tx, rx);
        // Two ranks share one duplex connection so data carries the ACKs too.
        if ring.link.duplex() {
            if rank == 0 { rx = Some(tx.as_ref().unwrap().try_clone().map_err(|e| e.to_string())?); }
            else { tx = Some(rx.as_ref().unwrap().try_clone().map_err(|e| e.to_string())?); }
        }
        for stream in tx.iter().chain(rx.iter()) { configure(stream)?; }
        ring.tx = tx; ring.rx = rx;
        Ok(ring)
    }
    pub fn rank(&self) -> usize { self.rank }
    /// Fail the complete service transport, including workers parked in command().
    pub fn abort(&mut self) {
        for stream in self.tx.iter().chain(self.rx.iter()) { let _=stream.shutdown(std::net::Shutdown::Both); }
        if let Some(aux)=&mut self.aux { aux.abort(); }
        #[cfg(target_os="linux")]
        { self.mailbox=[None,None];self.relay=[None,None];self.rdma=[None,None];self.peer_receive=None; }
    }
    pub fn world(&self) -> usize { self.world }
    pub fn topology(&self) -> &'static str { if self.link.is_path() { "path" } else { "ring" } }
    pub fn algorithm(&self) -> &'static str { "duplex" }
    pub fn uses_rdma(&self)->bool {#[cfg(target_os="linux")]{self.rdma.iter().any(Option::is_some)}#[cfg(not(target_os="linux"))]{false}}
    pub fn clear_registered(&mut self){#[cfg(target_os="linux")]{
        if self.peer_receive.is_some(){self.abort();}
        for e in self.rdma.iter_mut().flatten(){e.clear_regions();}for m in self.mailbox.iter_mut().chain(self.relay.iter_mut()).flatten(){m.clear_registered();}
    }if let Some(aux)=&mut self.aux{aux.clear_registered();}}

    fn profile(&self, kind: &str, sent: usize, received: usize, elapsed: Duration) {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ENABLED.get_or_init(|| std::env::var("SPARK_PROFILE").as_deref() == Ok("1")) {
            eprintln!("transfer kind={kind} rank={} transport={} configured_rails={} sent_bytes={sent} received_bytes={received} elapsed_s={:.9}",
                self.rank, if self.world==1{"local"}else if self.uses_rdma(){"rdma"}else{"tcp"}, 1+usize::from(self.aux.is_some()), elapsed.as_secs_f64());
        }
    }

    /// Keep fixed transfer slots registered for one worker's lifetime. This also
    /// covers worker threads without a current CUDA context/allocation identity.
    pub fn with_transfer_regions<R>(&mut self, work: impl FnOnce(&mut Self) -> Result<R>) -> Result<R> {
        #[cfg(target_os="linux")]
        for endpoint in self.rdma.iter_mut().chain(self.aux.iter_mut().flat_map(|r|r.rdma.iter_mut())).flatten() { endpoint.retain_regions(true); }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(self)));
        #[cfg(target_os="linux")]
        if result.is_err() { self.rdma = [None, None]; if let Some(aux)=&mut self.aux{aux.rdma=[None,None];} }
        else { for endpoint in self.rdma.iter_mut().chain(self.aux.iter_mut().flat_map(|r|r.rdma.iter_mut())).flatten() { endpoint.retain_regions(false); } }
        match result { Ok(result) => result, Err(panic) => std::panic::resume_unwind(panic) }
    }

    /// Register one complete host allocation before slicing it into transfer
    /// slots. Keep it pinned and fixed through the enclosing with_transfer_regions.
    ///
    /// # Safety
    /// `base` must point to `bytes` bytes of one host allocation. The memory
    /// must stay allocated, pinned and unmoved until the enclosing
    /// `with_transfer_regions` returns.
    pub unsafe fn reserve_transfer_region(&mut self, base: *mut u8, bytes: usize) -> Result<()> {
        #[cfg(target_os="linux")]
        for endpoint in self.rdma.iter_mut().chain(self.aux.iter_mut().flat_map(|r| r.rdma.iter_mut())).flatten() {
            unsafe { endpoint.reserve_region(base, bytes)?; }
        }
        #[cfg(not(target_os="linux"))]
        let _ = (base, bytes);
        Ok(())
    }

    /// Queue every piece on both rails before waiting. The caller keeps these
    /// disjoint slots pinned and unchanged until the complete batch returns.
    pub fn transfer_from_last_batch(&mut self, destination: usize, buffers: &mut [&mut [u8]]) -> Result<()> {
        if destination >= self.world || buffers.len() > 256 || buffers.iter().any(|b| b.len() > u32::MAX as usize) {
            return Err("invalid final-rank transfer batch".into());
        }
        if self.rank < destination || destination + 1 == self.world { return Ok(()); }
        #[cfg(target_os="linux")]
        if self.link.is_pair() && self.uses_rdma() {
            let bytes = buffers.iter().try_fold(0usize, |sum, b| sum.checked_add(b.len())).ok_or("transfer batch size overflow")?;
            if bytes == 0 { return Ok(()); }
            let start = Instant::now();
            let last = self.rank + 1 == self.world;
            let rails = if bytes >= DUAL_RAIL_FLOOR_BYTES { 1 + usize::from(self.aux.is_some()) } else { 1 };
            let result = (|| {
                for buffer in buffers.iter_mut().filter(|b| !b.is_empty()) {
                    let length = buffer.len();
                    for (rail, endpoints) in std::iter::once(&mut self.rdma).chain(self.aux.iter_mut().map(|r| &mut r.rdma)).take(rails).enumerate() {
                        let stripe = length * rail / rails..length * (rail + 1) / rails;
                        unsafe {
                            endpoints[0].as_mut().ok_or("missing hand-off rail")?.post(buffer.as_mut_ptr(), length, length, 1,
                                if last { stripe.clone() } else { 0..0 }, if last { 0..0 } else { stripe })?;
                        }
                    }
                }
                for endpoints in std::iter::once(&mut self.rdma).chain(self.aux.iter_mut().map(|r| &mut r.rdma)).take(rails) {
                    for endpoint in endpoints.iter_mut().flatten() { endpoint.finish()?; }
                }
                Ok(())
            })();
            // A failed post can leave earlier slots in flight. Destroy both QPs
            // before releasing the borrowed buffers, including on receive errors.
            if result.is_err() { self.abort(); }
            let elapsed = start.elapsed();
            self.exchanges += 1;
            self.elapsed += elapsed;
            self.profile("hybrid_handoff", if last { bytes } else { 0 }, if last { 0 } else { bytes }, elapsed);
            return result;
        }
        for buffer in buffers { self.transfer_from_last(destination, buffer)?; }
        Ok(())
    }

    /// Forward a buffer from the final rank down to one destination. Every rank
    /// calls in the same order; the final rank supplies bytes, others receive.
    pub fn transfer_from_last(&mut self, destination: usize, bytes: &mut [u8]) -> Result<()> {
        if destination >= self.world || bytes.len() > u32::MAX as usize {
            return Err("invalid final-rank transfer destination or length".into());
        }
        if self.rank < destination || destination + 1 == self.world || bytes.is_empty() { return Ok(()); }
        let start = Instant::now();
        let result = (|| {
            #[cfg(target_os="linux")]
            if self.link.is_pair() && bytes.len()>=64*1024 && self.aux.is_some() && self.uses_rdma() {
                let last=self.rank==1;
                let length=bytes.len();
                for (rail,endpoints) in std::iter::once(&mut self.rdma).chain(self.aux.iter_mut().map(|r|&mut r.rdma)).enumerate() {
                    let stripe=length*rail/2..length*(rail+1)/2;
                    unsafe { endpoints[0].as_mut().ok_or("missing hand-off rail")?.post(bytes.as_mut_ptr(),length,length,1,
                        if last{stripe.clone()}else{0..0},if last{0..0}else{stripe})?; }
                }
                for endpoint in self.rdma.iter_mut().chain(self.aux.iter_mut().flat_map(|r|r.rdma.iter_mut())).flatten(){endpoint.finish()?;}
                return Ok(());
            }
            // A relay forwards the preceding piece while receiving the next;
            // bounded duplex work keeps a long suffix flowing across every hop.
            let piece_bytes = if self.link.is_pair() { bytes.len() } else { 256 * 1024 };
            let pieces = bytes.len().div_ceil(piece_bytes);
            let last = self.rank + 1 == self.world;
            for step in 0..=pieces {
                let sending = if last { Some(step) } else { step.checked_sub(1) }
                    .filter(|&piece| piece < pieces && self.rank > destination);
                let receiving = (!last && step < pieces).then_some(step);
                let range = |piece: Option<usize>| piece.map_or(0..0, |p| p * piece_bytes..((p + 1) * piece_bytes).min(bytes.len()));
                let send = range(sending);
                let recv = range(receiving);
                if send.is_empty() && recv.is_empty() { continue; }
                #[cfg(target_os="linux")]
                if self.uses_rdma() {
                    if !recv.is_empty() {
                        let endpoint = self.rdma[0].as_mut().ok_or("missing right RDMA link")?;
                        unsafe { endpoint.post(bytes.as_mut_ptr(), bytes.len(), bytes.len(), 1, 0..0, recv)?; }
                    }
                    if !send.is_empty() {
                        let endpoint = self.rdma[if self.link.is_pair() { 0 } else { 1 }].as_mut().ok_or("missing left RDMA link")?;
                        unsafe { endpoint.post(bytes.as_mut_ptr(), bytes.len(), bytes.len(), 1, send, 0..0)?; }
                    }
                    for endpoint in self.rdma.iter_mut().flatten() { endpoint.finish()?; }
                    continue;
                }
                match (sending, receiving) {
                    (Some(_), Some(_)) => {
                        let (previous, next) = bytes.split_at_mut(recv.start);
                        exchange(self.rx.as_ref().ok_or("missing left transfer link")?, self.tx.as_ref().ok_or("missing right transfer link")?,
                            &previous[send], &mut next[..recv.len()])?;
                    }
                    (Some(_), None) => {
                        let socket = self.rx.as_ref().ok_or("missing left transfer link")?;
                        exchange(socket, socket, &bytes[send], &mut [])?;
                    }
                    (None, Some(_)) => {
                        let socket = self.tx.as_ref().ok_or("missing right transfer link")?;
                        exchange(socket, socket, &[], &mut bytes[recv])?;
                    }
                    (None, None) => unreachable!(),
                }
            }
            Ok(())
        })();
        #[cfg(target_os="linux")]
        if result.is_err() { self.rdma = [None, None]; if let Some(aux)=&mut self.aux{aux.rdma=[None,None];} }
        self.exchanges += 1;
        let elapsed = start.elapsed();
        self.elapsed += elapsed;
        self.profile("hybrid_handoff", if self.rank > destination { bytes.len() } else { 0 },
            if self.rank + 1 < self.world { bytes.len() } else { 0 }, elapsed);
        result
    }

    /// `buffer` already contains this rank's rows. Byte offsets are deterministic.
    /// Drop wire registrations before a registered device allocation is freed.
    pub fn release_transfer_regions(&mut self) {
        #[cfg(target_os="linux")]
        for endpoint in self.rdma.iter_mut().chain(self.aux.iter_mut().flat_map(|r|r.rdma.iter_mut())).flatten() {endpoint.retain_regions(false);}
    }
    pub fn all_gather(&mut self, buffer: &mut [u8], row_bytes: usize) -> Result<()> {
        if row_bytes == 0 || !buffer.len().is_multiple_of(row_bytes) { return Err("invalid all-gather row width".into()); }
        let total = buffer.len() / row_bytes;
        let world = self.world;
        self.gather_ranges(buffer, "tp_gather", |rank| {
            let r = rows(total, rank, world); r.start * row_bytes..r.end * row_bytes
        })
    }
    fn gather_ranges(&mut self, buffer: &mut [u8], kind: &str, range: impl Fn(usize) -> std::ops::Range<usize>) -> Result<()> {
        let before = self.elapsed;
        let result = self.gather_ranges_inner(buffer, &range);
        self.profile(kind, self.gather_sent(&range), buffer.len() - range(self.rank).len(), self.elapsed - before);
        result
    }
    fn gather_sent(&self, range: &impl Fn(usize) -> std::ops::Range<usize>) -> usize {
        if self.world <= 2 { return range(self.rank).len() * (self.world - 1); }
        let owner = |shift: isize| {
            let rank = self.rank as isize + shift;
            if self.link.is_path() && !(0..self.world as isize).contains(&rank) { 0 }
            else { range(rank.rem_euclid(self.world as isize) as usize).len() }
        };
        let mut bytes = 0;
        for step in 0..if self.link.is_path() { self.world - 1 } else { self.world / 2 } {
            if self.tx.is_some() { bytes += owner(-(step as isize)); }
            if self.rx.is_some() && (self.link.is_path() || step < (self.world - 1) / 2) { bytes += owner(step as isize); }
        }
        bytes
    }
    fn gather_ranges_inner(&mut self, buffer: &mut [u8], range: &impl Fn(usize) -> std::ops::Range<usize>) -> Result<()> {
        if self.world == 1 { return Ok(()); }
        #[cfg(target_os="linux")]
        if (0..self.world).all(|r|self.has_mailbox(range(r).len())){return self.gather_mailbox(buffer,&range);}
        #[cfg(target_os="linux")]
        if self.uses_rdma() {
            // Rank-major staging is one contiguous message per owner. Use the
            // same striped RDMA exchange as device projection blocks; a large
            // gather must not silently fall back to TCP when the mailbox fills.
            return self.gather_rdma_ranges(buffer.as_mut_ptr(),buffer.len(),buffer.len(),1,None,range);
        }
        let start = Instant::now();
        let result = if self.world > 2 || self.link.is_path() {
            self.gather_duplex(buffer, &range)
        } else {
            let rails = active_rdma_rails(self.world, self.link.is_pair(), false, false, self.aux.is_some(), range);
            let mut result = Ok(());
            for step in 0..self.world - 1 {
                let owner = (self.rank + self.world - step) % self.world;
                let previous = (owner + self.world - 1) % self.world;
                let a = range(owner);
                let b = range(previous);
                // Different ranks' contiguous ranges cannot overlap, including empty tails.
                let (own, peer): (&[u8], &mut [u8]) = if a.start >= b.end {
                    let (left, right) = buffer.split_at_mut(a.start);
                    (&right[..a.len()], &mut left[b])
                } else {
                    let (left, right) = buffer.split_at_mut(b.start);
                    (&left[a], &mut right[..b.len()])
                };
                result = if rails == 2 {
                    let aux = self.aux.as_ref().ok_or("missing auxiliary rail")?;
                    let (own0, own1) = own.split_at(rail_stripe(0..own.len(), 0, 2).end);
                    let (peer0, peer1) = peer.split_at_mut(rail_stripe(0..peer.len(), 0, 2).end);
                    exchange_pair_rails(&mut [
                        (self.tx.as_ref().unwrap(), self.rx.as_ref().unwrap(), own0, peer0),
                        (aux.tx.as_ref().ok_or("missing auxiliary send link")?, aux.rx.as_ref().ok_or("missing auxiliary receive link")?, own1, peer1),
                    ])
                } else {
                    exchange(self.tx.as_ref().unwrap(), self.rx.as_ref().unwrap(), own, peer)
                };
                if result.is_err() { break; }
            }
            result
        };
        if result.is_err() { self.abort(); return result; }
        self.exchanges += 1;
        self.elapsed += start.elapsed();
        Ok(())
    }
    fn gather_duplex(&mut self, buffer: &mut [u8], range: &impl Fn(usize) -> std::ops::Range<usize>) -> Result<()> {
        let(rank,world,path)=(self.rank,self.world,self.link.is_path());
        let rounds = if self.link.is_path() { self.world - 1 } else { self.world / 2 };
        for step in 0..rounds {
            let owner = |offset: isize| -> std::ops::Range<usize> {
                let rank = rank as isize + offset;
                if path && !(0..world as isize).contains(&rank) { 0..0 }
                else { range(rank.rem_euclid(world as isize) as usize) }
            };
            let clockwise = self.link.is_path() || step < self.world / 2;
            let counter = self.link.is_path() || step < (self.world - 1) / 2;
            let send_right = if clockwise && self.tx.is_some() { owner(-(step as isize)) } else { 0..0 };
            let recv_left = if clockwise && self.rx.is_some() { owner(-(step as isize) - 1) } else { 0..0 };
            let send_left = if counter && self.rx.is_some() { owner(step as isize) } else { 0..0 };
            let recv_right = if counter && self.tx.is_some() { owner(step as isize + 1) } else { 0..0 };
            // Received owners are distinct from each other and both sent owners.
            // The two immutable sends may share the local shard in round zero.
            let base = buffer.as_mut_ptr();
            unsafe {
                let send = |r: std::ops::Range<usize>| std::slice::from_raw_parts(base.add(r.start), r.len());
                let recv = |r: std::ops::Range<usize>| std::slice::from_raw_parts_mut(base.add(r.start), r.len());
                exchange_duplex(self.tx.as_ref(), self.rx.as_ref(), send(send_right), send(send_left), recv(recv_left), recv(recv_right))?;
            }
        }
        Ok(())
    }
    /// One collective for T row-major activation rows; rank blocks are contiguous
    /// on the wire and restored to token order after receipt.
    pub fn gather_batch_f32(&mut self, local: &[f32], units: usize, width: usize, tokens: usize) -> Result<Vec<f32>> {
        self.gather_batch(local, units, width, tokens).map(<[f32]>::to_vec)
    }
    pub fn release_scratch(&mut self) { self.packed = Vec::new(); self.unpacked = Vec::new(); }
    /// Register mapped rank slots and exchange keys outside the decode layer loop.
    ///
    /// # Safety
    /// `base` must point to `16 * block` bytes of mapped rank slots. The memory
    /// must stay allocated and unmoved while the peer can write it.
    pub unsafe fn prepare_peer_write(&mut self,base:*mut f32,block:usize)->Result<()> {
        #[cfg(target_os="linux")]
        if self.link.is_pair()&&self.uses_rdma()&&block<=65536 {
            let bytes=block.checked_mul(16).ok_or("mapped pair size overflow")?;
            unsafe{self.rdma[0].as_mut().unwrap().prepare_write(base.cast(),bytes)?;}
        }
        let _=(base,block);Ok(())
    }
    /// Prefix writes are unsignaled; only the final tail carries a completion.
    ///
    /// # Safety
    /// `base` and `block` must name the slots of the posted peer receive. The
    /// memory must stay allocated and unmoved until the write completes.
    pub unsafe fn write_peer_prefix(&mut self,base:*mut f32,block:usize,prefix:usize)->Result<()> {
        #[cfg(target_os="linux")]
        if self.link.is_pair()&&self.uses_rdma() {
            let Some((address,bytes,old,slot))=self.peer_receive else{return Err("missing mapped receive".into());};
            if address!=base as usize||bytes!=block*16||prefix<=old||prefix>=block{self.abort();return Err("invalid mapped prefix".into());}
            let start=Instant::now();let rank=self.rank;
            let result=unsafe{self.rdma[0].as_mut().unwrap().write(base.cast(),bytes,
                (rank*block+old)*4..(rank*block+prefix)*4,((rank+slot*2)*block+old)*4,false)};
            let elapsed=start.elapsed();self.elapsed+=elapsed;self.reduction_elapsed+=elapsed;
            if result.is_err(){self.abort();}else{self.peer_receive=Some((address,bytes,prefix,slot));}
            return result;
        }
        let _=(base,block,prefix);Ok(())
    }
    /// Post the peer slot before waiting for the local projection. The caller
    /// keeps both mapped slots alive through gather_blocks and must have finished
    /// the previous GPU read of the peer slot before posting its next receive.
    ///
    /// # Safety
    /// `base` must point to the mapped slots for `block` rows. The slots must
    /// stay allocated and unmoved through `gather_blocks`, and the previous GPU
    /// read of the peer slot must be complete.
    pub unsafe fn post_peer_receive(&mut self,base:*mut f32,block:usize,slot:usize)->Result<()> {
        #[cfg(target_os="linux")]
        {
            if !self.link.is_pair()||!self.uses_rdma(){return Ok(());}
            if self.peer_receive.is_some(){return Err("peer receive already posted".into());}
            let bytes=block.checked_mul(if block<=65536{16}else{8}).ok_or("peer receive size overflow")?;
            let start=Instant::now();let rank=self.rank;let rails=1+usize::from(self.aux.is_some());
            let result=(||{
                if block<=65536 {return self.rdma[0].as_mut().unwrap().receive_write();}
                for(rail,endpoints)in std::iter::once(&mut self.rdma).chain(self.aux.as_mut().map(|r|&mut r.rdma)).enumerate(){
                    let stripe=rows(block,rail,rails);let recv=((1-rank)*block+stripe.start)*4..((1-rank)*block+stripe.end)*4;
                    unsafe{endpoints[0].as_mut().unwrap().post(base.cast(),bytes,bytes,1,0..0,recv)?;}
                }
                Ok(())
            })();
            let elapsed=start.elapsed();self.elapsed+=elapsed;self.reduction_elapsed+=elapsed;
            if result.is_err(){self.abort();}else{self.peer_receive=Some((base as usize,bytes,0,slot));}
            result
        }
        #[cfg(not(target_os="linux"))]
        {let _=(base,block,slot);Ok(())}
    }
    pub fn gather_blocks(&mut self,output:&mut [f32],block:usize,_tokens:usize)->Result<()> {
        if block.checked_mul(self.world)!=Some(output.len()){return Err("rank block shape mismatch".into());}
        #[cfg(target_os="linux")]
        if self.uses_rdma()&&(self.peer_receive.is_some()||!self.has_mailbox(block*4)){
            let before=self.elapsed;
            let result=self.gather_rdma(output,self.world,block,1);
            self.reduction_bytes+=(block*4*(self.world-1)) as u64;self.reduction_elapsed+=self.elapsed-before;
            self.profile("tp_reduction",self.gather_sent(&|r|r*block*4..(r+1)*block*4),block*4*(self.world-1),self.elapsed-before);
            return result;
        }
        let bytes=unsafe{std::slice::from_raw_parts_mut(output.as_mut_ptr().cast(),output.len()*4)};
        self.gather_ranges(bytes,"tp_reduction",|r|r*block*4..(r+1)*block*4)
    }
    /// Small reductions fit in CPU cache; avoid three device copies and a
    /// reduction launch at each layer boundary. Addition remains rank ordered.
    pub fn reduce_small(&mut self,output:&mut [f32])->Result<()> {
        if self.world==1||output.is_empty(){return Ok(());}
        #[cfg(target_os="linux")]
        if self.link.is_pair()&&self.mailbox[0].as_ref().is_some_and(|m|m.supports(output.len())) {
            let start=Instant::now();
            let result=self.mailbox[0].as_mut().unwrap().reduce(output,crate::rdma::Wait::Collective);if result.is_err(){self.mailbox=[None,None];}
            let elapsed=start.elapsed();self.exchanges+=1;self.elapsed+=elapsed;
            self.profile("tp_reduction",output.len()*4,output.len()*4,elapsed);return result;
        }
        if self.link.is_pair() {
            let start=Instant::now();self.unpacked.resize(output.len(),0.);
            let send=unsafe{std::slice::from_raw_parts(output.as_ptr().cast(),output.len()*4)};
            let recv=unsafe{std::slice::from_raw_parts_mut(self.unpacked.as_mut_ptr().cast(),output.len()*4)};
            exchange(self.tx.as_ref().or(self.rx.as_ref()).unwrap(),self.rx.as_ref().or(self.tx.as_ref()).unwrap(),send,recv)?;
            let elapsed=start.elapsed();self.elapsed+=elapsed;self.exchanges+=1;
            self.profile("tp_reduction",output.len()*4,output.len()*4,elapsed);
            for(a,b)in output.iter_mut().zip(&self.unpacked){*a+=*b;}
        }else {
            let n=output.len();let world=self.world;let all=self.gather_batch(output,world,n,1)?;
            output.copy_from_slice(&all[..n]);
            for peer in all[n..].chunks_exact(n){for(a,b)in output.iter_mut().zip(peer){*a+=*b;}}
        }
        Ok(())
    }
    #[cfg(target_os="linux")]
    fn has_mailbox(&self,bytes:usize)->bool{self.mailbox.iter().flatten().next().is_some_and(|m|m.supports(bytes.div_ceil(4)))}
    #[cfg(target_os="linux")]
    fn has_relay(&self,bytes:usize)->bool{self.relay.iter().flatten().next().is_some_and(|m|m.supports(bytes.div_ceil(4)))}
    #[cfg(target_os="linux")]
    fn gather_mailbox(&mut self,buffer:&mut [u8],range:&impl Fn(usize)->std::ops::Range<usize>)->Result<()> {
        let start=Instant::now();let(rank,world,path)=(self.rank,self.world,self.link.is_path());
        let rounds=if self.link.is_pair(){1}else if path{world-1}else{world/2};
        let result=(||{
            for step in 0..rounds {
                if self.link.is_pair(){
                    let m=self.mailbox[0].as_mut().unwrap();
                    unsafe{m.post(&buffer[range(rank)])?;}m.finish(&mut buffer[range(1-rank)])?;
                    continue;
                }
                let owner=|offset:isize|{let r=rank as isize+offset;if path&&!(0..world as isize).contains(&r){0..0}else{range(r.rem_euclid(world as isize) as usize)}};
                let clockwise=path||step<world/2;let counter=path||step<(world-1)/2;
                let sr=if clockwise{owner(-(step as isize))}else{0..0};let rl=if clockwise{owner(-(step as isize)-1)}else{0..0};
                let sl=if counter{owner(step as isize)}else{0..0};let rr=if counter{owner(step as isize+1)}else{0..0};
                // Post both directions before waiting; send ranges stay unchanged.
                if let Some(m)=&mut self.mailbox[0]{unsafe{m.post(&buffer[sr])?;}}
                if let Some(m)=&mut self.mailbox[1]{unsafe{m.post(&buffer[sl])?;}}
                if let Some(m)=&mut self.mailbox[0]{m.finish(&mut buffer[rr])?;}
                if let Some(m)=&mut self.mailbox[1]{m.finish(&mut buffer[rl])?;}
            }Ok(())
        })();
        if result.is_err(){self.mailbox=[None,None];}
        self.elapsed+=start.elapsed();self.exchanges+=1;result
    }
    #[cfg(target_os="linux")]
    fn gather_rdma(&mut self,output:&mut [f32],units:usize,width:usize,tokens:usize)->Result<()> {
        let world=self.world;
        self.gather_rdma_ranges(output.as_mut_ptr().cast(),std::mem::size_of_val(output),units*width*4,tokens,Some(width),&|rank| {
            let r=rows(units,rank,world);r.start*width*4..r.end*width*4
        })
    }
    #[cfg(target_os="linux")]
    fn gather_rdma_ranges(&mut self,base:*mut u8,bytes:usize,stride:usize,tokens:usize,projection_width:Option<usize>,
        range:&impl Fn(usize)->std::ops::Range<usize>)->Result<()> {
        let start=Instant::now();
        let prepared=self.peer_receive.take();
        let width=projection_width.unwrap_or(0);
        if prepared.is_some_and(|(address,length,_,_)|projection_width.is_none()||(address,length)!=(base as usize,if width<=65536{bytes*2}else{bytes})) {self.abort();return Err("prepared peer receive allocation mismatch".into());}
        let(rank,world,path)=(self.rank,self.world,self.link.is_path());let mut aux=self.aux.take();
        // Short projections and short prompt groups keep one doorbell and CQ; large
        // block exchanges stripe across both cable rails when the sibling QP is open.
        let write_pair=prepared.is_some()&&self.link.is_pair()&&width<=65536;
        let rails=active_rdma_rails(world,self.link.is_pair(),prepared.is_some(),write_pair,aux.is_some(),range);
        let rounds=if self.link.is_pair(){1}else if self.link.is_path(){self.world-1}else{self.world/2};
        let result=(||{
            for step in 0..rounds {
                for(rail,endpoints)in std::iter::once(&mut self.rdma).chain(aux.as_mut().map(|r|&mut r.rdma)).take(rails).enumerate(){
                let owner=|offset:isize| {
                    let r=rank as isize+offset;
                    if path&&!(0..world as isize).contains(&r){return 0..0;}
                    rail_stripe(range(r.rem_euclid(world as isize) as usize),rail,rails)
                };
                if self.link.is_pair() {
                    let own=owner(0);let peer=owner(if rank==0{1}else{-1});
                    if let Some((_,allocation,prefix,slot))=prepared.filter(|_|width<=65536) {
                        let start=own.start.max((rank*width+prefix)*4).min(own.end);
                        unsafe{endpoints[0].as_mut().unwrap().write(base,allocation,start..own.end,start+slot*width*8,true)?;}
                    } else {unsafe{endpoints[0].as_mut().unwrap().post(base,bytes,stride,tokens,own,if prepared.is_some(){0..0}else{peer})?};}
                } else {
                    let clockwise=self.link.is_path()||step<self.world/2;let counter=self.link.is_path()||step<(self.world-1)/2;
                    let sr=if clockwise{owner(-(step as isize))}else{0..0};let rl=if clockwise{owner(-(step as isize)-1)}else{0..0};
                    let sl=if counter{owner(step as isize)}else{0..0};let rr=if counter{owner(step as isize+1)}else{0..0};
                    if let Some(e)=&mut endpoints[0]{unsafe{e.post(base,bytes,stride,tokens,sr,rr)?};}
                    if let Some(e)=&mut endpoints[1]{unsafe{e.post(base,bytes,stride,tokens,sl,rl)?};}
                }
                }
                if write_pair {self.rdma[0].as_mut().unwrap().finish()?;}
                else {for endpoints in std::iter::once(&mut self.rdma).chain(aux.as_mut().map(|r|&mut r.rdma)).take(rails){for e in endpoints.iter_mut().flatten(){e.finish()?;}}}
            }Ok(())
        })();
        self.aux=aux;
        if result.is_err(){self.abort();}
        self.elapsed+=start.elapsed();self.exchanges+=1;result
    }
    /// Exchange rank-major blocks in caller-owned pinned storage. GPU copies
    /// perform the row interleave after the NIC has completed both rails.
    pub fn gather_rank_blocks(&mut self,buffer:&mut [f32],units:usize,width:usize,tokens:usize)->Result<()> {
        if units.checked_mul(width).and_then(|n|n.checked_mul(tokens))!=Some(buffer.len()) {
            return Err("rank block shape mismatch".into());
        }
        let world=self.world;
        let bytes=unsafe{std::slice::from_raw_parts_mut(buffer.as_mut_ptr().cast(),std::mem::size_of_val(buffer))};
        self.gather_ranges(bytes,"tp_gather",|rank|{
            let r=rows(units,rank,world);r.start*width*tokens*4..r.end*width*tokens*4
        })
    }
    /// Reusable staging remains borrowed through the caller's device upload.
    pub fn gather_batch(&mut self, local: &[f32], units: usize, width: usize, tokens: usize) -> Result<&[f32]> {
        if rows(units,self.rank,self.world).len()*width*tokens!=local.len(){return Err("batch gather shape mismatch".into());}
        self.gather_batch_with(units,width,tokens,|out|{out.copy_from_slice(local);Ok(())})
    }
    /// Read device operands directly into the reusable rank-major exchange.
    pub fn gather_batch_with(&mut self, units:usize, width:usize, tokens:usize,
        read:impl FnOnce(&mut [f32])->Result<()>)->Result<&[f32]> {
        let total = units.checked_mul(width).and_then(|n| n.checked_mul(tokens)).ok_or("batch gather overflow")?;
        let own = rows(units, self.rank, self.world);
        if width == 0 || tokens == 0 { return Err("batch gather shape mismatch".into()); }
        let mut packed = std::mem::take(&mut self.packed);
        packed.resize(total, 0.);
        if let Err(error)=read(&mut packed[own.start*width*tokens..own.end*width*tokens]) {
            self.packed=packed;return Err(error);
        }
        let world = self.world;
        let bytes = unsafe { std::slice::from_raw_parts_mut(packed.as_mut_ptr().cast(), total * 4) };
        self.gather_ranges(bytes, "tp_gather", |rank| { let r = rows(units, rank, world); r.start * width * tokens * 4..r.end * width * tokens * 4 })?;
        if tokens == 1 || world == 1 { self.packed = packed; return Ok(&self.packed); }
        self.unpacked.resize(total, 0.);
        for rank in 0..world {
            let r = rows(units, rank, world); let n = r.len() * width;
            for token in 0..tokens {
                let src = r.start * width * tokens + token * n;
                let dst = token * units * width + r.start * width;
                self.unpacked[dst..dst + n].copy_from_slice(&packed[src..src + n]);
            }
        }
        self.packed = packed;
        Ok(&self.unpacked)
    }
    /// Small two-rank commands use the already-posted RDMA mailbox. Idle workers
    /// park between requests; large commands and other topologies retain TCP.
    pub fn command(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.command_phase(bytes, true)
    }
    /// Payloads follow a received header and must arrive within the deadline.
    pub fn command_payload(&mut self, bytes: &mut [u8]) -> Result<()> {
        self.command_phase(bytes, false)
    }
    /// Path selected for this command part, captured before a failure drops it.
    pub fn command_path(&self, bytes: usize) -> spark_engine::collective::Path {
        #[cfg(target_os="linux")]
        if command_route(self.link.is_pair(), || self.has_mailbox(bytes), || self.has_relay(bytes)) != CommandRoute::Tcp { return spark_engine::collective::Path::Mailbox; }
        #[cfg(not(target_os="linux"))]
        let _ = bytes;
        spark_engine::collective::Path::Tcp
    }
    fn command_phase(&mut self, bytes: &mut [u8], idle: bool) -> Result<()> {
        #[cfg(target_os="linux")]
        {
            let n=bytes.len();let route=command_route(self.link.is_pair(),||self.has_mailbox(n),||self.has_relay(n));
            let relay=route==CommandRoute::Relay;
            if route!=CommandRoute::Tcp {
                // The pair: rank 0 sends to rank 1. The relay: each rank takes the part from
                // rank - 1, then forwards the same bytes to rank + 1 and waits for that send.
                let (receive,send)=if relay {let [right,left]=&mut self.relay;(left.as_mut(),right.as_mut())}
                    else if self.rank==0 {(None,self.mailbox[0].as_mut())} else {(self.mailbox[0].as_mut(),None)};
                let mut result=match receive {Some(m) if idle=>m.finish_command(bytes),Some(m)=>m.finish(bytes),None=>Ok(())};
                if let (Ok(()),Some(mailbox))=(&result,send) {
                    result=unsafe{mailbox.post(bytes)}.and_then(|()|mailbox.finish(&mut []));
                }
                if result.is_err(){self.mailbox=[None,None];self.relay=[None,None];}
                if relay { return result.map_err(|e| format!("TP rank {} command relay phase: {e}", self.rank)); }
                return result.map_err(|e| format!("TP peer {} command mailbox phase: {e}", 1 - self.rank));
            }
        }
        // Sockets stay nonblocking between phases; only a parked fallback
        // switches one to blocking, and every socket is restored below.
        let deadline = Instant::now() + COMMAND_TIMEOUT;
        let result = (|| {
            if self.rank > 0 { command_receive(self.rx.as_ref().ok_or("missing previous rank")?, bytes, (!idle).then_some(deadline))?; }
            if self.rank + 1 < self.world { command_send(self.tx.as_ref().ok_or("missing next rank")?, bytes)?; }
            Ok(())
        })();
        for s in self.tx.iter().chain(self.rx.iter()) { s.set_nonblocking(true).map_err(|e| e.to_string())?; }
        result.map_err(|e: String| format!("TP peer {} command receive/send phase: {e}", if self.rank == 0 { 1 } else { self.rank - 1 }))
    }
    pub fn barrier(&mut self) -> Result<()> {
        if self.world == 1 { return Ok(()); }
        // Arrive toward rank zero, release toward the last rank. Model loading
        // gets a separate blocking deadline, without entering the spin pump.
        let start = Instant::now();
        for stream in self.tx.iter().chain(self.rx.iter()) {
            stream.set_nonblocking(false).map_err(|e| e.to_string())?;
            stream.set_read_timeout(Some(BARRIER_TIMEOUT)).map_err(|e| e.to_string())?;
            stream.set_write_timeout(Some(BARRIER_TIMEOUT)).map_err(|e| e.to_string())?;
        }
        let result = (|| {
            let mut ready = [1u8];
            if self.rank + 1 < self.world { self.tx.as_ref().unwrap().read_exact(&mut ready)?; }
            if self.rank > 0 {
                self.rx.as_ref().unwrap().write_all(&ready)?;
                self.rx.as_ref().unwrap().read_exact(&mut ready)?;
            }
            if self.rank + 1 < self.world { self.tx.as_ref().unwrap().write_all(&ready)?; }
            Ok::<(), std::io::Error>(())
        })().map_err(|e| format!("rank barrier: {e}"));
        for stream in self.tx.iter().chain(self.rx.iter()) {
            stream.set_read_timeout(None).map_err(|e| e.to_string())?;
            stream.set_write_timeout(None).map_err(|e| e.to_string())?;
            stream.set_nonblocking(true).map_err(|e| e.to_string())?;
        }
        self.exchanges += 1;
        self.elapsed += start.elapsed();
        result
    }
    /// The barrier that ends a served step (row_split.rs Exchange::complete). A pair with a mailbox trades ONE WORD
    /// on the mailbox both ranks already poll, so no step sleeps in the kernel for a receive interrupt of the control
    /// stream. MEASURED 2026-11-11 on two Sparks: that interrupt lands on one core for the life of a server, a
    /// different core at every start; when the core idles deep (exit 433 microseconds) every step waits 0.47 ms
    /// longer (14 of 22 starts). A rank proceeds only when the other rank's word has arrived, as before. Model
    /// load, three ranks or more, and a pair without a mailbox keep `barrier`. The deadline is BARRIER_TIMEOUT and
    /// the error starts with "rank barrier:" in both forms; a rank that is gone ends the wait within 100 ms.
    pub fn step_barrier(&mut self) -> Result<()> {
        #[cfg(target_os="linux")]
        {
            let pair = self.link.is_pair().then_some(()).filter(|()| step_barrier_on_mailbox());
            let Some(mailbox) = pair.and_then(|()| self.mailbox[0].as_mut()) else { return self.barrier(); };
            let mut word = [1f32];
            let result = mailbox.reduce(&mut word, crate::rdma::Wait::Barrier)
                .and_then(|()| word[0].eq(&2.0).then_some(()).ok_or_else(|| "the other rank's barrier word differs".to_string()));
            // A failed mailbox is dropped, as after any failed collective: the service has failed.
            result.map_err(|e| { self.mailbox = [None, None]; format!("rank barrier: {e}") })
        }
        #[cfg(not(target_os="linux"))]
        { self.barrier() }
    }
}

fn exchange_duplex(right: Option<&TcpStream>, left: Option<&TcpStream>, sr: &[u8], sl: &[u8], rl: &mut [u8], rr: &mut [u8]) -> Result<()> {
    exchange_duplex_timeout(right, left, sr, sl, rl, rr, TIMEOUT)
}
fn exchange_duplex_timeout(right: Option<&TcpStream>, left: Option<&TcpStream>, sr: &[u8], sl: &[u8], rl: &mut [u8], rr: &mut [u8], timeout: Duration) -> Result<()> {
    let mut sent = [0, 0];
    let mut received = [0, 0];
    let mut idle = 0u32;
    let mut slow = false;
    let deadline = Instant::now() + timeout;
    while sent != [sr.len(), sl.len()] || received != [rl.len(), rr.len()] {
        let before = sent[0] + sent[1] + received[0] + received[1];
        for (i, (stream, bytes)) in [(right, sr), (left, sl)].into_iter().enumerate() {
            if sent[i] < bytes.len() {
                match stream.unwrap().write(&bytes[sent[i]..]) {
                    Ok(0) => return Err("peer closed while sending".into()), Ok(n) => sent[i] += n,
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
        for (i, (stream, bytes)) in [(left, &mut *rl), (right, &mut *rr)].into_iter().enumerate() {
            if received[i] < bytes.len() {
                match stream.unwrap().read(&mut bytes[received[i]..]) {
                    Ok(0) => return Err("peer closed while receiving".into()), Ok(n) => received[i] += n,
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                    Err(e) => return Err(e.to_string()),
                }
            }
        }
        idle = idle.wrapping_add(1);
        if idle & 4095 == 0 {
            let now = Instant::now();
            if now > deadline { return Err("duplex exchange timed out".into()); }
            slow = deadline - now < timeout - Duration::from_millis(1);
        }
        if sent[0] + sent[1] + received[0] + received[1] != before { slow = false; }
        if slow { std::thread::sleep(Duration::from_micros(100)); }
        spin_loop();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    #[test]
    fn auxiliary_rdma_setup_stall_falls_back_after_tcp_connects() {
        fn stalled_setup(ring:&mut Ring,timeout:Duration)->Result<()> {
            assert_eq!(timeout,AUX_RDMA_TIMEOUT,"RDMA setup must get the auxiliary timeout before it starts");
            let socket=ring.tx.as_ref().unwrap();
            // No host RDMA device is needed. Both peers withhold the setup hello after TCP connects.
            exchange_timeout(socket,socket,&[],&mut [0;24],timeout)
        }
        let listeners=(0..4).map(|_|TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
        let addresses=listeners.iter().map(|s|s.local_addr().unwrap()).collect::<Vec<_>>();drop(listeners);
        let (primary,auxiliary)=addresses.split_at(addresses.len()/2);
        let started=Instant::now();
        std::thread::scope(|scope| {
            let mut workers=Vec::new();
            for rank in 0..primary.len() {
                workers.push(scope.spawn(move || {
                    let next=(rank+1)%primary.len();
                    let mut ring=Ring::connect_options(rank,primary.len(),primary[rank],primary[next],false).unwrap();
                    AUX_RDMA_SETUP.with(|hook|hook.set(Some(stalled_setup)));
                    ring.aux=Ring::open_configured_aux(rank,auxiliary.len(),auxiliary[rank],auxiliary[next],false,true);
                    AUX_RDMA_SETUP.with(|hook|hook.set(None));
                    assert!(ring.aux.is_none(),"stalled RDMA setup must reject the auxiliary rail");
                    let gathered=ring.gather_batch_f32(&[rank as f32],primary.len(),1,1).unwrap();
                    assert_eq!(gathered,(0..primary.len()).map(|i|i as f32).collect::<Vec<_>>());
                    ring.barrier().unwrap();
                }));
            }
            for worker in workers {worker.join().unwrap();}
        });
        // The setup did stall for its own timeout, and the whole fallback stays inside the auxiliary budget
        // (the wait of the connection, the setup, the probe), far below the 30 seconds of the main link.
        let elapsed=started.elapsed();let budget=AUX_CONNECT_TIMEOUT+2*AUX_RDMA_TIMEOUT;
        assert!(elapsed>=AUX_RDMA_TIMEOUT,"the stalled setup returned after {elapsed:?}, before its timeout");
        assert!(elapsed<budget && budget<TIMEOUT,"auxiliary RDMA setup took {elapsed:?}");
    }
    #[test]
    fn command_relay_only_from_four_ranks_and_the_pair_asks_main_only() {
        // The gate reads the configured world: never one, two or three ranks, never a ring.
        for world in 1..COMMAND_RELAY_RANKS { for path in [false, true] { assert!(!command_relay_world(world, path)); } }
        for world in COMMAND_RELAY_RANKS..=64 { assert!(command_relay_world(world, true) && !command_relay_world(world, false)); }
        // Call identity: main asks a pair `is_pair() && has_mailbox()` and nothing else. The
        // pair here asks has_mailbox once and the relay never, with main's result; every other
        // layout asks the relay once and has_mailbox never.
        for world in 2..=8 { for path in [false, true] { for up in [false, true] {
            let (mailbox, relay) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
            let pair = PeerLayout::new(world, path).is_pair();
            let route = command_route(pair, || { mailbox.set(mailbox.get() + 1); up }, || { relay.set(relay.get() + 1); up });
            let expected = match (pair, up) { (true, true) => CommandRoute::PairMailbox, (false, true) => CommandRoute::Relay, _ => CommandRoute::Tcp };
            assert_eq!((route, mailbox.get(), relay.get()), (expected, pair as usize, !pair as usize), "world {world} path {path} up {up}");        } } }
    }
    #[test]
    fn command_idle_and_payload_deadlines() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap(); let mut sender = TcpStream::connect(listener.local_addr().unwrap()).unwrap(); let (receiver, _) = listener.accept().unwrap();
        let timeout = Duration::from_millis(20);
        let send = std::thread::spawn(move || { std::thread::sleep(timeout * 3); sender.write_all(&[7]).unwrap(); sender });
        command_read(&receiver, &mut [0], None, timeout).unwrap(); let mut sender = send.join().unwrap();
        assert!(command_read(&receiver, &mut [0], Some(Instant::now() + timeout), timeout).is_err());
        sender.write_all(&[8]).unwrap(); assert!(command_read(&receiver, &mut [0; 2], None, timeout).is_err());
    }
    #[test]
    fn the_step_barrier_of_a_pair_without_a_mailbox_is_the_control_stream_barrier() {
        // Loopback has no fabric mailbox: the step barrier must fall back, on both ranks, and stay in step with `barrier`.
        let listeners = (0..2).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
        let addresses = listeners.iter().map(|s| s.local_addr().unwrap()).collect::<Vec<_>>();
        drop(listeners);
        std::thread::scope(|scope| {
            for rank in 0..2 {
                let addresses = &addresses;
                scope.spawn(move || {
                    let mut ring = Ring::connect_options(rank, 2, addresses[rank], addresses[(rank + 1) % 2], true).unwrap();
                    for _ in 0..3 { ring.step_barrier().unwrap(); ring.barrier().unwrap(); }
                    let mut values = vec![(rank + 1) as f32; 3]; ring.reduce_small(&mut values).unwrap();
                    assert_eq!(values, [3.0; 3]);
                    ring.step_barrier().unwrap();
                });
            }
        });
        // One rank alone has nothing to wait for.
        let alone = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        Ring::connect_options(0, 1, alone, alone, true).unwrap().step_barrier().unwrap();
    }
    #[test]
    fn odd_rings_uneven_and_empty_partitions() {
        for (world, path) in [1, 2, 3, 4, 5, 8].into_iter().flat_map(|n| [false, true].map(move |p| (n, p))) {
            let listeners = (0..world).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
            let addresses = listeners.iter().map(|s| s.local_addr().unwrap()).collect::<Vec<_>>();
            drop(listeners);
            std::thread::scope(|scope| {
                for rank in 0..world {
                    let addresses = &addresses;
                    scope.spawn(move || {
                        let mut ring = Ring::connect_options(rank, world, addresses[rank], addresses[(rank + 1) % world], path).unwrap();
                        ring.barrier().unwrap();
                        for units in [1, 3, 8, 19] { for tokens in [1, 7, 16] {
                            let width = 4;
                            let r = rows(units, rank, world);
                            let local = (0..tokens).flat_map(|t| (r.start*width..r.end*width).map(move |i| (t*units*width+i) as f32)).collect::<Vec<_>>();
                            let result = ring.gather_batch_f32(&local, units, width, tokens).unwrap();
                            assert!(result.iter().enumerate().all(|(i,&v)| v == i as f32));
                        }
                        let mut values=vec![(rank+1) as f32;19];ring.reduce_small(&mut values).unwrap();
                        assert!(values.iter().all(|&v|v==(world*(world+1)/2) as f32));
                        }
                        let mut command = if rank == 0 { [19u8; 24] } else { [0; 24] };
                        ring.command(&mut command).unwrap();
                        assert_eq!(command, [19; 24]);
                        // A payload beyond the socket buffers takes the blocking send
                        // fallback; a command after the spin window parks the reader.
                        let mut payload = vec![if rank == 0 { 7u8 } else { 0 }; 4 << 20];
                        ring.command_payload(&mut payload).unwrap();
                        assert!(payload.iter().all(|&b| b == 7));
                        if rank == 0 { std::thread::sleep(COMMAND_SPIN * 2); }
                        let mut late = if rank == 0 { [5u8; 8] } else { [0; 8] };
                        ring.command(&mut late).unwrap();
                        assert_eq!(late, [5; 8]);
                        ring.barrier().unwrap();
                    });
                }
            });
        }
    }
    /// A hand-off longer than one 256 KiB relay piece crosses every hop to each destination
    /// whole; ranks below the destination keep their bytes. The batch form sends each buffer.
    #[test]
    fn final_rank_transfer_relays_multi_piece_payloads() {
        let pattern = |len: usize, salt: usize| (0..len).map(|i| ((i * 31 + salt) % 251) as u8).collect::<Vec<_>>();
        for (world, path) in [3, 4].into_iter().flat_map(|n| [false, true].map(move |p| (n, p))) {
            let listeners = (0..world).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect::<Vec<_>>();
            let addresses = listeners.iter().map(|s| s.local_addr().unwrap()).collect::<Vec<_>>();
            drop(listeners);
            std::thread::scope(|scope| {
                for rank in 0..world {
                    let addresses = &addresses;
                    scope.spawn(move || {
                        let mut ring = Ring::connect_options(rank, world, addresses[rank], addresses[(rank + 1) % world], path).unwrap();
                        let last = rank == world - 1;
                        for destination in 0..world {
                            let mut bytes = if last { pattern(600_001, destination) } else { vec![0; 600_001] };
                            ring.transfer_from_last(destination, &mut bytes).unwrap();
                            let expect = if rank >= destination { pattern(600_001, destination) } else { vec![0; 600_001] };
                            assert!(bytes == expect, "world={world} path={path} rank={rank} destination={destination}");
                            let (mut a, mut b) = if last { (pattern(300_000, 1), pattern(300_000, 2)) } else { (vec![0; 300_000], vec![0; 300_000]) };
                            ring.transfer_from_last_batch(destination, &mut [&mut a[..], &mut [][..], &mut b[..]]).unwrap();
                            let received = rank >= destination;
                            assert!(a == if received { pattern(300_000, 1) } else { vec![0; 300_000] }, "batch a rank={rank} destination={destination}");
                            assert!(b == if received { pattern(300_000, 2) } else { vec![0; 300_000] }, "batch b rank={rank} destination={destination}");
                        }
                        assert!(ring.transfer_from_last(world, &mut [0; 4]).is_err());
                        ring.barrier().unwrap();
                    });
                }
            });
        }
    }
    #[test]
    fn dual_rail_floor_and_stripe_partition_stay_symmetric() {
        assert_eq!(DUAL_RAIL_FLOOR_BYTES, 64 << 10);
        assert_eq!(PAIR_GATHER_FLOOR_BYTES, 1 << 20);
        // Odd f32 counts keep 4-byte word boundaries and leave a shorter second stripe.
        assert_eq!(rail_stripe(0..(262_147 * 4), 0, 2), 0..(131_074 * 4));
        assert_eq!(rail_stripe(0..(262_147 * 4), 1, 2), (131_074 * 4)..(262_147 * 4));
        assert_eq!(rail_stripe(100..205, 0, 2), 100..153);
        assert_eq!(rail_stripe(100..205, 1, 2), 153..205);

        // Pair (is_pair=true): 15-token gathers (up to 960 KiB per owner) stay on one rail; >= 1 MiB stripes across two.
        assert_eq!(active_rdma_rails(2, true, false, false, true, &|part| part * (960 << 10)..(part + 1) * (960 << 10)), 1);
        assert_eq!(active_rdma_rails(2, true, false, false, true, &|part| part * PAIR_GATHER_FLOOR_BYTES..(part + 1) * PAIR_GATHER_FLOOR_BYTES), 2);
        // Uneven owner slices on a pair agree on one rail when either side falls below 1 MiB.
        assert_eq!(active_rdma_rails(2, true, false, false, true, &|part| if part == 0 { 0..PAIR_GATHER_FLOOR_BYTES } else { 0..(PAIR_GATHER_FLOOR_BYTES - 4) }), 1);
        assert_eq!(active_rdma_rails(2, true, false, false, false, &|part| part * (4 << 20)..(part + 1) * (4 << 20)), 1);
        assert_eq!(active_rdma_rails(2, true, true, true, true, &|part| part * (4 << 20)..(part + 1) * (4 << 20)), 1);
        assert_eq!(active_rdma_rails(2, true, true, false, true, &|part| part * 1024..(part + 1) * 1024), 2);
        // 3+ rank rings/paths (is_pair=false) keep main's immediate 2-rail gather behavior.
        assert_eq!(active_rdma_rails(4, false, false, false, true, &|part| part * 1024..(part + 1) * 1024), 2);
    }
}

