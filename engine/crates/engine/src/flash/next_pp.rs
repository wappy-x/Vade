//! Exact FP4-Local layer pipeline over a chain of full-duplex neighbor links.
//! Rank zero retains the frontend, head, sampling and drafting; each worker runs its
//! contiguous layer interval and relays final residuals back over the same cables.
//! Prompt tiles and decode steps preserve world-one kernels, FP32 bits and layer order.
//! Each boundary materializes the same per-element combine (see blocks.rs).
//! World two serves the tensor split (next_tp.rs) on this protocol; three or more ranks run it as a layer chain.
#[cfg(target_os = "linux")]
use super::*;
// Host-only protocol tests use the same fixed Flash-Next wire geometry, without CUDA.
#[cfg(not(target_os = "linux"))]
const LAYERS: usize = 48;
#[cfg(not(target_os = "linux"))]
const W: usize = 10240;
#[cfg(not(target_os = "linux"))]
const POOL_SIZE: usize = 16;
#[cfg(not(target_os = "linux"))]
type Result<T> = std::result::Result<T, String>;
#[cfg(not(target_os = "linux"))]
fn err(e: impl std::fmt::Display) -> String { e.to_string() }
use std::{
    collections::VecDeque,
    io::{ErrorKind, Read, Write},
    net::TcpStream,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};
#[cfg(target_os = "linux")]
use std::collections::BTreeMap;

#[path = "next_pp_prefetch.rs"]
mod prefetch;
use prefetch::{Incoming, Prefetch};
#[path = "next_pp_acks.rs"]
mod acks;

// Version 2: every request carries an exchange id that its reply echoes.
const MAGIC_REQUEST: u32 = 0x3250_5056; // "VPP2"
const MAGIC_RESPONSE: u32 = 0x3252_5056; // "VPR2"
const MAGIC_HELLO: u32 = 0x3448_5056; // "VPH4": explicit pooled-storage release on transitions
/// Request header words: magic, kind, rows, chats, exchange id (then 4 words per chat).
const REQUEST_HEAD: usize = 20;
/// Reply header words: magic, status, payload bytes, exchange id.
const REPLY_HEAD: usize = 16;
pub(super) const KIND_ROWS: u32 = 1;
pub(super) const KIND_POOLED: u32 = 2;
/// A prompt tile whose rows the front does not project; the back acks
/// it without returning rows, so the front runs its next tile while the back runs this one.
pub(super) const KIND_ROWS_ASYNC: u32 = 3;
/// Queued prompt tiles held for the sender thread at most.
const QUEUED: usize = 4;
/// Reply status: 0 done, 1 failed (payload = error text), 2 working: a heartbeat header with
/// no payload and the id of the request the back is running (sent on the TCP link).
const STATUS_WORKING: u32 = 2;
/// The back's heartbeat period: a quarter of the limit, at most 500 ms.
fn beat_period(limit: Duration) -> Duration {
    (limit / 4).min(Duration::from_millis(500))
}
fn working_head(id: u32) -> Vec<u8> {
    let mut head = Vec::with_capacity(REPLY_HEAD);
    for v in [MAGIC_RESPONSE, STATUS_WORKING, 0, id] {
        put(&mut head, v);
    }
    head
}
/// MTP verify rows for layers [split, 48); the back records its own GDN
/// tape and commits the accepted prefix when the chat's next request names its position.
pub(super) const KIND_VERIFY: u32 = 4;
/// A queued prompt tile whose rows the back keeps;
/// they return ahead of the rows of the next synchronous reply, and the front feeds them
/// to the drafters of their chats (the draft history world 1 writes for every prompt row).
/// When the peer is the last rank, the ack carries this tile's rows immediately. The
/// back writes that ack on `pp-replier` so the next tile's compute is not behind the copy.
pub(super) const KIND_ROWS_KEEP: u32 = 5;
/// Final prompt tile. The request rides the sender thread (TCP, in order with the queued
/// tiles) so its write overlaps the earlier acks instead of waiting behind `drain`.
/// The reply still carries rows, including kept rows owed by earlier tiles, and uses the
/// RDMA wire when that reply fits (same bound as a synchronous exchange).
pub(super) const KIND_ROWS_FINAL: u32 = 6;
/// The front's cohort controller authorized reclamation. Carried with the next
/// request (TCP or wire), after earlier queued work and before this request's work.
/// It also covers a filtered single row: request shape alone is not authorization.
const RELEASE_POOLED: u32 = 1 << 31;
/// Kept rows retained at most, on the back and on the front until their chats observe them
/// (32768 rows = 1.25 GiB of host memory); past it the front runs the tile synchronously, so
/// its rows enter the draft history at once (pp_keep_room).
const KEEP_ROWS: usize = 32768;

/// One chat's span of a kept tile: its slot, position, tokens and rows within the tile.
pub(super) struct KeepPart {
    pub(super) slot: u32,
    pub(super) pos: usize,
    pub(super) tokens: Vec<u32>,
    pub(super) start: usize,
}
/// A kept tile: its chats' spans, its row count, and its rows once they came back.
struct KeptTile {
    parts: Vec<KeepPart>,
    rows: usize,
    data: Option<Vec<f32>>,
}

/// Over RDMA (SEQPAR_RDMA unset or 1): a polled message wire (RoCE send/receive, built by spark-tp) for the
/// synchronous exchanges that fit `capacity` bytes (pooled decode steps, short projected
/// tiles). Messages and their order are the TCP ones; only the carrier changes.
pub trait PpWire: Send {
    fn capacity(&self) -> usize;
    /// Send one message made of `parts`.
    fn send(&mut self, parts: &[&[u8]]) -> Result<()>;
    /// Poll once; true when `out` now holds one whole message.
    fn try_recv(&mut self, out: &mut Vec<u8>) -> Result<bool>;
}

/// LINK_TIMEOUT (both ranks use the same; the hello
/// checks it): the longest silence a rank accepts from its peer. Once a message's first byte
/// arrived its rest must arrive within it, and a front send must make progress within it.
/// While the back runs a request it sends a heartbeat header every beat_period, so the
/// front's wait for a reply is bounded by the back's liveness, not by the work: a long
/// prompt tile is never cut off, a stopped or vanished back is noticed within the limit.
/// Heartbeats vouch for liveness, not speed: they come while the back process runs and its
/// serving thread is inside the request (a stop, a crash, or that thread returning or
/// unwinding ends them), with no cap on the request's work, so no tile at any context is
/// cut off for being slow. A GPU that wedges inside a request is a hang, as in world 1
/// (the wedge runbook's probes find it), never a wrong reply. A request on the
/// RDMA wire heartbeats on the TCP link too, and its clock starts once it is sent (after
/// the queued tiles' acks, whose waits have their own heartbeat clocks). Past the limit
/// without a heartbeat the link is dropped and the request fails. (The back waits for the
/// start of the next request without a limit: idle is normal.)
pub const LINK_TIMEOUT: Duration = Duration::from_secs(10);

/// Agree on the optional short-message wire before any GPU resources are opened.
/// Authenticated sockets still carry the authentication phase's short deadline.
pub fn negotiate_wire(stream: &mut TcpStream, want: bool, limit: Duration) -> Result<()> {
    // Wire setup advances from front to back. A middle/tail can finish admission
    // before earlier ranks finish connecting, so the authentication deadline must
    // not leak into this phase (including the RDMA endpoint exchange that follows).
    stream.set_read_timeout(Some(limit)).map_err(err)?;
    stream.set_write_timeout(Some(limit)).map_err(err)?;
    stream.write_all(&[u8::from(want)]).map_err(err)?;
    let mut peer = [0u8; 1];
    stream.read_exact(&mut peer).map_err(err)?;
    if peer[0] != u8::from(want) {
        return Err(format!("SEQPAR_RDMA differs across ranks (this {}, peer {})", u8::from(want), peer[0]));
    }
    Ok(())
}

/// Bound the three registered mailbox slots using context and prompt geometry.
/// Wider responses retain TCP; the 512 MiB slot cap funds at most 1.5 GiB per endpoint.
const MIN_WIRE_BYTES: usize = 8 << 20;
const MAX_WIRE_BYTES: usize = 512 << 20;
pub fn wire_capacity(context: usize, tile: usize) -> Result<usize> {
    if context == 0 || tile == 0 { return Err("pipeline wire needs nonzero context and tile".into()); }
    let rows = context.min(KEEP_ROWS.checked_add(tile).ok_or("pipeline wire row overflow")?)
        .max(tile.max(POOL_SIZE));
    let wanted = rows.checked_mul(W).and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(REQUEST_HEAD + POOL_SIZE * 16)).ok_or("pipeline wire capacity overflow")?;
    let slot = wanted.clamp(MIN_WIRE_BYTES, MAX_WIRE_BYTES);
    validate_wire_capacity(slot)?;
    Ok(slot)
}
fn validate_wire_capacity(slot: usize) -> Result<()> {
    if !(MIN_WIRE_BYTES..=MAX_WIRE_BYTES).contains(&slot) || u32::try_from(slot).is_err()
        || slot.checked_mul(3).is_none() {
        return Err("pipeline wire capacity outside supported bounds".into());
    }
    Ok(())
}
/// Agree before allocating/registering memory; mixed capacities cannot overrun a receive.
pub fn negotiate_capacity(stream: &mut TcpStream, slot: usize) -> Result<()> {
    validate_wire_capacity(slot)?;
    let mut hello = [0u8; 12];
    hello[..4].copy_from_slice(b"VPC1");
    hello[4..].copy_from_slice(&(slot as u64).to_le_bytes());
    stream.write_all(&hello).map_err(err)?;
    let mut peer = [0u8; 12];
    stream.read_exact(&mut peer).map_err(err)?;
    if peer != hello { return Err("pipeline wire capacity or protocol differs across ranks".into()); }
    Ok(())
}
fn wire_reply_size(owed: usize, bytes: usize) -> Option<usize> {
    owed.checked_mul(W)?.checked_mul(4)?.checked_add(bytes)?.checked_add(REPLY_HEAD)
}
fn wire_fits(capacity: usize, head: usize, bytes: usize, owed: usize) -> bool {
    head.checked_add(bytes).is_some_and(|n| n <= capacity)
        && wire_reply_size(owed, bytes).is_some_and(|n| n <= capacity)
}

/// Peer-supplied error text, made safe for logs and responses: printable ASCII, bounded.
fn peer_text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(&bytes[..bytes.len().min(512)])
        .chars()
        .map(|c| if c.is_ascii_graphic() || c == ' ' { c } else { '?' })
        .collect()
}

/// A contiguous stage of the layer chain. Geometry depends only on layer/rank counts.
/// The front keeps at least two layers because layer 1 consumes the frontend's PLE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stage {
    pub rank: usize,
    pub ranks: usize,
    pub first: usize,
    pub end: usize,
}
impl Stage {
    pub fn new(rank: usize, ranks: usize, split: Option<usize>) -> Result<Self> {
        if !(2..=LAYERS / 2).contains(&ranks) || rank >= ranks {
            return Err(format!("pipeline needs rank < WORLD in 2..={}", LAYERS / 2));
        }
        let (first, end) = if let Some(split) = split {
            // One explicit cut describes the intervals between these boundaries.
            let boundaries = [0, split, LAYERS];
            if ranks + 1 != boundaries.len() || !(2..LAYERS).contains(&split) {
                return Err("a two-rank cut needs two ranks and a cut in 2..48".into());
            }
            (boundaries[rank], boundaries[rank + 1])
        } else {
            (LAYERS * rank / ranks, LAYERS * (rank + 1) / ranks)
        };
        Ok(Self { rank, ranks, first, end })
    }
    pub fn successor(&self) -> Option<usize> {
        (self.rank + 1..self.ranks).next()
    }
    /// Every stage's layer interval in this geometry; each rank derives the same list (an
    /// explicit split moves the one cut between two ranks, which both of them hold).
    pub fn intervals(&self) -> Vec<std::ops::Range<usize>> {
        let mut cuts: Vec<usize> = (0..=self.ranks).map(|r| LAYERS * r / self.ranks).collect();
        (cuts[self.rank], cuts[self.rank + 1]) = (self.first, self.end);
        cuts.windows(2).map(|w| w[0]..w[1]).collect()
    }
    /// This rank's stage: the tensor split cuts after its first SPLIT layers (rank 1 runs every
    /// split pass from there); a longer chain cuts the layers evenly.
    pub fn for_rank(rank: usize, ranks: usize) -> Result<Self> {
        Self::new(rank, ranks, crate::flash_tp_switch::on().then_some(crate::flash_tp_switch::SPLIT))
    }
}

/// Snapshot once per process; tests supply this directly without changing global env.
#[derive(Clone, Copy)]
struct Recipe {
    limit: Duration,
    asynchronous: bool,
    mtp: bool, tp: bool,
}
impl Recipe {
    fn from_env() -> Result<Self> {
        // Every served pipeline queues prompt tiles and verifies MTP drafts across its ranks.
        Ok(Self { limit: LINK_TIMEOUT, asynchronous: true, mtp: true, tp: crate::flash_tp_switch::on() })
    }
}

/// A worker completes its pooled step at its own boundary. Front fragments complete only
/// with the final head, after the answer has returned from the entire chain.
pub(super) fn pool_advances_position(end: usize, head: bool) -> bool {
    !head || end == LAYERS
}

fn put(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
}
pub(super) fn f32_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
pub(super) fn f32_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast(), std::mem::size_of_val(v)) }
}

/// Read exactly `buf`. `idle`: the start of the next message may take any time (the back
/// between requests; `stop` is rechecked on each socket timeout) and a close before its
/// first byte is a clean end, Ok(false). Otherwise, and once the first byte arrived, the
/// rest must arrive within `limit`: a close or a stall mid-message is an error, never a
/// partial message. The socket's read timeout must be set (it paces the checks).
fn read_message(stream: &mut TcpStream, buf: &mut [u8], stop: Option<&AtomicBool>, limit: Duration, idle: bool) -> Result<bool> {
    let mut at = 0;
    let mut since = (!idle).then(Instant::now);
    while at < buf.len() {
        match stream.read(&mut buf[at..]) {
            Ok(0) if at == 0 && idle => return Ok(false),
            Ok(0) => return Err(format!("pipeline peer closed mid-message ({at} of {} bytes)", buf.len())),
            Ok(n) => {
                at += n;
                since.get_or_insert_with(Instant::now);
            }
            Err(e) if e.kind() == ErrorKind::Interrupted => (),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                if stop.is_some_and(|s| s.load(Ordering::Relaxed)) {
                    return Err("pipeline back stopped by signal".into());
                }
            }
            Err(e) => return Err(format!("pipeline read: {e}")),
        }
        if since.is_some_and(|t| t.elapsed() > limit) {
            return Err(format!("pipeline peer read timed out after {} ms ({at} of {} bytes)", limit.as_millis(), buf.len()));
        }
    }
    Ok(true)
}
/// One whole message part within `limit` (see read_message).
fn read_rest(stream: &mut TcpStream, buf: &mut [u8], stop: Option<&AtomicBool>, limit: Duration) -> Result<()> {
    read_message(stream, buf, stop, limit, false).map(|_| ())
}

/// Write all of `bytes`; a send that times out without progress (the back is running a
/// tile and not reading yet) retries until `stop` is set (the link failed or the server
/// stops). Other errors end the send.
fn send_all(writer: &mut TcpStream, mut bytes: &[u8], stop: &AtomicBool) -> std::io::Result<()> {
    while !bytes.is_empty() {
        match writer.write(bytes) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => bytes = &bytes[n..],
            Err(e) if e.kind() == ErrorKind::Interrupted => (),
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) && !stop.load(Ordering::Relaxed) => (),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The front's end of the link.
pub(super) struct Link {
    stream: TcpStream,
    pub(super) split: usize,
    pub(super) buffer: Vec<f32>,
    exchanges: u64,
    bytes: u64,
    wait: Duration,
    /// Queued (header, rows) messages for the sender thread, the
    /// thread's returned row buffers, and the acks still owed by the back.
    queue: Option<mpsc::SyncSender<(Vec<u8>, Vec<f32>)>>,
    free: Option<mpsc::Receiver<Vec<f32>>>,
    pending: usize,
    posted: u64,
    /// Over RDMA: the polled wire, its message buffer, exchanges it carried.
    fast: Option<Box<dyn PpWire>>,
    message: Vec<u8>,
    carried: u64,
    /// Set when the link fails: the front server stops (the pipeline cannot recover).
    down: &'static AtomicBool,
    /// After any failure the link is never used again (a late reply could pair with the
    /// wrong request).
    pub(super) poisoned: bool, pub(super) tp: bool, pub(super) tp_verify: bool, pub(super) tp_mail: bool, pub(super) tp_handoff: bool, pub(super) tp_tile: bool, pub(super) tp_heads: bool, pub(super) tp_head: bool,
    /// Exchange ids: the last request sent and the last reply read (replies come in order).
    sent: u32,
    replied: u32,
    limit: Duration,
    /// Heartbeat headers read (the back was running a long request), and late ones: a
    /// heartbeat rides TCP, so one of a wire exchange can arrive after the wire's reply.
    beats: u64,
    late: u64,
    /// The oldest exchange id whose heartbeat may still arrive: the back writes a request's
    /// heartbeats before its reply and before it reads the next request, so after a TCP
    /// reply (or a heartbeat) for id t no heartbeat for an id below t can follow. Ids from
    /// `floor` to `replied` are answered wire exchanges whose late heartbeats are ignored.
    floor: u32,
    /// Verify rows may cross (KIND_VERIFY).
    mtp: bool,
    /// Kept tiles in post order, and the rows the back still holds.
    kept: Vec<KeptTile>,
    owed: usize,
    /// Middle ranks relay the final stage's kept rows without interpreting them.
    relay_kept: Option<Vec<f32>>,
    /// Posted non-final tiles, oldest first. `Some(rows)` means the ack carries those
    /// rows (the peer is the last rank). `None` is an empty ack.
    acks: VecDeque<Option<usize>>,
    /// The peer is the last rank, so each KEEP ack carries that tile's rows.
    inline_kept: bool,
    /// A draft transition still has to release downstream pooled decode storage.
    release_pool: bool,
}
fn request_head(kind: u32, rows: usize, meta: &[[u32; 4]], id: u32) -> Vec<u8> {
    let mut head = Vec::with_capacity(REQUEST_HEAD + meta.len() * 16);
    for v in [MAGIC_REQUEST, kind, rows as u32, meta.len() as u32, id] {
        put(&mut head, v);
    }
    for m in meta {
        for &v in m {
            put(&mut head, v);
        }
    }
    head
}
impl Link {
    /// The downstream end of one chain edge, shared by the front and middle ranks.
    fn connect(mut stream: TcpStream, fast: Option<Box<dyn PpWire>>, stage: Stage, context: usize, active: usize, tile: usize, down: &'static AtomicBool, recipe: Recipe) -> Result<Self> {
        let limit = recipe.limit;
        stream.set_nonblocking(false).map_err(err)?;
        stream.set_nodelay(true).map_err(err)?;
        // Short socket timeouts pace the deadline checks; `limit` bounds each message.
        stream.set_read_timeout(Some(Duration::from_millis(100))).map_err(err)?;
        stream.set_write_timeout(Some(limit)).map_err(err)?;
        let mut hello = Vec::new();
        for v in [MAGIC_HELLO, stage.end as u32, context as u32, active as u32, tile as u32, u32::from(fast.is_some()), limit.as_millis() as u32, stage.ranks as u32, stage.rank as u32, u32::from(recipe.mtp) | u32::from(recipe.tp) << 1] {
            put(&mut hello, v);
        }
        stream.write_all(&hello).map_err(err)?;
        let mut ack = [0u8; 8];
        read_rest(&mut stream, &mut ack, None, limit)?;
        if word(&ack, 0) != MAGIC_HELLO || word(&ack, 1) != 0 {
            return Err("pipeline back refused the admission shape".into());
        }
        let (queue, free) = if recipe.asynchronous {
            let mut writer = stream.try_clone().map_err(err)?;
            let (queue, inbox) = mpsc::sync_channel::<(Vec<u8>, Vec<f32>)>(QUEUED);
            let (spare, free) = mpsc::channel::<Vec<f32>>();
            std::thread::Builder::new()
                .name("pp-sender".into())
                .spawn(move || {
                    for (head, rows) in inbox {
                        // A send that waits on a busy back retries; the front's main thread
                        // judges the back's liveness (heartbeats) and sets `down` on failure.
                        if let Err(e) = send_all(&mut writer, &head, down).and_then(|()| send_all(&mut writer, f32_bytes(&rows), down)) {
                            engine_model::events::note_line("pp_sender_error", &format!("flashnext pp sender: {e}"));
                            return;
                        }
                        let _ = spare.send(rows);
                    }
                })
                .map_err(err)?;
            eprintln!("flashnext pp front async tiles on");
            (Some(queue), Some(free))
        } else {
            (None, None)
        };
        Ok(Self {
            stream,
            split: stage.end,
            buffer: Vec::new(),
            exchanges: 0,
            bytes: 0,
            wait: Duration::ZERO,
            queue,
            free,
            pending: 0,
            posted: 0,
            fast,
            message: Vec::new(),
            carried: 0,
            down,
            poisoned: false, tp: recipe.tp, tp_verify: false, tp_mail: false, tp_handoff: false, tp_tile: false, tp_heads: false, tp_head: false,
            sent: 0,
            replied: 0,
            limit,
            beats: 0,
            late: 0,
            floor: 1,
            mtp: recipe.mtp,
            kept: Vec::new(),
            owed: 0,
            relay_kept: None,
            acks: VecDeque::new(),
            inline_kept: stage.rank + 2 == stage.ranks,
            release_pool: false,
        })
    }
    /// Any front-side pipeline error leaves the two ranks' session state unknown: the link is
    /// poisoned and the server stops.
    pub(super) fn fail(&mut self, e: String) -> String {
        self.poisoned = true;
        if !self.down.swap(true, Ordering::Relaxed) {
            engine_model::events::note_line("pp_link_failed", &format!("flashnext pp front: link failed ({e}); stopping the server"));
        }
        #[cfg(target_os = "linux")]
        spark_gpu_probe::teardown::arm();
        e
    }
    /// Check a reply header: magic and the exchange id it must answer; returns (status, len).
    fn reply_head(&mut self, reply: &[u8], expect: u32) -> Result<(u32, usize)> {
        if reply.len() < REPLY_HEAD || word(reply, 0) != MAGIC_RESPONSE {
            return Err("pipeline response magic mismatch".into());
        }
        let id = word(reply, 3);
        if id != expect || self.replied == self.sent {
            return Err(format!("pipeline reply out of order: id {id}, expected {expect}"));
        }
        self.replied = id;
        Ok((word(reply, 1), word(reply, 2) as usize))
    }
    /// Read one TCP header for exchange `expect`; it must arrive within `limit` of `heard`
    /// (the last sign of life). None for a heartbeat (the back is still running `expect`;
    /// `heard` moves) or a late one of an answered wire exchange (`heard` stays), else the
    /// reply's (status, len) (`heard` moves).
    fn frame(&mut self, expect: u32, heard: &mut Instant) -> Result<Option<(u32, usize)>> {
        let silent = || format!("pipeline back reply timed out after {} ms without a heartbeat", self.limit.as_millis());
        let left = self.limit.saturating_sub(heard.elapsed());
        if left.is_zero() {
            return Err(silent());
        }
        let mut reply = [0u8; REPLY_HEAD];
        read_rest(&mut self.stream, &mut reply, None, left).map_err(|e| if e.contains("timed out") { silent() } else { e })?;
        if word(&reply, 0) == MAGIC_RESPONSE && word(&reply, 1) == STATUS_WORKING {
            let id = word(&reply, 3);
            if word(&reply, 2) == 0 && id == expect && self.replied != self.sent {
                self.beats += 1;
                (self.floor, *heard) = (id, Instant::now());
                return Ok(None);
            }
            if word(&reply, 2) == 0 && id.wrapping_sub(self.floor) < self.replied.wrapping_add(1).wrapping_sub(self.floor) {
                self.late += 1;
                self.floor = id;
                return Ok(None);
            }
            return Err(format!("pipeline heartbeat out of order: id {id}, expected {expect}"));
        }
        let reply = self.reply_head(&reply, expect)?;
        (self.floor, *heard) = (expect.wrapping_add(1), Instant::now());
        Ok(Some(reply))
    }
    /// The reply header for exchange `expect` (sent just now): heartbeats are skipped, and
    /// the back must show life within `limit` of the send and of each heartbeat.
    fn await_reply(&mut self, expect: u32) -> Result<(u32, usize)> {
        let mut heard = Instant::now();
        loop {
            if let Some(reply) = self.frame(expect, &mut heard)? {
                return Ok(reply);
            }
        }
    }
    /// One header owed by the oldest queued tile: a heartbeat (false) or its ack (true).
    fn queued_frame(&mut self, heard: &mut Instant) -> Result<bool> {
        let Some((status, len)) = self.frame(self.replied.wrapping_add(1), heard)? else { return Ok(false) };
        self.pending -= 1;
        let expect_rows = self.acks.pop_front().ok_or("pipeline queued ack without a tile")?;
        if status != 0 {
            let mut text = vec![0u8; len.min(1 << 16)];
            read_rest(&mut self.stream, &mut text, None, self.limit)?;
            return Err(format!("pipeline back failed a queued tile: {}", peer_text(&text)));
        }
        if let Some(n) = expect_rows {
            if len != n * W * 4 {
                return Err("pipeline kept ack size mismatch".into());
            }
            let mut data = vec![0f32; n * W];
            read_rest(&mut self.stream, f32_bytes_mut(&mut data), None, self.limit)?;
            if let Some(relay) = self.relay_kept.as_mut() {
                relay.extend_from_slice(&data);
            } else {
                let tile = self.kept.iter_mut().find(|t| t.data.is_none()).ok_or("pipeline kept ack with no tile")?;
                if tile.rows != n {
                    return Err("pipeline kept ack does not match the tile".into());
                }
                tile.data = Some(data);
            }
            self.owed = self.owed.checked_sub(n).ok_or("pipeline kept ack exceeds owed rows")?;
        } else if len != 0 {
            return Err("pipeline queued-tile ack carries rows".into());
        }
        Ok(true)
    }
    /// Read the acks of every queued tile (in order); the first back failure is returned.
    /// Afterwards the sender thread has handed every queued byte to the socket.
    pub(super) fn drain(&mut self) -> Result<()> {
        let mut heard = Instant::now();
        while self.pending > 0 {
            self.queued_frame(&mut heard)?;
        }
        Ok(())
    }
    /// Queue a non-final prompt tile (rows already copied to `rows`) for the sender thread.
    /// `keep`: the back keeps the tile's rows for these chats' drafters (within KEEP_ROWS).
    pub(super) fn post(&mut self, meta: &[[u32; 4]], rows: Vec<f32>, keep: Option<Vec<KeepPart>>) -> Result<()> {
        if rows.len() % W != 0 || meta.is_empty() || meta.len() > POOL_SIZE {
            return Err("invalid pipeline exchange shape".into());
        }
        let n = rows.len() / W;
        if keep.is_some() && !self.keep_room(n) {
            return Err("pipeline kept rows past the limit".into());
        }
        let kind = if keep.is_some() { KIND_ROWS_KEEP } else if std::mem::take(&mut self.tp_verify) { crate::flash_tp_switch::KIND_VERIFY } else if std::mem::take(&mut self.tp_handoff) { crate::flash_tp_switch::KIND_HANDOFF } else { KIND_ROWS_ASYNC };
        let inline = keep.is_some() && self.inline_kept;
        if let Some(parts) = keep {
            self.kept.push(KeptTile { parts, rows: n, data: None });
            self.owed += n;
        }
        let mail = if std::mem::take(&mut self.tp_mail) { crate::flash_tp_switch::MAIL } else { 0 } | if std::mem::take(&mut self.tp_heads) { crate::flash_tp_switch::HEADS } else { 0 }
            | if std::mem::take(&mut self.tp_head) { crate::flash_tp_switch::HEAD } else { 0 }; self.enqueue(kind | mail, meta, rows)?;
        self.acks.push_back(inline.then_some(n));
        Ok(())
    }
    /// Queue the final prompt tile on the sender thread. Its reply carries rows, so
    /// `finish_final` collects it; `drain` must not (a queued ack with rows is an error).
    fn post_final(&mut self, meta: &[[u32; 4]], rows: Vec<f32>) -> Result<()> {
        if rows.len() % W != 0 || meta.is_empty() || meta.len() > POOL_SIZE {
            return Err("invalid pipeline exchange shape".into());
        }
        self.enqueue(KIND_ROWS_FINAL, meta, rows)
    }
    /// Read the acks of tiles queued before the final one, then the final tile's rows.
    /// The sender thread writes the final tile during the ack reads.
    fn finish_final(&mut self, residual: &mut [f32]) -> Result<()> {
        if self.pending == 0 {
            return Err("pipeline final tile was not posted".into());
        }
        let started = Instant::now();
        let mut heard = Instant::now();
        while self.pending > 1 {
            self.queued_frame(&mut heard)?;
        }
        let bytes = std::mem::size_of_val(residual);
        // Same fit test the back uses for a KIND_ROWS_FINAL reply. The request itself
        // stayed on TCP; only the row-bearing reply takes the wire.
        let fits = wire_reply_size(self.owed, bytes).is_some_and(|n| self.fast.as_ref().is_some_and(|f| n <= f.capacity()));
        if fits {
            let mut fast = self.fast.take().ok_or("pipeline wire missing")?;
            let result = self.wire_collect(&mut *fast, residual);
            self.fast = Some(fast);
            result?;
            self.carried += 1;
            self.pending -= 1;
            self.account(started, bytes as u64, residual.len() / W);
            return Ok(());
        }
        let len = self.take_tcp_reply(residual)?;
        self.pending -= 1;
        self.account(started, len as u64, residual.len() / W);
        Ok(())
    }
    /// The TCP reply of the request just sent (`self.sent`): status, then kept rows owed
    /// by earlier tiles, then `residual`. Returns the payload byte count.
    fn take_tcp_reply(&mut self, residual: &mut [f32]) -> Result<usize> {
        let (status, len) = self.await_reply(self.sent)?;
        if status != 0 {
            let mut text = vec![0u8; len.min(1 << 16)];
            read_rest(&mut self.stream, &mut text, None, self.limit)?;
            return Err(format!("pipeline back failed: {}", peer_text(&text)));
        }
        if len != self.owed * W * 4 + std::mem::size_of_val(residual) {
            return Err("pipeline response size mismatch".into());
        }
        if self.owed > 0 {
            if let Some(relay) = self.relay_kept.as_mut() {
                let mut back = vec![0f32; self.owed * W];
                read_rest(&mut self.stream, f32_bytes_mut(&mut back), None, self.limit)?;
                *relay = back;
            } else {
                // Validate before consuming payload, including tiles whose chats ended.
                let expected = self.kept.iter().filter(|t| t.data.is_none())
                    .try_fold(0usize, |n, t| n.checked_add(t.rows));
                if expected != Some(self.owed) {
                    return Err("pipeline kept row accounting mismatch".into());
                }
                // Receive in post order directly into each tile, under one payload deadline.
                let receiving = Instant::now();
                for t in self.kept.iter_mut().filter(|t| t.data.is_none()) {
                    let mut data = vec![0f32; t.rows * W];
                    let limit = self.limit.checked_sub(receiving.elapsed())
                        .ok_or("pipeline kept rows read timed out")?;
                    read_rest(&mut self.stream, f32_bytes_mut(&mut data), None, limit)?;
                    t.data = Some(data);
                }
                self.kept.retain(|t| !t.parts.is_empty());
            }
            self.owed = 0;
        }
        read_rest(&mut self.stream, f32_bytes_mut(residual), None, self.limit)?;
        Ok(len)
    }
    /// Queue the exact kind and metadata; used on every forward edge.
    fn enqueue(&mut self, kind: u32, meta: &[[u32; 4]], rows: Vec<f32>) -> Result<()> {
        self.sent = self.sent.wrapping_add(1);
        let flagged = kind | if self.release_pool { RELEASE_POOLED } else { 0 } | if self.tp_tile { crate::flash_tp_switch::TILE } else { 0 };
        let head = request_head(flagged, rows.len() / W, meta, self.sent);
        self.bytes += (rows.len() * 4) as u64;
        let queue = self.queue.clone().ok_or("pipeline queue missing")?;
        // At most QUEUED tiles wait for the socket; past that the front waits while the back
        // runs a tile. Meanwhile it reads the queued tiles' headers at once as they come
        // (heartbeats or acks: an unread KEEP ack holds the back) and fails once the back
        // stays silent for `limit`.
        let (mut item, mut heard) = ((head, rows), Instant::now());
        loop {
            match queue.try_send(item) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(back)) => {
                    item = back;
                    if self.pending > 0 && self.header_ready()? {
                        self.queued_frame(&mut heard)?;
                    } else {
                        std::thread::sleep(Duration::from_micros(200));
                    }
                    if heard.elapsed() > self.limit {
                        return Err(format!("pipeline back silent for {} ms while tiles wait", self.limit.as_millis()));
                    }
                }
                Err(mpsc::TrySendError::Disconnected(_)) => return Err("pipeline sender thread stopped".into()),
            }
        }
        self.pending += 1;
        self.posted += 1;
        self.release_pool = false;
        Ok(())
    }
    /// A middle runs its layers first, then forwards in the same exchange sequence.
    /// Only the last stage keeps final prompt rows; returning rows are copied as bytes.
    fn forward(&mut self, kind: u32, meta: &[[u32; 4]], rows: &mut [f32], out: &mut Vec<f32>) -> Result<()> {
        if kind == KIND_ROWS_FINAL {
            let mut buffer = self.spare(rows.len());
            f32_bytes_mut(&mut buffer).copy_from_slice(f32_bytes(rows));
            self.enqueue(KIND_ROWS_FINAL, meta, buffer)?;
            self.finish_final(rows)?;
            if let Some(kept) = self.relay_kept.as_mut() && !kept.is_empty() {
                out.append(kept);
                out.extend_from_slice(rows);
            }
            return Ok(());
        }
        if matches!(kind, KIND_ROWS_ASYNC | KIND_ROWS_KEEP) {
            let n = rows.len() / W;
            if kind == KIND_ROWS_KEEP {
                if self.owed + n > KEEP_ROWS { return Err("pipeline kept rows past the limit".into()); }
                self.owed += n;
            }
            let mut buffer = self.spare(rows.len());
            f32_bytes_mut(&mut buffer).copy_from_slice(f32_bytes(rows));
            self.enqueue(kind, meta, buffer)?;
            self.acks.push_back((kind == KIND_ROWS_KEEP && self.inline_kept).then_some(n));
            Ok(())
        } else {
            self.exchange(kind, meta, rows)?;
            if let Some(kept) = self.relay_kept.as_mut() && !kept.is_empty() {
                out.append(kept);
                out.extend_from_slice(rows);
            }
            Ok(())
        }
    }
    pub(super) fn spare(&mut self, len: usize) -> Vec<f32> {
        let mut v = self.free.as_ref().and_then(|f| f.try_recv().ok()).unwrap_or_default();
        v.resize(len, 0.0);
        v
    }
    pub(super) fn exchange(&mut self, kind: u32, meta: &[[u32; 4]], residual: &mut [f32]) -> Result<()> {
        if residual.len() % W != 0 || meta.is_empty() || meta.len() > POOL_SIZE {
            return Err("invalid pipeline exchange shape".into());
        }
        let started = Instant::now();
        self.drain()?;
        self.sent = self.sent.wrapping_add(1);
        let flagged = kind | if self.release_pool { RELEASE_POOLED } else { 0 } | if self.tp_tile { crate::flash_tp_switch::TILE } else { 0 };
        let head = request_head(flagged, residual.len() / W, meta, self.sent);
        let bytes = std::mem::size_of_val(residual);
        // Both directions must fit, including every kept row returned before current rows.
        let owed = self.owed;
        if let Some(mut fast) = self.fast.take_if(|f| wire_fits(f.capacity(), head.len(), bytes, owed)) {
            let result = self.wire_exchange(&mut *fast, &head, residual);
            self.fast = Some(fast);
            result?;
            self.release_pool = false;
            self.carried += 1;
            return Ok(self.account(started, (head.len() + bytes + wire_reply_size(owed, bytes).unwrap()) as u64, residual.len() / W));
        }
        let sent = (|| -> std::io::Result<()> {
            self.stream.write_all(&head)?;
            self.stream.write_all(f32_bytes(residual))
        })();
        sent.map_err(|e| format!("pipeline send: {e}"))?;
        let len = self.take_tcp_reply(residual)?;
        self.release_pool = false;
        self.account(started, 2 * len as u64, residual.len() / W);
        Ok(())
    }
    /// One exchange on the RDMA wire (after `drain`). Its clock starts at the send; while
    /// the back runs it, its heartbeats come on the TCP link, which is read (without
    /// blocking) every 50 ms of waiting, so a decode step never touches it. No sign of life
    /// within `limit`, a closed link or a TCP reply ends the exchange in an error.
    fn wire_exchange(&mut self, fast: &mut dyn PpWire, head: &[u8], residual: &mut [f32]) -> Result<()> {
        fast.send(&[head, f32_bytes(residual)])?;
        self.wire_collect(fast, residual)
    }
    /// Wait for the wire reply of `self.sent`. Heartbeats of that request stay on TCP.
    fn wire_collect(&mut self, fast: &mut dyn PpWire, residual: &mut [f32]) -> Result<()> {
        let (sent, mut heard, mut look) = (Instant::now(), Instant::now(), Duration::from_millis(50));
        while !fast.try_recv(&mut self.message)? {
            if sent.elapsed() > look {
                look = sent.elapsed() + Duration::from_millis(50);
                self.wire_beats(&mut heard)?;
            }
            if heard.elapsed() > self.limit {
                return Err(format!("pipeline back reply timed out after {} ms without a heartbeat", self.limit.as_millis()));
            }
            std::hint::spin_loop();
        }
        let message = std::mem::take(&mut self.message);
        let result = self.wire_reply(&message, residual);
        self.message = message;
        result
    }
    fn wire_reply(&mut self, message: &[u8], residual: &mut [f32]) -> Result<()> {
        let (status, len) = self.reply_head(message, self.sent)?;
        if status != 0 {
            return Err(format!("pipeline back failed: {}", peer_text(&message[REPLY_HEAD..(REPLY_HEAD + len).min(message.len())])));
        }
        let bytes = std::mem::size_of_val(residual);
        if Some(message.len()) != wire_reply_size(self.owed, bytes) || len != message.len() - REPLY_HEAD {
            return Err("pipeline response size mismatch".into());
        }
        let owed_bytes = len - bytes;
        let (kept, current) = message[REPLY_HEAD..].split_at(owed_bytes);
        if self.owed > 0 {
            if let Some(relay) = self.relay_kept.as_mut() {
                let mut rows = vec![0f32; owed_bytes / 4];
                f32_bytes_mut(&mut rows).copy_from_slice(kept);
                *relay = rows;
            } else {
                let expected = self.kept.iter().filter(|t| t.data.is_none())
                    .try_fold(0usize, |n, t| n.checked_add(t.rows));
                if expected != Some(self.owed) { return Err("pipeline kept row accounting mismatch".into()); }
                let mut at = 0;
                for t in self.kept.iter_mut().filter(|t| t.data.is_none()) {
                    let mut rows = vec![0f32; t.rows * W];
                    let end = at + rows.len() * 4;
                    f32_bytes_mut(&mut rows).copy_from_slice(&kept[at..end]);
                    t.data = Some(rows);
                    at = end;
                }
                self.kept.retain(|t| !t.parts.is_empty());
            }
            self.owed = 0;
        }
        f32_bytes_mut(residual).copy_from_slice(current);
        Ok(())
    }
    /// While a wire exchange waits: read the whole TCP headers already here (heartbeats of
    /// this exchange or late ones of earlier wire exchanges) without blocking. Only this
    /// thread reads the socket, and `drain` left no queued tile, so nothing else is owed.
    fn wire_beats(&mut self, heard: &mut Instant) -> Result<()> {
        loop {
            let mut probe = [0u8; REPLY_HEAD];
            self.stream.set_nonblocking(true).map_err(err)?;
            let seen = self.stream.peek(&mut probe);
            self.stream.set_nonblocking(false).map_err(err)?;
            match seen {
                Ok(0) => return Err("pipeline back closed".into()),
                Ok(n) if n >= REPLY_HEAD => {
                    if self.frame(self.sent, heard)?.is_some() {
                        return Err("pipeline reply on the TCP link for a wire exchange".into());
                    }
                }
                Ok(_) => return Ok(()),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => return Ok(()),
                Err(e) => return Err(format!("pipeline link: {e}")),
            }
        }
    }
    /// Kept rows of `slot` that came back, in post order: (position, tokens, rows). Stops at
    /// the first of its tiles still on the back, so the history keeps its order.
    fn take_kept(&mut self, slot: u32) -> Vec<(usize, Vec<u32>, Vec<f32>)> {
        let mut out = Vec::new();
        for t in self.kept.iter_mut() {
            if !t.parts.iter().any(|p| p.slot == slot) {
                continue;
            }
            let Some(data) = t.data.as_mut() else { break };
            let mut i = 0;
            while i < t.parts.len() {
                if t.parts[i].slot == slot {
                    let sole = t.parts.len() == 1;
                    let p = t.parts.remove(i);
                    let rows = if sole && p.start == 0 && p.tokens.len() * W == data.len() {
                        std::mem::take(data)
                    } else {
                        data[p.start * W..(p.start + p.tokens.len()) * W].to_vec()
                    };
                    out.push((p.pos, p.tokens, rows));
                } else {
                    i += 1;
                }
            }
        }
        self.kept.retain(|t| !t.parts.is_empty() || t.data.is_none());
        out
    }
    /// Forget `slot`'s kept spans (a new chat starts on it, or its chat ended).
    fn drop_kept(&mut self, slot: u32) {
        for t in self.kept.iter_mut() {
            t.parts.retain(|p| p.slot != slot);
        }
        self.kept.retain(|t| !t.parts.is_empty() || t.data.is_none());
    }
    /// True while `slot` has kept spans its chat has not observed.
    fn has_kept(&self, slot: u32) -> bool {
        self.kept.iter().any(|t| t.parts.iter().any(|p| p.slot == slot))
    }
    /// Room for `n` more kept rows: every retained tile counts (owed or back, not yet taken).
    fn keep_room(&self, n: usize) -> bool {
        self.kept.iter().map(|t| t.rows).sum::<usize>() + n <= KEEP_ROWS
    }
    fn account(&mut self, started: Instant, bytes: u64, rows: usize) {
        self.exchanges += 1;
        self.bytes += bytes;
        self.wait += started.elapsed();
        if self.exchanges.is_power_of_two() || self.exchanges % 4096 == 0 {
            eprintln!(
                "flashnext pp front exchanges={} posted={} rdma={} beats={} late_beats={} bytes={} wait_ms={:.1} last_rows={} last_ms={:.3}",
                self.exchanges,
                self.posted,
                self.carried,
                self.beats,
                self.late,
                self.bytes,
                self.wait.as_secs_f64() * 1e3,
                rows,
                started.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

/// A clean end closes in order: the front stops sending (the back, waiting for its next
/// request, reads the end), then reads and drops what the back still sent (a late heartbeat
/// of a wire exchange) until the back closes, within `limit`. Closing with unread bytes
/// would reset the link instead, and the back would report a failure. A failed link closes
/// at once.
impl Drop for Link {
    fn drop(&mut self) {
        if self.poisoned || self.stream.shutdown(std::net::Shutdown::Write).is_err() {
            return;
        }
        let (started, mut sink) = (Instant::now(), [0u8; 256]);
        while started.elapsed() < self.limit {
            match self.stream.read(&mut sink) {
                Ok(0) => return,
                Ok(_) => (),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => (),
                Err(_) => return,
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl<'a> NextModel<'a> {
    /// Record the controller's decision even if the front has no cached width.
    /// The worker may still have one; the flag rides its next ordered request.
    pub(super) fn pp_release_pooled_decode(&self) {
        if let Some(link) = self.pp.borrow_mut().as_mut() {
            link.release_pool = true;
        }
    }
    /// Front: agree admission with every stage before serving any request.
    pub fn set_pp_front(&self, stream: TcpStream, fast: Option<Box<dyn PpWire>>, stage: Stage, context: usize, active: usize, tile: usize, down: &'static AtomicBool) -> Result<()> {
        if stage.rank != 0 { return Err("pipeline front must be rank zero".into()); }
        let link = Link::connect(stream, fast, stage, context, active, tile, down, Recipe::from_env()?)?;
        self.pp_split.set(stage.end);
        *self.pp.borrow_mut() = Some(link);
        Ok(())
    }
    /// Front split when the link is up (prompt tiles and pooled steps call the back).
    pub(super) fn pp_front_split(&self) -> Option<usize> {
        self.pp.borrow().as_ref().filter(|l| !l.tp).map(|l| l.split)
    }
    /// Front: whether verify rows may cross the pipeline.
    pub(super) fn pp_mtp(&self) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.mtp)
    }
    /// Front: whether non-final prompt tiles are queued.
    pub(super) fn pp_async(&self) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.queue.is_some())
    }
    /// The peer is the last rank, so each KEEP ack already carries that tile's rows.
    pub(super) fn pp_inline_kept(&self) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.inline_kept)
    }
    /// Read one queued ack. `false` once only the final tile's reply is left.
    pub(super) fn pp_pull_kept_ack(&self) -> Result<bool> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        if link.pending <= 1 {
            return Ok(false);
        }
        let mut heard = Instant::now();
        let result = link.queued_frame(&mut heard);
        result.map(|_| true).map_err(|e| link.fail(e))
    }
    /// Front: copy `count` residual rows out and queue them; the back
    /// runs its layers while this rank runs the next tile. No rows come back now; with
    /// `keep` they come back with the next synchronous reply.
    pub(super) fn pp_post(&self, b: &Backend<'a>, meta: &[[u32; 4]], vector: &DeviceVector<'a>, count: usize, keep: Option<Vec<KeepPart>>) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut rows = link.spare(count * W);
        // The copy waits for this tile's layers; earlier tiles' acks are read meanwhile.
        let result = link.reading_acks(|| b.read_vector_range_into(vector, 0, &mut rows)).and_then(|()| link.post(meta, rows, keep));
        result.map_err(|e| link.fail(e))
    }
    /// Front: copy the final prompt tile and hand it to the sender thread. Rows come back
    /// from `pp_collect_final`, after the acks of the tiles posted before it.
    pub(super) fn pp_post_final(&self, b: &Backend<'a>, meta: &[[u32; 4]], vector: &DeviceVector<'a>, count: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut rows = link.spare(count * W);
        let result = link.reading_acks(|| b.read_vector_range_into(vector, 0, &mut rows)).and_then(|()| link.post_final(meta, rows));
        result.map_err(|e| link.fail(e))
    }
    /// Front: take the final tile's rows back into `vector` (see `Link::finish_final`).
    pub(super) fn pp_collect_final(&self, b: &Backend<'a>, vector: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut buffer = std::mem::take(&mut link.buffer);
        buffer.resize(count * W, 0.0);
        let result = link.finish_final(&mut buffer)
            .and_then(|()| b.write_vector_range(vector, 0, &buffer))
            .and_then(|()| b.synchronize());
        link.buffer = buffer;
        result.map_err(|e| link.fail(e))
    }
    /// Front: a non-link error after `pp_post_final` succeeded leaves the final reply
    /// outstanding on the link. Poison it (as link errors do) so no later prompt reuses
    /// the link with an orphaned reply in its accounting.
    pub(super) fn pp_fail_after_final(&self, e: String) -> String {
        match self.pp.borrow_mut().as_mut() {
            Some(link) => link.fail(e),
            None => e,
        }
    }
    /// Final prompt tile when asynchronous: the sender writes it while earlier acks are read.
    /// Verify rows and a synchronous pipeline stay on one exchange.
    pub(super) fn pp_return_rows(&self, b: &Backend<'a>, output: bool, verify: bool, meta: &[[u32; 4]], vector: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        if output && !verify && self.pp_async() {
            self.pp_post_final(b, meta, vector, count)?;
            self.pp_collect_final(b, vector, count)
        } else {
            let kind = if verify { KIND_VERIFY } else { KIND_ROWS };
            self.pp_exchange(b, kind, meta, vector, count)
        }
    }
    /// Front: the kept rows of `slot` that came back (see Link::take_kept).
    pub(super) fn pp_take_kept(&self, slot: u32) -> Vec<(usize, Vec<u32>, Vec<f32>)> {
        self.pp.borrow_mut().as_mut().map_or_else(Vec::new, |l| l.take_kept(slot))
    }
    /// Front: forget `slot`'s kept spans (a new chat starts on it, or its
    /// chat ended). Never panics: a link borrowed elsewhere keeps the spans until slot reuse.
    pub(super) fn pp_drop_kept(&self, slot: u32) {
        if let Ok(mut guard) = self.pp.try_borrow_mut() && let Some(l) = guard.as_mut() {
            l.drop_kept(slot);
        }
    }
    /// Front: true while `slot` has kept rows its chat has not observed.
    pub(super) fn pp_has_kept(&self, slot: u32) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.has_kept(slot))
    }
    /// Front: room to keep `n` more rows (else the tile runs synchronously).
    pub(super) fn pp_keep_room(&self, n: usize) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.keep_room(n))
    }
    /// Front: true once the link failed (the server stops; nothing can be served). `down` is
    /// the server's shared stop flag, which a signal also sets, so only `poisoned` (set by
    /// Link::fail) marks a failure; a SIGTERM shutdown stays a clean exit.
    pub fn pp_failed(&self) -> bool {
        self.pp.borrow().as_ref().is_some_and(|l| l.poisoned)
    }
    /// Front: true once set_pp_front configured the link (this rank runs layers [0, split)).
    pub fn pp_front(&self) -> bool {
        self.pp.borrow().is_some()
    }
    /// Front: ship `count` residual rows of `vector` to the back and take its rows back.
    pub(super) fn pp_exchange(&self, b: &Backend<'a>, kind: u32, meta: &[[u32; 4]], vector: &mut DeviceVector<'a>, count: usize) -> Result<()> {
        let mut guard = self.pp.borrow_mut();
        let link = guard.as_mut().filter(|l| !l.poisoned).ok_or("pipeline link failed earlier; the front is stopping")?;
        let mut buffer = std::mem::take(&mut link.buffer);
        buffer.resize(count * W, 0.0);
        let result = b
            .read_vector_range_into(vector, 0, &mut buffer)
            .and_then(|()| link.exchange(kind, meta, &mut buffer))
            .and_then(|()| b.write_vector_range(vector, 0, &buffer))
            .and_then(|()| b.synchronize());
        link.buffer = buffer;
        result.map_err(|e| link.fail(e))
    }
    /// Worker: mirror the front's sessions and execute just this stage's layer range.
    pub fn serve_pp_back(&self, b: &Backend<'a>, stream: TcpStream, fast: Option<Box<dyn PpWire>>, next: Option<(TcpStream, Option<Box<dyn PpWire>>)>, stage: Stage, context: usize, active: usize, tile: usize, stop: &'static AtomicBool) -> Result<()> {
        self.pp_split.set(stage.first);
        self.pp_end.set(stage.end);
        let mut sessions = BTreeMap::new();
        serve_stage(stream, fast, next, stage, context, active, tile, stop, Recipe::from_env()?,
            |n| self.set_session_limit(n),
            || self.release_pooled_decode(b),
            |kind, count, meta, residual| { self.tp_mail_next(kind);
                self.pp_back_step(b, &mut sessions, kind & !crate::flash_tp_switch::MAIL & !crate::flash_tp_switch::TILE & !crate::flash_tp_switch::HEADS & !crate::flash_tp_switch::HEAD, count, meta, tile, context, residual) })
    }
    #[allow(clippy::too_many_arguments)]
    fn pp_back_step<'w>(
        &'w self,
        b: &Backend<'a>,
        sessions: &mut BTreeMap<u32, NextSession<'a, 'w>>,
        kind: u32,
        count: usize,
        meta: &[[u32; 4]],
        tile: usize,
        context: usize,
        residual: &mut [f32],
    ) -> Result<()> {
        for &[slot, capacity, pos, len] in meta {
            let (capacity, pos) = (capacity as usize, pos as usize);
            if len == 0 || slot == 0 {
                return Err("pipeline row without a slot".into());
            }
            if capacity == 0 || capacity > context {
                return Err(format!("pipeline session capacity {capacity} outside 1..={context}"));
            }
            if sessions.get(&slot).is_none_or(|s| s.capacity != capacity) {
                sessions.remove(&slot);
                sessions.insert(slot, self.session_with_chunk(b, capacity, tile)?);
            }
            let s = sessions.get_mut(&slot).unwrap();
            if pos == 0 && (s.pos != 0 || !s.valid) {
                s.reset(b)?;
            }
            // After a verify, the chat's next request carries the position the
            // front committed; the back commits the same accepted prefix first.
            if s.retained > 0 && s.valid {
                s.commit_back(b, pos)?;
            }
            if s.pos != pos || !s.valid {
                return Err(format!("pipeline session slot {slot}: back position {} valid {} vs front {pos}", s.pos, s.valid));
            }
        }
        let mut by_slot: Vec<(u32, &mut NextSession<'a, 'w>)> = sessions.iter_mut().map(|(k, v)| (*k, v)).collect();
        let mut ordered: Vec<&mut NextSession<'a, 'w>> = Vec::with_capacity(meta.len());
        for m in meta {
            let i = by_slot.iter().position(|(k, _)| *k == m[0]).ok_or("pipeline cohort repeats a slot")?;
            ordered.push(by_slot.swap_remove(i).1);
        }
        let lens: Vec<usize> = meta.iter().map(|m| m[3] as usize).collect();
        match kind {
            KIND_ROWS | KIND_ROWS_ASYNC | KIND_ROWS_KEEP | KIND_ROWS_FINAL | KIND_VERIFY | crate::flash_tp_switch::KIND_VERIFY => {
                if lens.iter().sum::<usize>() != count {
                    return Err("pipeline rows do not match the residual".into());
                }
                NextSession::pp_back_rows(b, &mut ordered, &lens, residual, matches!(kind, KIND_VERIFY | crate::flash_tp_switch::KIND_VERIFY))
            }
            KIND_POOLED => {
                if lens.iter().any(|&n| n != 1) {
                    return Err("pipeline pooled step needs one row per chat".into());
                }
                NextSession::pp_back_pooled(b, &mut ordered, count, residual)
            }
            crate::flash_tp_switch::KIND_HANDOFF => self.tp_handoff(b, &mut ordered, &lens), _ => Err("pipeline request kind unknown".into()),
        }
    }
}

/// One TCP write from the back, in order: a heartbeat, an empty ack, or a row payload.
/// `Rows` is copied onto the socket by `pp-replier`, off the step thread.
enum ReplySend {
    Bytes(Vec<u8>),
    Rows(Vec<u8>, Vec<f32>),
}

/// Production worker loop, with GPU execution supplied separately for host protocol tests.
#[allow(clippy::too_many_arguments)]
fn serve_stage(
    mut stream: TcpStream, mut fast: Option<Box<dyn PpWire>>,
    next: Option<(TcpStream, Option<Box<dyn PpWire>>)>, stage: Stage,
    context: usize, active: usize, tile: usize, stop: &'static AtomicBool, recipe: Recipe,
    mut admit: impl FnMut(usize) -> Result<()>,
    mut release_pool: impl FnMut() -> Result<()>,
    mut step: impl FnMut(u32, usize, &[[u32; 4]], &mut [f32]) -> Result<()>,
) -> Result<()> {
    if stage.rank == 0 || next.is_some() != stage.successor().is_some() {
        return Err("pipeline worker links do not match its stage".into());
    }
    let limit = recipe.limit;
    stream.set_nonblocking(false).map_err(err)?;
    stream.set_nodelay(true).map_err(err)?;
    stream.set_read_timeout(Some(Duration::from_millis(100))).map_err(err)?;
    stream.set_write_timeout(Some(limit)).map_err(err)?;
    let mut hello = [0u8; 40];
    if !read_message(&mut stream, &mut hello, Some(stop), limit, true)? {
        return Err("pipeline front closed before hello".into());
    }
    let (split, f_context, f_active, f_tile) =
        (word(&hello, 1) as usize, word(&hello, 2) as usize, word(&hello, 3) as usize, word(&hello, 4) as usize);
    let refusal = if word(&hello, 0) != MAGIC_HELLO {
        Some("hello magic".to_string())
    } else if split != stage.first || word(&hello, 7) as usize != stage.ranks || word(&hello, 8) as usize + 1 != stage.rank
        || word(&hello, 9) != (u32::from(recipe.mtp) | u32::from(recipe.tp) << 1) || f_active == 0 || f_context != context || f_tile != tile || f_active > active || word(&hello, 5) != u32::from(fast.is_some())
        || word(&hello, 6) != limit.as_millis() as u32
    {
        Some(format!("shape front split={split} context={f_context} active={f_active} tile={f_tile} rdma={} timeout_ms={} back context={context} active={active} tile={tile} rdma={} timeout_ms={}", word(&hello, 5), word(&hello, 6), fast.is_some(), limit.as_millis()))
    } else {
        admit(f_active).err()
    };
    let mut refusal = refusal;
    let mut downstream = if refusal.is_none() {
        match next {
            Some((stream, fast)) => match Link::connect(stream, fast, stage, context, f_active, tile, stop, recipe) {
                Ok(mut link) => {
                    link.relay_kept = Some(Vec::new());
                    Some(link)
                }
                Err(error) => { refusal = Some(error); None }
            },
            None => None,
        }
    } else { None };
    let mut ack = Vec::new();
    put(&mut ack, MAGIC_HELLO);
    put(&mut ack, u32::from(refusal.is_some()));
    stream.write_all(&ack).map_err(err)?;
    if let Some(why) = refusal {
        return Err(format!("pipeline back refused: {why}"));
    }
    eprintln!("flashnext pp back ready split={split} context={context} active={f_active} tile={tile}");
    let prefetch = Prefetch::start(&stream, tile, stop, limit)?;
    let mut residual: Vec<f32> = Vec::new();
    let (mut served, mut busy, mut carried) = (0u64, Duration::ZERO, 0u64);
    let mut message: Vec<u8> = Vec::new();
    // KIND_ROWS_KEEP rows held for the front (they precede the next synchronous reply).
    let mut kept: Vec<f32> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    let mut last = Instant::now();
    // Exchange ids arrive 1, 2, 3, ... (TCP and wire share one sequence); anything else
    // is a broken link, and the back stops rather than answer the wrong request.
    let mut expect = 1u32;
    // While a request runs (TCP or wire), the heartbeat thread sends a STATUS_WORKING
    // header with its id on the TCP link every beat_period, for as long as it runs (no
    // cap: see LINK_TIMEOUT). It writes only under the lock and only while a request
    // is marked, and the mark is cleared before the reply is written, so a heartbeat never
    // splits a message and every heartbeat of a request is on the TCP link before its
    // reply (a wire reply can still overtake it: the front ignores such late heartbeats).
    // The Quit guard ends the thread when this function returns or unwinds.
    let working: Arc<Mutex<Option<(u32, Instant)>>> = Arc::new(Mutex::new(None));
    let quit = Arc::new(AtomicBool::new(false));
    let heartbeat_socket = stream.try_clone().map_err(err)?;
    // Drop order: the loop sender, then the heartbeat (its sender clone), then this
    // guard, which drops the last sender and joins pp-replier.
    // On the way out: stop the heartbeat (drops its sender), drop our sender, let
    // pp-replier finish the queued frames, then shut the socket down.
    struct ReplyShutdown {
        quit: Arc<AtomicBool>,
        heartbeat: Option<std::thread::JoinHandle<()>>,
        tx: Option<mpsc::SyncSender<ReplySend>>,
        join: Option<std::thread::JoinHandle<()>>,
        socket: TcpStream,
    }
    impl Drop for ReplyShutdown {
        fn drop(&mut self) {
            self.quit.store(true, Ordering::Relaxed);
            if let Some(worker) = self.heartbeat.take() { let _ = worker.join(); }
            self.tx.take();
            if let Some(worker) = self.join.take() { let _ = worker.join(); }
            let _ = self.socket.shutdown(std::net::Shutdown::Both);
        }
    }
    let (reply_tx, reply_rx) = mpsc::sync_channel::<ReplySend>(2);
    let mut reply_sock = stream.try_clone().map_err(err)?;
    let reply_quit = quit.clone();
    let reply_thread = std::thread::Builder::new().name("pp-replier".into()).spawn(move || {
        for item in reply_rx {
            // Same retry-until-stopped send as pp-sender: a front busy computing may
            // not read for a while, and one write timeout must not end the writer.
            let written = match item {
                ReplySend::Bytes(buf) => send_all(&mut reply_sock, &buf, &reply_quit),
                ReplySend::Rows(head, rows) => send_all(&mut reply_sock, &head, &reply_quit)
                    .and_then(|()| send_all(&mut reply_sock, f32_bytes(&rows), &reply_quit)),
            };
            if written.is_err() { return; }
        }
    }).map_err(err)?;
    let heartbeat = {
        let (working, quit, period, reply_tx) = (working.clone(), quit.clone(), beat_period(limit), reply_tx.clone());
        std::thread::Builder::new()
            .name("pp-heartbeat".into())
            .spawn(move || {
                while !quit.load(Ordering::Relaxed) {
                    std::thread::sleep(period / 5);
                    let mut mark = working.lock().unwrap_or_else(|p| p.into_inner());
                    if let Some((id, last)) = mark.as_mut() {
                        if last.elapsed() >= period {
                            // Full means a row payload is queued; the front is reading that frame.
                            let _ = reply_tx.try_send(ReplySend::Bytes(working_head(*id)));
                            *last = Instant::now();
                        }
                    }
                }
            })
            .map_err(err)?
    };
    let _shutdown = ReplyShutdown {
        quit,
        heartbeat: Some(heartbeat),
        tx: Some(reply_tx.clone()),
        join: Some(reply_thread),
        socket: heartbeat_socket,
    };
    let reply_tx = reply_tx;
    loop {
        // The reader alone owns TCP reads; changing nonblocking on any clone
        // would change its socket too. Poll the fast wire before completed TCP
        // requests. A valid wire request follows all preceding async acks.
        let mut incoming = None;
        let on_wire = loop {
            if stop.load(Ordering::Relaxed) {
                return Err("pipeline back stopped by signal".into());
            }
            if let Some(f) = fast.as_mut() {
                if f.try_recv(&mut message)? { break true; }
            }
            match prefetch.poll(fast.is_none())? {
                Some(Incoming::Request(request)) => { incoming = Some(request); break false; }
                Some(Incoming::Closed) => {
                    eprintln!("flashnext pp back: front closed after {served} requests ({carried} by rdma) busy_ms={:.1}", busy.as_secs_f64() * 1e3);
                    return Ok(());
                }
                None => (),
            }
            if stop.load(Ordering::Relaxed) {
                return Err("pipeline back stopped by signal".into());
            }
            if fast.is_some() {
                if last.elapsed() > Duration::from_millis(50) {
                    std::thread::sleep(Duration::from_micros(50));
                } else {
                    std::hint::spin_loop();
                }
            }
        };
        let (kind, count, meta) = if on_wire {
            let m = &message;
            if m.len() < REQUEST_HEAD {
                return Err("pipeline request header invalid".into());
            }
            let (magic, kind, count, n, id) = (word(m, 0), word(m, 1), word(m, 2) as usize, word(m, 3) as usize, word(m, 4));
            if magic != MAGIC_REQUEST || !(1..=POOL_SIZE).contains(&n) || count == 0 || count > tile.max(POOL_SIZE)
                || m.len() != REQUEST_HEAD + n * 16 + count * W * 4
            {
                return Err("pipeline request header invalid".into());
            }
            if id != expect {
                return Err(format!("pipeline request out of order: id {id}, expected {expect}"));
            }
            let meta: Vec<[u32; 4]> = (0..n).map(|i| std::array::from_fn(|j| word(m, 5 + i * 4 + j))).collect();
            residual.resize(count * W, 0.0);
            f32_bytes_mut(&mut residual).copy_from_slice(&m[REQUEST_HEAD + n * 16..]);
            carried += 1;
            (kind, count, meta)
        } else {
            let mut request = incoming.expect("TCP poll returned one request");
            if request.id != expect {
                return Err(format!("pipeline request out of order: id {}, expected {expect}", request.id));
            }
            std::mem::swap(&mut residual, &mut request.rows);
            prefetch.recycle(request.rows)?;
            (request.kind, request.count, request.meta)
        };
        let release = kind & RELEASE_POOLED != 0; let tp_bits = kind & (crate::flash_tp_switch::MAIL | crate::flash_tp_switch::TILE | crate::flash_tp_switch::HEADS | crate::flash_tp_switch::HEAD);
        let kind = kind & !RELEASE_POOLED & !tp_bits;
        if !matches!(kind, KIND_ROWS | KIND_POOLED | KIND_ROWS_ASYNC | KIND_ROWS_KEEP | KIND_ROWS_FINAL | KIND_VERIFY | crate::flash_tp_switch::KIND_VERIFY | crate::flash_tp_switch::KIND_HANDOFF)
            || matches!(kind, KIND_VERIFY | KIND_ROWS_KEEP | crate::flash_tp_switch::KIND_VERIFY) && !recipe.mtp || matches!(kind, crate::flash_tp_switch::KIND_VERIFY | crate::flash_tp_switch::KIND_HANDOFF) && !recipe.tp
            || matches!(kind, KIND_ROWS_ASYNC | KIND_ROWS_KEEP | KIND_ROWS_FINAL | crate::flash_tp_switch::KIND_VERIFY | crate::flash_tp_switch::KIND_HANDOFF) && (!recipe.asynchronous || on_wire)
        {
            return Err("pipeline request kind or carrier outside the agreed recipe".into());
        }
        expect = expect.wrapping_add(1);
        // Rows a successful final reply will carry ahead of this tile. Captured before
        // `forward` / the kept-row move, so an error uses the same carrier as success.
        // A middle counts rows already relayed into its buffer as well as rows still
        // owed downstream: acks consumed early moved rows from the second to the first.
        let prefix_rows = downstream.as_ref()
            .map(|link| link.owed + link.relay_kept.as_ref().map_or(0, |rows| rows.len() / W))
            .unwrap_or(kept.len() / W);
        let tile_bytes = count * W * 4;
        let started = Instant::now();
        *working.lock().unwrap_or_else(|p| p.into_inner()) = Some((expect.wrapping_sub(1), started));
        // Synchronize and drop pooled graphs/scratch before any session rollback,
        // verify allocation or execution. An error must prevent the step as well.
        let mut result = if release { release_pool() } else { Ok(()) }
            .and_then(|()| step(kind | tp_bits, count, &meta, &mut residual));
        if result.is_ok() && let Some(link) = downstream.as_mut() {
            link.release_pool |= release;
            result = link.forward(kind, &meta, &mut residual, &mut out);
            if result.is_err() { link.poisoned = true; }
        }
        *working.lock().unwrap_or_else(|p| p.into_inner()) = None;
        // Last rank: this tile's rows ride the ack. They are not held for the final reply.
        let inline_keep = downstream.is_none() && result.is_ok() && kind == KIND_ROWS_KEEP;
        let queued_empty = result.is_ok() && matches!(kind, KIND_ROWS_ASYNC | KIND_ROWS_KEEP | crate::flash_tp_switch::KIND_VERIFY | crate::flash_tp_switch::KIND_HANDOFF) && !inline_keep;
        if downstream.is_none() && result.is_ok() && !queued_empty && !inline_keep && !kept.is_empty() {
            let reply_bytes = kept.len().checked_add(residual.len()).and_then(|n| n.checked_mul(4))
                .and_then(|n| n.checked_add(REPLY_HEAD));
            if on_wire && !reply_bytes.is_some_and(|n| fast.as_ref().is_some_and(|f| n <= f.capacity())) {
                result = Err("pipeline kept reply exceeds wire capacity".into());
            } else {
                out.clear();
                out.append(&mut kept);
                out.extend_from_slice(&residual);
            }
        }
        busy += started.elapsed();
        served += 1;
        if let Err(e) = &result {
            engine_model::events::note_line("pp_back_request_failed", &format!("flashnext pp back request failed: {e}"));
        }
        let err_bytes = result.as_ref().err().map(|e| e.as_bytes().to_vec());
        let body = if err_bytes.is_some() {
            ReplySend::Bytes(err_bytes.clone().unwrap_or_default())
        } else if queued_empty {
            ReplySend::Bytes(Vec::new())
        } else if inline_keep {
            ReplySend::Rows(Vec::new(), std::mem::take(&mut residual))
        } else if !out.is_empty() {
            ReplySend::Rows(Vec::new(), std::mem::take(&mut out))
        } else {
            ReplySend::Rows(Vec::new(), std::mem::take(&mut residual))
        };
        let mut header = Vec::new();
        put(&mut header, MAGIC_RESPONSE);
        put(&mut header, u32::from(err_bytes.is_some()));
        let nbytes: u32 = match &body {
            ReplySend::Bytes(buf) => buf.len() as u32,
            ReplySend::Rows(_, rows) => (rows.len() * 4) as u32,
        };
        put(&mut header, nbytes);
        put(&mut header, expect.wrapping_sub(1));
        // A final tile arrives on TCP so it stays behind the queued tiles. Its reply uses
        // the wire when the rows fit, which is the same test the front's finish_final applies.
        // KEEP rows stay on TCP: the front reads those acks before the final wire reply.
        let final_on_wire = kind == KIND_ROWS_FINAL
            && wire_reply_size(prefix_rows, tile_bytes).is_some_and(|n| fast.as_ref().is_some_and(|f| n <= f.capacity()));
        if on_wire || final_on_wire {
            let payload_owned: Vec<u8> = match &body {
                ReplySend::Bytes(buf) => buf.clone(),
                ReplySend::Rows(_, rows) => f32_bytes(rows).to_vec(),
            };
            fast.as_mut().ok_or("pipeline wire missing")?.send(&[&header, &payload_owned])?;
        } else {
            let queued_send = match body {
                ReplySend::Bytes(buf) => {
                    header.extend_from_slice(&buf);
                    reply_tx.send(ReplySend::Bytes(header))
                }
                ReplySend::Rows(_, rows) => reply_tx.send(ReplySend::Rows(header, rows)),
            };
            queued_send.map_err(|_| "pipeline reply writer stopped")?;
        }
        result?;
        out.clear();
        last = Instant::now();
        if served.is_power_of_two() || served % 4096 == 0 {
            eprintln!("flashnext pp back served={served} busy_ms={:.1} last_kind={kind} last_rows={count}", busy.as_secs_f64() * 1e3);
        }
    }
}

#[cfg(test)]
#[path = "next_pp_tests.rs"]
mod tests;
