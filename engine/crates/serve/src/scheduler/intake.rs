//! Connection intake: which accepted sockets hold a place in the connection table.
//!
//! The key check runs on a complete request head. A peer with no key must not be able to fill
//! the table, or to stop the accept loop, with sockets that never pass that check. So a reader
//! holds a slot only after the check. The readers before it, and the refusals that they get, are
//! a bounded pool beside the table. When the pool is full, each new connection takes the place of
//! its oldest member, so the accept loop does not wait for a silent peer. A peer that sends its
//! request at once leaves the pool in the pass that accepts it.
//!
//! Before any member is pushed out, each member reads what has already arrived in its socket.
//! A head that is complete there passes the key check at that moment, and a reader that passed
//! it is never pushed out by readers that did not. What is not promised: a head with the key
//! that is still on its way while 64 newer connections arrive. To promise that, the intake
//! would have to know the key holder before it reads a head, which a plain socket cannot say.
//!
//! The same holds when the table is full. The control reserve keeps eight places for health and
//! cancel requests. A reader takes one of them only when it holds a request or gets a reply for
//! one; until then it waits in the pool. So sockets that send nothing cannot take the reserve
//! and keep a cancel out for their read time, wave after wave.
use super::{Duration, Instant, Limits, Reading, Socket, TcpListener, http};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

/// At capacity, the readers that are kept for health and cancel requests. The accept loop stops
/// while this many of them hold a request or drain a reply. The readers of the accept batches
/// before a reader pass (two a turn: its end and the next start) can add to them in that pass;
/// each has the short read time below.
const CONTROL_SLOTS: usize = 8;
const CONTROL_READ_TIMEOUT: Duration = Duration::from_millis(100);
/// The pool of readers that have not passed the key check: two accept batches, so one full batch
/// enters without pushing out a reader of the batch before it.
const UNPROVEN_SLOTS: usize = 64;

/// A reader that has not passed the key check, so it holds no slot and no place in the control
/// reserve: it has sent no complete request head, or it gets a refusal for such a request, or
/// its head is the health request that needs no key. A reply on a wire that the dispatcher
/// handed back has no head either, but it answers a request that passed the check: it holds
/// its slot. So does a request that showed the key and then failed a later check of its head
/// (`Wire::keyed`): it has no head, but it passed the key check, and it stays until its refusal
/// has drained.
fn unproven(input: &Reading) -> bool {
    if input.wire.socket.is_admin() || input.wire.keyed { return false; }
    match &input.head {
        None => !input.responding || input.wire.headless,
        Some((_, head)) => head.path == http::HEALTH_LIVE,
    }
}

/// One log line for the readers that a full pool pushed out, at most one line each second.
fn note_evicted(count: usize) {
    static PENDING: AtomicUsize = AtomicUsize::new(0);
    static NOTED_MS: AtomicU64 = AtomicU64::new(0);
    PENDING.fetch_add(count, Relaxed);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |since| since.as_millis() as u64);
    if now.saturating_sub(NOTED_MS.load(Relaxed)) >= 1000 {
        NOTED_MS.store(now, Relaxed);
        // One swap takes the whole count, so two callers at once cannot take more than there is.
        let pending = PENDING.swap(0, Relaxed);
        if pending != 0 { engine_model::events::note("intake_evicted", format_args!("count={pending} bound={UNPROVEN_SLOTS}")); }
    }
}

/// Never evict a reader that holds a slot, or a draining response. At capacity, reserve eight
/// bounded readers for health/cancel; other complete requests drain a 503. Overflow readers have
/// a short deadline. A reader that has not passed the key check holds no slot and no place in
/// the reserve (see `unproven`): only requests that passed it can fill the table or the reserve.
pub(super) fn accept_connections(listener: &TcpListener, reading: &mut Vec<Reading>, limits: &Limits, access: &http::AccessPolicy, slots: usize, attempts: usize) -> usize {
    let controls = reading.iter().filter(|input| input.wire.control_only && !unproven(input)).count();
    let mut waiting = reading.iter().filter(|input| unproven(input)).count();
    let control_only = reading.len() - waiting >= slots;
    let (mut accepted, mut evicted, mut read) = (0, 0, false);
    for _ in 0..attempts {
        if control_only && controls >= CONTROL_SLOTS {
            break;
        }
        let socket = match listener.accept() {
            Ok((socket, _)) => socket,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                eprintln!("accept: {e}");
                break;
            }
        };
        accepted += 1;
        match Reading::new(Socket::Tcp(socket), limits) {
            Ok(mut input) => {
                if control_only {
                    input.wire.control_only = true;
                    input.deadline = input.deadline.min(Instant::now() + CONTROL_READ_TIMEOUT);
                }
                // The pool is full. Before a member leaves, each member reads what has arrived
                // (once a call): a head that is complete in its socket passes the key check now.
                if waiting >= UNPROVEN_SLOTS && !read {
                    read = true;
                    for input in reading.iter_mut().filter(|input| unproven(input)) { input.arrived(limits, access); }
                    waiting = reading.iter().filter(|input| unproven(input)).count();
                }
                // Each new reader starts in the pool: it has shown no request yet.
                if waiting < UNPROVEN_SLOTS {
                    waiting += 1;
                } else if let Some(oldest) = (0..reading.len()).filter(|&at| unproven(&reading[at])).min_by_key(|&at| reading[at].wire.received) {
                    // The pool is full. The oldest member closes with no reply: it sent no request
                    // that passed the key check, and a reply would keep its place for one more pass.
                    reading.swap_remove(oldest);
                    evicted += 1;
                }
                reading.push(input);
            }
            Err(error) => eprintln!("connection setup: {error}"),
        }
    }
    if evicted != 0 {
        note_evicted(evicted);
    }
    accepted
}
