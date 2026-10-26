//! VADE_W2_TP=1 mailbox proxy: one thread per rank, owner of the mailbox's host endpoint,
//! moves every message the GPU publishes in its send slots over the tensor split's RDMA wire
//! and publishes every arriving message in the receive slots its kernels wait on. Messages
//! keep their order on both sides (sequence 1, 2, ... per direction). The thread does not
//! spin: while a pass runs it polls with yield_now (the core goes to any runnable thread) for
//! up to ACTIVE after the last message, then sleeps with a doubling backoff up to IDLE_MAX.
//! After any error it marks the mailbox failed (every waiting take stops with an error) and
//! tells the peer, so no rank ever computes on missing rows.
use spark_engine::flash::next_batched::PpWire;
use spark_gpu_probe::driver::TpMailboxHost;
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
}
