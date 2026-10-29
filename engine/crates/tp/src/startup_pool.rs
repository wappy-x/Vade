//! Agree the TP session count before the serving handshake, using its control stream.
use std::{io::{ErrorKind, Read, Write}, net::TcpStream, sync::atomic::{AtomicBool, Ordering}, time::{Duration, Instant}};

/// Allow the same model-loading time on each rank, before the serving protocol starts.
pub const LOAD_LIMIT: Duration = Duration::from_secs(600);
const POLL: Duration = Duration::from_millis(100);

/// Exchange local limits and return (peer limit, agreed limit). Both ranks send first;
/// the fixed frame fits in the TCP buffer. Restore timeouts before the next protocol.
/// `deadline` starts before model loading. Socket timeouts only pace stop/deadline checks.
pub fn agree(link: &mut TcpStream, local: usize, deadline: Instant, stop: &AtomicBool) -> Result<(usize, usize), String> {
    let peer = exchange(link, local, deadline, stop)?;
    if local == 0 || peer == 0 { return Err("session pool agreement requires a positive count on each rank".into()); }
    Ok((peer, local.min(peer)))
}
/// The cache room: the smaller of the two ranks' bytes. Zero is valid (no room).
pub fn agree_room(link: &mut TcpStream, local: usize, deadline: Instant, stop: &AtomicBool) -> Result<usize, String> {
    Ok(local.min(exchange(link, local, deadline, stop)?))
}
/// Send `local` and return the peer's value, the same frame for every agreement.
fn exchange(link: &mut TcpStream, local: usize, deadline: Instant, stop: &AtomicBool) -> Result<usize, String> {
    let before = (link.read_timeout().map_err(err)?, link.write_timeout().map_err(err)?);
    let agreed = (|| {
        let mut frame = [0u8; 16];
        frame[..8].copy_from_slice(b"VDPOOL01");
        frame[8..].copy_from_slice(&u64::try_from(local).map_err(|_| "session pool count exceeds u64")?.to_le_bytes());
        let mut at = 0;
        while at < frame.len() {
            link.set_write_timeout(Some(remaining(deadline, stop)?.min(POLL))).map_err(err)?;
            match link.write(&frame[at..]) {
                Ok(0) => return Err("session pool agreement: peer closed during write".into()),
                Ok(n) => at += n,
                Err(e) if retry(&e) => (),
                Err(e) => return Err(err(e)),
            }
        }
        at = 0;
        while at < frame.len() {
            link.set_read_timeout(Some(remaining(deadline, stop)?.min(POLL))).map_err(err)?;
            match link.read(&mut frame[at..]) {
                Ok(0) => return Err("session pool agreement: peer closed during read".into()),
                Ok(n) => at += n,
                Err(e) if retry(&e) => (),
                Err(e) => return Err(err(e)),
            }
        }
        remaining(deadline, stop)?;
        if &frame[..8] != b"VDPOOL01" { return Err("session pool agreement: invalid frame".into()); }
        usize::try_from(u64::from_le_bytes(frame[8..].try_into().unwrap()))
            .map_err(|_| "session pool peer count exceeds usize".into())
    })();
    let restored = link.set_read_timeout(before.0).and_then(|()| link.set_write_timeout(before.1)).map_err(err);
    agreed.and_then(|counts| restored.map(|()| counts))
}
fn remaining(deadline: Instant, stop: &AtomicBool) -> Result<Duration, String> {
    if stop.load(Ordering::Relaxed) { return Err("session pool agreement stopped by signal".into()); }
    deadline.checked_duration_since(Instant::now()).filter(|d| !d.is_zero())
        .ok_or_else(|| "session pool agreement: load deadline expired".into())
}
fn retry(e: &std::io::Error) -> bool { matches!(e.kind(), ErrorKind::Interrupted | ErrorKind::WouldBlock | ErrorKind::TimedOut) }
fn err(e: std::io::Error) -> String { format!("session pool agreement: {e}") }

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    static STOP: AtomicBool = AtomicBool::new(false);

    fn deadline() -> Instant { Instant::now() + Duration::from_secs(2) }

    fn pair() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let front = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        (front, listener.accept().unwrap().0)
    }
    #[test]
    fn fake_rank_counts_agree_before_the_next_message() {
        for (front_count, back_count) in [(5, 4), (4, 5), (5, 5)] {
            let (mut front, mut back) = pair();
            let timeout = Duration::from_secs(2);
            front.set_read_timeout(Some(timeout)).unwrap();
            let worker = std::thread::spawn(move || {
                let counts = agree(&mut back, back_count, deadline(), &STOP).unwrap();
                assert_eq!(counts, (front_count, front_count.min(back_count)));
                assert_eq!(back.read_timeout().unwrap(), None);
                assert_eq!(back.write_timeout().unwrap(), None);
                back.write_all(b"next hello").unwrap();
            });
            assert_eq!(agree(&mut front, front_count, deadline(), &STOP).unwrap(), (back_count, front_count.min(back_count)));
            assert_eq!(front.read_timeout().unwrap(), Some(timeout));
            assert_eq!(front.write_timeout().unwrap(), None);
            let mut next = [0u8; 10];
            front.read_exact(&mut next).unwrap();
            assert_eq!(&next, b"next hello");
            worker.join().unwrap();
        }
    }
    #[test]
    fn unfunded_rank_stops_both_ranks() {
        for counts in [(0, 4), (5, 0)] {
            let (mut front, mut back) = pair();
            let worker = std::thread::spawn(move || agree(&mut back, counts.1, deadline(), &STOP));
            assert!(agree(&mut front, counts.0, deadline(), &STOP).unwrap_err().contains("positive count"));
            assert!(worker.join().unwrap().unwrap_err().contains("positive count"));
            assert_eq!(front.read_timeout().unwrap(), None);
            assert_eq!(front.write_timeout().unwrap(), None);
        }
    }
    #[test]
    fn a_zero_room_on_either_rank_agrees_on_zero() {
        for (front_room, back_room, agreed) in [(0, 4096, 0), (4096, 0, 0), (0, 0, 0), (4096, 8192, 4096)] {
            let (mut front, mut back) = pair();
            let worker = std::thread::spawn(move || agree_room(&mut back, back_room, deadline(), &STOP));
            assert_eq!(agree_room(&mut front, front_room, deadline(), &STOP), Ok(agreed));
            assert_eq!(worker.join().unwrap(), Ok(agreed));
            assert_eq!(front.read_timeout().unwrap(), None);
            assert_eq!(front.write_timeout().unwrap(), None);
        }
    }
    #[test]
    fn rejects_another_protocol_and_restores_timeouts() {
        let (mut front, mut back) = pair();
        let worker = std::thread::spawn(move || {
            let mut frame = [0u8; 16];
            back.read_exact(&mut frame).unwrap();
            back.write_all(&[0; 16]).unwrap();
        });
        assert!(agree(&mut front, 5, deadline(), &STOP).unwrap_err().contains("invalid frame"));
        assert_eq!(front.read_timeout().unwrap(), None);
        assert_eq!(front.write_timeout().unwrap(), None);
        worker.join().unwrap();
    }
    #[test]
    fn front_ready_after_thirty_seconds_uses_the_load_deadline() {
        let (mut front, mut back) = pair();
        let serving = Duration::from_secs(10);
        front.set_read_timeout(Some(serving)).unwrap();
        back.set_read_timeout(Some(serving)).unwrap();
        let load = Instant::now() + Duration::from_secs(60);
        let worker = std::thread::spawn(move || {
            let counts = agree(&mut back, 4, load, &STOP).unwrap();
            assert_eq!(back.read_timeout().unwrap(), Some(serving));
            assert_eq!(back.write_timeout().unwrap(), None);
            back.write_all(b"next hello").unwrap();
            counts
        });
        // The worker has reached agreement and sent its frame before the front loads.
        let mut pending = [0u8; 1];
        assert_eq!(front.peek(&mut pending).unwrap(), pending.len());
        let start = Instant::now();
        std::thread::sleep(Duration::from_secs(30));
        assert_eq!(agree(&mut front, 5, load, &STOP).unwrap(), (4, 4));
        assert!(start.elapsed() >= Duration::from_secs(30));
        assert_eq!(worker.join().unwrap(), (5, 4));
        assert_eq!(front.read_timeout().unwrap(), Some(serving));
        assert_eq!(front.write_timeout().unwrap(), None);
        let mut hello = [0u8; 10];
        front.read_exact(&mut hello).unwrap();
        assert_eq!(&hello, b"next hello");
    }
    #[test]
    fn silent_and_partial_fronts_stop_at_the_load_deadline() {
        for prefix in [0, 1, 15] {
            let (mut front, mut back) = pair();
            let serving = Duration::from_secs(10);
            back.set_read_timeout(Some(serving)).unwrap();
            let mut frame = [0u8; 16];
            frame[..8].copy_from_slice(b"VDPOOL01");
            frame[8..].copy_from_slice(&5u64.to_le_bytes());
            front.write_all(&frame[..prefix]).unwrap();
            let start = Instant::now();
            let load = start + Duration::from_millis(250);
            let error = agree(&mut back, 4, load, &STOP).unwrap_err();
            assert!(error.contains("load deadline expired"), "{error}");
            assert!(start.elapsed() >= Duration::from_millis(250));
            assert!(start.elapsed() < Duration::from_secs(2));
            assert_eq!(back.read_timeout().unwrap(), Some(serving));
            assert_eq!(back.write_timeout().unwrap(), None);
        }
    }
    #[test]
    fn a_stopped_rank_does_not_wait_for_the_load_deadline() {
        let (mut front, mut back) = pair();
        let stop = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let worker = scope.spawn(|| agree(&mut back, 4, Instant::now() + LOAD_LIMIT, &stop));
            front.read_exact(&mut [0u8; 16]).unwrap();
            let start = Instant::now();
            stop.store(true, Ordering::Relaxed);
            assert!(worker.join().unwrap().unwrap_err().contains("stopped by signal"));
            assert!(start.elapsed() < Duration::from_secs(2));
        });
        assert_eq!(back.read_timeout().unwrap(), None);
        assert_eq!(back.write_timeout().unwrap(), None);
    }
    #[test]
    fn partial_progress_does_not_extend_the_load_deadline() {
        let (mut front, mut back) = pair();
        let start = Instant::now();
        let load = start + Duration::from_millis(250);
        let worker = std::thread::spawn(move || {
            let error = agree(&mut back, 4, load, &STOP).unwrap_err();
            assert_eq!(back.read_timeout().unwrap(), None);
            assert_eq!(back.write_timeout().unwrap(), None);
            (error, start.elapsed())
        });
        front.read_exact(&mut [0u8; 16]).unwrap();
        for byte in b"VDPOOL01" {
            if front.write_all(&[*byte]).is_err() { break; }
            std::thread::sleep(Duration::from_millis(60));
        }
        let (error, elapsed) = worker.join().unwrap();
        assert!(error.contains("load deadline expired"), "{error}");
        assert!(elapsed >= Duration::from_millis(250));
        assert!(elapsed < Duration::from_secs(2));
    }
}
