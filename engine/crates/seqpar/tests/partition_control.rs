//! Public sequence-parallel behavior that needs no GPU: prompt partitions and
//! the challenge-response rank admission, run over loopback TCP.
use spark_seqpar::{Partition, control};
use std::{io::Write, net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream}, thread, time::{Duration, Instant}};
use std::sync::{Arc, atomic::{AtomicBool, Ordering::Relaxed}};

fn assert_covers(partition: &Partition, tokens: usize, world: usize) {
    assert_eq!((partition.tokens(), partition.world()), (tokens, world));
    let offsets = partition.offsets();
    assert_eq!((offsets[0], offsets.len()), (0, world + 1));
    for rank in 0..world {
        let range = partition.range(rank);
        assert_eq!((range.start, range.end), (offsets[rank], offsets[rank + 1]));
        assert!(range.start <= range.end);
    }
}

#[test]
fn equal_and_balanced_partitions_cover_the_prompt_in_order() {
    let equal = Partition::equal(10, 3).unwrap();
    assert_eq!(equal.offsets(), [0, 3, 6, 10]);
    assert_covers(&equal, 10, 3);
    assert_eq!(Partition::equal(0, 2).unwrap().offsets(), [0, 0, 0]);
    // Attention makes later tokens dearer, so later ranks take fewer of them.
    for world in 2..=6 {
        let p = Partition::balanced(30000, world, 2.0).unwrap();
        assert_covers(&p, 30000, world);
        let lengths = (0..world).map(|rank| p.range(rank).len()).collect::<Vec<_>>();
        assert!(lengths.windows(2).all(|w| w[0] >= w[1]), "{lengths:?}");
        assert!(lengths[0] > lengths[world - 1], "{lengths:?}");
    }
    for (world, weight) in [(0, 0.0), (2, -1.0), (2, f64::NAN), (2, f64::INFINITY)] {
        assert!(Partition::balanced(100, world, weight).is_err(), "world {world} weight {weight}");
    }
}

#[test]
fn from_offsets_accepts_only_monotonic_ranges_from_zero() {
    assert_eq!(Partition::from_offsets(vec![0, 4, 4, 9]).unwrap().range(1), 4..4);
    for bad in [vec![], vec![0], vec![1, 4], vec![0, 5, 4]] {
        assert!(Partition::from_offsets(bad.clone()).is_err(), "{bad:?}");
    }
}

#[test]
fn aligned_partitions_keep_whole_chunks_and_refuse_short_prompts() {
    for (tokens, world) in [(4096, 2), (5000, 3), (1000, 4), (65537, 6)] {
        let p = Partition::balanced_aligned(tokens, world, 0.5, 128).unwrap();
        assert_covers(&p, tokens, world);
        assert!(p.offsets()[1..world].iter().all(|o| o % 128 == 0), "{:?}", p.offsets());
        assert!((0..world).all(|rank| !p.range(rank).is_empty()));
    }
    assert!(Partition::balanced_aligned(100, 1, 0.0, 128).is_ok(), "one rank takes any prompt");
    assert!(Partition::balanced_aligned(255, 2, 0.0, 128).is_err(), "shorter than world * alignment");
    assert!(Partition::balanced_aligned(1, 2, 0.0, 1).is_err(), "fewer tokens than ranks");
    assert!(Partition::balanced_aligned(1024, 2, 0.0, 0).is_err(), "zero alignment");
}

/// Prefill partitions are a pure function of (tokens, world, weight): every
/// rank computes the same split without talking to the others.
#[test]
fn prefill_partitions_are_aligned_covering_and_reproducible() {
    for tokens in [256, 1000, 4096, 6143, 6144, 20000, 32767, 40000] {
        let weight = Partition::prefill_weight(tokens);
        assert!(weight.is_finite() && weight >= 0.0, "{tokens}: {weight}");
        for world in 1..=4 {
            // Only a prompt shorter than one chunk per rank is refused.
            let Ok(p) = Partition::prefill(tokens, world, weight) else {
                assert!(tokens / world < Partition::ALIGNMENT, "{tokens}/{world}");
                continue;
            };
            assert_covers(&p, tokens, world);
            assert!(p.offsets()[1..world].iter().all(|o| o % Partition::ALIGNMENT == 0), "{tokens}/{world}: {:?}", p.offsets());
            assert!((0..world).all(|rank| !p.range(rank).is_empty()), "{tokens}/{world}: {:?}", p.offsets());
            assert_eq!(Partition::prefill(tokens, world, weight).unwrap().offsets(), p.offsets());
        }
    }
    assert_eq!(Partition::prefill(5000, 1, 1.0).unwrap().offsets(), [0, 5000]);
}

/// A connected loopback pair: (coordinator side, worker side).
fn pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    (client, listener.accept().unwrap().0)
}
fn handshake(coordinator: Option<&[u8]>, worker: Option<&[u8]>) -> (Result<(), String>, Result<(), String>) {
    let (mut a, mut b) = pair();
    let (coordinator, worker) = (coordinator.map(<[u8]>::to_vec), worker.map(<[u8]>::to_vec));
    let worker = thread::spawn(move || control::authenticate_worker(&mut b, worker.as_deref()));
    let coordinator = control::authenticate_coordinator(&mut a, coordinator.as_deref(), Instant::now() + Duration::from_secs(5));
    (coordinator, worker.join().unwrap())
}

#[test]
fn rank_admission_needs_the_same_secret_on_both_sides() {
    assert_eq!(handshake(Some(b"shared"), Some(b"shared")), (Ok(()), Ok(())));
    assert_eq!(handshake(None, None), (Ok(()), Ok(())), "loopback runs may omit the secret");
    let long = [7u8; 100];
    assert_eq!(handshake(Some(&long), Some(&long)), (Ok(()), Ok(())), "secrets longer than one HMAC block");
    let (coordinator, worker) = handshake(Some(b"shared"), Some(b"other"));
    assert!(coordinator.unwrap_err().contains("seqpar authentication failed"));
    assert!(worker.unwrap_err().contains("rejected"));
    for (a, b) in [(Some(&b"shared"[..]), None), (None, Some(&b"shared"[..]))] {
        let (coordinator, worker) = handshake(a, b);
        assert!(coordinator.unwrap_err().contains("mismatch") && worker.unwrap_err().contains("mismatch"));
    }
}

#[test]
fn two_coordinators_or_a_silent_peer_are_refused_within_the_deadline() {
    let (mut a, mut b) = pair();
    let other = thread::spawn(move || control::authenticate_coordinator(&mut b, Some(b"s"), Instant::now() + Duration::from_secs(5)));
    assert!(control::authenticate_coordinator(&mut a, Some(b"s"), Instant::now() + Duration::from_secs(5)).unwrap_err().contains("mismatch"));
    assert!(other.join().unwrap().is_err(), "a reflected role is never admitted");
    let (mut a, _silent) = pair();
    let start = Instant::now();
    assert!(control::authenticate_coordinator(&mut a, Some(b"s"), Instant::now() + Duration::from_secs(30)).is_err());
    assert!(start.elapsed() < Duration::from_secs(5), "the handshake has its own short limit: {:?}", start.elapsed());
    // A peer that trickles its 42-byte hello one byte per 50 ms (2.1 s in
    // all) cannot extend the deadline one read at a time.
    let (mut a, mut trickle) = pair();
    let feeder = thread::spawn(move || { for _ in 0..42 { if trickle.write_all(b"S").is_err() { break; } thread::sleep(Duration::from_millis(50)); } });
    let start = Instant::now();
    assert!(control::authenticate_worker(&mut a, None).is_err());
    assert!(start.elapsed() < Duration::from_millis(1500), "{:?}", start.elapsed());
    feeder.join().unwrap();
}

/// connect_peers reads SEQPAR_CONTROL_SECRET; the test peer mirrors whatever
/// the environment holds so it passes with or without the variable set.
fn environment_secret() -> Option<Vec<u8>> { control::control_secret().unwrap() }

#[test]
fn connect_peers_authenticates_the_outgoing_neighbour() {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
    let next = listener.local_addr().unwrap();
    let secret = environment_secret();
    let worker = thread::spawn(move || { let mut stream = listener.accept().unwrap().0; control::authenticate_worker(&mut stream, secret.as_deref()) });
    let (outgoing, incoming) = control::connect_peers(None, Some(next), Instant::now() + Duration::from_secs(10)).unwrap();
    worker.join().unwrap().unwrap();
    assert_eq!(outgoing.unwrap().peer_addr().unwrap(), next);
    assert!(incoming.is_none());
    assert_eq!(control::connect_peers(None, None, Instant::now() + Duration::from_secs(1)).map(|(a, b)| (a.is_none(), b.is_none())), Ok((true, true)));
}

/// connect_peers binds its listen address itself, so a test can only pick a free
/// port and release it; another process may take that port first. Each test
/// retries with a fresh port when the bind fails, at most five times.
fn with_free_port<T>(mut run: impl FnMut(SocketAddr) -> Option<T>) -> T {
    for _ in 0..5 {
        let listen = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap().local_addr().unwrap();
        if let Some(result) = run(listen) { return result; }
    }
    panic!("connect_peers could not bind a free loopback port in five tries");
}
fn bind_collision<T>(result: &Result<T, String>) -> bool { result.as_ref().is_err_and(|e| e.starts_with("listen:")) }

/// Connect to `address` until `end` or until `stop` is set; never loops past either.
fn connect_until(address: SocketAddr, end: Instant, stop: &AtomicBool) -> Result<TcpStream, String> {
    loop {
        let left = end.saturating_duration_since(Instant::now());
        if left.is_zero() || stop.load(Relaxed) { return Err(format!("no listener accepted at {address} before the deadline")); }
        match TcpStream::connect_timeout(&address, left.min(Duration::from_secs(1))) {
            Ok(stream) => return Ok(stream),
            Err(_) => thread::sleep(Duration::from_millis(10)),
        }
    }
}

/// A rejected or silent client never closes the admission window for the real one.
#[test]
fn connect_peers_admits_the_real_peer_after_rejected_clients() {
    let secret = environment_secret();
    let (incoming, real) = with_free_port(|listen| {
        let stop = Arc::new(AtomicBool::new(false));
        let (secret, flag) = (secret.clone(), stop.clone());
        let clients = thread::spawn(move || -> Result<SocketAddr, String> {
            let end = Instant::now() + Duration::from_secs(10);
            let _silent = connect_until(listen, end, &flag)?;
            let mut wrong = connect_until(listen, end, &flag)?;
            let _ = wrong.write_all(b"not a seqpar hello, but long enough to fill the 42-byte hello");
            // A worker claiming the coordinator's role is a mismatch for the listener.
            let mut reflected = connect_until(listen, end, &flag)?;
            let _ = control::authenticate_worker(&mut reflected, secret.as_deref());
            let mut real = connect_until(listen, end, &flag)?;
            control::authenticate_coordinator(&mut real, secret.as_deref(), end)?;
            real.local_addr().map_err(|e| e.to_string())
        });
        let peers = control::connect_peers(Some(listen), None, Instant::now() + Duration::from_secs(10));
        if bind_collision(&peers) { stop.store(true, Relaxed); let _ = clients.join(); return None; }
        let (outgoing, incoming) = peers.unwrap();
        assert!(outgoing.is_none());
        Some((incoming.expect("connect_peers admitted no incoming peer"), clients.join().unwrap().unwrap()))
    });
    assert_eq!(incoming.peer_addr().unwrap(), real);
}

#[test]
fn connect_peers_times_out_without_a_neighbour() {
    let (error, elapsed) = with_free_port(|listen| {
        let start = Instant::now();
        let peers = control::connect_peers(Some(listen), None, Instant::now() + Duration::from_millis(300));
        (!bind_collision(&peers)).then(|| (peers.map(|_| ()).unwrap_err(), start.elapsed()))
    });
    assert!(error.contains("timed out"), "{error}");
    assert!(elapsed < Duration::from_secs(3), "{elapsed:?}");
}
