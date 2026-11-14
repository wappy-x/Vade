//! Host transport, configured second-rail switch-on, and stripe exactness.
use engine_model::plan::layer::Topology;
use spark_gpu_probe::prompt_ops::{self, Shape};
use spark_tp::{
    runtime,
    sparse_heads::{self, Layout},
    transport::{PAIR_GATHER_FLOOR_BYTES, Ring},
};
use std::{
    io::{ErrorKind, Read, Write},
    net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering::SeqCst},
    },
    time::{Duration, Instant},
};

fn payload(slot: usize, tokens: usize, width: usize) -> Vec<f32> {
    (0..tokens * width)
        .map(|i| {
            f32::from_bits(
                [0, 0x80000000, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800000 + (slot * 1009 + i) as u32][i % 7],
            )
        })
        .collect()
}

fn free_ports(count: usize) -> Vec<SocketAddr> {
    let listeners = (0..count).map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap()).collect::<Vec<_>>();
    listeners.iter().map(|l| l.local_addr().unwrap()).collect()
}

struct Relay {
    entry: SocketAddr,
    bytes: Arc<AtomicUsize>,
    drop_after: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Relay {
    fn start(target: SocketAddr) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let entry = listener.local_addr().unwrap();
        let bytes = Arc::new(AtomicUsize::new(0));
        let drop_after = Arc::new(AtomicUsize::new(usize::MAX));
        let stop = Arc::new(AtomicBool::new(false));
        let (bytes_bg, drop_bg, stop_bg) = (Arc::clone(&bytes), Arc::clone(&drop_after), Arc::clone(&stop));
        let worker = std::thread::spawn(move || {
            let mut client = loop {
                if stop_bg.load(SeqCst) { return; }
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(e) if e.kind() == ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(2)),
                    Err(_) => return,
                }
            };
            let mut server = loop {
                if stop_bg.load(SeqCst) { return; }
                match TcpStream::connect_timeout(&target, Duration::from_millis(100)) {
                    Ok(stream) => break stream,
                    Err(_) => std::thread::sleep(Duration::from_millis(2)),
                }
            };
            for s in [&client, &server] {
                let _ = s.set_nodelay(true);
                let _ = s.set_nonblocking(true);
            }
            let mut buf = [0u8; 32 * 1024];
            while !stop_bg.load(SeqCst) {
                let mut moved = false;
                for c2s in [true, false] {
                    let (from, to) = if c2s { (&mut client, &mut server) } else { (&mut server, &mut client) };
                    match from.read(&mut buf) {
                        Ok(0) => {
                            let _ = to.shutdown(Shutdown::Both);
                            return;
                        }
                        Ok(n) => {
                            moved = true;
                            let limit = drop_bg.load(SeqCst);
                            let seen = bytes_bg.load(SeqCst);
                            let allowed = limit.saturating_sub(seen).min(n);
                            if allowed > 0 {
                                let _ = to.set_nonblocking(false);
                                if to.write_all(&buf[..allowed]).is_err() { return; }
                                let _ = to.set_nonblocking(true);
                                bytes_bg.fetch_add(allowed, SeqCst);
                            }
                            if bytes_bg.load(SeqCst) >= limit {
                                let _ = client.shutdown(Shutdown::Both);
                                let _ = server.shutdown(Shutdown::Both);
                                return;
                            }
                        }
                        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::Interrupted) => (),
                        Err(_) => {
                            let _ = client.shutdown(Shutdown::Both);
                            let _ = server.shutdown(Shutdown::Both);
                            return;
                        }
                    }
                }
                if !moved { std::thread::sleep(Duration::from_micros(100)); }
            }
        });
        Self { entry, bytes, drop_after, stop, worker: Some(worker) }
    }
    fn reset_counter(&self) {
        self.bytes.store(0, SeqCst);
        self.drop_after.store(usize::MAX, SeqCst);
    }
    fn count(&self) -> usize { self.bytes.load(SeqCst) }
    fn fail_after(&self, limit: usize) { self.drop_after.store(limit, SeqCst); }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.stop.store(true, SeqCst);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

#[test]
fn token_major_packets_cross_two_ranks_in_both_directions() {
    let parts = 2;
    for path in [false, true] {
        let addresses = free_ports(parts);
        std::thread::scope(|scope| {
            let handles = (0..parts).map(|slot| {
                let addresses = &addresses;
                scope.spawn(move || {
                    let mut ring = Ring::connect_options(slot, parts, addresses[slot], addresses[(slot + 1) % parts], path).unwrap();
                    for tokens in [1, 3, 16, 17, 31, 255, 256, 257, 1024] {
                        for width in [3, 4, 144] {
                            for _ in 0..3 {
                                let packets = (0..parts).map(|r| payload(r, tokens, width)).collect::<Vec<_>>();
                                let words = tokens * width;
                                let mut exchange = vec![f32::NAN; parts * words];
                                exchange[slot * words..(slot + 1) * words].copy_from_slice(&packets[slot]);
                                ring.gather_rank_blocks(&mut exchange, parts, 1, words).unwrap();
                                let shape = Shape::rank_rows(tokens, width, parts, true).unwrap();
                                let output = prompt_ops::reference(shape, [&exchange, &[], &[], &[], &[]], &[]).unwrap();
                                let layouts = (0..parts)
                                    .map(|r| Layout::new(parts as u64, 1, width as u64, Topology { rank: r as u32, parts: parts as u32 }).unwrap())
                                    .collect::<Vec<_>>();
                                let refs = layouts.iter().copied().zip(packets.iter().map(Vec::as_slice)).collect::<Vec<_>>();
                                let expected = sparse_heads::gather_attended(&refs, tokens).unwrap();
                                assert!(output.iter().zip(expected).all(|(a, b)| a.to_bits() == b.to_bits()), "slot={slot} tokens={tokens} width={width}");
                            }
                        }
                    }
                })
            }).collect::<Vec<_>>();
            for handle in handles { handle.join().unwrap(); }
        });
    }
}

#[test]
fn two_rail_stripe_word_order_short_tail_and_uneven_floor() {
    let parts = 2;
    let primary = free_ports(parts);
    let aux = free_ports(parts);
    let relay0 = Relay::start(primary[1]);
    let relay1 = Relay::start(aux[1]);
    std::thread::scope(|scope| {
        let r0 = &relay0;
        let r1 = &relay1;
        let primary = &primary;
        let aux = &aux;
        let left = scope.spawn(move || {
            Ring::connect_options_with_aux(0, parts, primary[0], r0.entry, Some((aux[0], r1.entry)), false).unwrap()
        });
        let right = scope.spawn(move || {
            Ring::connect_options_with_aux(1, parts, primary[1], primary[0], Some((aux[1], aux[0])), false).unwrap()
        });
        let (mut ring0, mut ring1) = (left.join().unwrap(), right.join().unwrap());
        relay0.reset_counter();
        relay1.reset_counter();

        // 1. Odd word count >= 1 MiB per owner: 262,147 words (1,048,588 B) splits into
        // 131,074 words (524,296 B) on rail 0 and a shorter 131,073-word block (524,292 B) on rail 1.
        let odd_words = 262_147usize;
        let p0 = payload(0, odd_words, 1);
        let p1 = payload(1, odd_words, 1);
        let (mut buf0, mut buf1) = (vec![f32::NAN; parts * odd_words], vec![f32::NAN; parts * odd_words]);
        buf0[..odd_words].copy_from_slice(&p0);
        buf1[odd_words..].copy_from_slice(&p1);
        std::thread::scope(|inner| {
            let a = inner.spawn(|| ring0.gather_rank_blocks(&mut buf0, parts, 1, odd_words).unwrap());
            let b = inner.spawn(|| ring1.gather_rank_blocks(&mut buf1, parts, 1, odd_words).unwrap());
            a.join().unwrap();
            b.join().unwrap();
        });
        assert_eq!(relay0.count(), 2 * 131_074 * 4, "rail 0 must carry the first word-aligned stripe");
        assert_eq!(relay1.count(), 2 * 131_073 * 4, "rail 1 must carry the short last stripe");
        for (idx, (&w0, &w1)) in buf0.iter().zip(&buf1).enumerate() {
            let expected = if idx < odd_words { p0[idx] } else { p1[idx - odd_words] };
            assert_eq!(w0.to_bits(), expected.to_bits(), "word mismatch on side 0 at {idx}");
            assert_eq!(w1.to_bits(), expected.to_bits(), "word mismatch on side 1 at {idx}");
        }

        // 2. Prompt-group packet interleave over two rails (1,024 rows x 512 cols = 2 MiB per owner).
        relay0.reset_counter();
        relay1.reset_counter();
        let (tokens, width) = (1024, 512);
        let words = tokens * width;
        let packets = [payload(0, tokens, width), payload(1, tokens, width)];
        let (mut ex0, mut ex1) = (vec![0.0f32; parts * words], vec![0.0f32; parts * words]);
        ex0[..words].copy_from_slice(&packets[0]);
        ex1[words..].copy_from_slice(&packets[1]);
        std::thread::scope(|inner| {
            let a = inner.spawn(|| ring0.gather_rank_blocks(&mut ex0, parts, 1, words).unwrap());
            let b = inner.spawn(|| ring1.gather_rank_blocks(&mut ex1, parts, 1, words).unwrap());
            a.join().unwrap();
            b.join().unwrap();
        });
        assert_eq!(relay0.count(), words * 4);
        assert_eq!(relay1.count(), words * 4);
        let shape = Shape::rank_rows(tokens, width, parts, true).unwrap();
        let out0 = prompt_ops::reference(shape, [&ex0, &[], &[], &[], &[]], &[]).unwrap();
        let out1 = prompt_ops::reference(shape, [&ex1, &[], &[], &[], &[]], &[]).unwrap();
        let layouts = (0..parts)
            .map(|r| Layout::new(parts as u64, 1, width as u64, Topology { rank: r as u32, parts: parts as u32 }).unwrap())
            .collect::<Vec<_>>();
        let refs = layouts.iter().copied().zip(packets.iter().map(Vec::as_slice)).collect::<Vec<_>>();
        let expected = sparse_heads::gather_attended(&refs, tokens).unwrap();
        assert!(out0.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(out1.iter().zip(&expected).all(|(a, b)| a.to_bits() == b.to_bits()));

        // 3. Uneven owner slices around the 1 MiB rule: side 0 owns 257 * 1,023 * 4 = 1,051,644 B (>= 1 MiB),
        // side 1 owns 256 * 1,023 * 4 = 1,047,552 B (< 1 MiB). Both sides must stay on rail 0 only.
        relay0.reset_counter();
        relay1.reset_counter();
        let (units, uneven_width) = (513usize, 1023usize);
        let slice0 = 257 * uneven_width;
        let slice1 = 256 * uneven_width;
        assert!(slice0 * 4 >= PAIR_GATHER_FLOOR_BYTES && slice1 * 4 < PAIR_GATHER_FLOOR_BYTES);
        let local0 = (0..slice0).map(|i| i as f32).collect::<Vec<_>>();
        let local1 = (0..slice1).map(|i| (slice0 + i) as f32).collect::<Vec<_>>();
        let (joined0, joined1) = std::thread::scope(|inner| {
            let a = inner.spawn(|| ring0.gather_batch_f32(&local0, units, uneven_width, 1).unwrap());
            let b = inner.spawn(|| ring1.gather_batch_f32(&local1, units, uneven_width, 1).unwrap());
            (a.join().unwrap(), b.join().unwrap())
        });
        assert_eq!(relay1.count(), 0, "uneven slice below 1 MiB must keep rail 1 completely idle");
        assert_eq!(relay0.count(), (slice0 + slice1) * 4);
        assert!(joined0.iter().enumerate().all(|(i, &v)| v == i as f32));
        assert_eq!(joined0, joined1);
    });
}

#[test]
fn mid_gather_failure_on_second_rail_is_bounded_and_loud() {
    let parts = 2;
    let primary = free_ports(parts);
    let aux = free_ports(parts);
    let relay0 = Relay::start(primary[1]);
    let relay1 = Relay::start(aux[1]);
    let (mut ring0, mut ring1) = std::thread::scope(|scope| {
        let (r0, r1, primary, aux) = (&relay0, &relay1, &primary, &aux);
        let left = scope.spawn(move || {
            Ring::connect_options_with_aux(0, parts, primary[0], r0.entry, Some((aux[0], r1.entry)), false).unwrap()
        });
        let right = scope.spawn(move || {
            Ring::connect_options_with_aux(1, parts, primary[1], primary[0], Some((aux[1], aux[0])), false).unwrap()
        });
        (left.join().unwrap(), right.join().unwrap())
    });
    relay0.reset_counter();
    relay1.reset_counter();
    // Cut rail 1 mid-stripe after 200 KiB (each side sends 1 MiB on rail 1).
    relay1.fail_after(200 * 1024);

    let words = 512 * 1024; // 2 MiB per owner -> 1 MiB per rail.
    let (mut buf0, mut buf1) = (vec![1.0f32; parts * words], vec![2.0f32; parts * words]);
    let start = Instant::now();
    let (res0, res1) = std::thread::scope(|scope| {
        let a = scope.spawn(|| ring0.gather_rank_blocks(&mut buf0, parts, 1, words));
        let b = scope.spawn(|| ring1.gather_rank_blocks(&mut buf1, parts, 1, words));
        (a.join().unwrap(), b.join().unwrap())
    });
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(5), "mid-gather rail failure took {elapsed:?}");
    let err0 = res0.expect_err("side 0 must fail loudly when rail 1 drops mid-gather");
    let err1 = res1.expect_err("side 1 must fail loudly when rail 1 drops mid-gather");
    assert!(err0.contains("rail") || err0.contains("closed") || err0.contains("reset") || err0.contains("pipe"), "unexpected error: {err0}");
    assert!(err1.contains("rail") || err1.contains("closed") || err1.contains("reset") || err1.contains("pipe"), "unexpected error: {err1}");
}

// Child helper invoked by subprocess wire tests below; ignored during normal test discovery.
#[test]
#[ignore]
fn child_ring_connect_probe() {
    let mut line = String::new();
    if std::io::stdin().read_line(&mut line).is_err() || line.trim().is_empty() { return; }
    let mut fields = line.split_whitespace();
    let slot: usize = fields.next().unwrap().parse().unwrap();
    let listen: SocketAddr = fields.next().unwrap().parse().unwrap();
    let next: SocketAddr = fields.next().unwrap().parse().unwrap();
    let mut ring = match Ring::connect(slot, 2, listen, next) {
        Ok(ring) => ring,
        Err(err) => {
            eprintln!("CONNECT_ERR:{err}");
            std::process::exit(2);
        }
    };
    ring.barrier().unwrap();
    let words = 300_000usize; // > 1 MiB per owner
    let mut data = vec![0.0f32; 2 * words];
    for i in 0..words { data[slot * words + i] = (slot * words + i) as f32; }
    ring.gather_rank_blocks(&mut data, 2, 1, words).unwrap();
    assert!(data.iter().enumerate().all(|(i, &v)| v == i as f32));
    ring.barrier().unwrap();
}

fn spawn_probe(run_id: &str, slot: usize, listen: SocketAddr, next: SocketAddr, aux: Option<(&str, Option<&str>)>) -> std::process::Child {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.args(["--ignored", "--exact", "--nocapture", "child_ring_connect_probe"]);
    cmd.env("TP_RDMA", "0").env("TP_TOPOLOGY", "ring").env("TP_RUN_ID", run_id);
    match aux {
        Some((al, Some(an))) => { cmd.env("TP_AUX_LISTEN", al).env("TP_AUX_NEXT", an); }
        Some((al, None)) => { cmd.env("TP_AUX_LISTEN", al).env_remove("TP_AUX_NEXT"); }
        None => { cmd.env_remove("TP_AUX_LISTEN").env_remove("TP_AUX_NEXT"); }
    }
    let mut child = cmd
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    writeln!(child.stdin.take().unwrap(), "{slot} {listen} {next}").unwrap();
    child
}

#[test]
fn configured_aux_unconfigured_main_half_broken_fallback_and_older_peer_handshakes() {
    // 1. Both sides configured with TP_AUX_*: connect both rails and stripe >= 1 MiB cleanly.
    let primary = free_ports(2);
    let aux = free_ports(2);
    let relay1 = Relay::start(aux[1]);
    let (a0, a1, r1e) = (aux[0].to_string(), aux[1].to_string(), relay1.entry.to_string());
    let c0 = spawn_probe("prompt-xchg-wire-test", 0, primary[0], primary[1], Some((&a0, Some(&r1e))));
    let c1 = spawn_probe("prompt-xchg-wire-test", 1, primary[1], primary[0], Some((&a1, Some(&a0))));
    let (o0, o1) = (c0.wait_with_output().unwrap(), c1.wait_with_output().unwrap());
    assert!(o0.status.success() && o1.status.success(), "o0={:?} o1={:?}", String::from_utf8_lossy(&o0.stderr), String::from_utf8_lossy(&o1.stderr));
    assert!(relay1.count() > 2 * 500_000, "configured TP_AUX must carry second-rail gather traffic, got {}", relay1.count());

    // 2. Opt-out / unconfigured (TP_AUX_* unset on both sides): exact main wire protocol with zero extra bytes.
    let main_ports = free_ports(2);
    let c_unconf = spawn_probe("main-wire-compat", 0, main_ports[0], main_ports[1], None);
    let c_unconf_peer = spawn_probe("main-wire-compat", 1, main_ports[1], main_ports[0], None);
    let (u0, u1) = (c_unconf.wait_with_output().unwrap(), c_unconf_peer.wait_with_output().unwrap());
    assert!(u0.status.success() && u1.status.success(), "unconfigured pair failed: {:?} {:?}", String::from_utf8_lossy(&u0.stderr), String::from_utf8_lossy(&u1.stderr));
    assert!(!String::from_utf8_lossy(&u0.stderr).contains("tp_aux"), "unconfigured start must stay silent on tp_aux");

    // Verify raw main wire framing directly in-process: connect_options exchanges ONLY 32-byte VADETP\0\x04
    // and then immediately enters 1-byte barrier with zero extra negotiation bytes.
    let raw_ports = free_ports(2);
    std::thread::scope(|scope| {
        let raw_ports_ref = &raw_ports;
        let older_main_peer = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (tx, rx) = spark_seqpar::control::connect_peers(Some(raw_ports_ref[1]), Some(raw_ports_ref[0]), deadline).unwrap();
            let mut hello = [0u8; 32];
            hello[..8].copy_from_slice(b"VADETP\0\x04");
            hello[8..16].copy_from_slice(&1u64.to_le_bytes());
            hello[16..24].copy_from_slice(&2u64.to_le_bytes());
            hello[24..].copy_from_slice(&runtime::run_hash().to_le_bytes());
            let (mut tx, mut rx) = (tx.unwrap(), rx.unwrap());
            tx.write_all(&hello).unwrap();
            let mut from_new = [0u8; 32];
            rx.read_exact(&mut from_new).unwrap();
            assert_eq!(&from_new[..8], b"VADETP\0\x04");
            rx.write_all(&hello).unwrap();
            let mut ack = [0u8; 32];
            tx.read_exact(&mut ack).unwrap();
            rx.write_all(&[1u8]).unwrap();
            let mut release = [0u8; 1];
            rx.read_exact(&mut release).unwrap();
            assert_eq!(release, [1u8]);
        });
        let mut new_rank = Ring::connect_options(0, 2, raw_ports[0], raw_ports[1], false).unwrap();
        new_rank.barrier().unwrap();
        older_main_peer.join().unwrap();
    });

    // 3a. Half-broken second rail (present on side 0, down/unconfigured on side 1):
    // falls back to one rail in < 5 s with one clear log line, never failing the start, and completes gathers cleanly.
    let half_ports = free_ports(2);
    let unused_aux = free_ports(2);
    let (ua0, ua1) = (unused_aux[0].to_string(), unused_aux[1].to_string());
    let start = Instant::now();
    let h0 = spawn_probe("half-broken-aux-test", 0, half_ports[0], half_ports[1], Some((&ua0, Some(&ua1))));
    let h1 = spawn_probe("half-broken-aux-test", 1, half_ports[1], half_ports[0], None);
    let (ho0, ho1) = (h0.wait_with_output().unwrap(), h1.wait_with_output().unwrap());
    let elapsed = start.elapsed();
    assert!(elapsed < Duration::from_secs(5), "half-broken aux fallback took {elapsed:?}");
    assert!(ho0.status.success() && ho1.status.success(), "half-broken aux must fall back to one rail, not fail start: {:?} {:?}", String::from_utf8_lossy(&ho0.stderr), String::from_utf8_lossy(&ho1.stderr));
    let he0 = String::from_utf8_lossy(&ho0.stderr);
    let fallback_lines = he0.lines().filter(|l| l.contains("tp_aux rank=0 world=2 fallback=single_rail")).count();
    assert_eq!(fallback_lines, 1, "expected exactly one clear fallback log line, got: {he0}");

    // 3b. Half-configured env (only TP_AUX_LISTEN set on side 0) or invalid aux address on side 1:
    // both fall back cleanly with one log line and complete gathers on rail 0.
    let bad_env_ports = free_ports(2);
    let b0 = spawn_probe("bad-aux-env-test", 0, bad_env_ports[0], bad_env_ports[1], Some((&ua0, None)));
    let b1 = spawn_probe("bad-aux-env-test", 1, bad_env_ports[1], bad_env_ports[0], Some(("not-a-socket", Some("not-a-socket"))));
    let (bo0, bo1) = (b0.wait_with_output().unwrap(), b1.wait_with_output().unwrap());
    assert!(bo0.status.success() && bo1.status.success());
    assert_eq!(String::from_utf8_lossy(&bo0.stderr).lines().filter(|l| l.contains("tp_aux rank=0 world=2 fallback=single_rail")).count(), 1);
    assert_eq!(String::from_utf8_lossy(&bo1.stderr).lines().filter(|l| l.contains("tp_aux rank=1 world=2 fallback=single_rail")).count(), 1);

    // 4. Older protocol version peer (VADETP\0\x03) and mismatched run-id peer: both refuse cleanly in < 5 s with handshake mismatch.
    let mismatch_ports = free_ports(2);
    let start = Instant::now();
    let m0 = spawn_probe("run-id-a", 0, mismatch_ports[0], mismatch_ports[1], None);
    let m1 = spawn_probe("run-id-b", 1, mismatch_ports[1], mismatch_ports[0], None);
    let (mo0, mo1) = (m0.wait_with_output().unwrap(), m1.wait_with_output().unwrap());
    assert!(start.elapsed() < Duration::from_secs(5));
    assert!(!mo0.status.success() && !mo1.status.success());
    assert!(String::from_utf8_lossy(&mo0.stderr).contains("handshake mismatch"));
    assert!(String::from_utf8_lossy(&mo1.stderr).contains("handshake mismatch"));

    let old_ports = free_ports(2);
    let start = Instant::now();
    std::thread::scope(|scope| {
        let old_ports_ref = &old_ports;
        let old_peer = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (tx, rx) = spark_seqpar::control::connect_peers(Some(old_ports_ref[1]), Some(old_ports_ref[0]), deadline).unwrap();
            let mut old_hello = [0u8; 32];
            old_hello[..8].copy_from_slice(b"VADETP\0\x03");
            old_hello[8..16].copy_from_slice(&1u64.to_le_bytes());
            old_hello[16..24].copy_from_slice(&2u64.to_le_bytes());
            old_hello[24..].copy_from_slice(&runtime::run_hash().to_le_bytes());
            let (mut tx, mut rx) = (tx.unwrap(), rx.unwrap());
            // Set the read timeouts before the first byte goes out: the engine closes both
            // sockets as soon as it refuses this hello, and macOS setsockopt on a socket the
            // peer has closed fails with EINVAL.
            tx.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            rx.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let _ = tx.write_all(&old_hello);
            let mut peer = [0u8; 32];
            let _ = rx.read_exact(&mut peer);
            let _ = rx.write_all(&old_hello);
            let mut echoed = [0u8; 32];
            let _ = tx.read_exact(&mut echoed);
            echoed
        });
        let err = match Ring::connect_options(0, 2, old_ports[0], old_ports[1], false) {
            Ok(_) => panic!("older protocol peer must be refused"),
            Err(err) => err,
        };
        assert!(err.contains("handshake mismatch"), "unexpected error for older peer: {err}");
        let echoed = old_peer.join().unwrap();
        assert_eq!(&echoed[..8], b"VADETP\0\x04");
    });
    assert!(start.elapsed() < Duration::from_secs(5));
}
