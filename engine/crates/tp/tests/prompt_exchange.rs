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

mod loopback { include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../test-support/loopback.rs")); }

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
fn token_major_prompt_packets_keep_every_word_on_two_and_four_ranks() {
    let heads = 4;
    for (parts, path) in [(2, false), (2, true), (4, false), (4, true)] {
        loopback::run_ranks(parts, 1, |slot, addresses| {
            Ring::connect_options(slot, parts, addresses[slot], addresses[(slot + 1) % parts], path)
        }, |slot, mut ring| {
            for tokens in [1, 2, 4, 8, 16, 17, 31, 255, 256, 257, 1024] {
                for width in [3, 4, 144] {
                    let full = payload(0, tokens, heads * width);
                    let local_width = heads / parts * width;
                    let words = tokens * local_width;
                    let local = full.chunks_exact(heads * width)
                        .flat_map(|row| &row[slot * local_width..(slot + 1) * local_width]);
                    let local = local.copied().collect::<Vec<_>>();
                    for _ in 0..3 {
                        let mut exchange = vec![f32::NAN; parts * words];
                        exchange[slot * words..(slot + 1) * words].copy_from_slice(&local);
                        ring.gather_rank_blocks(&mut exchange, parts, 1, words).unwrap();
                        let shape = Shape::rank_rows(tokens, local_width, parts, true).unwrap();
                        let output = prompt_ops::reference(shape, [&exchange, &[], &[], &[], &[]], &[]).unwrap();
                        assert_eq!(output.len(), full.len());
                        assert!(output.iter().zip(&full).all(|(a, b)| a.to_bits() == b.to_bits()), "parts={parts} path={path} slot={slot} tokens={tokens} width={width}");
                    }
                }
            }
        });
    }
}

#[test]
fn two_rail_stripe_word_order_short_tail_and_uneven_floor() {
    let parts = 2;
    let (mut ring0, mut ring1, relay0, relay1, _aux_return) = relayed_rings();
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
}

fn relayed_rings() -> (Ring, Ring, Relay, Relay, Relay) {
    loopback::with_ports(4, |addresses| {
        let (primary, aux) = addresses.split_at(2);
        let relay0 = Relay::start(primary[1]);
        let relay1 = Relay::start(aux[1]);
        let aux_return = Relay::start(aux[0]);
        let (entry0, entry1, return_entry) = (relay0.entry, relay1.entry, aux_return.entry);
        let rings = std::thread::scope(|scope| {
            let left = scope.spawn(move || {
                Ring::connect_options_with_aux(0, 2, primary[0], entry0, Some((aux[0], entry1)), false)
            });
            let right = scope.spawn(move || {
                Ring::connect_options_with_aux(1, 2, primary[1], primary[0], Some((aux[1], return_entry)), false)
            });
            (left.join().unwrap(), right.join().unwrap())
        });
        match rings {
            // Each side must have bound its port before it starts its outgoing link.
            (Ok(left), Ok(right)) if relay1.count() != 0 && aux_return.count() != 0 => {
                Ok((left, right, relay0, relay1, aux_return))
            }
            (Ok(_), Ok(_)) => Err(vec!["rank connection timed out".into()]),
            (left, right) => Err([left.err(), right.err()].into_iter().flatten().collect()),
        }
    })
}

#[test]
fn mid_gather_failure_on_second_rail_is_bounded_and_loud() {
    let parts = 2;
    let (mut ring0, mut ring1, relay0, relay1, _aux_return) = relayed_rings();
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

std::thread_local! {static EARLY_RECEIVE_DEVICE:std::cell::RefCell<Option<String>>=const {std::cell::RefCell::new(None)};}

/// The explicit device entry accepts its rank endpoints and fault through stdin. Normal tests never read stdin.
#[test]
#[ignore = "requires two RDMA ranks and fixture input"]
fn device_early_receive_fault_probe() {
    let mut line=String::new();std::io::stdin().read_line(&mut line).unwrap();assert!(!line.trim().is_empty());
    EARLY_RECEIVE_DEVICE.with(|v|*v.borrow_mut()=Some(line));
    a_ring_without_rdma_posts_no_early_receive_and_gathers_as_before();
    EARLY_RECEIVE_DEVICE.with(|v|*v.borrow_mut()=None);
}

/// RECEIVE FIRST on a link with no RDMA (every host test): nothing is posted, the answer says so, and the gather after it
/// exchanges the blocks as before. One rank alone asks; the other rank never does, which the wire must not notice.
#[test]
fn a_ring_without_rdma_posts_no_early_receive_and_gathers_as_before() {
    let parts = 2;
    // Opt-in device fixture: one process on each rank, using the normal production connection setup.
    // Stdin: rank, own socket, peer socket, fault (none, command, wrong-buffer, peer-abort).
    let device=EARLY_RECEIVE_DEVICE.with(|v|v.borrow().clone()).map(|line| {
        let fields=line.split_whitespace().collect::<Vec<_>>();assert_eq!(fields.len(),4);
        (fields[0].parse::<usize>().unwrap(),fields[1].parse::<SocketAddr>().unwrap(),fields[2].parse::<SocketAddr>().unwrap(),fields[3].to_owned())
    });
    let check = |slot: usize, mut ring: Ring, sync: Option<TcpListener>| {
    // A separate fixture socket confirms both receives are posted before a fault. It never uses the ring.
    if device.is_some() {assert!(ring.uses_rdma(),"device fixture requires RDMA");}
    for words in [3usize, 4096, 300_000] {
        for round in 0..3 {
            let mut exchange = vec![f32::NAN; parts * words];
            for i in 0..words { exchange[slot * words + i] = (round * 7 + slot * words + i) as f32; }
            let fault=device.as_ref().filter(|_|words==300_000 && round==2).map_or("none",|d|d.3.as_str());
            if slot == 0 || device.is_some() {
                // SAFETY: `exchange` stays allocated and fixed through the gather or abort below. No prior
                // reader remains. Fault cases abort the endpoints before this storage can be released.
                let posted = unsafe { ring.post_rank_blocks_receive(exchange.as_mut_ptr(), parts, 1, words) }.unwrap();
                if device.is_none() {assert!(!posted, "a link with no RDMA has no receive to post");}
                else if words==300_000 {assert!(posted,"large device exchange must post its receive");}
            }
            if fault!="none" {
                if let Some(listener)=&sync {
                    let (mut socket,_)=listener.accept().unwrap();socket.read_exact(&mut [0]).unwrap();socket.write_all(&[1]).unwrap();
                } else {
                    let mut address=device.as_ref().unwrap().2;address.set_port(address.port()+1);
                    let mut socket=TcpStream::connect(address).unwrap();socket.write_all(&[1]).unwrap();socket.read_exact(&mut [0]).unwrap();
                }
            }
            let start=std::time::Instant::now();
            if fault=="peer-abort" && slot==1 {ring.abort();println!("EARLY RECEIVE injected peer abort");return;}
            if fault!="none" && slot==0 {
                let result=match fault {
                    "command"=>ring.command(&mut [0]),
                    "wrong-buffer"=>ring.gather_rank_blocks(&mut vec![0.;parts*words],parts,1,words),
                    "peer-abort"=>ring.gather_rank_blocks(&mut exchange,parts,1,words),
                    _=>panic!("unknown fault"),
                };
                let error=result.expect_err("device fault must fail loudly");
                assert!(start.elapsed()<Duration::from_secs(5),"fault did not fail promptly");
                println!("EARLY RECEIVE FAULT PASS fault={fault} rank={slot} error={error}");return;
            }
            let result=ring.gather_rank_blocks(&mut exchange, parts, 1, words);
            if fault!="none" {assert!(result.is_err(),"peer must observe the aborted transport");println!("EARLY RECEIVE FAULT PASS fault={fault} rank={slot}");return;}
            result.unwrap();
            assert!(exchange.iter().enumerate().all(|(i, &v)| v == (round * 7 + i) as f32), "slot={slot} words={words} round={round}");
            println!("EARLY RECEIVE BYTES PASS rank={slot} words={words} round={round}");
        }
    }
    };
    if let Some(device) = &device {
        let sync = (device.0 == 0 && device.3 != "none").then(|| {
            let mut address = device.1;
            address.set_port(address.port() + 1);
            TcpListener::bind(address).unwrap()
        });
        let (slot, listen, next, _) = *device;
        check(slot, Ring::connect(slot, parts, listen, next).unwrap(), sync);
    } else {
        loopback::run_ranks(parts, 1, |rank, addresses| {
            Ring::connect_options(rank, parts, addresses[rank], addresses[1 - rank], false)
        }, |rank, ring| check(rank, ring, None));
    }
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

fn probe_pair(
    ports: &[SocketAddr],
    run_ids: [&str; 2],
    aux: [Option<(&str, Option<&str>)>; 2],
    mismatch_expected: bool,
    aux_required: bool,
) -> Result<[std::process::Output; 2], Vec<String>> {
    let children = [0, 1].map(|rank| {
        spawn_probe(run_ids[rank], rank, ports[rank], ports[1 - rank], aux[rank])
    });
    let outputs = children.map(|child| child.wait_with_output().unwrap());
    let mut errors = Vec::new();
    for output in &outputs {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Probe status 2 is reserved for connect errors. A failed gather or
        // assertion in either child must stop this test on its first attempt.
        assert!(
            output.status.success() || output.status.code() == Some(2),
            "wire probe failed: {stderr}"
        );
        for line in stderr.lines() {
            if let Some(error) = line.strip_prefix("CONNECT_ERR:")
                && (!mismatch_expected || !error.contains("handshake mismatch"))
            {
                errors.push(error.to_owned());
            }
            if line.starts_with("tp_aux ")
                && let Some((_, error)) = line.split_once("fallback=single_rail reason=")
                && (aux_required || error.starts_with("listen:"))
            {
                errors.push(error.to_owned());
            }
        }
    }
    if errors.is_empty() { Ok(outputs) } else { Err(errors) }
}

fn wire_pair(
    run_ids: [&str; 2],
    aux: [Option<(&str, Option<&str>)>; 2],
    mismatch_expected: bool,
) -> [std::process::Output; 2] {
    loopback::with_ports(2, |ports| probe_pair(ports, run_ids, aux, mismatch_expected, false))
}

#[test]
fn configured_aux_unconfigured_main_half_broken_fallback_and_older_peer_handshakes() {
    // 1. Both sides configured with TP_AUX_*: connect both rails and stripe >= 1 MiB cleanly.
    let ([o0, o1], relay1) = loopback::with_ports(4, |ports| {
        let relay = Relay::start(ports[3]);
        let (a0, a1, entry) = (ports[2].to_string(), ports[3].to_string(), relay.entry.to_string());
        let output = probe_pair(
            &ports[..2],
            ["prompt-xchg-wire-test"; 2],
            [Some((&a0, Some(&entry))), Some((&a1, Some(&a0)))],
            false,
            true,
        )?;
        Ok((output, relay))
    });
    assert!(o0.status.success() && o1.status.success(), "o0={:?} o1={:?}", String::from_utf8_lossy(&o0.stderr), String::from_utf8_lossy(&o1.stderr));
    assert!(relay1.count() > 2 * 500_000, "configured TP_AUX must carry second-rail gather traffic, got {}", relay1.count());

    // 2. Opt-out / unconfigured (TP_AUX_* unset on both sides): exact main wire protocol with zero extra bytes.
    let [u0, u1] = wire_pair(["main-wire-compat"; 2], [None; 2], false);
    assert!(u0.status.success() && u1.status.success(), "unconfigured pair failed: {:?} {:?}", String::from_utf8_lossy(&u0.stderr), String::from_utf8_lossy(&u1.stderr));
    assert!(!String::from_utf8_lossy(&u0.stderr).contains("tp_aux"), "unconfigured start must stay silent on tp_aux");

    // Verify raw main wire framing directly in-process: connect_options exchanges ONLY 32-byte VADETP\0\x04
    // and then immediately enters 1-byte barrier with zero extra negotiation bytes.
    loopback::with_ports(2, |ports| std::thread::scope(|scope| {
        let older_main_peer = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (tx, rx) = spark_seqpar::control::connect_peers(Some(ports[1]), Some(ports[0]), deadline)?;
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
            Ok(())
        });
        let result = Ring::connect_options(0, 2, ports[0], ports[1], false);
        if let Ok(mut ring) = result {
            ring.barrier().unwrap();
            older_main_peer.join().unwrap().map_err(|error| vec![error])
        } else {
            let peer: Result<(), String> = older_main_peer.join().unwrap();
            Err([result.err(), peer.err()].into_iter().flatten().collect())
        }
    }));

    // 3a. Half-broken second rail (present on side 0, down/unconfigured on side 1):
    // falls back to one rail in < 5 s with one clear log line, never failing the start, and completes gathers cleanly.
    let ([ho0, ho1], elapsed) = loopback::with_ports(3, |ports| {
        // Hold a separate socket without accepting, so the missing auxiliary peer cannot be stolen.
        let unavailable = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let (listen, peer) = (ports[2].to_string(), unavailable.local_addr().unwrap().to_string());
        let start = Instant::now();
        let output = probe_pair(
            &ports[..2], ["half-broken-aux-test"; 2], [Some((&listen, Some(&peer))), None], false, false,
        )?;
        Ok((output, start.elapsed()))
    });
    assert!(elapsed < Duration::from_secs(5), "half-broken aux fallback took {elapsed:?}");
    assert!(ho0.status.success() && ho1.status.success(), "half-broken aux must fall back to one rail, not fail start: {:?} {:?}", String::from_utf8_lossy(&ho0.stderr), String::from_utf8_lossy(&ho1.stderr));
    let he0 = String::from_utf8_lossy(&ho0.stderr);
    let fallback_lines = he0.lines().filter(|l| l.contains("tp_aux rank=0 world=2 fallback=single_rail")).count();
    assert_eq!(fallback_lines, 1, "expected exactly one clear fallback log line, got: {he0}");

    // 3b. Half-configured env (only TP_AUX_LISTEN set on side 0) or invalid aux address on side 1:
    // both fall back cleanly with one log line and complete gathers on rail 0.
    let [bo0, bo1] = wire_pair(
        ["bad-aux-env-test"; 2],
        [Some(("127.0.0.1:0", None)), Some(("not-a-socket", Some("not-a-socket")))],
        false,
    );
    assert!(bo0.status.success() && bo1.status.success());
    assert_eq!(String::from_utf8_lossy(&bo0.stderr).lines().filter(|l| l.contains("tp_aux rank=0 world=2 fallback=single_rail")).count(), 1);
    assert_eq!(String::from_utf8_lossy(&bo1.stderr).lines().filter(|l| l.contains("tp_aux rank=1 world=2 fallback=single_rail")).count(), 1);

    // 4. Older protocol version peer (VADETP\0\x03) and mismatched run-id peer: both refuse cleanly in < 5 s with handshake mismatch.
    let ([mo0, mo1], elapsed) = loopback::with_ports(2, |ports| {
        let start = Instant::now();
        let output = probe_pair(ports, ["run-id-a", "run-id-b"], [None; 2], true, false)?;
        Ok((output, start.elapsed()))
    });
    assert!(elapsed < Duration::from_secs(5));
    assert!(!mo0.status.success() && !mo1.status.success());
    assert!(String::from_utf8_lossy(&mo0.stderr).contains("handshake mismatch"));
    assert!(String::from_utf8_lossy(&mo1.stderr).contains("handshake mismatch"));

    loopback::with_ports(2, |ports| std::thread::scope(|scope| {
        let start = Instant::now();
        let old_peer = scope.spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            let (tx, rx) = spark_seqpar::control::connect_peers(Some(ports[1]), Some(ports[0]), deadline)?;
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
            Ok(echoed)
        });
        let err = match Ring::connect_options(0, 2, ports[0], ports[1], false) {
            Ok(_) => panic!("older protocol peer must be refused"),
            Err(err) => err,
        };
        let peer: Result<_, String> = old_peer.join().unwrap();
        let echoed = match peer {
            Ok(echoed) if err.contains("handshake mismatch") => echoed,
            result => return Err(std::iter::once(err).chain(result.err()).collect()),
        };
        assert_eq!(&echoed[..8], b"VADETP\0\x04");
        assert!(start.elapsed() < Duration::from_secs(5));
        Ok(())
    }));
}
