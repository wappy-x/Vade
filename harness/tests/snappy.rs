//! Opt-in Mac PTY latency benchmark; no model or accelerator required.
#![cfg(target_os = "macos")]
#[path = "support/home.rs"]
mod home;
#[path = "support/raw_ui.rs"]
mod raw_ui;
use std::{io::{Read, Write}, net::{Ipv4Addr, TcpListener}, sync::mpsc, thread, time::{Duration, Instant}};
fn child_cpu() -> f64 {
    unsafe extern "C" { fn getrusage(who: i32, usage: *mut [i64; 18]) -> i32; }
    let mut t = [0; 18]; assert_eq!(unsafe { getrusage(-1, &mut t) }, 0);
    (t[0] + t[2]) as f64 * 1e6 + (t[1] + t[3]) as f64
}
#[test]
#[ignore = "latency measurement; run release, serially, with --nocapture"]
fn snappy_pty() {
    for run in 0..5 { for mode in ["direct", "agent"] { for rate in [0, 60, 200] {
        let root = std::env::temp_dir().join(format!("vade-snappy-{}-{run}-{mode}-{rate}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let server = thread::spawn(move || {
            loop {
            let (mut socket, _) = listener.accept().unwrap();
            socket.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
            let mut byte = [0];
            socket.read_exact(&mut byte).unwrap();
            tx.send(Instant::now()).unwrap();
            let mut request = vec![byte[0]];
            while !request.ends_with(b"\r\n\r\n") { socket.read_exact(&mut byte).unwrap(); request.push(byte[0]); }
            let head = String::from_utf8(request).unwrap();
            if head.starts_with("GET ") { socket.write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n").unwrap(); continue; }
            let size: usize = head.lines().find_map(|s| s.to_ascii_lowercase().strip_prefix("content-length:").map(|s| s.trim().parse().unwrap())).unwrap();
            let mut body = vec![0; size]; socket.read_exact(&mut body).unwrap();
            let mut hash = std::hash::DefaultHasher::new(); std::hash::Hasher::write(&mut hash, &body);
            println!("request_bytes={} request_hash={:016x}", body.len(), std::hash::Hasher::finish(&hash));
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
            if rate > 0 { for token in 0..120 {
                write!(socket, "data: {{\"choices\":[{{\"delta\":{{\"content\":\"word{token} \"}}}}]}}\n\n").unwrap();
                thread::sleep(Duration::from_secs_f64(1. / rate as f64));
            }}
            socket.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n").unwrap();
            break;
            }
        });
        let before_cpu = child_cpu(); let start = Instant::now();
        let mut ui = raw_ui::Ui::start(mode, &endpoint, root.to_str().unwrap(), false);
        let cold = start.elapsed().as_secs_f64() * 1000.;
        let mut keys = Vec::new();
        let mut draft = String::new();
        for _ in 0..100 {
            draft.push('x'); let start = Instant::now(); ui.send(b"x");
            ui.until(&draft, Duration::from_secs(3)); keys.push(start.elapsed().as_secs_f64() * 1000.);
        }
        keys.sort_by(f64::total_cmp);
        ui.discard_output(); let start = Instant::now(); ui.resize(30, 100);
        ui.until(&format!("└{}┘", "─".repeat(97)), Duration::from_secs(3)); let resize = start.elapsed().as_secs_f64() * 1000.;
        ui.discard_output(); let start = Instant::now(); ui.send(b"\r");
        let wire = rx.recv_timeout(Duration::from_secs(10)).unwrap().duration_since(start).as_secs_f64() * 1000.;
        let output = ui.until("╰ ", Duration::from_secs(3)); ui.send(b"/quit\r"); ui.wait_exit();
        let cpu = child_cpu() - before_cpu;
        server.join().unwrap(); drop(ui); std::fs::remove_dir_all(root).unwrap();
        println!("snappy run={run} mode={mode} rate={rate} start_ms={cold:.3} key_p50_ms={:.3} key_p99_ms={:.3} enter_wire_ms={wire:.3} resize_ms={resize:.3} child_cpu_us={cpu:.0} stream_bytes={} frames={}", keys[49], keys[98], output.len(), output.matches("\x1b[?2026h").count());
    }}}
}

/// Opt-in performance contract: the first visible answer bypasses the sustained
/// redraw rate. Use a median and a wide margin below the old 33 ms wait, rather
/// than making scheduler-sensitive latency a default CI assertion.
#[test]
#[ignore = "first-frame latency check; run release, serially, with --nocapture"]
fn first_answer_frame_is_immediate() {
    for mode in ["direct", "agent"] {
        let mut samples = Vec::new();
        for trial in 0..7 {
            let root = std::env::temp_dir().join(format!(
                "vade-first-frame-{}-{mode}-{trial}", std::process::id()
            ));
            std::fs::create_dir(&root).unwrap();
            let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let (sent, started) = mpsc::channel();
            let (advance, observed) = mpsc::channel();
            let server = thread::spawn(move || {
                loop {
                    let (mut socket, _) = listener.accept().unwrap();
                    socket.set_nodelay(true).unwrap();
                    socket.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
                    socket.set_write_timeout(Some(Duration::from_secs(5))).unwrap();
                    let mut header = Vec::new();
                    while !header.ends_with(b"\r\n\r\n") {
                        let mut byte = [0];
                        socket.read_exact(&mut byte).unwrap();
                        header.push(byte[0]);
                        assert!(header.len() < 16384);
                    }
                    let header = String::from_utf8(header).unwrap();
                    let size: usize = header.lines().find_map(|line| {
                        line.to_ascii_lowercase().strip_prefix("content-length:")
                            .map(|value| value.trim().parse().unwrap())
                    }).unwrap_or(0);
                    assert!(size < 1024 * 1024);
                    socket.read_exact(&mut vec![0; size]).unwrap();
                    if header.starts_with("GET /v1/capabilities ") {
                        socket.write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n").unwrap();
                        continue;
                    }
                    assert!(header.starts_with("POST /v1/chat/completions "));
                    socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n").unwrap();
                    sent.send(Instant::now()).unwrap();
                    socket.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\"FIRST_FRAME_RECEIPT\"}}]}\n\n").unwrap();
                    // The answer cannot be buffered until completion: the
                    // terminal must render this first event before continuing.
                    observed.recv_timeout(Duration::from_secs(5)).unwrap();
                    socket.write_all(b"data: {\"choices\":[{\"delta\":{\"content\":\" done\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n").unwrap();
                    break;
                }
            });
            let mut ui = raw_ui::Ui::start(mode, &endpoint, root.to_str().unwrap(), false);
            ui.send(b"first frame\r");
            ui.until("FIRST_FRAME_RECEIPT", Duration::from_secs(5));
            let seen = Instant::now();
            samples.push(seen.duration_since(started.recv().unwrap()).as_secs_f64() * 1000.);
            advance.send(()).unwrap();
            ui.until("╰ ", Duration::from_secs(5));
            ui.send(b"/quit\r");
            ui.wait_exit();
            server.join().unwrap();
            drop(ui);
            std::fs::remove_dir_all(root).unwrap();
        }
        samples.sort_by(f64::total_cmp);
        println!("first_frame mode={mode} n=7 median_ms={:.3} samples_ms={samples:?}", samples[3]);
        assert!(samples[3] < 20., "first answer still waits for the sustained redraw budget: {samples:?}");
    }
}
