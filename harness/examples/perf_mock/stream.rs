use super::process::{self, LiveUsage, Output, OwnedChild, Pty};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read, Write},
    net::TcpListener,
    os::fd::{FromRawFd, IntoRawFd},
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU32, Ordering},
        mpsc,
    },
    thread,
    time::{Duration, Instant},
};
const BEGIN: &str = "BENCH_BEGIN_7F";
const END: &str = "BENCH_END_7F";
fn event(text: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{}}}}}]}}\n\n",
        vade_harness::json::quote(text)
    )
}
fn chunk(writer: &mut impl Write, value: &str) -> io::Result<()> {
    write!(writer, "{:x}\r\n{value}\r\n", value.len())
}

struct Mock {
    endpoint: String,
    first: mpsc::Receiver<(Instant, LiveUsage)>,
    proceed: mpsc::Sender<()>,
    stop: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<io::Result<()>>>,
}
impl Mock {
    fn start(
        pid: Arc<AtomicU32>,
        characters: usize,
        per_event: usize,
        progressive: bool,
        deadline: Instant,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        listener.set_nonblocking(true)?;
        let endpoint = format!("http://{}/v1/chat/completions", listener.local_addr()?);
        let (sent, first) = mpsc::channel();
        let (proceed, wait) = mpsc::channel();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let worker = thread::spawn(move || {
            loop {
                if stopping.load(Ordering::Relaxed) || Instant::now() >= deadline {
                    return Err(process::timeout());
                }
                let (socket, _) = match listener.accept() {
                    Ok(value) => value,
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                    Err(e) => return Err(e),
                };
                socket.set_nonblocking(false)?;
                socket.set_nodelay(true)?;
                socket.set_read_timeout(Some(Duration::from_secs(5)))?;
                socket.set_write_timeout(Some(Duration::from_secs(5)))?;
                let mut reader = BufReader::new(socket);
                let mut first_line = String::new();
                reader.read_line(&mut first_line)?;
                let mut length = 0;
                for index in 0..128 {
                    let mut line = String::new();
                    if reader.read_line(&mut line)? == 0 {
                        return Err(io::ErrorKind::UnexpectedEof.into());
                    }
                    if line == "\r\n" {
                        break;
                    }
                    if index == 127 {
                        return Err(io::Error::other("mock header count exceeded"));
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse::<usize>().map_err(io::Error::other)?;
                        }
                    }
                }
                if length > 8 * 1024 * 1024 {
                    return Err(io::Error::other("mock request exceeds limit"));
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body)?;
                let socket = reader.get_mut();
                if first_line == "GET /v1/capabilities HTTP/1.1\r\n" {
                    socket.write_all(
                        b"HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n",
                    )?;
                    continue;
                }
                if first_line != "POST /v1/chat/completions HTTP/1.1\r\n" {
                    return Err(io::Error::other("unexpected mock request route"));
                }
                socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n")?;
                while pid.load(Ordering::Relaxed) == 0 {
                    if stopping.load(Ordering::Relaxed) || Instant::now() >= deadline {
                        return Err(process::timeout());
                    }
                    thread::yield_now();
                }
                let start_usage = process::usage(pid.load(Ordering::Relaxed))?;
                sent.send((Instant::now(), start_usage))
                    .map_err(io::Error::other)?;
                chunk(socket, &event(&format!("{BEGIN}\n")))?;
                socket.flush()?;
                // First output cannot be mistaken for completion-buffered text:
                // the remaining response waits for the observer's receipt.
                if progressive {
                    wait.recv_timeout(
                        deadline
                            .saturating_duration_since(Instant::now())
                            .min(Duration::from_secs(10)),
                    )
                    .map_err(io::Error::other)?;
                }
                let delta = event(&"x".repeat(per_event));
                let framed = format!("{:x}\r\n{delta}\r\n", delta.len());
                let batch = framed.repeat(16);
                let mut left = characters / per_event;
                while left >= 16 {
                    socket.write_all(batch.as_bytes())?;
                    left -= 16;
                }
                for _ in 0..left {
                    socket.write_all(framed.as_bytes())?;
                }
                chunk(socket, &event(&format!("\n{END}\n")))?;
                chunk(
                    socket,
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
                )?;
                socket.write_all(b"0\r\n\r\n")?;
                return Ok(());
            }
        });
        Ok(Self {
            endpoint,
            first,
            proceed,
            stop,
            worker: Some(worker),
        })
    }
    fn finish(mut self) -> io::Result<()> {
        self.worker
            .take()
            .unwrap()
            .join()
            .map_err(|_| io::Error::other("mock server panicked"))?
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.proceed.send(());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn startup(binary: &Path, workspace: &Path, deadline: Instant) -> io::Result<()> {
    let mut command = Command::new(binary);
    command
        .args(["demo", "--workspace"])
        .arg(workspace)
        .stderr(Stdio::inherit());
    let mut pty = Pty::attach(&mut command)?;
    let started = Instant::now();
    let child = OwnedChild::spawn(&mut command)?;
    let mut output = Output::new(pty.input.try_clone()?, true);
    let deadline = deadline.min(Instant::now() + Duration::from_secs(10));
    let ready = output.until("│> ".as_bytes(), deadline)?;
    pty.send(b"/quit\r")?;
    let (_, _, rss) = output.finish(child, deadline)?;
    println!(
        "startup ready_ms={:.3} peak_rss_bytes={rss}",
        ready.duration_since(started).as_secs_f64() * 1e3
    );
    Ok(())
}

pub fn run(
    binary: &Path,
    workspace: &Path,
    mode: &str,
    characters: usize,
    per_event: usize,
    raw: bool,
    deadline: Instant,
) -> io::Result<()> {
    assert_eq!(characters % per_event, 0);
    let deadline = deadline.min(Instant::now() + Duration::from_secs(30));
    let pid = Arc::new(AtomicU32::new(0));
    // Redirected native output intentionally publishes only the authoritative
    // completed answer. Requiring provisional output there would deadlock the
    // fixture and mislabel completion latency as first-token overhead.
    let progressive = raw || mode == "direct";
    let mock = Mock::start(pid.clone(), characters, per_event, progressive, deadline)?;
    let mut command = Command::new(binary);
    command
        .args([
            mode,
            "--endpoint",
            &mock.endpoint,
            "--model",
            "mock",
            "--workspace",
        ])
        .arg(workspace)
        .stderr(Stdio::inherit());
    // This fixture intentionally emits 100k characters: disable only the
    // native host reply cap, whose normal job is to end long model turns.
    if mode == "agent" {
        command.args(["--reply-tokens", "0"]);
    }
    let mut pty = if raw {
        Some(Pty::attach(&mut command)?)
    } else {
        command
            .arg("--plain")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        None
    };
    let mut child = OwnedChild::spawn(&mut command)?;
    pid.store(child.child.id(), Ordering::Relaxed);
    let file = if let Some(pty) = &pty {
        pty.input.try_clone()?
    } else {
        unsafe { File::from_raw_fd(child.child.stdout.take().unwrap().into_raw_fd()) }
    };
    let mut output = Output::new(file, raw);
    if let Some(pty) = &mut pty {
        output.until("│> ".as_bytes(), deadline)?;
        pty.send(b"synthetic benchmark\r")?;
    } else {
        child
            .child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(b"synthetic benchmark\n")?;
    }
    let rendered = output.until(BEGIN.as_bytes(), deadline)?;
    let (sent, start_usage) = mock
        .first
        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(io::Error::other)?;
    if progressive {
        mock.proceed.send(()).map_err(io::Error::other)?;
    }
    let ended = output.until(END.as_bytes(), deadline)?;
    let end_usage = process::usage(child.child.id())?;
    let wall_s = ended.duration_since(sent).as_secs_f64();
    let stream_user_s =
        process::tick_seconds(end_usage.counters[0].saturating_sub(start_usage.counters[0]))?;
    let stream_system_s =
        process::tick_seconds(end_usage.counters[1].saturating_sub(start_usage.counters[1]))?;
    if let Some(pty) = &mut pty {
        output.until("╰ ".as_bytes(), deadline)?;
        pty.send(b"/quit\r")?;
    } else {
        drop(child.child.stdin.take());
        output.drain(deadline)?;
        let expected = format!("{BEGIN}\n{}\n{END}\n", "x".repeat(characters));
        if !output
            .all
            .windows(expected.len())
            .any(|value| value == expected.as_bytes())
        {
            return Err(io::Error::other(
                "plain output does not contain the complete exact fixture answer",
            ));
        }
    }
    let (user_s, system_s, rss) = output.finish(child, deadline)?;
    mock.finish()?;
    if stream_user_s + stream_system_s > user_s + system_s + 0.002 {
        return Err(io::Error::other(
            "live CPU accounting exceeds whole-process wait4 accounting",
        ));
    }
    println!(
        "stream mode={mode} raw_tui={raw} synthetic_characters={characters} characters_per_event={per_event} progressive={progressive} first_output_ms={:.3} wall_s={wall_s:.6} stream_cpu_user_s={stream_user_s:.6} stream_cpu_system_s={stream_system_s:.6} stream_cpu_percent={:.2} resident_start_bytes={} resident_end_bytes={} process_cpu_user_s={user_s:.6} process_cpu_system_s={system_s:.6} peak_rss_bytes={rss} output_bytes={} answer_check={}",
        rendered.duration_since(sent).as_secs_f64() * 1e3,
        100.0 * (stream_user_s + stream_system_s) / wall_s,
        start_usage.counters[6],
        end_usage.counters[6],
        output.all.len(),
        if raw {
            "framed_start_and_end"
        } else {
            "full_exact_text"
        }
    );
    Ok(())
}
