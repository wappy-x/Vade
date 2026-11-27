use super::*;
use std::sync::atomic::AtomicU64;

/// A back that reads one pooled request, heartbeats every `beat` for `work`, then
/// answers it (rows + 1) with the request's id.
pub(super) fn slow_answer(back: &mut TcpStream, beat: Option<Duration>, work: Duration, beat_id: Option<u32>) {
    let mut request = vec![0u8; REQUEST_HEAD + 16 + W * 4];
    assert!(read_message(back, &mut request, None, Duration::from_secs(5), true).unwrap());
    let id = word(&request, 4);
    let started = Instant::now();
    while started.elapsed() < work {
        match beat {
            Some(period) => {
                std::thread::sleep(period);
                if back.write_all(&working_head(beat_id.unwrap_or(id))).is_err() { return; }
            }
            None => std::thread::sleep(Duration::from_millis(10)),
        }
    }
    let mut reply = Vec::new();
    for v in [MAGIC_RESPONSE, 0, (W * 4) as u32, id] { put(&mut reply, v); }
    let rows: Vec<f32> = request[REQUEST_HEAD + 16..].chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) + 1.0).collect();
    reply.extend_from_slice(f32_bytes(&rows));
    let _ = back.write_all(&reply);
}

/// Run a test body in its own thread; a hang fails the test after 60 s of real time.
pub(super) fn within_a_minute(body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    let worker = std::thread::spawn(move || { body(); let _ = done.send(()); });
    match finished.recv_timeout(Duration::from_secs(60)) {
        Ok(()) => worker.join().unwrap(),
        Err(mpsc::RecvTimeoutError::Disconnected) => std::panic::resume_unwind(worker.join().unwrap_err()),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("heartbeat test hung: no result after 60 s of real time"),
    }
}

/// The reply to a request: its id, its rows + 1.
pub(super) fn plus_one(request: &[u8]) -> Vec<u8> {
    let rows: Vec<f32> = request[REQUEST_HEAD + 16..].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) + 1.0).collect();
    let mut reply = Vec::new();
    for v in [MAGIC_RESPONSE, 0, (rows.len() * 4) as u32, word(request, 4)] { put(&mut reply, v); }
    reply.extend_from_slice(f32_bytes(&rows));
    reply
}

/// A test back that owns the front's manual clock and steps it in lockstep with the front
/// (heartbeat::Steps). Its heartbeats come from the production heartbeat wake
/// (heartbeat::beat) at the production interval (beat_period), on the manual clock.
pub(super) struct Back {
    pub(super) tcp: TcpStream,
    /// The front's own socket (a clone), only peeked: what the back wrote is readable there.
    front: TcpStream,
    clock: Arc<AtomicU64>,
    origin: Instant,
    ready: mpsc::Receiver<()>,
    step: mpsc::Sender<usize>,
    working: Mutex<Option<(u32, Instant)>>,
    pub(super) period: Duration,
    queued: (mpsc::SyncSender<ReplySend>, mpsc::Receiver<ReplySend>),
    /// Heartbeats written.
    pub(super) sent: u64,
}
impl Back {
    pub(super) fn manual(link: &mut Link, tcp: TcpStream) -> Self {
        let (clock, origin) = (Arc::new(AtomicU64::new(0)), Instant::now());
        let ((ready_tx, ready), (step, step_rx)) = (mpsc::channel(), mpsc::channel());
        link.test_clock = Some((origin, Arc::clone(&clock)));
        link.test_steps = Some(heartbeat::Steps { ready: ready_tx, step: step_rx, unread: 0 });
        let working = Mutex::new(None);
        let front = link.stream.try_clone().unwrap();
        Back { tcp, front, clock, origin, ready, step, working, period: beat_period(link.limit), queued: mpsc::sync_channel(2), sent: 0 }
    }
    pub(super) fn now(&self) -> Duration { Duration::from_millis(self.clock.load(Ordering::Relaxed)) }
    fn instant(&self) -> Instant { self.origin + self.now() }
    /// Run request `id` from now (the production back marks it the same way), or stop.
    pub(super) fn mark(&self, id: Option<u32>) { *self.working.lock().unwrap() = id.map(|id| (id, self.instant())); }
    /// Read one TCP request of `rows` rows.
    pub(super) fn tcp_request(&mut self, rows: usize) -> Vec<u8> {
        let mut request = vec![0u8; REQUEST_HEAD + 16 + rows * W * 4];
        assert!(read_message(&mut self.tcp, &mut request, None, Duration::MAX, true).unwrap());
        request
    }
    pub(super) fn step(&mut self, by: Duration, beating: bool) -> bool { self.step_with(by, beating, |_| 0) }
    /// Answer on TCP in one step with no clock move: the reply header, then its payload.
    /// The front reads the payload on the manual clock (Link::test_payload_within); the step
    /// is released only once the whole reply is readable, so that read never waits on the back.
    pub(super) fn answer(&mut self, reply: &[u8]) -> bool {
        self.step_with(Duration::ZERO, false, |tcp| {
            tcp.write_all(&reply[..REPLY_HEAD]).unwrap();
            tcp.write_all(&reply[REPLY_HEAD..]).unwrap();
            1
        })
    }
    /// One step: wait until the front waits; advance the clock by `by`; if `beating`, run
    /// the production heartbeat wake and write what it queued; then `then` (it returns the
    /// TCP headers it wrote). Every byte written in the step is readable at the front before
    /// the step is released (the handoff). False once the front stopped waiting.
    pub(super) fn step_with(&mut self, by: Duration, beating: bool, then: impl FnOnce(&mut Written) -> usize) -> bool {
        if self.ready.recv().is_err() { return false; }
        self.clock.fetch_add(by.as_millis() as u64, Ordering::Relaxed);
        if beating { heartbeat::beat(&self.working, self.period, &self.queued.0, || self.instant()); }
        let mut headers = 0;
        let mut tcp = Written { tcp: &mut self.tcp, bytes: 0 };
        while let Ok(item) = self.queued.1.try_recv() {
            let ReplySend::Bytes(head) = item else { panic!("the heartbeat wake queued rows") };
            tcp.write_all(&head).unwrap();
            (headers, self.sent) = (headers + 1, self.sent + 1);
        }
        headers += then(&mut tcp);
        let written = tcp.bytes;
        self.until_readable(written);
        self.step.send(headers).is_ok()
    }
    /// Wait (real time, under the 60 s guard) until `bytes` are readable at the front. The
    /// front has read everything written before this step: on TCP it asks for a step only
    /// once no header is unread, and it reads a reply's payload right after its header.
    fn until_readable(&self, bytes: usize) {
        if bytes == 0 { return; }
        let mut probe = vec![0u8; bytes];
        loop {
            match self.front.peek(&mut probe) {
                Ok(0) => panic!("the front closed its socket"),
                Ok(n) if n >= bytes => return,
                Ok(_) => std::thread::yield_now(),
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted) => (),
                Err(e) => panic!("peek at the front: {e}"),
            }
        }
    }
}

/// The back's TCP writer for one step: counts the bytes written.
pub(super) struct Written<'a> {
    tcp: &'a mut TcpStream,
    bytes: usize,
}
impl Write for Written<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.tcp.write(buf)?;
        self.bytes += n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.tcp.flush() }
}
