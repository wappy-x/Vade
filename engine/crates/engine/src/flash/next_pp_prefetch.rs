//! One TCP reader overlaps receiving the next tile with the serving thread's work.
//! The wire still belongs to the serving thread, which checks the shared exchange IDs.
use super::*;

pub(super) struct Request {
    pub(super) id: u32,
    pub(super) kind: u32,
    pub(super) count: usize,
    pub(super) meta: Vec<[u32; 4]>,
    pub(super) rows: Vec<f32>,
}
pub(super) enum Incoming {
    Request(Request),
    Closed,
}

pub(super) struct Prefetch {
    socket: TcpStream,
    ready: Option<mpsc::Receiver<Result<Incoming>>>,
    recycle: Option<mpsc::SyncSender<Vec<f32>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl Prefetch {
    pub(super) fn start(stream: &TcpStream, tile: usize, stop: &'static AtomicBool, limit: Duration) -> Result<Self> {
        let (send, ready) = mpsc::sync_channel(1);
        let (recycle, spare) = mpsc::sync_channel(2);
        // Three row Vecs in total: the serving thread's current one, this spare,
        // and the reader's. At most one completed request waits in the channel.
        recycle.send(Vec::new()).map_err(err)?;
        let (socket, mut reader) = (stream.try_clone().map_err(err)?, stream.try_clone().map_err(err)?);
        let worker = std::thread::Builder::new().name("pp-prefetch".into()).spawn(move || {
            let mut rows = Vec::new();
            loop {
                let request = read_request(&mut reader, tile, stop, limit, rows);
                let terminal = !matches!(&request, Ok(Incoming::Request(_)));
                // EOF/errors follow complete requests; a full channel is bounded
                // backpressure. Drop disconnects it before joining this thread.
                if send.send(request).is_err() || terminal { return; }
                rows = match spare.recv() { Ok(rows) => rows, Err(_) => return };
            }
        }).map_err(err)?;
        Ok(Self { socket, ready: Some(ready), recycle: Some(recycle), worker: Some(worker) })
    }
    pub(super) fn poll(&self, wait: bool) -> Result<Option<Incoming>> {
        let ready = self.ready.as_ref().expect("live prefetch receiver");
        if wait {
            match ready.recv_timeout(Duration::from_millis(100)) {
                Ok(result) => result.map(Some),
                Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
                Err(mpsc::RecvTimeoutError::Disconnected) => Err("pipeline prefetch ended without a terminal result".into()),
            }
        } else {
            match ready.try_recv() {
                Ok(result) => result.map(Some),
                Err(mpsc::TryRecvError::Empty) => Ok(None),
                Err(mpsc::TryRecvError::Disconnected) => Err("pipeline prefetch ended without a terminal result".into()),
            }
        }
    }
    pub(super) fn recycle(&self, rows: Vec<f32>) -> Result<()> {
        match self.recycle.as_ref().expect("live prefetch recycler").try_send(rows) {
            Ok(()) | Err(mpsc::TrySendError::Disconnected(_)) => Ok(()),
            Err(mpsc::TrySendError::Full(_)) => Err("pipeline prefetch buffer accounting overflow".into()),
        }
    }
}
impl Drop for Prefetch {
    fn drop(&mut self) {
        // Wake either blocking send/receive before join. Shutdown affects this
        // socket's read half only, including a clone currently inside read().
        self.ready.take();
        self.recycle.take();
        let _ = self.socket.shutdown(std::net::Shutdown::Read);
        if let Some(worker) = self.worker.take() { let _ = worker.join(); }
    }
}

fn read_request(stream: &mut TcpStream, tile: usize, stop: &AtomicBool, limit: Duration, mut rows: Vec<f32>) -> Result<Incoming> {
    let mut head = [0u8; REQUEST_HEAD];
    if !read_message(stream, &mut head, Some(stop), limit, true)? { return Ok(Incoming::Closed); }
    // Preserve the preceding receiver's header deadline and its shared
    // metadata/payload deadline. Time waiting in the ready queue is not I/O time.
    let begun = Instant::now();
    let (magic, kind, count, n, id) = (word(&head, 0), word(&head, 1), word(&head, 2) as usize, word(&head, 3) as usize, word(&head, 4));
    if magic != MAGIC_REQUEST || !(1..=POOL_SIZE).contains(&n) || count == 0 || count > tile.max(POOL_SIZE) {
        return Err("pipeline request header invalid".into());
    }
    let elements = count.checked_mul(W).ok_or("pipeline request row size overflow")?;
    elements.checked_mul(4).ok_or("pipeline request byte size overflow")?;
    let left = || limit.saturating_sub(begun.elapsed()).max(Duration::from_millis(1));
    let mut raw = [0u8; POOL_SIZE * 16];
    read_rest(stream, &mut raw[..n * 16], Some(stop), left())?;
    let meta = (0..n).map(|i| std::array::from_fn(|j| word(&raw, i * 4 + j))).collect();
    rows.resize(elements, 0.0);
    read_rest(stream, f32_bytes_mut(&mut rows), Some(stop), left())?;
    Ok(Incoming::Request(Request { id, kind, count, meta, rows }))
}
