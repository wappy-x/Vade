//! Offset-sorted read plan for one prompt tile's host lookup rows (PLE nibbles,
//! PLE scales, embedding rows). Requests are sorted by (file, offset), merged
//! where they overlap or touch (duplicates collapse), read once per merged
//! extent by a pool of readers that first hint their extents to the kernel, and
//! scattered into each request's own destination. Every destination receives
//! exactly the file bytes its own pread would return, so outputs are unchanged;
//! only the syscall count and the number of reads in flight change.
//! Idea credit: llama.cpp PR 28136 (sorted, deduplicated parallel row preads)
//! and unsloth llama.cpp PR 137 (batched WILLNEED of the next step's pages).
use std::{
    fs::File,
    io,
    os::{fd::AsRawFd, unix::fs::FileExt},
    sync::{Arc, Mutex, OnceLock, mpsc},
};
/// Below this many extents per reader the thread hand-off costs more than it
/// overlaps (decode verify tiles stay on the calling thread).
const MIN_EXTENTS: usize = 1024;
#[derive(Clone, Copy, Debug)]
pub(crate) struct Request {
    pub file: u32,
    /// Absolute file offset.
    pub at: u64,
    pub len: u32,
    /// Offset in the caller's output buffer.
    pub dest: usize,
}
#[derive(Clone, Copy, Debug)]
struct Extent {
    file: u32,
    at: u64,
    len: u64,
}
pub(crate) struct Plan {
    extents: Vec<Extent>,
    /// (offset in the extent buffer, request) per request, in sorted order.
    pieces: Vec<(usize, Request)>,
    bytes: usize,
}
/// Reader threads per tile: VADE_PLE_READERS (default 16, at least 1).
pub(crate) fn readers() -> usize {
    static READERS: OnceLock<usize> = OnceLock::new();
    *READERS.get_or_init(|| {
        std::env::var("VADE_PLE_READERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(16)
            .clamp(1, 256)
    })
}
/// Foreground hints before the preads: VADE_PLE_HINTS (default 1; 0 = plain
/// preads, for measuring the page-cache-warm cost of the hint syscalls).
fn hints() -> bool {
    static HINTS: OnceLock<bool> = OnceLock::new();
    *HINTS.get_or_init(|| std::env::var("VADE_PLE_HINTS").map_or(true, |v| v != "0"))
}
impl Plan {
    pub(crate) fn new(mut requests: Vec<Request>) -> Self {
        requests.sort_unstable_by_key(|r| (r.file, r.at));
        let mut extents: Vec<Extent> = Vec::new();
        let mut pieces = Vec::with_capacity(requests.len());
        let mut bytes = 0usize;
        for r in requests {
            let end = r.at + u64::from(r.len);
            match extents.last_mut() {
                // The last extent owns the tail of the buffer, so it can grow.
                Some(e) if e.file == r.file && r.at <= e.at + e.len => {
                    let grow = end.saturating_sub(e.at + e.len);
                    let base = bytes - e.len as usize;
                    e.len += grow;
                    bytes += grow as usize;
                    pieces.push((base + (r.at - e.at) as usize, r));
                }
                _ => {
                    extents.push(Extent { file: r.file, at: r.at, len: u64::from(r.len) });
                    pieces.push((bytes, r));
                    bytes += r.len as usize;
                }
            }
        }
        Self { extents, pieces, bytes }
    }
    #[cfg(test)]
    pub(crate) fn extents(&self) -> usize {
        self.extents.len()
    }
    #[cfg(test)]
    pub(crate) fn extent_bytes(&self) -> usize {
        self.bytes
    }
    /// Read every extent (`readers` threads at most, each hinting its extents
    /// before its first pread) and scatter the requests into `out`.
    pub(crate) fn read(&self, files: &[Arc<File>], readers: usize, out: &mut [u8]) -> io::Result<()> {
        if let Some((_, r)) = self.pieces.iter().find(|(_, r)| r.dest + r.len as usize > out.len()) {
            return Err(io::Error::other(format!("read plan destination {} exceeds output", r.dest)));
        }
        if self.extents.iter().any(|e| e.file as usize >= files.len()) {
            return Err(io::Error::other("read plan names a missing file"));
        }
        let mut buf = vec![0u8; self.bytes];
        let groups = readers.min(self.extents.len().div_ceil(MIN_EXTENTS)).max(1);
        if groups == 1 {
            read_group(&self.extents, files, &mut buf)?;
        } else {
            let per = self.extents.len().div_ceil(groups);
            std::thread::scope(|scope| {
                let mut rest = &mut buf[..];
                let jobs: Vec<_> = self
                    .extents
                    .chunks(per)
                    .map(|group| {
                        let size = group.iter().map(|e| e.len as usize).sum();
                        let (mine, tail) = std::mem::take(&mut rest).split_at_mut(size);
                        rest = tail;
                        scope.spawn(move || read_group(group, files, mine))
                    })
                    .collect();
                jobs.into_iter().try_for_each(|job| {
                    job.join().map_err(|_| io::Error::other("PLE reader panicked"))?
                })
            })?;
        }
        for &(src, r) in &self.pieces {
            let len = r.len as usize;
            out[r.dest..r.dest + len].copy_from_slice(&buf[src..src + len]);
        }
        Ok(())
    }
    /// Page-cache hints only (no bytes move): at most `budget` extent bytes,
    /// lowest offsets first. Never blocks on I/O completion.
    pub(crate) fn hint(&self, files: &[Arc<File>], budget: usize) {
        let mut left = budget as u64;
        for e in &self.extents {
            if left < e.len {
                break;
            }
            left -= e.len;
            if let Some(file) = files.get(e.file as usize) {
                advise(file, e.at, e.len);
            }
        }
    }
}
fn read_group(group: &[Extent], files: &[Arc<File>], buf: &mut [u8]) -> io::Result<()> {
    if hints() {
        for e in group {
            advise(&files[e.file as usize], e.at, e.len);
        }
    }
    let mut at = 0;
    for e in group {
        let len = e.len as usize;
        files[e.file as usize].read_exact_at(&mut buf[at..at + len], e.at)?;
        at += len;
    }
    Ok(())
}
/// Ask the kernel to start reading the pages now and return (Linux
/// POSIX_FADV_WILLNEED; macOS F_RDADVISE). A hint: bytes are unchanged.
#[cfg(target_os = "linux")]
pub(crate) fn advise(file: &File, at: u64, len: u64) {
    unsafe extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    unsafe {
        posix_fadvise(file.as_raw_fd(), at as i64, len as i64, 3);
    }
}
#[cfg(target_os = "macos")]
pub(crate) fn advise(file: &File, at: u64, len: u64) {
    #[repr(C)]
    struct Radvisory {
        offset: i64,
        count: i32,
    }
    unsafe extern "C" {
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
    }
    let advice = Radvisory { offset: at as i64, count: len.min(i32::MAX as u64) as i32 };
    unsafe {
        fcntl(file.as_raw_fd(), 44, &advice as *const Radvisory);
    }
}
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn advise(_: &File, _: u64, _: u64) {}
/// A background job; its argument says whether it is the latest queued one.
type Job = Box<dyn FnOnce(bool) + Send>;
/// One long-lived background thread for read-ahead work: jobs run in order,
/// the caller only enqueues (never waits on hashing or I/O).
pub(crate) fn background(job: Job) {
    static PENDING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    static QUEUE: OnceLock<Mutex<mpsc::Sender<Job>>> = OnceLock::new();
    let queue = QUEUE.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("ple-read-ahead".into())
            .spawn(move || {
                for job in rx {
                    let left = PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel) - 1;
                    job(left == 0);
                }
            })
            .expect("spawn PLE read-ahead thread");
        Mutex::new(tx)
    });
    if let Ok(tx) = queue.lock() {
        PENDING.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        if tx.send(job).is_err() {
            PENDING.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        }
    }
}
/// Decode keeps the checkpoint drive awake. A drive idle for about 100 ms drops to a low-power
/// state, and the next cold PLE row then waits for its wake-up (about 9 ms in the one-Spark
/// trace, against about 0.6 ms awake). Two read paths tick: the pooled plain step
/// (Reader::prefetch_ple_rows) and the CPU frontend (Reader::frontend_prompt: verify passes,
/// observed passes, CPU prompt tiles). While a tick is at most WARM_IDLE_MS old, one thread
/// reads one WARM_BLOCK of the PLE table directly (page cache bypassed) every WARM_PERIOD_MS.
/// With no tick for WARM_IDLE_MS it parks until the next one and logs that burst's reads.
/// Reads only: no byte the engine uses changes.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
const WARM_PERIOD_MS: u64 = 50;
const WARM_IDLE_MS: u64 = 250;
const WARM_BLOCK: u64 = 4096;
static WARM_TICK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static WARM_THREAD: OnceLock<std::thread::Thread> = OnceLock::new();
fn warm_now() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64 + 1
}
/// A decode read happened now: start the warm thread once for `file` (the span
/// `start..start + len` is the table it reads in) and mark the drive busy.
pub(crate) fn keep_warm(file: &File, start: u64, len: u64) {
    static STARTED: OnceLock<()> = OnceLock::new();
    WARM_TICK.store(warm_now(), std::sync::atomic::Ordering::Relaxed);
    STARTED.get_or_init(|| spawn_warm(file, start, len));
    if let Some(thread) = WARM_THREAD.get() { thread.unpark(); }
}
/// A warm read is due while the last decode read is at most WARM_IDLE_MS old.
fn warm_due(now: u64, tick: u64) -> bool {
    tick != 0 && now.saturating_sub(tick) <= WARM_IDLE_MS
}
/// The k-th warm read's aligned block inside `start..start + len`, spread over the span.
fn warm_offset(k: u64, start: u64, len: u64) -> Option<u64> {
    let first = start.div_ceil(WARM_BLOCK) * WARM_BLOCK;
    let blocks = start.saturating_add(len).saturating_sub(first) / WARM_BLOCK;
    (blocks > 0).then(|| first + k.wrapping_mul(2_654_435_761) % blocks * WARM_BLOCK)
}
#[cfg(target_os = "linux")]
fn spawn_warm(file: &File, start: u64, len: u64) {
    use std::os::unix::fs::OpenOptionsExt;
    #[cfg(target_arch = "aarch64")]
    const O_DIRECT: i32 = 0o200000;
    #[cfg(not(target_arch = "aarch64"))]
    const O_DIRECT: i32 = 0o40000;
    // A second open file description, so the shared descriptor's buffered preads stay buffered.
    let direct = match std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT)
        .open(format!("/proc/self/fd/{}", file.as_raw_fd())) {
        Ok(direct) => direct,
        Err(error) => { eprintln!("flash_next ple_warm off: direct open failed: {error}"); return; }
    };
    let spawned = std::thread::Builder::new().name("ple-warm".into()).spawn(move || {
        let mut buf = vec![0u8; 2 * WARM_BLOCK as usize];
        let at = buf.as_ptr().align_offset(WARM_BLOCK as usize);
        if at >= WARM_BLOCK as usize { eprintln!("flash_next ple_warm off: no aligned buffer"); return; }
        let (mut reads, mut burst, mut since, mut failed) = (0u64, 0u64, 0u64, false);
        loop {
            if !warm_due(warm_now(), WARM_TICK.load(std::sync::atomic::Ordering::Relaxed)) {
                if burst > 0 {
                    eprintln!("flash_next ple_warm reads={reads} burst={burst} ms={}", warm_now().saturating_sub(since));
                    burst = 0;
                }
                std::thread::park(); // no decode: no reads until keep_warm unparks
                continue;
            }
            if burst == 0 { since = warm_now(); }
            std::thread::sleep(std::time::Duration::from_millis(WARM_PERIOD_MS));
            let Some(offset) = warm_offset(reads + 1, start, len) else { return };
            match direct.read_at(&mut buf[at..at + WARM_BLOCK as usize], offset) {
                Ok(_) => { reads += 1; burst += 1; }
                Err(error) if !failed => { failed = true; eprintln!("flash_next ple_warm read failed: {error}"); }
                Err(_) => {}
            }
        }
    });
    match spawned {
        Ok(handle) => { let _ = WARM_THREAD.set(handle.thread().clone()); eprintln!("flash_next ple_warm on"); }
        Err(error) => eprintln!("flash_next ple_warm off: no thread: {error}"),
    }
}
#[cfg(not(target_os = "linux"))]
fn spawn_warm(_: &File, _: u64, _: u64) {}
#[cfg(test)]
#[test]
fn warm_reads_only_while_decode_ticks_and_stay_inside_the_table() {
    assert!(!warm_due(1000, 0));
    assert!(warm_due(1000, 1000) && warm_due(1250, 1000));
    assert!(!warm_due(1251, 1000));
    assert!(warm_due(900, 1000)); // a tick newer than the check
    assert_eq!(warm_offset(1, 100, 4000), None); // no whole aligned block
    for (start, len) in [(0u64, 4096u64), (100, 1 << 20), (5000, 3 << 30)] {
        for k in 0..1000 {
            let o = warm_offset(k, start, len).unwrap();
            assert!(o.is_multiple_of(WARM_BLOCK) && o >= start && o + WARM_BLOCK <= start + len);
        }
    }
    let seen: std::collections::BTreeSet<_> = (0..64).map(|k| warm_offset(k, 0, 1 << 30).unwrap()).collect();
    assert!(seen.len() > 60);
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
    fn table(name: &str, bytes: usize, seed: u64) -> Arc<File> {
        let path = std::env::temp_dir().join(format!("vade-read-plan-{}-{name}", std::process::id()));
        let mut rng = Rng(seed);
        let data: Vec<u8> = (0..bytes).map(|_| rng.next() as u8).collect();
        File::create(&path).unwrap().write_all(&data).unwrap();
        let file = File::open(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        Arc::new(file)
    }
    /// Two "tensors" in one file (80 B rows then 10 B rows) plus a second file
    /// of 5120 B rows; random rows with repeats, adjacent and overlapping
    /// neighbours, the plan output must equal one pread per request.
    #[test]
    fn plan_matches_per_row_preads_at_every_pool_size() {
        let files = vec![table("ple", 80 * 4096 + 10 * 4096, 7), table("embed", 5120 * 64, 9)];
        let scales = 80 * 4096u64;
        for seed in 1..=6u64 {
            let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut requests = Vec::new();
            let mut dest = 0;
            let mut push = |file: u32, at: u64, len: u32, requests: &mut Vec<Request>| {
                requests.push(Request { file, at, len, dest });
                dest += len as usize;
            };
            for _ in 0..3000 {
                let row = match rng.next() % 4 {
                    0 => 5,                     // a hot repeated row
                    1 => rng.next() % 64,       // dense: adjacent neighbours
                    _ => rng.next() % 4096,     // sparse
                };
                push(0, row * 80, 80, &mut requests);
                push(0, scales + row * 10, 10, &mut requests);
                if rng.next().is_multiple_of(8) {
                    // Overlapping, unaligned piece straddling two rows.
                    push(0, row * 80 + 40, 80, &mut requests);
                }
                if rng.next().is_multiple_of(3) {
                    push(1, (rng.next() % 64) * 5120, 5120, &mut requests);
                }
            }
            let mut want = vec![0u8; dest];
            for r in &requests {
                files[r.file as usize]
                    .read_exact_at(&mut want[r.dest..r.dest + r.len as usize], r.at)
                    .unwrap();
            }
            let plan = Plan::new(requests.clone());
            assert!(plan.extents() < requests.len());
            for readers in [1, 4, 16] {
                let mut got = vec![0xa5u8; dest];
                plan.read(&files, readers, &mut got).unwrap();
                assert!(got == want, "seed {seed} readers {readers}");
            }
            // Hints (whole plan, then a small budget) never change bytes.
            plan.hint(&files, usize::MAX);
            plan.hint(&files, 4096);
            let mut got = vec![0u8; dest];
            plan.read(&files, 16, &mut got).unwrap();
            assert!(got == want, "seed {seed} after hints");
        }
    }
    #[test]
    fn plan_merges_touching_and_duplicate_requests_only() {
        let r = |at, len, dest| Request { file: 0, at, len, dest };
        let plan = Plan::new(vec![r(100, 10, 0), r(90, 10, 10), r(100, 10, 20), r(200, 5, 30)]);
        assert_eq!(plan.extents(), 2);
        assert_eq!(plan.extent_bytes(), 25);
        let other = Plan::new(vec![r(0, 10, 0), Request { file: 1, at: 10, len: 10, dest: 10 }]);
        assert_eq!(other.extents(), 2);
    }
    #[test]
    fn plan_rejects_short_files_and_destinations() {
        let files = vec![table("short", 100, 3)];
        let plan = Plan::new(vec![Request { file: 0, at: 96, len: 8, dest: 0 }]);
        assert!(plan.read(&files, 1, &mut [0u8; 8]).is_err());
        let plan = Plan::new(vec![Request { file: 0, at: 0, len: 8, dest: 4 }]);
        assert!(plan.read(&files, 1, &mut [0u8; 8]).is_err());
        let plan = Plan::new(vec![Request { file: 1, at: 0, len: 8, dest: 0 }]);
        assert!(plan.read(&files, 1, &mut [0u8; 8]).is_err());
    }
    #[test]
    fn background_jobs_run_in_order() {
        let (tx, rx) = mpsc::channel();
        for i in 0..8 {
            let tx = tx.clone();
            background(Box::new(move |_| tx.send(i).unwrap()));
        }
        let got: Vec<i32> = (0..8).map(|_| rx.recv().unwrap()).collect();
        assert_eq!(got, (0..8).collect::<Vec<_>>());
    }
}
