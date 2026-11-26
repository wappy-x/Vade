use std::time::Instant;

type Result<T> = std::result::Result<T, String>;

/// A rank's transport is kept separate from GPU ownership. Local execution
/// runs collectives concurrently, after each GPU readback has synchronized.
pub trait Exchange: Send {
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<()>;
    fn gather_f32(&mut self, values: &mut [f32], offsets: &[usize]) -> Result<()>;
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()>;
    fn profile_stats(&self) -> Option<ExchangeStats> { None }
    fn collective_profiling(&self) -> bool { false }
    fn profile_flash(&mut self, _ns: u64) {}
    fn profile_point(&mut self, _kind: usize, _ns: u64, _ok: bool) {}
    fn profile_begin_step(&mut self, _phase: &'static str, _rows: usize) {}
    fn profile_end_step(&mut self, _ok: bool, _device_bytes: usize) {}
    /// A distributed rank's RDMA mailbox memory (base, bytes), which the
    /// session page-locks so the GPU fills and drains the slots directly.
    fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> { None }
    /// Complete every returned producer/consumer event, including on error.
    /// Pieces keep wire order; the next producer precedes the current consumer.
    fn exchange_slots_pipeline<T>(&mut self, _sizes: &[(usize, usize)],
        _fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        _receive: impl FnMut(usize, &[u8]) -> Result<T>,
        _wait: impl FnMut(T) -> Result<()>) -> Result<()> { Err("transport has no mailbox pipeline".into()) }
    /// Rank-major pieces; callbacks address a rail's byte offset within its packet.
    fn gather_slots_pipeline<T>(&mut self, _offsets: &[Vec<usize>],
        _fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        _receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        _wait: impl FnMut(T) -> Result<()>) -> Result<()> { Err("transport has no gather mailbox pipeline".into()) }
}
#[derive(Clone, Copy)]
pub struct ExchangeStats {
    pub elapsed_ns: u128,
    pub sent: u64,
    pub received: u64,
    pub exchanges: u64,
}
/// All Flash exchange points share the same clock boundary, including row and
/// norm gathers, head candidates, host reductions and GPU mailbox pipelines.
/// The disabled branch never takes a timestamp or reacquires the transport lock.
#[repr(transparent)]
pub struct Profiled<E> { inner: E }
impl<E: Exchange> Profiled<E> {
    pub fn new(inner: E) -> Self { Self { inner } }
    fn exchange<T>(&mut self, work: impl FnOnce(&mut E) -> Result<T>) -> Result<T> {
        if !self.inner.collective_profiling() { return work(&mut self.inner); }
        let start = Instant::now();
        let result = work(&mut self.inner);
        let ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        self.inner.profile_flash(ns);
        result
    }
}
impl<E: Exchange> Exchange for Profiled<E> {
    fn gather_bytes(&mut self, values: &mut [u8], offsets: &[usize]) -> Result<()> {
        self.exchange(|rank| rank.gather_bytes(values, offsets))
    }
    fn gather_f32(&mut self, values: &mut [f32], offsets: &[usize]) -> Result<()> {
        self.exchange(|rank| rank.gather_f32(values, offsets))
    }
    fn reduce(&mut self, values: &mut [f32], scratch: &mut [f32]) -> Result<()> {
        self.exchange(|rank| rank.reduce(values, scratch))
    }
    fn exchange_slots_pipeline<T>(&mut self, sizes: &[(usize, usize)],
        fill: impl FnMut(usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.exchange(|rank| rank.exchange_slots_pipeline(sizes, fill, receive, wait))
    }
    fn profile_point(&mut self, kind: usize, ns: u64, ok: bool) { self.inner.profile_point(kind, ns, ok); }
    fn gather_slots_pipeline<T>(&mut self, offsets: &[Vec<usize>],
        fill: impl FnMut(usize, usize, &mut [u8]) -> Result<T>,
        receive: impl FnMut(usize, usize, usize, &[u8]) -> Result<T>,
        wait: impl FnMut(T) -> Result<()>) -> Result<()> {
        self.exchange(|rank| rank.gather_slots_pipeline(offsets, fill, receive, wait))
    }
    fn profile_stats(&self) -> Option<ExchangeStats> { self.inner.profile_stats() }
    fn collective_profiling(&self) -> bool { self.inner.collective_profiling() }
    fn mailbox_memory(&mut self) -> Option<(*mut u8, usize)> { self.inner.mailbox_memory() }
    fn profile_begin_step(&mut self, phase: &'static str, rows: usize) {
        if self.inner.collective_profiling() { self.inner.profile_begin_step(phase, rows); }
    }
    fn profile_end_step(&mut self, ok: bool, device_bytes: usize) {
        if self.inner.collective_profiling() { self.inner.profile_end_step(ok, device_bytes); }
    }
}
