//! Flash admission and elastic prefix-cache sizing; all bytes are per rank.

/// Product Flash context; an explicit launch context takes precedence.
pub const DEFAULT_CONTEXT_TOKENS: usize = 131_072;

/// Rank zero must fund optional copies before issuing any distributed command.
pub fn save_if_funded<T>(bytes: usize, retained: usize, cache: usize, headroom: usize,
    save: impl FnOnce() -> Result<T, String>) -> Result<Option<T>, String> {
    if bytes > cache.saturating_sub(retained) || bytes > headroom { return Ok(None); }
    save().map(Some)
}

/// Credit only private allocations already owned by admitted sessions.
pub fn snapshot_headroom(free: usize, reserved: usize, allocated: usize) -> usize {
    free.saturating_sub(reserved.saturating_sub(allocated))
}
/// Both votes are collective: partial copies drop on every rank after a miss.
pub fn save_collectively<T>(funded: bool, mut vote: impl FnMut(bool) -> Result<bool, String>,
    save: impl FnOnce() -> Result<T, String>) -> Result<Option<T>, String> {
    if !vote(funded)? { return Ok(None); }
    let saved = save().ok();
    if vote(saved.is_some())? { Ok(saved) } else { Ok(None) }
}

#[test] fn unfunded_save_never_dispatches_or_allocates() {
    for (bytes, retained, cache, headroom) in [(11, 90, 100, 20), (10, 0, 100, 9), (1, 0, 0, 100), (1, usize::MAX, 10, 100)] { assert_eq!(save_if_funded::<()>(bytes, retained, cache, headroom, || panic!("unfunded SAVE")), Ok(None)); }
    assert_eq!(save_if_funded(10, 90, 100, 10, || Ok(7)), Ok(Some(7)));
}

/// Post-load capacity includes private session graphs. Shared buffers and
/// graphs are funded first; the explicit ceiling covers all owned allocations.
/// Sessions stop TRANSIENT_RESERVE short of it for per-tick transients.
pub fn capacity_bytes(free: usize, allocated: usize, shared: usize, ceiling: Option<usize>) -> (usize, usize) {
    let safety = (free / 10).max(2 << 30);
    let remaining = ceiling.unwrap_or(usize::MAX).saturating_sub(allocated);
    let available = spark_serve::budget::after_transient_reserve(remaining.min(free.saturating_sub(safety)).saturating_sub(shared));
    (safety, available)
}

/// Explain reclaimable Linux pages that CUDA still cannot allocate.
pub fn file_cache_hint(unified_available: usize, cuda_free: usize) -> String {
    let cached = unified_available.saturating_sub(cuda_free);
    if cached <= 4 << 30 { return String::new(); }
    format!(" Reclaimable cache estimate: {}; this is not a guarantee that CUDA can allocate it.", spark_gpu_probe::admission::bytes(cached as u128))
}

#[derive(Clone, Copy, Default)]
pub struct Request {
    /// Explicit concurrency funds this many full contexts and caps live slots.
    /// Otherwise use all safe rank capacity and retain elastic request slots.
    pub chats: Option<usize>,
    /// Existing override: retained prefix-cache ceiling (zero disables cache).
    pub cache_bytes: Option<usize>,
}
#[derive(Debug)]
pub struct Plan {
    pub pool_bytes: usize,
    pub cache_bytes: usize,
    pub reservation: usize,
    /// Private publication headroom per full-context chat.
    pub publication_bytes: usize,
    pub minimum: usize,
    pub slots: usize,
}
impl Request {
    pub fn from_env() -> Result<Self, String> {
        let read = |name| match std::env::var(name) {
            Ok(value) => value.parse::<usize>().map(Some).map_err(|_| format!("invalid {name}")),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(error.to_string()),
        };
        let chats = read("FLASH_SERVE_MAX_ACTIVE")?;
        if chats.is_some_and(|n| !(1..=32).contains(&n)) {
            return Err("FLASH_SERVE_MAX_ACTIVE must be in 1..=32".into());
        }
        Ok(Self { chats, cache_bytes: read("FLASH_SERVE_KV_BYTES")? })
    }
    pub fn plan(self, context: usize, available: usize,
        reserve: impl Fn(usize, bool) -> Result<usize, String>,
        publication: impl Fn(usize) -> Result<usize, String>) -> Result<Plan, String> {
        let caching = self.cache_bytes != Some(0);
        let chats = self.chats.unwrap_or(1);
        let reservation = reserve(context, caching)?;
        let publication_bytes = if caching { publication(context)? } else { 0 };
        let full = reservation.checked_add(publication_bytes).ok_or("Flash session byte count overflow")?;
        let minimum = reserve(context.min(512), caching)?;
        if minimum == 0 || reservation < minimum { return Err("invalid Flash session reservation".into()); }
        let required = (full as u128 * chats as u128)
            .max(minimum as u128 + self.cache_bytes.unwrap_or(1) as u128);
        if required > available as u128 {
            // Keep the same worst-rank reservation and shared workspace bound
            // when suggesting a smaller context, so the suggestion is safe.
            let fits = |tokens: usize| -> Result<usize, String> {
                let cache_floor = reserve(tokens.min(512), caching)? as u128 + self.cache_bytes.unwrap_or(1) as u128;
                let full = reserve(tokens, caching)?.checked_add(if caching { publication(tokens)? } else { 0 })
                    .ok_or("Flash session byte count overflow")?;
                Ok(if cache_floor > available as u128 { 0 } else { available / full.max(1) })
            };
            let (mut low, mut high) = (0, context / 64);
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if fits(mid * 64)? >= chats { low = mid; } else { high = mid - 1; }
            }
            let cache = self.cache_bytes.map(|n| format!("{n} bytes")).unwrap_or_else(|| "automatic".into());
            return Err(spark_gpu_probe::admission::refusal(required, available,
                &format!("weights/rank already loaded; reservation/session {} x {chats} sessions (graphs/publication included); CONTEXT={context}; OUTPUT included in CONTEXT; retained cache {cache}; fits {context} x {} sessions or {} context x {chats} sessions; 32768 x {}, 131072 x {}", spark_gpu_probe::admission::bytes(full as u128), fits(context)?, low * 64, fits(32768)?, fits(131072)?),
                "the lowest rank's Linux MemAvailable (page cache counts as free), capped by CUDA total and remaining SEQPAR_MEMORY_BYTES; less safety reserve (10%, at least 2 GiB), weights, shared workspaces and shared graphs", spark_gpu_probe::admission::FLASH));
        }
        // An unspecified concurrency consumes the negotiated safe capacity.
        // Explicit chats fund their declared contexts; retained cache remains
        // an exact, separately reclaimable ceiling within either pool policy.
        let pool_bytes = if self.chats.is_none() { available } else { required as usize };
        let cache_bytes = match self.cache_bytes {
            Some(bytes) => bytes,
            // Retain a useful share of the negotiated pool, with enough room
            // for one complete configured context even when MTP is enabled.
            // Snapshot identities receive active-reservation funding credit;
            // cache token/logit vectors and index metadata remain private.
            // Include their publication headroom in both pool and retention.
            // Idle cache borrows these bytes and yields to live reservations,
            // so short-request slots need no permanent cache carve-out here.
            None => (pool_bytes / 2).max(full),
        };
        let slots = self.chats.unwrap_or(pool_bytes / minimum);
        Ok(Plan { pool_bytes, cache_bytes, reservation, publication_bytes, minimum, slots })
    }
}

#[test] fn snapshot_growth_protects_unopened_chats_and_each_rank() {
    for (free, reserved, allocated, expected) in [(100, 80, 30, 50), (100, 80, 0, 20), (40, 80, 30, 0), (100, 20, 30, 100), (0, usize::MAX, 1, 0)] { assert_eq!(snapshot_headroom(free, reserved, allocated), expected); }
    for free in [&[100, 40][..], &[40, 100][..], &[100, 100, 100, 40][..]] { let funded = free.iter().all(|&n| snapshot_headroom(n, 80, 30) >= 1); assert_eq!(save_collectively::<()>(true, |_| Ok(funded), || panic!("constrained rank allocated")), Ok(None)); } }
#[test] fn snapshot_allocation_failure_rolls_back_every_rank() {
    for failed in [0, 1, 3] { for rank in 0..4 { let mut phase=0; let drops=std::cell::Cell::new(0); struct Copy<'a>(&'a std::cell::Cell<usize>); impl Drop for Copy<'_> { fn drop(&mut self) { self.0.set(self.0.get()+1); } } let result=save_collectively(true, |_| { phase+=1; Ok(phase==1) }, || if rank==failed { Err("allocation failed".into()) } else { Ok(Copy(&drops)) }); assert!(result.unwrap().is_none()); assert_eq!(drops.get(),usize::from(rank!=failed)); assert_eq!(phase,2); } }
    assert_eq!(save_collectively(true, |ok| Ok(ok), || Err::<(), _>("allocation failed".into())), Ok(None)); assert_eq!(save_collectively(true, |ok| Ok(ok), || Ok(7)), Ok(Some(7))); assert!(save_collectively(true, |_| Err("transport".into()), || Ok(7)).is_err()); }
#[test] fn page_cache_funds_the_pool_and_low_mem_available_refuses() {
    use spark_gpu_probe::admission::{record_mem_available, unified_available};
    const GIB: usize = 1 << 30;
    let meminfo = |free: usize, available: usize, cached: usize| format!("MemTotal: {} kB\nMemFree: {} kB\nMemAvailable: {} kB\nCached: {} kB\n", 121 * GIB / 1024, free * GIB / 1024, available * GIB / 1024, cached * GIB / 1024);
    let plan = |info: &str| { let free = unified_available(info, 121 * GIB + GIB / 2).unwrap(); record_mem_available(free);
        let (_, available) = capacity_bytes(free, 78 * GIB, 2 * GIB, None);
        Request { chats: Some(8), cache_bytes: None }.plan(32768, available, |tokens, _| Ok(tokens * 48 * 1024), |_| Ok(0)) };
    // After the weights load: little MemFree, the checkpoint in page cache, 30 GiB MemAvailable.
    assert_eq!(plan(&meminfo(1, 30, 26)).unwrap().reservation, 32768 * 48 * 1024);
    let error = plan(&meminfo(1, 4, 1)).unwrap_err();
    for text in ["Requested size does not fit.", "Memory: MemAvailable 4.00 GiB (4294967296 bytes); required 12.00 GiB", "ceiling 0.00 GiB (0 bytes)", "Fix: "] { assert!(error.contains(text), "{error}"); } }
