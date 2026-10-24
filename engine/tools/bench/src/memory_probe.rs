//! Optional Linux CUDA allocation probe, compiled separately with rustc.
//! It forwards unchanged driver calls and records successful allocations only.
//! Enable for the benchmark server process with LD_PRELOAD, never system-wide.
use std::{collections::BTreeMap, ffi::{c_char, c_void}, sync::{Mutex, OnceLock}};

#[link(name = "dl")]
unsafe extern "C" { fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void; }
type Alloc = unsafe extern "C" fn(*mut u64, usize) -> i32;
type Managed = unsafe extern "C" fn(*mut u64, usize, u32) -> i32;
type Free = unsafe extern "C" fn(u64) -> i32;
struct State {
    allocations: BTreeMap<u64, usize>,
    current: usize,
    peak: usize,
    kv: usize,
    kv_peak: usize,
    other_peak: usize,
    weights: usize,
    kv_allocation: usize,
    kv_arena_token_bytes: usize,
    kv_max_tokens: usize,
    output: Option<String>,
}
fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| {
        let number = |name| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        Mutex::new(State { allocations: BTreeMap::new(), current: 0, peak: 0, kv: 0, kv_peak: 0, other_peak: 0,
            weights: number("VADE_BENCH_WEIGHTS_BYTES"), kv_allocation: number("VADE_BENCH_KV_ALLOCATION_BYTES"),
            kv_arena_token_bytes: number("VADE_BENCH_KV_ARENA_TOKEN_BYTES"), kv_max_tokens: number("VADE_BENCH_KV_MAX_TOKENS"),
            output: std::env::var("VADE_BENCH_MEMORY_FILE").ok() })
    })
}
fn is_kv(s: &State, bytes: usize) -> bool {
    if s.kv_arena_token_bytes == 0 { return bytes == s.kv_allocation; }
    let tokens = bytes / s.kv_arena_token_bytes;
    bytes % s.kv_arena_token_bytes == 0 && tokens > 0 && tokens <= s.kv_max_tokens
        && (tokens.is_power_of_two() || tokens == s.kv_max_tokens)
}
fn record(address: u64, allocation: Option<usize>) {
    let Ok(mut s) = state().lock() else { return; };
    if let Some(bytes) = allocation {
        s.allocations.insert(address, bytes);
        s.current += bytes;
        if is_kv(&s, bytes) { s.kv += bytes; }
    } else if let Some(bytes) = s.allocations.remove(&address) {
        s.current -= bytes;
        if is_kv(&s, bytes) { s.kv -= bytes; }
    }
    s.peak = s.peak.max(s.current);
    s.kv_peak = s.kv_peak.max(s.kv);
    s.other_peak = s.other_peak.max(s.current.saturating_sub(s.kv).saturating_sub(s.weights));
    if let Some(path) = &s.output {
        let tmp = format!("{path}.tmp");
        let json = format!("{{\"weights_bytes\":{},\"kv_peak_bytes\":{},\"scratch_peak_bytes\":{},\"device_peak_bytes\":{},\"device_current_bytes\":{},\"live_allocations\":{},\"kv_allocation_bytes\":{},\"kv_arena_token_bytes\":{},\"kv_max_tokens\":{},\"kv_current_bytes\":{},\"scope\":\"server-lifetime requested CUDA allocation peaks, excludes driver overhead; weights supplied from model ledger; KV identified by configured allocation size or growing power-of-two/capped arena sizes; scratch includes model auxiliary allocations; component peaks need not coincide; pinned loader staging is in host RSS\"}}\n", s.weights, s.kv_peak, s.other_peak, s.peak, s.current, s.allocations.len(), s.kv_allocation, s.kv_arena_token_bytes, s.kv_max_tokens, s.kv);
        if std::fs::write(&tmp, json).is_ok() { let _ = std::fs::rename(tmp, path); }
    }
}
// RTLD_NEXT resolves the real driver after this process-local preload library.
unsafe fn symbol(name: &[u8]) -> *mut c_void {
    unsafe { dlsym((-1isize) as *mut c_void, name.as_ptr().cast()) }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAlloc_v2(address: *mut u64, bytes: usize) -> i32 {
    static CALL: OnceLock<Option<Alloc>> = OnceLock::new();
    let Some(call) = CALL.get_or_init(|| {
        let p = unsafe { symbol(b"cuMemAlloc_v2\0") };
        if p.is_null() { None } else { Some(unsafe { std::mem::transmute::<*mut c_void, Alloc>(p) }) }
    }) else { return 999; };
    let code = unsafe { call(address, bytes) };
    if code == 0 { record(unsafe { *address }, Some(bytes)); }
    code
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemAllocManaged(address: *mut u64, bytes: usize, flags: u32) -> i32 {
    static CALL: OnceLock<Option<Managed>> = OnceLock::new();
    let Some(call) = CALL.get_or_init(|| {
        let p = unsafe { symbol(b"cuMemAllocManaged\0") };
        if p.is_null() { None } else { Some(unsafe { std::mem::transmute::<*mut c_void, Managed>(p) }) }
    }) else { return 999; };
    let code = unsafe { call(address, bytes, flags) };
    if code == 0 { record(unsafe { *address }, Some(bytes)); }
    code
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuMemFree_v2(address: u64) -> i32 {
    static CALL: OnceLock<Option<Free>> = OnceLock::new();
    let Some(call) = CALL.get_or_init(|| {
        let p = unsafe { symbol(b"cuMemFree_v2\0") };
        if p.is_null() { None } else { Some(unsafe { std::mem::transmute::<*mut c_void, Free>(p) }) }
    }) else { return 999; };
    let code = unsafe { call(address) };
    if code == 0 { record(address, None); }
    code
}
