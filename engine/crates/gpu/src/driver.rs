//! Minimal, manually declared NVIDIA CUDA Driver API ABI (not CUDA Runtime API).
//! Linux only; the runner owns a dedicated CUDA context on one host thread.
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::marker::PhantomData;
use std::ptr;
use std::rc::Rc;
use std::time::Instant;
use crate::launch_blocks;
use crate::teardown::{Progress, WaitKind};
#[path = "decode_profile.rs"]
mod decode_profile;
#[path = "jit_receipts.rs"]
mod jit_receipts;

type Handle = *mut c_void;
type DevicePtr = u64;
/// KV storage ABI shared by cache writes and attention kernels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum KvFormat { #[default] Fp32 = 0, Bf16 = 1, ScaledI16 = 4 }
impl KvFormat {
    /// Scaled I16 groups eight contiguous 128-value heads followed by their eight scales.
    pub fn storage_len(self, elements: usize) -> usize {
        match self { Self::Fp32 => elements, Self::ScaledI16 => elements.div_ceil(1024) * 520, Self::Bf16 => elements.div_ceil(2) }
    }
    pub fn element_capacity(self, words: usize) -> usize {
        match self { Self::Fp32 => words, Self::ScaledI16 => words / 520 * 1024, Self::Bf16 => words * 2 }
    }
}
impl From<bool> for KvFormat {
    fn from(bf16: bool) -> Self { if bf16 { Self::Bf16 } else { Self::Fp32 } }
}
#[path = "segments.rs"]
mod segments;
pub use crate::fp8_abi;
#[path = "fp4_abi.rs"]
pub mod fp4_abi;

/// CUgraphExecUpdateResultInfo.
#[repr(C)]
struct GraphExecUpdateInfo { result: c_int, error_node: Handle, error_from_node: Handle }

#[link(name = "cuda")]
extern "C" {
    fn cuInit(flags: c_uint) -> c_int;
    fn cuDriverGetVersion(version: *mut c_int) -> c_int;
    fn cuDeviceGet(device: *mut c_int, ordinal: c_int) -> c_int;
    fn cuDeviceGetName(name: *mut c_char, length: c_int, device: c_int) -> c_int;
    fn cuDeviceGetAttribute(value: *mut c_int, attribute: c_int, device: c_int) -> c_int;
    fn cuCtxCreate_v2(context: *mut Handle, flags: c_uint, device: c_int) -> c_int;
    fn cuCtxDestroy_v2(context: Handle) -> c_int;
    fn cuCtxSynchronize() -> c_int;
    fn cuCtxSetCurrent(context: Handle) -> c_int;
    fn cuMemAlloc_v2(address: *mut DevicePtr, bytes: usize) -> c_int;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> c_int;
    fn cuMemAllocManaged(address: *mut DevicePtr, bytes: usize, flags: c_uint) -> c_int;
    fn cuMemFree_v2(address: DevicePtr) -> c_int;
    fn cuMemHostAlloc(address: *mut Handle, bytes: usize, flags: c_uint) -> c_int;
    fn cuMemHostGetDevicePointer_v2(device: *mut DevicePtr, host: Handle, flags: c_uint) -> c_int;
    fn cuMemFreeHost(address: Handle) -> c_int;
    fn cuMemHostRegister_v2(address: *mut c_void, bytes: usize, flags: c_uint) -> c_int;
    fn cuMemHostUnregister(address: *mut c_void) -> c_int;
    fn cuMemcpyDtoH_v2(destination: *mut c_void, source: DevicePtr, bytes: usize) -> c_int;
    fn cuMemcpyHtoDAsync_v2(destination: DevicePtr, source: *const c_void, bytes: usize, stream: Handle) -> c_int;
    fn cuMemcpyDtoHAsync_v2(destination: *mut c_void, source: DevicePtr, bytes: usize, stream: Handle) -> c_int;
    fn cuMemcpyDtoDAsync_v2(destination: DevicePtr, source: DevicePtr, bytes: usize, stream: Handle) -> c_int;
    fn cuMemsetD8Async(destination: DevicePtr, value: u8, bytes: usize, stream: Handle) -> c_int;
    fn cuStreamCreate(stream: *mut Handle, flags: c_uint) -> c_int;
    fn cuStreamSynchronize(stream: Handle) -> c_int;
    fn cuStreamQuery(stream: Handle) -> c_int;
    fn cuStreamWaitEvent(stream: Handle, event: Handle, flags: c_uint) -> c_int;
    fn cuStreamDestroy_v2(stream: Handle) -> c_int;
    fn cuStreamBeginCapture_v2(stream: Handle, mode: c_int) -> c_int;
    fn cuStreamEndCapture(stream: Handle, graph: *mut Handle) -> c_int;
    fn cuGraphGetNodes(graph: Handle, nodes: *mut Handle, count: *mut usize) -> c_int;
    fn cuGraphInstantiate_v2(exec: *mut Handle, graph: Handle, error_node: *mut Handle, log: *mut c_char, bytes: usize) -> c_int;
    fn cuGraphLaunch(exec: Handle, stream: Handle) -> c_int;
    fn cuGraphExecUpdate_v2(exec: Handle, graph: Handle, info: *mut GraphExecUpdateInfo) -> c_int;
    fn cuGraphDestroy(graph: Handle) -> c_int;
    fn cuGraphExecDestroy(exec: Handle) -> c_int;
    fn cuEventCreate(event: *mut Handle, flags: c_uint) -> c_int;
    fn cuEventRecord(event: Handle, stream: Handle) -> c_int;
    fn cuEventSynchronize(event: Handle) -> c_int;
    fn cuEventQuery(event: Handle) -> c_int;
    fn cuEventElapsedTime(ms: *mut f32, start: Handle, end: Handle) -> c_int;
    fn cuEventDestroy_v2(event: Handle) -> c_int;
    fn cuModuleLoadData(module: *mut Handle, image: *const c_void) -> c_int;
    fn cuModuleLoadDataEx(module: *mut Handle, image: *const c_void, count: c_uint,
        options: *mut c_int, values: *mut *mut c_void) -> c_int;
    fn cuModuleUnload(module: Handle) -> c_int;
    fn cuModuleGetFunction(function: *mut Handle, module: Handle, name: *const c_char) -> c_int;
    fn cuFuncSetAttribute(function: Handle, attribute: c_int, value: c_int) -> c_int;
    fn cuFuncGetAttribute(value: *mut c_int, attribute: c_int, function: Handle) -> c_int;
    fn cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks: *mut c_int, function: Handle,
        block_size: c_int, dynamic_shared: usize) -> c_int;
    fn cuLaunchKernel(function: Handle, gx: c_uint, gy: c_uint, gz: c_uint,
        bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, stream: Handle,
        params: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
    fn cuLaunchKernelEx(config: *const LaunchConfig, function: Handle, params: *mut *mut c_void,
        extra: *mut *mut c_void) -> c_int;
    fn cuGetErrorName(error: c_int, name: *mut *const c_char) -> c_int;
    fn cuGetErrorString(error: c_int, message: *mut *const c_char) -> c_int;
}

#[repr(C)]
struct LaunchAttribute { id: c_uint, pad: c_uint, value: [c_uint; 16] }
#[repr(C)]
struct LaunchConfig { grid: [c_uint; 3], block: [c_uint; 3], shared: c_uint, stream: Handle,
    attrs: *const LaunchAttribute, count: c_uint }
/// cuLaunchKernel; `programmatic` adds CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
/// (6) = 1 through cuLaunchKernelEx: the kernel may start in the previous kernel's tail
/// and must griddepcontrol.wait before reading that kernel's output (fp4_bw_down).
unsafe fn launch_kernel(programmatic: bool, function: Handle, grid: [c_uint; 3], threads: c_uint,
    shared: c_uint, stream: Handle, params: *mut *mut c_void) -> c_int {
    if !programmatic {
        return unsafe { cuLaunchKernel(function, grid[0], grid[1], grid[2], threads, 1, 1, shared, stream, params, ptr::null_mut()) };
    }
    let mut value = [0; 16];
    value[0] = 1;
    let attr = LaunchAttribute { id: 6, pad: 0, value };
    let config = LaunchConfig { grid, block: [threads, 1, 1], shared, stream, attrs: &attr, count: 1 };
    unsafe { cuLaunchKernelEx(&config, function, params, ptr::null_mut()) }
}

fn check(code: c_int, operation: &str) -> Result<(), String> {
    if code == 0 { return Ok(()); }
    if matches!(code, 700 | 702 | 710 | 714..=719) { crate::teardown::arm(); }
    unsafe {
        let mut name = ptr::null();
        let mut message = ptr::null();
        let _ = cuGetErrorName(code, &mut name);
        let _ = cuGetErrorString(code, &mut message);
        let text = |p: *const c_char| if p.is_null() { "unknown".into() }
            else { CStr::from_ptr(p).to_string_lossy().into_owned() };
        Err(format!("{operation}: CUDA {code} {}: {}", text(name), text(message)))
    }
}

fn cleanup(code: c_int, operation: &str) {
    if let Err(error) = check(code, operation) { eprintln!("cleanup error: {error}"); }
}

struct Context {
    raw: Handle,
    device: c_int,
    multiprocessors: usize,
    // The current-context API is thread-affine. Do not permit Send or Sync.
    _thread: PhantomData<Rc<()>>,
    cleanup_errors: RefCell<Vec<String>>,
    execution_error: RefCell<Option<String>>,
    allocated: Cell<usize>,
    peak_allocated: Cell<usize>,
    graph_storage: Cell<usize>,
    peak_graph_storage: Cell<usize>,
    peak_accounted: Cell<usize>,
    graph_budget: Cell<usize>,
    pinned_staging: Cell<usize>,
    peak_pinned_staging: Cell<usize>,
    budget: Cell<usize>,
    // MemAvailable and allocated bytes at the first availability query.
    memory_baseline: Cell<Option<(usize, usize)>>,
    pending: Cell<bool>,
    // A verify segment capture is open (VADE_MTP_GRAPH): frees wait in `deferred_frees`.
    capturing: Cell<bool>,
    // A scoped resident queue retains dropped allocations until a real stream drain.
    resident_queued: Cell<bool>,
    deferred_frees: RefCell<Vec<(DevicePtr, usize, Option<Handle>, Reuse)>>,
    deferred_total: Cell<usize>,
    kv_reserve: RefCell<segments::Reserve>,
    // VADE_VECTOR_POOL: freed device vectors (Backend::vector, vector_arena, vector_workspace)
    // keyed by exact byte length; empty until a server configures it.
    vector_pool: RefCell<segments::Reserve>,
    progress: Progress,
    decode_profile: Option<Box<RefCell<decode_profile::DecodeProfile>>>,
    jit_receipts: Option<Box<RefCell<jit_receipts::Receipts>>>,
}
impl Context {
    fn open(ordinal: i32) -> Result<Self, String> {
        Self::open_reported(ordinal, true)
    }
    fn open_reported(ordinal: i32, report: bool) -> Result<Self, String> {
        let decode_profile = decode_profile::DecodeProfile::configured()?;
        let jit_receipts = jit_receipts::Receipts::configured()?;
        let progress = Progress::new()?;
        let initialization = progress.wait(WaitKind::Initialize);
        unsafe {
            check(cuInit(0), "cuInit")?;
            let mut device = 0;
            check(cuDeviceGet(&mut device, ordinal), "cuDeviceGet")?;
            let mut name = [0 as c_char; 128];
            let mut version = 0;
            let (mut major, mut minor) = (0, 0);
            check(cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, device), "cuDeviceGetName")?;
            check(cuDriverGetVersion(&mut version), "cuDriverGetVersion")?;
            // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR.
            check(cuDeviceGetAttribute(&mut major, 75, device), "compute capability major")?;
            check(cuDeviceGetAttribute(&mut minor, 76, device), "compute capability minor")?;
            // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT; cached before graph capture.
            let mut multiprocessors = 0;
            check(cuDeviceGetAttribute(&mut multiprocessors, 16, device), "multiprocessor count")?;
            if multiprocessors <= 0 { return Err("CUDA device has no multiprocessors".into()); }
            let mut raw = ptr::null_mut();
            check(cuCtxCreate_v2(&mut raw, 0, device), "cuCtxCreate_v2")?;
            drop(initialization);
            let context = Self { raw, device, multiprocessors: multiprocessors as usize, _thread: PhantomData, cleanup_errors: RefCell::new(Vec::new()), execution_error: RefCell::new(None), allocated: Cell::new(0), peak_allocated: Cell::new(0), graph_storage: Cell::new(0), peak_graph_storage: Cell::new(0), peak_accounted: Cell::new(0), graph_budget: Cell::new(usize::MAX), pinned_staging: Cell::new(0), peak_pinned_staging: Cell::new(0), budget: Cell::new(usize::MAX), memory_baseline: Cell::new(None), pending: Cell::new(false), capturing: Cell::new(false), resident_queued: Cell::new(false), deferred_frees: RefCell::new(Vec::new()), deferred_total: Cell::new(0), kv_reserve: RefCell::new(segments::Reserve::default()), vector_pool: RefCell::new(segments::Reserve::vectors()), progress, decode_profile, jit_receipts };
            if report {
                println!("device={:?} ordinal={ordinal} driver_api={version} compute={major}.{minor} multiprocessors={multiprocessors}",
                    CStr::from_ptr(name.as_ptr()).to_string_lossy());
            }
            Ok(context)
        }
    }
    fn record_cleanup(&self, code: c_int, operation: &str) {
        if let Err(error) = check(code, operation) {
            crate::teardown::arm();
            self.cleanup_errors.borrow_mut().push(error);
        }
    }
    fn accounted_bytes(&self) -> usize { self.allocated.get().saturating_add(self.graph_storage.get()) }
    /// A container's MemAvailable may not charge CUDA allocations, so also
    /// subtract this context's allocations since the first query. Bare metal
    /// already counts them; the smaller value is then MemAvailable itself.
    fn visible_available(&self, available: usize) -> usize {
        let allocated = self.allocated.get();
        let (first, at) = self.memory_baseline.get().unwrap_or((available, allocated));
        self.memory_baseline.set(Some((first, at)));
        available.min(first.saturating_sub(allocated.saturating_sub(at)))
    }
    fn record_allocation(&self, bytes: usize) {
        self.allocated.set(bytes);
        self.peak_allocated.set(self.peak_allocated.get().max(bytes));
        self.peak_accounted.set(self.peak_accounted.get().max(self.accounted_bytes()));
    }
    fn check_execution(&self, code: c_int, operation: &str) -> Result<(), String> {
        let result = check(code, operation);
        // Sticky CUDA execution faults require this context to be retired.
        if matches!(code, 700 | 702 | 710 | 714..=719) {
            let mut failure = self.execution_error.borrow_mut();
            if failure.is_none() { *failure = result.as_ref().err().cloned(); }
        }
        result
    }
    fn finish(mut self) -> Result<(), String> {
        let _wait = self.progress.wait(WaitKind::Cleanup);
        self.activate()?;
        self.capturing.set(false);
        self.release_deferred();
        self.clear_reserves();
        let raw = std::mem::replace(&mut self.raw, ptr::null_mut());
        unsafe { self.record_cleanup(cuCtxDestroy_v2(raw), "cuCtxDestroy_v2"); }
        if let Some(error) = self.execution_error.borrow_mut().take() { self.cleanup_errors.borrow_mut().push(error); }
        let errors = self.cleanup_errors.borrow();
        if errors.is_empty() { Ok(()) } else { Err(errors.join("; ")) }
    }
    fn activate(&self) -> Result<(), String> { unsafe { check(cuCtxSetCurrent(self.raw), "cuCtxSetCurrent") } }
    fn receipt_launch(&self, function: Handle, threads: u32, shared: u32) -> Result<(), String> {
        if let Some(receipts) = &self.jit_receipts { receipts.borrow_mut().launched(function, threads, shared)?; }
        Ok(())
    }
    fn decode_timer(&self, kind: usize) -> Option<Instant> { self.decode_profile.as_ref().and_then(|p| p.borrow().timer(kind)) }
    fn decode_event(&self, kind: usize, metric: usize, timer: Option<Instant>) {
        if timer.is_some() { self.decode_profile.as_ref().unwrap().borrow_mut().event(kind, metric, timer); }
    }
    fn decode_enqueue(&self) { if let Some(p) = &self.decode_profile { p.borrow_mut().enqueue(); } }
    /// A drop inside a stream capture may neither drain nor free: either one invalidates the
    /// capture. The allocation is released after the next drain (release_deferred).
    fn defer_free(&self, address: DevicePtr, bytes: usize, host: Option<Handle>, reuse: Reuse) {
        if self.deferred_total.replace(self.deferred_total.get() + 1) == 0 && self.capturing.get() {
            eprintln!("flash_next mtp_graph deferred_free bytes={bytes} (first; backtrace follows)\n{}", std::backtrace::Backtrace::force_capture());
        }
        self.deferred_frees.borrow_mut().push((address, bytes, host, reuse));
    }
    fn release_deferred(&self) {
        if self.capturing.get() || self.deferred_frees.borrow().is_empty() { return; }
        if self.pending.get() {
            if let Err(error) = self.sync() { self.cleanup_errors.borrow_mut().push(error); return; }
        }
        let frees = std::mem::take(&mut *self.deferred_frees.borrow_mut());
        for (address, bytes, host, reuse) in frees { self.free(address, bytes, host, reuse); }
    }
    fn free(&self, address: DevicePtr, bytes: usize, host: Option<Handle>, reuse: Reuse) {
        let spares = match reuse { Reuse::Kv => Some(&self.kv_reserve), Reuse::Vector => Some(&self.vector_pool), Reuse::None => None };
        if self.cleanup_errors.borrow().is_empty() {
            if let Some(spares) = spares {
                let mut reserve = spares.borrow_mut();
                reserve.prepare_return(self);
                if reserve.keep(address, bytes, false) { return; }
            }
        }
        let (code, operation) = unsafe { if let Some(host) = host { (cuMemFreeHost(host), "cuMemFreeHost") } else { (cuMemFree_v2(address), "cuMemFree_v2") } };
        self.record_cleanup(code, operation);
        if code == 0 {
            self.allocated.set(self.allocated.get() - bytes);
            if let Some(spares) = spares { spares.borrow_mut().direct_freed(bytes); }
        }
    }
    /// Release kept spares, the vector pool's first, until `bytes` more fit the budget.
    fn make_room(&self, bytes: usize) -> Result<(), String> {
        self.vector_pool.borrow_mut().make_room(self, bytes)?;
        self.kv_reserve.borrow_mut().make_room(self, bytes)
    }
    fn clear_reserves(&self) {
        self.vector_pool.borrow_mut().clear(self);
        self.kv_reserve.borrow_mut().clear(self);
    }
    fn sync(&self) -> Result<(), String> {
        let _wait = self.progress.wait(WaitKind::Context);
        self.activate()?;
        unsafe { self.check_execution(cuCtxSynchronize(), "cuCtxSynchronize")?; }
        self.pending.set(false);
        Ok(())
    }
}

/// Query memory without loading kernels or allocating weights or staging buffers.
pub fn device_memory_info(ordinal: i32) -> Result<(usize, usize), String> {
    let context = Context::open_reported(ordinal, false)?;
    let (mut free, mut total) = (0, 0);
    let query = unsafe { check(cuMemGetInfo_v2(&mut free, &mut total), "query device memory") };
    let cleanup = context.finish();
    query?;
    cleanup?;
    Ok((free, total))
}
/// Memory a model may claim on GB10's unified DRAM: Linux MemAvailable, which
/// counts reclaimable file cache (e.g. checkpoint pages) as free, capped at CUDA
/// total memory. Not CUDA free: in a container that excludes host cache.
pub fn unified_available_bytes(total: usize) -> Result<usize, String> {
    let info = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    crate::admission::record_mem_available(crate::admission::meminfo_bytes(&info, "MemAvailable:")?);
    crate::admission::unified_available(&info, total)
}
/// GB10 pairs ten Cortex-X925 cores with ten slower Cortex-A725 cores. A thread
/// that sleeps between requests otherwise wakes on either cluster, and long
/// prompts then parse and tokenize up to 1.7x slower on the first-token path.
/// Narrows only the calling thread to the fastest cores it may already use.
pub fn prefer_fast_cores() -> Result<(), String> {
    unsafe extern "C" {
        fn sched_getaffinity(pid: i32, bytes: usize, mask: *mut u64) -> i32;
        fn sched_setaffinity(pid: i32, bytes: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    if unsafe { sched_getaffinity(0, std::mem::size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
        return Err(format!("thread affinity: {}", std::io::Error::last_os_error()));
    }
    let speed = |cpu: usize| std::fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/cpufreq/cpuinfo_max_freq"))
        .ok().and_then(|value| value.trim().parse::<u64>().ok());
    let allowed: Vec<_> = (0..mask.len()*64).filter(|&cpu| mask[cpu/64] >> (cpu%64) & 1 == 1)
        .filter_map(|cpu| speed(cpu).map(|khz| (cpu, khz))).collect();
    let Some(fastest) = allowed.iter().map(|&(_, khz)| khz).max() else { return Ok(()) };
    if allowed.iter().all(|&(_, khz)| khz == fastest) { return Ok(()); }
    let mut fast = [0u64; 16];
    for &(cpu, _) in allowed.iter().filter(|&&(_, khz)| khz == fastest) { fast[cpu/64] |= 1 << (cpu%64); }
    if unsafe { sched_setaffinity(0, std::mem::size_of_val(&fast), fast.as_ptr()) } != 0 {
        return Err(format!("thread affinity: {}", std::io::Error::last_os_error()));
    }
    Ok(())
}
impl Drop for Context {
    fn drop(&mut self) {
        let _wait = self.progress.wait(WaitKind::Cleanup);
        if !self.raw.is_null() {
            self.capturing.set(false);
            if self.activate().is_ok() { self.release_deferred(); self.clear_reserves(); }
            unsafe { cleanup(cuCtxDestroy_v2(self.raw), "cuCtxDestroy_v2"); }
        }
    }
}

// Only these private types may be copied to/from GPU memory. They admit every
// bit pattern and contain no padding, pointers, or lifetimes.
trait DeviceElement: Copy {}
impl DeviceElement for f32 {}
impl DeviceElement for u16 {}
impl DeviceElement for u8 {}

/// Where a freed allocation waits for the next allocation of its exact byte length: nowhere
/// (freed at once), the KV reserve, or the vector pool (VADE_VECTOR_POOL).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Reuse { None, Kv, Vector }
struct Buffer<'a> { address: DevicePtr, bytes: usize, _context: &'a Context, owner: Option<Rc<Buffer<'a>>>, segments: Option<segments::Segments<'a>>, host: Option<Handle>, reuse: Reuse }
impl<'a> Buffer<'a> {
    fn new(context: &'a Context, bytes: usize) -> Result<Self, String> {
        Self::allocate(context, bytes, false)
    }
    /// A device vector's allocation (VADE_VECTOR_POOL): a freed one of the same byte length
    /// when the pool holds one, else a new one; either returns to the pool when dropped.
    /// Pure memory reuse: its old bytes stay (cuMemAlloc's bytes are unspecified too), so
    /// every vector is written before it is read, as without the pool.
    fn pooled(context: &'a Context, bytes: usize) -> Result<Self, String> {
        if bytes != 0 && context.cleanup_errors.borrow().is_empty() {
            if let Some(address) = context.vector_pool.borrow_mut().take(bytes, false) {
                return Ok(Self { address, bytes, _context: context, owner: None, segments: None, host: None, reuse: Reuse::Vector });
            }
        }
        let mut buffer = Self::new(context, bytes)?;
        buffer.reuse = Reuse::Vector;
        context.vector_pool.borrow_mut().allocated(bytes);
        Ok(buffer)
    }
    fn allocate(context: &'a Context, bytes: usize, managed: bool) -> Result<Self, String> {
        if bytes == 0 { return Err("zero byte allocation rejected".into()); }
        if !context.cleanup_errors.borrow().is_empty() { return Err("context poisoned by cleanup error".into()); }
        context.activate()?;
        context.make_room(bytes)?;
        let total = context.allocated.get().checked_add(bytes).ok_or("allocation accounting overflow")?;
        if total.checked_add(context.graph_storage.get()).is_none_or(|n| n > context.budget.get()) { return Err(crate::admission::allocation(total as u128 + context.graph_storage.get() as u128, context.budget.get(), &format!("allocation {bytes} bytes; live {} bytes; graphs {} bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to allocator", context.allocated.get(), context.graph_storage.get()))); }
        let mut address = 0;
        unsafe {
            if managed { check(cuMemAllocManaged(&mut address, bytes, 1), "cuMemAllocManaged")?; }
            else { check(cuMemAlloc_v2(&mut address, bytes), "cuMemAlloc_v2")?; }
        }
        context.record_allocation(total);
        Ok(Self { address, bytes, _context: context, owner: None, segments: None, host: None, reuse: Reuse::None })
    }
    fn mapped(context: &'a Context, bytes: usize) -> Result<Self, String> {
        context.activate()?;
        context.make_room(bytes)?;
        let total = context.allocated.get().checked_add(bytes).ok_or("allocation accounting overflow")?;
        if bytes == 0 || total.checked_add(context.graph_storage.get()).is_none_or(|n| n > context.budget.get()) { return Err(crate::admission::allocation(total as u128 + context.graph_storage.get() as u128, context.budget.get(), &format!("mapped allocation {bytes} bytes; live {} bytes; graphs {} bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to allocator", context.allocated.get(), context.graph_storage.get()))); }
        if !context.cleanup_errors.borrow().is_empty() { return Err("context poisoned by cleanup error".into()); }
        let (mut host, mut address) = (ptr::null_mut(), 0);
        unsafe {
            check(cuMemHostAlloc(&mut host, bytes, 2), "mapped activation allocation")?;
            if let Err(error) = check(cuMemHostGetDevicePointer_v2(&mut address, host, 0), "mapped activation device pointer") {
                context.record_cleanup(cuMemFreeHost(host), "cuMemFreeHost");
                return Err(error);
            }
        }
        // GB10 has one DRAM pool: mapped activation storage counts once with
        // device allocations, rather than again as pinned upload staging.
        context.record_allocation(total);
        Ok(Self { address, bytes, _context: context, owner: None, segments: None, host: Some(host), reuse: Reuse::None })
    }
    fn view(owner: &Rc<Self>, offset: usize, bytes: usize) -> Result<Self, String> {
        if offset.checked_add(bytes).is_none_or(|end| end > owner.bytes) { return Err("view outside weight arena".into()); }
        Ok(Self { address: owner.address + offset as u64, bytes, _context: owner._context, owner: Some(owner.clone()), segments: None, host: None, reuse: Reuse::None })
    }


}
impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        if self.owner.is_some() { return; }
        if (self._context.capturing.get() || (self._context.resident_queued.get() && self._context.pending.get())) && self.segments.is_none() {
            self._context.defer_free(self.address, self.bytes, self.host.take(), self.reuse);
            return;
        }
        let _wait = self._context.progress.wait(WaitKind::Cleanup);
        if self._context.pending.get() {
            if let Err(error) = self._context.sync() { self._context.cleanup_errors.borrow_mut().push(error); }
        }
        if let Err(error) = self._context.activate() { self._context.cleanup_errors.borrow_mut().push(error); return; }
        if let Some(mut segments) = self.segments.take() {
            if self._context.cleanup_errors.borrow().is_empty() && segments.keep(self.address, self._context) { return; }
            segments.release(self.address, self._context);
            return;
        }
        self._context.free(self.address, self.bytes, self.host, self.reuse);
    }
}

struct Pinned<'a> { address: Handle, bytes: usize, context: &'a Context }
impl<'a> Pinned<'a> {
    fn new(context: &'a Context, bytes: usize) -> Result<Self, String> {
        context.activate()?;
        let mut address = ptr::null_mut();
        unsafe { check(cuMemHostAlloc(&mut address, bytes, 0), "pinned staging allocation")?; }
        let total = context.pinned_staging.get() + bytes;
        context.pinned_staging.set(total);
        context.peak_pinned_staging.set(context.peak_pinned_staging.get().max(total));
        Ok(Self { address, bytes, context })
    }
}
impl Drop for Pinned<'_> {
    fn drop(&mut self) {
        if self.address.is_null() { return; }
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if self.context.pending.get() {
            if let Err(e) = self.context.sync() { self.context.cleanup_errors.borrow_mut().push(e); }
        }
        if let Err(e) = self.context.activate() { self.context.cleanup_errors.borrow_mut().push(e); }
        let code = unsafe { cuMemFreeHost(self.address) };
        self.context.record_cleanup(code, "cuMemFreeHost");
        if code == 0 { self.context.pinned_staging.set(self.context.pinned_staging.get() - self.bytes); }
    }
}
/// One reader pool for the entire model. Declaration order joins readers before
/// freeing pinned slots; uploads accept context-owned buffers, never raw pointers.
struct FileUploader<'a> { readers: crate::load::Readers, slots: Vec<Pinned<'a>> }
impl<'a> FileUploader<'a> {
    fn new(context: &'a Context) -> Result<Self, String> {
        Self::with_buffers(context, 1)
    }
    fn double_buffered(context: &'a Context) -> Result<Self, String> {
        Self::with_buffers(context, crate::staging::FILE_BUFFERS)
    }
    fn with_buffers(context: &'a Context, buffers: usize) -> Result<Self, String> {
        let mut slots = Vec::new();
        for _ in 0..crate::load::READERS * buffers {
            slots.push(Pinned::new(context, crate::load::CHUNK)?);
        }
        let addresses = slots.iter().map(|slot| slot.address as usize);
        let readers = unsafe { if buffers == 2 { crate::load::Readers::double_buffered(addresses) }
            else { crate::load::Readers::new(addresses) } }?;
        Ok(Self { readers, slots })
    }
    fn upload_selected(&self, shard: &crate::load::WeightFile<'_>, buffer: &Buffer<'a>,
        backend: &Backend<'a>, stream: &Stream<'a>) -> Result<(u64, u64), String> {
        use std::sync::atomic::Ordering::Relaxed;
        let source = crate::load::Source::open_selected(shard.file, false)?;
        let mut jobs = Vec::new();
        for range in shard.ranges {
            if range.destination.checked_add(range.bytes).is_none_or(|end| end > buffer.bytes) {
                return Err("selected weight upload exceeds arena".into());
            }
            jobs.extend(range.jobs(&source)?);
        }
        let jobs = crate::load::coalesce(jobs);
        let events = (0..self.slots.len()).map(|_| Event::ready(stream))
            .collect::<Result<Vec<_>, _>>()?;
        let send = |slot, index: usize| self.readers.send(slot, jobs[index].0.clone());
        let mut active = vec![0; self.slots.len()];
        let mut next = jobs.len().min(self.slots.len());
        let mut reading = next;
        for (slot, index) in active.iter_mut().enumerate().take(next) {
            *index = slot; send(slot, slot)?;
        }
        let mut copying = std::collections::VecDeque::new();
        for _ in 0..jobs.len() {
            let (slot, result) = self.readers.ready.recv().map_err(|_| "weight readers stopped")?;
            result?;
            reading -= 1;
            for &(at, destination, len) in &jobs[active[slot]].1 {
                let bytes = unsafe { std::slice::from_raw_parts(self.slots[slot].address.cast::<u8>().add(at), len) };
                unsafe { backend.queue_pinned_bytes(buffer, destination, bytes)?; }
            }
            events[slot].record_again(stream)?;
            copying.push_back(slot);
            // Four readers stay busy while up to four other slots are copied.
            // Wait for a slot's own event before giving its bytes back to disk.
            if next < jobs.len() && (copying.len() >= crate::load::READERS || reading == 0) {
                let slot = copying.pop_front().unwrap();
                events[slot].wait(false)?;
                active[slot] = next;
                send(slot, next)?;
                next += 1; reading += 1;
            }
        }
        backend.synchronize()?;
        Ok((source.read_bytes.load(Relaxed), source.buffered_bytes.load(Relaxed)))
    }
    fn upload(&self, file: &std::fs::File, ranges: &[(u64, usize, &Buffer<'a>, usize, usize, bool)],
        backend: &Backend<'a>, exact: bool) -> Result<(), String> {
        use crate::load::{CHUNK, ReadJob, Source};
        let source = Source::open_selected(file, exact)?;
        let mut jobs = Vec::new();
        for &(offset, bytes, buffer, destination, columns, fp8) in ranges {
            if bytes == 0 || offset.checked_add(bytes as u64).is_none_or(|n| n > source.len)
                || destination.checked_add(bytes).is_none_or(|n| n > buffer.bytes)
                || !ptr::eq(buffer._context, self.slots[0].context) { return Err("invalid weight upload range".into()); }
            let mut done = 0;
            while done < bytes {
                let at = offset + done as u64;
                let skip = if exact { 0 } else { (at % 4096) as usize };
                let tile = if columns == 0 { 1 } else { columns * if fp8 { 8 } else if !exact && columns % 64 == 0 {128} else {16} };
                let count = ((CHUNK - skip) / tile * tile).min(bytes - done);
                if count == 0 { return Err("packed weight row tile exceeds staging slot".into()); }
                let read_bytes = if exact { count } else { (skip + count).div_ceil(4096) * 4096 };
                jobs.push((at - skip as u64, read_bytes, skip, count, buffer, destination + done, columns, fp8));
                done += count;
            }
        }
        let send = |slot, index: usize| {
            let (offset, count, skip, payload, _, _, columns, fp8) = jobs[index];
            self.readers.send(slot, ReadJob { source: source.clone(), offset, count, skip, payload, columns, fp8, packed8: exact, row_bytes: 0, row_stride: 0 })
        };
        let mut active = [0usize; 4];
        let mut next = jobs.len().min(4);
        for (slot, index) in active.iter_mut().enumerate().take(next) { *index = slot; send(slot, slot)?; }
        for _ in 0..jobs.len() {
            let (slot, result) = self.readers.ready.recv().map_err(|_| "weight readers stopped")?;
            result?;
            let (_, _, skip, count, buffer, destination, _, _) = jobs[active[slot]];
            let bytes = unsafe { std::slice::from_raw_parts(self.slots[slot].address.cast::<u8>().add(skip), count) };
            unsafe { backend.write_pinned_bytes(buffer, destination, bytes)?; }
            if next < jobs.len() { active[slot] = next; send(slot, next)?; next += 1; }
        }
        Ok(())
    }
}

/// Allocation runs ahead by one shard. Only the worker touches its CUDA pointer;
/// the owning thread reserves the budget and adopts/frees it after joining.
struct PendingBuffer<'a> { worker: Option<std::thread::JoinHandle<Result<DevicePtr, String>>>, context: &'a Context, bytes: usize }
impl<'a> PendingBuffer<'a> {
    fn new(context: &'a Context, bytes: usize) -> Result<Self, String> {
        let total = context.allocated.get().checked_add(bytes).ok_or("allocation accounting overflow")?;
        if bytes == 0 || total.checked_add(context.graph_storage.get()).is_none_or(|n| n > context.budget.get()) { return Err(crate::admission::allocation(total as u128 + context.graph_storage.get() as u128, context.budget.get(), &format!("weight allocation {bytes} bytes; live {} bytes; graphs {} bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to allocator", context.allocated.get(), context.graph_storage.get()))); }
        if !context.cleanup_errors.borrow().is_empty() { return Err("context poisoned by cleanup error".into()); }
        let raw = context.raw as usize;
        let worker = std::thread::Builder::new().name("weight-allocate".into()).spawn(move || {
            let mut address = 0;
            unsafe {
                check(cuCtxSetCurrent(raw as Handle), "weight allocation context")?;
                check(cuMemAlloc_v2(&mut address, bytes), "weight arena allocation")?;
            }
            Ok(address)
        }).map_err(|e| e.to_string())?;
        context.record_allocation(total);
        Ok(Self { worker: Some(worker), context, bytes })
    }
    fn wait(mut self) -> Result<Buffer<'a>, String> { self.join() }
    fn join(&mut self) -> Result<Buffer<'a>, String> {
        let result = self.worker.take().unwrap().join().map_err(|_| "weight allocator panicked".to_string()).and_then(|r| r);
        match result {
            Ok(address) => Ok(Buffer { address, bytes: self.bytes, _context: self.context, owner: None, segments: None, host: None, reuse: Reuse::None }),
            Err(error) => { self.context.allocated.set(self.context.allocated.get() - self.bytes); Err(error) }
        }
    }
}
impl Drop for PendingBuffer<'_> {
    fn drop(&mut self) {
        if self.worker.is_some() {
            if let Err(error) = self.join() { self.context.cleanup_errors.borrow_mut().push(error); }
        }
    }
}

/// One nonblocking stream owns the queued forward pass. Buffers synchronize on
/// destruction if work is pending, including early error returns.
struct Stream<'a> { raw: Handle, context: &'a Context }
impl<'a> Stream<'a> {
    fn new(context: &'a Context) -> Result<Self, String> {
        context.activate()?;
        let mut raw = ptr::null_mut();
        unsafe { check(cuStreamCreate(&mut raw, 1), "cuStreamCreate")?; }
        Ok(Self { raw, context })
    }
    /// One deadline covers query polling and the following blocking driver call.
    fn sync(&self, spin: bool) -> Result<(), String> {
        let wait = self.context.progress.wait(WaitKind::Stream);
        self.context.activate()?;
        if spin {
            while unsafe { cuStreamQuery(self.raw) } == 600 {
                wait.poll(); std::hint::spin_loop();
            }
        }
        unsafe { self.context.check_execution(cuStreamSynchronize(self.raw), "cuStreamSynchronize")?; }
        self.context.pending.set(false);
        Ok(())
    }
}
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if let Err(error) = self.sync(false) { self.context.cleanup_errors.borrow_mut().push(error); }
        unsafe { self.context.record_cleanup(cuStreamDestroy_v2(self.raw), "cuStreamDestroy_v2"); }
    }
}
struct Event<'a> { raw: Handle, context: &'a Context, decode_kind: usize }
impl<'a> Event<'a> {
    fn record(stream: &Stream<'a>) -> Result<Self, String> {
        let mut raw = ptr::null_mut();
        unsafe { check(cuEventCreate(&mut raw, 0), "cuEventCreate")?; }
        let event = Self { raw, context: stream.context, decode_kind: 2 };
        unsafe { check(cuEventRecord(raw, stream.raw), "cuEventRecord")?; }
        Ok(event)
    }
    fn ready(stream: &Stream<'a>) -> Result<Self, String> {
        Self::ready_kind(stream, 2)
    }
    fn ready_kind(stream: &Stream<'a>, kind: usize) -> Result<Self, String> {
        let mut raw=ptr::null_mut();
        let timer = stream.context.decode_timer(kind);
        unsafe{check(cuEventCreate(&mut raw,2),"create readiness event")?;}
        stream.context.decode_event(kind, 0, timer);
        let event=Self{raw,context:stream.context,decode_kind:kind};event.record_again(stream)?;Ok(event)
    }
    fn record_again(&self,stream:&Stream<'a>)->Result<(),String> {
        let timer = self.context.decode_timer(self.decode_kind);
        let result = unsafe{check(cuEventRecord(self.raw,stream.raw),"record readiness event")};
        self.context.decode_event(self.decode_kind, 1, timer); result
    }
    fn sync(&self) -> Result<(), String> {
        self.wait(false)
    }
    /// Wait for this event alone; work queued behind it keeps running.
    fn wait(&self, spin: bool) -> Result<(), String> {
        let wait = self.context.progress.wait(WaitKind::Event);
        self.context.activate()?;
        let query = self.context.decode_timer(self.decode_kind);
        if spin {
            while unsafe { cuEventQuery(self.raw) } == 600 {
                wait.poll(); std::hint::spin_loop();
            }
        }
        self.context.decode_event(self.decode_kind, 3, query);
        let completion = self.context.decode_timer(self.decode_kind);
        let result = unsafe { self.context.check_execution(cuEventSynchronize(self.raw), "cuEventSynchronize") };
        self.context.decode_event(self.decode_kind, 4, completion);
        if result.is_ok() { if let Some(p) = &self.context.decode_profile { p.borrow_mut().completed(self.decode_kind, completion); } }
        result
    }
    fn elapsed_ns(&self, end: &Self) -> Result<u128, String> {
        let mut ms = 0.0f32;
        unsafe { check(cuEventElapsedTime(&mut ms, self.raw, end.raw), "cuEventElapsedTime")?; }
        Ok((ms as f64 * 1_000_000.0) as u128)
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        let timer = self.context.decode_timer(self.decode_kind);
        unsafe { self.context.record_cleanup(cuEventDestroy_v2(self.raw), "cuEventDestroy_v2"); }
        self.context.decode_event(self.decode_kind, 2, timer);
    }
}

/// The only platform-specific part of the shared module-loading state machine.
struct ModuleDriver<'a> { context: &'a Context }
impl crate::module_loading::Driver for ModuleDriver<'_> {
    type Handle = Handle;
    fn device_version(&mut self) -> Result<((i32, i32), i32), String> {
        let (mut major, mut minor, mut version) = (0, 0, 0);
        unsafe {
            check(cuDeviceGetAttribute(&mut major, 75, self.context.device), "cubin compute major")?;
            check(cuDeviceGetAttribute(&mut minor, 76, self.context.device), "cubin compute minor")?;
            check(cuDriverGetVersion(&mut version), "cubin driver API")?;
        }
        Ok(((major, minor), version))
    }
    fn cubin(&mut self, image: &[u8]) -> (i32, Option<Handle>) {
        let mut raw = ptr::null_mut();
        let code = unsafe { cuModuleLoadData(&mut raw, image.as_ptr().cast()) };
        (code, (!raw.is_null()).then_some(raw))
    }
    fn ptx(&mut self, ptx: &CStr, receipt: bool) -> Result<crate::module_loading::Loaded<Handle>, crate::module_loading::Failure<Handle>> {
        let mut raw = ptr::null_mut();
        let result = if receipt {
            self.context.jit_receipts.as_ref().unwrap().borrow_mut().load(&mut raw, ptx).map(Some)
        } else {
            unsafe { check(cuModuleLoadData(&mut raw, ptx.as_ptr().cast()), "cuModuleLoadData").map(|()| None) }
        };
        match result {
            Ok(receipt) => Ok(crate::module_loading::Loaded { handle: raw, receipt }),
            Err(error) => Err(crate::module_loading::Failure { handle: (!raw.is_null()).then_some(raw), error }),
        }
    }
    fn unload(&mut self, handle: Handle) -> Result<(), String> {
        unsafe { check(cuModuleUnload(handle), "failed module unload") }
    }
    fn error(&self, code: i32) -> String {
        check(code, "prebuilt cubin load").unwrap_err()
    }
    fn notice(&mut self, message: &str) { eprintln!("{message}"); }
}

/// `functions` caches each name's handle and the dynamic-shared limit last set
/// on it: a handle is fixed for the module's life, the receipt registers it on
/// first lookup, and setting an unchanged attribute again is a no-op, so the
/// cache removes the per-launch CString, cuModuleGetFunction and
/// cuFuncSetAttribute without changing what the driver launches.
struct Module<'a> {
    raw: Cell<Handle>,
    context: &'a Context,
    jit_module: Cell<Option<usize>>,
    functions: std::cell::RefCell<std::collections::HashMap<String, (Handle, Option<c_int>)>>,
}
impl<'a> Module<'a> {
    fn swap(&self, other: &Self) {
        self.raw.swap(&other.raw); self.jit_module.swap(&other.jit_module);
        self.functions.swap(&other.functions);
    }
    fn load(context: &'a Context, ptx: &CStr) -> Result<Self, String> {
        context.activate()?;
        let compiled = include_bytes!(concat!(env!("OUT_DIR"), "/kernel.cubin"));
        let receipt = context.jit_receipts.is_some();
        let force_ptx = std::env::var("CUDA_FORCE_PTX_JIT").as_deref() == Ok("1");
        let image = if !compiled.is_empty() && !receipt && !force_ptx {
            Some(crate::module_loading::Image { bytes: compiled, ptx: crate::layer_ptx()?.to_bytes(),
                minimum_driver: include!(concat!(env!("OUT_DIR"), "/cubin_driver.rs")) })
        } else { None };
        let loaded = crate::module_loading::load(&mut ModuleDriver { context }, ptx, image, receipt, force_ptx)?;
        Ok(Self { raw: Cell::new(loaded.handle), context, jit_module: Cell::new(loaded.receipt), functions: Default::default() })
    }
    fn function(&self, name: &str) -> Result<Handle, String> {
        self.context.activate()?;
        if let Some(&(raw, _)) = self.functions.borrow().get(name) {
            return Ok(raw);
        }
        let raw = self.lookup(name)?;
        self.functions.borrow_mut().insert(name.into(), (raw, None));
        Ok(raw)
    }
    fn lookup(&self, name: &str) -> Result<Handle, String> {
        let symbol = CString::new(name).map_err(|e| e.to_string())?;
        let mut raw = ptr::null_mut();
        unsafe { check(cuModuleGetFunction(&mut raw, self.raw.get(), symbol.as_ptr()), "cuModuleGetFunction")
            .map_err(|error| format!("{name}: {error}"))?; }
        if let Some(module) = self.jit_module.get() {
            self.context.jit_receipts.as_ref().unwrap().borrow_mut().resolved(raw, name, module)?;
        }
        Ok(raw)
    }
    fn function_with_shared(&self, name: &str, bytes: u32) -> Result<Handle, String> {
        let raw = self.function(name)?;
        let bytes = c_int::try_from(bytes).map_err(|_| "dynamic shared bytes exceed driver ABI")?;
        let mut functions = self.functions.borrow_mut();
        let entry = functions.get_mut(name).ok_or("missing cached kernel handle")?;
        if entry.1 != Some(bytes) {
            // CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES.
            unsafe { check(cuFuncSetAttribute(raw, 8, bytes), "cuFuncSetAttribute")?; }
            entry.1 = Some(bytes);
        }
        Ok(raw)
    }


}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if let Err(error) = self.context.activate() { self.context.cleanup_errors.borrow_mut().push(error); return; }
        unsafe { self.context.record_cleanup(cuModuleUnload(self.raw.get()), "cuModuleUnload"); }
    }
}

fn parameter<T>(value: &mut T) -> *mut c_void { (value as *mut T).cast() }
fn byte_count(elements: usize, width: usize) -> Result<usize, String> {
    elements.checked_mul(width).ok_or_else(|| "allocation size overflow".into())
}


#[path = "backend.rs"]
mod backend;
pub use backend::{with_backend, with_backend_mode, Backend, HostRegistration, Matrix, MatvecMode, DeviceVector, CacheRows, TopKScratch, ExpertTable, MoeScratch, MoeBatchScratch, MOE_ROUTING_ERROR,
    Fp8GroupedScratch, Fp8RoutingFlag, WeightArena, ExchangeSlot, TpMailbox, TpMailboxHost, TP_MAILBOX_SLOTS};
pub use backend::{BackendStats, Fence, FrontendStream};
pub use backend::ResidentTextEdges;
pub use backend::{ResidentMxfp8Projection,Mxfp8ProjectionPass,ResidentProjection};
pub use backend::{DecodeGraph, PromptBuffer, ResidentQueue};
pub use backend::{Bf16Vector,ResidentMhc,MhcPass,ResidentNorm,NormOutput};

/// VADE_VECTOR_POOL (default on; =0 off): whether a server configures the vector pool
/// (Backend::configure_vector_pool). Other values are refused.
pub fn vector_pool_from_env() -> Result<bool, String> {
    match std::env::var("VADE_VECTOR_POOL").as_deref() {
        Ok("0") => Ok(false),
        Err(std::env::VarError::NotPresent) | Ok("1") => Ok(true),
        _ => Err("VADE_VECTOR_POOL must be 0 or 1".into()),
    }
}
/// Read the test artifact once so rank identity and JIT use exactly the same bytes.
pub fn ab_ptx_from_env() -> Result<Option<CString>, String> {
    std::env::var_os("VADE_AB_PTX").map(|path| {
        CString::new(std::fs::read(path).map_err(|e| format!("VADE_AB_PTX: {e}"))?)
            .map_err(|e| format!("VADE_AB_PTX: {e}"))
    }).transpose()
}
