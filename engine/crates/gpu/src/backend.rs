//! Scoped GB10 backend. Persistent buffers and one ordered decode stream.
use super::*;
#[path = "next_profile.rs"] mod next_profile;
#[path = "frontend_stream.rs"] mod frontend_stream;
#[path = "verify_segment.rs"] mod verify_segment;
pub use frontend_stream::FrontendStream;
#[path = "batch.rs"] mod batch;
#[path = "multi.rs"] mod multi;
pub use multi::{CacheRows, TopKScratch};
#[path = "moe.rs"] mod moe;
pub use moe::{ExpertTable, MoeScratch, MoeBatchScratch, MOE_ROUTING_ERROR};
#[path = "fp8.rs"] mod fp8;
#[path = "tp_mailbox.rs"] mod tp_mailbox;
pub use tp_mailbox::TpMailbox;
pub use crate::tp_mailbox_host::{TpMailboxHost, TP_MAILBOX_SLOTS};
pub use fp8::{Fp8GroupedScratch, Fp8RoutingFlag};
#[path = "fp4.rs"] mod fp4;
#[path = "backend/device.rs"] mod device;
#[path = "backend/weights.rs"] mod weights;
#[path = "backend/kernels.rs"] mod kernels;
#[path = "backend/dense_tile.rs"] mod dense_tile;
#[path = "backend/dense.rs"] mod dense;
use kernels::AbKernels;

pub struct Matrix<'a> { buffer: Buffer<'a>, rows: usize, cols: usize, packed: bool, packed_blocked: bool, stride: usize, row_offset: usize, column_offset: usize, scalar_f32: bool, fp8_scales: Option<Buffer<'a>>, fp8_fragments: bool }
/// One official-dtype allocation; views are offsets and packing only permutes bytes.
pub struct WeightArena<'a> { buffer: Rc<Buffer<'a>> }
impl WeightArena<'_> {
    pub fn bytes(&self) -> usize { self.buffer.bytes }
    pub fn device_address(&self) -> u64 { self.buffer.address }
    /// Bounded inspection of stored bytes, after all prior device work completes.
    pub fn read(&self, offset: usize, out: &mut [u8]) -> Result<(), String> {
        if offset.checked_add(out.len()).is_none_or(|n| n > self.bytes()) {
            return Err("weight read outside arena".into());
        }
        let _wait = self.buffer._context.progress.wait(WaitKind::Readback);
        self.buffer._context.sync()?;
        unsafe { check(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(),
            self.buffer.address + offset as u64, out.len()), "weight arena read") }
    }
}
impl<'a> Matrix<'a> {
    /// Borrowed immutable address for the shared hybrid attention kernels.
    pub fn device_address(&self) -> u64 { self.buffer.address }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    fn embedding_layout(&self) -> u32 { if !self.packed { 0 } else if self.packed_blocked { 1 } else { 2 } }
    pub fn scalar_f32(&self) -> bool { self.scalar_f32 }
    pub fn is_fp8(&self) -> bool { self.fp8_scales.is_some() }
    /// Attach the checkpoint's BF16 block scales without copying either tensor.
    pub fn with_fp8_scales(mut self, scales: Self) -> Result<Self, String> {
        if self.packed || scales.packed || self.scalar_f32 || scales.scalar_f32
            || self.rows % 128 != 0 || self.cols % 128 != 0
            || scales.rows != self.rows / 128 || scales.cols != self.cols / 128
            || self.buffer.bytes != self.rows * self.cols || scales.buffer.bytes != scales.rows * scales.cols * 2 || scales.is_fp8()
            || !ptr::eq(self.buffer._context, scales.buffer._context) {
            return Err("invalid resident FP8 weight/scale layout".into());
        }
        self.fp8_scales = Some(scales.buffer);
        Ok(self)
    }
    /// Immutable views retain the loaded weight arena without copying weights.
    pub fn rows_view(&self, range: std::ops::Range<usize>) -> Result<Self, String> {
        if !self.packed || range.is_empty() || range.end > self.rows { return Err("invalid packed row view".into()); }
        let first = self.row_offset + range.start;
        let tile = if self.packed_blocked {64} else {8};
        let offset = first / tile * self.stride * tile * 2;
        Ok(Self { buffer: self.weight_view(offset)?, rows: range.len(), cols: self.cols,
            packed: true, packed_blocked: self.packed_blocked, stride: self.stride, row_offset: first % tile, column_offset: self.column_offset, scalar_f32: self.scalar_f32, fp8_scales: None, fp8_fragments: false })
    }

    fn weight_view(&self, offset: usize) -> Result<Buffer<'a>, String> {
        let owner = self.buffer.owner.as_ref().ok_or("weight view requires a shared loaded arena")?;
        let base = (self.buffer.address - owner.address) as usize;
        Buffer::view(owner, base + offset, self.buffer.bytes.checked_sub(offset).ok_or("weight view outside matrix")?)
    }
}
/// Page-locked caller-owned RDMA mailbox, mapped for kernels. Keep its storage
/// alive and unmoved until this registration drops.
pub struct HostRegistration<'a> { host: *mut u8, bytes: usize, device: DevicePtr, context: &'a Context }
impl HostRegistration<'_> {
    /// Kernel address of a host byte inside the registration.
    pub fn device_address(&self, host: *const u8) -> Result<u64, String> {
        let offset = (host as usize).checked_sub(self.host as usize).filter(|&offset| offset < self.bytes)
            .ok_or("host address outside the registration")?;
        Ok(self.device + offset as u64)
    }
}
impl Drop for HostRegistration<'_> {
    fn drop(&mut self) {
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if self.context.activate().is_ok() {
            self.context.record_cleanup(unsafe { cuMemHostUnregister(self.host.cast()) }, "cuMemHostUnregister");
        }
    }
}
pub struct DeviceVector<'a> { buffer: Buffer<'a>, len: usize, managed: bool,
    #[cfg(debug_assertions)] projection_scratch: Rc<Cell<(u64, Option<u64>)>>,
}
impl DeviceVector<'_> {
    // Shared by slices: a BF16 pack must invalidate every alias's FP8 receipt.
    fn invalidate_projection_scratch(&self) {
        #[cfg(debug_assertions)] {
            let (generation, _) = self.projection_scratch.get();
            self.projection_scratch.set((generation.wrapping_add(1), None));
        }
    }
    /// Address for model-specific kernels on this backend's ordered stream.
    pub fn device_address(&self) -> u64 { self.buffer.address }
    pub fn is_segmented(&self) -> bool { self.buffer.segments.is_some() }
    /// Complete immutable physical pages common to two virtual KV mappings.
    pub fn shared_prefix_bytes(&self, other: &Self) -> usize { self.buffer.shared_prefix_bytes(&other.buffer) }
    /// Complete prefix allocations survive a fork without a tail copy.
    pub fn prefix_allocations(&self, words: usize, out: &mut Vec<(usize, usize)>) { self.buffer.prefix_allocations(words * 4, out); }
    /// Stable identities for physical storage; shared views report their owner.
    pub fn allocations(&self, out: &mut Vec<(usize, usize)>) {
        if self.is_segmented() { self.buffer.segment_allocations(out); }
        else if let Some(owner) = &self.buffer.owner { out.push((Rc::as_ptr(owner) as usize, owner.bytes)); }
        else { out.push((&self.buffer as *const _ as usize, self.buffer.bytes)); }
    }
    pub fn writable_from(&self, start: usize) -> bool { self.buffer.writable_from(start * 4) }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    /// Change the active shape of scratch without reallocating its storage.
    pub fn resize_within_capacity(&mut self, len: usize) -> Result<(), String> {
        if self.is_segmented() || len == 0 || len > self.buffer.bytes / 4 {
            return Err("scratch shape exceeds contiguous storage".into());
        }
        self.len = len;
        Ok(())
    }
}

struct LayerKernels<'a> { _module: Module<'a>, functions: Cell<LayerFunctions> }
#[derive(Clone, Copy)]
struct LayerFunctions {
    embedding: Handle, embedding_selected: Handle, rms: Handle, residual_rms: Handle, rms_apply: Handle,
    rms_fused: Handle, residual_rms_fused: Handle,
    qk_rope: Handle, scores: Handle, scores_gqa4: Handle, softmax: Handle, values: Handle, value_split: Handle, value_gqa4: Handle, value_merge: Handle,
    model_rms: Handle, model_rms_apply: Handle, model_qkv: Handle, model_elementwise: Handle,
    bounded_scores: Handle, bounded_scores_grouped: Handle, bounded_softmax: Handle, bounded_values: Handle, bounded_flash: Handle, bounded_split: Handle, bounded_merge: Handle,
    swiglu: Handle, residual: Handle, logits_bf16: Handle, argmax: Handle, argmax_finish: Handle,
    top_k: Handle, top_k_finish: Handle,
}
impl<'a> LayerKernels<'a> {
    fn load(context: &'a Context) -> Result<Self, String> {
        let ptx = crate::layer_ptx()?;
        let module = Module::load(context, ptx)?;
        let functions = LayerFunctions::load(&module)?;
        Ok(Self { _module: module, functions: Cell::new(functions) })
    }
}
impl LayerFunctions {
    fn load(module: &Module<'_>) -> Result<Self, String> {
        Ok(Self {
            model_rms: module.function("model_rms_partials")?, model_rms_apply: module.function("model_rms_apply")?,
            model_qkv: module.function("model_qkv")?, model_elementwise: module.function("model_elementwise")?,
            bounded_flash: module.function_with_shared("attention_bounded_flash", 65536)?,
            bounded_scores_grouped: module.function("attention_bounded_scores_grouped")?,
            bounded_split: module.function("attention_bounded_values_split")?, bounded_merge: module.function("attention_bounded_merge")?,
            bounded_scores: module.function("attention_bounded_scores")?,
            bounded_softmax: module.function("attention_bounded_softmax")?, bounded_values: module.function("attention_bounded_values")?,
            embedding: module.function("embedding_bf16")?, embedding_selected: module.function("embedding_selected_bf16")?, rms: module.function("layer_rms")?,
            rms_apply: module.function("layer_rms_apply")?,
            rms_fused: module.function("layer_rms_fused")?, residual_rms_fused: module.function("residual_rms_fused")?,
            residual_rms: module.function("residual_rms")?, qk_rope: module.function("qk_rope_cache")?,
            scores: module.function("attention_scores")?, softmax: module.function("attention_softmax")?,
            scores_gqa4: module.function("attention_scores_gqa4")?,
            values: module.function("attention_values")?, swiglu: module.function("layer_swiglu")?,
            value_split: module.function("attention_values_split")?, value_merge: module.function("attention_values_merge")?,
            value_gqa4: module.function("attention_values_gqa4")?,
            residual: module.function("layer_residual")?, argmax: module.function("logits_argmax")?,
            argmax_finish: module.function("logits_argmax_finish")?, logits_bf16: module.function("logits_bf16")?,
            top_k: module.function("logits_top_k")?, top_k_finish: module.function("logits_top_k_finish")?,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatvecMode { Warp32, Vec8 }
struct UploadStaging<'a> { buffer: Pinned<'a>, capacity: usize, used: usize }
/// Pinned activation exchange; host waits for its download event, not later launches.
pub struct ExchangeSlot<'a> { buffer: Pinned<'a>, capacity: usize, len: usize, event: Event<'a>, queued: bool }
impl ExchangeSlot<'_> {
    pub fn bytes(&self) -> usize { self.capacity * 4 }
}
/// Allocation/synchronization wall times and optional CUDA event kernel times.
#[derive(Clone, Copy, Default, Debug)]
pub struct BackendStats {
    pub weight_bytes: u128, pub weight_read_upload_ns: u128,
    /// File read bytes and buffered fallback bytes for rank-selected arena loads.
    pub weight_file_bytes: u128, pub weight_buffered_bytes: u128,
    pub allocation_ns: u128, pub output_download_ns: u128,
    /// CUDA event times: matvec/GEMM, RMS, QK/RoPE, attention/scores, softmax, values,
    /// SwiGLU, residual, embedding, argmax, input packing, KV packing, copies, output head.
    /// QKV, O, decode gate/up, down are split out at 14..18 and prefill gate/up at 20;
    /// these projection subtotals are also included in matvec/GEMM.
    /// Router/grouping and expert gather/activation/merge occupy 18 and 19.
    /// Expert gate/up and down occupy 21 and 22, separate from dense projections.
    /// Enabled only while profiling. Allocation time includes KV mapping/growth.
    /// GDN normalization, chunk products, inverse and solve occupy 23..27;
    /// their subtotals are also included in GDN preparation at 18.
    /// Dense hybrid delta-net inputs, small gates and output occupy 27..30.
    /// Flash uses 14..18 for K=10240, K=320, N<=1024 and remaining dense
    /// projections; these are included in slot 0, with the output head in 13.
    pub device_kernel_ns: [u128; 30], pub device_launches: [u128; 30],
    pub queued_calls: u128, pub stream_wait_ns: u128, pub stream_wait_calls: u128,
    /// Host wall time in launch and event APIs; GPU backlog may block them.
    pub host_launch_ns: u128, pub host_profile_ns: u128,
    /// Staging wall envelopes, including CPU work overlapping GPU execution.
    pub host_pack_ns: u128,
    pub graph_launches: u128, pub graph_captures: u128, pub host_graph_launch_ns: u128,
}
impl BackendStats {
    pub fn since(self, before: Self) -> Self {
        Self {
            weight_bytes: self.weight_bytes.saturating_sub(before.weight_bytes),
            weight_read_upload_ns: self.weight_read_upload_ns.saturating_sub(before.weight_read_upload_ns),
            weight_file_bytes: self.weight_file_bytes.saturating_sub(before.weight_file_bytes),
            weight_buffered_bytes: self.weight_buffered_bytes.saturating_sub(before.weight_buffered_bytes),
            allocation_ns: self.allocation_ns.saturating_sub(before.allocation_ns),
            output_download_ns: self.output_download_ns.saturating_sub(before.output_download_ns),
            device_kernel_ns: std::array::from_fn(|i| self.device_kernel_ns[i].saturating_sub(before.device_kernel_ns[i])),
            device_launches: std::array::from_fn(|i| self.device_launches[i].saturating_sub(before.device_launches[i])),
            queued_calls: self.queued_calls.saturating_sub(before.queued_calls),
            stream_wait_ns: self.stream_wait_ns.saturating_sub(before.stream_wait_ns),
            stream_wait_calls: self.stream_wait_calls.saturating_sub(before.stream_wait_calls),
            host_launch_ns: self.host_launch_ns.saturating_sub(before.host_launch_ns),
            host_profile_ns: self.host_profile_ns.saturating_sub(before.host_profile_ns),
            host_pack_ns: self.host_pack_ns.saturating_sub(before.host_pack_ns),
            graph_launches: self.graph_launches.saturating_sub(before.graph_launches),
            graph_captures: self.graph_captures.saturating_sub(before.graph_captures),
            host_graph_launch_ns: self.host_graph_launch_ns.saturating_sub(before.host_graph_launch_ns),
        }
    }
}
/// Captured kernels borrow session storage; drop the exec before replacing it.
pub struct DecodeGraph<'a> { raw: Handle, context: &'a Context, storage_bytes: usize }
impl DecodeGraph<'_> {
    /// Conservative driver metadata allowance, not a CUDA allocation query.
    /// Flash serving funds the maximum for every retained segment graph slot.
    /// Whole-model captures use GraphAllowance::whole_model instead.
    pub const MAX_FLASH_NODES: usize = crate::GraphAllowance::FLASH_MAX_NODES;
    pub const EXEC_STORAGE_BYTES: usize = crate::GraphAllowance::EXEC_STORAGE_BYTES;
    pub const NODE_STORAGE_BYTES: usize = crate::GraphAllowance::NODE_STORAGE_BYTES;
    pub const MAX_RESERVATION_BYTES: usize = Self::EXEC_STORAGE_BYTES + Self::MAX_FLASH_NODES * Self::NODE_STORAGE_BYTES;
    /// Verification keeps ordered recurrence and lexical work for each row.
    /// The base covers a complete single-row segment; every additional row
    /// funds sixteen extra kernel/copy nodes, up to the Flash segment limit.
    pub fn node_limit(rows: usize) -> usize { 128usize.saturating_add(rows.saturating_sub(1).saturating_mul(16)).min(Self::MAX_FLASH_NODES) }
    pub fn reservation_bytes_for_rows(rows: usize) -> usize {
        Self::EXEC_STORAGE_BYTES + Self::node_limit(rows) * Self::NODE_STORAGE_BYTES
    }
}
impl Drop for DecodeGraph<'_> {
    fn drop(&mut self) {
        let _wait = self.context.progress.wait(WaitKind::Cleanup);
        if let Err(error) = self.context.activate() { self.context.cleanup_errors.borrow_mut().push(error); return; }
        let code = unsafe { cuGraphExecDestroy(self.raw) };
        self.context.record_cleanup(code, "cuGraphExecDestroy");
        // A failed destruction retains its charge and poisons future admission.
        if code == 0 { self.context.graph_storage.set(self.context.graph_storage.get() - self.storage_bytes); }
    }
}
pub struct PromptOrder<'b, 'a> { backend: &'b Backend<'a>, previous: bool }
pub struct PrefillClass<'b, 'a> { backend: &'b Backend<'a>, previous: Option<usize> }
impl Drop for PrefillClass<'_, '_> {
    fn drop(&mut self) { self.backend.context.decode_profile.as_ref().unwrap().borrow_mut().prefill_scope(self.previous); }
}
/// A recorded stream position (Backend::fence).
pub struct Fence<'a>(Event<'a>);
impl Fence<'_> {
    pub fn wait(&self) -> Result<(), String> { self.0.sync() }
}
/// One prompt slot. Its event protects both downloads and upload-source reuse;
/// its completion channel keeps CPU access exclusive, including on errors/drop.
pub struct PromptBuffer<'a> {
    pending: Option<std::sync::mpsc::Receiver<Result<u128, String>>>,
    ready: Event<'a>,
    buffer: Pinned<'a>,
    failed: Cell<bool>,
}
impl PromptBuffer<'_> {
    pub fn bytes(&self) -> usize { self.buffer.bytes }
    pub fn finish_pack(&mut self) -> Result<u128, String> {
        match self.pending.take() {
            Some(done) => done.recv().map_err(|_| "prompt preparation stopped")?,
            None => Ok(0),
        }
    }
    /// Reuse after a reported submission failure, only once all work is drained.
    pub fn reset_failed(&mut self) -> Result<(), String> {
        if self.failed.get() {
            self.finish_pack()?;
            self.buffer.context.sync()?;
            self.failed.set(false);
        }
        Ok(())
    }
    pub fn bytes_mut(&mut self) -> Result<&mut [u8], String> {
        if self.failed.get() || self.pending.is_some() { return Err("prompt slot is failed or still in flight".into()); }
        self.ready.wait(false)?;
        Ok(unsafe { std::slice::from_raw_parts_mut(self.buffer.address.cast(), self.buffer.bytes) })
    }
    pub fn start_pack(&mut self, profile: bool,
        run: impl FnOnce(&mut [u8]) -> Result<(), String> + Send + 'static) -> Result<(), String> {
        if self.failed.get() || self.pending.is_some() { return Err("prompt slot is failed or still in flight".into()); }
        self.ready.wait(false)?;
        let (send, done) = std::sync::mpsc::channel();
        self.pending = Some(done);
        let address = self.buffer.address as usize;
        let bytes = self.buffer.bytes;
        crate::host_prep::submit(Box::new(move || {
            let start = profile.then(Instant::now);
            // The slot owns the allocation and drains `done` before release.
            // No safe accessor permits CPU/GPU access while `pending` is set.
            // Debug/test unwind builds can report a panic; release uses panic=abort.
            #[cfg(panic = "unwind")]
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(unsafe { std::slice::from_raw_parts_mut(address as *mut u8, bytes) })
            })).unwrap_or_else(|_| Err("prompt preparation panicked in an unwind build".into()));
            #[cfg(panic = "abort")]
            let result = run(unsafe { std::slice::from_raw_parts_mut(address as *mut u8, bytes) });
            let _ = send.send(result.map(|_| elapsed(start)));
        }));
        Ok(())
    }
    fn record(&self, stream: &Stream<'_>) -> Result<(), String> {
        self.ready.record_again(stream).inspect_err(|_| self.failed.set(true))
    }
}
impl Drop for PromptBuffer<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.finish_pack() { self.buffer.context.cleanup_errors.borrow_mut().push(error); }
    }
}
impl Drop for PromptOrder<'_, '_> {
    fn drop(&mut self) { self.backend.prompt_order.set(self.previous); }
}
pub struct Backend<'a> {
    ab: RefCell<Option<AbKernels<'a>>>, arm_x: Cell<bool>,
    context: &'a Context, module: Module<'a>, mode: MatvecMode,
    profiling: Cell<bool>, profile_kind: Cell<Option<usize>>, stats: RefCell<BackendStats>,
    stream: Stream<'a>, matvec_function: Cell<Handle>, layers: Result<LayerKernels<'a>, String>,
    timings: RefCell<Vec<(usize, Option<usize>, Event<'a>, Event<'a>)>>,
    batch: RefCell<Option<batch::BatchKernels>>, vec8: Cell<Option<Handle>>, use_vec8: Cell<bool>,
    decode_partials: RefCell<Option<DeviceVector<'a>>>,
    flash_input: RefCell<Option<DeviceVector<'a>>>,
    flash_mixed: RefCell<Option<DeviceVector<'a>>>,
    dense_offer: Cell<Option<(u64, usize, usize, u64)>>,
    /// Open verify segment capture (VADE_MTP_GRAPH): saved profiling flag, layers and rows.
    segment: Cell<Option<(bool, usize, usize)>>,
    multi: RefCell<Option<multi::Kernels>>,
    moe: RefCell<Option<moe::Kernels>>,
    fp8: RefCell<Option<fp8::Kernels>>,
    fp4: RefCell<Option<fp4::Kernels>>,
    uploads: RefCell<UploadStaging<'a>>,
    capture_position: Cell<DevicePtr>, capture_base: Cell<usize>, prompt_order: Cell<bool>,
    spin_waits: Cell<bool>, programmatic: Cell<bool>,
}
fn elapsed(timer: Option<Instant>) -> u128 { timer.map_or(0, |start| start.elapsed().as_nanos()) }

/// All buffers/module are dropped before checked context destruction. Cleanup
/// errors turn an otherwise successful invocation into an error.
pub fn with_backend<R>(ordinal: i32, ptx: &CStr, budget: usize,
    run: impl FnOnce(&Backend<'_>) -> Result<R, String>) -> Result<R, String> {
    with_backend_mode(ordinal, ptx, budget, MatvecMode::Vec8, run)
}

/// Select arithmetic for raw matrix probes; model weights use blocked BF16.
pub fn with_backend_mode<R>(ordinal: i32, ptx: &CStr, budget: usize, mode: MatvecMode,
    run: impl FnOnce(&Backend<'_>) -> Result<R, String>) -> Result<R, String> {
    let start = Instant::now();
    let context = Context::open(ordinal)?;
    let context_ms = start.elapsed().as_secs_f64() * 1000.0;
    context.budget.set(budget);
    let result = (|| {
        let start = Instant::now();
        let module = Module::load(&context, ptx)?;
        let ptx_load_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let matvec_function = module.function("matvec_bf16_f32_warp32")?;
        let requested_vec8 = mode == MatvecMode::Vec8;
        let vec8 = module.function("matvec_bf16_f32_vec8").ok();
        if requested_vec8 && vec8.is_none() { return Err("PTX has no vec8 kernel".into()); }
        let backend = Backend { context: &context, module, mode, vec8: Cell::new(vec8), ab: RefCell::new(None), arm_x: Cell::new(false), use_vec8: Cell::new(requested_vec8), batch: RefCell::new(None), multi: RefCell::new(None),
            moe: RefCell::new(None), fp8: RefCell::new(None), fp4: RefCell::new(None), decode_partials: RefCell::new(None),
            flash_input: RefCell::new(None), flash_mixed: RefCell::new(None), dense_offer: Cell::new(None),
            uploads: RefCell::new(UploadStaging { buffer: Pinned::new(&context, 4 * 1024 * 1024)?, capacity: 4 * 1024 * 1024, used: 0 }),
            segment: Cell::new(None), capture_position: Cell::new(0), capture_base: Cell::new(0), prompt_order: Cell::new(false), programmatic: Cell::new(false),
            spin_waits: Cell::new(false), profiling: Cell::new(false), profile_kind: Cell::new(None), stats: RefCell::new(BackendStats::default()),
            stream: Stream::new(&context)?, matvec_function: Cell::new(matvec_function), layers: LayerKernels::load(&context), timings: RefCell::new(Vec::new()) };
        eprintln!("gpu_backend startup context_ms={context_ms:.3} ptx_load_ms={ptx_load_ms:.3} kernel_setup_ms={:.3} device_bytes={} pinned_staging_bytes={}",
            start.elapsed().as_secs_f64() * 1000.0, backend.allocated_bytes(), backend.pinned_staging_bytes());
        run(&backend).inspect_err(|_| crate::teardown::arm())
    })();
    if result.is_err() { crate::teardown::arm(); }
    let cleanup = context.finish();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!("{error}; cleanup: {cleanup}")),
    }
}
impl<'a> Backend<'a> {
    /// Device SM count queried once when its CUDA context is opened.
    pub fn multiprocessor_count(&self) -> usize { self.context.multiprocessors }
    /// This engine targets GB10. Refuse FP4 payload execution assumptions on
    /// other devices before allocating the packed weight arena.
    pub fn require_nvfp4_device(&self) -> Result<(), String> {
        let (mut major, mut minor) = (0, 0);
        unsafe {
            check(cuDeviceGetAttribute(&mut major, 75, self.context.device), "NVFP4 compute capability major")?;
            check(cuDeviceGetAttribute(&mut minor, 76, self.context.device), "NVFP4 compute capability minor")?;
        }
        if (major, minor) != (12, 1) {
            return Err(format!("NVFP4 requires GB10 compute capability 12.1; found {major}.{minor}"));
        }
        self.fp4_kernels()?;
        Ok(())
    }
    /// Startup admission memory. GB10 shares DRAM with Linux: MemAvailable
    /// (reclaimable file cache counts as free) capped at CUDA total memory,
    /// less this context's allocations a container's MemAvailable may omit.
    pub fn available_memory_bytes(&self) -> Result<usize, String> {
        let (_, total) = self.device_memory_info()?;
        Ok(self.context.visible_available(super::unified_available_bytes(total)?))
    }
    pub fn prepare_decode_graph(&self, split_projection: bool, experts: bool) -> Result<(), String> {
        self.require_batch_kernels()?;
        if experts { self.require_moe_kernels()?; }
        if split_projection && self.decode_partials.borrow().is_none() {
            *self.decode_partials.borrow_mut() = Some(self.vector(4 * 2048)?);
        }
        Ok(())
    }
    /// Capture only allocation-free kernels. Position uploads and readback stay
    /// outside the graph and share its ordered stream.
    /// Safety: every captured allocation must remain alive through all replays.
    pub unsafe fn capture_decode(&self, position: &DeviceVector<'_>, base: usize,
        layers: usize, rows: usize,
        run: impl FnOnce() -> Result<(), String>) -> Result<DecodeGraph<'a>, String> {
        self.check_vector(position, 4)?;
        unsafe { self.capture_batch(layers, rows, || {
            self.capture_position.set(position.buffer.address + 12);
            self.capture_base.set(base);
            let result = run();
            self.capture_position.set(0);
            result
        }) }
    }
    /// Capture a whole model whose dynamic inputs reside in device buffers.
    /// Layers and rows bound its graph metadata before any work is recorded.
    /// Safety: all captured allocations must remain valid for every replay.
    pub unsafe fn capture_batch(&self, layers: usize, rows: usize,
        run: impl FnOnce() -> Result<(), String>) -> Result<DecodeGraph<'a>, String> {
        unsafe { self.capture_with_allowance(crate::GraphAllowance::whole_model(layers, rows)?, run) }
    }
    /// Failed capture executes no kernels. Healthy contexts may use eager fallback;
    /// execution or cleanup failures remain fatal. Captured allocations must survive replay.
    pub unsafe fn try_capture_batch(&self, layers: usize, rows: usize,
        run: impl FnOnce() -> Result<(), String>) -> Result<Option<DecodeGraph<'a>>, String> {
        match unsafe { self.capture_batch(layers, rows, run) } {
            Ok(graph) => Ok(Some(graph)),
            Err(error) if self.context.execution_error.borrow().is_some()
                || !self.context.cleanup_errors.borrow().is_empty() => Err(error),
            Err(_) => Ok(None),
        }
    }
    /// Capture a Flash segment with its serving-funded node limit.
    /// Safety: captured buffers must survive every replay; the allowance
    /// must be funded for this retained slot before capture starts.
    pub unsafe fn capture_batch_bounded(&self, node_limit: usize,
        run: impl FnOnce() -> Result<(), String>) -> Result<DecodeGraph<'a>, String> {
        unsafe { self.capture_with_allowance(crate::GraphAllowance::flash(node_limit)?, run) }
    }
    unsafe fn capture_with_allowance(&self, allowance: crate::GraphAllowance,
        run: impl FnOnce() -> Result<(), String>) -> Result<DecodeGraph<'a>, String> {
        self.synchronize()?;
        if !self.context.cleanup_errors.borrow().is_empty() { return Err("context poisoned by cleanup error".into()); }
        // Charge before capture so the temporary graph and its executable also
        // fit the inclusive allocation ceiling. No captured callback allocates.
        let maximum = allowance.maximum_bytes();
        self.context.make_room(maximum)?;
        let storage = self.context.graph_storage.get().checked_add(maximum).ok_or("graph storage accounting overflow")?;
        if storage > self.context.graph_budget.get()
            || self.allocated_bytes().checked_add(storage).is_none_or(|n| n > self.context.budget.get()) {
            return Err(crate::admission::allocation(if storage > self.context.graph_budget.get() { storage as u128 } else { storage as u128 + self.allocated_bytes() as u128 }, if storage > self.context.graph_budget.get() { self.context.graph_budget.get() } else { self.context.budget.get() }, &format!("graph storage {storage} bytes (graph ceiling {} bytes); live {} bytes; weights/rank, sessions, CONTEXT and OUTPUT not supplied to graph allocator", self.context.graph_budget.get(), self.allocated_bytes())));
        }
        self.context.graph_storage.set(storage);
        self.context.peak_graph_storage.set(self.context.peak_graph_storage.get().max(storage));
        self.context.peak_accounted.set(self.context.peak_accounted.get().max(self.accounted_bytes()));
        if let Err(error) = unsafe { check(cuStreamBeginCapture_v2(self.stream.raw, 1), "cuStreamBeginCapture") } {
            self.context.graph_storage.set(storage - maximum);
            return Err(error);
        }
        let profiling = self.profiling.replace(false);
        let result = run();
        self.profiling.set(profiling);
        let mut graph = ptr::null_mut();
        let ended = unsafe { check(cuStreamEndCapture(self.stream.raw, &mut graph), "cuStreamEndCapture") };
        self.context.pending.set(false);
        let result = result.and(ended);
        if let Err(error) = result {
            self.discard_capture(graph, maximum);
            return Err(error);
        }
        let mut nodes = 0;
        let counted = unsafe { check(cuGraphGetNodes(graph, ptr::null_mut(), &mut nodes), "cuGraphGetNodes") }
            .and_then(|_| allowance.storage_bytes(nodes));
        let storage_bytes = match counted {
            Ok(bytes) => bytes,
            Err(error) => { self.discard_capture(graph, maximum); return Err(error); }
        };
        let mut raw = ptr::null_mut();
        let instantiated = unsafe { check(cuGraphInstantiate_v2(&mut raw, graph, ptr::null_mut(), ptr::null_mut(), 0), "cuGraphInstantiate") };
        let destroyed = unsafe { cuGraphDestroy(graph) };
        self.context.record_cleanup(destroyed, "cuGraphDestroy");
        if let Err(error) = instantiated {
            let released = if raw.is_null() { true } else {
                let code = unsafe { cuGraphExecDestroy(raw) };
                self.context.record_cleanup(code, "cuGraphExecDestroy");
                code == 0
            };
            if destroyed == 0 && released { self.context.graph_storage.set(self.context.graph_storage.get() - maximum); }
            return Err(error);
        }
        if destroyed != 0 {
            // The source graph leaked: retain its full charge until retirement.
            let code = unsafe { cuGraphExecDestroy(raw) };
            self.context.record_cleanup(code, "cuGraphExecDestroy");
            return Err("CUDA captured graph destruction failed".into());
        }
        self.context.graph_storage.set(self.context.graph_storage.get() - (maximum - storage_bytes));
        self.stats.borrow_mut().graph_captures += 1;
        Ok(DecodeGraph { raw, context: self.context, storage_bytes })
    }
    fn discard_capture(&self, graph: Handle, bytes: usize) {
        let code = if graph.is_null() { 0 } else { unsafe { cuGraphDestroy(graph) } };
        self.context.record_cleanup(code, "cuGraphDestroy");
        if code == 0 { self.context.graph_storage.set(self.context.graph_storage.get() - bytes); }
    }
    /// Safety: captured buffers and their mapped extents must still be valid.
    pub unsafe fn replay_decode(&self, graph: &DecodeGraph<'_>) -> Result<(), String> {
        if !ptr::eq(graph.context, self.context) { return Err("graph belongs to another context".into()); }
        self.context.activate()?;
        let start = Instant::now();
        self.context.decode_enqueue();
        unsafe { self.context.check_execution(cuGraphLaunch(graph.raw, self.stream.raw), "cuGraphLaunch")?; }
        self.context.pending.set(true);
        let mut stats = self.stats.borrow_mut();
        stats.graph_launches += 1;
        stats.host_graph_launch_ns += start.elapsed().as_nanos();
        Ok(())
    }
    pub fn decode_position(&self, position: usize) -> (u32, DevicePtr) {
        let pointer = self.capture_position.get();
        ((if pointer == 0 { position } else { position - self.capture_base.get() }) as u32, pointer)
    }

    /// Load exact rank-selected bytes into one device allocation. Four direct-I/O
    /// readers each alternate two 8 MiB pinned slots; column rows compact in
    /// place. Slot events protect asynchronous copies. All staging is released
    /// on return, including failures; no host weight mirror is retained.
    pub fn load_weight_arena(&self, bytes: usize,
        shards: &[crate::load::WeightFile<'_>]) -> Result<WeightArena<'a>, String> {
        self.synchronize()?;
        let mut end = 0;
        for range in shards.iter().flat_map(|shard| shard.ranges) {
            if range.destination < end { return Err("overlapping or unordered weight arena ranges".into()); }
            end = range.destination.checked_add(range.bytes).ok_or("weight arena range overflow")?;
            if end > bytes { return Err("selected weights exceed arena".into()); }
        }
        let start = Instant::now();
        let buffer = Buffer::new(self.context, bytes)?;
        self.stats.borrow_mut().allocation_ns += start.elapsed().as_nanos();
        let start = Instant::now();
        let uploader = FileUploader::double_buffered(self.context)?;
        // Preserve zero alignment padding without extra host buffers. The
        // memset is ordered before copies and overlaps the initial disk reads.
        unsafe { check(cuMemsetD8Async(buffer.address, 0, bytes, self.stream.raw), "weight arena padding")?; }
        self.context.pending.set(true);
        for shard in shards {
            let (file_bytes, buffered_bytes) = uploader.upload_selected(shard, &buffer, self, &self.stream)?;
            let mut stats = self.stats.borrow_mut();
            stats.weight_file_bytes += file_bytes as u128;
            stats.weight_buffered_bytes += buffered_bytes as u128;
        }
        self.synchronize()?;
        let mut stats = self.stats.borrow_mut();
        stats.weight_bytes += bytes as u128;
        stats.weight_read_upload_ns += start.elapsed().as_nanos();
        Ok(WeightArena { buffer: Rc::new(buffer) })
    }
    /// The shared buffer/graph ceiling also governs the weight loader's plan.
    pub fn allocation_budget(&self) -> usize { self.context.budget.get() }
    /// Refuse a ceiling below the currently owned buffers and graph charges.
    pub fn set_allocation_budget(&self, bytes: usize) -> Result<(), String> {
        if bytes < self.accounted_bytes() { return Err("device budget below live allocations and graph storage".into()); }
        self.context.budget.set(bytes);
        Ok(())
    }
    /// Serving reserves a separate graph sublimit within the inclusive ceiling.
    pub fn set_graph_storage_budget(&self, bytes: usize) -> Result<(), String> {
        if bytes < self.graph_storage_bytes() { return Err("graph budget below retained storage".into()); }
        self.context.graph_budget.set(bytes);
        Ok(())
    }
    /// Current free and total device bytes for this backend's CUDA context.
    pub fn device_memory_info(&self) -> Result<(usize, usize), String> {
        self.context.activate()?;
        let (mut free, mut total) = (0, 0);
        unsafe { check(cuMemGetInfo_v2(&mut free, &mut total), "cuMemGetInfo_v2")?; }
        Ok((free, total))
    }

    /// Attribute queued output-projection kernels to slot 13 while profiling.
    pub fn profile_output_head<R>(&self, run: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
        self.profile_phase(13, run)
    }
    /// Attribute a group of launches to the existing operation timing slots.
    pub fn profile_phase<R>(&self, kind: usize, run: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
        if !self.profiling.get() { return run(); }
        let previous = self.profile_kind.replace(Some(kind));
        let result = run();
        self.profile_kind.set(previous);
        result
    }
    pub fn decode_attention_span(&self, positions: usize) -> usize {
        if positions <= 4096 { 64 } else if positions <= 16384 { 256 } else { 1024 }
    }
    /// Maximum summary count through this position, including a batch that
    /// crosses a span change and still needs 64 short-span partitions.
    pub fn decode_attention_parts(&self, positions: usize) -> usize {
        if positions <= 4096 { positions.div_ceil(64) }
        else if positions <= 16384 { 64 }
        else { 64.max(positions.div_ceil(1024)) }
    }
    /// Capacity of the shared upload/exchange buffer, used before RDMA growth.
    pub fn pinned_upload_bytes(&self) -> usize { self.uploads.borrow().capacity }
    /// CUDA-owned pinned bytes: uploads, prompt slots and temporary readers.
    pub fn pinned_host_bytes(&self) -> usize { self.context.pinned_staging.get() }
    /// CUDA-owned transfer staging, including temporary weight-reader slots.
    pub fn pinned_staging_bytes(&self) -> usize { self.context.pinned_staging.get() }
    pub fn peak_pinned_staging_bytes(&self) -> usize { self.context.peak_pinned_staging.get() }
    pub fn prompt_buffer(&self, bytes: usize) -> Result<PromptBuffer<'a>, String> {
        Ok(PromptBuffer { pending: None, ready: Event::ready(&self.stream)?,
            buffer: Pinned::new(self.context, bytes)?, failed: Cell::new(false) })
    }
    /// Copy exactly the selected immutable weight bytes on the existing stream.
    /// (arena offset, prompt offset, bytes); validate the whole batch first.
    pub fn stage_prompt_rows(&self, arena: &WeightArena<'_>, slot: &mut PromptBuffer<'_>,
        rows: &[(usize, usize, usize)]) -> Result<(), String> {
        if !ptr::eq(arena.buffer._context, self.context) || !ptr::eq(slot.buffer.context, self.context)
            || rows.iter().any(|&(source, destination, bytes)| source.checked_add(bytes).is_none_or(|n| n > arena.bytes())
                || destination.checked_add(bytes).is_none_or(|n| n > slot.buffer.bytes)) {
            return Err("prompt row read outside arena, slot or context".into());
        }
        slot.bytes_mut()?;
        self.context.activate()?;
        let result = (|| {
            for &(source, destination, bytes) in rows {
                unsafe { check(cuMemcpyDtoHAsync_v2(slot.buffer.address.cast::<u8>().add(destination).cast(),
                    arena.buffer.address + source as u64, bytes, self.stream.raw), "prompt row read")?; }
                self.context.pending.set(true);
            }
            Ok(())
        })();
        // Even a partially submitted batch needs a release event.
        let recorded = slot.record(&self.stream);
        if result.is_err() { slot.failed.set(true); }
        result.and(recorded)
    }
    pub fn write_prompt_bytes(&self, vector: &mut DeviceVector<'_>, slot: &PromptBuffer<'_>,
        offset: usize, bytes: usize) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        if slot.failed.get() || slot.pending.is_some() || !ptr::eq(slot.buffer.context, self.context)
            || offset.checked_add(bytes).is_none_or(|n| n > slot.buffer.bytes) || bytes > vector.buffer.bytes {
            return Err("prompt upload outside ready slot or vector".into());
        }
        self.context.activate()?;
        if let Err(error) = unsafe { check(cuMemcpyHtoDAsync_v2(vector.buffer.address, slot.buffer.address.cast::<u8>().add(offset).cast(),
            bytes, self.stream.raw), "prompt row upload") } { slot.failed.set(true); return Err(error); }
        self.context.pending.set(true);
        slot.record(&self.stream)
    }
    /// Reserve a stable pinned exchange extent before registering it with a
    /// transport. The caller releases old registrations before growing it.
    pub fn reserve_exchange(&self, elements: usize) -> Result<(), String> {
        self.synchronize()?;
        self.reserve_upload(byte_count(elements, 4)?)
    }
    /// GPU-visible transport slots avoid host staging copies. Work synchronizes
    /// each slot before NIC access, joins transport, and clears registrations.
    pub fn with_exchange_slots<T>(&self, elements: usize, count: usize,
        work: impl FnOnce(&mut [DeviceVector<'a>], *mut f32) -> Result<T, String>) -> Result<T, String> {
        if count < 2 || count % 2 != 0 { return Err("mapped exchange requires an even slot count".into()); }
        let total = elements.checked_mul(count).ok_or("mapped exchange size overflow")?;
        let timer=Instant::now();
        let owner = Rc::new(Buffer::mapped(self.context, byte_count(total, 4)?)?);
        self.stats.borrow_mut().allocation_ns+=timer.elapsed().as_nanos();
        let bytes = byte_count(elements, 4)?;
        let mut slots = (0..count).map(|slot| Ok(DeviceVector {
            buffer: Buffer::view(&owner, slot * bytes, bytes)?, len: elements, managed: false,
            #[cfg(debug_assertions)] projection_scratch: Default::default(),
        })).collect::<Result<Vec<_>, String>>()?;
        let host = owner.host.unwrap().cast::<f32>();
        unsafe { std::ptr::write_bytes(host, 0, total); }
        let result = work(&mut slots, host);
        self.synchronize()?;
        result
    }
    /// Busy-poll the stream instead of the driver's default wait, whose wake-up
    /// latency dominates short GPU segments between host collectives.
    pub fn set_spin_waits(&self, spin: bool) { self.spin_waits.set(spin); }
    pub fn decode_profile_begin(&self, mode: usize, rank: usize) -> Result<(), String> {
        if let Some(p) = &self.context.decode_profile {
            if p.borrow_mut().begin(mode, rank, self.profiling.get())? { self.profiling.set(true); }
        }
        Ok(())
    }
    /// Lazy plan access: OFF performs only the absent-profiler guard.
    pub fn generic_profile_begin(&self, prefill: bool, plan: impl FnOnce() -> (usize, usize, &'static [&'static str], usize)) -> Result<(), String> {
        let Some(p) = &self.context.decode_profile else { return Ok(()); };
        if !p.borrow_mut().generic_enter() { return Ok(()); }
        if !prefill { if let Err(error) = self.decode_profile_begin(0, 0) { p.borrow_mut().generic_leave(); return Err(error); } }
        if prefill && p.borrow().gpu_enabled() || !prefill && p.borrow().collecting_bytes() {
            let (position, rows, names, ffn_class) = plan();
            self.prefill_profile_begin(position, rows, 0, names);
            p.borrow_mut().generic_begin(!prefill, ffn_class);
        }
        Ok(())
    }
    pub fn generic_profile_end(&self, prefill: bool, ok: bool) -> Result<(), String> {
        let Some(p) = &self.context.decode_profile else { return Ok(()); };
        if !p.borrow_mut().generic_leave() { return Ok(()); }
        let result = self.prefill_profile_end(ok);
        if !prefill { self.decode_profile_end(ok && result.is_ok())?; }
        result
    }
    pub fn generic_profile<R, E: From<String>>(&self, prefill: bool,
        plan: impl FnOnce() -> (usize, usize, &'static [&'static str], usize), run: impl FnOnce() -> Result<R, E>) -> Result<R, E> {
        if self.context.decode_profile.is_none() { return run(); }
        self.generic_profile_begin(prefill, plan).map_err(E::from)?;
        let result = run(); let end = self.generic_profile_end(prefill, result.is_ok()).map_err(E::from);
        end?; result
    }
    pub fn generic_read_bytes(&self, read: impl FnOnce() -> (usize, usize)) {
        if let Some(p) = &self.context.decode_profile { p.borrow_mut().generic_bytes(read); }
    }
    pub fn generic_plan(&self, run: impl FnOnce()) {
        if self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().generic_active()) { run(); }
    }
    pub fn generic_config<R>(&self, read: impl FnOnce() -> R) -> Option<R> {
        self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().gpu_enabled()).then(read)
    }
    pub fn generic_byte_probe(&self, class: usize, read: impl FnOnce() -> Result<usize, String>) -> Result<(), String> {
        if self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().generic_active()) {
            let bytes = read()?; self.generic_read_bytes(|| (class, bytes));
        }
        Ok(())
    }
    pub fn prefill_profile_begin(&self, position: usize, rows: usize, rank: usize, names: &'static [&'static str]) {
        if let Some(p) = &self.context.decode_profile {
            if p.borrow_mut().prefill_begin(position, rows, rank, self.profiling.get(), names) { self.profiling.set(true); }
        }
    }
    pub fn prefill_profile_end(&self, ok: bool) -> Result<(), String> {
        if self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().prefill_active()) {
            let result = self.synchronize();
            let previous = self.context.decode_profile.as_ref().unwrap().borrow_mut().prefill_end(ok && result.is_ok(), self.allocated_bytes()).unwrap();
            self.profiling.set(previous); result?;
        }
        Ok(())
    }
    pub fn prefill_profile_layer(&self, layer: impl FnOnce() -> &'static str) { if let Some(p) = &self.context.decode_profile {
        let mut p = p.borrow_mut(); if p.prefill_active() { p.prefill_layer(layer()); }
    } }
    pub fn prefill_class(&self, class: usize) -> Option<PrefillClass<'_, 'a>> {
        let p = self.context.decode_profile.as_ref()?;
        if !p.borrow().prefill_active() { return None; }
        let previous = p.borrow_mut().prefill_scope(Some(class)); Some(PrefillClass { backend: self, previous })
    }
    pub fn prefill_read_bytes(&self, class: usize, bytes: impl FnOnce() -> usize) {
        if let Some(p) = &self.context.decode_profile { let mut p = p.borrow_mut(); if !p.generic_active() { p.prefill_bytes(class, bytes); } }
    }
    pub fn prefill_collective<R>(&self, run: impl FnOnce() -> Result<R, String>) -> Result<R, String> {
        if !self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().prefill_active()) { return run(); }
        self.synchronize()?;
        let p = self.context.decode_profile.as_ref().unwrap();
        let before = p.borrow().prefill_event_total();
        let _scope = self.prefill_class(9);
        let start = Event::record(&self.stream)?;
        let class = p.borrow_mut().gpu_class(12, false);
        let result = run(); let end = Event::record(&self.stream)?;
        self.synchronize()?;
        let nested = p.borrow().prefill_event_total() - before;
        p.borrow_mut().kernel(class, start.elapsed_ns(&end)?.saturating_sub(nested)); result
    }
    pub fn decode_profile_end(&self, ok: bool) -> Result<(), String> {
        if let Some(p) = &self.context.decode_profile {
            let mut p = p.borrow_mut(); let result = p.end(ok, self.allocated_bytes());
            if result.is_err() { self.profiling.set(p.previous); }
            if let Some(previous) = result? { self.profiling.set(previous); }
        }
        Ok(())
    }
    pub fn decode_weight_read(&self, bytes: impl FnOnce() -> usize) {
        self.decode_read_bytes(if self.profile_kind.get() == Some(13) {3} else {0}, bytes);
    }
    pub fn decode_read_bytes(&self, class: usize, bytes: impl FnOnce() -> usize) {
        if let Some(p) = &self.context.decode_profile { p.borrow_mut().bytes(class, bytes); }
    }
    fn decode_byte_probe(&self, class: usize, read: impl FnOnce() -> Result<usize, String>) -> Result<(), String> {
        if self.context.decode_profile.as_ref().is_some_and(|p| p.borrow().collecting_bytes()) {
            let bytes = read()?; self.decode_read_bytes(class, || bytes);
        }
        Ok(())
    }
    fn decode_words(&self, address: u64, words: usize) -> Result<Vec<u32>, String> {
        let mut out = vec![0u32; words];
        unsafe { self.context.check_execution(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), address, words * 4), "decode byte probe readback")?; }
        Ok(out)
    }
    /// Useful persistent reads, once per query row; selected IDs are finalized on this stream.
    pub fn decode_qsa_read(&self, p: &crate::kernel_params::QsaAttention) -> Result<(), String> {
        self.decode_byte_probe(2, || {
            self.synchronize()?;
            let mut bytes = (128 + usize::from(p.query_heads != 0) * 256 + usize::from(p.key_heads != 0) * 256) * 4;
            if p.blocks > 512 { bytes += p.blocks as usize * 4 * 128 * 4 + 128 * 4; }
            if p.query_heads != 0 { for token in 0..p.tokens as usize {
                let length = p.past as usize + token + 1;
                let ids = if length / 4 > 512 { self.decode_words(p.ids + (token * p.blocks as usize * 4) as u64, 512)? } else { Vec::new() };
                bytes += decode_profile::selected_positions(length, &ids)? * p.key_heads as usize * 256 * 8;
            } }
            Ok(bytes)
        })
    }
    pub fn synchronize(&self) -> Result<(), String> {
        let start = Some(Instant::now());
        self.stream.sync(self.spin_waits.get())?;
        self.context.release_deferred();
        self.uploads.borrow_mut().used = 0;
        let mut stats = self.stats.borrow_mut();
        stats.stream_wait_ns += elapsed(start);
        stats.stream_wait_calls += 1;
        for (kind, class, start, end) in self.timings.borrow_mut().drain(..) {
            let ns = start.elapsed_ns(&end)?;
            if let Some(p) = &self.context.decode_profile { p.borrow_mut().kernel(class, ns); }
            stats.device_kernel_ns[kind] += ns;
            if (14..18).contains(&kind) || (20..23).contains(&kind) || (27..30).contains(&kind) { stats.device_kernel_ns[0] += ns; }
            if (23..27).contains(&kind) { stats.device_kernel_ns[18] += ns; }
        }
        Ok(())
    }
    pub fn fatal_error(&self) -> Option<String> {
        self.context.execution_error.borrow().clone()
            .or_else(|| self.context.cleanup_errors.borrow().first().cloned())
    }
    pub fn vector(&self, len: usize) -> Result<DeviceVector<'a>, String> {
        self.allocate_vector(len, false)
    }

    /// Page-lock and map caller-owned host memory (CU_MEMHOSTREGISTER_DEVICEMAP).
    /// Safety: `host` stays valid and unmoved for `bytes` until the registration drops.
    pub unsafe fn register_host(&self, host: *mut u8, bytes: usize) -> Result<HostRegistration<'a>, String> {
        self.context.activate()?;
        unsafe {
            check(cuMemHostRegister_v2(host.cast(), bytes, 2), "cuMemHostRegister")?;
            let mut device = 0;
            if let Err(error) = check(cuMemHostGetDevicePointer_v2(&mut device, host.cast(), 0), "registered host device pointer") {
                self.context.record_cleanup(cuMemHostUnregister(host.cast()), "cuMemHostUnregister");
                return Err(error);
            }
            Ok(HostRegistration { host, bytes, device, context: self.context })
        }
    }
    /// Copy FP32 values `vector[start..start + out.len() / 4]` into page-locked
    /// host bytes (a HostRegistration), returning their producer completion.
    /// Safety: retain the source and registered bytes, without CPU/NIC access, until the
    /// returned fence completes; drain it even if the collective fails.
    pub unsafe fn download_range(&self, vector: &DeviceVector<'_>, start: usize, out: &mut [u8]) -> Result<Fence<'a>, String> {
        self.check_vector(vector, vector.len)?;
        if out.len() % 4 != 0 || start.checked_add(out.len() / 4).is_none_or(|end| end > vector.len) {
            return Err("download range exceeds vector".into());
        }
        self.context.activate()?;
        unsafe { check(cuMemcpyDtoHAsync_v2(out.as_mut_ptr().cast(), vector.buffer.address + (start * 4) as u64,
            out.len(), self.stream.raw), "registered range read")?; }
        self.context.pending.set(true);
        self.fence()
    }
    /// GPU-visible host storage for a complete rank-major collective. The
    /// caller synchronizes GPU producers before NIC access and retains the
    /// allocation until all registrations and GPU consumers have finished.
    pub fn mapped_vector(&self,len:usize)->Result<DeviceVector<'a>,String> {
        if len==0||len>u32::MAX as usize{return Err("mapped vector shape".into());}
        let bytes=byte_count(len,4)?;
        let owner=Rc::new(Buffer::mapped(self.context,bytes)?);
        Ok(DeviceVector{buffer:Buffer::view(&owner,0,bytes)?,len,managed:false,
            #[cfg(debug_assertions)] projection_scratch: Default::default() })
    }
    /// Only transport setup uses this pointer; the caller orders GPU/host access
    /// while the allocation remains owned by these vectors.
    pub fn mapped_exchange_host(&self, vector: &DeviceVector<'_>) -> Result<*mut f32, String> {
        let owner=vector.buffer.owner.as_ref().ok_or("missing mapped exchange owner")?;
        owner.host.map(|host|host.cast()).ok_or_else(||"exchange is not mapped".into())
    }
    /// Mark the queued stream so the host can wait for the work before this
    /// point while later kernels keep running (a host collective overlaps them).
    pub fn fence(&self) -> Result<Fence<'a>, String> {
        match Event::ready(&self.stream) {
            Ok(event) => Ok(Fence(event)),
            Err(error) => { self.synchronize()?; Err(error) }
        }
    }
    pub fn wait_moe_readiness(&self) -> Result<(), String> {
        match Event::ready_kind(&self.stream, 0) {
            Ok(event) => self.wait_fence(Fence(event)),
            Err(error) => { self.synchronize()?; Err(error) }
        }
    }
    /// Wait for a producer or consumer, leaving later stream work running.
    pub fn wait_fence(&self, fence: Fence<'_>) -> Result<(), String> {
        let start = Some(Instant::now());
        fence.0.wait(self.spin_waits.get())?;
        let mut stats = self.stats.borrow_mut();
        stats.stream_wait_ns += elapsed(start);
        stats.stream_wait_calls += 1;
        Ok(())
    }
    pub fn segmented_vector(&self, maximum: usize, initial: usize) -> Result<DeviceVector<'a>, String> {
        if initial == 0 || initial > maximum || maximum > u32::MAX as usize { return Err("invalid segmented vector capacity".into()); }
        let timer = Some(Instant::now());
        let buffer = Buffer::segmented(self.context, byte_count(maximum, 4)?, byte_count(initial, 4)?)?;
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        Ok(DeviceVector { len: buffer.segment_capacity(byte_count(initial, 4)?)? / 4, buffer, managed: false,
            #[cfg(debug_assertions)] projection_scratch: Default::default() })
    }
    pub fn grow_vector(&self, vector: &mut DeviceVector<'a>, len: usize) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        let timer = Some(Instant::now());
        vector.buffer.grow_segments(byte_count(len, 4)?)?;
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        vector.len = vector.len.max(vector.buffer.segment_capacity(byte_count(len, 4)?)? / 4);
        Ok(())
    }
    /// Finish queued readers before releasing mapped suffix pages.
    pub fn trim_vector(&self, vector: &mut DeviceVector<'a>, len: usize) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        self.synchronize()?;
        vector.buffer.trim_segments(byte_count(len, 4)?)?;
        vector.len = vector.buffer.segment_capacity(byte_count(len, 4)?)? / 4;
        Ok(())
    }
    /// Alias complete prefix allocations and copy only the incomplete tail.
    /// Call writable_from before subsequent writes, including truncate/replay.
    pub fn fork_vector(&self, source: &DeviceVector<'a>, committed: usize, needed: usize) -> Result<DeviceVector<'a>, String> {
        self.check_vector(source, source.len)?;
        let timer = Some(Instant::now());
        let (buffer, shared) = source.buffer.fork_segments(byte_count(committed, 4)?, byte_count(needed, 4)?)?;
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        let mut fork = DeviceVector { len: buffer.segment_capacity(byte_count(needed, 4)?)? / 4, buffer, managed: false,
            #[cfg(debug_assertions)] projection_scratch: Default::default() };
        self.copy_range(source, &mut fork, shared / 4, shared / 4, committed - shared / 4)?;
        Ok(fork)
    }
    /// Restore contiguous views for callers without split-cache attention.
    pub fn fork_vectors(&self, sources: &[&DeviceVector<'a>], committed: usize, needed: usize, maximum: usize) -> Result<Vec<DeviceVector<'a>>, String> {
        let paged = byte_count(needed, 4)? >= 2 * 1024 * 1024;
        if paged && sources.iter().all(|source| source.is_segmented()) {
            return sources.iter().map(|source| self.fork_vector(source, committed, needed)).collect();
        }
        // Materializing a compact prefix must enter append-only pages before
        // it can become a second long arena during the next growth.
        let mut copies = if paged {
            sources.iter().map(|_| self.segmented_vector(maximum, needed)).collect::<Result<Vec<_>, String>>()?
        } else { self.kv_arena(&vec![needed; sources.len()])? };
        for (source, copy) in sources.iter().zip(&mut copies) { self.copy_prefix(source, copy, committed)?; }
        Ok(copies)
    }
    /// Allocate related vectors together. Views retain their arena independently.
    pub fn vector_arena(&self, lengths: &[usize]) -> Result<Vec<DeviceVector<'a>>, String> {
        self.arena(lengths, false)
    }
    /// Conversation KV arenas retain released chunks in the physical reserve.
    pub fn kv_arena(&self, lengths: &[usize]) -> Result<Vec<DeviceVector<'a>>, String> {
        self.arena(lengths, true)
    }
    fn arena(&self, lengths: &[usize], kv: bool) -> Result<Vec<DeviceVector<'a>>, String> {
        let sizes = lengths.iter().map(|&len| {
            if len == 0 || len > u32::MAX as usize { return Err("vector outside kernel ABI".into()); }
            Ok(byte_count(len, 4)?.div_ceil(64) * 64)
        }).collect::<Result<Vec<_>, String>>()?;
        let bytes = sizes.iter().try_fold(0usize, |n, &s| n.checked_add(s).ok_or("vector arena overflow"))?;
        let timer = Some(Instant::now());
        let arena = Rc::new(if kv { Buffer::kv_arena(self.context, bytes)? } else { Buffer::pooled(self.context, bytes)? });
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        let mut offset = 0;
        lengths.iter().zip(sizes).map(|(&len, size)| {
            let buffer = Buffer::view(&arena, offset, len * 4)?;
            offset += size;
            Ok(DeviceVector { buffer, len, managed: false,
                #[cfg(debug_assertions)] projection_scratch: Default::default() })
        }).collect()
    }
    /// Reusable scratch views, with offsets and lengths in FP32 elements.
    /// Overlapping views must have disjoint live ranges on the compute stream.
    pub fn vector_workspace(&self, views: &[(usize, usize)]) -> Result<Vec<DeviceVector<'a>>, String> {
        let mut end = 0;
        for &(offset, len) in views {
            if len == 0 || len > u32::MAX as usize || offset % 16 != 0 {
                return Err("invalid workspace view".into());
            }
            end = end.max(offset.checked_add(len).ok_or("workspace size overflow")?);
        }
        let timer = Some(Instant::now());
        let arena = Rc::new(Buffer::pooled(self.context, byte_count(end, 4)?)?);
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        // Workspace ranges may overlap, so their invalidation counter is shared.
        #[cfg(debug_assertions)] let projection_scratch: Rc<Cell<(u64, Option<u64>)>> = Default::default();
        views.iter().map(|&(offset, len)| Ok(DeviceVector {
            buffer: Buffer::view(&arena, byte_count(offset, 4)?, byte_count(len, 4)?)?, len, managed: false,
            #[cfg(debug_assertions)] projection_scratch: projection_scratch.clone(),
        })).collect()
    }
    /// CPU-readable unified storage; host access is allowed only after sync.
    pub fn shared_vector(&self, len: usize) -> Result<DeviceVector<'a>, String> {
        self.allocate_vector(len, true)
    }
    fn allocate_vector(&self, len: usize, managed: bool) -> Result<DeviceVector<'a>, String> {
        if len == 0 || len > u32::MAX as usize { return Err("vector outside kernel ABI".into()); }
        let timer = Some(Instant::now());
        let bytes = byte_count(len, 4)?;
        let buffer = if managed { Buffer::allocate(self.context, bytes, true)? } else { Buffer::pooled(self.context, bytes)? };
        self.stats.borrow_mut().allocation_ns += elapsed(timer);
        Ok(DeviceVector { buffer, len, managed,
            #[cfg(debug_assertions)] projection_scratch: Default::default() })
    }
    pub fn upload_vector(&self, values: &[f32]) -> Result<DeviceVector<'a>, String> {
        let mut vector = self.vector(values.len())?;
        self.write_vector(&mut vector, values)?;
        Ok(vector)
    }
    fn check_vector(&self, vector: &DeviceVector<'_>, len: usize) -> Result<(), String> {
        let same_context = ptr::eq(vector.buffer._context, self.context);
        crate::check_vector_shape(vector.len, vector.buffer.bytes, same_context, len)
    }
    /// Validate a working prefix without changing a reusable buffer's shape.
    fn check_vector_prefix(&self, vector: &DeviceVector<'_>, len: usize) -> Result<(), String> {
        if !std::ptr::eq(vector.buffer._context, self.context) || vector.len < len {
            return Err(format!("device vector prefix requires {len} elements, has {}", vector.len));
        }
        Ok(())
    }
    pub fn write_vector(&self, vector: &mut DeviceVector<'_>, values: &[f32]) -> Result<(), String> {
        self.check_vector(vector, values.len())?;
        self.write_buffer(&vector.buffer, 0, values)
    }
    /// Upload a compact official-dtype row without converting its stored bits.
    pub fn write_vector_bytes(
        &self,
        vector: &mut DeviceVector<'_>,
        values: &[u8],
    ) -> Result<(), String> {
        crate::check_vector_upload(
            vector.len,
            vector.buffer.bytes,
            ptr::eq(vector.buffer._context, self.context),
            values.len(),
        )?;
        self.write_buffer(&vector.buffer, 0, values)
    }
    /// Launch a model-specific Rust kernel from the supplied module.
    /// The caller must match its ABI and keep every referenced allocation alive
    /// until the ordered stream has completed the kernel.
    pub unsafe fn launch_flash(&self, name: &str, blocks: u32, threads: u32,
        args: &mut [*mut c_void]) -> Result<(), String> {
        unsafe { self.launch_flash_shared(name, blocks, threads, 0, args) }
    }
    /// Model-specific kernels whose shared tiles exceed the static 48 KiB cap.
    pub unsafe fn launch_flash_shared(&self, name: &str, blocks: u32, threads: u32, shared: u32,
        args: &mut [*mut c_void]) -> Result<(), String> {
        unsafe { self.launch_flash_rows(name, blocks, 1, threads, shared, args) }
    }
    /// Independent prompt descriptors occupy the second grid dimension.
    pub unsafe fn launch_flash_rows(&self, name: &str, blocks: u32, rows: u32, threads: u32, shared: u32,
        args: &mut [*mut c_void]) -> Result<(), String> {
        unsafe { self.launch_flash_grid(name, blocks, rows, 1, threads, shared, args) }
    }
    /// Kernels with separate slot planes and sessions use all three grid axes.
    pub unsafe fn launch_flash_grid(
        &self,
        name: &str,
        blocks: u32,
        rows: u32,
        planes: u32,
        threads: u32,
        shared: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), String> {
        let function = if shared == 0 {
            self.module.function(name)?
        } else {
            self.module.function_with_shared(name, shared)?
        };
        let kind = self
            .profile_kind
            .get()
            .unwrap_or_else(|| next_profile::flash_kind(name));
        self.profile_phase(kind, || unsafe {
            self.enqueue_dims_shared(function, blocks, rows, planes, threads, shared, args)
        })
    }
    fn reserve_upload(&self, bytes: usize) -> Result<(), String> {
        let bytes = bytes.div_ceil(64) * 64;
        let full = { let staging = self.uploads.try_borrow().map_err(|_| "pinned transfer already in use")?;
            bytes > staging.capacity - staging.used };
        if full {
            self.synchronize()?;
            let mut staging = self.uploads.borrow_mut();
            if bytes > staging.capacity {
                let capacity = bytes.checked_next_power_of_two().ok_or("pinned staging size overflow")?;
                *staging = UploadStaging { buffer: Pinned::new(self.context, capacity)?, capacity, used: 0 };
            }
        }
        Ok(())
    }
    // Caller supplies a pinned reader slot. Wait before returning it to the reader.
    pub(super) unsafe fn write_pinned_bytes(&self, buffer: &Buffer<'_>, offset: usize, bytes: &[u8]) -> Result<(), String> {
        self.queue_pinned_bytes(buffer, offset, bytes)?;
        self.synchronize()
    }
    /// Caller keeps the pinned source immutable until its slot event completes.
    pub(super) unsafe fn queue_pinned_bytes(&self, buffer: &Buffer<'_>, offset: usize, bytes: &[u8]) -> Result<(), String> {
        if !ptr::eq(buffer._context, self.context)
            || offset.checked_add(bytes.len()).is_none_or(|end| end > buffer.bytes) {
            return Err("pinned weight upload outside arena or context".into());
        }
        self.context.activate()?;
        check(cuMemcpyHtoDAsync_v2(buffer.address + offset as u64, bytes.as_ptr().cast(), bytes.len(), self.stream.raw), "packed weight upload")?;
        self.context.pending.set(true);
        Ok(())
    }
    // Staging ranges stay immutable until the compute stream completes. A token's
    // uploads are ordered with its kernels and need no additional stream wait.
    fn write_buffer<T: DeviceElement>(&self, buffer: &Buffer<'_>, offset: usize, values: &[T]) -> Result<(), String> {
        let bytes = std::mem::size_of_val(values);
        if offset.checked_add(bytes).is_none_or(|end| end > buffer.bytes) { return Err("upload exceeds buffer".into()); }
        if bytes == 0 { return Ok(()); }
        self.reserve_upload(bytes)?;
        let mut staging = self.uploads.borrow_mut();
        self.context.activate()?;
        unsafe {
            let host = staging.buffer.address.cast::<u8>().add(staging.used);
            ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), host, bytes);
            check(cuMemcpyHtoDAsync_v2(buffer.address + offset as u64, host.cast(), bytes, self.stream.raw), "stream upload")?;
        }
        staging.used += bytes.div_ceil(64) * 64;
        self.context.pending.set(true);
        Ok(())
    }
    /// Restart typed routing status and apply only the startup qualification hook.
    pub fn clear_moe_routing(&self, status: &mut DeviceVector<'_>, rows: usize) -> Result<(), String> {
        if rows == 0 || rows > status.len { return Err("routing status active row capacity".into()); }
        crate::routing::clear(rows, None, |row| match row { None => self.zero_vector(status), Some(row) => unsafe {
            check(cuMemsetD8Async(status.buffer.address + (row * 4) as u64, 1, 4, self.stream.raw), "routing row injection") } })
    }
    /// Clear state on the compute stream without a host zero buffer or upload.
    pub fn zero_vector(&self, vector: &mut DeviceVector<'_>) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        self.context.activate()?;
        unsafe { check(cuMemsetD8Async(vector.buffer.address, 0, byte_count(vector.len, 4)?, self.stream.raw), "state clear")?; }
        self.context.pending.set(true);
        Ok(())
    }
    /// Clear a contiguous activation range without creating an arena view.
    /// The allocation and logical shape stay unchanged on the ordered stream.
    pub fn zero_vector_range(&self, vector: &mut DeviceVector<'_>, range: std::ops::Range<usize>) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        let bytes = crate::vector_byte_range(vector.len, range)?;
        if bytes.is_empty() { return Ok(()); }
        if !vector.writable_from(bytes.start / 4) { return Err("device clear range is shared".into()); }
        self.context.activate()?;
        unsafe { check(cuMemsetD8Async(vector.buffer.address + bytes.start as u64, 0, bytes.len(),
            self.stream.raw), "state range clear")?; }
        self.context.pending.set(true);
        Ok(())
    }
    /// Exchange activations in reusable pinned memory at a compute-stream boundary.
    /// The callback can gather/reduce in place; its result is queued back to the GPU.
    pub fn exchange_vector(&self, vector: &mut DeviceVector<'_>, exchange: impl FnOnce(&mut [f32]) -> Result<(), String>) -> Result<(), String> {
        self.check_vector(vector, vector.len)?;
        let bytes = byte_count(vector.len, 4)?;
        self.reserve_upload(bytes)?;
        let host = { let staging = self.uploads.borrow();
            unsafe { staging.buffer.address.cast::<u8>().add(staging.used) } };
        self.context.activate()?;
        unsafe { check(cuMemcpyDtoHAsync_v2(host.cast(), vector.buffer.address, bytes, self.stream.raw), "pinned activation read")?; }
        self.context.pending.set(true);
        self.synchronize()?;
        let mut staging = self.uploads.borrow_mut();
        // Synchronization reset the cursor; the download can occupy a later range.
        let offset = host as usize - staging.buffer.address as usize;
        exchange(unsafe { std::slice::from_raw_parts_mut(host.cast::<f32>(), vector.len) })?;
        unsafe { check(cuMemcpyHtoDAsync_v2(vector.buffer.address, host.cast(), bytes, self.stream.raw), "pinned activation write")?; }
        staging.used = offset + bytes.div_ceil(64) * 64;
        self.context.pending.set(true);
        Ok(())
    }

    /// Read related operands at one ordered stream boundary into disjoint pinned spans.
    pub fn read_vectors_values(&self, vectors: &[DeviceVector<'_>]) -> Result<Vec<Vec<f32>>, String> {
        self.read_vector_refs(&vectors.iter().collect::<Vec<_>>())
    }
    fn read_vector_refs(&self, vectors: &[&DeviceVector<'_>]) -> Result<Vec<Vec<f32>>, String> {
        let bytes=vectors.iter().try_fold(0usize,|total,v| {
            self.check_vector(v,v.len)?;
            total.checked_add(byte_count(v.len,4)?).ok_or_else(||"read cohort extent overflow".to_owned())
        })?;
        self.reserve_upload(bytes)?;
        let host={let staging=self.uploads.borrow(); unsafe { staging.buffer.address.cast::<u8>().add(staging.used) }};
        let _wait=self.context.progress.wait(WaitKind::Readback);
        let download=self.profiling.get().then(Instant::now);
        self.context.activate()?;
        let copied=(|| {
            let mut offset=0;
            for v in vectors {
                if v.len!=0 { unsafe { check(cuMemcpyDtoHAsync_v2(host.add(offset).cast(),v.buffer.address,v.len*4,self.stream.raw),"pinned cohort read")?; }
                    self.context.pending.set(true); }
                offset+=v.len*4;
            }
            Ok(())
        })();
        // Drain even a partially submitted cohort before its pinned prefix can be reused.
        let waited=self.synchronize(); copied.and(waited)?;
        let mut offset=0;
        let output=vectors.iter().map(|v| {
            let values=unsafe { std::slice::from_raw_parts(host.add(offset).cast::<f32>(),v.len) }.to_vec();
            offset+=v.len*4; values
        }).collect();
        if self.profiling.get() { self.stats.borrow_mut().output_download_ns+=elapsed(download); }
        Ok(output)
    }

    pub fn read_vector(&self, vector: &DeviceVector<'_>) -> Result<Vec<f32>, String> {
        let output = self.read_vector_values(vector)?;
        if output.iter().any(|v| !v.is_finite()) { return Err("nonfinite device output".into()); }
        Ok(output)
    }
    /// Caller validates each row so a bad result can be attributed to its request.
    pub fn read_vector_values(&self, vector: &DeviceVector<'_>) -> Result<Vec<f32>, String> {
        self.check_vector(vector, vector.len)?;
        // A copy on the producer stream already follows every writer. Keep its
        // pinned destination disjoint from pending uploads and wait only after
        // the copy. Large reads retain the bounded, chunked staging path below.
        if !vector.managed && byte_count(vector.len, 4)? <= self.uploads.borrow().capacity {
            return self.read_vector_refs(&[vector]).map(|mut v| v.remove(0));
        }
        let _wait = self.context.progress.wait(WaitKind::Readback);
        self.synchronize()?;
        let download = self.profiling.get().then(Instant::now);
        let mut output = vec![0.0; vector.len];
        if vector.managed {
            unsafe { output.copy_from_slice(std::slice::from_raw_parts(vector.buffer.address as *const f32, vector.len)); }
        } else {
            // Pageable driver copies can stall while staging full-vocabulary
            // logits. Reuse the pinned upload prefix after its reads finish.
            let (host, capacity) = { let staging = self.uploads.borrow();
                (staging.buffer.address, staging.capacity / 4) };
            self.context.activate()?;
            for (chunk, values) in output.chunks_mut(capacity).enumerate() {
                unsafe { check(cuMemcpyDtoHAsync_v2(host, vector.buffer.address + (chunk * capacity * 4) as u64,
                    std::mem::size_of_val(values), self.stream.raw), "pinned vector read")?; }
                self.context.pending.set(true);
                self.synchronize()?;
                unsafe { values.copy_from_slice(std::slice::from_raw_parts(host.cast::<f32>(), values.len())); }
            }
        }
        if self.profiling.get() { self.stats.borrow_mut().output_download_ns += elapsed(download); }
        Ok(output)
    }

    pub fn copy_prefix(&self, source: &DeviceVector<'_>, destination: &mut DeviceVector<'_>, len: usize) -> Result<(), String> {
        self.copy_range(source, destination, 0, 0, len)
    }
    /// Copy an activation slice on the ordered stream without host staging.
    pub fn copy_range(&self, source: &DeviceVector<'_>, destination: &mut DeviceVector<'_>, source_offset: usize, destination_offset: usize, len: usize) -> Result<(), String> {
        self.check_vector(source, source.len)?;
        self.check_vector(destination, destination.len)?;
        if source_offset.checked_add(len).is_none_or(|n| n > source.len) || destination_offset.checked_add(len).is_none_or(|n| n > destination.len) { return Err("device copy exceeds vector".into()); }
        if len == 0 { return Ok(()); }
        self.context.activate()?;
        let timer = if self.profiling.get() { Some(Event::record(&self.stream)?) } else { None };
        self.context.decode_enqueue();
        unsafe { check(cuMemcpyDtoDAsync_v2(destination.buffer.address + (destination_offset * 4) as u64, source.buffer.address + (source_offset * 4) as u64, byte_count(len, 4)?, self.stream.raw), "cuMemcpyDtoDAsync_v2")?; }
        if let Some(start) = timer {
            let class = self.context.decode_profile.as_ref().and_then(|p| p.borrow_mut().gpu_class(12, false));
            self.timings.borrow_mut().push((12, class, start, Event::record(&self.stream)?));
        }
        self.context.pending.set(true);
        Ok(())
    }
    unsafe fn enqueue(&self, function: Handle, elements: usize, args: &mut [*mut c_void]) -> Result<(), String> {
        self.enqueue_grid(function, elements, 1, 1, args)
    }
    unsafe fn enqueue_grid(&self, function: Handle, elements: usize, rows: u32, planes: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        let blocks = launch_blocks(elements, 128)?;
        self.enqueue_dims(function, blocks, rows, planes, 128, args)
    }
    unsafe fn enqueue_dims(&self, function: Handle, blocks: u32, rows: u32, planes: u32, threads: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        self.enqueue_dims_shared(function, blocks, rows, planes, threads, 0, args)
    }
    /// Prompt rows keep the prefill GEMM order at every row count, so prompt KV
    /// and logits never depend on how a prompt was split into chunks or cached
    /// suffixes. Decode keeps its own order for 16 rows or fewer.
    pub fn prompt_order(&self, on: bool) -> PromptOrder<'_, 'a> {
        PromptOrder { backend: self, previous: self.prompt_order.replace(on) }
    }
    pub(crate) fn decode_rows(&self, tokens: usize) -> bool { tokens <= 64 && !self.prompt_order.get() }
    unsafe fn enqueue_dims_shared(&self, function: Handle, blocks: u32, rows: u32, planes: u32, threads: u32, shared: u32, args: &mut [*mut c_void]) -> Result<(), String> {
        if blocks == 0 { return Ok(()); }
        if threads == 0 || threads > 1024 || threads % 32 != 0 { return Err("kernel launch requires complete warps".into()); }
        if rows == 0 || rows > 65535 || planes == 0 || planes > 65535 { return Err("grid dimension exceeds driver limit".into()); }
        self.context.activate()?;
        let function = if self.capture_position.get() != 0 {
            self.batch.borrow().as_ref().and_then(|k| k.graph.iter().find(|(normal, _)| *normal == function).map(|(_, graph)| *graph)).unwrap_or(function)
        } else { function };
        let profile_start = self.profiling.get().then(Instant::now);
        let timer = if self.profiling.get() { Some(Event::record(&self.stream)?) } else { None };
        let launch_start = self.profiling.get().then(Instant::now);
        self.context.decode_enqueue(); self.context.receipt_launch(function, threads, shared)?;
        self.context.check_execution(launch_kernel(self.programmatic.replace(false), function, [blocks, rows, planes], threads,
            shared, self.stream.raw, args.as_mut_ptr()), "queued cuLaunchKernel")?;
        let launch_ns = elapsed(launch_start);
        self.context.pending.set(true);
        if let Some(start) = timer {
            let end = Event::record(&self.stream)?;
            let batch_kind = self.batch.borrow().as_ref().and_then(|k| k.kind(function))
                .or_else(|| self.multi.borrow().as_ref().and_then(|k| k.kind(function)))
                .or_else(|| self.moe.borrow().as_ref().and_then(|k| k.kind(function)))
                .or_else(|| self.fp8.borrow().as_ref().and_then(|k| k.kind(function)));
            let kind = if let Some(kind) = self.profile_kind.get().or(batch_kind) { kind } else if function == self.matvec_function.get() || Some(function) == self.vec8.get() { 0 } else {
                let k = self.layer_kernels()?;
                if [k.model_rms, k.model_rms_apply, k.rms, k.residual_rms, k.rms_apply, k.rms_fused, k.residual_rms_fused].contains(&function) { 1 }
                else if function == k.qk_rope || function == k.model_qkv { 2 } else if function == k.bounded_flash || function == k.bounded_scores_grouped || function == k.bounded_scores || function == k.scores || function == k.scores_gqa4 { 3 }
                else if function == k.softmax || function == k.bounded_softmax { 4 } else if function == k.bounded_split || function == k.bounded_merge || function == k.values || function == k.value_split || function == k.value_gqa4 || function == k.value_merge { 5 }
                else if function == k.swiglu { 6 } else if function == k.residual { 7 }
                else if function == k.embedding { 8 } else { 9 }
            };
            let class = self.context.decode_profile.as_ref().and_then(|p| { let mut p = p.borrow_mut(); let kind = if p.generic_active() { batch_kind.filter(|k| [18, 19, 21, 22].contains(k)).unwrap_or(kind) } else { kind }; p.gpu_class(kind,
                self.fp8.borrow().as_ref().is_some_and(|k| k.decode_expert(function))) });
            if let Some(p) = &self.context.decode_profile { p.borrow_mut().prefill_enqueue(class, launch_start.unwrap(), launch_ns); }
            self.timings.borrow_mut().push((kind, class, start, end));
            let mut stats = self.stats.borrow_mut();
            stats.queued_calls += 1;
            stats.device_launches[kind] += 1;
            if (14..18).contains(&kind) || (20..23).contains(&kind) || (27..30).contains(&kind) { stats.device_launches[0] += 1; }
            if (23..27).contains(&kind) { stats.device_launches[18] += 1; }
            stats.host_launch_ns += launch_ns;
            stats.host_profile_ns += elapsed(profile_start).saturating_sub(launch_ns);
        }
        Ok(())
    }
    pub fn mode(&self) -> MatvecMode { self.mode }
    pub fn use_vec8(&self, enabled: bool) -> Result<(), String> {
        if enabled && self.vec8.get().is_none() { return Err("PTX has no vec8 kernel".into()); }
        self.use_vec8.set(enabled);
        Ok(())
    }
    pub fn allocated_bytes(&self) -> usize { self.context.allocated.get() }
    pub fn peak_allocated_bytes(&self) -> usize { self.context.peak_allocated.get() }
    pub fn graph_storage_bytes(&self) -> usize { self.context.graph_storage.get() }
    pub fn peak_graph_storage_bytes(&self) -> usize { self.context.peak_graph_storage.get() }
    pub fn accounted_bytes(&self) -> usize { self.context.accounted_bytes() }
    pub fn peak_accounted_bytes(&self) -> usize { self.context.peak_accounted.get() }
    /// Enable bounded physical KV reuse for the server's memory safety margin.
    pub fn configure_kv_reserve(&self, safety_margin: usize) {
        self.context.kv_reserve.borrow_mut().configure(safety_margin);
    }
    /// VADE_VECTOR_POOL: freed device vectors (vector, vector_arena, vector_workspace) wait,
    /// keyed by exact byte length, for the next allocation of that length, under the KV
    /// reserve's limits (cap from kv_conversation_ended, release on demand, under host memory
    /// pressure and on clear). The KV reserve hooks below drive both reserves.
    pub fn configure_vector_pool(&self, safety_margin: usize) {
        self.context.vector_pool.borrow_mut().configure(safety_margin);
    }
    pub fn vector_pool_configured(&self) -> bool { self.context.vector_pool.borrow().configured() }
    pub fn kv_conversation_ended(&self, bytes: usize) {
        self.context.kv_reserve.borrow_mut().conversation_ended(bytes);
        self.context.vector_pool.borrow_mut().conversation_ended(bytes);
    }
    /// (host bytes left before KV reserve pressure, bytes the reserve may still keep); None: off.
    pub fn kv_reserve_room(&self) -> Result<Option<(i128, usize)>, String> {
        self.context.activate()?;
        self.context.kv_reserve.borrow_mut().room(self.context)
    }
    /// The server calls this while running and idle; no per-token allocation.
    pub fn maintain_kv_reserve(&self, idle: bool) -> Result<(), String> {
        self.context.activate()?;
        self.context.vector_pool.borrow_mut().maintain(self.context, idle)?;
        self.context.kv_reserve.borrow_mut().maintain(self.context, idle)
    }
    /// A completed session release invalidates the cached pressure verdict. Poll
    /// both reserves now, without weakening the margin, caps or ownership rules.
    pub fn refresh_kv_pressure(&self) -> Result<(), String> {
        self.context.activate()?;
        self.context.vector_pool.borrow_mut().refresh_pressure(self.context)?;
        self.context.kv_reserve.borrow_mut().refresh_pressure(self.context)
    }
    /// Model swaps and shutdown, after all sessions/graphs drop, release reserves.
    pub fn clear_kv_reserve(&self) -> Result<(), String> {
        self.synchronize()?;
        self.decode_partials.borrow_mut().take();
        self.context.clear_reserves();
        Ok(())
    }
    /// Return released loader/tokenizer heap pages to the OS between model
    /// lifetimes. Live host allocations and CUDA buffers remain owned as usual.
    pub fn trim_host_memory(&self) {
        extern "C" { fn malloc_trim(pad: usize) -> c_int; }
        unsafe { malloc_trim(0); }
    }
    pub fn set_profiling(&self, enabled: bool) { self.profiling.set(enabled); }
    pub fn profile_host_pack<R>(&self, run: impl FnOnce() -> R) -> R {
        let start = self.profiling.get().then(Instant::now);
        let result = run();
        if start.is_some() { self.stats.borrow_mut().host_pack_ns += elapsed(start); }
        result
    }
    pub fn record_host_pack(&self, ns: u128) {
        if self.profiling.get() { self.stats.borrow_mut().host_pack_ns += ns; }
    }
    pub fn is_profiling(&self) -> bool { self.profiling.get() }
    /// Byte probes synchronize and the rejection hook advances on the host.
    pub fn graph_capture_ready(&self) -> bool {
        !self.is_profiling() && self.context.decode_profile.is_none()
            && std::env::var_os("VADE_TEST_ROUTING_REJECT").is_none()
    }
    pub fn stats(&self) -> BackendStats { *self.stats.borrow() }
}
impl Drop for Backend<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() { self.context.cleanup_errors.borrow_mut().push(error); }
    }
}


impl<'a> Backend<'a> {
    pub fn exchange_slot(&self, capacity: usize) -> Result<ExchangeSlot<'a>, String> {
        if capacity == 0 || capacity > u32::MAX as usize { return Err("exchange slot outside kernel ABI".into()); }
        Ok(ExchangeSlot { buffer: Pinned::new(self.context, byte_count(capacity, 4)?)?, capacity, len: 0,
            event: Event::ready_kind(&self.stream, 1)?, queued: false })
    }
    /// Queue the download on the compute stream and mark its completion. A
    /// download left unread by a failed forward is simply superseded.
    pub fn exchange_begin(&self, slot: &mut ExchangeSlot<'a>, vector: &DeviceVector<'_>) -> Result<(), String> {
        self.prefill_read_bytes(9, || vector.len * 4);
        self.check_vector(vector, vector.len)?;
        if vector.len > slot.capacity { return Err("exchange slot is too small".into()); }
        self.context.activate()?;
        unsafe { check(cuMemcpyDtoHAsync_v2(slot.buffer.address, vector.buffer.address, byte_count(vector.len, 4)?, self.stream.raw), "pinned exchange read")?; }
        slot.event.record_again(&self.stream)?;
        self.context.pending.set(true);
        slot.len = vector.len;
        slot.queued = true;
        Ok(())
    }
    /// Wait for the queued download alone and expose it to the collective.
    pub fn exchange_wait<'s>(&self, slot: &'s mut ExchangeSlot<'a>) -> Result<&'s mut [f32], String> {
        if !slot.queued { return Err("exchange slot has no queued download".into()); }
        let start = Some(Instant::now());
        slot.event.wait(self.spin_waits.get())?;
        slot.queued = false;
        let mut stats = self.stats.borrow_mut();
        stats.stream_wait_ns += elapsed(start);
        stats.stream_wait_calls += 1;
        Ok(unsafe { std::slice::from_raw_parts_mut(slot.buffer.address.cast::<f32>(), slot.len) })
    }
    /// Queue the exchanged values back on the compute stream, ahead of later launches.
    pub fn exchange_upload(&self, slot: &ExchangeSlot<'a>, vector: &mut DeviceVector<'_>) -> Result<(), String> {
        self.prefill_read_bytes(9, || slot.len * 4);
        self.check_vector(vector, slot.len)?;
        if slot.queued { return Err("exchange slot download is still queued".into()); }
        self.context.activate()?;
        self.context.decode_enqueue();
        unsafe { check(cuMemcpyHtoDAsync_v2(vector.buffer.address, slot.buffer.address.cast_const(), byte_count(slot.len, 4)?, self.stream.raw), "pinned exchange write")?; }
        self.context.pending.set(true);
        Ok(())
    }
}
