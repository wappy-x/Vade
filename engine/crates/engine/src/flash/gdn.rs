//! Rank-local GatedDeltaNet: official BF16 projections, FP32 chunked prefill
//! and cached recurrence. Output projections are partials to reduce before HC.
use super::Weights;
use engine_model::safetensors::Dtype;
use spark_gpu_probe::gdn_params::{GdnConvBatch, GdnPrepareBatch, GdnSessionRow, GdnStepBatch};
use spark_gpu_probe::{
    driver::{Backend, DeviceVector},
    kernel_params::{
        GDN_SCAN_COLUMNS, GDN_SCAN_SHARED_BYTES, GDN_SCAN_THREADS, GDN_SYSTEM_SHARED_BYTES,
        GDN_SYSTEM_THREADS, GdnChunkPrompts, GdnGate, GdnHistory, GdnPrepare, GdnScanPrompts,
    },
};
use std::{ffi::c_void, marker::PhantomData};

type Result<T> = std::result::Result<T, String>;
const HIDDEN: usize = 2560;
const DIM: usize = 128;
const CHUNK: usize = 64;

#[path = "gdn_workspace.rs"]
mod workspace;
#[path = "gdn_replay.rs"]
mod replay;
#[path = "gdn_cohort.rs"]
mod cohort;
pub(crate) use cohort::{cohort_copies, cohort_kernel_entries, cohort_kernels, cohort_verify, hybrid_copy_rows, launch_cohort, CohortLaunches, COHORT_LAUNCH_BYTES, COPY_ROWS_PER_CHAT, COPY_ROW_WORDS};
#[path = "gdn_arena.rs"]
mod arena;
#[path = "gdn_half.rs"] mod half;
pub(crate) use arena::session_arena;
pub(super) use replay::GdnReplay;

/// Chunk-form GDN for Flash-Next (default on;
/// VADE_GDN_CHUNK=0 = the previous sequential scan). On: states carry the open 64-row chunk
/// after S (gdn_params::chunk_state_floats); prompt, MTP verify, replay and
/// pooled decode all run the flash_next_gdn_chunk_* kernels at absolute positions.
pub(crate) fn chunk_form() -> bool { engine_model::flash::gdn_chunk_form() }
/// MTP verify GDN layers: the replay tape's three projection copies, the conv history
/// snapshot and the scan descriptor upload go in one flash_gdn_verify_copy launch instead
/// of four device copies and one host upload (pure data movement, same bytes).
/// VADE_GDN_VERIFY_COPY=0 keeps the separate copies.
/// VADE_GDN_PREP_SPLIT (default on; =0 keeps one warp per key group): see execute's prep.
pub(crate) fn prep_split(b: &Backend<'_>) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_GDN_PREP_SPLIT").as_deref() != Ok("0")) && b.has_flash("flash_gdn_conv_prepare_tokens_khs")
}
pub(crate) fn fused_verify_copy(b: &Backend<'_>, descriptor: &DeviceVector<'_>) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VADE_GDN_VERIFY_COPY").is_ok_and(|v| v == "0"))
        && descriptor.device_address() % 8 == 0 && !descriptor.is_segmented() && descriptor.len() >= 6
        && b.has_flash("flash_gdn_verify_copy")
}
/// Chunk form: an MTP verify window of at most CHUNK_ROWS_LIMIT rows that
/// stays inside one chunk (no row 63) runs the rows kernel, which writes only
/// the window's own tail rows (never S, never an earlier row). Its entry needs
/// no state copy, and a rejection exchanges only the conv history: the replay
/// rewrites the accepted rows with the same bytes (verify == replay, proven by
/// the kbench split cases) and the rejected rows are stale by position.
pub(crate) fn shallow_verify(base: usize, tokens: usize) -> bool {
    chunk_form() && tokens <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT && base % CHUNK + tokens < CHUNK
}
/// Rows per chunk-form prompt launch pair (workspace bound; any cut gives the same bytes).
/// 8,192 = the widest prompt tile (next_rows::prompt_tile): every tile runs one prep +
/// scan pair (kbench gdn_cpipeq-8k-p4096 / -p8192); the A/P workspace is sized by the
/// tile (rows.min(CHUNK_PIECE)), so the default 4,096 tile funds what it did at 4,096
/// and VADE_PROMPT_TILE=2048 sizes and runs exactly as before.
const CHUNK_PIECE: usize = 8192;
/// flash_next_gdn_chunk_scan dynamic shared bytes (kernel SCAN_BYTES).
const CHUNK_SCAN_BYTES: u32 = spark_gpu_probe::gdn_params::CHUNK_SCAN_BYTES;
const CHUNK_ROWS_BYTES: u32 = spark_gpu_probe::gdn_params::CHUNK_ROWS_BYTES;

pub(crate) enum Arg { Ptr(u64), U32(u32), F32(f32) }
use Arg::{F32, Ptr, U32};
pub(crate) fn launch(b: &Backend<'_>, name: &str, threads: usize, args: &mut [Arg]) -> Result<()> {
    launch_threads(b, name, threads, 128, args)
}
pub(crate) fn launch_threads(b: &Backend<'_>, name: &str, threads: usize, block: u32, args: &mut [Arg]) -> Result<()> {
    launch_rows(b, name, threads, 1, block, 0, args)
}
fn launch_rows(b: &Backend<'_>, name: &str, threads: usize, rows: usize, block: u32, shared: u32, args: &mut [Arg]) -> Result<()> {
    if threads == 0 { return Ok(()); }
    let mut pointers = [std::ptr::null_mut(); 20];
    for (pointer, arg) in pointers.iter_mut().zip(args.iter_mut()) {
        *pointer = match arg { Ptr(v) => (v as *mut u64).cast::<c_void>(), U32(v) => (v as *mut u32).cast(), F32(v) => (v as *mut f32).cast() };
    }
    unsafe { b.launch_flash_rows(name, threads.div_ceil(block as usize) as u32, rows as u32, block, shared, &mut pointers[..args.len()]) }
}
pub(super) fn launch_batch<T>(b: &Backend<'_>, name: &str, threads: usize, mut params: T) -> Result<()> {
    if threads == 0 { return Ok(()); }
    unsafe { b.launch_flash(name, threads.div_ceil(128) as u32, 128,
        &mut [(&mut params as *mut T).cast()]) }
}
fn launch_gate(b: &Backend<'_>, name: &str, mut params: GdnGate, rows: usize) -> Result<()> {
    let blocks = (params.heads as usize * params.tokens as usize * 32).div_ceil(128) as u32;
    unsafe { b.launch_flash_rows(name, blocks, rows as u32, 128, 0,
        &mut [(&mut params as *mut GdnGate).cast()]) }
}
fn launch_chunk_prompts(b: &Backend<'_>, name: &str, blocks: usize, rows: usize,
    threads: u32, shared: u32, mut params: GdnChunkPrompts) -> Result<()> {
    unsafe { b.launch_flash_rows(name, blocks as u32, rows as u32, threads, shared,
        &mut [(&mut params as *mut GdnChunkPrompts).cast()]) }
}
fn launch_history(b: &Backend<'_>, mut params: GdnHistory, rows: usize) -> Result<()> {
    let name = if params.rows == 0 { "flash_gdn_history" } else { "flash_gdn_history_prompts" };
    unsafe { b.launch_flash_rows(name, params.channels.div_ceil(128), rows as u32, 128, 0,
        &mut [(&mut params as *mut GdnHistory).cast()]) }
}
/// Official BF16 projections share the split tensor-core order at every width.
pub(crate) fn projection(b: &Backend<'_>, w: u64, x: u64, y: u64, rows: usize, cols: usize, tokens: usize) -> Result<()> {
    super::blocks::bf16_rows(b, w, x, y, rows, cols, tokens, rows)
}
/// Two projections sharing split input operands, as (weight, output, rows).
pub(crate) fn projection_pair(b: &Backend<'_>, x: u64, cols: usize, tokens: usize,
    first: (u64, u64, usize), second: (u64, u64, usize)) -> Result<()> {
    super::blocks::bf16_row_pair(b, x, cols, tokens, (first.0, first.1, first.2, first.2),
        (second.0, second.1, second.2, second.2))
}
pub(crate) fn bf16(w: &Weights<'_>, name: &str, shape: &[u64], elements: usize) -> Result<u64> {
    if elements == 0 { return Ok(0); }
    let t = w.tensor(name)?;
    if t.tensor.dtype != Dtype::Bf16 || t.tensor.shape != shape || t.bytes != elements * 2 {
        return Err(format!("{name}: expected official BF16 {shape:?}, {elements} local elements"));
    }
    w.device_address(name)
}

/// Only projections are pooled; each decode row keeps its own FP32 recurrence
/// and convolution history. No chunk-prefill workspace or weight copy.
pub(super) struct GdnBatch<'a> {
    qkv: DeviceVector<'a>,
    z: DeviceVector<'a>,
    a: DeviceVector<'a>,
    b: DeviceVector<'a>,
    gated: DeviceVector<'a>,
    sessions: Option<DeviceVector<'a>>,
}
impl<'a> GdnBatch<'a> {
    fn lengths(w: &Weights<'_>, rows: usize) -> [usize; 5] {
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        [(2 * keys + heads) * DIM * rows, heads * DIM * rows,
            heads * rows, heads * rows, heads * DIM * rows].map(|n| n.max(1))
    }
    pub(super) fn new(b: &Backend<'a>, w: &Weights<'_>, rows: usize) -> Result<Self> {
        let mut v = b.vector_arena(&Self::lengths(w, rows))?.into_iter();
        Ok(Self { qkv: v.next().unwrap(), z: v.next().unwrap(), a: v.next().unwrap(),
            b: v.next().unwrap(), gated: v.next().unwrap(),
            sessions: Some(b.vector(rows * std::mem::size_of::<GdnSessionRow>() / 4)?) })
    }
    /// The prompt MLP partials are idle until attention has finished. Alias
    /// that arena for grouped projections; recurrence reads its rows in place.
    pub(super) fn workspace(w: &Weights<'_>, rows: usize, arena: &DeviceVector<'a>) -> Result<Self> {
        let mut offset = 0;
        let [qkv, z, a, b, gated] = Self::lengths(w, rows).map(|len| {
            let start = offset;
            offset += len.div_ceil(16) * 16;
            arena.slice(start..start + len)
        });
        Ok(Self { qkv: qkv?, z: z?, a: a?, b: b?, gated: gated?, sessions: None })
    }
    pub(super) fn upload_sessions(&mut self, b: &Backend<'_>, rows: &[GdnSessionRow]) -> Result<()> {
        let table = self.sessions.as_mut().ok_or("prompt GDN workspace has no decode descriptors")?;
        let bytes = unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), std::mem::size_of_val(rows)) };
        b.write_vector_bytes(table, bytes)
    }
    pub(super) fn device_bytes(&self) -> usize {
        self.sessions.as_ref().map_or(0, |v| (v.len() * 4).div_ceil(64) * 64) + [&self.qkv, &self.z, &self.a, &self.b, &self.gated].iter().map(|v| (v.len() * 4).div_ceil(64) * 64).sum::<usize>()
    }
}

/// One workspace is reused between layers. Only activation scratch includes
/// a full gathered norm vector; the original sharded weights stay resident once.
pub struct GdnScratch<'a> {
    hidden: usize,
    heads: usize,
    keys: usize,
    capacity: usize,
    pub(crate) qkv: DeviceVector<'a>,
    pub(crate) z: DeviceVector<'a>,
    pub(crate) a: DeviceVector<'a>,
    pub(crate) b: DeviceVector<'a>,
    convolved: DeviceVector<'a>,
    q: DeviceVector<'a>,
    k: DeviceVector<'a>,
    v: DeviceVector<'a>,
    beta: DeviceVector<'a>,
    decay: DeviceVector<'a>,
    core: DeviceVector<'a>,
    pub(crate) gated: DeviceVector<'a>,
    output: DeviceVector<'a>,
    norm: DeviceVector<'a>,
    /// use_norm's address, 0 = `norm`.
    norm_source: std::cell::Cell<u64>,
    cumulative: DeviceVector<'a>,
    system: DeviceVector<'a>,
    attention: DeviceVector<'a>,
    values: DeviceVector<'a>,
    decayed_keys: DeviceVector<'a>,
    chunk_decay: DeviceVector<'a>,
    bytes: usize,
    /// Set by the batched prompt step when every session's call is fully fused
    /// (fuses_fully): (dense input tile, rows of the out projection). The gated
    /// scans then write the out projection's packed BF16 input there instead of
    /// `gated`, and finish_next runs the prepacked GEMM (no pack pass).
    pub(crate) packed_out: std::cell::Cell<Option<(u64, usize)>>,
    /// Set by a cohort verify pass (cohort_verify): scan_next leaves the output gate
    /// to one Gdn::gate_rows launch over the whole pass.
    pub(crate) defer_gate: std::cell::Cell<bool>,
    /// Set by a cohort verify pass (cohort_copies): the entry conv histories were already
    /// copied, so scan_next skips that copy (S still copies here when the window is deep).
    pub(crate) entry_copied: std::cell::Cell<bool>,
    /// Set by a cohort verify layer (cohort_kernels): scan_next's launches, for launch_cohort.
    pub(crate) cohort: std::cell::RefCell<Option<CohortLaunches>>,
}
/// True when a chunk-form call of `tokens` rows runs only prep + fused scan
/// pieces (no piece small enough for the rows kernel), so its v rows, gates
/// and output gate are all fused.
pub(crate) fn fuses_fully(tokens: usize) -> bool {
    let limit = spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT;
    chunk_form() && tokens > limit && !(1..=limit).contains(&(tokens % CHUNK_PIECE))
}
impl<'a> GdnScratch<'a> {
    pub fn new(b: &Backend<'a>, w: &Weights<'_>, max_tokens: usize, full_norm: &[f32; DIM]) -> Result<Self> {
        if max_tokens == 0 || max_tokens > CHUNK {
            return Err("GDN workspace holds 1..=64 tokens; tile longer prefill at chunk boundaries".into());
        }
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        Self::allocate(b, HIDDEN, heads, keys, max_tokens, full_norm,
            Self::workspace_lengths(HIDDEN, heads, keys, max_tokens, true, false))
    }
    /// Flash prompt spans keep projections and gated output in GdnBatch rows.
    /// Only convolution and recurrence intermediates need separate storage.
    pub(super) fn group(b: &Backend<'a>, w: &Weights<'_>, max_tokens: usize) -> Result<Self> {
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        Self::allocate(b, HIDDEN, heads, keys, max_tokens, &[0.0; DIM], Self::group_lengths(heads, keys, max_tokens))
    }
    fn group_lengths(heads: usize, keys: usize, max_tokens: usize) -> [usize; 20] {
        let mut lengths = Self::workspace_lengths(HIDDEN, heads, keys, max_tokens, true, false);
        for i in [0, 1, 2, 3, 11, 12] { lengths[i] = 1; }
        lengths
    }
    pub(super) fn group_reservation_bytes(heads: usize, keys: usize, max_tokens: usize) -> usize {
        Self::group_lengths(heads, keys, max_tokens).iter().map(|n| (n * 4).div_ceil(64) * 64).sum()
    }
    /// Dense BF16 preparation fuses convolution and solve inputs in registers.
    pub fn with_dimensions(b: &Backend<'a>, hidden: usize, heads: usize, keys: usize, max_tokens: usize, full_norm: &[f32; DIM]) -> Result<Self> {
        Self::allocate(b, hidden, heads, keys, max_tokens, full_norm,
            Self::workspace_lengths(hidden, heads, keys, max_tokens, false, true))
    }
    pub(crate) fn shared_projections(b: &Backend<'a>, hidden: usize, heads: usize, keys: usize, max_tokens: usize) -> Result<Self> {
        let mut lengths = Self::workspace_lengths(hidden, heads, keys, max_tokens, false, false);
        lengths[4..11].fill(1);
        lengths[12] = 1;
        lengths[14..].fill(1);
        Self::allocate(b, hidden, heads, keys, max_tokens, &[0.0; DIM], lengths)
    }
    pub(crate) fn decode_workspace(b: &Backend<'a>, hidden: usize, heads: usize, keys: usize, max_tokens: usize) -> Result<Self> {
        let mut lengths = Self::workspace_lengths(hidden, heads, keys, max_tokens, true, false);
        lengths[12] = 1;
        lengths[14..].fill(1);
        Self::allocate(b, hidden, heads, keys, max_tokens, &[0.0; DIM],
            lengths)
    }
    /// BF16 prompts use pooled projections and keep only a solo decode row.
    /// FP8 prompts retain their per-prompt recurrence workspace.
    pub(crate) fn prompt_workspace(b: &Backend<'a>, hidden: usize, heads: usize, keys: usize, max_tokens: usize, fp8: bool) -> Result<Self> {
        let mut lengths = Self::workspace_lengths(hidden, heads, keys, max_tokens, false, true);
        let solo = Self::workspace_lengths(hidden, heads, keys, 1, false, true);
        if !fp8 { for i in [0, 1, 2, 3, 11, 12] { lengths[i] = solo[i]; } }
        Self::allocate(b, hidden, heads, keys, max_tokens, &[0.0; DIM], lengths)
    }
    fn allocate(b: &Backend<'a>, hidden: usize, heads: usize, keys: usize, max_tokens: usize,
        full_norm: &[f32; DIM], lengths: [usize; 20]) -> Result<Self> {
        if max_tokens == 0 || max_tokens > 32768 { return Err("GDN tile must hold 1..=32768 tokens".into()); }
        let bytes = lengths.iter().map(|n| (n * 4).div_ceil(64) * 64).sum();
        let mut v = b.vector_arena(&lengths)?.into_iter();
        let mut result = Self { hidden, heads, keys, capacity: max_tokens,
            qkv: v.next().unwrap(), z: v.next().unwrap(), a: v.next().unwrap(), b: v.next().unwrap(),
            convolved: v.next().unwrap(), q: v.next().unwrap(), k: v.next().unwrap(), v: v.next().unwrap(),
            beta: v.next().unwrap(), decay: v.next().unwrap(), core: v.next().unwrap(), gated: v.next().unwrap(),
            output: v.next().unwrap(), norm: v.next().unwrap(), norm_source: std::cell::Cell::new(0), cumulative: v.next().unwrap(),
            system: v.next().unwrap(), attention: v.next().unwrap(), values: v.next().unwrap(),
            decayed_keys: v.next().unwrap(), chunk_decay: v.next().unwrap(), bytes,
            packed_out: std::cell::Cell::new(None), defer_gate: std::cell::Cell::new(false),
            entry_copied: std::cell::Cell::new(false), cohort: std::cell::RefCell::new(None) };
        result.set_norm(b, full_norm)?;
        Ok(result)
    }
    pub fn reservation_bytes(hidden: usize, heads: usize, keys: usize, max_tokens: usize) -> usize {
        Self::workspace_lengths(hidden, heads, keys, max_tokens, false, true).iter()
            .map(|n| (n * 4).div_ceil(64) * 64).sum()
    }
    pub(crate) fn decode_reservation_bytes(hidden: usize, heads: usize, keys: usize, max_tokens: usize) -> usize {
        Self::workspace_lengths(hidden, heads, keys, max_tokens, true, false).iter()
            .map(|n| (n * 4).div_ceil(64) * 64).sum()
    }
    /// Dense (grouped) chunk prefill stores Q/K once per key head; a one-token
    /// step and batched decode still need one row per value head.
    fn workspace_lengths(hidden: usize, heads: usize, keys: usize, max_tokens: usize, materialized_inputs: bool, grouped: bool) -> [usize; 20] {
        let width = (2 * keys + heads) * DIM;
        let hd = heads * DIM;
        let qk = if grouped { (keys * max_tokens).max(heads) * DIM } else { hd * max_tokens };
        let chunks = max_tokens.div_ceil(CHUNK);
        [width * max_tokens, hd * max_tokens, heads * max_tokens, heads * max_tokens,
            if materialized_inputs { width * max_tokens } else { 1 }, qk, qk, if grouped { hd } else { hd * max_tokens },
            heads * max_tokens, heads * max_tokens, hd * max_tokens, hd * max_tokens,
            hidden * max_tokens, DIM, chunks * heads * CHUNK, chunks * heads * CHUNK * CHUNK, chunks * heads * CHUNK * CHUNK,
            chunks * heads * CHUNK * DIM, chunks * heads * CHUNK * DIM, chunks * heads].map(|n| n.max(1))
    }
    pub fn set_norm(&mut self, b: &Backend<'_>, full_norm: &[f32; DIM]) -> Result<()> {
        b.write_vector(&mut self.norm, full_norm)
    }
    /// Active shape is exactly tokens * 2560, including after a cached step.
    pub fn output(&self) -> &DeviceVector<'a> { &self.output }
    /// Row-major qkv, z, a, b projection outputs and the gated rows (rank chunk prefill).
    pub fn row_addresses(&self) -> [u64; 5] {
        [self.qkv.device_address(), self.z.device_address(), self.a.device_address(),
            self.b.device_address(), self.gated.device_address()]
    }
    pub fn device_bytes(&self) -> usize { self.bytes }
    /// Token-major scratch already reserved by the decode/prompt executor.
    pub(crate) fn token_workspace(&self, offset: usize, count: usize) -> Result<[u64; 6]> {
        let end = offset
            .checked_add(count)
            .filter(|&n| n <= self.capacity)
            .ok_or("GDN token workspace range")?;
        let p = |v: &DeviceVector<'_>, width: usize| {
            v.slice(offset * width..end * width)
                .map(|s| s.device_address())
        };
        let hd=self.heads*DIM;
        Ok([p(&self.q,hd)?,p(&self.k,hd)?,p(&self.v,hd)?,p(&self.beta,self.heads)?,p(&self.decay,self.heads)?,p(&self.core,hd)?])
    }
    pub(crate) fn step_batch_range(&mut self, b: &Backend<'_>, weights: [u64; 4],
        descriptors: &DeviceVector<'_>, offset: usize, count: usize, epsilon: f32,
        [keys, heads, key_start, value_start, precision]: [u32; 5]) -> Result<()> {
        let _class = b.prefill_class(1);
        let end = offset.checked_add(count).ok_or("GDN row range overflow")?;
        if count == 0 || end > self.capacity || keys as usize != self.keys || heads as usize != self.heads
            || keys == 0 || heads % keys != 0 || precision & 1 == 0 || precision > 15
            || value_start as u64 != key_start as u64 * (heads / keys) as u64
            || !epsilon.is_finite() || epsilon <= 0.0 || weights.contains(&0)
            || descriptors.is_segmented() || descriptors.len() != count * 6 { return Err("GDN step shape/owners/parameters".into()); }
        let width = (2 * self.keys + self.heads) * DIM;
        let hd = self.heads * DIM;
        if count.checked_mul(width.max(hd)).is_none_or(|n| n > u32::MAX as usize)
            || descriptors.device_address() % 8 != 0 || weights.iter().any(|w| w % 2 != 0) {
            return Err("GDN step launch extent/alignment".into());
        }
        for (v, width) in [(&self.qkv,width),(&self.z,hd),(&self.a,self.heads),(&self.b,self.heads),
            (&self.convolved,width),(&self.q,hd),(&self.k,hd),(&self.v,hd),(&self.beta,self.heads),
            (&self.decay,self.heads),(&self.core,hd),(&self.gated,hd)] {
            if v.is_segmented() || v.len() < end * width { return Err("GDN step operand extent".into()); }
        }
        let p = |v: &DeviceVector<'_>, width: usize| v.device_address() + (offset * width * 4) as u64;
        launch_batch(b, "flash_gdn_conv_batch", count * width, GdnConvBatch {
            x: p(&self.qkv, width), w: weights[0], rows: descriptors.device_address(),
            y: p(&self.convolved, width), sessions: 0, channels: width as u32, count: count as u32, precision: 1 })?;
        launch_batch(b, "flash_gdn_prepare_batch", count * self.heads * 32, GdnPrepareBatch {
            qkv: p(&self.convolved, width), a: p(&self.a, self.heads), b: p(&self.b, self.heads),
            alog: weights[1], dt: weights[2], q: p(&self.q, hd), k: p(&self.k, hd), v: p(&self.v, hd),
            beta: p(&self.beta, self.heads), g: p(&self.decay, self.heads), sessions: 0,
            key_heads: keys, value_heads: heads, key_start, value_start,
            count: count as u32, precision })?;
        launch_batch(b, "flash_gdn_step_batch", count * hd, GdnStepBatch {
            q: p(&self.q, hd), k: p(&self.k, hd), v: p(&self.v, hd), beta: p(&self.beta, self.heads),
            g: p(&self.decay, self.heads), rows: descriptors.device_address(), out: p(&self.core, hd),
            sessions: 0, heads: self.heads as u32, count: count as u32 })?;
        launch_gate(b, "flash_gdn_gate", GdnGate {
            x: p(&self.core, hd) as _, z: p(&self.z, hd) as _, norm: weights[3] as _,
            output: p(&self.gated, hd) as _, heads: (count * self.heads) as u32,
            tokens: 1, precision, packed: 0, epsilon, sessions: 0 }, 1)
    }
    pub(crate) fn prefill_batch(&self, b: &Backend<'_>, weights: [u64; 4], descriptors: &DeviceVector<'_>,
        count: usize, longest: usize, epsilon: f32, precision: u32) -> Result<()> {
        let _class = b.prefill_class(1);
        let h = self.heads as u32;
        let repeat = (self.heads / self.keys) as u32;
        let chunks = longest.div_ceil(CHUNK);
        b.profile_phase(23, || {
            launch_rows(b, "flash_gdn_conv_prepare_prompts", longest * self.keys * 32, count, 128, 0,
                &mut [Ptr(self.qkv.device_address()), Ptr(weights[0]), Ptr(self.a.device_address()), Ptr(self.b.device_address()),
                    Ptr(weights[1]), Ptr(weights[2]), Ptr(descriptors.device_address()), U32(self.keys as u32), U32(h), U32(precision)])
        })?;
        let params = GdnChunkPrompts { rows: descriptors.device_address(), heads: h, key_repeat: repeat,
            qkv: self.qkv.device_address(), conv: weights[0], keys: self.keys as u32, precision };
        b.profile_phase(24, || {
            launch_rows(b, "flash_gdn_cumulative_prompts", chunks * self.heads, count, 128, 0,
                &mut [Ptr(descriptors.device_address()), U32(h)])?;
            launch_chunk_prompts(b, "flash_gdn_system_prompts", chunks * self.heads * 4, count,
                GDN_SYSTEM_THREADS, GDN_SYSTEM_SHARED_BYTES, params)
        })?;
        b.profile_phase(26, || {
            launch_chunk_prompts(
                b,
                "flash_gdn_solve_prompts",
                chunks * self.heads * 2,
                count,
                512,
                0,
                params,
            )
        })?;
        b.profile_phase(23, || {
            launch_history(
                b,
                GdnHistory {
                    x: self.qkv.device_address() as _,
                    history: std::ptr::null_mut(),
                    rows: descriptors.device_address(),
                    channels: ((2 * self.keys + self.heads) * DIM) as u32,
                    tokens: 0,
                    precision: 1,
                },
                count,
            )
        })?;
        b.profile_phase(19, || {
            let mut params = GdnScanPrompts { rows: descriptors.device_address(), heads: h, key_repeat: repeat };
            unsafe { b.launch_flash_rows("flash_gdn_scan_prompts", (self.heads * (DIM / GDN_SCAN_COLUMNS)) as u32,
                count as u32, GDN_SCAN_THREADS, GDN_SCAN_SHARED_BYTES,
                &mut [(&mut params as *mut GdnScanPrompts).cast()]) }
        })?;
        launch_gate(b, "flash_gdn_gate_prompts", GdnGate {
            x: descriptors.device_address() as _, z: self.z.device_address() as _, norm: weights[3] as _,
            output: self.gated.device_address() as _, heads: h, tokens: longest as u32,
            precision, packed: 0, epsilon, sessions: 0 }, count)
    }
}

/// A layer's cache is four raw projection slots plus [local V heads,128,128].
/// The matrix has key coordinates first, value coordinates last, as in HF.
pub struct GdnState<'a> {
    layer: usize,
    rank: u64,
    state: DeviceVector<'a>,
    history: DeviceVector<'a>,
    cached: bool,
    /// Under the tensor split: the vectors hold one split rank's heads only (gdn_half.rs).
    half: bool,
}
impl<'a> GdnState<'a> {
    pub(super) fn pool_addresses(&self) -> [u64;3] { [self.state.device_address(),self.history.device_address(),0] }
    pub(super) fn commit_next(&mut self) { self.cached=true; }
    /// Tensor split state handoff: the recurrent state, the conv history and the cached flag as raw words.
    pub(super) fn handoff(&mut self) -> (&mut DeviceVector<'a>, &mut DeviceVector<'a>, &mut bool) { (&mut self.state, &mut self.history, &mut self.cached) }
    /// Rejection after a shallow verify (see shallow_verify): S and the tail
    /// rows below the window are live and unchanged, so only the conv history
    /// (and the cached flag) come back from the entry snapshot.
    pub(super) fn restore_history(&mut self, entry: &mut Self) {
        std::mem::swap(&mut self.history, &mut entry.history);
        self.cached = entry.cached;
    }

    pub(super) fn snapshot(&self, b: &Backend<'a>) -> Result<Self> {
        let mut next = Self { layer: self.layer, rank: self.rank,
            state: b.vector(self.state.len())?, history: b.vector(self.history.len())?, cached: self.cached, half: self.half };
        self.copy_into(b, &mut next)?;
        Ok(next)
    }
    /// Storage is populated by the verifier's scan before any commit can use it.
    pub(super) fn next_snapshot_storage(&self, b: &Backend<'a>) -> Result<Self> {
        Ok(Self {
            layer: self.layer,
            rank: self.rank,
            state: b.vector(self.state.len())?,
            history: b.vector(self.history.len())?,
            cached: self.cached,
            half: self.half,
        })
    }
    pub(crate) fn reset(&mut self, b: &Backend<'a>) -> Result<()> {
        b.zero_vector(&mut self.state)?;
        b.zero_vector(&mut self.history)?;
        self.cached = false;
        Ok(())
    }
    pub(crate) fn copy_into(&self, b: &Backend<'a>, next: &mut Self) -> Result<()> {
        b.copy_vector_range(&self.state, &mut next.state, 0, 0, self.state.len())?;
        b.copy_vector_range(&self.history, &mut next.history, 0, 0, self.history.len())?;
        next.cached = self.cached;
        Ok(())
    }
    pub(crate) fn dense_layers(b: &Backend<'a>, c: &engine_model::qwen3_5::Config) -> Result<Vec<Option<Self>>> {
        let state = (c.linear_value_heads * c.linear_key_dim * c.linear_value_dim) as usize;
        let history = (2 * c.linear_key_heads * c.linear_key_dim + c.linear_value_heads * c.linear_value_dim) as usize * 2;
        let lengths: Vec<_> = c.layers.iter().filter(|k| **k == engine_model::flash::LayerKind::DeltaNet)
            .flat_map(|_| [state, history]).collect();
        let mut vectors = b.vector_arena(&lengths)?.into_iter();
        c.layers.iter().enumerate().map(|(layer, kind)| {
            if *kind != engine_model::flash::LayerKind::DeltaNet { return Ok(None); }
            let mut state = vectors.next().unwrap();
            let mut history = vectors.next().unwrap();
            b.zero_vector(&mut state)?;
            b.zero_vector(&mut history)?;
            Ok(Some(Self { layer, rank: 0, state, history, cached: false, half: false }))
        }).collect()
    }
    pub(crate) fn device_bytes(&self) -> usize { (self.state.len() + self.history.len()) * 4 }
    pub(crate) fn descriptor(&self) -> [f32; 6] {
        self.decode_descriptor(self, None)
    }
    pub(crate) fn decode_descriptor(&self, history: &Self, output: Option<&Self>) -> [f32; 6] {
        let row = spark_gpu_probe::gdn_params::GdnDecodeRow { state: self.state.device_address(),
            history: history.history.device_address(), output_state: output.map_or(0, |s| s.state.device_address()) };
        [row.state as u32, (row.state >> 32) as u32, row.history as u32, (row.history >> 32) as u32,
            row.output_state as u32, (row.output_state >> 32) as u32].map(f32::from_bits)
    }
    pub(crate) fn history_destination(&self, next: &mut Self) -> u64 {
        next.cached = self.cached;
        next.history.device_address()
    }
    pub(crate) fn copy_history(&self, b: &Backend<'a>, next: &mut Self) -> Result<()> {
        b.copy_vector_range(&self.history, &mut next.history, 0, 0, self.history.len())?;
        next.cached = self.cached;
        Ok(())
    }
    pub(crate) fn prompt_descriptor(&self, s: &GdnScratch<'_>, offset: usize, tokens: usize) -> [f32; 30] {
        let pointers = [self.state.device_address(), self.history.device_address(), s.q.device_address(), s.k.device_address(),
            s.v.device_address(), s.beta.device_address(), s.decay.device_address(), s.core.device_address(),
            s.cumulative.device_address(), s.system.device_address(), s.attention.device_address(), s.values.device_address(),
            s.decayed_keys.device_address(), s.chunk_decay.device_address()];
        let mut words = [0; 30];
        for (i, p) in pointers.into_iter().enumerate() { words[i * 2] = p as u32; words[i * 2 + 1] = (p >> 32) as u32; }
        words[28] = offset as u32; words[29] = tokens as u32;
        words.map(f32::from_bits)
    }
    pub(crate) fn commit_prompt(&mut self) { self.cached = true; }
}

pub struct Gdn<'a, 'w> {
    hidden: usize,
    /// Dense checkpoints use packed projections, BF16 activations/history and swish.
    precision: bool,
    // BF16 activations in bit 0; FP32 A_log, dt_bias and norm in bits 1..3.
    parameter_precision: u32,
    norm: u64,
    epsilon: f32,
    layer: usize,
    rank: u64,
    heads: usize,
    keys: usize,
    key_start: usize,
    value_start: usize,
    qkv: u64,
    z: u64,
    a: u64,
    b: u64,
    conv: u64,
    alog: u64,
    dt: u64,
    out: u64,
    state: DeviceVector<'a>,
    history: DeviceVector<'a>,
    cached: bool,
    half: bool,
    _weights: PhantomData<&'w ()>,
}
impl<'a, 'w> Gdn<'a, 'w> {
    pub fn load(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize) -> Result<Self> {
        Self::load_cached(b, w, layer, None)
    }
    /// Reattach session state after replacing the resident layer's weights.
    pub fn load_cached(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, cached: Option<GdnState<'a>>) -> Result<Self> {
        if layer >= 48 || layer % 4 == 3 { return Err("layer is not GatedDeltaNet".into()); }
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        let width = (2 * keys + heads) * DIM;
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        let get = |part: &str, shape: &[u64], elements| bf16(w, &format!("{p}.{part}"), shape, elements);
        let fresh = cached.is_none();
        let cache = if let Some(cache) = cached {
            if cache.layer != layer || cache.rank != w.plan.rank
                || cache.state.len() != (heads * DIM * DIM).max(1) || cache.history.len() != (width * 4).max(1) {
                return Err("GDN cache does not belong to this layer and rank".into());
            }
            cache
        } else {
            let mut vectors = b.vector_arena(&[(heads * DIM * DIM).max(1), (width * 4).max(1)])?.into_iter();
            GdnState { layer, rank: w.plan.rank, state: vectors.next().unwrap(), history: vectors.next().unwrap(), cached: false, half: false }
        };
        let mut result = Self { hidden: HIDDEN, precision: false, parameter_precision: 0, norm: 0, epsilon: 1e-6,
            layer, rank: w.plan.rank, heads, keys, key_start: w.plan.key_heads.start as usize,
            value_start: w.plan.value_heads.start as usize,
            qkv: get("in_proj_qkv.weight", &[10240, 2560], width * HIDDEN)?,
            z: get("in_proj_z.weight", &[6144, 2560], heads * DIM * HIDDEN)?,
            a: get("in_proj_a.weight", &[48, 2560], heads * HIDDEN)?,
            b: get("in_proj_b.weight", &[48, 2560], heads * HIDDEN)?,
            conv: get("conv1d.weight", &[10240, 1, 4], width * 4)?,
            alog: get("A_log", &[48], heads)?, dt: get("dt_bias", &[48], heads)?,
            out: get("out_proj.weight", &[2560, 6144], HIDDEN * heads * DIM)?,
            state: cache.state, history: cache.history, cached: cache.cached, half: false,
            _weights: PhantomData };
        if fresh { result.reset(b)?; }
        Ok(result)
    }
    /// Mixed-checkpoint layer: MXFP8 projections arrive as expanded packed
    /// BF16 addresses [qkv, z, a, b, out]; conv/A_log/dt_bias stay BF16 in the
    /// resident arena. Arithmetic mode is the Flash FP32/sigmoid one.
    pub fn load_next(b: &Backend<'a>, w: &'w Weights<'_>, layer: usize, proj: [u64; 5],
        cached: Option<GdnState<'a>>) -> Result<Self> {
        if layer >= 48 || layer % 4 == 3 { return Err("layer is not GatedDeltaNet".into()); }
        if proj.iter().any(|a| a % 16 != 0) { return Err("expanded GDN projections need 16-byte alignment".into()); }
        let heads = (w.plan.value_heads.end - w.plan.value_heads.start) as usize;
        let keys = (w.plan.key_heads.end - w.plan.key_heads.start) as usize;
        let width = (2 * keys + heads) * DIM;
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        let get = |part: &str, shape: &[u64], elements| bf16(w, &format!("{p}.{part}"), shape, elements);
        let fresh = cached.is_none();
        let recurrent = if chunk_form() && heads != 0 {
            spark_gpu_probe::gdn_params::chunk_state_floats(heads, keys)
        } else { heads * DIM * DIM };
        let cache = if let Some(cache) = cached {
            if cache.layer != layer || cache.rank != w.plan.rank || cache.state.len() != recurrent.max(1)
                || cache.history.len() != (width * 4).max(1) { return Err("GDN cache does not belong to this layer and rank".into()); }
            cache } else {
            let mut vectors = b.vector_arena(&[recurrent.max(1), (width * 4).max(1)])?.into_iter();
            GdnState { layer, rank: w.plan.rank, state: vectors.next().unwrap(), history: vectors.next().unwrap(), cached: false, half: false } };
        let [qkv, z, a, bproj, out] = proj;
        let mut result = Self { hidden: HIDDEN, precision: false, parameter_precision: 0, norm: 0, epsilon: 1e-6,
            layer, rank: w.plan.rank, heads, keys, key_start: w.plan.key_heads.start as usize,
            value_start: w.plan.value_heads.start as usize, qkv, z, a, b: bproj,
            conv: get("conv1d.weight", &[10240, 1, 4], width * 4)?,
            alog: get("A_log", &[48], heads)?, dt: get("dt_bias", &[48], heads)?, out,
            state: cache.state, history: cache.history, cached: cache.cached, half: false, _weights: PhantomData };
        if fresh { result.reset(b)?; }
        Ok(result)
    }
    /// Borrow the dense cartridge's single packed weight arena. The kernel
    /// dimensions are checked by its loader; recurrence is shared with Flash.
    pub fn load_dense(b: &Backend<'a>, weights: &'w std::collections::BTreeMap<String, spark_gpu_probe::driver::Matrix<'a>>,
        c: &engine_model::qwen3_5::Config, layer: usize, cached: Option<GdnState<'a>>) -> Result<Self> {
        let heads = c.linear_value_heads as usize;
        let keys = c.linear_key_heads as usize;
        let width = (2 * keys + heads) * DIM;
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        let get = |part: &str| {
            weights
                .get(&format!("{p}.{part}"))
                .map(|m| m.device_address())
                .ok_or_else(|| format!("missing dense GDN {part}"))
        };
        let fresh = cached.is_none();
        let cache = if let Some(cache) = cached {
            if cache.layer != layer || cache.rank != 0 || cache.state.len() != heads * DIM * DIM || cache.history.len() != width * 2 {
                return Err("dense GDN cache belongs to another layer".into());
            }
            cache
        } else {
            let mut v = b.vector_arena(&[heads * DIM * DIM, width * 2])?.into_iter();
            GdnState { layer, rank: 0, state: v.next().unwrap(), history: v.next().unwrap(), cached: false, half: false }
        };
        let parameter_precision = 1 | (u32::from(weights[&format!("{p}.A_log")].scalar_f32()) << 1)
            | (u32::from(weights[&format!("{p}.dt_bias")].scalar_f32()) << 2)
            | (u32::from(weights[&format!("{p}.norm.weight")].scalar_f32()) << 3);
        let mut result = Self { hidden: c.hidden as usize, precision: true, parameter_precision, norm: get("norm.weight")?, epsilon: c.epsilon as f32,
            layer, rank: 0, heads, keys, key_start: 0, value_start: 0,
            qkv: get("in_proj_qkv.weight")?, z: get("in_proj_z.weight")?, a: get("in_proj_a.weight")?, b: get("in_proj_b.weight")?,
            conv: get("conv1d.weight")?, alog: get("A_log")?, dt: get("dt_bias")?, out: get("out_proj.weight")?,
            state: cache.state, history: cache.history, cached: cache.cached, half: false, _weights: PhantomData };
        // Grouped conv_prepare maps each key head to whole value-head groups.
        if keys == 0 || heads % keys != 0 || result.value_start % (heads / keys) != 0
            || result.key_start != result.value_start / (heads / keys) {
            return Err("dense GDN needs whole value-head groups per key head".into());
        }
        if fresh { result.reset(b)?; }
        Ok(result)
    }
    pub fn into_state(self) -> GdnState<'a> {
        GdnState { layer: self.layer, rank: self.rank, state: self.state, history: self.history, cached: self.cached, half: self.half }
    }
    pub(super) fn snapshot(&self, b: &Backend<'a>) -> Result<GdnState<'a>> {
        let mut state = b.vector(self.state.len())?;
        let mut history = b.vector(self.history.len())?;
        b.copy_range(&self.state, &mut state, 0, 0, self.state.len())?;
        b.copy_range(&self.history, &mut history, 0, 0, self.history.len())?;
        Ok(GdnState { layer: self.layer, rank: self.rank, state, history, cached: self.cached, half: self.half })
    }
    pub(super) fn save_into(&self, b: &Backend<'a>, saved: &mut Option<GdnState<'a>>) -> Result<()> {
        if let Some(saved) = saved {
            if saved.layer != self.layer || saved.rank != self.rank { return Err("GDN verification state ownership differs".into()); }
            b.copy_range(&self.state, &mut saved.state, 0, 0, self.state.len())?;
            b.copy_range(&self.history, &mut saved.history, 0, 0, self.history.len())?;
            saved.cached = self.cached;
        } else { *saved = Some(self.snapshot(b)?); }
        Ok(())
    }
    /// Share projections across consecutive verification rows, retaining the
    /// canonical single-token convolution and recurrence for every row.
    pub(super) fn verify_rows(&mut self, b: &Backend<'a>, s: &mut GdnScratch<'a>,
        projected: &mut GdnScratch<'a>, input: &DeviceVector<'a>, tokens: usize,
        saved: &mut [&mut Option<GdnState<'a>>]) -> Result<()> {
        if self.precision || (!self.cached && self.heads != 0) || tokens < 2 || tokens > projected.capacity || input.len() != tokens * self.hidden {
            return Err("Flash verification requires cached FP32 recurrent rows".into());
        }
        let width = (2 * self.keys + self.heads) * DIM;
        let hd = self.heads * DIM;
        projection_pair(b, input.device_address(), self.hidden, tokens,
            (self.qkv, projected.qkv.device_address(), width), (self.z, projected.z.device_address(), hd))?;
        projection_pair(b, input.device_address(), self.hidden, tokens,
            (self.a, projected.a.device_address(), self.heads), (self.b, projected.b.device_address(), self.heads))?;
        for t in 0..tokens {
            // Views keep the projection arena alive while recurrence consumes
            // one row directly. No staging copies enter the captured graph.
            let mut originals = Vec::with_capacity(5);
            for (from, to, n) in [(&projected.qkv, &mut s.qkv, width), (&projected.z, &mut s.z, hd),
                (&projected.a, &mut s.a, self.heads), (&projected.b, &mut s.b, self.heads),
                (&projected.gated, &mut s.gated, hd)] {
                let view = from.slice(t * n..(t * n + n).max(1))?;
                originals.push(std::mem::replace(to, view));
            }
            let result = self.forward(b, s, &input.slice(t * self.hidden..(t + 1) * self.hidden)?, 1, false, false, false, 0);
            for (to, original) in [&mut s.qkv, &mut s.z, &mut s.a, &mut s.b, &mut s.gated].into_iter().zip(originals) {
                *to = original;
            }
            result?;
            if t < saved.len() { self.save_into(b, saved[t])?; }
        }
        s.output.resize_within_capacity(tokens * self.hidden)?;
        if hd == 0 { b.write_vector(&mut s.output, &vec![0.0; tokens * self.hidden]) }
        else { projection(b, self.out, projected.gated.device_address(), s.output.device_address(), self.hidden, hd, tokens) }
    }
    pub fn recurrent_bytes(&self) -> usize { self.heads * DIM * DIM * 4 }
    pub fn convolution_bytes(&self) -> usize { (2 * self.keys + self.heads) * DIM * 4 * if self.precision { 2 } else { 4 } }
    pub fn state_bytes(&self) -> usize {
        self.recurrent_bytes().max(4).div_ceil(64) * 64
            + self.convolution_bytes().max(4).div_ceil(64) * 64
    }
    pub fn reset(&mut self, b: &Backend<'_>) -> Result<()> {
        b.zero_vector(&mut self.state)?;
        b.zero_vector(&mut self.history)?;
        self.cached = false;
        Ok(())
    }
    /// The in-projection members in project_next's order as (weight, rows), and K (the tensor split cuts them).
    pub(super) fn projection(&self) -> ([(u64, usize); 4], usize) { ([(self.qkv, (2 * self.keys + self.heads) * DIM), (self.z, self.heads * DIM), (self.a, self.heads), (self.b, self.heads)], self.hidden) }
    pub(super) fn project_next(&self, b: &Backend<'_>, s: &GdnScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > s.capacity || input.len() != tokens*self.hidden || self.precision || self.heads == 0 {
            return Err("Flash-Next projected GDN shape".into());
        }
        super::blocks::next_group(
            b,
            input.device_address(),
            self.hidden,
            tokens,
            [
                (
                    self.qkv,
                    s.qkv.device_address(),
                    (2 * self.keys + self.heads) * DIM,
                ),
                (self.z, s.z.device_address(), self.heads * DIM),
                (self.a, s.a.device_address(), self.heads),
                (self.b, s.b.device_address(), self.heads),
            ],
        )
    }
    pub(super) fn pool_next(
        b: &Backend<'_>,
        w: &Weights<'_>,
        proj: [u64; 5],
        s: &GdnScratch<'_>,
        input: &DeviceVector<'_>,
        output: &DeviceVector<'_>,
        norm: u64,
        mut step: spark_gpu_probe::next_math::SlotStep,
    ) -> Result<()> {
        use spark_gpu_probe::{
            gdn_params::GdnStepBatch,
            kernel_params::{GdnGate, GdnPrepare},
        };
        let n = step.rows as usize;
        super::blocks::next_rows_group(b, input.device_address(), HIDDEN, n, [
            (proj[0], s.qkv.device_address(), 10240),
            (proj[1], s.z.device_address(), 6144),
            (proj[2], s.a.device_address(), 48),
            (proj[3], s.b.device_address(), 48),
        ])?;
        let prefix=format!("model.language_model.layers.{}.linear_attn",step.layer);
        let get=|name: &str| w.device_address(&format!("{prefix}.{name}"));
        let [q,k,v,beta,g,core]=s.token_workspace(0,n)?;
        let mut p=GdnPrepare { x:s.qkv.device_address() as _,w:get("conv1d.weight")? as _,history:std::ptr::null(),
            a:s.a.device_address() as _,b:s.b.device_address() as _,alog:get("A_log")? as _,dt:get("dt_bias")? as _,
            q:q as _,k:k as _,v:v as _,beta:beta as _,g:g as _,key_heads:16,value_heads:48,key_start:0,value_start:0,tokens:1,precision:0,scale:0 };
        let mut scan=GdnStepBatch { q,k,v,beta,g,rows:0,out:core,sessions:0,heads:48,count:1 };
        // `norm` = the layer's gathered norm (no per-step copy into s.norm).
        let mut gate=GdnGate { x:core as _,z:s.z.device_address() as _,norm:norm as _,
            output:s.gated.device_address() as _,heads:48,tokens:1,precision:0,packed:0,epsilon:1e-6,sessions:0 };
        let step=(&mut step as *mut spark_gpu_probe::next_math::SlotStep).cast();
        unsafe {
            // prepare_slot + history_slot fused (same bits): one warp per qkv section.
            b.launch_flash_rows(b.chain_entry(n,"flash_next_gdn_prepare_hist_slot","flash_next_gdn_prepare_hist_slot_chain"),20,n as u32,128,0,&mut [(&mut p as *mut GdnPrepare).cast(),step])?;
            if chunk_form() {
                // The slot applies the output gate itself (flash_next_gdn_gate_slot's
                // words, zeros for skipped rows): one launch less per layer and step.
                let mut chunk = spark_gpu_probe::gdn_params::GdnChunkNext { q, k, v, beta, g, rows: 0, out: core,
                    a: 0, p: 0, gamma: 0, k0: 0, heads: 48, keys: 16, count: 1, base: 0,
                    z: s.z.device_address(), norm, gated: s.gated.device_address(), epsilon: gate.epsilon, spare: 0,
                    x: 0, w: 0, history: 0, channels: 0, x_start: 0, ga: 0, gb: 0, alog: 0, dt: 0, pk: 0, pk_tokens: 0, pk_row0: 0 };
                // The slot reads its state while prepare_hist_slot runs and waits
                // (griddepcontrol) before q/k/v/beta/g: same bytes.
                b.programmatic_next();
                b.launch_flash_rows(b.chain_entry(n, "flash_next_gdn_chunk_slot", "flash_next_gdn_chunk_slot_chain"), 48, n as u32, 128, 0,
                    &mut [(&mut chunk as *mut spark_gpu_probe::gdn_params::GdnChunkNext).cast(), step])?;
            } else {
                b.launch_flash_rows(
                    "flash_next_gdn_scan_slot",
                    96,
                    n as u32,
                    64,
                    0,
                    &mut [(&mut scan as *mut GdnStepBatch).cast(), step],
                )?;
                b.launch_flash_rows("flash_next_gdn_gate_slot",12,n as u32,128,0,&mut [(&mut gate as *mut GdnGate).cast(),step])?;
            }
        }
        super::blocks::next_rows(b,proj[4],s.gated.device_address(),output.device_address(),HIDDEN,6144,n)
    }
    /// Prompt/decode use the same ascending scan, including cold M=1.
    /// `base` = absolute position of the first row (chunk form only).
    pub(super) fn recur_next(
        &mut self,
        b: &Backend<'_>,
        s: &GdnScratch<'_>,
        start: usize,
        tokens: usize,
        descriptor: &mut DeviceVector<'_>,
        base: usize,
    ) -> Result<()> {
        self.scan_next(b, s, (start, tokens), descriptor, None, true, base, None, None)
    }
    pub(super) fn verify_next(
        &mut self,
        b: &Backend<'a>,
        s: &GdnScratch<'a>,
        start: usize,
        tokens: usize,
        descriptor: &mut DeviceVector<'a>,
        entry: &mut GdnState<'a>,
        base: usize,
        tape: Option<[(u64, u64, u32); 3]>,
        table: Option<u64>,
    ) -> Result<()> {
        if entry.layer != self.layer
            || entry.rank != self.rank
            || entry.state.len() != self.state.len()
            || entry.history.len() != self.history.len()
        {
            return Err("GDN entry snapshot ownership differs".into());
        }
        self.scan_next(b, s, (start, tokens), descriptor, Some(entry), true, base, tape, table)
    }
    /// `tape` (verify only, set when gdn::fused_verify_copy): the replay tape's (source,
    /// destination, words) copies, folded with the history snapshot and the descriptor
    /// into one flash_gdn_verify_copy launch.
    #[allow(clippy::too_many_arguments)]
    fn scan_next(
        &mut self,
        b: &Backend<'_>,
        s: &GdnScratch<'_>,
        rows: (usize, usize),
        descriptor: &mut DeviceVector<'_>,
        entry: Option<&mut GdnState<'a>>,
        emit: bool,
        base: usize,
        tape: Option<[(u64, u64, u32); 3]>,
        table: Option<u64>,
    ) -> Result<()> {
        use spark_gpu_probe::{
            gdn_params::GdnStepBatch,
            kernel_params::{GdnGate, GdnHistory, GdnPrepare},
        };
        let (start, tokens) = rows;
        if tokens == 0 || start.checked_add(tokens).is_none_or(|n| n > s.capacity)
            || self.precision || self.heads == 0 || self.keys == 0 || self.heads % self.keys != 0 {
            return Err("Flash-Next ordered GDN shape".into());
        }
        let hd = self.heads*DIM;
        let width = (2*self.keys+self.heads)*DIM;
        let at = |v: &DeviceVector<'_>, width: usize| v.device_address()+(start*width*4) as u64;
        let [q, k, v, beta, g, core] = s.token_workspace(start, tokens)?;
        let mut p = GdnPrepare { x: at(&s.qkv, width) as _, w: self.conv as _, history: self.history.device_address() as _,
            a: at(&s.a, self.heads) as _, b: at(&s.b, self.heads) as _, alog: self.alog as _, dt: self.dt as _,
            q: q as _, k: k as _, v: v as _, beta: beta as _, g: g as _, key_heads: self.keys as u32,
            value_heads: self.heads as u32, key_start: self.key_start as u32, value_start: self.value_start as u32,
            tokens: tokens as u32, precision: 0, scale: 0 };
        let address = |a: u64| [f32::from_bits(a as u32), f32::from_bits((a >> 32) as u32)];
        let chunked = chunk_form();
        // Chunk form with every piece through prep + scan (no piece small enough
        // for the rows kernel, which reads v): the scan convolves its value rows
        // from the projection rows itself and the prepare skips them.
        let fuse_v = chunked && fuses_fully(tokens);
        let packed_out = s.packed_out.get();
        if packed_out.is_some() && !(fuse_v && emit) { return Err("packed GDN output needs a fully fused call".into()); }
        if fuse_v { p.v = std::ptr::null_mut(); }
        if chunked {
            if self.state.len() != spark_gpu_probe::gdn_params::chunk_state_floats(self.heads, self.keys) {
                return Err("VADE_GDN_CHUNK=1 needs a chunk-form GDN state".into());
            }
            if let Some(e) = entry.as_ref() {
                if e.state.len() != self.state.len() { return Err("GDN entry snapshot is not chunk-form".into()); }
            }
        }
        // Chunk form snapshots S and the open chunk by a copy before the scan;
        // the sequential verify scan writes its snapshot itself.
        let snapshot = if chunked { 0 } else { entry.as_ref().map_or(0, |e| e.state.device_address()) };
        let addresses = [self.state.device_address(), self.history.device_address(), snapshot];
        if tape.is_some() && entry.is_none() { return Err("fused GDN verify copy needs an entry snapshot".into()); }
        // A cohort table (chunk form only) already holds this layer's words at `table`; a fused
        // verify copy rewrites them there (the same words). Otherwise the descriptor is uploaded,
        // or written by the fused copy.
        // The rows kernel takes the state address as a parameter: only larger pieces read the descriptor.
        let small = chunked && tokens <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT;
        // One chat's window inside a chunk, its tape already holding the rows: the history launch
        // writes the entry's storage and the two swap, so the entry keeps the old words with no copy.
        let swapped = tape.is_some_and(|t| t.iter().all(|r| r.2 == 0)) && shallow_verify(base, tokens);
        let rows = match table {
            Some(at) if chunked => at,
            Some(_) => return Err("GDN cohort descriptor table needs the chunk form".into()),
            None => {
                if tape.is_none() && !small {
                    b.write_vector(descriptor, &addresses.map(address).concat())?;
                }
                descriptor.device_address()
            }
        };
        let mut entry = entry;
        if let Some(entry) = entry.as_mut() {
            if chunked && !shallow_verify(base, tokens) { b.copy_range(&self.state, &mut entry.state, 0, 0, self.state.len())?; }
            if swapped {
                // No copy: flash_gdn_history_to below leaves the old words in place.
            } else if let Some(t) = tape {
                self.verify_copy(b, t, entry, if small { 0 } else { rows }, addresses)?;
            } else if !s.entry_copied.get() {
                // gdn::cohort_copies already copied this chat's entry history for the layer.
                b.copy_range(&self.history, &mut entry.history, 0, 0, self.history.len())?;
            }
            entry.cached = self.cached;
        }
        let mut scan = GdnStepBatch {
            q,
            k,
            v,
            beta,
            g,
            rows,
            out: core,
            sessions: 0,
            heads: self.heads as u32,
            count: tokens as u32,
        };
        let mut history = GdnHistory {
            x: at(&s.qkv, width) as _,
            history: self.history.device_address() as _,
            rows: if swapped { entry.as_ref().map_or(0, |e| e.history.device_address()) } else { 0 },
            channels: width as u32,
            tokens: tokens as u32,
            precision: 0,
        };
        let mut gate = GdnGate {
            x: core as _,
            z: at(&s.z, hd) as _,
            norm: s.norm_address() as _,
            output: at(&s.gated, hd) as _,
            heads: (self.heads * tokens) as u32,
            tokens: 1,
            precision: 0,
            packed: 0,
            epsilon: self.epsilon,
            sessions: 0,
        };
        unsafe {
            // Chunk form reads q/k on each key group's first value head; fuse_v calls convolve q/k/v and compute the gates in prep + scan;
            // fuse_rows windows do that in each head's rows CTA (own q/k slot, same words), and the output gate unless a cohort verify
            // pass defers it. A cohort verify layer (cohort_kernels) collects rows-kernel windows' launches (no prepare launch).
            let (collect, fuse_rows) = (s.cohort.borrow().is_some(), chunked && !fuse_v && tokens <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT);
            let fuse_gate = fuse_rows && emit && !s.defer_gate.get();
            if collect && (!chunked || fuse_v || tokens > spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT
                || !(emit && s.defer_gate.get())) {
                return Err("cohort GDN launches need deferred-gate rows-kernel windows".into());
            }
            if !fuse_v && !fuse_rows {
                // VADE_GDN_PREP_SPLIT (default on; =0 off): chunk-form windows of <= 16 rows
                // (verify) run one warp per value head (_khs), the same words.
                if chunked && tokens <= 16 && prep_split(b) {
                    b.launch_flash_shared("flash_gdn_conv_prepare_tokens_khs",
                        (self.keys*tokens*(self.heads/self.keys)*32).div_ceil(128) as u32, 128, 0, &mut [(&mut p as *mut GdnPrepare).cast()])?;
                } else {
                    b.launch_flash_shared(if chunked { "flash_gdn_conv_prepare_tokens_kh" } else { "flash_gdn_conv_prepare_tokens" },
                        (self.keys*tokens*32).div_ceil(128) as u32, 128, 0, &mut [(&mut p as *mut GdnPrepare).cast()])?;
                }
            }
            // Leading rows whose gate the chunk scan fused (whole CHUNK_PIECE pieces;
            // a short last piece, verify windows and replays use the launch).
            let mut gated_rows = 0usize;
            if chunked {
                // Pieces of <= CHUNK_PIECE rows at absolute positions; the open chunk
                // carries across pieces and calls (any cut, same bytes).
                let (hd4, h4) = ((self.heads * DIM * 4) as u64, (self.heads * 4) as u64);
                let (z0, gated0) = (at(&s.z, hd), at(&s.gated, hd));
                let x0 = if fuse_v || fuse_rows { at(&s.qkv, width) } else { 0 };
                let (a0, b0) = (at(&s.a, self.heads), at(&s.b, self.heads));
                let mut done = 0;
                while done < tokens {
                    let n = (tokens - done).min(CHUNK_PIECE);
                    let at = base + done;
                    let mut c = spark_gpu_probe::gdn_params::GdnChunkNext {
                        q: q + done as u64 * hd4, k: k + done as u64 * hd4, v: v + done as u64 * hd4,
                        beta: beta + done as u64 * h4, g: g + done as u64 * h4,
                        rows, out: core + done as u64 * hd4,
                        a: s.system.device_address(), p: s.attention.device_address(),
                        gamma: s.cumulative.device_address(),
                        k0: if n <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT { self.state.device_address() } else { s.decayed_keys.device_address() },
                        heads: self.heads as u32, keys: self.keys as u32, count: n as u32,
                        base: u32::try_from(at).map_err(|_| "GDN position past u32")?,
                        z: if fuse_gate { z0 + done as u64 * hd4 } else { 0 }, norm: if fuse_gate { s.norm_address() } else { 0 },
                        gated: if fuse_gate { gated0 + done as u64 * hd4 } else { 0 }, epsilon: self.epsilon, spare: 0,
                        x: x0, w: self.conv, history: self.history.device_address(),
                        channels: width as u32, x_start: done as u32,
                        // Fused q/k/gates (no prepare launch): in prep for fuse_v, in the rows kernel for fuse_rows.
                        ga: if fuse_v || fuse_rows { a0 + done as u64 * h4 } else { 0 }, gb: if fuse_v || fuse_rows { b0 + done as u64 * h4 } else { 0 }, alog: self.alog, dt: self.dt,
                        pk: packed_out.map_or(0, |o| o.0), pk_tokens: packed_out.map_or(0, |o| o.1 as u32),
                        pk_row0: (start + done) as u32,
                    };
                    if n <= spark_gpu_probe::gdn_params::CHUNK_ROWS_LIMIT {
                        // Verify windows and replays: the row function, same bytes. One launch never runs rows past a chunk's
                        // end (row 63): the key group's first head would overwrite tail key rows 0.. of the new chunk while
                        // the group's other CTAs still read the old ones for their close. The rest goes in a second launch (cut law).
                        let first = n.min(CHUNK - at % CHUNK);
                        c.count = first as u32;
                        let rest = if first < n {
                            let (r, f) = (first as u64 * hd4, first as u64 * h4);
                            Some(spark_gpu_probe::gdn_params::GdnChunkNext { q: c.q + r, k: c.k + r, v: c.v + r,
                                beta: c.beta + f, g: c.g + f, out: c.out + r, count: (n - first) as u32,
                                base: c.base + first as u32, x_start: c.x_start + first as u32, pk_row0: c.pk_row0 + first as u32,
                                ga: c.ga + f * (c.ga != 0) as u64, gb: c.gb + f * (c.gb != 0) as u64, z: c.z + r * (c.z != 0) as u64, gated: c.gated + r * (c.gated != 0) as u64, ..c })
                        } else { None };
                        if collect {
                            s.collect(|batch| { batch.rows[0].push(c); batch.rows[1].extend(rest); })?;
                        } else {
                            for mut c in std::iter::once(c).chain(rest) {
                                b.launch_flash_shared("flash_next_gdn_chunk_rows", self.heads as u32, 128, CHUNK_ROWS_BYTES, &mut [(&mut c as *mut spark_gpu_probe::gdn_params::GdnChunkNext).cast()])?;
                            }
                        }
                        done += n;
                        if fuse_gate { gated_rows = done; } continue;
                    }
                    let chunks = (at % CHUNK + n).div_ceil(CHUNK);
                    if s.system.len() < chunks * self.heads * CHUNK * CHUNK || s.decayed_keys.len() < self.keys * CHUNK * DIM {
                        return Err("chunk-form GDN workspace too small".into());
                    }
                    if emit {
                        // The scan applies the output gate itself (same words as
                        // flash_gdn_gate); these rows skip the gate launch below.
                        c.z = z0 + done as u64 * hd4;
                        c.norm = s.norm_address();
                        c.gated = gated0 + done as u64 * hd4;
                    }
                    let param = (&mut c as *mut spark_gpu_probe::gdn_params::GdnChunkNext).cast();
                    b.launch_flash_rows("flash_next_gdn_chunk_prep", chunks as u32, self.keys as u32, 256, 0, &mut [param])?;
                    let scan = if fuse_v { "flash_next_gdn_chunk_scan_cv" } else { "flash_next_gdn_chunk_scan" };
                    // The scan loads S and its first v rows in prep's tail; it waits
                    // (griddepcontrol) before reading prep's words, so bytes are unchanged.
                    b.programmatic_next();
                    b.launch_flash_shared(scan,self.heads as u32, 512, CHUNK_SCAN_BYTES, &mut [param])?;
                    done += n;
                    if emit { gated_rows = done; }
                }
            } else {
                b.launch_flash_shared(
                    if snapshot == 0 {
                        "flash_next_gdn_scan"
                    } else {
                        "flash_next_gdn_verify_scan"
                    },
                    (self.heads * 4) as u32,
                    32,
                    0,
                    &mut [(&mut scan as *mut GdnStepBatch).cast()],
                )?;
            }
            if collect {
                s.collect(|batch| batch.history.push(history))?;
            } else {
                b.launch_flash_shared(if swapped { "flash_gdn_history_to" } else { "flash_gdn_history" }, width.div_ceil(128) as u32, 128, 0,
                    &mut [(&mut history as *mut GdnHistory).cast()])?;
                if let Some(entry) = entry.as_mut().filter(|_| swapped) { std::mem::swap(&mut self.history, &mut entry.history); }
            }
            if emit && gated_rows < tokens && s.defer_gate.get() {
                // Cohort verify: Gdn::gate_rows gates the whole pass (no fused gate there).
                if gated_rows != 0 { return Err("deferred GDN gate after fused rows".into()); }
            } else if emit && gated_rows < tokens {
                let skip = gated_rows * hd;
                gate.x = gate.x.add(skip);
                gate.z = gate.z.add(skip);
                gate.output = gate.output.add(skip);
                gate.heads = (self.heads * (tokens - gated_rows)) as u32;
                b.launch_flash_shared(
                    "flash_gdn_gate",
                    (self.heads * (tokens - gated_rows) * 32).div_ceil(128) as u32,
                    128,
                    0,
                    &mut [(&mut gate as *mut GdnGate).cast()],
                )?;
            }
        }
        self.cached = true;
        Ok(())
    }
    /// Columns of the out projection's input (value heads x 128).
    pub(super) fn out_cols(&self) -> usize { self.heads * DIM }
    /// finish_next's out projection operands (BF16 weight, gated rows, K) when it reads the gated
    /// rows (the tensor split splits it); None while a fused scan packed its input.
    pub(super) fn out_operands(&self, s: &GdnScratch<'_>) -> Option<(u64, u64, usize)> {
        s.packed_out.get().is_none().then(|| (self.out, s.gated.device_address(), self.heads * DIM))
    }
    pub(super) fn finish_next(&self, b: &Backend<'_>, s: &GdnScratch<'_>, output: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if let Some((_, rows)) = s.packed_out.take() {
            // The gated scans already wrote flash_bf16_input's words for these rows.
            if rows != tokens { return Err("packed GDN output row count".into()); }
            return super::blocks::next_rows_prepacked(b, self.out, output.device_address(), self.hidden, self.heads*DIM, tokens);
        }
        super::blocks::next_rows(b, self.out, s.gated.device_address(), output.device_address(), self.hidden, self.heads*DIM, tokens)
    }
    /// Unpadded batch-one input [tokens,2560]. The first call and all multi-token
    /// calls use chunk64; exactly one cached token uses the recurrent algorithm.
    /// Synchronize and reduce scratch.output() before injecting into any HC.
    pub fn run(&mut self, b: &Backend<'_>, s: &mut GdnScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.forward(b, s, input, tokens, false, true, false, 0)
    }
    pub(super) fn project_batch(&self, b: &Backend<'_>, batch: &GdnBatch<'_>, input: &DeviceVector<'_>, count: usize, reuse: bool) -> Result<()> {
        if reuse && super::blocks::hc_packed(count) {
            super::blocks::bf16_row_pair_reuse(b, input.device_address(), HIDDEN, count,
                (self.qkv, batch.qkv.device_address(), (2 * self.keys + self.heads) * DIM, (2 * self.keys + self.heads) * DIM),
                (self.z, batch.z.device_address(), self.heads * DIM, self.heads * DIM))?;
        } else { projection_pair(b, input.device_address(), HIDDEN, count,
            (self.qkv, batch.qkv.device_address(), (2 * self.keys + self.heads) * DIM),
            (self.z, batch.z.device_address(), self.heads * DIM))?; }
        super::blocks::bf16_row_pair_reuse(b, input.device_address(), HIDDEN, count,
            (self.a, batch.a.device_address(), self.heads, self.heads),
            (self.b, batch.b.device_address(), self.heads, self.heads))
    }
    pub(super) fn session_row(&self, s: &GdnScratch<'_>, norm: &DeviceVector<'_>, row: usize) -> Result<GdnSessionRow> {
        if self.precision || !self.cached || s.heads != self.heads || s.keys != self.keys
            || s.hidden != self.hidden || norm.len() != DIM {
            return Err("Flash batch needs matching cached FP32 GDN state/scratch".into());
        }
        Ok(GdnSessionRow { state: self.state.device_address(), history: self.history.device_address(),
            convolved: s.convolved.device_address(), q: s.q.device_address(), k: s.k.device_address(),
            v: s.v.device_address(), beta: s.beta.device_address(), g: s.decay.device_address(),
            core: s.core.device_address(), norm: norm.device_address(), row: row as u32, padding: 0 })
    }
    pub(super) fn step_batch(&self, b: &Backend<'_>, batch: &GdnBatch<'_>, count: usize) -> Result<()> {
        let sessions = batch.sessions.as_ref().ok_or("missing GDN decode descriptors")?.device_address();
        let width = (2 * self.keys + self.heads) * DIM;
        launch_batch(b, "flash_gdn_conv_batch", count * width, GdnConvBatch {
            x: batch.qkv.device_address(), w: self.conv, rows: 0, y: 0, sessions,
            channels: width as u32, count: count as u32, precision: 0 })?;
        launch_batch(b, "flash_gdn_prepare_batch", count * self.heads * 32, GdnPrepareBatch {
            qkv: 0, a: batch.a.device_address(), b: batch.b.device_address(), alog: self.alog, dt: self.dt,
            q: 0, k: 0, v: 0, beta: 0, g: 0, sessions, key_heads: self.keys as u32,
            value_heads: self.heads as u32, key_start: self.key_start as u32, value_start: self.value_start as u32,
            count: count as u32, precision: self.parameter_precision })?;
        launch_batch(b, "flash_gdn_step_batch", count * self.heads * DIM, GdnStepBatch {
            q: 0, k: 0, v: 0, beta: 0, g: 0, rows: 0, out: 0, sessions,
            heads: self.heads as u32, count: count as u32 })?;
        launch_gate(b, "flash_gdn_gate", GdnGate { x: std::ptr::null(), z: batch.z.device_address() as _,
            norm: std::ptr::null(), output: batch.gated.device_address() as _, heads: self.heads as u32,
            tokens: 1, precision: self.parameter_precision, packed: 0, epsilon: self.epsilon, sessions }, count)
    }
    fn batch_inputs(&self, batch: &GdnBatch<'_>, row: usize) -> [u64; 5] {
        let at = |v: &DeviceVector<'_>, width: usize| v.device_address() + (row * width * 4) as u64;
        [at(&batch.qkv, (2 * self.keys + self.heads) * DIM), at(&batch.z, self.heads * DIM),
            at(&batch.a, self.heads), at(&batch.b, self.heads), at(&batch.gated, self.heads * DIM)]
    }
    /// Several canonical 64-row chunks per launch, carrying FP32 state in order.
    /// The caller ends spans at checkpoint boundaries before taking snapshots.
    pub(super) fn prefill_batch_rows(&mut self, b: &Backend<'_>, batch: &GdnBatch<'_>, start: usize,
        s: &GdnScratch<'_>, tokens: usize) -> Result<()> {
        if self.precision || tokens == 0 || tokens > s.capacity || start % CHUNK != 0
            || s.heads != self.heads || s.keys != self.keys || s.hidden != self.hidden {
            return Err("Flash prompt batch needs FP32 GDN state and a matching span workspace".into());
        }
        // finish_batch writes zero partials for a rank without value heads.
        if self.heads == 0 { return Ok(()); }
        let count = batch.z.len() / (self.heads * DIM);
        let packed = count > 64 && count % 16 == 0;
        let output = if packed { (b.flash_dense_input()? + (start * self.heads * DIM * 2) as u64,
            (count * self.heads * DIM) as u32) } else { (0, 0) };
        self.recur(b, s, tokens, true, false, output, self.batch_inputs(batch, start))?;
        self.cached = true;
        Ok(())
    }
    pub(super) fn finish_batch(&self, b: &Backend<'_>, batch: &GdnBatch<'_>, output: &mut DeviceVector<'_>, count: usize) -> Result<()> {
        if self.heads == 0 { return b.zero_vector(output); }
        if count > 64 && count % 16 == 0 {
            return super::blocks::bf16_row_pair_reuse(b, batch.gated.device_address(), self.heads * DIM, count,
                (self.out, output.device_address(), HIDDEN, HIDDEN), (0, 0, 0, 0));
        }
        projection_pair(b, batch.gated.device_address(), self.heads * DIM, count,
            (self.out, output.device_address(), HIDDEN), (0, 0, 0))
    }
    /// Tile a longer prefill at workspace boundaries, calling this for every tile.
    /// A final one-token tile must retain chunk arithmetic rather than decode.
    pub fn prefill(&mut self, b: &Backend<'_>, s: &mut GdnScratch<'_>, input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.forward(b, s, input, tokens, true, true, false, 0)
    }
    /// Dense prefill projects all inputs and prepares beta/decay in one launch.
    pub(crate) fn project_packed(&self, b: &Backend<'_>, s: &mut GdnScratch<'_>,
        matrices: [&spark_gpu_probe::driver::Matrix<'_>; 4], input: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        s.qkv.resize_within_capacity(tokens * matrices[0].rows())?;
        s.z.resize_within_capacity(tokens * matrices[1].rows())?;
        s.a.resize_within_capacity(tokens * self.heads)?;
        s.beta.resize_within_capacity(tokens * self.heads)?;
        s.decay.resize_within_capacity(tokens * self.heads)?;
        b.prefill_gdn_packed_bf16_device(matrices, input, [&s.qkv, &s.z, &s.a, &s.beta, &s.decay], self.alog, self.dt, tokens, self.parameter_precision)
    }
    /// Dense BF16 projections use the backend GEMMs. The caller fills QKV/Z/A/B
    /// and projects the gated recurrence output through its resident out matrix.
    pub(crate) fn run_projected(&mut self, b: &Backend<'_>, s: &mut GdnScratch<'_>, input: &DeviceVector<'_>, tokens: usize, prefill: bool, gates_ready: bool, packed_output: u64) -> Result<()> {
        self.forward(b, s, input, tokens, prefill, false, gates_ready, packed_output)
    }
    /// Native projections around the unchanged Flash FP32 recurrence.
    pub fn run_next_native(&mut self, b: &Backend<'_>, s: &mut GdnScratch<'_>,
        input: &DeviceVector<'_>, matrices: [super::next_native::Matrix; 5]) -> Result<()> {
        let ys = [s.qkv.device_address(), s.z.device_address(), s.a.device_address(), s.b.device_address()];
        super::next_native::project(b, &(0..4).map(|i| (matrices[i], input.device_address(), ys[i])).collect::<Vec<_>>(), 1, false, false)?;
        self.run_projected(b, s, input, 1, false, false, 0)?;
        super::next_native::project(b, &[(matrices[4], s.gated.device_address(), s.output.device_address())], 1, false, false)
    }
    /// Rank chunk prefill: one token's recurrence over row `row` of projected rows in `s`
    /// (qkv/z/a/b written by native projections of `rows` tokens): the launches of
    /// run_next_native's run_projected(1) on that row's addresses; gated row out.
    pub fn recur_row(&mut self, b: &Backend<'_>, s: &GdnScratch<'_>, row: usize, rows: usize) -> Result<()> {
        if self.heads == 0 || self.state.len() != self.heads * DIM * DIM || row >= rows || rows > s.capacity
            || s.heads != self.heads || s.keys != self.keys || s.hidden != self.hidden {
            return Err("GDN row recurrence does not match this rank".into());
        }
        let (width, hd) = ((2 * self.keys + self.heads) * DIM, self.heads * DIM);
        let at = |v: &DeviceVector<'_>, n: usize| v.device_address() + (row * n * 4) as u64;
        self.recur(b, s, 1, false, false, (0, 1), [at(&s.qkv, width), at(&s.z, hd), at(&s.a, self.heads),
            at(&s.b, self.heads), at(&s.gated, hd)])?;
        self.cached = true;
        Ok(())
    }
    /// Rank chunk prefill: cached rows start..start+tokens of projected rows in `s` through
    /// the ordered prompt scan (conv_prepare_tokens, flash_next_gdn_scan, history, gate): per
    /// row the conv/prepare words, recurrence column and gate of recur(1) on that row.
    pub fn scan_rows(&mut self, b: &Backend<'_>, s: &GdnScratch<'_>, start: usize, tokens: usize,
        descriptor: &mut DeviceVector<'_>) -> Result<()> {
        if !self.cached || chunk_form() || self.state.len() != self.heads * DIM * DIM {
            return Err("GDN row scan needs a cached row-form state".into());
        }
        self.scan_next(b, s, (start, tokens), descriptor, None, true, 0, None, None)
    }
    /// Whether the next token takes the cached recurrent step (else the cold path).
    pub fn is_cached(&self) -> bool { self.cached }
    fn forward(&mut self, b: &Backend<'_>, s: &mut GdnScratch<'_>, input: &DeviceVector<'_>, tokens: usize, force_chunk: bool, project_input: bool, gates_ready: bool, packed_output: u64) -> Result<()> {
        if self.heads != 0 && self.state.len() != self.heads * DIM * DIM {
            return Err("chunk-form GDN state (VADE_GDN_CHUNK=1) supports only the Flash-Next row paths".into());
        }
        if tokens == 0 || tokens > s.capacity || input.len() != tokens * self.hidden || input.is_segmented()
            || s.heads != self.heads || s.keys != self.keys || s.hidden != self.hidden {
            return Err("GDN input/scratch shape does not match this rank".into());
        }
        s.output.resize_within_capacity(tokens * self.hidden)?;
        if self.heads == 0 {
            return b.write_vector(&mut s.output, &vec![0.0; tokens * self.hidden]);
        }
        let width = (2 * self.keys + self.heads) * DIM;
        let hd = self.heads * DIM;
        let x = input.device_address();
        let project = |w, x, y, rows, cols| {
            if self.precision {
                launch(
                    b,
                    "flash_gdn_mv",
                    rows * tokens * 32,
                    &mut [
                        Ptr(w),
                        Ptr(x),
                        Ptr(y),
                        U32(rows as u32),
                        U32(cols as u32),
                        U32(tokens as u32),
                    ],
                )
            } else {
                projection(b, w, x, y, rows, cols, tokens)
            }
        };
        if project_input && !self.precision {
            projection_pair(b, x, self.hidden, tokens, (self.qkv, s.qkv.device_address(), width),
                (self.z, s.z.device_address(), hd))?;
            projection_pair(b, x, self.hidden, tokens, (self.a, s.a.device_address(), self.heads),
                (self.b, s.b.device_address(), self.heads))?;
        } else if project_input {
            project(self.qkv, x, s.qkv.device_address(), width, self.hidden)?;
            project(self.z, x, s.z.device_address(), hd, self.hidden)?;
            project(self.a, x, s.a.device_address(), self.heads, self.hidden)?;
            project(self.b, x, s.b.device_address(), self.heads, self.hidden)?;
        }
        // Batched decode reads/writes its projection row directly; recurrence
        // scratch, convolution history and state remain private to this session.
        let inputs = [s.qkv.device_address(), s.z.device_address(),
            s.a.device_address(), s.b.device_address(), s.gated.device_address()];
        self.recur(b, s, tokens, force_chunk, gates_ready, (packed_output, 1), inputs)?;
        if project_input { project(self.out, inputs[4], s.output.device_address(), self.hidden, hd)?; }
        self.cached = true;
        Ok(())
    }
    /// Convolution, canonical chunk recurrence, history and gate over projected rows.
    fn recur(&self, b: &Backend<'_>, s: &GdnScratch<'_>, tokens: usize, force_chunk: bool, gates_ready: bool,
        packed_output: (u64, u32), [qkv, z, a, beta_input, gated]: [u64; 5]) -> Result<()> {
        let _class = b.prefill_class(1);
        let width = (2 * self.keys + self.heads) * DIM;
        let hd = self.heads * DIM;
        let step = tokens == 1 && self.cached && !force_chunk;
        if !step {
            b.prefill_read_bytes(1, || {
                self.recurrent_bytes() * tokens.div_ceil(CHUNK)
                    + width * (4 * 2 + 6 * if self.precision { 2 } else { 4 })
                    + self.heads
                        * (if self.parameter_precision & 2 == 0 {
                            2
                        } else {
                            4
                        })
                    + self.heads
                        * (if self.parameter_precision & 4 == 0 {
                            2
                        } else {
                            4
                        })
                    + DIM
                        * if !self.precision || self.parameter_precision & 8 != 0 {
                            4
                        } else {
                            2
                        }
            });
        }
        // Grouped chunks convolve V inside solve; decode retains its one-row V scratch.
        let scale = u32::from(!step);
        b.profile_phase(23, || {
        if self.precision || (!step && self.key_repeat() == 3) {
            let mut params = GdnPrepare {
                x: qkv as _, w: self.conv as _, history: self.history.device_address() as _,
                a: if gates_ready { std::ptr::null() } else { a as _ }, b: beta_input as _,
                alog: self.alog as _, dt: self.dt as _, q: s.q.device_address() as _, k: s.k.device_address() as _,
                v: if step { s.v.device_address() as _ } else { std::ptr::null_mut() },
                beta: s.beta.device_address() as _, g: s.decay.device_address() as _,
                key_heads: self.keys as u32, value_heads: self.heads as u32, key_start: self.key_start as u32,
                value_start: self.value_start as u32, tokens: tokens as u32, precision: self.parameter_precision, scale };
            unsafe { b.launch_flash_shared("flash_gdn_conv_prepare", (self.keys * tokens * 32).div_ceil(128) as u32, 128, 0,
                &mut [(&mut params as *mut GdnPrepare).cast()]) }?;
        } else {
        launch(b, "flash_gdn_conv", width * tokens, &mut [Ptr(qkv), Ptr(self.conv),
            Ptr(self.history.device_address()), Ptr(s.convolved.device_address()), U32(width as u32), U32(tokens as u32), U32(u32::from(self.precision))])?;
        launch(b, "flash_gdn_prepare", self.heads * tokens * 32,
            &mut [Ptr(s.convolved.device_address()), Ptr(a), Ptr(beta_input),
                Ptr(self.alog), Ptr(self.dt), Ptr(s.q.device_address()), Ptr(s.k.device_address()),
                Ptr(s.v.device_address()), Ptr(s.beta.device_address()), Ptr(s.decay.device_address()),
                U32(self.keys as u32), U32(self.heads as u32), U32(self.key_start as u32),
                U32(self.value_start as u32), U32(tokens as u32), U32(self.parameter_precision), U32(scale)])?;
        }
        Ok(())
        })?;
        if step {
            // Three history taps in convolution, three in history update; weights exclude dense projections.
            b.decode_read_bytes(2, || {
                self.recurrent_bytes()
                    + width * (4 * 2 + 6 * if self.precision { 2 } else { 4 })
                    + self.heads
                        * (if self.parameter_precision & 2 == 0 {
                            2
                        } else {
                            4
                        })
                    + self.heads
                        * (if self.parameter_precision & 4 == 0 {
                            2
                        } else {
                            4
                        })
                    + DIM
                        * if !self.precision || self.parameter_precision & 8 != 0 {
                            4
                        } else {
                            2
                        }
            });
            b.profile_phase(18, || {
                launch(
                    b,
                    "flash_gdn_step",
                    hd,
                    &mut [
                        Ptr(s.q.device_address()),
                        Ptr(s.k.device_address()),
                        Ptr(s.v.device_address()),
                        Ptr(s.beta.device_address()),
                        Ptr(s.decay.device_address()),
                        Ptr(self.state.device_address()),
                        Ptr(s.core.device_address()),
                        U32(self.heads as u32),
                    ],
                )
            })?;
        } else {
            self.prepare_chunks(b, s, tokens, qkv)?;
            self.chunk(b, s, tokens, 0)?;
        }
        b.profile_phase(23, || {
            launch_history(
                b,
                GdnHistory {
                    x: qkv as _,
                    history: self.history.device_address() as _,
                    rows: 0,
                    channels: width as u32,
                    tokens: tokens as u32,
                    precision: u32::from(self.precision),
                },
                1,
            )
        })?;
        launch_gate(b, "flash_gdn_gate", GdnGate {
            x: s.core.device_address() as _, z: z as _,
            norm: if self.precision { self.norm as _ } else { s.norm_address() as _ },
            output: if packed_output.0 == 0 { gated as _ } else { packed_output.0 as _ },
            heads: self.heads as u32, tokens: tokens as u32, precision: self.parameter_precision,
            packed: if packed_output.0 == 0 { 0 } else { packed_output.1 }, epsilon: self.epsilon, sessions: 0 }, 1)
    }
    fn prepare_chunks(&self, b: &Backend<'_>, s: &GdnScratch<'_>, tokens: usize, qkv: u64) -> Result<()> {
        let h = self.heads as u32;
        let t = tokens as u32;
        let chunks = tokens.div_ceil(CHUNK);
        b.profile_phase(24, || {
        launch(b, "flash_gdn_cumulative", chunks * self.heads,
            &mut [Ptr(s.decay.device_address()), Ptr(s.cumulative.device_address()), Ptr(s.chunk_decay.device_address()), U32(h), U32(t)])?;
        {
            let mut params = spark_gpu_probe::kernel_params::GdnSystem {
                q: s.q.device_address() as *const f32, k: s.k.device_address() as *const f32,
                beta: s.beta.device_address() as *const f32, cumulative: s.cumulative.device_address() as *const f32,
                system: s.system.device_address() as *mut f32, attention: s.attention.device_address() as *mut f32,
                heads: h, tokens: t, key_repeat: self.key_repeat(),
            };
            unsafe { b.launch_flash_shared("flash_gdn_system", (chunks * self.heads * 4) as u32, GDN_SYSTEM_THREADS, GDN_SYSTEM_SHARED_BYTES,
                &mut [(&mut params as *mut spark_gpu_probe::kernel_params::GdnSystem).cast()]) }
        }
        })?;
        b.profile_phase(26, || {
            let mut params = spark_gpu_probe::kernel_params::GdnSolve {
                qkv: qkv as _, conv: self.conv as _, history: self.history.device_address() as _,
                keys: self.keys as u32, precision: self.parameter_precision,
                k: s.k.device_address() as _, v: if self.key_repeat() == 1 && !self.precision { s.v.device_address() as _ } else { std::ptr::null() },
                beta: s.beta.device_address() as _, cumulative: s.cumulative.device_address() as _,
                system: s.system.device_address() as _, new_values: s.values.device_address() as _,
                decayed_keys: s.decayed_keys.device_address() as _, heads: h, tokens: t, key_repeat: self.key_repeat(),
            };
            unsafe { b.launch_flash_shared("flash_gdn_solve", (chunks * self.heads * 2) as u32, 512, 0,
                &mut [(&mut params as *mut spark_gpu_probe::kernel_params::GdnSolve).cast()]) }
        })
    }
    /// Fused chunks require whole key-head groups; partial Flash groups stay expanded.
    fn key_repeat(&self) -> u32 {
        if self.precision { (self.heads / self.keys) as u32 }
        else if self.heads % 3 == 0 && self.value_start % 3 == 0 { 3 } else { 1 }
    }
    fn chunk(&self, b: &Backend<'_>, s: &GdnScratch<'_>, tokens: usize, start: usize) -> Result<()> {
        let h = self.heads as u32;
        let t = tokens as u32;
        let chunk = start / CHUNK;
        let start = start as u32;
        let chunk_decay = s.chunk_decay.device_address() + (chunk * self.heads * 4) as u64;
        let attention = s.attention.device_address() + (chunk * self.heads * CHUNK * CHUNK * 4) as u64;
        let values = s.values.device_address() + (chunk * self.heads * CHUNK * DIM * 4) as u64;
        let decayed_keys = s.decayed_keys.device_address() + (chunk * self.heads * CHUNK * DIM * 4) as u64;
        b.profile_phase(19, || {
            let mut params = spark_gpu_probe::kernel_params::GdnScan {
                q: s.q.device_address() as *const f32, k: s.k.device_address() as *const f32,
                attention: attention as *const f32, new_values: values as *const f32,
                decayed_keys: decayed_keys as *const f32, chunk_decay: chunk_decay as *const f32,
                cumulative: (s.cumulative.device_address() + (chunk * self.heads * CHUNK * 4) as u64) as *const f32,
                state: self.state.device_address() as *mut f32, output: s.core.device_address() as *mut f32,
                heads: h, tokens: t, start, key_repeat: self.key_repeat(),
            };
            unsafe { b.launch_flash_shared("flash_gdn_chunk_scan", (self.heads * (DIM / GDN_SCAN_COLUMNS)) as u32, GDN_SCAN_THREADS, GDN_SCAN_SHARED_BYTES,
                &mut [(&mut params as *mut spark_gpu_probe::kernel_params::GdnScan).cast()]) }
        })
    }
}
