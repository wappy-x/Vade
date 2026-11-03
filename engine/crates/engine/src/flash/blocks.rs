//! Four-stream residual connections and lexical projections. All activations
//! stay in FP32 storage; descriptors borrow the resident official BF16 weights.
use super::Weights;
use engine_model::{flash::plan::Selection, safetensors::Dtype};
use spark_gpu_probe::{driver::{Backend, DeviceVector}, kernel_params::{Bf16Rows, Bf16Pairs, FlashHcNorm, FlashHcAdd,
    FlashPleGate, FlashPleConv, BF16_ROWS_TOKENS, BF16_TILE_ROWS, BF16_TILE_TOKENS, BF16_MMA_ROWS, BF16_MMA_TOKENS, FLASH_BF16_MMA_SHARED, FLASH_BF16_WIDE_SHARED}};
use std::{ffi::c_void, marker::PhantomData};
use spark_gpu_probe::fp8_abi::MoeFinishParams;

type Result<T> = std::result::Result<T, String>;
pub const H: usize = 2560;
pub const W: usize = 4 * H;
const LOW: usize = 320;
/// Widest decode-arithmetic launch: batched decode rows and MTP verification
/// rows (draft depth + 1) both stay within it.
pub const DECODE_ROWS: usize = 16;
/// A crate::flash_draft_stack on/off switch: unset or 1 = on (the default since 2026-10-22),
/// 0 = off. NextModel's load refuses any other value before these are first read.
fn stack_on(name: &str) -> bool {
    crate::flash_draft_stack::switch_on(name).unwrap_or(true)
}
/// VADE_VERIFY_MX4 (default on; =0 off): first-party direct-input/K64-weight-ring port for
/// all-MXFP8 wide groups at M3/4. The fixed K16 graph is unchanged.
fn verify_mx4(tokens: usize, cols: usize, rows: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (3..=4).contains(&tokens) && cols <= 2560 && cols % 256 == 0 && rows > 4096
        && *ON.get_or_init(|| stack_on("VADE_VERIFY_MX4"))
}

/// VADE_VERIFY_MX4_MIXED (default on; =0 off): the M3-8 MoE projection group uses
/// direct-input K64 rings for its MXFP8 members, with the same BF16 router/scalar
/// path and four-range merge (M5-8: _mixed8; =0 keeps every member BF16 there, as
/// before). Other group shapes keep their production entry.
fn verify_mx4_mixed(tokens: usize, cols: usize, rows: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (3..=8).contains(&tokens) && cols == 2560 && rows == 1793
        && *ON.get_or_init(|| stack_on("VADE_VERIFY_MX4_MIXED"))
}


/// VADE_VERIFY_MX4_PART (default on; =0 off): cache one canonical input range
/// per CTA at M3-8 (M5-8: _part_group8, wide8's chains and merge; =0 keeps main's
/// wide8 there). Four K16 chains and their fixed merge are unchanged. Takes
/// precedence over VADE_VERIFY_MX4 when both are enabled, for integration.
fn verify_mx4_part(tokens: usize, cols: usize, rows: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (3..=8).contains(&tokens) && cols <= 2560 && cols % 256 == 0 && rows > 4096
        && *ON.get_or_init(|| stack_on("VADE_VERIFY_MX4_PART"))
}

/// VADE_VERIFY_MX_SOLO97 (default on; =0 off): exact sixteen-output-row
/// MXFP8 entry at M2-4. Each output keeps its K16 chains and fixed merge.
fn verify_mx_solo97(tokens: usize, rows: usize, cols: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (2..=4).contains(&tokens) && rows % 64 == 0 && cols % 256 == 0 && cols <= 6144
        && *ON.get_or_init(|| stack_on("VADE_VERIFY_MX_SOLO97"))
}


/// MXFP8 twins of expanded BF16 matrices live in crate::flash_mx8_registry, keyed by the
/// BF16 address: (twin, rows, cols). Decode launches (M <= 16) read the twin as stored; the
/// latency kernel expands each fragment in registers to the BF16 words of the load-time
/// expansion, so the output bytes are the BF16 matrix's. Prompt launches keep the BF16 copy.
/// Each entry is owned by the Mx8Twin that holds the twin's memory.
use crate::flash_mx8_registry as mx8_registry;
/// LBF12 twins: packed BF16 matrix address -> (its flash_bf16_lbf12_twin image, rows, cols).
static LBF12: std::sync::RwLock<std::collections::BTreeMap<u64, (u64, usize, usize)>> = std::sync::RwLock::new(std::collections::BTreeMap::new());
/// Build and register the LBF12 twin of a packed BF16 [rows][cols] matrix (rows % 64 == 0,
/// cols % 32 == 0, cols <= 4096): 12-bit slots where a slot's rows fit, flags + row bases;
/// decode kernels read it in place of the BF16 bytes and get the identical fragments.
pub(super) fn lbf12_twin<'a>(b: &Backend<'a>, weight: u64, rows: usize, cols: usize) -> Result<Option<DeviceVector<'a>>> {
    if rows % 64 != 0 || cols % 32 != 0 || cols > 4096 || !b.has_flash("flash_bf16_lbf12_twin") { return Ok(None); }
    let (slots, bytes) = (rows / 8 * (cols / 32), lbf12_bytes(rows, cols));
    let mut v = b.vector(bytes.div_ceil(4))?;
    b.write_vector_bytes(&mut v, &vec![0u8; bytes.div_ceil(4) * 4])?;
    let mut pack = spark_gpu_probe::kernel_params::FlashBf16Pack { source: weight, weight: v.device_address(), rows: rows as u32, cols: cols as u32 };
    unsafe {
        b.launch_flash("flash_bf16_lbf12_twin", (slots * 32).div_ceil(128) as u32, 128,
            &mut [(&mut pack as *mut spark_gpu_probe::kernel_params::FlashBf16Pack).cast()])?;
    }
    LBF12.write().map_err(|_| "LBF12 registry poisoned")?.insert(weight, (v.device_address(), rows, cols));
    Ok(Some(v))
}
fn lbf12_bytes(rows: usize, cols: usize) -> usize { rows / 8 * (cols / 32) * 392 + rows / 8 * (cols / 32).div_ceil(32) * 4 }
fn lbf12_of(weight: u64, size: Option<(usize, usize)>) -> Option<u64> { LBF12.read().ok().and_then(|m| m.get(&weight).filter(|t| size.is_none_or(|s| s == (t.1, t.2))).map(|t| t.0)) }
/// The FP8 rank path's LBF12 twins of its packed HC up weights ([W][LOW], replicated; built on every resident rank,
/// with no room check): K320 decode rows read them in place of the BF16 bytes (same B fragments). Dropping the set removes its entries.
#[derive(Default)] pub(super) struct Lbf12Twins<'a>(Vec<(u64, DeviceVector<'a>)>);
impl<'a> Lbf12Twins<'a> {
    pub(super) fn hc_up(b: &Backend<'a>, w: &Weights<'_>) -> Result<Self> {
        let mut twins = Self(Vec::new());
        for name in w.tensors.keys().filter(|name| name.ends_with(".input_mix_weight_up.weight")) {
            let Ok(address) = weight(w, name, &[W as u64, LOW as u64]) else { continue };
            if let Some(twin) = lbf12_twin(b, address, W, LOW)? { twins.0.push((address, twin)); }
        }
        b.synchronize()?; println!("flash_session hc_up_lbf12 matrices={} bytes={}", twins.0.len(), twins.0.iter().map(|t| t.1.len() * 4).sum::<usize>());
        Ok(twins)
    }
    /// hc_up's twins (no room check; the load plan does not count them), then the rank's attention projections' (native
    /// K2560/K3072 decode rows). Only the attention twins are checked: each only while it fits above the funded `reserve`.
    pub(super) fn fp8(b: &Backend<'a>, w: &Weights<'_>, reserve: usize) -> Result<Self> {
        let mut twins = Self::hc_up(b, w)?; let hc = twins.0.len();
        let mut room = b.allocation_budget().saturating_sub(b.allocated_bytes()).saturating_sub(reserve);
        for (address, rows, cols) in w.attention_matrices() { if lbf12_bytes(rows, cols) > room { continue }
            if let Some(twin) = lbf12_twin(b, address, rows, cols)? { room = room.saturating_sub(twin.len() * 4); twins.0.push((address, twin)); } }
        b.synchronize()?; println!("flash_session attention_lbf12 matrices={} bytes={}", twins.0.len() - hc, twins.0[hc..].iter().map(|t| t.1.len() * 4).sum::<usize>());
        Ok(twins)
    }
}
impl Drop for Lbf12Twins<'_> { fn drop(&mut self) { if let Ok(mut map) = LBF12.write() { for (address, _) in &self.0 { map.remove(address); } } } }
/// An MXFP8 twin's memory and its registry entry. Fields drop in order: the entry is
/// removed first, then the memory is freed (teardown and partial-load unwinding alike).
pub(super) struct Mx8Twin<'a> {
    _entry: mx8_registry::Registration,
    pub(super) twin: DeviceVector<'a>,
}
impl<'a> Mx8Twin<'a> {
    /// Register `twin` under the BF16 matrix at `bf16` ([rows][cols]) for as long as it lives.
    pub(super) fn register(bf16: u64, twin: DeviceVector<'a>, rows: usize, cols: usize) -> Self {
        Self { _entry: mx8_registry::register(bf16, twin.device_address(), rows, cols), twin }
    }
}
/// Whether `bf16` has a registered MXFP8 twin of exactly [rows][cols].
pub(super) fn mx8_has(bf16: u64, rows: usize, cols: usize) -> bool {
    mx8_registry::has(bf16, rows, cols)
}
/// VADE_FLASHNEXT_DRAFT_MX8_LAYER (default on; =0 off): the MTP layer's BF16 matrices get
/// drafter-only MXFP8 twins. Off, no drafter launch consults the registry.
pub(super) fn draft_mx8_layer() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| stack_on("VADE_FLASHNEXT_DRAFT_MX8_LAYER"))
}
/// Bytes of an MXFP8 twin of [rows][cols] (mx8_image's layout), rounded to 64.
pub(super) fn mx8_twin_bytes(rows: usize, cols: usize) -> usize {
    ((rows * cols).div_ceil(256) * 256 + rows.div_ceil(8) * (cols / 32) * 8).div_ceil(64) * 64
}
/// VADE_FLASHNEXT_DRAFT_MX8_LAYER (drafter only): build the MXFP8 twin of an official
/// BF16 MTP-layer matrix, stored packed (flash_bf16_element order) by the loader, with
/// flash_bf16_mx8_twin, and register it under the BF16 address for as long as the returned
/// Mx8Twin lives, so the matrix's M <= 4 latency launches read the twin.
pub(super) fn load_mx8_twin<'a>(b: &Backend<'a>, weight: u64, rows: usize, cols: usize) -> Result<Mx8Twin<'a>> {
    if rows % 8 != 0 || cols % 64 != 0 || !b.has_flash("flash_bf16_mx8_twin") {
        return Err(format!("no MXFP8 drafter twin for [{rows}][{cols}] (needs flash_bf16_mx8_twin, N % 8, K % 64)"));
    }
    let twin = b.vector(mx8_twin_bytes(rows, cols) / 4)?;
    let mut pack = spark_gpu_probe::kernel_params::FlashBf16Pack { source: weight, weight: twin.device_address(), rows: rows as u32, cols: cols as u32 };
    unsafe {
        b.launch_flash("flash_bf16_mx8_twin", (rows * (cols / 32)).div_ceil(128) as u32, 128,
            &mut [(&mut pack as *mut spark_gpu_probe::kernel_params::FlashBf16Pack).cast()])?;
    }
    Ok(Mx8Twin::register(weight, twin, rows, cols))
}
/// The latency kernel's weight word for a whole matrix: the twin flagged by bit 0.
/// Either storage gives the same bytes; the twin is picked where kbench measured it
/// faster: M <= 4 (M8/M16 run slower; with K32 stages, N 48 runs faster too).
fn mx8_weight(weight: u64, rows: usize, cols: usize, tokens: usize) -> u64 {
    // M5-16 reads MXFP8 only where the kernel keeps a CTA input copy (latency_mx8_wide,
    // latency_mx8_wide16).
    if tokens > 4 && !spark_gpu_probe::kernel_params::latency_mx8_wide(tokens, cols)
        && !spark_gpu_probe::kernel_params::latency_mx8_wide16(tokens, cols)
        && !spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols) {
        return weight;
    }
    match mx8_registry::get(weight, rows) {
        Some((twin, r, c)) if (r, c) == (rows, cols) => twin | 1,
        _ => weight,
    }
}
/// Host image of an MXFP8 twin: E4M3 bytes in the packed BF16 element order, then at
/// the next 256-byte boundary UE8M0 scales as [row / 8][K / 32][8] (padding rows 127).
/// None unless N % 4 == 0, K % 64 == 0 and every scale is in 10..=134 (the kernel's
/// exact range; the BF16 copy serves anything else).
pub(super) fn mx8_image(weights: &[u8], scales: &[u8], rows: usize, cols: usize) -> Option<Vec<u8>> {
    if rows % 4 != 0 || cols % 64 != 0 || weights.len() != rows * cols
        || scales.len() != rows * cols / 32 || scales.iter().any(|s| !(10..=134).contains(s))
    {
        return None;
    }
    let slot = (rows * cols).div_ceil(256) * 256;
    let mut out = vec![0u8; slot + rows.div_ceil(8) * (cols / 32) * 8];
    for r in 0..rows {
        for c in 0..cols {
            out[spark_gpu_probe::kernel_params::flash_bf16_element(r, c, cols, rows)] = weights[r * cols + c];
        }
    }
    out[slot..].fill(127);
    for r in 0..rows {
        for block in 0..cols / 32 {
            out[slot + (r / 8 * (cols / 32) + block) * 8 + r % 8] = scales[r * (cols / 32) + block];
        }
    }
    Some(out)
}
/// HC mix emits reusable prompt operands only for complete workspace tiles.
pub(super) fn hc_packed(tokens: usize) -> bool {
    tokens > DECODE_ROWS && tokens % 16 == 0 && tokens <= HcScratch::workspace_tokens()
}

enum Arg { Ptr(u64), U32(u32), F32(f32) }
impl Arg {
    fn address(&mut self) -> *mut c_void {
        match self {
            Self::Ptr(v) => (v as *mut u64).cast(),
            Self::U32(v) => (v as *mut u32).cast(),
            Self::F32(v) => (v as *mut f32).cast(),
        }
    }
}
use Arg::{F32, Ptr, U32};

fn launch(b: &Backend<'_>, name: &str, blocks: usize, args: &mut [Arg]) -> Result<()> {
    launch_tokens(b, name, blocks, 1, args)
}
/// Tile kernels take the token from the second grid dimension.
fn launch_tokens(b: &Backend<'_>, name: &str, blocks: usize, tokens: usize, args: &mut [Arg]) -> Result<()> {
    let mut pointers = [std::ptr::null_mut(); 9];
    for (p, arg) in pointers.iter_mut().zip(args.iter_mut()) { *p = arg.address(); }
    unsafe { b.launch_flash_rows(name, blocks as u32, tokens as u32, 128, 0, &mut pointers[..args.len()]) }
}
/// Resident official BF16 weights times [tokens, cols] FP32 rows through
/// split BF16 hi/lo tensor-core operands, with output y[token * stride + row].
pub fn bf16_rows(b: &Backend<'_>, weight: u64, x: u64, y: u64, rows: usize, cols: usize,
    tokens: usize, stride: usize) -> Result<()> {
    bf16_row_pair(b, x, cols, tokens, (weight, y, rows, stride), (0, 0, 0, 0))
}
/// Flash-Next uses this same K16 operation graph for every M, including one.
/// Prompt scheduling may change M/N tiling while keeping that operation graph.
pub(super) fn next_rows(b: &Backend<'_>, weight: u64, x: u64, y: u64, rows: usize, cols: usize, tokens: usize) -> Result<()> {
    next_rows_strided(b, weight, x, y, rows, cols, tokens, rows)
}
pub(super) fn next_rows_strided(
    b: &Backend<'_>,
    weight: u64,
    x: u64,
    y: u64,
    rows: usize,
    cols: usize,
    tokens: usize,
    stride: usize,
) -> Result<()> {
    if next_prefill(
        b,
        x,
        cols,
        tokens,
        [
            (weight, y, rows, stride),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ],
    )? {
        return Ok(());
    }
    // M17-32 with an MXFP8 twin, K <= 6144 and N % 64 == 0 (N > 640): flash_mx8_mma_latency_r
    // for both token tiles (one weight read for every token; each row's K16 chains and merge are
    // the M <= 16 ones, tied to flash_bf16_mma's rows by the row law). Same input pack format.
    let wide32 = tokens > 16 && rows > 640 && rows % 64 == 0 && cols % 64 == 0
        && spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols)
        && mx8_weight(weight, rows, cols, tokens) & 1 != 0;
    let packed = if tokens <= 16 || wide32 {
        b.pack_flash_latency_input(x, cols, tokens)?
    } else {
        b.pack_flash_dense_input(x, cols, tokens)?
    };
    let mut p = row_params(
        packed,
        cols,
        tokens,
        (weight, y, rows, stride),
        (0, 0, 0, 0),
        true,
        false,
    );
    if tokens <= 16 || wide32 {
        // latency_mx8_widen launches run flash_mx8_mma_latency_r, which needs N % 64 == 0
        // (the 8-row _w entries are retired); other N keep BF16 there.
        if cols % 64 == 0 && (tokens <= 4 || rows > 640)
            && !(spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols) && rows % 64 != 0) {
            p.weight = mx8_weight(weight, rows, cols, tokens);
        }
        // One MMA already writes both eight-row halves of the packed M16 tile.
        // The reserved packed-input tile has ample unused tail at M<=16.
        // Stream ordering protects this tail through both launches and replay.
        let mut partials = packed + p.input_plane_words * 8;
        debug_assert!(
            p.input_plane_words as usize * 8 + 4 * tokens * rows.min(640) * 4
                <= spark_gpu_probe::kernel_params::FLASH_BF16_INPUT_BYTES
        );
        // BF16 M1 with N <= 64 (latency_fold1): one 128-thread launch of the fold entry merges
        // the four ranges inside the CTA ((p0+p1)+(p2+p3), the merge pass's sums), no second launch.
        let fold = p.weight & 1 == 0 && spark_gpu_probe::kernel_params::latency_fold1(tokens, cols, rows);
        let threads = if rows <= 640 && !fold { 32 } else { 128 };
        // A flagged (MXFP8) matrix runs the twin entry; its scales follow its bytes.
        let mut scales = 0u64;
        unsafe {
            if p.weight & 1 != 0 && spark_gpu_probe::kernel_params::latency_mx8_range_warps(tokens, cols, rows) {
                let mut group = Bf16Pairs { pairs: [p, row_params(packed, cols, tokens, (0, 0, 0, 0), (0, 0, 0, 0), true, false)] };
                return b.launch_flash_rows("flash_mx8_mma_latency_group", 192, 1, 128, 4 * spark_gpu_probe::kernel_params::LATENCY_RING as u32,
                    &mut [(&mut group as *mut Bf16Pairs).cast(), (&mut partials as *mut u64).cast()]);
            }
            if p.weight & 1 != 0 {
                let r1 = threads == 128 && verify_mx_solo97(tokens, rows, cols);
                b.launch_flash_rows(
                    if r1 { b.chain_entry(tokens, "flash_verify_mx_solo97", "flash_verify_mx_solo97_chain") } else if threads == 128 && spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols) {
                        // Wide input staging in 16-row warp tasks (latency_rows.rs): their own
                        // entry, so C1's entry keeps main's code (N % 64 == 0 here, see the twin pick above).
                        "flash_mx8_mma_latency_r"
                    } else {
                        b.chain_solo(tokens, cols, threads)
                    },
                    192,
                    1,
                    threads,
                    if r1 { spark_gpu_probe::kernel_params::verify_solo97_shared(cols) }
                    else { spark_gpu_probe::kernel_params::latency_mx8_shared_rows(tokens, cols, threads as usize / 32, rows, false) } as u32,
                    &mut [
                        (&mut p as *mut Bf16Rows).cast(),
                        (&mut scales as *mut u64).cast(),
                        (&mut partials as *mut u64).cast(),
                    ],
                )?;
            } else {
                b.launch_flash_rows(
                    if fold { "flash_bf16_mma_latency_fold" } else { "flash_bf16_mma_latency" },
                    192,
                    1,
                    threads,
                    if fold { spark_gpu_probe::kernel_params::latency_fold_shared(tokens, cols, rows) as u32 }
                    else { threads / 32 * spark_gpu_probe::kernel_params::LATENCY_RING as u32 },
                    &mut [
                        (&mut p as *mut Bf16Rows).cast(),
                        (&mut partials as *mut u64).cast(),
                    ],
                )?;
            }
            if rows <= 640 && !fold {
                b.launch_flash_rows(
                    "flash_bf16_mma_latency_merge",
                    (rows * tokens).div_ceil(128) as u32,
                    1,
                    128,
                    0,
                    &mut [
                        (&mut p as *mut Bf16Rows).cast(),
                        (&mut partials as *mut u64).cast(),
                    ],
                )?;
            }
        }
        return Ok(());
    }
    unsafe {
        b.launch_flash_shared(
            "flash_bf16_mma",
            (rows.div_ceil(32) * tokens.div_ceil(16)) as u32,
            128,
            0,
            &mut [(&mut p as *mut Bf16Rows).cast()],
        )
    }
}

/// Draft head: rows of an MXFP8 image (flash_next_head_mx8 layout) over an FP32
/// input at M <= 4 (M 5-8 where the latency entry keeps its CTA input copy), through
/// the MXFP8 latency entry; logits FP32 [tokens][rows].
pub(super) fn mx8_rows(b: &Backend<'_>, twin: u64, x: u64, y: u64, rows: usize, cols: usize, tokens: usize) -> Result<()> {
    if tokens == 0 || tokens > 4 && !spark_gpu_probe::kernel_params::latency_mx8_wide(tokens, cols)
        || rows <= 640 || cols % 64 != 0 {
        return Err("MXFP8 draft head shape".into());
    }
    let packed = b.pack_flash_latency_input(x, cols, tokens)?;
    let mut p = row_params(packed, cols, tokens, (twin | 1, y, rows, rows), (0, 0, 0, 0), true, false);
    let (mut scales, mut partials) = (0u64, 0u64);
    unsafe {
        b.launch_flash_rows(
            "flash_mx8_mma_latency",
            192,
            1,
            128,
            spark_gpu_probe::kernel_params::latency_mx8_shared_rows(tokens, cols, 4, rows, false) as u32,
            &mut [
                (&mut p as *mut Bf16Rows).cast(),
                (&mut scales as *mut u64).cast(),
                (&mut partials as *mut u64).cast(),
            ],
        )
    }
}
/// Group independent projections over one FP32 input. Decode (M<=16) runs them in one
/// latency launch over one packed input; each output keeps its solo K16 graph.
pub(super) fn next_group(
    b: &Backend<'_>,
    x: u64,
    cols: usize,
    tokens: usize,
    rows: [(u64, u64, usize); 4],
) -> Result<()> {
    if next_prefill(b, x, cols, tokens, rows.map(|r| (r.0, r.1, r.2, r.2)))? {
        return Ok(());
    }
    next_rows_group(b, x, cols, tokens, rows)
}
/// Rows from which the prompt HC up runs flash_next_hc_up_tiled (same bytes as latency4;
/// kbench pgu-hc: 0.83x at 2,048 rows, slower at 65).
const TILED_HC_UP_ROWS: usize = 512;
/// Up to four decode projections of one FP32 input (rows 0 = absent; outputs stride =
/// rows): one packed input and one flash_bf16_mma_latency_group launch (groups of <= 640
/// rows merge their four ranges inside the CTA). Each output keeps the solo launch's
/// canonical ranges and (p0+p1)+(p2+p3) merge, so the bytes equal per-matrix next_rows.
pub(super) fn next_rows_group(b: &Backend<'_>, x: u64, cols: usize, tokens: usize, rows: [(u64, u64, usize); 4]) -> Result<()> {
    next_rows_group_at(b, x, cols, tokens, rows.map(|r| (r.0, r.1, r.2, r.2)))
}
/// next_rows_group with each output's row stride (weight, output, rows, stride): the tensor split
/// launches a rank's row half of a matrix into its columns of the full-width output.
pub(super) fn next_rows_group_at(b: &Backend<'_>, x: u64, cols: usize, tokens: usize, rows: [(u64, u64, usize, usize); 4]) -> Result<()> {
    // Prompt tiles share one packed input and the tiled launches (HC injection + down).
    if next_prefill(b, x, cols, tokens, rows)? {
        return Ok(());
    }
    // Small groups (latency_split: <= 4096 rows, e.g. HC [4, 320] x 10240) at 17-32 tokens: one
    // flash_bf16_mma_latency_group2 launch (the group entry's split path with both 16-token tiles
    // per weight fragment) instead of one flash_bf16_mma launch per matrix (HC at M20: 60 + 76 us).
    // Every (row, token) keeps the canonical K16 graph, so the bytes equal flash_bf16_mma's (kbench
    // law-* cases); these groups read BF16 at M > 4.
    let total: usize = rows.iter().map(|r| r.2).sum();
    if tokens > DECODE_ROWS && tokens <= 2 * DECODE_ROWS && cols % 64 == 0
        && rows.iter().filter(|r| r.2 != 0).count() >= 2
        && spark_gpu_probe::kernel_params::latency_split(total, true) {
        let packed = b.pack_flash_latency_input(x, cols, tokens)?;
        let at = |r: (u64, u64, usize, usize)| r;
        let mut p = Bf16Pairs {
            pairs: [
                row_params(packed, cols, tokens, at(rows[0]), at(rows[1]), true, false),
                row_params(packed, cols, tokens, at(rows[2]), at(rows[3]), true, false),
            ],
        };
        return unsafe {
            b.launch_flash_rows("flash_bf16_mma_latency_group2", 192, 1, 128,
                spark_gpu_probe::kernel_params::LATENCY_GROUP2_SHARED as u32,
                &mut [(&mut p as *mut Bf16Pairs).cast()])
        };
    }
    // Big all-MXFP8 groups (gdnin, qsa) at 17-32 tokens: one flash_mx8_mma_latency_w_group
    // launch (wide_n's two-tile mode; one weight read per matrix for every token, the 48-row GDN
    // a/b members ride along) instead of one launch per matrix. Each row's chains and merge are
    // the solo _w ones (same wide_n), tied to flash_bf16_mma's rows by the row law.
    if tokens > DECODE_ROWS && tokens <= 2 * DECODE_ROWS && cols % 64 == 0
        && rows.iter().filter(|r| r.2 != 0).count() >= 2
        && !spark_gpu_probe::kernel_params::latency_split(total, true)
        && spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols)
        && rows.iter().all(|r| r.2 == 0 || mx8_weight(r.0, r.2, cols, tokens) & 1 != 0) {
        let packed = b.pack_flash_latency_input(x, cols, tokens)?;
        let at = |r: (u64, u64, usize, usize)| r;
        let mut p = Bf16Pairs {
            pairs: [
                row_params(packed, cols, tokens, at(rows[0]), at(rows[1]), true, false),
                row_params(packed, cols, tokens, at(rows[2]), at(rows[3]), true, false),
            ],
        };
        for (q, r) in p.pairs.iter_mut().zip(rows.chunks(2)) {
            q.weight = mx8_weight(q.weight, r[0].2, cols, tokens);
            q.weight2 = mx8_weight(q.weight2, r[1].2, cols, tokens);
        }
        let mut partials = 0u64;
        // flash_mx8_mma_latency_r_group = _w_group with the N % 64 == 0 members in 16-row warp
        // tasks (_r) and the others (the 48-row GDN a/b) in _w_group's 8-row tasks, one launch
        // and one input pack (the earlier path ran _r per big member + group2 for the small ones).
        return unsafe {
            b.launch_flash_rows("flash_mx8_mma_latency_r_group", 192, 1, 128,
                spark_gpu_probe::kernel_params::latency_mx8_shared_rows(tokens, cols, 4, total, true) as u32,
                &mut [(&mut p as *mut Bf16Pairs).cast(), (&mut partials as *mut u64).cast()])
        };
    }
    if tokens > DECODE_ROWS || cols % 64 != 0 || rows.iter().filter(|r| r.2 != 0).count() < 2 {
        for (w, y, n, stride) in rows {
            if n != 0 && stride != n { return Err("strided group output outside the grouped decode launch".into()); }
            if n != 0 {
                next_rows(b, w, x, y, n, cols, tokens)?;
            }
        }
        return Ok(());
    }
    let packed = b.pack_flash_latency_input(x, cols, tokens)?;
    let at = |r: (u64, u64, usize, usize)| r;
    let mut p = Bf16Pairs {
        pairs: [
            row_params(packed, cols, tokens, at(rows[0]), at(rows[1]), true, false),
            row_params(packed, cols, tokens, at(rows[2]), at(rows[3]), true, false),
        ],
    };
    for (q, r) in p.pairs.iter_mut().zip(rows.chunks(2)) {
        q.weight = mx8_weight(q.weight, r[0].2, cols, tokens);
        q.weight2 = mx8_weight(q.weight2, r[1].2, cols, tokens);
    }
    // Every present matrix MXFP8: the single-storage entry (same ranges, chains, merges).
    let mx = rows.iter().zip([p.pairs[0].weight, p.pairs[0].weight2, p.pairs[1].weight, p.pairs[1].weight2]).all(|(r, w)| r.2 == 0 || w & 1 != 0);
    let total = rows.iter().map(|r| r.2).sum();
    let mixed = !mx && verify_mx4_mixed(tokens, cols, total);
    if tokens > 4 && !mx && !mixed {
        // M5-8 MXFP8 runs only in the single-storage group entries and the MoE mixed entry:
        // other mixed groups stay BF16.
        for (q, r) in p.pairs.iter_mut().zip(rows.chunks(2)) {
            (q.weight, q.weight2) = (r[0].0, r[1].0);
        }
    }
    let part = mx && verify_mx4_part(tokens, cols, total);
    let direct = mx && verify_mx4(tokens, cols, total);
    let wide = tokens > 4;
    let mut partials = 0u64;
    unsafe {
        b.launch_flash_rows(
            if part && wide { "flash_verify_mx4_part_group8" } else if part { b.chain_entry(tokens, "flash_verify_mx4_part_group", "flash_verify_mx4_part_group_chain") }
            else if direct { "flash_verify_mx4_group" }
            else if mixed && wide { "flash_verify_mx4_mixed8" } else if mixed { "flash_verify_mx4_mixed_r2" }
            else if mx && !spark_gpu_probe::kernel_params::latency_split(rows.iter().map(|r| r.2).sum(), true)
                && spark_gpu_probe::kernel_params::latency_mx8_widen(tokens, cols) {
                // 16-row warp tasks for the N % 64 == 0 members (same bytes as _w_group).
                "flash_mx8_mma_latency_r_group"
            } else { b.chain_group(tokens, mx, p.pairs.iter().all(|q| (q.weight | q.weight2) & 1 == 0), total, cols) },
            192,
            1,
            128,
            if part && wide { spark_gpu_probe::kernel_params::VERIFY_MX4_PART8_SHARED }
            else if part { spark_gpu_probe::kernel_params::VERIFY_MX4_PART_SHARED }
            else if direct { 4 * spark_gpu_probe::kernel_params::LATENCY_W8_RING }
            else if mx { spark_gpu_probe::kernel_params::latency_mx8_shared_rows(tokens, cols, 4, total, true) } else { 4 * spark_gpu_probe::kernel_params::LATENCY_RING } as u32,
            &mut [(&mut p as *mut Bf16Pairs).cast(), (&mut partials as *mut u64).cast()],
        )
    }
}
/// Prompt staging preserves every canonical K16 graph. COLD8K_R2 hoists
/// full-tile source addresses; disabling it retains the preceding u32 entry.
fn prompt_tiled_entry(tokens: usize) -> &'static str {
    let options = crate::flash_prompt_options::cached();
    if options.gemm_u32 {
        if options.cold8k_r2 && (8192..=32768).contains(&tokens) { "flash_bf16_mma_prefill_tiled_plan" }
        else { "flash_bf16_mma_prefill_tiled_u32" }
    } else {
        "flash_bf16_mma_prefill_tiled"
    }
}
/// next_prefill's tiled GEMM on an input a fused producer already wrote into the
/// dense input tile (b.flash_dense_input(), flash_bf16_input's layout and words,
/// e.g. the GDN gated scan): the same launch and parameters, without the pack.
/// (GDN owner, A1; for prefill_gemm review.)
pub(super) fn next_rows_prepacked(b: &Backend<'_>, weight: u64, y: u64, rows: usize, cols: usize, tokens: usize) -> Result<()> {
    if tokens < 64 || !cols.is_multiple_of(64) || rows < 64 || !rows.is_multiple_of(64) || !weight.is_multiple_of(16) {
        return Err("prepacked prefill shape".into());
    }
    let mut p = row_params(b.flash_dense_input()?, cols, tokens, (weight, y, rows, rows), (0, 0, 0, 0), true, false);
    unsafe {
        b.launch_flash_rows(
            prompt_tiled_entry(tokens),
            tokens.div_ceil(64) as u32,
            rows.div_ceil(128) as u32,
            256,
            spark_gpu_probe::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32,
            &mut [(&mut p as *mut Bf16Rows).cast()],
        )?;
    }
    Ok(())
}
fn next_prefill(
    b: &Backend<'_>,
    x: u64,
    cols: usize,
    tokens: usize,
    rows: [(u64, u64, usize, usize); 4],
) -> Result<bool> {
    next_prefill_packed(b, x, None, cols, tokens, rows)
}
/// next_prefill with the dense operand already packed by its producer (HC norm).
fn next_prefill_packed(
    b: &Backend<'_>,
    x: u64,
    prepacked: Option<u64>,
    cols: usize,
    tokens: usize,
    mut rows: [(u64, u64, usize, usize); 4],
) -> Result<bool> {
    if tokens < 64 {
        return Ok(false);
    }
    if !cols.is_multiple_of(16)
        || !x.is_multiple_of(8)
        || rows.iter().any(|r| !r.0.is_multiple_of(16))
    {
        return Ok(false);
    }
    // A prompt tile wider than the packed-input scratch runs in row groups.
    if let Some(step) = crate::flash_next_rows::pack_groups(cols, tokens) {
        let mut first = 0;
        while first < tokens {
            let n = step.min(tokens - first);
            let part = rows.map(|r| if r.2 == 0 { r } else { (r.0, r.1 + (first * r.3 * 4) as u64, r.2, r.3) });
            if !next_prefill_packed(b, x + (first * cols * 4) as u64, None, cols, n, part)? {
                return Err("prompt row group left the dense path".into());
            }
            first += n;
        }
        return Ok(true);
    }
    // Pack once for every large projection sharing this input, on this stream.
    // Narrow/irregular matrices retain fused staging and its edge handling.
    if cols.is_multiple_of(64) && rows.iter().any(|r| r.2 >= 64 && r.2.is_multiple_of(64)) {
        let packed = match prepacked {
            Some(packed) => packed,
            None => b.pack_flash_dense_input(x, cols, tokens)?,
        };
        let big = |r: &(u64, u64, usize, usize)| r.2 >= 64 && r.2.is_multiple_of(64);
        // A matrix under 64 rows rides in a tiled launch with its own K16 graph instead of
        // a pairs launch that rereads the FP32 input: in the unused rows of a last N tile
        // when it fits (HC injection beside HC down), else in extra N tiles of the last
        // launch still free (GDN a after z, b after qkv; MoE shared gate after shared up).
        let mut second = [None; 4];
        for j in 0..rows.len() {
            let n = rows[j].2;
            if n == 0 || n >= 64 {
                continue;
            }
            let free = |i: &usize| big(&rows[*i]) && second[*i].is_none();
            let spare = |i: &usize| rows[*i].2.next_multiple_of(128) - rows[*i].2 >= n.next_multiple_of(8);
            let host = (0..rows.len()).rev().filter(free).find(spare).or_else(|| (0..rows.len()).rev().find(free));
            if let Some(i) = host {
                second[i] = Some(j);
            }
        }
        for i in 0..rows.len() {
            let row = rows[i];
            if !big(&row) {
                continue;
            }
            let other = second[i].map_or((0, 0, 0, 0), |j| rows[j]);
            let mut p = row_params(packed, cols, tokens, row, other, true, false);
            unsafe {
                b.launch_flash_rows(
                    prompt_tiled_entry(tokens),
                    tokens.div_ceil(64) as u32,
                    (row.2 + other.2).div_ceil(128) as u32,
                    256,
                    spark_gpu_probe::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32,
                    &mut [(&mut p as *mut Bf16Rows).cast()],
                )?;
            }
        }
        for (i, j) in second.iter().enumerate() {
            if big(&rows[i]) {
                rows[i] = (0, 0, 0, 0);
            }
            if let Some(j) = *j {
                rows[j] = (0, 0, 0, 0);
            }
        }
    }
    if rows.iter().all(|r| r.2 == 0) {
        return Ok(true);
    }
    let pairs = [
        row_params(x, cols, tokens, rows[0], rows[1], false, false),
        row_params(x, cols, tokens, rows[2], rows[3], false, false),
    ];
    let blocks: usize = pairs
        .iter()
        .map(|&p| spark_gpu_probe::kernel_params::flash_bf16_prefill_blocks(p))
        .sum();
    let mut p = Bf16Pairs { pairs };
    unsafe {
        b.launch_flash_rows(
            "flash_bf16_mma_prefill_pairs",
            blocks as u32,
            1,
            256,
            FLASH_BF16_MMA_SHARED as u32,
            &mut [(&mut p as *mut Bf16Pairs).cast()],
        )?;
    }
    Ok(true)
}

/// Two weight matrices over the same tile in one launch: (weight, output, rows, stride).
pub(crate) fn bf16_row_pair(b: &Backend<'_>, x: u64, cols: usize, tokens: usize,
    first: (u64, u64, usize, usize), second: (u64, u64, usize, usize)) -> Result<()> {
    project_rows(b, x, cols, tokens, first, second, false, false, BF16_ROWS_TOKENS)
}
/// QSA alone reuses the same packed prompt input across both projection pairs.
pub(super) fn qsa_pair(b: &Backend<'_>, x: u64, tokens: usize,
    first: (u64, u64, usize), second: (u64, u64, usize), packed: bool) -> Result<()> {
    project_rows(b, x, H, tokens, (first.0, first.1, first.2, first.2),
        (second.0, second.1, second.2, second.2), packed, false, BF16_ROWS_TOKENS)
}
/// Reuse the preceding pair's packed prompt operands on the same ordered stream.
pub(crate) fn bf16_row_pair_reuse(b: &Backend<'_>, x: u64, cols: usize, tokens: usize,
    first: (u64, u64, usize, usize), second: (u64, u64, usize, usize)) -> Result<()> {
    if tokens == 0 || first.2 + second.2 == 0 { return Ok(()); }
    let packed = tokens > BF16_ROWS_TOKENS;
    let x = if packed { b.flash_dense_input()? } else { x };
    project_rows(b, x, cols, tokens, first, second, packed, false, BF16_ROWS_TOKENS)
}
fn row_params(x: u64, cols: usize, tokens: usize, first: (u64, u64, usize, usize),
    second: (u64, u64, usize, usize), input_bf16: bool, hc_silu: bool) -> Bf16Rows {
    Bf16Rows { weight: first.0, output: first.1, rows: first.2 as u32, stride: first.3 as u32,
        weight2: second.0, output2: second.1, rows2: second.2 as u32, stride2: second.3 as u32,
        input: x, cols: cols as u32, tokens: tokens as u32, input_bf16: u32::from(input_bf16), hc_silu: u32::from(hc_silu),
        input_plane_words: (tokens.div_ceil(16) * cols.div_ceil(16) * 128) as u64 }
}
#[cfg(test)]
#[test]
fn packed_pair_keeps_plane_and_output_strides() {
    let workspace = HcScratch::workspace_tokens();
    assert_eq!((0..=2 * workspace).filter(|&n| hc_packed(n)).collect::<Vec<_>>(), (32..=workspace).step_by(16).collect::<Vec<_>>());
    let p = row_params(16, H, 129, (32, 64, 24, 24), (48, 80, 24, 24), true, false);
    assert_eq!((p.input, p.input_bf16, p.input_plane_words, p.stride, p.stride2), (16, 1, 184320, 24, 24));
}
/// The MoE router + shared-expert gate and shared gate + up rows of one H-wide
/// tile, as two (weight, output, rows) pairs. Every decode, batch and
/// verification width (1..=DECODE_ROWS) runs one row kernel per
/// BF16_ROWS_TOKENS-token group (the MMA one serves both pairs in one launch),
/// so each output's sum depends only on its own row and column, never on how
/// many rows share the launch. Prompt tiles keep the tile kernels.
pub(crate) fn invariant_pairs(b: &Backend<'_>, x: u64, tokens: usize, rows: [(u64, u64, usize); 4]) -> Result<()> {
    let _route = b.prefill_class(3);
    let d = rows.map(|r| (r.0, r.1, r.2, r.2));
    if tokens > DECODE_ROWS {
        let reuse = hc_packed(tokens);
        project_rows(b, if reuse { b.flash_dense_input()? } else { x }, H, tokens, d[0], (0, 0, 0, 0), reuse, false, DECODE_ROWS)?;
        let packed = b.flash_dense_input()?;
        project_rows(b, packed, H, tokens, d[1], (0, 0, 0, 0), true, false, DECODE_ROWS)?;
        let _shared = b.prefill_class(6);
        return project_rows(b, packed, H, tokens, d[2], d[3], true, false, DECODE_ROWS);
    }
    if x % 8 != 0 || rows.iter().any(|r| r.0 % 16 != 0) {
        return Err("Flash BF16 decode pairs need aligned operands".into());
    }
    let blocks = (rows[0].2.div_ceil(8) + rows[1].2.div_ceil(8))
        .max(rows[2].2.div_ceil(8) + rows[3].2.div_ceil(8));
    for start in (0..tokens).step_by(BF16_ROWS_TOKENS) {
        b.decode_weight_read(|| rows.iter().map(|r| r.2).sum::<usize>() * H * 2);
        let at = |(w, y, rows, stride): (u64, u64, usize, usize)| (w, y + (start * stride * 4) as u64, rows, stride);
        let (input, n) = (x + (start * H * 4) as u64, BF16_ROWS_TOKENS.min(tokens - start));
        let mut p = Bf16Pairs { pairs: [row_params(input, H, n, at(d[0]), at(d[1]), false, false),
            row_params(input, H, n, at(d[2]), at(d[3]), false, false)] };
        b.profile_phase(16, || unsafe { b.launch_flash_rows("flash_bf16_mma1_2560_pairs", blocks as u32, 2, 128, 0,
            &mut [(&mut p as *mut Bf16Pairs).cast()]) })?;
    }
    Ok(())
}
/// One MoE shared-expert projection with output y[token * stride + row]; every
/// decode, batch and verification width shares one row kernel, as above.
pub(crate) fn invariant_rows(b: &Backend<'_>, weight: u64, x: u64, y: u64, rows: usize, cols: usize,
    tokens: usize, stride: usize) -> Result<()> {
    project_rows(b, x, cols, tokens, (weight, y, rows, stride), (0, 0, 0, 0), false, false, DECODE_ROWS)
}
/// Launches of up to `decode_width` tokens use the row (decode) kernels, in groups
/// of the BF16_ROWS_TOKENS tokens an MMA decode kernel holds; wider launches
/// use the prompt tile kernels.
fn project_rows(b: &Backend<'_>, x: u64, cols: usize, tokens: usize, first: (u64, u64, usize, usize),
    second: (u64, u64, usize, usize), input_bf16: bool, hc_silu: bool, decode_width: usize) -> Result<()> {
    if tokens == 0 || first.2 + second.2 == 0 { return Ok(()); }
    // Decode rows retain the same per-row K order at every batch width.
    let group = if tokens <= decode_width {
        BF16_ROWS_TOKENS
    } else if input_bf16 {
        tokens
    } else {
        crate::flash_next_rows::pack_groups(cols, tokens).unwrap_or(tokens)
    };
    if tokens > group {
        let width = if input_bf16 { 2 } else { 4 };
        for start in (0..tokens).step_by(group) {
            let slice = |(w, y, rows, stride): (u64, u64, usize, usize)|
                (w, y + (start * stride * 4) as u64, rows, stride);
            project_rows(b, x + (start * cols * width) as u64, cols, (tokens - start).min(group),
                slice(first), slice(second), input_bf16, hc_silu, decode_width)?;
        }
        return Ok(());
    }
    let mut params = row_params(x, cols, tokens, first, second, input_bf16, hc_silu);
    // Four ranks: a K-split shard computes only its two canonical K ranges (generic entries only).
    let ranges = crate::flash_split_ranges::launch(&mut params.cols, first.0, (second.2 != 0).then_some(second.0))?;
    b.decode_weight_read(|| (first.2 + second.2) * cols * 2);
    let rows = first.2 + second.2;
    let mut threads = 128;
    if cols < 64 || cols % 8 != 0 || x % 8 != 0 || first.0 % 16 != 0 || second.0 % 16 != 0 {
        return Err("Flash BF16 MMA needs K >= 64, a multiple of 8, and aligned operands".into());
    }
    if tokens > decode_width && !input_bf16 { params.input = b.pack_flash_dense_input(x, cols, tokens)?; }
    if tokens >= 128 && cols % 64 == 0 && first.2 % 64 == 0 && second.2 % 64 == 0 {
        return profile_projection(b, rows, cols, || {
            let full = tokens / 128 * 128;
            params.tokens = full as u32;
            unsafe { b.launch_flash_rows("flash_bf16_mma128", (full / 128) as u32, (rows / 64) as u32,
                256, FLASH_BF16_WIDE_SHARED as u32, &mut [(&mut params as *mut Bf16Rows).cast()])?; }
            let tail = tokens - full;
            if tail == 0 { return Ok(()); }
            params.tokens = tail as u32;
            // Keep the original plane stride: the low tail is equally far
            // from this sliced high-plane base as it was before slicing.
            params.input += (full * cols * 2) as u64;
            params.output += (full * first.3 * 4) as u64;
            params.output2 += (full * second.3 * 4) as u64;
            let (name, n, m, threads) = if tail >= 32 {
                ("flash_bf16_mma64", BF16_MMA_ROWS, BF16_MMA_TOKENS, 256)
            } else { ("flash_bf16_mma", BF16_TILE_ROWS, BF16_TILE_TOKENS, 128) };
            unsafe { b.launch_flash_rows(name, (rows.div_ceil(n) * tail.div_ceil(m)) as u32, 1,
                threads, if name == "flash_bf16_mma64" { FLASH_BF16_MMA_SHARED as u32 } else { 0 },
                &mut [(&mut params as *mut Bf16Rows).cast()]) }
        });
    }
    let (name, blocks) = if tokens <= decode_width {
        if tokens == 1 && cols == 2560 && rows < 64 && !input_bf16 && ranges == 0 {
            return b.profile_phase(16, || unsafe { b.launch_flash_rows("flash_bf16_mma1_2560_small", rows as u32, 1, 128, 0,
                &mut [(&mut params as *mut Bf16Rows).cast()]) });
        }
        let decode = if ranges != 0 { "flash_bf16_mma1" } else { match cols {
            320 if rows >= 1024 && first.2 % 64 == 0 && second.2 % 64 == 0 => {
                if input_bf16 { "flash_bf16_mma1_320_bf16" } else { "flash_bf16_mma1_320" }
            }
            2560 if rows < 1024 => "flash_bf16_mma1_2560_narrow",
            2560 if !input_bf16 && rows < 16384 && first.2 % 64 == 0 && second.2 % 64 == 0 => "flash_bf16_mma1_2560_native",
            2560 => "flash_bf16_mma1_2560",
            10240 if tokens == 1 && rows != 320 && rows != 324 => "flash_bf16_mma1",
            10240 => "flash_bf16_mma1_10240",
            3072 if !input_bf16 && rows >= 1024 && rows < 16384
                && first.2 % 64 == 0 && second.2 % 64 == 0 => "flash_bf16_mma1_3072_native",
            _ => "flash_bf16_mma1" } };
        let groups = first.2.div_ceil(8) + second.2.div_ceil(8);
        if !spark_gpu_probe::kernel_params::flash_bf16_decode_split(rows, cols) { threads = 64; (decode, groups.div_ceil(2)) }
        else { (decode, if ranges != 0 { groups.div_ceil(2) } else { groups }) } // two-range shards: two tiles a block
    }
    else if tokens >= 32 {
        threads = 256;
        ("flash_bf16_mma64", rows.div_ceil(BF16_MMA_ROWS) * tokens.div_ceil(BF16_MMA_TOKENS))
    }
    else { ("flash_bf16_mma", rows.div_ceil(BF16_TILE_ROWS) * tokens.div_ceil(BF16_TILE_TOKENS)) };
    // K320 decode rows of one matrix read its registered LBF12 twin (hc_silu bit 1).
    if let Some(twin) = lbf12_of(first.0, None).filter(|_| name.starts_with("flash_bf16_mma1_320") && second.2 == 0) { params.weight2 = twin; params.hc_silu |= 2; }
    // Native K2560/K3072 decode rows of a matrix with a registered LBF12 twin read the twin (hc_silu
    // bit 1) through the split, four warps per 8-row group; a pair with a twin runs as two launches.
    let twin = |m: (u64, u64, usize, usize)| lbf12_of(m.0, Some((m.2, cols))).filter(|_| name.ends_with("_native") && m.2 != 0);
    let mut launches = [(params, rows, blocks, threads), (params, 0, 0, 0)];
    if twin(first).is_some() || twin(second).is_some() { for (at, m) in launches.iter_mut().zip([first, second]) {
        let mut p = row_params(x, cols, tokens, m, (0, 0, 0, 0), input_bf16, hc_silu);
        *at = match twin(m) { Some(t) => { (p.weight2, p.hc_silu) = (t, p.hc_silu | 2); (p, m.2, m.2 / 8, 128) } None => (p, m.2, (m.2 / 8).div_ceil(2), 64) };
    } }
    for (mut params, rows, blocks, threads) in launches.into_iter().filter(|l| l.1 != 0) {
        profile_projection(b, rows, cols, || unsafe {
            b.launch_flash_rows(name, blocks as u32, 1, threads,
                if name == "flash_bf16_mma64" { FLASH_BF16_MMA_SHARED as u32 } else { 0 },
                &mut [(&mut params as *mut Bf16Rows).cast()])
        })?;
    }
    Ok(())
}
fn profile_projection(b: &Backend<'_>, rows: usize, cols: usize, run: impl FnOnce() -> Result<()>) -> Result<()> {
    if b.is_profiling() && rows < 16384 {
        let kind = if cols == W {14} else if cols == LOW {15} else if rows <= 1024 {16} else {17};
        b.profile_phase(kind, run)
    } else { run() }
}
fn norm(b: &Backend<'_>, x: u64, weight: u64, y: u64, tokens: usize, packed: u64, f32_weight: bool) -> Result<()> {
    let _class = b.prefill_class(7);
    b.prefill_read_bytes(7, || W * 2);
    // Prompt CTAs cache one stream each; small tiles retain the decode grid.
    let blocks = if tokens > DECODE_ROWS { 4 } else if x == y { 1 } else { 8 };
    let mut p = FlashHcNorm { input: x, weight, output: y, packed };
    let name = if f32_weight { "flash_hc_norm_f32" } else { "flash_hc_norm" };
    unsafe { b.launch_flash_rows(name, blocks, tokens as u32, 128, 0,
        &mut [(&mut p as *mut FlashHcNorm).cast()]) }
}
#[repr(C)]
struct AddRows { output: u64, input: u64, width: u32, repeat: u32 }
/// Add each input row once, or broadcast it across the residual streams.
pub(super) fn add_rows(b: &Backend<'_>, output: u64, input: u64, width: usize,
    tokens: usize, repeat: usize) -> Result<()> {
    let mut p = AddRows { output, input, width: width as u32, repeat: repeat as u32 };
    unsafe { b.launch_flash_rows("flash_add", width.div_ceil(128) as u32, tokens as u32, 128, 0,
        &mut [(&mut p.output as *mut u64).cast(), (&mut p.input as *mut u64).cast(),
            (&mut p.width as *mut u32).cast(), (&mut p.repeat as *mut u32).cast()]) }
}
fn weight(w: &Weights<'_>, name: &str, shape: &[u64]) -> Result<u64> {
    let t = w.tensor(name)?;
    if t.tensor.dtype != Dtype::Bf16 || t.tensor.shape != shape || !matches!(t.selection, Selection::All) {
        return Err(format!("{name}: expected replicated BF16 {shape:?}"));
    }
    w.device_address(name)
}
/// HC/PLE norm weights keep their stored dtype in mixed checkpoints (BF16 or
/// F32); the boolean selects the matching norm kernel, never a conversion.
fn norm_weight(w: &Weights<'_>, name: &str) -> Result<(u64, bool)> {
    let t = w.tensor(name)?;
    let f32_weight = match t.tensor.dtype { Dtype::Bf16 => false, Dtype::F32 => true,
        _ => return Err(format!("{name}: expected BF16/F32 norm weight")) };
    if t.tensor.shape != [W as u64] || !matches!(t.selection, Selection::All) || t.bytes != W * if f32_weight { 4 } else { 2 } {
        return Err(format!("{name}: expected replicated norm [{W}]")); }
    Ok((w.device_address(name)?, f32_weight)) }
fn dense(vector: &DeviceVector<'_>, len: usize) -> Result<u64> {
    if vector.len() != len || vector.is_segmented() {
        return Err(format!("Flash activation must be contiguous with {len} FP32 values"));
    }
    Ok(vector.device_address())
}
fn tile(vector: &DeviceVector<'_>, len: usize) -> Result<u64> {
    if vector.len() < len || vector.is_segmented() {
        return Err(format!("Flash tile must be contiguous with at least {len} FP32 values"));
    }
    Ok(vector.device_address())
}

/// One workspace is sufficient for every attention/MLP HC and the final mixer.
/// The original residual is not overwritten until the block has completed.
/// Normalized streams and gates die after mixing. Only the four injection
/// weights per token must survive the following attention or expert block.
pub struct HcScratch<'a> {
    normalized: DeviceVector<'a>,
    low: DeviceVector<'a>,
    gates: Option<DeviceVector<'a>>,
    inject: DeviceVector<'a>,
    mixed: DeviceVector<'a>,
    tokens: usize,
}
impl<'a> HcScratch<'a> {
    pub const DEVICE_BYTES: usize = (2 * W + LOW + 4 + H) * 4;
    /// Row-major BF16 HC-up scratch for a rank chunk (no one-board allocation).
    pub const RANK_UP_WORDS: usize = W * LOW / 2;
    pub(super) const WORKSPACE_TOKENS: usize = 4096;
    /// Workspace rows: 4096, or the prompt tile when it is wider (VADE_PROMPT_TILE=8192), so an
    /// 8192-row tile keeps the fused HC prompt paths (add_norm, planes, tiled HC up).
    pub(super) fn workspace_tokens() -> usize { crate::flash_next_rows::prompt_tile().max(Self::WORKSPACE_TOKENS) }
    pub(super) const WORKSPACE_TOKEN_BYTES: usize = (2 * W + LOW) * 4;
    pub fn new(b: &Backend<'a>) -> Result<Self> { Self::with_tokens(b, 1) }
    pub fn with_tokens(b: &Backend<'a>, tokens: usize) -> Result<Self> {
        let workspace = tokens.min(Self::workspace_tokens());
        let mut vectors = b.vector_arena(&[W * workspace, LOW * workspace, W * workspace, 4 * tokens, H])?.into_iter();
        Ok(Self {
            normalized: vectors.next().unwrap(), low: vectors.next().unwrap(),
            gates: Some(vectors.next().unwrap()), inject: vectors.next().unwrap(), mixed: vectors.next().unwrap(), tokens,
        })
    }
    /// Next rows fuse HC-up gates into the mixed output; no gates vector is used.
    /// Keep normalized as the owner shared with the subsequent GDN projections.
    pub(super) fn next_rows(b: &Backend<'a>, tokens: usize) -> Result<Self> {
        let mut vectors = b.vector_arena(&crate::flash_next_workspace::hc_lengths(W, LOW, H, tokens))?.into_iter();
        Ok(Self { normalized: vectors.next().unwrap(), low: vectors.next().unwrap(), gates: None,
            inject: vectors.next().unwrap(), mixed: vectors.next().unwrap(), tokens })
    }
    pub(super) fn normalized_workspace(&self) -> &DeviceVector<'a> { &self.normalized }
    /// Swap in other rows' injection words (a bank tile, flash/next_split_bank.rs); returns the
    /// words it held. The caller swaps them back before the scratch serves another pass.
    pub(super) fn swap_inject(&mut self, inject: DeviceVector<'a>) -> DeviceVector<'a> { std::mem::replace(&mut self.inject, inject) }
    pub fn device_bytes(&self) -> usize {
        if self.gates.is_none() {
            crate::flash_next_workspace::bytes(&crate::flash_next_workspace::hc_lengths(W, LOW, H, self.tokens))
        } else {
            self.tokens.min(Self::workspace_tokens()) * Self::WORKSPACE_TOKEN_BYTES + (4 * self.tokens + H) * 4
        }
    }
    pub fn mixed(&self) -> &DeviceVector<'a> { &self.mixed }
}

/// Lightweight views into the single resident rank allocation.
pub struct Hc<'w> {
    norm: u64,
    norm_f32: bool,
    next: bool,
    down: u64,
    up: u64,
    inject: Option<u64>,
    _weights: PhantomData<&'w ()>,
}
impl<'w> Hc<'w> {
    pub fn load(w: &'w Weights<'_>, prefix: &str) -> Result<Self> { Self::read(w, prefix, true, false) }
    pub fn mixer(w: &'w Weights<'_>, prefix: &str) -> Result<Self> { Self::read(w, prefix, false, false) }
    /// Mixed-checkpoint variant: the norm weight may be stored F32.
    pub fn load_next(w: &'w Weights<'_>, prefix: &str) -> Result<Self> { Self::read(w, prefix, true, true) }
    pub fn mixer_next(w: &'w Weights<'_>, prefix: &str) -> Result<Self> { Self::read(w, prefix, false, true) }
    pub(super) fn load_rows(w: &'w Weights<'_>, prefix: &str) -> Result<Self> {
        let mut hc = Self::load_next(w, prefix)?; hc.next = true; Ok(hc)
    }
    pub(super) fn mixer_rows(w: &'w Weights<'_>, prefix: &str) -> Result<Self> {
        let mut hc = Self::mixer_next(w, prefix)?; hc.next = true; Ok(hc)
    }
    fn read(w: &'w Weights<'_>, p: &str, inject: bool, next: bool) -> Result<Self> {
        let (norm, norm_f32) = if next { norm_weight(w, &format!("{p}.hc_norm.weight"))? }
            else { (weight(w, &format!("{p}.hc_norm.weight"), &[W as u64])?, false) };
        Ok(Self { norm, norm_f32, next: false,
            down: weight(w, &format!("{p}.input_mix_weight_down.weight"), &[LOW as u64, W as u64])?,
            up: weight(w, &format!("{p}.input_mix_weight_up.weight"), &[W as u64, LOW as u64])?,
            inject: if inject { Some(weight(w, &format!("{p}.block_inject_weight.weight"), &[4, W as u64])?) } else { None },
            _weights: PhantomData,
        })
    }
    /// Normalize four streams independently, then form their learned gated mean.
    /// TP/EP compact schedule; all operands remain session-owned.
    pub fn mix_native<'a>(&self,b:&Backend<'_>,s:&mut HcScratch<'a>,state:&mut DeviceVector<'a>,previous:&mut DeviceVector<'a>,residual:&mut DeviceVector<'a>)->Result<()> {
        let mut p=spark_gpu_probe::kernel_params::NextHc {input:residual.device_address(),previous:s.inject.device_address(),weight:self.norm,down:self.down,injection:self.inject.unwrap_or(0),up:self.up,state:state.device_address(),normalized:s.normalized.device_address(),gates:previous.device_address(),low:s.low.device_address(),output:s.mixed.device_address(),tokens:1,packed:1,norm_f32:u32::from(self.norm_f32),..Default::default()};
        for(name,grid)in [("flash_next_hc_pre",41),("flash_next_hc_up",320)] {unsafe{b.launch_flash_shared(name,grid,256,0,&mut[(&mut p as *mut spark_gpu_probe::kernel_params::NextHc).cast()])?;}}
        std::mem::swap(residual,state);std::mem::swap(&mut s.inject,previous);Ok(())
    }
    /// mix_native over `tokens` independent residual rows (grid Y = row): the same two
    /// launches and per-row arithmetic; mixed rows go to `output` (tokens * H).
    pub fn mix_native_rows<'a>(&self,b:&Backend<'_>,s:&mut HcScratch<'a>,state:&mut DeviceVector<'a>,previous:&mut DeviceVector<'a>,residual:&mut DeviceVector<'a>,output:u64,tokens:usize)->Result<()> {
        if tokens==0 || tokens>s.tokens || previous.len()<4*tokens || s.inject.len()<4*tokens { return Err("HC rows exceed their workspace".into()); }
        let mut p=spark_gpu_probe::kernel_params::NextHc {input:tile(residual,tokens*W)?,previous:s.inject.device_address(),weight:self.norm,down:self.down,injection:self.inject.unwrap_or(0),up:self.up,state:tile(state,tokens*W)?,normalized:s.normalized.device_address(),gates:previous.device_address(),low:s.low.device_address(),output,tokens:1,packed:1,norm_f32:u32::from(self.norm_f32),..Default::default()};
        for(name,grid)in [("flash_next_hc_pre",41),("flash_next_hc_up",320)] {unsafe{b.launch_flash_rows(name,grid,tokens as u32,256,0,&mut[(&mut p as *mut spark_gpu_probe::kernel_params::NextHc).cast()])?;}}
        std::mem::swap(residual,state);std::mem::swap(&mut s.inject,previous);Ok(())
    }
    /// Rank chunks: normalize once to the exclusive BF16 row, then the original down/inject
    /// dots and up. Only tp::next_chunk calls this; one-board and one-token paths stay native.
    pub fn mix_native_rank_rows<'a>(&self,b:&Backend<'_>,s:&mut HcScratch<'a>,state:&mut DeviceVector<'a>,previous:&mut DeviceVector<'a>,residual:&mut DeviceVector<'a>,output:u64,norm_bf16:u64,up_rowmajor:u64,tokens:usize)->Result<()> {
        if tokens==0 || tokens>s.tokens || previous.len()<4*tokens || s.inject.len()<4*tokens || norm_bf16==0 || up_rowmajor==0 { return Err("HC rank rows exceed their workspace".into()); }
        let mut p=spark_gpu_probe::kernel_params::NextHc {input:tile(residual,tokens*W)?,block:norm_bf16,previous:s.inject.device_address(),weight:self.norm,down:self.down,injection:self.inject.unwrap_or(0),up:self.up,state:tile(state,tokens*W)?,normalized:s.normalized.device_address(),gates:previous.device_address(),low:s.low.device_address(),output,tokens:1,packed:1,norm_f32:u32::from(self.norm_f32),..Default::default()};
        // Storage choice only: each row keeps exactly the same FMA/reduction plan.
        let rowmajor=tokens>=64;
        if rowmajor {p.previous=up_rowmajor;unsafe{b.launch_flash_shared("flash_next_hc_unpack_up",(W*LOW).div_ceil(256) as u32,256,0,&mut[(&mut p as *mut spark_gpu_probe::kernel_params::NextHc).cast()])?;}}
        for(name,grid)in [("flash_next_hc_norm_once",1),("flash_next_hc_pre_normed",41),(if rowmajor {"flash_next_hc_up_rowmajor"}else{"flash_next_hc_up_packed"},320)] {unsafe{b.launch_flash_rows(name,grid,tokens as u32,256,0,&mut[(&mut p as *mut spark_gpu_probe::kernel_params::NextHc).cast()])?;}}
        std::mem::swap(residual,state);std::mem::swap(&mut s.inject,previous);Ok(())
    }
    /// combine for `tokens` rows after mix_native_rows (flash_next_hc_add, row = grid Y).
    pub fn combine_rows(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &mut DeviceVector<'_>, block: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.combine_tiled(b, scratch, residual, block, tokens)
    }
    pub fn mix(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &DeviceVector<'_>) -> Result<()> {
        let mixed = scratch.mixed.device_address();
        self.mix_tile(b, scratch, dense(residual, W)?, mixed, 1)
    }
    /// Mix `tokens` consecutive residual rows into consecutive H-wide rows at `output`.
    pub(super) fn mix_tiled(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &DeviceVector<'_>,
        output: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.mix_tile(b, scratch, tile(residual, tokens * W)?, tile(output, tokens * H)?, tokens)
    }
    fn mix_tile(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, r: u64, output: u64, tokens: usize) -> Result<()> {
        let _class = b.prefill_class(7);
        if tokens == 0 || tokens > scratch.tokens { return Err("HC tile exceeds its workspace".into()); }
        for start in (0..tokens).step_by(HcScratch::workspace_tokens()) {
            self.mix_chunk(b, scratch, r + (start * W * 4) as u64, output + (start * H * 4) as u64,
                scratch.inject.device_address() + (start * 4 * 4) as u64,
                (tokens - start).min(HcScratch::workspace_tokens()), None, 0)?;
        }
        Ok(())
    }
    /// `prev.combine_tiled` then `self.mix_tiled` on one residual tile: prompt tiles run the
    /// combine, this mix's F32 norm and the projections' packed operand as one pass
    /// (flash_next_hc_add_norm; same per-element math and square order as the pair).
    /// `finish`: the MoE k16_finish that would have written `block`; it then runs inside the
    /// fused kernel (or first, on the unfused path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_mix_tiled(&self, prev: &Hc<'w>, b: &Backend<'_>, scratch: &mut HcScratch<'_>,
        residual: &mut DeviceVector<'_>, block: &DeviceVector<'_>, output: &DeviceVector<'_>, tokens: usize,
        finish: Option<MoeFinishParams>) -> Result<()> {
        if !(self.next && prev.next && self.norm_f32 && prev.inject.is_some() && (tokens >= 64 || tokens <= DECODE_ROWS && finish.is_none())
            && tokens <= HcScratch::workspace_tokens() && tokens <= scratch.tokens) {
            // Decode rows: the combine's add forms the finish per column (flash_next_hc_finish_add).
            let fused = finish.filter(|_| prev.next && b.has_flash("flash_next_hc_finish_add"));
            if let (Some(mut f), None) = (finish, fused) {
                unsafe { b.launch_flash_rows("flash_next_k16_finish", H.div_ceil(256) as u32, tokens as u32, 256, 0,
                    &mut [(&mut f as *mut MoeFinishParams).cast()])?; }
            }
            prev.combine_tile(b, scratch, tile(residual, tokens * W)?, tile(block, tokens * H)?, 0, 0, tokens, fused)?;
            return self.mix_tiled(b, scratch, residual, output, tokens);
        }
        let _class = b.prefill_class(7);
        let (r, y, o) = (tile(residual, tokens * W)?, tile(block, tokens * H)?, tile(output, tokens * H)?);
        self.mix_chunk(b, scratch, r, o, scratch.inject.device_address(), tokens, Some((y, finish)), 0)
    }
    /// Only the prompt MLP caller offers its existing BF16 expert-input buffer.
    /// A false return means the normal path ran and the caller must keep its a16 prepass.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_mix_tiled_a16(&self, prev:&Hc<'w>, b:&Backend<'_>, scratch:&mut HcScratch<'_>,
        residual:&mut DeviceVector<'_>, block:&DeviceVector<'_>, output:&DeviceVector<'_>,
        a16:&DeviceVector<'_>, tokens:usize)->Result<bool>{
        if !(self.next && prev.next && self.norm_f32 && prev.inject.is_some()
            && tokens>=TILED_HC_UP_ROWS && tokens<=HcScratch::workspace_tokens() && tokens<=scratch.tokens){
            self.combine_mix_tiled(prev,b,scratch,residual,block,output,tokens,None)?;
            return Ok(false);
        }
        if !b.has_flash("flash_next_hc_up_tiled_a16") {return Err("VADE_PROMPT_COLD8K_R2 requires flash_next_hc_up_tiled_a16".into());}
        let _class=b.prefill_class(7);
        let (r,y,o)=(tile(residual,tokens*W)?,tile(block,tokens*H)?,tile(output,tokens*H)?);
        let dest=tile(a16,tokens*H/2)?;
        self.mix_chunk(b,scratch,r,o,scratch.inject.device_address(),tokens,Some((y,None)),dest)?;
        Ok(true)
    }
    /// `fused` = the preceding combine's block rows: that combine runs inside the norm
    /// (its injection gates are `injection`, read before this mix's projections replace them).
    #[allow(clippy::too_many_arguments)]
    fn mix_chunk(&self, b: &Backend<'_>, scratch: &HcScratch<'_>, r: u64, output: u64,
        injection: u64, tokens: usize, fused: Option<(u64, Option<MoeFinishParams>)>, output_a16:u64) -> Result<()> {
        if self.next {
            let (u, low) = (scratch.normalized.device_address(), scratch.low.device_address());
            // The unfused norm runs the solo grid, with independent rows in grid Y; no prompt-only reduction.
            // Prompt tiles and fused decode rows: the F32 norm also writes the projections' packed operand.
            let decode_add = fused.is_some() && tokens <= DECODE_ROWS;
            let prepacked = if self.norm_f32 && (tokens >= 64 || decode_add) {
                b.prepare_flash_dense()?;
                Some(b.flash_dense_input()?)
            } else {
                None
            };
            let rows = [(self.inject.unwrap_or(0), injection, if self.inject.is_some() { 4 } else { 0 }, 4), (self.down, low, LOW, LOW), (0, 0, 0, 0), (0, 0, 0, 0)];
            // Prompt tiles: next_prefill_packed's HC down launch (injection in the last N tile)
            // whose epilogue applies flash_next_hc_activate and writes the activated low as the
            // HC up operand's planes in low's own words, offered to its pack. A last partial
            // 16-row group is safe: these kernels store only rows below `tokens`, and each
            // output row reads only its own operand row.
            let hc_planes = tokens >= TILED_HC_UP_ROWS && tokens <= HcScratch::workspace_tokens()
                && rows.iter().all(|r| r.0.is_multiple_of(16));
            // Fused prompt tiles whose HC down is tiled and whose HC up is tiled: the norm writes
            // only the residual rows and its scales into u; HC down forms its planes and HC up
            // the streams from them.
            let on_read = fused.is_some() && prepacked.is_some() && hc_planes;
            if let (Some((block, finish)), Some(packed)) = (fused, prepacked) {
                let mut p = spark_gpu_probe::kernel_params::FlashHcAddNorm { residual: r, block, gates: injection,
                    weight: self.norm, output: u, packed: if on_read { 0 } else { packed }, scales: if on_read { u } else { 0 } };
                let mut rows = tokens as u32;
                unsafe {
                    match finish {
                        // One CTA per token pair (t, t + 8) of each 16-row group.
                        Some(mut f) => {
                            b.launch_flash_rows(if b.has_flash("flash_next_hc_finish_add_norm_fast") { "flash_next_hc_finish_add_norm_fast" }
                                else { "flash_next_hc_finish_add_norm" }, 1, tokens.div_ceil(16) as u32 * 8, 256,
                                8 * H as u32 * 4 + 32 + 2 * H as u32 * 2 + 2 * 16 * 4,
                                &mut [(&mut p as *mut spark_gpu_probe::kernel_params::FlashHcAddNorm).cast(),
                                (&mut f as *mut MoeFinishParams).cast(), (&mut rows as *mut u32).cast()])?
                        }
                        // Decode rows: one CTA per (stream, row), which offers HC down its planes.
                        None => {
                            let (name, x, y, threads, shared) = if decode_add { ("flash_next_hc_add_norm_rows", 4, rows, 128, 0) } else { ("flash_next_hc_add_norm", 1, tokens.div_ceil(16) as u32 * 8, 256, 8 * H as u32 * 4 + 32) };
                            b.launch_flash_rows(name, x, y, threads, shared, &mut [(&mut p as *mut spark_gpu_probe::kernel_params::FlashHcAddNorm).cast(),
                                (&mut rows as *mut u32).cast()])?
                        }
                    }
                }
                if decode_add { b.offer_dense_planes(u, W, tokens, packed); }
            } else {
                let planes = if self.norm_f32 && prepacked.is_none() && tokens <= DECODE_ROWS { b.prepare_flash_dense()?; Some(b.flash_dense_input()?) } else { None }; // decode rows: + HC down's packed planes (its pack launch is skipped)
                let (mut p, name) = (FlashHcNorm { input: r, weight: self.norm, output: u, packed: prepacked.or(planes).unwrap_or(0) }, if self.norm_f32 { b.chain_entry(tokens, "flash_hc_norm_f32", "flash_hc_norm_f32_chain") } else { "flash_next_hc_norm" });
                unsafe { b.launch_flash_rows(name, 8, tokens as u32, 128, 0, &mut [(&mut p as *mut FlashHcNorm).cast()])?; }
                if let Some(planes) = planes { b.offer_dense_planes(u, W, tokens, planes); }
            }
            if let (true, Some(packed)) = (hc_planes, prepacked) {
                // on_read: the same A words, formed from the residual rows, the scales and the
                // norm weights; otherwise the norm's packed planes (scales and weights 0).
                let mut p = row_params(if on_read { r } else { packed }, W, tokens, rows[1], rows[0], true, false);
                let (mut scales, mut weight) = if on_read { (u, self.norm) } else { (0u64, 0u64) };
                unsafe {
                    b.launch_flash_rows("flash_bf16_mma_prefill_tiled_hc", tokens.div_ceil(64) as u32,
                        (LOW + rows[0].2).div_ceil(128) as u32, 256,
                        spark_gpu_probe::kernel_params::FLASH_BF16_PREFILL_ROWS_SHARED as u32,
                        &mut [(&mut p as *mut Bf16Rows).cast(), (&mut scales as *mut u64).cast(), (&mut weight as *mut u64).cast()])?;
                }
                b.offer_dense_planes(low, LOW, tokens, low);
            } else {
                let done = prepacked.is_some() && next_prefill_packed(b, u, prepacked, W, tokens, rows)?;
                // Decode BF16 rows: HC down activates its rows and writes HC up's planes, past its own input planes.
                let act = !done && tokens <= DECODE_ROWS && self.norm_f32 && self.inject.is_some() && rows.iter().all(|r| r.2 == 0 || mx8_weight(r.0, r.2, W, tokens) & 1 == 0);
                if act {
                    let packed = b.pack_flash_latency_input(u, W, tokens)?;
                    let planes = b.flash_dense_input()? + (2 * tokens.div_ceil(16) * (W / 16) * 128 * 4) as u64;
                    b.flash_hc_down_group(&mut Bf16Pairs { pairs: [row_params(packed, W, tokens, rows[0], rows[1], true, false), row_params(packed, W, tokens, rows[2], rows[3], true, false)] }, planes)?;
                    b.offer_dense_planes(low, LOW, tokens, planes);
                } else if done {
                } else if let Some(weight) = self.inject {
                    next_rows_group(b, u, W, tokens, [(weight, injection, 4), (self.down, low, LOW), (0, 0, 0), (0, 0, 0)])?;
                } else {
                    next_rows(b, self.down, u, low, LOW, W, tokens)?;
                }
                let planes = if tokens <= DECODE_ROWS { b.prepare_flash_dense()?; b.flash_dense_input()? } else { 0 }; // decode rows: + HC up's packed planes (its pack launch is skipped)
                if !act { launch(b, b.chain_entry(tokens, "flash_next_hc_activate", "flash_next_hc_activate_chain"), (LOW * tokens).div_ceil(128), &mut [Ptr(low),U32((LOW*tokens) as u32),Ptr(planes)])?; if planes != 0 { b.offer_dense_planes(low, LOW, tokens, planes); } }
            }
            let packed = b.pack_flash_dense_input(low, LOW, tokens)?;
            let mut p = row_params(packed, LOW, tokens, (self.up, output, W, H), (0,0,0,0), true, false);
            let mut normalized = u;
            let (mut residual, mut weight) = (if on_read { r } else { 0 }, self.norm);
            if tokens >= TILED_HC_UP_ROWS && tokens <= HcScratch::workspace_tokens() {
                // Prompt tiles: 64 tokens x 32 features of all four streams per CTA, the same
                // K16 gates and stream-ordered mix as latency4; the epilogue also writes the
                // next projections' dense planes, offered to their pack of `output`.
                let mut planes = b.flash_mixed_planes()?;
                p.output2=output_a16;
                unsafe {
                    b.launch_flash_rows(
                        if output_a16!=0 {"flash_next_hc_up_tiled_a16"} else {"flash_next_hc_up_tiled"},
                        tokens.div_ceil(64) as u32,
                        (H / 32 / spark_gpu_probe::kernel_params::FLASH_HC_UP_SLICES) as u32,
                        256,
                        spark_gpu_probe::kernel_params::FLASH_BF16_PREFILL_TILED_SHARED as u32,
                        &mut [
                            (&mut p as *mut Bf16Rows).cast(),
                            (&mut normalized as *mut u64).cast(),
                            (&mut planes as *mut u64).cast(),
                            (&mut residual as *mut u64).cast(),
                            (&mut weight as *mut u64).cast(),
                        ],
                    )?;
                }
                b.offer_dense_planes(output, H, tokens, planes);
                return Ok(());
            }
            // M <= 8 FP32 consumers never read the dense tile, so the second tile can serve their pack.
            let mut planes = if tokens <= BF16_ROWS_TOKENS && lbf12_of(self.up, None).is_some() { let t = b.flash_mixed_planes()?; b.offer_dense_planes(output, H, tokens, t); t } else { 0 };
            return unsafe {
                // One warp per stream (gates mixed in stream order by warp 0); at M1-16 the LBF12 twin of the up
                // weight, when built, replaces its BF16 bytes (M2-16: hc_up_gate's chains on the twin's fragments).
                match lbf12_of(self.up, None).filter(|_| tokens <= 16) {
                    Some(mut twin) => b.launch_flash_rows(if tokens == 1 { b.chain_entry(tokens, "flash_next_hc_up_latency4_lbf12", "flash_next_hc_up_latency4_lbf12_chain") } else { b.chain_entry(tokens, "flash_next_hc_up_latency16_lbf12", "flash_next_hc_up_latency16_lbf12_chain") }, H.div_ceil(8) as u32, 1, 128, 0,
                        &mut [(&mut p as *mut Bf16Rows).cast(), (&mut normalized as *mut u64).cast(), (&mut twin as *mut u64).cast(), (&mut planes as *mut u64).cast()]),
                    None => b.launch_flash_rows("flash_next_hc_up_latency4", H.div_ceil(8) as u32, tokens.div_ceil(16) as u32, 128, 0, &mut [(&mut p as *mut Bf16Rows).cast(), (&mut normalized as *mut u64).cast()]),
                }
            };
        }
        let u = scratch.normalized.device_address();
        let low = scratch.low.device_address();
        let gates = scratch.gates.as_ref().ok_or("Next-only HC scratch cannot run materialized gates")?.device_address();
        // Full MMA tiles can consume BF16 operands emitted by normalization; partial tiles keep the
        // standalone packer's zero padding. The F32 norm kernel emits no packed operands (FP32 packs).
        let packed = !self.norm_f32 && (tokens == 1 || tokens % 16 == 0);
        let decode_packed = packed && tokens == 1;
        let operand = if packed { b.flash_dense_input()? } else { 0 };
        norm(b, r, self.norm, u, tokens, operand, self.norm_f32)?;
        let x = if packed { operand } else { u };
        // Injection gates depend only on the normalized streams: one launch projects them beside the down rows.
        let (first, second) = match self.inject {
            Some(inject) => ((inject, injection, 4, 4), (self.down, low, LOW, LOW)),
            None => ((self.down, low, LOW, LOW), (0, 0, 0, 0)),
        };
        if tokens >= 128 && self.inject.is_some() {
            // Keep the compact four-row weights off mma128's unguarded tile loader.
            project_rows(b, x, W, tokens, second, (0, 0, 0, 0), packed, false, BF16_ROWS_TOKENS)?;
            project_rows(b, x, W, tokens, first, (0, 0, 0, 0), packed, false, BF16_ROWS_TOKENS)?;
        }
        else if !packed && self.inject.is_none() { bf16_rows(b, self.down, x, low, LOW, W, tokens, LOW)?; }
        // BF16-norm (FP8) HC at 2..8 FP32 rows: one latency group launch of injection + down (same graph).
        else if !packed && !self.norm_f32 && tokens <= BF16_ROWS_TOKENS { if x % 8 != 0 || first.0 % 16 != 0 || second.0 % 16 != 0 { return Err("Flash BF16 MMA needs K >= 64, a multiple of 8, and aligned operands".into()); }
            b.decode_weight_read(|| (first.2 + second.2) * W * 2); next_rows_group_at(b, x, W, tokens, [first, second, (0, 0, 0, 0), (0, 0, 0, 0)])?; } else { project_rows(b, x, W, tokens, first, second, packed, decode_packed, BF16_ROWS_TOKENS)?; }
        let up_packed = tokens > BF16_ROWS_TOKENS;
        let x = if decode_packed {
            // Down's epilogue already activated and packed this private row.
            // Keeping it separate from normalized input avoids inter-CTA aliasing.
            low
        } else if up_packed {
            b.pack_flash_hc_input(low, tokens)?
        } else {
            launch(b, "flash_silu", (LOW * tokens).div_ceil(128),
                &mut [Ptr(low), Ptr(0), U32((LOW * tokens) as u32), F32(4.0), U32(0)])?;
            low
        };
        if decode_packed || up_packed { project_rows(b, x, LOW, tokens, (self.up, gates, W, W), (0, 0, 0, 0), true, false, BF16_ROWS_TOKENS)?; }
        else { bf16_rows(b, self.up, x, gates, W, LOW, tokens, W)?; }
        launch_tokens(b, "flash_hc_mix", H.div_ceil(128), tokens,
            &mut [Ptr(u), Ptr(gates), Ptr(output), U32(H as u32),
                Ptr(if hc_packed(tokens) { b.flash_dense_input()? } else { 0 })])
    }
    /// Call after reducing the attention/MLP output across ranks. `scratch`
    /// must still contain this HC's most recent mix, before any subsequent HC.
    pub fn combine(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &mut DeviceVector<'_>, block: &DeviceVector<'_>) -> Result<()> {
        self.combine_tile(b, scratch, dense(residual, W)?, dense(block, H)?, 0, 0, 1, None)
    }
    /// Inject `tokens` reduced H-wide rows into their residual rows, after `mix_tiled`.
    pub(super) fn combine_tiled(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &mut DeviceVector<'_>,
        block: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        self.combine_tile(b, scratch, tile(residual, tokens * W)?, tile(block, tokens * H)?, 0, 0, tokens, None)
    }
    /// combine_tiled after a MoE whose k16_finish was deferred (`finish`, pooled decode): the
    /// add forms it per column (flash_next_hc_finish_add) when the PTX has that entry; otherwise
    /// k16_finish runs first, as moe_at would have launched it.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn combine_finish_tiled(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &mut DeviceVector<'_>,
        block: &DeviceVector<'_>, tokens: usize, finish: Option<MoeFinishParams>) -> Result<()> {
        let fused = finish.filter(|_| self.next && b.has_flash("flash_next_hc_finish_add"));
        if let (Some(mut f), None) = (finish, fused) {
            unsafe { b.launch_flash_rows("flash_next_k16_finish", H.div_ceil(256) as u32, tokens as u32, 256, 0,
                &mut [(&mut f as *mut MoeFinishParams).cast()])?; }
        }
        self.combine_tile(b, scratch, tile(residual, tokens * W)?, tile(block, tokens * H)?, 0, 0, tokens, fused)
    }
    /// Inject output-owned columns directly, without a separate reconstruction kernel.
    pub(super) fn combine_columns(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, residual: &mut DeviceVector<'_>,
        block: &DeviceVector<'_>, peer: &DeviceVector<'_>, rank: usize, tokens: usize) -> Result<()> {
        if rank >= 2 { return Err("HC column packets require two ranks".into()); }
        self.combine_tile(b, scratch, tile(residual, tokens * W)?, tile(block, tokens * H / 2)?,
            tile(peer, tokens * H / 2)?, rank as u32, tokens, None)
    }
    #[allow(clippy::too_many_arguments)]
    fn combine_tile(&self, b: &Backend<'_>, scratch: &mut HcScratch<'_>, r: u64, y: u64,
        peer: u64, rank: u32, tokens: usize, finish: Option<MoeFinishParams>) -> Result<()> {
        let _class = b.prefill_class(7);
        if self.inject.is_none() { return Err("the final HC mixer has no injection projection".into()); }
        if tokens == 0 || tokens > scratch.tokens { return Err("HC tile exceeds its workspace".into()); }
        let mut p = FlashHcAdd { residual: r, block: y, peer, gates: scratch.inject.device_address(), hidden: H as u32, rank };
        let columns = if self.next || tokens > DECODE_ROWS { H } else { W };
        debug_assert!(tokens <= DECODE_ROWS || H % 128 == 0);
        if let Some(mut f) = finish {
            return unsafe { b.launch_flash_rows(b.chain_entry(tokens, "flash_next_hc_finish_add", "flash_next_hc_finish_add"), columns.div_ceil(128) as u32, tokens as u32, 128, 0,
                &mut [(&mut p as *mut FlashHcAdd).cast(), (&mut f as *mut MoeFinishParams).cast()]) };
        }
        unsafe { b.launch_flash_rows(if self.next {b.chain_entry(tokens, "flash_next_hc_add", "flash_next_hc_add_chain")}else{"flash_hc_add"}, columns.div_ceil(128) as u32, tokens as u32, 128, 0,
            &mut [(&mut p as *mut FlashHcAdd).cast()]) }
    }
}

/// Decoder 1's lexical block. Row gathering and the two-token n-gram history
/// belong to the distributed input path; this owns only convolution history.
/// Scope is one unpadded sequence; tokens pass in prompt/decode order.
pub struct Ple<'a, 'w> {
    key: u64,
    value: u64,
    norms: [u64; 3],
    norm_f32: bool,
    conv: u64,
    history: DeviceVector<'a>,
    query_work: DeviceVector<'a>,
    /// Up to PLE_SLICE lexical rows and their key and value projections.
    lexical: DeviceVector<'a>,
    keys: DeviceVector<'a>,
    values: DeviceVector<'a>,
    _weights: PhantomData<&'w ()>,
}
/// Tokens gated and convolved per PLE tile.
pub const PLE_TOKENS: usize = 64;
/// Prompt rows whose key and value projections share one pass over the weights (four tiles).
pub const PLE_SLICE: usize = 4 * PLE_TOKENS;
/// Widest verify pass whose PLE rows stage at once (stage_pass, in DECODE_ROWS pieces).
pub const PLE_PASS_ROWS: usize = 2 * DECODE_ROWS;
impl<'a, 'w> Ple<'a, 'w> {
    pub(super) fn history_address(&self) -> u64 { self.history.device_address() }

    pub const HISTORY_BYTES: usize = W * 9 * 4;
    pub const DEVICE_BYTES: usize = Self::HISTORY_BYTES + PLE_TOKENS * W * 4 + PLE_SLICE * (W + 2 * H) * 4;
    pub fn load(b: &Backend<'a>, w: &'w Weights<'_>) -> Result<Self> {
        let p = "model.language_model.layers.1.ple";
        let key = weight(w, &format!("{p}.key_proj.weight"), &[W as u64, H as u64])?;
        let value = weight(w, &format!("{p}.value_proj.weight"), &[H as u64, H as u64])?;
        let loaded = ["norm_query", "norm_key", "norm_conv"].map(|n|
            norm_weight(w, &format!("{p}.{n}.weight")));
        let [query, key_norm, conv_norm] = loaded;
        let (query, f0) = query?; let (key_norm, f1) = key_norm?; let (conv_norm, f2) = conv_norm?;
        if f0 != f1 || f1 != f2 { return Err("PLE norm weights mix BF16 and F32 storage".into()); }
        let (norms, norm_f32) = ([query, key_norm, conv_norm], f0);
        let conv = weight(w, &format!("{p}.conv1d.weight"), &[W as u64, 1, 4])?;
        let mut vectors = b.vector_arena(&[W * 9, PLE_TOKENS * W, PLE_SLICE * H, PLE_SLICE * W, PLE_SLICE * H])?.into_iter();
        let mut result = Self {
            key, value, norms, norm_f32, conv,
            history: vectors.next().unwrap(), query_work: vectors.next().unwrap(),
            lexical: vectors.next().unwrap(), keys: vectors.next().unwrap(), values: vectors.next().unwrap(),
            _weights: PhantomData,
        };
        result.reset(b)?;
        Ok(result)
    }
    /// A new sequence resets all nine normalized convolution inputs. EOS only
    /// affects lexical hashing; it does not clear this causal convolution.
    pub fn reset(&mut self, b: &Backend<'_>) -> Result<()> {
        b.write_vector(&mut self.history, &vec![0.0; W * 9])
    }
    pub(super) fn history_len(&self) -> usize { self.history.len() }
    pub(super) fn save_history(&self, b: &Backend<'a>) -> Result<DeviceVector<'a>> {
        let mut copy = b.vector(self.history.len())?;
        b.copy_range(&self.history, &mut copy, 0, 0, self.history.len())?;
        Ok(copy)
    }
    pub(super) fn select_history(&mut self,saved: &mut DeviceVector<'a>) {std::mem::swap(&mut self.history,saved);}
    pub(super) fn copy_history(&self, b: &Backend<'a>, saved: &mut DeviceVector<'a>) -> Result<()> {
        b.copy_range(&self.history, saved, 0, 0, self.history.len())
    }
    pub(super) fn restore_history(&mut self, b: &Backend<'a>, saved: &DeviceVector<'a>) -> Result<()> {
        b.copy_range(saved, &mut self.history, 0, 0, saved.len())
    }
    /// Key and value projections of up to PLE_SLICE prompt rows in one pass over the
    /// weights; `finish_tile` then gates and convolves them a PLE tile at a time (each row
    /// keeps its canonical K16 projection, so the bytes equal per-tile projections).
    pub(super) fn project_slice(&mut self, b: &Backend<'_>, lexical: &DeviceVector<'_>, tokens: usize) -> Result<()> {
        if tokens == 0 || tokens > PLE_SLICE { return Err("PLE slice holds 1..=256 tokens".into()); }
        b.copy_range(lexical, &mut self.lexical, 0, 0, tokens * H)?;
        next_rows(b, self.key, self.lexical.device_address(), self.keys.device_address(), W, H, tokens)?;
        next_rows(b, self.value, self.lexical.device_address(), self.values.device_address(), H, H, tokens)
    }
    /// Gate and convolve `tokens` rows of the last projected slice, starting at its row `first`.
    pub(super) fn finish_tile(&mut self, b: &Backend<'_>, residual: &mut DeviceVector<'_>, first: usize, tokens: usize) -> Result<()> {
        let keys = self.keys.device_address() + (first * W * 4) as u64;
        let values = self.values.device_address() + (first * H * 4) as u64;
        self.finish_rows(b, residual.device_address(), keys, values, tokens)
    }
    /// The mixed-checkpoint path uses canonical K16 projections at every M.
    pub(super) fn project_next(&mut self,b: &Backend<'_>,residual: &mut DeviceVector<'_>,lexical: &DeviceVector<'_>,tokens: usize) -> Result<()> {
        if tokens==0 || tokens>PLE_TOKENS { return Err("PLE tile holds 1..=64 tokens".into()); }
        b.copy_range(lexical,&mut self.lexical,0,0,tokens*H)?;
        next_rows(b,self.key,self.lexical.device_address(),self.keys.device_address(),W,H,tokens)?;
        next_rows(b,self.value,self.lexical.device_address(),self.values.device_address(),H,H,tokens)?;
        self.finish_rows(b,residual.device_address(),self.keys.device_address(),self.values.device_address(),tokens)
    }
    pub(super) fn project_pool(&self,b: &Backend<'_>,residual: u64,lexical: u64,work: [u64;3],
        mut step: spark_gpu_probe::next_math::SlotStep) -> Result<()> {
        let n=step.rows as usize; let [keys,values,q]=work;
        next_rows(b,self.key,lexical,keys,W,H,n)?; next_rows(b,self.value,lexical,values,H,H,n)?;
        norm(b,residual,self.norms[0],q,n,0,self.norm_f32)?;
        norm(b,keys,self.norms[1],keys,n,0,self.norm_f32)?;
        let mut gate=FlashPleGate { query:q,key:keys,value:values,output:keys,width:H as u32 };
        unsafe { b.launch_flash_rows("flash_ple_gate",1,n as u32,128,0,&mut [(&mut gate as *mut FlashPleGate).cast()])?; }
        norm(b,keys,self.norms[2],q,n,0,self.norm_f32)?;
        let mut p=FlashPleConv { input:q,weight:self.conv,history:0,gated:keys,residual,width:W as u32,tokens:1 };
        unsafe { b.launch_flash_rows("flash_next_ple_slot",W.div_ceil(128) as u32,n as u32,128,0,
            &mut [(&mut p as *mut FlashPleConv).cast(),(&mut step as *mut spark_gpu_probe::next_math::SlotStep).cast()]) }
    }
    /// Consume the complete 16 x 160 lexical embedding gathered from row owners,
    /// then add the gated value and dilation-three convolution to the streams.
    pub fn project(&mut self, b: &Backend<'_>, residual: &mut DeviceVector<'_>, lexical: &DeviceVector<'_>) -> Result<()> {
        let r = dense(residual, W)?;
        self.set_lexical(b, lexical, 0)?;
        self.project_rows(b, r, 1)
    }
    /// Place one token's lexical embedding as row `token` of the next project_rows call.
    pub fn set_lexical(&mut self, b: &Backend<'_>, lexical: &DeviceVector<'_>, token: usize) -> Result<()> {
        if token >= PLE_TOKENS { return Err("PLE tile holds 64 tokens".into()); }
        b.copy_range(lexical, &mut self.lexical, 0, token * H, H)
    }
    /// Decode one token's raw lexical row at `raw` straight into row `token`
    /// (the same kernel and values as the front-end decode, without a copy).
    pub fn decode_lexical(&mut self, b: &Backend<'_>, raw: u64, scale: f32, token: usize) -> Result<()> {
        if token >= PLE_TOKENS { return Err("PLE tile holds 64 tokens".into()); }
        launch(b, "flash_ple_decode", H.div_ceil(128),
            &mut [Ptr(raw), Ptr(self.lexical.device_address() + (token * H * 4) as u64), F32(scale)])
    }
    /// The lexical block for `tokens` consecutive residual rows at `residual`,
    /// in place: their key and value projections in one pass over the weights
    /// (each token's sums as one matrix-vector product), then row-independent
    /// norms and gates. Convolution consumes the tile and the preceding history.
    pub fn project_rows(&mut self, b: &Backend<'_>, residual: u64, tokens: usize) -> Result<()> {
        let (keys, values) = self.project_lexical(b, tokens)?;
        self.finish_rows(b, residual, keys, values, tokens)
    }
    /// Key and value projections of the first `tokens` lexical rows, returned
    /// as the addresses of their [tokens, W] and [tokens, H] outputs.
    pub(super) fn project_lexical(&mut self, b: &Backend<'_>, tokens: usize) -> Result<(u64, u64)> {
        if tokens == 0 || tokens > PLE_TOKENS { return Err("PLE tile holds 1..=64 tokens".into()); }
        let (keys, values) = (self.keys.device_address(), self.values.device_address());
        bf16_row_pair(b, self.lexical.device_address(), H, tokens, (self.key, keys, W, W), (self.value, values, H, H))?;
        Ok((keys, values))
    }
    /// Norms, gate and convolution of this sequence's next `tokens` rows from
    /// their projections at `keys`/`values`, which hold the gated rows after.
    pub(super) fn finish_rows(&mut self, b: &Backend<'_>, residual: u64, keys: u64, values: u64, tokens: usize) -> Result<()> {
        self.gate_rows(b, residual, keys, values, tokens)?;
        self.convolve_rows(b, residual, keys, 0, tokens)
    }
    /// Row-independent work may precede the ordered convolution checkpoints.
    pub(super) fn gate_rows(&mut self, b: &Backend<'_>, residual: u64, keys: u64, values: u64, tokens: usize) -> Result<()> {
        self.gate_at(b, residual, keys, values, 0, tokens)
    }
    /// gate_rows for rows `start..start + tokens` of the query work.
    fn gate_at(&mut self, b: &Backend<'_>, residual: u64, keys: u64, values: u64, start: usize, tokens: usize) -> Result<()> {
        if tokens == 0 || start + tokens > PLE_TOKENS { return Err("PLE tile holds 1..=64 tokens".into()); }
        let q = self.query_work.device_address() + (start * W * 4) as u64;
        norm(b, residual, self.norms[0], q, tokens, 0, self.norm_f32)?;
        norm(b, keys, self.norms[1], keys, tokens, 0, self.norm_f32)?;
        // Each gate warp finishes its key dot product before overwriting that
        // stream. The dead keys hold the direct gated contribution until addition.
        let mut gate = FlashPleGate { query: q, key: keys, value: values, output: keys, width: H as u32 };
        unsafe { b.launch_flash_rows("flash_ple_gate", 1, tokens as u32, 128, 0,
            &mut [(&mut gate as *mut FlashPleGate).cast()])?; }
        norm(b, keys, self.norms[2], q, tokens, 0, self.norm_f32)
    }
    /// A verify pass's PLE rows of every chat at once (rows `lexical`/`residual`, decode row
    /// kernels, tokens <= PLE_PASS_ROWS): key and value projections and the row-independent
    /// norms and gate, in pieces of at most DECODE_ROWS rows so every row keeps the decode
    /// kernels' one-row sums and grids. Returns the conv inputs and gated rows; each row's
    /// ordered causal convolution follows with its own chat's history (convolve_row).
    pub(super) fn stage_pass(&mut self, b: &Backend<'_>, lexical: &DeviceVector<'_>, residual: u64, tokens: usize) -> Result<(u64, u64)> {
        if tokens == 0 || tokens > PLE_PASS_ROWS { return Err("PLE pass holds 1..=32 rows".into()); }
        b.copy_range(lexical, &mut self.lexical, 0, 0, tokens * H)?;
        for start in (0..tokens).step_by(DECODE_ROWS) {
            let n = (tokens - start).min(DECODE_ROWS);
            let at = |v: &DeviceVector<'_>, width: usize| v.device_address() + (start * width * 4) as u64;
            let (x, keys, values) = (at(&self.lexical, H), at(&self.keys, W), at(&self.values, H));
            next_rows(b, self.key, x, keys, W, H, n)?;
            next_rows(b, self.value, x, values, H, H, n)?;
            self.gate_at(b, residual + (start * W * 4) as u64, keys, values, start, n)?;
        }
        Ok((self.query_work.device_address(), self.keys.device_address()))
    }
    /// One row's causal convolution from staged rows (stage_pass), against this history.
    pub(super) fn convolve_row(&mut self, b: &Backend<'_>, input: u64, residual: u64, gated: u64) -> Result<()> {
        let mut conv = FlashPleConv { input, weight: self.conv, history: self.history.device_address(),
            gated, residual, width: W as u32, tokens: 1 };
        unsafe { b.launch_flash("flash_ple_conv", W.div_ceil(128) as u32, 128,
            &mut [(&mut conv as *mut FlashPleConv).cast()]) }
    }
    pub(super) fn convolve_rows(&mut self, b: &Backend<'_>, residual: u64, keys: u64, start: usize, tokens: usize) -> Result<()> {
        let mut conv = FlashPleConv { input: self.query_work.device_address() + (start * W * 4) as u64,
            weight: self.conv, history: self.history.device_address(),
            gated: keys, residual, width: W as u32, tokens: tokens as u32 };
        unsafe { b.launch_flash("flash_ple_conv", W.div_ceil(128) as u32, 128,
            &mut [(&mut conv as *mut FlashPleConv).cast()]) }
    }
}
