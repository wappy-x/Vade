//! SM121a BF16 tensor-core fused causal/sliding-window attention with relative bias, sink, and output gate
//! (`FLASH_WINDOW_ATTN`, kind 105 in `kernels-inkling`).
//!
//! Pure layout, admission, host checks, and exact CPU reference twin shared between host and `kernels-inkling`
//! (no dependency on `kernels/`, keeping Qwen and GLM byte-for-byte untouched).
//!
//! Determinism contract:
//! - For every contiguous chat span, key tiles start at absolute positions that are multiples of 32 and run in ascending order
//!   using `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for both `Q * K^T` (ascending `d = 0..128`)
//!   and `P * V` (ascending `t`).
//! - Online softmax initializes from the exact attention sink (`m = rn(sink), l = 1.0, O = 0.0`), uses
//!   fixed-order warp-butterfly reductions (`xor 1`, `xor 2`) and `fixed_math::exp` / `fixed_math::sigmoid` / `fixed_math::rn`.
//! - Zero FP atomics, zero materialized `[rows, heads, view]` score or weight buffers.

/// Threads per block of `FLASH_WINDOW_ATTN` (4 warps, 16 query rows per warp -> `BQ = 64`).
pub const FW_THREADS: usize = 128;
pub const FW_BQ: usize = 64;
pub const FW_BKV: usize = 32;
/// Specialized head dimension and relative position dimension (exact multiples of 16).
pub const FW_HEAD_DIM: usize = 128;
pub const FW_REL_DIM: usize = 16;
pub const FW_ROWS_MAX: usize = 4096;
pub const FW_HEADS_MAX: usize = 128;
pub const FW_VIEW_MAX: usize = 1 << 24;
pub const FW_EXTENT_MAX: usize = 65_536;
/// Minimum query rows in a chat span for `FLASH_WINDOW_ATTN` to replace the materialized path.
pub const FW_ROWS_MIN: usize = 1;

/// Payload word 16 of a `FLASH_WINDOW_ATTN` command: 0 (the tensor-core form) or `FW_EXACT_ONE`, the exact one-row
/// form. The exact form gives the words of `RelBias`, `WindowScores`, `SinkSoftmax` and `WindowValues` of the block
/// arms (one query row): the same products, sums in the same order and the same roundings, in one launch.
pub const FW_MODE_WORD: usize = 16;
pub const FW_EXACT_ONE: u64 = 1;
/// The largest live key count of the exact one-row form: its scores, exps and weights live in shared memory (6 blocks
/// of `block_attention::BLOCK` distances = 24,576 bytes, fitting inside the 44,288-byte static shared array alongside
/// Q, M, R and the 16 KB warp key stage).
pub const FW_ONE_VIEW_MAX: usize = 6144;
/// Blocks of one query head in the exact one-row form: each makes the scores and weights of the head and the values
/// of `FW_HEAD_DIM / FW_ONE_SPLIT` channels.
pub const FW_ONE_SPLIT: usize = 4;
/// The most query rows of one exact launch (the rows of a check pass of the draft model: the newest token and the
/// proposed ones). Each row has blocks of its own; all rows read the same key and value view.
pub const FW_ONE_ROWS_MAX: usize = 16;

/// Payload word of the exact form: the blocks of a query head of a row (1, 2 or `FW_ONE_SPLIT`; each makes the
/// scores of the head and the values of `FW_ONE_SPLIT / split` quarters of its channels).
pub const FW_SPLIT_WORD: usize = 22;
/// Optional key/value address pairs for independent rows. A zero count uses the shared planes.
pub const FW_PLANE_COUNT: usize = 23;
pub const FW_PLANES_AT: usize = 24;
/// Payload word of the tensor-core form: 0, or `FW_CLIP_IN` when query and moved are the products before their clips
/// and the block makes the clips itself (words 17 to 21 as in the exact form: the two scales, the position factors,
/// the two bounds). Each clipped word is the word of the prefill `HeadClip` launch. It follows the plane
/// addresses of the exact form, so no word has two meanings.
pub const FW_CLIP_WORD: usize = FW_PLANES_AT + 2 * FW_ONE_ROWS_MAX;
pub const FW_CLIP_IN: u64 = 1;
/// Bit 1 of `FW_CLIP_WORD` (tensor-core prefill form): pack the BF16-rounded output two values per u32 word
/// (`rows * heads * head_dim / 2` words), directly consumable by `PLAIN_TC_BIG` without `TC_MODE_PACK`.
pub const FW_HALF_OUT: u64 = 2;
/// Optional provisional written key/value planes of the exact form (mutually exclusive with `FW_PLANE_COUNT != 0`):
/// distances `d <= row` read `written[row - d]`, while `d > row` read the prior ring/plane `keys`/`values` unmodified.
pub const FW_WRITTEN_KEYS: usize = FW_CLIP_WORD + 1;
pub const FW_WRITTEN_VALUES: usize = FW_CLIP_WORD + 2;
const _: () = assert!((FW_WRITTEN_VALUES + 1) * 8 <= 1024);
/// The blocks a query head for `rows` rows: `FW_ONE_SPLIT` for one row, fewer as the rows fill the device (each block
/// makes the scores of its head again, so fewer blocks a head make fewer scores in all).
#[inline(always)]
pub const fn flash_window_one_split(rows: usize) -> usize { if rows <= 1 { FW_ONE_SPLIT } else if rows == 2 { 2 } else { 1 } }

/// Admission of the exact form: 1 to `FW_ONE_ROWS_MAX` query rows, `split` (1, 2 or `FW_ONE_SPLIT`) blocks a query
/// head of a row, `FW_THREADS` threads. The plane `view` may span up to `FW_VIEW_MAX` (e.g. a global KV buffer);
/// each row's live key count `n` is checked against `FW_ONE_VIEW_MAX` on device after `live_of`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn flash_window_one_admitted(rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, keys: usize, values: usize, threads: usize, grid: usize, split: usize) -> bool {
    (split == 1 || split == 2 || split == FW_ONE_SPLIT) &&
    rows >= 1 && rows <= FW_ONE_ROWS_MAX && heads >= 1 && heads <= FW_HEADS_MAX && kv_heads >= 1 && kv_heads <= heads && heads % kv_heads == 0
        && view >= 1 && view <= FW_VIEW_MAX && extent >= 1 && extent <= FW_EXTENT_MAX
        && threads == FW_THREADS && grid == rows * heads * split && (keys | values) & 15 == 0
}

/// Shape admission for `FLASH_WINDOW_ATTN`.
#[inline(always)]
pub const fn flash_window_shape_admitted(rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize) -> bool {
    rows >= 1 && rows <= FW_ROWS_MAX
        && heads >= 1 && heads <= FW_HEADS_MAX
        && kv_heads >= 1 && kv_heads <= heads && heads % kv_heads == 0
        && head_dim == FW_HEAD_DIM && rel_dim == FW_REL_DIM
        && view >= 1 && view <= FW_VIEW_MAX
        && extent >= 1 && extent <= FW_EXTENT_MAX
}

/// Query heads packed into one tensor-core prefill CTA (`3` when `group % 3 == 0`, `2` when `group % 2 == 0`, else `1`).
#[inline(always)]
pub const fn flash_window_gpack(heads: usize, kv_heads: usize) -> usize {
    let group = heads / kv_heads;
    if group % 3 == 0 { 3 } else if group % 2 == 0 { 2 } else { 1 }
}

/// Query positions per CTA for a packed-GQA tensor-core prefill launch (`32` for 2- or 3-head pack, `64` for 1 head).
#[inline(always)]
pub const fn flash_window_bq_pos(gpack: usize) -> usize {
    if gpack == 1 { FW_BQ } else { 32 }
}

/// Threads per block for tensor-core `FLASH_WINDOW_ATTN` (`192` for 3-head GQA pack, `128` otherwise).
#[inline(always)]
pub const fn flash_window_threads(heads: usize, kv_heads: usize) -> usize {
    if flash_window_gpack(heads, kv_heads) == 3 { 192 } else { FW_THREADS }
}

/// Device-side admission guard for `FLASH_WINDOW_ATTN`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn flash_window_admitted(rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, q: usize, k: usize, v: usize, moved: usize, gate: usize, threads: usize) -> bool {
    flash_window_shape_admitted(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent)
        && threads == flash_window_threads(heads, kv_heads)
        && (q | k | v | moved | gate) & 15 == 0
}

/// Grid blocks for `FLASH_WINDOW_ATTN` (legacy 1-head formula).
#[inline(always)]
pub const fn flash_window_blocks(rows: usize, heads: usize) -> u32 {
    (heads * rows.div_ceil(FW_BQ)) as u32
}

/// Grid blocks for packed-GQA `FLASH_WINDOW_ATTN`: `(heads / gpack) * ceil(rows / bq_pos)`.
#[inline(always)]
pub const fn flash_window_blocks_gqa(rows: usize, heads: usize, kv_heads: usize) -> u32 {
    let gpack = flash_window_gpack(heads, kv_heads);
    let bq = flash_window_bq_pos(gpack);
    ((heads / gpack) * rows.div_ceil(bq)) as u32
}

/// Query positions per CTA for high-occupancy 2-stage `PREFILL_FLASH_WINDOW` (kind 122).
pub const FW_PREFILL_BQ_POS: usize = 64;

/// Threads per block for `PREFILL_FLASH_WINDOW`: `gpack * 128` (`384` for 3:1 GQA, `256` for 2:1, `128` for 1:1).
#[inline(always)]
pub const fn flash_window_prefill_threads(heads: usize, kv_heads: usize) -> usize {
    if kv_heads == 0 || heads % kv_heads != 0 { return 0; }
    flash_window_gpack(heads, kv_heads) * FW_THREADS
}

/// Grid blocks for `PREFILL_FLASH_WINDOW`: `(heads / gpack) * ceil(rows / 64)`.
#[inline(always)]
pub const fn flash_window_prefill_blocks(rows: usize, heads: usize, kv_heads: usize) -> u32 {
    let gpack = flash_window_gpack(heads, kv_heads);
    ((heads / gpack) * rows.div_ceil(FW_PREFILL_BQ_POS)) as u32
}

/// Device/host admission guard for `PREFILL_FLASH_WINDOW` (kind 122).
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn flash_window_prefill_admitted(
    rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize,
    q: usize, k: usize, v: usize, moved: usize, rel_table: usize, gate: usize, output: usize, threads: usize,
) -> bool {
    rows >= FW_PREFILL_BQ_POS
        && flash_window_shape_admitted(rows, heads, kv_heads, FW_HEAD_DIM, FW_REL_DIM, view, extent)
        && threads == flash_window_prefill_threads(heads, kv_heads)
        && (q | k | v | moved | rel_table | gate | output) & 15 == 0
}

/// Split-KV GQA-packed tensor-core decode/verify attention (`FLASH_WINDOW_SPLIT16`, `FLASH_WINDOW_SPLIT32`,
/// `FLASH_WINDOW_REDUCE`): handles M=1..16 contiguous rows (including zero-copy provisional sliding-window deltas)
/// and C2..C16 independent per-chat planes with identical per-token floating-point arithmetic across M and C.
pub const FW_SPLIT_BKV: usize = 32;
pub const FW_SPLIT_ROWS_MAX: usize = 16;
pub const FW_SPLIT_GROUP_MAX: usize = 16;
pub const FW_SPLIT_PARTIAL_STRIDE: usize = FW_HEAD_DIM + 2;
/// Fixed per-layer-kind KV split sizes (multiples of `FW_SPLIT_BKV`, independent of M and C so M=1 == M=2..16 == C1..C16).
/// Ring (`window=1024`): `96` -> `11-12` splits x `8` kv_heads = `88-96` CTAs (~2.0 balanced waves on 48 SMs).
/// Global (`ctx=4096+`): `256` -> `16-17` splits x `8` kv_heads = `128-136` CTAs.
pub const FW_SPLIT_KV_RING: usize = 96;
pub const FW_SPLIT_KV_GLOBAL: usize = 256;
/// Payload word offsets for `FLASH_WINDOW_SPLIT16` / `FLASH_WINDOW_SPLIT32`.
pub const FW_SP_FLAGS_WORD: usize = 16;
pub const FW_SP_SPLIT_START_WORD: usize = 22;
pub const FW_SP_PLANE_COUNT_WORD: usize = 23;
pub const FW_SP_ROWS_PER_CTA_WORD: usize = 24;
pub const FW_SP_PLANES_AT: usize = 32;
pub const FW_SP_POS_AT: usize = 64;
pub const FW_SP_LIVE_AT: usize = 80;
const _: () = assert!((FW_SP_LIVE_AT + FW_SPLIT_ROWS_MAX) * 8 <= 1024);

/// Pack `(split_kv, num_splits, is_ring)` into `FW_SP_FLAGS_WORD`.
#[inline(always)]
pub const fn fw_split_pack_flags(split_kv: usize, num_splits: usize, is_ring: bool) -> u64 {
    (split_kv as u64 & 0xffff_ffff) | (((num_splits as u64) & 0x7fff_ffff) << 32) | ((is_ring as u64) << 63)
}

/// Fixed KV split size for a layer kind (`is_ring`: sliding window vs full global context).
#[inline(always)]
pub const fn flash_window_split_kv(is_ring: bool) -> usize {
    if is_ring { FW_SPLIT_KV_RING } else { FW_SPLIT_KV_GLOBAL }
}

/// Bucketed `grid.y` for `FLASH_WINDOW_SPLIT16` / `FLASH_WINDOW_SPLIT32` so split-count growth and ring-wrap
/// oscillation (`11 <-> 12`) keep stable CUDA Graph keys. Extra blocks (`block_idx_y >= num_splits`) exit at entry.
#[inline(always)]
pub const fn flash_window_split_grid_y(num_splits: usize) -> usize {
    if num_splits <= 2 { num_splits }
    else if num_splits <= 4 { 4 }
    else if num_splits <= 8 { 8 }
    else if num_splits <= 12 { 12 }
    else if num_splits <= 16 { 16 }
    else if num_splits <= 24 { 24 }
    else if num_splits <= 32 { 32 }
    else { num_splits.next_multiple_of(16) }
}

/// Choose `Q_GROUPS` (`1` for `<= 16` queries/KV-head, `2` for `> 16`) and `rows_per_cta` for a single-chat or multi-chat launch.
#[inline(always)]
pub const fn flash_window_split_q_groups(rows: usize, heads: usize, kv_heads: usize, multi_chat: bool) -> (usize, usize) {
    let group = heads / kv_heads;
    if multi_chat {
        return (1, 1);
    }
    let total_q = rows * group;
    if total_q <= 16 {
        (1, rows)
    } else {
        let cap = 32 / group;
        (2, if rows < cap { rows } else { cap })
    }
}

/// Shape & pointer admission for `FLASH_WINDOW_SPLIT16` / `FLASH_WINDOW_SPLIT32`.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub const fn flash_window_split_admitted(
    rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize,
    split_kv: usize, num_splits: usize, q_groups: usize, rows_per_cta: usize, plane_count: usize,
    q: usize, k: usize, v: usize, moved: usize, partials: usize, threads: usize,
) -> bool {
    let group = match heads.checked_div(kv_heads) { Some(g) => g, None => return false };
    rows >= 1 && rows <= FW_SPLIT_ROWS_MAX
        && heads >= 1 && heads <= FW_HEADS_MAX
        && kv_heads >= 1 && kv_heads <= heads && heads % kv_heads == 0
        && group >= 1 && group <= FW_SPLIT_GROUP_MAX
        && view >= 1 && view <= FW_VIEW_MAX
        && extent >= 1 && extent <= FW_EXTENT_MAX
        && split_kv >= FW_SPLIT_BKV && split_kv % FW_SPLIT_BKV == 0
        && num_splits >= 1 && num_splits <= 8192
        && (q_groups == 1 || q_groups == 2)
        && rows_per_cta >= 1 && rows_per_cta * group <= 16 * q_groups
        && (plane_count == 0 || (plane_count == rows && rows_per_cta == 1))
        && threads == FW_THREADS
        && (q | k | v | moved | partials) & 15 == 0
}

#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span { pub address: u64, pub bytes: u64 }
#[cfg(not(target_arch = "nvptx64"))]
impl Span {
    fn ends(self) -> bool { self.address.checked_add(self.bytes).is_some() }
    fn meets(self, other: Span) -> bool {
        self.address < other.address.saturating_add(other.bytes) && other.address < self.address.saturating_add(self.bytes)
    }
}

/// Host bounds, alignment, and non-overlap check for `FLASH_WINDOW_ATTN`.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape and nine device operands")]
pub fn check_flash_window(
    rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
    scale: f32, query: Span, keys: Span, values: Span, moved: Span, rel_table: Span, sink: Span, gate: Span, steps: Span, output: Span,
) -> Result<(), &'static str> {
    check_flash_window_form(rows, heads, kv_heads, head_dim, rel_dim, view, extent, scale, false, query, keys, values, moved, rel_table, sink, gate, steps, output)
}

/// Host bounds, alignment, and non-overlap check for `FLASH_WINDOW_ATTN` with optional packed BF16 output (`half_out`).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, output form and nine device operands")]
pub fn check_flash_window_form(
    rows: usize, heads: usize, kv_heads: usize, head_dim: usize, rel_dim: usize, view: usize, extent: usize,
    scale: f32, half_out: bool, query: Span, keys: Span, values: Span, moved: Span, rel_table: Span, sink: Span, gate: Span, steps: Span, output: Span,
) -> Result<(), &'static str> {
    if !flash_window_shape_admitted(rows, heads, kv_heads, head_dim, rel_dim, view, extent) {
        return Err("flash window: shape outside tensor-core attention envelope");
    }
    if !(scale > 0.0 && scale.is_finite()) { return Err("flash window: score scale must be positive finite"); }
    let expect_q = (rows as u64).checked_mul(heads as u64).and_then(|n| n.checked_mul(head_dim as u64)).and_then(|n| n.checked_mul(4));
    let expect_out = if half_out { expect_q.map(|b| b / 2) } else { expect_q };
    let expect_kv = (view as u64).checked_mul(kv_heads as u64).and_then(|n| n.checked_mul((head_dim / 2) as u64)).and_then(|n| n.checked_mul(4));
    let expect_m = (rows as u64).checked_mul(heads as u64).and_then(|n| n.checked_mul(rel_dim as u64)).and_then(|n| n.checked_mul(4));
    let expect_t = (rel_dim as u64).checked_mul(extent as u64).and_then(|n| n.checked_mul(4));
    let expect_s = (rows as u64).checked_mul(heads as u64).and_then(|n| n.checked_mul(4));
    let expect_st = (rows as u64).checked_mul(8);
    if Some(query.bytes) != expect_q || Some(gate.bytes) != expect_q || Some(output.bytes) != expect_out {
        return Err("flash window: query, gate or output byte count differs from rows x heads x dim");
    }
    if Some(keys.bytes) != expect_kv || Some(values.bytes) != expect_kv {
        return Err("flash window: key or value plane differs from view x kv_heads x dim/2 packed words");
    }
    if Some(moved.bytes) != expect_m { return Err("flash window: relative state differs from rows x heads x rel_dim"); }
    if Some(rel_table.bytes) != expect_t { return Err("flash window: relative table differs from rel_dim x extent"); }
    if Some(sink.bytes) != expect_s { return Err("flash window: sink differs from rows x heads"); }
    if Some(steps.bytes) != expect_st { return Err("flash window: steps differs from rows x 2"); }
    for sp in [query, keys, values, moved, gate] {
        if sp.address == 0 || !sp.ends() || sp.address % 16 != 0 {
            return Err("flash window: 16-byte aligned operand invalid");
        }
    }
    for sp in [rel_table, sink, steps, output] {
        if sp.address == 0 || !sp.ends() || sp.address % 4 != 0 {
            return Err("flash window: 4-byte aligned operand invalid");
        }
    }
    if half_out && output.address % 16 != 0 {
        return Err("flash window: half output not 16-byte aligned");
    }
    for inp in [query, keys, values, moved, rel_table, sink, gate, steps] {
        if output.meets(inp) { return Err("flash window: output overlaps an input"); }
    }
    Ok(())
}

/// CPU reference matching the exact online-softmax + sink + relative-bias + gate math of `FLASH_WINDOW_ATTN`
/// (used in unit tests and GPU validation checks).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments)]
pub fn flash_window_reference(
    rows: usize, heads: usize, kv_heads: usize, view: usize, extent: usize, scale: f32,
    query: &[f32], keys: &[u32], values: &[u32], moved: &[f32], rel_table: &[f32], sink: &[f32], gate: &[f32], steps: &[f32],
) -> Vec<f32> {
    use crate::fixed_math::{exp, rn, sigmoid};
    let group = heads / kv_heads;
    let mut out = vec![0.0f32; rows * heads * FW_HEAD_DIM];
    let unpack = |w: u32, hi: bool| f32::from_bits(if hi { w & 0xffff_0000 } else { w << 16 });
    for r in 0..rows {
        let live = steps[2 * r] as usize;
        let slot = steps[2 * r + 1] as usize;
        for h in 0..heads {
            let kvh = h / group;
            let q = &query[(r * heads + h) * FW_HEAD_DIM..(r * heads + h + 1) * FW_HEAD_DIM];
            let mv = &moved[(r * heads + h) * FW_REL_DIM..(r * heads + h + 1) * FW_REL_DIM];
            let g = &gate[(r * heads + h) * FW_HEAD_DIM..(r * heads + h + 1) * FW_HEAD_DIM];
            let sink_v = rn(sink[r * heads + h]);
            let mut scores = vec![0.0f32; live];
            let mut max_s = sink_v;
            for dist in 0..live {
                let phys = if dist <= slot { slot - dist } else { slot + view - dist };
                let krow = &keys[(phys * kv_heads + kvh) * (FW_HEAD_DIM / 2)..(phys * kv_heads + kvh + 1) * (FW_HEAD_DIM / 2)];
                let mut dot = 0.0f32;
                for j in 0..(FW_HEAD_DIM / 2) {
                    dot += rn(q[2 * j]) * unpack(krow[j], false);
                    dot += rn(q[2 * j + 1]) * unpack(krow[j], true);
                }
                let scaled = rn(rn(dot) * scale);
                let bias = if dist < extent {
                    let mut b = 0.0f32;
                    for j in 0..FW_REL_DIM { b += rn(mv[j]) * rel_table[j * extent + dist]; }
                    rn(b)
                } else { 0.0 };
                let sc = rn(scaled + bias);
                scores[dist] = sc;
                if sc > max_s { max_s = sc; }
            }
            let mut denom = exp(sink_v - max_s);
            let mut acc = [0.0f32; FW_HEAD_DIM];
            for (dist, &score) in scores.iter().enumerate() {
                let w = exp(score - max_s);
                denom += w;
                let w_bf16 = rn(w);
                let phys = if dist <= slot { slot - dist } else { slot + view - dist };
                let vrow = &values[(phys * kv_heads + kvh) * (FW_HEAD_DIM / 2)..(phys * kv_heads + kvh + 1) * (FW_HEAD_DIM / 2)];
                for j in 0..(FW_HEAD_DIM / 2) {
                    acc[2 * j] += w_bf16 * unpack(vrow[j], false);
                    acc[2 * j + 1] += w_bf16 * unpack(vrow[j], true);
                }
            }
            let inv = 1.0f32 / denom;
            let dst = &mut out[(r * heads + h) * FW_HEAD_DIM..(r * heads + h + 1) * FW_HEAD_DIM];
            for d in 0..FW_HEAD_DIM {
                dst[d] = rn(rn(acc[d] * inv) * sigmoid(rn(g[d])));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_and_host_checks() {
        assert!(flash_window_shape_admitted(4096, 24, 8, 128, 16, 8192, 4096));
        assert!(!flash_window_shape_admitted(0, 24, 8, 128, 16, 8192, 1024));
        assert!(!flash_window_shape_admitted(1024, 24, 8, 64, 16, 8192, 1024));
        assert!(!flash_window_shape_admitted(1024, 24, 5, 128, 16, 8192, 1024));
        assert_eq!(flash_window_blocks(4096, 24), 24 * 64);
        let sp = |a: u64, b: u64| Span { address: a, bytes: b };
        let q = sp(0x10000, 16 * 24 * 128 * 4);
        let k = sp(0x100000, 64 * 8 * 64 * 4);
        let v = sp(0x200000, 64 * 8 * 64 * 4);
        let m = sp(0x300000, 16 * 24 * 16 * 4);
        let t = sp(0x400000, 16 * 1024 * 4);
        let s = sp(0x500000, 16 * 24 * 4);
        let g = sp(0x600000, 16 * 24 * 128 * 4);
        let st = sp(0x700000, 16 * 8);
        let o = sp(0x800000, 16 * 24 * 128 * 4);
        assert!(check_flash_window(16, 24, 8, 128, 16, 64, 1024, 0.088_388_35, q, k, v, m, t, s, g, st, o).is_ok());
        assert!(check_flash_window(16, 24, 8, 128, 16, 64, 1024, 0.088_388_35, q, k, v, m, t, s, g, st, q).is_err());
        assert!(check_flash_window(16, 24, 8, 128, 16, 64, 1024, 0.0, q, k, v, m, t, s, g, st, o).is_err());
        assert_eq!(flash_window_split_grid_y(1), 1);
        assert_eq!(flash_window_split_grid_y(2), 2);
        assert_eq!((flash_window_split_grid_y(3), flash_window_split_grid_y(4)), (4, 4));
        assert_eq!((flash_window_split_grid_y(11), flash_window_split_grid_y(12)), (12, 12));
        assert_eq!((flash_window_split_grid_y(17), flash_window_split_grid_y(24)), (24, 24));
        assert_eq!((flash_window_split_grid_y(25), flash_window_split_grid_y(33)), (32, 48));
    }
}
