//! Fast zero-stack Inkling block-op kernels (kinds 107..=112 in `kernels-inkling`, Ruling R6h).
//!
//! Pure layout, admission, and host checks shared between host and `kernels-inkling` (zero dependency
//! on `kernels/` or `kernels-tc/`, keeping Qwen and GLM byte-for-byte untouched).
//!
//! Entries:
//! - `FAST_MIX_BLEND` (107): computes per-(row, query) softmax weights once into 256B smem and blends
//!   all channels with optional direct `[asked][rows][d]` transpose output (eliminating post-mix swaps).
//! - `FAST_ROW_DOTS` (108): warp-parallel 32-lane fixed-order dot products for list-form `RowDots`.
//! - `FAST_NORM` (109): zero-stack bit-exact `PlainNorm` (1 warp/row) and `RoundedNorm` (1 warp/row
//!   for `d == 128`, 4 warps/row for `d % 128 == 0`).
//! - `FAST_CAPPED_GATE` (110): zero-stack bit-exact `CappedGate`, 1 block/pick, zero divisions.
//! - `FAST_ROUTER_PICK` (111): warp-parallel bit-exact `RouterPick` (1 warp/token, computes `sigmoid` once).
//! - `FAST_STREAM_OPS` (112): zero-stack bit-exact `ShortConv` (window form), `RouteFold`, and `GateBlend`.

/// Multi-chat kept-ring descriptor tag packed into `first` for `STREAM_OP_SHORT_CONV_WIN` (`rows <= 16`, `taps <= 4`):
/// bits `0..16` hold `start_mask` (bit `r` set iff row `r` starts a chat), bits `16..48` hold 2-bit per-chat `first_slot`.
pub const SHORT_CONV_MULTI_TAG: usize = 1usize << 62;
const _: () = assert!(usize::BITS >= 64);

/// Encode a multi-chat kept-ring descriptor (`2..=16` chats, `rows <= 16`, `taps` in `2..=4`).
#[inline(always)]
pub const fn short_conv_multi_encode(rows: usize, taps: usize, start_mask: u16, slots: u32) -> Option<usize> {
    if rows < 2 || rows > 16 || taps < 2 || taps > 4 { return None; }
    if (start_mask & 1) == 0 || ((start_mask as u32) >> rows) != 0 { return None; }
    let chats = start_mask.count_ones() as usize;
    if chats < 2 || chats > rows { return None; }
    if chats < 16 && (slots >> (2 * chats)) != 0 { return None; }
    let mut c = 0usize;
    while c < chats {
        if ((slots >> (2 * c)) & 3) as usize >= taps { return None; }
        c += 1;
    }
    Some(SHORT_CONV_MULTI_TAG | (start_mask as usize) | ((slots as usize) << 16))
}

/// Decode and validate a multi-chat kept-ring descriptor, returning `(start_mask, slots, chats)`.
#[inline(always)]
pub const fn short_conv_multi_decode(rows: usize, taps: usize, packed: usize) -> Option<(u16, u32, usize)> {
    if (packed & SHORT_CONV_MULTI_TAG) == 0 || (packed & !(SHORT_CONV_MULTI_TAG | 0xffff_ffff_ffff)) != 0 {
        return None;
    }
    let start_mask = (packed & 0xffff) as u16;
    let slots = ((packed >> 16) & 0xffff_ffff) as u32;
    if short_conv_multi_encode(rows, taps, start_mask, slots).is_none() { return None; }
    Some((start_mask, slots, start_mask.count_ones() as usize))
}

/// Sub-operation selector for `FAST_STREAM_OPS` (kind 112).
pub const STREAM_OP_SHORT_CONV_WIN: u64 = 0;
pub const STREAM_OP_ROUTE_FOLD: u64 = 1;
pub const STREAM_OP_GATE_BLEND: u64 = 2;
pub const STREAM_OP_ROUTE_FOLD_CONV: u64 = 3;
pub const STREAM_OP_GATE_BLEND_NORM: u64 = 4;
pub const STREAM_OP_RING_CONV_ONE: u64 = 5;
/// Prefill fused `RouteFold (FOLD_HALF)` + `AfterExperts ShortConv` + `next_ring` over 64-row channel tiles.
pub const STREAM_OP_ROUTE_FOLD_CONV_PREFILL: u64 = 6;
/// Rows processed per CTA in 2D channel-tiled prefill `ShortConv` and `STREAM_OP_ROUTE_FOLD_CONV_PREFILL`.
pub const SHORT_CONV_PREFILL_TILE: usize = 64;
/// Maximum number of gates batched in one `STREAM_OP_GATE_BLEND_NORM` launch (`grid.y`).
pub const GATE_BLEND_NORM_GATES_MAX: usize = 3;
/// Payload word offset of the `GATE_BLEND_NORM` gate table (5 words per gate: gates, kept, msg, weight, out).
pub const GATE_BLEND_NORM_TABLE_AT: usize = 16;
const _: () = assert!(8 * (GATE_BLEND_NORM_TABLE_AT + 5 * GATE_BLEND_NORM_GATES_MAX) <= 1024);

/// Multi-row paired KV finish selector for `FAST_ROW_DOTS` (kind 108, word 0 == tag and word 1 == 0).
/// Fuses Key (`ShortConv` + `HeadClip` + `PackBf16` + optional `every`) and Value (`ShortConv` +
/// `RoundedNorm` + `PackBf16` + optional `every`) for `head_dim == 128` into one 128-thread launch.
pub const KV_FINISH_TAG: u64 = 0x4b56_4649_4e49_5348;
pub const KV_FINISH_HEAD_DIM: usize = 128;
pub const KV_FINISH_THREADS: usize = ROW_DOTS_THREADS;
pub const ROUTE_FOLD_CONV_ROWS_MAX: usize = 16;
pub const ROUTE_FOLD_CONV_TAPS_MAX: usize = 4;
pub const ROUTE_FOLD_CONV_WEIGHTS_MAX: usize = 128;
pub const ROUTE_FOLD_CONV_THREADS: usize = 32;
/// Form word of a `STREAM_OP_ROUTE_FOLD_CONV` command (`FOLD_HALF` bit for BF16 downs); sits after up to 16 `chat_rings` (words 17..=48).
pub const ROUTE_FOLD_CONV_FORM_WORD: usize = 50;
const _: () = assert!(8 * (ROUTE_FOLD_CONV_FORM_WORD + 1) <= 1024);
/// In the form word of a `RouteFold` (`FOLD_FORM_WORD`): the downs are BF16, two a word (the even channel in the low
/// half), the words of a `PLAIN_PACKED_TC_ROWS` launch with `packed_product::HALF_OUT`. Each read gives the FP32 value of
/// its BF16 bits, which is `rn` of the FP32 down the fold reads in its other form: the same sum, word for word.
pub const FOLD_HALF: u64 = 1;
/// The form word of a `RouteFold` command (words 9 and 10 are its weight stride and offset).
pub const FOLD_FORM_WORD: usize = 12;

/// Finish modes for the prefill window convolution written to a packed KV plane.
pub const CONV_KV_CLIP: u64 = 1;
pub const CONV_KV_NORM: u64 = 2;
pub const CONV_KV_DIM: usize = 128;
/// FAST_ROW_DOTS payload flag for the KV finish. The list-form pointers occupy words 7 through 70.
pub const KV_FINISH_WORD: usize = 80;
const _: () = assert!(7 + MIX_KEPT_MAX <= KV_FINISH_WORD && 8 * (KV_FINISH_WORD + 1) <= 1024);

#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy)]
pub struct KvWrite {
    pub first: usize,
    pub plane_rows: usize,
    pub clip: bool,
    pub param: f32,
}

/// One warp per head, four whole heads per block except for the last block.
pub const fn conv_kv_blocks(rows: usize, heads: usize) -> usize {
    (rows * heads).div_ceil(ROW_DOTS_THREADS / 32)
}

/// Threads per block of each fast kernel.
pub const MIX_BLEND_THREADS: usize = 256;
pub const ROW_DOTS_THREADS: usize = 128;
pub const NORM_THREADS: usize = 128;
pub const CAPPED_GATE_THREADS: usize = 256;
pub const CAPPED_GATE_TABLE_WORDS: usize = 1 << 16;
pub const ROUTER_PICK_THREADS: usize = 128;
/// Single-block threads for opt-in unique-expert compaction inside `FAST_ROUTER_PICK` (`rows <= 8`).
pub const ROUTER_PICK_COMPACT_THREADS: usize = 256;
pub const ROUTER_PICK_COMPACT_ROWS_MAX: usize = 8;
pub const ROUTER_PICK_COMPACT_PICKS_MAX: usize = 8;
/// Payload word offset of the optional compacted dispatch buffer in `FAST_ROUTER_PICK` (`0` = none).
pub const ROUTER_PICK_COMPACT_WORD: usize = 8;
/// 32-bit words per compacted entry (`indices_lo, indices_hi, expert_id, count | (ordinal << 16)`).
pub const ROUTER_PICK_COMPACT_ENTRY_WORDS: usize = 4;
pub const STREAM_OPS_THREADS: usize = 256;

/// Envelope limits.
pub const MIX_KEPT_MAX: usize = 64;
pub const FAST_ROWS_MAX: usize = 65_536;
pub const FAST_DIM_MAX: usize = 16_384;
pub const ROUTER_EXPERTS_MAX: usize = 128;
pub const ROUTER_PICKS_MAX: usize = 32;
pub const NORM_CHUNKS_MAX: usize = 32;

/// Blocks of one `(row, query)` of the mix blend: one channel a thread (a 2,304-wide row is 9 blocks). Each block makes
/// the same `kept` weights (thread 0, the same exp, sum and divisions in the same order) and blends its own channels,
/// `part * MIX_BLEND_THREADS + thread`, then every `parts * MIX_BLEND_THREADS`: one writer a word, the same words.
/// MEASURED (nsys): one block a `(row, query)` took 5.6 us a launch for 2,304 channels.
#[inline(always)]
pub const fn mix_blend_parts(dim: usize) -> usize { if dim == 0 { 1 } else { dim.div_ceil(MIX_BLEND_THREADS) } }
#[inline(always)]
pub const fn mix_blend_admitted(rows: usize, asked: usize, kept: usize, dim: usize, threads: usize) -> bool {
    threads == MIX_BLEND_THREADS
        && rows >= 1 && rows <= FAST_ROWS_MAX
        && asked >= 1 && asked <= 1024
        && kept >= 1 && kept <= MIX_KEPT_MAX
        && dim >= 1 && dim <= FAST_DIM_MAX
}

/// Fused plain-norm, dot-chain, softmax, and blend command (`Backend::fast_mix_one`): bit 1 of word 8 selects
/// this form, and bit 0 selects transposed `[asked][rows][dim]` output. Grid is `(asked * parts, rows, 1)` where
/// `parts = mix_one_parts(rows, asked, dim)`. Word 0 is `queries`, words `9..9 + kept` are the `kept - 1` list
/// buffers followed by `stream` (`[rows][dim]` each), word 40 is the optional `normed` output (0 when omitted),
/// and word 41 holds the norm epsilon bits.
pub const MIX_ONE: u64 = 2;
/// A list-form prefill mix whose blend block makes its own dots (`Backend::fast_mix_list`): word 8 of a `FAST_MIX_BLEND`
/// command holds this bit (and bit 0 the transpose flag), and word 0 is the queries, `[asked][dim]`, in place of the
/// scores. Block `(row, q)` makes the `kept` dots of query row `q` with the chain of `FAST_ROW_DOTS` (one warp a stream),
/// then the softmax and the blend of this kind: the words of the two launches, with no score buffer.
pub const MIX_DOTS: u64 = 4;
/// The most queries of a list-form prefill mix that takes `MIX_DOTS` (the attention mix asks 1 or 3). A mix with more
/// queries (the messages) keeps the dots launch: there each stream row is read once for all queries.
pub const MIX_DOTS_ASKED_MAX: usize = 4;
/// The transposed list mix with the gate norm of each blended row (`Backend::fast_mix_list_normed`): the blend words,
/// and the words of `fast_norm`'s rounded form with no weight (`rn(word * scale)`) in a second output. One launch for
/// the prefill messages and their gate norms, so the norm does not read the blend words back.
pub const MIX_LIST_NORM: u64 = 8;
/// Payload words of the list mix with the gate norm: the normed output, and the norm's epsilon bits.
pub const MIX_LIST_NORMED_AT: usize = 96;
pub const MIX_LIST_EPSILON_AT: usize = 97;
const _: () = assert!(9 + MIX_KEPT_MAX <= MIX_LIST_NORMED_AT && 8 * (MIX_LIST_EPSILON_AT + 1) <= 1024);
/// The rows the list mix with the gate norm takes: the rounded wide norm's (a multiple of 128 channels, at most
/// `NORM_CHUNKS_MAX` chunks), on blocks of `MIX_BLEND_THREADS` threads (warp `w` owns chunks `w, w + 8, ..`).
#[inline(always)]
pub const fn mix_list_norm_admitted(dim: usize, threads: usize) -> bool {
    threads == MIX_BLEND_THREADS && dim >= 128 && dim % 128 == 0 && dim / 128 <= NORM_CHUNKS_MAX
}
/// Threads per block for `MIX_ONE`: with fused post-blend `RoundedNorm`, launch 1 warp per 128-element
/// chunk (at least 8 warps = 256 threads so `kept <= 8` list dots run in parallel; 576 threads = 18 warps
/// for `dim = 2304`, giving each warp exactly 1 chunk of 4 floats with zero load imbalance).
#[inline(always)]
pub const fn mix_one_threads(dim: usize, post_norm: bool) -> usize {
    if post_norm && dim >= 128 && dim % 128 == 0 && dim <= MIX_ONE_DIM_MAX {
        let chunks = dim / 128;
        let warps = if chunks > 8 { chunks } else { 8 };
        warps * 32
    } else {
        MIX_BLEND_THREADS
    }
}
/// Admission for `MIX_ONE` with fused post-blend `RoundedNorm` (`dim <= MIX_ONE_DIM_MAX`, 1 warp/chunk).
#[inline(always)]
pub const fn mix_one_post_norm_admitted(dim: usize, threads: usize) -> bool {
    dim >= 128 && dim % 128 == 0 && dim <= MIX_ONE_DIM_MAX && threads == mix_one_threads(dim, true)
}
pub const MIX_ONE_NORMED_AT: usize = 40;
pub const MIX_ONE_EPSILON_AT: usize = 41;
/// Fused post-blend `RoundedNorm` flag in word 8 of a `MIX_ONE` command (bit 4).
pub const MIX_ONE_POST_NORM: u64 = 16;
/// With `MIX_ONE_POST_NORM`, skip writing the un-normed blend output (word 2 may be 0; bit 5).
pub const MIX_ONE_SKIP_MIXED: u64 = 32;
/// With `MIX_ONE_POST_NORM` and non-null `post_weight`, `post_weight` holds `[asked][dim]` words (bit 6).
pub const MIX_ONE_POST_WEIGHTS_PER_QUERY: u64 = 64;
/// With `MIX_ONE_POST_NORM`, write packed BF16 outputs (`rows * asked * dim / 2` u32 words, bit 7).
pub const MIX_ONE_POST_HALF_OUT: u64 = 128;
/// With `MIX_LIST_NORM`, write `normed` as packed BF16 (`rows * asked * dim / 2` u32 words, bit 8).
pub const MIX_LIST_NORM_HALF: u64 = 256;
/// Payload words for fused post-blend `RoundedNorm` in `MIX_ONE`: output, optional weight, epsilon bits.
pub const MIX_ONE_POST_OUT_AT: usize = 42;
pub const MIX_ONE_POST_WEIGHT_AT: usize = 43;
pub const MIX_ONE_POST_EPS_AT: usize = 44;
pub const MIX_ONE_KEPT_MAX: usize = 8;
pub const MIX_ONE_DIM_MAX: usize = 2560;
pub const MIX_ONE_ROWS_MAX: usize = 65_535;
const _: () = assert!(9 + MIX_ONE_KEPT_MAX <= MIX_ONE_NORMED_AT && 8 * (MIX_ONE_POST_EPS_AT + 1) <= 1024);
/// Channel-part split per `(row, query)` for `MIX_ONE`: 1 block of 256 threads per `(row, query)` so
/// 32-lane warp-butterfly dots avoid redundant reads and optional post-blend `RoundedNorm` fuses in registers.
#[inline(always)]
pub const fn mix_one_parts(_rows: usize, _asked: usize, _dim: usize) -> usize { 1 }
#[inline(always)]
pub const fn mix_one_admitted(rows: usize, asked: usize, kept: usize, dim: usize) -> bool {
    mix_blend_admitted(rows, asked, kept, dim, MIX_BLEND_THREADS)
        && rows <= MIX_ONE_ROWS_MAX
        && kept >= 2 && kept <= MIX_ONE_KEPT_MAX
        && dim % 32 == 0 && dim <= MIX_ONE_DIM_MAX
}

#[inline(always)]
pub const fn kv_finish_admitted(rows: usize, kv_heads: usize, taps: usize, threads: usize) -> bool {
    (threads == KV_FINISH_THREADS || (rows == 1 && threads == 32))
        && rows >= 1 && rows <= FAST_ROWS_MAX
        && kv_heads >= 1 && kv_heads <= 64
        && taps >= 2 && taps <= 16
}

#[inline(always)]
pub const fn row_dots_admitted(rows: usize, asked: usize, kept: usize, k: usize, threads: usize) -> bool {
    threads == ROW_DOTS_THREADS
        && rows >= 1 && rows <= FAST_ROWS_MAX
        && asked >= 1 && asked <= 1024
        && kept >= 1 && kept <= MIX_KEPT_MAX
        && k >= 1 && k <= FAST_DIM_MAX
}

#[inline(always)]
pub const fn norm_admitted(rows: usize, dim: usize, rounded: bool, threads: usize) -> bool {
    let ok_dim = if rounded {
        dim >= 128 && dim <= NORM_CHUNKS_MAX * 128 && dim % 128 == 0
    } else {
        dim >= 32 && dim <= FAST_DIM_MAX && dim % 32 == 0
    };
    threads == NORM_THREADS && rows >= 1 && rows <= 262_144 && ok_dim
}

#[inline(always)]
pub const fn norm_blocks(rows: usize, dim: usize, rounded: bool) -> u32 {
    if !rounded || dim == 128 { rows.div_ceil(4) as u32 } else { rows as u32 }
}

#[inline(always)]
pub const fn head_clip_admitted(heads: usize, dim: usize, threads: usize) -> bool {
    threads == NORM_THREADS && heads >= 1 && heads <= 262_144 && dim >= 1 && dim <= FAST_DIM_MAX
}

#[inline(always)]
pub const fn capped_gate_admitted(picks: usize, dim: usize, threads: usize) -> bool {
    threads == CAPPED_GATE_THREADS && picks >= 1 && picks <= 65_536 && dim >= 1 && dim <= FAST_DIM_MAX
}

#[inline(always)]
pub const fn router_pick_admitted(rows: usize, experts: usize, picks: usize, threads: usize) -> bool {
    threads == ROUTER_PICK_THREADS
        && rows >= 1 && rows <= FAST_ROWS_MAX
        && experts >= 2 && experts <= ROUTER_EXPERTS_MAX
        && picks >= 1 && picks <= ROUTER_PICKS_MAX && picks <= experts
}

#[inline(always)]
pub const fn router_pick_compact_admitted(rows: usize, experts: usize, picks: usize, threads: usize) -> bool {
    threads == ROUTER_PICK_COMPACT_THREADS
        && rows >= 1 && rows <= ROUTER_PICK_COMPACT_ROWS_MAX
        && experts >= 2 && experts <= ROUTER_EXPERTS_MAX
        && picks >= 1 && picks <= ROUTER_PICK_COMPACT_PICKS_MAX && picks <= experts
}

/// Blocks of one row of a stream op (`ShortConv` window, `RouteFold`, `GateBlend`): one channel a thread, so a 2,304-wide
/// row is 9 blocks and no thread walks the row in serial load-store rounds. Every channel of a row is in exactly one
/// block (`part * STREAM_OPS_THREADS + thread`), so each output word has one writer and the same arithmetic as with one
/// block a row. MEASURED (nsys): one block a row took 11.4 us a launch for 2,304 channels.
#[inline(always)]
pub const fn stream_ops_parts(dim: usize) -> usize { if dim == 0 { 1 } else { dim.div_ceil(STREAM_OPS_THREADS) } }
/// The grid of a stream op: `rows` rows of `stream_ops_parts(dim)` blocks each.
#[inline(always)]
pub const fn stream_ops_blocks(rows: usize, dim: usize) -> usize { rows * stream_ops_parts(dim) }
/// Threads per block for `STREAM_OP_GATE_BLEND_NORM`: 1 warp per 128-element norm chunk
/// (`576` threads = `18` warps for `dim = 2304`, `640` threads = `20` warps for `dim = 2560`),
/// giving each lane 4 channels with 1 barrier and zero multi-chunk register pressure.
#[inline(always)]
pub const fn gate_blend_norm_threads(dim: usize) -> usize {
    (dim / 128) * 32
}
#[inline(always)]
pub const fn stream_ops_admitted(subop: u64, rows: usize, dim: usize, aux: usize, threads: usize) -> bool {
    if rows == 0 || rows > FAST_ROWS_MAX || dim == 0 || dim > FAST_DIM_MAX {
        return false;
    }
    match subop {
        STREAM_OP_SHORT_CONV_WIN => threads == STREAM_OPS_THREADS && aux >= 1 && aux <= 16,
        STREAM_OP_ROUTE_FOLD => threads == STREAM_OPS_THREADS && aux >= 1 && aux <= ROUTER_PICKS_MAX,
        STREAM_OP_GATE_BLEND => threads == STREAM_OPS_THREADS,
        STREAM_OP_GATE_BLEND_NORM => {
            rows <= 65_535
                && aux >= 1 && aux <= GATE_BLEND_NORM_GATES_MAX
                && dim >= 128 && dim % 128 == 0 && dim <= MIX_ONE_DIM_MAX
                && threads == gate_blend_norm_threads(dim)
        }
        STREAM_OP_ROUTE_FOLD_CONV => {
            threads == ROUTE_FOLD_CONV_THREADS
                && rows <= ROUTE_FOLD_CONV_ROWS_MAX
                && aux >= 1 && aux <= ROUTER_PICKS_MAX
                && rows.saturating_mul(aux) <= ROUTE_FOLD_CONV_WEIGHTS_MAX
        }
        STREAM_OP_RING_CONV_ONE => {
            (threads == ROUTE_FOLD_CONV_THREADS || threads == 64)
                && rows == 1
                && aux >= 2 && aux <= ROUTE_FOLD_CONV_TAPS_MAX
        }
        STREAM_OP_ROUTE_FOLD_CONV_PREFILL => {
            threads == STREAM_OPS_THREADS
                && dim % 2 == 0
                && aux >= 1 && aux <= ROUTER_PICKS_MAX
        }
        _ => false,
    }
}

#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span { pub address: u64, pub bytes: u64 }
#[cfg(not(target_arch = "nvptx64"))]
impl Span {
    pub(crate) fn valid(self, expected: Option<u64>) -> bool {
        expected == Some(self.bytes) && self.address != 0 && self.address % 4 == 0 && self.address.checked_add(self.bytes).is_some()
    }
    pub(crate) fn meets(self, other: Span) -> bool {
        self.address < other.address.saturating_add(other.bytes) && other.address < self.address.saturating_add(self.bytes)
    }
}
#[cfg(not(target_arch = "nvptx64"))]
fn f32_bytes(words: usize) -> Option<u64> { (words as u64).checked_mul(4) }

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_mix_blend(rows: usize, asked: usize, kept: usize, dim: usize, scale: f32, scores: Span, joined: Span, output: Span) -> Result<(), &'static str> {
    if !mix_blend_admitted(rows, asked, kept, dim, MIX_BLEND_THREADS) { return Err("fast mix blend: shape outside envelope"); }
    if !(scale > 0.0 && scale.is_finite()) { return Err("fast mix blend: scale must be positive finite"); }
    let s_w = rows.checked_mul(asked).and_then(|n| n.checked_mul(kept));
    let j_w = kept.checked_mul(rows).and_then(|n| n.checked_mul(dim));
    let o_w = rows.checked_mul(asked).and_then(|n| n.checked_mul(dim));
    if !scores.valid(s_w.and_then(f32_bytes)) || !joined.valid(j_w.and_then(f32_bytes)) || !output.valid(o_w.and_then(f32_bytes)) {
        return Err("fast mix blend: operand size or alignment mismatch");
    }
    if output.meets(scores) || output.meets(joined) { return Err("fast mix blend: output overlaps input"); }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, norm parameters and operand spans")]
pub fn check_mix_rows(
    rows: usize, asked: usize, dim: usize, scale: f32, eps: f32,
    queries: Span, list: &[Span], stream: Span, normed: Option<Span>, output: Span,
) -> Result<(), &'static str> {
    check_mix_rows_normed(rows, asked, dim, scale, eps, None, queries, list, stream, normed, Some(output), None, None, false)
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, norm parameters, optional post-norm and operand spans")]
pub fn check_mix_rows_normed(
    rows: usize, asked: usize, dim: usize, scale: f32, eps: f32, post_eps: Option<f32>,
    queries: Span, list: &[Span], stream: Span, normed: Option<Span>, output: Option<Span>,
    post_out: Option<Span>, post_weight: Option<Span>, post_weight_per_query: bool,
) -> Result<(), &'static str> {
    check_mix_rows_normed_form(rows, asked, dim, scale, eps, post_eps, false, queries, list, stream, normed, output, post_out, post_weight, post_weight_per_query)
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, norm parameters, optional post-norm form and operand spans")]
pub fn check_mix_rows_normed_form(
    rows: usize, asked: usize, dim: usize, scale: f32, eps: f32, post_eps: Option<f32>, post_half: bool,
    queries: Span, list: &[Span], stream: Span, normed: Option<Span>, output: Option<Span>,
    post_out: Option<Span>, post_weight: Option<Span>, post_weight_per_query: bool,
) -> Result<(), &'static str> {
    let kept = list.len().checked_add(1).ok_or("fast mix one: kept overflow")?;
    if !mix_one_admitted(rows, asked, kept, dim) { return Err("fast mix one: shape outside envelope"); }
    if !(scale > 0.0 && scale.is_finite()) || !(eps > 0.0 && eps.is_finite()) {
        return Err("fast mix one: scale and epsilon must be positive finite");
    }
    let row_bytes = rows.checked_mul(dim).and_then(f32_bytes);
    let q_bytes = asked.checked_mul(dim).and_then(f32_bytes);
    let o_bytes = rows.checked_mul(asked).and_then(|n| n.checked_mul(dim)).and_then(f32_bytes);
    let post_bytes = if post_half { o_bytes.map(|b| b / 2) } else { o_bytes };
    if !queries.valid(q_bytes) || queries.address % 16 != 0
        || !stream.valid(row_bytes) || stream.address % 16 != 0
    {
        return Err("fast mix one: operand size or alignment mismatch");
    }
    for &entry in list {
        if !entry.valid(row_bytes) || entry.address % 16 != 0 {
            return Err("fast mix one: list entry size or alignment mismatch");
        }
    }
    if output.is_none() && post_out.is_none() {
        return Err("fast mix one: missing output");
    }
    if let Some(out) = output {
        if !out.valid(o_bytes) || out.meets(queries) || out.meets(stream) || list.iter().any(|&e| out.meets(e)) {
            return Err("fast mix one: output mismatch or overlap");
        }
    }
    match (post_eps, post_out) {
        (Some(pe), Some(po)) => {
            if !mix_one_post_norm_admitted(dim, mix_one_threads(dim, true)) || !(pe > 0.0 && pe.is_finite()) {
                return Err("fast mix one: post-norm shape or epsilon outside envelope");
            }
            if !po.valid(post_bytes) || (post_half && po.address % 16 != 0)
                || po.meets(queries) || po.meets(stream) || list.iter().any(|&e| po.meets(e))
                || output.is_some_and(|out| po.meets(out))
            {
                return Err("fast mix one: post-norm output mismatch or overlap");
            }
            if let Some(pw) = post_weight {
                let pw_words = if post_weight_per_query { asked.checked_mul(dim) } else { Some(dim) };
                if !pw.valid(pw_words.and_then(f32_bytes))
                    || po.meets(pw) || output.is_some_and(|out| out.meets(pw)) || normed.is_some_and(|n| n.meets(pw))
                {
                    return Err("fast mix one: post-norm weight mismatch or overlap");
                }
            }
        }
        (None, None) => {
            if post_weight.is_some() { return Err("fast mix one: post-norm weight without post-norm output"); }
        }
        _ => return Err("fast mix one: post-norm epsilon/output mismatch"),
    }
    if let Some(n) = normed {
        if !n.valid(row_bytes) || output.is_some_and(|out| n.meets(out)) || post_out.is_some_and(|po| n.meets(po))
            || n.meets(queries) || n.meets(stream) || list.iter().any(|&e| n.meets(e))
        {
            return Err("fast mix one: normed output mismatch or overlap");
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, scale, epsilon and five operands")]
pub fn check_mix_one(asked: usize, kept: usize, dim: usize, scale: f32, eps: f32, queries: Span, list: &[Span], stream: Span, normed: Span, output: Span) -> Result<(), &'static str> {
    if kept != list.len().saturating_add(1) { return Err("fast mix one: list length mismatch"); }
    check_mix_rows(1, asked, dim, scale, eps, queries, list, stream, Some(normed), output)
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_row_dots(rows: usize, asked: usize, kept: usize, k: usize, joined: Span, queries: Span, output: Span) -> Result<(), &'static str> {
    if !row_dots_admitted(rows, asked, kept, k, ROW_DOTS_THREADS) { return Err("fast row dots: shape outside envelope"); }
    let j_w = kept.checked_mul(rows).and_then(|n| n.checked_mul(k));
    let q_w = asked.checked_mul(k);
    let o_w = rows.checked_mul(asked).and_then(|n| n.checked_mul(kept));
    if !joined.valid(j_w.and_then(f32_bytes)) || !queries.valid(q_w.and_then(f32_bytes)) || !output.valid(o_w.and_then(f32_bytes)) {
        return Err("fast row dots: operand size or alignment mismatch");
    }
    if output.meets(joined) || output.meets(queries) { return Err("fast row dots: output overlaps input"); }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_norm(rows: usize, dim: usize, rounded: bool, eps: f32, input: Span, weight: Option<Span>, output: Span) -> Result<(), &'static str> {
    if !norm_admitted(rows, dim, rounded, NORM_THREADS) { return Err("fast norm: shape outside envelope"); }
    if !(eps > 0.0 && eps.is_finite()) { return Err("fast norm: epsilon must be positive finite"); }
    if !rounded && weight.is_some() { return Err("fast norm: plain norm takes no weight"); }
    let n_w = rows.checked_mul(dim);
    if !input.valid(n_w.and_then(f32_bytes)) || !output.valid(n_w.and_then(f32_bytes)) {
        return Err("fast norm: operand size or alignment mismatch");
    }
    if let Some(w) = weight {
        if !w.valid(f32_bytes(dim)) || output.meets(w) { return Err("fast norm: bad weight span"); }
    }
    if output.meets(input) { return Err("fast norm: output overlaps input"); }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_capped_gate_caps(gate_cap: f32, up_cap: f32, inv_gate: f32, inv_up: f32) -> Result<(), &'static str> {
    if gate_cap > 0.0 && up_cap > 0.0 && inv_gate.is_finite() && inv_up.is_finite() {
        Ok(())
    } else {
        Err("fast capped gate: bad caps")
    }
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_capped_gate(picks: usize, dim: usize, gate_cap: f32, up_cap: f32, inv_gate: f32, inv_up: f32, input: Span, output: Span) -> Result<(), &'static str> {
    if !capped_gate_admitted(picks, dim, CAPPED_GATE_THREADS) { return Err("fast capped gate: shape outside envelope"); }
    check_capped_gate_caps(gate_cap, up_cap, inv_gate, inv_up)?;
    let i_w = picks.checked_mul(2).and_then(|n| n.checked_mul(dim));
    let o_w = picks.checked_mul(dim);
    if !input.valid(i_w.and_then(f32_bytes)) || !output.valid(o_w.and_then(f32_bytes)) || output.meets(input) {
        return Err("fast capped gate: operand mismatch or overlap");
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_router_pick(rows: usize, experts: usize, picks: usize, bound: f32, logits: Span, bias: Span, output: Span) -> Result<(), &'static str> {
    if !router_pick_admitted(rows, experts, picks, ROUTER_PICK_THREADS) { return Err("fast router pick: shape outside envelope"); }
    if !(bound > 0.0 && bound.is_finite()) { return Err("fast router pick: bad bound"); }
    let l_w = rows.checked_mul(experts);
    let o_w = rows.checked_mul(2 * picks);
    if !logits.valid(l_w.and_then(f32_bytes)) || !bias.valid(f32_bytes(experts)) || !output.valid(o_w.and_then(f32_bytes)) {
        return Err("fast router pick: operand size or alignment mismatch");
    }
    if output.meets(logits) || output.meets(bias) { return Err("fast router pick: output overlaps input"); }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, bound, operands and compacted dispatch output")]
pub fn check_router_pick_compact(
    rows: usize, experts: usize, picks: usize, bound: f32,
    logits: Span, bias: Span, output: Span, compact: Span,
) -> Result<(), &'static str> {
    if !router_pick_compact_admitted(rows, experts, picks, ROUTER_PICK_COMPACT_THREADS) {
        return Err("fast router pick compact: shape outside envelope");
    }
    if !(bound > 0.0 && bound.is_finite()) { return Err("fast router pick compact: bad bound"); }
    let l_w = rows.checked_mul(experts);
    let o_w = rows.checked_mul(2 * picks);
    let c_w = rows.checked_mul(picks).and_then(|n| n.checked_mul(ROUTER_PICK_COMPACT_ENTRY_WORDS));
    if !logits.valid(l_w.and_then(f32_bytes)) || !bias.valid(f32_bytes(experts))
        || !output.valid(o_w.and_then(f32_bytes))
        || !compact.valid(c_w.and_then(f32_bytes)) || compact.address % 16 != 0
    {
        return Err("fast router pick compact: operand size or alignment mismatch");
    }
    if output.meets(logits) || output.meets(bias) || compact.meets(logits) || compact.meets(bias) || compact.meets(output) {
        return Err("fast router pick compact: output overlaps operand");
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape and four spans")]
pub fn check_short_conv_win(rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32, window: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span, ring: Option<(Span, usize)>, output: Span, every: Option<(Span, usize)>) -> Result<(), &'static str> {
    check_short_conv_form(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, ring, output, every, None)
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, form and operand spans")]
pub(crate) fn check_short_conv_form(rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
    window: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span, ring: Option<(Span, usize)>,
    output: Span, every: Option<(Span, usize)>, kv: Option<KvWrite>,
) -> Result<(), &'static str> {

    let width = heads.checked_mul(dim).ok_or("fast short conv: width overflow")?;
    if !stream_ops_admitted(STREAM_OP_SHORT_CONV_WIN, rows, width, taps, STREAM_OPS_THREADS) { return Err("fast short conv: shape outside envelope"); }
    if !(bound > 0.0 && bound.is_finite()) || form >= 8 || (form & 4 != 0 && form & 2 == 0) { return Err("fast short conv: bad form or bound"); }
    let history = if ring.is_some() { 0 } else { taps - 1 };
    let w_w = rows.checked_add(history).and_then(|n| n.checked_mul(width));
    let f_w = width.checked_mul(taps);
    let o_w = rows.checked_mul(width);
    let out_words = match kv {
        Some(kv) => {
            if dim != CONV_KV_DIM || ring.is_none() || every.is_some() || addend.is_some()
                || rows < taps || rows < 2 || kv.plane_rows < rows || kv.first >= kv.plane_rows
                || !(kv.param.is_finite() && kv.param > 0.0) {
                return Err("fast short conv: invalid KV finish");
            }
            kv.plane_rows.checked_mul(width / 2)
        }
        None => o_w,
    };
    if !window.valid(w_w.and_then(f32_bytes)) || !fixed_taps.valid(f_w.and_then(f32_bytes)) || !output.valid(out_words.and_then(f32_bytes)) {
        return Err("fast short conv: operand size or alignment mismatch");
    }
    if (form & 1 != 0) != token_taps.is_some() || (form & 2 != 0) != addend.is_some() {
        return Err("fast short conv: optional operand mismatch");
    }
    if let Some(tt) = token_taps {
        if !tt.valid(rows.checked_mul(heads).and_then(|n| n.checked_mul(taps)).and_then(f32_bytes)) || output.meets(tt) {
            return Err("fast short conv: bad token_taps span");
        }
    }
    if let Some(ad) = addend {
        if !ad.valid(o_w.and_then(f32_bytes)) || output.meets(ad) { return Err("fast short conv: bad addend span"); }
    }
    if output.meets(window) || output.meets(fixed_taps) { return Err("fast short conv: output overlaps input"); }
    if let Some((ring, first)) = ring {
        let ring_chats = if let Some((_, _, chats)) = short_conv_multi_decode(rows, taps, first) {
            if kv.is_some() || every.is_none() { return Err("fast short conv: multi-chat ring requires kept output without kv"); }
            chats
        } else {
            if first >= taps { return Err("fast short conv: bad ring span or slot"); }
            1
        };
        if !ring.valid(ring_chats.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes)) || output.meets(ring) {
            return Err("fast short conv: bad ring span or slot");
        }
    }
    // The rings after each row of a step that keeps each row: written, so disjoint from every other operand.
    // Supported both with a joined window and with a separate ring + inputs (same first slot or multi-chat descriptor).
    if let Some((every, first)) = every {
        let spans = [Some(window), token_taps, addend, Some(fixed_taps), ring.map(|(s, _)| s), Some(output)];
        let multi = short_conv_multi_decode(rows, taps, first).is_some();
        if ring.is_some_and(|(_, r_first)| r_first != first) || (!multi && first >= taps) || (multi && ring.is_none())
            || !every.valid(rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes))
            || spans.iter().flatten().any(|s| every.meets(*s))
        {
            return Err("fast short conv: bad kept rings span or slot");
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "paired key/value shapes, scalars, and device operand spans")]
pub fn check_kv_finish_pair(
    rows: usize, kv_heads: usize, taps: usize, first_slot: usize,
    key_bound: f32, val_eps: f32, key_scale: f32, val_scale: f32,
    key_in: Span, key_ring: Span, key_fixed: Span, key_made: Option<Span>, key_packed: Span, key_every: Option<Span>,
    val_in: Span, val_ring: Span, val_fixed: Span, val_made: Option<Span>, val_packed: Span, val_every: Option<Span>,
) -> Result<(), &'static str> {
    if !kv_finish_admitted(rows, kv_heads, taps, KV_FINISH_THREADS) || first_slot >= taps {
        return Err("fast kv finish: shape or slot outside envelope");
    }
    if !(key_bound > 0.0 && key_bound.is_finite() && val_eps > 0.0 && val_eps.is_finite()
        && key_scale > 0.0 && key_scale.is_finite() && val_scale > 0.0 && val_scale.is_finite())
    {
        return Err("fast kv finish: non-finite or non-positive scalar");
    }
    let width = kv_heads.checked_mul(KV_FINISH_HEAD_DIM).ok_or("fast kv finish: width overflow")?;
    let in_b = rows.checked_mul(width).and_then(f32_bytes);
    let ring_b = taps.checked_mul(width).and_then(f32_bytes);
    let fixed_b = width.checked_mul(taps).and_then(f32_bytes);
    let made_b = rows.checked_mul(kv_heads).and_then(|n| n.checked_mul(taps)).and_then(f32_bytes);
    let pack_b = rows.checked_mul(width / 2).and_then(f32_bytes);
    let every_b = rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes);
    for (inp, rng, fix, mad, pck, ev) in [
        (key_in, key_ring, key_fixed, key_made, key_packed, key_every),
        (val_in, val_ring, val_fixed, val_made, val_packed, val_every),
    ] {
        if !inp.valid(in_b) || !rng.valid(ring_b) || !fix.valid(fixed_b) || !pck.valid(pack_b) {
            return Err("fast kv finish: operand size or alignment mismatch");
        }
        if let Some(m) = mad { if !m.valid(made_b) { return Err("fast kv finish: bad made taps span"); } }
        if let Some(e) = ev { if !e.valid(every_b) { return Err("fast kv finish: bad every span"); } }
    }
    let reads = [Some(key_in), Some(key_ring), Some(key_fixed), key_made, Some(val_in), Some(val_ring), Some(val_fixed), val_made];
    let writes = [Some(key_packed), key_every, Some(val_packed), val_every];
    for (i, w) in writes.iter().enumerate() {
        let Some(ws) = *w else { continue; };
        if reads.iter().flatten().any(|r| ws.meets(*r)) {
            return Err("fast kv finish: output overlaps input");
        }
        for ow in writes.iter().skip(i + 1).flatten() {
            if ws.meets(*ow) { return Err("fast kv finish: outputs overlap"); }
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, strided route layout, ring slot, and device operand spans")]
pub fn check_route_fold_conv(
    rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
    taps: usize, first_slot: usize, bound: f32,
    route: Span, downs: Span, residual: Span, fixed: Span, ring: Span, output: Span, every: Option<Span>,
) -> Result<(), &'static str> {
    check_route_fold_conv_form(rows, picks, dim, weight_stride, weight_offset, taps, first_slot, bound, route, downs, residual, fixed, ring, output, every, false)
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, strided route layout, ring slot, device operand spans and down form")]
pub fn check_route_fold_conv_form(
    rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize,
    taps: usize, first_slot: usize, bound: f32,
    route: Span, downs: Span, residual: Span, fixed: Span, ring: Span, output: Span, every: Option<Span>, half: bool,
) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_ROUTE_FOLD_CONV, rows, dim, picks, ROUTE_FOLD_CONV_THREADS)
        || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || first_slot >= taps
        || weight_offset.checked_add(picks).is_none_or(|end| end > weight_stride)
        || !(bound > 0.0 && bound.is_finite())
        || (half && !dim.is_multiple_of(2))
    {
        return Err("fast route fold conv: shape or parameter outside envelope");
    }
    let r_b = rows.checked_mul(weight_stride).and_then(f32_bytes);
    let d_w = rows.checked_mul(picks).and_then(|n| n.checked_mul(dim));
    let d_b = if half { d_w.and_then(|n| (n as u64).checked_mul(2)) } else { d_w.and_then(f32_bytes) };
    let o_b = rows.checked_mul(dim).and_then(f32_bytes);
    let f_b = dim.checked_mul(taps).and_then(f32_bytes);
    let ring_b = taps.checked_mul(dim).and_then(f32_bytes);
    if !route.valid(r_b) || !downs.valid(d_b) || !residual.valid(o_b) || !fixed.valid(f_b)
        || !ring.valid(ring_b) || !output.valid(o_b)
    {
        return Err("fast route fold conv: operand size or alignment mismatch");
    }
    let reads = [route, downs, residual, fixed, ring];
    if reads.iter().any(|r| output.meets(*r)) { return Err("fast route fold conv: output overlaps input"); }
    if let Some(ev) = every {
        let e_b = rows.checked_mul(taps).and_then(|n| n.checked_mul(dim)).and_then(f32_bytes);
        if !ev.valid(e_b) || ev.meets(output) || reads.iter().any(|r| ev.meets(*r)) {
            return Err("fast route fold conv: bad every span");
        }
    }
    Ok(())
}

/// Check prefill fused `RouteFold (FOLD_HALF)` + `AfterExperts ShortConv` + `next_ring`.
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, ring parameters, and eight device operand spans")]
pub fn check_route_fold_conv_prefill(
    rows: usize, picks: usize, dim: usize, taps: usize, first_slot: usize, bound: f32,
    weights: Span, downs_half: Span, residual: Span, fixed: Span, ring: Span, output: Span, next_ring: Span,
) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_ROUTE_FOLD_CONV_PREFILL, rows, dim, picks, STREAM_OPS_THREADS)
        || !(2..=ROUTE_FOLD_CONV_TAPS_MAX).contains(&taps) || rows < taps || first_slot >= taps
        || !(bound > 0.0 && bound.is_finite())
    {
        return Err("fast route fold conv prefill: shape or parameter outside envelope");
    }
    let w_b = rows.checked_mul(picks).and_then(f32_bytes);
    let d_b = rows.checked_mul(picks).and_then(|n| n.checked_mul(dim)).and_then(|n| (n as u64).checked_mul(2));
    let o_b = rows.checked_mul(dim).and_then(f32_bytes);
    let f_b = dim.checked_mul(taps).and_then(f32_bytes);
    let r_b = taps.checked_mul(dim).and_then(f32_bytes);
    if !weights.valid(w_b) || !downs_half.valid(d_b) || !residual.valid(o_b) || !fixed.valid(f_b)
        || !ring.valid(r_b) || !output.valid(o_b) || !next_ring.valid(r_b)
    {
        return Err("fast route fold conv prefill: operand size or alignment mismatch");
    }
    let reads = [weights, downs_half, residual, fixed, ring];
    if reads.iter().any(|r| output.meets(*r) || next_ring.meets(*r)) || output.meets(next_ring) {
        return Err("fast route fold conv prefill: output overlaps input or next_ring");
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_route_fold(rows: usize, picks: usize, dim: usize, weights: Span, downs: Span, output: Span) -> Result<(), &'static str> {
    check_route_fold_strided(rows, picks, dim, picks, 0, weights, downs, output, false)
}

/// `check_route_fold` with strided route weights (`weight_stride` words a row, from `weight_offset`) and the down form:
/// `half` (`FOLD_HALF`) needs an even `dim`, and `downs` then holds `rows * picks * dim` BF16 values (2 bytes each).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, strided route offsets, operand spans and down form")]
pub fn check_route_fold_strided(rows: usize, picks: usize, dim: usize, weight_stride: usize, weight_offset: usize, weights: Span, downs: Span, output: Span, half: bool) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_ROUTE_FOLD, rows, dim, picks, STREAM_OPS_THREADS) { return Err("fast route fold: shape outside envelope"); }
    if weight_offset.checked_add(picks).is_none_or(|end| end > weight_stride) { return Err("fast route fold: bad weight stride or offset"); }
    if half && dim % 2 != 0 { return Err("fast route fold: half downs need an even width"); }
    let w_w = rows.checked_mul(weight_stride);
    let d_w = rows.checked_mul(picks).and_then(|n| n.checked_mul(dim));
    let o_w = rows.checked_mul(dim);
    let d_bytes = if half { d_w.and_then(|n| (n as u64).checked_mul(2)) } else { d_w.and_then(f32_bytes) };
    if !weights.valid(w_w.and_then(f32_bytes)) || !downs.valid(d_bytes) || !output.valid(o_w.and_then(f32_bytes)) {
        return Err("fast route fold: operand size or alignment mismatch");
    }
    if output.meets(weights) || output.meets(downs) { return Err("fast route fold: output overlaps input"); }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_gate_blend(rows: usize, dim: usize, gates: Span, kept: Span, msg: Span, output: Span) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_GATE_BLEND, rows, dim, 0, STREAM_OPS_THREADS) { return Err("fast gate blend: shape outside envelope"); }
    let g_w = rows.checked_mul(2).and_then(|n| n.checked_mul(dim));
    let v_w = rows.checked_mul(dim);
    if !gates.valid(g_w.and_then(f32_bytes)) || !kept.valid(v_w.and_then(f32_bytes)) || !msg.valid(v_w.and_then(f32_bytes)) || !output.valid(v_w.and_then(f32_bytes)) {
        return Err("fast gate blend: operand size or alignment mismatch");
    }
    if output.meets(gates) || output.meets(kept) || output.meets(msg) { return Err("fast gate blend: output overlaps input"); }
    Ok(())
}

/// Check 1..=3 batched `(gates, kept, msg, optional weight, output)` tuples for fused `GateBlend + RoundedNorm`.
#[cfg(not(target_arch = "nvptx64"))]
pub fn check_gate_blend_norm(
    rows: usize, dim: usize, eps: f32,
    entries: &[(Span, Span, Span, Option<Span>, Span)],
) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_GATE_BLEND_NORM, rows, dim, entries.len(), gate_blend_norm_threads(dim))
        || !(eps > 0.0 && eps.is_finite())
    {
        return Err("fast gate blend norm: shape or epsilon outside envelope");
    }
    let g_b = rows.checked_mul(2).and_then(|n| n.checked_mul(dim)).and_then(f32_bytes);
    let v_b = rows.checked_mul(dim).and_then(f32_bytes);
    let w_b = f32_bytes(dim);
    for &(gates, kept, msg, weight, output) in entries {
        if !gates.valid(g_b) || !kept.valid(v_b) || !msg.valid(v_b) || !output.valid(v_b) {
            return Err("fast gate blend norm: operand size or alignment mismatch");
        }
        if let Some(wt) = weight {
            if !wt.valid(w_b) { return Err("fast gate blend norm: bad weight span"); }
        }
    }
    for (i, &(_, _, _, _, out)) in entries.iter().enumerate() {
        for &(gates, kept, msg, weight, _) in entries {
            if out.meets(gates) || out.meets(kept) || out.meets(msg) || weight.is_some_and(|wt| out.meets(wt)) {
                return Err("fast gate blend norm: output overlaps input");
            }
        }
        for &(_, _, _, _, other_out) in entries.iter().skip(i + 1) {
            if out.meets(other_out) { return Err("fast gate blend norm: outputs overlap"); }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn span(address: u64, words: usize) -> Span { Span { address, bytes: words as u64 * 4 } }
    /// The blocks of a stream op cover every channel of every row once (the kernel's map: block b is part b % parts of
    /// row b / parts, channels part * threads + tid, then every parts * threads).
    #[test]
    fn the_parts_of_a_stream_row_cover_each_channel_once() {
        assert_eq!((stream_ops_parts(2304), stream_ops_parts(256), stream_ops_parts(257), stream_ops_parts(1), stream_ops_blocks(3, 2304)), (9, 1, 2, 1, 27));
        for dim in [1usize, 31, 255, 256, 257, 2304, 4608, FAST_DIM_MAX] {
            let parts = stream_ops_parts(dim);
            let mut seen = vec![0u8; dim];
            for part in 0..parts { for tid in 0..STREAM_OPS_THREADS { let mut ch = part * STREAM_OPS_THREADS + tid; while ch < dim { seen[ch] += 1; ch += parts * STREAM_OPS_THREADS; } } }
            assert!(seen.iter().all(|&n| n == 1), "dim {dim}");
        }
    }
    /// The one-row mix: the decode shapes (1 or 3 queries, 2 to 8 kept rows, 2,304 wide) pass; a short or unaligned
    /// operand, a count of rows the fold does not take, a wider row than the stage, or an output that meets an input
    /// or the other output is refused.
    #[test]
    fn the_one_row_mix_takes_only_its_checked_operands() {
        let d = 2304;
        let list = |n: usize| (0..n).map(|e| span(1 << 24 | (e as u64) << 16, d)).collect::<Vec<_>>();
        let ok = |asked: usize, kept: usize, list: &[Span], stream: Span, normed: Span, output: Span|
            check_mix_one(asked, kept, d, 0.02, 1e-6, span(1 << 20, asked * d), list, stream, normed, output);
        for (asked, kept) in [(1, 2), (3, 5), (1, MIX_ONE_KEPT_MAX)] {
            assert_eq!(ok(asked, kept, &list(kept - 1), span(1 << 28, d), span(1 << 30, d), span(1 << 32, asked * d)), Ok(()), "{asked} {kept}");
        }
        assert!(ok(1, 1, &[], span(1 << 28, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "one kept row: no fold");
        assert!(ok(1, 9, &list(8), span(1 << 28, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "9 kept rows");
        assert!(ok(1, 3, &[span(1 << 24, d), span(1 << 25 | 4, d)], span(1 << 28, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "unaligned list row");
        assert!(ok(1, 3, &list(1), span(1 << 28, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "list and kept differ");
        assert!(ok(1, 2, &list(1), span(1 << 28 | 4, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "unaligned stream");
        assert!(ok(1, 2, &list(1), span(1 << 28, d - 1), span(1 << 30, d), span(1 << 32, d)).is_err(), "short stream");
        assert!(ok(1, 2, &list(1), span(1 << 28, d), span(1 << 30, d), span(1 << 30, d)).is_err(), "outputs meet");
        assert!(ok(1, 2, &list(1), span(1 << 28, d), span(1 << 28, d), span(1 << 32, d)).is_err(), "normed meets the stream");
        assert!(ok(1, 2, &list(1), span(1 << 28, d), span(1 << 30, d), span(1 << 24, d)).is_err(), "output meets a list row");
        assert!(check_mix_one(1, 2, MIX_ONE_DIM_MAX + 32, 0.02, 1e-6, span(1 << 20, MIX_ONE_DIM_MAX + 32), &[span(1 << 24, MIX_ONE_DIM_MAX + 32)],
            span(1 << 28, MIX_ONE_DIM_MAX + 32), span(1 << 30, MIX_ONE_DIM_MAX + 32), span(1 << 32, MIX_ONE_DIM_MAX + 32)).is_err(), "wider than the stage");
        assert!(check_mix_one(1, 2, d, 0.0, 1e-6, span(1 << 20, d), &list(1), span(1 << 28, d), span(1 << 30, d), span(1 << 32, d)).is_err(), "scale");
    }
    #[test]
    fn fast_ops_checks_admit_inkling_shapes_and_refuse_bad_ones() {
        assert_eq!(check_mix_blend(4096, 48, 3, 2304, 0.02, span(1 << 20, 4096 * 48 * 3), span(1 << 28, 3 * 4096 * 2304), span(1 << 32, 4096 * 48 * 2304)), Ok(()));
        assert!(check_mix_blend(4096, 48, 65, 2304, 0.02, span(1 << 20, 4096 * 48 * 65), span(1 << 28, 65 * 4096 * 2304), span(1 << 32, 4096 * 48 * 2304)).is_err());
        assert_eq!(check_row_dots(4096, 48, 3, 2304, span(1 << 28, 3 * 4096 * 2304), span(1 << 20, 48 * 2304), span(1 << 32, 4096 * 48 * 3)), Ok(()));
        assert_eq!(check_norm(4096, 2304, true, 1e-6, span(1 << 20, 4096 * 2304), Some(span(1 << 28, 2304)), span(1 << 30, 4096 * 2304)), Ok(()));
        assert_eq!(check_norm(32768, 128, true, 1e-6, span(1 << 20, 32768 * 128), None, span(1 << 30, 32768 * 128)), Ok(()));
        assert_eq!(check_norm(4096, 2304, false, 1e-6, span(1 << 20, 4096 * 2304), None, span(1 << 30, 4096 * 2304)), Ok(()));
        assert!(check_norm(4096, 2304, false, 1e-6, span(1 << 20, 4096 * 2304), Some(span(1 << 28, 2304)), span(1 << 30, 4096 * 2304)).is_err());
        assert_eq!(check_capped_gate(32768, 1152, 4.0, 25.0, 0.25, 0.04, span(1 << 20, 32768 * 2304), span(1 << 30, 32768 * 1152)), Ok(()));
        assert!(check_capped_gate_caps(4.0, 25.0, f32::INFINITY, 0.04).is_err());
        assert!(check_capped_gate_caps(4.0, 0.0, 0.25, 0.04).is_err());
        assert_eq!(check_router_pick(4096, 128, 8, 8.0, span(1 << 20, 4096 * 128), span(1 << 28, 128), span(1 << 30, 4096 * 16)), Ok(()));
        assert_eq!(check_router_pick_compact(4, 128, 8, 8.0, span(1 << 20, 4 * 128), span(1 << 28, 128), span(1 << 30, 4 * 16), span(1 << 31, 4 * 8 * ROUTER_PICK_COMPACT_ENTRY_WORDS)), Ok(()));
        assert!(check_router_pick_compact(9, 128, 8, 8.0, span(1 << 20, 9 * 128), span(1 << 28, 128), span(1 << 30, 9 * 16), span(1 << 31, 9 * 8 * ROUTER_PICK_COMPACT_ENTRY_WORDS)).is_err());
        assert!(check_router_pick_compact(4, 128, 8, 8.0, span(1 << 20, 4 * 128), span(1 << 28, 128), span(1 << 30, 4 * 16), span((1 << 31) | 4, 4 * 8 * ROUTER_PICK_COMPACT_ENTRY_WORDS)).is_err());
        assert!(check_router_pick_compact(4, 128, 8, 8.0, span(1 << 20, 4 * 128), span(1 << 28, 128), span(1 << 30, 4 * 16), span(1 << 30, 4 * 8 * ROUTER_PICK_COMPACT_ENTRY_WORDS)).is_err());
        assert_eq!(check_route_fold_conv_form(4, 8, 2304, 16, 8, 4, 0, 0.5, span(1 << 20, 4 * 16), span(1 << 22, 4 * 8 * 2304 / 2), span(1 << 24, 4 * 2304), span(1 << 26, 4 * 2304), span(1 << 28, 4 * 2304), span(1 << 30, 4 * 2304), Some(span(1 << 31, 4 * 4 * 2304)), true), Ok(()));
        assert!(check_route_fold_conv_form(4, 8, 2303, 16, 8, 4, 0, 0.5, span(1 << 20, 4 * 16), span(1 << 22, 4 * 8 * 2303 / 2), span(1 << 24, 4 * 2303), span(1 << 26, 4 * 2303), span(1 << 28, 4 * 2303), span(1 << 30, 4 * 2303), None, true).is_err());
        assert!(stream_ops_admitted(STREAM_OP_RING_CONV_ONE, 1, 2304, 4, ROUTE_FOLD_CONV_THREADS));
        assert!(!stream_ops_admitted(STREAM_OP_RING_CONV_ONE, 2, 2304, 4, ROUTE_FOLD_CONV_THREADS));
        assert!(!stream_ops_admitted(STREAM_OP_RING_CONV_ONE, 1, 2304, 5, ROUTE_FOLD_CONV_THREADS));
        assert_eq!(check_short_conv_win(4096, 1, 2304, 4, 6, 0.2, span(1 << 20, 4099 * 2304), None, Some(span(1 << 28, 4096 * 2304)), span(1 << 29, 4 * 2304), None, span(1 << 30, 4096 * 2304), None), Ok(()));
        let packed = short_conv_multi_encode(16, 3, 0x5555, 0x0000_2102).unwrap();
        assert_eq!(short_conv_multi_decode(16, 3, packed), Some((0x5555, 0x0000_2102, 8)));
        assert_eq!(check_short_conv_win(16, 1, 2304, 3, 6, 0.2, span(1 << 20, 16 * 2304), None, Some(span(1 << 22, 16 * 2304)), span(1 << 24, 3 * 2304),
            Some((span(1 << 26, 8 * 3 * 2304), packed)), span(1 << 28, 16 * 2304), Some((span(1 << 30, 16 * 3 * 2304), packed))), Ok(()));
        assert!(short_conv_multi_encode(16, 3, 0x5555, 0x0000_3000).is_none());
        assert!(short_conv_multi_encode(17, 3, 0x0003, 0).is_none());
        assert_eq!(check_route_fold(4096, 8, 2304, span(1 << 20, 4096 * 8), span(1 << 24, 4096 * 8 * 2304), span(1 << 30, 4096 * 2304)), Ok(()));
        // Half downs: two BF16 values a word; the FP32 size, or an odd width, is refused.
        assert_eq!(check_route_fold_strided(4096, 8, 2304, 8, 0, span(1 << 20, 4096 * 8), span(1 << 24, 4096 * 8 * 2304 / 2), span(1 << 30, 4096 * 2304), true), Ok(()));
        assert!(check_route_fold_strided(4096, 8, 2304, 8, 0, span(1 << 20, 4096 * 8), span(1 << 24, 4096 * 8 * 2304), span(1 << 30, 4096 * 2304), true).is_err());
        assert!(check_route_fold_strided(2, 8, 2303, 8, 0, span(1 << 20, 2 * 8), span(1 << 24, 8 * 2303), span(1 << 30, 2 * 2303), true).is_err());
        assert_eq!(check_gate_blend(4096, 2304, span(1 << 20, 4096 * 4608), span(1 << 28, 4096 * 2304), span(1 << 29, 4096 * 2304), span(1 << 30, 4096 * 2304)), Ok(()));
        assert_eq!(check_gate_blend_norm(4, 2304, 1e-6, &[
            (span(1 << 20, 4 * 4608), span(1 << 22, 4 * 2304), span(1 << 23, 4 * 2304), Some(span(1 << 24, 2304)), span(1 << 25, 4 * 2304)),
            (span(1 << 26, 4 * 4608), span(1 << 22, 4 * 2304), span(1 << 27, 4 * 2304), Some(span(1 << 24, 2304)), span(1 << 28, 4 * 2304)),
            (span(1 << 29, 4 * 4608), span(1 << 22, 4 * 2304), span(1 << 30, 4 * 2304), None, span(1 << 31, 4 * 2304)),
        ]), Ok(()));
        assert!(check_gate_blend_norm(4, 2304, 1e-6, &[
            (span(1 << 20, 4 * 4608), span(1 << 22, 4 * 2304), span(1 << 23, 4 * 2304), None, span(1 << 25, 4 * 2304)),
            (span(1 << 26, 4 * 4608), span(1 << 22, 4 * 2304), span(1 << 27, 4 * 2304), None, span(1 << 25, 4 * 2304)),
        ]).is_err(), "overlapping gate_blend_norm outputs");
        assert_eq!(check_mix_rows_normed(4, 3, 2304, 0.02, 1e-6, Some(1e-6), span(1 << 20, 3 * 2304), &[span(1 << 24, 4 * 2304)],
            span(1 << 26, 4 * 2304), None, None, Some(span(1 << 28, 12 * 2304)), Some(span(1 << 30, 3 * 2304)), true), Ok(()));
        assert!(check_mix_rows_normed(4, 1, 2304, 0.02, 1e-6, None, span(1 << 20, 2304), &[span(1 << 24, 4 * 2304)],
            span(1 << 26, 4 * 2304), None, None, None, None, false).is_err(), "missing both outputs");
    }
}
