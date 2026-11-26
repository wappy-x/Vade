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

/// Multi-chat kept-ring descriptor tag packed into `first` (`word 13` and `word 15`) for `STREAM_OP_SHORT_CONV_WIN`
/// (`rows <= 256`, `chats <= 64`, `taps in 2..=4`). Low bits hold `chats`; payload words `16, 17, 84, 85` hold
/// the 256-bit `start_mask[0..4]`, words `18..=19` hold 2-bit per-chat initial ring slots, and words `20..=83` hold
/// optional per-chat ring device pointers (when word 12 is 0).
pub const SHORT_CONV_MULTI_TAG: usize = 1usize << 62;
pub const SHORT_CONV_MULTI_ROWS_MAX: usize = 256;
pub const SHORT_CONV_MULTI_CHATS_MAX: usize = 64;
pub const SHORT_CONV_MULTI_MASK_0: usize = 16;
pub const SHORT_CONV_MULTI_MASK_1: usize = 17;
pub const SHORT_CONV_MULTI_SLOTS_LO: usize = 18;
pub const SHORT_CONV_MULTI_SLOTS_HI: usize = 19;
/// Optional per-chat ring device pointers in `STREAM_OP_SHORT_CONV_WIN` multi-chat mode when word 12 is 0
/// (`chats <= SHORT_CONV_MULTI_CHATS_MAX`, eliminating `compute.join` ring copies across 2..=64 chats).
pub const SHORT_CONV_MULTI_RINGS_AT: usize = 20;
pub const SHORT_CONV_MULTI_MASK_2: usize = 84;
pub const SHORT_CONV_MULTI_MASK_3: usize = 85;
const _: () = assert!(usize::BITS >= 64 && SHORT_CONV_MULTI_RINGS_AT + SHORT_CONV_MULTI_CHATS_MAX <= SHORT_CONV_MULTI_MASK_2 && 8 * (SHORT_CONV_MULTI_MASK_3 + 1) <= 1024);

/// Multi-chat kept-ring descriptor for `STREAM_OP_SHORT_CONV_WIN` (`2..=64` chats, `2..=256` rows, `taps in 2..=4`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShortConvMulti {
    pub start_mask: [u64; 4],
    pub slots_lo: u64,
    pub slots_hi: u64,
    pub chats: usize,
}

/// Ring slot descriptor for `short_conv_keep`: single chat (`One`) or packed multi-chat (`Multi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShortConvRing {
    One(usize),
    Multi(ShortConvMulti),
}

/// Device-side O(1) structural check for `ShortConvMulti` masks and chat count (`2 <= chats <= rows <= 256`, `taps in 2..=4`).
#[inline(always)]
pub const fn short_conv_multi_valid(rows: usize, taps: usize, chats: usize, m: [u64; 4]) -> bool {
    if rows < 2 || rows > SHORT_CONV_MULTI_ROWS_MAX || taps < 2 || taps > 4 { return false; }
    if chats < 2 || chats > SHORT_CONV_MULTI_CHATS_MAX || chats > rows { return false; }
    if (m[0] & 1) == 0 { return false; }
    let mut w = 0usize;
    while w < 4 {
        let base = w * 64;
        if rows <= base {
            if m[w] != 0 { return false; }
        } else if rows < base + 64 && (m[w] >> (rows - base)) != 0 {
            return false;
        }
        w += 1;
    }
    (m[0].count_ones() + m[1].count_ones() + m[2].count_ones() + m[3].count_ones()) as usize == chats
}

impl ShortConvMulti {
    /// Validate and construct a multi-chat kept-ring descriptor (`2..=64` chats, `2..=256` rows, `taps in 2..=4`).
    #[inline(always)]
    pub const fn encode(rows: usize, taps: usize, m: [u64; 4], s_lo: u64, s_hi: u64) -> Option<Self> {
        let chats = (m[0].count_ones() + m[1].count_ones() + m[2].count_ones() + m[3].count_ones()) as usize;
        if !short_conv_multi_valid(rows, taps, chats, m) { return None; }
        if chats < 32 {
            if (s_lo >> (2 * chats)) != 0 || s_hi != 0 { return None; }
        } else if chats == 32 {
            if s_hi != 0 { return None; }
        } else if chats < 64 && (s_hi >> (2 * (chats - 32))) != 0 {
            return None;
        }
        let mut c = 0usize;
        while c < chats {
            let sw = if c < 32 { s_lo } else { s_hi };
            if ((sw >> (2 * (c & 31))) & 3) as usize >= taps { return None; }
            c += 1;
        }
        Some(Self { start_mask: m, slots_lo: s_lo, slots_hi: s_hi, chats })
    }
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
/// Fused `Softcap` (`cap(rn(x), eps, bound)`) + greedy argmax (`[rows][3]`: `best, id, finite`),
/// eliminating the `[rows][vocab]` capped-logits DRAM write/read on greedy decode and verify steps.
pub const STREAM_OP_SOFTCAP_ARGMAX: u64 = 7;
pub const SOFTCAP_ARGMAX_COLS_MAX: usize = 1 << 24;
pub const SOFTCAP_ARGMAX_PARTS_MAX: usize = 256;
/// Row-split factor for `STREAM_OP_SOFTCAP_ARGMAX`: 1 block/row once `rows >= 48` (single launch, no partials),
/// or enough parts to saturate 48 SMs when `rows < 48` (followed by a tiny 1-block/row finish launch).
#[inline(always)]
pub const fn softcap_argmax_parts(rows: usize, cols: usize) -> usize {
    if rows == 0 || cols == 0 || rows >= 48 { return 1; }
    let target = 192 / rows;
    let by_cols = cols.div_ceil(4096);
    let p = if target < by_cols { target } else { by_cols };
    if p < 1 { 1 } else if p > SOFTCAP_ARGMAX_PARTS_MAX { SOFTCAP_ARGMAX_PARTS_MAX } else { p }
}
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
/// Extra sub-modes dispatched inside `FAST_NORM` (kind 109, word 5).
pub const NORM_MODE_KV_DIRECT: u64 = 3;
pub const NORM_MODE_SHORT_CONV_DIRECT: u64 = 4;
/// Maximum independent chats per `NORM_MODE_KV_DIRECT` / `NORM_MODE_SHORT_CONV_DIRECT` launch.
pub const DIRECT_CHATS_MAX: usize = 16;
/// Maximum convolution taps for `NORM_MODE_KV_DIRECT` / `NORM_MODE_SHORT_CONV_DIRECT`.
pub const DIRECT_TAPS_MAX: usize = 4;

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
pub const fn kv_finish_direct_admitted(rows: usize, kv_heads: usize, taps: usize, threads: usize) -> bool {
    threads == NORM_THREADS
        && rows >= 2 && rows <= DIRECT_CHATS_MAX
        && kv_heads >= 1 && kv_heads <= 64
        && taps >= 2 && taps <= DIRECT_TAPS_MAX
}

#[inline(always)]
pub const fn short_conv_direct_admitted(rows: usize, heads: usize, dim: usize, taps: usize, form: u32, threads: usize) -> bool {
    let width = heads.saturating_mul(dim);
    threads == NORM_THREADS
        && rows >= 2 && rows <= DIRECT_CHATS_MAX
        && heads >= 1 && dim >= 1 && width >= 1 && width <= FAST_DIM_MAX
        && taps >= 2 && taps <= DIRECT_TAPS_MAX
        && form < 8 && (form & 4 == 0 || form & 2 != 0)
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
    if subop == STREAM_OP_SOFTCAP_ARGMAX {
        return rows >= 1 && rows <= FAST_ROWS_MAX
            && dim >= 1 && dim <= SOFTCAP_ARGMAX_COLS_MAX
            && aux >= 1 && aux <= SOFTCAP_ARGMAX_PARTS_MAX
            && threads == STREAM_OPS_THREADS;
    }
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
    check_short_conv_form(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, ring, output, every, None, None)
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, multi-chat descriptor and operand spans")]
pub fn check_short_conv_multi(rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32, window: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span, ring: Span, output: Span, every: Span, multi: ShortConvMulti) -> Result<(), &'static str> {
    check_short_conv_form(rows, heads, dim, taps, form, bound, window, token_taps, addend, fixed_taps, Some((ring, 0)), output, Some((every, 0)), None, Some(multi))
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, multi-chat descriptor, direct per-chat rings and operand spans")]
pub fn check_short_conv_multi_rings(
    rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
    window: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span,
    rings: &[Span], output: Span, every: Span, multi: ShortConvMulti,
) -> Result<(), &'static str> {
    let width = heads.checked_mul(dim).ok_or("fast short conv: width overflow")?;
    if !stream_ops_admitted(STREAM_OP_SHORT_CONV_WIN, rows, width, taps, STREAM_OPS_THREADS) {
        return Err("fast short conv: shape outside envelope");
    }
    if !(bound > 0.0 && bound.is_finite()) || form >= 8 || (form & 4 != 0 && form & 2 == 0) {
        return Err("fast short conv: bad form or bound");
    }
    if ShortConvMulti::encode(rows, taps, multi.start_mask, multi.slots_lo, multi.slots_hi) != Some(multi)
        || rings.len() != multi.chats
    {
        return Err("fast short conv: invalid multi-chat kept ring descriptor");
    }
    let w_b = rows.checked_mul(width).and_then(f32_bytes);
    let f_b = width.checked_mul(taps).and_then(f32_bytes);
    let r_b = taps.checked_mul(width).and_then(f32_bytes);
    let e_b = rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes);
    if !window.valid(w_b) || !fixed_taps.valid(f_b) || !output.valid(w_b) || !every.valid(e_b) {
        return Err("fast short conv: operand size or alignment mismatch");
    }
    if (form & 1 != 0) != token_taps.is_some() || (form & 2 != 0) != addend.is_some() {
        return Err("fast short conv: optional operand mismatch");
    }
    if let Some(tt) = token_taps {
        if !tt.valid(rows.checked_mul(heads).and_then(|n| n.checked_mul(taps)).and_then(f32_bytes))
            || output.meets(tt) || every.meets(tt)
        {
            return Err("fast short conv: bad token_taps span");
        }
    }
    if let Some(ad) = addend {
        if !ad.valid(w_b) || output.meets(ad) || every.meets(ad) {
            return Err("fast short conv: bad addend span");
        }
    }
    if output.meets(window) || output.meets(fixed_taps)
        || every.meets(window) || every.meets(fixed_taps) || every.meets(output)
    {
        return Err("fast short conv: output overlaps input");
    }
    for &ring in rings {
        if !ring.valid(r_b) || output.meets(ring) || every.meets(ring) {
            return Err("fast short conv: bad ring span or overlap");
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, softcap parameters and operand spans")]
pub fn check_softcap_argmax(
    rows: usize, cols: usize, parts: usize, finish: bool, bound: f32, eps: f32, input: Span, output: Span,
) -> Result<(), &'static str> {
    if !stream_ops_admitted(STREAM_OP_SOFTCAP_ARGMAX, rows, cols, parts, STREAM_OPS_THREADS) {
        return Err("fast softcap argmax: shape outside envelope");
    }
    if !(bound > 0.0 && bound.is_finite() && eps > 0.0 && eps.is_finite()) {
        return Err("fast softcap argmax: bad cap or reciprocal");
    }
    let in_w = if finish {
        rows.checked_mul(parts).and_then(|n| n.checked_mul(3))
    } else {
        rows.checked_mul(cols)
    };
    let out_w = if finish {
        rows.checked_mul(3)
    } else {
        rows.checked_mul(parts).and_then(|n| n.checked_mul(3))
    };
    if !input.valid(in_w.and_then(f32_bytes)) || !output.valid(out_w.and_then(f32_bytes)) || output.meets(input) {
        return Err("fast softcap argmax: operand mismatch or overlap");
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, form and operand spans")]
pub(crate) fn check_short_conv_form(rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
    window: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span, ring: Option<(Span, usize)>,
    output: Span, every: Option<(Span, usize)>, kv: Option<KvWrite>, multi: Option<ShortConvMulti>,
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
    if let Some(m) = multi {
        if ShortConvMulti::encode(rows, taps, m.start_mask, m.slots_lo, m.slots_hi) != Some(m)
            || kv.is_some() || ring.is_none() || every.is_none()
        {
            return Err("fast short conv: invalid multi-chat kept ring descriptor");
        }
    }
    if let Some((ring, first)) = ring {
        let ring_chats = if let Some(m) = multi {
            m.chats
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
        if multi.is_none() && (ring.is_some_and(|(_, r_first)| r_first != first) || first >= taps) {
            return Err("fast short conv: bad kept rings span or slot");
        }
        if !every.valid(rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes))
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

/// Host operand check for `NORM_MODE_KV_DIRECT` (multi-chat decode KV finish + direct plane row writes).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "paired key/value shapes, scalars, and per-chat ring/plane spans")]
pub fn check_kv_finish_direct(
    rows: usize, kv_heads: usize, taps: usize, plane_rows: usize,
    key_bound: f32, val_eps: f32, key_scale: f32, val_scale: f32,
    key_in: Span, key_fixed: Span, key_made: Option<Span>, key_every: Span,
    val_in: Span, val_fixed: Span, val_made: Option<Span>, val_every: Span,
    chats: &[(Span, Span, usize, Span, Span, usize)],
) -> Result<(), &'static str> {
    if !kv_finish_direct_admitted(rows, kv_heads, taps, NORM_THREADS) || chats.len() != rows || plane_rows == 0 {
        return Err("fast kv direct: shape or chat count outside envelope");
    }
    if !(key_bound > 0.0 && key_bound.is_finite() && val_eps > 0.0 && val_eps.is_finite()
        && key_scale > 0.0 && key_scale.is_finite() && val_scale > 0.0 && val_scale.is_finite())
    {
        return Err("fast kv direct: non-finite or non-positive scalar");
    }
    let width = kv_heads.checked_mul(KV_FINISH_HEAD_DIM).ok_or("fast kv direct: width overflow")?;
    let in_b = rows.checked_mul(width).and_then(f32_bytes);
    let ring_b = taps.checked_mul(width).and_then(f32_bytes);
    let fixed_b = width.checked_mul(taps).and_then(f32_bytes);
    let made_b = rows.checked_mul(kv_heads).and_then(|n| n.checked_mul(taps)).and_then(f32_bytes);
    let every_b = rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes);
    let plane_b = plane_rows.checked_mul(width / 2).and_then(f32_bytes);
    let row_b = f32_bytes(width / 2).ok_or("fast kv direct: row byte overflow")?;
    for (inp, fix, mad, ev) in [(key_in, key_fixed, key_made, key_every), (val_in, val_fixed, val_made, val_every)] {
        if !inp.valid(in_b) || !fix.valid(fixed_b) || !ev.valid(every_b) {
            return Err("fast kv direct: operand size or alignment mismatch");
        }
        if let Some(m) = mad { if !m.valid(made_b) { return Err("fast kv direct: bad made taps span"); } }
    }
    let mut reads = [Span { address: 0, bytes: 0 }; 6 + 2 * DIRECT_CHATS_MAX];
    let mut n_reads = 0usize;
    for s in [Some(key_in), Some(key_fixed), key_made, Some(val_in), Some(val_fixed), val_made].into_iter().flatten() {
        reads[n_reads] = s; n_reads += 1;
    }
    let mut writes = [Span { address: 0, bytes: 0 }; 2 + 2 * DIRECT_CHATS_MAX];
    writes[0] = key_every;
    writes[1] = val_every;
    let mut n_writes = 2usize;
    for &(kr, vr, rslot, kp, vp, pslot) in chats {
        if rslot >= taps || pslot >= plane_rows || !kr.valid(ring_b) || !vr.valid(ring_b) || !kp.valid(plane_b) || !vp.valid(plane_b) {
            return Err("fast kv direct: chat ring, plane, or slot mismatch");
        }
        let k_row = Span { address: kp.address + (pslot as u64) * row_b, bytes: row_b };
        let v_row = Span { address: vp.address + (pslot as u64) * row_b, bytes: row_b };
        reads[n_reads] = kr; n_reads += 1;
        reads[n_reads] = vr; n_reads += 1;
        writes[n_writes] = k_row; n_writes += 1;
        writes[n_writes] = v_row; n_writes += 1;
    }
    for i in 0..n_writes {
        let ws = writes[i];
        if reads[..n_reads].iter().any(|r| ws.meets(*r)) {
            return Err("fast kv direct: output overlaps input");
        }
        if writes[i + 1..n_writes].iter().any(|ow| ws.meets(*ow)) {
            return Err("fast kv direct: outputs overlap");
        }
    }
    Ok(())
}

/// Host operand check for `NORM_MODE_SHORT_CONV_DIRECT` (multi-chat decode ShortConv without `row_rings` join).
#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments, reason = "shape, form, scalars, and per-chat ring spans")]
pub fn check_short_conv_direct(
    rows: usize, heads: usize, dim: usize, taps: usize, form: u32, bound: f32,
    inputs: Span, token_taps: Option<Span>, addend: Option<Span>, fixed_taps: Span,
    output: Span, every: Span, chat_rings: &[(Span, usize)],
) -> Result<(), &'static str> {
    if !short_conv_direct_admitted(rows, heads, dim, taps, form, NORM_THREADS) || chat_rings.len() != rows {
        return Err("fast short conv direct: shape or chat count outside envelope");
    }
    if !(bound > 0.0 && bound.is_finite()) { return Err("fast short conv direct: bad bound"); }
    if (form & 1 != 0) != token_taps.is_some() || (form & 2 != 0) != addend.is_some() {
        return Err("fast short conv direct: optional operand mismatch");
    }
    let width = heads.checked_mul(dim).ok_or("fast short conv direct: width overflow")?;
    let io_b = rows.checked_mul(width).and_then(f32_bytes);
    let fixed_b = width.checked_mul(taps).and_then(f32_bytes);
    let ring_b = taps.checked_mul(width).and_then(f32_bytes);
    let every_b = rows.checked_mul(taps).and_then(|n| n.checked_mul(width)).and_then(f32_bytes);
    if !inputs.valid(io_b) || !fixed_taps.valid(fixed_b) || !output.valid(io_b) || !every.valid(every_b) {
        return Err("fast short conv direct: operand size or alignment mismatch");
    }
    if let Some(tt) = token_taps {
        if !tt.valid(rows.checked_mul(heads).and_then(|n| n.checked_mul(taps)).and_then(f32_bytes)) {
            return Err("fast short conv direct: bad token taps span");
        }
    }
    if let Some(ad) = addend {
        if !ad.valid(io_b) { return Err("fast short conv direct: bad addend span"); }
    }
    if output.meets(every) { return Err("fast short conv direct: outputs overlap"); }
    for s in [Some(inputs), token_taps, addend, Some(fixed_taps)].into_iter().flatten() {
        if output.meets(s) || every.meets(s) { return Err("fast short conv direct: output overlaps input"); }
    }
    for &(ring, slot) in chat_rings {
        if slot >= taps || !ring.valid(ring_b) || output.meets(ring) || every.meets(ring) {
            return Err("fast short conv direct: bad chat ring span or overlap");
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
        let m16 = ShortConvMulti::encode(16, 3, [0x5555, 0, 0, 0], 0x0000_2102, 0).unwrap();
        assert_eq!(m16.chats, 8);
        assert_eq!(check_short_conv_multi(16, 1, 2304, 3, 6, 0.2, span(1 << 20, 16 * 2304), None, Some(span(1 << 22, 16 * 2304)), span(1 << 24, 3 * 2304),
            span(1 << 26, 8 * 3 * 2304), span(1 << 28, 16 * 2304), span(1 << 30, 16 * 3 * 2304), m16), Ok(()));
        let m256 = ShortConvMulti::encode(256, 4, [0x0101_0101_0101_0101; 4], 0x1b1b_1b1b_1b1b_1b1b, 0).unwrap();
        assert_eq!(m256.chats, 32);
        assert_eq!(check_short_conv_multi(256, 1, 2304, 4, 6, 0.2, span(1 << 20, 256 * 2304), None, Some(span(1 << 22, 256 * 2304)), span(1 << 24, 4 * 2304),
            span(1 << 26, 32 * 4 * 2304), span(1 << 28, 256 * 2304), span(1 << 30, 256 * 4 * 2304), m256), Ok(()));
        assert!(ShortConvMulti::encode(16, 3, [0x5555, 0, 0, 0], 0x0000_3000, 0).is_none());
        assert!(ShortConvMulti::encode(257, 3, [0x0003, 0, 0, 0], 0, 0).is_none());
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
        let direct_rings: Vec<Span> = (0..8).map(|c| span((1 << 26) + (c as u64) * 65_536, 3 * 2304)).collect();
        assert_eq!(check_short_conv_multi_rings(16, 1, 2304, 3, 6, 0.2, span(1 << 20, 16 * 2304), None, Some(span(1 << 22, 16 * 2304)),
            span(1 << 24, 3 * 2304), &direct_rings, span(1 << 28, 16 * 2304), span(1 << 30, 16 * 3 * 2304), m16), Ok(()));
        assert!(check_short_conv_multi_rings(16, 1, 2304, 3, 6, 0.2, span(1 << 20, 16 * 2304), None, Some(span(1 << 22, 16 * 2304)),
            span(1 << 24, 3 * 2304), &direct_rings[..7], span(1 << 28, 16 * 2304), span(1 << 30, 16 * 3 * 2304), m16).is_err());
        assert_eq!(check_softcap_argmax(128, 200_058, 1, false, 30.0, 1.0 / 30.0, span(1 << 20, 128 * 200_058), span(1 << 32, 128 * 3)), Ok(()));
        assert_eq!(check_softcap_argmax(16, 200_058, 12, false, 30.0, 1.0 / 30.0, span(1 << 20, 16 * 200_058), span(1 << 32, 16 * 12 * 3)), Ok(()));
        assert_eq!(check_softcap_argmax(16, 200_058, 12, true, 30.0, 1.0 / 30.0, span(1 << 32, 16 * 12 * 3), span(1 << 33, 16 * 3)), Ok(()));
        assert!(check_softcap_argmax(16, 200_058, 0, false, 30.0, 1.0 / 30.0, span(1 << 20, 16 * 200_058), span(1 << 32, 16 * 3)).is_err());
        assert!(check_softcap_argmax(16, 200_058, 1, false, 0.0, 1.0 / 30.0, span(1 << 20, 16 * 200_058), span(1 << 32, 16 * 3)).is_err());
        assert_eq!(softcap_argmax_parts(128, 200_058), 1);
        assert_eq!(softcap_argmax_parts(64, 200_058), 1);
        assert_eq!(softcap_argmax_parts(16, 200_058), 12);
        assert_eq!(softcap_argmax_parts(1, 200_058), 49);
        let chats_kv = [
            (span(1 << 20, 4 * 1024), span(1 << 21, 4 * 1024), 1usize, span(1 << 22, 64 * 512), span(1 << 23, 64 * 512), 5usize),
            (span(1 << 24, 4 * 1024), span(1 << 25, 4 * 1024), 2usize, span(1 << 26, 64 * 512), span(1 << 27, 64 * 512), 7usize),
        ];
        assert_eq!(check_kv_finish_direct(2, 8, 4, 64, 1.0, 1e-6, 0.5, 0.5,
            span(1 << 28, 2 * 1024), span(1 << 29, 4 * 1024), None, span(1 << 30, 8 * 1024),
            span(1 << 31, 2 * 1024), span(1 << 32, 4 * 1024), None, span(1 << 33, 8 * 1024), &chats_kv), Ok(()));
        let chats_sc = [(span(1 << 20, 4 * 2304), 0usize), (span(1 << 21, 4 * 2304), 3usize)];
        assert_eq!(check_short_conv_direct(2, 1, 2304, 4, 6, 0.5, span(1 << 22, 2 * 2304), None, Some(span(1 << 23, 2 * 2304)),
            span(1 << 24, 4 * 2304), span(1 << 25, 2 * 2304), span(1 << 26, 8 * 2304), &chats_sc), Ok(()));
    }
    /// Exhaustively verify that for every finite BF16 value in `[1e-6 / eps, 5.0 / eps]`, `tanh(r * eps) * bound`
    /// is strictly increasing across adjacent BF16 values — proving the skip bound of `STREAM_OP_SOFTCAP_ARGMAX`
    /// never alters any argmax or tie-break bit-for-bit.
    #[test]
    fn softcap_bf16_strictly_increasing_below_skip_threshold() {
        for bound in [1.0f32, 10.0, 30.0, 32.0, 50.0, 100.0] {
            let eps = (1.0f64 / f64::from(bound)) as f32;
            let lo = 1.0e-6f32 / eps;
            let hi = 5.0f32 / eps;
            let mut prev = f32::NEG_INFINITY;
            for b in 0u16..=0x7f7fu16 {
                let r = f32::from_bits(u32::from(b) << 16);
                let c = crate::fixed_math::tanh(r * eps) * bound;
                assert!(c >= prev, "non-monotonic cap at bound={bound} r={r}");
                if r >= lo && r <= hi {
                    assert!(c > prev, "cap tie inside skip window at bound={bound} r={r} c={c}");
                }
                prev = c;
            }
        }
    }
}
