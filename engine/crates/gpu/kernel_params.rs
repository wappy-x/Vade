//! Launch payloads shared by the host and Rust NVPTX kernels.
/// One dense NVFP4 MMA: A `[16,64]`, transposed B `[8,64]`, scales per 16 K.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Fp4MmaTile {
    pub a: u64, pub b: u64, pub scale_a: u64, pub scale_b: u64,
    pub c: u64, pub d: u64,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Fp4Dequant {
    pub input: u64, pub scales: u64, pub output: u64,
    pub elements: u32, pub global_scale: f32,
}

/// Bit-exact W4A16 NVFP4 projection: BF16 input pairs in u32 words
/// [tokens, cols/2], packed E2M1 weights [rows, cols/2], expanded E4M3
/// scales [rows, cols/16], FP32 output [tokens, rows]. Ascending-K FMA.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Fp4ExactProject {
    pub input: u64, pub weights: u64, pub scales: u64, pub output: u64,
    pub global_scale: f32, pub tokens: u32, pub rows: u32, pub cols: u32,
}

/// BF16(silu(gate) * up) for `pairs` adjacent element pairs; output packs
/// two BF16 values per u32 word, low half first.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Fp4ExactSwiglu {
    pub gate: u64, pub up: u64, pub output: u64, pub pairs: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct LogitsTopK {
    pub logits: u64, pub partials: u64, pub result: u64,
    pub width: u32, pub k: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrefillAttention {
    pub query: u64, pub keys: u64, pub values: u64, pub output: u64,
    pub heads: u32, pub kv_heads: u32, pub rows: u32, pub position: u32,
    /// Nonzero selects the ex2 softmax form (every GQA8 prefill chunk).
    pub capacity: u32, pub exp2: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrefillAttentionMulti {
    pub query: u64, pub metadata: u64, pub tiles: u64, pub output: u64,
    pub heads: u32, pub kv_heads: u32, pub exp2: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BatchEmbedding {
    pub table: u64,
    pub ids: u64,
    pub output: u64,
    pub width: u32,
    pub rows: u32,
    /// 0: row major; 1: blocked64; 2: hybrid packed8.
    pub packed: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct SelectedIds {
    pub result: u64,
    pub ids: u64,
    pub rows: u32,
    pub result_rows: u32,
}

/// Device IDs are FP32 exact integers. Check before converting or indexing;
/// NaN, infinities, negative/fractional IDs and out-of-range sentinels are invalid.
#[inline(always)]
pub fn id_index(id: f32, rows: u32) -> Option<usize> {
    if !id.is_finite() || id < 0.0 || id >= rows as f32 { return None; }
    let index = id as u32;
    if index as f32 != id { return None; }
    Some(index as usize)
}

/// Dense router flags (`renormalize` below): bit 0 renormalizes the top-k
/// weights, bit 1 rounds scores and weights to BF16, and ROUTE_FIRST_LAYER
/// marks a forward's first routed layer, which restarts every row's status.
pub const ROUTE_FIRST_LAYER: u32 = 4;

/// Close one dense router row. A row whose scores are not all finite has no
/// softmax or top-k: it takes experts 0..k, each once and with zero weight, so
/// expert tables and grouped row lists stay in range, and its status word
/// stays 1 for the rest of the forward. The host fails that row's request.
#[inline(always)]
pub unsafe fn finish_route(indices: *mut u32, weights: *mut f32, status: *mut u32,
    selected: u32, valid: bool, first_layer: bool) {
    unsafe {
        if !valid {
            for slot in 0..selected as usize { *indices.add(slot) = slot as u32; *weights.add(slot) = 0.0; }
        }
        if first_layer { *status = u32::from(!valid); } else if !valid { *status = 1; }
    }
}

/// Allocation-free single-row expert launches, also used by decode graphs.
/// `status` is one u32 per routed row (see finish_route).
#[repr(C)]
pub struct MoeRouter {
    pub matrix: u64, pub input: u64, pub scores: u64,
    pub indices: u64, pub weights: u64, pub completed: u64, pub status: u64,
    pub experts: u32, pub width: u32, pub selected: u32, pub renormalize: u32,
}

#[repr(C)]
pub struct MoeRoute {
    pub scores: u64, pub indices: u64, pub weights: u64, pub status: u64,
    pub experts: u32, pub selected: u32, pub renormalize: u32,
}

#[repr(C)]
pub struct MoeSingleProjection {
    pub tables: u64, pub indices: u64, pub input: u64, pub output: u64,
    pub width: u32, pub input_width: u32,
}

#[repr(C)]
pub struct MoeMerge {
    pub input: u64, pub weights: u64, pub output: u64,
    pub hidden: u32, pub selected: u32,
}

#[repr(C)]
pub struct MoeRouteBatch {
    pub scores: u64, pub indices: u64, pub weights: u64, pub status: u64,
    pub experts: u32, pub selected: u32, pub tokens: u32, pub renormalize: u32,
}

#[repr(C)]
pub struct MoeMergeBatch {
    pub projected: u64, pub weights: u64, pub inverse: u64, pub output: u64,
    pub hidden: u32, pub selected: u32, pub tokens: u32, pub bf16: u32,
}

#[repr(C)]
pub struct HybridSharedExpert {
    pub input: u64, pub gate: u64, pub shared: u64, pub output: u64,
    pub hidden: u32, pub tokens: u32,
}

#[repr(C)]
pub struct MoePack {
    pub input: u64,
    pub rows: u64,
    pub counts: u64,
    pub offsets: u64,
    pub tile_experts: u64,
    pub packed: u64,
    pub hidden: u32,
    pub selected: u32,
    pub tokens: u32,
    pub experts: u32,
    pub input_packed: u32,
    pub tile_rows: u32,
}

#[repr(C)]
pub struct MoeRouteGroup {
    pub scores: u64,
    pub indices: u64,
    pub weights: u64,
    pub status: u64,
    pub rows: u64,
    pub counts: u64,
    pub offsets: u64,
    pub inverse: u64,
    pub experts: u32,
    pub selected: u32,
    pub tokens: u32,
    pub renormalize: u32,
}

#[repr(C)]
pub struct MoeGroupOffsets {
    pub rows: u64,
    pub counts: u64,
    pub offsets: u64,
    pub inverse: u64,
    pub tile_experts: u64,
    pub experts: u32,
    pub tokens: u32,
    pub alignment: u32,
}

#[repr(C)]
pub struct MoeDecode {
    pub tables: u64,
    pub counts: u64,
    pub offsets: u64,
    pub tile_experts: u64,
    pub packed: u64,
    pub output: u64,
    pub width: u32,
    pub input_width: u32,
    pub experts: u32,
    pub bf16: u32,
}

#[repr(C)]
pub struct Cascade {
    pub query: u64,
    pub metadata: u64,
    pub parts: u64,
    pub scores: u64,
    pub kv_heads: u32,
    pub rows: u32,
    pub heads: u32,
    /// Scalar-order cohorts run scores (0), then values (1) after softmax.
    pub values: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrefillResidual {
    pub weights: u64, pub input: u64, pub output: u64, pub partials: u64,
    pub width: u32, pub inner: u32, pub rows: u32, pub stride: u32,
}

/// Typed storage for the existing flat QK/RoPE launch ABI. A negative epsilon
/// selects the dense Mistral path without QK normalization.
#[derive(Default)]
#[repr(C)]
pub struct QkRope {
    pub query: u64, pub key: u64, pub value: u64,
    pub q_weight: u64, pub k_weight: u64, pub rope: u64,
    pub keys: u64, pub values: u64, pub metadata: u64, pub replay_position: u64,
    pub heads: u32, pub kv_heads: u32, pub dim: u32, pub position: u32,
    pub epsilon: f32, pub format: u32, pub first_head: u32, pub last_head: u32, pub cache_base: u32,
}
impl QkRope {
    pub fn cache_args(&mut self) -> [*mut core::ffi::c_void; 13] {
        [(&mut self.query as *mut u64).cast(), (&mut self.key as *mut u64).cast(),
         (&mut self.value as *mut u64).cast(), (&mut self.q_weight as *mut u64).cast(),
         (&mut self.k_weight as *mut u64).cast(), (&mut self.rope as *mut u64).cast(),
         (&mut self.keys as *mut u64).cast(), (&mut self.values as *mut u64).cast(),
         (&mut self.heads as *mut u32).cast(), (&mut self.kv_heads as *mut u32).cast(),
         (&mut self.dim as *mut u32).cast(), (&mut self.position as *mut u32).cast(),
         (&mut self.epsilon as *mut f32).cast()]
    }
    pub fn batch_args(&mut self) -> [*mut core::ffi::c_void; 18] {
        let a = self.cache_args();
        [a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10],a[11],a[12],
         (&mut self.format as *mut u32).cast(), (&mut self.first_head as *mut u32).cast(),
         (&mut self.last_head as *mut u32).cast(), (&mut self.cache_base as *mut u32).cast(),
         (&mut self.replay_position as *mut u64).cast()]
    }
    pub fn multi_args(&mut self) -> [*mut core::ffi::c_void; 11] {
        let a = self.cache_args();
        [a[0],a[1],a[2],a[3],a[4],a[5],(&mut self.metadata as *mut u64).cast(),a[8],a[9],a[10],a[12]]
    }
}

#[repr(C)]
pub struct SwiGlu {
    pub gate: u64, pub up: u64, pub width: u32, pub bf16: u32,
}

/// Canonical four-range projection with an optional BF16 output boundary.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct PrefillProjection {
    pub weights: u64, pub input: u64, pub output: u64,
    pub width: u32, pub inner: u32, pub rows: u32, pub stride: u32,
    pub round_bf16: u32,
}

/// Causal/windowed attention for any even head width. Positions are absolute;
/// cache_capacity maps them to the bounded token-major BF16 ring.
/// round_bf16=2 uses BF16 scores and metadata as packed BF16 probability storage;
/// other modes use FP32 scores and metadata as optional per-row cache bindings.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct BoundedAttention {
    pub query: u64, pub keys: u64, pub values: u64, pub scores: u64, pub output: u64, pub metadata: u64,
    pub heads: u32, pub kv_heads: u32, pub dim: u32, pub position: u32,
    pub rows: u32, pub capacity: u32, pub window: u32, pub cache_capacity: u32,
    pub scale: f32, pub round_bf16: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ModelRms {
    pub input: u64, pub weight: u64, pub partials: u64, pub output: u64,
    pub width: u32, pub rows: u32, pub epsilon: f32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ModelQkv {
    pub query: u64, pub key: u64, pub value: u64, pub q_weight: u64, pub k_weight: u64,
    pub rope: u64, pub keys: u64, pub values: u64, pub metadata: u64,
    pub heads: u32, pub kv_heads: u32, pub dim: u32, pub position: u32,
    pub rows: u32, pub cache_capacity: u32, pub epsilon: f32,
}
/// BF16 model boundaries: 0 round/scale, 1 residual, 2 gated GELU, 3 logits cap.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct ModelElementwise {
    pub input: u64, pub peer: u64, pub output: u64,
    pub elements: u32, pub operation: u32, pub scalar: f32,
}

#[derive(Clone, Copy)]
#[repr(C)]
/// Balanced four-range sum; optionally round the completed projection to BF16.
pub struct ProjectionMerge { pub input: u64, pub output: u64, pub elements: u32, pub round_bf16: u32 }

#[derive(Clone, Copy)]
#[repr(C)]
pub struct BoundedCache { pub keys: u64, pub values: u64, pub position: u32, pub capacity: u32 }
#[repr(C)]
pub struct DecodeStream {
    pub query: u64, pub keys: u64, pub values: u64, pub output: u64,
    pub kv_heads: u32, pub base: u32, pub parts: u32, pub span: u32, pub format: u32,
}

/// Storage for the flat grouped-score and canonical 64-key value launch ABIs.
#[repr(C)]
pub struct GroupedAttention {
    pub query: u64, pub keys: u64, pub values: u64, pub scores: u64,
    pub partials: u64, pub output: u64, pub replay_position: u64,
    pub heads: u32, pub kv_heads: u32, pub dim: u32, pub position: u32,
    pub capacity: u32, pub format: u32, pub rows: u32, pub chunks: u32, pub width: u32,
}
impl GroupedAttention {
    pub fn score_args(&mut self) -> [*mut core::ffi::c_void; 11] {
        [(&mut self.query as *mut u64).cast(), (&mut self.keys as *mut u64).cast(),
         (&mut self.scores as *mut u64).cast(), (&mut self.heads as *mut u32).cast(),
         (&mut self.kv_heads as *mut u32).cast(), (&mut self.dim as *mut u32).cast(),
         (&mut self.position as *mut u32).cast(), (&mut self.capacity as *mut u32).cast(),
         (&mut self.format as *mut u32).cast(), (&mut self.rows as *mut u32).cast(),
         (&mut self.replay_position as *mut u64).cast()]
    }
    pub fn value_args(&mut self) -> [*mut core::ffi::c_void; 11] {
        let a = self.score_args();
        [(&mut self.scores as *mut u64).cast(), (&mut self.values as *mut u64).cast(),
         (&mut self.partials as *mut u64).cast(), a[3],a[4],a[5],a[6],a[7],a[8],
         (&mut self.chunks as *mut u32).cast(),a[10]]
    }
    pub fn merge_args(&mut self) -> [*mut core::ffi::c_void; 4] {
        [(&mut self.partials as *mut u64).cast(), (&mut self.output as *mut u64).cast(),
         (&mut self.width as *mut u32).cast(), (&mut self.chunks as *mut u32).cast()]
    }
}

#[repr(C)]
pub struct MultiStream {
    pub query: u64, pub metadata: u64, pub tiles: u64, pub parts: u64,
    pub kv_heads: u32, pub heads: u32,
}

// Launch layouts shared by the host and Rust NVPTX kernels.
#[repr(C)]
pub struct Fp8PrefillWeights {
    pub weights: *const u8,
    pub scales: *const u16,
    pub output: *mut u32,
    pub rows: u32,
    pub cols: u32,
    pub experts: u32,
    pub weight_fragments: u32,
}

/// Long-prefill GEMMs reading resident FP8 fragments and BF16 block scales.
/// Q/K/V uses all three parts; other projections use part 0.
#[repr(C)]
pub struct Fp8PrefillGemm {
    pub weights: [*const u8; 3],
    pub scales: [*const u16; 3],
    pub input: *const u32,
    pub outputs: [*mut f32; 3],
    pub partials: *mut f32,
    pub widths: [u32; 3],
    pub k: u32,
    pub tokens: u32,
}

/// Launch shape of both FP8 prefill tiles (256x128x64, two stages).
#[allow(dead_code)] // host only
pub const FP8_PREFILL_THREADS: u32 = 512;
#[allow(dead_code)] // host only
pub const FP8_PREFILL_SHARED_BYTES: u32 = 98304;

/// Dynamic shared bytes of fp4_prefill_tc_gate: two 128-row BF16 A stages (32,768 B)
/// plus a 3-deep ring of packed gate/up B (8,192 B) and scale words (1,024 B).
pub const FP4_PREFILL_GATE_SHARED_BYTES: u32 = 32768 + 3 * 9216;

#[repr(C)]
pub struct QsaGate {
    pub query_gate: *const f32, pub attention: *mut f32,
    /// Null keeps token-major FP32 storage; otherwise emit the BF16 MMA operand.
    pub packed: *mut u16,
    pub width: u32, pub tokens: u32,
}

/// Largest query-head group per key head that flash_dense_values stages.
pub const DENSE_VALUE_GROUP: u32 = 16;
#[repr(C)]
pub struct DenseAttention {
    pub query: u64, pub query_gate: u64, pub output: u64, pub rows: u64,
    pub scores: u64, pub replay_position: u64,
    pub keys: u64, pub values: u64,
    pub query_heads: u32, pub key_heads: u32, pub count: u32, pub score_stride: u32, pub score_tiles: u32,
    pub past: u32,
}

#[repr(C)]
pub struct GdnPrepare {
    pub x: *const f32, pub w: *const u16, pub history: *const f32,
    pub a: *const f32, pub b: *const f32, pub alog: *const u16, pub dt: *const u16,
    pub q: *mut f32, pub k: *mut f32, pub v: *mut f32, pub beta: *mut f32, pub g: *mut f32,
    pub key_heads: u32, pub value_heads: u32, pub key_start: u32, pub value_start: u32,
    pub tokens: u32, pub precision: u32, pub scale: u32,
}
/// Independent decode rows; each grid-Y slot owns its history and scratch row.
#[repr(C)]
pub struct GdnPrepareRows { pub prepare: GdnPrepare, pub rows: u64 }
/// One-token conv-prepare plus its history update; `rows` is zero for a direct
/// history or a GdnDecodeRow table by grid-Y. `snapshot` optionally receives
/// the updated history words, exactly as the separate history launch wrote.
#[repr(C)]
pub struct GdnPrepareHist { pub prepare: GdnPrepare, pub rows: u64, pub snapshot: u64 }
/// One verify GDN layer's pre-scan copies in one launch (flash_gdn_verify_copy): up to four
/// word ranges (the replay tape's qkv/a/b rows and the conv history snapshot; words 0 =
/// absent) plus, when `descriptor` != 0, the scan's three-address descriptor, written as
/// the same little-endian words write_vector uploads. Pure data movement.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GdnVerifyCopy {
    pub source: [u64; 4], pub destination: [u64; 4], pub words: [u32; 4],
    pub descriptor: u64, pub addresses: [u64; 3],
}
#[repr(C)]
pub struct GdnHistory {
    // Direct history: rows is an optional snapshot destination; prompts: descriptor table.
    pub x: *const f32, pub history: *mut f32, pub rows: u64,
    pub channels: u32, pub tokens: u32, pub precision: u32,
}

#[repr(C)]
pub struct GdnSystem {
    pub q: *const f32,
    pub k: *const f32,
    pub beta: *const f32,
    pub cumulative: *const f32,
    pub system: *mut f32,
    pub attention: *mut f32,
    pub heads: u32,
    pub tokens: u32,
    /// Value heads per stored Q/K head (1 when Q/K are expanded per value head).
    pub key_repeat: u32,
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct GdnChunkPrompts {
    pub rows: u64, pub heads: u32, pub key_repeat: u32,
    pub qkv: u64, pub conv: u64, pub keys: u32, pub precision: u32,
}

#[repr(C)]
pub struct GdnSolve {
    pub qkv: *const f32, pub conv: *const u16, pub history: *const f32,
    pub keys: u32, pub precision: u32,
    pub k: *const f32, pub v: *const f32,
    pub beta: *const f32, pub cumulative: *const f32,
    pub system: *const f32,
    pub new_values: *mut f32, pub decayed_keys: *mut f32,
    pub heads: u32, pub tokens: u32, pub key_repeat: u32,
}

#[repr(C)]
pub struct GdnScan {
    pub q: *const f32,
    pub k: *const f32,
    pub attention: *const f32,
    pub new_values: *const f32,
    pub decayed_keys: *const f32,
    pub chunk_decay: *const f32,
    pub cumulative: *const f32,
    pub state: *mut f32,
    pub output: *mut f32,
    pub heads: u32,
    pub tokens: u32,
    pub start: u32,
    pub key_repeat: u32,
}

pub const GDN_SYSTEM_THREADS: u32 = 256;
/// Transposed K (128 x 33) plus row-major Q and beta*K (32 x 128), FP32.
#[allow(dead_code)] // launch size: host only
pub const GDN_SYSTEM_SHARED_BYTES: u32 = 4 * (128 * 33 + 2 * 32 * 128);
pub const GDN_SCAN_COLUMNS: usize = 64;
pub const GDN_SCAN_THREADS: u32 = 512;
#[allow(dead_code)] // launch size: host only
/// State slice (128 x columns) and two 64 x 128 operand tiles, FP32.
/// After prediction: attention, final-decay keys and corrections reuse the tiles.
pub const GDN_SCAN_SHARED_BYTES: u32 = (4 * (128 * GDN_SCAN_COLUMNS + 2 * 64 * 128)) as u32;

#[repr(C)]
pub struct GdnScanPrompts {
    pub rows: u64,
    pub heads: u32,
    pub key_repeat: u32,
}

#[repr(C)]
pub struct GdnGate {
    pub x: *const f32,
    pub z: *const f32,
    pub norm: *const f32,
    pub output: *mut f32,
    pub heads: u32,
    pub tokens: u32,
    pub precision: u32,
    pub packed: u32,
    pub epsilon: f32,
    pub sessions: u64,
}
/// Hybrid RMS boundary; order 0 is short prefill, 1 is the packed long
/// prefill reduction, and 2 is canonical batch-one CUDA decode arithmetic.
#[repr(C)]
pub struct HybridNorm {
    pub input: *mut f32,
    pub residual: *const f32,
    pub weight: *const u16,
    pub output: *mut f32,
    pub width: u32,
    pub zero_centered: u32,
    pub epsilon: f32,
    pub order: u32,
}

/// Hybrid activation boundary, including dynamically quantized MTP projections.
#[repr(C)]
pub struct HybridSilu {
    pub input: *mut f32,
    pub up: *const f32,
    pub count: u32,
    pub divisor: f32,
    pub precision: u32,
}

/// 512 selected four-token groups and at most three unpooled tail positions.
pub const QSA_MAX_POSITIONS: usize = 4 * 512 + 3;

/// Complete index groups are averaged, normalized and rotated in FP32.
#[repr(C)]
pub struct QsaIndexPool {
    pub raw_keys: u64, pub weight: u64, pub pooled: u64,
    pub blocks: u32, pub position_offset: u32,
}

/// Index scores are [tokens, blocks]. Selection reuses each row's first 512
/// words for ascending group IDs; queries with at most 512 groups need no IDs.
#[repr(C)]
pub struct QsaIndex {
    pub query: u64, pub pooled: u64, pub scores: u64,
    pub tokens: u32, pub past: u32, pub blocks: u32, pub padding: u32,
}

/// Neighboring prompt queries sharing one pooled index key per score warp.
pub const QSA_INDEX_QUERIES: usize = 4;
/// Independent pooled keys sharing each query load in prompt score warps.
pub const QSA_INDEX_GROUPS: usize = 4;

/// Score warps share keys across this many globally aligned query heads.
/// All twelve query heads per KV head reuse the same selected key loads.
pub const QSA_SCORE_HEADS: usize = 12;
/// Visible keys per score warp, sharing one query load across all four dots.
pub const QSA_SCORE_KEYS: usize = 4;
/// Query heads sharing one selected FP32 value tile for a KV head.
pub const QSA_VALUE_HEADS: usize = 12;
/// Flash indexed attention (flash_qsa_attention_scores, _softmax, _values) per
/// query head and token. `scores` holds [rows, stride] FP32 scores in ascending
/// selected-slot order (then their probabilities), never dense context order.
/// `ids` holds the indexer's selected groups; `denominators` one value per row.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct QsaAttention {
    pub query: u64, pub query_gate: u64, pub keys: u64, pub values: u64,
    pub ids: u64, pub output: u64, pub scores: u64,
    pub query_heads: u32, pub key_heads: u32, pub query_start: u32, pub key_start: u32,
    pub tokens: u32, pub past: u32, pub blocks: u32, pub queries_per_key: u32,
    pub stride: u32, pub padding: u32,
    /// One FP32 softmax denominator per query row (split indexed attention).
    pub denominators: u64,
}

/// Independent one-token caches and scratch, uploaded before batch replay.
/// No weight or history is copied into the batch's stable descriptor table.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct QsaBatchRow {
    pub keys: u64, pub values: u64, pub raw_index: u64, pub query: u64,
    pub index_query: u64, pub pooled: u64, pub ids: u64, pub scores: u64,
    pub denominators: u64, pub norm: u64,
    pub past: u32, pub stride: u32,
}

/// Sessions occupy grid-z for scores, whose grid-y selects slot planes;
/// every other stage selects its session on grid-y. Padded sessions are inert.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct QsaBatch {
    pub rows: u64, pub query_gate: u64, pub key: u64, pub value: u64,
    pub index: u64, pub output: u64,
    pub query_heads: u32, pub key_heads: u32, pub query_start: u32, pub key_start: u32,
    pub count: u32, pub padding: u32,
}

/// Canonical split BF16 tensor-core projections. Rows below
/// `rows` come from `weight`, the next `rows2` from `weight2`; both read the
/// same `input` tile and write `output[token * stride + row]`.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Bf16Rows {
    pub weight: u64, pub output: u64, pub rows: u32, pub stride: u32,
    pub weight2: u64, pub output2: u64, pub rows2: u32, pub stride2: u32,
    pub input: u64, pub cols: u32, pub tokens: u32,
    /// Single-row input already split into row-major BF16 hi/lo planes by its producer.
    pub input_bf16: u32,
    /// HC down rows produce split SiLU(x/4); four injection rows stay FP32.
    pub hc_silu: u32,
    /// Packed A-plane stride in u32 words, retained when a prompt tile is sliced.
    pub input_plane_words: u64,
}
/// Two independent projection pairs sharing one decode input and launch.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct Bf16Pairs {
    pub pairs: [Bf16Rows; 2],
}
/// Scheduling only: both shapes execute the canonical per-output K16 graph.
pub fn flash_bf16_prefill_blocks(p: Bf16Rows) -> usize {
    (p.tokens as usize).div_ceil(64)
        * ((p.rows + p.rows2) as usize).div_ceil(64)
}
/// Tokens that share one weight pass in canonical Flash MMA decode.
#[cfg(not(target_arch = "nvptx64"))]
pub const BF16_ROWS_TOKENS: usize = 8;
/// Output rows and tokens of the small canonical Flash MMA prompt tile.
#[cfg(not(target_arch = "nvptx64"))]
pub const BF16_TILE_ROWS: usize = 32;
#[cfg(not(target_arch = "nvptx64"))]
pub const BF16_TILE_TOKENS: usize = 16;
/// Large canonical Flash MMA prompt tile; smaller batches use the tile above.
#[cfg(not(target_arch = "nvptx64"))]
pub const BF16_MMA_ROWS: usize = 64;
#[cfg(not(target_arch = "nvptx64"))]
pub const BF16_MMA_TOKENS: usize = 64;

/// Pack from a bounded source tile, or in place through shared memory when source is zero.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashBf16Pack { pub source: u64, pub weight: u64, pub rows: u32, pub cols: u32 }

#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashBf16Input {
    pub input: u64, pub output: u64, pub tokens: u32, pub cols: u32,
    /// Pack HC up's split SiLU(x/4) directly from its FP32 down output.
    pub hc_silu: u32,
    /// A-fragment lanes written per K16 tile (4, 8, 16; anything else = all 32): a
    /// flash_bf16_mma_latency consumer with K % 64 == 0 stages only
    /// flash_latency_lanes(M) lanes, so the unread lanes of each tile are left as they are.
    pub lanes: u32,
}
/// Input lanes per K16 tile that flash_bf16_mma_latency's K % 64 == 0 path stages at M
/// tokens (the L of its dispatch in dense_latency.rs: rows lane/4 and lane/4 + 8 < M).
pub const fn flash_latency_lanes(tokens: usize) -> usize {
    match tokens {
        1 => 4,
        2 => 8,
        3 | 4 => 16,
        _ => 32,
    }
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashHcNorm { pub input: u64, pub weight: u64, pub output: u64, pub packed: u64 }
/// A zero peer selects contiguous block rows; otherwise each packet holds H/2 columns.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashHcAdd {
    pub residual: u64, pub block: u64, pub peer: u64, pub gates: u64, pub hidden: u32, pub rank: u32,
}
/// Prompt HC combine fused with the next mix's F32 norm (flash_next_hc_add_norm): residual
/// `[M][4H]` in place, block `[M][H]`, gates `[M][4]`, F32 norm weights `[4H]`, normalized output
/// `[M][4H]` and optional dense hi/lo planes (packed = 0: none). scales != 0: the norm scales
/// `[M][4]` go there instead of the normalized output (prompt HC up forms the streams).
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashHcAddNorm {
    pub residual: u64, pub block: u64, pub gates: u64, pub weight: u64, pub output: u64, pub packed: u64, pub scales: u64,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashPleGate { pub query: u64, pub key: u64, pub value: u64, pub output: u64, pub width: u32 }
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashPleConv {
    pub input: u64, pub weight: u64, pub history: u64, pub gated: u64, pub residual: u64,
    pub width: u32, pub tokens: u32,
}
pub const TERMS: usize = 2;
/// Dynamic shared bytes per warp of flash_bf16_mma_latency(_group) and
/// flash_mx8_mma_latency (each warp's private K16 stage ring).
pub const LATENCY_RING: usize = 6144;
/// MXFP8 at M5-8 with K <= 2560 (K % 256 == 0) and four warps: the CTA keeps one compact
/// copy of the input (the transposed MMA's a0/a2 words, 512 bytes per K16 tile) and each
/// warp a LATENCY_W8_RING-byte weight ring: K / 16 * 512 + 4 * 4864 <= 101,376 bytes.
pub const LATENCY_W8_RING: usize = 4864;
pub const fn latency_mx8_wide(tokens: usize, cols: usize) -> bool {
    tokens >= 5 && tokens <= 8 && cols <= 2560 && cols % 256 == 0
}
/// MXFP8 at M9-16 with K <= 2560 (K % 256 == 0) and four warps: the CTA keeps one
/// canonical range of the input at a time (16 tokens, 1 KB per K16 tile) and fragment-only
/// rings; the four ranges merge (p0+p1)+(p2+p3) in registers.
pub const fn latency_mx8_wide16(tokens: usize, cols: usize) -> bool {
    tokens >= 9 && tokens <= 16 && cols <= 2560 && cols % 256 == 0
}
/// The wide path of flash_mx8_mma_latency_w(_group), including
/// M17-32 with both 16-token tiles per chunk: MXFP8 at M9-32 with K <= 6144 and at M5-8 with
/// 2560 < K <= 6144 (M5-8 with K <= 2560 keeps wide8), K % 256 == 0.
pub const fn latency_mx8_widen(tokens: usize, cols: usize) -> bool {
    cols <= 6144 && cols % 256 == 0 && (tokens >= 9 && tokens <= 32 || tokens >= 5 && tokens <= 8 && cols > 2560)
}
/// K16 tiles per CTA input chunk of wide_n's per-range copies (g 4-token groups, 256 bytes per
/// group per tile): the whole canonical range when it fits 81920 bytes and 64 tiles, else equal
/// chunks of whole K64 quads.
pub const fn latency_wide_chunk_tiles(cols: usize, g: usize) -> usize {
    let range = cols / 64;
    let (bytes, tiles) = ((range * g * 256 + 81919) / 81920, (range + 63) / 64);
    let chunks = if bytes > tiles { bytes } else { tiles };
    ((range + chunks - 1) / chunks + 3) / 4 * 4
}
/// flash_bf16_mma_latency_group2 (small BF16 groups at M17-32, latency_group2.rs): 4 warps x
/// 8 ring stages x (two 16-token input tiles, hi and lo planes, 2048 bytes + two 256-byte B fragments
/// for the task's two 8-row groups).
pub const LATENCY_GROUP2_SHARED: usize = 4 * 8 * 2560;
/// Kernel split predicate: grouped launches merge inside the CTA.
pub const fn latency_split(total_rows: usize, grouped: bool) -> bool {
    total_rows <= if grouped { 4096 } else { 640 }
}
/// BF16 solo M1 launches with N <= 64 in 128 threads: the CTA input copy and K64
/// fragment rings of dense_latency_fold.rs (N % 4 == 0: packed rows). Wider N keep the
/// K16 ring: every CTA copies the whole input, which cost N 320-640 +0.4..+1.1 us.
pub const fn latency_fold1(tokens: usize, cols: usize, rows: usize) -> bool {
    tokens == 1 && cols % 256 == 0 && cols <= 10240 && rows % 4 == 0 && rows <= 64
}
/// Dynamic shared bytes of flash_bf16_mma_latency_fold (latency_fold1 shapes; else 4 RING).
pub const fn latency_fold_shared(tokens: usize, cols: usize, rows: usize) -> usize {
    if latency_fold1(tokens, cols, rows) { 4 * LATENCY_RING + cols / 16 * 64 } else { 4 * LATENCY_RING }
}
/// K64 stages plus the range's scale burst must fit one warp's ring.
pub const fn latency_mx4(cols: usize) -> bool {
    cols % 256 == 0 && 5 * 1040 + cols / 16 <= LATENCY_RING
}
/// Conservative size for callers without row counts (either kernel path may run).
pub const fn latency_mx8_shared(tokens: usize, cols: usize, warps: usize) -> usize {
    let wide = latency_mx8_shared_rows(tokens, cols, warps, usize::MAX, false);
    if wide < warps * LATENCY_RING { warps * LATENCY_RING } else { wide }
}
/// Size from the same total-row split predicate as dense_latency::fast.
pub const fn latency_mx8_shared_rows(tokens: usize, cols: usize, warps: usize, total_rows: usize, grouped: bool) -> usize {
    if latency_split(total_rows, grouped) { return warps * LATENCY_RING; }
    if warps == 4 && latency_mx8_wide(tokens, cols) { cols / 16 * 512 + 4 * LATENCY_W8_RING }
    else if warps == 4 && latency_mx8_wide16(tokens, cols) { (cols / 64 + 1) * 1024 + 4 * LATENCY_W8_RING }
    else if warps == 4 && latency_mx8_widen(tokens, cols) {
        // M17-32: both 16-token tiles per chunk (eight 4-token groups, 2 KB per K16 tile).
        let two = tokens > 16;
        latency_wide_chunk_tiles(cols, if two { 8 } else { 4 }) * if two { 2048 } else { 1024 } + 4 * LATENCY_W8_RING
    }
    else if warps == 4 && latency_mx8_wide1(tokens, cols) { cols / 16 * 64 + 4 * LATENCY_W8_RING }
    else { warps * LATENCY_RING }
}
/// MXFP8 at M1 in the same structure (CTA input copy of token 0's a0/a2 words, 64 bytes
/// per K16 tile, plus the fragment-only rings) for K % 256 == 0.
pub const fn latency_mx8_wide1(tokens: usize, cols: usize) -> bool {
    tokens == 1 && cols % 256 == 0 && cols <= 10240
}
pub const FLASH_BF16_INPUT_BYTES: usize = 4096 * 10240 * 2 * TERMS;
pub const FLASH_BF16_MMA_SHARED: usize = 2 * (64 * 128 + TERMS * 64 * 128);
/// Dense hi/lo planes of a prompt HC up output: up to 4,096 rows x K 2,560 (flash_mixed_planes).
pub const FLASH_BF16_MIXED_BYTES: usize = 4096 * 2560 * 2 * TERMS;
/// flash_bf16_mma_prefill_tiled: three 32 KiB stages (hi/lo A planes 16 KiB + B 16 KiB) and,
/// after them, the stages' full/empty mbarriers (6 x 8 bytes, padded to 64).
pub const FLASH_BF16_PREFILL_TILED_SHARED: usize = 3 * 32768 + 64;
/// flash_bf16_mma_prefill_tiled_hc (its norm != 0 form fills it; the planes form fits): two
/// stages of A planes 16 KiB + B 16 KiB + the FP32 A tile 16 KiB + its 64 norm weights (256 bytes).
pub const FLASH_BF16_PREFILL_ROWS_SHARED: usize = 2 * (3 * 16384 + 256);
/// Feature slices per prompt HC up CTA (flash_next_hc_up_tiled; H/32 = 80 is a multiple).
pub const FLASH_HC_UP_SLICES: usize = 1;
pub const FLASH_BF16_WIDE_STAGE: usize = TERMS * 8192 + 4096;
pub const FLASH_BF16_WIDE_SHARED: usize = 3 * FLASH_BF16_WIDE_STAGE;
const _: () = assert!(TERMS == 2 && FLASH_BF16_WIDE_SHARED <= 101376 && FLASH_BF16_MMA_SHARED <= 101376
    && FLASH_BF16_PREFILL_TILED_SHARED <= FLASH_BF16_PREFILL_ROWS_SHARED && FLASH_BF16_PREFILL_ROWS_SHARED <= 101376);
pub const fn flash_bf16_decode_split(rows: usize, cols: usize) -> bool {
    rows < 1024 && cols > 512 || cols == 320 && rows >= 1024
}

pub const fn flash_bf16_packed(rows: usize, cols: usize) -> bool {
    rows != 0 && rows % 4 == 0 && cols != 0 && cols % 16 == 0
}
pub const fn flash_bf16_element(row: usize, col: usize, width: usize, rows: usize) -> usize {
    if rows % 64 == 0 {
        let (rt, ct, kt) = (row / 8, col / 16, width / 16);
        let tile = if kt % 4 == 0 { (rt / 8 * (kt / 4) + ct / 4) * 32 + rt % 8 * 4 + ct % 4 }
            else { rt * kt + ct };
        return (tile * 32 + row % 8 * 4 + col % 8 / 2) * 4 + col % 2 + col % 16 / 8 * 2;
    }
    ((row / 4 * (width / 16) + col / 16) * 16 + row % 4 * 4 + col % 8 / 2) * 4
        + col % 2 + col % 16 / 8 * 2
}

/// A K-split projection keeps the two-rank split's sum tree at every rank
/// count: K divides into two shards of four canonical ranges (eight leaves). A
/// four-rank shard computes only its two ranges, and the ranks fold as a pair
/// tree.
pub const SPLIT_LEAVES: usize = 8;
/// Canonical ranges in a K shard of `shard` of `whole` columns: 2 for a shard
/// of exactly two leaves (four ranks); any other shard keeps four of its own (0).
#[inline]
pub const fn shard_ranges(shard: usize, whole: usize) -> u32 {
    if whole != 0 && shard * SPLIT_LEAVES == 2 * whole { 2 } else { 0 }
}
/// Bf16Rows.cols bit where a K-split shard's ranges start. Only the generic
/// projection entries read it; zero keeps every launch byte-identical to a
/// launch without ranges.
pub const BF16_RANGES_SHIFT: u32 = 28;
/// The Bf16Rows.cols word of a K-split shard: K, plus its ranges when not four.
#[inline]
pub const fn bf16_cols_word(cols: u32, ranges: u32) -> u32 { cols | ranges << BF16_RANGES_SHIFT }
/// K and the ranges from a Bf16Rows.cols word.
#[inline]
pub const fn bf16_cols_ranges(word: u32) -> (u32, u32) {
    (word & ((1 << BF16_RANGES_SHIFT) - 1), word >> BF16_RANGES_SHIFT)
}
/// K16 tiles of merge part `part` (0..4) in a shard with `ranges` ranges (2, or
/// any other value for four). Parts 1 and 3 of a two-range shard are None:
/// kernels give them -0.0, the exact FP32 identity, so `(p0 + p1) + (p2 + p3)`
/// equals the shard's own subtree of the canonical tree.
#[inline]
pub const fn canonical_span(tiles: usize, part: usize, ranges: u32) -> Option<(usize, usize)> {
    if ranges != 2 { return Some((tiles * part / 4, tiles * (part + 1) / 4)); }
    if part % 2 != 0 { return None; }
    Some((tiles * (part / 2) / 2, tiles * (part / 2 + 1) / 2))
}
/// Rank counts fold_ranks holds: rank 65535 would need a 17th level.
pub const FOLD_RANKS_LIMIT: usize = 1 << 16;
/// FP32 partials of `count` ranks folded as a pair tree: ranks 2i and 2i+1
/// first, then pairs of pairs. Two and three ranks keep the sequential order.
/// Level j holds the latest complete block of 2^j ranks while bit j of the
/// ranks taken is set. A rank adds the levels of its trailing one bits, older
/// block first, and the result adds the set levels of `count` from low to high:
/// the additions of the in-place pairwise passes. The level steps are written
/// out, so every index is a constant and a kernel keeps the levels in registers.
#[inline]
pub fn fold_ranks(count: usize, value: impl Fn(usize) -> f32) -> f32 {
    debug_assert!(count < FOLD_RANKS_LIMIT, "fold_ranks holds at most 65535 ranks");
    let mut levels = [0.0f32; 16];
    for rank in 0..count {
        let (next, mut carry) = (rank + 1, value(rank));
        // Bits 0..=bit of rank + 1 are zero while bit is a trailing one of rank.
        macro_rules! steps { ($($bit:literal)*) => { $(
            let low = next & ((2 << $bit) - 1);
            if low == 0 { carry = levels[$bit] + carry; } else if low == 1 << $bit { levels[$bit] = carry; }
        )* } }
        steps!(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14);
        // Level 15 only takes a block: a 16th trailing one needs rank 65535.
        if next & 0xffff == 1 << 15 { levels[15] = carry; }
    }
    let mut sum = None;
    for (bit, level) in levels.iter().enumerate() {
        if count >> bit & 1 != 0 { sum = Some(sum.map_or(*level, |sum| *level + sum)); }
    }
    sum.unwrap_or(0.0)
}

/// Raw BF16 KV words retained at a canonical hybrid prefill boundary.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct HybridCopy {
    pub source: *const u32,
    pub destination: *mut u32,
    pub words: u32,
    pub padding: u32,
    pub rows: u64,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct HybridCopyRow {
    pub source: u64, pub destination: u64,
    pub words: u32, pub padding: u32,
}

/// Shared QSA normalization, rotary positions and KV insertion.
#[repr(C)]
pub struct QsaPrepare {
    pub query_gate: u64, pub key: u64, pub value: u64,
    pub query_weight: u64, pub key_weight: u64,
    pub query: u64, pub keys: u64, pub values: u64,
    pub heads: u32, pub kv_heads: u32, pub tokens: u32, pub past: u32,
    pub position_offset: u32, pub frequencies: u64, pub precision: u32,
    pub epsilon: f32, pub replay_position: u64,
}

/// Ragged canonical Q/K normalization, RoPE and KV insertion.
#[repr(C)]
pub struct MultiQk {
    pub query: u64, pub key: u64, pub value: u64,
    pub q_weight: u64, pub k_weight: u64, pub rope: u64, pub metadata: u64,
    pub heads: u32, pub kv_heads: u32, pub dim: u32, pub epsilon: f32,
}
#[repr(C)]
pub struct DraftAttention {
    pub query: u64, pub metadata: u64, pub tiles: u64, pub output: u64,
    pub kv_heads: u32,
}

#[repr(C)]
pub struct GatherRows {
    pub input: u64, pub indices: u64, pub output: u64,
    pub width: u32, pub rows: u32,
}
/// NVFP4 checkpoint layout: row-major low-nibble-first E2M1 pairs, row-major
/// E4M3 scales per 16 logical columns, and pointers to the FP32 global scales.
/// Rows/columns describe the rank-local matrix; starts retain global channels.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct NvFp4Matrix {
    pub weight: u64,
    pub block_scales: u64,
    pub global_scale: u64,
    pub input_scale: u64,
    pub rows: u32,
    pub cols: u32,
    pub row_start: u32,
    pub column_start: u32,
}

/// Bound hd256 operands before BF16 conversion and FP32 QK/PV accumulation.
/// Normalized model activations are far below this conservative ceiling.
#[inline(always)]
pub fn hd256_operand(value: f32) -> bool { value.is_finite() && value.abs() <= 1.0e18 }
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AttentionValidate {
    pub query: u64, pub keys: u64, pub values: u64, pub error: u64,
    pub query_elements: u32, pub kv_elements: u32,
}

/// Flash router probabilities and selections, with one warp per token.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct FlashMoeRoute {
    pub logits: u64, pub ids: u64, pub weights: u64,
    pub tokens: u32, pub norm_topk: u32,
}

/// FP4-Local prototype ABI: one ascending K16 chain at every row count.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextMatrix { pub weight: u64, pub scales: u64, pub global: f32, pub reserved: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextExpert { pub matrices: [NextMatrix; 3] }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextDense { pub weight: u64, pub weight2: u64, pub input: u64, pub output: u64,
    pub rows: u32, pub cols: u32, pub tokens: u32, pub packed: u32, pub rounded: u32, pub reserved: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextMoe { pub experts: u64, pub input: u64, pub logits: u64, pub ids: u64, pub weights: u64,
    pub tiles: u64, pub tile_count: u64, pub rows: u64, pub positions: u64, pub gathered: u64,
    pub gate_up: u64, pub middle: u64, pub down: u64, pub shared_down: u64, pub shared_gate: u64,
    pub combined: u64, pub output: u64, pub tokens: u32, pub reserved: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextRouted { pub experts: u64, pub input: u64, pub output: u64, pub tiles: u64,
    pub tile_count: u64, pub rows: u32, pub cols: u32, pub down: u32, pub reserved: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextActivation { pub input: u64, pub output: u64, pub rows: u32, pub reserved: u32 }
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct NextHc { pub input: u64, pub block: u64, pub previous: u64, pub weight: u64,
    pub down: u64, pub injection: u64, pub up: u64, pub state: u64, pub normalized: u64,
    pub gates: u64, pub low: u64, pub output: u64, pub tokens: u32, pub packed: u32,
    pub norm_f32: u32, pub reserved: u32 }

/// Same allocation as sixteen-output-row's original M9-sized launch, independent of the
/// wide-row owner's helper/dispatch. M2-4, K%256=0, K<=6144, four warps.
pub const fn verify_solo97_shared(cols: usize) -> usize {
    let tiles = if cols <= 2560 { cols / 64 + 1 } else {
        let range = cols / 64;
        let bytes = (range * 4 * 256 + 81919) / 81920;
        let tiles = (range + 63) / 64;
        let chunks = if bytes > tiles { bytes } else { tiles };
        ((range + chunks - 1) / chunks + 3) / 4 * 4
    };
    4 * LATENCY_W8_RING + tiles * 1024
}
/// The original qualified MX4 canonical-range cache allocation.
pub const VERIFY_MX4_PART_SHARED: usize = 4 * LATENCY_W8_RING + (2560 / 16 / 4) * 32 * 8;
/// flash_verify_mx4_part_group8 (M5-8): the same rings plus one canonical range of both
/// token halves (512 bytes per K16 tile at K2560).
pub const VERIFY_MX4_PART8_SHARED: usize = 4 * LATENCY_W8_RING + (2560 / 16 / 4) * 64 * 8;
