//! Shared host/NVPTX row descriptor for canonical cached recurrence.
#[repr(C)]
pub struct GdnDecodeRow {
    pub state: u64,
    pub history: u64,
    pub output_state: u64,
}

/// Flash keeps scratch and FP32 state in the owning session. Graphs read this
/// table at replay time, so admitting or retiring a chat never bakes its address.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct GdnSessionRow {
    pub state: u64, pub history: u64, pub convolved: u64,
    pub q: u64, pub k: u64, pub v: u64, pub beta: u64, pub g: u64,
    pub core: u64, pub norm: u64,
    // padding=1 retires this slot without touching its recurrent state.
    pub row: u32, pub padding: u32,
}

#[repr(C)]
pub struct GdnConvBatch {
    pub x: u64, pub w: u64, pub rows: u64, pub y: u64, pub sessions: u64,
    pub channels: u32, pub count: u32, pub precision: u32,
}

#[repr(C)]
pub struct GdnPrepareBatch {
    pub qkv: u64, pub a: u64, pub b: u64, pub alog: u64, pub dt: u64,
    pub q: u64, pub k: u64, pub v: u64, pub beta: u64, pub g: u64, pub sessions: u64,
    pub key_heads: u32, pub value_heads: u32, pub key_start: u32, pub value_start: u32,
    pub count: u32, pub precision: u32,
}

#[repr(C)]
pub struct GdnStepBatch {
    pub q: u64, pub k: u64, pub v: u64, pub beta: u64, pub g: u64,
    pub rows: u64, pub out: u64, pub sessions: u64,
    pub heads: u32, pub count: u32,
}

/// One-token step plus its gate, for descriptor rows only (no sessions).
/// `out` is the gated token-major output the separate gate launch wrote.
#[repr(C)]
pub struct GdnStepGateBatch {
    pub q: u64, pub k: u64, pub v: u64, pub beta: u64, pub g: u64,
    pub rows: u64, pub z: u64, pub norm: u64, pub out: u64,
    pub heads: u32, pub count: u32, pub precision: u32, pub epsilon: f32,
}

/// Chunk-form GDN (flash_next_gdn_chunk_prep/scan, switch VADE_GDN_CHUNK).
/// rows -> GdnDecodeRow whose state is a chunk-form state (chunk_state_floats),
/// updated in place; base = absolute position of the call's first row (the open
/// chunk holds base % 64 rows). Workspace per (call chunk, value head): a holds A
/// in the substitution layout [64 j][4 m%4][16 m/4], p holds qk per (call chunk,
/// key head) as [64 j][64 i] packed (the scan forms P from it and gamma),
/// gamma holds [64]; k0 holds the first chunk's tail key rows [keys][64][128].
#[repr(C)]
pub struct GdnChunkNext {
    pub q: u64, pub k: u64, pub v: u64, pub beta: u64, pub g: u64,
    pub rows: u64, pub out: u64,
    pub a: u64, pub p: u64, pub gamma: u64, pub k0: u64,
    pub heads: u32, pub keys: u32, pub count: u32, pub base: u32,
    /// Fused output gate (flash_next_gdn_chunk_scan only, when gated != 0):
    /// gated = rms(o) * norm * sigmoid(z) word for word as flash_gdn_gate
    /// computes it from o, written instead of o. z/gated are token-major
    /// [rows][heads][128] like out; norm holds 128 FP32 weights.
    pub z: u64, pub norm: u64, pub gated: u64, pub epsilon: f32, pub spare: u32,
    /// Fused value convolution (flash_next_gdn_chunk_scan only, when x != 0):
    /// v rows are the causal 4-tap conv + SiLU of the call's projection rows
    /// x [rows][channels] from the conv call's first row (weights w: BF16
    /// [channels][4], history: FP32 [channels][4] for taps before that row);
    /// this call's rows start at x row x_start. Word for word as
    /// flash_gdn_conv_prepare_tokens computes them at precision 0; v unused.
    pub x: u64, pub w: u64, pub history: u64, pub channels: u32, pub x_start: u32,
    /// Fused q/k/gate prepare (flash_next_gdn_chunk_prep only, when ga != 0, with
    /// x/w/history/channels/x_start set): prep convolves, L2-normalizes and
    /// writes this call's q/k rows (key group head h0) and beta/g from the gate
    /// projections ga/gb [rows][heads] and A_log/dt_bias (BF16 [heads]), word for
    /// word as flash_gdn_conv_prepare_tokens_kh; the prepare launch is skipped.
    pub ga: u64, pub gb: u64, pub alog: u64, pub dt: u64,
    /// Packed out-projection input (flash_next_gdn_chunk_scan_cv with the fused
    /// gate, when pk != 0): instead of the FP32 gated row, write its split2_pair
    /// hi/lo BF16 words into the A-fragment tile flash_bf16_input would build
    /// from it (rows of the projection call: this call's rows start at row
    /// pk_row0 of pk_tokens), word for word the same.
    pub pk: u64, pub pk_tokens: u32, pub pk_row0: u32,
}
/// Dynamic shared bytes of flash_next_gdn_chunk_scan: S [128][128], v' rows
/// [64][128], e/beta/f/gamma [4][64] and the state pointer (16 bytes); the kernel
/// asserts its layout matches.
pub const CHUNK_SCAN_BYTES: u32 = (128 * 128 + 64 * 128 + 4 * 64) * 4 + 16;
/// Calls of at most this many rows run flash_next_gdn_chunk_rows (the decode row
/// function per row, S column in registers) instead of prep + scan; the bytes
/// are the same either way (kernel-bench split checks).
pub const CHUNK_ROWS_LIMIT: usize = 8;
/// Static shared bytes of flash_next_gdn_chunk_rows and _slot (launch with 0
/// dynamic bytes): kd [64][128] (the slot kernel stages tail keys there as
/// [128][65]), q_i and k_i [128], A_ij, P_ij and f_j [64].
pub const CHUNK_ROW_BYTES: u32 = (128 * 65 + 2 * 128 + 3 * 64) * 4;
/// FP32 words of one layer's chunk-form GDN state: S [heads][128][128], then the
/// open chunk's tail: key rows [keys][64][128], v' rows [heads][64][128] and
/// gamma [heads][64]. Rows at or past base % 64 are stale and never read.
pub const fn chunk_state_floats(heads: usize, keys: usize) -> usize {
    heads * 128 * 128 + keys * 64 * 128 + heads * 64 * 128 + heads * 64
}
