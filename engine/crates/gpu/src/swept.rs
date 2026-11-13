//! The tuples the device sweeps of lane P's reading rows run (`kernel-bench --reading-probe list-*` iterates these lists and nothing
//! else), so that a host test can compare them with the predicates of `prompt_admission` (the rule: every admitted tuple is swept).
//! Each list holds the model's shape and every token count from 1 to GROUP_MAX (1 to MOVE_MAX for the two movement kernels; every
//! row count from 128 to GROUP_MAX for the mHC project). Host and bench only; the kernels never see this file.
// Self-contained: the bench includes this file by path and has no access to the gpu crate. A host test pins these to the
// constants of `prompt_admission`.
pub const GROUP_MAX:usize=256;
pub const GROUP_BIG:[usize;3]=[512,768,1024];
pub fn group_tokens()->impl Iterator<Item=usize> {(1..=GROUP_MAX).chain(GROUP_BIG)}
/// Most tokens of a group for the two movement kernels and the transpose (they copy words; the largest prompt group).
pub const MOVE_MAX:usize=1024;
pub const MHC_MIN_ROWS:usize=128;
pub const ROUTED:usize=8;

/// Permute: (rows, cols, source rows) for the expert inputs of a group of `tokens`.
pub fn permute_cases()->Vec<(usize,usize,usize)> {(1..=MOVE_MAX).flat_map(|m|[2048usize,4096].into_iter().map(move |cols|(m*ROUTED,cols,m))).collect()}
/// Place: (tokens, local rows, ranks).
pub fn place_cases()->Vec<(usize,usize,usize)> {(1..=MOVE_MAX).flat_map(|m|[1024usize,2048].into_iter().map(move |d|(m,d,2))).collect()}
/// Head counts of the recurrent prompt kernels (`prompt_admission::recurrent_heads`): 64 on one box, 32 a rank on two.
pub const RECURRENT_HEADS:[usize;2]=[32,64];
/// Wrapped Prepare (true) and GatedNorm (false): (tokens, heads, dim, is_prepare).
pub fn prepare_cases()->Vec<(usize,usize,usize,bool)> {
    group_tokens().flat_map(|m|RECURRENT_HEADS.into_iter().flat_map(move |h|[true,false].into_iter().map(move |p|(m,h,128,p)))).collect()
}
/// Wrapped Route: (tokens, experts, chosen, normalized).
pub fn route_cases()->Vec<(usize,usize,usize,usize)> {group_tokens().map(|m|(m,288,ROUTED,1)).collect()}
/// Wrapped ExpertFold: (tokens, routed rows, width, shared).
pub fn fold_cases()->Vec<(usize,usize,usize,usize)> {group_tokens().map(|m|(m,ROUTED,4096,1)).collect()}
/// ConvRows: (tokens, heads, dim, taps).
pub fn conv_cases()->Vec<(usize,usize,usize,usize)> {group_tokens().flat_map(|m|RECURRENT_HEADS.into_iter().map(move |h|(m,h,128,4))).collect()}
/// mHC project tile: (rows, streams, hidden).
pub fn mhc_cases()->Vec<(usize,usize,usize)> {(MHC_MIN_ROWS..=GROUP_MAX).chain(GROUP_BIG).map(|r|(r,4,4096)).collect()}

/// PoolChannels (the pooled-latent channels of one pool, one unit a channel): the dims and taps the device sweep covers, and the
/// largest row (slot) count. The predicate `causal_ops::fast_admitted` admits exactly these dims and taps, rows 1 to the maximum
/// (an interval: the sweep takes both edges, every value next to a tap multiple, and the values below).
pub const POOL_CHANNEL_DIMS:[usize;8]=[1,2,31,32,33,127,128,129];
pub const POOL_CHANNEL_TAPS:[usize;9]=[1,2,3,4,5,8,127,128,129];
pub const POOL_CHANNEL_ROWS_MAX:usize=16384;
/// The row counts swept for one tap count.
pub fn pool_channel_rows(taps:usize)->Vec<usize> {
    let mut rows=vec![1usize,2,3,4,5,31,32,33,127,128,129,1000,2051,4096,8191,8192,8193,16383,POOL_CHANNEL_ROWS_MAX,
        taps.saturating_sub(1),taps,taps+1,2*taps-1,2*taps,2*taps+1,3*taps+2,64*taps];
    rows.retain(|&r|(1..=POOL_CHANNEL_ROWS_MAX).contains(&r));rows.sort_unstable();rows.dedup();rows
}

// ---- Axis lists of the byte sweep of `mx_rows_tile` (Grok's finding of 2026-11-06). ----
// Each admitted dimension is swept at EVERY value of its admitted range (the lists below hold all of them); the cross products are covered by
// full grids of two axes at a small third, the tile-edge grids, and all eight corners. A host test in `mx_rows` / `plain_tile` walks the
// predicate over each axis and fails if an admitted value is missing from its list here.
pub const MX_ROWS_MAX:usize=8192;
pub const MX_COLS_MAX:usize=8192;
pub const MX_TOKENS_MAX:usize=1024;
pub fn mx_rows_axis()->Vec<usize> {(1..=MX_ROWS_MAX).collect()}
/// The one admitted column count past `MX_COLS_MAX`: the dense down projection of layers 0 to 2.
pub const MX_COLS_DENSE:usize=12288;
pub fn mx_cols_axis()->Vec<usize> {(1..=MX_COLS_MAX/32).map(|i|i*32).chain([MX_COLS_DENSE]).collect()}
pub fn mx_tokens_axis()->Vec<usize> {(1..=MX_TOKENS_MAX).collect()}
/// The eight corners (rows, cols, tokens) of the admitted box, and the four of the dense column count.
pub fn mx_corners()->Vec<(usize,usize,usize)> {
    let mut v=vec![];for r in [1usize,MX_ROWS_MAX] {for c in [32usize,MX_COLS_MAX,MX_COLS_DENSE] {for t in [1usize,MX_TOKENS_MAX] {v.push((r,c,t));}}}v
}

// ---- Axis lists of the byte sweep of `plain_rows_narrow` (Astra's finding of 2026-11-06). ----
// Every admitted value of each axis is on a list below; two-axis grids at a small third, and the eight corners, cover the cross products.
// A host test in `plain_tile` walks the predicate over each axis and fails if an admitted value is missing here.
pub const NARROW_ROWS_MAX:usize=192;
pub const NARROW_TOKENS_MAX:usize=1024;
pub const NARROW_COLS_MAX:usize=4096;
pub fn narrow_rows_axis()->Vec<usize> {(1..=NARROW_ROWS_MAX).collect()}
pub fn narrow_tokens_axis()->Vec<usize> {(1..=NARROW_TOKENS_MAX).collect()}
pub fn narrow_cols_axis()->Vec<usize> {(1..=NARROW_COLS_MAX/4).map(|i|i*4).collect()}
pub fn narrow_corners()->Vec<(usize,usize,usize)> {
    let mut v=vec![];for r in [1usize,NARROW_ROWS_MAX] {for c in [4usize,NARROW_COLS_MAX] {for t in [1usize,NARROW_TOKENS_MAX] {v.push((r,c,t));}}}v
}

// ---- The pairs the byte sweep of `transpose_smem` walks (KV expansion, Astra's findings of 10:37). ----
// The admitted set of `prompt_admission::transpose_admitted` IS this list: the model's matrix row counts (whole 4 x 4 tiles, at most 32,768) times the token
// counts 4, 8, .. 256 and the whole 32-token tiles 288, 320, .. 1024. A host test in `prompt_admission` checks it both ways over a universe around it; the device sweep byte-compares every pair, with and
// without the finite check, against the generic kernel.
pub const TRANSPOSE_ROWS:[usize;21]=[4,12,16,24,32,64,128,144,256,288,512,768,1024,1536,2048,4096,6144,8192,12288,16384,32768];
pub const TRANSPOSE_COLS_MAX:usize=256;
pub const TRANSPOSE_WIDE_STEP:usize=32;
pub fn transpose_pairs()->Vec<(usize,usize)> {
    TRANSPOSE_ROWS.iter().flat_map(|&r|(1..=TRANSPOSE_COLS_MAX/4).map(move |i|(r,i*4)).chain((TRANSPOSE_COLS_MAX/TRANSPOSE_WIDE_STEP+1..=MOVE_MAX/TRANSPOSE_WIDE_STEP).map(move |i|(r,i*TRANSPOSE_WIDE_STEP)))).collect()
}
