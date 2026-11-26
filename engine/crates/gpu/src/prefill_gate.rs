//! Shared ABI and host validation for fused prefill message-gate kernels (`PREFILL_GATE_FUSE`, kind 123).
//!
//! Two sub-operations selected by payload word `PG_OP_WORD` (0):
//! - `PG_OP_MSG_META` (0): replaces `MIX_LIST_NORM_HALF` at `Token::enter` by computing only the compact
//!   `msg_weights[asked, rows, kept]` and `msg_nscales[asked, rows]` (2.36 MB instead of 2,718 MB per pass at 4K).
//! - `PG_OP_FULL` (1): fuses `GateDown + GateOuter + GateBlend + RoundedNorm` for `1..=3` gates in one `BM = 16`
//!   full-`HMMA.16816` launch reading lazy sources `S_0..S_{kept-1}` directly (zero intermediate `messages`,
//!   `msg_normed_half`, `inner`, or `GateForget/GateGain` global buffers), bit-for-bit identical to the separate path.
//!
//! Shared by path with `kernels-inkling`; `no_std`-compatible.

pub const PG_OP_WORD: usize = 0;
pub const PG_OP_MSG_META: u64 = 0;
pub const PG_OP_FULL: u64 = 1;

pub const PG_BM: usize = 8;
pub const PG_K_MAX: usize = 64;
pub const PG_KEPT_MAX: usize = 4;
pub const PG_GATES_MAX: usize = 3;
pub const PG_DIM_MAX: usize = 2432; // 19 x 128-element RoundedNorm chunks (11 SMEM + 8 reg warps)
pub const PG_META_THREADS: usize = 256; // 8 warps
pub const PG_FULL_THREADS: usize = 256; // 8 warps (2 blocks/SM @ <=128 regs, 47,424 B SMEM)
pub const PG_FLAG_HALF_OUT: u64 = 1;
/// First payload u64 word of the per-gate descriptor table in `PG_OP_FULL` (`PG_GATES_MAX * PG_GATE_STRIDE = 24` words).
pub const PG_GATE_AT: usize = 12;
pub const PG_GATE_STRIDE: usize = 8;

/// Rows batched per CTA in `PG_OP_MSG_META` so each query row read from L2 is reused across multiple token rows
/// while keeping total SMEM <= 47,424 B (2 blocks/SM on SM121a).
pub const fn msg_meta_batch_rows(kept: usize) -> usize {
    if kept <= 2 { 2 } else { 1 }
}

/// Grid X block count for `PG_OP_MSG_META`.
pub const fn msg_meta_blocks(rows: usize, kept: usize) -> u32 {
    let batch = msg_meta_batch_rows(kept);
    rows.div_ceil(batch) as u32
}

/// True when `PG_OP_MSG_META` can execute this shape.
pub const fn msg_meta_admitted(rows: usize, asked: usize, kept: usize, dim: usize, threads: usize) -> bool {
    rows >= 1
        && rows <= 65_536
        && asked >= 1
        && asked <= 256
        && kept >= 1
        && kept <= PG_KEPT_MAX
        && dim >= 128
        && dim <= PG_DIM_MAX
        && dim % 128 == 0
        && threads == PG_META_THREADS
}

/// Grid X block count for `PG_OP_FULL` (interleaved `(r_tile, gate)` so sibling gates share L2 on `S_0..S_{kept-1}`).
pub const fn full_blocks(rows: usize, count: usize) -> u32 {
    (rows.div_ceil(PG_BM) * count) as u32
}

/// True when `PG_OP_FULL` can execute this shape.
pub const fn full_admitted(rows: usize, dim: usize, k: usize, count: usize, kept: usize, threads: usize) -> bool {
    rows >= 1
        && rows <= 65_536
        && dim >= 128
        && dim <= PG_DIM_MAX
        && dim % 128 == 0
        && k >= 16
        && k <= PG_K_MAX
        && k % 16 == 0
        && count >= 1
        && count <= PG_GATES_MAX
        && kept >= 1
        && kept <= PG_KEPT_MAX
        && threads == PG_FULL_THREADS
}

#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy)]
pub struct Span {
    pub address: u64,
    pub bytes: u64,
}

#[cfg(not(target_arch = "nvptx64"))]
fn check_span(name: &str, s: Span, need_bytes: u64, align: u64) -> Result<(), String> {
    if s.address == 0 || !s.address.is_multiple_of(align) {
        return Err(format!("prefill gate: {name} is null or unaligned to {align} bytes"));
    }
    if s.bytes < need_bytes {
        return Err(format!("prefill gate: {name} has {} bytes, need {need_bytes}", s.bytes));
    }
    if s.address.checked_add(s.bytes).is_none() {
        return Err(format!("prefill gate: {name} span overflows address space"));
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
fn disjoint(a: Span, b: Span) -> bool {
    a.address + a.bytes <= b.address || b.address + b.bytes <= a.address
}

#[cfg(not(target_arch = "nvptx64"))]
#[allow(clippy::too_many_arguments)]
pub fn check_msg_meta(
    rows: usize,
    asked: usize,
    dim: usize,
    scale: f32,
    eps: f32,
    queries: Span,
    sources: &[Span],
    weights_out: Span,
    nscales_out: Span,
) -> Result<(), String> {
    let kept = sources.len();
    if !msg_meta_admitted(rows, asked, kept, dim, PG_META_THREADS) {
        return Err(format!("prefill gate msg_meta: shape rows={rows} asked={asked} kept={kept} dim={dim} not admitted"));
    }
    if !scale.is_finite() || !(eps > 0.0 && eps.is_finite()) {
        return Err("prefill gate msg_meta: non-finite scale or non-positive eps".into());
    }
    let q_bytes = (asked * dim * 4) as u64;
    let s_bytes = (rows * dim * 4) as u64;
    let w_bytes = (asked * rows * kept * 4) as u64;
    let n_bytes = (asked * rows * 4) as u64;
    check_span("queries", queries, q_bytes, 16)?;
    for &src in sources {
        check_span("source", src, s_bytes, 16)?;
    }
    check_span("weights_out", weights_out, w_bytes, 4)?;
    check_span("nscales_out", nscales_out, n_bytes, 4)?;
    if !disjoint(weights_out, nscales_out) {
        return Err("prefill gate msg_meta: weights_out and nscales_out overlap".into());
    }
    for out in [weights_out, nscales_out] {
        if !disjoint(out, queries) {
            return Err("prefill gate msg_meta: output overlaps queries".into());
        }
        for &src in sources {
            if !disjoint(out, src) {
                return Err("prefill gate msg_meta: output overlaps source".into());
            }
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "nvptx64"))]
#[derive(Clone, Copy)]
pub struct GateSpans {
    pub w_down: Span,
    pub w_outer: Span,
    pub kept_val: Span,
    pub kept_normed_half: Span,
    pub msg_weights: Span,
    pub msg_nscales: Span,
    pub norm_w: Option<Span>,
    pub output: Span,
}

#[cfg(not(target_arch = "nvptx64"))]
pub fn check_full(
    rows: usize,
    dim: usize,
    k: usize,
    eps: f32,
    half_out: bool,
    sources: &[Span],
    gates: &[GateSpans],
) -> Result<(), String> {
    let (count, kept) = (gates.len(), sources.len());
    if !full_admitted(rows, dim, k, count, kept, PG_FULL_THREADS) {
        return Err(format!("prefill gate full: shape rows={rows} dim={dim} k={k} count={count} kept={kept} not admitted"));
    }
    if !(eps > 0.0 && eps.is_finite()) {
        return Err("prefill gate full: eps must be positive finite".into());
    }
    let s_bytes = (rows * dim * 4) as u64;
    let wd_bytes = (k * 2 * dim * 2) as u64;
    let wo_bytes = (2 * dim * k * 2) as u64;
    let kh_bytes = (rows * (dim / 2) * 4) as u64;
    let mw_bytes = (rows * kept * 4) as u64;
    let mn_bytes = (rows * 4) as u64;
    let nw_bytes = (dim * 4) as u64;
    let out_bytes = if half_out { kh_bytes } else { s_bytes };
    let out_align = if half_out { 16 } else { 4 };
    for &src in sources {
        check_span("source", src, s_bytes, 16)?;
    }
    for (i, g) in gates.iter().enumerate() {
        check_span("w_down", g.w_down, wd_bytes, 16)?;
        check_span("w_outer", g.w_outer, wo_bytes, 16)?;
        check_span("kept_val", g.kept_val, s_bytes, 16)?;
        check_span("kept_normed_half", g.kept_normed_half, kh_bytes, 16)?;
        check_span("msg_weights", g.msg_weights, mw_bytes, 4)?;
        check_span("msg_nscales", g.msg_nscales, mn_bytes, 4)?;
        if let Some(nw) = g.norm_w {
            check_span("norm_w", nw, nw_bytes, 16)?;
        }
        check_span("output", g.output, out_bytes, out_align)?;
        for &src in sources {
            if !disjoint(g.output, src) {
                return Err(format!("prefill gate full: output {i} overlaps source"));
            }
        }
        for other in gates {
            for inp in [
                other.w_down,
                other.w_outer,
                other.kept_val,
                other.kept_normed_half,
                other.msg_weights,
                other.msg_nscales,
            ] {
                if !disjoint(g.output, inp) {
                    return Err(format!("prefill gate full: output {i} overlaps input"));
                }
            }
            if other.norm_w.is_some_and(|nw| !disjoint(g.output, nw)) {
                return Err(format!("prefill gate full: output {i} overlaps norm_w"));
            }
        }
        for other in &gates[i + 1..] {
            if !disjoint(g.output, other.output) {
                return Err(format!("prefill gate full: output {i} overlaps another output"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_rules() {
        assert!(msg_meta_admitted(4096, 48, 2, 2304, PG_META_THREADS));
        assert!(msg_meta_admitted(1, 48, 4, 2432, PG_META_THREADS));
        assert!(!msg_meta_admitted(4096, 48, 5, 2304, PG_META_THREADS));
        assert!(!msg_meta_admitted(4096, 48, 2, 2304, 128));
        assert!(full_admitted(4096, 2304, 64, 3, 2, PG_FULL_THREADS));
        assert!(full_admitted(17, 1280, 64, 1, 4, PG_FULL_THREADS));
        assert!(!full_admitted(4096, 2304, 80, 3, 2, PG_FULL_THREADS));
        assert!(!full_admitted(4096, 2304, 64, 4, 2, PG_FULL_THREADS));
        assert_eq!(msg_meta_blocks(4096, 2), 2048);
        assert_eq!(msg_meta_blocks(4096, 3), 4096);
        assert_eq!(full_blocks(4096, 3), 1536);
    }
}
