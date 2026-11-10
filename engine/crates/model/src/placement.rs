//! Pure ownership calculations in global model coordinates, independent of links.
use std::ops::Range;

/// Divide blocks contiguously, rotating remainder ownership by item ID.
/// Callers supply a positive rank count and a rank below it, as in RankPlan.
/// Ring and path topologies use the same ownership; only transport differs.
pub fn rotating_blocks(blocks: u64, rank: u64, ranks: u64, item: u64) -> Range<u64> {
    let base = blocks / ranks;
    let extra = blocks % ranks;
    let first = item % ranks;
    let before_wrap = extra.min(ranks - first);
    let wrapped = extra - before_wrap;
    let owned_before =
        rank.min(first + before_wrap) - rank.min(first) + rank.min(wrapped);
    let start = rank * base + owned_before;
    let owns_extra = (first..first + before_wrap).contains(&rank) || rank < wrapped;
    start..start + base + u64::from(owns_extra)
}
