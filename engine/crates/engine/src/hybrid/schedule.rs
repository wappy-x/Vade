//! Scheduling changes row placement, never an operator's arithmetic.
pub const PROMPT_ROWS: usize = 256;
/// FixedLinear ranges retain the proven 1024-row projection shape.
pub const PROMPT_GROUP: usize = 4 * PROMPT_ROWS;
/// Routed experts and ordered GDN scan share a larger serialized prompt pass.
/// Attention and logits retain their canonical 256-row tiles.
pub const ROUTED_GROUP: usize = 4096;
pub fn ranges(rows: usize) -> impl Iterator<Item=std::ops::Range<usize>> { (0..rows).step_by(PROMPT_GROUP).map(move |start| start..start+PROMPT_GROUP.min(rows-start)) }
pub fn tokens(lengths: &[usize]) -> Vec<(usize,usize)> { (0..lengths.iter().copied().max().unwrap_or(0)).flat_map(|step| lengths.iter().enumerate().filter_map(move |(session,&length)| (step<length).then_some((session,step)))).collect() }

/// Router consensus may share the FF download only for decode/verify rows.
pub fn fused_router_boundary(distributed: bool, rows: usize, verify_rows: usize) -> bool {
    distributed && rows <= verify_rows
}
