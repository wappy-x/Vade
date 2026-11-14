//! Host-testable ownership arithmetic. No allocation or operator policy changes.
use engine_model::qwen3_5::Config;
pub const PREFIX_SNAPSHOTS: usize = 4;

pub fn pool_prompt(fixed: bool, _drafting: bool, _active: Option<usize>) -> bool {
    fixed
}

/// Non-serving callers may own a full prompt executor; serving always pools it.
pub fn private_rows(pooled: bool) -> usize {
    if pooled { 16 } else { 4096 }
}

pub fn workspace(c: &Config, rows: usize, context: usize, model_bound: usize) -> Option<usize> {
    if context == 0 || context as u64 > c.max_positions { return None; }
    // Two spans: live scores plus their replacement during growth.
    // Score spans hold one 256-row canonical tile at most.
    let unused = (c.max_positions as usize - context).checked_mul(rows.min(256))?
        .checked_mul(c.query_heads as usize)?.checked_mul(8)?;
    model_bound.checked_sub(unused)
}

pub fn host_logits(c: &Config, rows: usize) -> Option<usize> {
    rows.checked_add(1)?.checked_mul(c.vocab as usize)?.checked_mul(8)
}

/// Context caps KV capacity; row capacity caps the private executor and readback.
/// Prompt storage is a separate shared owner when the serving adapter pools it.
pub fn request(c: &Config, tokens: usize, ceiling: usize, rows: usize,
    scratch: usize, workspace: usize, draft: bool, draft_bytes: usize) -> Option<usize> {
    request_with_retention(c,tokens,ceiling,rows,scratch,workspace,draft,draft_bytes,true)
}
/// Unretained sessions never allocate the bounded prefix snapshot ring.
pub fn request_with_retention(c: &Config, tokens: usize, ceiling: usize, rows: usize,
    scratch: usize, workspace: usize, draft: bool, draft_bytes: usize, retain: bool) -> Option<usize> {
    if tokens == 0 || tokens > ceiling || ceiling as u64 > c.max_positions { return None; }
    let capacity = tokens.checked_next_power_of_two()?.max(64).min(ceiling);
    let states = (c.state_bytes() as usize).checked_mul((if draft { 17 } else { 2 }) + if retain { PREFIX_SNAPSHOTS } else { 0 })?;
    let pages = c.layers.iter().filter(|&&k| k == engine_model::flash::LayerKind::SparseAttention)
        .count().checked_mul(8 * 1024 * 1024)?;
    let owners = [
        scratch,
        states,
        capacity.checked_mul(c.kv_bytes_per_token() as usize)?,
        pages,
        workspace,
        capacity.checked_mul(32)?,
        host_logits(c, rows.min(256))?,
        1024 * 1024,
        draft_bytes,
    ];
    owners.into_iter().try_fold(0usize, usize::checked_add)
}
