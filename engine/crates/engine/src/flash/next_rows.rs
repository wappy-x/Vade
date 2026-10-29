//! Host-side row admission and deterministic router order, shared with device execution.
#[cfg(any(target_os = "linux", test))]
use std::ops::Range;
/// Shared layer traversal for prompt tiles and world-N independent rows.
pub fn layers(count: usize, mut phase: impl FnMut(usize, bool) -> Result<()>) -> Result<()> {
    for layer in 0..count {
        phase(layer, true)?;
        phase(layer, false)?;
    }
    Ok(())
}
type Result<T> = std::result::Result<T, String>;
/// VADE_POOL_GRAPH_FALLBACK=1: a world-one pooled step may run the same
/// kernels eagerly when a healthy context cannot retain its graph pair.
pub fn pool_graph_fallback_value(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => Err("VADE_POOL_GRAPH_FALLBACK must be unset, 0 or 1".into()),
    }
}
#[cfg(test)]
#[test]
fn pooled_graph_fallback_is_explicit_and_defaults_off() {
    assert_eq!(pool_graph_fallback_value(None), Ok(false));
    assert_eq!(pool_graph_fallback_value(Some("0")), Ok(false));
    assert_eq!(pool_graph_fallback_value(Some("1")), Ok(true));
    for v in ["", "2", "true", " 1"] { assert!(pool_graph_fallback_value(Some(v)).is_err()); }
}
/// A retained workspace retries refused capture infrequently, independently
/// for each arm. Replacing the workspace starts a new address/capture epoch.
#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Default)]
pub(crate) struct CaptureBackoff { remaining: usize, refusals: u32 }
#[cfg(any(target_os = "linux", test))]
impl CaptureBackoff {
    pub(crate) fn attempt(&mut self) -> bool {
        if self.remaining == 0 { true } else { self.remaining -= 1; false }
    }
    pub(crate) fn refused(&mut self) {
        self.refusals = self.refusals.saturating_add(1);
        self.remaining = 64usize << self.refusals.saturating_sub(1).min(6);
    }
    pub(crate) fn captured(&mut self) { *self = Self::default(); }
}
#[cfg(test)]
#[test]
fn refused_capture_backoff_skips_sync_and_capture_until_retry_boundary() {
    let mut retry = CaptureBackoff::default();
    for refusal in 0..9 {
        assert!(retry.attempt()); retry.refused();
        let wait = 64usize << refusal.min(6);
        for _ in 0..wait { assert!(!retry.attempt()); }
    }
    let mut independent_arm = CaptureBackoff::default(); assert!(independent_arm.attempt());
    retry.captured(); assert!(retry.attempt()); retry.refused();
    for _ in 0..64 { assert!(!retry.attempt()); }
    assert!(retry.attempt());
}
/// Neither graph has executed yet. Release a partial pair before eager work;
/// an execution/cleanup error from capture still aborts the step.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn capture_pair<G>(mut capture: impl FnMut(bool) -> Result<Option<G>>)
    -> Result<Option<(G, G)>>
{
    let Some(head) = capture(true)? else { return Ok(None) };
    Ok(capture(false)?.map(|tail| (head, tail)))
}
#[cfg(test)]
#[test]
fn capture_pair_releases_partial_storage_and_preserves_fatal_errors() {
    use std::cell::RefCell;
    struct Graph<'a>(&'a RefCell<Vec<&'static str>>);
    impl Drop for Graph<'_> { fn drop(&mut self) { self.0.borrow_mut().push("drop"); } }
    let events = RefCell::new(Vec::new());
    let pair = capture_pair(|head| {
        events.borrow_mut().push(if head { "head" } else { "tail" });
        Ok(head.then(|| Graph(&events)))
    }).unwrap();
    assert!(pair.is_none());
    assert_eq!(*events.borrow(), ["head", "tail", "drop"]);
    events.borrow_mut().clear();
    let failed = capture_pair(|head| {
        events.borrow_mut().push(if head { "head" } else { "tail" });
        if head { Ok(Some(Graph(&events))) } else { Err("poisoned context".into()) }
    });
    assert_eq!(failed.err().as_deref(), Some("poisoned context"));
    assert_eq!(*events.borrow(), ["head", "tail", "drop"]);
    let mut calls = 0;
    assert!(capture_pair::<()>(|_| { calls += 1; Ok(None) }).unwrap().is_none());
    assert_eq!(calls, 1);
    assert_eq!(capture_pair(|head| Ok(Some(if head { 1 } else { 2 }))).unwrap(), Some((1, 2)));
}
/// Fund every reachable padded graph width and all admitted sessions together: `needed` bytes
/// per slot, and one `full` session in the pool (the startup check), since the shared graph
/// storage grows with the slot count. The one-slot fallback preserves the caller's normal
/// insufficient-budget refusal.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn fit_session_pool(available: usize, needed: usize, full: usize, mut shared: impl FnMut(usize) -> Result<usize>)
    -> Result<(usize, usize, usize)>
{
    for sessions in (1usize..=16).rev() {
        let reserve = shared(sessions)?;
        let pool = available.saturating_sub(reserve);
        if sessions == 1 || full <= pool && needed.checked_mul(sessions).is_some_and(|n| n <= pool) {
            return Ok((sessions, pool, reserve));
        }
    }
    unreachable!()
}
#[cfg(test)]
#[test]
fn session_pool_funds_graph_cliffs_and_refuses_shortfall() {
    let shared = |n: usize| Ok(1000 + n.next_power_of_two() * 60);
    assert_eq!(fit_session_pool(20000, 2257, 2257, shared).unwrap(), (8, 18520, 1480));
    assert_eq!(fit_session_pool(100, 2257, 2257, shared).unwrap(), (1, 0, 1060));
    assert_eq!(fit_session_pool(usize::MAX, usize::MAX, usize::MAX, |_| Ok(0)).unwrap().0, 1);
    // Small chats fund 16 slots (pool 18040), but one full session of 18100 bytes fits only
    // below the 9..16 storage width: the plan takes 8 slots, which pass the startup check.
    assert_eq!(fit_session_pool(20000, 1000, 18000, shared).unwrap(), (16, 18040, 1960));
    assert_eq!(fit_session_pool(20000, 1000, 18100, shared).unwrap(), (8, 18520, 1480));
}
/// Missing owners cannot be reconstructed while published pool addresses remain live.
/// Such a session must be replaced; reset and pooled admission share this check.
/// `layers`: the owned interval (all 48, or a pipeline stage's own layers).
#[cfg(any(target_os = "linux", test))]
pub(crate) fn check_owners<G, Q>(gdn: &[Option<G>], qsa: &[Option<Q>], layers: std::ops::Range<usize>) -> Result<()> {
    if gdn.len() != 48
        || qsa.len() != 48
        || layers.is_empty()
        || layers.end > 48
        || layers.clone().any(|li| {
            if li % 4 == 3 {
                qsa[li].is_none()
            } else {
                gdn[li].is_none()
            }
        })
    {
        return Err("Flash-Next state owner lost after failure; replace session".into());
    }
    Ok(())
}
/// Returned by execution, so readback never re-evaluates a mutable indexer hook.
#[cfg(any(target_os = "linux", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResultSource {
    Pooled,
    Session,
}
#[cfg(any(target_os = "linux", test))]
impl ResultSource {
    pub(crate) fn select(single: bool, sparse: impl FnOnce() -> bool) -> Self {
        if single && sparse() {
            Self::Pooled
        } else {
            Self::Session
        }
    }
    pub(crate) fn select_for_policy(single: bool, permitted: bool, sparse: impl FnOnce() -> bool) -> Self {
        Self::select(single && permitted, sparse)
    }
    pub(crate) fn read<T>(
        self,
        pooled: impl FnOnce() -> Result<T>,
        session: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match self {
            Self::Pooled => pooled(),
            Self::Session => session(),
        }
    }
}
#[cfg(test)]
#[test]
fn filtered_single_rows_during_fairness_holds_do_not_grow_pooled_storage() {
    use std::cell::Cell;
    let storage=Cell::new(3usize); let before=storage.get();
    for _ in 0..128 {
        // A retained drafting path forbids pooling even if all proposals filter
        // down to one target row. Both routing and readback use this decision.
        let route=ResultSource::select_for_policy(true,false,|| true);
        let id=route.read(|| { storage.set(storage.get()+1); Ok(7) },|| Ok(7)).unwrap();
        assert_eq!(id,7); assert_eq!(storage.get(),before);
    }
    assert_eq!(ResultSource::select_for_policy(true,true,|| true),ResultSource::Pooled);
}
/// The widest admitted verifier is sixteen chats with three proposals each.
#[cfg(any(target_os = "linux", test))]
pub(crate) const MAX_HEAD_ROWS: usize = 64;
#[cfg(any(target_os = "linux", test))]
pub(crate) const HEAD_TILE_ROWS: usize = 16;
/// Refuse unsupported widths before preparing input or launching a head kernel.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn head_tiles(count: usize) -> Result<impl Iterator<Item = Range<usize>>> {
    if !(1..=MAX_HEAD_ROWS).contains(&count) {
        return Err("Flash-Next head supports 1..=64 rows in 16-row slices".into());
    }
    let width = if count > HEAD_TILE_ROWS && head32() { 32 } else { HEAD_TILE_ROWS };
    Ok((0..count)
        .step_by(width)
        .map(move |start| start..(start + width).min(count)))
}
/// VADE_MTP_HEAD32=1 (default off): M17..32 share one vocabulary weight pass.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn head32() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VADE_MTP_HEAD32").as_deref() == Ok("1"))
}
/// Scratch belongs to the largest-context session, whose admission funds it.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn workspace_owner(capacities: impl Iterator<Item = usize>) -> Result<usize> {
    capacities
        .enumerate()
        .max_by_key(|&(_, capacity)| capacity)
        .map(|(owner, _)| owner)
        .ok_or_else(|| "empty workspace cohort".into())
}
/// Actual Next GDN owners retain four raw convolution slots per layer.
#[cfg(any(target_os = "linux", test))]
pub(crate) const RECURRENT_BYTES: usize = 36 * (48 * 128 * 128 + (2 * 16 + 48) * 128 * 4) * 4;

/// Session KV + recurrent bytes a rank funds for the layers it owns: `kv` is the
/// loader plan's KV for all twelve QSA layers (every 4th). All 48 owned: exactly
/// `kv + RECURRENT_BYTES`; a pipeline stage pays its own QSA and GDN layers only.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn owned_state_bytes(kv: usize, layers: std::ops::Range<usize>) -> usize {
    let qsa = layers.clone().filter(|li| li % 4 == 3).count();
    let gdn = layers.len() - qsa;
    (kv * qsa).div_ceil(12) + RECURRENT_BYTES / 36 * gdn
}
#[cfg(test)]
#[test]
fn owned_state_bytes_split_whole_model_state_across_stages() {
    let kv = 12 * 32768 * 4608;
    assert_eq!(owned_state_bytes(kv, 0..48), kv + RECURRENT_BYTES);
    for world in [2, 4] {
        let stages = (0..world).map(|r| owned_state_bytes(kv, r * 48 / world..(r + 1) * 48 / world)).collect::<Vec<_>>();
        assert_eq!(stages.iter().sum::<usize>(), kv + RECURRENT_BYTES, "world {world}");
        assert!(stages.iter().all(|&b| b == stages[0]), "world {world}: {stages:?}");
    }
    // An uneven cut (the tensor split's) still sums to the whole model's state.
    assert_eq!(owned_state_bytes(kv, 0..20) + owned_state_bytes(kv, 20..48), kv + RECURRENT_BYTES);
    assert_eq!(owned_state_bytes(kv, 0..24), kv / 2 + RECURRENT_BYTES / 2);
}
/// A stage's own interval must be one of `stages`, which tile the 48 layers in order.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn check_stage_charge(own: std::ops::Range<usize>, stages: &[std::ops::Range<usize>]) -> Result<()> {
    let tiled = stages.first().is_some_and(|s| s.start == 0) && stages.last().is_some_and(|s| s.end == 48)
        && stages.iter().all(|s| s.start < s.end) && stages.windows(2).all(|w| w[0].end == w[1].start);
    if !tiled || !stages.contains(&own) {
        return Err(format!("pipeline stage state {own:?} is not one of the stages {stages:?}"));
    }
    Ok(())
}
/// Per-chat state funding: the costliest stage, so every rank charges the same bytes
/// and admits the same number of chats from the same budget (one stage = 0..48: all).
#[cfg(any(target_os = "linux", test))]
pub(crate) fn stage_charge_bytes(kv: usize, stages: &[std::ops::Range<usize>]) -> usize {
    stages.iter().map(|s| owned_state_bytes(kv, s.clone())).max().unwrap_or(kv + RECURRENT_BYTES)
}
#[cfg(test)]
#[test]
fn uneven_stages_fund_the_same_chat_count_on_every_rank() {
    let kv = 12 * 32768 * 4608;
    // The review's case: split 20 owns 5 QSA + 15 GDN layers in front, 7 + 21 behind.
    let stages = [0..20, 20..48];
    let (front, back) = (owned_state_bytes(kv, 0..20), owned_state_bytes(kv, 20..48));
    assert!(front < back);
    let charge = stage_charge_bytes(kv, &stages);
    assert_eq!(charge, back);
    // Same budget and shared bytes on both ranks: own-stage charges would admit more chats
    // in front than behind (the hello refuses that); the common charge admits equal counts.
    let (budget, shared, fixed) = (10_737_418_240usize, 1 << 30, 300 << 20);
    let chats = |per: usize| (budget - shared) / (per + fixed);
    assert!(chats(front) > chats(back), "{} vs {}", chats(front), chats(back));
    // Each rank passes the same interval list, so front and back charge identical bytes.
    assert_eq!(chats(charge), chats(back));
    assert!(check_stage_charge(0..20, &stages).is_ok() && check_stage_charge(20..48, &stages).is_ok());
    for world in [2, 4] {
        let stages: Vec<_> = (0..world).map(|r| r * 48 / world..(r + 1) * 48 / world).collect();
        // Even stages: the common charge equals each stage's own state (no loss at the default).
        assert!(stages.iter().all(|s| stage_charge_bytes(kv, &stages) == owned_state_bytes(kv, s.clone())));
        assert!(stages.iter().all(|s| check_stage_charge(s.clone(), &stages).is_ok()));
    }
    assert_eq!(stage_charge_bytes(kv, &[0..48]), kv + RECURRENT_BYTES);
    assert!(check_stage_charge(0..20, &[0..24, 24..48]).is_err());
    assert!(check_stage_charge(0..24, &[0..24, 25..48]).is_err());
    assert!(check_stage_charge(0..24, &[]).is_err());
}
/// Prefix checkpoints retain the loader's complete GDN layout, including an open
/// 64-row tail in chunk form. The legacy constant also defines small PLE terms;
/// keep it unchanged and charge the actual state separately for opt-in retention.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn recurrent_bytes() -> usize {
    recurrent_bytes_for_chunk(engine_model::flash::gdn_chunk_form())
}
#[cfg(any(target_os = "linux", test))]
fn recurrent_bytes_for_chunk(chunked: bool) -> usize {
    if chunked {
        36 * (spark_gpu_probe::gdn_params::chunk_state_floats(48, 16)
            + (2 * 16 + 48) * 128 * 4) * 4
    } else { RECURRENT_BYTES }
}
/// The existing session bound funds legacy live/rollback states. A retaining
/// request must fund their chunk tails too; other requests keep their old bound.
#[cfg(any(target_os = "linux", test))]
pub fn prefix_retention_extra_bytes(resident_draft: bool) -> usize {
    (recurrent_bytes() - RECURRENT_BYTES) * if resident_draft { 2 } else { 1 }
}
#[cfg(test)]
#[test]
fn retained_prefix_funds_chunk_state_without_growing_ple_terms() {
    let old = recurrent_bytes_for_chunk(false);
    let actual = recurrent_bytes_for_chunk(true);
    assert_eq!(old, 119_144_448);
    assert_eq!(actual, 195_084_288);
    assert_eq!(actual - old, 75_939_840);
    assert_eq!(2 * (actual - old), 151_879_680);
    assert_eq!(PREFIX_BYTES - RECURRENT_BYTES, 409_600);
}
#[cfg(any(target_os = "linux", test))]
pub(crate) const PREFIX_BYTES: usize = RECURRENT_BYTES + 10240 * (9 + 1) * 4;

/// Widest prompt tile (rows per layer pass), read once per process. Auto by
/// default; VADE_PROMPT_TILE=4096 keeps fixed-width tiles (each MoE layer streams its experts once per tile: cold 8K -0.28 s, 32K
/// -1.46 s vs 2048); VADE_PROMPT_TILE=2048 or 8192 selects the others. Every tile runs in
/// the model's one prompt workspace, funded once per server, so the tile adds nothing to
/// a session's admission. Every kernel is row-exact, so outputs must not depend on the
/// tile (chunked == unchunked; verified per tile value). VADE_PROMPT_TILE=auto funds the
/// 8192 tile and picks each prompt's tile with prompt_tile_for.
pub fn prompt_tile() -> usize {
    static TILE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TILE.get_or_init(|| match std::env::var("VADE_PROMPT_TILE").ok().as_deref() {
        Some("2048") => 2048,
        None | Some("8192") | Some("auto") => 8192,
        _ => 4096,
    })
}
/// Auto by default; VADE_PROMPT_TILE=4096 disables auto, read once per process.
pub fn prompt_tile_auto() -> bool {
    static AUTO: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AUTO.get_or_init(|| matches!(std::env::var("VADE_PROMPT_TILE").ok().as_deref(), None | Some("auto")))
}
/// Prompts ending at or beyond this many tokens take 8192-row tiles under auto.
pub const AUTO_TILE_FROM: usize = 8192;
/// Rows per tile for prompt work ending at token `end` in a session admitted at `chunk`:
/// `chunk` itself, except under VADE_PROMPT_TILE=auto, where a prompt ending below
/// AUTO_TILE_FROM keeps the 4096 tile (main's tiles and launches) and a longer one runs
/// 8192-row tiles (cold 8K / 32K). Rows are exact per tile, so outputs do not change.
pub fn prompt_tile_for(end: usize, chunk: usize) -> usize {
    if prompt_tile_auto() { auto_tile(end, chunk) } else { chunk }
}
fn auto_tile(end: usize, chunk: usize) -> usize {
    chunk.min(if end >= AUTO_TILE_FROM { 8192 } else { 4096 })
}
#[cfg(test)]
#[test]
fn auto_tile_keeps_short_prompts_on_4096() {
    for (end, tile) in [(1, 4096), (1024, 4096), (3072, 4096), (8191, 4096), (8192, 8192), (32768, 8192)] {
        assert_eq!(auto_tile(end, 8192), tile);
    }
    // Explicit smaller admissions stay as they were.
    assert_eq!(auto_tile(32768, 2048), 2048);
    // A cold 6K prompt runs main's 4096 + 2048 tiles; a cold 8K one 8192-row tile.
    assert_eq!(prompt_rows(0, 6144, auto_tile(6144, 8192)), 4096);
    assert_eq!(prompt_rows(4096, 2048, auto_tile(6144, 8192)), 2048);
    assert_eq!(prompt_rows(0, 8192, auto_tile(8192, 8192)), 8192);
    if std::env::var_os("VADE_PROMPT_TILE").is_none() {
        assert_eq!(prompt_tile_for(32768, 4096), 4096);
        assert_eq!(prompt_tile_for(1024, 4096), 4096);
    }
}

/// VADE_VERIFY_ROWS: rows in one VADE_MTP_BATCH verify tick (unset = 16 here; the world-one
/// executable resolves unset to 32 through verify_rows_setting). Explicit 16 or 32; other values
/// are errors. A 32-row tick verifies up to 32 rows in one pass instead of groups of 16, and such
/// a pass stages its PLE rows at once (in 16-row pieces).
pub fn verify_rows() -> Result<usize> {
    verify_rows_setting().map(|rows| rows.unwrap_or(16))
}
/// VADE_VERIFY_ROWS as set: None when unset, else 16 or 32.
pub fn verify_rows_setting() -> Result<Option<usize>> {
    static ROWS: std::sync::OnceLock<Result<Option<usize>>> = std::sync::OnceLock::new();
    ROWS.get_or_init(|| match std::env::var("VADE_VERIFY_ROWS") {
        Err(std::env::VarError::NotPresent) => Ok(None),
        Ok(v) => verify_rows_value(&v).map(Some),
        Err(e) => Err(format!("VADE_VERIFY_ROWS: {e}")),
    })
    .clone()
}
fn verify_rows_value(value: &str) -> Result<usize> {
    match value {
        "16" => Ok(16),
        "32" => Ok(32),
        v => Err(format!("VADE_VERIFY_ROWS must be unset, 16 or 32 (got {v:?})")),
    }
}
#[cfg(test)]
#[test]
fn verify_rows_accepts_only_the_two_tick_widths() {
    assert_eq!(verify_rows_value("16"), Ok(16));
    assert_eq!(verify_rows_value("32"), Ok(32));
    for v in ["", "0", "8", "17", "64", " 32"] {
        assert!(verify_rows_value(v).unwrap_err().contains("VADE_VERIFY_ROWS"), "{v:?}");
    }
}

/// Scratch width funded for a session and every cohort it can own.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn reservation_rows(capacity: usize, chunk: usize) -> Result<usize> {
    if !(1..=262144).contains(&capacity) || !(1..=prompt_tile()).contains(&chunk) {
        return Err("Flash-Next capacity/chunk outside limits".into());
    }
    // A cohort can contain more rows than one short session can consume.
    Ok(chunk.max(64))
}
#[cfg(test)]
#[test]
fn chunk_reservation_covers_public_tiles_and_cohorts() {
    for capacity in [1, 16, 512, 32768, 262144] {
        for chunk in [1, 32, 512, 1024, 2048] {
            let funded = reservation_rows(capacity, chunk).unwrap();
            assert!(funded >= chunk);
            assert!(funded >= prompt_rows(0, capacity.min(2048), chunk));
        }
    }
    assert_eq!(reservation_rows(32768, 512).unwrap(), 512);
    assert_eq!(reservation_rows(32768, 2048).unwrap(), 2048);
    for (capacity, chunk) in [(0, 512), (262145, 512), (32768, 0), (32768, prompt_tile() + 1)] {
        assert!(reservation_rows(capacity, chunk).is_err());
    }
}
/// Preserve the selected scheduling budget while respecting every owner's funding.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn prefill_budget(
    chunks: impl Iterator<Item = (usize, usize)>,
    live: usize,
) -> Result<usize> {
    if !(1..=16).contains(&live) {
        return Err("invalid prefill cohort width".into());
    }
    let tile = prompt_tile();
    let (mut selected, mut funded) = (0, tile);
    for (chunk, limit) in chunks {
        if chunk == 0 || chunk > limit || limit > tile { return Err("chunk exceeds admission".into()); }
        selected = selected.max(chunk);
        funded = funded.min(limit.max(64));
    }
    if selected == 0 { return Err("empty prefill cohort".into()); }
    Ok(selected.clamp(512, tile).min(funded) / live)
}
#[cfg(test)]
#[test]
fn prefill_budget_preserves_default_and_funds_larger_chunks() {
    assert_eq!(prefill_budget([(512,2048);8].into_iter(),8).unwrap(),64);
    assert_eq!(prefill_budget([(512,512);8].into_iter(),8).unwrap(),64);
    assert_eq!(prefill_budget([(2048,2048);8].into_iter(),8).unwrap(),256);
    assert_eq!(prefill_budget([(32,2048);8].into_iter(),8).unwrap(),64);
    assert_eq!(prefill_budget([(1,1);8].into_iter(),8).unwrap(),8);
    assert!(prefill_budget([(2048,512)].into_iter(),1).is_err());
}
/// A row is either the next causal token of one prompt, or one independent chat.
/// Validate the complete cohort before any lexical/recurrent history advances.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn validate(
    positions: &[usize],
    tokens: &[&[u32]],
    capacities: &[usize],
    vocab: u64,
) -> Result<Vec<Range<usize>>> {
    if tokens.is_empty()
        || tokens.len() > 16
        || positions.len() != tokens.len()
        || capacities.len() != tokens.len()
    {
        return Err("invalid Flash-Next row cohort".into());
    }
    let mut offset = 0usize;
    tokens
        .iter()
        .zip(positions)
        .zip(capacities)
        .map(|((row, &pos), &cap)| {
            if row.is_empty()
                || pos.checked_add(row.len()).is_none_or(|end| end > cap)
                || row.iter().any(|&t| t as u64 >= vocab)
            {
                return Err("Flash-Next token or context outside limits".into());
            }
            let end = offset
                .checked_add(row.len())
                .ok_or("Flash-Next row count overflow")?;
            let range = offset..end;
            offset = end;
            Ok(range)
        })
        .collect()
}
/// A cohort uploads exactly one padded slot vector; zero never owns state.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn padded_slots(slots: &[u32]) -> Result<Vec<u32>> {
    if slots.is_empty()
        || slots.len() > 16
        || slots
            .iter()
            .enumerate()
            .any(|(i, &id)| id == 0 || id > 16 || slots[..i].contains(&id))
    {
        return Err("invalid Flash-Next pool slots".into());
    }
    let mut out = slots.to_vec();
    out.resize(slots.len().next_power_of_two(), 0);
    Ok(out)
}
/// A multi-token tile starts only on an absolute 64-token boundary.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn prompt_rows(position: usize, remaining: usize, limit: usize) -> usize {
    let n = remaining.min(limit);
    if position % 64 != 0 {
        n.min(1)
    } else if n >= 64 {
        n / 64 * 64
    } else {
        n
    }
}
#[cfg(test)]
pub(crate) fn routes(logits: &[f32], experts: usize) -> Result<Vec<Vec<(usize, f32)>>> {
    if experts < 10
        || logits.is_empty()
        || logits.len() % experts != 0
        || logits.iter().any(|x| !x.is_finite())
    {
        return Err("nonfinite or invalid Flash-Next router row".into());
    }
    Ok(logits
        .chunks_exact(experts)
        .map(|row| {
            let (ids, weights) = engine_model::flash::refcpu::route(row, 10);
            let mut pairs: Vec<_> = ids.into_iter().zip(weights).collect();
            pairs.sort_by_key(|&(id, _)| id);
            pairs
        })
        .collect())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verifier_head_slices_cover_c1_c8_c16_k0_through_k3() {
        const VOCAB: usize = 248320;
        for chats in [1, 8, 16] {
            for depth in 0..=3 {
                for unequal in [false, true] {
                    let widths: Vec<_> = (0..chats)
                        .map(|i| {
                            if unequal {
                                1 + i % (depth + 1)
                            } else {
                                depth + 1
                            }
                        })
                        .collect();
                    let count = widths.iter().sum();
                    let mut writes = vec![0u8; count];
                    let mut logits = vec![f32::NAN; count * VOCAB];
                    let mut results = vec![(u32::MAX, f32::NAN); count];
                    for tile in head_tiles(count).unwrap() {
                        let groups = tile.len().div_ceil(8);
                        // The actual kernel stores four warps of groups*8 triples.
                        assert!(4 * groups * 8 * 12 <= 1024);
                        for local in 0..tile.len() {
                            let row = tile.start + local;
                            for warp in 0..4 {
                                let read_end = (warp * groups * 8 + local) * 12 + 12;
                                assert!(read_end <= 4 * groups * 8 * 12);
                            }
                            writes[row] += 1;
                            let full = &mut logits[row * VOCAB..(row + 1) * VOCAB];
                            for (token, value) in full.iter_mut().enumerate() {
                                // A row-local four-range fold, including ties. Moving
                                // a row across a slice must preserve every f32 bit.
                                let parts = [row as f32 * 0.125, (token % 131) as f32, -0.25, 0.5];
                                *value = (parts[0] + parts[1]) + (parts[2] + parts[3]);
                            }
                            results[row] = (crate::greedy_token(full).unwrap(), full[130]);
                        }
                    }
                    assert!(writes.iter().all(|&n| n == 1));
                    for row in (0..count).rev() {
                        for (token, value) in
                            logits[row * VOCAB..(row + 1) * VOCAB].iter().enumerate()
                        {
                            let expected =
                                (row as f32 * 0.125 + (token % 131) as f32) + (-0.25 + 0.5);
                            assert_eq!(value.to_bits(), expected.to_bits());
                        }
                        assert_eq!(results[row].0, 130);
                    }
                }
            }
        }
        for count in [0, 65, usize::MAX] {
            assert!(head_tiles(count).is_err());
        }
    }
    #[test]
    fn mixed_capacity_workspace_is_charged_to_the_largest_owner() {
        for capacities in [
            vec![256, 262144, 262144, 262144, 262144, 262144, 262144, 262144],
            vec![32768],
            vec![262144, 256],
        ] {
            let owner = workspace_owner(capacities.iter().copied()).unwrap();
            assert_eq!(capacities[owner], *capacities.iter().max().unwrap());
        }
        assert!(workspace_owner(std::iter::empty()).is_err());
        assert_eq!(RECURRENT_BYTES, 119144448);
        assert_eq!(PREFIX_BYTES, 119554048);
    }
    #[test]
    fn lost_owner_failure_reset_and_pooled_admission_reject_stale_slots() {
        for layer in [0, 3, 7, 47] {
            let mut gdn: Vec<_> = (0..48)
                .map(|li| (li % 4 != 3).then(|| Box::new(li)))
                .collect();
            let mut qsa: Vec<_> = (0..48)
                .map(|li| (li % 4 == 3).then(|| Box::new(li)))
                .collect();
            check_owners(&gdn, &qsa, 0..48).unwrap();
            let slot = if layer % 4 == 3 {
                &mut qsa[layer]
            } else {
                &mut gdn[layer]
            };
            let published = &**slot.as_ref().unwrap() as *const i32;
            fn inject<T>(slot: &mut Option<T>) -> Result<()> {
                let _owner = slot.take();
                Err("injected recoverable launch failure".into())
            }
            let failed = inject(slot);
            assert!(failed.is_err());
            // reset uses this gate before publishing valid=true, without rebuilding storage.
            assert!(
                check_owners(&gdn, &qsa, 0..48)
                    .unwrap_err()
                    .contains("replace session")
            );
            let mut used_addresses = Vec::new();
            let pooled = check_owners(&gdn, &qsa, 0..48).map(|()| used_addresses.push(published));
            assert!(pooled.is_err());
            assert!(used_addresses.is_empty());
        }
    }
    #[test]
    fn stage_owners_cover_only_the_owned_layers() {
        for (first, end) in [(0, 24), (24, 48), (0, 12), (36, 48)] {
            let own = |li: usize| (first..end).contains(&li);
            let mut gdn: Vec<_> = (0..48).map(|li| (li % 4 != 3 && own(li)).then(|| Box::new(li))).collect();
            let qsa: Vec<_> = (0..48).map(|li| (li % 4 == 3 && own(li)).then(|| Box::new(li))).collect();
            check_owners(&gdn, &qsa, first..end).unwrap();
            assert!(check_owners(&gdn, &qsa, 0..48).is_err());
            gdn[first] = None;
            assert!(check_owners(&gdn, &qsa, first..end).unwrap_err().contains("replace session"));
        }
        let gdn: Vec<Option<u8>> = vec![None; 48];
        assert!(check_owners(&gdn, &gdn, 0..0).is_err());
        assert!(check_owners(&gdn, &gdn, 40..49).is_err());
    }
    #[test]
    fn changing_indexer_reads_actual_source_after_another_requests_warm_pool() {
        use std::cell::Cell;
        let calls = Cell::new(0);
        let hook = || {
            let n = calls.get() + 1;
            calls.set(n);
            n != 13
        };
        let pooled = Cell::new(41u32); // previous request
        let session = Cell::new(0u32);
        for (token, expected_source) in [(17, ResultSource::Pooled), (23, ResultSource::Session)] {
            let source = ResultSource::select(true, || (0..12).all(|_| hook()));
            match source {
                ResultSource::Pooled => pooled.set(token),
                ResultSource::Session => session.set(token),
            }
            assert_eq!(source, expected_source);
            assert_eq!(
                source
                    .read(|| Ok(pooled.get()), || Ok(session.get()))
                    .unwrap(),
                token
            );
        }
        assert_eq!(calls.get(), 13);
        assert_eq!(pooled.get(), 17); // eager step must not return this stale result
    }
    #[test]
    fn chunks_and_chat_permutations_preserve_admission_and_routes() {
        for pos in 0..130 {
            for limit in [1, 7, 63, 64, 65, 127, 512, 2048] {
                for remaining in [1, 17, 512, 2051, 8192] {
                    let n = prompt_rows(pos, remaining, limit);
                    assert!((1..=remaining.min(limit)).contains(&n));
                    assert!(n == 1 || pos % 64 == 0);
                    if n >= 64 {
                        assert_eq!(n % 64, 0);
                    }
                }
            }
        }
        for n in 1..=16 {
            let live: Vec<_> = (1..=n as u32).rev().collect();
            let padded = padded_slots(&live).unwrap();
            assert_eq!(&padded[..n], &live);
            assert_eq!(padded.len(), n.next_power_of_two());
            assert!(padded[n..].iter().all(|&id| id == 0));
            let step = spark_gpu_probe::next_math::SlotStep {
                slots: padded.as_ptr() as u64,
                table: 0,
                layer: 0,
                rows: padded.len() as u32,
            };
            let mut history = [0u32; 17];
            history[0] = 0xdeadbeef;
            for row in 0..padded.len() + 1 {
                if let Some(slot) = unsafe { step.selected(row) } {
                    history[slot] += 1;
                }
            }
            assert_eq!(history[0], 0xdeadbeef);
            assert!(history[1..=n].iter().all(|&v| v == 1));
            // Reuse a retired slot in a different row without changing any other slot.
            let reused = padded_slots(&[live[n - 1]]).unwrap();
            assert_eq!(reused, [live[n - 1]]);
        }
        for invalid in [vec![], vec![0], vec![17], vec![1, 1], vec![1; 17]] {
            assert!(padded_slots(&invalid).is_err());
        }
        assert_eq!(prompt_rows(0, 2051, 2048), 2048);
        assert_eq!(prompt_rows(2048, 3, 2048), 3);
        let input: Vec<u32> = (0..513).collect();
        for chunk in [1, 2, 7, 16, 63, 64, 65, 127, 128, 256, 512] {
            let mut pos = 0;
            for row in input.chunks(chunk) {
                assert_eq!(
                    validate(&[pos], &[row], &[513], 1024).unwrap(),
                    [0..row.len()]
                );
                pos += row.len();
            }
            assert_eq!(pos, 513);
        }
        let logits: Vec<f32> = (0..16 * 512)
            .map(|i| ((i * 137 % 997) as f32 - 499.0) / 23.0)
            .collect();
        let solo: Vec<_> = logits
            .chunks_exact(512)
            .map(|r| routes(r, 512).unwrap().remove(0))
            .collect();
        for n in [1, 2, 8, 16] {
            for rows in logits.chunks(n * 512) {
                let got = routes(rows, 512).unwrap();
                for (row, route) in rows.chunks_exact(512).zip(got) {
                    assert_eq!(route, routes(row, 512).unwrap()[0]);
                }
            }
            let permuted: Vec<_> = (0..n)
                .rev()
                .flat_map(|i| logits[i * 512..(i + 1) * 512].iter().copied())
                .collect();
            assert_eq!(
                routes(&permuted, 512).unwrap(),
                solo[..n].iter().rev().cloned().collect::<Vec<_>>()
            );
        }
        assert!(validate(&[0], &[&[1024]], &[1], 1024).is_err());
        assert!(validate(&[usize::MAX], &[&[0]], &[usize::MAX], 1024).is_err());
        assert_eq!(
            validate(&[0, 0], &[&[1, 2], &[3]], &[4, 4], 1024).unwrap(),
            [0..2, 2..3]
        );
        assert!(routes(&vec![f32::NAN; 512], 512).is_err());
    }
}

#[cfg(test)]
mod mtp_tests {
    use crate::{
        draft::{Config, ModelProposal, Speculator, Target},
        greedy_token,
        sample::Sampling,
    };
    use spark_gpu_probe::next_math as m;
    #[derive(Clone)]
    struct State {
        cells: [f32; 128],
        conv: [f32; 9],
        logits: Vec<f32>,
    }
    impl State {
        fn new() -> Self {
            Self {
                cells: [0.; 128],
                conv: [0.; 9],
                logits: vec![0.; 8],
            }
        }
        fn step(&mut self, t: u32) -> Vec<f32> {
            let q = [(t + 1) as f32 / 1024.; 128];
            let k = [0.015625; 128];
            let value = m::column(&mut self.cells, &q, &k, t as f32 / 8., 0.25, 0.9921875);
            let conv = self.conv[0] * 0.125 + self.conv[3] * 0.25 + self.conv[6] * 0.5 + t as f32;
            self.conv.copy_within(1..9, 0);
            self.conv[8] = value;
            let sig = |v: f32| 1. / (1. + m::exp(-v));
            let hc = m::hc_mix([value, conv, 0.5, -0.25], [0.5, -0.5, 0.25, -0.25], sig);
            let x = m::hc_add(value, 0.5, hc, sig);
            let packed = [m::bf16(x) | (m::bf16(value) << 16); 16];
            let w = [0x21; 16];
            let scales = [56; 2];
            let expert =
                unsafe { m::project(packed.as_ptr(), w.as_ptr(), scales.as_ptr(), 0.5, 32) };
            self.logits = (0..8)
                .map(|i| ((i + t as usize * 3) % 8) as f32 + expert * 0.001)
                .collect();
            self.logits.clone()
        }
        fn bytes(&self) -> Vec<u32> {
            self.cells
                .iter()
                .chain(&self.conv)
                .chain(&self.logits)
                .map(|v| v.to_bits())
                .collect()
        }
    }
    #[derive(Clone)]
    struct Oracle {
        state: State,
        history: Vec<u32>,
        saved: Vec<(usize, State)>,
        depth: usize,
        lookup: bool,
    }
    impl Target for Oracle {
        fn checkpoint(&self) -> usize {
            self.history.len()
        }
        fn truncate(&mut self, p: usize) -> Result<(), String> {
            if p != self.history.len() {
                let entry = self
                    .saved
                    .first()
                    .filter(|s| s.0 <= p && p <= self.history.len())
                    .ok_or("prefix missing")?;
                self.state = entry.1.clone();
                for &token in &self.history[entry.0..p] {
                    self.state.step(token);
                }
                self.history.truncate(p);
            }
            Ok(())
        }
        fn step(&mut self, t: u32) -> Result<Vec<f32>, String> {
            self.history.push(t);
            Ok(self.state.step(t))
        }
        fn verify_batch(&mut self, ts: &[u32]) -> Result<Vec<Vec<f32>>, String> {
            self.saved.clear();
            self.saved.push((self.history.len(), self.state.clone()));
            ts.iter().map(|&t| self.step(t)).collect()
        }
        fn model_draft_width(&self) -> usize {
            self.depth
        }
        fn max_verify_rows(&self) -> usize {
            if self.lookup { 16 } else { self.depth + 1 }
        }
        fn native_draft_policy(&self) -> bool {
            true
        }
        fn propose_model(
            &mut self,
            history: &[u32],
            mut t: u32,
            n: usize,
        ) -> Result<Option<ModelProposal>, String> {
            let copied = if self.lookup {
                crate::flash_next_lookup::Lookup::default().propose(history, t, n)
            } else {
                Vec::new()
            };
            if !copied.is_empty() {
                return Ok(Some(ModelProposal {
                    tokens: copied,
                    proposal_ms: 0.,
                    catchup_ms: 0.,
                }));
            }
            let mut copy = self.clone();
            let tokens = (0..n.min(self.depth))
                .map(|i| {
                    t = greedy_token(&copy.step(t).unwrap()).unwrap();
                    if (self.history.len() + i).is_multiple_of(5) {
                        t = (t + 1) % 8;
                    }
                    t
                })
                .collect();
            Ok(Some(ModelProposal {
                tokens,
                proposal_ms: 0.,
                catchup_ms: 0.,
            }))
        }
    }
    #[test]
    fn mtp_oracle_cards_64_k123_c1_c8_c16_and_prefix_replay() {
        for prompt in [
            "What is the capital of France? Describe the city, its history and three places to visit.",
            "Explain how to build a consistent daily reading habit. Give a practical plan with examples.",
            "Write a Rust function that merges two sorted slices into a sorted Vec. Explain its complexity.",
            "A train travels 120 km at 60 km/h and returns at 40 km/h. Derive its average speed and explain why the arithmetic mean is wrong.",
            "Edit only the return value; copy the rest: fn merge(a: &[u32], b: &[u32]) -> Vec<u32> { let mut out = Vec::new(); out.extend(a); out.extend(b); out }",
            "Repeat this tool call with only the file changed: {\"name\":\"apply_patch\",\"arguments\":{\"path\":\"src/lib.rs\",\"old\":\"let count = 1;\",\"new\":\"let count = 2;\"}}",
        ] {
            let mut base = Oracle {
                state: State::new(),
                history: Vec::new(),
                saved: Vec::new(),
                depth: 0,
                lookup: false,
            };
            for t in prompt.bytes() {
                base.step((t % 8) as u32).unwrap();
            }
            let mut plain = base.clone();
            let mut next = greedy_token(&plain.state.logits).unwrap();
            let mut expected = Vec::new();
            for _ in 0..64 {
                expected.push(next);
                next = greedy_token(&plain.step(next).unwrap()).unwrap();
            }
            for width in [1, 4, 8, 16] {
                let input = &expected[..width];
                let mut serial = base.clone();
                let reference: Vec<_> = input.iter().map(|&t| serial.step(t).unwrap()).collect();
                let mut rows = base.clone();
                assert_eq!(rows.verify_batch(input).unwrap(), reference);
                for kept in 0..=width {
                    let mut selected = rows.clone();
                    selected.truncate(base.history.len() + kept).unwrap();
                    let mut replay = base.clone();
                    for &t in &input[..kept] {
                        replay.step(t).unwrap();
                    }
                    assert_eq!(selected.state.bytes(), replay.state.bytes());
                    for &t in &expected[width..] {
                        assert_eq!(selected.step(t).unwrap(), replay.step(t).unwrap());
                    }
                }
            }
            for depth in [1, 2, 3, 15] {
                for chats in [1, 8, 16] {
                    let mut targets = vec![base.clone(); chats];
                    let mut specs = Vec::new();
                    for target in &mut targets {
                        target.depth = if depth == 15 { 2 } else { depth };
                        target.lookup = depth == 15;
                        let mut config = Config::strict(64);
                        config.max_rows = depth + 1;
                        config.adaptive = false;
                        specs.push(
                            Speculator::new(
                                &target.history,
                                &target.state.logits,
                                config,
                                Sampling::greedy(),
                            )
                            .unwrap(),
                        );
                    }
                    while specs.iter().any(|s| !s.is_done()) {
                        for (s, t) in specs.iter_mut().zip(&mut targets).rev() {
                            if !s.is_done() {
                                s.step(t, depth + 1, |_| false).unwrap();
                                let mut replay = base.clone();
                                for &token in &t.history[base.history.len()..] {
                                    replay.step(token).unwrap();
                                }
                                assert_eq!(t.state.bytes(), replay.state.bytes());
                            }
                        }
                    }
                    for spec in specs {
                        assert_eq!(spec.tokens(), expected);
                        assert!(spec.stats.accepted > 0);
                        assert!(spec.stats.rollbacks > 0);
                        if depth == 15 {
                            assert_eq!(spec.stats.max_rows, 16);
                        }
                    }
                }
            }
        }
    }
}
