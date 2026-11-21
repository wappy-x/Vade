//! Bound a temporary prompt workspace before a synchronous scheduler slice.
use super::{Active,Config,Engine,Ready,Retired,Tokenizer,reservations,shared_prefix};

const CONTENDED_PROMPT_ROWS: usize = 512;

pub(super) fn contention<E:Engine,T:Tokenizer>(jobs:&[Active<'_, '_, E,T>],i:usize,errors:&[Option<String>],queued:&[Ready],config:&Config,shared_prompt_limit:usize)->(bool,bool,bool) {
    let source = &jobs[i];
    // Waiters for this owner's uncommitted prefix cannot run yet.
    // Keep the large prefill tile until an independent request is due.
    let queued_runnable = queued.iter().any(|waiting| !source.automatic_cache
        || !config.prefix_cache || !waiting.request.prefix_cache || source.ready.scope != waiting.scope
        || shared_prefix(&source.ready.prompt, &waiting.prompt)
            <= source.generation.state.tokens.len() + shared_prompt_limit);
    let other_runnable = jobs.iter().enumerate().any(|(n, j)| n != i && !j.done
        && errors[n].is_none() && j.ready.wire.bytes.is_empty()) || queued_runnable;
    let decoding = jobs.iter().enumerate().any(|(n, j)| n != i && !j.done
        && errors[n].is_none() && j.ready.wire.bytes.is_empty()
        && j.generation.state.tokens.len() >= j.ready.prompt.len());
    let shared_waiters = !queued.is_empty() && !queued_runnable;
    (other_runnable,decoding,shared_waiters)
}
/// A cold prompt that arrives behind a live decoder and would take exactly two contended slices
/// (the engine's minimum tile, then the rest) finishes in one turn when the engine's own idle chunk
/// funds it, nothing waits on its prefix and no other chat holds priority. One prompt runs per turn,
/// so due decode still runs before the next prompt. Longer prompts and idle or lone chats keep their path.
pub(super) fn whole_short_prompt(remaining:usize,context:usize,rows:usize,funded:usize,decoding:bool,shared_waiters:bool,foreground_other:bool)->bool {
    decoding && context==0 && !shared_waiters && !foreground_other && remaining>rows && remaining<=2*rows && remaining<=funded
}
/// Widen long cold reads behind live decoders within the already-funded group.
pub(super) fn prompt_chunk(
    remaining: usize, context: usize, (rows, funded): (usize, usize), decoding: bool,
    shared_waiters: bool, foreground_other: bool, cold_long: bool,
) -> usize {
    if whole_short_prompt(remaining, context, rows, funded, decoding, shared_waiters, foreground_other) {
        return remaining;
    }
    let rows = if cold_long && decoding && !shared_waiters && !foreground_other {
        rows.max(CONTENDED_PROMPT_ROWS.min(funded))
    } else { rows };
    rows.min(remaining)
}
pub(super) fn prepare<'e,'t,E:Engine,T:Tokenizer>(engine:&'e E,(pool,retained):(usize,usize),jobs:&[Active<'e,'t,E,T>],active:&[Active<'e,'t,E,T>],retired:&[Retired<E::Session<'e>>],alone:bool,request_hold:impl Fn(&Ready)->usize)->Result<(),String> {
    let reserved = reservations(jobs.iter().map(|j| &j.generation.state)
        .chain(active.iter().map(|j| &j.generation.state))
        .chain(retired.iter().map(|r| &r.state))).0;
    let hold = engine.cache_hold_bytes() + jobs.iter().chain(active.iter()).map(|j| request_hold(&j.ready)).sum::<usize>();
    let available = if alone && jobs.len() + active.len() == 1 && retired.is_empty() {
        pool.saturating_sub(reserved).saturating_sub(retained).saturating_sub(hold)
    } else { 0 };
    engine.prepare_prompt_workspace(available)?;
    Ok(())
}
