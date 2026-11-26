//! Retire optional state before its invoice can block another admission.
use super::{Active, Engine, Ready, Session, Tokenizer};

/// Whether `request` and the requests it would overlap all keep their draft state: the engine keeps it for up to
/// `contended_draft_chats` overlapping requests when each of them can draft beside the others (`drafts_contended`).
/// With the default of one chat this is never true beside a running request.
pub(super) fn keeps<E: Engine, T: Tokenizer>(engine: &E, active: &[Active<'_, '_, E, T>], request: &crate::protocol::Request) -> bool {
    let mut running = active.iter().filter(|job| !job.done);
    running.clone().count() < engine.contended_draft_chats() && engine.drafts_contended(request)
        && running.all(|job| !job.reservation_plain && engine.drafts_contended(&job.ready.request))
}
/// Whether `request` takes the plain reservation because it overlaps a running request (`discard_contended_draft`).
pub(super) fn contended<E: Engine, T: Tokenizer>(engine: &E, active: &[Active<'_, '_, E, T>], request: &crate::protocol::Request) -> bool {
    engine.discard_contended_draft() && active.iter().any(|job| !job.done) && !keeps(engine, active, request)
}

/// Priority or affinity can select a younger request that cannot keep the queue head's draft
/// overlap. A true result makes dispatch retire before allocation and price the queue again.
pub(super) fn retires_selected<E: Engine, T: Tokenizer>(engine: &E, active: &[Active<'_, '_, E, T>],
    request: &crate::protocol::Request, plain: bool,
) -> bool {
    engine.discard_contended_draft() && active.iter().any(|job| !job.reservation_plain)
        && (plain || contended(engine, active, request))
}

pub(super) fn retire<E: Engine, T: Tokenizer>(
    engine: &E,
    active: &mut [Active<'_, '_, E, T>],
    reservation_for: impl Fn(&Ready, usize, bool, Option<usize>) -> Option<usize>,
) -> Result<(), String> {
    if !engine.discard_contended_draft() { return Ok(()); }
    // Waiting until decode would let the old draft invoice prevent admission.
    for job in active.iter_mut().filter(|job| !job.reservation_plain) {
        job.generation.state.session.disable_model_draft();
        if let Some(error) = engine.fatal_error() { return Err(error); }
        job.generation.prefer_plain_shared_prefix();
        job.generation.state.reserved_session_bytes = reservation_for(&job.ready,job.reserved_tokens,true,None)
            .ok_or("missing plain reservation")?;
        job.reservation_plain = true;
    }
    Ok(())
}
