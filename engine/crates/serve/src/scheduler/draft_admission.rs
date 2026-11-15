//! Retire optional state before its invoice can block another admission.
use super::{Active, Engine, Ready, Session, Tokenizer};

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
