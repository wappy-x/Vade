//! Optional per-request storage is released on both ranks before its invoice shrinks.
use super::*;
pub(super) const PREDICTOR_MODE:u32=31;
pub(super) fn request_can_draft(r:&spark_serve::protocol::Request)->bool {
    use spark_serve::protocol::DraftMode;
    let trial=matches!(r.draft,None|Some(DraftMode::Auto)) && r.mtp_depth.is_some_and(|depth|depth>=trial::TRIAL_DEPTH)
        && r.max_tokens>trial::TRIAL_START as usize;
    (r.draft==Some(DraftMode::Strict) || trial) && spark_engine::draft::plain_greedy(r.sampling)
}
pub(super) fn set_mode<'f,C:Compute>(decoder:&mut Decoder<'f,C>,state:&mut Option<State<C::Value>>,open:&mut Option<Verify<C::Value>>,prepared:&crate::bounded_decoder::Prepared<'f>,enabled:bool)->Result<()> {
    if !enabled {
        // Finish any selected prefix under the old, still-funded invoice. Dropping Verify
        // releases the predictor cache, row snapshots and hidden trail together.
        if let Some(rows)=open.take() {*state=Some(rows.settle(decoder)?);}
        // Draft passes can grow expanded rows past the prompt. A plain invoice funds only
        // prompt-held rows; drop their owners rather than retain an oversized prefix view.
        if let Some(state)=state {state.expanded.clear();}
    }
    decoder.set_prepared_predictor(prepared,enabled);Ok(())
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    pub(super) fn predictor_mode(&mut self,enabled:bool)->Result<()> {
        if self.kept.is_some() {return Ok(());}
        let enabled=enabled && self.factory.predictor>0;
        if self.decoder.predictor_enabled()==enabled && (enabled || self.open.is_none() && self.factory.tail_rows(self.id)==0) {return Ok(());}
        let f=self.factory;if let Some(error)=f.failure_reason() {return Err(error);}
        f.finish(f.split.work(|| {
            f.failed.set(true);f.split.command_for(PREDICTOR_MODE,u32::from(enabled),self.id)?;
            set_mode(&mut self.decoder,&mut self.state,&mut self.open,&f.prepared,enabled)?;
            if !enabled {f.tails.borrow_mut().remove(&self.id);}
            f.split.exchange.borrow_mut().complete()?;f.failed.set(false);Ok(())
        }))
    }
}
