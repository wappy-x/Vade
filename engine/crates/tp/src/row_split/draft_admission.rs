//! Optional per-request storage is released on both ranks before its invoice shrinks.
use super::*;
pub(super) const PREDICTOR_MODE:u32=31;
/// The command's word: 0 turns the predictor off and drops the expanded rows (the request's invoice shrinks to the
/// plain one), 1 turns it on, HANDOFF turns it off for a request that keeps its drafting invoice to its end
/// (`Session::draft_over`): the rows its prompt and its passes expanded stay, and its plain steps read them.
pub(super) const HANDOFF:u32=2;
pub(super) fn request_can_draft(r:&spark_serve::protocol::Request)->bool {
    use spark_serve::protocol::DraftMode;
    let trial=matches!(r.draft,None|Some(DraftMode::Auto)) && r.mtp_depth.is_some_and(|depth|depth>=trial::TRIAL_DEPTH)
        && r.max_tokens>trial::TRIAL_START as usize;
    (r.draft==Some(DraftMode::Strict) || trial) && spark_engine::draft::plain_greedy(r.sampling)
}
/// GROUP DRAFTING (pair.rs): whether a request keeps its draft state beside another chat. A strict request of depth
/// 2 or more, and a request that drafts by default (no draft mode or "auto", the trial's depth, an output cap above
/// TRIAL_START) while DRAFT CONFIDENCE is on (`confident`). Greedy sampling only.
pub(super) fn drafts_beside(r:&spark_serve::protocol::Request,confident:bool)->bool {
    use spark_serve::protocol::DraftMode;
    let default=confident && matches!(r.draft,None|Some(DraftMode::Auto)) && r.mtp_depth.is_some_and(|depth|depth>=trial::TRIAL_DEPTH) && r.max_tokens>trial::TRIAL_START as usize;
    (r.draft==Some(DraftMode::Strict) && r.mtp_depth.is_some_and(|depth|depth>=2) || default) && spark_engine::draft::plain_greedy(r.sampling)
}
pub(super) fn set_mode<'f,C:Compute>(decoder:&mut Decoder<'f,C>,state:&mut Option<State<C::Value>>,open:&mut Option<Verify<C::Value>>,prepared:&crate::bounded_decoder::Prepared<'f>,word:u32)->Result<()> {
    let enabled=word==1;
    if !enabled {
        // Finish any selected prefix under the old, still-funded invoice. Dropping Verify
        // releases the predictor cache, row snapshots and hidden trail together.
        if let Some(rows)=open.take() {*state=Some(rows.settle(decoder)?);}
        // Draft passes can grow expanded rows past the prompt. A plain invoice funds only
        // prompt-held rows; drop their owners rather than retain an oversized prefix view.
        // A HANDOFF keeps them: its invoice does not shrink, and without them every later step expands the prompt's
        // rows again (two Sparks, 2026-11-16, run duox-g-c1-1: steps of two chats 64.69 ms against 64.24 ms).
        if word!=HANDOFF && let Some(state)=state {state.expanded.clear();}
    }
    decoder.set_prepared_predictor(prepared,enabled);Ok(())
}
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    pub(super) fn predictor_mode(&mut self,enabled:bool)->Result<()> {self.predictor_word(u32::from(enabled))}
    /// The predictor off for a request that goes on by plain steps under its drafting invoice (HANDOFF).
    pub(super) fn predictor_handoff(&mut self)->Result<()> {self.predictor_word(HANDOFF)}
    fn predictor_word(&mut self,word:u32)->Result<()> {
        if self.kept.is_some() {return Ok(());}
        let enabled=word==1 && self.factory.predictor>0;let word=if enabled {1} else if word==HANDOFF {HANDOFF} else {0};
        if self.decoder.predictor_enabled()==enabled && (enabled || self.open.is_none() && self.factory.tail_rows(self.id)==0 && (word!=0 || !self.draft_handoff)) {return Ok(());}
        let f=self.factory;if let Some(error)=f.failure_reason() {return Err(error);}
        let result=f.finish(f.split.work(|| {
            f.failed.set(true);f.split.command_for(PREDICTOR_MODE,word,self.id)?;
            set_mode(&mut self.decoder,&mut self.state,&mut self.open,&f.prepared,word)?;
            if !enabled {f.tails.borrow_mut().remove(&self.id);}
            f.split.exchange.borrow_mut().complete()?;f.failed.set(false);Ok(())
        }));
        if result.is_ok() {self.draft_handoff=word==HANDOFF;}result
    }
}
#[cfg(test)] mod tests {
    use super::*;
    use crate::bounded_decoder::{Host,TestCheckpoint};
    use spark_serve::{Engine as _,Session as _};
    #[derive(Default)] struct Modes(Vec<u32>);
    impl Exchange for Modes {
        fn topology(&self)->Topology {Topology {rank:0,parts:1}}
        fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
        fn command(&mut self,words:&mut [u8])->Result<()> {
            if words.len()==16 && u32::from_le_bytes(words[..4].try_into().unwrap())==PREDICTOR_MODE {self.0.push(u32::from_le_bytes(words[4..8].try_into().unwrap()));}Ok(())
        }
        fn complete(&mut self)->Result<()> {Ok(())}
    }
    #[test] fn explicit_retirement_after_handoff_drops_expanded_owners_before_plain_repricing() {
        let c=TestCheckpoint::new_with_predictor(96);
        let engine=Factory::new(&c.admission,Split::new(Host,Modes::default()).unwrap(),96,vec![0]).unwrap().with_predictor(3).unwrap();
        let mut session=engine.session().unwrap();session.predictor_mode(true).unwrap();session.prefill(&[1,2,3]).unwrap();
        let held=session.state.as_ref().unwrap().expanded.values().next().expect("predictor prompt retains expanded rows");
        let owner=std::rc::Rc::downgrade(&held.0);
        session.predictor_handoff().unwrap();assert!(owner.upgrade().is_some(),"handoff keeps funded expanded storage");
        assert!(!session.decoder.predictor_enabled() && session.open.is_none() && engine.tail_rows(session.id)==0);
        let before=engine.split.exchange.borrow().0.len();
        // Competing admission calls this before replacing the drafting invoice with its plain invoice.
        session.disable_model_draft();assert_eq!(engine.fatal_error(),None);
        assert!(owner.upgrade().is_none(),"explicit retirement must drop the owner before admission can shrink its invoice");
        assert!(session.state.as_ref().unwrap().expanded.is_empty());
        assert_eq!(&engine.split.exchange.borrow().0[before..],&[0],"word 0 must also retire the worker");
        let after=engine.split.exchange.borrow().0.len();session.disable_model_draft();
        assert_eq!(engine.split.exchange.borrow().0.len(),after,"completed plain retirement is a no-op");
        drop(session);engine.shutdown().unwrap();
    }
}
