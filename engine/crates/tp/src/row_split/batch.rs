//! The served batch step: one decode step for several chats in plain decode, on the controller
//! (`batch_tokens`) and on the worker (`worker_batch`, command 13).
use super::*;
/// A worker's live chats: the decoder, the state (absent while a step holds it) and open verify rows.
pub(super) type WorkerSessions<'a,C,E>=BTreeMap<u64,(Decoder<'a,&'a Split<C,E>>,Option<State<<C as Compute>::Value>>,Option<Verify<<C as Compute>::Value>>)>;
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    /// One batch step on both ranks for chats in plain decode: `tokens[i]` is the input token of
    /// `sessions[i]`. Each chat gets the logits and the ID its own plain step gives, and its next state.
    /// `greedy`: no chat needs its logits; they come back empty (`Decoder::step_batch_greedy`).
    pub(super) fn batch_tokens(&self,sessions:&mut [&mut Session<'_,C,E>],tokens:&[u32],greedy:bool)->Result<Vec<(Vec<f32>,u32)>> {
        if let Some(error)=self.failure_reason() {return Err(error);}
        let vocab=self.checkpoint.description().vocab;
        if sessions.len()!=tokens.len() || !(2..=crate::bounded_decoder::BATCH_ROWS).contains(&sessions.len()) {return Err("row split batch step exceeds its bound".into());}
        let outside=|session:&Session<'_,C,E>,token:u32|session.state.as_ref().is_none_or(|s|s.position>=self.context as u64) || u64::from(token)>=vocab;
        if sessions.iter().zip(tokens).any(|(session,&token)|outside(session,token)) {return Err("row split request exceeds vocabulary or context".into());}
        self.finish(self.split.work(|| {
            // Print only: the time since the last batch step ended (scheduler tick, sampling, stream writes, other work).
            let since=self.batch_ended.take().map_or(0.0,|at|at.elapsed().as_secs_f64());
            let started=std::time::Instant::now();
            if self.failed.get() {return Err("row split service previously failed".into());}
            self.failed.set(true);self.split.command_for(13,sessions.len() as u32,0)?;
            let mut bytes=sessions.iter().zip(tokens).flat_map(|(session,token)|session.id.to_le_bytes().into_iter().chain(token.to_le_bytes())).collect::<Vec<_>>();
            self.split.exchange.borrow_mut().command(&mut bytes)?;
            let results={let rows=sessions.iter().zip(tokens).map(|(session,&token)|Ok((token,session.state.as_ref().ok_or("row split session has no state")?))).collect::<Result<Vec<_>>>()?;
                if greedy {sessions[0].decoder.step_batch_greedy(&rows)?} else {sessions[0].decoder.step_batch(&rows)?}};
            self.split.exchange.borrow_mut().complete()?;
            let mut rows=Vec::with_capacity(results.len());
            // A batch step is not a steady plain step: the proposer's own step time must not learn from it.
            for (session,(logits,best,next)) in sessions.iter_mut().zip(results) {session.state=Some(next);session.after_step=false;rows.push((logits,best));}
            self.failed.set(false);
            let ids=sessions.iter().map(|session|session.id).collect::<Vec<_>>();
            eprintln!("SERVED BATCH rank={} rows={} sessions={ids:?} seconds={:.9} since_last_seconds={since:.9} mhc_fused_row_launches={}",self.split.topology.rank,rows.len(),started.elapsed().as_secs_f64(),self.split.mhc_row_launches());
            self.batch_ended.set(Some(std::time::Instant::now()));
            self.split.cost_report("decode");Ok(rows)
        }))
    }
    /// The worker's side of one batch step of `rows` chats: each row follows the command as a session
    /// ID and its input token.
    pub(super) fn worker_batch(&self,sessions:&mut WorkerSessions<'_,C,E>,rows:usize)->Result<()> {
        if !(2..=crate::bounded_decoder::BATCH_ROWS).contains(&rows) {return Err("worker batch step exceeds its bound".into());}
        self.split.work(|| {
            let mut bytes=vec![0;rows*12];self.split.exchange.borrow_mut().command(&mut bytes)?;
            let chats=bytes.as_chunks::<12>().0.iter().map(|b|(u64::from_le_bytes(b[..8].try_into().unwrap()),u32::from_le_bytes(b[8..].try_into().unwrap()))).collect::<Vec<_>>();
            // Each chat's state leaves its session for the step: a chat named twice has none the second time.
            let mut held=Vec::with_capacity(rows);
            for &(id,_) in &chats {held.push(sessions.get_mut(&id).ok_or("worker batch step has no session")?.1.take().ok_or("worker batch session has no state")?);}
            let results={let decoder=&sessions.get(&chats[0].0).ok_or("worker batch step has no session")?.0;
                // The worker keeps each chat's next state and nothing else: it never reads the logits.
                decoder.step_batch_greedy(&chats.iter().zip(&held).map(|(&(_,token),state)|(token,state)).collect::<Vec<_>>())?};
            self.split.exchange.borrow_mut().complete()?;
            for (&(id,_),(_,_,next)) in chats.iter().zip(results) {sessions.get_mut(&id).ok_or("worker batch step has no session")?.1=Some(next);}
            Ok(())
        })
    }
}
