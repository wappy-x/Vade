//! ONE PASS on two ranks (command 33): the proposals of the chats of a group of two or four in one go. Each chat of a group
//! of two or four live chats (pair.rs `joined`) defers its proposal (`Session::defers`); the scheduler then makes ONE `Engine::propose_batch` call a
//! tick for all of them (`Factory::propose_together`). There each chat first decides what it proposes, in the
//! scheduler's order, as its own `propose_model` does; then the chains of all of them run word by word together
//! (`bounded_decoder::propose_group`: each chat's steps on its own values, one vocabulary head pass a word); then each
//! chat records its chain and chooses the words its rows check, in the same order. MEASURED (trace draftm-jump-1, two
//! Sparks): a chat's word is about 3.05 ms, of it the head 1.42 to 1.47 ms; four chats took 43 ms of proposals a tick,
//! one after another (draftm-policy29-1). Proposals only choose which tokens a check tries: no kept token depends on
//! this file.
use super::*;
use spark_engine::draft::ModelProposal;
/// Command 33: the chains of `token` chats in one go. Byte `i` of the session field is chat `i`'s depth. The payload:
/// each chat's session ID, then each chat's token word (its `next` with its stop flags, `confidence::chain_stop`).
pub(super) const GROUP_PROPOSE:u32=33;
impl<C:Compute,E:Exchange> Session<'_,C,E> {
    /// Some(words) when this chat's proposal of this tick waits for the scheduler's one call (`propose_together`): a
    /// chat of a group of two or four live chats that may still draft; a group of three proposes one chat at a time. `words` is the most its own `propose_model` could propose now (its depth,
    /// and a group's share of one check pass), so the tick's rows are reserved as that proposal could take them.
    pub(super) fn defers(&self,limit:usize)->Option<usize> {
        let live=self.factory.active.get();
        if !pair::joined(live) || self.phase==Phase::Over || !self.decoder.predictor_enabled() || !self.factory.pair.borrow().member(self.id,live) {return None;}
        Some(limit.min(self.depth.min(self.factory.predictor)).min((crate::bounded_decoder::VERIFY_ROWS/live).saturating_sub(1)))
    }
}
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    /// The proposals of `rows`, each what its chat's own `propose_model` returns, with the chains made in one go. Each
    /// chat's part of the chains' time is an equal share of it (its own catch-up apart).
    pub(super) fn propose_together(&self,rows:&mut [spark_serve::DraftRows<'_,Session<'_,C,E>>])->Result<Vec<ModelProposal>> {
        let mut limits=Vec::with_capacity(rows.len());
        for row in rows.iter_mut() {limits.push(row.session.model_limit(row.history,row.next,row.limit)?);}
        let entries=rows.iter().map(|row|row.session.model_entry()).collect::<Vec<_>>();
        let started=std::time::Instant::now();
        let made={let mut chats=rows.iter_mut().zip(&limits).map(|(row,&limit)|(&mut *row.session,row.next,limit)).collect::<Vec<_>>();self.propose_group(&mut chats)?};
        let total=started.elapsed().as_secs_f64();
        let chains=limits.iter().filter(|&&limit|limit>0).count().max(1);let waited=made.iter().map(|chain|chain.2).sum::<f64>();
        let share=(total-waited).max(0.0)/chains as f64;
        let mut proposals=Vec::with_capacity(rows.len());
        for (((row,limit),(entry,waiting)),(tokens,mass,waited)) in rows.iter_mut().zip(limits).zip(entries).zip(made) {
            proposals.push(if limit==0 {ModelProposal {tokens:Vec::new(),proposal_ms:0.0,catchup_ms:0.0}}
                else {row.session.model_made(entry,waiting,started,tokens,&mass,share*1000.0,waited*1000.0)});
        }
        Ok(proposals)
    }
    /// `Session::propose` of each of `chats` (session, next, words), the chains of two or more of them in ONE command
    /// on both ranks. A chat with no chain now (`chain_plan`) gets none; a lone chain takes its own command.
    #[allow(clippy::type_complexity)] // each chat's proposals, their scores and its catch-up seconds.
    fn propose_group(&self,chats:&mut [(&mut Session<'_,C,E>,u32,usize)])->Result<Vec<(Vec<u32>,Vec<Option<f32>>,f64)>> {
        if let Some(error)=self.failure_reason() {return Err(error);}
        let plans=chats.iter().map(|(session,next,limit)|if *limit==0 {None} else {session.chain_plan(*next,*limit)}).collect::<Vec<_>>();
        let count=plans.iter().flatten().count();
        if count<2 {return chats.iter_mut().map(|(session,next,limit)|if *limit==0 {Ok((Vec::new(),Vec::new(),0.0))} else {session.propose(*next,*limit)}).collect();}
        if count>LIVE_SESSIONS {return Err("row split proposal group exceeds its bound".into());}
        let packed=plans.iter().flatten().enumerate().fold(0u64,|packed,(at,(depth,_))|packed|(*depth as u64)<<(8*at));
        let mut bytes=chats.iter().zip(&plans).filter(|(_,plan)|plan.is_some()).flat_map(|((session,_,_),_)|session.id.to_le_bytes())
            .chain(chats.iter().zip(&plans).filter_map(|((_,next,_),plan)|plan.map(|(_,flags)|next|flags)).flat_map(u32::to_le_bytes)).collect::<Vec<_>>();
        #[allow(clippy::type_complexity)]
        self.finish((|| -> Result<Vec<(Vec<u32>,Vec<Option<f32>>,f64)>> {self.failed.set(true);
            self.split.command_for(GROUP_PROPOSE,count as u32,packed)?;self.split.exchange.borrow_mut().command(&mut bytes)?;
            let made=self.split.work(|| {
                let (mut decoder,mut group)=(None,Vec::with_capacity(count));
                for ((session,next,_),plan) in chats.iter_mut().zip(&plans) {
                    let Some((depth,flags))=*plan else {continue;};let session=&mut **session;
                    // The layer runs now: the pass after it is not a steady step, whatever its row count.
                    session.after_step=false;
                    let open=session.open.as_mut().ok_or("row split proposal has no open rows")?;
                    if let Some(groups)=self.tails.borrow_mut().remove(&session.id) {open.seed_trail(&session.decoder,&groups)?;}
                    // Any chat's decoder runs the chains: the chats of one service share its weights and its plans.
                    if decoder.is_none() {decoder=Some(&session.decoder);}
                    let (token,stop)=confidence::chain_stop(*next|flags);group.push((open,token,depth,stop));
                }
                crate::bounded_decoder::propose_group(decoder.ok_or("row split proposal group has no chat")?,&mut group)
            })?;
            self.split.exchange.borrow_mut().complete()?;self.failed.set(false);
            eprintln!("SERVED ONE PASS GROUP rank={} chains={count}", self.split.topology.rank);
            let mut made=made.into_iter();
            plans.iter().map(|plan|if plan.is_some() {made.next().ok_or_else(||"row split proposal group lost a chain".into())} else {Ok((Vec::new(),Vec::new(),0.0))}).collect()
        })())
    }
    /// The worker's side of command 33: `chats` chains, their depths a byte a chat in `packed`.
    pub(super) fn worker_propose_group(&self,sessions:&mut batch::WorkerSessions<'_,C,E>,chats:usize,packed:u64)->Result<()> {
        if !(2..=LIVE_SESSIONS).contains(&chats) {return Err("worker proposal group exceeds its bound".into());}
        let depths=(0..chats).map(|at|((packed>>(8*at))&0xff) as usize).collect::<Vec<_>>();
        // A chain deeper than this rank's loaded predictor is a command the controller never sends.
        if depths.iter().any(|&depth|depth==0 || depth>self.predictor) || packed.checked_shr(8*chats as u32).is_some_and(|rest|rest!=0) {return Err("worker proposal group exceeds its depth".into());}
        self.split.work(|| {
            let mut bytes=vec![0;chats*12];self.split.exchange.borrow_mut().command_payload(&mut bytes)?;
            let ids=bytes[..chats*8].as_chunks::<8>().0.iter().map(|b|u64::from_le_bytes(*b)).collect::<Vec<_>>();
            let words=bytes[chats*8..].as_chunks::<4>().0.iter().map(|b|u32::from_le_bytes(*b)).collect::<Vec<_>>();
            if ids.iter().enumerate().any(|(at,id)|ids[..at].contains(id)) {return Err("worker proposal group repeats a session".into());}
            // Each chat's prompt tail first, in the controller's order, as its own chain seeds it.
            for id in &ids {let (decoder,_,open)=sessions.get_mut(id).ok_or("worker proposal group has no session")?;
                let open=open.as_mut().ok_or("worker proposal has no open rows")?;
                if let Some(groups)=self.tails.borrow_mut().remove(id) {open.seed_trail(decoder,&groups)?;}}
            // Each chat's open rows leave its session for the chains and come back after them (as a shared pass does).
            let mut opens=Vec::with_capacity(chats);
            for id in &ids {opens.push(sessions.get_mut(id).and_then(|session|session.2.take()).ok_or("worker proposal has no open rows")?);}
            let result={let decoder=&sessions.get(&ids[0]).ok_or("worker proposal group has no session")?.0;
                let mut group=opens.iter_mut().zip(&words).zip(&depths).map(|((open,&word),&depth)|{let (token,stop)=confidence::chain_stop(word);(open,token,depth,stop)}).collect::<Vec<_>>();
                crate::bounded_decoder::propose_group(decoder,&mut group).map(|_|())};
            for (id,open) in ids.iter().zip(opens) {if let Some(session)=sessions.get_mut(id) {session.2=Some(open);}}
            result?;self.split.exchange.borrow_mut().complete()?;
            eprintln!("SERVED ONE PASS GROUP rank={} chains={chats}", self.split.topology.rank);
            Ok(())
        })
    }
}
