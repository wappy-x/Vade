//! SHARED PASS on two ranks (command 32): the check rows of the drafts of several chats in ONE pass
//! (`bounded_decoder::shared_pass`), on the controller (`Factory::shared_pass`) and on the worker (`worker_shared`).
//!
//! Until now two chats that draft did so in turns: each chat's check rows ran in a pass of their own (pair.rs), and
//! every weight was read once a chat. In a shared pass every step that reads a weight runs once for the rows of all
//! chats, and only the steps that read a chat's own history run once a chat. Each chat's rows have the bits of its own
//! pass, each chat keeps its own open rows, and each takes its own verdict afterwards (command 7): one chat may keep
//! its whole pass while the other keeps one row.
use super::*;
/// Command 32: one shared pass. `token` is the number of chats; byte `i` of the session field is the number of new
/// rows of chat `i`. The payload: each chat's session ID, then every new token, chat after chat.
pub(super) const SHARED_PASS:u32=32;
impl<C:Compute,E:Exchange> Factory<'_,C,E> {
    /// Whether the passes these chats ask for in one tick can be ONE shared pass: two chats or more, each a strict
    /// chat that drafts with its predictor on, and no more rows in all (rows they carry included) than one check
    /// pass takes. Else each chat keeps its own pass, as before.
    pub(super) fn shares_pass(&self,sessions:&[&Session<'_,C,E>],tokens:&[&[u32]])->bool {
        let rows=sessions.iter().zip(tokens).map(|(session,rows)|session.open.as_ref().map_or(rows.len(),|open|open.shared_count(rows.len()))).sum::<usize>();
        (2..=LIVE_SESSIONS).contains(&sessions.len()) && rows<=crate::bounded_decoder::VERIFY_ROWS
            && sessions.iter().all(|session|session.strict && !matches!(session.phase,Phase::Over|Phase::Trial|Phase::Alone) && session.kept.is_none() && session.decoder.predictor_enabled())
    }
    /// One shared pass on both ranks: `tokens[i]` are the new rows of `sessions[i]`. Each chat gets the logits of its
    /// own rows, with the bits of its own pass (`Session::pass`). The caller validated every row (`validate_row`).
    pub(super) fn shared_pass(&self,sessions:&mut [&mut Session<'_,C,E>],tokens:&[&[u32]])->Result<Vec<Vec<Vec<f32>>>> {
        if let Some(error)=self.failure_reason() {return Err(error);}
        let n=sessions.len();
        if n!=tokens.len() || !(2..=LIVE_SESSIONS).contains(&n) || tokens.iter().any(|rows|rows.is_empty() || rows.len()>crate::bounded_decoder::VERIFY_ROWS) {return Err("row split shared pass exceeds its bound".into());}
        self.finish(self.split.work(|| {
            let started=std::time::Instant::now();
            if self.failed.get() {return Err("row split service previously failed".into());}
            for session in sessions.iter_mut() {
                if session.open.is_none() {let mut fresh=Verify::new(session.state.take().ok_or("row split session has no state")?);if session.decoder.predictor_enabled() {fresh.track_proposer();}session.open=Some(fresh);}
            }
            let places=sessions.iter().map(|session|{let open=session.open.as_ref().unwrap();(open.position(),open.carried())}).collect::<Vec<_>>();
            self.failed.set(true);
            let packed=tokens.iter().enumerate().fold(0u64,|packed,(at,rows)|packed|(rows.len() as u64)<<(8*at));
            self.split.command_for(SHARED_PASS,n as u32,packed)?;
            let mut bytes=sessions.iter().flat_map(|session|session.id.to_le_bytes()).chain(tokens.iter().flat_map(|rows|rows.iter().flat_map(|token|token.to_le_bytes()))).collect::<Vec<_>>();
            self.split.exchange.borrow_mut().command(&mut bytes)?;
            let whole=self.split.rows_peak_begin();
            let logits={
                // Any chat's decoder runs the pass: the chats of one service share its weights and its plans.
                let mut decoder=None;let mut chats=Vec::with_capacity(n);
                for (session,rows) in sessions.iter_mut().zip(tokens) {
                    let session=&mut **session;if decoder.is_none() {decoder=Some(&session.decoder);}
                    chats.push((session.open.as_mut().ok_or("row split shared pass has no open rows")?,*rows));
                }
                crate::bounded_decoder::shared_pass(decoder.ok_or("row split shared pass has no chat")?,&mut chats)?};
            let peak=self.split.rows_peak_end(whole);
            self.split.exchange.borrow_mut().complete()?;self.failed.set(false);
            let seconds=started.elapsed().as_secs_f64();let total=tokens.iter().map(|rows|rows.len()).sum::<usize>();let live=self.active.get();
            {let mut counts=self.shared_counts.get();counts[0]+=1;counts[1]+=total as u64;counts[2]=counts[2].max(n as u64);self.shared_counts.set(counts);}
            for ((session,rows),&(position,carried)) in sessions.iter_mut().zip(tokens).zip(&places) {
                // Each chat's part of the pass, by its rows: the parts add up to the pass, and a pair counts both (pair.rs).
                let share=seconds*rows.len() as f64/total as f64;
                session.gate.pass(rows.len(),share*1000.0,false);
                if rows.len()==1 {self.pair.borrow_mut().turned(session.id,live,0,1,share*1000.0);}
                // Not a steady step, and not a pass of this chat alone: its own step and pass prices do not learn from it.
                session.after_step=false;
                eprintln!("SERVED VERIFY rank={} session={} phase=decode position={position} rows={} carried={carried} picked=0 peak_bytes={peak} seconds={share:.9}",self.split.topology.rank,session.id,rows.len());
            }
            eprintln!("SERVED SHARED rank={} chats={n} rows={total} sessions={:?} seconds={seconds:.9}",self.split.topology.rank,sessions.iter().map(|session|session.id).collect::<Vec<_>>());
            // The same line on both ranks: a stress proves each rank's peak against its own admission line.
            eprintln!("SERVED SHARED PEAK rank={} chats={n} rows={total} peak_bytes={peak}",self.split.topology.rank);
            self.split.cost_report("decode");Ok(logits)
        }))
    }
    /// The worker's side of one shared pass of `chats` chats (`packed`: the new rows of each, a byte a chat).
    pub(super) fn worker_shared(&self,sessions:&mut batch::WorkerSessions<'_,C,E>,chats:usize,packed:u64)->Result<()> {
        if !(2..=LIVE_SESSIONS).contains(&chats) {return Err("worker shared pass exceeds its bound".into());}
        let counts=(0..chats).map(|at|((packed>>(8*at))&0xff) as usize).collect::<Vec<_>>();let total=counts.iter().sum::<usize>();
        if counts.contains(&0) || total>crate::bounded_decoder::VERIFY_ROWS || packed.checked_shr(8*chats as u32).is_some_and(|rest|rest!=0) {return Err("worker shared pass exceeds its rows".into());}
        self.split.work(|| {
            let mut bytes=vec![0;chats*8+total*4];self.split.exchange.borrow_mut().command_payload(&mut bytes)?;
            let ids=bytes[..chats*8].as_chunks::<8>().0.iter().map(|b|u64::from_le_bytes(*b)).collect::<Vec<_>>();
            let tokens=bytes[chats*8..].as_chunks::<4>().0.iter().map(|b|u32::from_le_bytes(*b)).collect::<Vec<_>>();
            if ids.iter().enumerate().any(|(at,id)|ids[..at].contains(id)) {return Err("worker shared pass repeats a session".into());}
            // Each chat's open rows leave its session for the pass and come back after it.
            let mut opens=Vec::with_capacity(chats);
            for id in &ids {let (decoder,state,open)=sessions.get_mut(id).ok_or("worker shared pass has no session")?;
                opens.push(match open.take() {Some(open)=>open,None=>{let mut fresh=Verify::new(state.take().ok_or("worker session has no state")?);if decoder.predictor_enabled() {fresh.track_proposer();}fresh}});}
            let whole=self.split.rows_peak_begin();
            let result={let decoder=&sessions.get(&ids[0]).ok_or("worker shared pass has no session")?.0;let mut at=0;
                let mut rows=opens.iter_mut().zip(&counts).map(|(open,&count)|{let own=&tokens[at..at+count];at+=count;(open,own)}).collect::<Vec<_>>();
                crate::bounded_decoder::shared_pass(decoder,&mut rows).map(|_|())};
            let peak=self.split.rows_peak_end(whole);
            for (id,open) in ids.iter().zip(opens) {if let Some(session)=sessions.get_mut(id) {session.2=Some(open);}}
            result?;
            eprintln!("SERVED SHARED PEAK rank={} chats={chats} rows={total} peak_bytes={peak}",self.split.topology.rank);
            self.split.exchange.borrow_mut().complete()
        })
    }
}
