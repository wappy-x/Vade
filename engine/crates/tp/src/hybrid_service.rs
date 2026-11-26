//! Root-ordered typed sessions; workers execute the same existing arithmetic.
use super::*;
use spark_serve::{Engine as _,Session as _,BatchOutput,Predictions};
use spark_engine::{hybrid::{Model as HybridModel,Session as Local},draft::Target as _};
enum Reply { Empty,Rows(Vec<Vec<f32>>),Plan(Option<usize>),Proposal(Option<spark_engine::draft::ModelProposal>),Drafts(Vec<Option<Vec<u32>>>),Rejected(String) }
pub struct Controller<'m,'a,'b> {
    model:&'m HybridModel<'a,'b>, local:&'m spark_serve::hybrid::Hybrid<'m,'a,'b>, ring:RefCell<Ring>, root:bool,
    protocol:RefCell<HybridProtocol<'m>>, sessions:RefCell<BTreeMap<u64,Local<'m,'a,'b>>>, next:Cell<u64>,
    // Fund every request/cache owner at the largest rank bound; no local-page credit.
    reservation:usize, shutdown:AtomicBool,
}
impl<'m,'a,'b> Controller<'m,'a,'b> {
    pub fn new(model:&'m HybridModel<'a,'b>,local:&'m spark_serve::hybrid::Hybrid<'m,'a,'b>,ring:Ring,ledgers:&[spark_serve::budget::HybridRankLedger])->Result<Self> {
        let group=model.rank_group()?;
        if group.rank()!=ring.rank()||group.split().ranks as usize!=ring.world()||ledgers.len()!=group.members().len(){return Err("hybrid control/data rank mismatch".into());}
        let reservation=ledgers.iter().map(|l|l.request).max().filter(|&n|n!=0).ok_or("missing hybrid request owners")?;
        let root=ring.rank()==0;
        Ok(Self{model,local,ring:RefCell::new(ring),root,protocol:RefCell::new(HybridProtocol::for_model(model)?),sessions:RefCell::new(BTreeMap::new()),next:Cell::new(0),reservation,shutdown:AtomicBool::new(false)})
    }
    fn id(&self)->Result<u64>{let id=self.next.get().checked_add(1).ok_or("hybrid session id overflow")?;self.next.set(id);Ok(id)}
    fn abort(&self){self.shutdown.store(true,Ordering::Relaxed);self.model.rank_group().unwrap().invalidate();self.ring.borrow_mut().abort();}
    fn perform(&self,op:u64,id:u64,arg:u64,tokens:&[u32])->Result<Reply> {
        if self.shutdown.load(Ordering::Relaxed){return Err("hybrid service invalid".into());}
        let result=(|| {
            if self.root {
                let mut header=[0;32];for(out,n)in header.as_chunks_mut::<8>().0.iter_mut().zip([op,id,arg,tokens.len() as u64]){out.copy_from_slice(&n.to_le_bytes());}
                self.ring.borrow_mut().command(&mut header)?;
                if !tokens.is_empty(){self.ring.borrow_mut().command_payload(&mut tokens.iter().flat_map(|n|n.to_le_bytes()).collect::<Vec<_>>())?;}
            }
            let mut protocol=self.protocol.borrow_mut();let mut sessions=self.sessions.borrow_mut();
            let live=(op!=VERIFY||tokens.len()<=5)&&sessions.len()==protocol.state.sessions.len()&&sessions.iter().all(|(id,s)|protocol.state.sessions.get(id)==Some(&s.len())&&s.cache_identity().is_ok_and(|k|k==protocol.identity));
            let previous=protocol.state.clone();let mut expected=previous.clone();let valid=expected.validate(op,id,arg,tokens,self.model.max_tokens(),self.model.config().vocab as usize).is_ok();
            if valid{expected.apply(op,id,arg,tokens);}
            let mut group=self.model.rank_group()?;
            protocol.execute_deciding(&mut group,op,id,arg,tokens,self.model.max_tokens(),self.model.config().vocab as usize,live,|| {
                let executed:Result<Reply>=(||Ok(match op {
                    OPEN=>{sessions.insert(id,self.local.session()?);Reply::Empty},
                    FORK=>{let fork=sessions.get(&id).ok_or("unknown hybrid session")?.fork()?;sessions.insert(arg,fork);Reply::Empty},
                    CLOSE=>{drop(sessions.remove(&id));Reply::Empty}, STOP=>Reply::Empty,
                    _=>{let s=sessions.get_mut(&id).ok_or("unknown hybrid session")?;match op {
                        STEP=>Reply::Rows(vec![s.step(arg as u32)?]), GREEDY=>Reply::Rows(vec![vec![s.step_greedy(arg as u32)? as f32]]),
                        PREFILL=>Reply::Rows(vec![s.prefill(tokens)?]), PREFILL_ADVANCE=>{s.prefill_advance(tokens)?;Reply::Empty},
                        VERIFY=>Reply::Rows(s.verify_mtp(tokens)?), TRUNCATE=>{spark_engine::draft::Target::truncate(s,arg as usize)?;Reply::Empty},
                        PREFIX=>{s.truncate_prefix(arg as usize)?;Reply::Empty}, EXPAND=>Reply::Rows(s.decode_logits(arg as usize)?.unwrap_or_default()),
                        DRAFT_DISABLE=>{s.disable_model_draft();Reply::Empty}, DRAFT_AUTO=>{s.automatic_draft(arg as usize,(u64::from(tokens[0])|(u64::from(tokens[1])<<32)) as usize);Reply::Empty},
                        DRAFT_BUDGET=>{s.model_draft_budget(arg as usize)?;Reply::Empty}, DRAFT_PLAN=>Reply::Plan(s.plan_model_draft(tokens,arg as usize)?),
                        DRAFT_PROPOSE=>Reply::Proposal(s.propose_model(&tokens[..tokens.len()-1],tokens[tokens.len()-1],arg as usize)?),
                        DRAFT_PREPARE=>{let token=[arg as u32];Reply::Drafts(self.local.prepare_drafts(&mut [spark_serve::BatchRows{session:s,tokens:&token,output:BatchOutput::All,greedy:true,top_k:0}]).map_err(|e|e.message)?)},
                        _=>return Err("unsupported hybrid operation".into()),
                    }}
                }))();
                let reply=match executed {Err(e) if e==spark_gpu_probe::driver::MOE_ROUTING_ERROR=>{expected=previous;Reply::Rejected(e)},v=>v?};
                if let Reply::Rows(rows)=&reply {for row in rows{finite(row)?;}}
                if sessions.len()!=expected.sessions.len()||sessions.iter().any(|(id,s)|expected.sessions.get(id)!=Some(&s.len())) {return Err("hybrid live positions differ after execution".into());}
                let schedule=match &reply{Reply::Plan(v)=>vec![1,v.map_or(u64::MAX,|n|n as u64)],Reply::Proposal(v)=>std::iter::once(2).chain(v.iter().flat_map(|p|p.tokens.iter().map(|&t|u64::from(t)))).collect(),Reply::Drafts(v)=>std::iter::once(3).chain(v.iter().flat_map(|v|std::iter::once(v.as_ref().map_or(u64::MAX,|v|v.len() as u64)).chain(v.iter().flatten().map(|&t|u64::from(t))))).collect(),_=>vec![0]};
                self.model.rank_group()?.agree(&schedule,&[])?;
                let commit=!matches!(reply,Reply::Rejected(_));Ok((reply,commit))
            })
        })();
        if self.protocol.borrow().poisoned||result.as_ref().is_err_and(|e|e!="hybrid command rejected by rank group"){self.abort();}result
    }
    pub fn worker(&self)->Result<()> {
        if self.root{return Err("hybrid root must drive the scheduler".into());}
        let result=(||{loop {
            let mut header=[0;32];self.ring.borrow_mut().command(&mut header)?;
            let w=header.as_chunks::<8>().0.iter().map(|b|u64::from_le_bytes(*b)).collect::<Vec<_>>();
            if w[3]>self.protocol.borrow().state.payload_limit(w[0],self.model.max_tokens()) as u64{return Err("hybrid command extent".into());}
            let mut data=vec![0;w[3] as usize*4];if !data.is_empty(){self.ring.borrow_mut().command_payload(&mut data)?;}
            let tokens=data.as_chunks::<4>().0.iter().map(|b|u32::from_le_bytes(*b)).collect::<Vec<_>>();
            match self.perform(w[0],w[1],w[2],&tokens){Err(e) if e=="hybrid command rejected by rank group"=>continue,v=>{v?;}}
            if w[0]==STOP{return Ok(());}
        }})();if result.is_err(){self.abort();}result
    }
    fn validate_rows(&self,rows:&[spark_serve::BatchRows<'_,Session<'_,'m,'a,'b>>])->std::result::Result<(),spark_serve::BatchError>{
        let mut seen=std::collections::BTreeSet::new();for(i,r)in rows.iter().enumerate(){
            if r.session.closed||!std::ptr::eq(r.session.engine,self)||!seen.insert(r.session.id)||r.tokens.is_empty()||r.tokens.iter().any(|&t|t as usize>=self.vocab_size())||r.session.observe(|s|r.tokens.len()>self.context_limit().saturating_sub(s.len()))!=Some(false){return Err(spark_serve::BatchError{request:Some(i),message:"invalid hybrid batch row".into()});}
        }Ok(())
    }
    pub fn shutdown(&self)->Result<()>{self.perform(STOP,0,0,&[]).map(|_|())}
    pub fn shutdown_requested(&self)->&AtomicBool{&self.shutdown}
}
impl Drop for Controller<'_, '_, '_>{fn drop(&mut self){self.ring.borrow_mut().abort();}}
pub struct Session<'e,'m,'a,'b>{engine:&'e Controller<'m,'a,'b>,id:u64,closed:bool}
impl<'e,'m,'a,'b> Session<'e,'m,'a,'b> {
    fn call(&self,op:u64,arg:u64,tokens:&[u32])->Result<Reply>{if self.closed{return Err("hybrid session closed".into());}self.engine.perform(op,self.id,arg,tokens)}
    fn rows(&self,op:u64,arg:u64,tokens:&[u32])->Result<Vec<Vec<f32>>>{match self.call(op,arg,tokens)?{Reply::Rows(v)=>Ok(v),Reply::Rejected(e)=>Err(e),_=>Err("hybrid output type".into())}}
    // CLOSE removes `id` from `sessions`: never index a closed or absent entry.
    fn observe<T>(&self,f:impl FnOnce(&Local<'m,'a,'b>)->T)->Option<T>{observe_row(&self.engine.sessions.borrow(),self.id,self.closed,f)}
}
fn observe_row<S,T>(sessions:&BTreeMap<u64,S>,id:u64,closed:bool,f:impl FnOnce(&S)->T)->Option<T>{if closed{None}else{sessions.get(&id).map(f)}}
impl Drop for Session<'_, '_, '_, '_>{fn drop(&mut self){let _=spark_serve::Session::finish(self);}}
impl spark_serve::Session for Session<'_, '_, '_, '_> {
    fn step(&mut self,t:u32)->Result<Vec<f32>>{self.rows(STEP,t as u64,&[]).map(|mut v|v.remove(0))}
    fn prefill(&mut self,t:&[u32])->Result<Vec<f32>>{self.rows(PREFILL,t.len() as u64,t).map(|mut v|v.remove(0))}
    fn prefill_advance(&mut self,t:&[u32])->Result<()>{let len=self.observe(|s|s.len()).ok_or("hybrid session closed")?;self.call(PREFILL_ADVANCE,len as u64+t.len() as u64,t).map(|_|())}
    fn prefill_slice(&mut self,t:&[u32],last:bool,_:Option<std::time::Duration>,_:usize)->Result<Option<Vec<f32>>>{if last{self.prefill(t).map(Some)}else{self.prefill_advance(t).map(|_|Some(Vec::new()))}}
    fn prefill_chunk_size(&self,end:usize)->usize{self.observe(|s|spark_serve::Session::prefill_chunk_size(s,end)).unwrap_or(512)}
    fn prefill_chunk_min(&self)->usize{if self.closed{1}else{self.prefill_chunk_size(0)}}
    fn cached_prompt_tokens(&self,n:usize)->Option<usize>{self.observe(|s|spark_serve::Session::cached_prompt_tokens(s,n)).flatten()}
    fn can_fork(&self)->bool{self.observe(|s|s.can_fork()).unwrap_or(false)}
    fn can_fork_prefix(&self,p:usize)->bool{self.observe(|s|spark_serve::Session::can_fork_prefix(s,p)).unwrap_or(false)}
    fn fork(&self)->Result<Self>{let id=self.engine.id()?;self.call(FORK,id,&[])?;Ok(Self{engine:self.engine,id,closed:false})}
    fn fork_prefix(&self,p:usize)->Result<Self>{let s=self.fork()?;s.call(PREFIX,p as u64,&[])?;Ok(s)}
    fn truncate(&mut self,p:usize)->Result<()>{self.call(TRUNCATE,p as u64,&[]).map(|_|())}
    fn decode_logits(&mut self,n:usize)->Result<Option<Vec<Vec<f32>>>>{self.rows(EXPAND,n as u64,&[]).map(|v|(!v.is_empty()).then_some(v))}
    fn finish(&mut self)->Result<()>{if self.closed{return Ok(());}let result=self.call(CLOSE,0,&[]);self.closed=true;result.map(|_|())}
    fn retained_bytes(&self)->Option<usize>{Some(self.engine.reservation)}
    fn memory_bytes(&self)->Option<(usize,usize)>{self.observe(spark_serve::Session::memory_bytes).flatten()}
    fn draft_model_memory_bytes(&self)->Option<(usize,usize)>{self.observe(spark_serve::Session::draft_model_memory_bytes).flatten()}
    fn checkpoint_reusable_tokens(&self,n:usize)->usize{self.observe(|s|spark_serve::Session::checkpoint_reusable_tokens(s,n)).unwrap_or(usize::MAX)}
    fn disable_model_draft(&mut self){let _=self.call(DRAFT_DISABLE,0,&[]);}
    fn automatic_draft(&mut self,p:usize,r:usize){let _=self.call(DRAFT_AUTO,p as u64,&[r as u32,(r as u64>>32) as u32]);}
    fn draft_target(&mut self)->Option<&mut dyn spark_engine::draft::Target>{(self.model_draft_width()!=0).then_some(self)}
}
impl spark_engine::draft::Target for Session<'_, '_, '_, '_> {
    fn checkpoint(&self)->usize{self.observe(|s|s.len()).unwrap_or(0)}
    fn truncate(&mut self,p:usize)->Result<()>{spark_serve::Session::truncate(self,p)}
    fn step(&mut self,t:u32)->Result<Vec<f32>>{spark_serve::Session::step(self,t)}
    fn step_greedy(&mut self,t:u32)->Result<u32>{self.rows(GREEDY,t as u64,&[]).map(|v|v[0][0] as u32)}
    fn verify_batch(&mut self,t:&[u32])->Result<Vec<Vec<f32>>>{self.rows(VERIFY,t.len() as u64,t)}
    fn kv_bf16(&self)->bool{true}
    fn max_verify_rows(&self)->usize{5}
    fn model_draft_width(&self)->usize{self.observe(|s|s.model_draft_width()).unwrap_or(0)}
    fn native_draft_policy(&self)->bool{true}
    fn model_draft_budget(&mut self,r:usize)->Result<()>{self.call(DRAFT_BUDGET,r as u64,&[]).map(|_|())}
    fn plan_model_draft(&mut self,h:&[u32],n:usize)->Result<Option<usize>>{match self.call(DRAFT_PLAN,n as u64,h)?{Reply::Plan(v)=>Ok(v),_=>Err("hybrid draft plan type".into())}}
    fn propose_model(&mut self,h:&[u32],t:u32,n:usize)->Result<Option<spark_engine::draft::ModelProposal>>{let mut h=h.to_vec();h.push(t);match self.call(DRAFT_PROPOSE,n as u64,&h)?{Reply::Proposal(v)=>Ok(v),_=>Err("hybrid draft proposal type".into())}}
}
impl<'m,'a,'b> spark_serve::Engine for Controller<'m,'a,'b> {
    type Session<'s>=Session<'s,'m,'a,'b> where Self:'s;
    fn session(&self)->Result<Self::Session<'_>>{let id=self.id()?;self.perform(OPEN,id,0,&[])?;Ok(Session{engine:self,id,closed:false})}
    fn context_limit(&self)->usize{self.local.context_limit()}
    fn vocab_size(&self)->usize{self.local.vocab_size()}
    fn eos_token_ids(&self)->&[u32]{self.local.eos_token_ids()}
    fn fatal_error(&self)->Option<String>{self.shutdown.load(Ordering::Relaxed).then(||"hybrid service invalid".into()).or_else(||self.local.fatal_error())}
    fn session_reservation_bytes(&self)->Option<usize>{Some(self.reservation)}
    fn request_reservation_bytes(&self,_:usize)->Option<usize>{Some(self.reservation)}
    fn shared_batch(&self)->bool{true}
    fn shared_prefill_limit(&self)->usize{self.local.shared_prefill_limit()}
    fn shared_prefill_rows(&self)->usize{self.local.shared_prefill_rows()}
    fn reserve_foreground_slot(&self)->bool{false}
    fn shared_model_drafts(&self)->bool{self.local.shared_model_drafts()}
    fn native_draft_policy(&self)->bool{true}
    fn automatic_draft_limit(&self)->usize{1}
    fn automatic_draft_min_output(&self)->usize{self.local.automatic_draft_min_output()}
    fn decode_row_limit(&self,n:usize)->usize{self.local.decode_row_limit(n)}
    fn prepare_drafts<'s>(&'s self,rows:&mut [spark_serve::BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<Option<Vec<u32>>>,spark_serve::BatchError>{self.validate_rows(rows)?;rows.iter().map(|r|match r.session.call(DRAFT_PREPARE,r.tokens[0] as u64,&[]){Ok(Reply::Drafts(mut v))=>Ok(v.remove(0)),Ok(_)=>Err(spark_serve::BatchError{request:None,message:"hybrid draft output type".into()}),Err(message)=>Err(spark_serve::BatchError{request:None,message})}).collect()}
    fn step_batch<'s>(&'s self,rows:&mut [spark_serve::BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<Predictions>,spark_serve::BatchError>{self.validate_rows(rows)?;rows.iter_mut().map(|r|{
        let result=match r.output{BatchOutput::All if r.tokens.len()>1||r.session.model_draft_width()>0=>r.session.verify_batch(r.tokens).map(Predictions::Logits),BatchOutput::All=>spark_serve::Session::step(r.session,r.tokens[0]).map(|v|Predictions::Logits(vec![v])),BatchOutput::Last=>r.session.prefill(r.tokens).map(|v|Predictions::Logits(vec![v])),BatchOutput::None=>r.session.prefill_advance(r.tokens).map(|_|Predictions::Logits(Vec::new()))};
        match result{Err(e) if e==spark_gpu_probe::driver::MOE_ROUTING_ERROR=>Ok(Predictions::Rejected(e)),v=>v.map_err(|message|spark_serve::BatchError{request:None,message})}
    }).collect()}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_or_removed_hybrid_session_returns_none_instead_of_indexing() {
        let mut sessions = BTreeMap::from([(7_u64, 128_usize)]);
        assert_eq!(observe_row(&sessions, 7, false, |&n| n * 2), Some(256));
        assert_eq!(observe_row(&sessions, 7, true, |&n| n * 2), None);
        sessions.remove(&7);
        assert_eq!(observe_row(&sessions, 7, false, |&n| n * 2), None);
        assert_eq!(observe_row(&sessions, 7, true, |&n| n * 2), None);
    }
}
