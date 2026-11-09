//! Rank-zero session commands, shared by service and speculative callers.
use crate::{Result,model::{Model,Multi,Session},transport::Ring};
use std::{cell::{Cell,RefCell},collections::BTreeMap,sync::atomic::{AtomicBool,Ordering}};
use crate::service_commands::*;
pub const SEQUENCE_THRESHOLD:usize=6144;

pub struct Engine<'m,'a,'b> {
    model:&'m Model<'a,'b>, ring:RefCell<Ring>, next:Cell<u64>, failure:RefCell<Option<String>>, shutdown:AtomicBool, eos:[u32;2],
    sequence:RefCell<Option<spark_seqpar::Ring>>, sequence_enabled:bool,
    multi:RefCell<Option<Multi<'m,'a,'b>>>,
    state:RefCell<CommandState>, previous_state:RefCell<Option<CommandState>>,
}
impl<'m,'a,'b> Engine<'m,'a,'b> {
    pub fn new(model:&'m Model<'a,'b>,ring:Ring)->Result<Self> {
        if ring.rank()!=0{return Err("service controller must be rank zero".into());}
        Ok(Self{model,ring:RefCell::new(ring),next:Cell::new(0),failure:RefCell::new(None),shutdown:AtomicBool::new(false),eos:[model.config().eos_token as u32,model.config().bos_token as u32],sequence:RefCell::new(None),sequence_enabled:false,multi:RefCell::new(None),state:RefCell::new(CommandState::default()),previous_state:RefCell::new(None)})
    }
    pub fn new_with_sequence(model:&'m Model<'a,'b>,ring:Ring,sequence:Option<spark_seqpar::Ring>)->Result<Self>{
        let mut engine=Self::new(model,ring)?;engine.sequence=RefCell::new(sequence);engine.sequence_enabled=model.sequence_prefill();Ok(engine)
    }
    fn id(&self)->Result<u64>{let id=self.next.get().checked_add(1).ok_or("session id overflow")?;self.next.set(id);Ok(id)}
    fn result<T>(&self,result:Result<T>)->Result<T>{
        if let Err(error)=&result {
            if error==OUTPUT_ERROR {
                if let Some(previous)=self.previous_state.borrow_mut().take(){*self.state.borrow_mut()=previous;}
                return result;
            }
            spark_gpu_probe::teardown::arm();
            if self.failure.borrow().is_none(){
                eprintln!("TP rank zero failed: {error}");
                *self.failure.borrow_mut()=Some(error.clone());
                self.shutdown.store(true,Ordering::Relaxed);
                self.ring.borrow_mut().abort();
                if let Some(ring)=self.sequence.borrow_mut().as_mut(){ring.abort();}
            }
        }
        result
    }
    pub fn shutdown_requested(&self)->&AtomicBool{&self.shutdown}
    fn command(&self,op:u64,id:u64,arg:u64,tokens:&[u32])->Result<()> {
        if let Some(error)=self.failure.borrow().as_ref(){return Err(error.clone());}
        self.state.borrow().validate(op,id,arg,tokens,self.model.max_tokens(),self.model.config().vocab_size as usize)?;
        let result=(||{
            let mut header=[0u8;32];
            for(dst,n)in header.chunks_exact_mut(8).zip([op,id,arg,tokens.len() as u64]){dst.copy_from_slice(&n.to_le_bytes());}
            self.ring.borrow_mut().command(&mut header)?;
            if !tokens.is_empty(){let mut data=tokens.iter().flat_map(|id|id.to_le_bytes()).collect::<Vec<_>>();self.ring.borrow_mut().command_payload(&mut data)?;}
            *self.previous_state.borrow_mut()=Some(self.state.borrow().clone());
            self.state.borrow_mut().apply(op,id,arg,tokens);
            Ok(())
        })();
        self.result(result)
    }
    pub fn session(&self)->Result<RemoteSession<'_,'m,'a,'b>> {
        let id=self.id()?;self.command(OPEN,id,0,&[])?;
        let mut session=RemoteSession{engine:self,id,local:self.model.session(&self.ring),closed:false};
        // KV at the model's precision (BF16), as on one Spark. Hybrid sessions keep FP32 KV only with the
        // opt-in sequence prefill (Model::sequence_prefill), whose hand-off changes output.
        if self.model.full_model().is_some()&&!self.model.sequence_prefill(){session.configure_kv_bf16(true)?;}
        Ok(session)
    }
    /// One decode token per chat, sharing the shard projections and reductions.
    pub fn decode_batch<'e>(&'e self,sessions:&mut [&mut RemoteSession<'e,'m,'a,'b>],tokens:&[u32],greedy:bool)->Result<Vec<Vec<f32>>>{
        if sessions.len()!=tokens.len(){return Err("decode batch session/token count mismatch".into());}
        if sessions.is_empty(){return Ok(Vec::new());}
        for(i,(session,&token))in sessions.iter().zip(tokens).enumerate(){
            if !std::ptr::eq(session.engine,self){return Err("decode batch belongs to another engine".into());}
            if sessions[..i].iter().any(|s|s.id==session.id){return Err("duplicate decode batch session".into());}
            session.check(&[token])?;
        }
        let result=(||{
            if sessions.len()==1||self.model.full_model().is_some(){
                return sessions.iter_mut().zip(tokens).map(|(session,&token)|
                    if greedy{session.step_greedy(token).map(|id|vec![id as f32])}else{session.step(token)}).collect();
            }
            let mut multi=self.multi.borrow_mut();
            if multi.is_none(){*multi=Some(Multi::new(self.model)?);}
            let mut command=Vec::with_capacity(sessions.len()*3);
            for(session,&token)in sessions.iter().zip(tokens){command.extend_from_slice(&[session.id as u32,(session.id>>32) as u32,token]);}
            self.command(DECODE_BATCH,0,greedy as u64,&command)?;
            let mut local=sessions.iter_mut().map(|s|&mut s.local).collect::<Vec<_>>();
            multi.as_mut().unwrap().step(&mut local,tokens,greedy)
        })();
        self.result(result)
    }
    /// Waiting prompts share each shard projection; every chat keeps its own KV.
    pub fn prefill_batch<'e>(&'e self,sessions:&mut [&mut RemoteSession<'e,'m,'a,'b>],tokens:&[&[u32]],greedy:bool)->Result<Vec<Vec<f32>>>{
        if sessions.len()!=tokens.len(){return Err("prefill batch session/token count mismatch".into());}
        for(i,(session,tokens))in sessions.iter().zip(tokens).enumerate(){
            if !std::ptr::eq(session.engine,self){return Err("prefill batch belongs to another engine".into());}
            if sessions[..i].iter().any(|s|s.id==session.id){return Err("duplicate prefill batch session".into());}
            session.check(tokens)?;
        }
        let result=(||{
            let mut output=Vec::with_capacity(sessions.len());
            let mut first=0;
            while first<sessions.len(){
                let mut end=first;let mut rows=0;
                while end<sessions.len()&&rows+tokens[end].len()<=2048{rows+=tokens[end].len();end+=1;}
                end=end.max(first+1);
                if end-first==1||self.model.full_model().is_some(){
                    for(session,tokens)in sessions[first..end].iter_mut().zip(&tokens[first..end]){
                        let logits=session.prefill(tokens)?;
                        output.push(if greedy{vec![spark_engine::greedy_token(&logits)? as f32]}else{logits});
                    }
                }else{
                    let mut multi=self.multi.borrow_mut();
                    if multi.is_none(){*multi=Some(Multi::new(self.model)?);}
                    let mut command=Vec::with_capacity(rows+3*(end-first));
                    for(session,tokens)in sessions[first..end].iter().zip(&tokens[first..end]){
                        command.extend_from_slice(&[session.id as u32,(session.id>>32) as u32,tokens.len() as u32]);
                        command.extend_from_slice(tokens);
                    }
                    self.command(PREFILL_BATCH,0,greedy as u64,&command)?;
                    let mut local=sessions[first..end].iter_mut().map(|s|&mut s.local).collect::<Vec<_>>();
                    output.extend(multi.as_mut().unwrap().prefill(&mut local,&tokens[first..end],greedy)?);
                }
                first=end;
            }
            Ok(output)
        })();
        self.result(result)
    }
    pub fn shutdown(&self)->Result<()>{self.command(STOP,0,0,&[])}
}

pub struct RemoteSession<'e,'m,'a,'b>{engine:&'e Engine<'m,'a,'b>,id:u64,local:Session<'m,'e,'a,'b>,closed:bool}
impl<'e,'m,'a,'b> RemoteSession<'e,'m,'a,'b>{
    fn result<T>(&self,result:Result<T>)->Result<T>{self.engine.result(result)}
    fn check(&self,tokens:&[u32])->Result<()> {
        if self.closed{return Err("TP session is closed".into());}
        if !self.local.valid(){return Err("TP session invalid after failed collective".into());}
        if tokens.is_empty()||tokens.len()>self.engine.model.max_tokens().saturating_sub(self.local.len())||tokens.iter().any(|&t|t as u64>=self.engine.model.config().vocab_size){return Err("batch outside vocabulary or capacity".into());}Ok(())
    }
    pub fn len(&self)->usize{self.local.len()}
    pub fn is_empty(&self)->bool{self.local.is_empty()}
    pub fn checkpoint(&self)->usize{self.local.checkpoint()}
    pub fn cache_bytes(&self)->usize{self.local.cache_bytes()}
    pub fn step(&mut self,token:u32)->Result<Vec<f32>>{
        self.check(&[token])?;self.engine.command(STEP,self.id,token as u64,&[])?;
        let result=self.local.step(token);self.result(result)
    }
    pub fn step_greedy(&mut self,token:u32)->Result<u32>{
        self.check(&[token])?;self.engine.command(GREEDY,self.id,token as u64,&[])?;
        let result=self.local.step_greedy(token);self.result(result)
    }
    pub fn forward_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>>{self.verify_batch(tokens)}
    pub fn verify_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>>{
        if tokens.len()>512{return Err("verify_batch accepts at most 512 rows".into());}
        self.check(tokens)?;self.engine.command(VERIFY,self.id,tokens.len() as u64,tokens)?;
        let result=self.local.verify_batch(tokens);self.result(result)
    }
    /// Explicit approximate rows; speculative acceptance requires a guarded fallback.
    pub fn forward_batch_fast(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>>{
        if tokens.len()>512{return Err("forward_batch_fast accepts at most 512 rows".into());}
        self.check(tokens)?;self.engine.command(FAST,self.id,tokens.len() as u64,tokens)?;
        let result=self.local.forward_batch_fast(tokens);self.result(result)
    }
    pub fn configure_kv_bf16(&mut self,enabled:bool)->Result<()>{
        if self.closed||self.cache_bytes()!=0||!self.is_empty(){return Err("KV format must be selected before first forward".into());}
        self.engine.command(KV_FORMAT,self.id,enabled as u64,&[])?;
        let result=self.local.configure_kv_bf16(enabled);self.result(result)
    }
    pub fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>>{
        self.prefill_chunk(tokens,self.len()+tokens.len(),true)
    }
    pub fn prefill_chunk(&mut self,tokens:&[u32],prompt_end:usize,final_chunk:bool)->Result<Vec<f32>>{
        self.check(tokens)?;
        if prompt_end<self.len()+tokens.len()||prompt_end>self.engine.model.max_tokens()||final_chunk&&prompt_end!=self.len()+tokens.len(){return Err("invalid prefill prompt end".into());}
        let sequence=self.engine.sequence_enabled&&prompt_end>=SEQUENCE_THRESHOLD&&self.local.can_sequence();
        let op=match(sequence,final_chunk){(true,true)=>SEQUENCE_END,(true,false)=>SEQUENCE_ADVANCE,(false,true)=>PREFILL_END,(false,false)=>PREFILL_ADVANCE};
        self.engine.command(op,self.id,prompt_end as u64,tokens)?;
        let result=if sequence{self.local.prefill_sequence(tokens,prompt_end,final_chunk,self.engine.sequence.borrow_mut().as_mut())}
            else{self.local.prefill_chunk(tokens,prompt_end,final_chunk)};
        self.result(result)
    }
    pub fn truncate(&mut self,position:usize)->Result<()>{
        if self.closed||position>self.len(){return Err("invalid truncate position".into());}
        if position==self.len() && self.local.can_fork() && self.engine.failure.borrow().is_none() { return Ok(()); }
        self.engine.command(TRUNCATE,self.id,position as u64,&[])?;
        let result=self.local.truncate(position);self.result(result)
    }
    pub fn fork(&self)->Result<Self>{
        if self.closed{return Err("TP session is closed".into());}
        // A local allocation failure must not leave workers with an orphan fork.
        let local=self.local.fork()?;
        let id=self.engine.id()?;self.engine.command(FORK,self.id,id,&[])?;
        Ok(Self{engine:self.engine,id,local,closed:false})
    }
    fn start_sequence(&mut self,tokens:&[u32])->Result<()> {
        self.check(tokens)?;
        if !self.local.can_sequence()||!self.local.is_empty(){return Err("invalid resumable sequence prompt".into());}
        self.engine.command(SEQUENCE_START,self.id,tokens.len() as u64,tokens)?;
        let result=self.local.start_sequence(tokens,self.engine.sequence.borrow().as_ref());
        self.result(result)
    }
    fn advance_sequence(&mut self,budget:std::time::Duration)->Result<Option<Vec<f32>>> {
        if self.closed{return Err("TP session is closed".into());}
        if !self.local.valid(){return Err("TP session invalid after failed collective".into());}
        let micros=budget.as_micros().min(u64::MAX as u128) as u64;
        self.engine.command(SEQUENCE_SLICE,self.id,micros,&[])?;
        let result=self.local.advance_sequence(std::time::Duration::from_micros(micros),self.engine.sequence.borrow_mut().as_mut());
        self.result(result)
    }
    pub fn finish(&mut self)->Result<()>{
        if self.closed{return Ok(());}self.closed=true;
        #[cfg(target_os="linux")]
        if let Some(ring)=self.engine.sequence.borrow_mut().as_mut(){ring.retain_regions(false);}
        eprintln!("tp_session_closed rank=0 session={} kv_bytes={} session_device_peak_bytes={} pinned_peak_bytes={}",self.id,self.local.cache_bytes(),self.local.device_peak_bytes(),self.local.pinned_peak_bytes());
        self.engine.command(CLOSE,self.id,0,&[])
    }
}
impl Drop for RemoteSession<'_,'_,'_,'_>{fn drop(&mut self){let _=self.finish();}}

impl spark_serve::Session for RemoteSession<'_,'_,'_,'_>{
    fn draft_target(&mut self)->Option<&mut dyn spark_engine::draft::Target>{Some(self)}
    fn memory_bytes(&self)->Option<(usize,usize)>{Some((self.local.cache_bytes(),self.local.device_peak_bytes()))}
    fn retained_bytes(&self)->Option<usize>{self.local.cache_allocations().map(|a|a.iter().map(|(_,bytes)|bytes).sum())}
    fn retained_allocations(&self)->Option<Vec<(usize,usize)>>{self.local.cache_allocations()}
    fn prefix_allocations(&self,tokens:usize)->Option<Vec<(usize,usize)>>{self.local.prefix_allocations(tokens)}
    fn step(&mut self,token:u32)->Result<Vec<f32>>{Self::step(self,token)}
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>>{Self::prefill(self,tokens)}
    fn prefill_advance(&mut self,tokens:&[u32])->Result<()>{Self::prefill(self,tokens).map(|_|())}
    fn prefill_chunk_size(&self,prompt_end:usize)->usize{self.local.prefill_chunk_size(prompt_end)}
    fn prefill_slice(&mut self,tokens:&[u32],final_chunk:bool,_budget:Option<std::time::Duration>,prompt_end:usize)->Result<Option<Vec<f32>>>{Self::prefill_chunk(self,tokens,prompt_end,final_chunk).map(Some)}
    fn fork(&self)->Result<Self>{Self::fork(self)}
    fn can_fork(&self)->bool{!self.closed&&self.local.can_fork()}
    fn truncate(&mut self,position:usize)->Result<()>{Self::truncate(self,position)}
    fn finish(&mut self)->Result<()>{Self::finish(self)}
}
impl spark_engine::draft::Target for RemoteSession<'_,'_,'_,'_>{
    fn checkpoint(&self)->usize{Self::checkpoint(self)}
    fn truncate(&mut self,position:usize)->Result<()>{Self::truncate(self,position)}
    fn step(&mut self,token:u32)->Result<Vec<f32>>{Self::step(self,token)}
    fn step_greedy(&mut self,token:u32)->Result<u32>{Self::step_greedy(self,token)}
    fn verify_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>>{Self::verify_batch(self,tokens)}
    fn forward_batch(&mut self,tokens:&[u32])->Result<Vec<Vec<f32>>>{Self::forward_batch_fast(self,tokens)}
    fn kv_bf16(&self)->bool{self.local.kv_bf16()}
}
struct SequencePrefill { tokens:Vec<u32>, started:bool }
impl spark_serve::LongPrefill<RemoteSession<'_,'_,'_,'_>> for SequencePrefill {
    fn advance(&mut self,session:&mut RemoteSession<'_,'_,'_,'_>,budget:std::time::Duration)->Result<Option<Vec<f32>>> {
        if !self.started {
            session.start_sequence(&self.tokens)?;
            self.tokens.clear();self.started=true;
        }
        session.advance_sequence(budget)
    }
}
impl<'m,'a,'b> spark_serve::Engine for Engine<'m,'a,'b>{
    type Session<'s>=RemoteSession<'s,'m,'a,'b> where Self:'s;
    fn session(&self)->Result<Self::Session<'_>>{Self::session(self)}
    fn prefill_long<'s>(&'s self,session:&Self::Session<'s>,tokens:&[u32])
        ->Result<Option<Box<dyn spark_serve::LongPrefill<Self::Session<'s>>+'s>>> {
        if !self.sequence_enabled||tokens.len()<SEQUENCE_THRESHOLD||!session.local.can_sequence()||!session.is_empty(){return Ok(None);}
        Ok(Some(Box::new(SequencePrefill{tokens:tokens.to_vec(),started:false})))
    }
    fn shared_batch(&self)->bool{self.model.full_model().is_none()}
    fn shared_prompt_limit(&self)->usize{if self.shared_batch(){256}else{8}}
    fn prefill_batch<'s>(&'s self,rows:&mut [spark_serve::BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<spark_serve::Predictions>,spark_serve::BatchError>{
        use spark_serve::{BatchOutput,Predictions,BatchError};
        for(i,row)in rows.iter().enumerate(){
            if matches!(row.output,BatchOutput::All){return Err(BatchError{request:Some(i),message:"prefill accepts Last or None output".into()});}
            row.session.check(row.tokens).map_err(|message|BatchError{request:Some(i),message})?;
        }
        let greedy=rows.iter().all(|r|r.greedy);
        let tokens=rows.iter().map(|r|r.tokens).collect::<Vec<_>>();
        let mut sessions=rows.iter_mut().map(|r|&mut *r.session).collect::<Vec<_>>();
        let result=self.prefill_batch(&mut sessions,&tokens,greedy).map_err(|message|BatchError{request:None,message})?;
        Ok(result.into_iter().zip(rows).map(|(v,row)|match(row.output,greedy){
            (BatchOutput::None,_)=>Predictions::Logits(Vec::new()),
            (_,true)=>Predictions::Greedy(vec![v[0] as u32]),
            _=>Predictions::Logits(vec![v]),
        }).collect())
    }
    fn step_batch<'s>(&'s self,rows:&mut [spark_serve::BatchRows<'_,Self::Session<'s>>])->std::result::Result<Vec<spark_serve::Predictions>,spark_serve::BatchError>{
        use spark_serve::{BatchOutput,Predictions,BatchError};
        // Validate before advancing any chat; a rank failure invalidates every row.
        let mut seen=std::collections::BTreeSet::new();
        for(i,row)in rows.iter().enumerate(){
            if !std::ptr::eq(row.session.engine,self)||!seen.insert(row.session.id){
                return Err(BatchError{request:Some(i),message:"invalid TP batch session".into()});
            }
            row.session.check(row.tokens).map_err(|message|BatchError{request:Some(i),message})?;
        }
        if rows.iter().all(|r|r.tokens.len()==1&&matches!(r.output,BatchOutput::All)){
            let greedy=rows.iter().all(|r|r.greedy);
            let tokens=rows.iter().map(|r|r.tokens[0]).collect::<Vec<_>>();
            let mut sessions=rows.iter_mut().map(|r|&mut *r.session).collect::<Vec<_>>();
            return self.decode_batch(&mut sessions,&tokens,greedy)
                .map(|output|output.into_iter().map(|row|if greedy{Predictions::Greedy(vec![row[0] as u32])}else{Predictions::Logits(vec![row])}).collect())
                .map_err(|message|BatchError{request:None,message});
        }
        rows.iter_mut().map(|row|{
            let result=if row.greedy&&matches!(row.output,BatchOutput::All)&&row.tokens.len()==1 {
                row.session.step_greedy(row.tokens[0]).map(|token|Predictions::Greedy(vec![token]))
            }else{
                match row.output {
                    BatchOutput::All=>row.tokens.iter().map(|&token|row.session.step(token)).collect(),
                    BatchOutput::Last=>row.session.prefill(row.tokens).map(|v|vec![v]),
                    BatchOutput::None=>row.session.prefill(row.tokens).map(|_|Vec::new()),
                }.map(Predictions::Logits)
            };
            result.map_err(|message|BatchError{request:None,message})
        }).collect()
    }
    fn fatal_error(&self)->Option<String>{self.failure.borrow().clone().or_else(||self.model.fatal_error())}
    fn validate_request(&self,prompt:&[u32],request:&spark_serve::protocol::Request)->Result<()>{
        validate_request(prompt,request,self.model.max_tokens(),self.model.config().vocab_size as usize)
    }
    fn context_limit(&self)->usize{self.model.max_tokens()}
    fn vocab_size(&self)->usize{self.model.config().vocab_size as usize}
    fn eos_token_ids(&self)->&[u32]{&self.eos}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize>{Some(self.model.request_reservation_bytes(tokens))}
    fn session_reservation_bytes(&self)->Option<usize>{Some(self.model.session_reservation_bytes())}
    fn memory_bytes(&self)->Option<(usize,usize,usize)>{Some(self.model.memory_bytes())}
}

/// Nonzero ranks own their own model/KV and sleep between root commands.
pub fn worker(model:&Model<'_,'_>,ring:Ring)->Result<()> {
    worker_with_sequence(model,ring,None)
}
pub fn worker_with_sequence(model:&Model<'_,'_>,ring:Ring,mut sequence:Option<spark_seqpar::Ring>)->Result<()> {
    if ring.rank()==0{return Err("rank zero must drive the session API".into());}
    let ring=RefCell::new(ring);let mut sessions=BTreeMap::new();let mut multi=None;
    let mut state=CommandState::default();
    let mut reported_peak=model.memory_bytes().1;
    let mut memory_report=std::time::Instant::now();
    let result=(||{loop{
        let mut header=[0u8;32];ring.borrow_mut().command(&mut header)?;
        if memory_report.elapsed()>=std::time::Duration::from_secs(60) {
            let (current,peak,pinned)=model.memory_bytes();
            eprintln!("tp_worker_memory rank={} device_bytes={current} peak_device_bytes={peak} pinned_host_bytes={pinned} process_rss_bytes={}",ring.borrow().rank(),crate::runtime::host_memory().0);
            memory_report=std::time::Instant::now();
        }
        let mut words=header.chunks_exact(8).map(|b|u64::from_le_bytes(b.try_into().unwrap()));
        let(op,id,arg,count)=(words.next().unwrap(),words.next().unwrap(),words.next().unwrap(),words.next().unwrap());
        if count>state.payload_limit(op,model.max_tokens()) as u64{return Err("TP rank command header inconsistency".into());}
        let mut data=vec![0u8;count as usize*4];
        if count!=0{ring.borrow_mut().command_payload(&mut data)?;}
        let tokens=data.chunks_exact(4).map(|b|u32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        state.validate(op,id,arg,&tokens,model.max_tokens(),model.config().vocab_size as usize)
            .map_err(|_|"TP rank command/session inconsistency")?;
        if state.sessions.len()!=sessions.len()||state.sessions.iter().any(|(id,&position)|
            sessions.get(id).is_none_or(|s:&Session<'_, '_, '_, '_>|s.len()!=position)) {
            return Err("TP rank command/session inconsistency".into());
        }
        // Fatal faults abort both transports; a gathered numerical rejection
        // leaves this command unapplied and affects only its request(s).
        let executed:Result<()>=(||{
        match op{
            OPEN=>{let mut s=model.session(&ring);s.collect_logits=false;if sessions.insert(id,s).is_some(){return Err("duplicate session id".into());}},
            STOP=>return Ok(()),
            CLOSE=>{
                #[cfg(target_os="linux")]
                if let Some(ring)=sequence.as_mut(){ring.retain_regions(false);}
                let s=sessions.remove(&id).ok_or("unknown session close")?;let(kv,peak,pinned)=(s.cache_bytes(),s.device_peak_bytes(),s.pinned_peak_bytes());drop(s);eprintln!("tp_session_closed rank={} session={id} kv_bytes={kv} session_device_peak_bytes={peak} pinned_peak_bytes={pinned} device_bytes={}",ring.borrow().rank(),model.memory_bytes().0);
            },
            FORK=>{let session=sessions.get(&id).ok_or("unknown fork session")?.fork()?;if sessions.insert(arg,session).is_some(){return Err("duplicate fork id".into());}},
            DECODE_BATCH=>{
                let count=usize::try_from(count).map_err(|_|"decode batch count overflow")?;
                if count==0||count%3!=0||count/3>sessions.len()||arg>1{return Err("invalid remote decode batch".into());}
                let mut active=Vec::with_capacity(count/3);let mut tokens=Vec::with_capacity(count/3);
                for row in data.chunks_exact(12){
                    let id=u64::from_le_bytes(row[..8].try_into().unwrap());
                    active.push((id,sessions.remove(&id).ok_or("unknown or duplicate decode session")?));
                    tokens.push(u32::from_le_bytes(row[8..].try_into().unwrap()));
                }
                if multi.is_none(){multi=Some(Multi::new(model)?);}
                let mut local=active.iter_mut().map(|(_,s)|s).collect::<Vec<_>>();
                let result=multi.as_mut().unwrap().step(&mut local,&tokens,arg!=0);
                for(id,session)in active{sessions.insert(id,session);}
                result?;
            },
            PREFILL_BATCH=>{
                let count=usize::try_from(count).map_err(|_|"prefill batch count overflow")?;
                if count==0||count>8192||arg>1{return Err("invalid remote prefill batch".into());}
                let words=&tokens;
                let mut active=Vec::new();let mut tokens=Vec::new();let mut at=0;let mut rows=0;
                while at<words.len(){
                    if words.len()-at<3{return Err("truncated remote prefill batch".into());}
                    let id=u64::from(words[at])|(u64::from(words[at+1])<<32);
                    let n=words[at+2] as usize;at+=3;
                    if n==0||n>words.len()-at||n>2048-rows{return Err("invalid remote prefill rows".into());}
                    active.push((id,sessions.remove(&id).ok_or("unknown or duplicate prefill session")?));
                    tokens.push(&words[at..at+n]);at+=n;rows+=n;
                }
                if multi.is_none(){multi=Some(Multi::new(model)?);}
                let mut local=active.iter_mut().map(|(_,s)|s).collect::<Vec<_>>();
                let result=multi.as_mut().unwrap().prefill(&mut local,&tokens,arg!=0);
                for(id,session)in active{sessions.insert(id,session);}
                result?;
            },
            _=>{
                let s=sessions.get_mut(&id).ok_or("unknown session")?;
                match op{
                    STEP=>{s.step(u32::try_from(arg).map_err(|_|"invalid token")?)?;},
                    GREEDY=>{s.step_greedy(u32::try_from(arg).map_err(|_|"invalid token")?)?;},
                    KV_FORMAT=>s.configure_kv_bf16(arg!=0)?,
                    TRUNCATE=>s.truncate(usize::try_from(arg).map_err(|_|"invalid position")?)?,
                    SEQUENCE_SLICE=>{s.advance_sequence(std::time::Duration::from_micros(arg),sequence.as_mut())?;},
                    PREFILL|VERIFY|FAST|PREFILL_ADVANCE|PREFILL_END|SEQUENCE_ADVANCE|SEQUENCE_END|SEQUENCE_START=>{
                        let count=usize::try_from(count).map_err(|_|"batch count overflow")?;
                        if count==0||count>model.max_tokens()-s.len(){return Err("remote batch exceeds capacity".into());}
                        if op==SEQUENCE_START{s.start_sequence(&tokens,sequence.as_ref())?;}
                        else if op==SEQUENCE_ADVANCE||op==SEQUENCE_END{s.prefill_sequence(&tokens,usize::try_from(arg).map_err(|_|"prompt end overflow")?,op==SEQUENCE_END,sequence.as_mut())?;}
                        else if op==PREFILL{s.prefill(&tokens)?;}else if op==FAST{s.forward_batch_fast(&tokens)?;}else if op==VERIFY{s.verify_batch(&tokens)?;}else{s.prefill_chunk(&tokens,usize::try_from(arg).map_err(|_|"prompt end overflow")?,op==PREFILL_END)?;}
                    },
                    _=>return Err("unknown TP session command".into()),
                }
            }
        }
        Ok(())})();
        if let Err(error)=executed {
            if error==OUTPUT_ERROR {eprintln!("tp_serve worker rank={} session={id} op={op} request_failed: {error}",ring.borrow().rank());continue;}
            return Err(error);
        }
        state.apply(op,id,arg,&tokens);
        if op==STOP{return Ok(());}
        let (device,peak,pinned)=model.memory_bytes();
        if peak>reported_peak {
            eprintln!("tp_memory rank={} device_bytes={device} device_peak_bytes={peak} pinned_bytes={pinned}",ring.borrow().rank());
            reported_peak=peak;
        }
    }})();
    if result.is_err(){spark_gpu_probe::teardown::arm();ring.borrow_mut().abort();if let Some(ring)=sequence.as_mut(){ring.abort();}}
    #[cfg(target_os="linux")]
    if let Some(ring)=sequence.as_mut(){ring.retain_regions(false);}
    result
}

/// Typed hybrid commands share one immutable checkpoint/plan/rank incarnation.
/// Execution errors poison the group: this protocol advertises no state recovery.
pub struct HybridProtocol<'p> {
    identity: spark_engine::hybrid::cache::Identity<'p>,
    binding: spark_engine::hybrid::tp::Binding<'p>,
    state: CommandState, epoch: u64, poisoned: bool, domain: Vec<u64>,
}
impl<'p> HybridProtocol<'p> {
    pub fn new(checkpoint: usize, plan: &'p engine_model::plan::ModelPlan,
        split: &'p engine_model::split::SplitPlan, rank: u32, members: &[u64], incarnation: u64) -> Result<Self> {
        let metadata=format!("{plan:?}{split:?}");
        Ok(Self { identity: spark_engine::hybrid::cache::Identity::new(checkpoint,plan,
            spark_engine::hybrid::linear::ARITHMETIC_ID,split,rank)?.with_group(members,incarnation)?,
            binding: spark_engine::hybrid::tp::Binding::new(plan,split,rank as usize)?,
            state: CommandState::default(), epoch: 0, poisoned: false,
            domain: [incarnation,members.len() as u64,metadata.len() as u64].into_iter().chain(members.iter().copied()).chain(
                metadata.as_bytes().chunks(8).map(|s|{let mut b=[0;8];b[..s.len()].copy_from_slice(s);u64::from_le_bytes(b)})).collect() })
    }
    pub fn for_model(model:&'p spark_engine::hybrid::Model<'_, '_>)->Result<Self> {
        let group=model.rank_group()?;
        Self::new(model as *const _ as usize,model.model_plan(),group.split(),group.rank() as u32,group.members(),group.incarnation())
    }
    /// Single-session commands check the actual cache domain and ledger position.
    /// Rollback uses this same command agreement and only publishes after all ranks succeed.
    pub fn execute_session<T>(&mut self,model:&spark_engine::hybrid::Model<'_, '_>,session:&mut spark_engine::hybrid::Session<'_, '_,'_>,
        op:u64,id:u64,arg:u64,tokens:&[u32],work:impl FnOnce(&mut spark_engine::hybrid::Session<'_, '_,'_>)->Result<T>)->Result<T> {
        let live=session.cache_identity().is_ok_and(|key|key==self.identity) && model.cache_identity().is_ok_and(|key|key==self.identity)
            && if op==OPEN { session.len()==0 } else { self.state.sessions.get(&id)==Some(&session.len()) };
        let mut group=model.rank_group()?; let mut expected=self.state.clone();
        let result=self.execute(&mut group,op,id,arg,tokens,model.max_tokens(),model.config().vocab as usize,live,|| { let result=work(session)?; expected.apply(op,id,arg,tokens);
            if op!=CLOSE && expected.sessions.get(&id)!=Some(&session.len()) { return Err("hybrid command left an uncommitted position".into()); } Ok(result) });
        if self.poisoned { group.invalidate(); } result
    }
    pub fn command_words(&self,payload:usize)->Result<usize>{self.domain.len().checked_add(6).and_then(|n|n.checked_add(payload)).ok_or("hybrid command envelope overflow".into())}
    pub fn identity(&self) -> &spark_engine::hybrid::cache::Identity<'p> { &self.identity }
    /// `live` confirms the command's session/snapshot affinity and exact position.
    /// All ranks vote before any mutation, and commit the ledger only together.
    pub fn execute<E: engine_fabric::exchange::Exchange, T>(&mut self, exchange: &mut E,
        op: u64, id: u64, arg: u64, tokens: &[u32], context: usize, vocab: usize,
        live: bool, work: impl FnOnce() -> Result<T>) -> Result<T> {
        self.execute_deciding(exchange,op,id,arg,tokens,context,vocab,live,||work().map(|v|(v,true)))
    }
    pub(super) fn execute_deciding<E: engine_fabric::exchange::Exchange,T>(&mut self,exchange:&mut E,
        op:u64,id:u64,arg:u64,tokens:&[u32],context:usize,vocab:usize,live:bool,work:impl FnOnce()->Result<(T,bool)>)->Result<T> {
        if self.poisoned { return Err("hybrid rank group invalid".into()); }
        let epoch=self.epoch.checked_add(1).ok_or("hybrid command epoch overflow")?;
        let mut command=vec![epoch,op,id,arg,tokens.len() as u64];
        command.extend(tokens.iter().map(|&t|u64::from(t)));
        if self.epoch==0 { command.extend_from_slice(&self.domain); }
        let validation=self.state.validate(op,id,arg,tokens,context,vocab);
        let vote=self.binding.agree(exchange,&command,&[!live || validation.is_err()]);
        let rejected=match vote { Ok(v)=>v[0], Err(error)=>{self.poisoned=true;return Err(error);} };
        self.epoch=epoch;
        if rejected { return Err("hybrid command rejected by rank group".into()); }
        let result=work();
        command.push(u64::from(result.as_ref().is_ok_and(|(_,commit)|*commit)));
        let vote=self.binding.agree(exchange,&command,&[result.is_err()]);
        if !vote.as_ref().is_ok_and(|v|!v[0]) { self.poisoned=true; return Err("hybrid execution failed; rank group invalid".into()); }
        let (value,commit)=result?; if commit { self.state.apply(op,id,arg,tokens); } Ok(value)
    }
}

#[path="hybrid_service.rs"]
pub mod typed;
