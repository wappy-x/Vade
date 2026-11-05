//! Scalar full-stack service loop with real command and gather ordering.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::{bounded_decoder::{Compute,Decoder,Host},row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
#[path="support/attention_checkpoint.rs"] mod fixture;
struct Wire {topology:Topology,controller:bool,tx:SyncSender<Vec<u8>>,rx:Receiver<Vec<u8>>}
#[test]
fn shared_chat_parser_keeps_plain_greedy_streaming_and_explicit_sampling() {
    let limits=spark_tp::row_split::limits(512,4).unwrap();
    let request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":4,"stream":true,"mtp_depth":0}"#,true,&limits).unwrap();
    assert!(request.stream);assert_eq!(request.max_tokens,4);assert_eq!(request.mtp_depth,Some(0));
    assert_eq!(request.sampling.temperature,0.0);assert_eq!(request.sampling.top_k,0);assert_eq!(request.sampling.top_p,1.0);
    assert_eq!(request.sampling.repetition_penalty,1.0);assert_eq!(request.sampling.presence_penalty,0.0);assert_eq!(request.sampling.frequency_penalty,0.0);
    assert!(matches!(request.prompt,spark_serve::protocol::Prompt::Chat(_)));
    let request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"temperature":0.25,"seed":7}"#,true,&limits).unwrap();
    assert_eq!(request.sampling.temperature,0.25);assert_eq!(request.sampling.seed,7);
    assert!(spark_tp::row_split::limits(0,4).is_err());assert!(spark_tp::row_split::limits(8,0).is_err());
    let card=spark_tp::row_split::limits(32768,8192).unwrap();
    let request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8192}"#,true,&card).unwrap();
    assert_eq!(request.max_tokens,8192);assert!(!request.ignore_eos);assert_eq!(card.context_tokens,32768);
    assert_eq!(card.generation_timeout,std::time::Duration::from_secs(86400));
}
struct Prepared {
    tx:std::sync::mpsc::Sender<()>,rx:std::sync::mpsc::Receiver<()>,calls:std::sync::Arc<std::sync::atomic::AtomicUsize>,
    batches:std::sync::Arc<std::sync::atomic::AtomicUsize>,working:std::cell::Cell<bool>,
}
impl Compute for Prepared {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    fn work<T>(&self,run:impl FnOnce()->Result<T>)->Result<T> {
        assert!(!self.working.replace(true),"execution owners must not nest");
        let result=run();assert!(self.working.replace(false));result
    }
    fn prepare(&self,_:&engine_model::plan::admission::CheckpointAdmission)->Result<()> {
        self.tx.send(()).map_err(|e|e.to_string())?;
        self.rx.recv_timeout(std::time::Duration::from_secs(2)).map_err(|e|format!("session preparation is not paired: {e}"))?;
        self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);Ok(())
    }
    fn projection_batch(&self,checkpoint:&engine_model::plan::admission::CheckpointAdmission,input:&[(&str,&Vec<f32>)],round:bool)->Result<Option<Vec<Vec<f32>>>> {
        assert!(self.working.get(),"projection consumes buffers inside its execution owner");
        self.batches.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        input.iter().map(|(name,input)| {
            let upload=checkpoint.upload(name).map_err(|error|error.to_string())?;
            if upload.storage==engine_model::plan::storage::WeightStorage::Mxfp8 {
                let activation=spark_gpu_probe::dynamic_mxfp8::Activation::from_f32(input)?;
                let shape=spark_gpu_probe::mxfp8_projection::Shape::new(upload.shape[0] as usize,upload.shape[1] as usize,spark_gpu_probe::norm_boundary::WeightType::F32)?;
                return Ok(shape.reference(&upload.weight,&upload.scales,activation.decoded())?.1);
            }
            assert_eq!(upload.storage,engine_model::plan::storage::WeightStorage::Plain);
            let weight=match upload.dtype {
                Dtype::Bf16=>upload.weight.chunks_exact(2).map(|word|f32::from_bits(u32::from(u16::from_le_bytes(word.try_into().unwrap()))<<16)).collect::<Vec<_>>(),
                Dtype::F32=>upload.weight.chunks_exact(4).map(|word|f32::from_le_bytes(word.try_into().unwrap())).collect(),_=>return Err("batch fixture requires plain weights".into()),
            };
            let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:upload.shape[0] as u32,cols:upload.shape[1] as u32,epsilon:0.0,bound:0.0};
            let value=Host.operation(shape,[Some(input),None,None,None],&weight)?;
            if round {Host.round(&value)} else {Ok(value)}
        }).collect::<Result<Vec<_>>>().map(Some)
    }
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {Host.read(v)}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Host.range(v,start,len)}
    fn join(&self,v:&[&Vec<f32>])->Result<Vec<f32>> {Host.join(v)}
    fn zeros(&self,n:usize)->Result<Vec<f32>> {Host.zeros(n)}
    fn operation(&self,s:spark_gpu_probe::causal_ops::Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->Result<Vec<f32>> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,w:&[f32])->Result<Vec<f32>> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->Result<(Self::Pass,Vec<f32>)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&Vec<f32>)->Result<Vec<f32>> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&Vec<f32>)->Result<Vec<f32>> {Host.mx(p,x)}
    fn round(&self,x:&Vec<f32>)->Result<Vec<f32>> {Host.round(x)}
    fn mean(&self,x:&Vec<f32>,s:u32,h:u32)->Result<Vec<f32>> {Host.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->Result<Vec<f32>> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&Vec<f32>,m:usize,w:&[f32])->Result<Vec<f32>> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->Result<(Self::Pass,Vec<f32>)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&Vec<f32>,m:usize)->Result<Vec<f32>> {Host.rows_finish(p,x,m)}
}

impl Exchange for Wire {
    fn topology(&self)->Topology {self.topology}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {
        self.tx.send(local.iter().flat_map(|v|v.to_le_bytes()).collect()).map_err(|e|e.to_string())?;
        let other=self.rx.recv().map_err(|e|e.to_string())?;
        let peer=other.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        let mut out=vec![0.0;rows];let r=spark_tp::rows(rows,self.topology.rank as usize,self.topology.parts as usize);out[r].copy_from_slice(local);
        let r=spark_tp::rows(rows,1-self.topology.rank as usize,self.topology.parts as usize);out[r].copy_from_slice(&peer);Ok(out)
    }
    fn command(&mut self,v:&mut [u8])->Result<()> {
        if self.controller {self.tx.send(v.to_vec()).map_err(|e|e.to_string())} else {let p=self.rx.recv().map_err(|e|e.to_string())?;v.copy_from_slice(&p);Ok(())}
    }
    fn complete(&mut self)->Result<()> {self.tx.send(Vec::new()).map_err(|e|e.to_string())?;assert!(self.rx.recv().map_err(|e|e.to_string())?.is_empty());Ok(())}
}
#[test]
fn scalar_service_generates_four_exact_tokens_across_row_parts_and_new_sessions() {
    let _constructor:fn(Dtype)->fixture::Checkpoint=fixture::Checkpoint::new;
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let _=&c.operands;
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let (ap_tx,bp_rx)=std::sync::mpsc::channel();let (bp_tx,ap_rx)=std::sync::mpsc::channel();
    let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let batches=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker_calls=calls.clone();
        let worker_batches=batches.clone();
        let worker=scope.spawn(move || {
            let prepared=Prepared {tx:bp_tx,rx:bp_rx,calls:worker_calls,batches:worker_batches,working:std::cell::Cell::new(false)};
            let split=Split::new(&prepared,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,8,vec![0]).unwrap();
            assert_eq!(factory.negotiate_available_bytes((1usize<<40)+123).unwrap(),(1usize<<40)+122);factory.worker().unwrap();
        });
        let prepared=Prepared {tx:ap_tx,rx:ap_rx,calls:calls.clone(),batches:batches.clone(),working:std::cell::Cell::new(false)};
        let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let service=Factory::new(checkpoint,split,8,vec![0]).unwrap();let reference=Decoder::new(checkpoint,Host,8).unwrap();
        assert_eq!(service.negotiate_available_bytes((1usize<<40)+122).unwrap(),(1usize<<40)+122);
        assert!(service.request_reservation_bytes(8).unwrap()>=service.request_reservation_bytes(1).unwrap());assert!(service.request_reservation_bytes(9).is_none());
        for repeat in 0..3 {
            let mut session=service.session().unwrap();let extra=service.session().unwrap();drop(extra);assert!(!session.can_fork());
            let mut state=reference.zero_state().unwrap();let mut token=1;
            if repeat%2==0 {
                let prompt=[1,3,2];let mut expected=Vec::new();
                for id in prompt {let (logits,id,next)=reference.step(id,&state,|_,_|Ok(())).unwrap();expected=logits;state=next;token=id;}
                let logits=session.prefill(&prompt).unwrap();
                assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
            }
            for _ in 0..4 {
                let (expected,id,next)=reference.step(token,&state,|_,_|Ok(())).unwrap();let logits=session.step(token).unwrap();
                assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                let best=(0..logits.len()).reduce(|a,b|if logits[b]>logits[a] {b} else {a}).unwrap();assert_eq!(best as u32,id);
                state=next;token=id;
            }
            assert!(session.step(u32::MAX).is_err());
        }
        service.shutdown().unwrap();worker.join().unwrap();
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),14);
        assert!(batches.load(std::sync::atomic::Ordering::SeqCst)>0);
    });
}
#[test]
fn interleaved_two_four_eight_sessions_match_alone_full_logits_and_ids() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();Factory::new(checkpoint,split,80,vec![0]).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();
        // Generated rows grow ordinary state, but retain no new prompt-only cache.
        let reserve=|prompt,tokens|engine.request_reservation_bytes_for_prompt(prompt,tokens,true,None,false).unwrap();
        let d=checkpoint.description();let plan=|tokens|engine_model::plan::state::StatePlan::new(d,Topology {rank:0,parts:1},tokens,isize::MAX as usize).unwrap();
        assert_eq!(reserve(2,80)-reserve(2,66),2*(plan(80).state_layout().bytes()-plan(66).state_layout().bytes()));
        assert!(reserve(20,80)>reserve(2,80));
        assert!(engine.request_reservation_bytes_for_prompt(81,80,true,None,false).is_none());
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let prompts=(0..8).map(|n|vec![1+n as u32,(n as u32*3+2)%16]).collect::<Vec<_>>();
        let mut alone=Vec::new();
        for prompt in &prompts {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();for at in 0..64 {let id=best(&logits);rows.push((id,bits(&logits)));if at<63 {logits=session.step(id).unwrap();}}alone.push(rows);}
        for count in [2,4,8] {
            let mut sessions=(0..count).map(|_|engine.session().unwrap()).collect::<Vec<_>>();
            if count==8 {assert!(engine.session().is_err());}
            let mut logits=sessions.iter_mut().zip(&prompts).map(|(s,p)|s.prefill(p).unwrap()).collect::<Vec<_>>();
            for at in 0..64 {let order=if at%2==0 {(0..count).collect::<Vec<_>>()} else {(0..count).rev().collect()};
                for i in order {let id=best(&logits[i]);assert_eq!((id,bits(&logits[i])),alone[i][at],"cohort={count} chat={i} token={at}");if at<63 {logits[i]=sessions[i].step(id).unwrap();}}
            }
            while let Some(session)=sessions.pop() {drop(session);}
        }
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

struct FaultWire {fault:std::rc::Rc<std::cell::Cell<bool>>,aborted:std::rc::Rc<std::cell::Cell<bool>>}
impl Exchange for FaultWire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {if self.fault.get() {Err("injected peer failure".into())} else {Ok(())}}
    fn complete(&mut self)->Result<()> {Ok(())}
    fn abort(&mut self) {self.aborted.set(true);}
}
#[test]
fn permanent_exchange_failure_is_sticky_for_live_and_future_chats() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let fault=std::rc::Rc::new(std::cell::Cell::new(false));let aborted=std::rc::Rc::new(std::cell::Cell::new(false));
    let split=Split::new(Host,FaultWire {fault:fault.clone(),aborted:aborted.clone()}).unwrap();
    let engine=Factory::new(&c.admission,split,8,vec![0]).unwrap();
    let mut a=engine.session().unwrap();let mut b=engine.session().unwrap();a.step(1).unwrap();b.step(1).unwrap();
    fault.set(true);let first=a.step(1).unwrap_err();assert_eq!(first,"injected peer failure");assert!(aborted.get());
    fault.set(false);assert_eq!(b.step(1).unwrap_err(),first);assert_eq!(engine.fatal_error(),Some(first.clone()));
    assert_eq!(engine.session().err(),Some(first.clone()));assert_eq!(engine.shutdown().unwrap_err(),first);
    assert!(engine.reserve_full_output());
}
#[test]
fn row_split_deadline_retires_the_session_without_poisoning_the_factory() {
    use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::atomic::AtomicBool,time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let worker=std::thread::spawn(move|| {
        let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
        let split=Split::new(Host,FaultWire {fault:std::rc::Rc::new(std::cell::Cell::new(false)),aborted:std::rc::Rc::new(std::cell::Cell::new(false))}).unwrap();
        let engine=Factory::new(&c.admission,split,8,vec![0]).unwrap();let mut limits=spark_tp::row_split::limits(8,4).unwrap();limits.generation_timeout=Duration::from_millis(1);
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:8,logits_per_entry:16,idle_ttl:Duration::from_secs(1)};
        let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<24,session_budget_bytes:1<<24,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"deadline",1).unwrap();
        assert_eq!(engine.fatal_error(),None);let next=engine.session().unwrap();drop(next);engine.shutdown().unwrap();
    });
    let body=r#"{"prompt_token_ids":[1],"stream":true,"max_tokens":4,"ignore_eos":true,"cache":false}"#;
    let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();
    assert!(reply.contains("generation_deadline"),"{reply}");worker.join().unwrap();
}
