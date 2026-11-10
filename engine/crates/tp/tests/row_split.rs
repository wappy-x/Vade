//! Scalar full-stack service loop with real command and gather ordering.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::{bounded_decoder::{Compute,Decoder,Host},row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
#[path="support/attention_checkpoint.rs"] mod fixture;
struct Wire {topology:Topology,controller:bool,tx:SyncSender<Vec<u8>>,rx:Receiver<Vec<u8>>}
#[test]
fn compact_head_survives_reference_and_split_wrappers() {
    let checkpoint=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(6),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();let input=vec![-0.0,17.0];
    for result in [prepared.greedy_projection(&checkpoint.admission,"lm_head.weight",&input),<&Prepared as Compute>::greedy_projection(&&prepared,&checkpoint.admission,"lm_head.weight",&input),split.greedy_projection(&checkpoint.admission,"lm_head.weight",&input),<&Split<&Prepared,Wire> as Compute>::greedy_projection(&&split,&checkpoint.admission,"lm_head.weight",&input)] {assert_eq!(result.unwrap(),17);}
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),4);prepared.attention_probe.set(7);
    assert_eq!(split.greedy_projection(&checkpoint.admission,"lm_head.weight",&input).unwrap_err(),"compact head probe failure");
}
#[test]
fn vector_allocation_counts_survive_reference_and_split_wrappers() {
    let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(4),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    for counts in [prepared.vector_allocations(),<&Prepared as Compute>::vector_allocations(&&prepared),split.vector_allocations(),<&Split<&Prepared,Wire> as Compute>::vector_allocations(&&split)] {
        assert_eq!(counts,Some((33,156_237_824)));
    }
    for counts in [prepared.vector_pool_counts(),<&Prepared as Compute>::vector_pool_counts(&&prepared),split.vector_pool_counts(),<&Split<&Prepared,Wire> as Compute>::vector_pool_counts(&&split)] {
        assert_eq!(counts,Some((33,156_237_824,21)));
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),8);
}
#[test]
fn cache_append_survives_reference_and_split_wrappers() {
    let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(3),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    let (old,new)=(vec![-0.0,17.0],vec![f32::from_bits(1)]);
    for result in [prepared.cache_append(&old,&new,2,1,128),<&Prepared as Compute>::cache_append(&&prepared,&old,&new,2,1,128),split.cache_append(&old,&new,2,1,128),<&Split<&Prepared,Wire> as Compute>::cache_append(&&split,&old,&new,2,1,128)] {
        assert_eq!(result.unwrap().iter().map(|x|x.to_bits()).collect::<Vec<_>>(),vec![0x80000000,17.0f32.to_bits(),1]);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),4);
}
#[test]
fn projection_stage_survives_reference_and_split_wrappers_without_fallback() {
    use spark_tp::projection_stage::Job;
    let checkpoint=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(1),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    let input=vec![0.0,-0.0,f32::from_bits(1),17.0];let jobs=[Job::Project {input:vec![("weight",&input)],round:true}];
    for result in [prepared.projection_stage(&checkpoint.admission,&jobs),<&Prepared as Compute>::projection_stage(&&prepared,&checkpoint.admission,&jobs),split.projection_stage(&checkpoint.admission,&jobs),<&Split<&Prepared,Wire> as Compute>::projection_stage(&&split,&checkpoint.admission,&jobs)] {
        assert_eq!(result.unwrap().unwrap()[0].iter().map(|value|value.to_bits()).collect::<Vec<_>>(),input.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),4);prepared.attention_probe.set(0);assert!(split.projection_stage(&checkpoint.admission,&jobs).unwrap().is_none());prepared.attention_probe.set(2);
    assert_eq!(split.projection_stage(&checkpoint.admission,&jobs).unwrap_err(),"projection stage probe failure");
    assert_eq!(<&Split<&Prepared,Wire> as Compute>::projection_stage(&&split,&checkpoint.admission,&jobs).unwrap_err(),"projection stage probe failure");
    assert!(Host.projection_stage(&checkpoint.admission,&jobs).unwrap().is_none());assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),6);
}
#[test]
fn device_route_survives_reference_and_split_wrappers_without_fallback() {
    let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(1),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    let ids=vec![7.0,0.0];let input=vec![-0.0,17.0];
    for result in [prepared.routed_experts(&[],&ids,&input),<&Prepared as Compute>::routed_experts(&&prepared,&[],&ids,&input),
        split.routed_experts(&[],&ids,&input),<&Split<&Prepared,Wire> as Compute>::routed_experts(&&split,&[],&ids,&input)] {
        let result=result.unwrap().unwrap();assert_eq!(result[0].iter().map(|value|value.to_bits()).collect::<Vec<_>>(),input.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),4);
    prepared.attention_probe.set(0);assert!(split.routed_experts(&[],&ids,&input).unwrap().is_none());
    prepared.attention_probe.set(2);assert_eq!(split.routed_experts(&[],&ids,&input).unwrap_err(),"device route probe failure");
    assert_eq!(<&Split<&Prepared,Wire> as Compute>::routed_experts(&&split,&[],&ids,&input).unwrap_err(),"device route probe failure");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),6);
    assert!(Host.routed_experts(&[],&ids,&input).unwrap().is_none());
}
#[test]
fn paired_reads_survive_reference_and_split_wrappers_once() {
    let (sender,receiver)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx:sender,rx:receiver,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(1),known:Default::default()};
    let (wire_tx,wire_rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx:wire_tx,rx:wire_rx}).unwrap();
    let first=vec![0.0,-0.0,f32::from_bits(1),f32::MIN_POSITIVE,1.0];let second=vec![17.0];
    let results=[prepared.read_pair(&first,&second),<&Prepared as Compute>::read_pair(&&prepared,&first,&second),
        split.read_pair(&first,&second),<&Split<&Prepared,Wire> as Compute>::read_pair(&&split,&first,&second)];
    for result in results {let (actual_first,actual_second)=result.unwrap();
        assert_eq!(actual_first.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),first.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
        assert_eq!(actual_second,second);}
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),4);
    prepared.attention_probe.set(2);
    assert_eq!(split.read_pair(&first,&second).unwrap_err(),"paired read probe failure");
    assert_eq!(<&Split<&Prepared,Wire> as Compute>::read_pair(&&split,&first,&second).unwrap_err(),"paired read probe failure");
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),6);
}
#[test]
fn known_selection_survives_reference_and_split_wrappers_once() {
    let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls,batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(0),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    let (ids,count)=(vec![0.0,1.0,-1.0],vec![2.0]);let row=[0.0,1.0,-1.0,2.0];
    prepared.select_known(&ids,&count,&row).unwrap();assert_eq!(prepared.known.load(std::sync::atomic::Ordering::SeqCst),1);
    <&Prepared as Compute>::select_known(&&prepared,&ids,&count,&row).unwrap();assert_eq!(prepared.known.load(std::sync::atomic::Ordering::SeqCst),2);
    split.select_known(&ids,&count,&row).unwrap();assert_eq!(prepared.known.load(std::sync::atomic::Ordering::SeqCst),3);
    <&Split<&Prepared,Wire> as Compute>::select_known(&&split,&ids,&count,&row).unwrap();assert_eq!(prepared.known.load(std::sync::atomic::Ordering::SeqCst),4);
    prepared.attention_probe.set(2);
    assert_eq!(<&Prepared as Compute>::select_known(&&prepared,&ids,&count,&row).unwrap_err(),"known selection probe failure");
    assert_eq!(split.select_known(&ids,&count,&row).unwrap_err(),"known selection probe failure");
    assert_eq!(<&Split<&Prepared,Wire> as Compute>::select_known(&&split,&ids,&count,&row).unwrap_err(),"known selection probe failure");
    assert_eq!(prepared.known.load(std::sync::atomic::Ordering::SeqCst),4);
}
/// The service's own decoder, on two ranks: Decoder over &Split over the backend, as flash-serve builds it.
/// Every plain step up to the boundary reaches the backend, with the row the selection really computed.
#[test]
fn served_two_rank_steps_take_the_known_selection_up_to_the_boundary_only() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let (ap_tx,bp_rx)=std::sync::mpsc::channel();let (bp_tx,ap_rx)=std::sync::mpsc::channel();
    let (rank0,rank1)=(std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let worker_known=rank1.clone();
        let worker=scope.spawn(move || {
            let prepared=Prepared {tx:bp_tx,rx:bp_rx,calls:Default::default(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(0),known:worker_known};
            let split=Split::new(&prepared,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,8,vec![0]).unwrap();
            factory.negotiate_available_bytes(1usize<<40).unwrap();factory.worker().unwrap();
        });
        let prepared=Prepared {tx:ap_tx,rx:ap_rx,calls:Default::default(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(0),known:rank0.clone()};
        let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let service=Factory::new(checkpoint,split,8,vec![0]).unwrap();let reference=Decoder::new(checkpoint,Host,8).unwrap();
        service.negotiate_available_bytes(1usize<<40).unwrap();
        let mut session=service.session().unwrap();let mut state=reference.zero_state().unwrap();let mut token=1;
        // Budget 4, pool 2, tail on: contexts 1 to 5 keep every pool, 6 to 8 select for real.
        for context in 1..=8usize {
            let (expected,id,next)=reference.step(token,&state,|_,_|Ok(())).unwrap();let logits=session.step(token).unwrap();
            assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"context {context}");
            assert_eq!(rank0.load(std::sync::atomic::Ordering::SeqCst),context.min(5),"known selections at context {context}");
            state=next;token=id;
        }
        drop(session);service.shutdown().unwrap();worker.join().unwrap();
        assert_eq!(rank1.load(std::sync::atomic::Ordering::SeqCst),5);
    });
}
#[test]
fn indirect_attention_survives_reference_and_split_wrappers() {
    let (tx,rx)=std::sync::mpsc::channel();let calls=std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let prepared=Prepared {tx,rx,calls:calls.clone(),batches:Default::default(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(1),known:Default::default()};
    let (tx,rx)=sync_channel(1);let split=Split::new(&prepared,Wire {topology:Topology {rank:0,parts:1},controller:true,tx,rx}).unwrap();
    let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::ScoresIndirect,heads:64,dim:256,taps:2,rows:2,cols:256,epsilon:0.0,bound:0.0};
    let (first,second,query)=(vec![1.0],vec![2.0],vec![3.0]);let parts=[&first,&second];
    for reference in [false,true] {
        let result=if reference {<&Prepared as Compute>::indirect_attention(&&prepared,shape,&parts,&query)} else {split.indirect_attention(shape,&parts,&query)};
        assert_eq!(result.unwrap().unwrap().into_iter().map(f32::to_bits).collect::<Vec<_>>(),vec![(-0.0f32).to_bits(),17.0f32.to_bits()]);
    }
    assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst),2);
    prepared.attention_probe.set(0);assert!(split.indirect_attention(shape,&parts,&query).unwrap().is_none());
    prepared.attention_probe.set(2);assert_eq!(split.indirect_attention(shape,&parts,&query).unwrap_err(),"indirect probe failure");
    assert_eq!(<&Prepared as Compute>::indirect_attention(&&prepared,shape,&parts,&query).unwrap_err(),"indirect probe failure");
}
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
    batches:std::sync::Arc<std::sync::atomic::AtomicUsize>,working:std::cell::Cell<bool>,attention_probe:std::cell::Cell<u32>,
    known:std::sync::Arc<std::sync::atomic::AtomicUsize>,
}
impl Compute for Prepared {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    fn projects_complete_rows(&self)->bool {matches!(self.attention_probe.get(),6|7)}
    fn greedy_projection(&self,c:&engine_model::plan::admission::CheckpointAdmission,name:&str,input:&Vec<f32>)->Result<u32> {
        if !self.projects_complete_rows() {return Host.greedy_projection(c,name,input);}
        assert_eq!(name,"lm_head.weight");assert_eq!(input,&vec![-0.0,17.0]);self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        if self.attention_probe.get()==7 {Err("compact head probe failure".into())} else {Ok(17)}
    }
    fn projection_stage(&self,_:&engine_model::plan::admission::CheckpointAdmission,jobs:&[spark_tp::projection_stage::Job<'_,Vec<f32>>])->Result<Option<Vec<Vec<f32>>>> {
        if self.attention_probe.get()==0 {return Ok(None);}
        self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        if self.attention_probe.get()==2 {return Err("projection stage probe failure".into());}
        let [spark_tp::projection_stage::Job::Project {input,round:true}]=jobs else {panic!()};Ok(Some(vec![input[0].1.clone()]))
    }
    fn routed_experts(&self,steps:&[engine_model::plan::layer::Step],ids:&Vec<f32>,input:&Vec<f32>)->Result<Option<Vec<Vec<f32>>>> {
        if self.attention_probe.get()==0 {return Ok(None);}
        assert!(steps.is_empty());assert_eq!(ids,&vec![7.0,0.0]);self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        if self.attention_probe.get()==2 {return Err("device route probe failure".into());}Ok(Some(vec![input.clone()]))
    }
    fn indirect_attention(&self,shape:spark_gpu_probe::causal_ops::Shape,parts:&[&Vec<f32>],query:&Vec<f32>)->Result<Option<Vec<f32>>> {
        if self.attention_probe.get()==0 {return Ok(None);}
        assert_eq!(shape.op,spark_gpu_probe::causal_ops::Op::ScoresIndirect);assert_eq!(shape.rows,2);
        assert_eq!(parts.len(),2);assert_eq!(parts[0],&vec![1.0]);assert_eq!(parts[1],&vec![2.0]);assert_eq!(query,&vec![3.0]);
        self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
        if self.attention_probe.get()==2 {return Err("indirect probe failure".into());}Ok(Some(vec![-0.0,17.0]))
    }
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
                Dtype::Bf16=>upload.weight.as_chunks::<2>().0.iter().map(|word|f32::from_bits(u32::from(u16::from_le_bytes(*word))<<16)).collect::<Vec<_>>(),
                Dtype::F32=>upload.weight.as_chunks::<4>().0.iter().map(|word|f32::from_le_bytes(*word)).collect(),_=>return Err("batch fixture requires plain weights".into()),
            };
            let shape=spark_gpu_probe::causal_ops::Shape {op:spark_gpu_probe::causal_ops::Op::Project,heads:0,dim:0,taps:0,rows:upload.shape[0] as u32,cols:upload.shape[1] as u32,epsilon:0.0,bound:0.0};
            let value=Host.operation(shape,[Some(input),None,None,None],&weight)?;
            if round {Host.round(&value)} else {Ok(value)}
        }).collect::<Result<Vec<_>>>().map(Some)
    }
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {Host.upload(v)}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {Host.read(v)}
    /// The row the host would assume must be the bits the selection really computed.
    fn select_known(&self,ids:&Vec<f32>,count:&Vec<f32>,row:&[f32])->Result<()> {
        if self.attention_probe.get()==2 {return Err("known selection probe failure".into());}
        let real=ids.iter().chain(count).map(|v|v.to_bits()).collect::<Vec<_>>();
        assert_eq!(real,row.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"assumed row differs from the computed selection");
        self.known.fetch_add(1,std::sync::atomic::Ordering::SeqCst);Ok(())
    }
    fn prompt_end(&self)->Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn read_pair(&self,first:&Vec<f32>,second:&Vec<f32>)->Result<(Vec<f32>,Vec<f32>)> {
        if self.attention_probe.get()!=0 {self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);
            if self.attention_probe.get()==2 {return Err("paired read probe failure".into());}}
        Ok((Host.read(first)?,Host.read(second)?))
    }
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Host.range(v,start,len)}
    fn join(&self,v:&[&Vec<f32>])->Result<Vec<f32>> {Host.join(v)}
    fn vector_allocations(&self)->Option<(usize,usize)> {if self.attention_probe.get()==4 {self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);}Some((33,156_237_824))}
    fn vector_pool_counts(&self)->Option<(usize,usize,usize)> {if self.attention_probe.get()==4 {self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);}Some((33,156_237_824,21))}
    fn cache_append(&self,old:&Vec<f32>,new:&Vec<f32>,rows:usize,width:usize,room:usize)->Result<Vec<f32>> {
        if self.attention_probe.get()==3 {assert_eq!((rows,width,room),(2,1,128));self.calls.fetch_add(1,std::sync::atomic::Ordering::SeqCst);}
        Host.cache_append(old,new,rows,width,room)
    }
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
        let peer=other.as_chunks::<4>().0.iter().map(|v|f32::from_le_bytes(*v)).collect::<Vec<_>>();
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
            let prepared=Prepared {tx:bp_tx,rx:bp_rx,calls:worker_calls,batches:worker_batches,working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(0),known:Default::default()};
            let split=Split::new(&prepared,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,8,vec![0]).unwrap();
            assert_eq!(factory.negotiate_available_bytes((1usize<<40)+123).unwrap(),(1usize<<40)+122);factory.worker().unwrap();
        });
        let prepared=Prepared {tx:ap_tx,rx:ap_rx,calls:calls.clone(),batches:batches.clone(),working:std::cell::Cell::new(false),attention_probe:std::cell::Cell::new(0),known:Default::default()};
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
            for step in 0..4 {
                let (expected,id,next)=reference.step(token,&state,|_,_|Ok(())).unwrap();
                if step%2==0 {let tokens=[token];let mut rows=[spark_serve::BatchRows {session:&mut session,tokens:&tokens,output:spark_serve::BatchOutput::All,greedy:true,top_k:0}];
                    let output=service.step_batch(&mut rows).unwrap_or_else(|_|panic!("compact batch failed"));
                    assert!(matches!(&output[..],[spark_serve::Predictions::Greedy(ids)] if *ids==vec![id]));
                } else {let logits=session.step(token).unwrap();
                    assert_eq!(logits.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
                    let best=(0..logits.len()).reduce(|a,b|if logits[b]>logits[a] {b} else {a}).unwrap();assert_eq!(best as u32,id);
                }
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
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
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
/// Returns a non-finite word from one gather when armed, as a bad exchange would.
struct BadWordWire {armed:std::rc::Rc<std::cell::Cell<bool>>}
impl Exchange for BadWordWire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);let mut out=local.to_vec();if self.armed.get() {out[0]=f32::INFINITY;}Ok(out)}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
    fn abort(&mut self) {}
}
#[test]
fn non_finite_exchanged_word_stops_the_request_before_any_token() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let armed=std::rc::Rc::new(std::cell::Cell::new(false));
    let split=Split::new(Host,BadWordWire {armed:armed.clone()}).unwrap();
    let engine=Factory::new(&c.admission,split,8,vec![0]).unwrap();
    let mut chat=engine.session().unwrap();
    armed.set(true);
    let error=chat.prefill(&[1,2,3]).expect_err("a non-finite exchanged word must stop the prompt, with no logits");
    assert_eq!(error,"row gather returned invalid output");
    // The failure is sticky: the server reports it, and no later chat can start.
    armed.set(false);
    assert_eq!(engine.fatal_error(),Some(error.clone()));assert_eq!(engine.session().err(),Some(error.clone()));
    assert_eq!(chat.step(1).unwrap_err(),error);
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
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn strict_drafting_keeps_the_plain_ids_and_logits_on_two_ranks() {
    use spark_engine::draft::{Config,Speculator};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();Factory::new(checkpoint,split,80,vec![0]).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();let vocab=engine.vocab_size() as u32;let count=48;
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let request=|draft:&str|spark_serve::protocol::parse(format!(r#"{{"messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":48,"draft":"{draft}"}}"#).as_bytes(),true,&limits).unwrap();
        let prompt=vec![3u32,5,1];
        let mut session=engine.session().unwrap();let mut logits=session.prefill(&prompt).unwrap();let mut plain=Vec::new();
        for at in 0..count {let id=best(&logits);plain.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}drop(session);
        // Only the explicit strict option opens the target. Other draft modes are refused.
        let mut session=engine.session().unwrap();assert!(session.configure_request(&request("fast")).is_err());assert!(session.configure_request(&request("sampled")).is_err());
        session.configure_request(&request("plain")).unwrap();assert!(session.draft_target().is_none());
        session.configure_request(&request("auto")).unwrap();assert!(session.draft_target().is_none());drop(session);
        // Passes with right and wrong proposals: each kept row has the bits of the plain step.
        let mut session=engine.session().unwrap();session.configure_request(&request("strict")).unwrap();
        let first=session.prefill(&prompt).unwrap();assert_eq!(bits(&first),plain[0].1);
        let target=session.draft_target().unwrap();assert_eq!(target.checkpoint(),prompt.len());
        let (mut at,mut pass,mut partial)=(1,0,0);
        while at<count {let rows=(1+pass%5).min(count-at);let kept=1+(pass*7)%rows;partial+=usize::from(kept<rows);
            let tokens=(0..rows).map(|i|if i<kept {plain[at+i-1].0} else {(plain[at+i-1].0+1)%vocab}).collect::<Vec<_>>();
            let entry=target.checkpoint();let logits=target.verify_batch(&tokens).unwrap();assert_eq!(target.checkpoint(),entry+rows);
            target.truncate(entry+kept).unwrap();assert_eq!(target.checkpoint(),entry+kept);assert!(target.truncate(entry+kept+1).is_err());
            for row in &logits[..kept] {assert_eq!(bits(row),plain[at].1,"row {at}");at+=1;}pass+=1;}
        assert!(partial>5,"the schedule must drop rows");
        // The engine was failed by the refused truncation above only if it reached the wire. It did not.
        assert_eq!(engine.fatal_error(),None);drop(session);
        // The server's speculator on the same target chooses the plain ids.
        let mut session=engine.session().unwrap();session.configure_request(&request("strict")).unwrap();
        let first=session.prefill(&prompt).unwrap();
        let mut spec=Speculator::from_selected(&prompt,prompt.len(),best(&first),vocab as usize,Config::strict(count-1),request("strict").sampling).unwrap();
        let target=session.draft_target().unwrap();while !spec.is_done() {spec.step(target,16,|_|false).unwrap();}
        assert_eq!(spec.tokens(),plain[..count-1].iter().map(|p|p.0).collect::<Vec<_>>());
        eprintln!("speculator cycles={} rows={} proposed={} accepted={} rollbacks={}",spec.stats.cycles,spec.stats.rows,spec.stats.proposed,spec.stats.accepted,spec.stats.rollbacks);
        drop(session);
        // A plain chat after the drafting chats is unchanged.
        let mut session=engine.session().unwrap();let mut logits=session.prefill(&prompt).unwrap();
        for at in 0..8 {assert_eq!((best(&logits),bits(&logits)),plain[at]);logits=Session::step(&mut session,plain[at].0).unwrap();}drop(session);
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}
#[test]
fn served_strict_request_streams_the_plain_ids() {
    use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::atomic::AtomicBool,time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let worker=std::thread::spawn(move|| {
        let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
        let split=Split::new(Host,FaultWire {fault:std::rc::Rc::new(std::cell::Cell::new(false)),aborted:std::rc::Rc::new(std::cell::Cell::new(false))}).unwrap();
        let engine=Factory::new(&c.admission,split,80,vec![0]).unwrap();let limits=spark_tp::row_split::limits(80,48).unwrap();
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:8,logits_per_entry:16,idle_ttl:Duration::from_secs(1)};
        let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<24,session_budget_bytes:1<<24,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"strict",2).unwrap();
        assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();
    });
    let ask=|draft:&str| {let body=format!(r#"{{"prompt_token_ids":[3,5,1],"stream":true,"max_tokens":40,"ignore_eos":true,"cache":false,"draft":"{draft}"}}"#);
        let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
    let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect::<Vec<_>>();
    let plain=ask("plain");let strict=ask("strict");worker.join().unwrap();
    assert_eq!(ids(&plain).len(),40,"{plain}");assert_eq!(ids(&strict),ids(&plain),"{strict}");
    assert!(strict.contains("\"draft\":\"strict\""),"{strict}");assert!(plain.contains("\"draft\":\"plain\""),"{plain}");
    eprintln!("strict tail: {}",&strict[strict.len().saturating_sub(700)..]);
}
#[test]
fn model_proposals_keep_the_plain_ids_on_two_ranks() {
    use spark_engine::draft::{Config,Speculator};
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();let before=engine.request_reservation_bytes(80).unwrap();
        let engine=engine.with_predictor(3).unwrap();assert_eq!(engine.predictor_depth(),3);
        assert!(engine.request_reservation_bytes(80).unwrap()>before,"the predictor's state and waiting rows are charged");
        let vocab=engine.vocab_size() as u32;let count=40;
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let mut limits=spark_tp::row_split::limits(80,64).unwrap();limits.next_mtp_depth=Some(3);
        let request=|depth:usize|spark_serve::protocol::parse(format!(r#"{{"messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":40,"mtp_depth":{depth}}}"#).as_bytes(),true,&limits).unwrap();
        let prompt=vec![3u32,5,1,7,2,6];
        let mut session=engine.session().unwrap();let mut logits=session.prefill(&prompt).unwrap();let mut plain=Vec::new();
        for at in 0..count {let id=best(&logits);plain.push(id);if at+1<count {logits=Session::step(&mut session,id).unwrap();}}drop(session);
        // A depth above the loaded predictor is refused; depths 1 to 3 give the plain ids.
        let mut session=engine.session().unwrap();
        assert!(session.configure_request(&spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":40,"mtp_depth":3,"draft":"fast"}"#,true,&limits).unwrap()).is_err());drop(session);
        for depth in 1..=3 {
            let mut session=engine.session().unwrap();session.configure_request(&request(depth)).unwrap();
            let first=session.prefill(&prompt).unwrap();
            let mut spec=Speculator::from_selected(&prompt,prompt.len(),best(&first),vocab as usize,Config::strict(count-1),request(depth).sampling).unwrap();
            let target=session.draft_target().unwrap();assert_eq!(target.model_draft_width(),depth);
            while !spec.is_done() {spec.step(target,16,|_|false).unwrap();}
            assert_eq!(spec.tokens(),&plain[..count-1],"depth {depth}");
            eprintln!("depth {depth}: cycles={} rows={} model proposals={} accepted={}",spec.stats.cycles,spec.stats.rows,spec.stats.model_proposals,spec.stats.accepted);
            drop(session);assert_eq!(engine.fatal_error(),None);
            // By hand, so that proposals are made whatever the chooser thinks of host timings: every
            // kept row has the plain id, and both ranks stay in step through proposals and verdicts.
            let mut session=engine.session().unwrap();session.configure_request(&request(depth)).unwrap();
            let first=session.prefill(&prompt).unwrap();assert_eq!(best(&first),plain[0]);
            let target=session.draft_target().unwrap();let (mut at,mut proposed,mut kept_proposals)=(1,0,0);
            while at<count {target.limit_model_depth(depth);
                let mut rows=vec![plain[at-1]];rows.extend(target.propose_model(&prompt,plain[at-1],depth).unwrap().unwrap().tokens);
                rows.truncate(count-at);proposed+=rows.len()-1;
                let entry=target.checkpoint();let logits=target.verify_batch(&rows).unwrap();
                let kept=1+rows[1..].iter().zip(&plain[at..]).take_while(|(a,b)|a==b).count();kept_proposals+=kept-1;
                target.truncate(entry+kept).unwrap();for row in &logits[..kept] {assert_eq!(best(row),plain[at],"depth {depth} row {at}");at+=1;}}
            eprintln!("depth {depth} by hand: proposed={proposed} kept={kept_proposals}");assert!(proposed>0,"the predictor made proposals at depth {depth}");
            drop(session);assert_eq!(engine.fatal_error(),None);
        }
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}
#[test]
fn served_model_drafting_keeps_the_plain_ids_at_caps_ends_and_beside_a_second_chat() {
    use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::atomic::AtomicBool,time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let caps=[1usize,2,3,4,5,6,7,9];let requests=caps.len()*2+2+2+2+2;
    let worker=std::thread::spawn(move|| {
        let c=fixture::Checkpoint::new_with_predictor(80);
        let split=Split::new(Host,FaultWire {fault:std::rc::Rc::new(std::cell::Cell::new(false)),aborted:std::rc::Rc::new(std::cell::Cell::new(false))}).unwrap();
        let engine=Factory::new(&c.admission,split,80,vec![0]).unwrap().with_predictor(3).unwrap();
        let mut limits=spark_tp::row_split::limits(80,48).unwrap();limits.next_mtp_depth=Some(engine.predictor_depth());
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:8,logits_per_entry:16,idle_ttl:Duration::from_secs(1)};
        let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<26,session_budget_bytes:1<<26,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"drafting",requests).unwrap();
        assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();
    });
    let ask=move|prompt:&str,tokens:usize,ignore_eos:bool,strict:bool| {
        let draft=if strict {r#""draft":"strict","mtp_depth":3"#} else {r#""draft":"plain""#};
        let body=format!(r#"{{"prompt_token_ids":{prompt},"stream":true,"max_tokens":{tokens},"ignore_eos":{ignore_eos},"cache":false,{draft}}}"#);
        let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
    let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect::<Vec<_>>();
    let finish=|reply:&str|reply.split("\"finish_reason\":\"").nth(1).map(|part|part.split('"').next().unwrap().to_owned());
    // An output cap inside a pass: the answer has exactly that many tokens, the plain ones.
    for cap in caps {let (plain,strict)=(ask("[3,5,1]",cap,true,false),ask("[3,5,1]",cap,true,true));
        assert_eq!(ids(&plain).len(),cap,"{plain}");assert_eq!(ids(&strict),ids(&plain),"cap {cap}: {strict}");assert_eq!(finish(&strict),finish(&plain),"cap {cap}");}
    // An end of the answer by the model's own stop token, if it comes: the same ids and the same reason.
    let (plain,strict)=(ask("[3,5,1]",40,false,false),ask("[3,5,1]",40,false,true));
    assert_eq!(ids(&strict),ids(&plain),"{strict}");assert_eq!(finish(&strict),finish(&plain));
    // An answer that ends one token under the context limit of 80: 31 prompt tokens and 48 outputs.
    let long=format!("[{}]",(0..31).map(|i|(1+i%7).to_string()).collect::<Vec<_>>().join(","));
    let (plain,strict)=(ask(&long,48,true,false),ask(&long,48,true,true));
    assert_eq!(ids(&plain).len(),48,"{plain}");assert_eq!(ids(&strict),ids(&plain),"{strict}");
    // A plain chat joins a drafting chat and leaves first: each has the ids it has alone.
    let (alone,other)=(ask("[3,5,1]",40,true,false),ask("[2,6,4]",12,true,false));
    let first=std::thread::spawn(move||ask("[3,5,1]",40,true,true));std::thread::sleep(Duration::from_millis(300));
    let second=ask("[2,6,4]",12,true,false);let first=first.join().unwrap();worker.join().unwrap();
    assert_eq!(ids(&alone).len(),40);assert_eq!(ids(&first),ids(&alone),"{first}");assert_eq!(ids(&second),ids(&other),"{second}");
}
