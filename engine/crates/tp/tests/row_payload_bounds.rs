use engine_model::{plan::layer::Topology,safetensors::Dtype};

use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
// The shared fixture has makers this file does not call.
#[allow(dead_code)]
#[path="support/attention_checkpoint.rs"] mod fixture;
#[allow(dead_code)]
#[path="support/watch_compute.rs"] mod watch;
struct Wire {topology:Topology,controller:bool,tx:SyncSender<Vec<u8>>,rx:Receiver<Vec<u8>>}
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

struct Marked(Wire,std::sync::Arc<std::sync::Mutex<Vec<(bool,usize)>>>);
impl Exchange for Marked {
    fn topology(&self)->Topology {self.0.topology()}
    fn gather(&mut self,v:&[f32],n:usize)->Result<Vec<f32>> {self.0.gather(v,n)}
    fn command(&mut self,v:&mut [u8])->Result<()> {self.1.lock().unwrap().push((false,v.len()));self.0.command(v)}
    fn command_payload(&mut self,v:&mut [u8])->Result<()> {self.1.lock().unwrap().push((true,v.len()));self.0.command(v)}
    fn complete(&mut self)->Result<()> {self.0.complete()}
}
#[test]
fn worker_prompt_payload_uses_bounded_read() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let reads=std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let seen=reads.clone();
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Marked(Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx},seen)).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().worker().unwrap();
        });
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();
        let mut session=engine.session().unwrap();session.prefill(&[3,5,1,7,2,9]).unwrap();
        assert_eq!(reads.lock().unwrap().iter().filter(|r|r.1==24).copied().collect::<Vec<_>>(),vec![(true,24)]);
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let request=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":40,"draft":"strict","temperature":0}"#,false,&limits).unwrap();
        session.configure_request(&request).unwrap();session.draft_target().unwrap().verify_batch(&[7,2,9]).unwrap();
        let mut sibling=engine.session().unwrap();sibling.prefill(&[2,4]).unwrap();
        let mut rows=[spark_serve::BatchRows {session:&mut session,tokens:&[1],output:spark_serve::BatchOutput::All,greedy:true,top_k:0},
            spark_serve::BatchRows {session:&mut sibling,tokens:&[2],output:spark_serve::BatchOutput::All,greedy:true,top_k:0}];
        engine.step_batch(&mut rows).map_err(|e|e.message).unwrap();
        drop(session);drop(sibling);engine.shutdown().unwrap();worker.join().unwrap();
    });
    let reads=reads.lock().unwrap();
    assert_eq!(reads.iter().filter(|r|r.0).copied().collect::<Vec<_>>(),vec![(true,24),(true,12),(true,8),(true,24)],"{reads:?}");
}
#[test]
fn invalid_sibling_preserves_open_verify_rows() {
    for bad_tokens in [vec![u32::MAX],vec![1;79]] {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let result=std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().worker().unwrap();
        });
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let request=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":40,"draft":"strict","temperature":0}"#,false,&limits).unwrap();
        let mut a=engine.session().unwrap();let mut b=engine.session().unwrap();
        a.configure_request(&request).unwrap();a.prefill(&[3,5,1]).unwrap();b.prefill(&[2,4]).unwrap();
        a.draft_target().unwrap().verify_batch(&[7,2,9]).unwrap();
        let rejected={let mut rows=[spark_serve::BatchRows {session:&mut a,tokens:&[1],output:spark_serve::BatchOutput::All,greedy:true,top_k:0},
            spark_serve::BatchRows {session:&mut b,tokens:&bad_tokens,output:spark_serve::BatchOutput::All,greedy:true,top_k:0}];engine.step_batch(&mut rows).is_err()};
        let restored=Session::truncate(&mut a,4);
        if restored.is_ok() {
            let mut reference=engine.session().unwrap();reference.prefill(&[3,5,1,7]).unwrap();
            assert_eq!(a.step(1).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>(),reference.step(1).unwrap().iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        }
        drop(a);drop(b);engine.shutdown().unwrap();worker.join().unwrap();
        (rejected,restored)
    });
    assert!(result.0);assert_eq!(result.1,Ok(()),"invalid sibling consumed healthy verify rows");
    }
}

#[derive(Clone,Copy,Debug)]
enum Malformed {EmptyLast,EmptyNone,EmptyStrict,LongStrict,OpenLast,OpenNone}
fn malformed_sibling_is_atomic(count:usize,kind:Malformed) {
    use spark_serve::{BatchOutput,BatchRows};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let commands=std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().worker()
        });
        let split=Split::new(Host,Marked(Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx},commands.clone())).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let strict=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":40,"draft":"strict","temperature":0}"#,false,&limits).unwrap();
        let open_bad=matches!(kind,Malformed::OpenLast|Malformed::OpenNone);
        let mut sessions=(0..count).map(|_|engine.session().unwrap()).collect::<Vec<_>>();
        for (i,session) in sessions.iter_mut().enumerate() {
            session.configure_request(&strict).unwrap();
            session.prefill(if i+1==count {&[2,4]} else {&[3,5,1]}).unwrap();
            if i+1<count || open_bad {session.draft_target().unwrap().verify_batch(&[7,2,9]).unwrap();}
        }
        let positions=sessions.iter_mut().map(|session|session.draft_target().unwrap().checkpoint()).collect::<Vec<_>>();
        let before=commands.lock().unwrap().len();
        let (tokens,output,error)=match kind {
            Malformed::EmptyLast=>(vec![],BatchOutput::Last,"empty row split prompt"),
            Malformed::EmptyNone=>(vec![],BatchOutput::None,"empty row split prompt"),
            Malformed::EmptyStrict=>(vec![],BatchOutput::All,"row split verify pass exceeds its rows or vocabulary"),
            Malformed::LongStrict=>(vec![1;spark_tp::bounded_decoder::VERIFY_ROWS+1],BatchOutput::All,"row split verify pass exceeds its rows or vocabulary"),
            Malformed::OpenLast=>(vec![1],BatchOutput::Last,"row split prompt has no state"),
            Malformed::OpenNone=>(vec![1],BatchOutput::None,"row split prompt has no state"),
        };
        let result={let mut rows=sessions.iter_mut().enumerate().map(|(i,session)|BatchRows {
            session,tokens:if i+1==count {&tokens} else {&[1]},output:if i+1==count {output} else {BatchOutput::All},greedy:true,top_k:0}).collect::<Vec<_>>();
            engine.step_batch(&mut rows).map(|_|()).map_err(|error|error.message)};
        assert_eq!(result,Err(error.into()),"{kind:?}, chats={count}");
        let after=commands.lock().unwrap().len();
        assert_eq!(after,before,"refused {kind:?} published a command, chats={count}");
        assert_eq!(engine.fatal_error(),None,"refused {kind:?} latched a service failure");
        assert_eq!(sessions.iter_mut().map(|session|session.draft_target().unwrap().checkpoint()).collect::<Vec<_>>(),positions,"{kind:?} changed a position");
        let bits=|row:Vec<f32>|row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        for (i,session) in sessions.iter_mut().enumerate() {
            let prefix=if i+1<count {Session::truncate(session,4).unwrap();vec![3,5,1,7]}
                else if open_bad {Session::truncate(session,3).unwrap();vec![2,4,7]} else {vec![2,4]};
            let mut reference=engine.session().unwrap();reference.prefill(&prefix).unwrap();
            assert_eq!(bits(session.step(1).unwrap()),bits(reference.step(1).unwrap()),"{kind:?}, chat={i}");
        }
        drop(sessions);engine.shutdown().unwrap();drop(engine);worker.join().unwrap().unwrap();
    });
}
#[test]
fn empty_prompt_sibling_preserves_open_verify_rows() {
    for kind in [Malformed::EmptyLast,Malformed::EmptyNone] {malformed_sibling_is_atomic(2,kind);}
}
#[test]
fn malformed_four_chat_sibling_preserves_every_verify_position() {
    for kind in [Malformed::EmptyLast,Malformed::EmptyNone,Malformed::EmptyStrict,Malformed::LongStrict,Malformed::OpenLast,Malformed::OpenNone] {malformed_sibling_is_atomic(4,kind);}
}
#[test]
fn malformed_strict_sibling_preserves_open_verify_rows() {
    for kind in [Malformed::LongStrict,Malformed::EmptyStrict] {malformed_sibling_is_atomic(2,kind);}
}
#[test]
fn prompt_on_open_verify_sibling_preserves_every_chat() {
    for kind in [Malformed::OpenLast,Malformed::OpenNone] {malformed_sibling_is_atomic(2,kind);}
}
#[test]
fn empty_plain_all_is_a_noop_with_committed_or_open_state() {
    use spark_serve::{BatchOutput,BatchRows,Predictions};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let commands=std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().worker()
        });
        let split=Split::new(Host,Marked(Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx},commands.clone())).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let strict=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":40,"draft":"strict","temperature":0}"#,false,&limits).unwrap();
        let plain=spark_serve::protocol::parse(br#"{"prompt":"probe","max_tokens":40,"draft":"plain","temperature":0}"#,false,&limits).unwrap();
        for open in [false,true] {
            let mut a=engine.session().unwrap();let mut b=engine.session().unwrap();
            for session in [&mut a,&mut b] {session.prefill(&[3,5,1]).unwrap();if open {
                session.configure_request(&strict).unwrap();session.draft_target().unwrap().verify_batch(&[7,2,9]).unwrap();session.configure_request(&plain).unwrap();}}
            let before=commands.lock().unwrap().len();
            let mut rows=[BatchRows {session:&mut a,tokens:&[],output:BatchOutput::All,greedy:true,top_k:0}];
            let output=engine.step_batch(&mut rows).map_err(|error|error.message).unwrap();
            assert!(matches!(&output[..],[Predictions::Logits(rows)] if rows.is_empty()));
            let after=commands.lock().unwrap().len();assert_eq!(after,before);assert_eq!(engine.fatal_error(),None);
            let bits=|row:Vec<f32>|row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(a.step(1).unwrap()),bits(b.step(1).unwrap()));
            drop(a);drop(b);
        }
        engine.shutdown().unwrap();drop(engine);worker.join().unwrap().unwrap();
    });
}
