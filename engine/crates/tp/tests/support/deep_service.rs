//! Deep requests through the real scheduler, cache and two-rank command stream.
use super::*;
use std::{io::{Read,Write},net::{TcpListener,TcpStream,Ipv4Addr},sync::{Arc,Mutex,atomic::{AtomicBool,Ordering::SeqCst}},time::Duration};
#[derive(Default)]
struct Seen {depths:Mutex<Vec<usize>>,pause:AtomicBool,paused:Mutex<u64>,rows:Mutex<std::collections::BTreeMap<u64,usize>>,released:Mutex<std::collections::BTreeMap<u64,usize>>,fail:AtomicBool,fail_check:AtomicBool,dropped:AtomicBool}
struct Observed {wire:Wire,seen:Arc<Seen>,deep:bool,check:bool,payload:u32,entered:SyncSender<()>,resume:Receiver<()>}
impl Exchange for Observed {
    fn topology(&self)->Topology {self.wire.topology()}
    fn gather(&mut self,v:&[f32],rows:usize)->Result<Vec<f32>> {self.wire.gather(v,rows)}
    fn command(&mut self,v:&mut [u8])->Result<()> {
        self.wire.command(v)?;
        if self.payload!=0 {self.count_payload(v);return Ok(());}
        let code=u32::from_le_bytes(v[..4].try_into().unwrap());
        self.payload=if matches!(code,5|6|13|15|16) {code} else {0};self.check=code==6;
        let token=u32::from_le_bytes(v[4..8].try_into().unwrap());let id=u64::from_le_bytes(v[8..16].try_into().unwrap());
        let rows=match code {2|4|8|12|14=>1,5=>(token&!(1<<31)) as usize,6=>token as usize,_=>0};
        if rows>0 {*self.seen.rows.lock().unwrap().entry(id).or_default()+=rows;}
        if code==3 {let rows=self.seen.rows.lock().unwrap().remove(&id).unwrap_or(0);self.seen.released.lock().unwrap().insert(id,rows);}
        let depth=match code {9..=11=>Some((code-8) as usize),21..=24=>Some((code-17) as usize),_=>None};
        self.deep=depth.is_some_and(|n|n>3);
        if let Some(depth)=depth {self.seen.depths.lock().unwrap().push(depth);if self.deep && self.seen.pause.load(SeqCst) {*self.seen.paused.lock().unwrap()=id;}}Ok(())
    }
    fn command_payload(&mut self,v:&mut [u8])->Result<()> {self.wire.command(v)?;self.count_payload(v);Ok(())}
    fn complete(&mut self)->Result<()> {
        self.wire.complete()?;
        if self.deep && self.seen.fail.swap(false,SeqCst) || self.check && self.seen.fail_check.swap(false,SeqCst) {return Err("injected deep operation completion failure".into());}
        // Both ranks finished the proposal. Hold the controller inside the cycle until the client
        // has written the joining request or cancellation. No scheduling assumption or sleep.
        if self.deep && self.seen.pause.swap(false,SeqCst) {self.entered.send(()).map_err(|e|e.to_string())?;self.resume.recv_timeout(Duration::from_secs(60)).map_err(|e|e.to_string())?;}
        self.deep=false;Ok(())
    }
}
impl Observed {
    fn count_payload(&mut self,v:&[u8]) {
        if std::mem::take(&mut self.payload)==13 {
            assert_eq!(v.len()%12,0);
            let mut rows=self.seen.rows.lock().unwrap();
            for row in v.as_chunks::<12>().0 {let id=u64::from_le_bytes(row[..8].try_into().unwrap());*rows.entry(id).or_default()+=1;}
        }
    }
}
impl Drop for Observed {fn drop(&mut self) {self.seen.dropped.store(true,SeqCst);}}
fn send(address:std::net::SocketAddr,path:&str,body:&str)->TcpStream {
    let mut s=TcpStream::connect(address).unwrap();s.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
    write!(s,"POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();s
}
fn finish(mut s:TcpStream)->String {let mut reply=String::new();s.read_to_string(&mut reply).unwrap();reply}
fn body(mode:&str,cache:bool)->String {format!(r#"{{"prompt_token_ids":[3,5,1],"stream":true,"max_tokens":88,"ignore_eos":true,"cache":{cache}{mode}}}"#)}
fn ids(reply:&str)->Vec<u32> {reply.split("\"token_id\":").skip(1).map(|p|p.chars().take_while(char::is_ascii_digit).collect::<String>().parse().unwrap()).collect()}
#[test]
fn served_depth_limits_cache_reuse_cancel_and_join_on_two_ranks() {deep_lifecycle(true);}
#[test]
#[should_panic(expected="the disconnected request must stop before")]
fn connected_deep_request_fails_the_cancellation_assertion() {deep_lifecycle(false);}
fn deep_lifecycle(disconnect:bool) {
    let c=fixture::Checkpoint::new_with_predictor(144);
    let listener=TcpListener::bind((Ipv4Addr::LOCALHOST,0)).unwrap();let address=listener.local_addr().unwrap();
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let (entered_tx,entered_rx)=sync_channel(1);let (resume_tx,resume_rx)=sync_channel(1);
    let seen=Arc::new(Seen::default());let worker_seen=Arc::new(Seen::default());
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let counts=worker_seen.clone();
        let worker=scope.spawn(move|| {
            let (tx,_)=sync_channel(1);let (_,rx)=sync_channel(1);
            let split=Split::new(Host,Observed {wire:Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx},seen:counts,deep:false,check:false,payload:0,entered:tx,resume:rx}).unwrap();
            let engine=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(7).unwrap().with_prompt_keep();engine.worker().unwrap();assert_eq!(engine.kept_prompts(),(0,0));
        });
        let counts=seen.clone();
        let server=scope.spawn(move|| {
            let split=Split::new(Host,Observed {wire:Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx},seen:counts,deep:false,check:false,payload:0,entered:entered_tx,resume:resume_rx}).unwrap();
            let engine=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(7).unwrap().with_prompt_keep();engine.set_trial_always();engine.set_deep_always(true);
            let mut limits=spark_tp::row_split::limits(144,96).unwrap();limits.next_mtp_depth=Some(7);
            let cache=spark_serve::cache::CacheLimits {entries:8,retained_bytes:1<<24,tokens_per_entry:144,logits_per_entry:16,idle_ttl:Duration::from_secs(300)};
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<28,session_budget_bytes:1<<28,draft:spark_serve::protocol::DraftMode::Strict,mtp_depth:3,draft_global_bank:false,prefix_cache:true,sharing:Default::default()};
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"deep lifecycle",11).unwrap();
            assert_eq!(engine.fatal_error(),None);assert_eq!(engine.kept_prompts(),(0,0));engine.shutdown().unwrap();
        });
        let ask=|mode:&str,cache|finish(send(address,"/v1/completions",&body(mode,cache)));
        let plain=ids(&ask(r#", "draft":"plain""#,false));assert_eq!(plain.len(),88);
        for depth in [3,4,5] {
            seen.depths.lock().unwrap().clear();
            let reply=ask(&format!(r#", "mtp_depth":{depth}"#),false);assert_eq!(ids(&reply),plain);
            let depths=seen.depths.lock().unwrap();assert!(!depths.is_empty());assert!(depths.iter().all(|&n|n<=depth),"explicit {depth}: {depths:?}");
        }
        seen.depths.lock().unwrap().clear();assert_eq!(ids(&ask("",false)),plain);
        assert!(seen.depths.lock().unwrap().contains(&7),"omitted depth uses loaded depth after trial");
        let strict=r#", "draft":"strict","mtp_depth":7"#;
        assert_eq!(ids(&ask(strict,true)),plain);
        let reused=ask(strict,true);assert_eq!(ids(&reused),plain);assert!(reused.contains("\"cached_tokens\":3"),"{reused}");
        seen.pause.store(true,SeqCst);let first=send(address,"/v1/completions",&body(strict,true));
        entered_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        let second=send(address,"/v1/completions",&body(r#", "draft":"plain""#,false));resume_tx.send(()).unwrap();
        assert_eq!(ids(&finish(first)),plain);assert_eq!(ids(&finish(second)),plain);
        let cancel=spark_serve::cache::SessionKey::generate().unwrap().expose();
        seen.pause.store(true,SeqCst);let first=send(address,"/v1/completions",&body(&format!(r#"{strict},"cancel":"{cancel}""#),true));
        entered_rx.recv_timeout(Duration::from_secs(60)).unwrap();
        // A disconnect cancels the live request after the deep proposal, even if its output was buffered.
        let cancelled_session=*seen.paused.lock().unwrap();
        let _connected=if disconnect {first.shutdown(std::net::Shutdown::Both).unwrap();drop(first);None} else {Some(first)};resume_tx.send(()).unwrap();
        assert_eq!(ids(&ask(strict,true)),plain,"request after cancelled deep cycle");
        server.join().unwrap();worker.join().unwrap();
        for rank in [&seen,&worker_seen] {
            let released=rank.released.lock().unwrap();let rows=*released.get(&cancelled_session).expect("both ranks must release the disconnected session");
            assert!(rows<87,"the disconnected request must stop before the 87 decode inputs needed to finish 88 tokens: {rows}");
        }
        assert!(worker_seen.depths.lock().unwrap().contains(&7),"the worker ran deep proposals too");
    });
}
#[test]
fn normalized_request_keeps_explicit_depth_without_a_draft_mode() {
    let c=fixture::Checkpoint::new_with_predictor(144);
    let split=Split::new(Host,FaultWire {fault:Default::default(),aborted:Default::default()}).unwrap();
    let engine=Factory::new(&c.admission,split,144,vec![0]).unwrap().with_predictor(7).unwrap();
    let mut limits=spark_tp::row_split::limits(144,96).unwrap();limits.next_mtp_depth=Some(7);
    let mut request=spark_serve::protocol::parse(body(r#", "mtp_depth":3"#,false).as_bytes(),false,&limits).unwrap();
    request.draft=None;let mut session=engine.session().unwrap();session.configure_request(&request).unwrap();
    assert_eq!(spark_engine::draft::Target::model_draft_width(&session),3);
    drop(session);
    // The depth-7 server keeps the depth-3 request bound and funds only a lone draft.
    let plain_split=Split::new(Host,FaultWire {fault:Default::default(),aborted:Default::default()}).unwrap();
    let plain=Factory::new(&c.admission,plain_split,144,vec![0]).unwrap();
    let base=plain.request_reservation_bytes_for_prompt(3,99,true,None,false).unwrap();
    request.draft=Some(spark_serve::protocol::DraftMode::Strict);
    let price=|contended|engine.request_reservation_bytes_for_request(3,99,contended,None,false,&request).unwrap();
    assert!(price(false)>base,"a lone explicit depth-3 request funds the predictor");
    assert_eq!(price(true),base,"depth 7 must not charge optional state to contended chats");
    let mut chats=Vec::new();
    for index in 0..8 {
        if index==1 {
            spark_serve::Session::disable_model_draft(&mut chats[0]);
            assert!(spark_serve::Session::draft_over(&mut chats[0]),"retirement must hand back the pending canonical token");
        }
        let mut chat=engine.session().unwrap();chat.configure_request(&request).unwrap();
        assert_eq!(spark_engine::draft::Target::model_draft_width(&chat),3);
        if index>0 {chat.disable_model_draft();assert!(chat.draft_over());}
        chats.push(chat);
    }
    assert_eq!(chats.len()*price(true),8*base,"all eight invoices are plain");
    for chat in &mut chats {assert!(chat.draft_target().is_none(),"retired chats must stay plain");}
    drop(chats);plain.shutdown().unwrap();engine.shutdown().unwrap();
}

#[test]
fn failed_deep_operations_stop_both_ranks_and_a_fresh_service_keeps_ids() {
    let c=fixture::Checkpoint::new_with_predictor(144);let mut answers=Vec::new();
    for fault in [0,1,2,0] {
        let fail=fault!=0;
        let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
        let seen=Arc::new(Seen::default());let worker_seen=Arc::new(Seen::default());
        std::thread::scope(|scope| {
            let checkpoint=&c.admission;let counts=worker_seen.clone();
            let worker=scope.spawn(move|| {
                let (tx,_)=sync_channel(1);let (_,rx)=sync_channel(1);
                let split=Split::new(Host,Observed {wire:Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx},seen:counts,deep:false,check:false,payload:0,entered:tx,resume:rx}).unwrap();
                let engine=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(7).unwrap();
                let result=engine.worker();
                if fail {assert_eq!(result.unwrap_err(),"injected deep operation completion failure");} else {result.unwrap();}
                // Returning from worker drops its complete session map, including open Verify.
                drop(engine);
            });
            let (tx,_)=sync_channel(1);let (_,rx)=sync_channel(1);
            let split=Split::new(Host,Observed {wire:Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx},seen:seen.clone(),deep:false,check:false,payload:0,entered:tx,resume:rx}).unwrap();
            let engine=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(7).unwrap();engine.set_deep_always(true);
            let mut limits=spark_tp::row_split::limits(144,96).unwrap();limits.next_mtp_depth=Some(7);
            let mode=if fail {r#", "draft":"strict","mtp_depth":7"#} else {r#", "draft":"plain""#};
            let request=spark_serve::protocol::parse(body(mode,false).as_bytes(),false,&limits).unwrap();
            let mut session=engine.session().unwrap();session.configure_request(&request).unwrap();
            let mut logits=session.prefill(&[3,5,1]).unwrap();
            if fail {
                Session::step(&mut session,7).unwrap();Session::step(&mut session,7).unwrap();
                if fault==1 {seen.fail.store(true,SeqCst);worker_seen.fail.store(true,SeqCst);}
                let target=session.draft_target().unwrap();target.limit_model_depth(7);
                let proposal=target.propose_model(&[3,5,1,7],2,7);
                let error=if fault==1 {proposal.err().expect("injected proposal must fail")} else {
                    assert!(!proposal.unwrap().unwrap().tokens.is_empty());
                    seen.fail_check.store(true,SeqCst);worker_seen.fail_check.store(true,SeqCst);
                    target.verify_batch(&[2,6,4,1,7,6,3,2]).unwrap_err()
                };
                assert_eq!(error,"injected deep operation completion failure");
                assert_eq!(engine.fatal_error(),Some(error.clone()));
                assert_eq!(engine.session().err(),Some(error.clone()),"a failed factory must never be reused");
                drop(session);assert_eq!(engine.shutdown().unwrap_err(),error);
            } else {
                let mut ids=Vec::new();
                for _ in 0..88 {
                    let mut best=0;for i in 1..logits.len() {if logits[i]>logits[best] {best=i;}}
                    ids.push(best as u32);logits=Session::step(&mut session,best as u32).unwrap();
                }
                answers.push(ids);drop(session);engine.shutdown().unwrap();
            }
            drop(engine);worker.join().unwrap();
            if fail {for rank in [&seen,&worker_seen] {assert!(rank.depths.lock().unwrap().contains(&7));assert!(!rank.fail.load(SeqCst));assert!(!rank.fail_check.load(SeqCst));}}
            assert!(seen.dropped.load(SeqCst));assert!(worker_seen.dropped.load(SeqCst));
        });
    }
    assert_eq!(answers.len(),2);assert_eq!(answers[0],answers[1],"fresh service after failure must preserve every ID");
}
