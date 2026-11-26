//! Startup must lock the capacity it charges. Host rows check the lifecycle wiring.
use engine_model::{plan::{layer::Topology,description::TokenMixer},safetensors::Dtype};
use spark_tp::{row_split::{Exchange,Factory,Split},projection_memo::{Plan,Summary,FULL},Result};
use spark_serve::{Engine,Session};
#[allow(dead_code)] #[path="support/attention_checkpoint.rs"] mod fixture;
#[allow(dead_code)] #[path="support/watch_compute.rs"] mod watch;
struct Local;
impl Exchange for Local {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,v:&[f32],_:usize)->Result<Vec<f32>> {Ok(v.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn summary(c:&fixture::Checkpoint)->Summary {
    let mut s=Summary::default();let d=c.admission.description();
    for (layer,l) in d.layers.iter().enumerate() {if let TokenMixer::LatentAttention(a)=l.mixer {
        let name=format!("model.language_model.layers.{layer}.self_attn.kv_b_proj.weight");
        let slots=a.indexer.token_budget+if a.indexer.select_tail {a.indexer.pool-1} else {0};
        s.add((a.heads*(a.key_dim+a.value_dim)) as usize,a.kv_rank as usize,slots as usize,name.len()).unwrap();
    }}s
}
#[test]
fn startup_charge_equals_locked_capacity_through_fork_resume_and_reopen() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    for metadata in [None,Some(Summary::default()),Some(summary(&c))] {
        let compute=watch::Watch::default();compute.memo.set(metadata);
        let factory=Factory::new(&c.admission,Split::new(&compute,Local).unwrap(),80,vec![0]).unwrap().with_prompt_keep();
        let workspace=factory.workspace_reservation_bytes();let full=factory.full_state_reservation_bytes();
        let engine=factory.with_memo_bound().unwrap();let plan=compute.memo_bound.plan().unwrap();
        assert_eq!(workspace-engine.workspace_reservation_bytes(),FULL-plan.bytes());
        assert_eq!(full,engine.full_state_reservation_bytes());assert_eq!(plan,Plan::new(metadata,8));
        let bits=|row:Vec<f32>|row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
        let fresh={let mut s=engine.session().unwrap();s.prefill(&[1,3,2,5,7,4]).unwrap();bits(s.step(8).unwrap())};
        let kept={let mut s=engine.session().unwrap();s.prefill(&[1,3,2,5,7,4]).unwrap();s.fork().unwrap()};
        for _ in 0..3 {let mut resumed=kept.fork().unwrap();assert_eq!(bits(resumed.step(8).unwrap()),fresh);assert_eq!(compute.memo_bound.plan(),Some(plan));}
        drop(kept);engine.shutdown().unwrap();
    }
}
#[test]
fn a_device_cannot_acknowledge_another_charge_and_startup_cannot_apply_twice() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let compute=watch::Watch::default();
    compute.memo.set(Some(Summary::default()));compute.wrong_memo.set(true);
    let make=||Factory::new(&c.admission,Split::new(&compute,Local).unwrap(),80,vec![0]).unwrap();
    assert!(make().with_memo_bound().err().unwrap().contains("differs"));
    compute.wrong_memo.set(false);
    let engine=make().with_memo_bound().unwrap();assert!(engine.with_memo_bound().err().unwrap().contains("already"));
    assert!(make().with_memo_bound().unwrap().with_predictor(3).err().unwrap().contains("memo admission"));
}
#[test]
fn memo_admission_refuses_a_second_charge_a_live_session_and_a_later_prompt_group() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let compute=watch::Watch::default();compute.memo.set(Some(Summary::default()));
    let make=||Factory::new(&c.admission,Split::new(&compute,Local).unwrap(),80,vec![0]).unwrap();
    // A session the caller forgets stays counted as live; its borrow of the factory ends with the forget.
    let live=make();std::mem::forget(live.session().unwrap());
    assert_eq!(live.with_memo_bound().err().as_deref(),Some("memo admission requires no live session"));assert_eq!(compute.memo_bound.plan(),None);
    assert_eq!(make().with_memo_bound().unwrap().with_memo_bound().err().as_deref(),Some("memo startup charge is already set"));
    assert_eq!(make().with_memo_bound().unwrap().negotiate_prompt_group(usize::MAX).err().as_deref(),Some("prompt group selection follows memo admission"));
}

#[test]
fn cache_and_scheduler_reclaim_before_full_entitlements_open() {
    use std::{io::{Read,Write},net::{Ipv4Addr,TcpListener,TcpStream},sync::{Barrier,atomic::{AtomicBool,Ordering}},time::Duration};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let listener=TcpListener::bind((Ipv4Addr::LOCALHOST,0)).unwrap();let address=listener.local_addr().unwrap();let stop=AtomicBool::new(false);
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Release);}}
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let stop=&stop;
        let _stop=Stop(stop);
        let server=scope.spawn(move|| {
            let compute=watch::Watch::default();compute.memo.set(Some(Summary::default()));
            let engine=Factory::new(checkpoint,Split::new(&compute,Local).unwrap(),80,vec![0]).unwrap().with_prompt_keep().with_memo_bound().unwrap();
            assert!(engine.reserve_full_output());
            let full=engine.full_state_reservation_bytes();
            let charge={let mut s=engine.session().unwrap();s.prefill(&[1,3,2,5,7,4]).unwrap();s.retained_bytes().unwrap()};
            let metadata=spark_serve::cache::publication_metadata_bytes(6,engine.vocab_size(),0,0,0).unwrap();
            let pool=full+4*(charge+metadata);let cap=engine.keep_within(pool);assert_eq!(cap+full,pool);
            let cache=spark_serve::cache::CacheLimits {entries:8,retained_bytes:cap,tokens_per_entry:80,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:16,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,
                draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:true,sharing:spark_serve::scheduler::Sharing {private_counts:true,..Default::default()}};
            let limits=spark_tp::row_split::limits(80,48).unwrap();
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,stop,"fixture",18).unwrap();
            assert_eq!(engine.kept_prompts(),(0,0));assert_eq!(engine.fatal_error(),None);assert_eq!(compute.memo_bound.plan().unwrap().bytes(),0);engine.shutdown().unwrap();
        });
        let ask=|ids:&[u32],output:usize| {
            let body=format!("{{\"prompt_token_ids\":{ids:?},\"stream\":true,\"temperature\":0,\"draft\":\"plain\",\"max_tokens\":{output},\"ignore_eos\":true,\"stream_options\":{{\"include_usage\":true}}}}");
            let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
            let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();assert!(reply.starts_with("HTTP/1.1 200 "),"{reply}");
            assert!(!reply.contains("capacity_exhausted") && !reply.contains("\"error\""),"{reply}");
            let tokens=reply.matches("\"token_id\":").count();assert_eq!(tokens,output,"{reply}");reply
        };
        for i in 1..=8 {ask(&[i,3,2,5,7,4],2);}
        let replay=ask(&[8,3,2,5,7,4],2);assert!(replay.contains("\"cached_tokens\":6"),"{replay}");
        let long=(0..79).map(|i|1+(i%15) as u32).collect::<Vec<_>>();ask(&long,1);
        let barrier=Barrier::new(8);
        std::thread::scope(|scope| {let handles=(1..=8).map(|i| {let ask=&ask;let barrier=&barrier;scope.spawn(move|| {barrier.wait();ask(&[i,9,6,1,3,2],32);})}).collect::<Vec<_>>();for h in handles {h.join().unwrap();}});
        server.join().unwrap();
    });
}
