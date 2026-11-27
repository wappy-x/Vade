//! The live request count controls both ranks of an individual strict verifier command.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_serve::{Engine,Session};
use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
#[allow(dead_code)]
#[path="support/attention_checkpoint.rs"] mod fixture;
include!("support/row_wire.rs");

// This integration binary has only this test. Each multi-row command completes one
// first layer per rank. The process-wide execution counts therefore rise by two.
fn plan_counts()->[u64;2] {
    let line=spark_gpu_probe::prompt_launches::line();
    ["verify_solo_plan=","verify_owned_plan="].map(|name|
        line.split_whitespace().find_map(|part|part.strip_prefix(name)).unwrap().parse().unwrap())
}
#[test]
fn strict_rows_only_borrow_with_one_live_request_on_both_ranks_including_resumes() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep().worker().unwrap();
        });
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let limits=spark_tp::row_split::limits(80,32).unwrap();
        let strict=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":16,"draft":"strict"}"#,true,&limits).unwrap();
        let prompt=[3,5,1,7];let tokens=[2,6,4];
        let mut plain=engine.session().unwrap();plain.prefill(&prompt).unwrap();
        let expected=tokens.map(|t|Session::step(&mut plain,t).unwrap().iter().map(|x|x.to_bits()).collect::<Vec<_>>());
        drop(plain);
        let mut source=engine.session().unwrap();source.prefill(&prompt[..3]).unwrap();
        let kept=source.fork().unwrap();drop(source);
        for (cached,concurrent) in [(false,false),(false,true),(true,true),(true,false)] {
            let mut request=if cached {kept.fork().unwrap()} else {engine.session().unwrap()};
            request.configure_request(&strict).unwrap();request.prefill(if cached {&prompt[3..]} else {&prompt}).unwrap();
            let beside=concurrent.then(||engine.session().unwrap());
            let before=plan_counts();
            let result=request.draft_target().unwrap().verify_batch(&tokens).unwrap();
            let after=plan_counts();let delta=[after[0]-before[0],after[1]-before[1]];
            let bits=result.iter().map(|row|row.iter().map(|x|x.to_bits()).collect::<Vec<_>>()).collect::<Vec<_>>();
            assert_eq!(bits.as_slice(),&expected);
            assert_eq!(delta,if concurrent {[0,2]} else {[2,0]},"cached={cached}, concurrent={concurrent}");
            drop(beside);drop(request);
        }
        drop(kept);engine.shutdown().unwrap();worker.join().unwrap();
    });
}
