//! A drafting request's own numbers on two ranks: the plain price its chooser reads, and the
//! proposer's account. Neither can change a token; both decide what a served drafting test reports.
use engine_model::plan::layer::Topology;
use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use spark_serve::{BatchOutput,BatchRows,Engine,Predictions,Session};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
// The shared fixture has makers this file does not call.
#[allow(dead_code)]
#[path="support/attention_checkpoint.rs"] mod fixture;
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
type Service<'c>=Factory<'c,Host,Wire>;
/// One pass as the scheduler sends it: a greedy request's rows.
fn one_pass<'s,E:Engine>(engine:&'s E,session:&mut E::Session<'s>,tokens:&[u32])->Predictions {
    let mut rows=[BatchRows {session,tokens,output:BatchOutput::All,greedy:true,top_k:0}];
    engine.step_batch(&mut rows).map_err(|e|e.message).unwrap().pop().unwrap()
}
/// The chooser's price of a plain step of this request, by its bits. None until a steady step is timed.
fn price<S:Session>(session:&mut S)->Option<u64> {session.draft_target().unwrap().model_verify_ms(1).map(f64::to_bits)}
/// A proposal of depth 3 for the tokens after `next`, and the position at its entry.
fn propose<S:Session>(session:&mut S,prompt:&[u32],next:u32)->(Vec<u32>,usize) {
    let target=session.draft_target().unwrap();target.limit_model_depth(3);
    let entry=target.checkpoint();(target.propose_model(prompt,next,3).unwrap().unwrap().tokens,entry)
}
/// A two-rank service with the predictor layer, a strict request of depth 3, a prompt and the
/// plain tokens after it (`plain[0]` is the prompt's own next token).
fn with_service(run:impl FnOnce(&Service<'_>,&spark_serve::protocol::Request,&[u32],&[u32])) {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let mut limits=spark_tp::row_split::limits(80,64).unwrap();limits.next_mtp_depth=Some(3);
        let request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":40,"mtp_depth":3}"#,true,&limits).unwrap();
        let prompt=vec![3u32,5,1,7,2,6];let mut plain=Vec::new();
        {let mut session=engine.session().unwrap();let mut logits=session.prefill(&prompt).unwrap();
            for at in 0..24 {plain.push(best(&logits));if at+1<24 {logits=Session::step(&mut session,plain[at]).unwrap();}}}
        run(&engine,&request,&prompt,&plain);
        assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();worker.join().unwrap();
    });
}
/// The chooser's price of a plain step is this request's own steady steps: a pass of one row
/// that follows a pass of one row. A proposal runs the predictor layer and a verdict that drops
/// rows reaches both ranks, so the pass of one row after either is not a steady step, and it
/// must leave the price alone. (The flag that says "the call before was a step" went stale
/// across both: only a pass reset it.)
#[test]
fn a_one_row_pass_after_a_proposal_or_a_dropped_row_leaves_the_plain_price_alone() {
    with_service(|engine,request,prompt,plain| {
        let mut session=engine.session().unwrap();session.configure_request(request).unwrap();session.prefill(prompt).unwrap();
        // The first pass after the prompt is not timed; the second and the third are.
        assert_eq!(price(&mut session),None);
        for at in 0..3 {let Predictions::Greedy(ids)=one_pass(engine,&mut session,&plain[at..at+1]) else {panic!("the pass was not picked")};assert_eq!(ids,[plain[at+1]]);}
        let steady=price(&mut session);assert!(steady.is_some());
        // A proposal, then every proposed row is cut (the scheduler does this at a stop token): the pass has one row.
        assert!(!propose(&mut session,prompt,plain[3]).0.is_empty());
        let Predictions::Greedy(ids)=one_pass(engine,&mut session,&plain[3..4]) else {panic!("the pass was not picked")};assert_eq!(ids,[plain[4]]);
        assert_eq!(price(&mut session),steady,"a pass of one row right after a proposal was priced as a steady step");
        // The passes after it follow a pass of one row: they are timed again.
        one_pass(engine,&mut session,&plain[4..5]);one_pass(engine,&mut session,&plain[5..6]);
        let steady=price(&mut session);assert!(steady.is_some());
        // The scheduler's way back from a picked token the tokenizer does not hold: drop the row, run it again by its logits.
        Session::truncate(&mut session,prompt.len()+5).unwrap();Session::step(&mut session,plain[5]).unwrap();
        assert_eq!(price(&mut session),steady,"the row run again after a dropped row was priced as a steady step");
    });
}
/// The account keeps the proposal's length (the tokens the predictor made, which the proposal
/// time paid for) and the draft's length (the proposed rows the pass verified) side by side,
/// and its sums cover the same cycles as its counts. A proposal whose rows were all cut before
/// the pass is counted as unused: its tokens and its time are in no mean of the proposing cycles.
#[test]
fn the_account_keeps_the_proposal_length_and_the_draft_length_of_the_same_cycles() {
    with_service(|engine,request,prompt,plain| {
        let mut session=engine.session().unwrap();session.configure_request(request).unwrap();session.prefill(prompt).unwrap();
        for at in 0..3 {one_pass(engine,&mut session,&plain[at..at+1]);}
        // Cycle one: three tokens proposed, every row cut. The pass has one row and no verdict follows.
        let (cut,_)=propose(&mut session,prompt,plain[3]);assert_eq!(cut.len(),3);
        one_pass(engine,&mut session,&plain[3..4]);
        let account=session.proposer_account();
        assert_eq!((account.cycles,account.proposed_tokens,account.drafted_rows,account.kept_rows),(0,0,0,0),"a proposal with no verdict is in no cycle sum");
        assert_eq!((account.unused_proposals,account.unused_tokens),(1,3));assert!(account.unused_ms>0.0);
        assert_eq!((account.one_row_passes,account.picked_on_device),(4,4));
        // Cycle two: three tokens proposed, the scheduler cuts the proposal after its first token. The pass has two rows.
        let (tokens,entry)=propose(&mut session,prompt,plain[4]);assert_eq!(tokens.len(),3);
        let rows=[plain[4],tokens[0]];let kept=1+usize::from(tokens[0]==plain[5]);
        let Predictions::Logits(logits)=one_pass(engine,&mut session,&rows) else {panic!("a pass of rows answers with logits")};assert_eq!(logits.len(),2);
        session.draft_target().unwrap().truncate(entry+kept).unwrap();
        let account=session.proposer_account();
        assert_eq!((account.cycles,account.proposed_tokens,account.drafted_rows,account.kept_rows),(1,3,1,kept as u64));
        assert!(account.proposal_ms>0.0);assert_eq!((account.unused_proposals,account.unused_tokens),(1,3));
        // Cycle three: the whole proposal is verified.
        let at=4+kept;let (tokens,entry)=propose(&mut session,prompt,plain[at]);assert_eq!(tokens.len(),3);
        let mut rows=vec![plain[at]];rows.extend(&tokens);let kept=1+tokens.iter().zip(&plain[at+1..]).take_while(|(a,b)|a==b).count();
        one_pass(engine,&mut session,&rows);session.draft_target().unwrap().truncate(entry+kept).unwrap();
        let after=session.proposer_account();
        assert_eq!((after.cycles,after.proposed_tokens,after.drafted_rows),(2,6,4));assert_eq!(after.kept_rows,account.kept_rows+kept as u64);
        assert!(after.kept_rows<=after.drafted_rows+u64::from(after.cycles) && after.drafted_rows<=after.proposed_tokens);
    });
}
