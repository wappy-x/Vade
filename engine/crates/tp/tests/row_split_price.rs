//! How deep a drafting request proposes (row_split/price.rs), on two ranks: a request of depth 2
//! or more goes by its own record and proposes its whole depth or nothing; a chooser that picks
//! less cannot take it off that depth. A request of depth 1 takes the chooser's depth. Every kept
//! row has the plain ID.
use engine_model::plan::layer::Topology;
use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
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
/// The test plays the cycles by hand, with a chooser that picks no proposal or one token, in turn.
/// That cannot take a request of depth 2 or 3 off its whole depth:
/// - its first four cycles propose its whole depth (the test makes them lose every proposed row);
/// - it then takes plain steps and makes one whole try within the longest wait (the wait itself is
///   pinned by the unit tests of price.rs: the times of a loaded test host do not decide it there);
/// - after a cycle that kept its whole proposal it proposes its whole depth again;
/// - no proposal is shorter than its whole depth.
///
/// A request of depth 1 takes the chooser's depth. Every kept row has the plain ID.
#[test]
fn a_request_of_depth_two_or_more_proposes_its_whole_depth_or_nothing_by_its_own_record() {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let count=72;
        let mut limits=spark_tp::row_split::limits(80,count).unwrap();limits.next_mtp_depth=Some(3);
        let body=|depth:usize|format!(r#"{{"messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":{count},"mtp_depth":{depth}}}"#);
        let request=|depth:usize|spark_serve::protocol::parse(body(depth).as_bytes(),true,&limits).unwrap();
        let prompt=vec![3u32,5,1,7,2,6];
        let mut session=engine.session().unwrap();let mut logits=session.prefill(&prompt).unwrap();let mut plain=Vec::new();
        for at in 0..count {let id=best(&logits);plain.push(id);if at+1<count {logits=Session::step(&mut session,id).unwrap();}}drop(session);
        for depth in 1..=3 {
            let mut session=engine.session().unwrap();session.configure_request(&request(depth)).unwrap();
            let first=session.prefill(&prompt).unwrap();assert_eq!(best(&first),plain[0]);
            // `whole` = the last cycle kept its whole proposal; `waited` = plain steps since the first four cycles ended.
            let (mut at,mut verdicts,mut cycle,mut whole,mut waited,mut tried)=(1,0u32,0usize,false,0u32,false);
            while at<count {
                let target=session.draft_target().unwrap();let priced=target.model_verify_ms(1).is_some();
                // As the Speculator does: a pass may not make more tokens than the request has left, and the
                // depth is asked only when a proposal has room.
                let pick=cycle%2;cycle+=1;let limit=depth.min(count-at-1);let room=limit==depth;
                if limit>0 {target.limit_model_depth(pick);}
                let proposal=target.propose_model(&prompt,plain[at-1],limit).unwrap().unwrap().tokens;
                if depth==1 {assert_eq!(proposal.len(),if priced && room {pick.min(1)} else {0},"a request of depth 1 takes the chooser's depth (row {at})");}
                else {
                    assert!(proposal.is_empty() || proposal.len()==limit,"depth {depth}: a proposal of {} tokens (row {at})",proposal.len());
                    if !priced {assert!(proposal.is_empty());}
                    else if verdicts<4 {assert!(!proposal.is_empty(),"depth {depth}: cycle {verdicts} of the first four made no proposal (row {at})");}
                    else if whole && room {assert!(!proposal.is_empty(),"depth {depth}: no proposal after a cycle that kept its whole proposal (row {at})");}
                    else if !tried && proposal.is_empty() {waited+=1;assert!(waited<=64,"depth {depth}: no try within the longest wait");}
                }
                // The first four cycles verify rows that are not the plain tokens, so each keeps its first row only.
                // Every later cycle verifies the plain tokens in the place of the proposal, so it keeps them all.
                let mut rows=vec![plain[at-1]];
                if verdicts<4 {rows.extend(proposal.iter().enumerate().map(|(i,_)|u32::from(plain[(at+i).min(count-1)]==0)));}
                else {rows.extend(&plain[at..(at+proposal.len()).min(count)]);}
                rows.truncate(count-at);
                let entry=target.checkpoint();let logits=target.verify_batch(&rows).unwrap();
                let kept=1+rows[1..].iter().zip(&plain[at..]).take_while(|(a,b)|a==b).count();
                target.truncate(entry+kept).unwrap();
                if rows.len()>1 {if verdicts>=4 {tried=true;}verdicts+=1;whole=rows.len()==depth+1 && kept==rows.len();}
                for row in &logits[..kept] {assert_eq!(best(row),plain[at],"depth {depth} row {at}");at+=1;}
            }
            let account=session.proposer_account();assert_eq!(account.cycles,verdicts);
            eprintln!("depth {depth}: {verdicts} proposing cycles, {waited} plain steps before the try, account {account:?}");
            // Depth 2 and 3: the four first cycles, the try and the whole cycles after it. Depth 1: only the chooser's.
            if depth>=2 {assert!(tried && verdicts>=4+3 && account.unused_proposals==0,"depth {depth}: {verdicts} cycles");}
            drop(session);assert_eq!(engine.fatal_error(),None);
        }
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}
