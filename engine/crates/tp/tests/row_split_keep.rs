//! Step 3 of the prompt cache on two ranks (host twins, the test wire): what a kept prompt is charged, the byte
//! cap, the vote of the two ranks on a keep and on a resume, the shared cache above the engine, and what a
//! stranger can see. The first two steps' tests are in row_split.rs.
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

/// Step 3, memory: the charge and the byte cap. A kept prompt is charged every plane it holds and the room behind
/// its row buffers, and holds no expanded rows. A keep past the byte cap is refused before any command: the chat goes
/// on with the bits of a fresh read and nothing is kept. A chat that fills its whole context beside kept prompts
/// at the cap reads the same bits, and the kept prompts still restore.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn a_kept_prompt_is_charged_all_it_holds_and_a_keep_past_the_byte_cap_is_refused_cleanly() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank charged and gave back the same bytes for every kept prompt");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        // The engine's own storage plan, asked apart from the service: the planes of a state of that length and the room behind them.
        let plan=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let charge=|tokens:usize|plan.bytes(tokens).unwrap()+plan.append_reservation_bytes();
        assert!(plan.append_reservation_bytes()>0 && charge(13)>charge(10) && charge(10)>charge(7));
        let prompts=[vec![1u32,3,2,7,5,11,4,9,6,13,8,2,10],vec![2u32,6,4,1,15,3,8,12,5,9],vec![3u32,9,1,14,6,2,11]];
        let extras=[vec![12u32,1,14],vec![5u32,5,9,2],vec![7u32,13]];
        let expected=(0..3).map(|chat|fresh(&[prompts[chat].as_slice(),extras[chat].as_slice()].concat(),6)).collect::<Vec<_>>();
        let plain=fresh(&prompts[2],5);
        // A chat that fills the whole context of 80 tokens: 79 prompt tokens and one answer token.
        let long=(0..79).map(|at|1+(at*7%15) as u32).collect::<Vec<_>>();let long_expected=fresh(&long,2);
        let mut live=(0..3).map(|chat|{let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();(session,logits)}).collect::<Vec<_>>();
        // THE CHARGE. The cache asks the live session before it forks, and the kept session after: one number.
        assert_eq!(live[0].0.retained_bytes(),Some(charge(13)));assert_eq!(live[0].0.kept_expanded_rows(),None);
        let kept0=live[0].0.fork().unwrap();
        assert_eq!(kept0.retained_bytes(),Some(charge(13)));assert_eq!(engine.kept_prompts(),(1,charge(13)));
        // It held expanded rows for all 13 prompt rows and dropped them at the keep. Each resume builds 13 again.
        assert_eq!(kept0.kept_expanded_rows(),Some(13));
        // THE BYTE CAP: the pool less one full session. Here the pool has room for a full chat and these two prompts.
        let full=engine.full_state_reservation_bytes();assert_eq!(engine.request_reservation_bytes(80),Some(full));
        assert_eq!(engine.kept_limit_bytes(),usize::MAX,"no byte cap until the server sets one");
        let pool=full+charge(13)+charge(10);engine.keep_within(pool);assert_eq!(engine.kept_limit_bytes(),charge(13)+charge(10));
        let kept1=live[1].0.fork().unwrap();assert_eq!(engine.kept_prompts(),(2,charge(13)+charge(10)));
        // The third keep does not fit. It is refused on rank 0 before any command, so the service is whole.
        assert!(live[2].0.can_fork());let refused=live[2].0.fork().err().unwrap();assert!(refused.contains("no bytes for one more kept prompt"),"{refused}");
        assert_eq!(engine.kept_prompts(),(2,charge(13)+charge(10)),"nothing was kept and nothing was charged");
        // The refused chat goes on and answers with the bits of a fresh read.
        {let (session,logits)=&mut live[2];for at in 0..5 {let id=best(logits);assert_eq!((id,bits(logits)),plain[at],"refused chat token {at}");if at+1<5 {*logits=Session::step(session,id).unwrap();}}}
        // A LIVE CHAT THAT FILLS ITS CONTEXT beside kept prompts at the cap. The cap leaves a full session of the pool.
        assert!(engine.kept_prompts().1+full<=pool);
        assert_eq!(fresh(&long,2),long_expected,"a full-size chat beside kept prompts at the byte cap");
        // The kept prompts are what they were: each chat's next turn has the bits of a fresh read.
        let kept=[Some(&kept0),Some(&kept1)];
        for chat in 0..2 {let mut session=kept[chat].unwrap().fork().unwrap();let mut logits=session.prefill(&extras[chat]).unwrap();
            for at in 0..6 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[chat][at],"chat {chat} token {at}");if at+1<6 {logits=Session::step(&mut session,id).unwrap();}}}
        // A released prompt gives its bytes back on both ranks, and then the refused keep fits (the chat is 11 tokens long now).
        drop(kept0);assert_eq!(engine.kept_prompts(),(1,charge(10)));
        let kept2=live[2].0.fork().unwrap();assert_eq!(engine.kept_prompts(),(2,charge(10)+charge(11)));assert_eq!(kept2.retained_bytes(),Some(charge(11)));
        // A prompt kept from a chat that itself started at a kept prompt: its expanded rows were rebuilt, then
        // extended by the new rows, then dropped again. The turn after it still has the bits of a fresh read.
        engine.keep_within(pool+charge(20));
        let mut second=kept1.fork().unwrap();let two=[prompts[1].as_slice(),extras[1].as_slice()].concat();second.prefill(&extras[1]).unwrap();
        let chained=second.fork().unwrap();drop(second);assert_eq!(chained.kept_expanded_rows(),Some(two.len()));assert_eq!(chained.retained_bytes(),Some(charge(two.len())));
        let third=[two.as_slice(),&[4u32,8,1]].concat();let third_expected=fresh(&third,5);
        let mut session=chained.fork().unwrap();let mut logits=session.prefill(&[4,8,1]).unwrap();
        for at in 0..5 {let id=best(&logits);assert_eq!((id,bits(&logits)),third_expected[at],"chained token {at}");if at+1<5 {logits=Session::step(&mut session,id).unwrap();}}
        drop(session);drop(chained);drop(kept2);drop(kept1);drop(live);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, memory: the server's shared prompt cache over this service, on two ranks. The cache holds kept prompts by
/// BYTES, with the charge the service reports. A prompt that does not fit beside the kept ones makes the cache give
/// back the least recently used one first, on both ranks, before the new one is kept. A prompt too large for the
/// cache alone is skipped and no command is sent. The scheduler lowers the same byte bound by the live chats'
/// reservations before it admits a chat (`rebalance_within`, tested in spark_serve), so that rule gives back in this order too.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn the_shared_cache_holds_kept_prompts_by_bytes_and_gives_back_the_least_recently_used_first() {
    use spark_serve::cache::{CacheLimits,CachedSession,SessionCache,SessionKey};
    use std::time::{Duration,Instant};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every prompt the cache gave back");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let plan=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let private=plan.bytes(10).unwrap()+plan.append_reservation_bytes();let vocab=engine.vocab_size();
        // Three chats, each with its own key and a prompt of 10 tokens. One cache entry: the kept bytes and the cache's own notes.
        let prompts=[vec![1u32,3,2,7,5,11,4,9,6,13],vec![2u32,6,4,1,15,3,8,12,5,9],vec![3u32,9,1,14,6,2,11,7,10,4]];let extra=[12u32,1,14];
        let keys=[SessionKey::generate().unwrap(),SessionKey::generate().unwrap(),SessionKey::generate().unwrap()];
        let expected=(0..3).map(|chat|fresh(&[prompts[chat].as_slice(),extra.as_slice()].concat(),5)).collect::<Vec<_>>();
        let entry=spark_serve::cache::publication_metadata_bytes(10,vocab,private,0,0).unwrap();
        // Room for two entries and a half: the third prompt fits only when one is given back.
        let limits=|bytes|CacheLimits {entries:8,retained_bytes:bytes,tokens_per_entry:80,logits_per_entry:vocab,idle_ttl:Duration::from_secs(300)};
        let mut cache=SessionCache::new(limits(2*entry+entry/2)).unwrap();let start=Instant::now();let at=|seconds|start+Duration::from_secs(seconds);
        // A chat reads its prompt, the cache snapshots it at the prompt's end, and the chat ends.
        let read=|cache:&mut SessionCache<_>,chat:usize,now:Instant| {let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();
            let state=CachedSession {checkpoint_lease:None,session,tokens:prompts[chat].clone(),next_logits:logits,reserved_session_bytes:engine.request_reservation_bytes(80).unwrap()};
            cache.snapshot(keys[chat],&state,now);};
        read(&mut cache,0,at(0));read(&mut cache,1,at(1));
        assert_eq!((cache.len(),engine.kept_prompts()),(2,(2,2*private)));assert!(cache.retained_bytes()>=2*private && cache.retained_bytes()<=2*entry+entry/2);
        // Chat 0 asks for its prompt again (an exact replay): its entry is now the more recently used one.
        let replay=cache.fork_prefix(&keys[0],&prompts[0],at(2));assert!(replay.failed.is_empty());
        let replay=replay.state.expect("the kept prompt restores");assert_eq!((replay.tokens.len(),bits(&replay.next_logits)),(10,fresh(&prompts[0],1)[0].1.clone()));drop(replay);
        // The third prompt does not fit beside both. The cache gives back chat 1's prompt, the least recently used:
        // the one a chat is least likely to ask for next. Both ranks release it before the new prompt is kept.
        read(&mut cache,2,at(3));
        assert_eq!((cache.len(),cache.counts().evicted,engine.kept_prompts()),(2,1,(2,2*private)));
        let whole=|chat:usize|[prompts[chat].as_slice(),extra.as_slice()].concat();
        assert!(cache.fork_prefix(&keys[1],&whole(1),at(4)).state.is_none(),"chat 1's prompt was given back: its next turn reads fresh");
        for chat in [0,2] {
            let forked=cache.fork_prefix(&keys[chat],&whole(chat),at(4)).state.expect("a kept prompt restores");assert_eq!(forked.tokens,prompts[chat]);
            let mut session=forked.session;let mut logits=session.prefill(&extra).unwrap();
            for at in 0..5 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[chat][at],"chat {chat} token {at}");if at+1<5 {logits=Session::step(&mut session,id).unwrap();}}
        }
        // A cache with less room than one prompt needs keeps nothing: the snapshot is skipped before any fork.
        let mut small=SessionCache::new(limits(private-1)).unwrap();read(&mut small,1,at(5));
        assert_eq!((small.len(),small.counts().skipped[1],engine.kept_prompts()),(0,1,(2,2*private)));
        // The service is whole, and dropping the cache releases every kept prompt on both ranks.
        assert_eq!(fresh(&whole(1),5),expected[1]);
        drop(small);drop(cache);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, memory: with the predictor loaded a kept prompt is charged the predictor layer's planes too (charged, not
/// held: they live with a chat's open rows, `Factory::kept_charge`). A chat that starts from it, plain or strict with
/// no depth, has the bits of a fresh read. Strict at depth 3 and default drafting from a kept prompt:
/// `kept_prompts_under_default_drafting` below.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn a_kept_prompt_with_the_predictor_loaded_is_charged_the_layer_and_restores_the_same_bits() {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(1).unwrap().with_prompt_keep();factory.worker().unwrap();assert_eq!(factory.kept_prompts(),(0,0));});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(1).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let plain=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let mut layered=checkpoint.description().with_draft_layer().unwrap().expect("the fixture declares a predictor layer");layered.residual=checkpoint.description().residual;
        let layered=spark_tp::admission_plan::StateChargePlan::new(&layered,80).unwrap();
        let prompt=vec![1u32,3,2,7,5,11,4,9,6,13,8,2,10];let extra=[12u32,1,14,3];let expected=fresh(&[prompt.as_slice(),extra.as_slice()].concat(),6);
        let mut first=engine.session().unwrap();first.prefill(&prompt).unwrap();let kept=first.fork().unwrap();drop(first);
        let charge=layered.bytes(13).unwrap()+layered.append_reservation_bytes();
        assert_eq!(kept.retained_bytes(),Some(charge));assert_eq!(engine.kept_prompts(),(1,charge));
        assert!(charge>plain.bytes(13).unwrap()+plain.append_reservation_bytes(),"the predictor layer's planes are charged");
        // A plain turn from the kept prompt.
        let mut session=kept.fork().unwrap();let mut logits=session.prefill(&extra).unwrap();
        for at in 0..6 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[at],"plain token {at}");if at+1<6 {logits=Session::step(&mut session,id).unwrap();}}
        drop(session);
        // A strict turn from the kept prompt: its rows go through verify passes and give the plain IDs.
        let limits=spark_tp::row_split::limits(80,64).unwrap();
        let strict=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":8,"draft":"strict"}"#,true,&limits).unwrap();
        let mut session=kept.fork().unwrap();session.configure_request(&strict).unwrap();let mut logits=session.prefill(&extra).unwrap();
        for at in 0..6 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[at],"strict token {at}");if at+1<6 {logits=Session::step(&mut session,id).unwrap();}}
        drop(session);drop(kept);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, review fault 2: one rank that cannot finish a keep (command 15) or a resume (command 16) refuses that one
/// request on BOTH ranks. Before the fix the refusing rank marked the whole service failed (rank 0) or left its loop
/// with an error (rank 1), and every live chat died. Now the ranks agree before either records anything
/// (`Factory::kept_vote`): nothing is kept or opened on either rank, the kept prompt is as it was, the service is not
/// marked failed, the chat beside goes on with the bits of a fresh read, and the next keep and resume work.
/// Each rank refuses each command once here: the worker rank its work numbers 0 and 3, rank 0 the ones it arms.
#[test]
fn a_keep_or_a_resume_one_rank_refuses_is_refused_on_both_and_every_chat_goes_on() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.refuse_kept_work(0b1001);
            factory.worker().expect("the worker rank goes on after each refusal and stops only when it is told to");
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank recorded no refused keep and gave back the kept one");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let plan=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let charge=plan.bytes(13).unwrap()+plan.append_reservation_bytes();
        let prompt=vec![1u32,3,2,7,5,11,4,9,6,13,8,2,10];let extra=vec![12u32,1,14];let other=vec![3u32,9,1,14,6,2,11];
        let expected=fresh(&[prompt.as_slice(),extra.as_slice()].concat(),6);let plain=fresh(&prompt,6);let beside=fresh(&other,8);
        // The chat beside: it takes one decode step after every refusal, and each has the bits of its fresh read.
        let mut by=engine.session().unwrap();let mut by_logits=by.prefill(&other).unwrap();let mut by_at=0;
        let mut goes_on=|what:&str| {
            assert!(engine.fatal_error().is_none(),"{what}: the service is not marked failed");
            let id=best(&by_logits);assert_eq!((id,bits(&by_logits)),beside[by_at],"{what}: the chat beside, token {by_at}");
            by_logits=Session::step(&mut by,id).unwrap();by_at+=1;};
        let mut live=engine.session().unwrap();let mut logits=live.prefill(&prompt).unwrap();
        // KEEP refused by the worker rank. Rank 0 parked its own state, heard the refusal, and dropped it again.
        let refused=live.fork().err().unwrap();assert!(refused.contains("the peer refused the kept prompt work"),"{refused}");
        assert_eq!(engine.kept_prompts(),(0,0));goes_on("keep refused by the worker rank");
        // KEEP refused by rank 0. The worker rank parked its own state, heard the refusal, and dropped it again.
        engine.refuse_kept_work(1);let refused=live.fork().err().unwrap();assert!(refused.contains("refused by the test hook"),"{refused}");
        assert_eq!(engine.kept_prompts(),(0,0));goes_on("keep refused by rank 0");
        // The same keep now works on both ranks, and the chat it was refused for is untouched.
        assert!(live.can_fork());let kept=live.fork().unwrap();assert_eq!((engine.kept_prompts(),kept.kept_expanded_rows()),((1,charge),Some(13)));
        // RESUME refused by the worker rank: rank 0 rebuilt the rows and opened nothing. The kept prompt is as it was.
        let refused=kept.fork().err().unwrap();assert!(refused.contains("the peer refused the kept prompt work"),"{refused}");
        assert_eq!((engine.kept_prompts(),kept.kept_expanded_rows()),((1,charge),Some(13)));goes_on("resume refused by the worker rank");
        // RESUME refused by rank 0: the worker rank rebuilt the rows and opened nothing.
        engine.refuse_kept_work(1);let refused=kept.fork().err().unwrap();assert!(refused.contains("refused by the test hook"),"{refused}");
        assert_eq!((engine.kept_prompts(),kept.kept_expanded_rows()),((1,charge),Some(13)));assert!(kept.can_fork());goes_on("resume refused by rank 0");
        // The kept prompt still restores, twice, and each turn from it has the bits of a fresh read.
        for turn in 0..2 {let mut session=kept.fork().unwrap();let mut logits=session.prefill(&extra).unwrap();
            for (at,row) in expected.iter().enumerate() {let id=best(&logits);assert_eq!(&(id,bits(&logits)),row,"turn {turn} token {at}");if at+1<6 {logits=Session::step(&mut session,id).unwrap();}}
            goes_on("a resumed turn");}
        // The chat whose keep was refused twice decodes on with the bits of its fresh read.
        for (at,row) in plain.iter().enumerate() {let id=best(&logits);assert_eq!(&(id,bits(&logits)),row,"refused chat token {at}");if at+1<6 {logits=Session::step(&mut live,id).unwrap();}}
        goes_on("the end");
        // Every refused resume gave its session slot back: a stop with a live session would be refused.
        drop(kept);drop(live);drop(by);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, review fault 1: the cache's byte bound must be the engine's byte cap (`Factory::keep_within` returns it, and
/// the server passes that number on). With the whole pool as the bound, an idle cache saw room for a third prompt, gave
/// nothing back, and the engine then refused the keep: the newest prompt was lost and no old one made way. With the
/// engine's cap as the bound the cache gives back its least recently used prompt first, on both ranks, and the keep
/// works. Both wirings run here over the same service: the old one shows the fault, the new one the rule.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn the_cache_bound_is_the_engine_cap_so_the_least_recently_used_prompt_makes_way_for_a_new_one() {
    use spark_serve::cache::{CacheLimits,CachedSession,SessionCache,SessionKey};
    use std::time::{Duration,Instant};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every prompt the cache gave back");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let plan=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let private=plan.bytes(10).unwrap()+plan.append_reservation_bytes();let vocab=engine.vocab_size();
        let prompts=[vec![1u32,3,2,7,5,11,4,9,6,13],vec![2u32,6,4,1,15,3,8,12,5,9],vec![3u32,9,1,14,6,2,11,7,10,4]];let extra=[12u32,1,14];
        let keys=[SessionKey::generate().unwrap(),SessionKey::generate().unwrap(),SessionKey::generate().unwrap()];
        let whole=|chat:usize|[prompts[chat].as_slice(),extra.as_slice()].concat();
        let expected=(0..3).map(|chat|fresh(&whole(chat),5)).collect::<Vec<_>>();
        // One cache entry: the kept bytes and the cache's own notes. So the cache never counts less than the engine.
        let entry=spark_serve::cache::publication_metadata_bytes(10,vocab,private,0,0).unwrap();assert!(entry>=private);
        // THE SERVER'S POOL: one full session and two cache entries. The engine's cap is the pool less the full session.
        let full=engine.full_state_reservation_bytes();let pool=full+2*entry;let cap=engine.keep_within(pool);
        assert_eq!((cap,engine.kept_limit_bytes()),(2*entry,2*entry));assert!(3*private>cap,"a third kept prompt is past the engine's cap");
        let limits=|bytes|CacheLimits {entries:8,retained_bytes:bytes,tokens_per_entry:80,logits_per_entry:vocab,idle_ttl:Duration::from_secs(300)};
        let start=Instant::now();let at=|seconds|start+Duration::from_secs(seconds);
        let read=|cache:&mut SessionCache<_>,chat:usize,now:Instant| {let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();
            let state=CachedSession {checkpoint_lease:None,session,tokens:prompts[chat].clone(),next_logits:logits,reserved_session_bytes:engine.request_reservation_bytes(80).unwrap()};
            cache.snapshot(keys[chat],&state,now);};
        // THE OLD WIRING, bound = the whole pool, on an idle server (no live chat holds any of it). The cache sees room
        // for the third prompt, gives nothing back, and the engine refuses the keep. The newest prompt is lost.
        let mut old=SessionCache::new(limits(pool)).unwrap();read(&mut old,0,at(0));read(&mut old,1,at(1));
        assert_eq!((old.len(),engine.kept_prompts()),(2,(2,2*private)));
        read(&mut old,2,at(2));
        assert_eq!((old.len(),old.counts().evicted,old.counts().skipped[2],engine.kept_prompts()),(2,0,1,(2,2*private)),"the fault: no prompt made way and the keep was refused by the engine");
        assert!(old.fork_prefix(&keys[2],&whole(2),at(3)).state.is_none(),"the fault: the newest prompt is not kept");
        drop(old);assert_eq!(engine.kept_prompts(),(0,0));
        // THE WIRING NOW, bound = the engine's cap. The cache gives back the least recently used prompt on both ranks
        // BEFORE it asks the engine for one more, so the engine never refuses a keep the cache had room for.
        let mut cache=SessionCache::new(limits(cap)).unwrap();read(&mut cache,0,at(10));read(&mut cache,1,at(11));
        assert_eq!((cache.len(),engine.kept_prompts()),(2,(2,2*private)));
        // Chat 0 asks for its prompt again: chat 1's is now the least recently used.
        let replay=cache.fork_prefix(&keys[0],&prompts[0],at(12));assert!(replay.failed.is_empty() && replay.state.is_some());drop(replay);
        read(&mut cache,2,at(13));
        assert_eq!((cache.len(),cache.counts().evicted,cache.counts().skipped,engine.kept_prompts()),(2,1,[0;4],(2,2*private)),"one prompt made way, nothing was skipped, the engine refused nothing");
        assert!(cache.retained_bytes()<=cap && engine.kept_prompts().1<=cache.retained_bytes(),"the engine holds no more than the cache counts");
        assert!(cache.fork_prefix(&keys[1],&whole(1),at(14)).state.is_none(),"chat 1's prompt made way: its next turn reads fresh");
        for chat in [0,2] {
            let forked=cache.fork_prefix(&keys[chat],&whole(chat),at(14)).state.expect("a kept prompt restores");assert_eq!(forked.tokens,prompts[chat]);
            let mut session=forked.session;let mut logits=session.prefill(&extra).unwrap();
            for at in 0..5 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[chat][at],"chat {chat} token {at}");if at+1<5 {logits=Session::step(&mut session,id).unwrap();}}
        }
        // A live chat of the full size still fits beside the kept prompts: the cap left it the pool's last full session.
        assert!(engine.kept_prompts().1+full<=pool);
        drop(cache);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, the count of kept prompts (blind read of step 1, claims C3 and C4). The server gives the cache the engine's
/// own count, KEPT_PROMPTS, as its entry bound. So the ninth prompt makes the cache give back its least recently used
/// one on both ranks BEFORE it asks the engine: the engine's count never refuses a keep the cache asked for, and the
/// cache does not stall at eight. And a session that has read nothing is not kept at all: no slot, no charge.
#[test]
fn the_ninth_prompt_makes_the_cache_give_one_back_and_an_empty_session_is_never_kept() {
    use spark_serve::cache::{CacheLimits,CachedSession,SessionCache,SessionKey};
    use std::time::{Duration,Instant};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every prompt the cache gave back");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let count=spark_tp::row_split::KEPT_PROMPTS;let vocab=engine.vocab_size();
        // AN EMPTY SESSION: it says it cannot be kept, and a keep is refused before any command. Nothing is counted.
        {let empty=engine.session().unwrap();assert!(!empty.can_fork());let refused=empty.fork().err().unwrap();
            assert!(refused.contains("has read no prompt to keep"),"{refused}");assert_eq!(engine.kept_prompts(),(0,0));assert!(engine.fatal_error().is_none());}
        // Nine chats, each with its own key and its own prompt of 6 tokens. The cache's entry bound is the engine's count.
        let prompts=(0..=count).map(|chat|(0..6).map(|at|1+((at*3+chat*5+at*chat)%15) as u32).collect::<Vec<_>>()).collect::<Vec<_>>();
        let keys=(0..=count).map(|_|SessionKey::generate().unwrap()).collect::<Vec<_>>();
        let mut cache=SessionCache::new(CacheLimits {entries:count,retained_bytes:usize::MAX/4,tokens_per_entry:80,logits_per_entry:vocab,idle_ttl:Duration::from_secs(300)}).unwrap();
        let start=Instant::now();let at=|seconds:usize|start+Duration::from_secs(seconds as u64);
        let mut ends=Vec::new();
        for chat in 0..=count {let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();ends.push(bits(&logits));
            let state=CachedSession {checkpoint_lease:None,session,tokens:prompts[chat].clone(),next_logits:logits,reserved_session_bytes:engine.request_reservation_bytes(80).unwrap()};
            cache.snapshot(keys[chat],&state,at(chat));
            let held=(chat+1).min(count);
            assert_eq!((cache.len(),engine.kept_prompts().0,cache.counts().skipped),(held,held,[0;4]),"after chat {chat}: the engine refused no keep");}
        // The ninth made exactly one prompt give way: chat 0's, the least recently used. The eight newest restore.
        assert_eq!(cache.counts().evicted,1);
        assert!(cache.fork_prefix(&keys[0],&prompts[0],at(20)).state.is_none(),"chat 0's prompt made way");
        for chat in 1..=count {let replay=cache.fork_prefix(&keys[chat],&prompts[chat],at(21)).state.expect("a kept prompt restores");assert_eq!(bits(&replay.next_logits),ends[chat],"chat {chat}");}
        drop(cache);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, eviction before a live chat waits or fails, over the real two-rank service. Before the scheduler admits a
/// chat (or lets one grow) it lowers the cache's byte bound to the pool less the live reservations
/// (`SessionCache::rebalance_within`, dispatch.rs). This test makes that same call. A full-size chat fits beside kept
/// prompts at the engine's cap and nothing gives way. When a second chat arrives, the least recently used kept prompt
/// is given back on both ranks BEFORE that chat opens, so live reservations plus kept bytes stay inside the pool. The
/// second chat starts from the prompt that stayed. Both chats have the bits of a fresh read.
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the arithmetic order.
fn kept_prompts_give_way_least_recently_used_first_before_a_live_chat_is_admitted() {
    use spark_serve::cache::{CacheLimits,CachedSession,SessionCache,SessionKey};
    use std::time::{Duration,Instant};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every prompt the cache gave back");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let plan=spark_tp::admission_plan::StateChargePlan::new(checkpoint.description(),80).unwrap();
        let private=plan.bytes(10).unwrap()+plan.append_reservation_bytes();let vocab=engine.vocab_size();
        let prompts=[vec![1u32,3,2,7,5,11,4,9,6,13],vec![2u32,6,4,1,15,3,8,12,5,9]];let extra=[12u32,1,14];
        let keys=[SessionKey::generate().unwrap(),SessionKey::generate().unwrap()];
        let whole=|chat:usize|[prompts[chat].as_slice(),extra.as_slice()].concat();
        let expected=fresh(&whole(0),5);
        let long=(0..79).map(|at|1+(at*7%15) as u32).collect::<Vec<_>>();let long_expected=fresh(&long,2);
        let entry=spark_serve::cache::publication_metadata_bytes(10,vocab,private,0,0).unwrap();
        // THE POOL: one full-size chat, one small chat (13 prompt tokens and 5 answer tokens), and ONE kept prompt.
        let full=engine.full_state_reservation_bytes();let small=engine.request_reservation_bytes_for_prompt(13,18,false,None,false).unwrap();
        assert!(entry<=small && small<full);
        let pool=full+small+entry;let cap=engine.keep_within(pool);assert_eq!(cap,small+entry);
        let mut cache=SessionCache::new(CacheLimits {entries:8,retained_bytes:cap,tokens_per_entry:80,logits_per_entry:vocab,idle_ttl:Duration::from_secs(300)}).unwrap();
        let start=Instant::now();let at=|seconds|start+Duration::from_secs(seconds);
        for chat in 0..2 {let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();
            let state=CachedSession {checkpoint_lease:None,session,tokens:prompts[chat].clone(),next_logits:logits,reserved_session_bytes:engine.request_reservation_bytes(80).unwrap()};
            cache.snapshot(keys[chat],&state,at(chat as u64));}
        // Chat 0 asks for its prompt again, so chat 1's is the least recently used. An idle pass gives nothing back.
        drop(cache.fork_prefix(&keys[0],&prompts[0],at(2)));cache.rebalance_within(pool,None,Default::default(),at(3));
        assert_eq!((cache.len(),cache.counts().evicted,engine.kept_prompts()),(2,0,(2,2*private)));
        // A FULL-SIZE CHAT ARRIVES. The bound is the pool less its reservation: both kept prompts stay, and it runs.
        cache.rebalance_within(pool-full,None,Default::default(),at(4));
        assert_eq!((cache.len(),cache.counts().evicted,engine.kept_prompts()),(2,0,(2,2*private)),"the cap left a full-size chat its room: nothing gives way");
        let mut first=engine.session().unwrap();let mut first_logits=first.prefill(&long).unwrap();
        // A SECOND CHAT ARRIVES. The bound drops to one entry: chat 1's prompt is given back on both ranks before it opens.
        cache.rebalance_within(pool-full-small,None,Default::default(),at(5));
        assert_eq!((cache.len(),cache.counts().evicted,engine.kept_prompts()),(1,1,(1,private)),"the least recently used kept prompt gave way before the chat was admitted");
        assert!(full+small+engine.kept_prompts().1<=pool,"live reservations and kept bytes are inside the pool");
        assert!(engine.fatal_error().is_none());
        assert!(cache.fork_prefix(&keys[1],&whole(1),at(6)).state.is_none(),"chat 1's prompt made way: its next turn reads fresh");
        // The second chat starts from the prompt that stayed, beside the full-size chat. Both have fresh bits.
        let forked=cache.fork_prefix(&keys[0],&whole(0),at(6)).state.expect("the kept prompt that stayed restores");assert_eq!(forked.tokens,prompts[0]);
        let mut second=forked.session;let mut logits=second.prefill(&extra).unwrap();
        for at in 0..5 {let id=best(&logits);assert_eq!((id,bits(&logits)),expected[at],"second chat token {at}");if at+1<5 {logits=Session::step(&mut second,id).unwrap();}
            if at<2 {let id=best(&first_logits);assert_eq!((id,bits(&first_logits)),long_expected[at],"full-size chat token {at}");if at<1 {first_logits=Session::step(&mut first,id).unwrap();}}}
        drop(second);drop(first);drop(cache);assert_eq!(engine.kept_prompts(),(0,0));
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// Step 3, second review, fault 1. A resumed chat's decoder is prepared before the ranks vote, and with no other chat
/// open that preparation talks to the other rank (the shared budget's collective). A failure there is a wire failure,
/// or leaves the ranks waiting on different rings, so it must NOT be voted away as a clean refusal: the service stops
/// loudly on both ranks. The test hook fails that preparation on one rank, as the failed collective would. Rank 0 is
/// marked failed with that first error and opens no new chat; the worker rank leaves its loop with an error. The
/// rebuild itself, which runs on one rank alone, is still voted (the test above).
#[test]
fn a_failure_where_a_resumed_chat_is_prepared_stops_the_service_on_both_ranks() {
    for failing in [1u32,0] {
        let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
        let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
        std::thread::scope(|scope| {
            let checkpoint=&c.admission;
            let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
                let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();if failing==1 {factory.fail_resume_prepare(1);}
                factory.worker().err()});
            {
                let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
                let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();if failing==0 {engine.fail_resume_prepare(1);}
                let prompt=vec![1u32,3,2,7,5,11,4,9,6,13,8,2,10];
                let mut live=engine.session().unwrap();live.prefill(&prompt).unwrap();
                let kept=live.fork().unwrap();assert_eq!(engine.kept_prompts().0,1);assert!(engine.fatal_error().is_none());
                let error=kept.fork().err().expect("the resume fails");
                assert!(!error.contains("the peer refused the kept prompt work"),"rank {failing}: the failure was voted away as a refusal: {error}");
                assert_eq!(engine.fatal_error(),Some(error.clone()),"rank {failing}: rank 0 is marked failed with the first error");
                assert_eq!(engine.session().err(),Some(error.clone()),"rank {failing}: no new chat opens on a service that failed");
                assert!(kept.fork().is_err() && live.fork().is_err(),"rank {failing}: no keep and no resume after the failure");
            }
            let stopped=worker.join().unwrap();assert!(stopped.is_some(),"rank {failing}: the worker rank left its loop with an error");
        });
    }
}

/// KEPT PROMPTS ARE PRIVATE TO THEIR CALLERS (step 2 reviews). The served scheduler over
/// the real two-rank service, with the settings the text service uses under `--prompt-keep`:
/// - `/health` asks no credential. It says the cache is on and shows NO live counter (entries, hits, reused tokens,
///   retained bytes each tell a stranger how long another caller's prompt is), before and after a hit.
/// - A caller with another credential never starts from this caller's kept prompt: it reads the prompt fresh, is told
///   0 cached tokens, and gets the same IDs. The first caller then starts from its own kept prompt, same IDs.
/// - `/v1/capabilities` tells a caller with no credential that its scope is shared, and a caller with one that the
///   scope is its credential.
#[test]
fn a_served_kept_prompt_shows_a_stranger_no_count_and_no_hit_on_two_ranks() {
    use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::atomic::AtomicBool,time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    // A failed check below must end the server too, or the test would wait for ever instead of failing.
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,std::sync::atomic::Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (checkpoint,stop)=(&c.admission,&stop);let _stop=Stop(stop);
        scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every kept prompt when the server stopped");});
        scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
            let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();let limits=spark_tp::row_split::limits(80,48).unwrap();
            let pool=(1usize<<24).max(4*engine.full_state_reservation_bytes());let cap=engine.keep_within(pool);
            let cache=spark_serve::cache::CacheLimits {entries:8,retained_bytes:cap,tokens_per_entry:80,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
            // As text.rs sets it with the switch: prefix_cache on, private counts on.
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:true,
                sharing:spark_serve::scheduler::Sharing {private_counts:true,..Default::default()}};
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,stop,"private",8).unwrap();
            assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();});
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let ask=|credential:&str| {let body=r#"{"prompt_token_ids":[3,5,1,7,2,9],"stream":true,"max_tokens":6,"ignore_eos":true,"temperature":0,"stream_options":{"include_usage":true}}"#;
            send(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\n{credential}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()))};
        let get=|path:&str,credential:&str|send(format!("GET {path} HTTP/1.1\r\nHost: {address}\r\n{credential}\r\n"));
        let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect::<Vec<_>>();
        let cached=|reply:&str|reply.rsplit("\"cached_tokens\":").next().unwrap().chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
        let no_count=|health:&str| {assert!(health.starts_with("HTTP/1.1 200 "),"{health}");assert!(health.contains("\"prefix_cache\":{\"enabled\":true}"),"{health}");
            for name in ["entries","hits","misses","reused_tokens","snapshots","evicted","retained_bytes","room_bytes"] {assert!(!health.contains(&format!("\"{name}\":")),"/health shows the live cache counter {name}: {health}");}};
        let other="Authorization: Bearer another-caller-of-this-server\r\n";
        // The first caller sends no credential. Its prompt is read fresh and kept at its end on both ranks.
        let first=ask("");assert_eq!(ids(&first).len(),6,"{first}");assert!(first.contains("\"cached_tokens\":"),"{first}");assert_eq!(cached(&first),0,"{first}");
        no_count(&get("/health",""));
        // Another caller, the same prompt: no hit, no count of the first caller's tokens, the same answer.
        let stranger=ask(other);assert_eq!(cached(&stranger),0,"another caller started from a prompt it did not send: {stranger}");assert_eq!(ids(&stranger),ids(&first),"{stranger}");
        // The first caller again: its own kept prompt serves it, with the IDs of the fresh read.
        let again=ask("");assert_eq!(cached(&again),6,"{again}");assert_eq!(ids(&again),ids(&first),"{again}");
        no_count(&get("/health",""));no_count(&get("/health",other));
        let shared=get("/v1/capabilities","");assert!(shared.contains("\"cache_scope\":\"shared\"") && shared.contains("\"prefix_cache\":true"),"{shared}");
        let scoped=get("/v1/capabilities",other);assert!(scoped.contains("\"cache_scope\":\"authorization\""),"{scoped}");
    });
}

/// A REAL failure inside a resume's own work, not the hook after it: rank 0's rebuild of the expanded rows fails at
/// an append, as a refused device allocation would, after it shared the kept planes. That one request is refused on
/// both ranks (the worker rank had built its rows and drops them), the service is not marked failed, and the kept
/// prompt is whole. Then, as the scheduler does after a refused resume, the prompt is read COLD in a new session: it
/// has the bits of a fresh read. The kept prompt still restores, and a chat that started from it decodes on with
/// fresh bits AFTER the kept prompt is given back on both ranks (it holds its own handles and its own rebuilt rows).
#[test]
fn a_rebuild_that_fails_on_one_rank_is_refused_on_both_and_a_chat_outlives_its_kept_prompt() {
    use watch::{Event,Watch};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let compute=Watch::default();
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
            factory.worker().expect("the worker rank goes on after the refusal and stops only when it is told to");
            assert_eq!(factory.kept_prompts(),(0,0));});
        let split=Split::new(&compute,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let prompt=vec![1u32,3,2,7,5,11,4,9,6,13,8,2,10];let extra=vec![12u32,1,14];let whole=[prompt.as_slice(),extra.as_slice()].concat();
        let expected=fresh(&whole,6);
        let mut live=engine.session().unwrap();live.prefill(&prompt).unwrap();let kept=live.fork().unwrap();drop(live);
        let before=engine.kept_prompts();assert_eq!((before.0,kept.kept_expanded_rows()),(1,Some(13)));
        // The next append of expanded rows on rank 0 fails: it is the rebuild's.
        compute.take();compute.fail_extend.set(Some(compute.extends()));
        let refused=kept.fork().err().unwrap();assert_eq!(refused,"watched device refused the expanded rows");compute.fail_extend.set(None);
        let events=compute.take();assert_eq!((events.first(),events.last()),(Some(&Event::Pause(true)),Some(&Event::Pause(false))),"recording is resumed after the failed rebuild: {events:?}");
        assert_eq!(engine.fatal_error(),None,"a rebuild one rank could not finish is a refusal, not a failure of the service");
        assert_eq!(engine.kept_prompts(),before);assert!(kept.can_fork());
        // The cold read the scheduler falls back to.
        assert_eq!(fresh(&whole,6),expected,"a cold read after a refused resume");
        // The kept prompt restores, and the chat from it takes six tokens.
        let mut turn=kept.fork().unwrap();let mut logits=turn.prefill(&extra).unwrap();
        for (at,row) in expected.iter().enumerate() {
            let id=best(&logits);assert_eq!(&(id,bits(&logits)),row,"token {at}");if at+1<expected.len() {logits=Session::step(&mut turn,id).unwrap();}
        }
        drop(kept);assert_eq!(engine.kept_prompts(),(0,0));
        // Given back on both ranks: the chat still appends rows and decodes with fresh bits.
        let longer=fresh(&whole,12);let mut logits=Session::step(&mut turn,longer[5].0).unwrap();
        for (at,row) in longer.iter().enumerate().skip(6) {let id=best(&logits);assert_eq!(&(id,bits(&logits)),row,"token {at} after the kept prompt was given back");if at+1<longer.len() {logits=Session::step(&mut turn,id).unwrap();}}
        drop(turn);engine.shutdown().unwrap();worker.join().unwrap();
    });
}

/// A wire that marks each read of the worker rank: by the idle wait of a header (`command`, false) or by the bounded
/// read of a payload (`command_payload`, true), with its length.
struct Marked(Wire,std::sync::Arc<std::sync::Mutex<Vec<(bool,usize)>>>);
impl Exchange for Marked {
    fn topology(&self)->Topology {self.0.topology()}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {self.0.gather(local,rows)}
    fn command(&mut self,v:&mut [u8])->Result<()> {self.1.lock().unwrap().push((false,v.len()));self.0.command(v)}
    fn command_payload(&mut self,v:&mut [u8])->Result<()> {self.1.lock().unwrap().push((true,v.len()));self.0.command(v)}
    fn complete(&mut self)->Result<()> {self.0.complete()}
}
/// Review fault (the unbounded wait): the session number that follows the header of a keep (command 15) and of a
/// resume (command 16) is owed at once. The worker rank reads it with the bounded read of a payload, never with the
/// idle wait of a header, which has no end: a number lost on the wire stops that rank at the command deadline
/// instead of holding it for ever. Each 16-byte header still takes the idle wait: it waits for the next request.
#[test]
fn the_worker_reads_the_session_number_of_a_keep_and_of_a_resume_with_the_bounded_read() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    let reads=std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let seen=reads.clone();
        let worker=scope.spawn(move || {let split=Split::new(Host,Marked(Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx},seen)).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        // Six tokens: their 24 bytes are no 8-byte read.
        let mut first=engine.session().unwrap();first.prefill(&[3,5,1,7,2,9]).unwrap();
        let kept=first.fork().unwrap();let turn=kept.fork().unwrap();
        drop(turn);drop(first);drop(kept);engine.shutdown().unwrap();worker.join().unwrap();
    });
    let reads=reads.lock().unwrap();
    assert_eq!(reads.iter().filter(|read|read.1==8).collect::<Vec<_>>(),[&(true,8),&(true,8)],"the number of the keep and of the resume: {reads:?}");
    let headers=reads.iter().filter(|read|read.1==16).collect::<Vec<_>>();
    // Open, the prompt group, keep, resume, the two drops, the release, stop.
    assert_eq!(headers.len(),8,"{reads:?}");assert!(headers.iter().all(|read|!read.0),"a header waits for the next request: {reads:?}");
}

/// A wire that counts the batch steps (command 13) the controller sends.
struct Batches(Wire,std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl Exchange for Batches {
    fn topology(&self)->Topology {self.0.topology()}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {self.0.gather(local,rows)}
    fn command(&mut self,v:&mut [u8])->Result<()> {
        if v.len()==16 && u32::from_le_bytes(v[..4].try_into().unwrap())==13 {self.1.fetch_add(1,std::sync::atomic::Ordering::SeqCst);}
        self.0.command(v)
    }
    fn complete(&mut self)->Result<()> {self.0.complete()}
}
/// KEPT PROMPTS MEET DEFAULT DRAFTING, through the scheduler on two ranks, as flash-serve runs with the predictor
/// loaded and `--prompt-keep`: the server default is strict at TRIAL_DEPTH, so a request that names no draft mode
/// runs its trial, and the cache keeps each prompt at its end. Every answer must have the IDs of a plain read with
/// the cache off, and the cached tokens the reply names show that the chat really started from a kept prompt.
/// - The same prompt again: the chat starts at the kept prompt's end and reads no row, so the predictor has NO prompt
///   row to start from (`Factory::tails` has none for it). Its trial probes on decoded rows alone.
/// - A second turn (the prompt and 5 more tokens): only the 5 new rows seed the predictor.
/// - Strict at depth 3 from a kept prompt: real proposals, verify passes of several rows, verdicts that drop rows.
/// - Two chats at once, each from its kept prompt: they ride batch steps (command 13).
///
/// `force`: every trial starts real drafting at its first closed shadow, so the probe is proven to have run on
/// both ranks and the request drafts from a predictor that saw no prompt row.
fn kept_prompts_under_default_drafting(force:bool) {
    use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::atomic::{AtomicBool,AtomicUsize,Ordering::SeqCst},time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let c=fixture::Checkpoint::new_with_predictor(144);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,SeqCst);}}
    let stop=AtomicBool::new(false);let batches=std::sync::Arc::new(AtomicUsize::new(0));
    std::thread::scope(|scope| {
        let (checkpoint,stop)=(&c.admission,&stop);let _stop=Stop(stop);
        scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(3).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank forgot every kept prompt when the server stopped");});
        let counter=batches.clone();
        scope.spawn(move || {let split=Split::new(Host,Batches(Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx},counter)).unwrap();
            let engine=Factory::new(checkpoint,split,144,vec![0]).unwrap().with_predictor(3).unwrap().with_prompt_keep();if force {engine.set_trial_always();}
            let mut limits=spark_tp::row_split::limits(144,96).unwrap();limits.next_mtp_depth=Some(engine.predictor_depth());
            let pool=(1usize<<26).max(4*engine.full_state_reservation_bytes());let cap=engine.keep_within(pool);
            let cache=spark_serve::cache::CacheLimits {entries:spark_tp::row_split::KEPT_PROMPTS,retained_bytes:cap,tokens_per_entry:144,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
            // As text.rs sets it with a loaded predictor and the switch.
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,
                draft:spark_serve::protocol::DraftMode::Strict,mtp_depth:spark_tp::row_split::TRIAL_DEPTH,draft_global_bank:false,prefix_cache:true,
                sharing:spark_serve::scheduler::Sharing {private_counts:true,..Default::default()}};
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,stop,"kept and drafting",10).unwrap();
            assert_eq!(engine.fatal_error(),None);let [started,_,gave_up]=engine.trial_counts();
            // Four lone requests name no draft mode (the first read, the same prompt again, the second turn, and at
            // least the first of the two at once runs alone for a while or ends its trial beside the other).
            if force {assert!(started>=3,"a trial from a kept prompt started real drafting: started {started}");} else {assert_eq!(started,0);assert!(gave_up>=3,"gave up {gave_up}");}
            engine.shutdown().unwrap();});
        // `mode`: the cache and draft fields of the request body; empty is a request that names neither.
        let ask=move|prompt:&str,tokens:usize,mode:&str| {
            let body=format!(r#"{{"prompt_token_ids":{prompt},"stream":true,"max_tokens":{tokens},"ignore_eos":true,"stream_options":{{"include_usage":true}}{mode}}}"#);
            let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect::<Vec<_>>();
        let cached=|reply:&str|reply.rsplit("\"cached_tokens\":").next().unwrap().chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
        let (prompt,turn)=("[3,5,1,7,2,9]","[3,5,1,7,2,9,4,6,2,8,1]");let (plain_cold,strict)=(r#","cache":false,"draft":"plain""#,r#","draft":"strict","mtp_depth":3"#);
        // The reference: a plain read with the cache off.
        let want=ids(&ask(prompt,88,plain_cold));assert_eq!(want.len(),88);
        // No draft mode: read fresh and kept at the prompt's end, then the same prompt again from the kept prompt.
        let first=ask(prompt,88,"");assert_eq!(cached(&first),0,"{first}");assert_eq!(ids(&first),want,"a default request, read fresh: {first}");
        let again=ask(prompt,88,"");assert_eq!(cached(&again),6,"{again}");assert_eq!(ids(&again),want,"a default request from the kept prompt, no new row: {again}");
        // Strict at depth 3 from the kept prompt: its first proposal comes from a predictor that saw no prompt row.
        let drafted=ask(prompt,88,strict);assert_eq!(cached(&drafted),6,"{drafted}");assert!(drafted.contains("\"mtp_depth\":3"),"{drafted}");assert_eq!(ids(&drafted),want,"strict at depth 3 from the kept prompt: {drafted}");
        let plain=ask(prompt,88,r#","draft":"plain""#);assert_eq!(cached(&plain),6,"{plain}");assert_eq!(ids(&plain),want,"plain from the kept prompt: {plain}");
        // The second turn: the kept prompt and 5 new tokens.
        let want_turn=ids(&ask(turn,80,plain_cold));assert_eq!(want_turn.len(),80);
        let second=ask(turn,80,"");assert_eq!(cached(&second),6,"{second}");assert_eq!(ids(&second),want_turn,"a default second turn from the kept prompt: {second}");
        let second_drafted=ask(turn,80,strict);assert_eq!(cached(&second_drafted),11,"{second_drafted}");assert_eq!(ids(&second_drafted),want_turn,"strict at depth 3 from the kept second turn: {second_drafted}");
        // Two chats at once, each from its kept prompt, no draft mode: batch steps, each chat's own IDs.
        let before=batches.load(SeqCst);
        let chats=[(prompt,88,6,&want),(turn,80,11,&want_turn)].map(|(prompt,tokens,kept,want)|(scope.spawn(move||ask(prompt,tokens,"")),kept,want));
        for (chat,kept,want) in chats {let reply=chat.join().unwrap();assert_eq!(cached(&reply),kept,"{reply}");assert_eq!(&ids(&reply),want,"beside another chat from a kept prompt: {reply}");}
        assert!(batches.load(SeqCst)>before,"two chats from kept prompts shared batch steps");
    });
}
#[test]
fn default_requests_from_kept_prompts_give_the_plain_ids_on_two_ranks() {kept_prompts_under_default_drafting(false);}
#[test]
fn trials_from_kept_prompts_that_start_real_drafting_give_the_plain_ids_on_two_ranks() {kept_prompts_under_default_drafting(true);}
