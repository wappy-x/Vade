//! GROUP DRAFTING (row_split/pair.rs): two or three chats that share the server each draft by their own record, in
//! one shared pass: requests that name "strict" and requests that draft by default. Each chat's tokens, and the bits
//! of every row it keeps, are those of that chat alone by plain steps: by hand on two ranks, and served through the
//! scheduler. Host arithmetic and rank protocol; speed and device memory need the served gate.
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
/// One rank that holds every row: the served test's service.
struct Solo;
impl Exchange for Solo {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn best(v:&[f32])->u32 {(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32}
fn bits(v:&[f32])->Vec<u32> {v.iter().map(|v|v.to_bits()).collect()}
/// A chat alone by plain steps: each token it chooses and the bits of the row that chose it.
fn alone<E:Engine>(engine:&E,prompt:&[u32],count:usize)->Vec<(u32,Vec<u32>)> {
    let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut plain=Vec::new();
    for at in 0..count {let id=best(&logits);plain.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}plain
}
/// One scheduler step for the given chats: each chat's rows and their logits.
fn step<'s,E:Engine>(engine:&'s E,sessions:&mut [E::Session<'s>],tokens:&[Vec<u32>])->Vec<Vec<Vec<f32>>> {
    let mut rows=sessions.iter_mut().zip(tokens).map(|(session,tokens)|BatchRows {session,tokens,output:BatchOutput::All,greedy:false,top_k:0}).collect::<Vec<_>>();
    engine.step_batch(&mut rows).map_err(|e|e.message).unwrap().into_iter().map(|output|match output {Predictions::Logits(rows)=>rows,_=>panic!("a request that reads logits answers with logits")}).collect()
}
/// The rows chat `chat` sends in tick `tick`, after its proposer ran: `form` 0 is the proposer's own proposal, 1 is
/// a proposal that is right in every row, 2 is one that is wrong at its first row (and right after it).
fn rows_of(form:usize,next:u32,proposal:Vec<u32>,plain:&[(u32,Vec<u32>)],at:usize,vocab:u32)->Vec<u32> {
    let mut rows=vec![next];
    if proposal.is_empty() {return rows;}
    match form {
        0=>rows.extend(proposal),
        1=>rows.extend(plain[at..at+3].iter().map(|row|row.0)),
        _=>{rows.push((plain[at].0+1)%vocab);rows.extend(plain[at+1..at+3].iter().map(|row|row.0));},
    }
    rows
}
/// One tick in turns: each chat's proposer runs, then ONE scheduler step holds both chats' rows, then each chat's
/// verdict. Every kept row has the bits of that chat alone. `forms[chat]` sets the chat's rows (`rows_of`).
/// Returns the rows each chat sent and the rows each kept.
#[allow(clippy::needless_range_loop)] // index loops keep the two chats' rows side by side.
fn tick<'s,E:Engine>(engine:&'s E,sessions:&mut [E::Session<'s>;2],at:&mut [usize;2],forms:[usize;2],plain:&[Vec<(u32,Vec<u32>)>;2],prompts:&[Vec<u32>;2],vocab:u32)->(Vec<usize>,[usize;2]) {
    let (mut rows,mut entries)=(Vec::new(),Vec::new());
    for chat in 0..2 {
        let target=sessions[chat].draft_target().unwrap();target.limit_model_depth(3);entries.push(target.checkpoint());
        let next=plain[chat][at[chat]-1].0;let proposal=target.propose_model(&prompts[chat],next,3).unwrap().unwrap().tokens;
        rows.push(rows_of(forms[chat],next,proposal,&plain[chat],at[chat],vocab));
    }
    let logits=step(engine,sessions,&rows);let mut kept=[0;2];
    for chat in 0..2 {
        kept[chat]=1+rows[chat][1..].iter().zip(&plain[chat][at[chat]..]).take_while(|(a,b)|**a==b.0).count();
        sessions[chat].draft_target().unwrap().truncate(entries[chat]+kept[chat]).unwrap();
        for row in &logits[chat][..kept[chat]] {assert_eq!(bits(row),plain[chat][at[chat]].1,"chat {chat} row {}",at[chat]);at[chat]+=1;}
    }
    (rows.iter().map(Vec::len).collect(),kept)
}
#[test]
#[allow(clippy::needless_range_loop)] // index loops keep the two chats' rows side by side.
fn two_strict_chats_in_turns_keep_their_own_bits_through_each_others_rollbacks_on_two_ranks() {
    let c=fixture::Checkpoint::new_with_predictor(80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;let vocab=checkpoint.description().vocab as u32;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_predictor(3).unwrap();
        // The test control: every chat proposes and checks its whole depth in every tick. The rule itself is tested in pair.rs and below.
        engine.set_pair_always(true);
        let mut limits=spark_tp::row_split::limits(80,64).unwrap();limits.next_mtp_depth=Some(3);
        let request=|fields:&str|spark_serve::protocol::parse(format!(r#"{{"messages":[{{"role":"user","content":"Say hello."}}],"max_tokens":70{fields}}}"#).as_bytes(),true,&limits).unwrap();
        let strict=request(r#","draft":"strict","mtp_depth":3"#);
        // Which requests keep their draft state beside another: a strict one of depth 2 or more (a request that names
        // a depth and no mode is strict), and one that drafts by default: no draft mode or "auto", the depth the
        // scheduler gives it (the service's default) or its own of 3 or more, greedy, more than 64 tokens. No other.
        assert!(engine.drafts_contended(&strict));assert!(engine.drafts_contended(&request(r#","mtp_depth":3"#)));assert_eq!(engine.contended_draft_chats(),4);assert_eq!(engine.draft_chat_limit(),4);
        let default=|fields:&str,tokens:usize| {let mut request=request(fields);request.mtp_depth.get_or_insert(3);request.max_tokens=tokens;request};
        for kept in [default("",70),default(r#","draft":"auto""#,70),default(r#","draft":"auto","mtp_depth":3"#,70)] {assert!(engine.drafts_contended(&kept),"{:?} {:?} {}",kept.draft,kept.mtp_depth,kept.max_tokens);}
        assert!(!engine.drafts_contended(&default("",64)),"a short answer never drafts by default");assert!(!engine.drafts_contended(&default(r#","temperature":0.7"#,70)),"a sampled request");
        // A request with no depth of its own or of the service does not draft, as before.
        for other in ["",r#","draft":"auto","mtp_depth":2"#,r#","draft":"plain""#,r#","draft":"strict","mtp_depth":1"#,r#","draft":"strict","mtp_depth":3,"temperature":0.7"#] {
            assert!(!engine.drafts_contended(&request(other)),"{other}");}
        // With DRAFT CONFIDENCE off (a test control) a default request runs its trial alone and is plain beside another.
        engine.set_confidence(false);assert!(!engine.drafts_contended(&default("",70)) && engine.drafts_contended(&strict));engine.set_confidence(true);
        let count=40;let prompts=[vec![3u32,5,1,7,2,6],vec![2u32,6,4,1]];
        let plain=[alone(&engine,&prompts[0],count),alone(&engine,&prompts[1],count)];
        let third=alone(&engine,&[4,4,2],12);
        let open=|chat:usize| {let mut session=engine.session().unwrap();session.configure_request(&strict).unwrap();
            assert_eq!(bits(&session.prefill(&prompts[chat]).unwrap()),plain[chat][0].1);session};
        let tick=|sessions:&mut [_;2],at:&mut [usize;2],forms:[usize;2]|tick(&engine,sessions,at,forms,&plain,&prompts,vocab);
        let mut sessions=[open(0),open(1)];let mut at=[1usize,1];let (mut whole,mut rejected,mut own)=(0,0,0);
        for round in 0.. {
            if at[0]+8>=count || at[1]+8>=count {break;}
            // The two chats' forms move apart each round, so every pair of forms meets: both right, one wrong at
            // its first row while the other keeps every row (each way round), and the proposers' own proposals.
            let forms=[round%3,(round+1+(round/3)%2)%3];
            let (sent,kept)=tick(&mut sessions,&mut at,forms);
            for chat in 0..2 {
                if sent[chat]==1 {continue;}assert_eq!(sent[chat],4,"a proposal of depth 3 is a pass of four rows");
                match forms[chat] {1=>{assert_eq!(kept[chat],4);whole+=u32::from(forms[1-chat]==2 && sent[1-chat]==4 && kept[1-chat]==1);},2=>{assert_eq!(kept[chat],1);rejected+=1;},_=>own+=1}
            }
        }
        assert!(whole>=2 && rejected>=4 && own>=2,"one chat kept every row beside a chat rejected at row 1 {whole} times; rejected {rejected}; own proposals {own}");
        assert_eq!(engine.fatal_error(),None);
        // SHARED PASS (row_split/shared.rs): once each chat has its own timed step, the two chats' passes of a tick are
        // ONE pass on both ranks. The rows above (a whole pass kept beside a pass rejected at row 1) came out of it.
        let shared=engine.shared_counts();assert!(shared[0]>=8 && shared[1]>=shared[0]*5,"{} shared passes of {} rows",shared[0],shared[1]);
        // A chat that is cancelled between its proposal and its pass: its proposer ran, its rows never do. The other
        // chat's pass and every later row of it keep its own bits. The pair ends and prints its account.
        let [first,mut second]=sessions;let mut first=first;
        let target=first.draft_target().unwrap();target.limit_model_depth(3);target.propose_model(&prompts[0],plain[0][at[0]-1].0,3).unwrap();
        let target=second.draft_target().unwrap();target.limit_model_depth(3);let entry=target.checkpoint();
        let next=plain[1][at[1]-1].0;let proposal=target.propose_model(&prompts[1],next,3).unwrap().unwrap().tokens;
        drop(first);
        let counts=engine.pair_counts();assert_eq!(counts[0],1,"one pair ended");assert!(counts[1]>=8,"{} ticks in turns",counts[1]);assert_eq!(counts[2],0,"no step of two chats: the pair was in turns");
        let rows=rows_of(1,next,proposal,&plain[1],at[1],vocab);
        let logits=step(&engine,std::slice::from_mut(&mut second),std::slice::from_ref(&rows)).pop().unwrap();
        second.draft_target().unwrap().truncate(entry+rows.len()).unwrap();
        for row in &logits {assert_eq!(bits(row),plain[1][at[1]].1,"the other chat's row {}",at[1]);at[1]+=1;}
        // The chat that is left goes on alone, to its end.
        while at[1]<count {let next=vec![plain[1][at[1]-1].0];
            let logits=step(&engine,std::slice::from_mut(&mut second),std::slice::from_ref(&next)).pop().unwrap();
            assert_eq!(bits(&logits[0]),plain[1][at[1]].1,"alone again, row {}",at[1]);at[1]+=1;}
        drop(second);assert_eq!(engine.fatal_error(),None);
        // A third chat arrives: the scheduler ends both chats' drafting before it admits the third (draft_admission),
        // and the three ride one step. Each has its own bits, and the pair's record is closed.
        let mut sessions=[open(0),open(1)];let mut at=[1usize,1];
        for round in 0..5 {tick(&mut sessions,&mut at,[round%3,(round+1)%3]);}
        for session in &mut sessions {session.disable_model_draft();}
        assert_eq!(engine.pair_counts()[0],2,"the second pair ended when its chats gave their drafting up");
        let mut late=engine.session().unwrap();assert_eq!(bits(&late.prefill(&[4,4,2]).unwrap()),third[0].1);
        let [first,second]=sessions;let mut three=[first,second,late];
        for row in 1..8 {
            let tokens=vec![vec![plain[0][at[0]-1].0],vec![plain[1][at[1]-1].0],vec![third[row-1].0]];
            let logits=step(&engine,&mut three,&tokens);
            assert_eq!(bits(&logits[0][0]),plain[0][at[0]].1,"first chat beside two, row {}",at[0]);assert_eq!(bits(&logits[1][0]),plain[1][at[1]].1,"second chat beside two, row {}",at[1]);
            assert_eq!(bits(&logits[2][0]),third[row].1,"third chat, row {row}");at[0]+=1;at[1]+=1;
        }
        drop(three);assert_eq!(engine.fatal_error(),None);
        engine.shutdown().unwrap();worker.join().unwrap();
    });
}
/// The same through the scheduler: requests that run together are served by shared passes, and each answer is the
/// answer of that request alone. `always`: the test control makes every chat propose and check its whole depth in
/// every tick (so the counts below are exact enough to assert); without it each chat's record and the group's
/// measured prices decide, and only the answers are asserted. `default`: the chats name no draft mode, on a service
/// that drafts by default as flash-serve does (its scheduler names strict at the trial's depth); else they name strict.
fn served_pair(always:bool,default:bool) {
    use std::{io::{BufRead,BufReader,Read,Write},net::{TcpListener,TcpStream},sync::atomic::AtomicBool,time::Duration};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let requests=3+2+2+2+2+3+4+2+3+if default {2} else {0};let (counts_tx,counts_rx)=sync_channel(1);
    // A default request drafts only with an output cap above 64 tokens.
    let (context,long,short,third):(usize,usize,usize,usize)=if default {(96,70,66,68)} else {(80,40,9,32)};
    let worker=std::thread::spawn(move|| {
        let c=fixture::Checkpoint::new_with_predictor(context as u64);
        let engine=Factory::new(&c.admission,Split::new(Host,Solo).unwrap(),context,vec![0]).unwrap().with_predictor(3).unwrap();engine.set_pair_always(always);
        let mut limits=spark_tp::row_split::limits(context,context-8).unwrap();limits.next_mtp_depth=Some(engine.predictor_depth());
        let (mode,depth)=if default {(spark_serve::protocol::DraftMode::Strict,spark_tp::row_split::TRIAL_DEPTH)} else {(spark_serve::protocol::DraftMode::Plain,0)};
        let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:8,logits_per_entry:16,idle_ttl:Duration::from_secs(1)};
        let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<26,session_budget_bytes:1<<26,draft:mode,mtp_depth:depth,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"pair",requests).unwrap();
        assert_eq!(engine.fatal_error(),None);counts_tx.send((engine.pair_counts(),engine.shared_counts())).unwrap();engine.shutdown().unwrap();
    });
    let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect::<Vec<_>>();
    // `draft`: "strict" (depth 3), "plain", or "" for a request that names no draft mode. `cancel`: the client closes
    // its connection after that many tokens.
    let send=|prompt:&str,tokens:usize,draft:&str| {
        let draft=match draft {"strict"=>r#","draft":"strict","mtp_depth":3"#,"plain"=>r#","draft":"plain""#,_=>""};
        let body=format!(r#"{{"prompt_token_ids":{prompt},"stream":true,"max_tokens":{tokens},"ignore_eos":true,"cache":false{draft}}}"#);
        let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
        write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
        BufReader::new(stream)
    };
    fn read_tokens(reader:&mut BufReader<TcpStream>,reply:&mut String,want:usize) {
        while reply.matches("\"token_id\":").count()<want {
            assert!(reader.read_line(reply).unwrap()>0,"stream ended before token {want}");
        }
    }
    let finish=|mut reader:BufReader<TcpStream>,cancel:Option<usize>,mut reply:String| {
        match cancel {
            Some(most)=>read_tokens(&mut reader,&mut reply,most),
            None=>{reader.read_to_string(&mut reply).unwrap();}
        }
        reply
    };
    let ask=|prompt,tokens,draft,cancel|finish(send(prompt,tokens,draft),cancel,String::new());
    let together=|chats:Vec<(&str,usize,&str,Option<usize>,bool)>| {
        // Send the whole cohort before waiting for any response. A late chat waits for a token from each earlier chat.
        let mut pending=Vec::new();
        for (prompt,tokens,draft,cancel,late) in chats {
            if late {for (reader,_,reply) in &mut pending {read_tokens(reader,reply,1);}}
            pending.push((send(prompt,tokens,draft),cancel,String::new()));
        }
        pending.into_iter().map(|(reader,cancel,reply)|ids(&finish(reader,cancel,reply))).collect::<Vec<_>>()
    };
    let (p1,p2,p3)=("[3,5,1]","[2,6,4]","[4,4,2,1]");
    // `kind`: what the chats that draft name.
    let kind=if default {""} else {"strict"};
    // Each chat alone, by plain steps.
    let (a1,a2,a3)=(ids(&ask(p1,long,"plain",None)),ids(&ask(p2,long,"plain",None)),ids(&ask(p3,third,"plain",None)));
    assert_eq!((a1.len(),a2.len(),a3.len()),(long,long,third));
    // Two different prompts.
    let got=together(vec![(p1,long,kind,None,false),(p2,long,kind,None,false)]);assert_eq!(got,vec![a1.clone(),a2.clone()],"two chats");
    // The same prompt twice.
    let got=together(vec![(p1,long,kind,None,false),(p1,long,kind,None,false)]);assert_eq!(got,vec![a1.clone(),a1.clone()],"the same prompt twice");
    // A second chat joins a chat that drafts already: the first keeps its draft state, both draft.
    let got=together(vec![(p1,long,kind,None,false),(p2,long,kind,None,true)]);assert_eq!(got,vec![a1.clone(),a2.clone()],"a chat joins a drafting one");
    // One chat ends early, inside the other's answer: the other goes on alone and drafts on.
    let got=together(vec![(p1,long,kind,None,false),(p2,short,kind,None,false)]);assert_eq!(got,vec![a1.clone(),a2[..short].to_vec()],"one chat ends early");
    // A third chat arrives while two draft: the three are a new group, and all have their own answers.
    let got=together(vec![(p1,long,kind,None,false),(p2,long,kind,None,false),(p3,third,kind,None,true)]);assert_eq!(got,vec![a1.clone(),a2.clone(),a3.clone()],"a third chat arrives");
    // Four chats can share check rows under the forced control, and each keeps its own answer.
    let got=together(vec![(p1,long,kind,None,false),(p2,long,kind,None,false),(p3,third,kind,None,false),(p1,long,kind,None,false)]);
    assert_eq!(got,vec![a1.clone(),a2.clone(),a3.clone(),a1.clone()],"four chats");
    // A strict chat beside a request that names no draft mode. On a service that drafts by default the two are a
    // group. On one that does not, the request that names nothing is plain, and so both are, as before.
    let got=together(vec![(p1,long,"strict",None,false),(p2,long,"",None,false)]);assert_eq!(got,vec![a1.clone(),a2.clone()],"a strict chat beside a default one");
    // A chat that drafts by default beside a plain one: plain rows for both, as before.
    if default {let got=together(vec![(p1,long,"",None,false),(p2,long,"plain",None,false)]);assert_eq!(got,vec![a1.clone(),a2.clone()],"a default chat beside a plain one");}
    // One chat is cancelled in the middle: the other has its whole answer, and the service takes the next request.
    let got=together(vec![(p1,long,kind,Some(7),false),(p2,long,kind,None,false)]);
    assert_eq!(got[0][..7],a1[..7],"the cancelled chat's tokens up to the cancel");assert_eq!(got[1],a2,"the chat beside a cancelled one");
    assert_eq!(ids(&ask(p1,long,kind,None)),a1,"a chat alone after the groups");
    let (counts,shared)=counts_rx.recv_timeout(Duration::from_secs(120)).unwrap();worker.join().unwrap();
    eprintln!("served group always={always} default={default}: groups={} ticks with a checked row={} steps of all chats={} shared passes={} rows={} most chats={}",counts[0],counts[1],counts[2],shared[0],shared[1],shared[2]);
    // Every group of chats ran: this is what fails without group drafting, where a chat beside another gives its
    // draft state up and no group exists.
    assert!(counts[0]>=3,"{counts:?}");
    // With the control every chat of a group brought its whole depth in every tick.
    if always {assert!(counts[1]>=20,"{counts:?}");assert!(shared[0]>=20 && shared[2]==4,"four chats shared a pass: {shared:?}");}
    // Without it each chat's record decides (pair.rs): the answers above are what is asserted.
}
#[test]
fn served_strict_chats_share_their_passes_and_each_chat_keeps_its_own_ids() {served_pair(true,false);}
#[test]
fn served_strict_chats_under_the_rule_keep_their_own_ids() {served_pair(false,false);}
#[test]
fn served_default_chats_share_their_passes_and_each_chat_keeps_its_own_ids() {served_pair(true,true);}
#[test]
fn served_default_chats_under_the_rule_keep_their_own_ids() {served_pair(false,true);}
/// Ticks of chats under the rule itself, as the scheduler runs them: each chat's proposer runs while its request
/// drafts, ONE scheduler step holds every chat's rows, each chat keeps the rows that are its own answer, and a chat
/// whose drafting is over hands its token back. Every token a chat gets is its token alone (`expected`). Returns the
/// ticks that held a checked row, and the checked rows.
fn drive<'s,E:Engine>(engine:&'s E,sessions:&mut [E::Session<'s>],at:&mut [usize],expected:&[Vec<(u32,Vec<u32>)>],until:usize)->(usize,usize) {
    let (mut wide,mut checked)=(0,0);
    while at.iter().all(|&at|at+4<until) {
        let (mut rows,mut entries)=(Vec::new(),Vec::new());
        for (chat,session) in sessions.iter_mut().enumerate() {
            let next=expected[chat][at[chat]-1].0;let mut tokens=vec![next];
            match session.draft_target() {
                Some(target)=>{target.limit_model_depth(3);entries.push(Some(target.checkpoint()));tokens.extend(target.propose_model(&[],next,3).unwrap().unwrap().tokens);},
                None=>entries.push(None)}
            rows.push(tokens);
        }
        let results={let mut batch=sessions.iter_mut().zip(&rows).map(|(session,tokens)|BatchRows {session,tokens,output:BatchOutput::All,greedy:true,top_k:0}).collect::<Vec<_>>();
            engine.step_batch(&mut batch).map_err(|e|e.message).unwrap()};
        if rows.iter().any(|tokens|tokens.len()>1) {wide+=1;}
        for (chat,result) in results.into_iter().enumerate() {
            let ids=match result {Predictions::Greedy(ids)=>ids,Predictions::Logits(logits)=>logits.iter().map(|row|best(row)).collect(),_=>panic!("a decode row answers with its token or its logits")};
            assert_eq!(ids.len(),rows[chat].len());checked+=rows[chat].len()-1;
            let kept=1+rows[chat][1..].iter().zip(&expected[chat][at[chat]..]).take_while(|(a,b)|**a==b.0).count();
            if let Some(entry)=entries[chat] {sessions[chat].draft_target().unwrap().truncate(entry+kept).unwrap();}
            for id in &ids[..kept] {assert_eq!(*id,expected[chat][at[chat]].0,"chat {chat} token {}",at[chat]);at[chat]+=1;}
            sessions[chat].draft_over();
        }
    }
    (wide,checked)
}
#[test]
fn a_strict_and_a_default_chat_under_the_rule_keep_plain_ids_on_two_ranks_and_survive_one_chat_leaving() {
    let c=fixture::Checkpoint::new_with_predictor(96);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move|| {
            let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            Factory::new(checkpoint,split,96,vec![0]).unwrap().with_predictor(3).unwrap().worker().unwrap();
        });
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,96,vec![0]).unwrap().with_predictor(3).unwrap();
        let mut limits=spark_tp::row_split::limits(96,90).unwrap();limits.next_mtp_depth=Some(3);
        let strict=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":90,"draft":"strict","mtp_depth":3}"#,true,&limits).unwrap();
        // A request that names no draft mode, with the depth the scheduler gives it on a service that drafts by default.
        let mut default=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":90}"#,true,&limits).unwrap();default.mtp_depth=Some(3);
        assert!(engine.drafts_contended(&strict) && engine.drafts_contended(&default));
        let prompts=[vec![3,5,1],vec![2,6,4,1]];
        let expected=[alone(&engine,&prompts[0],80),alone(&engine,&prompts[1],80)];
        let open=|requests:[&spark_serve::protocol::Request;2]|prompts.iter().zip(requests).map(|(prompt,request)| {let mut s=engine.session().unwrap();s.configure_request(request).unwrap();s.prefill(prompt).unwrap();s}).collect::<Vec<_>>();
        // The rule itself: five ticks time the group, then one chat at a time looks in its answer text (a proposal
        // that no row checks, right when the tokens that come are its words), and the chain stop rides the proposal
        // commands to the other rank. This fixture's draft head is seldom sure, so the looks may never pass: every
        // token is the chat's own either way.
        let mut sessions=open([&strict,&default]);
        let mut at=[1usize,1];let (wide,checked)=drive(&engine,&mut sessions,&mut at,&expected,60);
        eprintln!("two ranks under the rule: ticks with a checked row={wide} checked rows={checked} shared passes={:?}",engine.shared_counts());
        // One chat leaves: the other goes on alone to its end, drafting or plain, with its own tokens.
        sessions.pop();
        let mut at=[at[0]];drive(&engine,&mut sessions,&mut at,&expected[..1],80);assert!(at[0]>=75);drop(sessions);
        // With DRAFT CONFIDENCE off (a test control) a look asks no price, and it checks no word either: like every
        // look it waits for the tokens that come, and a drafting starts only when each chat's look was right. On this
        // fixture that may never happen. Whatever happens, a row is checked only in a drafting, which is a shared pass
        // of both chats on two ranks, and every token is the chat's own (`drive`). (The test control's shared passes on
        // two ranks are the first case of this file.)
        engine.set_confidence(false);let before=engine.shared_counts()[0];
        let mut sessions=open([&strict,&strict]);let mut at=[1usize,1];let (wide,checked)=drive(&engine,&mut sessions,&mut at,&expected,60);
        assert!((wide==0)==(checked==0) && (checked==0 || engine.shared_counts()[0]>before),"rows are checked only in shared passes: {wide} {checked} {:?}",engine.shared_counts());
        drop(sessions);assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();worker.join().unwrap();
    });
}

// Observe the rank protocol: a final handoff must retire optional predictor storage, not only its handle.
// Its word is 2 (HANDOFF, draft_admission.rs): the predictor goes, the rows the prompt expanded stay.
struct ModeSolo(std::rc::Rc<std::cell::Cell<usize>>);
impl Exchange for ModeSolo {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
    fn command(&mut self,words:&mut [u8])->Result<()> {
        if words.len()==16 && u32::from_le_bytes(words[..4].try_into().unwrap())==31 && words[4..8]==[2,0,0,0] {self.0.set(self.0.get()+1);}
        Ok(())
    }
    fn complete(&mut self)->Result<()> {Ok(())}
}
/// One tick of `sessions` with teacher-forced inputs that deliberately do not follow the predictor: valid target
/// rows, but every cycle keeps its own row only. Returns which chats had proposed rows checked (a look has none)
/// and which handed their token back.
fn forced<'s,E:Engine>(engine:&'s E,sessions:&mut [E::Session<'s>],at:usize)->(Vec<bool>,Vec<bool>) {
    let (mut rows,mut entries)=(Vec::new(),Vec::new());
    for (chat,session) in sessions.iter_mut().enumerate() {
        let next=((at*5+3*chat)%11+1) as u32;let mut tokens=vec![next];
        match session.draft_target() {
            Some(target)=>{target.limit_model_depth(3);entries.push(Some(target.checkpoint()));tokens.extend(target.propose_model(&[],next,3).unwrap().unwrap().tokens);},
            None=>entries.push(None)}
        rows.push(tokens);
    }
    {let mut batch=sessions.iter_mut().zip(&rows).map(|(session,tokens)|BatchRows {session,tokens,output:BatchOutput::All,greedy:true,top_k:0}).collect::<Vec<_>>();
        engine.step_batch(&mut batch).map_err(|e|e.message).unwrap();}
    let over=sessions.iter_mut().zip(&entries).map(|(session,entry)| {
        if let Some(entry)=entry {session.draft_target().unwrap().truncate(entry+1).unwrap();}
        session.draft_over()}).collect();
    (rows.iter().map(|tokens|tokens.len()>1).collect(),over)
}
#[test]
fn chats_whose_rows_do_not_pay_keep_their_draft_state_and_a_chat_that_cannot_draft_on_hands_off_once() {
    let c=fixture::Checkpoint::new_with_predictor(96);
    let retired=std::rc::Rc::new(std::cell::Cell::new(0));
    let engine=Factory::new(&c.admission,Split::new(Host,ModeSolo(retired.clone())).unwrap(),96,vec![0]).unwrap().with_predictor(3).unwrap();
    let mut limits=spark_tp::row_split::limits(96,90).unwrap();limits.next_mtp_depth=Some(3);
    let request=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":90,"draft":"strict","mtp_depth":3}"#,true,&limits).unwrap();
    // The shortest answer that keeps its draft state: the ticks that time the group and one look.
    let mut short=request.clone();short.max_tokens=31;assert!(!engine.drafts_contended(&short));
    short.max_tokens=32;assert!(engine.drafts_contended(&short));
    let open=|request:&spark_serve::protocol::Request| {let mut s=engine.session().unwrap();s.configure_request(request).unwrap();s.prefill(&[1,2,3]).unwrap();s};
    // GROUP DRAFTING (pair.rs) with no price (DRAFT CONFIDENCE off, a test control). No chat looks before the net
    // has room for a look (38 plain ticks). A look with no price checks no word: it waits for the tokens that come,
    // and these inputs do not follow it, so it fails and no drafting starts: no row is checked in all 64 ticks. The
    // group never gives its draft state up: no chat hands off, and the predictor is never retired.
    engine.set_confidence(false);
    let mut sessions=vec![open(&request),open(&request)];let mut checked=[0usize;2];
    for at in 0..64 {
        let (proposed,over)=forced(&engine,&mut sessions,at);assert!(!over.contains(&true),"no handoff under the rule, tick {at}");
        for chat in 0..2 {checked[chat]+=usize::from(proposed[chat]);}
    }
    assert_eq!(checked,[0,0],"a look with no price checks no word, and no look here is right");
    assert!(sessions.iter_mut().all(|session|session.draft_target().is_some()));assert_eq!(retired.get(),0);
    drop(sessions);
    // Two chats that draft by default are a group too. A third chat that does not draft arrives: the two cannot
    // draft beside it, each ends its drafting, hands its pending token back once, and retires its predictor state.
    engine.set_confidence(true);
    let mut default=spark_serve::protocol::parse(br#"{"messages":[{"role":"user","content":"Continue."}],"max_tokens":90}"#,true,&limits).unwrap();default.mtp_depth=Some(3);
    assert!(engine.drafts_contended(&default));
    let mut sessions=vec![open(&default),open(&default)];
    for at in 0..4 {assert_eq!(forced(&engine,&mut sessions,at),(vec![false;2],vec![false;2]),"the group is being timed, tick {at}");}
    assert!(sessions.iter_mut().all(|session|session.draft_target().is_some()));
    let mut late=engine.session().unwrap();late.prefill(&[4,4,2]).unwrap();sessions.push(late);
    assert_eq!(forced(&engine,&mut sessions,4),(vec![false;3],vec![true,true,false]),"each default chat hands its pending token back");
    assert_eq!(forced(&engine,&mut sessions,5),(vec![false;3],vec![false;3]),"handoff happens once");
    assert!(sessions.iter_mut().all(|session|session.draft_target().is_none()));
    assert_eq!(retired.get(),2,"both handoffs retire predictor state exactly once");
    // The chat goes on by plain steps with its own bits.
    let plain=Factory::new(&c.admission,Split::new(Host,Solo).unwrap(),96,vec![0]).unwrap();
    let mut reference=plain.session().unwrap();let mut prefix=vec![1,2,3];
    prefix.extend((0..6).map(|at|(at*5%11+1) as u32));reference.prefill(&prefix).unwrap();
    for token in [6,4,1,5] {assert_eq!(bits(&Session::step(&mut sessions[0],token).unwrap()),bits(&Session::step(&mut reference,token).unwrap()));}
    drop(reference);plain.shutdown().unwrap();
    drop(sessions);assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();
}
