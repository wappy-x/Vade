//! Host tests of the text service on the made-up checkpoints of `fixture.rs`: the real HTTP scheduler on a loopback
//! port, the driver on the host backend, and the settings that the service start gives (`Service::schedule`, `limits`).
//!
//! The reference of every served token ID is the DIRECT DECODE: the driver alone, with no service, fed the same
//! prompt and then its own top rows, one row a step (`Driver::new`). The service of these tests takes steps of up to
//! four rows (`LIMITS`), so a prompt of more than four tokens is read in pieces and two chats share a step. The
//! windows of the made-up models are 4 and 3 rows, so the rings wrap inside these chats.
use super::*;
use crate::{bounded_decoder::Host,loop_decoder::fixture::Fixture};
use spark_serve::Engine;
use std::{io::{Read,Write},net::{SocketAddr,TcpListener,TcpStream},sync::atomic::{AtomicBool,Ordering}};

/// Tokens of one chat in these tests. The made-up model takes 64.
const CONTEXT:usize=48;
/// Bytes of one chat in the ledger of these tests, and the room for the values of a step and for the chats.
const CHAT_BYTES:usize=1<<20;
const ROOM:usize=64<<20;
/// The limits of a step in these tests: four rows a step, and a small group of values over the keys.
pub(super) const LIMITS:Limits=Limits {step_rows:4,key_bytes:1<<20};
/// Chats at one time in these tests.
const CHATS:usize=2;
/// What a service of these tests is built with: at most `most` chats.
fn seats(most:usize)->Seats {Seats {limits:LIMITS,chat:CHAT_BYTES,room:ROOM,most}}
/// Prompts of token IDs that the head of the small model can write (it writes rows 0 to 19).
const PROMPTS:[&[u32];4]=[&[3,5,2,7,11,13],&[4,8,1,6],&[19,0,17],&[9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9]];

/// The direct decode: the driver alone reads `prompt` and then writes `count` tokens, each its own top row.
fn direct(fixture:&Fixture,prompt:&[u32],count:usize)->Vec<u32> {direct_at(fixture,CONTEXT,prompt,count)}
pub(super) fn direct_at(fixture:&Fixture,context:usize,prompt:&[u32],count:usize)->Vec<u32> {
    let driver=Driver::new(&Host,fixture,&fixture.description,context as u64).unwrap();let mut state=driver.clean().unwrap();
    let mut ids=Vec::new();let mut top=None;
    for &token in prompt {top=Some(driver.step(&mut state,token).unwrap().top);}
    while ids.len()<count {let id=top.expect("a prompt holds a token");ids.push(id);top=Some(driver.step(&mut state,id).unwrap().top);}
    ids
}
/// One server over the small made-up model with the stop set `eos`. It answers `requests` requests and stops.
/// `tokenizer`: a model folder for requests that carry text; without one only token ID prompts are served.
fn served(fixture:&Fixture,eos:Vec<u32>,requests:usize,tokenizer:Option<&spark_serve::tokenizer::ModelTokenizer>,client:impl FnOnce(&(dyn Fn(String)->String+Sync),SocketAddr)) {
    served_at(fixture,CONTEXT,eos,requests,tokenizer,client)
}
/// The same server with a chat of `context` tokens.
fn served_at(fixture:&Fixture,context:usize,eos:Vec<u32>,requests:usize,tokenizer:Option<&spark_serve::tokenizer::ModelTokenizer>,client:impl FnOnce(&(dyn Fn(String)->String+Sync),SocketAddr)) {
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    // A failed check in the client must end the server too, or the test would wait for ever.
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let stop=&stop;let _stop=Stop(stop);
        scope.spawn(move || {
            let service=Service::new(&Host,fixture,&fixture.description,context,eos,seats(CHATS)).unwrap();
            assert_eq!(service.chats(),CHATS);
            let limits=limits(context,24).unwrap();let (cache,config)=service.schedule();
            spark_serve::scheduler::serve_requests(listener,&service,tokenizer,&limits,&cache,&config,stop,"made-up",requests).unwrap();
            assert_eq!((service.fatal_error(),service.live()),(None,0),"the service is not marked failed and every chat is closed");
        });
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();
            if let Err(error)=stream.read_to_string(&mut reply) {reply.push_str(&format!("\nREAD ERROR {error}"));}reply};
        client(&send,address);
    });
}
pub(super) fn post(send:&dyn Fn(String)->String,address:SocketAddr,path:&str,body:&str)->String {
    send(format!("POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()))
}
/// A completion request of token IDs. `fields`: more JSON members, each with its comma in front.
fn ask(send:&dyn Fn(String)->String,address:SocketAddr,prompt:&[u32],max_tokens:usize,fields:&str)->String {
    let prompt=prompt.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    post(send,address,"/v1/completions",&format!(r#"{{"prompt_token_ids":[{prompt}],"stream":true,"max_tokens":{max_tokens}{fields},"stream_options":{{"include_usage":true}}}}"#))
}
/// Every token ID of a streamed reply, in order. A value that is not a whole number fails the test.
pub(super) fn ids(reply:&str)->Vec<u32> {
    reply.split("\"token_id\":").skip(1).map(|part|{let digits=part.chars().take_while(char::is_ascii_digit).collect::<String>();
        digits.parse::<u32>().unwrap_or_else(|_|panic!("a token_id is no whole number: {}",&part[..part.len().min(24)]))}).collect()
}
pub(super) fn ok(reply:&str)->bool {reply.starts_with("HTTP/1.1 200 ")}
pub(super) fn health(send:&dyn Fn(String)->String,address:SocketAddr)->String {send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"))}

/// THE FIRST CHECK: a served answer holds a list of whole-number token IDs that is not empty, and the list equals the
/// direct decode of the same prompt, ID for ID. A request that names no sampling field gives the same list: it is
/// decoded greedy. The same chat three times gives the same list three times.
#[test]
fn a_served_answer_holds_the_token_ids_of_the_direct_decode() {
    let fixture=Fixture::small();
    served(&fixture,vec![1],PROMPTS.len()*4,None,|send,address| {
        for prompt in PROMPTS {
            let expected=direct(&fixture,prompt,10);assert_eq!(expected.len(),10);
            let reply=ask(send,address,prompt,10,r#","temperature":0,"ignore_eos":true"#);assert!(ok(&reply),"{reply}");
            let served=ids(&reply);
            assert!(!served.is_empty(),"the answer holds no token ID: {reply}");
            assert_eq!(served,expected,"prompt {prompt:?}: the served token IDs differ from the direct decode: {reply}");
            let unsampled=ask(send,address,prompt,10,r#","ignore_eos":true"#);
            assert_eq!(ids(&unsampled),expected,"prompt {prompt:?}: a request that names no sampling field is decoded greedy: {unsampled}");
            for run in 0..2 {let again=ask(send,address,prompt,10,r#","temperature":0,"ignore_eos":true"#);assert_eq!(ids(&again),expected,"prompt {prompt:?} run {run}: {again}");}
        }
    });
    // The made-up model does not write one token for every prompt: the lists above are not all the same.
    let lists=PROMPTS.map(|prompt|direct(&fixture,prompt,10));assert!(lists.iter().any(|list|list!=&lists[0]),"{lists:?}");
}
#[test]
fn kept_prompts_restore_exact_replay_and_branch_at_a_committed_boundary() {
    let fixture=Fixture::small();
    served(&fixture,vec![1],6,None,|send,address| {
        let prompt=[3,5,2,7,11,13,4,6,8,10,12];
        let mut branch=prompt;
        branch[9]=19;
        let mut continuation=prompt.to_vec();
        continuation.extend([14,15]);
        for (tokens,cache,reused) in [
            (prompt.as_slice(),false,0),(prompt.as_slice(),true,0),(prompt.as_slice(),true,prompt.len()),
            (branch.as_slice(),true,8),(continuation.as_slice(),true,prompt.len()),(branch.as_slice(),false,0),
        ] {
            let reply=ask(send,address,tokens,8,&format!(r#","temperature":0,"ignore_eos":true,"cache":{cache}"#));
            assert!(ok(&reply) && reply.contains(&format!(r#""cached_tokens":{reused}}}"#)),"{reply}");
            assert_eq!(ids(&reply),direct(&fixture,tokens,8));
        }
    });
}

#[test]
fn kept_prompts_yield_to_live_reservations_without_taking_chat_slots() {
    use spark_serve::{Session,cache::{CachedSession,SessionCache,SessionKey}};
    let fixture=Fixture::small();
    let service=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],seats(1)).unwrap();
    let mut cache=SessionCache::new(service.schedule().0).unwrap();
    let (first,second)=(SessionKey::generate().unwrap(),SessionKey::generate().unwrap());
    let now=Instant::now();
    for (key,&prompt) in [first,second].into_iter().zip(&PROMPTS) {
        let mut live=service.session().unwrap();
        let next_logits=live.prefill(prompt).unwrap();
        let state=CachedSession {checkpoint_lease:None,session:live,tokens:prompt.to_vec(),
            next_logits,reserved_session_bytes:service.chat_bytes()};
        cache.snapshot(key,&state,now);
    }
    assert_eq!((cache.len(),service.live()),(2,0));
    let mut newer=cache.fork_prefix(&second,PROMPTS[1],now).state.unwrap();
    assert_eq!(service.live(),1);
    let room=cache.retained_bytes()-1;
    cache.rebalance_within(room,None,Default::default(),now);
    assert_eq!((cache.len(),cache.counts().evicted),(1,1));
    assert!(cache.fork_prefix(&first,PROMPTS[0],now).state.is_none());
    cache.rebalance_within(0,None,Default::default(),now);
    assert_eq!((cache.retained_bytes(),service.live()),(0,1));
    let next=direct(&fixture,PROMPTS[1],2);
    let logits=newer.session.step(next[0]).unwrap();
    assert_eq!(logits[next[1] as usize],logits.iter().copied().fold(f32::NEG_INFINITY,f32::max));
}
/// The second made-up model (16 visits a token, layers after the loop) through the same service.
#[test]
fn the_wide_model_is_served_with_the_ids_of_its_direct_decode() {
    let fixture=Fixture::wide();let rows=fixture.description.head_rows as u32;
    let prompt=[rows-1,0,rows/2,1];let expected=direct(&fixture,&prompt,8);
    served(&fixture,vec![2],2,None,|send,address| {
        for run in 0..2 {let reply=ask(send,address,&prompt,8,r#","temperature":0,"ignore_eos":true"#);assert!(ok(&reply),"{reply}");assert_eq!(ids(&reply),expected,"run {run}: {reply}");}
    });
}
/// A chat stops at a token of the stop set, and only there. The stop token is one that the direct decode writes
/// after at least two other tokens, so the test knows where the answer must end.
#[test]
fn a_chat_stops_at_a_token_of_the_stop_set() {
    let fixture=Fixture::small();
    let (prompt,decoded,place)=PROMPTS.iter().find_map(|prompt|{let decoded=direct(&fixture,prompt,16);
        (2..decoded.len()).find(|&place|!decoded[..place].contains(&decoded[place])).map(|place|(*prompt,decoded,place))}).expect("one prompt writes a new token after two others");
    let stop=decoded[place];
    served(&fixture,vec![stop],3,None,|send,address| {
        let reply=ask(send,address,prompt,16,r#","temperature":0"#);assert!(ok(&reply),"{reply}");
        assert_eq!(ids(&reply),decoded[..place],"the answer ends before the stop token and holds every token before it: {reply}");
        assert!(reply.contains(r#""finish_reason":"stop""#),"{reply}");
        // With the stop set ignored the same chat goes on to its token limit.
        let ignored=ask(send,address,prompt,16,r#","temperature":0,"ignore_eos":true"#);
        assert_eq!(ids(&ignored),decoded,"{ignored}");assert!(ignored.contains(r#""finish_reason":"length""#),"{ignored}");
        // A cap below the stop place ends the answer by length.
        let capped=ask(send,address,prompt,place-1,r#","temperature":0"#);
        assert_eq!(ids(&capped),decoded[..place-1],"{capped}");assert!(capped.contains(r#""finish_reason":"length""#),"{capped}");
    });
}
/// A prompt that fills the context or is longer than it is refused with the limit in the text, and so is a token
/// ID that the head cannot write. The service answers the next request.
#[test]
fn a_prompt_over_the_context_is_refused_with_the_limit() {
    let fixture=Fixture::small();
    served(&fixture,vec![1],6,None,|send,address| {
        for tokens in [CONTEXT,CONTEXT+1,CONTEXT*4] {
            let reply=ask(send,address,&vec![3;tokens],4,r#","temperature":0"#);
            assert!(reply.starts_with("HTTP/1.1 400 "),"{tokens} tokens: {reply}");
            assert!(reply.contains(&format!("input={tokens};")) && reply.contains(&format!("Available: {} tokens",CONTEXT-1)) && reply.contains(r#""code":"context_length_exceeded""#),
                "{tokens} tokens: the refusal names the prompt length and the limit: {reply}");
            assert!(ids(&reply).is_empty(),"{reply}");
        }
        // Row 20 is a stored embedding row that the head cannot write: it is no token of this service.
        let outside=ask(send,address,&[3,fixture.description.head_rows as u32],4,r#","temperature":0"#);
        assert!(outside.starts_with("HTTP/1.1 400 ") && outside.contains("prompt token outside vocabulary") && ids(&outside).is_empty(),"{outside}");
        assert!(health(send,address).contains("\"status\":\"ready\""));
        // The longest prompt that leaves room for one token is served, and its answer is cut to that room.
        let prompt=vec![3;CONTEXT-1];let reply=ask(send,address,&prompt,4,r#","temperature":0,"ignore_eos":true"#);
        assert!(ok(&reply),"{reply}");assert_eq!(ids(&reply),direct(&fixture,&prompt,1),"{reply}");
    });
}
/// A request with a temperature is answered from the logits of the driver. The same seed gives the same token IDs.
#[test]
fn a_sampled_request_is_served_from_the_logits() {
    let fixture=Fixture::small();let rows=fixture.description.head_rows as u32;
    served(&fixture,vec![1],3,None,|send,address| {
        let fields=r#","temperature":1.5,"seed":7,"ignore_eos":true"#;
        let first=ask(send,address,PROMPTS[0],12,fields);assert!(ok(&first),"{first}");
        let sampled=ids(&first);assert_eq!(sampled.len(),12,"{first}");assert!(sampled.iter().all(|&id|id<rows),"{sampled:?}");
        assert_eq!(ids(&ask(send,address,PROMPTS[0],12,fields)),sampled,"the same seed gives the same token IDs");
        // A temperature of zero on the logits path of another field is still the direct decode.
        let named=ask(send,address,PROMPTS[0],12,r#","temperature":0,"top_p":0.5,"ignore_eos":true"#);
        assert_eq!(ids(&named),direct(&fixture,PROMPTS[0],12),"{named}");
    });
}
/// A request that asks for drafting is refused with a clear text, and the service goes on.
#[test]
fn a_request_for_drafting_is_refused() {
    let fixture=Fixture::small();
    served(&fixture,vec![1],3,None,|send,address| {
        for (fields,reason) in [(r#","temperature":0,"draft":"strict""#,"drafting is not built for this model family"),(r#","temperature":0,"mtp_depth":2"#,"use mtp_depth: 0")] {
            let reply=ask(send,address,PROMPTS[0],4,fields);
            assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains(reason) && ids(&reply).is_empty(),"{fields}: {reply}");
        }
        let reply=ask(send,address,PROMPTS[0],4,r#","temperature":0,"draft":"plain","ignore_eos":true"#);
        assert_eq!(ids(&reply),direct(&fixture,PROMPTS[0],4),"{reply}");
    });
}
/// Requests at the same time: the service runs two chats at one time and a third request waits for a free chat.
/// Each gets the token IDs of its own direct decode, whatever chats share its steps. Three rounds, so the chats
/// meet at other places of their answers.
#[test]
fn requests_at_one_time_share_steps_and_each_gets_its_own_ids() {
    let fixture=Fixture::small();let expected=PROMPTS.map(|prompt|direct(&fixture,prompt,12));
    served(&fixture,vec![1],PROMPTS.len()*3,None,|send,address| {
        for round in 0..3 {
            let replies=std::thread::scope(|scope| {
                let workers=PROMPTS.map(|prompt|scope.spawn(move ||ask(send,address,prompt,12,r#","temperature":0,"ignore_eos":true"#)));
                workers.map(|worker|worker.join().unwrap())
            });
            for (reply,expected) in replies.iter().zip(&expected) {assert!(ok(reply),"round {round}: {reply}");assert_eq!(&ids(reply),expected,"round {round}: {reply}");}
        }
    });
}
/// The batch call of the engine itself, with no scheduler: rows of several chats in one call are packed into
/// steps of at most the step row limit, and every answer is the answer of the chat alone.
#[test]
fn a_batch_of_rows_of_several_chats_gives_each_chat_its_own_answer() {
    use spark_serve::Session;
    let fixture=Fixture::small();let d=&fixture.description;
    let service=Service::new(&Host,&fixture,d,CONTEXT,vec![1],seats(3)).unwrap();
    assert_eq!((service.chats(),service.limits(),service.shared_decode_rows()),(3,LIMITS,3));
    let expected=[PROMPTS[0],PROMPTS[1],PROMPTS[3]].map(|prompt|direct(&fixture,prompt,10));
    let mut chats=[service.session().unwrap(),service.session().unwrap(),service.session().unwrap()];
    assert!(service.session().err().unwrap().contains("every chat of this service is in use"));
    // The prompts in one call: 6, 4 and 19 tokens are 29 rows, so the call takes 8 steps of at most 4 rows, and a
    // prompt goes on in the next step. The first chat asks for the top row after its prompt, the second for the
    // logits, the third for nothing but its last token: that one is left for the call below.
    let last=PROMPTS[3].len()-1;
    let answers={
        let [a,b,c]=&mut chats;
        service.step_batch(&mut [BatchRows {session:a,tokens:PROMPTS[0],output:BatchOutput::Last,greedy:true,top_k:0},
            BatchRows {session:b,tokens:PROMPTS[1],output:BatchOutput::Last,greedy:false,top_k:0},
            BatchRows {session:c,tokens:&PROMPTS[3][..last],output:BatchOutput::None,greedy:true,top_k:0}]).map_err(|error|error.message).unwrap()
    };
    let top=|logits:&[f32]|crate::bounded_decoder::greedy_logits(logits,logits.len()).unwrap();
    let mut next=[0u32;3];
    match &answers[..] {
        [Predictions::Greedy(first),Predictions::Logits(second),Predictions::Logits(third)]=>{
            assert_eq!(first,&expected[0][..1]);assert_eq!(second.len(),1);assert_eq!(second[0].len(),d.head_rows as usize);assert_eq!(top(&second[0]),expected[1][0]);
            assert!(third.is_empty(),"a row that wants no answer gets none");next[0]=first[0];next[1]=expected[1][0];
        }
        _=>panic!("the answers have another form than the rows asked for"),
    }
    // The last prompt token of the third chat with every row answered, beside one new token of the two others.
    let answers={
        let [a,b,c]=&mut chats;
        service.step_batch(&mut [BatchRows {session:a,tokens:&next[..1],output:BatchOutput::Last,greedy:true,top_k:0},
            BatchRows {session:b,tokens:&next[1..2],output:BatchOutput::All,greedy:true,top_k:0},
            BatchRows {session:c,tokens:&PROMPTS[3][last..],output:BatchOutput::All,greedy:false,top_k:0}]).map_err(|error|error.message).unwrap()
    };
    match &answers[..] {
        [Predictions::Greedy(first),Predictions::Greedy(second),Predictions::Logits(third)]=>{
            assert_eq!((first,second),(&expected[0][1..2].to_vec(),&expected[1][1..2].to_vec()));assert_eq!(top(&third[0]),expected[2][0]);
            next=[first[0],second[0],expected[2][0]];
        }
        _=>panic!("the answers have another form than the rows asked for"),
    }
    // New tokens of the three chats, one step a call, to the end of the direct decodes.
    for place in 2..8 {
        let [a,b,c]=&mut chats;
        let answers=service.step_batch(&mut [BatchRows {session:a,tokens:&next[..1],output:BatchOutput::Last,greedy:true,top_k:0},
            BatchRows {session:b,tokens:&next[1..2],output:BatchOutput::Last,greedy:true,top_k:0},
            BatchRows {session:c,tokens:&next[2..],output:BatchOutput::Last,greedy:true,top_k:0}]).map_err(|error|error.message).unwrap();
        let ids:Vec<u32>=answers.iter().map(|answer|match answer {Predictions::Greedy(ids) if ids.len()==1=>ids[0],_=>panic!("a greedy row of one token gets one top row")}).collect();
        assert_eq!(ids,[expected[0][place],expected[1][place],expected[2][place-1]],"place {place}");
        next=[ids[0],ids[1],ids[2]];
    }
    // The third chat waits while the two others take a step: at the end of that call it gets rings of its own,
    // one time. A second call without it copies nothing more. Then it goes on alone with its own IDs.
    assert_eq!(service.detached(),0,"no chat waited so far: every chat was in every call");
    assert_eq!(service.token_steps(),0,"a step of several chats runs the pass of rows");
    for (place,(&first,&second)) in expected[0].iter().zip(&expected[1]).enumerate().skip(8) {
        let [a,b,_]=&mut chats;
        let answers=service.step_batch(&mut [BatchRows {session:a,tokens:&next[..1],output:BatchOutput::Last,greedy:true,top_k:0},
            BatchRows {session:b,tokens:&next[1..2],output:BatchOutput::Last,greedy:true,top_k:0}]).map_err(|error|error.message).unwrap();
        let ids:Vec<u32>=answers.iter().map(|answer|match answer {Predictions::Greedy(ids) if ids.len()==1=>ids[0],_=>panic!("a greedy row of one token gets one top row")}).collect();
        assert_eq!(ids,[first,second],"place {place}");
        assert_eq!(service.detached(),1,"place {place}: the waiting chat is detached once");
        next[0]=ids[0];next[1]=ids[1];
    }
    {
        let c=&mut chats[2];
        let answers=service.step_batch(&mut [BatchRows {session:c,tokens:&next[2..],output:BatchOutput::Last,greedy:true,top_k:0}]).map_err(|error|error.message).unwrap();
        assert!(matches!(&answers[..],[Predictions::Greedy(ids)] if ids[..]==expected[2][7..8]),"the chat that waited goes on with its own IDs");
        // Its rings are its own since it was detached: one new token of one chat alone runs the token pass, with the
        // words of the pass of rows (the IDs of the direct decode).
        assert_eq!(service.token_steps(),1,"one token of one chat alone runs the token pass");
        let mut id=expected[2][7];
        for place in 8..10 {
            // The scheduler asks a decode row for `All`.
            let answers=service.step_batch(&mut [BatchRows {session:&mut *c,tokens:&[id],output:BatchOutput::All,greedy:true,top_k:0}]).map_err(|error|error.message).unwrap();
            let [Predictions::Greedy(ids)]=&answers[..] else {panic!("a greedy row of one token gets one top row")};
            assert_eq!(ids[..],expected[2][place..place+1],"place {place}");id=ids[0];
        }
        assert_eq!(service.token_steps(),3);
    }
    // A chat alone in pieces: a prompt of 19 tokens through `prefill` is 5 steps, and its logits are those of the direct decode.
    drop(chats);assert_eq!(service.live(),0);
    let mut alone=service.session().unwrap();
    assert_eq!(alone.prefill_chunk_size(PROMPTS[3].len()),LIMITS.step_rows);
    assert_eq!(top(&alone.prefill(PROMPTS[3]).unwrap()),expected[2][0]);
    assert!(alone.prefill(&[]).is_err());
    // A row with no token that asks for an answer is refused before any step.
    let refused=service.step_batch(&mut [BatchRows {session:&mut alone,tokens:&[],output:BatchOutput::Last,greedy:true,top_k:0}]).err().unwrap();
    assert!(refused.message.contains("a row with no token asks for an answer"),"{}",refused.message);
}
#[test]
fn retiring_draft_history_returns_the_entire_cohort_in_one_batch_step() {
    use spark_serve::Session;
    use super::super::draft::DraftChat;
    struct DraftModel;
    impl mtp::Model<Host,Fixture> for DraftModel {
        fn config(&self)->&engine_model::mtp_draft::MtpConfig {panic!("plain batches need no taps")}
        fn chat(&self)->DraftChat<Vec<f32>> {DraftChat::new(0,CONTEXT)}
        fn push(&self,_:&Host,_:&mut DraftChat<Vec<f32>>,_:&[Vec<f32>],_:usize)->Result<()> {panic!("plain batches need no history")}
        fn propose(&self,_:&Host,_:&Fixture,_:&DraftChat<Vec<f32>>,_:u32,_:&Vec<f32>,_:usize)->Result<Vec<u32>> {panic!("plain batches need no proposals")}
    }
    let fixture=Fixture::small();
    let seats=Seats {limits:Limits {step_rows:16,..LIMITS},..seats(10)};
    let mut service=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],seats).unwrap();
    service.model=Some(&DraftModel);service.depth=7;
    assert_eq!(service.draft_chat_limit(),8);
    assert_eq!(service.contended_draft_chats(),8);
    assert!(service.discard_contended_draft());
    let mut limits=limits(CONTEXT,24).unwrap();limits.next_mtp_depth=Some(7);
    let request=spark_serve::protocol::parse(br#"{"prompt_token_ids":[3],"max_tokens":4,"temperature":0,"draft":"auto"}"#,false,&limits).unwrap();
    let strict=spark_serve::protocol::parse(br#"{"prompt_token_ids":[3],"max_tokens":4,"temperature":0,"draft":"strict"}"#,false,&limits).unwrap();
    let plain=spark_serve::protocol::parse(br#"{"prompt_token_ids":[3],"max_tokens":4,"temperature":0,"draft":"plain"}"#,false,&limits).unwrap();
    assert!(service.permits_prefix_cache(&plain) && service.permits_prefix_cache(&request) && !service.permits_prefix_cache(&strict));
    // A chat restored from a cached prefix (`position > 0`) must not attach a fresh draft state at position 0.
    {
        let mut warm=service.session().unwrap();
        warm.advance(PROMPTS[0][0]).unwrap();
        warm.configure_request(&request).unwrap();
        assert!(warm.draft_target().is_none());
    }
    for width in [2,4,8,10] {
        let mut chats=Vec::new();
        for _ in 0..width {
            let mut chat=service.session().unwrap();
            chat.configure_request(&request).unwrap();
            assert!(chat.draft_target().is_some());
            // The scheduler retires the first chat on overlap and disables draft state on admission for the rest.
            chat.disable_model_draft();
            chat.disable_model_draft();
            assert!(chat.draft_over());
            assert!(!chat.draft_over());
            chats.push(chat);
        }
        let mut rows:Vec<_>=chats.iter_mut().enumerate().map(|(i,session)|BatchRows {
            session,tokens:&PROMPTS[i%PROMPTS.len()][..1],output:BatchOutput::All,greedy:true,top_k:0,
        }).collect();
        let before=service.batch_steps.get();
        let answers=service.step_batch(&mut rows).map_err(|error|error.message).unwrap();
        assert_eq!(service.batch_steps.get()-before,1,"{width} chats");
        assert_eq!(answers.len(),width);
        for (i,answer) in answers.iter().enumerate() {
            assert!(matches!(answer,Predictions::Greedy(ids) if *ids==direct(&fixture,&PROMPTS[i%PROMPTS.len()][..1],1)));
        }
    }
}
#[test]
fn driver_verify_many_matches_direct_decode_across_mixed_accept_lengths() {
    let fixture=Fixture::small();
    let driver=Driver::with_limits(&Host,&fixture,&fixture.description,CONTEXT as u64,Limits {step_rows:32,key_bytes:1<<20}).unwrap();
    let expected=PROMPTS.map(|p|direct(&fixture,p,12));
    let mut states=[driver.clean().unwrap(),driver.clean().unwrap(),driver.clean().unwrap(),driver.clean().unwrap()];
    let taps=[0u64,fixture.description.visits.len() as u64];
    // Prefill all 4 chats with `step_rows_taps` and grab initial top tokens.
    let mut next=[0u32;4];
    {
        let [s0,s1,s2,s3]=&mut states;
        let mut steps=[
            ChatStep {state:s0,tokens:PROMPTS[0],wanted:Wanted::Last,logits:false},
            ChatStep {state:s1,tokens:PROMPTS[1],wanted:Wanted::Last,logits:false},
            ChatStep {state:s2,tokens:PROMPTS[2],wanted:Wanted::Last,logits:false},
            ChatStep {state:s3,tokens:PROMPTS[3],wanted:Wanted::Last,logits:false},
        ];
        let (res,got_taps)=driver.step_rows_taps(&mut steps,&taps).unwrap();
        assert_eq!(got_taps.len(),taps.len());
        next.copy_from_slice(&res.tops);
    }
    let mut emitted:[Vec<u32>;4]=std::array::from_fn(|i|vec![next[i]]);
    // Run mixed multi-chat verify rounds: chat 0 full accept (3 rows), chat 1 partial accept (3 rows, keep 2),
    // chat 2 reject-all (3 rows, keep 1), chat 3 single-row peer (1 row, keep 1).
    let mut step_idx=[1usize;4];
    while step_idx.iter().any(|&k|k<10) {
        let mut toks:[Vec<u32>;4]=std::array::from_fn(|i|vec![next[i]]);
        // Chat 0: 2 correct drafts (3 rows total) -> accepts all 3.
        if step_idx[0]+2<expected[0].len() {toks[0].extend_from_slice(&expected[0][step_idx[0]..step_idx[0]+2]);}
        // Chat 1: 1 correct + 1 wrong draft (3 rows total) -> accepts 2, rolls back 1.
        if step_idx[1]+2<expected[1].len() {
            toks[1].push(expected[1][step_idx[1]]);
            toks[1].push((expected[1][step_idx[1]+1]+1)%fixture.description.head_rows as u32);
        }
        // Chat 2: 2 wrong drafts (3 rows total) -> accepts 1, rolls back 2.
        if step_idx[2]+2<expected[2].len() {
            toks[2].push((expected[2][step_idx[2]]+1)%fixture.description.head_rows as u32);
            toks[2].push((expected[2][step_idx[2]+1]+2)%fixture.description.head_rows as u32);
        }
        // Chat 3: 1-row peer riding inside verify_many.
        let [s0,s1,s2,s3]=&mut states;
        let mut pair=[(s0,toks[0].as_slice()),(s1,toks[1].as_slice()),(s2,toks[2].as_slice()),(s3,toks[3].as_slice())];
        let (tops,got_taps)=driver.verify_many(&mut pair,&taps).unwrap();
        let total:usize=toks.iter().map(Vec::len).sum();
        assert_eq!(tops.len(),total);
        assert_eq!(got_taps[0].len(),total*fixture.description.hidden as usize);
        let mut off=0usize;
        for i in 0..4 {
            let n=toks[i].len();
            let row_tops=&tops[off..off+n];
            off+=n;
            let mut keep=1usize;
            while keep<n && row_tops[keep-1]==toks[i][keep] {keep+=1;}
            driver.accept(&mut states[i],n,keep).unwrap();
            for &t in &toks[i][1..keep] {emitted[i].push(t);step_idx[i]+=1;}
            next[i]=row_tops[keep-1];
            emitted[i].push(next[i]);
            step_idx[i]+=1;
        }
    }
    for i in 0..4 {
        assert_eq!(&emitted[i][..10],&expected[i][..10],"chat {i} verify_many drifted from direct decode");
    }
}
#[test]
fn multi_chat_auto_and_strict_drafting_matches_direct_decode() {
    use super::super::draft::DraftChat;
    struct MockDraft {cfg:engine_model::mtp_draft::MtpConfig,proposals:std::sync::atomic::AtomicUsize}
    impl mtp::Model<Host,Fixture> for MockDraft {
        fn config(&self)->&engine_model::mtp_draft::MtpConfig {&self.cfg}
        fn chat(&self)->DraftChat<Vec<f32>> {DraftChat::new(0,CONTEXT)}
        fn push(&self,_:&Host,chat:&mut DraftChat<Vec<f32>>,taps:&[Vec<f32>],rows:usize)->Result<()> {
            assert_eq!(taps.len(),self.cfg.target_layer_ids.len());
            assert!(taps.iter().all(|t|t.len()==rows*self.cfg.hidden_size));
            chat.advance_len(rows);
            Ok(())
        }
        fn push_batch(&self,c:&Host,items:&mut [(&mut DraftChat<Vec<f32>>,&[Vec<f32>],usize)])->Result<()> {
            for (chat,taps,rows) in items {self.push(c,chat,taps,*rows)?;}
            Ok(())
        }
        fn propose(&self,_:&Host,_:&Fixture,_:&DraftChat<Vec<f32>>,anchor:u32,_:&Vec<f32>,depth:usize)->Result<Vec<u32>> {
            self.proposals.fetch_add(1,Ordering::Relaxed);
            Ok((0..depth).map(|k|(anchor+1+k as u32)%20).collect())
        }
        fn propose_confident_batch(&self,_:&Host,_:&Fixture,items:&mut [(&mut DraftChat<Vec<f32>>,u32,&Vec<f32>,usize)])->Result<Vec<(Vec<u32>,Vec<f32>)>> {
            self.proposals.fetch_add(items.len(),Ordering::Relaxed);
            Ok(items.iter().map(|(_,anchor,_,depth)| {
                let ids=(0..*depth).map(|k|(anchor+1+k as u32)%20).collect();
                let probs=vec![0.96;*depth];
                (ids,probs)
            }).collect())
        }
    }
    let fixture=Fixture::small();
    let cfg=engine_model::mtp_draft::MtpConfig {
        hidden_size:fixture.description.hidden as usize,intermediate_size:16,num_layers:0,num_heads:2,num_kv_heads:2,
        head_dim:4,sliding_window_size:4,block_size:8,markov_rank:4,vocab_size:fixture.description.vocab as usize,
        unpadded_vocab_size:fixture.description.head_rows as usize,max_position_embeddings:CONTEXT,
        rms_norm_eps:1e-6,rope_theta:10000.0,final_logit_softcapping:30.0,local_layer_ids:Vec::new(),
        target_layer_ids:vec![0,fixture.description.visits.len()],
    };
    let model=MockDraft {cfg,proposals:std::sync::atomic::AtomicUsize::new(0)};
    let expected=PROMPTS.map(|p|direct(&fixture,p,10));
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let stop=&stop;let _stop=Stop(stop);
        let model_ref=&model;let fixture_ref=&fixture;
        scope.spawn(move || {
            let s=Seats {limits:Limits {step_rows:16,key_bytes:1<<20},chat:CHAT_BYTES,room:ROOM,most:8};
            let mut service=Service::new(&Host,fixture_ref,&fixture_ref.description,CONTEXT,vec![1],s).unwrap();
            service.model=Some(model_ref);service.depth=7;
            let mut lim=limits(CONTEXT,24).unwrap();lim.next_mtp_depth=Some(7);
            let (cache,config)=service.schedule();
            spark_serve::scheduler::serve_requests(listener,&service,None::<&spark_serve::tokenizer::ModelTokenizer>,&lim,&cache,&config,stop,"made-up",PROMPTS.len()*3).unwrap();
            assert_eq!((service.fatal_error(),service.live()),(None,0));
        });
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();
            if let Err(error)=stream.read_to_string(&mut reply) {reply.push_str(&format!("\nREAD ERROR {error}"));}reply};
        for mode in ["plain","auto","strict"] {
            let replies=std::thread::scope(|inner| {
                let workers=PROMPTS.map(|prompt|inner.spawn(move ||ask(&send,address,prompt,10,&format!(r#","temperature":0,"draft":"{mode}","ignore_eos":true"#))));
                workers.map(|w|w.join().unwrap())
            });
            for (reply,want) in replies.iter().zip(&expected) {
                assert!(ok(reply),"mode {mode}: {reply}");
                assert_eq!(&ids(reply),want,"mode {mode}: served IDs differ from direct decode: {reply}");
            }
        }
    });
    assert!(model.proposals.load(Ordering::Relaxed)>0,"multi-chat batched drafter never ran");
}
#[test]
fn drafted_prefix_cache_restores_draft_state_and_matches_direct_decode() {
    use super::super::draft::DraftChat;
    use spark_serve::Session;
    struct MockDraft {cfg:engine_model::mtp_draft::MtpConfig,proposals:std::sync::atomic::AtomicUsize}
    impl mtp::Model<Host,Fixture> for MockDraft {
        fn config(&self)->&engine_model::mtp_draft::MtpConfig {&self.cfg}
        fn chat(&self)->DraftChat<Vec<f32>> {DraftChat::new(0,CONTEXT)}
        fn push(&self,_:&Host,chat:&mut DraftChat<Vec<f32>>,taps:&[Vec<f32>],rows:usize)->Result<()> {
            assert_eq!(taps.len(),self.cfg.target_layer_ids.len());
            assert!(taps.iter().all(|t|t.len()==rows*self.cfg.hidden_size));
            chat.advance_len(rows);
            Ok(())
        }
        fn propose(&self,_:&Host,_:&Fixture,chat:&DraftChat<Vec<f32>>,anchor:u32,_:&Vec<f32>,depth:usize)->Result<Vec<u32>> {
            assert!(!chat.is_empty(),"restored draft chat must carry committed prefix length");
            self.proposals.fetch_add(1,Ordering::Relaxed);
            Ok((0..depth).map(|k|(anchor+1+k as u32)%20).collect())
        }
        fn propose_confident(&self,c:&Host,w:&Fixture,chat:&mut DraftChat<Vec<f32>>,anchor:u32,embed:&Vec<f32>,depth:usize,_:f32)->Result<(Vec<u32>,Vec<f32>)> {
            let ids=self.propose(c,w,chat,anchor,embed,depth)?;
            let probs=vec![0.96;ids.len()];
            Ok((ids,probs))
        }
    }
    let fixture=Fixture::small();
    let cfg=engine_model::mtp_draft::MtpConfig {
        hidden_size:fixture.description.hidden as usize,intermediate_size:16,num_layers:0,num_heads:2,num_kv_heads:2,
        head_dim:4,sliding_window_size:4,block_size:8,markov_rank:4,vocab_size:fixture.description.vocab as usize,
        unpadded_vocab_size:fixture.description.head_rows as usize,max_position_embeddings:CONTEXT,
        rms_norm_eps:1e-6,rope_theta:10000.0,final_logit_softcapping:30.0,local_layer_ids:Vec::new(),
        target_layer_ids:vec![0,fixture.description.visits.len()],
    };
    {
        let model=MockDraft {cfg:cfg.clone(),proposals:std::sync::atomic::AtomicUsize::new(0)};
        let mut service=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],seats(2)).unwrap();
        service.model=Some(&model);service.depth=7;
        let mut lim=limits(CONTEXT,24).unwrap();lim.next_mtp_depth=Some(7);
        let auto_req=spark_serve::protocol::parse(br#"{"prompt_token_ids":[3],"max_tokens":4,"temperature":0}"#,false,&lim).unwrap();
        let plain_req=spark_serve::protocol::parse(br#"{"prompt_token_ids":[3],"max_tokens":4,"temperature":0,"draft":"plain"}"#,false,&lim).unwrap();
        let mut chat=service.session().unwrap();
        chat.configure_request(&auto_req).unwrap();
        chat.prefill(PROMPTS[0]).unwrap();
        assert!(chat.can_fork(),"drafted chat is forkable right after prefill");
        let page=Host.plane_page_bytes();
        let target_only=state::prefix_bytes::<Vec<f32>>(service.driver.layouts(),PROMPTS[0].len() as u64,page);
        let draft_only=DraftChat::<Vec<f32>>::prefix_bytes(&cfg,PROMPTS[0].len(),CONTEXT,page);
        assert_eq!(chat.retained_bytes(),Some(target_only+draft_only),"retained_bytes includes draft prefix");
        let _=chat.step(PROMPTS[1][0]).unwrap();
        assert!(!chat.can_fork(),"lagging draft chat cannot fork before settle_prefix");
        chat.settle_prefix().unwrap();
        assert!(chat.can_fork(),"settle_prefix catches up lagged draft taps");
        let kept=chat.fork().unwrap();
        assert_eq!(kept.retained_bytes(),chat.retained_bytes(),"kept snapshot accounts exact same bytes");
        drop(chat);
        let mut warm_draft=kept.fork().unwrap();
        warm_draft.configure_request(&auto_req).unwrap();
        assert!(warm_draft.draft_target().is_some(),"restored drafted snapshot enables drafting");
        warm_draft.prefill(&[PROMPTS[1][1]]).unwrap();
        assert!(warm_draft.can_fork(),"1-token continuation prefill flushes draft immediately");
        drop(warm_draft);
        let mut warm_plain=kept.fork().unwrap();
        warm_plain.configure_request(&plain_req).unwrap();
        assert!(warm_plain.draft_target().is_none(),"plain request drops restored draft state");
    }
    let model=MockDraft {cfg,proposals:std::sync::atomic::AtomicUsize::new(0)};
    let turn1:Vec<u32>=vec![3,5,2,7,11,13,4,6,8,10,12];
    let mut turn2=turn1.clone();turn2.extend([14,15,16]);
    let mut turn3=turn2.clone();turn3.extend([17]);
    let exp1=direct(&fixture,&turn1,10);
    let exp2=direct(&fixture,&turn2,10);
    let exp3=direct(&fixture,&turn3,10);
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let stop=&stop;let _stop=Stop(stop);
        let model_ref=&model;let fixture_ref=&fixture;
        scope.spawn(move || {
            let s=Seats {limits:Limits {step_rows:4,key_bytes:1<<20},chat:CHAT_BYTES,room:ROOM,most:2};
            let mut service=Service::new(&Host,fixture_ref,&fixture_ref.description,CONTEXT,vec![1],s).unwrap();
            service.model=Some(model_ref);service.depth=7;
            let mut lim=limits(CONTEXT,24).unwrap();lim.next_mtp_depth=Some(7);
            let (cache,config)=service.schedule();
            spark_serve::scheduler::serve_requests(listener,&service,None::<&spark_serve::tokenizer::ModelTokenizer>,&lim,&cache,&config,stop,"made-up",5).unwrap();
            assert_eq!((service.fatal_error(),service.live()),(None,0));
        });
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(120))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();
            if let Err(error)=stream.read_to_string(&mut reply) {reply.push_str(&format!("\nREAD ERROR {error}"));}reply};
        let r1=ask(&send,address,&turn1,10,r#","temperature":0,"ignore_eos":true"#);
        assert!(ok(&r1) && r1.contains(r#""cached_tokens":0}"#),"{r1}");
        assert_eq!(ids(&r1),exp1);
        let p_after_t1=model.proposals.load(Ordering::Relaxed);
        assert!(p_after_t1>0,"turn 1 must draft");
        let r2=ask(&send,address,&turn2,10,r#","temperature":0,"ignore_eos":true"#);
        assert!(ok(&r2) && r2.contains(&format!(r#""cached_tokens":{}}}"#,turn1.len())),"{r2}");
        assert_eq!(ids(&r2),exp2);
        let p_after_t2=model.proposals.load(Ordering::Relaxed);
        assert!(p_after_t2>p_after_t1,"turn 2 must keep drafting after cache restore");
        let r3=ask(&send,address,&turn3,10,r#","temperature":0,"ignore_eos":true"#);
        assert!(ok(&r3) && r3.contains(&format!(r#""cached_tokens":{}}}"#,turn2.len())),"{r3}");
        assert_eq!(ids(&r3),exp3);
        let p_after_t3=model.proposals.load(Ordering::Relaxed);
        assert!(p_after_t3>p_after_t2,"turn 3 must keep drafting after 1-token suffix cache restore");
        let r2_plain=ask(&send,address,&turn2,10,r#","temperature":0,"draft":"plain","ignore_eos":true"#);
        assert!(ok(&r2_plain) && r2_plain.contains(&format!(r#""cached_tokens":{}}}"#,turn2.len())),"{r2_plain}");
        assert_eq!(ids(&r2_plain),exp2);
        let r2_cold=ask(&send,address,&turn2,10,r#","temperature":0,"cache":false,"ignore_eos":true"#);
        assert!(ok(&r2_cold) && r2_cold.contains(r#""cached_tokens":0}"#),"{r2_cold}");
        assert_eq!(ids(&r2_cold),exp2);
    });
}
#[test]
fn a_draft_failure_keeps_plain_peers_batched_and_their_answers_in_order() {
    let fixture=Fixture::small();
    let service=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],seats(3)).unwrap();
    let (first,last)=(&PROMPTS[0][..1],&PROMPTS[1][..1]);
    for prefill in [true,false] {
        let (mut a,mut draft,mut b)=(service.session().unwrap(),service.session().unwrap(),service.session().unwrap());
        // Inject a draft context without its model so the tapped call fails.
        draft.draft=Some(mtp::Context::new(super::super::draft::DraftChat::new(0,CONTEXT),1));
        let mut rows=[
            BatchRows {session:&mut a,tokens:first,output:BatchOutput::All,greedy:true,top_k:0},
            BatchRows {session:&mut draft,tokens:first,output:BatchOutput::All,greedy:true,top_k:0},
            BatchRows {session:&mut b,tokens:last,output:BatchOutput::All,greedy:true,top_k:0},
        ];
        let answers=if prefill {service.prefill_batch(&mut rows)} else {service.step_batch(&mut rows)};
        let answers=answers.map_err(|error|error.message).unwrap();
        let [Predictions::Greedy(left),Predictions::Rejected(message),Predictions::Greedy(right)]=answers.as_slice() else {
            panic!("plain results and the draft failure must stay in request order");
        };
        assert_eq!(left,&direct(&fixture,first,1));
        assert_eq!(right,&direct(&fixture,last,1));
        assert_eq!(message,"draft model missing");
        assert_eq!((a.position(),draft.position(),b.position()),(1,0,1));
        let slots=service.slots.borrow();
        for peer in [&a,&b] {
            assert!(slots[peer.active_slot().unwrap()].as_ref().unwrap().shared);
        }
    }
}
/// Deferred model drafting must preserve a solo chat's full `15`-token (`M=16`) verify window
/// for wide neural drafting and suffix extension, without truncating it to `gamma = 7`.
#[test]
fn defer_model_draft_preserves_wide_solo_limit_above_base_gamma() {
    use spark_engine::draft::Target;
    let fixture=Fixture::small();
    let service=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],seats(1)).unwrap();
    let mut chat=service.session().unwrap();
    assert_eq!(chat.defer_model_draft(15),None,"plain session never defers model drafting");
    chat.draft=Some(mtp::Context::new(super::super::draft::DraftChat::new(0,CONTEXT),7));
    assert_eq!(chat.defer_model_draft(15),Some(15),"solo chat keeps full 15-token wide/extension window");
    assert_eq!(chat.defer_model_draft(7),Some(7),"C2 cohort cap passes through unchanged");
    assert_eq!(chat.defer_model_draft(3),Some(3),"C4 cohort cap passes through unchanged");
    assert_eq!(chat.defer_model_draft(1),Some(1),"C8 cohort cap passes through unchanged");
    assert_eq!(chat.defer_model_draft(32),Some(15),"defer limit clamps to MAX_POS=15");
}
/// The engine itself: its count of chats, the vocabulary is the rows that the head can write, the stop set is the
/// given list, and a chat that ends frees its place.
#[test]
fn the_engine_holds_its_chats_and_reports_its_limits() {
    use spark_serve::Session;
    let fixture=Fixture::small();let d=&fixture.description;
    let service=Service::new(&Host,&fixture,d,CONTEXT,vec![1,7],seats(1)).unwrap();
    assert_eq!((service.context_limit(),service.vocab_size(),service.eos_token_ids()),(CONTEXT,d.head_rows as usize,&[1u32,7][..]));
    let min_chat=service.chat_bytes();
    assert!(min_chat<CHAT_BYTES,"paged straight planes reserve less than full context at 1 token");
    assert_eq!((service.session_reservation_bytes(),service.request_reservation_bytes(1),service.request_reservation_bytes(CONTEXT)),(Some(min_chat),Some(min_chat),Some(CHAT_BYTES)));
    assert_eq!((service.request_reservation_bytes(0),service.request_reservation_bytes(CONTEXT+1)),(None,None));
    let mut chat=service.session().unwrap();assert_eq!(service.live(),1);
    assert!(service.session().err().unwrap().contains("every chat of this service is in use"));
    assert!(!chat.can_fork() && chat.fork().is_err() && chat.truncate(0).is_err());
    // A step gives the logits of the rows that the head can write, and their top row is the driver's.
    let expected=direct(&fixture,PROMPTS[0],1);chat.prefill_advance(&PROMPTS[0][..PROMPTS[0].len()-1]).unwrap();
    let logits=chat.step(*PROMPTS[0].last().unwrap()).unwrap();assert_eq!(logits.len(),d.head_rows as usize);
    assert_eq!(crate::bounded_decoder::greedy_logits(&logits,logits.len()).unwrap(),expected[0]);
    drop(chat);assert_eq!(service.live(),0);assert!(service.session().is_ok());
    // What a start must not give.
    let refused=|context,eos:Vec<u32>|Service::new(&Host,&fixture,d,context,eos,seats(1)).err().unwrap();
    assert!(refused(0,vec![1]).contains("the context is zero or above"));assert!(refused(d.max_positions as usize+1,vec![1]).contains("the context is zero or above"));
    assert!(refused(CONTEXT,vec![]).contains("the stop set is empty"));
    assert!(refused(CONTEXT,vec![d.head_rows as u32]).contains("outside the rows that the head can write"));
    assert!(refused(CONTEXT,vec![1,1]).contains("names one token twice"));
    // The count of chats is at least one and at most the step row limit, and the ledger must hold the state of a chat.
    let built=|chat,room,most|Service::new(&Host,&fixture,d,CONTEXT,vec![1],Seats {limits:LIMITS,chat,room,most});
    let counted=|chat,most|built(chat,ROOM,most).err().unwrap();
    assert!(counted(CHAT_BYTES,0).contains("the count of chats is zero or above the step row limit"));
    assert!(counted(CHAT_BYTES,LIMITS.step_rows+1).contains("the count of chats is zero or above the step row limit"));
    assert!(counted(service.state_bytes()-1,1).contains("the state of one chat is larger than the bytes that the ledger reserves"));
    assert_eq!(built(service.state_bytes(),ROOM,LIMITS.step_rows).unwrap().chats(),LIMITS.step_rows);
    // The scheduler uses the same pool for live and retained state.
    let (cache,config)=service.schedule();
    assert_eq!((config.active,config.prefill_chunk,config.mtp_depth,config.prefix_cache,config.active_bytes,cache.retained_bytes,cache.tokens_per_entry),(1,LIMITS.step_rows,0,true,service.pool(),service.pool(),CONTEXT));
    assert!(matches!(config.draft,DraftMode::Plain));
    // Prompts that wait at one time are read together, in steps of the step row limit; decode stays per chat.
    assert_eq!((service.shared_prefill_batch(),service.shared_batch(),service.shared_prefill_limit(),service.shared_prefill_rows()),(true,false,LIMITS.step_rows,LIMITS.step_rows));
    // The room: the driver's bound of the values of one step comes off once, and the chats that fit in the rest
    // are the chats of the service. A chat of a service of several chats reserves one more set of rings for each
    // other chat.
    let step=service.step_bytes();
    assert!(step>0 && step<ROOM);assert_eq!((service.pool(),service.chat_bytes()),(ROOM-step,min_chat));
    let wide=built(CHAT_BYTES,ROOM,3).unwrap();
    assert_eq!((wide.schedule().1.active,wide.shared_decode_rows(),wide.chats()),(3,3,3));
    // The step of a service of more chats holds more: the rings of every chat wait in it.
    assert!(wide.step_bytes()>step && wide.pool()==ROOM-wide.step_bytes(),"{} {step}",wide.step_bytes());
    let rings=wide.chat_bytes()-min_chat;
    assert!(rings>0 && wide.chat_bytes()==min_chat+rings,"{}",wide.chat_bytes());
    assert_eq!((wide.session_reservation_bytes(),wide.request_reservation_bytes(1),wide.request_reservation_bytes(CONTEXT)),(Some(min_chat+rings),Some(min_chat+rings),Some(CHAT_BYTES+rings)));
    // Room for exactly two paged chats gives two, also when the caller takes up to four. One byte less gives one.
    let pair=built(CHAT_BYTES,ROOM,2).unwrap();
    let two=pair.step_bytes()+2*pair.chat_bytes();assert_eq!(pair.chat_bytes(),min_chat+rings);
    assert_eq!(built(CHAT_BYTES,two,LIMITS.step_rows).unwrap().chats(),2);
    let one=built(CHAT_BYTES,two-1,LIMITS.step_rows).unwrap();assert_eq!((one.chats(),one.chat_bytes(),one.pool()),(1,min_chat,two-1-step));
    assert_eq!(built(CHAT_BYTES,step+CHAT_BYTES,LIMITS.step_rows).unwrap().chats(),1);
    // No room for one chat beside the values of a step: the start is refused with the numbers.
    let none=built(CHAT_BYTES,step+CHAT_BYTES-1,LIMITS.step_rows).err().unwrap();
    assert!(none.starts_with("text service memory refusal: CONTEXT=48; ") && none.contains(&format!("step_values={step};")) && none.contains("no chat fits beside the values of one step of 4 rows"),"{none}");
}
/// A request that names no sampling field is decoded greedy, and a named field sits on a greedy base.
#[test]
fn the_limits_give_greedy_when_no_sampling_is_named() {
    let limits=limits(CONTEXT,24).unwrap();
    assert!(limits.greedy_if_unsampled);assert_eq!(limits.default_sampling,Some(spark_engine::sample::Sampling::greedy()));
    assert_eq!((limits.context_tokens,limits.input_tokens,limits.output_tokens,limits.next_mtp_depth),(CONTEXT,CONTEXT-1,24,Some(0)));
}
/// The memory rule: the ledger of the load plan against a budget. A context that does not fit is refused with the
/// largest context that fits, before any device work.
#[test]
fn a_context_is_admitted_by_the_ledger_and_refused_with_the_largest_that_fits() {
    let fixture=Fixture::small();let d=&fixture.description;
    let load=LoadBytes {device:1<<20,..Default::default()};let bucket=4;
    let plan=|context|LoopStatePlan::for_steps(d,context,bucket,LIMITS.step_rows as u64).unwrap();
    let total=|context|LoopLedger::new(&load,&plan(context)).unwrap().total;
    // A budget that holds exactly the ledger of 48 tokens admits 48 and every smaller context.
    let admitted=admit(d,&load,48,total(48),bucket,LIMITS).unwrap();
    assert_eq!((admitted.context,admitted.ledger.total,admitted.ledger.weights,admitted.largest_context),(48,total(48),1<<20,48));
    assert_eq!((admitted.budget,admitted.limits),(total(48),LIMITS));
    assert_eq!(admitted.room().unwrap() as u64,total(48)-(1<<20));assert_eq!(admitted.chat_bytes().unwrap() as u64,total(48)-(1<<20));
    assert_eq!(admitted.chat_bytes().unwrap() as u64,plan(48).chat_bytes().unwrap());
    assert_eq!(admitted.seats().unwrap(),Seats {limits:LIMITS,chat:admitted.chat_bytes().unwrap(),room:admitted.room().unwrap(),most:LIMITS.step_rows});
    assert!(admit(d,&load,8,total(48),bucket,LIMITS).is_ok());
    // The state of a chat in the ledger has the extra ring rows of the step row limit: three more rows in the
    // key ring and in the value ring of each window visit than the plan of one row a step.
    let (one,rows)=(LoopStatePlan::new(d,48,bucket).unwrap(),plan(48));
    let extra:u64=rows.visits.iter().filter(|visit|visit.window.is_some()).map(|visit|2*visit.row_bytes*(LIMITS.step_rows as u64-1)).sum();
    assert!(extra>0);assert_eq!(rows.fixed_bytes().unwrap(),one.fixed_bytes().unwrap()+extra);assert_eq!(rows.plane_bytes().unwrap(),one.plane_bytes().unwrap());
    assert_eq!(LoopStatePlan::for_steps(d,48,bucket,1).unwrap(),one);assert!(LoopStatePlan::for_steps(d,48,bucket,0).is_err());
    // The driver allocates for a chat the state bytes of the ledger less the extra rows of the wider window rings: its
    // state keeps window rings, and a chat with many rows in a step makes its wider rings in the step (step bytes).
    for context in [8,48,64] {
        let driver=Driver::with_limits(&Host,&fixture,d,context,LIMITS).unwrap();let plan=plan(context);
        assert_eq!(driver.state_words() as u64*4+extra,plan.fixed_bytes().unwrap()+plan.plane_bytes().unwrap(),"context {context}");
    }
    // One byte less: 48 is refused, and the text names 44, the largest context of whole buckets that fits.
    let refusal=admit(d,&load,48,total(48)-1,bucket,LIMITS).unwrap_err();
    assert!(refusal.starts_with("text service memory refusal: CONTEXT=48; ") && refusal.ends_with("reduce CONTEXT to 44 or less"),"{refusal}");
    assert!(admit(d,&load,44,total(48)-1,bucket,LIMITS).is_ok());
    // A budget below the weights fits no context.
    let none=admit(d,&load,8,1<<19,bucket,LIMITS).unwrap_err();assert!(none.contains("no context fits"),"{none}");
    // A context that the model does not take is refused with the model's range.
    for context in [0,d.max_positions as usize+1] {
        let refusal=admit(d,&load,context,u64::MAX,bucket,LIMITS).unwrap_err();
        assert!(refusal.starts_with("text service context refusal: ") && refusal.contains(&format!("the model takes 1 to {} tokens",d.max_positions)),"{refusal}");
    }
}

/// A model folder for the request reader and the identity of a service: the small host tokenizer with the control
/// tokens of this family's chat template, the checked template itself, an EOS list of the end-of-turn token, a
/// config and one weight file with no tensor.
pub(super) struct Folder {pub root:std::path::PathBuf,pub stop:u32}
impl Folder {
    pub fn new(name:&str)->Self {
        const HOST:&str=include_str!("../../../../tokenizer/tests/fixtures/host_edges/tokenizer.json");
        const STOP:&str="<|content_model_end_sampling|>";
        const CONTROLS:[&str;12]=["<|message_user|>","<|message_model|>","<|message_system|>","<|message_tool|>","<|content_text|>","<|content_thinking|>","<|content_xml|>",
            "<|content_invoke_tool_json|>","<|end_message|>",STOP,"<|content_tool_error|>","<|content_invoke_tool_text|>"];
        let root=std::env::temp_dir().join(format!("vade-loop-service-{name}-{}",std::process::id()));std::fs::create_dir(&root).unwrap();
        let extra=CONTROLS.iter().enumerate().map(|(place,text)|format!(r#"{{"id":{},"content":"{text}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}"#,300+place)).collect::<Vec<_>>().join(",");
        // The host tokenizer in the byte-level form with no normalizer, which loads as a text-only tokenizer.
        let tokenizer=HOST.replace(r#""normalizer":{"type":"NFC"}"#,r#""normalizer":null"#).replace(r#""trim_offsets":false"#,r#""trim_offsets":true"#)
            .replace(r#""decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}"#,r#""decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}"#)
            .replacen(r#""added_tokens":["#,&format!(r#""added_tokens":[{extra},"#),1);
        let stop=300+CONTROLS.iter().position(|&text|text==STOP).unwrap() as u32;
        for (name,text) in [("tokenizer.json",tokenizer.as_str()),("tokenizer_config.json","{}"),("chat_template.jinja",spark_serve::tokenizer::inkling::TEMPLATE),("generation_config.json",&format!(r#"{{"eos_token_id":{stop}}}"#)),
            ("config.json",crate::loop_decoder::fixture::SMALL)] {
            std::fs::write(root.join(name),text).unwrap();
        }
        // The identity of a service reads the header of each weight file. This one has an empty header and no tensor.
        std::fs::write(root.join("model.safetensors"),[2u64.to_le_bytes().as_slice(),b"{}"].concat()).unwrap();
        Self {root,stop}
    }
}
impl Drop for Folder {fn drop(&mut self) {let removed=std::fs::remove_dir_all(&self.root);if !std::thread::panicking() {removed.unwrap();}}}
/// Tokens of one chat in the tests with text: a rendered chat is longer than the token ID prompts above.
pub(super) const CHAT_CONTEXT:usize=160;
/// A made-up model for requests with text: the small model with a head that can write every token of the folder's
/// tokenizer, and a longer chat.
fn chat_model()->Fixture {chat_model_with(1)}
/// The same model with `eos` as the EOS token of its config.
pub(super) fn chat_model_with(eos:u32)->Fixture {
    use crate::loop_decoder::fixture::SMALL;
    let config=SMALL.replace(r#""vocab_size":24,"unpadded_vocab_size":20"#,r#""vocab_size":320,"unpadded_vocab_size":312"#).replace(r#""model_max_length":64"#,r#""model_max_length":256"#)
        .replace(r#""eos_token_id":1,"#,&format!(r#""eos_token_id":{eos},"#));
    let fixture=Fixture::new(&config).unwrap();let d=&fixture.description;assert_eq!((d.head_rows,d.max_positions,d.eos_tokens.as_slice()),(312,256,&[u64::from(eos)][..]));fixture
}
/// A CHAT WITH TEXT, END TO END: the request reader of the model folder renders the messages with the checked
/// template, the driver decodes, and the answer streams. The served token IDs equal the direct decode of the
/// rendered prompt, ID for ID. The stop set of the service is the EOS list of the folder: the scheduler refuses
/// every request of a service whose stop set differs from it.
#[test]
fn a_chat_with_text_is_rendered_and_served_with_the_ids_of_the_direct_decode() {
    let fixture=chat_model();let folder=Folder::new("chat");
    let tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(&folder.root).unwrap();
    assert!(spark_serve::Tokenizer::token_channels(&tokenizer),"the folder selects the reply reader of this family");
    let eos=spark_serve::Tokenizer::checkpoint_eos_token_ids(&tokenizer).unwrap().unwrap().to_vec();assert_eq!(eos,[folder.stop]);
    let body=|fields:&str|format!(r#"{{"messages":[{{"role":"system","content":"Be brief."}},{{"role":"user","content":"Hello"}}],"max_tokens":8,"stream":true,"ignore_eos":true{fields},"stream_options":{{"include_usage":true}}}}"#);
    // The prompt as the request reader renders it: the system message, the effort message, the user message, and no start mark for the model.
    let request=spark_serve::protocol::parse(body(r#","temperature":0"#).as_bytes(),true,&limits(CHAT_CONTEXT,24).unwrap()).unwrap();
    let spark_serve::protocol::Prompt::Chat(input)=&request.prompt else {panic!("the request is a chat")};
    let prompt=spark_serve::Tokenizer::encode_chat(&tokenizer,input,1<<20).unwrap();
    let text=tokenizer.format_chat_input(input,1<<20).unwrap();
    assert_eq!(text,"<|message_system|><|content_text|>Be brief.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Hello<|end_message|>");
    assert!(prompt.len()>9 && prompt.len()<CHAT_CONTEXT-8 && prompt.iter().all(|&id|id<312),"{prompt:?}");
    let expected=direct_at(&fixture,CHAT_CONTEXT,&prompt,8);
    served_at(&fixture,CHAT_CONTEXT,eos.clone(),4,Some(&tokenizer),|send,address| {
        for fields in [r#","temperature":0"#,""] {
            let reply=post(send,address,"/v1/chat/completions",&body(fields));assert!(ok(&reply),"{reply}");
            let served=ids(&reply);assert!(!served.is_empty(),"the answer holds no token ID: {reply}");
            assert_eq!(served,expected,"the served token IDs differ from the direct decode of the rendered prompt: {reply}");
            assert!(reply.contains(&format!("\"prompt_tokens\":{}",prompt.len())) && reply.contains("\"completion_tokens\":8"),"{reply}");
        }
        // Image input is refused with a clear text before the engine sees the request: this engine runs the text part only.
        for part in [r#"{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}"#,r#"{"type":"image","image":"AAAA"}"#] {
            let image=format!(r#"{{"messages":[{{"role":"user","content":[{{"type":"text","text":"What is this?"}},{part}]}}],"max_tokens":4,"temperature":0}}"#);
            let reply=post(send,address,"/v1/chat/completions",&image);
            assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains("image input is unsupported by this text-only engine"),"{reply}");
        }
    });
    // A service whose stop set is not the EOS list of the folder answers no request.
    served_at(&fixture,CHAT_CONTEXT,vec![1],1,Some(&tokenizer),|send,address| {
        let reply=post(send,address,"/v1/chat/completions",&body(r#","temperature":0"#));
        assert!(reply.starts_with("HTTP/1.1 400 ") && reply.contains("backend stop tokens do not match the checkpoint EOS set"),"{reply}");
    });
}
/// Paged straight KV planes grow across page boundaries (`16 -> 32 -> 48` on `Host`) and run mixed-length cohorts
/// in both `step_batch` and `verify_many` with zero token-ID drift vs direct single-chat decode.
#[test]
fn paged_mixed_length_cohort_and_prefix_cache_match_direct_decode() {
    use spark_serve::Session;
    for fixture in [Fixture::small(),Fixture::wide()] {
        let d=&fixture.description;
        let service=Service::new(&Host,&fixture,d,CONTEXT,vec![1],Seats {limits:Limits {step_rows:16,key_bytes:1<<20},chat:CHAT_BYTES,room:ROOM,most:4}).unwrap();
        let long_prompt:Vec<u32>=(0..34).map(|i|((i*7+3)%(d.head_rows as usize)) as u32).collect();
        let mid_prompt:Vec<u32>=(0..19).map(|i|((i*5+2)%(d.head_rows as usize)) as u32).collect();
        let short_prompt:Vec<u32>=(0..4).map(|i|((i*3+1)%(d.head_rows as usize)) as u32).collect();
        let want_long=direct(&fixture,&long_prompt,8);
        let want_mid=direct(&fixture,&mid_prompt,8);
        let want_short=direct(&fixture,&short_prompt,8);
        // Three chats at 3 different mapped straight-plane page counts (48, 32, 16 rows) share one batch step.
        let (mut a,mut b,mut c)=(service.session().unwrap(),service.session().unwrap(),service.session().unwrap());
        let top=|logits:&[f32]|crate::bounded_decoder::greedy_logits(logits,logits.len()).unwrap();
        let mut next=[top(&a.prefill(&long_prompt).unwrap()),top(&b.prefill(&mid_prompt).unwrap()),top(&c.prefill(&short_prompt).unwrap())];
        assert_eq!(next,[want_long[0],want_mid[0],want_short[0]]);
        // Short chat keeps its 1-page straight planes while long chat has grown to 3 pages.
        {
            let slots=service.slots.borrow();
            let layouts=service.driver.layouts();
            let straight=layouts.iter().position(|l|!l.ring).unwrap();
            let sa=slots[a.active_slot().unwrap()].as_ref().unwrap().state.visits[straight].keys.len()/layouts[straight].row_words;
            let sb=slots[b.active_slot().unwrap()].as_ref().unwrap().state.visits[straight].keys.len()/layouts[straight].row_words;
            let sc=slots[c.active_slot().unwrap()].as_ref().unwrap().state.visits[straight].keys.len()/layouts[straight].row_words;
            assert!(sc<sb && sb<sa,"distinct page counts: short={sc} mid={sb} long={sa}");
        }
        // Mixed-length cohort steps stay bit-exact with direct single-chat decode.
        for step in 1..8 {
            let answers=service.step_batch(&mut [
                BatchRows {session:&mut a,tokens:&next[0..1],output:BatchOutput::All,greedy:true,top_k:0},
                BatchRows {session:&mut b,tokens:&next[1..2],output:BatchOutput::All,greedy:true,top_k:0},
                BatchRows {session:&mut c,tokens:&next[2..3],output:BatchOutput::All,greedy:true,top_k:0},
            ]).map_err(|e|e.message).unwrap();
            let got:Vec<u32>=answers.into_iter().map(|p|match p {Predictions::Greedy(v)=>v[0],_=>panic!("greedy")}).collect();
            assert_eq!(got,[want_long[step],want_mid[step],want_short[step]],"mixed-length step {step}");
            next=[got[0],got[1],got[2]];
        }
        // Prefix cache snapshot and restore across page boundaries (> 1 page) stays bit-exact, and reservations scale with pages.
        let mut src=service.session().unwrap();
        let _=src.prefill(&long_prompt).unwrap();
        let kept=src.fork().unwrap();
        drop(src);
        assert!(service.request_reservation_bytes(short_prompt.len()).unwrap()<service.request_reservation_bytes(long_prompt.len()).unwrap());
        let big_layout=state::Layout {rows:262_144,window:262_144,step_rows:4096,key_bytes:1<<30,ring:false,row_words:512,kv_channels:1024,kv_taps:4,channels:4096,taps:4};
        let p_short=state::prefix_bytes::<Vec<f32>>(&[big_layout],1024,2*1024*1024);
        let p_long=state::prefix_bytes::<Vec<f32>>(&[big_layout],65_536,2*1024*1024);
        assert!(p_short<p_long,"paged prefix cache scales with 2 MiB VMM pages: {p_short} < {p_long}");
        let mut restored=kept.fork().unwrap();
        let next_id=top(&restored.step(want_long[0]).unwrap());
        assert_eq!(next_id,want_long[1],"forked multi-page prefix stays bit-exact");
    }
}
/// Long-context fixes: tokenizer limits accept > 1 MiB text when configured from `limits()`, `request_body_bytes`
/// covers 512K-1M token ID arrays, and `CacheLimits` keeps full `context` tokens per entry past 262,144.
#[test]
fn long_context_limits_tokenizer_and_cache_caps_scale_past_256k() {
    let folder=Folder::new("long-ctx-limits");
    let mut tokenizer=spark_serve::tokenizer::ModelTokenizer::from_directory(&folder.root).unwrap();
    let big_text="ab ".repeat(400_000);
    assert!(big_text.len()>1_048_576);
    assert!(spark_serve::Tokenizer::encode(&tokenizer,&big_text).is_err(),"default 1 MiB limit refuses >1 MiB text");
    let lim=limits(524_288,8192).unwrap();
    assert_eq!(lim.body_bytes,8*1024*1024);
    tokenizer.set_limits(lim.tokenizer_limits()).unwrap();
    let encoded=spark_serve::Tokenizer::encode(&tokenizer,&big_text).unwrap();
    assert!(!encoded.is_empty(),"wired tokenizer_limits accepts >1 MiB text");
    let lim_1m=limits(1_048_576,8192).unwrap();
    assert_eq!(lim_1m.body_bytes,16*1024*1024);
    // 524K token-ID JSON body (> 4 MiB) parses cleanly under 512K service limits.
    let ids_body=format!(r#"{{"prompt_token_ids":[{}],"max_tokens":1,"temperature":0}}"#,vec!["12345";400_000].join(","));
    assert!(ids_body.len()>2*1024*1024 && ids_body.len()<lim.body_bytes);
    assert!(spark_serve::protocol::parse(ids_body.as_bytes(),false,&lim).is_ok());
    // Active prefix cache permits entries up to 1M tokens and > 128 entries (bounded by retained bytes, no arbitrary cap).
    for ctx in [262_144usize,524_288,1_048_576] {
        let cl=spark_serve::cache::CacheLimits {entries:256,retained_bytes:1<<30,tokens_per_entry:ctx,logits_per_entry:200_000,idle_ttl:Duration::from_secs(60)};
        assert!(spark_serve::cache::SessionCache::<()>::new(cl).is_ok(),"cache limits accept context {ctx}");
    }
    // Service admission derives chat count and cache entries from actual memory budget, with no arbitrary 32-chat or 8-entry cap.
    let fixture=Fixture::small();
    let wide=Service::new(&Host,&fixture,&fixture.description,CONTEXT,vec![1],Seats {limits:Limits {step_rows:64,key_bytes:1<<20},chat:CHAT_BYTES,room:256<<20,most:64}).unwrap();
    assert_eq!(wide.chats(),64,"memory-derived chat count is not capped at 32");
    assert_eq!(wide.schedule().0.entries,64,"prefix cache entry count scales with admitted chats");
}
