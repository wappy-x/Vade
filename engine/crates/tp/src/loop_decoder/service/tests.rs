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
/// The engine itself: its count of chats, the vocabulary is the rows that the head can write, the stop set is the
/// given list, and a chat that ends frees its place.
#[test]
fn the_engine_holds_its_chats_and_reports_its_limits() {
    use spark_serve::Session;
    let fixture=Fixture::small();let d=&fixture.description;
    let service=Service::new(&Host,&fixture,d,CONTEXT,vec![1,7],seats(1)).unwrap();
    assert_eq!((service.context_limit(),service.vocab_size(),service.eos_token_ids()),(CONTEXT,d.head_rows as usize,&[1u32,7][..]));
    assert_eq!((service.session_reservation_bytes(),service.request_reservation_bytes(1),service.request_reservation_bytes(CONTEXT)),(Some(CHAT_BYTES),Some(CHAT_BYTES),Some(CHAT_BYTES)));
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
    // The scheduler settings: the chats of the service, a prompt piece of the step row limit, plain decoding, no kept prompt.
    let (cache,config)=service.schedule();
    assert_eq!((config.active,config.prefill_chunk,config.mtp_depth,config.prefix_cache,config.active_bytes,cache.retained_bytes,cache.tokens_per_entry),(1,LIMITS.step_rows,0,false,service.pool(),0,CONTEXT));
    assert!(matches!(config.draft,DraftMode::Plain));
    // The room: the driver's bound of the values of one step comes off once, and the chats that fit in the rest
    // are the chats of the service. A chat of a service of several chats reserves one more set of rings for each
    // other chat.
    let step=service.step_bytes();
    assert!(step>0 && step<ROOM);assert_eq!((service.pool(),service.chat_bytes()),(ROOM-step,CHAT_BYTES));
    let wide=built(CHAT_BYTES,ROOM,3).unwrap();
    assert_eq!((wide.schedule().1.active,wide.shared_decode_rows(),wide.chats()),(3,3,3));
    // The step of a service of more chats holds more: the rings of every chat wait in it.
    assert!(wide.step_bytes()>step && wide.pool()==ROOM-wide.step_bytes(),"{} {step}",wide.step_bytes());
    let rings=(wide.chat_bytes()-CHAT_BYTES)/2;
    assert!(rings>0 && wide.chat_bytes()==CHAT_BYTES+2*rings,"{}",wide.chat_bytes());
    assert_eq!((wide.session_reservation_bytes(),wide.request_reservation_bytes(1)),(Some(CHAT_BYTES+2*rings),Some(CHAT_BYTES+2*rings)));
    // Room for exactly two chats gives two, also when the caller takes up to four. One byte less gives one.
    let pair=built(CHAT_BYTES,ROOM,2).unwrap();
    let two=pair.step_bytes()+2*pair.chat_bytes();assert_eq!(pair.chat_bytes(),CHAT_BYTES+rings);
    assert_eq!(built(CHAT_BYTES,two,LIMITS.step_rows).unwrap().chats(),2);
    let one=built(CHAT_BYTES,two-1,LIMITS.step_rows).unwrap();assert_eq!((one.chats(),one.chat_bytes(),one.pool()),(1,CHAT_BYTES,two-1-step));
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
