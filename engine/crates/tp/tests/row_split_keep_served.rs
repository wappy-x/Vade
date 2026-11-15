//! The prompt cache as the two-Spark text service serves it by default (host twins, the test wire, the real
//! scheduler): prompt keep on and private counts on. What a resume that fails costs the caller.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use spark_serve::Engine;
use std::{io::{Read,Write},net::{TcpListener,TcpStream},sync::{atomic::AtomicBool,mpsc::{sync_channel,SyncSender,Receiver}},time::Duration};
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
/// One two-rank server with the settings text.rs gives the service by default (prefix cache on, private counts on).
/// It answers `requests` requests and stops. `refuse`: the worker rank's kept work to refuse (`Factory::refuse_kept_work`:
/// bit n is its keep or resume number n), as a rank does whose own part of that work failed. `client` gets `send`.
fn served(requests:usize,refuse:u64,client:impl FnOnce(&dyn Fn(String)->String,std::net::SocketAddr)) {served_with(requests,true,move|worker|worker.refuse_kept_work(refuse),client)}
/// The same server; `arm` sets the worker rank's test hooks before it serves. `private` false: `/health` shows the
/// cache's counters, so a test can count its entries, hits and misses.
fn served_with(requests:usize,private:bool,arm:impl FnOnce(&Factory<'_,Host,Wire>)+Send,client:impl FnOnce(&dyn Fn(String)->String,std::net::SocketAddr)) {served_cap(requests,private,false,arm,client)}
/// The same server. `no_room`: the pool is exactly one full-size chat, so the engine's byte cap (`keep_within`) and the
/// cache's room are 0 while prompt keep stays on, as on a box whose free memory at start funds one chat and nothing more.
fn served_cap(requests:usize,private:bool,no_room:bool,arm:impl FnOnce(&Factory<'_,Host,Wire>)+Send,client:impl FnOnce(&dyn Fn(String)->String,std::net::SocketAddr)) {served_full(requests,private,no_room,true,arm,client)}
/// The same server. `keep` false: started as text.rs starts a service given `--no-prompt-keep` (no prompt keep on
/// either rank, a cache of one entry and 0 bytes, `prefix_cache` off).
fn served_full(requests:usize,private:bool,no_room:bool,keep:bool,arm:impl FnOnce(&Factory<'_,Host,Wire>)+Send,client:impl FnOnce(&dyn Fn(String)->String,std::net::SocketAddr)) {
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    // A failed check below must end the server too, or the test would wait for ever instead of failing.
    struct Stop<'a>(&'a AtomicBool);impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,std::sync::atomic::Ordering::Relaxed);}}
    let stop=AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (checkpoint,stop)=(&c.admission,&stop);let _stop=Stop(stop);
        scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap();let factory=if keep {factory.with_prompt_keep()} else {factory};arm(&factory);
            factory.worker().expect("the worker rank goes on after a refusal and stops only when it is told to");});
        scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
            let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap();let engine=if keep {engine.with_prompt_keep()} else {engine};let limits=spark_tp::row_split::limits(80,48).unwrap();
            let pool=if no_room {engine.full_state_reservation_bytes()} else {(1usize<<24).max(4*engine.full_state_reservation_bytes())};let cap=if keep {engine.keep_within(pool)} else {0};
            assert_eq!(cap==0,no_room || !keep,"the byte cap is 0 exactly when the pool holds one chat, or with no prompt keep");
            let cache=spark_serve::cache::CacheLimits {entries:if keep {8} else {1},retained_bytes:cap,tokens_per_entry:80,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:keep,
                sharing:spark_serve::scheduler::Sharing {private_counts:private,..Default::default()}};
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,stop,"private",requests).unwrap();
            assert_eq!(engine.fatal_error(),None,"the service is not marked failed");if !keep {assert_eq!(engine.kept_prompts(),(0,0),"no prompt was kept");}engine.shutdown().unwrap();});
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        client(&send,address);
    });
}
fn ask(send:&dyn Fn(String)->String,address:std::net::SocketAddr,prompt:&str)->String {ask_with(send,address,prompt,r#""temperature":0"#)}
/// `fields`: the request's sampling (and any other) fields, as JSON members.
fn ask_with(send:&dyn Fn(String)->String,address:std::net::SocketAddr,prompt:&str,fields:&str)->String {
    let body=format!(r#"{{"prompt_token_ids":[{prompt}],"stream":true,"max_tokens":6,"ignore_eos":true,{fields},"stream_options":{{"include_usage":true}}}}"#);
    send(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()))
}
fn ids(reply:&str)->Vec<u32> {reply.split("\"token_id\":").skip(1).map(|part|part.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap()).collect()}
fn cached(reply:&str)->usize {reply.rsplit("\"cached_tokens\":").next().unwrap().chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap()}

/// The early read's point 3, the part a full device can cause. A turn hits a kept prompt and one rank cannot finish
/// its part of the resume: on a device that is a refused allocation while it builds the chat's expanded rows again,
/// which is a plain error (only a sticky execution fault of the driver is fatal). The ranks vote (`Factory::kept_vote`),
/// so this ONE request is refused the kept prompt on both ranks, the scheduler reads its prompt cold, and the answer
/// has the IDs of the first read with 0 cached tokens. The service is not marked failed. Both ranks drop the refused
/// prompt, the cold read keeps the prompt again, and the next ask starts from that new kept prompt. So a memory
/// squeeze in a resume costs one cold read.
#[test]
fn a_resume_one_rank_cannot_finish_is_read_cold_with_the_same_ids_and_the_server_goes_on() {
    // The worker rank's kept work: number 0 is the keep of the first request, number 1 the resume of the second.
    served(5,0b10,|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        let refused=ask(send,address,"3,5,1,7,2,9");assert!(refused.starts_with("HTTP/1.1 200 "),"the request is answered: {refused}");
        assert_eq!((ids(&refused),cached(&refused)),(ids(&first),0),"a refused resume is a cold read with the same IDs: {refused}");
        let health=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(health.contains("\"status\":\"ready\""),"{health}");
        let next=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&next),cached(&next)),(ids(&first),6),"the next ask starts from the kept prompt: {next}");
        let other=ask(send,address,"4,8,1,6");assert_eq!((ids(&other).len(),cached(&other)),(6,0),"another prompt is served as before: {other}");
    });
}

/// A turn hits a kept prompt that the worker rank has LOST. Before the readiness word (`Factory::kept_ready`) the
/// worker left its loop with "worker resume has no kept prompt" and every later request failed. Now this one request
/// reads its prompt cold: http 200, 0 cached tokens, the IDs of the first read. Both ranks have forgotten that prompt,
/// so the next ask reads cold too and is kept again, and the ask after it starts from the new kept prompt.
#[test]
fn a_resume_of_a_prompt_the_worker_rank_lost_is_read_cold_and_the_server_goes_on() {
    served_with(6,true,|worker|worker.lose_kept_before_resume(1),|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        let lost=ask(send,address,"3,5,1,7,2,9");assert!(lost.starts_with("HTTP/1.1 200 "),"the request is answered: {lost}");
        assert_eq!((ids(&lost),cached(&lost)),(ids(&first),0),"a lost kept prompt is a cold read with the same IDs: {lost}");
        let health=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(health.contains("\"status\":\"ready\""),"{health}");
        let other=ask(send,address,"4,8,1,6");assert_eq!((ids(&other).len(),cached(&other)),(6,0),"another prompt is served: {other}");
        let again=ask(send,address,"3,5,1,7,2,9");assert_eq!(ids(&again),ids(&first),"{again}");
        let later=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&later),cached(&later)),(ids(&first),6),"the prompt is kept again and serves the next ask: {later}");
    });
}

/// THE SAME PROMPT REFUSED N TIMES IN A ROW (3 here). Each refused resume drops its kept prompt on both ranks, and the
/// cache lets the dead entry go at once, so the cold read that follows keeps the prompt as ONE fresh entry: the cache
/// never holds two copies of the prompt, and a request never tries more than one. Before, a refused prompt stayed,
/// the cold read kept a second copy beside it, and a later ask tried every copy: up to 8 failed resumes a request.
/// The worker rank's kept work, in order: keep (0), then for each refused ask a resume and the keep of its cold read
/// (1 and 2, 3 and 4, 5 and 6), then a resume that works (7). It refuses numbers 1, 3 and 5. If any ask tried a second
/// copy, that try would take the next number and the asks below would not read 0, 0, 0 and then 6 cached tokens.
#[test]
fn a_prompt_refused_three_times_in_a_row_is_one_entry_and_each_ask_tries_one_resume() {
    served_with(11,false,|worker|worker.refuse_kept_work(0b10_1010),|send,address| {
        let health=|| {let reply=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(reply.contains("\"status\":\"ready\""),"{reply}");
            let count=|name:&str|reply.split(&format!("\"{name}\":")).nth(1).unwrap_or_else(||panic!("/health has no {name}: {reply}")).chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
            (count("entries"),count("hits"),count("misses"))};
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");assert_eq!(health(),(1,0,1));
        for refusal in 1..=3 {
            let refused=ask(send,address,"3,5,1,7,2,9");assert!(refused.starts_with("HTTP/1.1 200 "),"refusal {refusal}: {refused}");
            assert_eq!((ids(&refused),cached(&refused)),(ids(&first),0),"refusal {refusal}: one cold read with the same IDs: {refused}");
            assert_eq!(health(),(1,0,1+refusal),"refusal {refusal}: one entry for the prompt, and no hit");
        }
        let hit=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&hit),cached(&hit)),(ids(&first),6),"the fourth ask starts from the one kept prompt: {hit}");
        assert_eq!(health(),(1,1,4));
    });
}

/// A byte cap of 0 with prompt keep ON (the pool funds one full-size chat and nothing beside it). The start line then
/// reads `cache=true kept_cap_bytes=0`. Every request is still looked up (only `--no-prompt-keep` skips the lookup),
/// but nothing is kept: no entry, no retained byte, no hit, and every answer has the IDs of the first read.
#[test]
fn a_cap_of_zero_with_the_cache_on_keeps_nothing_and_every_ask_reads_cold() {
    served_cap(6,false,true,|_|{},|send,address| {
        let health=|| {let reply=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(reply.contains("\"status\":\"ready\""),"{reply}");
            let count=|name:&str|reply.split(&format!("\"{name}\":")).nth(1).unwrap_or_else(||panic!("/health has no {name}: {reply}")).chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
            (count("entries"),count("retained_bytes"),count("hits"),count("misses"))};
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        assert_eq!(health(),(0,0,0,1),"looked up once, nothing kept");
        for again in 1..=2 {
            let next=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&next),cached(&next)),(ids(&first),0),"ask {again}: a cold read with the same IDs: {next}");
            assert_eq!(health(),(0,0,0,1+again),"ask {again}: still nothing kept and no hit");
        }
    });
}

/// A later turn that changes its sampling after a kept first turn. The kept prompt holds state only: the new request's
/// sampling (temperature, top_p, seed) is applied after the restore, so the turn that starts from the kept prompt gives
/// the IDs of the same request read cold (`"cache": false`). Two seeds, so the check is not met by chance.
#[test]
fn a_turn_that_changes_sampling_after_a_kept_turn_has_the_ids_of_its_cold_read() {
    served_with(7,true,|_|{},|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        for seed in [7,11] {
            let sampling=format!(r#""temperature":0.8,"top_p":0.9,"seed":{seed}"#);
            let kept=ask_with(send,address,"3,5,1,7,2,9",&sampling);assert_eq!((ids(&kept).len(),cached(&kept)),(6,6),"seed {seed}: starts from the kept prompt: {kept}");
            let cold=ask_with(send,address,"3,5,1,7,2,9",&format!(r#"{sampling},"cache":false"#));assert_eq!(cached(&cold),0,"seed {seed}: {cold}");
            assert_eq!(ids(&kept),ids(&cold),"seed {seed}: the kept turn has the IDs of its cold read");
        }
    });
}

/// One request from its own connection, for requests sent at the same time from several threads.
fn post(address:std::net::SocketAddr,prompt:&str,fields:&str)->String {
    let body=format!(r#"{{"prompt_token_ids":[{prompt}],"stream":true,"max_tokens":6,"ignore_eos":true,{fields},"stream_options":{{"include_usage":true}}}}"#);
    let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    stream.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).unwrap();
    let mut reply=String::new();stream.read_to_string(&mut reply).expect("an answer, never a hung request");reply
}
fn health_counts(send:&dyn Fn(String)->String,address:std::net::SocketAddr)->(usize,usize,usize) {
    let reply=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(reply.contains("\"status\":\"ready\""),"{reply}");
    let count=|name:&str|reply.split(&format!("\"{name}\":")).nth(1).unwrap_or_else(||panic!("/health has no {name}: {reply}")).chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
    (count("entries"),count("hits"),count("misses"))
}

/// TWO RESUMES OF ONE KEPT PROMPT AT ONCE. The kept state is shared and never written: each request builds its own
/// chat from it. Both start from the kept prompt (6 cached tokens), both have the IDs of the first read, and the kept
/// prompt is still there for a third ask.
#[test]
fn two_resumes_of_one_kept_prompt_at_once_both_start_from_it_with_the_ids_of_the_first_read() {
    served_with(6,false,|_|{},|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        let (a,b)=std::thread::scope(|s| {let a=s.spawn(||post(address,"3,5,1,7,2,9",r#""temperature":0"#));std::thread::sleep(Duration::from_millis(20));
            let b=s.spawn(||post(address,"3,5,1,7,2,9",r#""temperature":0"#));(a.join().unwrap(),b.join().unwrap())});
        for (name,reply) in [("a",&a),("b",&b)] {
            assert!(reply.starts_with("HTTP/1.1 200 "),"{name}: {reply}");
            assert_eq!((ids(reply),cached(reply)),(ids(&first),6),"{name}: starts from the kept prompt with the first read's IDs: {reply}");
        }
        assert_eq!(health_counts(send,address).1,2,"two hits");
        let third=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&third),cached(&third)),(ids(&first),6),"{third}");
    });
}

/// A RESUME WHILE THE SAME PROMPT IS BEING KEPT. Chat A reads a prompt nothing holds yet and keeps it at the prompt's
/// end; while A still decodes (40 tokens, its chat shares the kept state), B asks the same prompt. B starts from the
/// kept prompt (6 cached tokens) and has the IDs of the prompt's cold read; A's answer is unchanged by B; the cache
/// holds ONE entry for the prompt.
#[test]
fn a_resume_while_the_same_prompt_is_being_kept_starts_from_it_with_the_cold_ids() {
    served_with(5,false,|_|{},|send,address| {
        let cold=ask_with(send,address,"3,5,1,7,2,9",r#""temperature":0,"cache":false"#);assert_eq!((ids(&cold).len(),cached(&cold)),(6,0),"{cold}");
        let body=r#"{"prompt_token_ids":[3,5,1,7,2,9],"stream":true,"max_tokens":40,"ignore_eos":true,"temperature":0,"stream_options":{"include_usage":true}}"#;
        let mut a=TcpStream::connect(address).unwrap();a.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        a.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).unwrap();
        let mut seen=Vec::new();let mut chunk=[0u8;4096];
        while !String::from_utf8_lossy(&seen).contains("\"token_id\":") {let n=a.read(&mut chunk).expect("A answers");assert!(n>0,"A ended before its first token");seen.extend_from_slice(&chunk[..n]);}
        let b=post(address,"3,5,1,7,2,9",r#""temperature":0"#);
        a.read_to_end(&mut seen).expect("A finishes");let a=String::from_utf8_lossy(&seen).into_owned();
        assert!(b.starts_with("HTTP/1.1 200 "),"{b}");assert_eq!((ids(&b),cached(&b)),(ids(&cold),6),"B starts from the prompt A is keeping: {b}");
        assert_eq!((ids(&a).len(),ids(&a)[..6].to_vec()),(40,ids(&cold)),"A is unchanged by B: {a}");
        assert_eq!(health_counts(send,address).0,1,"one entry for one prompt");
    });
}

/// A RESUME AFTER THE WORKER RANK LOST THE PROMPT, WHILE ANOTHER CHAT RUNS. The lost prompt's request ends in ONE
/// cold read with the right IDs (0 cached tokens), never a hung request (every read has a 60 s limit), and the other
/// chat, served at the same time, has the IDs of its own read alone. The service goes on.
#[test]
fn a_resume_of_a_lost_prompt_beside_a_live_chat_is_one_cold_read_and_the_other_chat_is_unchanged() {
    served_with(6,true,|worker|worker.lose_kept_before_resume(1),|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        let alone=ask_with(send,address,"4,8,1,6",r#""temperature":0,"cache":false"#);assert_eq!(ids(&alone).len(),6,"{alone}");
        let (other,lost)=std::thread::scope(|s| {let o=s.spawn(||post(address,"4,8,1,6",r#""temperature":0"#));std::thread::sleep(Duration::from_millis(5));
            let l=s.spawn(||post(address,"3,5,1,7,2,9",r#""temperature":0"#));(o.join().unwrap(),l.join().unwrap())});
        assert!(lost.starts_with("HTTP/1.1 200 "),"{lost}");assert_eq!((ids(&lost),cached(&lost)),(ids(&first),0),"one cold read with the right IDs: {lost}");
        assert!(other.starts_with("HTTP/1.1 200 "),"{other}");assert_eq!(ids(&other),ids(&alone),"the other chat: {other}");
        let health=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));assert!(health.contains("\"status\":\"ready\""),"{health}");
        let again=ask(send,address,"3,5,1,7,2,9");assert_eq!(ids(&again),ids(&first),"{again}");
    });
}

/// A LIVE SERVER STARTED LIKE `--no-prompt-keep` (text.rs: no prompt keep on either rank, `prefix_cache` off, a cache
/// of one entry and 0 bytes). It has no prompt cache to look in: `/health` reports `"prefix_cache":null` (with the
/// cache on it reports counts or `{"enabled":true}`), the same prompt asked twice reads cold twice with the same IDs,
/// and neither rank holds a kept prompt at the end.
#[test]
fn a_live_no_prompt_keep_server_has_no_cache_to_look_in_and_reads_every_prompt_cold() {
    served_full(3,false,false,false,|_|{},|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        let again=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&again),cached(&again)),(ids(&first),0),"{again}");
        let health=send(format!("GET /health HTTP/1.1\r\nHost: {address}\r\n\r\n"));
        assert!(health.contains("\"prefix_cache\":null"),"no prompt cache, so no lookup: {health}");
    });
    // This server's off settings are the ones text.rs gives a rank started with --no-prompt-keep.
    let source=include_str!("../src/bin/flash-serve/text.rs");let code=&source[..source.find("#[cfg(test)]").unwrap()];
    assert!(code.contains("prefix_cache:keep,") && code.contains("entries:if keep {spark_tp::row_split::KEPT_PROMPTS} else {1},") && code.contains("let kept_cap=if keep {engine.keep_within(pool)} else {0};"));
}

/// A CANCEL IN THE MIDDLE OF A RESUME. The client closes its connection after the first bytes of a reply that started
/// from a kept prompt. The chat that resume opened is released (the service shuts down at the end with no live
/// session and no kept prompt, `Factory::shutdown`), the kept prompt stays usable, and the next ask starts from it.
#[test]
fn a_cancel_in_the_middle_of_a_resume_releases_its_chat_and_the_kept_prompt_stays_usable() {
    served_with(3,true,|_|{},|send,address| {
        let first=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&first).len(),cached(&first)),(6,0),"{first}");
        {   let body=r#"{"prompt_token_ids":[3,5,1,7,2,9],"stream":true,"max_tokens":40,"ignore_eos":true,"temperature":0}"#;
            let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            stream.write_all(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).unwrap();
            let mut head=[0u8;64];let n=stream.read(&mut head).unwrap();assert!(n>0 && head.starts_with(b"HTTP/1.1 200 "),"the resumed reply started");
            stream.shutdown(std::net::Shutdown::Both).unwrap();}
        std::thread::sleep(Duration::from_millis(200));
        let next=ask(send,address,"3,5,1,7,2,9");assert_eq!((ids(&next),cached(&next)),(ids(&first),6),"the kept prompt is still usable: {next}");
    });
}
