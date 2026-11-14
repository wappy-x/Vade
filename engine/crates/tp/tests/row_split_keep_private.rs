//! Kept prompts on two ranks (host twins, the test wire): what a caller of another cache scope is shown of them (no
//! count, event line or device bytes; the cache's ROOM is still one pool for all callers, see engine/docs/server.md),
//! and how long an unused one lives. The scheduler's own tests of both are in spark_serve.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::{bounded_decoder::Host,row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
use std::sync::mpsc::{sync_channel,SyncSender,Receiver};
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

/// THE TWO ROUTES FROM A SECOND SCOPE, on the real two-rank service with the settings the text service uses (prefix
/// cache on, private counts). The server asks no key here, so a caller's credential header is its cache scope. The
/// first caller's prompt is read (a miss), asked again (a hit), pushed out by another prompt (the cache holds one
/// entry: an eviction) and read once more (a miss). After each step a caller of ANOTHER scope asks `/health` and
/// `/v1/events`: neither may hold a `prefix_cache_` name, which is what every count and line of the cache starts
/// with (they carry prompt lengths, hits and kept bytes). The second caller's own reply may not hold one either.
/// Before the fix `/health` counted prefix_cache_miss and prefix_cache_status for every caller and `/v1/events`
/// showed their lines.
#[test]
fn a_second_scope_sees_no_cache_event_of_the_first_on_two_ranks() {
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
            // One entry: the second prompt pushes the first out.
            let cache=spark_serve::cache::CacheLimits {entries:1,retained_bytes:cap,tokens_per_entry:80,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)};
            let config=spark_serve::scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:pool,session_budget_bytes:pool,draft:spark_serve::protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:true,
                sharing:spark_serve::scheduler::Sharing {private_counts:true,..Default::default()}};
            // 4 asks of the first caller, 2 routes after each, 1 ask of the second caller.
            spark_serve::scheduler::serve_requests(listener,&engine,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,stop,"private",13).unwrap();
            assert_eq!(engine.fatal_error(),None);engine.shutdown().unwrap();});
        let send=|text:String| {let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
            stream.write_all(text.as_bytes()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply};
        let ask=|credential:&str,prompt:&str| {let body=format!(r#"{{"prompt_token_ids":[{prompt}],"stream":true,"max_tokens":4,"ignore_eos":true,"temperature":0,"stream_options":{{"include_usage":true}}}}"#);
            send(format!("POST /v1/completions HTTP/1.1\r\nHost: {address}\r\n{credential}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()))};
        let get=|path:&str,credential:&str|send(format!("GET {path} HTTP/1.1\r\nHost: {address}\r\n{credential}\r\n"));
        let cached=|reply:&str|reply.rsplit("\"cached_tokens\":").next().unwrap().chars().take_while(char::is_ascii_digit).collect::<String>().parse::<usize>().unwrap();
        let (first,second)=("Authorization: Bearer the-first-caller\r\n","Authorization: Bearer the-second-caller\r\n");
        for (step,prompt,kept) in [("a miss","3,5,1,7,2,9",0),("a hit","3,5,1,7,2,9",6),("an eviction","4,8,1,6,2,7",0),("a miss after the eviction","3,5,1,7,2,9",0)] {
            let reply=ask(first,prompt);assert!(reply.starts_with("HTTP/1.1 200 "),"{step}: {reply}");assert_eq!(cached(&reply),kept,"{step}: {reply}");
            for route in ["/health","/v1/events"] {
                let seen=get(route,second);assert!(seen.starts_with("HTTP/1.1 200 "),"{step} {route}: {seen}");
                assert!(!seen.contains("prefix_cache_"),"after {step}, {route} shows the second scope a cache event of the first: {seen}");
                for word in ["hits=","misses=","reused_tokens=","evicted_tokens=","scoped_entries="] {assert!(!seen.contains(word),"after {step}, {route} shows {word}: {seen}");}
                // The device bytes grow with a kept prompt: with private counts /health shows none.
                if route=="/health" {assert!(seen.contains("\"device_bytes\":null"),"after {step}, /health shows device bytes: {seen}");}
            }
        }
        let own=ask(second,"6,5,2");assert!(own.starts_with("HTTP/1.1 200 "),"{own}");
        assert!(!own.contains("prefix_cache_"),"the second caller's own reply counts the first caller's cache events: {own}");
    });
}

/// THE LIFETIME OF AN UNUSED KEPT PROMPT, with a clock the test drives, and with NO memory pressure (the cache may
/// keep far more than it holds). The shared cache above the two-rank service keeps two prompts. One second before a
/// prompt has been unused for its lifetime (300 s) it is still there; at 300 s the next pass of the scheduler's cache
/// balance drops it, and BOTH ranks release it, by the release an eviction sends (command 17). A use starts the time
/// again. A lookup drops an expired prompt too: it cannot start a chat from it. After all of it the service reads the
/// prompt cold with the bits of a fresh read. Before the fix an unused prompt stayed until memory ran short.
#[test]
fn a_prompt_unused_for_its_lifetime_is_released_on_both_ranks_with_no_memory_pressure() {
    use spark_serve::cache::{CacheLimits,CachedSession,SessionCache,SessionKey};
    use std::{collections::BTreeSet,time::{Duration,Instant}};
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);
    let (a_tx,b_rx)=sync_channel(1);let (b_tx,a_rx)=sync_channel(1);
    std::thread::scope(|scope| {
        let checkpoint=&c.admission;
        let worker=scope.spawn(move || {let split=Split::new(Host,Wire {topology:Topology {rank:1,parts:2},controller:false,tx:b_tx,rx:b_rx}).unwrap();
            let factory=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();factory.worker().unwrap();
            assert_eq!(factory.kept_prompts(),(0,0),"the worker rank released every prompt that expired");});
        let split=Split::new(Host,Wire {topology:Topology {rank:0,parts:2},controller:true,tx:a_tx,rx:a_rx}).unwrap();
        let engine=Factory::new(checkpoint,split,80,vec![0]).unwrap().with_prompt_keep();
        let best=|v:&[f32]|(0..v.len()).reduce(|a,b|if v[b]>v[a] {b} else {a}).unwrap() as u32;
        let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
        let fresh=|prompt:&[u32],count:usize| {let mut session=engine.session().unwrap();let mut logits=session.prefill(prompt).unwrap();let mut rows=Vec::new();
            for at in 0..count {let id=best(&logits);rows.push((id,bits(&logits)));if at+1<count {logits=Session::step(&mut session,id).unwrap();}}rows};
        let prompts=[vec![1u32,3,2,7,5,11,4,9,6,13],vec![2u32,6,4,1,15,3,8,12,5,9]];
        let keys=[SessionKey::generate().unwrap(),SessionKey::generate().unwrap()];
        let expected=fresh(&prompts[1],4);
        // No pressure: the cache may hold 64 MiB and holds two small prompts.
        let room=64usize<<20;
        let mut cache=SessionCache::new(CacheLimits {entries:8,retained_bytes:room,tokens_per_entry:80,logits_per_entry:engine.vocab_size(),idle_ttl:Duration::from_secs(300)}).unwrap();
        let start=Instant::now();let at=|seconds|start+Duration::from_secs(seconds);
        let read=|cache:&mut SessionCache<_>,chat:usize,now:Instant| {let mut session=engine.session().unwrap();let logits=session.prefill(&prompts[chat]).unwrap();
            let state=CachedSession {checkpoint_lease:None,session,tokens:prompts[chat].clone(),next_logits:logits,reserved_session_bytes:engine.request_reservation_bytes(80).unwrap()};
            cache.snapshot(keys[chat],&state,now);};
        // The scheduler's pass with nothing to give back: every byte the cache may hold is free.
        let pass=|cache:&mut SessionCache<_>,now:Instant|cache.rebalance_within(room,None,BTreeSet::new(),now);
        read(&mut cache,0,at(0));read(&mut cache,1,at(100));assert_eq!((cache.len(),engine.kept_prompts().0),(2,2));
        // 299 s after prompt 0 was kept: both are there. 300 s: prompt 0 is dropped and both ranks release it.
        pass(&mut cache,at(299));assert_eq!((cache.len(),cache.counts().expired,engine.kept_prompts().0),(2,0,2),"one second before the lifetime");
        pass(&mut cache,at(300));assert_eq!((cache.len(),cache.counts().expired,cache.counts().evicted,engine.kept_prompts().0),(1,1,0,1),"at the lifetime, with no pressure");
        assert!(cache.fork_prefix(&keys[0],&prompts[0],at(300)).state.is_none(),"an expired prompt starts no chat");
        // Prompt 1 is used at 399 s (299 s after it was kept): the use starts its time again.
        let used=cache.fork_prefix(&keys[1],&prompts[1],at(399)).state.expect("a prompt younger than its lifetime restores");
        assert_eq!(bits(&used.next_logits),expected[0].1,"the restored prompt has the bits of a fresh read");drop(used);
        pass(&mut cache,at(399+299));assert_eq!((cache.len(),engine.kept_prompts().0),(1,1),"299 s after its last use");
        // No pass comes at 300 s after the use: the LOOKUP itself drops the prompt and finds nothing.
        assert!(cache.fork_prefix(&keys[1],&prompts[1],at(399+300)).state.is_none(),"a lookup cannot start from an expired prompt");
        assert_eq!((cache.len(),cache.counts().expired,engine.kept_prompts()),(0,2,(0,0)),"both prompts are released on rank 0");
        assert_eq!(engine.fatal_error(),None);
        // The service is whole: the prompt reads cold with the bits of the first read.
        assert_eq!(fresh(&prompts[1],4),expected);
        drop(cache);engine.shutdown().unwrap();worker.join().unwrap();
    });
}
