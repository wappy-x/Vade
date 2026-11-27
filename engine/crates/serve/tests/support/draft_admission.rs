/// One draft entitlement costs four units; a retired plain request costs one.
/// The first retirement must precede construction of the second session.
#[derive(Clone)]
struct DraftAdmission { live: Arc<AtomicUsize>, first_step: Arc<AtomicUsize>, retired: Arc<AtomicUsize>, first_retire: Arc<AtomicUsize> }
struct AdmissionKv { model: DraftAdmission, optional: bool }
impl Drop for AdmissionKv { fn drop(&mut self) { self.model.live.fetch_sub(1,Ordering::SeqCst); } }
impl Session for AdmissionKv {
    fn advance(&mut self,_:u32)->Result<(),String> { self.model.first_step.compare_exchange(0,self.model.live.load(Ordering::SeqCst),Ordering::SeqCst,Ordering::SeqCst).ok();Ok(()) }
    fn step(&mut self,token:u32)->Result<Vec<f32>,String> {self.advance(token)?;Ok(vec![0.0,1.0])}
    fn disable_model_draft(&mut self) {if self.optional {self.optional=false;self.model.first_retire.compare_exchange(0,self.model.live.load(Ordering::SeqCst),Ordering::SeqCst,Ordering::SeqCst).ok();self.model.retired.fetch_add(1,Ordering::SeqCst);}}
}
impl Engine for DraftAdmission {
    type Session<'a> = AdmissionKv;
    fn session(&self)->Result<AdmissionKv,String> {self.live.fetch_add(1,Ordering::SeqCst);Ok(AdmissionKv {model:self.clone(),optional:true})}
    fn vocab_size(&self)->usize {2}
    fn context_limit(&self)->usize {128}
    fn eos_token_ids(&self)->&[u32] {&[]}
    fn discard_contended_draft(&self)->bool {true}
    fn single_chat_draft(&self)->bool {true}
    fn shared_batch(&self)->bool {true}
    fn shared_prefill_batch(&self)->bool {true}
    fn shared_decode_rows(&self)->usize {8}
    fn reserve_foreground_slot(&self)->bool {false}
    fn session_reservation_bytes(&self)->Option<usize> {Some(4)}
    fn request_reservation_bytes_for_prompt(&self,_:usize,_:usize,plain:bool,_:Option<usize>,_:bool)->Option<usize> {Some(if plain {1} else {4})}
}
#[test]
fn draft_admission_refunds_before_the_next_chat_fits_and_preserves_a_lone_entitlement() {
    for count in [1,8] {
        let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
        let model=DraftAdmission {live:AtomicUsize::new(0).into(),first_step:AtomicUsize::new(0).into(),retired:AtomicUsize::new(0).into(),first_retire:AtomicUsize::new(0).into()};
        let mut clients=Vec::new();
        for _ in 0..count {
            let mut socket=TcpStream::connect(address).unwrap();socket.set_read_timeout(Some(HANG_GUARD)).unwrap();
            let body=r#"{"prompt_token_ids":[1],"max_tokens":4,"temperature":0,"stream":true,"draft":"strict","cache":false}"#;
            write!(socket,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();clients.push(socket);
        }
        let server_model=model.clone();
        let worker=std::thread::spawn(move || {
            let limits=Limits {context_tokens:128,..Limits::default()};
            let cache=CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:128,logits_per_entry:2,idle_ttl:Duration::from_secs(1)};
            let config=scheduler::Config {active:8,connections:16,prefill_chunk:1,active_bytes:8,session_budget_bytes:8,draft:protocol::DraftMode::Strict,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default(),foreground_rows:8,background_rows:8};
            scheduler::serve_requests(listener,&server_model,None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"draft-admission",count).unwrap();
        });
        for client in &mut clients {let mut reply=String::new();client.read_to_string(&mut reply).unwrap();assert_eq!(tokens(&reply),vec![1;4],"{reply}");}
        worker.join().unwrap();
        assert_eq!(model.first_step.load(Ordering::SeqCst),count,"every funded chat must enter before the first model step");
        assert_eq!(model.retired.load(Ordering::SeqCst),if count==1 {0} else {count});
        assert_eq!(model.first_retire.load(Ordering::SeqCst),if count==1 {0} else {1},"the lone draft state must retire before a second session is constructed");
        assert_eq!(model.live.load(Ordering::SeqCst),0);
    }
}

struct PairAdmission(DraftAdmission);
impl Engine for PairAdmission {
    type Session<'a> = AdmissionKv;
    fn session(&self)->Result<AdmissionKv,String> {self.0.session()}
    fn vocab_size(&self)->usize {2}
    fn context_limit(&self)->usize {128}
    fn eos_token_ids(&self)->&[u32] {&[]}
    fn discard_contended_draft(&self)->bool {true}
    fn contended_draft_chats(&self)->usize {2}
    fn drafts_contended(&self,r:&protocol::Request)->bool {r.draft==Some(protocol::DraftMode::Strict)}
    fn shared_batch(&self)->bool {true}
    fn shared_prefill_batch(&self)->bool {true}
    fn shared_decode_rows(&self)->usize {8}
    fn reserve_foreground_slot(&self)->bool {false}
    fn session_reservation_bytes(&self)->Option<usize> {Some(4)}
    fn request_reservation_bytes_for_prompt(&self,_:usize,_:usize,plain:bool,_:Option<usize>,_:bool)->Option<usize> {Some(if plain {1} else {4})}
}
#[test]
fn selected_plain_priority_request_retires_the_live_pair_entitlement_before_admission() {
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let model=DraftAdmission {live:AtomicUsize::new(0).into(),first_step:AtomicUsize::new(0).into(),retired:AtomicUsize::new(0).into(),first_retire:AtomicUsize::new(0).into()};
    let mut clients=Vec::new();
    // The first strict request enters. The next queue head is compatible, but a younger plain
    // request wins by priority. The live entitlement must retire before constructing that session.
    // Intake removes completed readers by swap_remove, so the final socket is prepared second.
    for (priority,draft) in [(2,"strict"),(1,"plain"),(0,"strict")] {
        let mut socket=TcpStream::connect(address).unwrap();socket.set_read_timeout(Some(HANG_GUARD)).unwrap();
        let body=format!(r#"{{"prompt_token_ids":[1],"max_tokens":4,"temperature":0,"stream":true,"draft":"{draft}","priority":{priority},"cache":false}}"#);
        write!(socket,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();clients.push(socket);
    }
    let server_model=model.clone();
    let worker=std::thread::spawn(move || {
        let limits=Limits {context_tokens:128,..Limits::default()};
        let cache=CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:128,logits_per_entry:2,idle_ttl:Duration::from_secs(1)};
        let config=scheduler::Config {active:4,connections:8,prefill_chunk:1,active_bytes:12,session_budget_bytes:12,draft:protocol::DraftMode::Strict,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:scheduler::Sharing {client_priority_cap:2,..Default::default()},foreground_rows:8,background_rows:8};
        scheduler::serve_requests(listener,&PairAdmission(server_model),None::<&spark_serve::tokenizer::ModelTokenizer>,&limits,&cache,&config,&AtomicBool::new(false),"pair-admission",3).unwrap();
    });
    for client in &mut clients {let mut reply=String::new();client.read_to_string(&mut reply).unwrap();assert_eq!(tokens(&reply),vec![1;4],"{reply}");}
    worker.join().unwrap();
    assert_eq!(model.first_step.load(Ordering::SeqCst),3,"all requests must be admitted before model work");
    assert_eq!(model.first_retire.load(Ordering::SeqCst),1,"selected plain request must retire the running strict request before construction");
    assert_eq!(model.live.load(Ordering::SeqCst),0);
}
