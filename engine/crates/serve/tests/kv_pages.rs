// Admission by use: the scheduler charges each chat the KV pages it holds (kv_pages::Geometry),
// grows the charge with the chat and refuses one chat cleanly when no page is free for it.
use spark_serve::{Decoder, Engine, Limits, Session, Tokenizer, cache::CacheLimits, kv_pages::{Format, Geometry}, protocol, scheduler};
use std::{
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}},
    time::Duration,
};

/// Live sessions, the most live at once, and the tokens each ended session was funded for.
#[derive(Default)]
struct Ledger { live: AtomicUsize, peak: AtomicUsize, funded: Mutex<Vec<usize>>, admitted: Mutex<Vec<usize>> }
/// `charge`: None charges every chat the full window, as before this change.
struct Paged { geometry: Geometry, ahead: u64, full: Option<usize>, context: usize, step_sleep: Duration, ledger: Arc<Ledger> }
struct Chat { history: Vec<u32>, funded: usize, sleep: Duration, ledger: Arc<Ledger> }
impl Drop for Chat {
    fn drop(&mut self) {
        self.ledger.live.fetch_sub(1, Ordering::SeqCst);
        self.ledger.funded.lock().unwrap().push(self.funded);
    }
}
impl Session for Chat {
    fn advance(&mut self, token: u32) -> Result<(), String> { self.history.push(token); Ok(()) }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
        std::thread::sleep(self.sleep);
        self.advance(token)?;
        Ok(vec![0., 1., 0., 0., 0., 0., 0., 0.])
    }
    fn truncate(&mut self, position: usize) -> Result<(), String> { self.history.truncate(position); Ok(()) }
    fn fork(&self) -> Result<Self, String> { Err("no fork".into()) }
    fn reserve_model_draft(&mut self, tokens: usize) { self.funded = tokens; }
}
impl Engine for Paged {
    type Session<'a> = Chat;
    fn session(&self) -> Result<Chat, String> {
        let live = self.ledger.live.fetch_add(1, Ordering::SeqCst) + 1;
        self.ledger.peak.fetch_max(live, Ordering::SeqCst);
        Ok(Chat { history: Vec::new(), funded: 0, sleep: self.step_sleep, ledger: self.ledger.clone() })
    }
    fn session_for_admission(&self, _: &protocol::Request, tokens: usize) -> Result<Chat, String> {
        self.ledger.admitted.lock().unwrap().push(tokens);
        self.session()
    }
    fn automatic_draft_allowed(&self, _: usize) -> bool { false }
    fn vocab_size(&self) -> usize { 8 }
    fn context_limit(&self) -> usize { self.context }
    fn eos_token_ids(&self) -> &[u32] { &[7] }
    fn session_reservation_bytes(&self) -> Option<usize> { self.request_reservation_bytes(1) }
    fn request_reservation_bytes(&self, tokens: usize) -> Option<usize> {
        self.full.or_else(|| usize::try_from(self.geometry.charge(tokens as u64, self.ahead)?).ok())
    }
}
struct Codec;
impl Decoder for Codec {
    fn push(&mut self, _: u32) -> Result<String, String> { Ok("x".into()) }
    fn finish(&mut self) -> Result<String, String> { Ok(String::new()) }
}
impl Tokenizer for Codec {
    type Decoder<'a> = Codec;
    fn encode(&self, _: &str) -> Result<Vec<u32>, String> { Ok(vec![1]) }
    fn decoder(&self) -> Codec { Codec }
    fn has_token(&self, id: u32) -> bool { id < 8 }
    fn encode_chat(&self, _: &protocol::ChatInput, _: usize) -> Result<Vec<u32>, String> { Ok(vec![1]) }
}

/// Serve `bodies` on a scheduler with `active` slots and an `active_bytes` pool; send all of
/// them at once and return each response.
fn serve<E: Engine + Send + 'static>(engine: E, limits: Limits, active: usize, active_bytes: usize, bodies: &[String]) -> Vec<String> {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let shutdown = stop.clone();
    let requests = bodies.len();
    let worker = std::thread::spawn(move || {
        scheduler::serve_requests(listener, &engine, Some(&Codec), &limits,
            &CacheLimits { entries: 32, retained_bytes: 1 << 20, tokens_per_entry: 1024, logits_per_entry: 8, idle_ttl: Duration::from_secs(60) },
            &scheduler::Config { active, connections: 32, prefill_chunk: 256, foreground_rows: 8, background_rows: 16,
                active_bytes, session_budget_bytes: active_bytes.max(2 << 20), draft: protocol::DraftMode::Plain, mtp_depth: 0,
                draft_global_bank: false, prefix_cache: false, sharing: scheduler::Sharing::default() },
            &shutdown, "paged", requests).unwrap()
    });
    let sockets: Vec<TcpStream> = bodies.iter().map(|body| send(address, body)).collect();
    let responses = sockets.into_iter().map(|mut socket| {
        let mut output = String::new();
        socket.read_to_string(&mut output).unwrap();
        output
    }).collect();
    worker.join().unwrap();
    responses
}
fn send(address: SocketAddr, body: &str) -> TcpStream {
    let mut socket = TcpStream::connect(address).unwrap();
    socket.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
    write!(socket, "POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    socket
}
fn complete(response: &str) -> bool { response.contains("data: [DONE]") && !response.contains("\"error\"") }

/// The Flash-Next rank geometry at 131072 (12 indexed attention layers and the draft layer);
/// `full` is the 10-27 two-Spark session charge and `pool` its session pool.
const PAGE: u64 = 2 << 20;
const FULL: u64 = 8_165_864_772;
const POOL: usize = 21_279_044_229;
fn flash_next(full: bool, ledger: &Arc<Ledger>) -> Paged {
    let geometry = Geometry::new(PAGE, (0..13).flat_map(|_| Geometry::attention(2, 256, 128, Format::Fp32)).collect(), FULL - 59_904 * 131_072, 131_072).unwrap();
    Paged { geometry, ahead: 64 + 7 + 1, full: full.then_some(FULL as usize), context: 131_072, step_sleep: Duration::from_millis(3), ledger: ledger.clone() }
}

#[test]
fn eight_short_chats_run_together_at_131072() {
    let limits = Limits { input_tokens: 131_072, context_tokens: 131_072, output_tokens: 32, body_bytes: 1 << 20, ..Limits::default() };
    let prompt = vec![1u32; 1275];
    let bodies = vec![format!("{{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":{prompt:?},\"max_tokens\":32}}"); 8];
    for (full, peak) in [(true, 2), (false, 8)] {
        let ledger = Arc::new(Ledger::default());
        let responses = serve(flash_next(full, &ledger), limits.clone(), 16, POOL, &bodies);
        assert!(responses.iter().all(|r| complete(r)), "{responses:?}");
        assert_eq!((ledger.peak.load(Ordering::SeqCst), ledger.live.load(Ordering::SeqCst)), (peak, 0), "window charge: {full}");
        assert_eq!(*ledger.admitted.lock().unwrap(), [1275 + 32; 8]);
    }
}

#[test]
fn growth_past_the_pool_refuses_one_chat_and_the_others_get_their_pages() {
    // 256 bytes per token in 1,024-byte pages: 257 tokens charge 66,560 bytes, 401 tokens
    // 103,424. Nine chats start at 257 (599,040) and grow to 401 at 240 output tokens;
    // eight grown and one not (893,952) fit the 900,000-byte pool, nine grown do not.
    let ledger = Arc::new(Ledger::default());
    let engine = Paged { geometry: Geometry::new(1024, vec![256], 0, 4096).unwrap(), ahead: 0, full: None, context: 4096,
        step_sleep: Duration::from_millis(1), ledger: ledger.clone() };
    let limits = Limits { output_tokens: 400, ..Limits::default() };
    let body = "{\"stream\":true,\"temperature\":0,\"prompt_token_ids\":[1]}".to_string();
    let responses = serve(engine, limits, 9, 900_000, &vec![body; 9]);
    assert_eq!(ledger.peak.load(Ordering::SeqCst), 9);
    let refused: Vec<_> = responses.iter().filter(|r| !complete(r)).collect();
    assert_eq!(refused.len(), 1, "{responses:?}");
    assert!(refused[0].contains("capacity_exhausted"), "{}", refused[0]);
    let mut funded = ledger.funded.lock().unwrap().clone();
    funded.sort();
    assert_eq!(funded, [257, 401, 401, 401, 401, 401, 401, 401, 401]);
}

struct Funded(Paged);
impl Engine for Funded {
    type Session<'a> = Chat;
    fn session(&self)->Result<Chat,String> {self.0.session()}
    fn session_for_admission(&self,r:&protocol::Request,tokens:usize)->Result<Chat,String> {self.0.session_for_admission(r,tokens)}
    fn vocab_size(&self)->usize {8}
    fn context_limit(&self)->usize {600}
    fn eos_token_ids(&self)->&[u32] {&[7]}
    fn session_reservation_bytes(&self)->Option<usize> {Some(1)}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize> {Some(tokens)}
    fn reserve_full_output(&self)->bool {true}
    fn automatic_draft_allowed(&self,_:usize)->bool {false}
}
#[test]
fn eight_implicit_output_chats_finish_like_solo_when_full_budget_queues_the_eighth() {
    let run=|count| {
        let ledger=Arc::new(Ledger::default());
        let engine=Funded(Paged {geometry:Geometry::new(1024,vec![256],0,4096).unwrap(),ahead:0,full:None,context:600,step_sleep:Duration::ZERO,ledger:ledger.clone()});
        // The old 256-token window admitted all eight, then killed one at growth.
        let body=r#"{"stream":true,"temperature":0,"cache":false,"prompt_token_ids":[1]}"#.to_owned();
        let replies=serve(engine,Limits {output_tokens:257,..Limits::default()},8,8*257,&vec![body;count]);
        assert!(ledger.admitted.lock().unwrap().iter().all(|&n|n==258));
        assert_eq!(ledger.peak.load(Ordering::SeqCst),count.min(7));replies
    };
    let ids=|reply:&str|reply.split("\"token_id\":").skip(1).map(|s|s.split(|c:char|!c.is_ascii_digit()).next().unwrap().parse::<u32>().unwrap()).collect::<Vec<_>>();
    let solo=run(1);let expected=ids(&solo[0]);assert_eq!(expected.len(),257);
    for reply in run(8) {assert!(complete(&reply),"{reply}");assert!(!reply.contains("capacity_exhausted"));assert_eq!(ids(&reply),expected);}
}

struct Poisoned {engine:Funded,failure:Arc<Mutex<Option<String>>>}
impl Engine for Poisoned {
    type Session<'a> = Chat;
    fn session(&self)->Result<Chat,String> {if let Some(error)=self.fatal_error() {Err(error)} else {self.engine.session()}}
    fn vocab_size(&self)->usize {8}
    fn context_limit(&self)->usize {600}
    fn eos_token_ids(&self)->&[u32] {&[7]}
    fn session_reservation_bytes(&self)->Option<usize> {Some(1)}
    fn request_reservation_bytes(&self,n:usize)->Option<usize> {Some(n)}
    fn fatal_error(&self)->Option<String> {self.failure.lock().unwrap().clone()}
}
#[test]
fn permanent_failure_reports_poisoned_health_and_the_same_refusal_reason() {
    let failure=Arc::new(Mutex::new(None));let injected=failure.clone();
    let engine=Poisoned {engine:Funded(Paged {geometry:Geometry::new(1024,vec![256],0,4096).unwrap(),ahead:0,full:None,context:600,step_sleep:Duration::ZERO,ledger:Arc::new(Ledger::default())}),failure};
    let listener=TcpListener::bind("127.0.0.1:0").unwrap();let address=listener.local_addr().unwrap();
    let worker=std::thread::spawn(move||scheduler::serve_requests(listener,&engine,Some(&Codec),&Limits::default(),&CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:600,logits_per_entry:8,idle_ttl:Duration::from_secs(60)},
        &scheduler::Config {active:8,connections:32,prefill_chunk:4,foreground_rows:1,background_rows:1,active_bytes:1<<20,session_budget_bytes:1<<20,draft:protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()},&AtomicBool::new(false),"fault",0));
    let read=|method:&str,path:&str,body:&str| {
        let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        write!(stream,"{method} {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();let mut reply=String::new();stream.read_to_string(&mut reply).unwrap();reply
    };
    assert!(read("GET","/health","").contains("\"status\":\"ready\""));
    *injected.lock().unwrap()=Some("injected permanent backend failure".into());
    let mut reply=String::new();for _ in 0..20 {reply=read("GET","/health","");if reply.contains("\"status\":\"poisoned\"") {break;}std::thread::sleep(Duration::from_millis(1));}
    assert!(reply.starts_with("HTTP/1.1 503"),"{reply}");assert!(reply.contains("\"status\":\"poisoned\""),"{reply}");assert!(reply.contains("\"reason\":\"injected permanent backend failure\""),"{reply}");
    for _ in 0..2 {let reply=read("POST","/v1/completions","{}");assert!(reply.starts_with("HTTP/1.1 503"),"{reply}");assert!(reply.contains("injected permanent backend failure"),"{reply}");assert!(!reply.contains("retry when ready"),"{reply}");}
    assert_eq!(worker.join().unwrap().unwrap_err(),"injected permanent backend failure");
}
