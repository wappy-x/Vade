//! An unmapped model winner reaches the client through the production tokenizer filter.
use spark_serve::{BatchError,BatchOutput,BatchRows,Engine,Limits,Predictions,Session,Tokenizer,cache::CacheLimits,protocol,scheduler};
use std::{io::{Read,Write},net::{Ipv4Addr,TcpListener,TcpStream},sync::{Arc,atomic::{AtomicBool,AtomicUsize,Ordering}},time::Duration};
const VOCAB:usize=154_880;
const PADDING:u32=154_856;
const MESSAGE:&str="The model chose an output slot that is not a token.";
#[derive(Clone,Copy)]
enum Fault { Prompt, Decode }
struct Model {fault:Fault,decodes:Arc<AtomicUsize>}
struct Kv {fault:Fault,decodes:Arc<AtomicUsize>}
fn logits(best:u32)->Vec<f32> {let mut row=vec![-1.0;VOCAB];row[best as usize]=9.0;row}
impl Session for Kv {
    fn reject_unmapped_greedy(&self)->bool {true}
    fn prefill(&mut self,_:&[u32])->Result<Vec<f32>,String> {
        Ok(logits(if matches!(self.fault,Fault::Prompt) {PADDING} else {0}))
    }
    fn step(&mut self,_:u32)->Result<Vec<f32>,String> {self.decodes.fetch_add(1,Ordering::Relaxed);Ok(logits(PADDING))}
}
impl Engine for Model {
    type Session<'a>=Kv;
    fn session(&self)->Result<Kv,String> {Ok(Kv{fault:self.fault,decodes:self.decodes.clone()})}
    fn vocab_size(&self)->usize {VOCAB}
    fn context_limit(&self)->usize {256}
    fn eos_token_ids(&self)->&[u32] {&[]}
    fn session_reservation_bytes(&self)->Option<usize> {Some(1<<20)}
    fn step_batch(&self,rows:&mut [BatchRows<'_,Kv>])->Result<Vec<Predictions>,BatchError> {
        rows.iter_mut().map(|row|match row.output {
            BatchOutput::All=>{row.session.decodes.fetch_add(1,Ordering::Relaxed);Ok(Predictions::Greedy(vec![PADDING]))},
            BatchOutput::Last=>row.session.prefill(row.tokens).map(|r|Predictions::Logits(vec![r])),
            BatchOutput::None=>Ok(Predictions::Logits(Vec::new())),
        }.map_err(|message|BatchError{request:None,message})).collect()
    }
}
fn request(stream:bool,fault:Fault)->String {
    let listener=TcpListener::bind((Ipv4Addr::LOCALHOST,0)).unwrap();let address=listener.local_addr().unwrap();
    let decodes=Arc::new(AtomicUsize::new(0));let observed=decodes.clone();
    let reply=std::thread::scope(|scope| {
        let server=scope.spawn(move|| {
            let tokenizer=engine_tokenizer::Tokenizer::from_json(include_bytes!("../../tokenizer/tests/glm-tokenizer-slice.json"),Default::default()).unwrap();
            assert!(tokenizer.has_token(154_855));assert!(tokenizer.has_token(0));
            for id in PADDING..VOCAB as u32 {assert!(!tokenizer.has_token(id));}
            let engine=Model{fault,decodes};
            let limits=Limits{input_tokens:128,output_tokens:8,context_tokens:256,greedy_if_unsampled:true,
                default_sampling:Some(spark_engine::sample::Sampling::greedy()),..Default::default()};
            let cache=CacheLimits{entries:1,retained_bytes:0,tokens_per_entry:256,logits_per_entry:VOCAB,idle_ttl:Duration::from_secs(60)};
            let config=scheduler::Config{active:2,connections:8,prefill_chunk:8,foreground_rows:1,background_rows:1,active_bytes:1<<24,session_budget_bytes:1<<24,
                draft:protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
            scheduler::serve_requests(listener,&engine,Some(&tokenizer),&limits,&cache,&config,&AtomicBool::new(false),"padding test",1).unwrap();
        });
        let body=format!(r#"{{"prompt_token_ids":[0],"max_tokens":2,"cache":false,"stream":{stream}}}"#);
        let mut socket=TcpStream::connect(address).unwrap();socket.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
        write!(socket,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).unwrap();
        let mut reply=String::new();socket.read_to_string(&mut reply).unwrap();server.join().unwrap();reply
    });
    assert_eq!(observed.load(Ordering::Relaxed),usize::from(matches!(fault,Fault::Decode)));
    reply
}
#[test]
fn nonstream_padding_winner_explains_invalid_model_output() {
    for fault in [Fault::Prompt,Fault::Decode] {
        let reply=request(false,fault);
        assert!(reply.starts_with("HTTP/1.1 422"),"{reply}");
        assert!(reply.contains("\"type\":\"invalid_model_output\"") && reply.contains(MESSAGE),"{reply}");
        assert!(!reply.contains(spark_engine::draft::UNMAPPED_GREEDY) && !reply.contains("generation_error"),"{reply}");
    }
}
#[test]
fn stream_padding_winner_explains_invalid_model_output() {
    for fault in [Fault::Prompt,Fault::Decode] {
        let reply=request(true,fault);
        assert!(reply.starts_with("HTTP/1.1 200"),"{reply}");
        assert!(reply.contains("data: {\"error\":{\"type\":\"invalid_model_output\"") && reply.contains(MESSAGE),"{reply}");
        assert!(reply.ends_with("data: [DONE]\n\n"),"{reply}");
        assert!(!reply.contains(spark_engine::draft::UNMAPPED_GREEDY) && !reply.contains("generation_error"),"{reply}");
    }
}
