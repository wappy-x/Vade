//! Real scheduler intake while a synchronous large prompt slice holds scratch.
use crate::{Engine,Session,protocol,cache::CacheLimits,scheduler::{self,Config},Limits};
use std::{cell::{Cell,RefCell},net::{TcpListener,TcpStream,Ipv4Addr},io::{Read,Write},sync::{atomic::AtomicBool,mpsc},time::Duration};
struct Model {live:Cell<usize>,room:Cell<usize>,scratch:RefCell<Vec<u8>>,offers:RefCell<Vec<(usize,usize)>>,arrival:RefCell<Option<(mpsc::Sender<()>,mpsc::Receiver<()>)>>}
struct Chat<'a>(&'a Model);
impl Drop for Chat<'_> {fn drop(&mut self) {self.0.live.set(self.0.live.get()-1);}}
impl Session for Chat<'_> {
    fn step(&mut self,_:u32)->Result<Vec<f32>,String> {Ok(vec![1.0,0.0])}
    fn prefill(&mut self,_:&[u32])->Result<Vec<f32>,String> {
        if self.0.room.get()>0 {
            self.0.scratch.replace(vec![0;self.0.room.get()]);
            if let Some((sent,written))=self.0.arrival.borrow_mut().take() {sent.send(()).unwrap();written.recv().unwrap();}
        }
        self.0.scratch.replace(Vec::new());self.0.room.set(0);Ok(vec![1.0,0.0])
    }
    fn prefill_chunk_size(&self,_:usize)->usize {if self.0.room.get()>0 {512} else {256}}
    fn prefill_owns_chunks(&self)->bool {true}
}
impl Engine for Model {
    type Session<'a>=Chat<'a>;
    fn session(&self)->Result<Chat<'_>,String> {
        assert_eq!(self.scratch.borrow().capacity(),0,"intake admitted while borrowed allocation was live");
        self.live.set(self.live.get()+1);Ok(Chat(self))
    }
    fn prepare_prompt_workspace(&self,available:usize)->Result<(),String> {
        assert_eq!(self.scratch.borrow().capacity(),0);
        if self.live.get()>1 {assert_eq!(available,0,"concurrent chats borrowed pool bytes");}
        self.offers.borrow_mut().push((self.live.get(),available));self.room.set(available);Ok(())
    }
    fn finish_prompt_workspace(&self)->Result<(),String> {assert_eq!(self.scratch.borrow().capacity(),0);self.room.set(0);Ok(())}
    fn shared_prefill_batch(&self)->bool {true}
    fn shared_prefill_limit(&self)->usize {256}
    fn shared_prefill_max_tokens(&self)->usize {256}
    fn reserve_full_output(&self)->bool {true}
    fn context_limit(&self)->usize {2048}
    fn vocab_size(&self)->usize {2}
    fn eos_token_ids(&self)->&[u32] {&[0]}
    fn session_reservation_bytes(&self)->Option<usize> {Some(100)}
    fn request_reservation_bytes(&self,_:usize)->Option<usize> {Some(100)}
    fn cache_hold_bytes(&self)->usize {50}
}
fn send(address:std::net::SocketAddr)->TcpStream {
    let prompt=vec![1;1024];let body=format!("{{\"prompt_token_ids\":{prompt:?},\"max_tokens\":1,\"temperature\":0,\"cache\":false}}");
    let mut socket=TcpStream::connect(address).unwrap();socket.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    write!(socket,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();socket
}
#[test]
fn arrival_during_large_slice_returns_allocation_before_eight_admissions() {
    let listener=TcpListener::bind((Ipv4Addr::LOCALHOST,0)).unwrap();let address=listener.local_addr().unwrap();
    let (started,wait)=mpsc::channel();let (written,resume)=mpsc::channel();
    let engine=Model {live:Cell::new(0),room:Cell::new(0),scratch:RefCell::new(Vec::new()),offers:RefCell::new(Vec::new()),arrival:RefCell::new(Some((started,resume)))};
    let stop=AtomicBool::new(false);let stopping=&stop;
    std::thread::scope(|scope| {
        let clients=scope.spawn(move || {
            let _stop=Stop(stopping);
            let first=send(address);wait.recv_timeout(Duration::from_secs(30)).unwrap();
            let mut sockets=vec![first];for _ in 0..7 {sockets.push(send(address));}written.send(()).unwrap();
            for mut socket in sockets {let mut text=String::new();socket.read_to_string(&mut text).unwrap();assert!(text.starts_with("HTTP/1.1 200"),"{text}");}
        });
        let limits=Limits {input_tokens:2047,context_tokens:2048,output_tokens:1,..Default::default()};
        let cache=CacheLimits {entries:1,retained_bytes:0,tokens_per_entry:2048,logits_per_entry:2,idle_ttl:Duration::from_secs(300)};
        let config=Config {active:8,connections:16,prefill_chunk:256,foreground_rows:1,background_rows:1,active_bytes:1000,session_budget_bytes:1000,draft:protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:false,sharing:Default::default()};
        scheduler::serve_requests(listener,&engine,None::<&Codec>,&limits,&cache,&config,&stop,"fixture",8).unwrap();clients.join().unwrap();
    });
    assert!(engine.offers.borrow().contains(&(1,850)),"live reservation and hold were not subtracted: {:?}",engine.offers.borrow());
    assert!(engine.offers.borrow().iter().any(|&(live,room)|live>1 && room==0));
    assert_eq!(engine.live.get(),0);assert_eq!(engine.scratch.borrow().capacity(),0);
}

struct Codec;
impl crate::Decoder for Codec {fn push(&mut self,_:u32)->Result<String,String> {Ok(String::new())}fn finish(&mut self)->Result<String,String> {Ok(String::new())}}
impl crate::Tokenizer for Codec {
    type Decoder<'a>=Self;
    fn encode(&self,_:&str)->Result<Vec<u32>,String> {Ok(vec![1])}
    fn decoder(&self)->Self {Self}
    fn has_token(&self,id:u32)->bool {id<2}
    fn encode_chat(&self,_:&protocol::ChatInput,_:usize)->Result<Vec<u32>,String> {Ok(vec![1])}
}

struct Stop<'a>(&'a AtomicBool);
impl Drop for Stop<'_> {fn drop(&mut self) {self.0.store(true,std::sync::atomic::Ordering::Relaxed);}}
