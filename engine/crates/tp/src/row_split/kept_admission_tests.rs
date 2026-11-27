//! The real cache and scheduler with shape-funded sessions instead of host matrix work.
use super::*;
use super::kept_charge_tests::{fixture,Local};
use engine_model::plan::ConfigPlan;
use spark_serve::{Engine,Session as ServingSession,Decoder as TextDecoder,Tokenizer,Limits,protocol,scheduler,cache::CacheLimits};
use std::{rc::{Rc,Weak},net::{TcpListener,TcpStream,Ipv4Addr},io::{Read,Write},sync::atomic::AtomicBool,time::Duration};
const POOL:usize=6_400_000_000;
const FULL:usize=4_930_916_352;
const OWNER:usize=227_934_208;
#[derive(Default)]
struct Ledger {owners:RefCell<BTreeMap<u32,Weak<Owner>>>,reserved:Cell<usize>,evicted:RefCell<Vec<u32>>,full:RefCell<Option<(usize,usize,Vec<u32>)>>}
struct Owner {id:u32,kept:Cell<usize>}
impl Ledger {
    fn retained(&self)->usize {self.owners.borrow().values().filter_map(Weak::upgrade).filter(|o|o.kept.get()>0).count()*OWNER}
}
struct Model<'a> {charge:&'a dyn Fn(u64)->usize,ledger:Rc<Ledger>,short:usize,vocab:usize}
struct Chat<'a> {charge:&'a dyn Fn(u64)->usize,ledger:Rc<Ledger>,vocab:usize,position:usize,owner:Option<Rc<Owner>>,kept:bool,reserved:usize}
impl Drop for Chat<'_> {
    fn drop(&mut self) {
        if self.kept {let owner=self.owner.as_ref().unwrap();owner.kept.set(owner.kept.get()-1);if owner.kept.get()==0 {self.ledger.evicted.borrow_mut().push(owner.id);}}
        self.retire();
    }
}
impl ServingSession for Chat<'_> {
    fn step(&mut self,token:u32)->Result<Vec<f32>> {self.prefill(&[token])}
    fn prefill(&mut self,tokens:&[u32])->Result<Vec<f32>> {
        if self.owner.is_none() {let owner=Rc::new(Owner {id:tokens[0],kept:Cell::new(0)});self.ledger.owners.borrow_mut().insert(owner.id,Rc::downgrade(&owner));self.owner=Some(owner);}
        self.position+=tokens.len();let mut logits=vec![0.0;self.vocab];logits[0]=1.0;Ok(logits)
    }
    fn fork(&self)->Result<Self> {
        let owner=self.owner.as_ref().ok_or("empty snapshot")?.clone();owner.kept.set(owner.kept.get()+1);
        assert!(self.ledger.reserved.get()+self.ledger.retained()<=POOL,"publication exceeded the physical pool");
        Ok(Self {charge:self.charge,ledger:self.ledger.clone(),vocab:self.vocab,position:self.position,owner:Some(owner),kept:true,reserved:0})
    }
    fn can_fork(&self)->bool {self.position>0}
    fn retained_bytes(&self)->Option<usize> {Some((self.charge)(self.position as u64))}
    fn retire(&mut self) {self.ledger.reserved.set(self.ledger.reserved.get()-self.reserved);self.reserved=0;}
    fn retirement_bytes(&self)->Option<usize> {Some(0)}
}
impl Engine for Model<'_> {
    type Session<'a>=Chat<'a> where Self:'a;
    fn session(&self)->Result<Chat<'_>> {Ok(Chat {charge:self.charge,ledger:self.ledger.clone(),vocab:self.vocab,position:0,owner:None,kept:false,reserved:0})}
    fn session_for_admission(&self,_:&protocol::Request,tokens:usize)->Result<Chat<'_>> {
        let reservation=self.request_reservation_bytes(tokens).unwrap();let retained=self.ledger.retained();
        if tokens==16384 {*self.ledger.full.borrow_mut()=Some((retained,reservation,self.ledger.evicted.borrow().clone()));}
        if reservation+retained>POOL {return Err("retained owners and full request exceed pool".into());}
        let mut chat=self.session()?;chat.reserved=reservation;self.ledger.reserved.set(self.ledger.reserved.get()+reservation);Ok(chat)
    }
    fn reserve_full_output(&self)->bool {true}
    fn context_limit(&self)->usize {16384}
    fn vocab_size(&self)->usize {self.vocab}
    fn eos_token_ids(&self)->&[u32] {&[0]}
    fn session_reservation_bytes(&self)->Option<usize> {Some(FULL)}
    fn request_reservation_bytes(&self,tokens:usize)->Option<usize> {Some(if tokens==16384 {FULL} else {self.short})}
}
struct Codec;
impl TextDecoder for Codec {fn push(&mut self,_:u32)->Result<String> {Ok(String::new())}fn finish(&mut self)->Result<String> {Ok(String::new())}}
impl Tokenizer for Codec {
    type Decoder<'a>=Self;
    fn encode(&self,_:&str)->Result<Vec<u32>> {Ok(vec![1])}
    fn decoder(&self)->Self {Self}
    fn has_token(&self,id:u32)->bool {id<154880}
    fn encode_chat(&self,_:&protocol::ChatInput,_:usize)->Result<Vec<u32>> {Ok(vec![1])}
}
fn run(charge:&dyn Fn(u64)->usize,short:usize,vocab:usize)->(Vec<String>,Rc<Ledger>) {
    let ledger=Rc::new(Ledger::default());let model=Model {charge,ledger:ledger.clone(),short,vocab};
    let listener=TcpListener::bind((Ipv4Addr::LOCALHOST,0)).unwrap();let address=listener.local_addr().unwrap();
    let responses=std::thread::scope(|scope| {
        let clients=scope.spawn(move || {
            (1..=9).map(|id| {
                let prompt=if id==9 {vec![9;16383]} else {vec![id;256]};
                let output=1;let cache=id!=9;
                let body=format!("{{\"prompt_token_ids\":{prompt:?},\"max_tokens\":{output},\"temperature\":0,\"cache\":{cache}}}");
                let mut stream=TcpStream::connect(address).unwrap();stream.set_read_timeout(Some(Duration::from_secs(60))).unwrap();
                write!(stream,"POST /v1/completions HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).unwrap();
                let mut response=String::new();stream.read_to_string(&mut response).unwrap();response
            }).collect::<Vec<_>>()
        });
        let limits=Limits {input_tokens:16383,context_tokens:16384,output_tokens:16383,..Default::default()};
        let cache=CacheLimits {entries:8,retained_bytes:POOL-FULL,tokens_per_entry:16384,logits_per_entry:vocab,idle_ttl:Duration::from_secs(300)};
        let config=scheduler::Config {active:1,connections:2,prefill_chunk:256,foreground_rows:1,background_rows:1,active_bytes:POOL,session_budget_bytes:POOL,
            draft:protocol::DraftMode::Plain,mtp_depth:0,draft_global_bank:false,prefix_cache:true,sharing:Default::default()};
        scheduler::serve_requests(listener,&model,Some(&Codec),&limits,&cache,&config,&AtomicBool::new(false),"fixture",9).unwrap();
        clients.join().unwrap()
    });(responses,ledger)
}
#[test]
fn eight_real_charge_snapshots_evict_before_full_context_scheduler_admission() {
    struct Scratch(std::path::PathBuf);impl Drop for Scratch {fn drop(&mut self) {std::fs::remove_dir_all(&self.0).unwrap();}}
    let root=Scratch(std::env::temp_dir().join(format!("vade-kept-admission-{}",std::process::id())));
    let mut small=fixture::description();small.max_positions=16384;
    let index=fixture::checkpoint(&root.0,&small).unwrap();let checkpoint=small.open_headers(&root.0,&index).unwrap();
    let ConfigPlan::Deferred(real)=ConfigPlan::from_config(include_bytes!("../../../model/tests/fixtures/glm/config.json")).unwrap() else {panic!()};
    let mut predictor=real.with_draft_layer().unwrap().unwrap();predictor.residual=real.residual;
    let old=crate::admission_plan::StateChargePlan::new(&predictor,16384).unwrap();
    let plain=crate::admission_plan::SessionChargePlan::for_sparse_heads(&real,Topology {rank:0,parts:2},16384).unwrap();let short=plain.session(256,257).unwrap();
    let mut factory=Factory::new(&checkpoint,Split::new(crate::bounded_decoder::Host,Local).unwrap(),16384,vec![0]).unwrap();
    factory.charge_plan=plain;factory.with_layer=Some(crate::admission_plan::StateChargePlan::new(&predictor,16384).unwrap());factory.kept_tail=super::kept_charge::Tail::new(&real,Topology {rank:0,parts:2}).unwrap();
    // One host allocation supplies the measured device owner's byte total.
    // The scheduler model below tracks each publication as a separate holder.
    // Use the production parked-charge method, so restoring the loose invoice
    // makes this exact-capacity and admission regression fail.
    let state=State {position:256,layers:vec![BTreeMap::from([("owners".into(),vec![0.0;OWNER/4])])],expanded:BTreeMap::new()};
    let parked=crate::bounded_decoder::park_state(&crate::bounded_decoder::Host,&state).unwrap();drop(state);
    let (responses,ledger)=run(&|_|factory.parked_charge(&parked),short,real.vocab as usize);
    assert!(responses.iter().all(|r|r.starts_with("HTTP/1.1 200")),"{responses:?}");
    let (retained,reserved,evicted)=ledger.full.borrow().clone().expect("the scheduler opened the full-context request");
    assert_eq!((retained,reserved),(6*OWNER,FULL));
    for id in 1..=2 {assert!(evicted.contains(&id),"prompt {id} was not evicted before admission");}
    assert!(retained+reserved<=POOL);
    println!("KEPT ADMISSION HOST charge=exact published=8 retained_owners={} evicted_before_open={} full_context=16384 reserved_bytes={reserved} retained_owner_bytes={retained} combined_bytes={} pool_bytes={POOL} request_opened=true",retained/OWNER,evicted.len(),retained+reserved);
    let (responses,ledger)=run(&|position|factory.kept_charge(position),short,real.vocab as usize);
    assert!(responses.iter().all(|r|r.starts_with("HTTP/1.1 200")));
    let (retained,reserved,evicted)=ledger.full.borrow().clone().unwrap();assert_eq!((retained,reserved,evicted.len()),(4*OWNER,FULL,4));
    println!("KEPT ADMISSION HOST charge=loose published=8 retained_owners={} evicted_before_open={} combined_bytes={} request_opened=true",retained/OWNER,evicted.len(),retained+reserved);
    let (responses,ledger)=run(&|position|old.bytes(position as usize).unwrap()+old.append_reservation_bytes(),short,real.vocab as usize);
    assert!(responses[..8].iter().all(|r|r.starts_with("HTTP/1.1 200")));
    assert!(!responses[8].starts_with("HTTP/1.1 200"),"the old charge must not pass the physical admission check");
    let (retained,reserved,evicted)=ledger.full.borrow().clone().unwrap();
    assert_eq!((retained,reserved,evicted.len()),(8*OWNER,FULL,0));assert!(retained+reserved>POOL);
    println!("KEPT ADMISSION HOST charge=main published=8 retained_owners={} evicted_before_open={} full_context=16384 reserved_bytes={reserved} retained_owner_bytes={retained} combined_bytes={} pool_bytes={POOL} request_opened=false",retained/OWNER,evicted.len(),retained+reserved);
}
