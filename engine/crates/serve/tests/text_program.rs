use spark_serve::{Engine,Session,Tokenizer,Decoder,text_program::{Factory,TokenProgram,TextEngine}};
use engine_model::{plan::{ConfigPlan,TextDescription,description::DecoderDescription,text_edges::TextEdges},safetensors::{Tensor,Dtype}};
use std::{collections::BTreeMap,rc::Rc,cell::RefCell};
struct Codec(Vec<u32>);
struct Decode;
impl Decoder for Decode {fn push(&mut self,_:u32)->Result<String,String>{Ok(String::new())} fn finish(&mut self)->Result<String,String>{Ok(String::new())}}
impl Tokenizer for Codec {
    type Decoder<'a> = Decode;
    fn encode(&self,_:&str)->Result<Vec<u32>,String>{Ok(vec![0])}
    fn decoder(&self)->Decode{Decode}
    fn has_token(&self,id:u32)->bool{id<8}
    fn checkpoint_eos_token_ids(&self)->Result<Option<&[u32]>,String>{Ok(Some(&self.0))}
    fn encode_chat(&self,_:&spark_serve::protocol::ChatInput,_:usize)->Result<Vec<u32>,String>{Ok(vec![0])}
}
#[derive(Clone)]
struct Mock {log:Rc<RefCell<Vec<String>>>,fail:&'static str}
impl Mock {fn event(&self,name:String)->Result<(),String>{self.log.borrow_mut().push(name.clone());if self.fail==name{Err(name)}else{Ok(())}}}
impl Factory for Mock {
    type Program<'a> = Self;
    fn create(&self,_:&TextDescription,_:&TextEdges,_:usize)->Result<Self,String>{self.event("create".into())?;Ok(self.clone())}
}
impl TokenProgram for Mock {
    type Residual = (); type Hidden = ();
    fn embed(&mut self,id:u32,pos:usize)->Result<(),String>{self.event(format!("embed:{id}:{pos}"))}
    fn layer(&mut self,i:usize,_:&DecoderDescription,_:())->Result<(),String>{self.event(format!("layer:{i}"))}
    fn mean(&mut self,_:())->Result<(),String>{self.event("mean".into())}
    fn norm(&mut self,_:())->Result<(),String>{self.event("norm".into())}
    fn head(&mut self,_:())->Result<Vec<f32>,String>{self.event("head".into())?;Ok(match self.fail {"nan"=>vec![f32::NAN;8],"short"=>vec![0.0;7],_=>{let mut v=vec![0.0;8];v[3]=1.0;v}})}
    fn commit(&mut self)->Result<(),String>{self.event("commit".into())}
    fn abort(&mut self)->Result<(),String>{self.event("abort".into())}
}
fn source()->(TextDescription,BTreeMap<String,Tensor>){
    let ConfigPlan::Deferred(mut d)=ConfigPlan::from_config(include_bytes!("../../model/tests/fixtures/glm/config.json")).unwrap() else{panic!()};
    d.hidden=32;d.vocab=8;d.eos_tokens=vec![3,5,6];d.layers.truncate(2);
    let h=[("model.language_model.embed_tokens.weight",Dtype::Bf16,vec![8,32]),("model.language_model.norm.weight",Dtype::F32,vec![32]),
        ("lm_head.weight",Dtype::F8E4M3,vec![8,32]),("lm_head.weight_scale",Dtype::U8,vec![8,1])].into_iter()
        .map(|(n,dtype,shape)|{let end=shape.iter().product::<u64>()*dtype.bytes();(n.into(),Tensor{dtype,shape,start:0,end})}).collect();(d,h)
}
fn engine(fail:&'static str)->(TextEngine<Mock>,Rc<RefCell<Vec<String>>>){
    let log=Rc::new(RefCell::new(Vec::new()));let factory=Mock{log:log.clone(),fail};let(d,h)=source();
    (TextEngine::new(factory,d,&h,&Codec(vec![6,3,5]),2).unwrap(),log)
}
#[test]
fn full_token_order_and_context_are_fixed(){
    let(e,log)=engine("");assert_eq!(e.eos_token_ids(),[3,5,6]);let mut s=e.session().unwrap();
    assert!(!s.can_fork());assert!(s.truncate(1).is_err());assert!(s.step(8).is_err());
    s.step(1).unwrap();s.step(2).unwrap();let before=log.borrow().len();assert!(s.step(0).is_err());assert_eq!(log.borrow().len(),before);
    assert_eq!(*log.borrow(),["create","embed:1:0","layer:0","layer:1","mean","norm","head","commit",
        "embed:2:1","layer:0","layer:1","mean","norm","head","commit"]);
    assert!(s.truncate(2).is_ok());assert!(s.truncate(0).is_err());
}
#[test]
fn every_operator_error_aborts_and_retires_session(){
    for fail in ["embed:1:0","layer:0","layer:1","mean","norm","head","commit","nan","short"]{
        let(e,log)=engine(fail);let mut s=e.session().unwrap();assert!(s.step(1).is_err());assert_eq!(log.borrow().last().unwrap(),"abort");
        let before=log.borrow().len();assert!(s.step(1).is_err());assert!(s.truncate(0).is_err());assert_eq!(log.borrow().len(),before);
        if matches!(fail,"nan"|"short"){assert!(!log.borrow().iter().any(|x|x=="commit"));}
    }
}
#[test]
fn validates_headers_and_complete_stop_set_before_factory(){
    for eos in [vec![],vec![3],vec![3,5,6,7],vec![3,3,6]]{
        let(d,h)=source();let log=Rc::new(RefCell::new(Vec::new()));let factory=Mock{log:log.clone(),fail:""};
        assert!(TextEngine::new(factory,d,&h,&Codec(eos),2).is_err());assert!(log.borrow().is_empty());
    }
    let(d,mut h)=source();h.remove("lm_head.weight_scale");let factory=Mock{log:Default::default(),fail:""};
    assert!(TextEngine::new(factory,d,&h,&Codec(vec![3,5,6]),2).is_err());
}
#[test]
fn shared_generator_stops_on_the_bound_program_eos(){
    use spark_serve::generation::{self,Event,FinishReason};
    let(e,log)=engine("");let mut events=Vec::new();
    let usage=generation::generate(&e,&[1],1,&Default::default(),||Ok(()),|event|{events.push(event);Ok(())}).unwrap();
    assert_eq!(usage.completion_tokens,1);assert!(!events.iter().any(|e|matches!(e,Event::Token(_))));
    assert_eq!(events.last(),Some(&Event::Finished(FinishReason::Stop,usage)));
    assert_eq!(log.borrow().last().unwrap(),"commit");
}
