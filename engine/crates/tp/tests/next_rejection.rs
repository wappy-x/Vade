use spark_tp::{next_http::{Causal,Commands,Engine,Runtime},Result};
use spark_serve::{Engine as _,Session as _,BatchRows,BatchOutput};
use std::{cell::RefCell,rc::Rc};
struct Fake;
impl Causal for Fake {fn token(&mut self,t:u32)->Result<u32> {Ok((t+1)%16)}}
#[derive(Default)]
struct Trace {frames:usize,rejections:usize}
struct Wire(Rc<RefCell<Trace>>);
impl Commands for Wire {
    fn broadcast(&mut self,_:&mut [u8])->Result<()> {self.0.borrow_mut().frames+=1;Ok(())}
    fn verdict(&mut self,ok:bool)->Result<()> {if !ok {self.0.borrow_mut().rejections+=1;return Err("rank rejected".into());}Ok(())}
}
#[test]
fn invalid_chunk_does_not_publish_or_poison_other_sessions() {
    let trace=Rc::new(RefCell::new(Trace::default()));
    let engine=Engine::new(Wire(trace.clone()),Runtime::new(16,Box::new(|_|Ok(Fake))),vec![0],0);
    let mut healthy=engine.session().unwrap();let mut bad=engine.session().unwrap();
    let before=trace.borrow().frames;
    assert!(bad.prefill(&[16]).is_err());
    assert_eq!(trace.borrow().frames,before,"invalid chunk published a command");
    assert_eq!(trace.borrow().rejections,0);assert_eq!(engine.fatal_error(),None);
    assert_eq!(healthy.step(7).unwrap()[8],0.0);
}
#[test]
fn invalid_batch_does_not_publish_or_poison_other_sessions() {
    let trace=Rc::new(RefCell::new(Trace::default()));
    let engine=Engine::new(Wire(trace.clone()),Runtime::new(16,Box::new(|_|Ok(Fake))),vec![0],0);
    let mut healthy=engine.session().unwrap();let mut bad=engine.session().unwrap();
    let before=trace.borrow().frames;
    let mut rows=[BatchRows {session:&mut healthy,tokens:&[7],output:BatchOutput::Last,greedy:true,top_k:0},
        BatchRows {session:&mut bad,tokens:&[16],output:BatchOutput::Last,greedy:true,top_k:0}];
    assert!(engine.step_batch(&mut rows).is_err());
    assert_eq!(trace.borrow().frames,before);assert_eq!(trace.borrow().rejections,0);
    assert_eq!(engine.fatal_error(),None);assert_eq!(healthy.step(7).unwrap()[8],0.0);
}
