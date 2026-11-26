use super::*;
use crate::bounded_decoder::{Host,TestCheckpoint};
use engine_model::safetensors::Dtype;
use spark_serve::{BatchOutput,BatchRows,Engine as _,Session as _};
std::thread_local! {pub(super) static VALIDATIONS:Cell<usize>=const {Cell::new(0)};}
struct Wire {batches:usize}
impl Exchange for Wire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
    fn command(&mut self,words:&mut [u8])->Result<()> {
        if words.len()==16 && u32::from_le_bytes(words[..4].try_into().unwrap())==13 {self.batches+=1;}Ok(())
    }
    fn complete(&mut self)->Result<()> {Ok(())}
}
#[test]
fn riding_rows_validate_once_before_the_shared_batch() {
    let checkpoint=TestCheckpoint::new_with_capacity(Dtype::Bf16,true,80);
    let split=Split::new(Host,Wire {batches:0}).unwrap();
    let engine=Factory::new(&checkpoint.admission,split,80,vec![0]).unwrap();
    let mut a=engine.session().unwrap();let mut b=engine.session().unwrap();
    a.prefill(&[3,5,1]).unwrap();b.prefill(&[2,4]).unwrap();
    VALIDATIONS.with(|count|count.set(0));
    let mut rows=[BatchRows {session:&mut a,tokens:&[7],output:BatchOutput::All,greedy:true,top_k:0},
        BatchRows {session:&mut b,tokens:&[2],output:BatchOutput::All,greedy:true,top_k:0}];
    let result=engine.step_batch(&mut rows).map_err(|error|error.message);
    assert_eq!(VALIDATIONS.with(Cell::get),rows.len(),"each riding row validates once");
    assert_eq!(result.unwrap().len(),rows.len());
    assert_eq!(engine.split.exchange.borrow().batches,1,"the rows must take the shared batch path");
    drop(a);drop(b);engine.shutdown().unwrap();
}
