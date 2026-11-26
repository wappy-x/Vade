//! GROUP SPARES: a service tells its compute, before each prompt group, whether the prompt goes on after that group
//! (`Compute::rows_follow`), and tells it when a session leaves (`Compute::rows_settle`). A device keeps the vector
//! pool's spares at the end of a group whose prompt goes on, and trims them when a session leaves. What this does NOT
//! run: the device's pool. That needs a pair.
use engine_model::{plan::layer::Topology,safetensors::Dtype};
use spark_tp::{row_split::{Exchange,Factory,Split},Result};
use spark_serve::{Engine,Session};
#[allow(dead_code)] #[path="support/attention_checkpoint.rs"] mod fixture;
#[allow(dead_code)] #[path="support/watch_compute.rs"] mod watch;
struct Local;
impl Exchange for Local {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,v:&[f32],_:usize)->Result<Vec<f32>> {Ok(v.to_vec())}
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
#[test]
fn a_service_names_each_group_whose_prompt_goes_on_and_settles_when_a_session_leaves() {
    let c=fixture::Checkpoint::new_with_capacity(Dtype::Bf16,true,80);let compute=watch::Watch::default();
    let engine=Factory::new(&c.admission,Split::new(&compute,Local).unwrap(),80,vec![0]).unwrap();
    let bits=|row:Vec<f32>|row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    // One prompt read in three parts: the first two do not end it, the last one does. Then the same prompt in one part.
    let parts={let mut s=engine.session().unwrap();s.prefill_advance(&[1,3,2]).unwrap();s.prefill_advance(&[5,7]).unwrap();
        assert_eq!(*compute.follows.borrow(),[true,true]);assert_eq!(compute.settles.get(),0,"no session left yet");
        let last=s.prefill(&[4]).unwrap();assert_eq!(*compute.follows.borrow(),[true,true,false]);
        let row=bits(s.step(8).unwrap());assert_eq!(compute.settles.get(),0);drop(s);(bits(last),row)};
    assert_eq!(compute.settles.get(),1,"a session that leaves settles the spares once");
    let whole={let mut s=engine.session().unwrap();let last=s.prefill(&[1,3,2,5,7,4]).unwrap();
        assert_eq!(*compute.follows.borrow(),[true,true,false,false]);(bits(last),bits(s.step(8).unwrap()))};
    assert_eq!(compute.settles.get(),2);
    // The words do not depend on where the prompt was cut or on what its groups were told.
    assert_eq!(parts,whole);
    // A session that leaves in the middle of its prompt settles too: its last group was told the prompt goes on.
    {let mut s=engine.session().unwrap();s.prefill_advance(&[1,3,2]).unwrap();assert_eq!(compute.follows.borrow().last(),Some(&true));}
    assert_eq!(compute.settles.get(),3);
    engine.shutdown().unwrap();
}
