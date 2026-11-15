use super::*;
use crate::bounded_decoder::{Host,TestCheckpoint};
use engine_model::safetensors::Dtype;
use spark_serve::{Engine as _,Session as _};
#[derive(Default)]
struct Observed {spares:RefCell<Vec<u8>>,returns:Cell<usize>,fail:Cell<bool>}
// Delegate arithmetic to the host twin. Only scratch ownership and faults differ.
macro_rules! host {($(fn $name:ident($($arg:ident:$ty:ty),*)->$out:ty;)*)=>{$(
    fn $name(&self,$($arg:$ty),*)->$out {Host.$name($($arg),*)}
)*};}
impl Compute for Observed {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    host! {
        fn mhc_row_launches()->u64;
        fn prepare_beside(c:&CheckpointAdmission,live:usize)->Result<()>;
        fn rows_greedy(x:&Self::Value,rows:usize)->Result<Option<Vec<u32>>>;
        fn has_latent_absorption(name:&str)->bool;
        fn latent_absorption(name:&str,s:Shape,a:&Self::Value,b:&Self::Value,c:&Self::Value,n:usize)->Result<Self::Value>;
        fn latent_absorption_rows(name:&str,s:spark_gpu_probe::prompt_ops::Shape,a:&Self::Value,b:&Self::Value,c:&Self::Value)->Result<Self::Value>;
        fn latent_absorption_chats(name:&str,s:spark_gpu_probe::prompt_ops::Shape,a:&[crate::bounded_decoder::ChatGather<'_,Self::Value>],n:usize,b:&Self::Value,c:&Self::Value)->Result<Option<Self::Value>>;
        fn upload(x:&[f32])->Result<Self::Value>;
        fn read(x:&Self::Value)->Result<Vec<f32>>;
        fn select_known(a:&Self::Value,b:&Self::Value,x:&[f32])->Result<()>;
        fn range(x:&Self::Value,start:usize,len:usize)->Result<Self::Value>;
        fn words(x:&Self::Value)->usize;
        fn join(x:&[&Self::Value])->Result<Self::Value>;
        fn zeros(n:usize)->Result<Self::Value>;
        fn operation(s:Shape,x:[Option<&Self::Value>;4],w:&[f32])->Result<Self::Value>;
        fn gather_absorbed(s:Shape,x:[Option<&Self::Value>;4],n:usize,f:f32)->Result<Self::Value>;
        fn norm(name:&str,s:NormShape,x:&Self::Value,w:&[f32])->Result<Self::Value>;
        fn begin(key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value)->Result<(Self::Pass,Self::Value)>;
        fn finish(p:Self::Pass,x:&Self::Value)->Result<Self::Value>;
        fn mx(p:&Mxfp8Projection<'_>,x:&Self::Value)->Result<Self::Value>;
        fn round(x:&Self::Value)->Result<Self::Value>;
        fn mean(x:&Self::Value,streams:u32,hidden:u32)->Result<Self::Value>;
        fn prompt(s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Self::Value>;5],w:&[f32])->Result<Self::Value>;
        fn prompt_end()->Result<()>;
        fn prompt_seeds()->bool;
        fn rows_norm(name:&str,s:NormShape,x:&Self::Value,m:usize,w:&[f32])->Result<Self::Value>;
        fn rows_begin(key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Self::Value,m:usize)->Result<(Self::Pass,Self::Value)>;
        fn rows_finish(p:Self::Pass,x:&Self::Value,m:usize)->Result<Self::Value>;
    }
    fn state_allocations(&self,v:&Self::Value,out:&mut Vec<(usize,usize)>)->bool {Host.state_allocations(v,out)}
    fn memory_bytes(&self)->Option<(usize,usize,usize)> {Some((self.spares.borrow().capacity(),0,0))}
    fn rows_keep(&self)->Result<()> {self.spares.replace(Vec::new());self.returns.set(self.returns.get()+1);Ok(())}
    fn rows_embedding(&self,_:&[u32],_:usize)->Result<Option<Self::Value>> {
        self.spares.replace(vec![0;4096]);
        if self.fail.replace(false) {return Err("injected prompt allocation failure".into());}Ok(None)
    }
}
struct Wire {peer:Cell<usize>,commands:RefCell<Vec<u32>>}
impl Wire {fn new()->Self {Self {peer:Cell::new(1024),commands:RefCell::new(Vec::new())}}}
impl Exchange for Wire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],n:usize)->Result<Vec<f32>> {
        assert_eq!(local.len(),n);let mut out=local.to_vec();
        if n==4 {let own=(0..4).map(|i|(local[i] as u64)<<(16*i)).sum::<u64>() as usize;let min=own.min(self.peer.get());
            for (i,v) in out.iter_mut().enumerate() {*v=((min as u64>>(16*i))&65535) as f32;}}
        Ok(out)
    }
    fn command(&mut self,words:&mut [u8])->Result<()> {if words.len()==16 {self.commands.borrow_mut().push(u32::from_le_bytes(words[..4].try_into().unwrap()));}Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
fn engine(c:&CheckpointAdmission)->Factory<'_,Observed,Wire> {
    let mut e=Factory::new(c,Split::new(Observed::default(),Wire::new()).unwrap(),8193,vec![0]).unwrap().with_prompt_keep();
    // Test geometry has no draft layer. Exercise the workspace policy without
    // enabling predictor arithmetic in the tiny host model.
    e.predictor=3;e.memo_admitted=true;e.init_prompt_lease_plan().unwrap();e
}
#[test]
fn returned_lease_releases_actual_host_scratch_allocation() {
    let c=checkpoint();let e=engine(&c.admission);let mut s=e.session().unwrap();
    let workspace=e.workspace;let bytes=e.prompt_lease_extra(1024).unwrap();e.prepare_prompt_workspace(bytes).unwrap();assert_eq!(s.prefill_chunk_size(8192),1024);
    s.prefill_advance(&vec![1;1024]).unwrap();assert_eq!(e.split.compute.spares.borrow().capacity(),0);assert!(e.split.compute.returns.get()>0);
    assert_eq!((e.lease_bytes.get(),e.lease_rows.get(),e.workspace),(0,0,workspace));assert_eq!(s.prefill_chunk_size(8192),256);
}
#[test]
fn unequal_rank_offers_choose_smaller_and_eight_chats_cannot_borrow() {
    let c=checkpoint();let e=engine(&c.admission);let a=e.session().unwrap();
    e.split.exchange.borrow().peer.set(512);e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),512);e.finish_prompt_workspace().unwrap();
    e.split.exchange.borrow().peer.set(1024);let others=(0..7).map(|_|e.session().unwrap()).collect::<Vec<_>>();
    e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),256);assert_eq!(e.lease_bytes.get(),0);drop(others);drop(a);
}
#[test]
fn arrival_after_large_group_has_original_pool_and_group() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    e.prepare_prompt_workspace(usize::MAX).unwrap();a.prefill_advance(&vec![1;1024]).unwrap();
    // Scheduler intake resumes only after this synchronous call. The arriving
    // session must see freed scratch, not just a lower ledger charge.
    let b=e.session().unwrap();assert_eq!(e.split.compute.spares.borrow().capacity(),0);
    e.prepare_prompt_workspace(0).unwrap();assert_eq!(a.prefill_chunk_size(8192),256);drop(b);
}
#[test]
fn unused_offer_cancel_and_mid_prompt_error_release_once() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    e.prepare_prompt_workspace(usize::MAX).unwrap();let n=e.split.compute.returns.get();
    e.finish_prompt_workspace().unwrap();e.finish_prompt_workspace().unwrap();assert_eq!(e.split.compute.returns.get(),n+1);
    e.prepare_prompt_workspace(usize::MAX).unwrap();e.split.compute.fail.set(true);
    assert!(a.prefill_advance(&vec![1;1024]).unwrap_err().contains("injected"));
    assert_eq!(e.split.compute.spares.borrow().capacity(),0);assert_eq!(e.lease_bytes.get(),0);assert!(e.failed.get());
}
#[test]
fn cached_8192_resume_borrows_only_for_its_suffix() {
    let c=checkpoint();let e=engine(&c.admission);let mut a=e.session().unwrap();
    for _ in 0..8 {e.prepare_prompt_workspace(usize::MAX).unwrap();a.prefill_advance(&vec![1;1024]).unwrap();}
    let kept=a.fork().unwrap();drop(a);let mut resumed=kept.fork().unwrap();
    assert_eq!(resumed.state.as_ref().unwrap().position,8192);
    e.prepare_prompt_workspace(usize::MAX).unwrap();resumed.prefill(&[1]).unwrap();
    assert_eq!(resumed.state.as_ref().unwrap().position,8193);assert_eq!(e.lease_bytes.get(),0);assert_eq!(e.split.compute.spares.borrow().capacity(),0);
}
#[test]
fn predictor_off_never_sends_lease_command() {
    let c=checkpoint();let mut e=engine(&c.admission);e.predictor=0;let a=e.session().unwrap();
    e.prepare_prompt_workspace(usize::MAX).unwrap();assert_eq!(e.current_prompt_group(),256);
    assert!(!e.split.exchange.borrow().commands.borrow().contains(&25));drop(a);
}
#[test]
fn exact_remainder_threshold_preserves_the_startup_workspace_charge() {
    let c=checkpoint();let e=engine(&c.admission);let a=e.session().unwrap();
    let base=e.workspace;let half=e.prompt_lease_extra(512).unwrap();let full=e.prompt_lease_extra(1024).unwrap();assert!(full>half && half>0);
    for (room,rows) in [(half-1,256),(half,512),(full-1,512),(full,1024)] {
        e.prepare_prompt_workspace(room).unwrap();assert_eq!(e.current_prompt_group(),rows);assert_eq!(e.workspace,base);
        e.finish_prompt_workspace().unwrap();assert_eq!(e.lease_bytes.get(),0);
    }drop(a);
}

fn checkpoint()->TestCheckpoint {
    TestCheckpoint::build_with(Dtype::Bf16,true,8193,false,|d| {
        d.hidden=32;d.layers.truncate(1);
        if let engine_model::plan::description::TokenMixer::Kda(k)=&mut d.layers[0].mixer {k.heads=1;}
    })
}
#[test]
fn memo_startup_prepares_the_immutable_lease_plan() {
    let c=checkpoint();let mut e=engine(&c.admission);let expected=e.lease_extra;
    e.memo_admitted=false;e.lease_extra=[0;2];let e=e.with_memo_bound().unwrap();
    assert_eq!(e.lease_extra,expected);assert!(expected.iter().all(|&bytes|bytes>0));
    let a=e.session().unwrap();e.prepare_prompt_workspace(expected[1]).unwrap();assert_eq!(e.current_prompt_group(),1024);
    e.finish_prompt_workspace().unwrap();drop(a);
}
