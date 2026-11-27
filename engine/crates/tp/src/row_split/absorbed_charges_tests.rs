//! The charges of a server whose latent layers are all absorbed, through the real startup calls of `Factory`.
use super::*;
use crate::bounded_decoder::{Host,LatentRows,TestCheckpoint,group_workspace_bytes,group_workspace_bytes_with,PROMPT_GROUP_MAX};
use engine_model::safetensors::Dtype;
/// The host twin, except that it answers the load question the way a device does: `base` for the checkpoint's latent
/// layers, `draft` for the predictor layer (the names under `prefix`). No test here runs a row.
struct Loaded {base:Cell<bool>,draft:Cell<bool>,prefix:String,memo:Option<crate::projection_memo::Summary>}
macro_rules! host {($(fn $name:ident($($arg:ident:$ty:ty),*)->$out:ty;)*)=>{$(
    fn $name(&self,$($arg:$ty),*)->$out {Host.$name($($arg),*)}
)*};}
impl Compute for Loaded {
    type Value=Vec<f32>;type Pass=<Host as Compute>::Pass;
    host! {
        fn mhc_row_launches()->u64;
        fn prepare_beside(c:&CheckpointAdmission,live:usize)->Result<()>;
        fn rows_greedy(x:&Self::Value,rows:usize)->Result<Option<Vec<u32>>>;
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
    fn has_latent_absorption(&self,name:&str)->bool {if name.starts_with(&self.prefix) {self.draft.get()} else {self.base.get()}}
    fn memo_summary(&self,_:&CheckpointAdmission)->Option<crate::projection_memo::Summary> {self.memo}
    fn freeze_memo(&self,plan:crate::projection_memo::Plan)->Result<crate::projection_memo::Plan> {Ok(plan)}
    fn state_allocations(&self,v:&Self::Value,out:&mut Vec<(usize,usize)>)->bool {Host.state_allocations(v,out)}
}
/// Two ranks: every gathered word comes back as the smaller of this rank's and `peer`.
struct Wire {peer:Cell<usize>}
impl Exchange for Wire {
    fn topology(&self)->Topology {Topology {rank:0,parts:1}}
    fn gather(&mut self,local:&[f32],n:usize)->Result<Vec<f32>> {
        assert_eq!(local.len(),n);let mut out=local.to_vec();
        if n==4 {let own=(0..4).map(|i|(local[i] as u64)<<(16*i)).sum::<u64>() as usize;let min=own.min(self.peer.get());
            for (i,v) in out.iter_mut().enumerate() {*v=((min as u64>>(16*i))&65535) as f32;}}
        Ok(out)
    }
    fn command(&mut self,_:&mut [u8])->Result<()> {Ok(())}
    fn complete(&mut self)->Result<()> {Ok(())}
}
const CONTEXT:usize=8193;
fn checkpoint(predictor:bool)->TestCheckpoint {TestCheckpoint::build_with(Dtype::Bf16,true,CONTEXT as u64,predictor,|_|{})}
fn engine(c:&CheckpointAdmission,base:bool,draft:bool,peer:usize)->Factory<'_,Loaded,Wire> {
    engine_at(c,base,draft,peer,CONTEXT)
}
fn engine_at(c:&CheckpointAdmission,base:bool,draft:bool,peer:usize,context:usize)->Factory<'_,Loaded,Wire> {
    let prefix=format!("model.language_model.layers.{}.",c.description().layers.len());
    Factory::new(c,Split::new(Loaded {base:Cell::new(base),draft:Cell::new(draft),prefix,memo:None},Wire {peer:Cell::new(peer)}).unwrap(),context,vec![0]).unwrap()
}
#[test]
fn prompt_group_requires_the_plain_floor_after_the_memo_bound() {
    use crate::{projection_memo::{FULL,Plan,Summary},service_memory::{with_memo_bound,workspace_for_prompt_group_with}};
    let c=checkpoint(false);
    let summary=Summary {unabsorbed:1,selection:FULL/12-1,chunk:1};
    let make=|| {
        let mut e=engine(&c.admission,true,true,usize::MAX).with_absorbed_charges().unwrap();
        e.split.compute.memo=Some(summary);
        e.reservation=1;
        e
    };
    let e=make();
    let plan=Plan::new(Some(summary),LIVE_SESSIONS);
    let floor=LIVE_SESSIONS*e.charge_plan.request(PROMPT_GROUP_MAX/LIVE_SESSIONS,PROMPT_GROUP_MAX).unwrap();
    let threshold=|rows| {
        let before=workspace_for_prompt_group_with(&c.admission,CONTEXT,e.workspace,rows,e.latent_rows).unwrap();
        let after=with_memo_bound(before,plan).unwrap();
        assert!(after<before && e.resident+after+floor>e.resident+before+e.reservation);
        e.resident+after+floor
    };
    let (half,wide)=(threshold(512),threshold(1024));
    for (budget,rows) in [(0,256),(half-1,256),(half,512),(wide-1,512),(wide,1024)] {
        let (e,actual,pool)=make().with_prompt_group_and_memo(budget).unwrap();
        assert_eq!((e.group,actual,pool),(rows,floor,budget.saturating_sub(e.resident).saturating_sub(e.workspace)));
    }
}
#[test]
fn prompt_group_still_requires_the_full_session_charge() {
    use crate::service_memory::workspace_for_prompt_group_with;
    let c=checkpoint(false);
    let mut e=engine(&c.admission,true,true,usize::MAX).with_absorbed_charges().unwrap();
    let wide=workspace_for_prompt_group_with(&c.admission,CONTEXT,e.workspace,PROMPT_GROUP_MAX,e.latent_rows).unwrap();
    let half=workspace_for_prompt_group_with(&c.admission,CONTEXT,e.workspace,PROMPT_GROUP_MAX/2,e.latent_rows).unwrap();
    let floor=e.prompt_group_pool_floor().unwrap();
    // The memo keeps its full charge in this fixture. The wider group meets the floor exactly.
    assert_eq!(e.memo_plan().unwrap(),crate::projection_memo::Plan::full());
    e.reservation=floor+1;
    let budget=e.resident+wide+floor;
    assert!(budget<e.resident+wide+e.reservation);
    assert!(budget>=e.resident+half+e.reservation);
    let (e,actual_floor,pool)=e.with_prompt_group_and_memo(budget).unwrap();
    assert_eq!((e.group,actual_floor,pool),(PROMPT_GROUP_MAX/2,floor,budget-e.resident-half));
}
#[test]
fn short_context_keeps_the_widest_group_plain_floor() {
    let c=checkpoint(false);
    for absorbed in [false,true] {
        let full=engine(&c.admission,absorbed,true,usize::MAX).with_absorbed_charges().unwrap();
        let expected=LIVE_SESSIONS*full.charge_plan.request(PROMPT_GROUP_MAX/LIVE_SESSIONS,PROMPT_GROUP_MAX).unwrap();
        let short=engine_at(&c.admission,absorbed,true,usize::MAX,PROMPT_GROUP_MAX/2).with_absorbed_charges().unwrap();
        assert_eq!(short.latent_rows,full.latent_rows);
        assert!(short.charge_plan.request(PROMPT_GROUP_MAX/LIVE_SESSIONS,PROMPT_GROUP_MAX).is_err());
        let (e,floor,pool)=short.with_prompt_group_and_memo(usize::MAX).unwrap();
        assert_eq!((e.group,floor,pool),(PROMPT_GROUP_MAX,expected,usize::MAX-e.resident-e.workspace));
    }
}
#[test]
fn a_server_that_expands_rows_keeps_every_charge() {
    let c=checkpoint(false);let e=engine(&c.admission,false,true,usize::MAX);
    let (workspace,reservation,short)=(e.workspace,e.reservation,e.prompt_state_reservation(512,640).unwrap());
    let e=e.with_absorbed_charges().unwrap();
    assert_eq!((e.latent_rows,e.workspace,e.reservation,e.prompt_state_reservation(512,640).unwrap()),(LatentRows::Expanded,workspace,reservation,short));
}
#[test]
fn an_absorbed_server_is_not_charged_the_rows_it_never_makes() {
    let c=checkpoint(false);let e=engine(&c.admission,true,true,usize::MAX);
    let (workspace,reservation,short)=(e.workspace,e.reservation,e.prompt_state_reservation(512,640).unwrap());
    let held=e.charge_plan.expanded_bytes(512);assert!(held>0,"the fixture charges no expanded rows: the test proves nothing");
    let e=e.with_absorbed_charges().unwrap();let rows=LatentRows::Absorbed {draft:false};assert_eq!(e.latent_rows,rows);
    // The workspace: the group's own buffers in their absorbed form, nothing else.
    let (old,new)=(group_workspace_bytes(&c.admission,CONTEXT).unwrap(),group_workspace_bytes_with(&c.admission,CONTEXT,256,rows).unwrap());
    assert!(new<old);assert_eq!(e.workspace,workspace-old+new);
    // A request: no held rows, and still both copies of its state.
    assert_eq!(e.charge_plan.expanded_bytes(512),0);assert_eq!(e.charge_plan.expanded_bytes(CONTEXT),0);
    // A short prompt was charged its held rows once or twice (the late or the early peak): those bytes are gone, and
    // nothing else is. What is left still covers both copies of the state and the logits.
    let now=e.prompt_state_reservation(512,640).unwrap();
    assert!(short-2*held<=now && now<=short-held,"released {} of {held} held bytes",short-now);
    assert!(now>=2*e.charge_plan.state_bytes(640).unwrap()+e.charge_plan.append_reservation_bytes());
    assert!(e.reservation<reservation && e.reservation>=2*e.charge_plan.state_bytes(CONTEXT).unwrap());
    assert_eq!(e.reservation,e.prompt_state_reservation(CONTEXT,CONTEXT).unwrap());
    // The prompt group and the lease read the same form.
    let budget=usize::MAX;let e=e.negotiate_prompt_group(budget).unwrap();assert_eq!(e.group,PROMPT_GROUP_MAX);
    let copies=crate::service_memory::state_copy_bytes(&c.admission).unwrap();
    assert_eq!(e.workspace,workspace-old+group_workspace_bytes_with(&c.admission,CONTEXT,PROMPT_GROUP_MAX,rows).unwrap()+copies);
    // Set once, and before the group.
    assert!(e.with_absorbed_charges().is_err());
}
#[test]
fn one_rank_that_expands_keeps_every_charge_on_both() {
    // This rank absorbed every layer; the other rank's word is 0.
    let c=checkpoint(false);let e=engine(&c.admission,true,true,0);let (workspace,reservation)=(e.workspace,e.reservation);
    let e=e.with_absorbed_charges().unwrap();assert_eq!((e.latent_rows,e.workspace,e.reservation),(LatentRows::Expanded,workspace,reservation));
}
#[test]
fn an_unabsorbed_predictor_layer_keeps_one_row_of_its_own() {
    // The predictor layer is declared and stored; only its load form is asked here (no predictor arithmetic runs).
    let c=checkpoint(true);
    let charge=|draft:bool| {let mut e=engine(&c.admission,true,draft,usize::MAX);e.predictor=3;let before=e.workspace;let e=e.with_absorbed_charges().unwrap();(e.latent_rows,before,e.workspace)};
    let (rows,before,expands)=charge(false);assert_eq!(rows,LatentRows::Absorbed {draft:true});
    let (plain,same,absorbed)=charge(true);assert_eq!(plain,LatentRows::Absorbed {draft:false});assert_eq!(before,same);
    assert!(absorbed<=expands && expands<=before);
    assert_eq!(expands-absorbed,group_workspace_bytes_with(&c.admission,CONTEXT,256,rows).unwrap()-group_workspace_bytes_with(&c.admission,CONTEXT,256,plain).unwrap());
    // Without a predictor nobody asks for its form.
    let e=engine(&c.admission,true,false,usize::MAX).with_absorbed_charges().unwrap();assert_eq!(e.latent_rows,LatentRows::Absorbed {draft:false});
}
