//! ROWS FOLD and the rule that a decode step makes no host allocation: the fold adds none of its own.
//!
//! WHAT IS COUNTED. This test binary runs on the crate's tracking allocator (`runtime::TrackedAllocator`, the one the
//! servers run on). A window is clean when the live host bytes never rise above their level at its start
//! (`reset_heap_peak` then `heap_peak`). The binary holds ONE test, so no other test thread allocates meanwhile.
//!
//! 1. The row pass, on the real `Decoder::step_batch`: from the moment the compute hands back this rank's shared rows
//!    (`rows_project_local`) to the moment the decoder asks for the fold stage (`rows_routed_fold`). Inside that window
//!    the decoder keeps the rows for the fold, walks to the fold step, finds the waiting down step and its inputs. On the
//!    head before this test a String key and a map entry were allocated there.
//! 2. The device function's own host storage, as `apply_routed_rows_fold` uses it (`resident_weights::rows_fold`): a
//!    token's planes, the folded handles, the message parts and their rows, for 1 to 8 tokens.
//!
//! WHAT IS NOT. The device function itself is Linux and GPU code: its call sites, the kernels it launches and the gather
//! it calls (`gather_joined`, which builds its layout as on main) are not run here. And a rise of live bytes is counted,
//! not allocation calls: an allocation behind a larger free inside one window would pass.
use spark_tp::{bounded_decoder::{Compute,Decoder,Host,State,BATCH_ROWS},runtime::{TrackedAllocator,heap_peak,reset_heap_peak},resident_weights::{Rows,rows_fold}};
use engine_model::{plan::{admission::CheckpointAdmission,layer::Step},safetensors::Dtype};
use spark_gpu_probe::causal_ops::Shape;
use std::cell::Cell;
#[global_allocator] static ALLOCATOR:TrackedAllocator=TrackedAllocator;
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// The host compute as a device that folds: it keeps the shared rows of every routed layer on their rank and answers each
/// fold stage. Between the two calls it measures the decoder: `window` holds the live bytes when the shared rows were
/// handed back; `rise` sums how far the live bytes rose above that before the fold stage was asked; `windows` counts.
#[derive(Default)]
struct Measuring {window:Cell<Option<usize>>,rise:Cell<usize>,windows:Cell<usize>}
impl Compute for Measuring {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Self::Value,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn routes_rows(&self,m:usize)->bool {(3..=BATCH_ROWS).contains(&m)}
    fn folds_rows(&self,m:usize)->bool {self.routes_rows(m)}
    fn rows_project_local(&self,_:&CheckpointAdmission,_:&str,_:&V,m:usize,_:bool)->spark_tp::Result<Option<V>> {
        // The rows are made before the window opens: they are the compute's, not the decoder's.
        let rows=vec![7.0;3*m];
        reset_heap_peak();self.window.set(Some(heap_peak()));
        Ok(Some(rows))
    }
    fn rows_routed_fold(&self,_:&Step,_:&V,_:&V,_:&V,shared:Option<&V>,fold:Shape,m:usize)->spark_tp::Result<Option<V>> {
        // The window closes before this call makes its own answer.
        let start=self.window.take().expect("the fold stage was asked before the shared rows were kept");
        self.rise.set(self.rise.get()+(heap_peak()-start));self.windows.set(self.windows.get()+1);
        assert_eq!(shared.map(Vec::len),Some(3*m),"the fold gets the rows this rank kept");
        Ok(Some(vec![0.0;m*fold.cols as usize]))
    }
    fn select_known(&self,ids:&V,count:&V,row:&[f32])->spark_tp::Result<()> {Host.select_known(ids,count,row)}
    fn prompt_end(&self)->spark_tp::Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&CheckpointAdmission,_:usize)->spark_tp::Result<()> {self.prepare(c)}
    fn upload(&self,v:&[f32])->spark_tp::Result<V> {Host.upload(v)}
    fn read(&self,v:&V)->spark_tp::Result<V> {Host.read(v)}
    fn words(&self,v:&V)->usize {v.len()}
    fn range(&self,v:&V,start:usize,len:usize)->spark_tp::Result<V> {Host.range(v,start,len)}
    fn join(&self,v:&[&V])->spark_tp::Result<V> {Host.join(v)}
    fn zeros(&self,n:usize)->spark_tp::Result<V> {Host.zeros(n)}
    fn operation(&self,s:Shape,x:[Option<&V>;4],w:&[f32])->spark_tp::Result<V> {Host.operation(s,x,w)}
    fn norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,w:&[f32])->spark_tp::Result<V> {Host.norm(name,s,x,w)}
    fn begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V)->spark_tp::Result<(Self::Pass,V)> {Host.begin(key,p,x)}
    fn finish(&self,p:Self::Pass,x:&V)->spark_tp::Result<V> {Host.finish(p,x)}
    fn mx(&self,p:&spark_tp::mxfp8_projection::Mxfp8Projection<'_>,x:&V)->spark_tp::Result<V> {Host.mx(p,x)}
    fn round(&self,x:&V)->spark_tp::Result<V> {Host.round(x)}
    fn mean(&self,x:&V,s:u32,h:u32)->spark_tp::Result<V> {Host.mean(x,s,h)}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&V>;5],w:&[f32])->spark_tp::Result<V> {Host.prompt(s,x,w)}
    fn rows_norm(&self,name:&str,s:spark_gpu_probe::norm_boundary::NormShape,x:&V,m:usize,w:&[f32])->spark_tp::Result<V> {Host.rows_norm(name,s,x,m,w)}
    fn rows_begin(&self,key:(usize,usize),p:&spark_tp::layer_operands::ResidualBlockOperands,x:&V,m:usize)->spark_tp::Result<(Self::Pass,V)> {Host.rows_begin(key,p,x,m)}
    fn rows_finish(&self,p:Self::Pass,x:&V,m:usize)->spark_tp::Result<V> {Host.rows_finish(p,x,m)}
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:Shape,x:[Option<&V>;4],reserve:usize,expected:f32)->spark_tp::Result<V> {Host.gather_absorbed(s,x,reserve,expected)}
    fn latent_absorption(&self,name:&str,s:Shape,q:&V,l:&V,n:&V,slots:usize)->spark_tp::Result<V> {Host.latent_absorption(name,s,q,l,n,slots)}
    fn latent_absorption_rows(&self,name:&str,s:spark_gpu_probe::prompt_ops::Shape,q:&V,l:&V,n:&V)->spark_tp::Result<V> {Host.latent_absorption_rows(name,s,q,l,n)}
}
fn states<C:Compute<Value=V>>(decoder:&Decoder<'_,C>,prompts:&[&[u32]])->Vec<State<V>> {
    prompts.iter().map(|prompt| {
        let mut state=decoder.zero_state().unwrap();
        for &token in *prompt {state=decoder.step(token,&state,|_,_|Ok(())).unwrap().2;}
        state
    }).collect()
}
#[test]
fn the_fold_of_a_batch_step_allocates_nothing_of_its_own() {
    // 1. The row pass.
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
    let routed=c.admission.description().layers.iter().filter(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Experts(_))).count();assert!(routed>0);
    let prompts:[&[u32];3]=[&[1],&[7,2],&[3,4,5]];let tokens=[2u32,5,1];
    let compute=Measuring::default();let decoder=Decoder::new(&c.admission,&compute,8).unwrap();let held=states(&decoder,&prompts);
    let rows=tokens.iter().zip(&held).map(|(&token,state)|(token,state)).collect::<Vec<_>>();
    // The warm-up step: the decoder finds each layer's plan and its fold steps once.
    decoder.step_batch(&rows).unwrap();
    assert_eq!(compute.windows.get(),routed,"every routed layer of a batch step keeps its shared rows and folds");
    compute.rise.set(0);compute.windows.set(0);
    for _ in 0..3 {decoder.step_batch(&rows).unwrap();}
    assert_eq!(compute.windows.get(),3*routed);
    assert_eq!(compute.rise.get(),0,"the decoder allocated host memory between keeping the shared rows and asking for the fold ({} bytes over {} folds)",compute.rise.get(),compute.windows.get());
    // 2. The device function's own host storage: the planes of each token, the folded handles, the message and its rows.
    let split=Rows {start:2048,count:2048,total:4096,split:true};
    for m in 1..=rows_fold::TOKENS {for shared in [None,Some(m*split.count),Some(m*split.total)] {
        // The handles are made before the window opens (on the device they are the kernels' output vectors).
        let mut handles=(0..m).map(|token|vec![token as f32;4]).collect::<Vec<_>>();handles.reverse();
        reset_heap_peak();let start=heap_peak();
        let mut folded=rows_fold::Folded::default();let mut words=0usize;
        for token in 0..m {
            let at=rows_fold::planes(token,m,8,split,shared).unwrap();
            words+=at.picked.len()+at.projected.len()+at.own.map_or(0,|own|own.len());
            folded.push(handles.pop().unwrap()).unwrap();
        }
        let (parts,list)=(folded.parts().unwrap(),rows_fold::message_rows(split));
        let message=(&parts[..m],&list[..m]);
        let rise=heap_peak()-start;
        assert_eq!(rise,0,"the fold's host storage allocated {rise} bytes at {m} tokens");
        assert_eq!((folded.len(),message.0.len(),message.1.len()),(m,m,m));
        assert!(message.0.iter().enumerate().all(|(token,(row,round))|row[0]==token as f32 && !*round),"the message is the folded rows in token order, none to round");
        assert!(message.1.iter().all(|rows|*rows==split));
        assert_eq!(words,m*(8+8*split.count+shared.map_or(0,|_|split.count)));
    }}
    // More tokens than a batch step are refused, not grown.
    let mut folded=rows_fold::Folded::default();
    for token in 0..rows_fold::TOKENS {folded.push(token).unwrap();}
    assert!(folded.push(rows_fold::TOKENS).is_err());assert!(rows_fold::Folded::<usize>::default().parts().is_err());
}
