//! FOLD HONESTY: a routed down step must not skip LOCAL FOLD because an earlier projection stage was refused. With every
//! stage refused, the shared expert's row runs after the routed down in the plan, so the fold's last input is absent when
//! the walk reaches the down step. The decoder must then run the down step with its fold at the fold step, never alone.
use spark_tp::bounded_decoder::{Compute,Decoder,Host};
use engine_model::{plan::admission::CheckpointAdmission,safetensors::Dtype};
use spark_gpu_probe::causal_ops::Shape;
#[allow(dead_code,reason="the shared fixture has more forms than this test uses")]
#[path="support/attention_checkpoint.rs"] mod fixture;
type V=Vec<f32>;
/// The host compute with an attached device exchange: every projection stage is refused (the default), and a routed
/// fold is counted. `answer` true answers each fold with zeros, false refuses it (the caller keeps the two steps).
#[derive(Default)]
struct Folding {with_shared:std::cell::Cell<usize>,without_shared:std::cell::Cell<usize>,answer:bool}
impl Compute for Folding {
    type Value=V;type Pass=<Host as Compute>::Pass;
    fn mhc_row_launches(&self)->u64 {0}
    fn rows_greedy(&self,_:&Self::Value,_:usize)->spark_tp::Result<Option<Vec<u32>>> {Ok(None)}
    fn folds_routed(&self)->bool {true}
    fn resident_weights_ready(&self)->bool {true}
    fn routed_fold(&self,_:&[engine_model::plan::layer::Step],_:&V,_:&V,_:&V,shared:Option<&V>,fold:Shape)->spark_tp::Result<Option<V>> {
        let count=if shared.is_some() {&self.with_shared} else {&self.without_shared};count.set(count.get()+1);
        Ok(self.answer.then(||vec![0.0;fold.cols as usize]))
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
#[test]
fn a_refused_stage_still_folds_the_routed_down_with_the_shared_row() {
    let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let bits=|v:&[f32]|v.iter().map(|v|v.to_bits()).collect::<Vec<_>>();
    let routed=c.admission.description().layers.iter().filter(|layer|matches!(layer.ffn,engine_model::shapes::Ffn::Experts(_))).count();assert!(routed>0);
    let plain=Decoder::new(&c.admission,Host,8).unwrap();
    // The fold refused: the down step runs at the fold step on the old path, and every logit keeps its bits.
    let refused=Folding::default();let decoder=Decoder::new(&c.admission,&refused,8).unwrap();
    let (mut a,mut b)=(plain.zero_state().unwrap(),decoder.zero_state().unwrap());
    for token in [1,7,2] {
        let (expected,_,next_a)=plain.step(token,&a,|_,_|Ok(())).unwrap();let (actual,_,next_b)=decoder.step(token,&b,|_,_|Ok(())).unwrap();
        assert_eq!(bits(&expected),bits(&actual),"token {token}");a=next_a;b=next_b;
    }
    assert_eq!(refused.without_shared.get(),0,"a routed fold was asked without its shared row");
    assert_eq!(refused.with_shared.get(),3*routed,"every routed down step of every decode step must reach the fold with its shared row");
    // The fold answered: its output is the layer's expert output (the fold step is not run again).
    let answered=Folding {answer:true,..Folding::default()};let decoder=Decoder::new(&c.admission,&answered,8).unwrap();
    let (expected,_,_)=plain.step(1,&plain.zero_state().unwrap(),|_,_|Ok(())).unwrap();
    let (actual,_,_)=decoder.step(1,&decoder.zero_state().unwrap(),|_,_|Ok(())).unwrap();
    assert_eq!(answered.with_shared.get(),routed);assert_ne!(bits(&expected),bits(&actual),"the folded output was not used");
}
