use super::*;
use spark_gpu_probe::{mhc_reference,mhc_boundary};
pub struct Host;
// Test builds only: the next stages the decoder named to `begin_next`, in call order. Ordinary
// builds keep the default `begin_next` and record nothing.
#[cfg(test)]
thread_local! {static SHADOW_TARGETS:std::cell::RefCell<Vec<Vec<String>>>=const {std::cell::RefCell::new(Vec::new())};}
#[cfg(test)]
fn take_shadow_targets()->Vec<Vec<String>> {SHADOW_TARGETS.with(|targets|std::mem::take(&mut *targets.borrow_mut()))}
pub struct Pass {input:Vec<f32>,coefficients:Vec<f32>,streams:usize,hidden:usize}
fn finish_rows(p:Pass,x:&[f32],m:usize)->Vec<f32> {
    let wide=p.streams*p.hidden;let count=p.streams*(p.streams+2);let mut out=Vec::with_capacity(m*wide);
    for token in 0..m {let input=&p.input[token*wide..(token+1)*wide];let coefficients=&p.coefficients[token*count..(token+1)*count];let block=&x[token*p.hidden..(token+1)*p.hidden];
        out.extend((0..wide).map(|j|mhc_boundary::combine(input,coefficients,block[j%p.hidden],p.streams,p.hidden,j/p.hidden,j%p.hidden)));}out
}
impl Compute for Host {
    type Value=Vec<f32>;type Pass=Pass;
    fn has_latent_absorption(&self,_:&str)->bool {false}
    fn gather_absorbed(&self,s:Shape,x:[Option<&Vec<f32>>;4],_:usize,expected:f32)->Result<Vec<f32>> {
        need(x[2].is_some_and(|count|count.len()==1 && count[0].to_bits()==expected.to_bits()),"absorbed selection count differs")?;self.operation(s,x,&[])
    }
    fn latent_absorption(&self,_:&str,_:Shape,_:&Self::Value,_:&Self::Value,_:&Self::Value,_:usize)->Result<Self::Value> {Err("host fixture has no resident latent weights".into())}
    fn latent_absorption_rows(&self,_:&str,_:spark_gpu_probe::prompt_ops::Shape,_:&Self::Value,_:&Self::Value,_:&Self::Value)->Result<Self::Value> {Err("host fixture has no resident latent weights".into())}
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {need(v.iter().all(|v|v.is_finite()),"bounded host value is not finite")?;Ok(v.to_vec())}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {self.upload(v)}
    /// Host reads cost nothing and are always the computed bits.
    fn select_known(&self,_:&Vec<f32>,_:&Vec<f32>,_:&[f32])->Result<()> {Ok(())}
    fn prompt_end(&self)->Result<()> {Ok(())} fn prompt_seeds(&self)->bool {false}
    fn prepare_beside(&self,c:&engine_model::plan::admission::CheckpointAdmission,_:usize)->Result<()> {self.prepare(c)}
    fn words(&self,v:&Vec<f32>)->usize {v.len()}
    fn range(&self,v:&Vec<f32>,start:usize,len:usize)->Result<Vec<f32>> {Ok(v.get(start..start.checked_add(len).ok_or("bounded range overflow")?).ok_or("bounded host range differs")?.to_vec())}
    fn join(&self,parts:&[&Vec<f32>])->Result<Vec<f32>> {Ok(parts.iter().flat_map(|p|p.iter().copied()).collect())}
    fn zeros(&self,len:usize)->Result<Vec<f32>> {Ok(vec![0.0;len])}
    fn operation(&self,s:Shape,x:[Option<&Vec<f32>>;4],w:&[f32])->Result<Vec<f32>> {spark_gpu_probe::causal_ops::reference(s,x.map(|v|v.map_or(&[][..],Vec::as_slice)),w)}
    fn norm(&self,_:&str,s:NormShape,x:&Vec<f32>,w:&[f32])->Result<Vec<f32>> {Ok(causal_layer::norm(s,x,w))}
    fn begin(&self,_:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>)->Result<(Pass,Vec<f32>)> {
        let m=p.mhc();let s=m.shape();let mut normalized=vec![0.0;s.wide()];let mut coefficients=vec![0.0;s.coefficients()];
        mhc_reference::prepare(s,x,m.weights(),m.base(),m.scale(),&mut normalized,&mut coefficients);
        let collapsed=(0..s.hidden as usize).map(|j|mhc_boundary::collapse(x,&coefficients,s.streams as usize,s.hidden as usize,j)).collect::<Vec<_>>();
        let norm=causal_layer::norm(p.norm().shape(),&collapsed,p.norm().weight());
        Ok((Pass {input:x.clone(),coefficients,streams:s.streams as usize,hidden:s.hidden as usize},norm))
    }
    #[cfg(test)]
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>,next:&[&str])->Result<(Pass,Vec<f32>)> {
        SHADOW_TARGETS.with(|targets|targets.borrow_mut().push(next.iter().map(|name|(*name).to_owned()).collect()));self.begin(key,p,x)
    }
    fn finish(&self,p:Pass,x:&Vec<f32>)->Result<Vec<f32>> {Ok((0..p.streams*p.hidden).map(|j|mhc_boundary::combine(&p.input,&p.coefficients,x[j%p.hidden],p.streams,p.hidden,j/p.hidden,j%p.hidden)).collect())}
    fn mx(&self,p:&Mxfp8Projection<'_>,x:&Vec<f32>)->Result<Vec<f32>> {Ok(p.dynamic_reference_positive_zero(x)?.2)}
    fn round(&self,x:&Vec<f32>)->Result<Vec<f32>> {Ok(x.iter().copied().map(spark_gpu_probe::bf16::rounded).collect())}
    fn mean(&self,x:&Vec<f32>,streams:u32,hidden:u32)->Result<Vec<f32>> {let mut s=shape(Op::StreamMean);s.heads=streams;s.dim=hidden;self.operation(s,[Some(x),None,None,None],&[])}
    fn prompt(&self,s:spark_gpu_probe::prompt_ops::Shape,x:[Option<&Vec<f32>>;5],w:&[f32])->Result<Vec<f32>> {spark_gpu_probe::prompt_ops::reference(s,x.map(|v|v.map_or(&[][..],Vec::as_slice)),w)}
    fn rows_norm(&self,_:&str,s:NormShape,x:&Vec<f32>,m:usize,w:&[f32])->Result<Vec<f32>> {
        let width=x.len()/m;Ok(x.chunks_exact(width).flat_map(|row|causal_layer::norm(s,row,w)).collect())
    }
    fn rows_begin(&self,_:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>,m:usize)->Result<(Pass,Vec<f32>)> {
        let h=p.mhc().shape();let wide=h.wide();let mut coefficients=Vec::new();let mut norm=Vec::new();
        for row in x.chunks_exact(wide) {let (pass,value)=self.begin((0,0),p,&row.to_vec())?;coefficients.extend_from_slice(&pass.coefficients);norm.extend_from_slice(&value);}
        need(coefficients.len()==m*h.coefficients(),"prompt residual row count differs")?;
        Ok((Pass {input:x.clone(),coefficients,streams:h.streams as usize,hidden:h.hidden as usize},norm))
    }
    fn rows_finish(&self,p:Pass,x:&Vec<f32>,m:usize)->Result<Vec<f32>> {Ok(finish_rows(p,x,m))}
}
#[cfg(test)]
#[allow(dead_code)]
#[path="../../tests/support/attention_checkpoint.rs"] pub(super) mod shadow_fixture;
#[cfg(test)]
mod shadow_tests {
    use super::{Host,take_shadow_targets,shadow_fixture as fixture};
    use crate::{bounded_decoder::{Compute,Decoder,Verify},row_split::{Exchange,Split},Result};
    use engine_model::plan::layer::Topology;
    use engine_model::safetensors::Dtype;
    struct LocalExchange;
    impl Exchange for LocalExchange {
        fn topology(&self)->Topology {Topology {rank:0,parts:1}}
        fn gather(&mut self,local:&[f32],rows:usize)->Result<Vec<f32>> {assert_eq!(local.len(),rows);Ok(local.to_vec())}
        fn command(&mut self,_:&mut [u8])->Result<()> {Err("host shadow test does not send commands".into())}
        fn complete(&mut self)->Result<()> {Err("host shadow test does not synchronize ranks".into())}
    }
    fn assert_split_shadow_targets(compute:impl Compute<Value=Vec<f32>>) {
        let checkpoint=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);
        let direct=Decoder::new(&checkpoint.admission,Host,8).unwrap();let _=take_shadow_targets();
        let (expected_logits,_,_)=direct.step(1,&direct.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let expected=take_shadow_targets();
        assert!(!expected.is_empty() && expected.iter().all(|names|!names.is_empty()),"direct Host targets {expected:?}");
        let wrapped=Decoder::new(&checkpoint.admission,compute,8).unwrap();
        let (actual_logits,_,_)=wrapped.step(1,&wrapped.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let actual=take_shadow_targets();
        assert_eq!(actual,expected,"Split must forward every CHAIN-SHADOW target in order");
        assert_eq!(actual_logits.iter().map(|value|value.to_bits()).collect::<Vec<_>>(),expected_logits.iter().map(|value|value.to_bits()).collect::<Vec<_>>());
    }
    #[test]
    fn chain_shadow_targets_survive_owned_split() {
        assert_split_shadow_targets(Split::new(Host,LocalExchange).unwrap());
    }
    #[test]
    fn chain_shadow_targets_survive_borrowed_split() {
        let split=Split::new(Host,LocalExchange).unwrap();assert_split_shadow_targets(&split);
    }
    #[test]
    fn chain_shadow_is_named_on_the_plain_step_and_never_on_verify_or_prompt_rows() {
        let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let decoder=Decoder::new(&c.admission,Host,8).unwrap();
        let _=take_shadow_targets();
        // Plain decode: every boundary names its next stage.
        let (expected,_,_)=decoder.step(1,&decoder.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let plain=take_shadow_targets();
        assert!(!plain.is_empty() && plain.iter().all(|names|!names.is_empty()),"plain step targets {plain:?}");
        // Single-row verification with nothing carried runs the same arithmetic, but names nothing.
        let mut open=Verify::new(decoder.zero_state().unwrap());let logits=open.rows(&decoder,&[1]).unwrap();
        let verify=take_shadow_targets();
        assert_eq!(verify.len(),plain.len(),"verify reaches the same boundaries");assert!(verify.iter().all(Vec::is_empty),"verify targets {verify:?}");
        assert_eq!(logits[0].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        // A single-row prompt names nothing either.
        let _=decoder.prefill(&[1],&decoder.zero_state().unwrap()).unwrap();
        let prompt=take_shadow_targets();assert!(prompt.iter().all(Vec::is_empty),"prompt targets {prompt:?}");
    }
}
