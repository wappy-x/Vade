use super::*;
use spark_gpu_probe::{mhc_reference,mhc_boundary};
pub struct Host;
// Test builds only: the next stages the decoder named to `begin_next`, in call order. Ordinary
// builds keep the default `begin_next` and record nothing.
#[cfg(test)]
thread_local! {static SHADOW_TARGETS:std::cell::RefCell<Vec<Option<String>>>=const {std::cell::RefCell::new(Vec::new())};}
#[cfg(test)]
fn take_shadow_targets()->Vec<Option<String>> {SHADOW_TARGETS.with(|targets|std::mem::take(&mut *targets.borrow_mut()))}
pub struct Pass {input:Vec<f32>,coefficients:Vec<f32>,streams:usize,hidden:usize}
fn finish_rows(p:Pass,x:&[f32],m:usize)->Vec<f32> {
    let wide=p.streams*p.hidden;let count=p.streams*(p.streams+2);let mut out=Vec::with_capacity(m*wide);
    for token in 0..m {let input=&p.input[token*wide..(token+1)*wide];let coefficients=&p.coefficients[token*count..(token+1)*count];let block=&x[token*p.hidden..(token+1)*p.hidden];
        out.extend((0..wide).map(|j|mhc_boundary::combine(input,coefficients,block[j%p.hidden],p.streams,p.hidden,j/p.hidden,j%p.hidden)));}out
}
impl Compute for Host {
    type Value=Vec<f32>;type Pass=Pass;
    fn upload(&self,v:&[f32])->Result<Vec<f32>> {need(v.iter().all(|v|v.is_finite()),"bounded host value is not finite")?;Ok(v.to_vec())}
    fn read(&self,v:&Vec<f32>)->Result<Vec<f32>> {self.upload(v)}
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
    fn begin_next(&self,key:(usize,usize),p:&crate::layer_operands::ResidualBlockOperands,x:&Vec<f32>,next:Option<&str>)->Result<(Pass,Vec<f32>)> {
        SHADOW_TARGETS.with(|targets|targets.borrow_mut().push(next.map(str::to_owned)));self.begin(key,p,x)
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
#[path="../../tests/support/attention_checkpoint.rs"] mod shadow_fixture;
#[cfg(test)]
mod shadow_tests {
    use super::{Host,take_shadow_targets,shadow_fixture as fixture};
    use crate::bounded_decoder::{Decoder,Verify};
    use engine_model::safetensors::Dtype;
    #[test]
    fn chain_shadow_is_named_on_the_plain_step_and_never_on_verify_or_prompt_rows() {
        let c=fixture::Checkpoint::new_with_head(Dtype::Bf16,true);let decoder=Decoder::new(&c.admission,Host,8).unwrap();
        let _=take_shadow_targets();
        // Plain decode: every boundary names its next stage.
        let (expected,_,_)=decoder.step(1,&decoder.zero_state().unwrap(),|_,_|Ok(())).unwrap();
        let plain=take_shadow_targets();
        assert!(!plain.is_empty() && plain.iter().all(Option::is_some),"plain step targets {plain:?}");
        // Single-row verification with nothing carried runs the same arithmetic, but names nothing.
        let mut open=Verify::new(decoder.zero_state().unwrap());let logits=open.rows(&decoder,&[1]).unwrap();
        let verify=take_shadow_targets();
        assert_eq!(verify.len(),plain.len(),"verify reaches the same boundaries");assert!(verify.iter().all(Option::is_none),"verify targets {verify:?}");
        assert_eq!(logits[0].iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        // A single-row prompt names nothing either.
        let _=decoder.prefill(&[1],&decoder.zero_state().unwrap()).unwrap();
        let prompt=take_shadow_targets();assert!(prompt.iter().all(Option::is_none),"prompt targets {prompt:?}");
    }
}
