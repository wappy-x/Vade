use super::*;
use spark_gpu_probe::{mhc_reference,mhc_boundary};
pub struct Host;
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
