//! Resident KDA/dense layer owners. Publish causal state only after full success.
use super::*;
use spark_gpu_probe::driver::{Backend,DeviceVector,Bf16Vector};
use crate::{dynamic_projection::ResidentDynamicProjection,layer_operands::device::{ResidentResidualBlock,ResidualBlockPass}};
use std::collections::BTreeMap;

enum Resident<'b,'d> {Residual(ResidentResidualBlock<'b,'d>),Projection(ResidentDynamicProjection<'b,'d>),
    Operation {shape:Shape,weight:Option<DeviceVector<'d>>,round:bool},Combine,Skip}
pub struct ResidentLayer<'b,'d> {backend:&'b Backend<'d>,plan:LayerPlan,operations:Vec<Resident<'b,'d>>}
pub struct DeviceState<'d> {history:DeviceVector<'d>,recurrent:DeviceVector<'d>,position:u64}
impl DeviceState<'_> {
    pub fn position(&self)->u64 {self.position}
    pub fn history(&self)->&DeviceVector<'_> {&self.history}
    pub fn recurrent(&self)->&DeviceVector<'_> {&self.recurrent}
}
impl Layer<'_> {
    pub fn upload<'b,'d>(&self,backend:&'b Backend<'d>)->Result<ResidentLayer<'b,'d>> {
        let mut operations=Vec::new();
        for p in &self.operations {operations.push(match p {
            Prepared::Residual(p)=>Resident::Residual(p.upload(backend)?),
            Prepared::Projection(p)=>Resident::Projection(p.load_dynamic_positive_zero(backend)?),
            Prepared::Plain {shape,weight}|Prepared::Operation {shape,weight}=>Resident::Operation {shape:*shape,
                weight:if weight.is_empty() {None} else {Some(backend.upload_vector(weight)?)},round:matches!(p,Prepared::Plain {..})},
            Prepared::Combine=>Resident::Combine,Prepared::Skip=>Resident::Skip,
        });}
        backend.synchronize()?;Ok(ResidentLayer {backend,plan:self.plan.clone(),operations})
    }
}
fn copy<'d>(b:&Backend<'d>,input:&DeviceVector<'_>)->Result<DeviceVector<'d>> {
    let mut out=b.vector(input.len())?;b.copy_prefix(input,&mut out,input.len())?;b.synchronize()?;Ok(out)
}
impl<'b,'d> ResidentLayer<'b,'d> {
    pub fn zero_state(&self)->Result<DeviceState<'d>> {
        let size=|name|self.plan.inputs.iter().find(|v|v.name==name).ok_or("causal state input is absent")?.elements().map_err(|e|e.to_string());
        let b=self.backend;let mut history=b.vector(size("kda.history")?)?;let mut recurrent=b.vector(size("kda.state")?)?;
        b.zero_vector(&mut history)?;b.zero_vector(&mut recurrent)?;b.synchronize()?;Ok(DeviceState {history,recurrent,position:0})
    }
    /// Input state stays immutable. A failure drops all unpublished next state.
    /// Each output producer completes before observation or a later consumer.
    pub fn step(&self,residual:&Bf16Vector<'_>,state:&DeviceState<'_>,mut observe:impl FnMut(&str,&DeviceVector<'_>)->Result<()>)->Result<(Bf16Vector<'d>,DeviceState<'d>)> {
        let b=self.backend;b.require_resident_stream()?;let mut values=BTreeMap::new();
        values.insert("residual".to_owned(),copy(b,residual.vector())?);
        values.insert("kda.history".into(),copy(b,&state.history)?);values.insert("kda.state".into(),copy(b,&state.recurrent)?);
        let mut passes:BTreeMap<&str,ResidualBlockPass<'b,'d>>=BTreeMap::new();
        for (step,p) in self.plan.steps.iter().zip(&self.operations) {
            let input=step.inputs.iter().map(|n|values.get(n).ok_or_else(||format!("missing resident causal input {n}"))).collect::<Result<Vec<_>>>()?;
            let output:Vec<(String,DeviceVector<'d>)>=match p {
                Resident::Residual(p)=>{
                    let pass=p.begin(b.round_bf16_lanes(input[0])?)?;let side=step.id.split('.').next().unwrap();
                    let output=vec![(format!("{side}.normalized"),copy(b,pass.mhc().normalized())?),
                        (format!("{side}.coefficients"),copy(b,pass.mhc().coefficients())?),
                        (format!("{side}.collapsed"),copy(b,pass.mhc().collapsed().vector())?),
                        (if side=="attn" {"input_norm"} else {"post_norm"}.into(),copy(b,pass.normalized().vector())?)];
                    passes.insert(side,pass);output
                },
                Resident::Projection(p)=>{
                    let projected=p.apply_vector(input[0])?;
                    vec![(step.outputs[0].name.clone(),copy(b,projected.projection().output().vector())?)]
                },
                Resident::Operation {shape,weight,round}=>{
                    let packed = if shape.op==Op::Conv {
                        let count=input[..3].iter().map(|v|v.len()).sum();let mut out=b.vector(count)?;let mut offset=0;
                        for v in &input[..3] {b.copy_range(v,&mut out,0,offset,v.len())?;offset+=v.len();}b.synchronize()?;Some(out)
                    } else {None};
                    let args=if let Some(packed)=packed.as_ref() {[Some(packed),Some(input[3]),None,None]}
                        else {std::array::from_fn(|i|input.get(i).copied())};
                    let result=b.causal_operation(*shape,args,weight.as_ref())?;
                    let results=self.unpack(*shape,&result)?;
                    let mut output=Vec::new();for (v,spec) in results.into_iter().zip(&step.outputs) {
                        let v=if *round {copy(b,b.round_bf16_lanes(&v)?.vector())?} else {v};output.push((spec.name.clone(),v));
                    }output
                },
                Resident::Combine=>{
                    let side=step.id.split('.').next().unwrap();let pass=passes.remove(side).ok_or("resident residual pass is absent")?;
                    let combined=pass.finish(b.round_bf16_lanes(input[2])?)?;vec![(step.outputs[0].name.clone(),copy(b,combined.vector())?)]
                },
                Resident::Skip=>continue,
            };
            for (name,value) in output {observe(&name,&value)?;values.insert(name,value);}
        }
        let output=values.remove("output").ok_or("resident causal layer has no output")?;
        let history=values.remove("next_history").ok_or("resident causal layer has no next history")?;
        let recurrent=values.remove("next_state").ok_or("resident causal layer has no next state")?;
        b.synchronize()?;if let Some(e)=b.fatal_error() {return Err(e);}
        Ok((b.round_bf16_lanes(&output)?,DeviceState {history,recurrent,position:state.position.checked_add(1).ok_or("resident causal position overflow")?}))
    }
    fn unpack(&self,s:Shape,result:&DeviceVector<'_>)->Result<Vec<DeviceVector<'d>>> {
        let b=self.backend;let (_,_,units,words)=s.lengths().ok_or("invalid resident causal output shape")?;
        let (first,second,swap)=match s.op {
            Op::Conv=>(s.heads as usize*s.dim as usize,words-s.heads as usize*s.dim as usize,false),
            Op::Step=>(s.dim as usize*s.dim as usize,s.dim as usize,true),
            _=>return Ok(vec![copy(b,result)?]),
        };
        let mut a=b.vector(units*first)?;let mut c=b.vector(units*second)?;
        for i in 0..units {b.copy_range(result,&mut a,i*words,i*first,first)?;b.copy_range(result,&mut c,i*words+first,i*second,second)?;}
        b.synchronize()?;Ok(if swap {vec![c,a]} else {vec![a,c]})
    }
}
