use super::*;
use crate::projection_stage::{self,Job};
struct Collect(std::rc::Rc<std::cell::Cell<bool>>);
impl Drop for Collect {fn drop(&mut self) {self.0.set(false);}}
pub fn gather_stage<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&[DeviceVector<'d>],rows:&[Rows])->Result<Vec<DeviceVector<'d>>> {
    let words=projection_stage::checked_rows(rows)?;
    if local.is_empty() || local.len()!=rows.len() || local.iter().zip(rows).any(|(value,row)|value.len()!=row.count) {return Err("projection stage local row extent differs".into());}
    if words.checked_mul(4).is_none_or(|bytes|bytes>mailbox.slot_bytes()) {return Err("projection stage payload exceeds mailbox slot".into());}
    let parts=local.iter().map(|value|(value,value.len())).collect::<Vec<_>>();let packed=b.resident_join(&parts,&mut None,0)?;
    gather_device(b,mailbox,sequence,topology,&packed,rows)
}
impl<'b,'d> Weights<'b,'d> {
    fn stage_rows(&self,jobs:&[Job<'_,DeviceVector<'d>>])->Result<Option<(Vec<Rows>,Vec<usize>)>> {
        if self.collecting.get() {return Err("nested projection stage".into());}
        let Some(mailbox)=self.mailbox else {return Ok(None);};
        let mut rows=Vec::new();let mut counts=Vec::new();
        for job in jobs {
            match job {
                Job::Project {input,..}=>{
                    if input.is_empty() || input.len()>9 {return Ok(None);}
                    for (name,value) in input {
                        let Some(matrix)=self.matrices.get(*name) else {return Ok(None);};
                        if value.len()!=matrix.cols || value.is_segmented() || input.len()>1 && !matches!(matrix.storage,Storage::Mx(_)|Storage::Plain {..}|Storage::Packed {shape:Shape {op:Op::NvProject|Op::Fp8Serial,..},..}) {return Ok(None);}
                        rows.push(matrix.rows);counts.push(1);
                    }
                },
                Job::Routed {steps,ids,input}=>{
                    let Some(first)=steps.first().and_then(|step|step.weights.first()) else {return Ok(None);};
                    if !(1..=2).contains(&steps.len()) || !self.routes.get(&first.name).is_some_and(Option::is_some) {return Ok(None);}
                    let Some(matrix)=self.matrices.get(&first.name) else {return Ok(None);};
                    let width=if steps[0].kind==Kind::ExpertDown {matrix.cols.checked_mul(ids.len())} else {Some(matrix.cols)};
                    if ids.len()!=8 || width!=Some(input.len()) || ids.is_segmented() || input.is_segmented() {return Ok(None);}
                    rows.extend(std::iter::repeat_n(matrix.rows,steps.len()*ids.len()));counts.extend(std::iter::repeat_n(ids.len(),steps.len()));
                },
                Job::Gated {name,gate,up,shape}=>{
                    let Some(matrix)=self.matrices.get(*name) else {return Ok(None);};
                    if shape.op!=spark_gpu_probe::causal_ops::Op::SwiGlu || gate.len()!=matrix.cols || up.len()!=matrix.cols || shape.rows as usize!=matrix.cols
                        || gate.is_segmented() || up.is_segmented() {return Ok(None);}
                    rows.push(matrix.rows);counts.push(1);
                },
            }
        }
        let Ok(words)=projection_stage::checked_rows(&rows) else {return Ok(None);};
        if projection_stage::checked_messages(jobs.len(),words,mailbox.slot_bytes()).is_err() {return Ok(None);}
        Ok(Some((rows,counts)))
    }
    pub fn apply_stage(&mut self,b:&Backend<'d>,jobs:&[Job<'_,DeviceVector<'d>>])->Result<Option<Vec<Vec<DeviceVector<'d>>>>> {
        let Some((rows,counts))=self.stage_rows(jobs)? else {return Ok(None);};
        self.collecting.set(true);let guard=Collect(self.collecting.clone());
        let mut output=Vec::new();
        for job in jobs {
            match job {
                Job::Project {input,round}=>{
                    if input.len()==1 {output.push(self.apply(input[0].0,b,input[0].1,*round)?);}
                    else {let Some(values)=self.apply_batch(b,input,*round)? else {return Ok(None);};output.extend(values);}
                },
                Job::Routed {steps,ids,input}=>{
                    let Some(values)=self.apply_routed(b,steps,ids,input)? else {return Ok(None);};output.extend(values);
                },
                Job::Gated {name,gate,up,shape}=>{
                    let value=b.causal_operation(*shape,[Some(gate),Some(up),None,None],None)?;
                    output.push(self.apply(name,b,&value,true)?);
                },
            }
        }
        drop(guard);
        if output.len()!=rows.len() || output.iter().zip(&rows).any(|(value,row)|value.len()!=row.count) {return Err("projection stage local row extent differs".into());}
        let words=output.iter().map(DeviceVector::len).sum();
        let values=gather_stage(b,self.mailbox.ok_or("projection stage mailbox is absent")?,&mut self.sequence,self.topology,&output,&rows)?;
        projection_stage::accepted(jobs.len(),words);
        let mut values=values.into_iter();let result=counts.into_iter().map(|count|values.by_ref().take(count).collect::<Vec<_>>()).collect::<Vec<_>>();
        if values.next().is_some() || result.iter().any(Vec::is_empty) {return Err("projection stage result extent differs".into());}Ok(Some(result))
    }
}
