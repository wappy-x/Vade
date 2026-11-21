use super::*;
use crate::projection_stage::{self,Job};
impl<C:Compute> Decoder<'_,C> {
    pub(super) fn projection_stage(&self,index:usize,end:usize,plan:&LayerPlan,values:&mut BTreeMap<String,C::Value>)->Result<Vec<usize>> {
        if !self.compute.resident_weights_ready() {return Ok(Vec::new());}
        let available=projection_stage::ready_names(values);
        let Some(mut frontier)=projection_stage::frontier(index,plan,&available) else {return Ok(Vec::new());};
        let mut jobs=Vec::new();
        let mut routed_middle=None;
        for group in &frontier.groups {
            let first=&plan.steps[group.indices[0]];
            if let Some(middle)=group.middle {
                let middle=&plan.steps[middle];let mut shape=shape(Op::SwiGlu);
                shape.rows=middle.outputs[0].elements().map_err(|e|e.to_string())?.try_into().map_err(|_|"projection stage middle exceeds ABI")?;
                shape.bound=self.checkpoint.description().swiglu_limit as f32;
                jobs.push(Job::Gated {name:&first.weights[0].name,gate:&values[&middle.inputs[0]],up:&values[&middle.inputs[1]],shape});
            } else if first.kind==Kind::Project {
                let round=!plan.steps.iter().any(|step|step.kind==Kind::Route && step.inputs.contains(&first.outputs[0].name));
                jobs.push(Job::Project {input:group.indices.iter().map(|&at| {let step=&plan.steps[at];(step.weights[0].name.as_str(),&values[&step.inputs[0]])}).collect(),round});
            } else {
                let last=*group.indices.last().ok_or("empty routed stage")?;
                let middle=if group.indices[0]==index {
                    projection_stage::routed_middle(plan,index,last,end,&available).map(|at| {
                        let mut shape=shape(Op::SwiGlu);
                        shape.rows=plan.steps[at].outputs[0].elements().map_err(|e|e.to_string())?.try_into().map_err(|_|"routed middle exceeds ABI")?;
                        shape.bound=self.checkpoint.description().swiglu_limit as f32;
                        routed_middle=Some((jobs.len(),at));
                        Ok::<_,String>(shape)
                    }).transpose()?
                } else {None};
                jobs.push(Job::Routed {steps:&plan.steps[group.indices[0]..=last],ids:&values[&first.inputs[1]],input:&values[&first.inputs[0]],middle});
            }
        }
        let mut output=self.compute.projection_stage(self.checkpoint,&jobs)?;
        if let Some((job,at))=routed_middle {
            if output.is_some() {
                let position=frontier.outputs.iter().position(|&step|step==index).ok_or("routed stage output is absent")?;
                frontier.outputs[position]=at;
                frontier.outputs.remove(position+1);
                frontier.skipped.push(at);
            } else if let Job::Routed {middle,..}=&mut jobs[job] {
                *middle=None;
                output=self.compute.projection_stage(self.checkpoint,&jobs)?;
            }
        }
        let Some(output)=output else {return Ok(Vec::new());};
        need(output.len()==frontier.outputs.len(),"projection stage output count differs")?;
        for (stage,value) in frontier.outputs.iter().zip(output) {values.insert(plan.steps[*stage].outputs[0].name.clone(),value);}
        Ok(frontier.skipped)
    }
}
