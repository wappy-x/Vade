use super::*;
use crate::projection_stage::{self,Job};
impl<C:Compute> Decoder<'_,C> {
    pub(super) fn projection_stage(&self,index:usize,plan:&LayerPlan,values:&mut BTreeMap<String,C::Value>)->Result<Vec<usize>> {
        if !self.compute.resident_weights_ready() {return Ok(Vec::new());}
        let Some(frontier)=projection_stage::frontier(index,plan,&projection_stage::ready_names(values)) else {return Ok(Vec::new());};
        let mut jobs=Vec::new();
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
                jobs.push(Job::Routed {steps:&plan.steps[group.indices[0]..=last],ids:&values[&first.inputs[1]],input:&values[&first.inputs[0]]});
            }
        }
        let Some(output)=self.compute.projection_stage(self.checkpoint,&jobs)? else {return Ok(Vec::new());};
        need(output.len()==frontier.outputs.len(),"projection stage output count differs")?;
        for (stage,value) in frontier.outputs.iter().zip(output) {values.insert(plan.steps[*stage].outputs[0].name.clone(),value);}
        Ok(frontier.skipped)
    }
}
