use crate::Result;
use engine_model::plan::{layer::{Kind,LayerPlan,Step},storage::WeightStorage};
use spark_gpu_probe::causal_ops::Shape;
use std::collections::{BTreeMap,BTreeSet};

pub enum Job<'a,V> {
    Project {input:Vec<(&'a str,&'a V)>,round:bool},
    Routed {steps:&'a [Step],ids:&'a V,input:&'a V},
    Gated {name:&'a str,gate:&'a V,up:&'a V,shape:Shape},
}
#[derive(Debug,PartialEq,Eq)]
pub struct Group {pub indices:Vec<usize>,pub middle:Option<usize>}
#[derive(Debug,PartialEq,Eq)]
pub struct Frontier {pub groups:Vec<Group>,pub outputs:Vec<usize>,pub skipped:Vec<usize>}
fn sole_output(step:&Step)->Option<&str> {if step.outputs.len()==1 {Some(&step.outputs[0].name)} else {None}}
fn mixed(step:&Step,plan:&LayerPlan)->bool {
    step.kind==Kind::Project && step.weights.len()==1 && step.weights[0].shape.len()==2 && step.weights[0].shape[1]%32==0
        && matches!(step.weights[0].storage,WeightStorage::Mxfp8|WeightStorage::Plain)
        && !plan.steps.iter().any(|route|route.kind==Kind::Route && route.inputs.iter().any(|input|Some(input.as_str())==sole_output(step)))
}
fn compatible(first:&Step,next:&Step,plan:&LayerPlan)->bool {
    if first.inputs!=next.inputs {return false;}
    if mixed(first,plan) && mixed(next,plan) {return true;}
    // Block-FP8 matrices of one storage and one shape (the shared MLP's gate and up) are one two-job launch, like the NVFP4 pairs: each job runs the
    // same kernel on the same bytes, so only the launch count changes.
    matches!(first.weights[0].storage,WeightStorage::Nvfp4 {..}|WeightStorage::Fp8Block {..}) && first.weights[0].storage==next.weights[0].storage
        && first.weights[0].shape==next.weights[0].shape
        && !plan.steps.iter().any(|route|route.kind==Kind::Route && (route.inputs.contains(&first.outputs[0].name) || route.inputs.contains(&next.outputs[0].name)))
}
pub fn frontier(index:usize,plan:&LayerPlan,available:&BTreeSet<&str>)->Option<Frontier> {
    if index>=plan.steps.len() {return None;}
    let mut writes=BTreeSet::new();
    if plan.steps.iter().flat_map(|step|&step.outputs).any(|out|!writes.insert(out.name.as_str())) {return None;}
    let mut groups:Vec<Group>=Vec::new();let mut outputs=Vec::new();let mut skipped=Vec::new();let mut taken=BTreeSet::new();
    for stage in index..plan.steps.len() {
        let step=&plan.steps[stage];let Some(output)=sole_output(step) else {continue;};
        if taken.contains(&stage) || available.contains(output) {continue;}
        if step.kind==Kind::Project && step.inputs.len()==1 && step.weights.len()==1 {
            let source=step.inputs[0].as_str();
            let middle=if available.contains(source) {None} else {
                let Some((at,producer))=plan.steps.iter().enumerate().find(|(_,producer)|sole_output(producer)==Some(source)) else {continue;};
                if at<index || producer.kind!=Kind::SwiGlu || producer.inputs.len()!=2 || !producer.weights.is_empty()
                    || producer.inputs.iter().any(|input|!available.contains(input.as_str())) || plan.returns.iter().any(|name|name==source)
                    || plan.steps.iter().filter(|consumer|consumer.inputs.iter().any(|input|input==source)).count()!=1 {continue;}
                Some(at)
            };
            if middle.is_none() {
                if let Some(group)=groups.iter_mut().find(|group|group.middle.is_none() && group.indices.len()<9
                    && plan.steps[group.indices[0]].kind==Kind::Project && compatible(&plan.steps[group.indices[0]],step,plan)) {
                    group.indices.push(stage);outputs.push(stage);taken.insert(stage);continue;
                }
            }
            groups.push(Group {indices:vec![stage],middle});outputs.push(stage);taken.insert(stage);
            if let Some(middle)=middle {skipped.push(middle);}
        } else if matches!(step.kind,Kind::ExpertProject|Kind::ExpertDown) && step.inputs.len()==2
            && step.inputs.iter().all(|input|available.contains(input.as_str())) {
            let span=if step.kind==Kind::ExpertProject && plan.steps.get(stage+1).is_some_and(|next|next.kind==step.kind
                && next.inputs==step.inputs && sole_output(next).is_some_and(|name|!available.contains(name))) {2} else {1};
            let indices=(stage..stage+span).collect::<Vec<_>>();outputs.extend(&indices);taken.extend(&indices);groups.push(Group {indices,middle:None});
        }
    }
    if groups.len()<2 || groups.len()>16 || !outputs.contains(&index) {return None;}
    outputs=groups.iter().flat_map(|group|group.indices.iter().copied()).collect();skipped.extend(outputs.iter().copied().filter(|&at|at>index));
    Some(Frontier {groups,outputs,skipped})
}
#[derive(Clone,Copy,Debug,Default,PartialEq,Eq)]
pub struct Counts {pub accepted:u64,pub original_messages:u64,pub words:u64}
std::thread_local! {static COUNTS:std::cell::Cell<Counts>=const {std::cell::Cell::new(Counts {accepted:0,original_messages:0,words:0})};}
pub fn counts()->Counts {COUNTS.with(std::cell::Cell::get)}
#[cfg(target_os="linux")]
pub(crate) fn accepted(messages:usize,words:usize) {COUNTS.with(|counts| {let old=counts.get();counts.set(Counts {accepted:old.accepted+1,original_messages:old.original_messages+messages as u64,words:old.words+words as u64});});}
pub fn ready_names<V>(values:&BTreeMap<String,V>)->BTreeSet<&str> {values.keys().map(String::as_str).collect()}
pub fn checked_messages(jobs:usize,words:usize,slot_bytes:usize)->Result<()> {
    if !(2..=16).contains(&jobs) || words==0 || words.checked_mul(4).is_none_or(|bytes|bytes>slot_bytes) {return Err("projection stage is outside the bounded exchange envelope".into());}Ok(())
}
pub fn checked_rows(rows:&[crate::resident_weights::Rows])->Result<usize> {
    if rows.is_empty() || rows.len()>32 || rows.iter().any(|row|!row.split || row.count==0 || row.total%row.count!=0
        || row.start%row.count!=0 || row.start.checked_add(row.count).is_none_or(|end|end>row.total)) {return Err("projection stage rows are outside the split envelope".into());}
    let chunks=rows[0].total/rows[0].count;
    if chunks<2 || rows.iter().any(|row|row.total/row.count!=chunks || row.start/row.count!=rows[0].start/rows[0].count) {return Err("projection stage row groups differ".into());}
    let words=rows.iter().try_fold(0usize,|sum,row|sum.checked_add(row.count)).ok_or("projection stage rows overflow")?;
    words.checked_mul(chunks).ok_or("projection stage gathered extent overflow")?;Ok(words)
}
