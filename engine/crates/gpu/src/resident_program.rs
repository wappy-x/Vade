//! Checked ordered graph pieces. Slot indices name inputs or earlier outputs.
use crate::causal_ops::{Shape,Op};
#[derive(Clone)]
pub struct Node {pub shape:Shape,pub inputs:[Option<usize>;4]}
pub struct Program {pub inputs:Vec<usize>,pub nodes:Vec<Node>}
impl Program {
    pub fn validate(&self)->Result<Vec<usize>,String> {
        if self.nodes.is_empty() || self.nodes.len()>128 || self.inputs.iter().any(|&n|n==0 || n>u32::MAX as usize) {return Err("resident program geometry exceeds limits".into());}
        let mut slots=self.inputs.clone();
        for node in &self.nodes {
            let (lengths,_,units,words)=node.shape.lengths().ok_or("resident program operation shape differs")?;
            for (slot,words) in node.inputs.iter().zip(lengths) {match slot {
                Some(i) if words>0 && slots.get(*i)==Some(&words)=>{},None if words==0=>{},_=>return Err("resident program input does not name an earlier exact slot".into()),
            }}
            slots.push(units.checked_mul(words).ok_or("resident program output overflow")?);
        }
        Ok(slots)
    }
    /// Preserve the existing channel/column work followed by its layout kernel.
    pub fn operation(shape:Shape)->Result<Self,String> {
        let (lengths,_,_,_)=shape.lengths().ok_or("resident program shape differs")?;
        let mut inputs=Vec::new();let links=lengths.map(|n|if n==0 {None} else {let i=inputs.len();inputs.push(n);Some(i)});
        let phases=match shape.op {Op::Step=>Some((Op::StepColumns,Op::StepLayout)),Op::Conv=>Some((Op::ConvChannels,Op::ConvLayout)),Op::Pool=>Some((Op::PoolChannels,Op::PoolLayout)),_=>None};
        let nodes=if let Some((first,last))=phases {vec![Node {shape:Shape {op:first,..shape},inputs:links},Node {shape:Shape {op:last,..shape},inputs:[Some(inputs.len()),None,None,None]}]}
            else {vec![Node {shape,inputs:links}]};
        let program=Self {inputs,nodes};program.validate()?;Ok(program)
    }
}
#[cfg(test)] mod tests {
    use super::*;
    fn shape(op:Op)->Shape {Shape {op,heads:2,dim:4,taps:3,rows:4,cols:4,epsilon:1e-6,bound:-1.0}}
    #[test] fn expanded_chains_keep_original_extents() {
        for op in [Op::Step,Op::Conv,Op::Pool] {let s=shape(op);let p=Program::operation(s).unwrap();let slots=p.validate().unwrap();let (_,_,u,w)=s.lengths().unwrap();assert_eq!(slots.last(),Some(&(u*w)));assert_eq!(p.nodes.len(),2);}
    }
    #[test] fn rejects_forward_edges_wrong_shapes_and_empty_programs() {
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.nodes[0].inputs[0]=Some(p.inputs.len());assert!(p.validate().is_err());
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.inputs[0]+=1;assert!(p.validate().is_err());
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.nodes.clear();assert!(p.validate().is_err());
    }
}
