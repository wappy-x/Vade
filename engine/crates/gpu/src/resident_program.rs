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
    /// Keep the split layouts except for the byte-swept direct Step and Conv outputs.
    pub fn operation(shape:Shape)->Result<Self,String> {
        let (lengths,_,_,_)=shape.lengths().ok_or("resident program shape differs")?;
        let mut inputs=Vec::new();let links=lengths.map(|n|if n==0 {None} else {let i=inputs.len();inputs.push(n);Some(i)});
        let phases=match shape.op {Op::Step if !crate::causal_ops::fast_admitted(shape,crate::causal_ops::FAST_THREADS)=>Some((Op::StepColumns,Op::StepLayout)),Op::Conv if !crate::causal_ops::fast_admitted(shape,crate::causal_ops::FAST_THREADS)=>Some((Op::ConvChannels,Op::ConvLayout)),Op::Pool=>Some((Op::PoolChannels,Op::PoolLayout)),_=>None};
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
    #[test] fn direct_conv_uses_one_node_only_for_the_swept_family() {
        for (heads,dim,taps) in [(64,128,4),(1,1,2),(8,3,5),(65,129,3)] {
            let spec=Shape {heads,dim,taps,..shape(Op::Conv)};let program=Program::operation(spec).unwrap();
            let (w,t)=((heads*dim) as usize,taps as usize);
            assert_eq!(program.nodes.len(),1);assert_eq!(program.nodes[0].shape.op,Op::Conv);assert_eq!(program.nodes[0].inputs,[Some(0),Some(1),None,None]);
            assert_eq!(program.validate().unwrap(),vec![3*w,3*(t-1)*w,3*w*t]);
        }
        for (heads,dim,taps) in [(4,128,4),(64,100,4),(64,128,6),(64,128,1)] {
            let program=Program::operation(Shape {heads,dim,taps,..shape(Op::Conv)}).unwrap();
            assert_eq!(program.nodes.len(),2);assert_eq!(program.nodes[0].shape.op,Op::ConvChannels);assert_eq!(program.nodes[1].shape.op,Op::ConvLayout);
        }
    }
    #[test] fn planar_forms_are_single_node_programs() {
        for (op,heads,dim,taps) in [(Op::StepPlanar,64,128,0),(Op::ConvParts,64,128,4)] {
            let spec=Shape {op,heads,dim,taps,..shape(Op::Step)};let program=Program::operation(spec).unwrap();
            assert_eq!(program.nodes.len(),1);assert_eq!(program.nodes[0].shape.op,op);assert_eq!(program.validate().unwrap().last().copied(),Some(spec.lengths().unwrap().3));
        }
    }
    #[test] fn rejects_forward_edges_wrong_shapes_and_empty_programs() {
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.nodes[0].inputs[0]=Some(p.inputs.len());assert!(p.validate().is_err());
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.inputs[0]+=1;assert!(p.validate().is_err());
        let mut p=Program::operation(shape(Op::Step)).unwrap();p.nodes.clear();assert!(p.validate().is_err());
    }
    #[test] fn direct_step_uses_one_node_only_for_the_swept_family() {
        for heads in [1,2,3,17,63,64] {
            let spec=Shape {heads,dim:128,..shape(Op::Step)};let program=Program::operation(spec).unwrap();
            assert_eq!(program.nodes.len(),1);assert_eq!(program.nodes[0].shape.op,Op::Step);assert_eq!(program.nodes[0].inputs,[Some(0),Some(1),None,None]);
            assert_eq!(program.validate().unwrap(),vec![heads as usize*513,heads as usize*16384,heads as usize*16512]);
        }
        for (heads,dim) in [(4,128),(32,128),(65,128),(64,127),(64,129)] {
            let program=Program::operation(Shape {heads,dim,..shape(Op::Step)}).unwrap();
            assert_eq!(program.nodes.len(),2);assert_eq!(program.nodes[0].shape.op,Op::StepColumns);assert_eq!(program.nodes[1].shape.op,Op::StepLayout);
        }
    }
}
