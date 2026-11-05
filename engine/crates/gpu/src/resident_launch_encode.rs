//! Numeric field serialization preserves every original ABI field and zeroes padding.
use crate::resident_launch::{Command,Kind};
macro_rules! fields {($c:ident,$t:ty,$p:expr,$($f:ident),+)=>{{let p=$p;$( $c.put(core::mem::offset_of!($t,$f),&p.$f.to_ne_bytes()); )+}}}
pub(super) fn causal(p:crate::causal_ops::Arguments)->Command {
    use crate::causal_ops::{Arguments,Shape};let mut c=Command::new(Kind::Causal);
    for (i,v) in p.input.into_iter().enumerate(){c.word(i,v);}fields!(c,Arguments,p,weight,output);
    let base=core::mem::offset_of!(Arguments,shape);let mut s=Command::new(Kind::Causal);
    s.put(core::mem::offset_of!(Shape,op),&(p.shape.op as u32).to_ne_bytes());fields!(s,Shape,p.shape,heads,dim,taps,rows,cols,epsilon,bound);
    c.put(base,&s.payload[..core::mem::size_of::<Shape>()]);c
}
fn expert_arguments(p:crate::expert_ops::Arguments)->Command {
    use crate::expert_ops::{Arguments,Shape};let mut c=Command::new(Kind::Expert);
    for (i,v) in p.input.into_iter().enumerate(){c.word(i,v);}fields!(c,Arguments,p,weight,scales,output);
    let base=core::mem::offset_of!(Arguments,shape);let mut s=Command::new(Kind::Expert);
    s.put(core::mem::offset_of!(Shape,op),&(p.shape.op as u32).to_ne_bytes());fields!(s,Shape,p.shape,rows,cols,block_rows,block_cols,tokens,global,activation,bound);
    c.put(base,&s.payload[..core::mem::size_of::<Shape>()]);c
}
pub(super) fn expert(p:crate::expert_ops::LaunchArguments)->Command {let mut c=expert_arguments(p.single);fields!(c,crate::expert_ops::LaunchArguments,p,batch,count);c}
pub(super) fn mx(p:crate::next_params::MxLayer)->Command {
    use crate::next_params::{MxLayer,MxProjection};let mut c=Command::new(Kind::Mx);
    for (i,p) in p.projections.into_iter().enumerate(){let mut s=Command::new(Kind::Mx);fields!(s,MxProjection,p,weights,scales,input,output,rows,cols,tokens,input_bf16,round);c.put(i*core::mem::size_of::<MxProjection>(),&s.payload[..core::mem::size_of::<MxProjection>()]);}
    for (i,v) in p.ends.into_iter().enumerate(){c.put(core::mem::offset_of!(MxLayer,ends)+i*4,&v.to_ne_bytes());}
    fields!(c,MxLayer,p,count);c
}
pub(super) fn prepare(p:crate::mhc_params::Prepare)->Command {
    use crate::mhc_params::{Prepare,Shape};let mut c=Command::new(Kind::Prepare);fields!(c,Prepare,p,input,weights,base,scale,normalized,coefficients,collapsed);
    let mut s=Command::new(Kind::Prepare);fields!(s,Shape,p.shape,rows,streams,hidden,iterations,rms_eps,eps);
    c.put(core::mem::offset_of!(Prepare,shape),&s.payload[..core::mem::size_of::<Shape>()]);c
}
pub(super) fn round(p:crate::mhc_boundary::LaneCast)->Command {let mut c=Command::new(Kind::Round);fields!(c,crate::mhc_boundary::LaneCast,p,input,output,words);c}
pub(super) fn collapse(p:crate::mhc_params::Collapse)->Command {let mut c=Command::new(Kind::Collapse);fields!(c,crate::mhc_params::Collapse,p,input,coefficients,output,rows,streams,hidden);c}
pub(super) fn combine(p:crate::mhc_params::Combine)->Command {let mut c=Command::new(Kind::Combine);fields!(c,crate::mhc_params::Combine,p,input,coefficients,block,output,rows,streams,hidden);c}
pub(super) fn rms(p:crate::kernel_params::ModelRms,kind:Kind)->Command {let mut c=Command::new(kind);fields!(c,crate::kernel_params::ModelRms,p,input,weight,partials,output,width,rows,epsilon);c}
pub(super) fn elementwise(p:crate::kernel_params::ModelElementwise)->Command {let mut c=Command::new(Kind::Elementwise);fields!(c,crate::kernel_params::ModelElementwise,p,input,peer,output,elements,operation,scalar);c}
pub(super) fn copy(p:crate::kernel_params::HybridCopy)->Command {let mut c=Command::new(Kind::Copy);c.word(0,p.source as u64);c.word(1,p.destination as u64);fields!(c,crate::kernel_params::HybridCopy,p,words,padding,rows);c}
#[cfg(test)] mod tests {
    use super::*;
    fn u64_at(c:&Command,at:usize)->u64 {u64::from_ne_bytes(c.payload[at..at+8].try_into().unwrap())}
    fn u32_at(c:&Command,at:usize)->u32 {u32::from_ne_bytes(c.payload[at..at+4].try_into().unwrap())}
    #[test] fn original_numeric_abis_and_zero_padding_are_serialized_explicitly() {
        use core::mem::{offset_of,size_of};
        let shape=crate::causal_ops::Shape {op:crate::causal_ops::Op::Step,heads:3,dim:7,taps:4,rows:2,cols:6,epsilon:-0.0,bound:-1.0};
        let p=crate::causal_ops::Arguments {input:[16,32,48,64],weight:80,output:96,shape};let c=causal(p);
        assert_eq!(u64_at(&c,offset_of!(crate::causal_ops::Arguments,output)),96);assert_eq!(u64_at(&c,24),64);
        let at=offset_of!(crate::causal_ops::Arguments,shape);assert_eq!(u32_at(&c,at),shape.op as u32);assert_eq!(u32_at(&c,at+24),(-0.0f32).to_bits());
        assert!(c.payload[size_of::<crate::causal_ops::Arguments>()..].iter().all(|&v|v==0));
        let p=crate::expert_ops::Arguments {input:[16,32,48,64],weight:80,scales:96,output:112,shape:crate::expert_ops::Shape {op:crate::expert_ops::Op::NvProject,rows:3,cols:16,block_rows:1,block_cols:16,tokens:1,global:0.5,activation:2.0,bound:-0.0}};
        let c=expert(crate::expert_ops::LaunchArguments {single:p,batch:128,count:3});assert_eq!(u64_at(&c,offset_of!(crate::expert_ops::Arguments,output)),112);assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::Arguments,shape)+32),(-0.0f32).to_bits());
        assert_eq!(u64_at(&c,offset_of!(crate::expert_ops::LaunchArguments,batch)),128);assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::LaunchArguments,count)),3);
        assert_eq!(u32_at(&c,offset_of!(crate::expert_ops::Arguments,shape)+offset_of!(crate::expert_ops::Shape,tokens)),1);
        let mut p=crate::next_params::MxLayer::default();p.push(crate::next_params::MxProjection {weights:16,scales:32,input:48,output:64,rows:3,cols:32,tokens:1,input_bf16:0,round:0}).unwrap();let c=mx(p);
        assert_eq!(u64_at(&c,24),64);assert_eq!(u32_at(&c,offset_of!(crate::next_params::MxLayer,count)),1);
        for (i,v) in p.ends.iter().enumerate(){assert_eq!(u32_at(&c,offset_of!(crate::next_params::MxLayer,ends)+i*4),*v);}
        let p=crate::mhc_params::Prepare {input:16,weights:32,base:48,scale:64,normalized:80,coefficients:96,collapsed:112,shape:crate::mhc_params::Shape {rows:1,streams:2,hidden:3,iterations:4,rms_eps:1e-6,eps:1e-5}};let c=prepare(p);
        assert_eq!(u64_at(&c,40),96);assert_eq!(u64_at(&c,offset_of!(crate::mhc_params::Prepare,collapsed)),112);
        assert_eq!(u32_at(&c,offset_of!(crate::mhc_params::Prepare,shape)+offset_of!(crate::mhc_params::Shape,iterations)),4);assert!(c.payload[size_of::<crate::mhc_params::Prepare>()..].iter().all(|&v|v==0));
        let c=round(crate::mhc_boundary::LaneCast {input:16,output:32,words:9});assert_eq!(u64_at(&c,8),32);assert_eq!(u32_at(&c,16),9);
        let c=collapse(crate::mhc_params::Collapse {input:16,coefficients:32,output:48,rows:1,streams:2,hidden:3});assert_eq!(u32_at(&c,32),3);assert!(c.payload[36..40].iter().all(|&v|v==0));
        let c=combine(crate::mhc_params::Combine {input:16,coefficients:32,block:48,output:64,rows:1,streams:2,hidden:3});assert_eq!(u64_at(&c,24),64);assert_eq!(u32_at(&c,40),3);
        let c=rms(crate::kernel_params::ModelRms {input:16,weight:32,partials:48,output:64,width:5,rows:1,epsilon:-0.0},Kind::RmsApply);assert_eq!(u32_at(&c,40),(-0.0f32).to_bits());assert_eq!(u32_at(&c,44),0);
        let c=elementwise(crate::kernel_params::ModelElementwise {input:16,peer:32,output:48,elements:5,operation:4,scalar:-0.0});assert_eq!(u32_at(&c,32),(-0.0f32).to_bits());
        let c=copy(crate::kernel_params::HybridCopy {source:16 as _,destination:32 as _,words:5,padding:0,rows:48});assert_eq!(u64_at(&c,0),16);assert_eq!(u64_at(&c,8),32);assert_eq!(u64_at(&c,24),48);
    }
}
