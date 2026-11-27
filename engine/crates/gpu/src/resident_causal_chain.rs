use crate::{causal_ops::{Arguments,Op,Shape},resident_launch::{Command,Kind}};
fn word(command:&Command,offset:usize)->u64 {u64::from_ne_bytes(command.payload[offset..offset+8].try_into().unwrap())}
fn field(command:&Command,offset:usize)->u32 {let offset=core::mem::offset_of!(Arguments,shape)+offset;u32::from_ne_bytes(command.payload[offset..offset+4].try_into().unwrap())}
pub(crate) fn fuse(prepare:&Command,step:&Command,norm:&Command)->Option<(Command,u32)> {
    if [prepare,step,norm].iter().any(|command|command.kind!=Kind::CausalFast as u32) {return None;}
    if field(prepare,0)!=Op::Prepare as u32 || field(step,0)!=Op::StepPlanar as u32 || field(norm,0)!=Op::GatedNorm as u32 {return None;}
    let heads=field(prepare,core::mem::offset_of!(Shape,heads));let dim=field(prepare,core::mem::offset_of!(Shape,dim));
    if heads==0 || dim!=128 || [step,norm].iter().any(|command|field(command,core::mem::offset_of!(Shape,heads))!=heads || field(command,core::mem::offset_of!(Shape,dim))!=dim) {return None;}
    if word(step,0)!=word(prepare,core::mem::offset_of!(Arguments,output)) {return None;}
    let vector=word(step,core::mem::offset_of!(Arguments,output)).checked_add(heads as u64*dim as u64*dim as u64*4)?;
    if word(norm,0)!=vector {return None;}
    let mut command=Command::new(Kind::CausalChain);
    for (offset,source) in [(0,prepare),(256,step),(512,norm)] {command.put(offset,&source.payload[..core::mem::size_of::<Arguments>()]);}
    Some((command,heads))
}
