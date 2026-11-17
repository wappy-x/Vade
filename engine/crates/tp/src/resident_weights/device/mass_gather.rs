//! DRAFT CONFIDENCE: the mailbox gather of the draft head's four-word picks (`logits_argmax_mass`).
use super::*;
use super::super::argmax_mass::merge_argmax_mass;
/// DRAFT CONFIDENCE: `gather_argmax` of the four-word parts of `logits_argmax_mass`: the pick and its probability.
pub fn gather_argmax_mass<'d>(b:&Backend<'d>,mailbox:&spark_gpu_probe::driver::TpMailbox<'d>,sequence:&mut u64,topology:Topology,local:&DeviceVector<'d>,owners:&[Rows])->Result<(u32,f32)> {
    let parts=topology.parts as usize;let total=parts.checked_mul(4).ok_or("resident argmax exchange overflow")?;
    if local.len()!=4 || owners.len()!=parts {return Err("resident argmax part extent differs".into());}
    let rows=Rows {start:topology.rank as usize*4,count:4,total,split:true};
    let gathered=gather_device(b,mailbox,sequence,topology,&[(local,false)],&[rows])?;
    merge_argmax_mass(&b.read_vector_values(&gathered[0])?,owners)
}
