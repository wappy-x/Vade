//! Block arms of the routed experts and of the output: `CappedGate`, `RouteFold`, `Softcap`. Safe shared code for
//! the host and the GPU.
//!
//! Shape fields: `d` = dim, `r` = rows. Counts are 32-bit words.
//! - `CappedGate`: `r` picks, `d` = width of one half; `epsilon` = cap of the gate half, `bound` = cap of the up half
//!   (both above zero). Input: [r*2*d], for each pick its gate half and then its up half. Weight: 2 words, the
//!   reciprocal of the gate cap and then the reciprocal of the up cap. Units `r*d`, words 1.
//! - `RouteFold`: `r` picks, `d` = width. Inputs: the weights [r] in pick order, the down rows [r*d]. Units `d`,
//!   words 1.
//! - `Softcap`: `r` values; `bound` = the cap, `epsilon` = the reciprocal of the cap (both above zero). Input: [r].
//!   Units `r`, words 1.
//!
//! No arm divides by a cap (ruling R8). Where the source divides a 32-bit tensor by a number, its GPU answer is one
//! 32-bit product with the reciprocal (measured on the source, lab receipt `divform.json` of 2026-11-22). The host
//! makes each reciprocal one time, in 64-bit, rounded one time to 32-bit, and gives it in the weight or in the
//! shape. An arm does not test that a reciprocal goes with its cap.
//!
//! Number rules. Every value is one FP32 word. A product and its sum are two rounded steps, never a fused step.
//! An input that its producer holds as BF16 is read through `rn`; on a word that is BF16 already `rn` changes
//! nothing. Thus a dense product can leave its FP32 sum in memory, and the bytes are those of "round, then use".
//! `sigmoid`, `tanh` and `rn` come from `fixed_math`.
//!
//! What each arm computes, for one output word, with `cap(x, r, c) = tanh(x * r) * c`:
//! - `CappedGate`, element `j` of pick `p`: `gate` and `up` are read through `rn`. The result is
//!   `rn((cap(gate, weight[0], epsilon) * sigmoid(gate)) * cap(up, weight[1], bound))`. The sigmoid reads the gate,
//!   not the capped gate.
//! - `RouteFold`, channel `c`: the sum of `rn(down[p][c]) * weight[p]` from +0.0 in the order of the picks, then one
//!   `rn`. The weights are FP32 and are not rounded when they are read. The router fixes the order of the picks.
//! - `Softcap`: `cap(rn(x), epsilon, bound)`. The result stays 32-bit.
//!
//! Source of the math (read for the math, no code copied): the public Transformers release v5.24.0, file
//! `models/inkling2_loop/modeling_inkling2_loop.py`. Expert activation: lines 751-771. Sum of the picked experts:
//! lines 787-807. Cap of the output: lines 1814-1816.
use crate::{block_ops::{Lengths,Op,Shape},fixed_math::{rn,sigmoid,tanh}};
/// The soft cap in the ruled form: `tanh(x * reciprocal) * cap`. A product, then the shared `tanh`, then a product.
#[inline(always)]
fn cap(x:f32,reciprocal:f32,cap:f32)->f32 {tanh(x*reciprocal)*cap}
/// Word counts of an arm of this file. `None` refuses the shape.
#[inline]
pub fn lengths(s:Shape)->Option<Lengths> {
    let (d,r)=(s.dim as usize,s.rows as usize);
    match s.op {
        Op::CappedGate if r>0 && d>0 && s.epsilon>0.0 && s.bound>0.0=>Some(([r.checked_mul(2)?.checked_mul(d)?,0,0,0],2,r.checked_mul(d)?,1)),
        Op::RouteFold if r>0 && d>0=>Some(([r,r.checked_mul(d)?,0,0],0,d,1)),
        Op::Softcap if r>0 && s.epsilon>0.0 && s.bound>0.0=>Some(([r,0,0,0],0,r,1)),
        _=>None,
    }
}
/// Write one output unit of an arm of this file. The caller has checked every extent with `lengths`.
#[inline]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    let d=s.dim as usize;
    match s.op {
        Op::CappedGate=>{
            let (pick,element)=(index/d,index%d);let gate=rn(x[0][pick*2*d+element]);let up=rn(x[0][pick*2*d+d+element]);
            out[0]=rn((cap(gate,w[0],s.epsilon)*sigmoid(gate))*cap(up,w[1],s.bound));
        },
        Op::RouteFold=>{
            let mut sum=0.0f32;for (pick,&weight) in x[0].iter().enumerate() {sum+=rn(x[1][pick*d+index])*weight;}
            out[0]=rn(sum);
        },
        Op::Softcap=>{out[0]=cap(rn(x[0][index]),s.epsilon,s.bound);},
        _=>{},
    }
}
