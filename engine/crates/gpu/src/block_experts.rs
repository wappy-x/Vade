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
//! - `RouterPick` (read arm 2): the router pick of one token, the fields of `Op::Route` in its one-group form:
//!   `heads` = `dim` = 1, `r` experts (2 to `PICK_EXPERTS_MAX`), `cols` = `k` picks (1 to `r`), `taps` = 1 when the picked scores
//!   are divided by their sum, `bound` = the route scale (above zero). Input: the router logits [r]. Weight: the
//!   bias [r]. Units 1, words `2k`: the picked expert numbers, then their weights, in pick order. These are the first
//!   `2k` words that `Op::Route` writes, bit for bit; its scratch words after them are not written (host test).
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
//! - `RouterPick`: `score[e] = sigmoid(logit[e])`. Pick `k` times the expert of the largest `score + bias` not yet
//!   picked, scanned in rising number with a strict `>` from minus infinity (so the lower number first on equal
//!   values, and expert 0 when no value is above minus infinity). Its weight is its score; with `taps` = 1 each
//!   weight is divided by `sum + 1e-20`, the sum of the picked scores in pick order from +0.0; then each weight is
//!   multiplied by the scale. These are the steps of `Op::Route` in the same order (`latent_ops.rs`); a division by
//!   a sum, not by a cap, so ruling R8 does not apply.
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
        Op::RouterPick if s.heads==1 && d==1 && (2..=PICK_EXPERTS_MAX).contains(&r) && k_of(s)>0 && k_of(s)<=r && s.taps<=1 && s.bound>0.0=>Some(([r,0,0,0],r,1,k_of(s).checked_mul(2)?)),
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
        Op::RouterPick=>{
            let k=k_of(s);let (ids,weights)=out.split_at_mut(k);
            let mut sum=0.0f32;
            for slot in 0..k {
                let (mut id,mut best)=(0usize,f32::NEG_INFINITY);
                for (expert,&logit) in x[0].iter().enumerate() {
                    // An expert picked before is out, by its number: each word of `ids` is a whole number below
                    // PICK_EXPERTS_MAX, so the float holds it exactly (Op::Route marks it in its scratch instead).
                    if ids[..slot].iter().any(|&picked|picked as usize==expert) {continue;}
                    let choice=sigmoid(logit)+w[expert];
                    if choice>best {best=choice;id=expert;}
                }
                let score=sigmoid(x[0][id]);ids[slot]=id as f32;weights[slot]=score;sum+=score;
            }
            for v in weights.iter_mut() {if s.taps!=0 {*v/=sum+1e-20;}*v*=s.bound;}
        },
        _=>{},
    }
}
/// The most experts of `RouterPick`: 65,536, the expert cap of the index rows of kind 96 (lead order 22:39 on
/// review REVARM2 finding 2). It is far below 2^24, so every expert number and every picked word is exact as a
/// float and below the expert count.
pub const PICK_EXPERTS_MAX:usize=1<<16;
/// The picks `k` of `RouterPick` (field `cols`).
#[inline(always)]
fn k_of(s:Shape)->usize {s.cols as usize}
