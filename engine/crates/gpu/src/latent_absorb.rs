//! Fixed-order latent attention arithmetic for admitted checkpoint projections, one token (the step form).
//! The projection keeps BF16 checkpoint operands. Absorbed query and latent output round to BF16.
//! The word function is the tensor-core fold of `absorb_tc` (twin: `tc_twin_bf16::fold`), the same as the prompt form.
use crate::causal_ops::{Op,Shape};
/// BF16 round of an absorbed word that keeps every NaN a NaN (quiet, sign and top payload bits kept).
/// The shared round adds a carry that can reach the sign bit: the device's canonical NaN 0x7fffffff becomes -0.0.
/// Finite and infinite words round exactly as `causal_ops::round`. Contract: NaN locations agree, payloads do not.
#[inline(always)]
pub fn round(x:f32)->f32 {if x.is_nan() {f32::from_bits((x.to_bits()|0x0040_0000)&0xffff_0000)} else {crate::causal_ops::round(x)}}
pub fn is_absorb(op:Op)->bool {matches!(op,Op::AbsorbQuery|Op::AbsorbScores|Op::AbsorbValues|Op::AbsorbProject)}
pub fn lengths(s:Shape)->Option<([usize;4],usize,usize,usize)> {
    let (h,d,v,r,k)=(s.heads as usize,s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    if h==0 || d==0 || v==0 || r==0 || k==0 || k%2!=0 {return None;}
    let weights=h.checked_mul(d.checked_add(v)?)?.checked_mul(k)?/2;
    match s.op {
        Op::AbsorbQuery=>Some(([h.checked_mul(d)?,0,0,0],weights,h.checked_mul(k)?,1)),
        Op::AbsorbScores=>Some(([h.checked_mul(k)?,r.checked_mul(k)?,1,0],0,h.checked_mul(r)?,1)),
        Op::AbsorbValues=>Some(([h.checked_mul(r)?,r.checked_mul(k)?,1,0],0,h.checked_mul(k)?,1)),
        Op::AbsorbProject=>Some(([h.checked_mul(k)?,0,0,0],weights,h.checked_mul(v)?,1)),
        _=>None,
    }
}
#[cfg(not(target_arch="nvptx64"))]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32]) {
    use crate::absorb_tc::{Form,word};
    let Some(f)=Form::step(s) else {return;};
    // An output unit is [head][feature] of the one token.
    let width=f.m();let (head,feature)=(index/width,index%width);
    let (outer,n)=if f.attention() {(0,head)} else {(head,0)};
    out[0]=word(f,outer,feature,n,x[0],x[1],x[2],w);
}
#[cfg(test)]
mod tests {
    #[test]
    fn round_keeps_nan_and_finite_words() {
        for bits in [0x7fff_ffffu32,0xffff_ffff,0x7fc0_0000,0xffeb_1234,0x7f80_0001,0x7fff_8000] {assert!(super::round(f32::from_bits(bits)).is_nan(),"{bits:08x}");}
        assert_eq!(crate::causal_ops::round(f32::from_bits(0x7fff_ffff)).to_bits(),0x8000_0000);
        for x in [0.0f32,-0.0,1.0,-1.5,3.0e38,f32::MAX,f32::INFINITY,f32::NEG_INFINITY,f32::from_bits(1),3.140_625e-3] {assert_eq!(super::round(x).to_bits(),crate::causal_ops::round(x).to_bits());}
    }
}
