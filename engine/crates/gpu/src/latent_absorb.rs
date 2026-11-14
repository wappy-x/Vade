//! Fixed-order latent attention arithmetic for admitted checkpoint projections.
//! The projection keeps BF16 checkpoint operands. Absorbed query and latent output round to BF16.
use crate::causal_ops::{Op,Shape};
/// SUM-ORDER v1: 32 strips, eight adjacent terms per strip per period.
pub const LANES:usize=32;
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
fn weight(w:&[f32],i:usize)->f32 {let bits=w[i/2].to_bits();f32::from_bits(if i.is_multiple_of(2) {bits<<16} else {bits&0xffff_0000})}
#[cfg(not(target_arch="nvptx64"))]
#[inline(always)]
pub fn unit(s:Shape,index:usize,x:[&[f32];4],w:&[f32],out:&mut[f32],mul:impl Fn(f32,f32)->f32,add:impl Fn(f32,f32)->f32) {
    let (d,v,r,k)=(s.dim as usize,s.taps as usize,s.rows as usize,s.cols as usize);
    let (head,feature,n)=match s.op {
        Op::AbsorbQuery=>(index/k,index%k,d),
        Op::AbsorbScores=>(index/r,index%r,k),
        Op::AbsorbValues=>(index/k,index%k,x[2][0] as usize),
        Op::AbsorbProject=>(index/v,index%v,k),
        _=>return,
    };
    if s.op==Op::AbsorbScores && feature>=x[2][0] as usize {out[0]=0.0;return;}
    let mut sums=[0.0f32;LANES];let mut at=0;
    while at<n {let j=at;let sum=&mut sums[(j/8)%LANES];
        let (a,b)=match s.op {
            Op::AbsorbQuery=>(x[0][head*d+j],weight(w,head*(d+v)*k+feature*d+j)),
            Op::AbsorbScores=>(x[0][head*k+j],x[1][feature*k+j]),
            Op::AbsorbValues=>(x[0][head*r+j],x[1][j*k+feature]),
            Op::AbsorbProject=>(x[0][head*k+j],weight(w,(head*(d+v)+d+feature)*k+j)),
            _=>return,
        };*sum=add(*sum,mul(a,b));
    at+=1;}
    for distance in [16,8,4,2,1] {for lane in 0..distance {sums[lane]=add(sums[lane],sums[lane+distance]);}}
    let sum=sums[0];
    out[0]=match s.op {
        Op::AbsorbScores=>mul(sum,1.0/crate::causal_ops::root(d as f32)),
        _=>round(sum),
    };
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
