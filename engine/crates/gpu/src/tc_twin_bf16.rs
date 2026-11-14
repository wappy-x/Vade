//! Kernel kit, dense family: CPU reference of a BF16 tensor-core product in the CANONICAL order
//! (`kernels/src/canonical.rs`). Integer arithmetic only, independent of the kernels.
//!
//! The word function of one output word (one weight row, one token row), for `cols` a multiple of 16:
//! 1. Each activation `x` is rounded once to BF16, round to nearest even (`pack_f32_bf16_mma_input`). There is no
//!    low part: measured on a box (night 1, `dense-tc-rows`), a model with a second BF16 value for `x - bf16(x)`
//!    differs from the device on general FP32 activations, and this model equals it on every checked word.
//! 2. The `cols / 16` steps are cut into four ranges: range `p` = steps `[n*p/4, n*(p+1)/4)`.
//! 3. In a range, one ascending chain of 16-column steps from +0.0. A step is the GB10 instruction
//!    `mma.sync m16n8k16 bf16`: the 16 products and the carried FP32 accumulator are lined up on the largest
//!    exponent, bits under 25 fraction bits are cut toward zero term by term, the sum is exact, and it is cut
//!    toward zero to FP32.
//! 4. `acc = fl(fl(part0 + part1) + fl(part2 + part3))`. `fl` is one IEEE FP32 add.
//! 5. `word = bf16(acc)`.
#[derive(Clone,Copy)]
struct Term {negative:bool,magnitude:u64,exponent:i32,top:i32}
fn operand(v:u16)->(bool,u64,i32) {let e=i32::from((v>>7)&255);let f=u64::from(v&127);(v&0x8000!=0,if e==0 {f} else {128+f},if e==0 {-126} else {e-127})}
fn carried(bits:u32)->Term {
    let e=((bits>>23)&255) as i32;let f=u64::from(bits&0x7f_ffff);
    Term {negative:bits>>31!=0,magnitude:if e==0 {f} else {(1<<23)+f},exponent:if e==0 {-149} else {e-150},top:if e==0 {-126} else {e-127}}
}
fn truncate(negative:bool,magnitude:u64,exponent:i32)->u32 {
    let sign=u32::from(negative)<<31;if magnitude==0 {return sign;}
    let e=63-magnitude.leading_zeros() as i32+exponent;
    let low=if e< -126 {-149} else {e-23};let shift=low-exponent;
    let kept=if shift<=0 {magnitude<<(-shift)} else if shift>=64 {0} else {magnitude>>shift};
    if e< -126 {return sign|kept as u32;}
    if e>127 {return sign|0x7f80_0000;}
    sign|((e+127) as u32)<<23|(kept as u32&0x7f_ffff)
}
/// One 16-column step: 16 BF16 weights, 16 BF16 activations, the carried accumulator. Returns the new accumulator.
pub fn step(weight:&[u16],input:&[u16],acc:u32)->u32 {
    assert!(weight.len()==16 && input.len()==16);
    if acc&0x7f80_0000==0x7f80_0000 {return acc;}
    let mut terms=[carried(acc);17];let mut top=terms[16].top;
    for k in 0..16 {let (na,sa,ea)=operand(weight[k]);let (nb,sb,eb)=operand(input[k]);
        terms[k]=Term {negative:na!=nb,magnitude:sa*sb,exponent:ea+eb-14,top:ea+eb};top=top.max(ea+eb);
    }
    let unit=top-25;let mut sum=0i64;
    for t in terms {let shift=t.exponent-unit;
        let value=if shift>=0 {t.magnitude<<shift} else if shift<= -64 {0} else {t.magnitude>>(-shift)};
        sum+=if t.negative {-(value as i64)} else {value as i64};
    }
    truncate(sum<0,sum.unsigned_abs(),unit)
}
/// Round to nearest even to BF16 (the 16 high bits of the FP32 pattern).
pub fn bf16(x:f32)->u16 {let b=x.to_bits();(b.wrapping_add(0x7fff+((b>>16)&1))>>16) as u16}
/// Steps of range `part` of `steps` 16-column steps (`canonical::range`).
pub fn range(steps:usize,part:usize)->(usize,usize) {(steps*part/4,steps*(part+1)/4)}
/// The four raw range accumulators of one word, before the join. When `cols / 16` is a multiple of 4 the ranges are
/// the column quarters: this is what the owners of a plain BF16 SECOND matrix hold in a column split (an owner sends
/// `tc_twin_fold::partial` of them), so 1, 2 and 4 owners give the bits of `canonical`.
pub fn parts(weight:&[u16],input:&[f32])->[u32;4] {
    assert!(weight.len()==input.len() && weight.len().is_multiple_of(16));
    let rounded:Vec<u16>=input.iter().map(|&x|bf16(x)).collect();
    let steps=weight.len()/16;let mut parts=[0u32;4];
    for (part,slot) in parts.iter_mut().enumerate() {
        let (start,end)=range(steps,part);let mut acc=0u32;
        for s in start..end {acc=step(&weight[s*16..s*16+16],&rounded[s*16..s*16+16],acc);}
        *slot=acc;
    }
    parts
}
/// The FP32 accumulator of one word before its BF16 boundary: `weight` is one row of BF16 patterns.
pub fn canonical(weight:&[u16],input:&[f32])->u32 {
    let parts=parts(weight,input).map(f32::from_bits);
    ((parts[0]+parts[1])+(parts[2]+parts[3])).to_bits()
}
/// The output word: the accumulator rounded to BF16, held in FP32.
pub fn word(acc:u32)->u32 {u32::from(bf16(f32::from_bits(acc)))<<16}
#[cfg(test)]
mod tests {
    use super::*;
    fn b(x:f32)->u16 {assert_eq!(f32::from_bits(u32::from(bf16(x))<<16),x,"not a BF16 value");bf16(x)}
    #[test]
    fn small_integers_give_the_exact_dot_product() {
        // Every product and partial sum is an integer below 2^20: nothing is cut, in any order.
        let weight:Vec<u16>=(0..256).map(|i|b(((i*7+3)%31) as f32-15.0)).collect();
        let input:Vec<f32>=(0..256).map(|i|((i*5+1)%29) as f32-14.0).collect();
        let want:f32=weight.iter().zip(&input).map(|(&w,&x)|f32::from_bits(u32::from(w)<<16)*x).sum();
        assert_eq!(f32::from_bits(canonical(&weight,&input)),want);
        assert_eq!(f32::from_bits(word(canonical(&weight,&input))),f32::from_bits(u32::from(bf16(want))<<16));
    }
    #[test]
    fn an_activation_is_rounded_once_to_bf16_nearest_even() {
        assert_eq!(bf16(1.0),0x3f80);assert_eq!(bf16(f32::from_bits(0x3f80_8000)),0x3f80);assert_eq!(bf16(f32::from_bits(0x3f81_8000)),0x3f82);
        assert_eq!(bf16(f32::from_bits(0x3f80_8001)),0x3f81);assert_eq!(bf16(-0.0),0x8000);
        // A general FP32 activation gives the word of its BF16 rounding: there is no low part.
        let weight:Vec<u16>=(0..64).map(|i|b(((i*7+3)%31) as f32-15.0)).collect();
        let general:Vec<f32>=(0..64).map(|i|f32::from_bits(0x3f81_2345+(i as u32)*0x0001_0101)).collect();
        let rounded:Vec<f32>=general.iter().map(|&x|f32::from_bits(u32::from(bf16(x))<<16)).collect();
        assert_ne!(general,rounded);assert_eq!(canonical(&weight,&general),canonical(&weight,&rounded));
    }
    #[test]
    fn a_step_cuts_toward_zero_under_25_fraction_bits_of_the_largest_term() {
        // 2^10 * 1 carried, plus 1 * (1 + 2^-7): the small product is cut at 2^(10-25).
        let mut w=[0u16;16];let mut x=[0u16;16];w[0]=b(1.0);x[0]=b(1.0+2f32.powi(-7));
        assert_eq!(f32::from_bits(step(&w,&x,1024f32.to_bits())),1025.0+2f32.powi(-7));
        // 2^20 carried: the product's 2^-7 part is under 2^(20-25) and is cut; FP32 keeps 2^(20-23).
        assert_eq!(f32::from_bits(step(&w,&x,1048576f32.to_bits())),1048577.0);
        // Negative small term against a large one: cut toward zero term by term, so the result rounds up in magnitude of the large term.
        w[0]=b(-1.0);x[0]=b(2f32.powi(-20));
        assert_eq!(f32::from_bits(step(&w,&x,64f32.to_bits())),64.0);
        // An infinite accumulator stays.
        assert_eq!(step(&w,&x,f32::INFINITY.to_bits()),f32::INFINITY.to_bits());
    }
    #[test]
    fn the_four_ranges_cover_every_step_once() {
        for steps in [4usize,5,7,8,32,96,256,768] {
            let mut next=0;for part in 0..4 {let (s,e)=range(steps,part);assert_eq!(s,next);next=e;}assert_eq!(next,steps);
        }
    }
}
