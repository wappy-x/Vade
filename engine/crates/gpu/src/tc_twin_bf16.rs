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
//!    toward zero to FP32. Two facts measured on a box (2026-11-15, `kernel-bench --plain-tc probe`, 589,824
//!    short-chain words of tiny, subnormal and zero operands; this model is the only one of six with no wrong word):
//!    a term that is ZERO (a product with a zero operand, a zero carried value) brings no exponent to the largest
//!    exponent; a subnormal operand brings its exponent field's 2^-126 (it is not normalized). And a result that is
//!    zero is +0.0, also where a negative sum is cut to zero.
//! 4. `acc = fl(fl(part0 + part1) + fl(part2 + part3))`. `fl` is one IEEE FP32 add.
//! 5. `word = bf16(acc)`.
#[derive(Clone,Copy)]
struct Term {negative:bool,magnitude:u64,exponent:i32,top:Option<i32>}
fn operand(v:u16)->(bool,u64,i32) {let e=i32::from((v>>7)&255);let f=u64::from(v&127);(v&0x8000!=0,if e==0 {f} else {128+f},if e==0 {-126} else {e-127})}
fn carried(bits:u32)->Term {
    let e=((bits>>23)&255) as i32;let f=u64::from(bits&0x7f_ffff);
    Term {negative:bits>>31!=0,magnitude:if e==0 {f} else {(1<<23)+f},exponent:if e==0 {-149} else {e-150},top:if e!=0 {Some(e-127)} else if f!=0 {Some(-126)} else {None}}
}
fn truncate(negative:bool,magnitude:u64,exponent:i32)->u32 {
    let sign=u32::from(negative)<<31;if magnitude==0 {return 0;}
    let e=63-magnitude.leading_zeros() as i32+exponent;
    let low=if e< -126 {-149} else {e-23};let shift=low-exponent;
    let kept=if shift<=0 {magnitude<<(-shift)} else if shift>=64 {0} else {magnitude>>shift};
    if e< -126 {return if kept==0 {0} else {sign|kept as u32};}
    if e>127 {return sign|0x7f80_0000;}
    sign|((e+127) as u32)<<23|(kept as u32&0x7f_ffff)
}
/// One 16-column step: 16 BF16 weights, 16 BF16 activations, the carried accumulator. Returns the new accumulator.
pub fn step(weight:&[u16],input:&[u16],acc:u32)->u32 {
    assert!(weight.len()==16 && input.len()==16);
    if acc&0x7f80_0000==0x7f80_0000 {return acc;}
    let mut terms=[carried(acc);17];
    for k in 0..16 {let (na,sa,ea)=operand(weight[k]);let (nb,sb,eb)=operand(input[k]);
        terms[k]=Term {negative:na!=nb,magnitude:sa*sb,exponent:ea+eb-14,top:if sa*sb==0 {None} else {Some(ea+eb)}};
    }
    // Every term is zero: the result is +0.0.
    let Some(top)=terms.iter().filter_map(|t|t.top).max() else {return 0;};
    let unit=top-25;let mut sum=0i64;
    for t in terms {if t.magnitude==0 {continue;}let shift=t.exponent-unit;
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
/// The BF16 operand a tensor-core kernel takes from an FP32 word: its high 16 bits. No rounding happens at the read,
/// in any path; a BF16 value (low 16 bits zero) is its own operand.
pub fn high(x:f32)->u16 {(x.to_bits()>>16) as u16}
fn special(v:u16)->bool {v&0x7f80==0x7f80}
/// A step with an infinity or a NaN among its 33 inputs, as IEEE sums the 16 products and the carried accumulator:
/// `None` when every input is finite. NaN: a NaN input, zero times infinity, or infinities of both signs. The NaN
/// pattern is not part of the contract (NaN places must agree, payloads need not).
fn step_special(weight:&[u16],input:&[u16],acc:u32)->Option<u32> {
    let carried=acc&0x7f80_0000==0x7f80_0000;
    if !carried && !weight.iter().chain(input).any(|&v|special(v)) {return None;}
    let (mut nan,mut plus,mut minus)=(carried && acc&0x7f_ffff!=0,false,false);
    if carried && acc&0x7f_ffff==0 {if acc>>31==0 {plus=true;} else {minus=true;}}
    for (&w,&x) in weight.iter().zip(input) {
        if !special(w) && !special(x) {continue;}
        let (nan_w,nan_x)=(special(w) && w&0x7f!=0,special(x) && x&0x7f!=0);
        if nan_w || nan_x || w&0x7fff==0 || x&0x7fff==0 {nan=true;} else if (w^x)&0x8000==0 {plus=true;} else {minus=true;}
    }
    Some(if nan || plus && minus {0x7fff_ffff} else if plus {0x7f80_0000} else {0xff80_0000})
}
/// One 16-term step for any BF16 inputs: `step` on finite inputs, the IEEE result on an infinity or a NaN.
pub fn step_any(weight:&[u16],input:&[u16],acc:u32)->u32 {step_special(weight,input,acc).unwrap_or_else(||step(weight,input,acc))}
/// The joined accumulator of one sum of `terms` products `a(j) * b(j)`, for ANY term count. The terms fill 16-term steps
/// in ascending order; the last step is filled with +0.0 operands. The `terms.div_ceil(16)` steps are cut into four
/// ranges by the step count only (`range`); a range is one ascending chain from +0.0 (an empty range is +0.0); the join
/// is `fl(fl(p0 + p1) + fl(p2 + p3))`.
pub fn fold(terms:usize,a:impl Fn(usize)->u16,b:impl Fn(usize)->u16)->u32 {
    let steps=terms.div_ceil(16);let mut parts=[0.0f32;4];
    for (part,slot) in parts.iter_mut().enumerate() {
        let (start,end)=range(steps,part);let mut acc=0u32;
        for s in start..end {
            let (mut left,mut right)=([0u16;16],[0u16;16]);
            for j in 0..16 {let t=s*16+j;if t<terms {left[j]=a(t);right[j]=b(t);}}
            acc=step_any(&left,&right,acc);
        }
        *slot=f32::from_bits(acc);
    }
    ((parts[0]+parts[1])+(parts[2]+parts[3])).to_bits()
}
#[cfg(test)]
mod tests {
    use super::*;
    fn b(x:f32)->u16 {assert_eq!(f32::from_bits(u32::from(bf16(x))<<16),x,"not a BF16 value");bf16(x)}
    #[test]
    fn a_zero_term_brings_no_exponent_and_a_zero_result_is_positive() {
        // One live product, 255 * 2^-133 times 255 * 2^-4 = 65,025 * 2^-137, beside fifteen zero products whose
        // exponent fields say 2^(-126 + 20). Were a zero product part of the alignment, the cut at 2^(-106 - 25) would
        // leave 65,024 * 2^-137. The device keeps the whole product (measured), and so does this model.
        let mut w=[0u16;16];let mut x=[0u16;16];
        w[0]=0x00ff;x[0]=0x417f;for k in 1..16 {w[k]=0;x[k]=b(1048576.0);}
        let exact=(65025.0f64*2f64.powi(-137)) as f32;
        assert_eq!(f32::from_bits(step(&w,&x,0)),exact);
        // A zero carried value brings no exponent either, whatever its sign.
        x[1..].fill(0);
        assert_eq!(f32::from_bits(step(&w,&x,0)),exact);assert_eq!(f32::from_bits(step(&w,&x,0x8000_0000)),exact);
        // A subnormal operand is not normalized: it brings the 2^-126 of its exponent field. Here 2^-133 times 2^10
        // sets the alignment at 2^(-126 + 10), so the cut at 2^(-116 - 25) removes the second product
        // (65,025 * 2^-160) whole. A normalized operand would align at 2^-123 and keep 15 * 2^-148 of it.
        let mut w=[0u16;16];let mut x=[0u16;16];w[0]=0x0001;x[0]=b(1024.0);w[1]=0x00ff;x[1]=0x35ff;
        assert_eq!(f32::from_bits(step(&w,&x,0)),2f32.powi(-123));
        // A negative sum under the smallest FP32 value is cut to +0.0, not -0.0; a step of zero products is +0.0.
        let mut w=[0u16;16];let mut x=[0u16;16];w[0]=0x8001;x[0]=0x0001;
        assert_eq!(step(&w,&x,0),0);
        w[0]=0x8000;x[0]=b(3.0);assert_eq!(step(&w,&x,0),0);
    }
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
    fn fold_equals_canonical_on_whole_steps_and_pads_a_short_last_step_with_zero() {
        let weight:Vec<u16>=(0..256).map(|i|b(((i*7+3)%31) as f32*0.125-1.75)).collect();
        let input:Vec<f32>=(0..256).map(|i|f32::from_bits(u32::from(b(((i*5+1)%29) as f32*0.0625-0.875))<<16)).collect();
        for cols in [16usize,32,64,128,256] {
            assert_eq!(fold(cols,|j|weight[j],|j|high(input[j])),canonical(&weight[..cols],&input[..cols]),"{cols}");
        }
        // A short last step: the same word as the row padded with +0.0 operands to whole steps.
        for terms in [0usize,1,2,15,17,31,33,63,65,100,255] {
            let whole=terms.div_ceil(16)*16;
            let padded=fold(whole,|j|if j<terms {weight[j]} else {0},|j|if j<terms {high(input[j])} else {0});
            assert_eq!(fold(terms,|j|weight[j],|j|high(input[j])),padded,"{terms}");
        }
        assert_eq!(fold(0,|_|0,|_|0),0);
        // The four ranges follow the step count: 5 steps cut as 1, 1, 1, 2.
        let ones=|n:usize|fold(n,|_|b(1.0),|_|b(1.0));assert_eq!(f32::from_bits(ones(80)),80.0);assert_eq!(f32::from_bits(ones(2051)),2051.0);
    }
    #[test]
    fn an_operand_is_the_high_half_of_its_word() {
        assert_eq!(high(1.0),0x3f80);assert_eq!(high(f32::from_bits(0x3f80_ffff)),0x3f80);assert_eq!(high(-0.0),0x8000);assert_eq!(high(f32::NAN),0x7fc0);
    }
    #[test]
    fn a_step_with_an_infinity_or_a_nan_follows_ieee() {
        let (mut w,mut x)=([0u16;16],[0u16;16]);let (inf,nan,one)=(0x7f80u16,0x7fc0u16,0x3f80u16);
        let is_nan=|v:u32|v&0x7f80_0000==0x7f80_0000 && v&0x7f_ffff!=0;
        w[3]=inf;x[3]=one;assert_eq!(step_any(&w,&x,0),0x7f80_0000);
        x[3]=one|0x8000;assert_eq!(step_any(&w,&x,0),0xff80_0000);
        assert!(is_nan(step_any(&w,&x,0x7f80_0000)));
        x[3]=0;assert!(is_nan(step_any(&w,&x,0)));
        w[3]=nan;x[3]=one;assert!(is_nan(step_any(&w,&x,0)));
        w[3]=one;assert_eq!(step_any(&w,&x,0x7f80_0000),0x7f80_0000);assert!(is_nan(step_any(&w,&x,0x7fc0_0000)));
        assert_eq!(step_any(&w,&x,1.0f32.to_bits()),step(&w,&x,1.0f32.to_bits()));
    }
    #[test]
    fn the_four_ranges_cover_every_step_once() {
        for steps in [4usize,5,7,8,32,96,256,768] {
            let mut next=0;for part in 0..4 {let (s,e)=range(steps,part);assert_eq!(s,next);next=e;}assert_eq!(next,steps);
        }
    }
}
