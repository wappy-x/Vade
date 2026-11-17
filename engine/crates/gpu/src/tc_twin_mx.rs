//! Kernel kit, dense family: CPU reference of the MXFP8 tensor-core word (`engine/docs/tc-dense-contract.md`, section
//! MXFP8). Integer arithmetic for the tensor-core step, plain IEEE FP32 operations for the rest. Independent of the
//! kernels (`kernels/src/mx_tc.rs`) and of the activation quantizer (`mxfp8_quantize.rs`).
//!
//! The word of one weight row and one token row, for `cols` a multiple of 32 (`groups = cols / 32`):
//! 1. Weight operand: the stored E4M3 codes, as they are. Nothing is converted, so nothing can be inexact.
//! 2. Activation operand, a group of 32 columns at a time: `m` = the largest magnitude of the group's 32 FP32 values;
//!    `xs` = the scale code of `m` by the activation quantizer's own rule (`activation_scale_code`); each value times
//!    `2^(127 - xs)` is rounded to its E4M3 code, nearest, ties to even, saturating (`e4m3_code`; the device uses
//!    `cvt.rn.satfinite.e4m3x2.f32`). For a group the MXFP8 activation quantizer wrote, every value is on the E4M3 grid
//!    of that scale, so this step is exact: the codes are a second spelling of the same 32 numbers.
//! 3. A sub-chain is up to `SUB` groups in ascending order from +0.0: `acc = step(acc, weight codes, activation codes,
//!    ws, xs)`. `step` is the GB10 instruction `mma.sync m16n8k32 kind::mxf8f6f4 block_scale scale_vec::1X` (E4M3 times
//!    E4M3, one UE8M0 scale code a weight row and one a token for the step's 32 columns) with the old sum as its
//!    accumulator. Measured rule (probe runs of 2026-11-16, every word of 13 classes): each of the 32 products is
//!    `significand * significand * 2^(exponent fields + ws + xs - 254)`; the products that are not zero and the old sum
//!    (when it is not zero) are lined up on the largest exponent; bits under `FRACTION` fraction bits of that exponent
//!    are cut toward zero, term by term; the sum is exact; it is cut toward zero to FP32. A result of zero is +0.0. A
//!    step with no live product leaves the old sum as it is. `ws` is the weight's UE8M0 scale code of the group, `xs`
//!    the activation's (step 2), or `NOT_FINITE` when a value of the group is not finite.
//! 4. The `groups` are cut into four ranges by the group count only: range `p` = groups `[groups*p/4, groups*(p+1)/4)`.
//!    A boundary is a group boundary, so it never cuts a scale group of the weight or of the activation. A range is cut
//!    into sub-chains of `SUB` groups counted from the range's first group (the last one may be shorter). The range's
//!    part is the sum of its sub-chains in ascending order from +0.0, each add one IEEE FP32 add, round to nearest.
//! 5. `word = fl(fl(part0 + part1) + fl(part2 + part3))`, IEEE adds, never fused. The stored word is this FP32 value
//!    (the callers round to BF16 where they did before).
//!
//! The scales act inside the step: no multiply and no add stands between two steps of a sub-chain.
//!
//! Why sub-chains: the step cuts toward zero, so every step of a chain can lose up to one unit of the sum's last
//! place, always in the same direction. One chain over a whole range (32 to 96 steps) fails the lab's word accuracy
//! count; with `SUB` = 8 the cut sums stay short and the adds between them round to nearest. MEASURED 2026-11-16 on
//! the 3,182,592 real MXFP8 words of three decode steps, words not correctly rounded, on the three captured chains:
//! today's scalar word 144 / 146 / 159; one chain a range 213 / 226 / 244; sub-chains of 8: 139 / 167 / 183.
//!
//! NaN: a scale code 255 (weight or activation), a NaN old sum, and an E4M3 NaN code among the 64 operands of a step
//! make the step `STEP_NAN`. An infinite old sum stays. Products that leave FP32 at the top give an infinity.
//! Results under 2^-126 follow the same rule: the last cut stores a subnormal (device run `r3-proof-1`: 0 of 1,048,576
//! probe words differ, one class of the probe aimed at that range; a floor of the cut at 2^-149 or 2^-151 does differ).
//!
//! A group with a non-finite activation value, and a weight scale code 255, make the word NaN. A weight code 0x7f or
//! 0xff (the E4M3 NaN) is outside the contract: loaders refuse it.

/// Fraction bits under the largest product exponent that a step keeps.
pub const FRACTION:i32=25;
/// A zero product brings no alignment exponent (the measured device rule).
pub const ZEROS_COUNT:bool=false;
/// The smallest exponent of the unit the terms of a step are cut at: none (the unit is always `FRACTION` bits under the
/// largest exponent, MEASURED device run `r3-proof-1`). The probe also counts the models -149 and -151: they differ.
pub const LOW:i32=i32::MIN;

/// (negative, significand with 3 fraction bits, exponent) of an E4M3 code: the magnitude is `significand * 2^(exponent - 3)`.
pub fn e4m3_parts(code:u8)->(bool,u64,i32) {
    let e=i32::from((code>>3)&15);let m=u64::from(code&7);
    (code&0x80!=0,if e==0 {m} else {8+m},if e==0 {-6} else {e-7})
}
/// The FP32 value of an E4M3 code (exact). The NaN codes give NaN.
pub fn e4m3_value(code:u8)->f32 {
    if code&0x7f==0x7f {return f32::NAN;}
    let (negative,significand,exponent)=e4m3_parts(code);
    let magnitude=significand as f32*f32::from_bits(((exponent-3+127) as u32)<<23);
    if negative {-magnitude} else {magnitude}
}
/// `2^(code - 127)` of a UE8M0 scale code. Code 0 is `2^-127` (a subnormal FP32 value), code 255 is NaN.
pub fn scale(code:u8)->f32 {f32::from_bits(if code==0 {0x0040_0000} else if code==255 {0x7fc0_0000} else {u32::from(code)<<23})}
/// The scale code of a group whose largest magnitude has the FP32 pattern `maximum` (sign bit clear, finite): the
/// smallest code whose top value, 448 times its scale, is not below the maximum. 0 to 247.
pub fn activation_scale_code(maximum:u32)->u8 {
    ((maximum>>23)+u32::from(maximum&0x007f_ffff>0x0060_0000)).saturating_sub(8) as u8
}
/// The E4M3 code of a finite FP32 value: the nearest grid value, ties to the even code, saturating at 448. The grid
/// has steps of 2^-9 below 2^-5 and 8 steps a binade above. The sign bit is kept, also on a zero.
pub fn e4m3_code(value:f32)->u8 {
    let bits=value.to_bits();let sign=((bits>>31) as u8)<<7;let magnitude=bits&0x7fff_ffff;
    if magnitude>0x7f80_0000 {return sign|0x7f;}
    if magnitude==0 {return sign;}
    if magnitude>=0x43e0_0000 {return sign|126;}
    let exponent=if magnitude>>23==0 {-127} else {(magnitude>>23) as i32-127};
    let field=(exponent+7).clamp(1,16);
    // Steps of 2^(field - 10): the scaled magnitude is exact in FP64.
    let scaled=f64::from(f32::from_bits(magnitude))*2f64.powi(10-field);
    let floor=scaled.floor();let rest=scaled-floor;
    let steps=floor as u64+u64::from(rest>0.5 || rest==0.5 && (floor as u64)&1==1);
    let code=(field as u64-1)*8+steps;
    sign|if code>126 {126} else {code as u8}
}
/// The activation operand of one group of 32 FP32 values: (scale code, the FP32 scale `2^(code - 127)` or NaN when a
/// value is not finite, the 32 E4M3 codes).
pub fn encode_group(input:&[f32])->(u8,f32,[u8;32]) {
    assert_eq!(input.len(),32);
    let maximum=input.iter().map(|x|x.to_bits()&0x7fff_ffff).max().unwrap();
    if maximum>=0x7f80_0000 {return (0,f32::NAN,[0;32]);}
    let code=activation_scale_code(maximum);
    // 2^(127 - code): the exponent field 254 - code is 7 to 254, always a normal FP32 value.
    let factor=f32::from_bits((254-u32::from(code))<<23);
    let mut codes=[0u8;32];
    for (slot,&x) in codes.iter_mut().zip(input) {*slot=e4m3_code(x*factor);}
    (code,scale(code),codes)
}
fn cut(negative:bool,magnitude:u64,exponent:i32,nearest:bool)->u32 {
    let sign=u32::from(negative)<<31;if magnitude==0 {return 0;}
    let e=63-magnitude.leading_zeros() as i32+exponent;
    let low=if e< -126 {-149} else {e-23};let shift=low-exponent;
    let mut kept=if shift<=0 {magnitude<<(-shift)} else if shift>=64 {0} else {magnitude>>shift};
    if nearest && shift>0 && shift<64 {
        let rest=magnitude&((1u64<<shift)-1);let half=1u64<<(shift-1);
        if rest>half || rest==half && kept&1==1 {kept+=1;}
    }
    // A carry of the rounding into the next binade keeps the same pattern arithmetic: the fraction overflows into the exponent.
    if e< -126 {return if kept==0 {0} else {sign|kept as u32};}
    if e>127 {return sign|0x7f80_0000;}
    sign|((((e+127) as u32)<<23)+(kept as u32-0x80_0000))
}
/// The NaN a step writes.
pub const STEP_NAN:u32=0x7fff_ffff;
/// The activation scale code of a group with a value that is not finite: the step then writes NaN.
pub const NOT_FINITE:u8=255;
/// One step with the model as arguments (the device probe of `kernel-bench --mx-tc` counts, for a list of models, the
/// words that differ from the device). `acc`: the old sum; `ws`, `xs`: the two scale codes; `model` = (kept fraction
/// bits, `zeros_count` as `ZEROS_COUNT`, `nearest`: true = the last cut to FP32 rounds to nearest even in place of
/// toward zero, `low`: the smallest exponent of the unit the terms are cut at, `LOW` in the twin).
pub fn step_model(acc:u32,weight:&[u8],input:&[u8],ws:u8,xs:u8,model:(i32,bool,bool,i32))->u32 {
    let (fraction,zeros_count,nearest,low)=model;
    assert!(weight.len()==32 && input.len()==32);
    if acc&0x7fff_ffff>0x7f80_0000 || ws==255 || xs==255 || weight.iter().chain(input).any(|code|code&0x7f==0x7f) {return STEP_NAN;}
    if acc&0x7fff_ffff==0x7f80_0000 {return acc;}
    let scales=i32::from(ws)+i32::from(xs)-254;
    let mut terms=[(false,0u64,0i32);32];let mut top=i32::MIN;
    for k in 0..32 {
        let (na,sa,ea)=e4m3_parts(weight[k]);let (nb,sb,eb)=e4m3_parts(input[k]);
        terms[k]=(na!=nb,sa*sb,ea+eb+scales);
        if (sa*sb!=0 || zeros_count) && ea+eb+scales>top {top=ea+eb+scales;}
    }
    // The old sum is `old * 2^(old_exponent - 23)`; a subnormal keeps the exponent -126.
    let (field,fraction_bits)=(((acc>>23)&255) as i32,u64::from(acc&0x007f_ffff));
    let (old,old_exponent)=if field==0 {(fraction_bits,-126)} else {(fraction_bits|1<<23,field-127)};
    if old!=0 && old_exponent>top {top=old_exponent;}
    if top==i32::MIN {return 0;}
    // A product is `magnitude * 2^(exponent - 6)`. Everything in units of 2^(top - fraction), cut toward zero.
    let unit=(top-fraction).max(low);let mut sum=0i64;
    let place=|magnitude:u64,shift:i32|if magnitude==0 || shift<= -64 {0} else if shift>=0 {magnitude<<shift} else {magnitude>>(-shift)};
    for (negative,magnitude,exponent) in terms {
        let value=place(magnitude,exponent-6-unit) as i64;
        sum+=if negative {-value} else {value};
    }
    let value=place(old,old_exponent-23-unit) as i64;
    sum+=if acc>>31!=0 {-value} else {value};
    cut(sum<0,sum.unsigned_abs(),unit,nearest)
}
/// One step: the old sum, 32 E4M3 weight codes, 32 E4M3 activation codes, the weight's and the activation's scale code.
/// Returns the FP32 pattern of the new sum.
pub fn step(acc:u32,weight:&[u8],input:&[u8],ws:u8,xs:u8)->u32 {step_model(acc,weight,input,ws,xs,(FRACTION,ZEROS_COUNT,false,LOW))}
/// Groups of range `part` of `groups` 32-column groups.
pub fn range(groups:usize,part:usize)->(usize,usize) {(groups*part/4,groups*(part+1)/4)}
/// Groups of a sub-chain (256 columns): the tensor-core step carries its sum this far, then one FP32 add takes it.
pub const SUB:usize=8;
/// The activation operand of one group of 32 FP32 values as the step takes it: (scale code, or `NOT_FINITE` when a value
/// is not finite; the 32 E4M3 codes). The codes of a group that is not finite are not part of the contract.
pub fn operand_group(input:&[f32])->(u8,[u8;32]) {let (code,factor,codes)=encode_group(input);(if factor.is_nan() {NOT_FINITE} else {code},codes)}
/// The activation operand of one token row: (scale code, codes) of every group of 32 columns.
pub fn encode_row(input:&[f32])->Vec<(u8,[u8;32])> {
    assert!(input.len().is_multiple_of(32));
    input.as_chunks::<32>().0.iter().map(|group|operand_group(group)).collect()
}
/// The packed token row of `mx_tc_pack`: the row's codes, then one scale code a group.
pub fn pack_row(input:&[f32])->Vec<u8> {
    let operand=encode_row(input);
    operand.iter().flat_map(|(_,codes)|codes.iter().copied()).chain(operand.iter().map(|(code,_)|*code)).collect()
}
/// The chain of the groups `[start, end)` of one word, from +0.0: `weight` is one row of E4M3 codes, `scales` its UE8M0
/// codes (one a group of 32 columns), `encoded` one token row from `encode_row`.
pub fn chain(weight:&[u8],scales:&[u8],encoded:&[(u8,[u8;32])],start:usize,end:usize)->u32 {
    assert!(weight.len()==encoded.len()*32 && scales.len()==encoded.len() && start<=end && end<=encoded.len());
    let mut acc=0u32;
    for g in start..end {acc=step(acc,&weight[g*32..g*32+32],&encoded[g].1,scales[g],encoded[g].0);}
    acc
}
/// One range of one word, groups `[start, end)`: the sum, from +0.0 and in ascending order, of its sub-chains of `sub`
/// groups counted from `start` (FP32 adds, round to nearest). `sub` = `SUB` is the contract; another value is a control.
pub fn part_encoded(weight:&[u8],scales:&[u8],encoded:&[(u8,[u8;32])],start:usize,end:usize,sub:usize)->u32 {
    let (mut total,mut at)=(0.0f32,start);
    while at<end {let stop=(at+sub).min(end);total+=f32::from_bits(chain(weight,scales,encoded,at,stop));at=stop;}
    total.to_bits()
}
/// The four range accumulators of one word, before the join.
pub fn parts_encoded(weight:&[u8],scales:&[u8],encoded:&[(u8,[u8;32])])->[u32;4] {
    let mut parts=[0u32;4];
    for (part,slot) in parts.iter_mut().enumerate() {let (start,end)=range(encoded.len(),part);*slot=part_encoded(weight,scales,encoded,start,end,SUB);}
    parts
}
/// The FP32 pattern of one word of an encoded token row.
pub fn word_encoded(weight:&[u8],scales:&[u8],encoded:&[(u8,[u8;32])])->u32 {
    let parts=parts_encoded(weight,scales,encoded).map(f32::from_bits);
    ((parts[0]+parts[1])+(parts[2]+parts[3])).to_bits()
}
/// The four range accumulators of one word: `input` is one token row of FP32 values.
pub fn parts(weight:&[u8],scales:&[u8],input:&[f32])->[u32;4] {parts_encoded(weight,scales,&encode_row(input))}
/// The FP32 pattern of one word.
pub fn word(weight:&[u8],scales:&[u8],input:&[f32])->u32 {word_encoded(weight,scales,&encode_row(input))}
/// Small deterministic generator for fixtures (the bench and the host tests draw the same inputs).
#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self)->u64 {self.0^=self.0<<13;self.0^=self.0>>7;self.0^=self.0<<17;self.0}
    pub fn below(&mut self,n:u64)->u64 {(self.next_u64()>>11)%n}
}
/// Weight codes of class `class % 6` (never an E4M3 NaN code): 0 near one; 1 any exponent; 2 subnormals and zeros among
/// small values; 3 signed zeros among values; 4 plus and minus one in turn; 5 mantissas of ones near one.
pub fn weight_code(r:&mut Rng,index:usize,class:usize)->u8 {
    let h=r.next_u64();let sign=((h>>40) as u8)&0x80;
    match class%6 {
        0=>sign|(0x30+(h%24) as u8),
        1=>{let c=(h%127) as u8;sign|c},
        2=>sign|match h&3 {0=>0,1=>(h>>8) as u8%8,_=>8+(h>>8) as u8%24},
        3=>if h&7==0 {sign} else {sign|(0x28+(h>>8) as u8%40)},
        4=>if index.is_multiple_of(2) {0x38} else {0xb8},
        _=>sign|((5+(h>>8) as u8%4)<<3)|7,
    }
}
/// One activation group of class `class % 7` as FP32 values. Classes 0 to 5 are MXFP8 groups (codes of `weight_code`'s
/// classes times one power-of-two scale: what the activation quantizer writes); class 6 is general FP32 values, which
/// the quantizer never writes (the encode then rounds: outside the served contract, inside the word function).
pub fn activation_group(r:&mut Rng,index:usize,class:usize)->[f32;32] {
    let mut group=[0.0f32;32];
    if class%7==6 {
        let base=100+r.below(50) as u32;
        for slot in group.iter_mut() {let h=r.next_u64();*slot=f32::from_bits(((h as u32)&0x807f_ffff)|((base+(h>>40) as u32%9)<<23));}
        return group;
    }
    // Classes 0, 3, 4 and 5 keep the group scales of a row within a factor of 8, so that every group counts in the word.
    let code=match class%7 {1=>1+r.below(246) as u8,2=>[0u8,1,2,9,10,11][r.below(6) as usize],_=>125+r.below(4) as u8};
    for (k,slot) in group.iter_mut().enumerate() {*slot=e4m3_value(weight_code(r,index+k,class))*scale(code);}
    group
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_terms_have_no_exponent_and_cut_zero_is_positive() {
        let mut w=[0u8;32];let mut x=[0x7eu8;32];
        w[0]=1;x[0]=1;
        const _:()=assert!(!ZEROS_COUNT);
        assert_eq!(step(0,&w,&x,127,127),2f32.powi(-18).to_bits());
        assert_eq!(step(0,&[0x80;32],&x,127,127),0);
        // E4M3 products cannot underflow FP32, but the integer cut must still implement the rule.
        assert_eq!(cut(true,1,-150,false),0);
        assert_eq!(cut(true,1,-149,false),0x8000_0001);
    }
    #[test]
    fn codes_and_values_follow_the_format() {
        for (code,value) in [(0u8,0.0f32),(1,1.0/512.0),(7,7.0/512.0),(8,1.0/64.0),(0x38,1.0),(0x7e,448.0),(0xb8,-1.0),(0xfe,-448.0)] {
            assert_eq!(e4m3_value(code).to_bits(),value.to_bits(),"code {code}");
        }
        assert_eq!(e4m3_value(0x80).to_bits(),(-0.0f32).to_bits());assert!(e4m3_value(0x7f).is_nan() && e4m3_value(0xff).is_nan());
        // Every code is the code of its value.
        for code in 0..=255u8 {if code&0x7f!=0x7f {assert_eq!(e4m3_code(e4m3_value(code)),code,"code {code}");}}
        assert_eq!(scale(127),1.0);assert_eq!(scale(128),2.0);assert_eq!(scale(0).to_bits(),0x0040_0000);assert!(scale(255).is_nan());
    }
    #[test]
    fn the_code_of_a_value_is_the_nearest_grid_value_with_ties_to_even() {
        // Between 1.0 (0x38) and 1.125 (0x39): the middle goes to the even code.
        assert_eq!(e4m3_code(1.0625),0x38);assert_eq!(e4m3_code(1.0625+1e-6),0x39);assert_eq!(e4m3_code(1.1875),0x3a);
        // The subnormal steps: 2^-10 is the middle of 0 and 2^-9.
        assert_eq!(e4m3_code(1.0/1024.0),0);assert_eq!(e4m3_code(1.5/1024.0),1);assert_eq!(e4m3_code(3.0/1024.0),2);
        // A carry into the next binade, and saturation.
        assert_eq!(e4m3_code(1.97),0x40);assert_eq!(e4m3_code(447.0),0x7e);assert_eq!(e4m3_code(1e9),0x7e);assert_eq!(e4m3_code(-1e9),0xfe);
        assert_eq!(e4m3_code(-0.0),0x80);assert_eq!(e4m3_code(f32::from_bits(1)),0);
        // Against the nearest search over the 127 values, on a spread of magnitudes.
        let mut r=Rng(77);
        for _ in 0..200_000 {
            let x=f32::from_bits((r.next_u64() as u32&0x007f_ffff)|((110+r.below(30) as u32)<<23));
            let mut best=0u8;
            for code in 1..=126u8 {
                let (old,new)=((f64::from(e4m3_value(best))-f64::from(x)).abs(),(f64::from(e4m3_value(code))-f64::from(x)).abs());
                if new<old || new==old && code%2==0 {best=code;}
            }
            assert_eq!(e4m3_code(x),best,"value {x}");
        }
    }
    #[test]
    fn the_scale_code_is_the_smallest_whose_top_value_holds_the_maximum() {
        for code in 1..=246u8 {
            let top=448.0*scale(code);
            assert_eq!(activation_scale_code(top.to_bits()),code);
            assert_eq!(activation_scale_code(top.to_bits()+1),code+1);
            assert_eq!(activation_scale_code((448.0*scale(code-1)).to_bits()+1),code);
        }
        assert_eq!(activation_scale_code(0),0);assert_eq!(activation_scale_code(1),0);assert_eq!(activation_scale_code(f32::MAX.to_bits()),247);
        let (code,factor,codes)=encode_group(&[0.0;32]);assert_eq!((code,factor.to_bits(),codes),(0,0x0040_0000,[0;32]));
    }
    #[test]
    fn small_integers_give_the_exact_dot_product() {
        // Codes of small integers and unit scales: nothing is cut in a step, and every partial sum is an integer below 2^24.
        let integer=|v:i32|e4m3_code(v as f32);
        let weight:Vec<u8>=(0..256).map(|i|integer((i*7+3)%15-7)).collect();
        let input:Vec<f32>=(0..256).map(|i|((i*5+1)%13) as f32-6.0).collect();
        let want:f32=weight.iter().zip(&input).map(|(&w,&x)|e4m3_value(w)*x).sum();
        assert_eq!(f32::from_bits(word(&weight,&[127;8],&input)),want);
        // The weight scale and the activation scale are exact powers of two on the group's term.
        let scaled:Vec<f32>=input.iter().map(|x|x*0.25).collect();
        assert_eq!(f32::from_bits(word(&weight,&[130;8],&scaled)),want*8.0*0.25);
    }
    #[test]
    fn a_step_cuts_toward_zero_under_the_kept_fraction_bits_of_the_largest_product() {
        let mut w=[0u8;32];let mut x=[0u8;32];
        // 448 * 448 (exponent 16) and 2^-9 * 2^-9 * 49 (subnormal codes 7 and 7, exponent -12): the small product is
        // 49 * 2^-18, the unit is 2^(16-25) = 2^-9, so the small product is cut to nothing.
        w[0]=0x7e;x[0]=0x7e;w[1]=7;x[1]=7;
        assert_eq!(f32::from_bits(step(0,&w,&x,127,127)),448.0*448.0);
        // With the large product away, the small one is exact.
        w[0]=0;assert_eq!(f32::from_bits(step_model(0,&w,&x,127,127,(FRACTION,false,false,LOW))),49.0/262144.0);
        // The sum of 32 products is exact before the last cut: 32 times 225 * 2^-6 (1.875 * 1.875).
        assert_eq!(f32::from_bits(step(0,&[0x3f;32],&[0x3f;32],127,127)),32.0*1.875*1.875);
        // A sum that needs more than 24 bits is cut toward zero, not rounded: 448*448 + 3 products of 2^-6 * 1.875.
        let mut w=[0u8;32];let mut x=[0u8;32];w[0]=0x7e;x[0]=0x7e;for k in 1..4 {w[k]=0x0f;x[k]=0x38;}
        let exact=448.0f64*448.0+3.0*1.875/64.0;let cut=f32::from_bits(step(0,&w,&x,127,127));
        assert!(f64::from(cut)<=exact && exact-f64::from(cut)<f64::from(cut)*2f64.powi(-23));
        assert_eq!(step(0,&[0;32],&[0x38;32],127,127),0);assert_eq!(step(0,&[0x80;32],&[0x38;32],127,127),0);
    }
    #[test]
    fn the_old_sum_and_the_scales_enter_the_step() {
        let one=|w:u8,x:u8|{let (mut a,mut b)=([0u8;32],[0u8;32]);a[5]=w;b[5]=x;(a,b)};
        let bits=|v:f32|v.to_bits();
        // The scale codes are exponents on every product: 1 * 1 * 2^(130 - 127) * 2^(125 - 127) = 2.
        let (w,x)=one(0x38,0x38);
        assert_eq!(step(0,&w,&x,130,125),bits(2.0));
        // The old sum is one more term of the same sum: 3 + 1 = 4, and -1 + 1 = +0.0.
        assert_eq!(step(bits(3.0),&w,&x,127,127),bits(4.0));
        assert_eq!(step(bits(-1.0),&w,&x,127,127),0);
        // Lined up on the old sum: a product under its 25 kept bits is cut to nothing (2^30 keeps down to 2^5).
        assert_eq!(step(bits(1073741824.0),&w,&x,127,127),bits(1073741824.0));
        // 224 is inside the 25 kept bits; the last cut to FP32 (steps of 128 at 2^30) is toward zero: 2^30 + 128.
        let (w,x)=one(0x7e,0x30);
        assert_eq!(step(bits(1073741824.0),&w,&x,127,127),bits(1073741824.0+128.0));
        // The last cut is toward zero, also for a sum that shrinks: 2^24 - 0.5 is 16777215.5, stored as 16777215.
        let (w,x)=one(0xb8,0x30);
        assert_eq!(step(bits(16777216.0),&w,&x,127,127),bits(16777215.0));
        // A step with no live product leaves the old sum as it is; a zero of either sign becomes +0.0.
        assert_eq!(step(bits(-7.25),&[0x80;32],&[0x38;32],3,250),bits(-7.25));
        assert_eq!(step(0x8000_0000,&[0;32],&[0x38;32],127,127),0);
        // NaN: a scale code 255, a NaN old sum, an E4M3 NaN code. An infinite old sum stays.
        let (w,x)=one(0x38,0x38);
        assert_eq!(step(0,&w,&x,255,127),STEP_NAN);assert_eq!(step(0,&w,&x,127,NOT_FINITE),STEP_NAN);
        assert_eq!(step(0,&[0;32],&[0;32],255,127),STEP_NAN);
        assert_eq!(step(0x7fc0_0000,&w,&x,127,127),STEP_NAN);
        assert_eq!(step(0xff80_0000,&w,&x,127,127),0xff80_0000);
        let (w,x)=one(0x7f,0x38);assert_eq!(step(0,&w,&x,127,127),STEP_NAN);
        // Products that leave FP32 at the top give an infinity.
        let (w,x)=one(0x7e,0xfe);assert_eq!(step(0,&w,&x,254,254),0xff80_0000);
    }
    #[test]
    fn the_packed_row_is_the_codes_then_the_scale_codes() {
        let mut r=Rng(5);
        let mut input:Vec<f32>=(0..4).flat_map(|g|activation_group(&mut r,g*32,g)).collect();
        input[70]=f32::NAN;
        let packed=pack_row(&input);assert_eq!(packed.len(),4*33);
        for g in 0..4 {
            let (code,codes)=operand_group(&input[g*32..g*32+32]);
            assert_eq!(packed[128+g],code);
            if g!=2 {assert_eq!(&packed[g*32..g*32+32],&codes[..]);}
        }
        assert_eq!(packed[128+2],NOT_FINITE);
    }
    #[test]
    fn the_four_ranges_cover_every_group_once_and_cut_no_scale_group() {
        for groups in [1usize,2,3,4,5,7,8,128,256,384] {
            let mut next=0;for part in 0..4 {let (s,e)=range(groups,part);assert_eq!(s,next);next=e;}assert_eq!(next,groups);
        }
        // The model's shapes: the ranges are the column quarters, as for the other dense formats.
        for cols in [4096usize,8192,12288] {for part in 0..4 {assert_eq!(range(cols/32,part),(cols/128*part,cols/128*(part+1)));}}
    }
    #[test]
    fn the_join_is_two_pair_sums_then_one_sum_and_a_range_is_a_sum_of_sub_chains() {
        // Four groups, one a range: 2^24, 1, 1, -2^24. The pair sums are fl(2^24 + 1) = 2^24 and fl(1 - 2^24), which is
        // exact, so the word is 1. One chain over all four would lose both ones and give 0.
        let big=e4m3_code(256.0);
        let mut weight=vec![0u8;128];let mut input=vec![0.0f32;128];
        for (g,(w,x)) in [(big,65536.0f32),(0x38,1.0),(0x38,1.0),(big,-65536.0)].into_iter().enumerate() {weight[g*32]=w;input[g*32]=x;}
        let four=parts(&weight,&[127;4],&input).map(f32::from_bits);
        assert_eq!(four,[16777216.0,1.0,1.0,-16777216.0]);
        assert_eq!(f32::from_bits(word(&weight,&[127;4],&input)),1.0);
        assert_eq!(((four[0]+four[1])+four[2])+four[3],0.0);
        // Eight groups: each range chains its two groups in ascending order.
        let mut weight=vec![0u8;256];let mut input=vec![0.0f32;256];
        for (g,(w,x)) in [(big,65536.0f32),(0x38,1.0)].into_iter().enumerate() {weight[g*32]=w;input[g*32]=x;}
        assert_eq!(f32::from_bits(parts(&weight,&[127;8],&input)[0]),16777216.0);
        // 64 groups: a range is 16 groups = two sub-chains of 8. Groups 0, 7, 8: 2^24, 1.5, 1.5. The first sub-chain cuts
        // 2^24 + 1.5 toward zero to 2^24; the second is 1.5; their FP32 add rounds 2^24 + 1.5 to nearest: 2^24 + 2.
        // One chain over the 16 groups would cut twice and give 2^24 (the control below).
        let mut weight=vec![0u8;2048];let mut input=vec![0.0f32;2048];
        for (g,w,x) in [(0usize,big,65536.0f32),(7,0x38,1.5),(8,0x38,1.5)] {weight[g*32]=w;input[g*32]=x;}
        let encoded=encode_row(&input);
        assert_eq!(f32::from_bits(parts_encoded(&weight,&[127;64],&encoded)[0]),16777218.0);
        assert_eq!(part_encoded(&weight,&[127;64],&encoded,0,16,SUB),parts_encoded(&weight,&[127;64],&encoded)[0]);
        assert_eq!(f32::from_bits(chain(&weight,&[127;64],&encoded,0,16)),16777216.0);
        assert_eq!(f32::from_bits(part_encoded(&weight,&[127;64],&encoded,0,16,16)),16777216.0);
        // The sub-chains are counted from the range's first group: at 48 groups range 1 is groups 12..24, cut at 20.
        // Groups 12, 17, 20: 2^24, 1.5, 1.5 give fl(2^24 + 1.5) = 2^24 + 2. Counted from group 0 (cuts at 16 and 24)
        // it would be fl(2^24 + 3.0) = 2^24 + 4.
        let mut weight=vec![0u8;48*32];let mut input=vec![0.0f32;48*32];
        for (g,w,x) in [(12usize,big,65536.0f32),(17,0x38,1.5),(20,0x38,1.5)] {weight[g*32]=w;input[g*32]=x;}
        assert_eq!(f32::from_bits(parts(&weight,&[127;48],&input)[1]),16777218.0);
        // A short range (5 groups a range at 20 groups) is one short sub-chain; a range of 12 is 8 and then 4.
        let mut weight=vec![0u8;48*32];let mut input=vec![0.0f32;48*32];
        for (g,w,x) in [(0usize,big,65536.0f32),(8,0x38,1.5),(11,0x38,1.5)] {weight[g*32]=w;input[g*32]=x;}
        let encoded=encode_row(&input);
        // 2^24, then the sub-chain of groups 8..12 = 3.0 exactly: 2^24 + 3 rounds to nearest even = 2^24 + 4.
        assert_eq!(f32::from_bits(parts_encoded(&weight,&[127;48],&encoded)[0]),16777220.0);
    }
    #[test]
    fn a_word_reads_only_its_own_rows_and_non_finite_inputs_give_nan() {
        let mut r=Rng(9);
        let weight:Vec<u8>=(0..512).map(|i|weight_code(&mut r,i,1)).collect();
        let scales:Vec<u8>=(0..16).map(|_|100+r.below(40) as u8).collect();
        let input:Vec<f32>=(0..16).flat_map(|g|activation_group(&mut r,g*32,g)).collect();
        let first=word(&weight,&scales,&input);
        assert_eq!(word(&weight,&scales,&input),first);
        let mut bad=input.clone();bad[40]=f32::INFINITY;assert!(f32::from_bits(word(&weight,&scales,&bad)).is_nan());
        let mut bad=scales.clone();bad[3]=255;assert!(f32::from_bits(word(&weight,&bad,&input)).is_nan());
    }
}
