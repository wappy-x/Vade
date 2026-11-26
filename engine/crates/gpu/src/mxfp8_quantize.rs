//! Dynamic MXFP8 lanes with the host recipe's BF16 and nearest-even boundaries.
//! Each lane reads its complete group. Groups and dot products do not change.
fn source(v:f32)->f32 {crate::bf16::rounded(v)}
fn scale(code:u8)->f32 {f32::from_bits(if code==0 {0x00400000} else {u32::from(code)<<23})}
fn value(code:u8)->f32 {
    let exponent=(code>>3)&15;let fraction=code&7;
    if exponent==0 {fraction as f32/512.0}
    else {f32::from_bits(((u32::from(exponent)+120)<<23)|(u32::from(fraction)<<20))}
}
pub fn lane(input:&[f32],index:usize,plane:u32)->f32 {
    quantize(input,index,plane,false)
}
pub fn positive_zero_lane(input:&[f32],index:usize,plane:u32)->f32 {
    quantize(input,index,plane,true)
}
fn quantize(input:&[f32],index:usize,plane:u32,positive_zero:bool)->f32 {
    let start=index/32*32;let mut maximum=0.0f32;
    for &v in &input[start..start+32] {let v=source(v);if !v.is_finite() {return f32::NAN;}maximum=maximum.max(v.abs());}
    finish(maximum,input[index],plane,positive_zero)
}
/// The scale code of a group: the smallest code whose top value (448 times its scale, 1.75 x 2^(code-119)) is not below
/// the group maximum. Floats that are not negative order as their bit patterns, and the top value of code c has the
/// exponent field c+8 and the fraction 0x600000. So the code is the maximum's exponent field less 8, one more when its
/// fraction is above 0x600000, and 0 below. This is the answer of the binary search over the 255 codes that stood here
/// (`tests::finish_search`), for every bit pattern: +infinity gives 254; a NaN or a negative pattern gives 0.
#[inline(always)]
fn scale_code(maximum:f32)->u8 {
    let bits=maximum.to_bits();
    if bits>=0x7f80_0000 {return if bits==0x7f80_0000 {254} else {0};}
    ((bits>>23)+u32::from(bits&0x007f_ffff>0x0060_0000)).saturating_sub(8) as u8
}
/// The E4M3 code (0 to 126) of |source| / 2^(scale-127): the nearest value of the grid, ties to the even code. The grid
/// has steps of 2^-9 below 2^-5 and 8 steps a binade above, to 448 (code 126). `source` is finite. The quotient is exact
/// (a division by a power of two), so the rounding is one shift of the significand with round-half-to-even. This is the
/// answer of the binary search over the 127 values and its tie rule that stood here (`tests::finish_search`).
#[inline(always)]
fn magnitude_code(source:f32,scale:u8)->u8 {
    let bits=source.to_bits()&0x7fff_ffff;
    if bits==0 {return 0;}
    let exponent=bits>>23;
    let significand=if exponent==0 {bits} else {bits&0x007f_ffff|0x0080_0000};
    // |source| = significand x 2^(e-23), with e = exponent-127, or -126 for a subnormal. Scaled: significand x 2^t.
    let t=(if exponent==0 {-126} else {exponent as i32-127})-23+127-scale as i32;
    let top=31-significand.leading_zeros() as i32;
    // The E4M3 exponent field of the scaled magnitude; the subnormal steps share the step of field 1.
    let field=if top+t+7>1 {top+t+7} else {1};
    // Steps of 2^(field-10) in the magnitude: significand x 2^(t-field+10), to nearest, ties to even.
    let shift=field-10-t;
    let steps=if shift<=0 {if shift< -7 {u32::MAX>>8} else {significand<<(-shift) as u32}} else if shift>=32 {0} else {
        let (floor,rest,half)=(significand>>shift,significand&((1u32<<shift)-1),1u32<<(shift-1));
        floor+u32::from(rest>half || rest==half && floor&1==1)};
    let code=(field-1) as u32*8+steps;
    if code>126 {126} else {code as u8}
}
/// Everything after the group maximum: the scale code, the lane's code and the requested plane. `maximum` is the largest
/// absolute BF16-rounded value of the lane's group (all finite), `raw` the lane's own input word. The device version that
/// shares one maximum between the 32 lanes of a warp calls this with the same arguments, so it writes the same bytes.
/// The two codes are integer steps on the bit patterns; they give the codes of the FP64 binary searches they replace
/// (host tests: every BF16 maximum; every BF16 source at every scale code).
pub fn finish(maximum:f32,raw:f32,plane:u32,positive_zero:bool)->f32 {
    let encoded_scale=scale_code(maximum);let descale=scale(encoded_scale);let source=source(raw);
    let low=magnitude_code(source,encoded_scale);
    // The legacy (shared) rule is main's exact expression. Option B is GLM only.
    let negative=if positive_zero {raw<0.0} else {source.is_sign_negative()};
    let code=low|if negative {128} else {0};
    match plane {1=>code as f32,2=>encoded_scale as f32,_=>{
        let v=if code&128==0 {value(code&127)} else {-value(code&127)};
        (v as f64*descale as f64) as f32
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
    /// `finish` as it stood before the integer steps: the two binary searches in FP64. The reference of the tests below.
    fn finish_search(maximum:f32,raw:f32,plane:u32,positive_zero:bool)->f32 {
        let (mut low,mut high)=(0u8,254u8);
        while low<high {let middle=low+(high-low)/2;
            if maximum as f64>448.0*scale(middle) as f64 {low=middle+1;} else {high=middle;}}
        let encoded_scale=low;let descale=scale(encoded_scale);let source=source(raw);
        let scaled=(source as f64/descale as f64) as f32;let magnitude=scaled.abs();
        let (mut low,mut high)=(0u8,126u8);
        while low<high {let middle=low+(high-low)/2;if value(middle)<magnitude {low=middle+1;} else {high=middle;}}
        if low>0 {let lower=low-1;let down=magnitude as f64-value(lower) as f64;let up=value(low) as f64-magnitude as f64;
            if down<up || down==up && lower%2==0 {low=lower;}}
        let negative=if positive_zero {raw<0.0} else {source.is_sign_negative()};
        let code=low|if negative {128} else {0};
        match plane {1=>code as f32,2=>encoded_scale as f32,_=>{
            let v=if code&128==0 {value(code&127)} else {-value(code&127)};
            (v as f64*descale as f64) as f32
        }}
    }
    /// The top value of a scale code (448 times the scale): the largest group maximum that takes the code. Code 247 is the
    /// last a finite maximum takes; its top is past the largest float, so the largest BF16 value stands for it.
    fn top(code:u8)->f32 {if code>=247 {f32::from_bits(0x7f7f_0000)} else {f32::from_bits((u32::from(code)+8)<<23|0x0060_0000)}}
    fn sources_agree(codes:&[u8]) {
        for &code in codes {
            let maximum=top(code);
            assert_eq!(finish_search(maximum,0.0,2,false),f32::from(code));
            for pattern in 0..=u16::MAX {
                let raw=f32::from_bits(u32::from(pattern)<<16);
                if !raw.is_finite() {continue;}
                for positive_zero in [false,true] {for plane in 0..3 {
                    assert_eq!(finish(maximum,raw,plane,positive_zero).to_bits(),finish_search(maximum,raw,plane,positive_zero).to_bits(),"code {code} source {pattern:#06x} plane {plane} {positive_zero}");
                }}
            }
        }
    }
    #[test]
    fn the_scale_code_is_the_search_for_every_bf16_maximum() {
        // Every BF16 pattern that is not negative (the finite ones are the group maxima that occur), and every f32 neighbour
        // of each code's top value.
        for pattern in 0..0x8000u32 {let maximum=f32::from_bits(pattern<<16);
            assert_eq!(f32::from(scale_code(maximum)),finish_search(maximum,0.0,2,false),"maximum {pattern:#06x}");}
        for code in 0..=246u8 {let edge=top(code).to_bits();
            for bits in [edge-1,edge,edge+1] {let maximum=f32::from_bits(bits);assert_eq!(f32::from(scale_code(maximum)),finish_search(maximum,0.0,2,false),"edge {bits:#010x}");}
            assert_eq!(scale_code(top(code)),code);}
        for bits in [0u32,1,0x007f_ffff,0x0080_0000,0x7f7f_ffff,0x7f80_0000,0x7fc0_0000,0x8000_0000,0xbf80_0000,0xff80_0000] {let maximum=f32::from_bits(bits);
            assert_eq!(f32::from(scale_code(maximum)),finish_search(maximum,0.0,2,false),"special {bits:#010x}");}
    }
    #[test]
    fn every_bf16_source_takes_the_search_code_at_the_edge_scales() {
        // Every finite BF16 source, both zero rules, the three planes, at the scale codes around every boundary of the
        // arithmetic: the lowest, the subnormal source range, the middle, and the highest a finite maximum takes.
        sources_agree(&[0,1,2,7,8,9,22,23,24,100,118,119,120,126,127,128,135,136,200,245,246,247]);
        // A raw word below BF16's smallest step rounds to a signed zero; the two zero rules differ on it.
        for raw in [f32::from_bits(1),-f32::from_bits(1),1.0e-41,-1.0e-41,-0.0,0.0] {for code in [0u8,119,127,247] {for positive_zero in [false,true] {for plane in 0..3 {
            assert_eq!(finish(top(code),raw,plane,positive_zero).to_bits(),finish_search(top(code),raw,plane,positive_zero).to_bits());}}}}
    }
    /// The whole table: every finite BF16 source at every scale code (48.6 million triples a zero rule). Run on demand:
    /// cargo test --release -p spark-gpu-probe --lib mxfp8_quantize -- --ignored
    #[test]
    #[ignore]
    fn every_bf16_source_takes_the_search_code_at_every_scale() {
        let codes:Vec<u8>=(0..=247).collect();sources_agree(&codes);
    }
    #[test]
    fn exact_zeros_are_positive_and_negative_underflow_remains_negative() {
        let mut input=[0.0;96];input[..8].copy_from_slice(&[448.0,0.0,-0.0,1.0/2048.0,-1.0/2048.0,f32::from_bits(1),-f32::from_bits(1),-1.0/1024.0]);
        input[32..64].fill(-0.0);input[64..96].fill(-f32::from_bits(1));
        let expected=[0x7eu8,0,0,0,0x80,0,0x80,0x80];
        for (index,&code) in expected.iter().enumerate() {assert_eq!(positive_zero_lane(&input,index,1) as u8,code);}
        for index in 32..96 {let negative=index>=64;assert_eq!(positive_zero_lane(&input,index,1) as u8,if negative {128} else {0});assert_eq!(positive_zero_lane(&input,index,0).to_bits(),if negative {(-0.0f32).to_bits()} else {0});assert_eq!(positive_zero_lane(&input,index,2),0.0);}
        for index in 0..input.len() {assert_eq!((lane(&input,index,1) as u8)^(positive_zero_lane(&input,index,1) as u8),if input[index].to_bits()==(-0.0f32).to_bits() {128} else {0});}
    }
    #[test]
    fn lanes_match_the_independent_host_bytes_and_decoded_bits() {
        let mut inputs=Vec::new();let mut seed=0x1d3a5b79u32;
        for exponent in 0..255u32 {let mut row=[0.0f32;32];for v in &mut row {
            seed=seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v=f32::from_bits((seed&0x807fffff)|(exponent<<23));
        }if crate::dynamic_mxfp8::Activation::from_f32(&row).is_ok() {inputs.extend(row);}
        else {assert!((0..32).any(|i|!lane(&row,i,0).is_finite()));}}
        inputs.extend([0.0;32]);inputs.extend([-0.0;32]);
        let mut edge=[0.0;32];edge[..8].copy_from_slice(&[448.0,1.0625,1.1875,-1.0625,-0.0,1.0/1024.0,3.0/1024.0,450.0]);inputs.extend(edge);
        let expected=crate::dynamic_mxfp8::Activation::from_f32(&inputs).unwrap();
        for i in 0..inputs.len() {assert_eq!(lane(&inputs,i,0).to_bits(),expected.decoded()[i].to_bits(),"lane {i}");
            assert_eq!(lane(&inputs,i,1) as u8,expected.values()[i],"code {i}");assert_eq!(lane(&inputs,i,2) as u8,expected.scales()[i/32],"scale {i}");}
        let expected=crate::dynamic_mxfp8::Activation::from_f32_positive_zero(&inputs).unwrap();
        for index in 0..inputs.len() {assert_eq!(positive_zero_lane(&inputs,index,0).to_bits(),expected.decoded()[index].to_bits());
            assert_eq!(positive_zero_lane(&inputs,index,1) as u8,expected.values()[index]);assert_eq!(positive_zero_lane(&inputs,index,2) as u8,expected.scales()[index/32]);}
    }
}
