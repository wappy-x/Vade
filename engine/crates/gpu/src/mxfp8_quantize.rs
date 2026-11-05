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
    let (mut low,mut high)=(0u8,254u8);
    while low<high {let middle=low+(high-low)/2;
        if maximum as f64>448.0*scale(middle) as f64 {low=middle+1;} else {high=middle;}}
    let encoded_scale=low;let descale=scale(encoded_scale);let source=source(input[index]);
    let scaled=(source as f64/descale as f64) as f32;let magnitude=scaled.abs();
    let (mut low,mut high)=(0u8,126u8);
    while low<high {let middle=low+(high-low)/2;if value(middle)<magnitude {low=middle+1;} else {high=middle;}}
    if low>0 {let lower=low-1;let down=magnitude as f64-value(lower) as f64;let up=value(low) as f64-magnitude as f64;
        if down<up || down==up && lower%2==0 {low=lower;}}
    // The legacy (shared) rule is main's exact expression. Option B is GLM only.
    let negative=if positive_zero {input[index]<0.0} else {source.is_sign_negative()};
    let code=low|if negative {128} else {0};
    match plane {1=>code as f32,2=>encoded_scale as f32,_=>{
        let v=if code&128==0 {value(code&127)} else {-value(code&127)};
        (v as f64*descale as f64) as f32
    }}
}

#[cfg(test)]
mod tests {
    use super::*;
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
