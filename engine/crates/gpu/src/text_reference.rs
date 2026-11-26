//! Host oracle for stored BF16 rows and the candidate single-row MXFP8 head.
//! This checks primitive arithmetic. It does not qualify a model activation policy.

/// Widen original BF16 bytes. Do not round or normalize the stored row.
pub fn embedding(bytes:&[u8]) -> Result<Vec<f32>,String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(2) {return Err("invalid BF16 row extent".into());}
    let values:Vec<_>=bytes.as_chunks::<2>().0.iter().map(|b|
        f32::from_bits(u32::from(u16::from_le_bytes(*b))<<16)).collect();
    if values.iter().any(|v|!v.is_finite()) {return Err("nonfinite BF16 row".into());}
    Ok(values)
}
pub(crate) fn e4m3(byte:u8)->f32 {
    let exponent=(byte>>3)&15; let fraction=byte&7;
    let value=if exponent==0 {fraction as f32/512.0}
        else {f32::from_bits(((u32::from(exponent)+120)<<23)|(u32::from(fraction)<<20))};
    if byte&128==0 {value} else {-value}
}
pub(crate) fn scale(byte:u8)->f32 {f32::from_bits(if byte==0 {0x00400000} else {u32::from(byte)<<23})}

/// Follow next_decode.rs::row: 8 columns per lane, ascending 256-column tiles,
/// fused multiply-add, then a fixed down-shuffle tree. Input stays FP32.
/// Each scale covers 32 columns. Scale zero denotes 2^-127, not zero.
pub fn project_row(weights:&[u8],scales:&[u8],input:&[f32])->Result<f32,String> {
    let hidden=input.len();
    if hidden==0 || !hidden.is_multiple_of(32) || weights.len()!=hidden || scales.len()!=hidden/32 {
        return Err("invalid MXFP8 reference row extent".into());
    }
    if input.iter().any(|x|!x.is_finite()) || scales.contains(&255)
        || weights.iter().any(|w|w&127==127) {return Err("nonfinite MXFP8 reference operand".into());}
    let mut lanes=[0.0f32;32];
    for (lane,sum) in lanes.iter_mut().enumerate() {
        for tile in 0..hidden.div_ceil(256) {
            let start=tile*256+lane*8;
            for col in start..(start+8).min(hidden) {
                let weight=e4m3(weights[col])*scale(scales[col/32]);
                *sum=weight.mul_add(input[col],*sum);
            }
        }
    }
    for distance in [16,8,4,2,1] {
        let old=lanes;
        for lane in 0..32-distance {lanes[lane]+=old[lane+distance];}
    }
    if !lanes[0].is_finite() {return Err("nonfinite MXFP8 reference output".into());}
    Ok(lanes[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_bf16_bits_and_refusals() {
        let values=embedding(&[0x80,0x3f,0x00,0x80,0x01,0x00]).unwrap();
        assert_eq!(values.iter().map(|x|x.to_bits()).collect::<Vec<_>>(),[0x3f800000,0x80000000,0x00010000]);
        for bad in [&[][..],&[0][..],&[0x80,0x7f][..]] {assert!(embedding(bad).is_err());}
    }
    #[test]
    fn hand_checked_encodings_and_scale_boundaries() {
        for (raw,value) in [(0,0.0f32),(1,1.0/512.0),(7,7.0/512.0),(8,1.0/64.0),
            (0x38,1.0),(0x7e,448.0),(0xb8,-1.0),(0xfe,-448.0)] {
            assert_eq!(e4m3(raw).to_bits(),value.to_bits());
        }
        assert_eq!(e4m3(128).to_bits(),(-0.0f32).to_bits());
        assert_eq!(project_row(&[0x38;32],&[127],&[0.5;32]).unwrap(),16.0);
        assert_eq!(project_row(&[0x38;32],&[126],&[0.5;32]).unwrap(),8.0);
        assert_eq!(project_row(&[0x38;32],&[0],&[1.0;32]).unwrap().to_bits(),0x02800000);
    }
    #[test]
    fn fixed_tree_is_not_a_sequential_dot_product() {
        let mut input=vec![0.0;512];
        input[0]=16_777_216.0;input[8]= -16_777_216.0;input[16]=1.0;input[256]=1.0;
        assert_eq!(project_row(&vec![0x38;512],&[127;16],&input).unwrap(),0.0);
        let sequential=input.iter().fold(0.0,|sum,&x|1.0f32.mul_add(x,sum));
        assert_eq!(sequential,2.0);
        assert!(project_row(&[0x7f;32],&[127],&[0.0;32]).is_err());
        assert!(project_row(&[0x38;32],&[255],&[0.0;32]).is_err());
        assert!(project_row(&[0x38;32],&[127],&[f32::NAN;32]).is_err());
        assert!(project_row(&[0x38;32],&[127],&[f32::MAX;32]).is_err());
        assert!(project_row(&[0x38;32],&[],&[0.0;32]).is_err());
    }
}
