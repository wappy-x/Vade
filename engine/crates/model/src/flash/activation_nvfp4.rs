//! Calibrated NVFP4 input recipe. This does not change logical-f32-v1.
//! Use complete BF16-source groups and the retained positive FP32 input scale.
use crate::{Result,invalid,flash::nvfp4::{e2m1,e4m3,f32_to_bf16,bf16_to_f32}};
pub struct Activation {codes:Vec<u8>,scales:Vec<u8>,global:f32}
fn nearest(value:f32,decode:impl Fn(u8)->f32,last:u8)->u8 {
    let mut lower=0u8;while lower<last && decode(lower+1)<=value {lower+=1;}
    if lower==last {return lower;}
    let a=value as f64-decode(lower) as f64;let b=decode(lower+1) as f64-value as f64;
    if a<b || a==b && lower.is_multiple_of(2) {lower} else {lower+1}
}
impl Activation {
    pub fn from_bf16_source(input:&[f32],global:f32)->Result<Self> {
        if input.is_empty() || !input.len().is_multiple_of(16) || !global.is_finite() || global<=0.0 || input.iter().any(|v|!v.is_finite()) {
            return Err(invalid("NVFP4 input requires complete finite groups and a positive calibrated scale"));
        }
        let mut codes=Vec::with_capacity(input.len()/2);let mut scales=Vec::with_capacity(input.len()/16);
        for group in input.as_chunks::<16>().0 {
            let values=group.iter().map(|&v|bf16_to_f32(f32_to_bf16(v))).collect::<Vec<_>>();
            if values.iter().any(|v|!v.is_finite()) {return Err(invalid("NVFP4 source overflows BF16"));}
            let maximum=values.iter().fold(0.0f32,|m,&v|m.max(v.abs()));
            let raw=maximum/global/6.0;
            if !raw.is_finite() {return Err(invalid("NVFP4 block scale exceeds the calibrated range"));}
            let encoded=nearest(raw,e4m3,126);let block=e4m3(encoded);scales.push(encoded);
            for (pair,original) in values.as_chunks::<2>().0.iter().zip(group.as_chunks::<2>().0.iter()) {
                let quantize=|v:f32,source:f32| {let magnitude=if block==0.0 {0.0} else {v.abs()/global/block};
                    nearest(magnitude,e2m1,7)|if source<0.0 {8} else {0}};
                codes.push(quantize(pair[0],original[0])|(quantize(pair[1],original[1])<<4));
            }
        }
        Ok(Self {codes,scales,global})
    }
    pub fn codes(&self)->&[u8] {&self.codes}
    pub fn scales(&self)->&[u8] {&self.scales}
    pub fn global(&self)->f32 {self.global}
    /// Fold global scales once. Visit complete K in ascending feature order.
    /// Both code products and both block scales precede the fixed FP32 fold.
    pub fn project(&self,weight:&[u8],scales:&[u8],global:f32)->Result<Vec<f32>> {
        let cols=self.codes.len()*2;
        if cols==0 || weight.is_empty() || !weight.len().is_multiple_of(self.codes.len()) || !global.is_finite() || global<=0.0 {
            return Err(invalid("NVFP4 reference matrix or global scale differs"));
        }
        let rows=weight.len()/self.codes.len();
        if scales.len()!=rows*cols/16 || scales.iter().any(|&v|!e4m3(v).is_finite() || e4m3(v)<0.0) {
            return Err(invalid("NVFP4 reference block scale extent or value differs"));
        }
        let combined=global*self.global;if !combined.is_finite() {return Err(invalid("NVFP4 combined global scale is not finite"));}
        let mut out=Vec::with_capacity(rows);
        for row in 0..rows {let mut sum=0.0;
            for group in 0..cols/16 {let block=(e4m3(scales[row*cols/16+group])*e4m3(self.scales[group]))*combined;
                for feature in group*16..(group+1)*16 {
                    let a=self.codes[feature/2]>>((feature%2)*4);let b=weight[row*cols/2+feature/2]>>((feature%2)*4);
                    sum+=(e2m1(a)*e2m1(b))*block;
                }
            }
            let value=bf16_to_f32(f32_to_bf16(sum));if !value.is_finite() {return Err(invalid("NVFP4 reference epilogue is not finite"));}out.push(value);
        }Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibrated_groups_canonicalize_exact_zero_and_keep_nearest_even_codes() {
        let mut input=[0.0;16];input[..8].copy_from_slice(&[6.0,0.25,0.75,1.25,1.75,2.5,3.5,-0.0]);
        let a=Activation::from_bf16_source(&input,1.0).unwrap();assert_eq!(a.scales(),[0x38]);assert_eq!(&a.codes()[..4],&[0x07,0x22,0x44,0x06]);
        let zero=Activation::from_bf16_source(&[-0.0;16],2.0).unwrap();assert_eq!(zero.scales(),[0]);assert_eq!(zero.codes(),[0;8]);
        assert_eq!(a.global(),1.0);assert!(Activation::from_bf16_source(&input,0.0).is_err());
    }
    #[test]
    fn exact_zero_and_negative_underflow_have_distinct_signs_in_each_block() {
        for source in [0.0,-0.0] {let q=Activation::from_bf16_source(&[source;16],1.0).unwrap();assert_eq!(q.codes(),[0;8]);assert_eq!(q.scales(),[0]);}
        let mut input=[0.0;64];input[..8].copy_from_slice(&[6.0,0.0,-0.0,0.125,-0.125,f32::from_bits(1),-f32::from_bits(1),-0.25]);
        input[16..32].fill(-0.0);input[32..48].fill(-f32::from_bits(1));input[48..64].fill(-f32::from_bits(0x0001_0000));
        let q=Activation::from_bf16_source(&input,1.0).unwrap();assert_eq!(&q.codes()[..4],&[0x07,0x00,0x08,0x88]);
        assert_eq!(&q.codes()[8..16],&[0;8]);assert_eq!(&q.codes()[16..],&[0x88;16]);assert_eq!(&q.scales()[1..],&[0;3]);
        for (index,block) in input.as_chunks::<16>().0.iter().enumerate() {let single=Activation::from_bf16_source(block,1.0).unwrap();assert_eq!(single.codes(),&q.codes()[index*8..index*8+8]);assert_eq!(single.scales(),&q.scales()[index..index+1]);}
    }
    #[test]
    fn global_scales_fold_once_and_epilogue_rounds_after_complete_k() {
        let a=Activation::from_bf16_source(&[1.0;16],0.5).unwrap();let actual=a.project(&[0x22;8],&[0x38],2.0).unwrap();
        assert_eq!(actual.len(),1);assert!(actual[0]>0.0);assert_eq!(actual,a.project(&[0x22;8],&[0x38],2.0).unwrap());
        assert!(a.project(&[0x22;8],&[],2.0).is_err());assert!(a.project(&[0x22;8],&[0x7f],2.0).is_err());
    }
}
