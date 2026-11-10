//! Dynamic row-major MXFP8 activations from a complete BF16 feature row.
//! Each group has 32 E4M3FN values and one unsigned E8M0 scale.
use crate::{bf16,text_reference::{e4m3,scale}};

pub struct Activation {values:Vec<u8>,scales:Vec<u8>,decoded:Vec<f32>}
impl Activation {
    /// Round the producer's values to the recipe's BF16 source boundary first.
    /// Scale uses ceil(log2(amax/448)), clamped to the finite E8M0 range.
    /// E4M3FN uses nearest-even rounding and retains the source sign.
    pub fn from_f32(input:&[f32])->Result<Self,String> {
        Self::from_source(input,false)
    }
    pub fn from_f32_positive_zero(input:&[f32])->Result<Self,String> {
        Self::from_source(input,true)
    }
    fn from_source(input:&[f32],positive_zero:bool)->Result<Self,String> {
        if input.is_empty() || !input.len().is_multiple_of(32) || input.iter().any(|v|!v.is_finite()) {
            return Err("dynamic MXFP8 requires complete finite feature groups".into());
        }
        let source=input.iter().map(|&v|bf16::rounded(v)).collect::<Vec<_>>();
        if source.iter().any(|v|!v.is_finite()) {return Err("dynamic MXFP8 source overflows BF16".into());}
        let mut values=Vec::with_capacity(input.len());let mut scales=Vec::with_capacity(input.len()/32);
        let mut decoded=Vec::with_capacity(input.len());
        for (block,original) in source.as_chunks::<32>().0.iter().zip(input.as_chunks::<32>().0.iter()) {
            let maximum=block.iter().fold(0.0f32,|a,&b|a.max(b.abs()));
            // BF16 maxima are exact. Compare powers of two without log library
            // rounding. This also covers zero and BF16 subnormal values.
            let mut exponent=-127;
            while exponent<127 && maximum as f64>448.0*2f64.powi(exponent) {exponent+=1;}
            let encoded=(exponent+127) as u8;let descale=scale(encoded);
            scales.push(encoded);
            for (&v,&original) in block.iter().zip(original) {
                let scaled=(v as f64/descale as f64) as f32;
                // The legacy (shared) rule is main's exact expression. Option B is GLM only.
                let byte=if positive_zero {nearest(scaled.abs())|if original<0.0 {128} else {0}} else {nearest(scaled)};let value=(e4m3(byte) as f64*descale as f64) as f32;
                if !value.is_finite() {return Err("dynamic MXFP8 decoded activation is not finite".into());}
                values.push(byte);decoded.push(value);
            }
        }
        Ok(Self {values,scales,decoded})
    }
    pub fn values(&self)->&[u8] {&self.values}
    pub fn scales(&self)->&[u8] {&self.scales}
    /// Exact decoded FP32 lanes for the fixed projection kernel.
    pub fn decoded(&self)->&[f32] {&self.decoded}
}
fn nearest(value:f32)->u8 {
    let sign=if value.is_sign_negative() {128} else {0};let x=value.abs();
    let mut low=0u8;let mut high=126u8;
    while low<high {
        let mid=low+(high-low)/2;
        if e4m3(mid)<x {low=mid+1;} else {high=mid;}
    }
    let upper=low;
    let chosen=if upper==0 {0} else {
        let lower=upper-1;let down=x as f64-e4m3(lower) as f64;let up=e4m3(upper) as f64-x as f64;
        if down<up || down==up && lower.is_multiple_of(2) {lower} else {upper}
    };
    sign|chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hand_checked_scales_ties_signs_and_source_boundary() {
        let mut input=[0.0;64];
        input[..8].copy_from_slice(&[448.0,1.0625,1.1875,-1.0625,-0.0,1.0/1024.0,3.0/1024.0,1.0+1.0/1024.0]);
        input[32]=896.0;input[33]=2.125;
        let q=Activation::from_f32(&input).unwrap();
        assert_eq!(q.scales(),[127,128]);
        assert_eq!(&q.values()[..8],&[0x7e,0x38,0x3a,0xb8,0x80,0,2,0x38]);
        assert_eq!(&q.values()[32..34],&[0x7e,0x38]);
        assert_eq!(q.decoded()[4].to_bits(),(-0.0f32).to_bits());
        assert_eq!(&q.decoded()[..4],&[448.0,1.0,1.25,-1.0]);
        let mut boundary=[0.0;32];boundary[0]=450.0;
        assert_eq!(Activation::from_f32(&boundary).unwrap().scales(),[128]);
    }
    #[test]
    fn zero_subnormal_and_block_placement_are_exact() {
        let zero=Activation::from_f32(&[0.0;32]).unwrap();assert_eq!(zero.scales(),[0]);assert_eq!(zero.values(),[0;32]);
        let mut input=[0.0;64];input[0]=f32::from_bits(0x00010000);input[32]=448.0;
        let q=Activation::from_f32(&input).unwrap();assert_eq!(q.scales(),[0,127]);assert_eq!(q.values()[0],8);
        assert_eq!(q.decoded()[0].to_bits(),input[0].to_bits());
        let first=Activation::from_f32(&input[..32]).unwrap();let last=Activation::from_f32(&input[32..]).unwrap();
        assert_eq!(q.values(),[first.values(),last.values()].concat());
        assert_eq!(q.scales(),[first.scales(),last.scales()].concat());
    }
    #[test]
    fn exact_zero_and_negative_underflow_keep_their_required_block_signs() {
        for source in [0.0,-0.0] {let q=Activation::from_f32_positive_zero(&[source;32]).unwrap();assert_eq!(q.values(),[0;32]);assert_eq!(q.scales(),[0]);assert!(q.decoded().iter().all(|v|v.to_bits()==0));}
        let mut input=[0.0;96];input[..8].copy_from_slice(&[448.0,0.0,-0.0,1.0/2048.0,-1.0/2048.0,f32::from_bits(1),-f32::from_bits(1),-1.0/1024.0]);
        input[32..64].fill(-0.0);input[64..96].fill(-f32::from_bits(1));
        let q=Activation::from_f32_positive_zero(&input).unwrap();assert_eq!(&q.values()[..8],&[0x7e,0,0,0,0x80,0,0x80,0x80]);
        assert_eq!(&q.values()[32..64],&[0;32]);assert_eq!(&q.values()[64..],&[0x80;32]);assert_eq!(&q.scales()[1..],&[0;2]);
        for (index,block) in input.as_chunks::<32>().0.iter().enumerate() {let single=Activation::from_f32_positive_zero(block).unwrap();assert_eq!(single.values(),&q.values()[index*32..index*32+32]);assert_eq!(single.scales(),&q.scales()[index..index+1]);}
        let legacy=Activation::from_f32(&input).unwrap();assert_eq!(legacy.scales(),q.scales());
        for (index,&source) in input.iter().enumerate() {let changed=source.to_bits()==(-0.0f32).to_bits();assert_eq!(legacy.values()[index]^q.values()[index],if changed {128} else {0});}
    }
    #[test]
    fn every_finite_encoding_rounds_back_at_unit_scale() {
        for byte in 0..=254u8 {
            if byte&127==127 {continue;}
            assert_eq!(nearest(e4m3(byte)),byte);
        }
        for input in [vec![],vec![1.0;31],vec![f32::NAN;32],vec![f32::INFINITY;32],vec![f32::MAX;32]] {
            assert!(Activation::from_f32(&input).is_err());
        }
    }
}
