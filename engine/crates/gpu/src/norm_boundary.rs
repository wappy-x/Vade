//! Candidate ordinary RMSNorm boundary. Loaded parameter precision is explicit.
//! This checks operands. It does not prove a checkpoint's load policy or reductions.
use crate::bf16;

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum WeightType { Bf16, F32 }
#[derive(Debug,Clone,Copy,PartialEq)]
pub struct NormShape { width: usize, epsilon: f32, weight_type: WeightType }
impl NormShape {
    pub fn new(width: usize,epsilon: f32,weight_type: WeightType) -> Result<Self,String> {
        if width==0 || width>u32::MAX as usize || width>isize::MAX as usize/4
            || !epsilon.is_finite() || epsilon<=0.0 {return Err("invalid resident norm shape".into());}
        Ok(Self {width,epsilon,weight_type})
    }
    pub fn width(self) -> usize {self.width}
    pub fn epsilon(self) -> f32 {self.epsilon}
    pub fn weight_type(self) -> WeightType {self.weight_type}
    pub fn check_weights(self,weights: &[f32]) -> Result<(),String> {
        if weights.len()!=self.width || weights.iter().any(|&v| !v.is_finite()
            || self.weight_type==WeightType::Bf16 && bf16::rounded(v).to_bits()!=v.to_bits()) {
            return Err("resident norm weights differ from the declared precision or extent".into());
        } Ok(())
    }
    /// Pointwise contract only. The supplied value precedes the normalized cast.
    /// A stored-precision candidate must not silently narrow an FP32 parameter.
    pub fn weight(self,normalized: f32,weight: f32) -> f32 {
        let product=bf16::rounded(normalized)*weight;
        match self.weight_type {WeightType::Bf16=>bf16::rounded(product),WeightType::F32=>product}
    }
}
