//! Stored MXFP8 candidate: preserve norm lanes, fixed FP32 tree, BF16 epilogue.
//! The checkpoint's Linear replacement and effective load policy stay unqualified.
use crate::{bf16,norm_boundary::WeightType,text_reference};

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Shape {rows:usize,cols:usize,input:WeightType}
impl Shape {
    pub fn new(rows:usize,cols:usize,input:WeightType)->Result<Self,String> {
        if rows==0 || cols==0 || cols%32!=0 || rows>u32::MAX as usize || cols>u32::MAX as usize
            || rows.checked_mul(cols).is_none_or(|n|n>isize::MAX as usize)
            || rows.checked_mul(4).is_none_or(|n|n>isize::MAX as usize)
            || cols.checked_mul(4).is_none_or(|n|n>isize::MAX as usize) {
            return Err("invalid complete MXFP8 projection dimensions".into());
        }Ok(Self {rows,cols,input})
    }
    pub fn rows(self)->usize {self.rows}
    pub fn cols(self)->usize {self.cols}
    pub fn input(self)->WeightType {self.input}
    pub fn weight_bytes(self)->usize {self.rows*self.cols}
    pub fn scale_bytes(self)->usize {self.weight_bytes()/32}
    pub fn arena_bytes(self)->Result<usize,String> {
        self.weight_bytes().checked_add(self.scale_bytes()).filter(|&n|n<=isize::MAX as usize)
            .ok_or_else(||"MXFP8 arena size overflow".into())
    }
    pub fn check_input(self,input:&[f32])->Result<(),String> {
        if input.len()!=self.cols || input.iter().any(|&v| !v.is_finite()
            || self.input==WeightType::Bf16 && bf16::rounded(v).to_bits()!=v.to_bits()) {
            return Err("MXFP8 input differs from its stored norm contract".into());
        }Ok(())
    }
    /// Return the accumulator and its one RN-even BF16 epilogue separately.
    /// No activation quantizer or input narrowing is part of this candidate.
    pub fn reference(self,weights:&[u8],scales:&[u8],input:&[f32])->Result<(Vec<f32>,Vec<f32>),String> {
        self.check_input(input)?;
        if weights.len()!=self.weight_bytes() || scales.len()!=self.scale_bytes() {
            return Err("MXFP8 reference requires complete selected rows and scale groups".into());
        }
        let mut accumulator=Vec::with_capacity(self.rows);let mut output=Vec::with_capacity(self.rows);
        for (w,s) in weights.chunks_exact(self.cols).zip(scales.chunks_exact(self.cols/32)) {
            let v=text_reference::project_row(w,s,input)?;let rounded=bf16::rounded(v);
            if !rounded.is_finite() {return Err("MXFP8 epilogue is not finite BF16".into());}
            accumulator.push(v);output.push(rounded);
        }Ok((accumulator,output))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp32_norm_residual_reaches_the_accumulator_before_the_bf16_epilogue() {
        let s=Shape::new(1,32,WeightType::F32).unwrap();let mut weights=[0;32];weights[0]=0x38;
        let mut input=[0.0;32];input[0]=1.0+1.0/1024.0;
        let (acc,out)=s.reference(&weights,&[127],&input).unwrap();
        assert_eq!(acc[0].to_bits(),input[0].to_bits());assert_eq!(out,[1.0]);
        assert!(Shape::new(1,32,WeightType::Bf16).unwrap().reference(&weights,&[127],&input).is_err());
        input[0]=1.0+1.0/256.0;assert_eq!(s.reference(&weights,&[127],&input).unwrap().1,[1.0]);
        input[0]=1.0+3.0/256.0;assert_eq!(s.reference(&weights,&[127],&input).unwrap().1,[1.0+1.0/64.0]);
    }
    #[test]
    fn complete_k_and_fixed_tree_do_not_follow_row_or_token_count() {
        let mut input=vec![0.0;512];input[0]=16_777_216.0;input[8]=-16_777_216.0;input[16]=1.0;input[256]=1.0;
        let s=Shape::new(3,512,WeightType::F32).unwrap();
        let all=s.reference(&vec![0x38;1536],&[127;48],&input).unwrap();assert_eq!(all.0,[0.0;3]);
        let row=Shape::new(1,512,WeightType::F32).unwrap().reference(&vec![0x38;512],&[127;16],&input).unwrap();
        for _ in 0..3 {assert_eq!(row.0[0].to_bits(),all.0[2].to_bits());}
        input.fill(0.0);input[511]=2.0;
        assert_eq!(s.reference(&vec![0x38;1536],&[127;48],&input).unwrap().1,[2.0;3]);
    }
    #[test]
    fn malformed_extents_nonfinite_values_and_bf16_overflow_refuse() {
        for (r,c) in [(0,32),(1,0),(1,31),(usize::MAX,32),(1,usize::MAX)] {assert!(Shape::new(r,c,WeightType::F32).is_err());}
        let s=Shape::new(1,32,WeightType::F32).unwrap();
        assert!(s.reference(&[0x38;31],&[127],&[1.0;32]).is_err());
        assert!(s.reference(&[0x38;32],&[],&[1.0;32]).is_err());
        assert!(s.reference(&[0x7f;32],&[127],&[1.0;32]).is_err());
        assert!(s.reference(&[0x38;32],&[255],&[1.0;32]).is_err());
        assert!(s.reference(&[0x38;32],&[127],&[f32::NAN;32]).is_err());
        let mut w=[0;32];w[0]=0x38;let mut x=[0.0;32];x[0]=f32::MAX;
        assert!(s.reference(&w,&[127],&x).unwrap_err().contains("epilogue"));
        assert_eq!(s.reference(&[0x38;32],&[0],&[1.0;32]).unwrap().0[0].to_bits(),0x02800000);
    }
}
