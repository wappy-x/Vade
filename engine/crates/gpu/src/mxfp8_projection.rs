//! Stored MXFP8 candidate: preserve norm lanes, fixed FP32 tree, BF16 epilogue.
//! The checkpoint's Linear replacement and effective load policy stay unqualified.
use crate::{bf16,norm_boundary::WeightType,text_reference};

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub struct Shape {rows:usize,cols:usize,input:WeightType}
impl Shape {
    pub fn new(rows:usize,cols:usize,input:WeightType)->Result<Self,String> {
        if rows==0 || cols==0 || !cols.is_multiple_of(32) || rows>u32::MAX as usize || cols>u32::MAX as usize
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
    /// `reference` on the tensor-core word (`next_params::MX_TC`; CPU twin `tc_twin_mx`, contract
    /// `docs/tc-dense-contract.md`, section MXFP8): the accumulator of every row and its one RN-even BF16 epilogue. The
    /// same refusals as `reference` (extents, a non-finite input, a NaN scale or weight code, a non-finite word).
    /// The word encodes each 32-column activation group to E4M3 codes and one scale; for the groups the activation
    /// quantizer wrote (every dynamic projection's input) that is a second spelling of the same values.
    pub fn reference_tc(self,weights:&[u8],scales:&[u8],input:&[f32])->Result<(Vec<f32>,Vec<f32>),String> {
        self.check_input(input)?;
        if weights.len()!=self.weight_bytes() || scales.len()!=self.scale_bytes() {
            return Err("MXFP8 reference requires complete selected rows and scale groups".into());
        }
        if scales.contains(&255) || weights.iter().any(|w|w&127==127) {return Err("nonfinite MXFP8 reference operand".into());}
        let encoded=crate::tc_twin_mx::encode_row(input);
        let mut accumulator=Vec::with_capacity(self.rows);let mut output=Vec::with_capacity(self.rows);
        for (w,s) in weights.chunks_exact(self.cols).zip(scales.chunks_exact(self.cols/32)) {
            let v=f32::from_bits(crate::tc_twin_mx::word_encoded(w,s,&encoded));
            if !v.is_finite() {return Err("nonfinite MXFP8 reference output".into());}
            let rounded=bf16::rounded(v);
            if !rounded.is_finite() {return Err("MXFP8 epilogue is not finite BF16".into());}
            accumulator.push(v);output.push(rounded);
        }Ok((accumulator,output))
    }
    /// `reference` or `reference_tc`, by the word function of the binding (`next_params::MX_SCALAR` or `MX_TC`).
    pub fn reference_mode(self,mode:u32,weights:&[u8],scales:&[u8],input:&[f32])->Result<(Vec<f32>,Vec<f32>),String> {
        match mode {
            crate::next_params::MX_SCALAR=>self.reference(weights,scales,input),
            crate::next_params::MX_TC=>self.reference_tc(weights,scales,input),
            _=>Err("unknown MXFP8 word function".into()),
        }
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
        let mut input=vec![0.0;512];input[0]=16_777_216.0;input[8]= -16_777_216.0;input[16]=1.0;input[256]=1.0;
        let s=Shape::new(3,512,WeightType::F32).unwrap();
        let all=s.reference(&vec![0x38;1536],&[127;48],&input).unwrap();assert_eq!(all.0,[0.0;3]);
        let row=Shape::new(1,512,WeightType::F32).unwrap().reference(&vec![0x38;512],&[127;16],&input).unwrap();
        for _ in 0..3 {assert_eq!(row.0[0].to_bits(),all.0[2].to_bits());}
        input.fill(0.0);input[511]=2.0;
        assert_eq!(s.reference(&vec![0x38;1536],&[127;48],&input).unwrap().1,[2.0;3]);
    }
    /// The tensor-core word on a quantized row is the twin's word, is the same for a row alone and among rows, and differs from
    /// the scalar word in its sum order (a cancelling row that the scalar tree and the four ranges split differently).
    #[test]
    fn the_tensor_core_reference_is_the_twin_word_for_each_row_alone() {
        use crate::tc_twin_mx as twin;
        let (rows,cols)=(5usize,512usize);let mut r=twin::Rng(77);
        let weights:Vec<u8>=(0..rows*cols).map(|i|twin::weight_code(&mut r,i,i/cols)).collect();
        let scales:Vec<u8>=(0..rows*cols/32).map(|i|(120+i%9) as u8).collect();
        let mut input=Vec::new();for group in 0..cols/32 {input.extend(twin::activation_group(&mut r,group*32,group%7));}
        // The projection's input contract: finite values (the activation classes of the twin include non-finite ones).
        for v in &mut input {if !v.is_finite() {*v=1.5;}}
        let s=Shape::new(rows,cols,WeightType::F32).unwrap();
        let (acc,out)=s.reference_tc(&weights,&scales,&input).unwrap();
        for row in 0..rows {
            let (w,sc)=(&weights[row*cols..(row+1)*cols],&scales[row*cols/32..(row+1)*cols/32]);
            assert_eq!(acc[row].to_bits(),twin::word(w,sc,&input),"row {row}");
            assert_eq!(out[row].to_bits(),bf16::rounded(acc[row]).to_bits());
            let alone=Shape::new(1,cols,WeightType::F32).unwrap().reference_tc(w,sc,&input).unwrap();
            assert_eq!(alone.0[0].to_bits(),acc[row].to_bits());
        }
        assert_eq!(s.reference_mode(crate::next_params::MX_TC,&weights,&scales,&input).unwrap().0,acc);
        assert_eq!(s.reference_mode(crate::next_params::MX_SCALAR,&weights,&scales,&input).unwrap().0,s.reference(&weights,&scales,&input).unwrap().0);
        assert!(s.reference_mode(2,&weights,&scales,&input).is_err());
        // Refusals, as the scalar reference.
        let one=Shape::new(1,32,WeightType::F32).unwrap();
        assert!(one.reference_tc(&[0x38;31],&[127],&[1.0;32]).is_err() && one.reference_tc(&[0x38;32],&[],&[1.0;32]).is_err());
        assert!(one.reference_tc(&[0x7f;32],&[127],&[1.0;32]).is_err() && one.reference_tc(&[0x38;32],&[255],&[1.0;32]).is_err());
        assert!(one.reference_tc(&[0x38;32],&[127],&[f32::NAN;32]).is_err());
        // 32 ones times 0.5: one step of 16.0, scales 2^0.
        assert_eq!(one.reference_tc(&[0x38;32],&[127],&[0.5;32]).unwrap().0,[16.0]);
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
