//! Plain FP32 projection with the logical-f32-v1 product and add order.
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error { UnsupportedStorage, Shape, Layout, Extent, Address, Overlap, Context, Device(String) }
impl fmt::Display for Error {
    fn fmt(&self,f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::UnsupportedStorage=>"ordered projection requires plain FP32 weights",
            Self::Shape=>"ordered projection dimensions exceed the nonempty kernel ABI",
            Self::Layout=>"ordered projection requires complete contiguous rows",
            Self::Extent=>"ordered projection operand extent differs",
            Self::Address=>"ordered projection address is unaligned or overflows",
            Self::Overlap=>"ordered projection output overlaps an input",
            Self::Context=>"ordered projection operands belong to another context",
            Self::Device(why)=>why,
        })
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape { rows: usize, cols: usize, weight_bytes: usize }
impl Shape {
    pub fn new(rows: usize,cols: usize) -> Result<Self,Error> {
        if rows==0 || cols==0 || rows>u32::MAX as usize || cols>u32::MAX as usize { return Err(Error::Shape); }
        let weight_bytes=rows.checked_mul(cols).and_then(|n| n.checked_mul(4))
            .filter(|&n| n<=isize::MAX as usize).ok_or(Error::Shape)?;
        Ok(Self {rows,cols,weight_bytes})
    }
    pub fn rows(self) -> usize {self.rows}
    pub fn cols(self) -> usize {self.cols}
    pub fn weight_bytes(self) -> usize {self.weight_bytes}
    /// The loaded matrix must retain FP32 bytes and complete input features.
    pub fn layout(self,plain_f32: bool,stride: usize,offsets: [usize;2],bytes: usize) -> Result<(),Error> {
        if !plain_f32 {return Err(Error::UnsupportedStorage);}
        if stride!=self.cols || offsets!=[0,0] {return Err(Error::Layout);}
        if bytes!=self.weight_bytes {return Err(Error::Extent);}
        Ok(())
    }
    /// Each span is (address, mapped bytes). Input and output lengths are exact.
    pub fn operands(self,weights: (u64,usize),input: (u64,usize),output: (u64,usize)) -> Result<(),Error> {
        let expected=[self.weight_bytes,self.cols*4,self.rows*4];
        let spans=[weights,input,output];let mut ends=[0;3];
        for (i,&(address,bytes)) in spans.iter().enumerate() {
            if bytes!=expected[i] {return Err(Error::Extent);}
            if address==0 || address%4!=0 {return Err(Error::Address);}
            ends[i]=address.checked_add(u64::try_from(bytes).map_err(|_| Error::Address)?).ok_or(Error::Address)?;
        }
        for i in 0..2 {if output.0<ends[i] && spans[i].0<ends[2] {return Err(Error::Overlap);}}
        Ok(())
    }
    /// Separate FP32 products and ascending feature adds. No activation cast.
    pub fn reference(self,weights: &[f32],input: &[f32]) -> Result<Vec<f32>,Error> {
        if weights.len()!=self.weight_bytes/4 || input.len()!=self.cols {return Err(Error::Extent);}
        Ok(weights.chunks_exact(self.cols).map(|row| {
            let mut sum=0.0f32;for j in 0..self.cols {sum+=row[j]*input[j];} sum
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn rejects_shape_storage_layout_and_extents() {
        for (r,c) in [(0,1),(1,0),(usize::MAX,2),(u32::MAX as usize,u32::MAX as usize)] {
            assert_eq!(Shape::new(r,c),Err(Error::Shape));
        }
        let s=Shape::new(3,5).unwrap();assert_eq!(s.weight_bytes(),60);
        assert_eq!(s.layout(false,5,[0,0],60),Err(Error::UnsupportedStorage));
        assert_eq!(s.layout(true,6,[0,0],60),Err(Error::Layout));
        assert_eq!(s.layout(true,5,[1,0],60),Err(Error::Layout));
        assert_eq!(s.layout(true,5,[0,1],60),Err(Error::Layout));
        assert_eq!(s.layout(true,5,[0,0],59),Err(Error::Extent));
        assert_eq!(s.layout(true,5,[0,0],60),Ok(()));
        assert_eq!(s.reference(&[1.0;14],&[1.0;5]),Err(Error::Extent));
        assert_eq!(s.reference(&[1.0;15],&[1.0;4]),Err(Error::Extent));
    }
    #[test] fn checks_alignment_overflow_aliases_and_exact_vector_bytes() {
        let s=Shape::new(3,5).unwrap();
        assert_eq!(s.operands((4,60),(64,20),(84,12)),Ok(()));
        assert_eq!(s.operands((4,60),(64,20),(80,12)),Err(Error::Overlap));
        assert_eq!(s.operands((4,60),(64,20),(8,12)),Err(Error::Overlap));
        assert_eq!(s.operands((4,60),(65,20),(88,12)),Err(Error::Address));
        assert_eq!(s.operands((u64::MAX-3,60),(64,20),(88,12)),Err(Error::Address));
        assert_eq!(s.operands((0,60),(64,20),(88,12)),Err(Error::Address));
        assert_eq!(s.operands((4,60),(64,24),(88,12)),Err(Error::Extent));
    }
    #[test] fn keeps_separate_products_and_ascending_adds() {
        let s=Shape::new(3,3).unwrap();
        let a=f32::from_bits(0x3f800001);let b=f32::from_bits(0x3f7ffffe);
        let w=[-1.0,a,0.0,16777216.0,1.0,-16777216.0,-16777216.0,16777216.0,1.0];
        let y=s.reference(&w,&[1.0,b,1.0]).unwrap();
        assert_eq!(y[0].to_bits(),0.0f32.to_bits());
        assert_ne!(a.mul_add(b,-1.0).to_bits(),y[0].to_bits());
        assert_eq!(s.reference(&w,&[1.0;3]).unwrap()[1..],[0.0,1.0]);
    }
}
