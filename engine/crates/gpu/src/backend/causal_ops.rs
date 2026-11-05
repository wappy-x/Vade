//! Owned single-token operation outputs. Borrowed inputs live through completion.
use super::*;
use crate::causal_ops::{Shape,Arguments};
impl<'a> Backend<'a> {
    pub fn causal_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let (lengths,weights,units,words)=shape.lengths().ok_or("invalid causal operation shape")?;
        let mut addresses=[4u64;4];
        for (i,(vector,count)) in input.into_iter().zip(lengths).enumerate() {
            match vector {Some(v) if count>0=>{self.check_vector(v,count)?;
                if v.is_segmented() {return Err("causal operation input must be contiguous".into());}addresses[i]=v.device_address();},
                None if count==0=>{},_=>return Err("causal operation input count differs".into())}
        }
        let address=match weight {Some(w) if weights>0=>{self.check_vector(w,weights)?;
            if w.is_segmented() {return Err("causal operation weight must be contiguous".into());}w.device_address()},
            None if weights==0=>4,_=>return Err("causal operation weight count differs".into())};
        self.require_arm_kernels(&["causal_operation","project_f32_ordered"])?;
        let mut output=self.vector(units*words)?;
        self.zero_vector(&mut output)?;self.resident_completion()?;
        let mut args=Arguments {input:addresses,weight:address,output:output.device_address(),shape};
        let (mut w,mut x,mut y)=(address,addresses[0],output.device_address());
        let (mut rows,mut cols)=(shape.rows,shape.cols);
        // SAFETY: Shape bounds every indexed read and output unit. check_vector
        // checks all nonempty owners, contexts, and exact lengths. Inputs are
        // read-only. The private output is initialized, disjoint, and contiguous. Empty spans
        // use aligned nonnull addresses. Owners live through stream completion.
        let launched=unsafe {if shape.op==crate::causal_ops::Op::Project {
            // This resident-only operation keeps separate RN FP32 products and
            // ascending adds. The complete row and weight extents are checked above.
            self.launch_flash("project_f32_ordered",(units as u32).div_ceil(128),128,
                &mut [parameter(&mut w),parameter(&mut x),parameter(&mut y),parameter(&mut rows),parameter(&mut cols)])
        } else {self.launch_flash("causal_operation",(units as u32).div_ceil(128),128,&mut [parameter(&mut args)])}};
        let completed=self.resident_completion();launched.and(completed)?;Ok(output)
    }
}
