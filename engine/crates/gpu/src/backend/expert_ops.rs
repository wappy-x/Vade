//! Checked borrowed expert operands. Every launch completes before return.
use super::*;
use crate::expert_ops::{Shape,Arguments};
impl<'a> Backend<'a> {
    pub fn expert_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let (lengths,wb,sb,units,words)=shape.lengths().ok_or("invalid expert operation shape")?;
        let mut addresses=[4u64;4];
        for (i,(v,n)) in input.into_iter().zip(lengths).enumerate() {match v {
            Some(v) if n>0=>{self.check_vector(v,n)?;if v.is_segmented() {return Err("expert input must be contiguous".into());}addresses[i]=v.device_address();},
            None if n==0=>{},_=>return Err("expert input count differs".into()),
        }}
        let mut raw=[4u64;2];
        for (i,(v,n)) in [weight,scales].into_iter().zip([wb,sb]).enumerate() {match v {
            Some(v) if n>0=>{self.check_vector(v,n.div_ceil(4))?;if v.is_segmented() {return Err("expert bytes must be contiguous".into());}raw[i]=v.device_address();},
            None if n==0=>{},_=>return Err("expert raw byte count differs".into()),
        }}
        self.require_arm_kernels(&["expert_operation"])?;
        let output=self.upload_vector(&vec![0.0;units*words])?;
        let mut p=Arguments {input:addresses,weight:raw[0],scales:raw[1],output:output.device_address(),shape};
        let rows=if shape.op==crate::expert_ops::Op::NvProject {8} else {32};
        // SAFETY: Shape checks all slice extents and disjoint output units.
        // check_vector checks exact lengths and context. All buffers are
        // contiguous. Raw owners fund each byte. Output is initialized private
        // storage. All immutable inputs and output owners live through completion.
        let launched=unsafe {self.launch_flash("expert_operation",(units as u32).div_ceil(rows),32,&mut [parameter(&mut p)])};
        let completed=self.synchronize();launched.and(completed)?;Ok(output)
    }
}
