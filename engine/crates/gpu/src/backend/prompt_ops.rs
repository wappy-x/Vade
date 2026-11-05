//! Layer-major prompt rows on the ordered stream. Readback waits for completion.
use super::*;
use crate::prompt_ops::{Shape,Arguments};
impl<'a> Backend<'a> {
    /// Enqueue one prompt operation. Later consumers on this stream follow it.
    /// Dropping an owner drains the stream first, so borrowed inputs stay live.
    pub fn prompt_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;5],weight:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let (lengths,weights,units,words)=shape.lengths().ok_or("invalid prompt row shape")?;
        let mut addresses=[4u64;5];
        for (i,(vector,count)) in input.into_iter().zip(lengths).enumerate() {
            match vector {Some(v) if count>0=>{self.check_vector_prefix(v,count)?;
                if v.is_segmented() {return Err("prompt row input must be contiguous".into());}addresses[i]=v.device_address();},
                _ if count==0=>{},_=>return Err("prompt row input count differs".into())}
        }
        let address=match weight {Some(w) if weights>0=>{self.check_vector(w,weights)?;
            if w.is_segmented() {return Err("prompt row weight must be contiguous".into());}w.device_address()},
            _ if weights==0=>4,_=>return Err("prompt row weight count differs".into())};
        self.require_arm_kernels(&["prompt_rows"])?;
        let total=units.checked_mul(words).ok_or("prompt row output overflow")?;
        let mut output=self.vector(total.max(1))?;self.zero_vector(&mut output)?;
        if units==0 {return Ok(output);}
        let mut args=Arguments {input:addresses,lengths:lengths.map(|n|n as u64),weight:address,weights:weights as u64,
            output:output.device_address(),units:units as u64,words:words as u64,shape};
        // SAFETY: Shape bounds every indexed read and output unit, and the kernel
        // receives those exact extents. check_vector_prefix checks owners and
        // contexts. Inputs are read-only. The private output is zeroed earlier on
        // this ordered stream, disjoint and contiguous. Buffer::drop drains the
        // stream, so all owners outlive the launch. The argument is copied at launch.
        unsafe {self.launch_flash("prompt_rows",u32::try_from(units.div_ceil(128)).map_err(|_|"prompt row grid overflow")?,128,&mut [parameter(&mut args)])}?;
        Ok(output)
    }
}
