//! Checked borrowed expert operands. Ordinary calls complete; an owned queue retains dropped allocations until its drain.
use super::*;
use crate::expert_ops::{Shape,Arguments,LaunchArguments,Op};
impl<'a> Backend<'a> {
    pub fn expert_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let mut arguments=self.expert_arguments(shape,input,weight,scales)?;
        let (_,_,_,units,words)=shape.lengths().ok_or("invalid expert operation shape")?;
        let output=self.upload_vector(&vec![0.0;units*words])?;arguments.output=output.device_address();
        let rows=if shape.op==Op::NvProject {8} else {32};
        self.launch_experts(LaunchArguments {single:arguments,batch:0,count:1},(units as u32).div_ceil(rows))?;Ok(output)
    }
    pub fn expert_projection_batch(&self,jobs:&[(Shape,&DeviceVector<'_>,&DeviceVector<'_>,&DeviceVector<'_>)])->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("expert projection batch is empty")?.0;
        if jobs.iter().any(|(shape,_,_,_)|shape.op!=Op::NvProject || shape.rows!=first.rows || shape.cols!=first.cols) {
            return Err("expert projection batch requires equal complete NVFP4 rows".into());
        }
        let count=u32::try_from(jobs.len()).map_err(|_|"expert projection batch exceeds ABI")?;
        let grid=first.rows.div_ceil(8).checked_mul(count).ok_or("expert projection batch grid overflow")?;
        let words=(first.rows as usize).checked_mul(jobs.len()).filter(|&words|words<=u32::MAX as usize && words<=isize::MAX as usize/4).ok_or("expert projection batch output overflow")?;
        let mut descriptors=jobs.iter().map(|(shape,input,weight,scales)|self.expert_arguments(*shape,[Some(input),None,None,None],Some(weight),Some(scales))).collect::<Result<Vec<_>,_>>()?;
        let output=self.upload_vector(&vec![0.0;words])?;
        for (index,arguments) in descriptors.iter_mut().enumerate() {arguments.output=output.device_address()+(index*first.rows as usize*4) as u64;}
        let table=self.upload_vector(&descriptors.iter().flat_map(|arguments|arguments.descriptor_words().map(f32::from_bits)).collect::<Vec<_>>())?;
        self.launch_experts(LaunchArguments {single:descriptors[0],batch:table.device_address(),count},grid)?;Ok(output)
    }
    fn expert_arguments(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<Arguments,String> {
        let (lengths,wb,sb,_,_)=shape.lengths().ok_or("invalid expert operation shape")?;
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
        Ok(Arguments {input:addresses,weight:raw[0],scales:raw[1],output:0,shape})
    }
    fn launch_experts(&self,mut arguments:LaunchArguments,grid:u32)->Result<(),String> {
        self.require_arm_kernels(&["expert_operation"])?;
        // SAFETY: Shape checks all slice extents and disjoint output units.
        // check_vector checks exact lengths and context. All buffers are
        // contiguous. Raw owners fund each byte. Output is initialized private
        // storage. By-value arguments are copied at submission. Equal row extents and the
        // checked grid give each batch job a disjoint initialized output span. A single
        // launch keeps lane A's queued completion: queued owner drops defer allocations
        // through the real drain. A batch keeps lane S's full completion, because its
        // descriptor table is a local owner; ordinary calls complete before return.
        let launched=unsafe {self.launch_flash("expert_operation",grid,32,&mut [parameter(&mut arguments)])};
        let completed=if arguments.batch==0 {self.resident_completion()} else {self.synchronize()};launched.and(completed)
    }
}
