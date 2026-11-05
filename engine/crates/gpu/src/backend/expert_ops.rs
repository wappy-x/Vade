//! Checked borrowed expert operands. Ordinary calls complete; an owned queue retains dropped allocations until its drain.
use super::*;
use crate::expert_ops::{Shape,Arguments,GroupArguments,GroupEntry,LaunchArguments,Op};
impl<'a> Backend<'a> {
    pub fn expert_operation(&self,shape:Shape,input:[Option<&DeviceVector<'_>>;4],weight:Option<&DeviceVector<'_>>,scales:Option<&DeviceVector<'_>>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let mut arguments=self.expert_arguments(shape,input,weight,scales)?;
        let (_,_,_,units,words)=shape.lengths().ok_or("invalid expert operation shape")?;
        let mut output=self.vector(units*words)?;self.zero_vector(&mut output)?;arguments.output=output.device_address();
        let rows=if shape.op==Op::NvProject {8} else {32};
        self.launch_experts(LaunchArguments {single:arguments,batch:0,count:1},(units as u32).div_ceil(rows))?;Ok(output)
    }
    pub fn expert_projection_batch(&self,jobs:&[(Shape,&DeviceVector<'_>,&DeviceVector<'_>,&DeviceVector<'_>)])->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        let first=jobs.first().ok_or("expert projection batch is empty")?.0;
        if !matches!(first.op,Op::NvProject|Op::Fp8Project|Op::Fp8Serial|Op::PlainSerial) || jobs.iter().any(|(shape,_,_,_)|shape.op!=first.op || shape.rows!=first.rows || shape.cols!=first.cols) {
            return Err("expert projection batch requires equal complete projection rows".into());
        }
        let count=u32::try_from(jobs.len()).map_err(|_|"expert projection batch exceeds ABI")?;
        let rows=if matches!(first.op,Op::Fp8Serial|Op::PlainSerial) {32} else {8};
        let grid=first.rows.div_ceil(rows).checked_mul(count).ok_or("expert projection batch grid overflow")?;
        let words=(first.rows as usize).checked_mul(jobs.len()).filter(|&words|words<=u32::MAX as usize && words<=isize::MAX as usize/4).ok_or("expert projection batch output overflow")?;
        let mut descriptors=jobs.iter().map(|(shape,input,weight,scales)|self.expert_arguments(*shape,[Some(input),None,None,None],Some(weight),if shape.op==Op::PlainSerial {None} else {Some(scales)})).collect::<Result<Vec<_>,_>>()?;
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
        // through the real drain, including a batch's local descriptor table.
        // Ordinary calls complete before return.
        let launched=unsafe {self.launch_flash("expert_operation",grid,32,&mut [parameter(&mut arguments)])};
        let completed=self.resident_completion();launched.and(completed)
    }
    /// One launch for one layer's experts. Each part is (shape, raw weight and
    /// scale owners, first input row). Parts own contiguous output units.
    pub fn expert_group(&self,parts:&[(Shape,Option<(&DeviceVector<'_>,&DeviceVector<'_>)>,usize)],input:&DeviceVector<'_>,stride:usize,lut:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;self.require_arm_kernels(&["expert_group"])?;
        self.check_vector(lut,crate::expert_ops::LUT_WORDS)?;if lut.is_segmented() {return Err("expert decode table must be contiguous".into());}
        if parts.is_empty() || input.is_segmented() || stride==0 {return Err("expert group is empty or its input is segmented".into());}
        self.check_vector(input,input.len())?;
        let mut table=Vec::with_capacity(parts.len());let mut units=0usize;let mut word_size=None;let mut end_row=0usize;
        for &(shape,raw,first_row) in parts {
            let (lengths,wb,sb,count,words)=shape.lengths().ok_or("invalid expert group shape")?;
            if !matches!(shape.op,crate::expert_ops::Op::QuantizeRows|crate::expert_ops::Op::NvRows|crate::expert_ops::Op::Fp8Rows) {return Err("expert group accepts token rows only".into());}
            if *word_size.get_or_insert(words)!=words || lengths[1..].iter().any(|&n|n!=0) {return Err("expert group word size differs".into());}
            let (weight,scales)=match raw {
                Some((w,s))=>{self.check_vector(w,wb.div_ceil(4))?;self.check_vector(s,sb.div_ceil(4))?;if w.is_segmented() || s.is_segmented() {return Err("expert bytes must be contiguous".into());}(w.device_address(),s.device_address())},
                None if wb==0 && sb==0=>(4,4),_=>return Err("expert group raw byte count differs".into()),
            };
            let tokens=shape.tokens as usize;
            if first_row.checked_mul(stride).and_then(|n|n.checked_add(lengths[0])).is_none_or(|n|n>input.len()) || lengths[0]!=tokens*stride {return Err("expert group input span differs".into());}
            end_row=end_row.max(first_row+tokens);
            table.push(GroupEntry {shape,weight,scales,first:u32::try_from(units).map_err(|_|"expert group units exceed ABI")?,first_row:first_row as u32,
                input_words:lengths[0] as u32,weight_bytes:wb as u32,scale_bytes:sb as u32,pad:0});
            units=units.checked_add(count).ok_or("expert group units overflow")?;
        }
        let words=word_size.ok_or("expert group is empty")?;let total=units.checked_mul(words).ok_or("expert group output overflow")?;
        if u32::try_from(total).is_err() || end_row*stride>input.len() {return Err("expert group output exceeds ABI".into());}
        let bytes=table.iter().flat_map(|e|{let mut raw=Vec::with_capacity(std::mem::size_of::<GroupEntry>());
            raw.extend_from_slice(&(e.shape.op as u32).to_le_bytes());for v in [e.shape.rows,e.shape.cols,e.shape.block_rows,e.shape.block_cols,e.shape.tokens] {raw.extend_from_slice(&v.to_le_bytes());}
            for v in [e.shape.global,e.shape.activation,e.shape.bound] {raw.extend_from_slice(&v.to_le_bytes());}raw.extend_from_slice(&0u32.to_le_bytes());
            raw.extend_from_slice(&e.weight.to_le_bytes());raw.extend_from_slice(&e.scales.to_le_bytes());
            for v in [e.first,e.first_row,e.input_words,e.weight_bytes,e.scale_bytes,e.pad] {raw.extend_from_slice(&v.to_le_bytes());}raw}).collect::<Vec<_>>();
        if bytes.len()!=table.len()*std::mem::size_of::<GroupEntry>() {return Err("expert group table layout differs".into());}
        let mut owner=self.upload_vector(&vec![0.0;bytes.len().div_ceil(4)])?;self.write_vector_bytes(&mut owner,&bytes)?;
        let mut output=self.vector(total)?;self.zero_vector(&mut output)?;
        let mut args=GroupArguments {table:owner.device_address(),input:input.device_address(),output:output.device_address(),lut:lut.device_address(),count:table.len() as u32,units:units as u32,words:words as u32,stride:stride as u32};
        // SAFETY: Each part's shape, raw owners, context, extents and input span
        // are checked above. The table bytes follow the repr(C) layout, checked
        // by size. Buffer::drop drains the stream before any owner is freed. The
        // private output is zeroed earlier on this ordered stream and disjoint.
        let launched=unsafe {self.launch_flash("expert_group",(units as u32).div_ceil(32),32,&mut [parameter(&mut args)])};
        launched?;drop(owner);Ok(output)
    }
}
