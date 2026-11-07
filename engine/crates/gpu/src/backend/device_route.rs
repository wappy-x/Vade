use super::*;
use crate::{device_route as route,expert_ops::{Arguments,LaunchArguments,Op,Shape}};
pub struct DeviceRouteTable<'a> {table:DeviceVector<'a>,owners:Vec<DeviceVector<'a>>,shapes:Vec<Shape>,quantized:DeviceVector<'a>,quantizers:DeviceVector<'a>,projects:DeviceVector<'a>,first:Arguments,
    experts:u32,picks:u32,phases:u32,down:bool,shared:bool}
impl<'a> Backend<'a> {
    pub fn device_route_table(&self,parts:&[(Shape,&DeviceVector<'a>,&DeviceVector<'a>)],experts:u32,picks:u32,phases:u32,down:bool)->Result<Option<DeviceRouteTable<'a>>,String> {
        let shapes=parts.iter().map(|part|part.0).collect::<Vec<_>>();
        let addresses=parts.iter().map(|(_,weight,scales)|(weight.device_address(),scales.device_address())).collect::<Vec<_>>();
        if down && phases!=1 || !route::admitted(&shapes,experts,picks,phases,&addresses) {return Ok(None);}
        self.require_resident_stream()?;let mut owners=Vec::with_capacity(parts.len()*2);let mut entries=Vec::with_capacity(parts.len());
        for &(shape,weight,scales) in parts {
            let (_,weight_bytes,scale_bytes,_,_)=shape.lengths().ok_or("device route shape differs")?;
            self.check_vector(weight,weight_bytes.div_ceil(4))?;self.check_vector(scales,scale_bytes.div_ceil(4))?;
            if weight.is_segmented() || scales.is_segmented() {return Err("device route raw owner is segmented".into());}
            owners.push(weight.resident_range(0..weight.len())?);owners.push(scales.resident_range(0..scales.len())?);
            entries.push(Arguments {input:[4;4],weight:weight.device_address(),scales:scales.device_address(),output:4,shape});
        }
        let first=entries[0];let shared=!down && shapes.iter().all(|shape|shape.activation.to_bits()==first.shape.activation.to_bits());
        let words=entries.iter().flat_map(|entry|entry.descriptor_words()).map(f32::from_bits).collect::<Vec<_>>();
        let table=self.upload_vector(&words)?;
        let count=picks*phases;let quant_count=if shared {1} else {count};
        let mut quantized=self.vector(first.shape.cols as usize/16*17*quant_count as usize)?;self.zero_vector(&mut quantized)?;let quantized=quantized.into_resident_shared()?;
        let mut quantizers=self.vector(24*quant_count as usize)?;self.zero_vector(&mut quantizers)?;
        let mut projects=self.vector(24*count as usize)?;self.zero_vector(&mut projects)?;
        Ok(Some(DeviceRouteTable {table,owners,shapes,quantized,quantizers,projects,first,experts,picks,phases,down,shared}))
    }
    pub fn device_route_projection(&self,table:&DeviceRouteTable<'a>,ids:&DeviceVector<'_>,input:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        self.check_vector(&table.table,table.experts as usize*table.phases as usize*24)?;
        self.check_vector(ids,table.picks as usize)?;
        self.check_vector(input,table.first.shape.cols as usize*if table.down {table.picks as usize} else {1})?;
        if ids.is_segmented() || input.is_segmented() || table.owners.len()!=table.experts as usize*table.phases as usize*2 {return Err("device route owner layout differs".into());}
        let count=table.picks*table.phases;let quantized=&table.quantized;let quantizers=&table.quantizers;let projects=&table.projects;
        let mut output=self.vector(table.first.shape.rows as usize*count as usize)?;self.zero_vector(&mut output)?;
        let mut build=route::Build {table:table.table.device_address(),ids:ids.device_address(),input:input.device_address(),quantized:quantized.device_address(),
            quantizers:quantizers.device_address(),projects:projects.device_address(),output:output.device_address(),experts:table.experts,picks:table.picks,phases:table.phases,
            down:u32::from(table.down),shared:u32::from(table.shared)};
        self.require_arm_kernels(&["device_route_build","expert_operation","nv_tile"])?;
        self.launch_checked_expert_payload("device_route_build",1,1,32,0,&mut build)?;
        let quant_shape=Shape {op:Op::Quantize,..table.first.shape};
        let quant_first=Arguments {input:[input.device_address(),4,4,4],weight:4,scales:4,output:quantized.device_address(),shape:quant_shape};
        for phase in 0..if table.shared {1} else {table.phases} {
            let jobs=if table.shared {1} else {table.picks};
            let mut launch=LaunchArguments {single:quant_first,batch:quantizers.device_address()+u64::from(phase*table.picks)*96,count:jobs};
            let grid=table.first.shape.cols.div_ceil(128).checked_mul(jobs).ok_or("device route quantize grid overflow")?;
            self.launch_checked_expert_payload("expert_operation",grid,1,crate::expert_ops::QUANTIZE_LANES_THREADS,0,&mut launch)?;
        }
        let mut launch=LaunchArguments {single:table.first,batch:projects.device_address(),count};
        let blocks=crate::nv_stream::tile_blocks(table.first.shape.rows as usize,count as usize).ok_or("device route tile grid overflow")?;
        self.launch_checked_expert_payload("nv_tile",blocks,1,crate::nv_stream::TILE_THREADS,crate::nv_stream::TILE_SHARED,&mut launch)?;
        self.resident_completion()?;let mut stats=self.stats.borrow_mut();stats.projection_dispatches[6]+=1;
        stats.projection_dispatches[if table.shared {7} else {8}]+=u128::from(if table.shared {1} else {table.phases});
        route::record(table.down);Ok(output)
    }
    pub fn check_device_route_descriptors(&self,table:&DeviceRouteTable<'a>,ids:&[usize],input:&DeviceVector<'a>,output:&DeviceVector<'_>)->Result<usize,String> {
        if ids.len()!=table.picks as usize || ids.iter().enumerate().any(|(slot,&id)|id>=table.experts as usize || ids[..slot].contains(&id)) {return Err("route descriptor check requires distinct bounded IDs".into());}
        let count=table.picks as usize*table.phases as usize;let cols=table.first.shape.cols as usize;let quant_words=cols/16*17;
        self.check_vector(output,count*table.first.shape.rows as usize)?;
        let mut quantizers=Vec::new();let mut projects=Vec::new();
        for job in 0..count {
            let slot=job%table.picks as usize;let phase=job/table.picks as usize;let entry=phase*table.experts as usize+ids[slot];let shape=table.shapes[entry];
            let source=if table.down {input.resident_range(slot*cols..(slot+1)*cols)?} else {input.resident_range(0..cols)?};
            let quant=if table.shared {0} else {job};let quantized=table.quantized.resident_range(quant*quant_words..(quant+1)*quant_words)?;
            if !table.shared || job==0 {
                let mut arguments=self.expert_arguments(Shape {op:Op::Quantize,..shape},[Some(&source),None,None,None],None,None)?;
                arguments.output=quantized.device_address();quantizers.extend(arguments.descriptor_words());
            }
            let mut arguments=self.expert_arguments(shape,[Some(&quantized),None,None,None],Some(&table.owners[entry*2]),Some(&table.owners[entry*2+1]))?;
            arguments.output=output.device_address()+(job*shape.rows as usize*4) as u64;projects.extend(arguments.descriptor_words());
        }
        let (actual_quantizers,actual_projects)=self.read_resident_pair(&table.quantizers,&table.projects)?;
        for (name,actual,expected) in [("Quantize",actual_quantizers,quantizers),("NvProject",actual_projects,projects)] {
            if actual.len()!=expected.len() || actual.iter().map(|value|value.to_bits()).ne(expected.into_iter()) {return Err(format!("device route {name} descriptors differ from original host builder"));}
        }
        Ok((table.quantizers.len()+table.projects.len())*4)
    }
}
