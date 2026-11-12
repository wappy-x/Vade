use super::*;
use crate::{device_route as route,expert_ops::{Arguments,LaunchArguments,Op,Shape}};
pub struct DeviceRouteTable<'a> {table:DeviceVector<'a>,owners:Vec<DeviceVector<'a>>,shapes:Vec<Shape>,quantized:DeviceVector<'a>,quantizers:DeviceVector<'a>,projects:DeviceVector<'a>,first:Arguments,
    experts:u32,picks:u32,phases:u32,down:bool,shared:bool,raw:u32}
impl DeviceRouteTable<'_> {
    /// The pair split's form of a down route: the projector writes `quarters` (4, 2 or 1) RAW quarter accumulators a
    /// row (`nv_cols_q4` / `q2` / `q1`), `[job][row][quarter]`, in place of one finished word. The table then holds
    /// the owner's column range of every second matrix, as column tiles.
    pub fn with_quarters(mut self,quarters:usize)->Result<Self,String> {
        let first=self.first.shape;
        if !self.down || self.phases!=1 || first.role!=crate::expert_ops::NV_SECOND || crate::pair_split::quarters_entry(quarters).is_none()
            || !crate::expert_ops::nv_tiled(&first) || !(first.cols as usize/64).is_multiple_of(quarters) {
            return Err("a quarter route needs a down table of column-tile second matrices of whole quarters".into());
        }
        self.raw=quarters as u32;Ok(self)
    }
    /// Words a job writes for one output row: 1, or the quarters of a quarter route.
    fn row_words(&self)->usize {self.raw.max(1) as usize}
}
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
        Ok(Some(DeviceRouteTable {table,owners,shapes,quantized,quantizers,projects,first,experts,picks,phases,down,shared,raw:0}))
    }
    pub fn device_route_projection(&self,table:&DeviceRouteTable<'a>,ids:&DeviceVector<'_>,input:&DeviceVector<'_>)->Result<DeviceVector<'a>,String> {
        self.require_resident_stream()?;
        self.check_vector(&table.table,table.experts as usize*table.phases as usize*24)?;
        self.check_vector(ids,table.picks as usize)?;
        self.check_vector(input,table.first.shape.cols as usize*if table.down {table.picks as usize} else {1})?;
        if ids.is_segmented() || input.is_segmented() || table.owners.len()!=table.experts as usize*table.phases as usize*2 {return Err("device route owner layout differs".into());}
        let count=table.picks*table.phases;let quantized=&table.quantized;let quantizers=&table.quantizers;let projects=&table.projects;
        // No fill: the region is fully written by nv_tile before any read (job * rows offsets, one raw store a word; a bad route traps).
        // A quarter route writes rows * quarters words a job, every one of them (rows are whole tiles).
        let mut output=self.vector(table.first.shape.rows as usize*count as usize*table.row_words())?;self.skip_fill(&mut output)?;
        let mut build=route::Build {table:table.table.device_address(),ids:ids.device_address(),input:input.device_address(),quantized:quantized.device_address(),
            quantizers:quantizers.device_address(),projects:projects.device_address(),output:output.device_address(),experts:table.experts,picks:table.picks,phases:table.phases,
            down:u32::from(table.down),shared:u32::from(table.shared),raw:table.raw};
        // The projection kernel follows the storage layout of the table's matrices (all equal, checked at admission).
        let tiled=crate::expert_ops::nv_tiled(&table.first.shape);
        let projector=if table.raw!=0 {crate::pair_split::quarters_entry(table.raw as usize).ok_or("quarter route entry differs")?} else if tiled {"nv_cols"} else {"nv_tile"};
        self.require_arm_kernels(&["device_route_build",projector])?;
        self.prefetch_route_window(build,table.first.shape.lengths().ok_or("route shadow shape differs")?.1);
        // ROUTE-PREPARE: one launch writes the descriptors (block 0) and quantizes every job's input (the other
        // blocks, ceil(cols/128) a job; each computes its own descriptor from the route table).
        let grid=table.first.shape.cols.div_ceil(128).checked_mul(if table.shared {1} else {count}).and_then(|blocks|blocks.checked_add(1)).ok_or("device route prepare grid overflow")?;
        self.launch_checked_expert_payload("device_route_build",grid,1,crate::expert_ops::QUANTIZE_LANES_THREADS,0,&mut build)?;
        let mut launch=LaunchArguments {single:table.first,batch:projects.device_address(),count};
        let (blocks,threads,shared)=if tiled {
            if table.first.input[0]&3!=0 || quantized.device_address()&3!=0 {return Err("device route column input is not word aligned".into());}
            (crate::expert_ops::nv_cols_blocks(table.first.shape.rows,count).ok_or("device route column grid overflow")?,crate::expert_ops::NV_COLS_THREADS,crate::expert_ops::nv_cols_shared(table.first.shape.cols))
        } else {(crate::nv_stream::tile_blocks(table.first.shape.rows as usize,count as usize).ok_or("device route tile grid overflow")?,crate::nv_stream::TILE_THREADS,crate::nv_stream::TILE_SHARED)};
        self.launch_checked_expert_payload(projector,blocks,1,threads,shared,&mut launch)?;
        self.resident_completion()?;let mut stats=self.stats.borrow_mut();stats.projection_dispatches[6]+=1;
        stats.projection_dispatches[if table.shared {7} else {8}]+=u128::from(if table.shared {1} else {table.phases});
        route::record(table.down);Ok(output)
    }
    pub fn check_device_route_descriptors(&self,table:&DeviceRouteTable<'a>,ids:&[usize],input:&DeviceVector<'a>,output:&DeviceVector<'_>)->Result<usize,String> {
        if ids.len()!=table.picks as usize || ids.iter().enumerate().any(|(slot,&id)|id>=table.experts as usize || ids[..slot].contains(&id)) {return Err("route descriptor check requires distinct bounded IDs".into());}
        let count=table.picks as usize*table.phases as usize;let cols=table.first.shape.cols as usize;let quant_words=cols/16*17;
        self.check_vector(output,count*table.first.shape.rows as usize*table.row_words())?;
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
            arguments.output=output.device_address()+(job*shape.rows as usize*table.row_words()*4) as u64;projects.extend(arguments.descriptor_words());
        }
        let (actual_quantizers,actual_projects)=self.read_resident_pair(&table.quantizers,&table.projects)?;
        for (name,actual,expected) in [("Quantize",actual_quantizers,quantizers),("NvProject",actual_projects,projects)] {
            if actual.len()!=expected.len() || actual.iter().map(|value|value.to_bits()).ne(expected) {return Err(format!("device route {name} descriptors differ from original host builder"));}
        }
        Ok((table.quantizers.len()+table.projects.len())*4)
    }
}
