//! Native expert products whose routing descriptors never leave the device.
use super::*;
use crate::{expert_ops::{Shape,Op},prompt_route::{NativeBuild,NativeSet,THREADS},tc_abi::{NvRows,NvQuantizeGather}};
pub struct PromptExpertTable<'a> {templates:DeviceVector<'a>,owners:Vec<DeviceVector<'a>>,shape:Shape,activations:Vec<u32>}
pub struct PromptRowScales<'a> {routes:Vec<Rc<PromptRoute<'a>>>,table:Rc<PromptExpertTable<'a>>,values:DeviceVector<'a>,starts:Vec<usize>,headers:DeviceVector<'a>}
/// `merge`: PAIRED PARTS, the `NvRows::merge` of the product launch (0, or the set count when one work interval an
/// expert covers the tokens of all its sets).
/// `rowmap`: DIRECT PLACE, one u64 output address a sorted row (the product stores in original pair order).
pub struct PromptNativeBatch<'a> {scales:Rc<PromptRowScales<'a>>,table:Rc<PromptExpertTable<'a>>,parts:DeviceVector<'a>,work:DeviceVector<'a>,headers:DeviceVector<'a>,outputs:Vec<Rc<DeviceVector<'a>>>,capacity:u32,merge:u32,rowmap:Option<DeviceVector<'a>>}
/// Retain immutable submission metadata without retaining consumed product outputs.
pub struct PromptNativeKeep<'a> {scales:Rc<PromptRowScales<'a>>,table:Rc<PromptExpertTable<'a>>,parts:DeviceVector<'a>,work:DeviceVector<'a>,headers:DeviceVector<'a>,rowmap:Option<DeviceVector<'a>>}
pub enum PromptOperandSource<'a> {Gathered(Rc<DeviceVector<'a>>),Packed(Rc<DeviceVector<'a>>)}
pub struct PromptPacked<'a> {scales_owner:Rc<PromptRowScales<'a>>,codes:DeviceVector<'a>,scales:DeviceVector<'a>,inputs:Vec<PromptOperandSource<'a>>}
impl PromptExpertTable<'_> {
    pub fn experts(&self)->usize {self.activations.len()}
    pub fn geometry(&self)->(usize,usize) {(self.shape.rows as usize,self.shape.cols as usize)}
    pub fn same_calibration(&self,other:&Self)->bool {self.geometry()==other.geometry() && self.shape.role==other.shape.role && self.activations==other.activations}
    pub fn capacity_bytes(&self)->usize {page_words(self.templates.len())}
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {self.templates.allocations(out);for v in &self.owners {v.allocations(out);}}
}
impl<'a> PromptRowScales<'a> {
    pub fn routes(&self)->&[Rc<PromptRoute<'a>>] {&self.routes}
    pub fn values(&self)->&DeviceVector<'a> {&self.values}
    pub fn capacity_bytes(&self)->usize {page_words(self.values.len())+page_words(self.headers.len())}
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {self.values.allocations(out);self.headers.allocations(out);self.table.allocations(out);for r in &self.routes {r.allocations(out);}}
}
impl<'a> PromptNativeBatch<'a> {
    pub fn outputs(&self)->&[Rc<DeviceVector<'a>>] {&self.outputs}
    pub fn row_scales(&self)->&Rc<PromptRowScales<'a>> {&self.scales}
    /// Parts, work and submission headers only; product outputs and shared
    /// route/scales owners keep their separate charges.
    pub fn metadata_capacity_bytes(&self)->usize {page_words(self.parts.len())+page_words(self.work.len())+page_words(self.headers.len())+self.rowmap.as_ref().map_or(0,|v|page_words(v.len()))}
    /// After the product is submitted, give its outputs to the next stream
    /// consumer and retain the small metadata owner through completion.
    pub fn into_outputs(self)->(Vec<Rc<DeviceVector<'a>>>,PromptNativeKeep<'a>) {
        let Self {scales,table,parts,work,headers,outputs,rowmap,..}=self;(outputs,PromptNativeKeep {scales,table,parts,work,headers,rowmap})
    }
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {
        self.scales.allocations(out);self.table.allocations(out);for v in [&self.parts,&self.work,&self.headers] {v.allocations(out);}for v in &self.outputs {v.allocations(out);}
        if let Some(v)=&self.rowmap {v.allocations(out);}
    }
}
impl PromptNativeKeep<'_> {
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {self.scales.allocations(out);self.table.allocations(out);for v in [&self.parts,&self.work,&self.headers] {v.allocations(out);}if let Some(v)=&self.rowmap {v.allocations(out);}}
}
impl PromptPacked<'_> {
    pub fn capacity_bytes(&self)->usize {page_words(self.codes.len())+page_words(self.scales.len())}
    pub fn allocations(&self,out:&mut Vec<(usize,usize)>) {self.scales_owner.allocations(out);self.codes.allocations(out);self.scales.allocations(out);for v in &self.inputs {match v {PromptOperandSource::Gathered(v)|PromptOperandSource::Packed(v)=>v.allocations(out)}}}
}
fn page_words(words:usize)->usize {words.div_ceil(1024)*4096}
fn header_words(v:&[NativeSet])->Vec<f32> {
    v.iter().flat_map(|s| {
        let mut words=Vec::with_capacity(16);for p in [s.counts,s.starts,s.work,s.output,s.activations,s.order] {words.extend([p as u32,(p>>32) as u32]);}words.extend([s.first,s.pairs,s.work_capacity,s.pad]);words
    }).map(f32::from_bits).collect()
}
impl<'a> Backend<'a> {
    /// The complete immutable expert inventory, in ascending expert ID order.
    /// Its shared views retain every weight and scale owner.
    pub fn prompt_expert_table(&self,parts:&[(Shape,&DeviceVector<'a>,&DeviceVector<'a>)])->Result<Option<PromptExpertTable<'a>>,String> {
        let Some(&(head,_,_))=parts.first() else {return Ok(None);};
        let chains=crate::expert_ops::nv_chains(&head);
        if parts.len()>crate::prompt_route::MAX_EXPERTS as usize || !self.tc_image_present() || !crate::expert_ops::nv_tiled(&head) || head.rows==0 || head.rows%128!=0 || head.rows/128>65535
            || head.cols>crate::tc_abi::NV_ROWS_MAX_COLS || crate::tc_contract::chain_steps(head.cols as usize,chains).is_none()
            || parts.iter().any(|(s,_,_)|!matches!(s.op,Op::NvProject|Op::NvRows) || !crate::expert_ops::nv_tiled(s) || s.rows!=head.rows || s.cols!=head.cols || s.role!=head.role || s.block_cols!=16) {return Ok(None);}
        self.require_resident_stream()?;let mut owners=Vec::with_capacity(2*parts.len());let mut words=Vec::with_capacity(10*parts.len());let mut activations=Vec::with_capacity(parts.len());
        for &(shape,weight,scales) in parts {
            let (_,weights,scale_bytes,_,_)=shape.lengths().ok_or("prompt expert weight shape differs")?;
            self.check_vector(weight,weights.div_ceil(4))?;self.check_vector(scales,scale_bytes.div_ceil(4))?;
            if weight.is_segmented() || scales.is_segmented() || (weight.device_address()|scales.device_address())&15!=0 {return Err("prompt expert owner layout differs".into());}
            owners.push(weight.resident_range(0..weight.len())?);owners.push(scales.resident_range(0..scales.len())?);
            for p in [weight.device_address(),scales.device_address(),0] {words.extend([p as u32,(p>>32) as u32]);}words.extend([0,0,shape.global.to_bits(),shape.activation.to_bits()]);activations.push(shape.activation.to_bits());
        }
        let templates=self.upload_vector(&words.into_iter().map(f32::from_bits).collect::<Vec<_>>())?;
        Ok(Some(PromptExpertTable {templates,owners,shape:head,activations}))
    }
    /// Calibrated scales for every sorted row, emitted from the resident inventory.
    /// The complete owner is retained; this call reads no route value.
    pub fn prompt_row_scales(&self,routes:Vec<Rc<PromptRoute<'a>>>,table:Rc<PromptExpertTable<'a>>)->Result<PromptRowScales<'a>,String> {
        self.require_resident_stream()?;
        if routes.is_empty() || routes.len()>crate::prompt_route::MAX_SETS as usize || routes.iter().any(|r|r.shape().experts as usize!=table.experts()) {return Err("prompt native row sets differ".into());}
        self.check_vector(&table.templates,table.experts()*10)?;
        let mut starts=Vec::with_capacity(routes.len());let mut total=0usize;
        for route in &routes {self.check_vector(route.ids(),route.shape().pairs())?;starts.push(total);total=total.checked_add(route.shape().pairs()).ok_or("prompt native row count overflow")?;}
        let mut values=self.vector(total)?;self.skip_fill(&mut values)?;
        let headers=routes.iter().zip(&starts).map(|(r,&first)|Ok(NativeSet {counts:r.counts().device_address(),starts:r.starts().device_address(),work:0,output:0,activations:values.device_address()+(first*4) as u64,order:0,
            first:u32::try_from(first).map_err(|_|"prompt native row offset exceeds ABI")?,pairs:r.shape().pairs() as u32,work_capacity:0,pad:0})).collect::<Result<Vec<_>,String>>()?;
        let headers=self.upload_vector(&header_words(&headers))?;
        let mut args=NativeBuild {templates:table.templates.device_address(),sets:headers.device_address(),table:0,work:0,rowmap:0,count:routes.len() as u32,experts:table.experts() as u32,rows:0,tokens:0,capacity:0,merge:0};
        self.require_arm_kernels(&["prompt_route_native"])?;
        self.unrecorded(||self.launch_checked_expert_payload("prompt_route_native",args.experts,1,THREADS,0,&mut args))?;
        crate::prompt_launches::record("prompt_route_native");Ok(PromptRowScales {routes,table,values:values.into_resident_shared()?,starts,headers})
    }
    /// A cohort of all sets of one expert phase. Parts and tile work are expert/set
    /// interleaved, as the host-built cohort was. Empty experts write no products.
    /// PAIRED PARTS (two or more sets, unless a check tool turned it off): one work
    /// interval an expert over the tokens of all its sets, so a block copies an
    /// expert's weight tile once for both sets. The same part records, outputs and
    /// words; the grid shrinks to experts + ceil(all pairs / tile) records.
    /// DIRECT PLACE (`pair_order`): the build also writes one output address a
    /// sorted row, so the product stores each set's output in original pair order
    /// ([pair][rows], what ExpertPlaceTile with the route's placed table makes),
    /// and the caller skips that placement launch.
    pub fn prompt_native_batch(&self,scales:Rc<PromptRowScales<'a>>,table:Rc<PromptExpertTable<'a>>,pair_order:bool)->Result<PromptNativeBatch<'a>,String> {
        self.require_resident_stream()?;
        self.check_vector(&scales.values,scales.values.len())?;
        if scales.table.geometry()!=table.geometry() || scales.table.activations!=table.activations || scales.table.shape.role!=table.shape.role {return Err("prompt native calibrated inventory differs".into());}
        self.check_vector(&table.templates,table.experts()*10)?;let sets=scales.routes.len();let rows=table.shape.rows as usize;let chains=crate::expert_ops::nv_chains(&table.shape);let (_,tokens)=tc_family::nv_rows_entry(chains);
        let merge=sets>1 && !self.prompt_paired_off();
        let capacity=if merge {crate::prompt_route::merged_work_capacity(table.experts() as u32,scales.values.len(),tokens)} else {scales.routes.iter().map(|r|r.shape().work_capacity(tokens)).sum::<usize>()};
        let capacity=u32::try_from(capacity).map_err(|_|"prompt native work count exceeds ABI")?;
        if capacity>65535 {return Err("prompt native work grid exceeds driver limit".into());}
        let mut parts=self.vector(table.experts()*sets*10)?;self.skip_fill(&mut parts)?;
        let mut work=self.vector(capacity as usize*2)?;self.skip_fill(&mut work)?;
        let rowmap=if pair_order {let mut v=self.vector(scales.values.len().checked_mul(2).ok_or("prompt native row map overflow")?)?;self.skip_fill(&mut v)?;Some(v)} else {None};
        let outputs=scales.routes.iter().map(|r| {let mut v=self.vector(r.shape().pairs().checked_mul(rows).ok_or("prompt native output overflow")?)?;self.skip_fill(&mut v)?;Ok(Rc::new(v))}).collect::<Result<Vec<_>,String>>()?;
        let headers=scales.routes.iter().zip(&scales.starts).zip(&outputs).map(|((r,&first),out)|NativeSet {counts:r.counts().device_address(),starts:r.starts().device_address(),
            work:if tokens==128 {r.work128().device_address()} else {r.work64().device_address()},output:out.device_address(),activations:0,
            order:if pair_order {r.pairs().device_address()} else {0},first:first as u32,pairs:r.shape().pairs() as u32,work_capacity:r.shape().work_capacity(tokens) as u32,pad:0}).collect::<Vec<_>>();
        let headers=self.upload_vector(&header_words(&headers))?;
        let mut args=NativeBuild {templates:table.templates.device_address(),sets:headers.device_address(),table:parts.device_address(),work:work.device_address(),rowmap:rowmap.as_ref().map_or(0,|v|v.device_address()),count:sets as u32,experts:table.experts() as u32,rows:rows as u32,tokens,capacity,merge:u32::from(merge)};
        self.require_arm_kernels(&["prompt_route_native"])?;self.unrecorded(||self.launch_checked_expert_payload("prompt_route_native",args.experts,1,THREADS,0,&mut args))?;
        crate::prompt_launches::record("prompt_route_native");if merge {crate::prompt_launches::record("paired_parts");}
        if pair_order {crate::prompt_launches::record("direct_place");}
        Ok(PromptNativeBatch {scales,table,parts,work,headers,outputs,capacity,merge:if merge {sets as u32} else {0},rowmap})
    }
    /// Same quantizer and packed layout as the existing native cohort. Inputs and
    /// metadata remain owned through both gate/up products or the down product.
    pub fn prompt_route_operands(&self,scales:Rc<PromptRowScales<'a>>,inputs:Vec<PromptOperandSource<'a>>)->Result<PromptPacked<'a>,String> {
        self.require_resident_stream()?;
        self.check_vector(&scales.values,scales.values.len())?;
        if inputs.len()!=scales.routes.len() {return Err("prompt native operand set count differs".into());}
        let cols=scales.table.shape.cols as usize;let total=scales.values.len();let mut codes=self.vector(total.checked_mul(cols/8).ok_or("prompt native operand overflow")?)?;let mut packed_scales=self.vector(total.checked_mul(cols/64).ok_or("prompt native operand overflow")?)?;
        self.skip_fill(&mut codes)?;self.skip_fill(&mut packed_scales)?;
        for ((input,route),&first) in inputs.iter().zip(&scales.routes).zip(&scales.starts) {
            let n=route.shape().pairs();match input {
                PromptOperandSource::Gathered(input)=>{
                    if input.is_segmented() || input.len()!=route.shape().rows as usize*cols {return Err("prompt native gathered input extent differs".into());}self.check_vector(input,input.len())?;
                    let mut args=NvQuantizeGather {input:input.device_address(),codes:codes.device_address()+(first*cols/2) as u64,scales:packed_scales.device_address()+(first*cols/16) as u64,
                        activations:scales.values.device_address()+(first*4) as u64,index:route.order().device_address(),rows:n as u32,cols:cols as u32,source_rows:route.shape().rows,pad:0};
                    self.require_arm_kernels(&["nv_quantize_gather"])?;let blocks=crate::tc_abi::nv_quantize_pack_blocks(n,cols).ok_or("prompt native quantizer grid overflow")?;
                    self.unrecorded(||self.launch_checked_expert_payload("nv_quantize_gather",blocks,1,crate::tc_abi::NV_QUANTIZE_PACK_THREADS,0,&mut args))?;crate::prompt_launches::record("nv_quantize_gather");
                },
                PromptOperandSource::Packed(input)=>{
                    let (a,b)=(n*(cols/8),n*(cols/64));if input.is_segmented() || input.len()!=a+b {return Err("prompt native packed input extent differs".into());}self.check_vector(input,a+b)?;
                    self.unrecorded(|| {self.copy_range(input,&mut codes,0,first*(cols/8),a)?;self.copy_range(input,&mut packed_scales,a,first*(cols/64),b)})?;
                },
            }
        }
        Ok(PromptPacked {scales_owner:scales,codes,scales:packed_scales,inputs})
    }
    pub fn prompt_route_product(&self,batch:&PromptNativeBatch<'a>,packed:&PromptPacked<'a>)->Result<(),String> {
        self.require_resident_stream()?;
        self.check_vector(&batch.parts,batch.parts.len())?;self.check_vector(&packed.codes,packed.codes.len())?;
        if !Rc::ptr_eq(&batch.scales,&packed.scales_owner) {return Err("prompt native product operand epoch differs".into());}
        let shape=batch.table.shape;let chains=crate::expert_ops::nv_chains(&shape);let (entry,_)=tc_family::nv_rows_entry(chains);
        let mut args=NvRows {table:batch.parts.device_address(),codes:packed.codes.device_address(),scales:packed.scales.device_address(),count:(batch.table.experts()*batch.outputs.len()) as u32,
            rows:shape.rows,cols:shape.cols,layout:1,skip:1,raw:0,chains,work:batch.work.device_address(),work_count:batch.capacity,merge:batch.merge,rowmap:batch.rowmap.as_ref().map_or(0,|v|v.device_address())};
        self.require_arm_kernels_tc(&[entry])?;
        self.unrecorded(||self.launch_checked_expert_payload(entry,shape.rows/128,batch.capacity,crate::tc_abi::NV_ROWS_THREADS,0,&mut args))?;
        crate::prompt_launches::record("fp4_cohort");crate::prompt_launches::record("prompt_route_product");Ok(())
    }
}
